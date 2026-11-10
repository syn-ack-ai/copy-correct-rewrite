#!/bin/bash
# Addendum 202 (user 11-09: "yes, queue the live test after the imitable run"): the LoRA from addendum 201 (online setup D, seed 1, with 199's
# 4x corrector; Colab session xqoD4x) live on the same 80 held-out-repo agent tasks as 196 (box tmp/s4/live80.txt), on QAT E2B as Kaggle
# serves it (s4_plain.sh with MREPO/MREV, LORA, SUB=submissions/s1t1024_s4b; vLLM on the 4090's port 8000, never 8080; rollouts on box), two
# attempts (x202_D4x_a, x202_D4x_b), then the loop gate and a per-task comparison with 196's arms (plain, D, C; the same tasks and harness).
# Runs on the Mac. Waits for 201's adapter (and its session to end), then 200's exact end line (IMIT200_DONE) and the GPU free twice; gives
# up at 2026-11-10 03:00. Ends by writing exactly "HH:MM X202_DONE" or "HH:MM X202_HALTED" to the 4090's queue log, which 192
# (online_rep3d_4090.sh) waits for; an EXIT trap writes X202_HALTED if the runner dies first. Log runs/s4/x202.log.
#   kaggle_g4agent/dreamer/detach.sh /tmp/x202.out bash kaggle_g4agent/dreamer/x202_live.sh
# To stop it, kill its whole process group (kill -TERM -PGID), so s4_plain.sh's trap stops vLLM on port 8000 (review 202).
K=~/systemone/kaggle_g4agent; X=$K/runs/s4; L=$X/x202.log; log() { echo "$(date +%H:%M) $*" >> $L; }
dr() { ssh -o ControlPath=none -o ServerAliveInterval=30 dreamer "$@" < /dev/null; }
QL=/home/skypanther/toyq/behave/logs/online_rep.log
ended() { dr "grep -qE '^[0-9]{2}:[0-9]{2} X202_(DONE|HALTED)\$' $QL"; }
mark() { local i; for i in $(seq 1 120); do   # until the exact line is on the 4090 (192 waits for it)
  dr "echo \"\$(date +%H:%M) $1\" >> $QL"; dr "grep -qE '^[0-9]{2}:[0-9]{2} $1\$' $QL" && { log "$1"; return 0; }; sleep 60; done; log "could not write $1 on the 4090"; }
halt() { log "$1"; ended || mark X202_HALTED; trap - EXIT; exit 0; }
trap 'ended || { log "202 runner ended unexpectedly"; mark X202_HALTED; }' EXIT
E2B="MREPO=google/gemma-4-E2B-it-qat-w4a16-ct MREV=971342c08f607aa7779983f6b5289778b5d271a7"; LD=/home/skypanther/toyq/behave/data_x202
A=$K/runs/xqoD4x/lora_onl_bridge_keep_replay; dl=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-10 03:00" +%s)
log "202 queued: waits for 201's adapter, then 200's end line (IMIT200_DONE) and the GPU free twice; gives up at 2026-11-10 03:00"
until [ -f $A/adapter_model.safetensors ] && grep -qE "^exit" $K/runs/xqoD4x/status 2>/dev/null; do
  [ "$(date +%s)" -gt "$dl" ] && halt "202 not run: no 4x adapter by 2026-11-10 03:00"
  grep -qE "xqoD4x: gave up" $K/runs/xql/side.log && halt "202 not run: 201 never got a Colab GPU"
  if ! [ -f $A/adapter_model.safetensors ]; then   # review 202: 201 ended with no adapter (a failed run, a lost session, the launcher done)
    grep -qE "^exit" $K/runs/xqoD4x/status 2>/dev/null && ! grep -qE "^exit (3 no runtime|4 )" $K/runs/xqoD4x/status && halt "202 not run: 201 ended without an adapter ($(cat $K/runs/xqoD4x/status))"
    grep -qE "XQ_ONL7_DONE" $K/runs/xql/side.log && halt "202 not run: 201's launcher ended without an adapter"
  fi
  sleep 120; done
log "201 ended: $(cat $K/runs/xqoD4x/status); adapter present"
dr "mkdir -p $LD && rm -rf $LD/lora_onl_bridge_keep_replay_4x" && scp -q -r $A dreamer:$LD/lora_onl_bridge_keep_replay_4x && dr "test -f $LD/lora_onl_bridge_keep_replay_4x/adapter_model.safetensors" || halt "202 not run: the adapter copy to the 4090 failed"
until dr "grep -qE '^[0-9]{2}:[0-9]{2} IMIT200_DONE\$' $QL"; do [ "$(date +%s)" -gt "$dl" ] && halt "202 not run: 200 had not ended by 2026-11-10 03:00"; sleep 120; done
gpu_free() { local free=0; until [ $free -ge 2 ]; do   # no compute process on the GPU and under 2 GB used, twice in a row
  [ "$(date +%s)" -gt "$dl" ] && [ ! -e $X/x202_started ] && halt "202 not run: the GPU was not free by 2026-11-10 03:00"   # review 202
  set -- $(dr "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
  if [ "${1:-1}" = 0 ] && [ "${2:-99999}" -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done; }
gpu_free
log "202 start"; touch $X/x202_started
run() {   # NAME [LORA]: one attempt of one arm on the 80 tasks; s4_plain.sh stops its vLLM when it ends
  local n=$1 l=$2 k n0 new r
  for k in 1 2; do
    n0=$(wc -l < $X/plain.log)   # review 196: only this arm's new lines count
    if [ -z "$l" ]; then env $E2B $K/dreamer/s4_plain.sh IDS=live80.txt NAME=$n
    else env $E2B $K/dreamer/s4_plain.sh LORA=$l SUB=submissions/s1t1024_s4b IDS=live80.txt NAME=$n; fi
    new=$(tail -n +$((n0 + 1)) $X/plain.log); r=$(echo "$new" | grep -E "^[0-9:]+ $n done: [0-9]+/80 resolved$" | tail -n 1)
    [ -n "$r" ] && { log "${r#* }"; return 0; }
    if [ $k = 1 ] && echo "$new" | grep -q "vLLM not up: HALTED"; then   # e.g. the user's port-8080 server holding the GPU: wait, retry once
      log "$n: vLLM did not start; retry once when the GPU is free"; gpu_free; continue; fi
    halt "$n: no valid result line: $(echo "$new" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done; }
for a in a b; do run x202_D4x_$a $LD/lora_onl_bridge_keep_replay_4x; done
log "loop gate: $(ssh -o ControlPath=none box 'python3 - ~/g4agent/rollouts x202_D4x_a x202_D4x_b x196_plain_a x196_plain_b x196_D_a x196_D_b x196_C_a x196_C_b' < $K/dreamer/live_loops.py 2>&1 | tr '\n' ' ' | cut -c1-3000)"
log "per task: $(ssh -o ControlPath=none box 'cd ~/g4agent/rollouts && python3 -' < $K/dreamer/x202_pertask.py 2>&1 | tr '\n' ' ' | cut -c1-1500)"
mark X202_DONE; trap - EXIT
