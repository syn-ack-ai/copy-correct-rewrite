#!/bin/bash
# Addendum 196 (user 11-08: "yes, set up D's live test ahead of 195"): online setup D's LoRA live on 80 held-out-repo agent tasks (box
# tmp/s4/live80.txt = live40 + conf40), against C's LoRA and plain E2B, all on QAT E2B as Kaggle serves it (s4_plain.sh with MREPO/MREV; vLLM on
# the 4090's port 8000, never 8080; rollouts on box), full strength, two attempts per arm in the order plain, D, C, plain, D, C; then the loop
# gate (live_loops.py). Runs on the Mac. Starts after the 189 queue's exact end line (196a: before 192) with the GPU free twice. Ends by writing exactly
# "HH:MM X196_DONE" (or "HH:MM X196_HALTED") to the 4090's ~/x195/logs/x196.log, which 195's runner waits for. Log runs/s4/x196.log.
#   kaggle_g4agent/dreamer/detach.sh /tmp/x196.out bash kaggle_g4agent/dreamer/x196_live.sh
K=~/systemone/kaggle_g4agent; X=$K/runs/s4; L=$X/x196.log; log() { echo "$(date +%H:%M) $*" >> $L; }
dr() { ssh -o ControlPath=none -o ServerAliveInterval=30 dreamer "$@" < /dev/null; }
mark() { local i; for i in $(seq 1 120); do   # review 196: until the exact line is on the 4090 (195 waits for it)
  dr "echo \"\$(date +%H:%M) $1\" >> ~/x195/logs/x196.log"; dr "grep -qE '^[0-9]{2}:[0-9]{2} $1\$' ~/x195/logs/x196.log" && { log "$1"; return 0; }; sleep 60; done; log "could not write $1 on the 4090"; }
halt() { log "$1"; mark X196_HALTED; exit 0; }
E2B="MREPO=google/gemma-4-E2B-it-qat-w4a16-ct MREV=971342c08f607aa7779983f6b5289778b5d271a7"; LD=/home/skypanther/toyq/behave/data3/out
log "196 queued (196a): waits for the 189 queue's exact end line on the 4090 (after D seed 3; before 192), then the GPU free twice"
until dr "grep -qE '^[0-9]{2}:[0-9]{2} ONLINE_REP_DONE\$' ~/toyq/behave/logs/online_rep.log"; do sleep 120; done
gpu_free() { local free=0; until [ $free -ge 2 ]; do   # no compute process on the GPU and under 2 GB used, twice in a row
  set -- $(dr "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
  if [ "${1:-1}" = 0 ] && [ "${2:-99999}" -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done; }
gpu_free
for x in lora_onl_bridge_keep_replay lora_onl_direct_keep_replay; do dr "test -f $LD/$x/adapter_model.safetensors" || halt "missing $LD/$x"; done
log "196 start"
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
for a in a b; do
  run x196_plain_$a; run x196_D_$a $LD/lora_onl_bridge_keep_replay; run x196_C_$a $LD/lora_onl_direct_keep_replay
done
log "loop gate: $(ssh -o ControlPath=none box 'python3 - ~/g4agent/rollouts x196_plain_a x196_plain_b x196_D_a x196_D_b x196_C_a x196_C_b' < $K/dreamer/live_loops.py 2>&1 | tr '\n' ' ' | cut -c1-2500)"
mark X196_DONE
