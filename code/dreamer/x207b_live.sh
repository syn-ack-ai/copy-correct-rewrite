#!/bin/bash
# Addendum 207 (user 11-09: "yes, go on all three": a larger live test ahead of 203): setup D's LoRA live again on the 80 held-out-repo agent
# tasks of 196 (box tmp/s4/live80.txt), on QAT E2B as Kaggle serves it (s4_plain.sh; vLLM on the 4090's port 8000, never 8080; rollouts on
# box), with the control 196 lacked: a null adapter (D's adapter with every lora_B zeroed: plain E2B's outputs through the same adapter
# path and submission, s1t1024_s4b). Attempts, in this order (one s4_plain.sh run each):
#   x207_N_a  x207_D_c  x207_plain_c  x207_N_b  x207_D_d  x207_plain_d      null adapter; D seed 1 (the 4090 adapter of 196); plain
#   x207_CA_a x207_DA_a x207_CA_b x207_DA_b                                 C and D seed 1 from the A100 reruns (xqoC1, xqoD1)
# Then the loop gate and x207_pertask.py over 196, 202 and 207. Runs on the Mac. x207b (amendment 207b): an early block (N_a, plain_c, N_b; no
# early adapter attempt after 18:40 on Nov 9, no early plain attempt after 19:40) runs at once while the 4090 is idle; the rest waits for 202's exact end line on the 4090's queue log
# (X202_DONE or X202_HALTED) and the GPU free twice; nothing after 202 if it has not ended by 2026-11-10 12:00; no new attempt after 19:00. Ends by
# writing exactly "HH:MM X207_DONE" or "HH:MM X207_HALTED" to the 4090's queue log, which 203 (dense203b_4090.sh) waits for; an EXIT trap
# writes X207_HALTED if the runner dies first. Log runs/s4/x207.log.
#   kaggle_g4agent/dreamer/detach.sh /tmp/x207.out bash kaggle_g4agent/dreamer/x207b_live.sh
# To stop it, kill its whole process group (kill -TERM -PGID), so s4_plain.sh's trap stops vLLM on port 8000.
K=~/systemone/kaggle_g4agent; X=$K/runs/s4; L=$X/x207.log; log() { echo "$(date +%H:%M) $*" >> $L; }
dr() { ssh -o ControlPath=none -o ServerAliveInterval=30 dreamer "$@" < /dev/null; }
QL=/home/skypanther/toyq/behave/logs/online_rep.log
ended() { dr "grep -qE '^[0-9]{2}:[0-9]{2} X207_(DONE|HALTED)\$' $QL"; }
mark() { local i; for i in $(seq 1 120); do   # until the exact line is on the 4090 (203 waits for it); one ssh, so a failed check cannot add it twice
  dr "grep -qE '^[0-9]{2}:[0-9]{2} $1\$' $QL || echo \"\$(date +%H:%M) $1\" >> $QL; grep -qE '^[0-9]{2}:[0-9]{2} $1\$' $QL" && { log "$1"; return 0; }; sleep 60; done; log "could not write $1 on the 4090"; }
halt() { log "$1"; ended || mark X207_HALTED; trap - EXIT; exit 0; }
novllm() { local i; for i in $(seq 1 20); do dr 'pgrep -f "[v]llm serve.*--port 8000" > /dev/null' || return 0; sleep 30; done; }   # review 207: the marker only after vLLM on 8000 is gone
trap 'ended || { log "207 runner ended unexpectedly"; novllm; mark X207_HALTED; }' EXIT
E2B="MREPO=google/gemma-4-E2B-it-qat-w4a16-ct MREV=971342c08f607aa7779983f6b5289778b5d271a7"
LD=/home/skypanther/toyq/behave/data_x207; D1=/home/skypanther/toyq/behave/data3/out/lora_onl_bridge_keep_replay
dl=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 12:00:00" +%s); last=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 19:00:00" +%s)
until dr true; do sleep 30; done                                        # review 207: a failed ssh is not a missing file
if ended; then log "207 not started: an X207 end line is already in the 4090's queue log (remove it before a rerun)"; trap - EXIT; exit 1; fi
rm -f $X/x207_started
log "207 (x207b) queued: the early block starts when the GPU is free twice; the rest after 202's end line"
for x in $D1 $LD/lora_null $LD/lora_C1_a100 $LD/lora_D1_a100; do dr "test -f $x/adapter_model.safetensors && test -f $x/adapter_config.json" || halt "207 not run: missing $x"; done
gpu_free() { local free=0; until [ $free -ge 2 ]; do   # no compute process on the GPU and under 2 GB used, twice in a row
  [ "$(date +%s)" -gt "$dl" ] && [ ! -e $X/x207_started ] && halt "207 not run: the GPU was not free by 2026-11-10 12:00"
  [ "$(date +%s)" -gt "$last" ] && { log "207: the GPU was not free by 19:00 on Nov 10"; return 1; }   # review 207b: never wait for good
  set -- $(dr "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
  if [ "${1:-1}" = 0 ] && [ "${2:-99999}" -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done; }
gpu_free
log "207 start (early block before 202, while the 4090 is idle)"; touch $X/x207_started
run() {   # NAME [LORA]: one attempt of one arm on the 80 tasks; s4_plain.sh stops its vLLM when it ends
  local n=$1 l=$2 k n0 new r t0
  for k in 1 2; do
    n0=$(wc -l < $X/plain.log); t0=$(date +%s)   # review 196: only this arm's new lines count; review 207: the time of this try only
    if [ -z "$l" ]; then env $E2B $K/dreamer/s4_plain.sh IDS=live80.txt NAME=$n
    else env $E2B $K/dreamer/s4_plain.sh LORA=$l SUB=submissions/s1t1024_s4b IDS=live80.txt NAME=$n; fi
    new=$(tail -n +$((n0 + 1)) $X/plain.log); r=$(echo "$new" | grep -E "^[0-9:]+ $n done: [0-9]+/80 resolved$" | tail -n 1)
    [ -n "$r" ] && { log "${r#* } in $(( ($(date +%s) - t0) / 60 )) min"; return 0; }
    if [ $k = 1 ] && [ -z "$EARLY" ] && echo "$new" | grep -q "vLLM not up: HALTED"; then   # review 207b: no blind retry in the early block (202 may be about to start)   # e.g. the user's port-8080 server holding the GPU: wait, retry once
      log "$n: vLLM did not start; retry once when the GPU is free"; gpu_free; continue; fi
    log "$n: no valid result line: $(echo "$new" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; return 1
  done; return 1; }
DONE=""; FAIL=0; SKIP=""
e1=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-09 18:40:00" +%s); e2=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-09 19:40:00" +%s)   # last early start: an adapter attempt (up to about 2 h), a plain attempt (about 50 min)
x202line() { dr "grep -qE '^[0-9]{2}:[0-9]{2} X202_(DONE|HALTED)\$' $QL"; }
J=$K/runs/xqoD4x; x202near() { [ -e $J/job.log ] || [ -e $J/out.tgz ] || [ -e $J/lora_onl_bridge_keep_replay ] || [ -e $X/x202_started ] || x202line; }   # review 207b: 201's session has ended (202 is about to poll the GPU), or 202 has started or ended
attempt() {   # NAME [LORA]: one attempt, unless it is done already; a failure is counted and vLLM is waited out
  case " $DONE " in *" $1 "*) return 0;; esac
  if run "$@"; then DONE="$DONE $1"; else FAIL=$((FAIL + 1)); novllm; return 1; fi; }
# amendment 207b: the early block. The 4090 is idle until 202 (about 20:45): the null adapter and plain run now. No early adapter attempt
# starts after 18:40 and no early plain attempt after 19:40; none after 202's end line or a failed attempt; 202 then finds the GPU free.
EARLY=1
for a in "x207_N_a $LD/lora_null" "x207_plain_c" "x207_N_b $LD/lora_null"; do
  set -- $a
  c=$e1; [ -z "$2" ] && c=$e2
  [ "$(date +%s)" -gt "$c" ] && continue; x202near && break; [ $FAIL -ge 1 ] && break
  attempt "$@"
done
EARLY=""
log "early block done:$DONE; now waits for 202's end line"
until x202line; do [ "$(date +%s)" -gt "$dl" ] && { log "207: 202 had not ended by 2026-11-10 12:00; the remaining attempts are not run"; SKIP=1; break; }; sleep 120; done
if [ -z "$SKIP" ] && gpu_free; then
  for a in "x207_N_a $LD/lora_null" "x207_D_c $D1" "x207_plain_c" "x207_N_b $LD/lora_null" "x207_D_d $D1" "x207_plain_d" \
           "x207_CA_a $LD/lora_C1_a100" "x207_DA_a $LD/lora_D1_a100" "x207_CA_b $LD/lora_C1_a100" "x207_DA_b $LD/lora_D1_a100"; do
    set -- $a
    [ "$(date +%s)" -gt "$last" ] && { log "207: no new attempt after 19:00; not run: $1 and the ones after it"; break; }
    [ $FAIL -ge 2 ] && { log "207: two failed attempts; not run: $1 and the ones after it"; break; }
    attempt "$@" || { [ $FAIL -ge 2 ] || gpu_free; }   # review 207b: after a failure, wait for the GPU as before
  done
fi
log "attempts done:$DONE"
log "loop gate: $(ssh -o ControlPath=none -o ServerAliveInterval=30 -o ConnectTimeout=20 box "python3 - ~/g4agent/rollouts $DONE x196_plain_a x196_plain_b x196_D_a x196_D_b x196_C_a x196_C_b" < $K/dreamer/live_loops.py 2>&1 | tr '\n' ' ' | cut -c1-4000)"
log "per task: $(ssh -o ControlPath=none -o ServerAliveInterval=30 -o ConnectTimeout=20 box 'cd ~/g4agent/rollouts && python3 -' < $K/dreamer/x207_pertask.py 2>&1 | tr '\n' ' ' | cut -c1-4000)"
mark X207_DONE; trap - EXIT
