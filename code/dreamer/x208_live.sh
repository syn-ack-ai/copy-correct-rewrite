#!/bin/bash
# Addendum 208 (user 11-10: "improve this as much as we can before submission. the focus should be the novel pattern"): the self-corrector
# students' LoRAs live on the same 80 held-out-repo agent tasks as 196, 202 and 207 (box tmp/s4/live80.txt), on QAT E2B as Kaggle serves it
# (s4_plain.sh; vLLM on the 4090's port 8000, never 8080; rollouts on box). Does the live gain of D's LoRA need E4B's states, or does it come
# with the corrector's form? Attempts, in this order (one s4_plain.sh run each):
#   x208_S_a  x208_S4_a  x208_S_b  x208_S4_b
#   S  = the 1x self-corrector's student, seed 1, trained on the 4090 (189; data3/out/lora_onl_control_keep_replay): the twin of 196's D adapter
#   S4 = the 4x self-corrector's student, seed 1, trained on a G4 (206, xqgSCx1; data_x208/lora_S4_g4): the twin of 202's 4x D adapter (A100)
# Starts after 207's exact end line on the 4090's queue log (X207_DONE or X207_HALTED) and the GPU free twice; not started if 207 has not ended
# by 2026-11-10 18:00; no new attempt after 20:00. Ends by writing exactly "HH:MM X208_DONE" or "HH:MM X208_HALTED" to the 4090's queue log,
# which 203 (dense203c_4090.sh) waits for; an EXIT trap writes X208_HALTED if the runner dies first. Log runs/s4/x208.log.
#   kaggle_g4agent/dreamer/detach.sh /tmp/x208.out bash kaggle_g4agent/dreamer/x208_live.sh
# To stop it, kill its whole process group (kill -TERM -PGID), so s4_plain.sh's trap stops vLLM on port 8000.
K=~/systemone/kaggle_g4agent; X=$K/runs/s4; L=$X/x208.log; log() { echo "$(date +%H:%M) $*" >> $L; }
dr() { ssh -o ControlPath=none -o ServerAliveInterval=30 dreamer "$@" < /dev/null; }
QL=/home/skypanther/toyq/behave/logs/online_rep.log
ended() { dr "grep -qE '^[0-9]{2}:[0-9]{2} X208_(DONE|HALTED)\$' $QL"; }
mark() { local i; for i in $(seq 1 120); do   # until the exact line is on the 4090 (203 waits for it); one ssh, so a failed check cannot add it twice
  dr "grep -qE '^[0-9]{2}:[0-9]{2} X208_(DONE|HALTED)\$' $QL || echo \"\$(date +%H:%M) $1\" >> $QL; grep -qE '^[0-9]{2}:[0-9]{2} X208_(DONE|HALTED)\$' $QL" && { log "$1"; return 0; }; sleep 60; done; log "could not write $1 on the 4090"; }   # review 208: never a second X208 line, of either kind
halt() { log "$1"; ended || mark X208_HALTED; trap - EXIT; exit 0; }
novllm() { local i; for i in $(seq 1 20); do dr 'pgrep -f "[v]llm serve.*--port 8000" > /dev/null; [ $? = 1 ]' && return 0; sleep 30; done; }   # the marker only after vLLM on 8000 is gone (review 208: a failed ssh is not "gone")
trap 'ended || { log "208 runner ended unexpectedly"; novllm; mark X208_HALTED; }' EXIT
E2B="MREPO=google/gemma-4-E2B-it-qat-w4a16-ct MREV=971342c08f607aa7779983f6b5289778b5d271a7"
LD=/home/skypanther/toyq/behave/data_x208; S1=/home/skypanther/toyq/behave/data3/out/lora_onl_control_keep_replay
dl=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 18:00:00" +%s); last=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 20:00:00" +%s)
until dr true; do sleep 30; done                                        # a failed ssh is not a missing file
if ended; then log "208 not started: an X208 end line is already in the 4090's queue log (remove it before a rerun)"; trap - EXIT; exit 1; fi
rm -f $X/x208_started
log "208 queued: waits for 207's end line (X207_DONE or X207_HALTED) and the GPU free twice; not started after 2026-11-10 18:00"
for x in $S1 $LD/lora_S4_g4; do ok=""; for k in 1 2 3; do dr "test -f $x/adapter_model.safetensors && test -f $x/adapter_config.json" && { ok=1; break; }; sleep 20; done; [ -n "$ok" ] || halt "208 not run: missing $x"; done   # review 208: three tries
x207line() { dr "grep -qE '^[0-9]{2}:[0-9]{2} X207_(DONE|HALTED)\$' $QL"; }
until x207line; do [ "$(date +%s)" -gt "$dl" ] && halt "208 not run: 207 had not ended by 2026-11-10 18:00"; sleep 120; done
gpu_free() { local free=0; until [ $free -ge 2 ]; do   # no compute process on the GPU and under 2 GB used, twice in a row
  [ "$(date +%s)" -gt "$dl" ] && [ ! -e $X/x208_started ] && halt "208 not run: the GPU was not free by 2026-11-10 18:00"
  [ "$(date +%s)" -gt "$last" ] && { log "208: the GPU was not free by 20:00 on Nov 10"; return 1; }   # never wait for good
  set -- $(dr "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
  if [ "${1:-1}" = 0 ] && [ "${2:-99999}" -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done; }
dr 'pgrep -f "[d]ense203b_4090.sh" > /dev/null; [ $? = 1 ]' || halt "208 not run: the old 203 runner (dense203b_4090.sh) is still alive on the 4090, or the check failed"   # review 208: 203b does not wait for 208
gpu_free || halt "208 not run: the GPU was not free in time"
log "208 start"; touch $X/x208_started
run() {   # NAME LORA: one attempt of one arm on the 80 tasks; s4_plain.sh stops its vLLM when it ends
  local n=$1 l=$2 k n0 new r t0
  for k in 1 2; do
    n0=$(wc -l < $X/plain.log); t0=$(date +%s)   # only this arm's new lines count; the time of this try only
    env $E2B $K/dreamer/s4_plain.sh LORA=$l SUB=submissions/s1t1024_s4b IDS=live80.txt NAME=$n
    new=$(tail -n +$((n0 + 1)) $X/plain.log); r=$(echo "$new" | grep -E "^[0-9:]+ $n done: [0-9]+/80 resolved$" | tail -n 1)
    [ -n "$r" ] && { log "${r#* } in $(( ($(date +%s) - t0) / 60 )) min"; return 0; }
    if [ $k = 1 ] && echo "$new" | grep -q "vLLM not up: HALTED"; then   # e.g. the user's port-8080 server holding the GPU: wait, retry once
      log "$n: vLLM did not start; retry once when the GPU is free"; gpu_free || return 1; continue; fi
    log "$n: no valid result line: $(echo "$new" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; return 1
  done; return 1; }
DONE=""; FAIL=0
attempt() {   # NAME LORA: one attempt; a failure is counted and vLLM is waited out
  if run "$@"; then DONE="$DONE $1"; else FAIL=$((FAIL + 1)); novllm; return 1; fi; }
for a in "x208_S_a $S1" "x208_S4_a $LD/lora_S4_g4" "x208_S_b $S1" "x208_S4_b $LD/lora_S4_g4"; do
  set -- $a
  [ "$(date +%s)" -gt "$last" ] && { log "208: no new attempt after 20:00; not run: $1 and the ones after it"; break; }
  [ $FAIL -ge 2 ] && { log "208: two failed attempts; not run: $1 and the ones after it"; break; }
  attempt "$@" || { [ $FAIL -ge 2 ] || gpu_free || break; }   # after a failure, wait for the GPU as before
done
log "attempts done:$DONE"
log "loop gate: $(ssh -o ControlPath=none -o ServerAliveInterval=30 -o ConnectTimeout=20 box "python3 - ~/g4agent/rollouts $DONE x207_N_a x207_N_b x196_D_a x196_D_b x207_D_c x207_D_d x202_D4x_a x202_D4x_b" < $K/dreamer/live_loops.py 2>&1 | tr '\n' ' ' | cut -c1-4000)"
log "per task: $(ssh -o ControlPath=none -o ServerAliveInterval=30 -o ConnectTimeout=20 box 'cd ~/g4agent/rollouts && python3 -' < $K/dreamer/x208_pertask.py 2>&1 | tr '\n' ' ' | cut -c1-8000)"
novllm; mark X208_DONE; trap - EXIT   # review 208: vLLM gone before the marker on the normal path too
