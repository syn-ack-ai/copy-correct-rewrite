#!/bin/bash
# Addendum 189 on the 4090 (user 11-08: "yes, queue D seeds 2-3 and the controls after the gate"): online setup D (corrector teacher + keep +
# replay) seeds 2-3, and two seed-1 controls with the same keep + replay: the self-corrector as the teacher (control) and a blend of E4B and base
# E2B whose starting pull matches the corrector's (blend). First the four seed-1 adapters (A, B, C, D) are gated again with per-point output
# (their earlier gate files backed up), for paired tests. Code: code7 = code5 with addendum 189's online.py. Order: re-gates, D seed 2, control,
# blend, D seed 3; a small pass before each run. Runs before the RSI pilot (rsi4090_run.sh waits for ONLINE_REP_DONE). Log
# ~/toyq/behave/logs/online_rep.log; ends with ONLINE_REP_DONE (also after failures).
cd ~/toyq/behave/code7 || exit 1; LG=~/toyq/behave/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; OD=~/toyq/behave/data3/out; log() { echo "$(date +%H:%M) $*" >> $L; }
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$HOME/toyq/behave/data3 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
try() { local lg=$1; shift; env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
gate() { try gate_$1 $G ON_LORA_NAME=lora_$1 $P online.py lora gate && log "$1 gate: $(tail -n 2 $LG/gate_$1.log | head -n 1 | cut -c1-1500)"; }
arm() {   # arm TEACHER SEED: small pass, the 600-item run, its gate
  local t=$1 s=$2 tag="" se=""; [ "$s" != 1 ] && { tag=_s$s; se="ON_SEED=$s"; }; local n=onl_${t}_keep_replay$tag
  if try onl_small_${t}$tag $G $se ON_SMALL=1 ON_ONL_TEACH=$t ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_small189$tag $P online.py online; then
    if try train_$n $G $se ON_ONL_TEACH=$t ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 ON_BEH_TAG=$tag $P online.py online; then
      log "$n: $(grep -E 'blend calibration' $LG/train_$n.log | tail -n 1 | cut -c1-500)"
      log "$n: $(grep -E 'online eval' $LG/train_$n.log | tail -n 1 | cut -c1-500)"; gate $n
    fi
  else log "small pass failed: $n not run"; fi
  log "disk free $(df -h --output=avail / | tail -1)"
}
log "addendum 189: D seeds 2-3 and the control and blend teachers; waiting for a free GPU"
free=0; until [ $free -ge 2 ]; do                                       # no process on the GPU (the pilot's CPU smoke test may still run) and under 2 GB used, twice in a row
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
log "start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
mkdir -p $OD/gate_backup_189 && cp $OD/gate_lora_onl_direct.json $OD/gate_lora_onl_bridge.json $OD/gate_lora_onl_direct_keep_replay.json $OD/gate_lora_onl_bridge_keep_replay.json $OD/gate_backup_189/ && log "seed-1 gate files backed up"
for n in onl_direct onl_bridge onl_direct_keep_replay onl_bridge_keep_replay; do gate $n; done   # per-point output for paired tests; must reproduce the stored gates
arm bridge 2
arm control 1
arm blend 1
arm bridge 3
log "ONLINE_REP_DONE"
