#!/bin/bash
# Addendum 189b on the 4090 (user 11-08: "yes, queue C seeds 2-3 after D seed 3"): setup C (E4B's own distribution + keep + replay) seeds 2-3,
# so D and C both have three seeds. After online_rep_4090.sh (ONLINE_REP_DONE) with the GPU free twice; the same code7, settings, small pass
# and gate as its arms. Log ~/toyq/behave/logs/online_rep.log (the same queue log); ends with ONLINE_REP2_DONE, which the RSI pilot waits for.
cd ~/toyq/behave/code7 || exit 1; LG=~/toyq/behave/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$HOME/toyq/behave/data3 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
try() { local lg=$1; shift; env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
gate() { try gate_$1 $G ON_LORA_NAME=lora_$1 $P online.py lora gate && log "$1 gate: $(tail -n 2 $LG/gate_$1.log | head -n 1 | cut -c1-1500)"; }
arm() {   # arm TEACHER SEED: small pass, the 600-item run, its gate (as online_rep_4090.sh)
  local t=$1 s=$2 tag="" se=""; [ "$s" != 1 ] && { tag=_s$s; se="ON_SEED=$s"; }; local n=onl_${t}_keep_replay$tag
  if try onl_small_${t}$tag $G $se ON_SMALL=1 ON_ONL_TEACH=$t ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_small189$tag $P online.py online; then
    if try train_$n $G $se ON_ONL_TEACH=$t ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 ON_BEH_TAG=$tag $P online.py online; then
      log "$n: $(grep -E 'online eval' $LG/train_$n.log | tail -n 1 | cut -c1-500)"; gate $n
    fi
  else log "small pass failed: $n not run"; fi
  log "disk free $(df -h --output=avail / | tail -1)"
}
log "addendum 189b: C seeds 2-3 queued after the 189 queue (ONLINE_REP_DONE)"
until grep -q ONLINE_REP_DONE $L 2>/dev/null; do sleep 120; done
free=0; until [ $free -ge 2 ]; do                                       # no process on the GPU and under 2 GB used, twice in a row
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
log "189b start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
arm direct 2
arm direct 3
log "ONLINE_REP2_DONE"
