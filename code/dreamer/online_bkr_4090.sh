#!/bin/bash
# Addendum 183 (user 11-08: "yes, queue the bridge keep-and-replay arm after resdec"): the online test's arm D, the bridge teacher (base E2B
# bridged by the no-penalty gall translator, as arm B) + keep 1.0 (KL to base E2B on one untaught row per item) + replay every 4th item (as
# arm C). The same code (code5), settings, stream and gate as addendum 170's three arms. After the resdec queue (RESDEC_DONE) and a free GPU
# (twice); a small pass first. Then gate_lora on the 240 points. Log ~/toyq/behave/logs/online_bkr.log; ends with ONLINE_BKR_DONE (also after
# a failure, so the RSI pilot, which waits for it, still starts).
cd ~/toyq/behave/code5 || exit 1; LG=~/toyq/behave/logs; L=$LG/online_bkr.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$HOME/toyq/behave/data3 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
try() { local lg=$1; shift; env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
log "online arm D (bridge + keep + replay): waiting for the resdec queue (RESDEC_DONE)"
until grep -q RESDEC_DONE $LG/resdec.log 2>/dev/null; do sleep 120; done
free=0; until [ $free -ge 2 ]; do                                       # as online_4090.sh: no python online.py process, no vLLM, the GPU free, twice in a row
  if ! pgrep -f "bin/python[0-9.]* online\.py " > /dev/null && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
log "start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
if try onl_small_bkr $G ON_SMALL=1 ON_ONL_TEACH=bridge ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_small $P online.py online; then
  n=onl_bridge_keep_replay
  if try train_$n $G ON_ONL_TEACH=bridge ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 $P online.py online; then
    log "$n: $(grep -E 'online eval' $LG/train_$n.log | tail -n 1 | cut -c1-400)"
    try gate_$n $G ON_LORA_NAME=lora_$n $P online.py lora gate && log "$n gate: $(tail -n 2 $LG/gate_$n.log | head -n 1 | cut -c1-1500)"
  fi
else log "small pass failed: arm D not run"; fi
log "ONLINE_BKR_DONE"
