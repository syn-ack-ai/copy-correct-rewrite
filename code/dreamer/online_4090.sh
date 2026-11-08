#!/bin/bash
# Addendum 170 (user 11-07: "yes, set up the online learning test on the 4090"): after the arms queue and the library sync (sync_libs.log:
# SYNC_DONE), E2B learns online from a stream of 600 unseen agent decision points (online.py online; code5): small passes (the bridge arm, the
# keep + replay arm), then arms A direct, B bridge (the no-penalty gall translator, data3; translator5 chose data3, tag _g0), C direct + keep 1.0
# + replay every 4th item; each then gate_lora on the 240 points. Review 11-07: held curve every 50 items is the main learning curve. Log ~/toyq/behave/logs/online.log; ONLINE_RUN_DONE or a line ending in HALTED.
cd ~/toyq/behave/code5 || exit 1; LG=~/toyq/behave/logs; L=$LG/online.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$HOME/toyq/behave/data3 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
try() { local lg=$1; shift; env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
log "online: waiting for the arms queue and the library sync"
until grep -q SYNC_DONE $LG/sync_libs.log 2>/dev/null; do sleep 120; done
$P -c "import transformers, trl; assert transformers.__version__ == '5.22.2'" 2>/dev/null || log "warning: the library sync did not install transformers 5.22.2 (see sync_libs.log); running on what is installed"
free=0; until [ $free -ge 2 ]; do                                       # review 11-07: no python online.py process and the GPU free, twice in a row
  if ! pgrep -f "bin/python[0-9.]* online\.py " > /dev/null && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
log "start: $($P -c 'import torch, transformers, peft; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)' 2>/dev/null) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
try onl_small_bridge $G ON_SMALL=1 ON_ONL_TEACH=bridge ON_BEH_TAG=_small $P online.py online || { log "small bridge pass failed: HALTED"; exit 1; }
try onl_small_keep $G ON_SMALL=1 ON_ONL_TEACH=direct ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_small $P online.py online || { log "small keep pass failed: HALTED"; exit 1; }
log "small passes ok"
for arm in "direct::0:0" "bridge::0:0" "direct:_keep_replay:1.0:4"; do
  IFS=: read t suf k rp <<< "$arm"; n=onl_${t}${suf}
  if try train_$n $G ON_ONL_TEACH=$t ON_ONL_KEEP=$k ON_ONL_REPLAY=$rp $P online.py online; then
    log "$n: $(grep -E 'online eval' $LG/train_$n.log | tail -n 1 | cut -c1-400)"
    try gate_$n $G ON_LORA_NAME=lora_$n $P online.py lora gate && log "$n gate: $(tail -n 2 $LG/gate_$n.log | head -n 1 | cut -c1-1500)"
  fi
done
log "ONLINE_RUN_DONE"
