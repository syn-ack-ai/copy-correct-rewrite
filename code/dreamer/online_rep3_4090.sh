#!/bin/bash
# Addendum 192 on the 4090 (user 11-08: "is online a better teacher (with less breaking) then SFT?"; "yes, add both and queue them before the
# graft"): seed 1 on the online test's 600 stream items: offline SFT (plain cross-entropy on E4B's logged replies, about two passes, cosine lr)
# and online SFT (each item once, keep + replay as D), each with a small pass first and the same 240-point gate. After the 189b queue
# (ONLINE_REP2_DONE) with the GPU free twice; code7. Log ~/toyq/behave/logs/online_rep.log; ends with ONLINE_REP3_DONE (the RSI pilot waits for it).
cd ~/toyq/behave/code7 || exit 1; LG=~/toyq/behave/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$HOME/toyq/behave/data3 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
try() { local lg=$1; shift; env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
gate() { try gate_$1 $G ON_LORA_NAME=lora_$1 $P online.py lora gate && log "$1 gate: $(tail -n 2 $LG/gate_$1.log | head -n 1 | cut -c1-1500)"; }
log "addendum 192: offline and online SFT queued after the 189b queue"
until grep -qE '^[0-9]{2}:[0-9]{2} ONLINE_REP2_DONE$' $L 2>/dev/null; do sleep 120; done   # review 192: the exact marker line, not a mention of it
free=0; until [ $free -ge 2 ]; do                                       # no process on the GPU and under 2 GB used, twice in a row
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
log "192 start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
if try lora_small_sft_ce $G ON_SMALL=1 ON_LORA_LOSS=ce ON_LORA_POOL=stream ON_LORA_OUT=lora_sft_ce_small $P online.py lora train; then   # 1. offline SFT
  try train_sft_ce $G ON_LORA_LOSS=ce ON_LORA_POOL=stream ON_LORA_OUT=lora_sft_ce $P online.py lora train && gate sft_ce
else log "small pass failed: offline SFT not run"; fi
if try onl_small_sft $G ON_SMALL=1 ON_ONL_TEACH=sft ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_small192 $P online.py online; then      # 2. online SFT
  n=onl_sft_keep_replay
  if try train_$n $G ON_ONL_TEACH=sft ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 $P online.py online; then
    log "$n: $(grep -E 'online eval' $LG/train_$n.log | tail -n 1 | cut -c1-500)"; gate $n
  fi
else log "small pass failed: online SFT not run"; fi
log "disk free $(df -h --output=avail / | tail -1)"; log "ONLINE_REP3_DONE"
