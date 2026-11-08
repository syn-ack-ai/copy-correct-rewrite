#!/bin/bash
# Addendum 180 (user 11-07: "YEs, do this: Should I queue resdec_g0's control and replication seeds on the 4090 for after the online test?"):
# resdec_g0 (state pull along the decision basis toward the gall-bridged E4B states) fixed 27 of 52 gate decisions, the best of the series, with
# no control. After the online test (ONLINE_RUN_DONE), with the synced libraries (transformers 5.22.2, peft 0.21.2) for every arm, so seed 1
# is rerun too: for seeds 1-3, resdec (bridge source) and resctldec (the same pull toward E2B's own control-translator states), each gated with
# per-point output (code6), then rep2_summary.py: fixed counts and paired exact sign tests per seed and pooled. Log ~/toyq/behave/logs/resdec.log.
cd ~/toyq/behave/code6 || exit 1; LG=~/toyq/behave/logs; L=$LG/resdec.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
D3=$HOME/toyq/behave/data3; G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D3 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
try() { local lg=$1; shift; env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
gate() { try gate_$1 $2 ON_LORA_NAME=$3 $P online.py lora gate && log "$1 gate: $(tail -n 2 $LG/gate_$1.log | head -n 1 | cut -c1-600)"; }
log "waiting for the online test (ONLINE_RUN_DONE in online.log)"
until grep -q ONLINE_RUN_DONE $LG/online.log 2>/dev/null; do sleep 120; done
free=0; until [ $free -ge 2 ]; do if ! pgrep -f "[v]llm serve.*--port 8000" > /dev/null && ! pgrep -f "bin/python[0-9.]* online\.py " > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 30; done
[ -f $D3/out/resonance_decbasis.pt ] || { log "no decision basis in $D3/out: stop"; exit 1; }
for s in 1 2 3; do
  SE=""; [ $s != 1 ] && SE="ON_SEED=$s"   # seed 1 = the original seeds (offset 0)
  try train_resdec_g0n$s $G $SE ON_RES_BASIS=dec ON_BEH_TAG=_g0n$s $P online.py behave res && gate resdec_g0n$s "$G" lora_beh_resdec_g0n$s
  try train_resctldec_g0n$s $G $SE ON_RES_SRC=control ON_RES_BASIS=dec ON_BEH_TAG=_g0n$s $P online.py behave res && gate resctldec_g0n$s "$G" lora_beh_resctldec_g0n$s
  $P ~/toyq/behave/code6/rep2_summary.py > $LG/rep2_summary.json 2>&1; log "summary after seed $s: $(cut -c1-1500 $LG/rep2_summary.json)"
done
log "RESDEC_DONE"
