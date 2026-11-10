#!/bin/bash
# Amendment 200c (2026-11-09): 200's runner (imit200_4090.sh) ended at 14:20 when the imitable fit's small pass ran out of memory, before
# step 2. This runs step 2 unchanged (realize.py, code11, the same commands and environment; its small pass r200_small passed at 14:20):
# the 1x pair (data4_1x) and the 4x pair (data4). Writes "HH:MM REAL200_DONE" at the end (never IMIT200_DONE). About 1 h.
#   (cd ~/toyq/behave && exec setsid nohup bash real200_4090.sh) > /dev/null 2>&1 < /dev/null &
B=~/toyq/behave; LG=$B/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
trap 'log "REAL200_DONE"' EXIT
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
try() { local lg=$1; shift; timeout ${TO:-1h} env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED (or over ${TO:-1h}): $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
D1=$B/data4_1x; D4=$B/data4; EV="TR_SRC_DEPTHS=0.5,0.65"
cd $B/code11 || exit 1
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] || { log "200c: GPU busy, realizability not run"; exit 0; }
log "200c: realizability of the 1x and 4x pairs (200 step 2, unchanged)"
try r200_1x $EV ON_DIR=$D1 $P realize.py && cp $D1/out/realize_gall.json $D1/out/realize_1x.json
try r200_4x $EV ON_DIR=$D4 $P realize.py
