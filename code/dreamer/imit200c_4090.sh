#!/bin/bash
# Amendment 200c (2026-11-09): 200's imitable correctors after the out-of-memory stop at 14:20. code13 = code12 + the memory fix in online.py
# (the penalty reads the unpadded writes; the same values) and a small pass that trains the longest row again once the penalty is on.
# After real200_4090.sh (REAL200_DONE). Steps as imit200_4090.sh 1 and 3: the small imitable fit (the penalty must become active), then the
# beta 1 and beta 10 cross correctors (data3 rows, 600 steps, no size term), each with the rung, the suite and realize.py. Beta 10 starts only
# before 18:00, so the GPU is free before 201's adapter (about 20:30) and 202; review 200c: every step's timeout is capped at the time left
# before 20:25, and a step that cannot fit (a fit under 75 min left, an evaluation under 10) is skipped. No online run here. Ends with "HH:MM IMIT200C_DONE".
#   (cd ~/toyq/behave && exec setsid nohup bash imit200c_4090.sh) > /dev/null 2>&1 < /dev/null &
B=~/toyq/behave; LG=$B/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
trap 'log "IMIT200C_DONE"' EXIT
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
END=$(date -d "2026-11-09 20:25" +%s); left() { echo $(( END - $(date +%s) )); }
cap() { local want=$1 need=$2 l; l=$(left); [ $l -ge $need ] || return 1; [ $l -lt $want ] && echo $l || echo $want; }   # seconds for a step, or fail
try() { local lg=$1; shift; timeout ${TO:-1h} env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED (or over ${TO:-1h}): $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
EV="TR_SRC_DEPTHS=0.5,0.65"; GT="TR_SRC_DEPTHS=0.5,0.65 ON_TRMIX=1 ON_TR_SIZE=0"
log "200c: imitable correctors queued after the realizability check (REAL200_DONE)"
until grep -qE '^[0-9]{2}:[0-9]{2} REAL200_DONE$' $L 2>/dev/null; do sleep 60; done
[ -f $B/code13/READY ] || { log "200c not run: code13/READY missing"; exit 0; }
cd $B/code13 || exit 0
free=0; until [ $free -ge 2 ]; do                                       # as 200: no process on the GPU, no vLLM, under 2 GB used, twice in a row
  [ $(left) -lt 5400 ] && { log "200c not run: the GPU was not free by 18:55"; exit 0; }
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 30
done
for b in 1 10; do [ -e $B/data5_b$b/out/gall_e4b.pt ] && { log "200c not run: $B/data5_b$b/out/gall_e4b.pt exists already"; exit 0; }; done
TO=$(cap 3600 900) || { log "200c not run: too late for the small pass"; exit 0; }
TO=$TO try i200c_small $GT ON_DIR=$B/data5_bsmall ON_SMALL=1 ON_IMIT_BETA=1 $P online.py train gall_e4b || { log "200c small pass failed"; exit 0; }
grep -o 'unpredicted share [0-9.]*' $LG/i200c_small.log | tail -n 1 | awk '{exit !($3 > 0)}' || { log "200c small pass: the penalty never became active"; exit 0; }
log "200c small imitable fit: $(grep 'unpredicted share' $LG/i200c_small.log | tail -n 1 | cut -c1-200); $(grep -o 'peak memory.*' $LG/i200c_small.log | tail -n 1)"
for b in 1 10; do Dd=$B/data5_b$b
  if [ $b = 10 ] && [ $(date +%s) -gt $(date -d "2026-11-09 18:00" +%s) ]; then log "200c: beta 10 not started (after 18:00)"; break; fi
  TO=$(cap 7200 4500) || { log "200c: beta $b fit not started (under 75 min before 20:25)"; break; }
  if TO=$TO try i200_tr_b$b $GT ON_DIR=$Dd ON_IMIT_BETA=$b ON_STEPS=600 $P online.py train gall_e4b; then
    log "200 beta $b fit: $(grep -E 'gall_e4b step 599 ' $LG/i200_tr_b$b.log | cut -c1-260)"
    for st in rung mix real; do
      TO=$(cap 3600 600) || { log "200c: beta $b $st not run (under 10 min before 20:25)"; continue; }
      case $st in rung) TO=$TO try i200_rung_b$b $EV ON_DIR=$Dd $P online.py rung gall;; mix) TO=$TO try i200_mix_b$b $EV ON_DIR=$Dd $P online.py mixgate gall;; real) TO=$TO try i200_real_b$b $EV ON_DIR=$Dd $P realize.py;; esac
    done
  fi
done
log "200c done"
