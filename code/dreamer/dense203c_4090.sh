#!/bin/bash
# dense203c_4090.sh (amendment 203c, 2026-11-10): dense203b_4090.sh, now also after 208's live test (X208_DONE or X208_HALTED); nothing else changes.
# dense203b_4090.sh (amendment 203b, 2026-11-09): dense203_4090.sh, now also after 207's live test (X207_DONE or X207_HALTED), with the cutoffs moved
# (not started after 2026-11-10 23:00; no new arm after 2026-11-11 03:00), because the freeze moved to the morning of Nov 11. Nothing else changes.
# Addendum 203 (user 11-09: "Yes, go ahead", to a dense rank-64 LoRA toward the 4x corrector): offline rank-64 LoRAs on E2B (MLP, layers
# 12-34, alpha 128, lr 3e-4, 1,000 steps x 2 rows of the agent training pool), each trained with a dense forward KL (top 32 + tail bin) over the
# reply and up to 512 context positions before it, along E4B's logged replies, toward: the 4x cross corrector (199, data4/out/gall_e4b), E4B
# itself, and the 4x self-corrector (data4/out/gall_e2b, if 199 finished it). Each gated on the 240 points. On the 4090 after 200 (IMIT200_DONE)
# and 202 (X202 line), before 195; skipped if not started by its cutoff (see the top). Always ends with exactly "HH:MM DENSE203_DONE" (online_rep3f
# then writes ONLINE_REP3_DONE for 195). code12 = code11 + train_lora's dense mode. Log ~/toyq/behave/logs/online_rep.log.
#   (cd ~/toyq/behave && exec setsid nohup bash dense203c_4090.sh) > /dev/null 2>&1 < /dev/null &
B=~/toyq/behave; LG=$B/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
done_line() { grep -qE '^[0-9]{2}:[0-9]{2} DENSE203_DONE$' $L 2>/dev/null; }
fin() { log "$1"; done_line || log "DENSE203_DONE"; exit 0; }
trap 'done_line || { log "203 runner ended unexpectedly"; log "DENSE203_DONE"; }' EXIT
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
try() { local lg=$1; shift; timeout ${TO:-1h} env $BASE "$@" > $LG/$lg.log 2>&1 && { log "$lg ok"; return 0; }; log "$lg FAILED (or over ${TO:-1h}): $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
D4=$B/data4; G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D4 ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"; DL="ON_LORA_LOSS=dense ON_RANK=64 ON_LORA_LAYERS=12-34 ON_DENSE_CTX=512"
log "addendum 203 (runner 203c): dense rank-64 LoRAs toward the 4x corrector, E4B and the 4x self-corrector, after 200, 202, 207 and 208, before 195"
dl=$(date -d "2026-11-10 23:00" +%s)
until grep -qE '^[0-9]{2}:[0-9]{2} IMIT200_DONE$' $L 2>/dev/null && grep -qE '^[0-9]{2}:[0-9]{2} X202_(DONE|HALTED)$' $L 2>/dev/null && grep -qE '^[0-9]{2}:[0-9]{2} X207_(DONE|HALTED)$' $L 2>/dev/null && grep -qE '^[0-9]{2}:[0-9]{2} X208_(DONE|HALTED)$' $L 2>/dev/null; do   # amendment 203b: after 207's live test; 203c: and 208's
  [ "$(date +%s)" -gt "$dl" ] && fin "203 not run: no X202, X207 and X208 end lines by 23:00"; sleep 120; done   # review 207: never wait for good (195 has its own free-GPU check)
until [ -f $B/code12/READY ]; do [ "$(date +%s)" -gt "$dl" ] && fin "203 not run: code12/READY missing at 23:00"; sleep 60; done
cd $B/code12 || fin "203 not run: no code12"
free=0; until [ $free -ge 2 ]; do                                       # no process on the GPU, no vLLM, under 2 GB used, twice in a row
  [ "$(date +%s)" -gt "$dl" ] && fin "203 not run: the GPU was not free by 23:00"
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
[ -f $D4/out/gall_e4b.pt ] || fin "203 not run: no 4x cross corrector"
for x in lora_d203_bridge lora_d203_e4b lora_d203_control; do [ -e $D4/out/$x ] && fin "203 not run: $D4/out/$x exists already"; done
log "203 start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
if ! try d203_small $G $DL ON_LORA_TEACH=bridge ON_SMALL=1 ON_LORA_OUT=lora_d203_small $P online.py lora train; then   # review 203: one fallback, fewer context positions
  DL="ON_LORA_LOSS=dense ON_RANK=64 ON_LORA_LAYERS=12-34 ON_DENSE_CTX=256"; log "203: small pass failed at 512 context positions; retrying at 256"
  try d203_small256 $G $DL ON_LORA_TEACH=bridge ON_SMALL=1 ON_LORA_OUT=lora_d203_small256 $P online.py lora train || fin "203 small pass failed at 256 too"
fi
log "203 small pass: $(grep -hE 'peak GPU memory' $LG/d203_small*.log | tail -n 1 | cut -c1-80) | $DL"
arm() {   # TEACHER: train (1,000 steps), then the gate
  TO=4h try d203_train_$1 $G $DL ON_LORA_TEACH=$1 ON_STEPS=1000 ON_LORA_OUT=lora_d203_$1 $P online.py lora train && \
    try d203_gate_$1 $G ON_LORA_NAME=lora_d203_$1 $P online.py lora gate && log "203 $1 gate: $(tail -n 2 $LG/d203_gate_$1.log | head -n 1 | cut -c1-1200)"; }
arm bridge
[ "$(date +%s)" -gt "$(date -d '2026-11-11 03:00' +%s)" ] && fin "203: stopped after the 4x arm (past 03:00)"
arm e4b
[ "$(date +%s)" -gt "$(date -d '2026-11-11 03:00' +%s)" ] && fin "203: stopped after the e4b arm (past 03:00)"
if [ -f $D4/out/gall_e2b.pt ]; then arm control; else log "203: no 4x self-corrector; control arm not run"; fi
fin "203 done"
