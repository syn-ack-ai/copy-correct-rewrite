#!/bin/bash
# side v14, 11-07 22:30 (the RSI pilot leaves Colab; user: "dont run the rsi on the colab, use the 4090 with the smaller models that fit there"). Formerly side v13, 11-07 22:05 (adds addendum 181 before the RSI pilot). Formerly side v12, 11-07 19:50
# (the four-arm Qwen RSI pilot after 179). One launcher, one slot counter (at most 3 Colab sessions; runs/*/status "starting|running"; a lock so
# branches never take the same slot). v12's 179 sessions xqlO1-3 were started by v12 and keep running; v13 replaces only v12's waiting chain.
# Order: 181 seed s starts when 179 seed s (xqlO$s) has ended and the reviewed code is on HF (runs/xqlCF/READY): its 179 adapters (sft, opd_c,
# mix_c) go to HF xq/prior/s$s/ first, for paired evaluation. Then 178 seeds 2, 3, 1. Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqcode xqrA xqrB xqrC xqlB xqlS1 xqlS2 xqlS3 xqs xqt xqcv2 xqcv3 xqw1 xqw2 xqw3 xqrsiS xqlShort xqlO1 xqlO2 xqlO3 xqrsiS1 xqrsiT1 xqlCF1 xqlCF2 xqlCF3"
running() { local n=0; for s in $SESS; do grep -qE "^(starting|running)" $R/$s/status 2>/dev/null && n=$((n + 1)); done; echo $n; }
start() { local s=$1 d=$2; shift 2; mkdir -p $R/$s; echo starting > $R/$s/status
  ( JOB_DESC="$d" $CR -s $s -g G4 -o /content/out -d $R/$s "$@" > $R/$s/colab_run.out 2>&1; log "$s ended: $(cat $R/$s/status)" ) &
  log "started $s ($d)"; }
start_when_free() {   # same arguments as start
  while true; do
    if mkdir $LOCK 2>/dev/null; then
      if [ $(running) -lt 3 ]; then start "$@"; sleep 30; rmdir $LOCK; return; fi
      rmdir $LOCK
    fi
    sleep 60
  done
}
prior() {   # $1 = seed: upload 179's adapters of that seed for the paired evaluation (a missing one is skipped; sft is then retrained in the session)
  local s=$1 a
  for a in sft opd_c mix_c; do
    if [ -f $R/xqlO$s/xl_${a}_s$s/adapter_model.safetensors ]; then
      hf upload SkyPanther/g4agent-job $R/xqlO$s/xl_${a}_s$s xq/prior/s$s/xl_$a --repo-type dataset > /dev/null 2>&1 && log "uploaded 179 seed $s $a to xq/prior/s$s" || log "upload of 179 seed $s $a FAILED"
    else log "179 seed $s: no $a adapter"; fi
  done
}
log "side v14: chain 181 seeds 1-3 (each after its 179 seed) -> 178 seeds 2, 3, 1 (the RSI pilot moves to the 4090)"
(
  until [ -f $R/xqlCF/READY ]; do sleep 60; done; log "181 code ready"
  for s in 1 2 3; do
    until grep -qE "^exit" $R/xqlO$s/status 2>/dev/null; do sleep 60; done
    prior $s
    start_when_free xqlCF$s "181 refit translator: opd_cf + rewrite rw_cf, seed $s (paired with 179 seed $s)" $K/colab/xq_learn.sh STAGE=cf XB_SEED=$s
  done
  for s in 2 3 1; do start_when_free xqw$s "178 wide coding eval of the existing LoRAs, seed $s" $K/colab/xq_wide.sh XB_SEED=$s; done
) &
wait; log "XQ_SIDE14_DONE"
