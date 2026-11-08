#!/bin/bash
# side v17, 11-08 (185, user: "yes, run the SFT + KL-to-original baseline on colab"): arm sft_self_s (the shortest rows + 1.0 x KL(original
# Qwen || student), no translator) for seeds 1-3 on Colab, each with 179's sft as the in-session anchor (xq_learn.sh STAGE=rb XL_ARMS=sft_self_s
# XL_TAG=sfself); paired afterwards with the stored sft_s, sft_c_s (transfer/rb_pair.py RUNS sft_self_s xqlSS sfself). Slot counter, lock and balance guard as v15 (a
# session needs MIN_CU for itself and each running session). Waits for runs/xqlRBC/READY. Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-20}
log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqw1 xqw2 xqw3 xqlRB0 xqlRB1 xqlRB2 xqlRB3 xqlRBC1 xqlRBC2 xqlRBC3 xqlSS1 xqlSS2 xqlSS3"
running() { local n=0; for s in $SESS; do grep -qE "^(starting|running)" $R/$s/status 2>/dev/null && n=$((n + 1)); done; echo $n; }
balance() { local b k; for k in 1 2 3; do
    b=$( (cd ~ && /usr/bin/perl -e 'alarm 120; exec @ARGV' ~/.local/bin/colab --auth oauth2 usage 2>/dev/null) | sed -n 's/^Current balance: \([0-9.]*\).*/\1/p' | head -n 1)
    [ -n "$b" ] && { echo $b; return; }; sleep 30; done; }
start() { local s=$1 d=$2; shift 2; mkdir -p $R/$s; echo starting > $R/$s/status
  ( JOB_DESC="$d" $CR -s $s -g G4 -o /content/out -d $R/$s "$@" > $R/$s/colab_run.out 2>&1; rc=$?
    grep -qE "^exit" $R/$s/status 2>/dev/null || echo "exit $rc launcher (colab_run ended before writing a status)" > $R/$s/status
    log "$s ended: $(cat $R/$s/status)" ) &
  log "started $s ($d)"; }
start_when_free() { local told=""
  while true; do
    if mkdir $LOCK 2>/dev/null; then
      local n=$(running)
      if [ $n -lt 3 ]; then
        local b=$(balance) need=$((MIN_CU * (n + 1)))
        if [ -n "$b" ] && awk "BEGIN{exit !($b >= $need)}"; then start "$@"; sleep 30; rmdir $LOCK; return 0; fi
        [ -z "$told" ] && log "balance '${b:-unknown}' below $need CU ($n running): $1 waits"; told=1; rmdir $LOCK; sleep 600; continue
      fi
      rmdir $LOCK
    fi
    sleep 60
  done
}
log "side v17: 185 arm sft_self_s seeds 1-3 on Colab; balance guard $MIN_CU CU per session"
until [ -f $R/xqlSS/READY ]; do sleep 60; done; log "185 code ready"
for s in 1 2 3; do start_when_free xqlSS$s "185 arm sft_self_s (shortest rows + KL to the original student), seed $s, 179's sft as anchor" $K/colab/xq_learn.sh STAGE=rb XL_ARMS=sft_self_s XL_TAG=sfself XB_SEED=$s; done
wait; log "XQ_SIDE17_DONE"
