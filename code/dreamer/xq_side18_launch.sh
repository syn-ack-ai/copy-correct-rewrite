#!/bin/bash
# side v18, 11-08 (186, user: "replicate with e2b"; "added 100 more cu"): Gemma 4 E2B as the coding student, arms sft, sft_s, rb, seeds 1-3,
# one Colab session per seed (xq_learn.sh STAGE=e2b XB_SEED=s); each session also measures drift from the original E2B. Slot counter, lock and balance guard as v15 (a
# session needs MIN_CU for itself and each running session). Waits for runs/xqlRBC/READY. Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-20}
log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqw1 xqw2 xqw3 xqlRB0 xqlRB1 xqlRB2 xqlRB3 xqlRBC1 xqlRBC2 xqlRBC3 xqlSS1 xqlSS2 xqlSS3 xqlE2B1 xqlE2B2 xqlE2B3"
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
log "side v18: 186 E2B student (sft, sft_s, rb) seeds 1-3 on Colab; balance guard $MIN_CU CU per session"
until [ -f $R/xqlE2B/READY ]; do sleep 60; done; log "186 code ready"
for s in 1 2 3; do start_when_free xqlE2B$s "186 Gemma 4 E2B as the coding student (sft, sft_s, rb), seed $s" $K/colab/xq_learn.sh STAGE=e2b XB_SEED=$s; done
wait; log "XQ_SIDE18_DONE"
