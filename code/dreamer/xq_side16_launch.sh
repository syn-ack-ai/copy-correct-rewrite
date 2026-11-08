#!/bin/bash
# side v16, 11-08 (184c, user: "use the remaining CU to run rb_c on colab"): arm rb_c (the runbook rows + the old control translator's KL over
# the span) for seeds 1-3 on Colab, each session with 179's sft as the in-session anchor (xq_learn.sh STAGE=rb XL_ARMS=rb_c XL_TAG=rbc); paired
# afterwards with the stored rb, sft_s and sft_c_s (transfer/rb_pair.py RUNS rb_c xqlRBC rbc). Slot counter, lock and balance guard as v15 (a
# session needs MIN_CU for itself and each running session). Waits for runs/xqlRBC/READY. Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-20}
log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqw1 xqw2 xqw3 xqlRB0 xqlRB1 xqlRB2 xqlRB3 xqlRBC1 xqlRBC2 xqlRBC3"
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
log "side v16: 184c arm rb_c seeds 1-3 on Colab; balance guard $MIN_CU CU per session"
until [ -f $R/xqlRBC/READY ]; do sleep 60; done; log "184c code ready"
for s in 1 2 3; do start_when_free xqlRBC$s "184c arm rb_c (runbook rows + self-bridge KL), seed $s, 179's sft as anchor" $K/colab/xq_learn.sh STAGE=rb XL_ARMS=rb_c XL_TAG=rbc XB_SEED=$s; done
wait; log "XQ_SIDE16_DONE"
