#!/bin/bash
# side v15, 11-08 (addendum 182, user: "yes, build the runbook arm"; the user added 100 CU). One launcher, one slot counter (at most 3 Colab
# sessions; runs/*/status "starting|running"; a lock so branches never take the same slot), and a balance guard: a session starts only while the
# Colab balance covers MIN_CU for it and for each running session (otherwise it waits). Order: once the reviewed code is on HF (runs/xqlRB/READY), the runbook rows are built once (xqlRB0, STAGE=rbuild,
# with the write token so the rows go to HF xq/learn/) and 178 seed 1 (xqw1, which found no runtime at 0 CU on 11-08) starts beside it; when
# xqlRB0 has uploaded the rows (RB_UPLOADED; not on RB_FEW), arm rb runs for seeds 1-3 (xqlRB1-3, STAGE=rb, 179's sft as the in-session anchor).
# Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-20}
log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqcode xqrA xqrB xqrC xqlB xqlS1 xqlS2 xqlS3 xqs xqt xqcv2 xqcv3 xqw1 xqw2 xqw3 xqrsiS xqlShort xqlO1 xqlO2 xqlO3 xqrsiS1 xqrsiT1 xqlCF1 xqlCF2 xqlCF3 xqlRB0 xqlRB1 xqlRB2 xqlRB3"
running() { local n=0; for s in $SESS; do grep -qE "^(starting|running)" $R/$s/status 2>/dev/null && n=$((n + 1)); done; echo $n; }
balance() { local b k; for k in 1 2 3; do   # review 11-08: a 120 s limit per call (the CLI can hang), retried when the balance comes back empty
    b=$( (cd ~ && /usr/bin/perl -e 'alarm 120; exec @ARGV' ~/.local/bin/colab --auth oauth2 usage 2>/dev/null) | sed -n 's/^Current balance: \([0-9.]*\).*/\1/p' | head -n 1)
    [ -n "$b" ] && { echo $b; return; }; sleep 30; done; }
start() { local s=$1 d=$2; shift 2; mkdir -p $R/$s; echo starting > $R/$s/status
  ( JOB_DESC="$d" $CR -s $s -g G4 -o /content/out -d $R/$s "$@" > $R/$s/colab_run.out 2>&1; rc=$?
    grep -qE "^exit" $R/$s/status 2>/dev/null || echo "exit $rc launcher (colab_run ended before writing a status)" > $R/$s/status   # review 11-08: no phantom slot
    log "$s ended: $(cat $R/$s/status)" ) &
  log "started $s ($d)"; }
start_when_free() {   # same arguments as start. Review 11-08: a new session needs MIN_CU for itself and for every session already running;
  local told=""        # below that (or the balance unknown) it waits and checks again every 10 minutes (the user may add CU), logging once
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
log "side v15: 182 runbook rows (xqlRB0) + 178 seed 1 (xqw1), then arm rb seeds 1-3; balance guard $MIN_CU CU"
(
  until [ -f $R/xqlRB/READY ]; do sleep 60; done; log "182 code ready"
  start_when_free xqlRB0 "182 runbook rows: E4B writes code-free plans from its verified solutions, Qwen re-solves with them (built once)" -w $K/colab/xq_learn.sh STAGE=rbuild
  until grep -qE "^exit" $R/xqlRB0/status 2>/dev/null; do sleep 60; done
  if cat $R/xqlRB0/job.log $R/xqlRB0/job.tail 2>/dev/null | grep -q "RB_UPLOADED"; then
    for s in 1 2 3; do start_when_free xqlRB$s "182 arm rb (Qwen's own runbook-guided solutions), seed $s, 179's sft as anchor" $K/colab/xq_learn.sh STAGE=rb XB_SEED=$s; done
  else log "182: no runbook rows uploaded (RB_PROBE_FAIL, RB_FEW or a failure in xqlRB0): no seed sessions"; fi
) &
(
  until [ -f $R/xqlRB/READY ]; do sleep 60; done; sleep 120   # after xqlRB0 has its slot
  start_when_free xqw1 "178 wide coding eval of the existing LoRAs, seed 1" $K/colab/xq_wide.sh XB_SEED=1
) &
wait; log "XQ_SIDE15_DONE"
