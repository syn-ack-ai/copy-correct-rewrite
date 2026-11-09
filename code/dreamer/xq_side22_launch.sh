#!/bin/bash
# side v22, 11-08 (193 KL strength sweep and 194 longer-cap re-grade, both chosen by the user from the Fable review): xq_learn.sh STAGE=e2b
# XL_ARMS=sft_self_s_l03,sft_self_s_l3 (xqlKL1-3) and STAGE=regrade (xqlRG1-3), seeds 1-3; each set waits for its own READY flag (the re-grade for the adapter upload)
# (sft stopped at Qwen's final training loss 0.24, and sft at a tenth of the learning rate; each session re-scores 186's sft adapter for its seed and
# measures drift for all three). Starts only after v19 has started all of 187's sessions (v19's slot counter does not see these), then the same
# slot counter, lock and balance guard (MIN_CU for itself and each running session). Waits for runs/xqlML/READY. Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-25}
log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqlTW2b1 xqlTW2b2 xqlTW2b3 xqlTW4b1 xqlTW4b2 xqlTW4b3 xqlML1 xqlML2 xqlML3 xqlMT1 xqlMT2 xqlMT3 xqlKL1 xqlKL2 xqlKL3 xqlRG1 xqlRG2 xqlRG3"
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
log "side v22: 193 KL sweep and 194 re-grade, 6 sessions; balance guard $MIN_CU CU per session"
until [ -f $R/xqlKL/READY ]; do sleep 60; done; log "193 code ready"
for s in 1 2 3; do start_when_free xqlKL$s "193 KL strength sweep E2B: sft_self_s at lambda 0.3 and 3, 186 sft re-scored, seed $s" $K/colab/xq_learn.sh STAGE=e2b XL_ARMS=sft_self_s_l03,sft_self_s_l3 XB_SEED=$s; done
until [ -f $R/xqlRG/READY ]; do sleep 60; done; log "194 code and adapters ready"
for s in 1 2 3; do start_when_free xqlRG$s "194 longer-cap re-grade: the lost problems of 11 Qwen recipes at 1,536 tokens, seed $s" $K/colab/xq_learn.sh STAGE=regrade XB_SEED=$s; done
wait; log "XQ_SIDE22_DONE"
