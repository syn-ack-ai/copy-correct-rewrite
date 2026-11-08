#!/bin/bash
# side v19, 11-08 (187, user: two-way E4B <-> E2B, "I think we should replecate all of that, yeah, including the 600 trained correctors for both"):
# after the 186 E2B sessions end, E2B's own greedy answers (xqlE2B1/xl_gens_qwen.jsonl, the student alone) go to HF xq/learn/xl_gens_e2b_alone.jsonl
# (direction 2's rows) and each seed's 186 E2B sft adapter to xq/prior_e2b/sS/xl_sft (direction 1's anchor); then six sessions, xq_learn.sh STAGE=two:
# XL_DIR=e2b (E4B teaches E2B) and XL_DIR=e4b (E2B teaches E4B), seeds 1-3, each fitting its own 600-step self- and cross-model correctors. Slot counter, lock and balance guard as v15 (a
# session needs MIN_CU for itself and each running session). Waits for runs/xqlTW/READY. Log runs/xql/side.log.
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-20}
log() { echo "$(date +%H:%M) $*" >> $L; }
SESS="xqw1 xqw2 xqw3 xqlRB0 xqlRB1 xqlRB2 xqlRB3 xqlRBC1 xqlRBC2 xqlRBC3 xqlSS1 xqlSS2 xqlSS3 xqlE2B1 xqlE2B2 xqlE2B3 xqlTW2b1 xqlTW2b2 xqlTW2b3 xqlTW4b1 xqlTW4b2 xqlTW4b3"
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
log "side v19: 187 two-way E4B <-> E2B (STAGE=two), 6 sessions after 186; balance guard $MIN_CU CU per session"
until [ -f $R/xqlTW/READY ]; do sleep 60; done; log "187 code ready"
for s in 1 2 3; do until grep -qE "^exit" $R/xqlE2B$s/status 2>/dev/null; do sleep 60; done; done; log "186 sessions ended"
SRC=""; for s in 1 2 3; do grep -q "^exit 0" $R/xqlE2B$s/status && [ -s $R/xqlE2B$s/xl_gens_qwen.jsonl ] && [ -s $R/xqlE2B$s/xl_eval_e2b_s$s.json ] && { SRC=$s; break; }; done   # review 187: any finished seed (E2B alone is deterministic)
DO4B=1
if [ -z "$SRC" ]; then log "187: no finished E2B seed with answers: direction e4b skipped"; DO4B=0
else
  NROWS=$(python3 -c "import json; b = json.load(open('$R/xqlB/xl_base.json')); ok = json.load(open('$R/xqlE2B$SRC/xl_eval_e2b_s$SRC.json'))['per problem']['qwen']; print(sum(1 for i in b['rows'] if ok.get(i)))")
  log "E2B (seed $SRC) passes $NROWS of the 299 row problems alone"
  [ "$NROWS" -ge 30 ] || { log "187: fewer than 30 E2B rows: direction e4b skipped"; DO4B=0; }
  if [ $DO4B = 1 ]; then
    hf upload SkyPanther/g4agent-job $R/xqlE2B$SRC/xl_gens_qwen.jsonl xq/learn/xl_gens_e2b_alone.jsonl --repo-type dataset > /dev/null 2>&1 && hf upload SkyPanther/g4agent-job $R/xqlE2B$SRC/xl_eval_e2b_s$SRC.json xq/learn/xl_eval_e2b_alone.json --repo-type dataset > /dev/null 2>&1 && log "uploaded E2B's own answers and pass flags (seed $SRC)" || { log "upload of E2B's answers FAILED: direction e4b skipped"; DO4B=0; }
  fi
fi
for s in 1 2 3; do
  if [ -f $R/xqlE2B$s/xl_sft_e2b_s$s/adapter_model.safetensors ]; then hf upload SkyPanther/g4agent-job $R/xqlE2B$s/xl_sft_e2b_s$s xq/prior_e2b/s$s/xl_sft --repo-type dataset > /dev/null 2>&1 && log "uploaded E2B sft seed $s" || log "upload of E2B sft seed $s FAILED (the session trains its own)"
  else log "no E2B sft adapter for seed $s (the session trains its own)"; fi
done
for s in 1 2 3; do
  start_when_free xqlTW2b$s "187 two-way: E4B teaches E2B (600-step self + cross correctors, anchors), seed $s" $K/colab/xq_learn.sh STAGE=two XL_DIR=e2b XB_SEED=$s
  [ $DO4B = 1 ] && start_when_free xqlTW4b$s "187 two-way: E2B teaches E4B (600-step self + cross correctors, anchors), seed $s" $K/colab/xq_learn.sh STAGE=two XL_DIR=e4b XB_SEED=$s
done
wait; log "XQ_SIDE19_DONE"
