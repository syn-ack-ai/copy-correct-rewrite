#!/bin/bash
# Addendum 177 (transfer/rsi.py): the RSI pilot. A loop of attempt -> verify -> update on LiveCodeBench (train pool: test.jsonl-test4.jsonl easy
# and medium; held: test5 + test6 easy and medium, never trained on), ROUNDS rounds per loop arm, one Python process per arm (the student stays
# in memory; each round's LoRA is merged into it). Arms SOURCE:UPDATE: S = the student's own verified wins, T = own wins + the stronger model's
# verified rows (TROWS) for the train-pool problems it failed that round; U1 ReST (SFT on the wins), U2 SFT + KL to the previous model, U3 the
# self-bridge (control translator on the student's own states fitted to the wins, then a LoRA toward the corrected model). Model 0 is measured
# once per session and shared by its arms. Small pass first for every arm (6 train / 4 held problems, 384 tokens, 4 steps per phase, 2 rounds),
# then the full run, arm after arm (a failed arm does not stop the next one; the exit code says if any failed).
#   STUDENT=qwen|e4b  ARMS=S:U1,S:U3,T:U1,T:U3  ROUNDS=3  XB_SEED=1  TROWS=hf:xq/learn/xl_rows.jsonl (the qwen default: 174's E4B rows) | /path
#   optional: RSI_BATCH (64) RSI_UPD_STEPS (400) RSI_TR_STEPS (600) RSI_TR_CLEAN_STEPS (0; "300,300" = 175's two clean rounds) RSI_K (4)
#             RSI_LAMBDA (1.0, U2) RSI_FINAL_SAMPLE (1) RSI_R0 (a model-0 file from another session, same seed and settings)
#             SMALL_ONLY=1: stop after the small pass (a cheap check of the whole pipeline on Colab)
#             RSI_UPLOAD=xq/rsi/<name>: upload each finished arm's results to that HF path (needs colab_run -w)
#   first (from the Studio): hf upload SkyPanther/g4agent-job bench/archetypes/transfer/rsi.py xq/code/rsi.py --repo-type dataset
#   v2/infra/colab/colab_run.sh -s xqrsi -g G4 -o /content/out -d kaggle_g4agent/runs/xqrsi kaggle_g4agent/colab/xq_rsi.sh STUDENT=qwen
# Split across sessions: one session per ARMS subset (e.g. ARMS=S:U1,S:U3 and ARMS=T:U1,T:U3); each measures its own model 0. Put every arm's
# rsi_<student>_<arm>.json in one out/ folder and run `XB_DIR=<folder> RSI_NAME=<student> python rsi.py summary` anywhere (CPU, no model).
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
STUDENT=${STUDENT:-qwen}; ARMS=${ARMS:-S:U1,S:U3,S:U5,T:U1,T:U3,T:U5}; ROUNDS=${ROUNDS:-3}; export XB_SEED=${XB_SEED:-1}
case $STUDENT in qwen) TROWS=${TROWS:-hf:xq/learn/xl_rows.jsonl};; e4b) TROWS=${TROWS:-};; *) echo "STUDENT must be qwen or e4b"; exit 1;; esac
NEED_T=0; [[ ",$ARMS," == *",T:"* ]] && NEED_T=1
[ $NEED_T = 1 ] && [ -z "$TROWS" ] && { echo "T arms need TROWS (the stronger model's verified rows); none exist yet for STUDENT=$STUDENT"; exit 1; }
B=/content/bench/transfer; O=/content/out; mkdir -p $B $O
INC=(--include "s3/*" --include "xq/*.py" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/code/*")
[[ $TROWS == hf:* ]] && INC+=(--include "${TROWS#hf:}")
hf download SkyPanther/g4agent-job --repo-type dataset "${INC[@]}" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
cp /content/inx/s3/*.py /content/inx/xq/*.py /content/inx/xq/code/*.py $B/
[ -f $B/rsi.py ] || { echo "rsi.py missing: upload it to xq/code/ on HF first"; exit 1; }
if [ $NEED_T = 1 ]; then
  TR=$TROWS; [[ $TROWS == hf:* ]] && TR=/content/inx/${TROWS#hf:}
  [ -s "$TR" ] || { echo "stronger-model rows missing or empty: $TR"; exit 1; }; export RSI_TROWS=$TR; echo "stronger-model rows: $(wc -l < $TR) ($TROWS)"
fi
NEED_LCB=0; for f in test test2 test3 test4 test5 test6; do [ -f /content/lcb/$f.jsonl ] || NEED_LCB=1; done   # earlier jobs may have fetched test5/6 only
[ $NEED_LCB = 0 ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 \
  --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
if [ $STUDENT = e4b ]; then
  if [ ! -f /content/me4b_qat/config.json ]; then
    hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --local-dir /content/me4b_qat_ct > /dev/null || { echo "QAT download failed"; exit 1; }
    uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
    python $B/dequant.py /content/me4b_qat_ct /content/me4b_qat > $O/dequant_e4b.log 2>&1 && grep DEQUANT_OK $O/dequant_e4b.log || { echo "dequantize FAILED"; exit 1; }
  fi
  export XB_STUDENT=/content/me4b_qat XB_TEACHER=/content/me4b_qat
else
  [ -f /content/mqwen/config.json ] || hf download Qwen/Qwen3-1.7B --local-dir /content/mqwen > /dev/null || { echo "Qwen download failed"; exit 1; }
  GT=/content/me4b_qat
  if [ ! -f $GT/tokenizer.json ]; then GT=/content/me4b_tok
    [ -f $GT/tokenizer.json ] || hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --include "tokenizer*" \
      --include "config.json" --include "special_tokens_map.json" --include "chat_template.jinja" --local-dir $GT > /dev/null || { echo "Gemma tokenizer download failed"; exit 1; }
  fi
  export XB_STUDENT=/content/mqwen XB_TEACHER=$GT   # XB_TEACHER: only E4B's tokenizer (decodes the off-task s3 rows); no teacher model is loaded
fi
[ -f /content/xb/s3_ids.npy ] || python $B/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/
export XC_DATA=/content/lcb XB_SIZE=0.03 XB_HARM=0.5 RSI_NAME=$STUDENT RSI_BATCH=${RSI_BATCH:-64} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
run() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; "$@" > $O/rsi_${STUDENT}_$lg.log 2>&1; local rc=$?
        grep -E "RSI |wins for update|model 0 (greedy|samples|: reusing)|final attempt|merged round|decision rules|LoRA step [0-9]+ of|translator phase [0-9]+ step" $O/rsi_${STUDENT}_$lg.log | tail -n 14 | cut -c1-330
        [ $rc = 0 ] || { echo "$lg FAILED (rc $rc)"; tail -n 25 $O/rsi_${STUDENT}_$lg.log; }; return $rc; }
setup() { rm -rf $1/out; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/; export XB_DIR=$1; }
echo "student $STUDENT | arms $ARMS | rounds $ROUNDS | seed $XB_SEED | batch $RSI_BATCH"
export XB_SMALL=1; unset RSI_ROUNDS; setup /content/rsi_small                                   # small pass: every arm, 2 rounds
for a in ${ARMS//,/ }; do run small_${a/:/-} python rsi.py loop $a || exit 1
  grep -q "merged round 1" $O/rsi_${STUDENT}_small_${a/:/-}.log || { echo "small $a: no update ran in round 1 (review 11-07: the small pass must reach an update)"; exit 1; }; done
run small_summary python rsi.py summary || exit 1; mkdir -p $O/small; cp $XB_DIR/out/rsi_*.json $O/small/
[ "${SMALL_ONLY:-0}" = 1 ] && { echo "$(date +%H:%M) XQRSI_SMALL_DONE"; exit 0; }
unset XB_SMALL; export RSI_ROUNDS=$ROUNDS; setup /content/rsi_s$XB_SEED; FAIL=0; mkdir -p $O/s$XB_SEED   # the full run, arm after arm
for a in ${ARMS//,/ }; do
  t=${a/:/-}; run $t python rsi.py loop $a || FAIL=1
  cp -r $XB_DIR/out/rsi_${STUDENT}_${t}* $XB_DIR/out/rsi_${STUDENT}_m0* $O/s$XB_SEED/ 2>/dev/null   # results survive a later failure
  if [ -n "$RSI_UPLOAD" ]; then   # review 11-07: with colab_run -w, every finished arm's results go to HF at once (a lost session keeps them)
    mkdir -p /content/up_rsi && rm -rf /content/up_rsi/* && cp $XB_DIR/out/rsi_${STUDENT}_${t}*.json $XB_DIR/out/rsi_${STUDENT}_m0*.json /content/up_rsi/ 2>/dev/null
    cp $O/rsi_${STUDENT}_$t.log /content/up_rsi/ 2>/dev/null; hf upload SkyPanther/g4agent-job /content/up_rsi $RSI_UPLOAD/s$XB_SEED --repo-type dataset > /dev/null && echo "$(date +%H:%M) $a uploaded to $RSI_UPLOAD" || echo "$a upload FAILED"
  fi
done
run summary python rsi.py summary && cp $XB_DIR/out/rsi_${STUDENT}_summary.json $O/s$XB_SEED/
echo "$(date +%H:%M) XQRSI_DONE (any arm failed: $FAIL)"; exit $FAIL
