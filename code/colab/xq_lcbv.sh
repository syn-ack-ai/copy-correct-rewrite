#!/bin/bash
# Amendment 209e (user 11-10: "clean up the data that we train the corrector on", "make sure we use clean, working code, verified both either
# from e4b or e2b", "do a smoke test/sanity check of the data before we train on it"): xq_lcb.sh with verified text only.
#   STAGE=fit TEACH=bridge          fits the 4x cross-model corrector (2,400 steps, 199's settings) on verified solutions only: kind fit_lcb
#                                   (lcb_own.py rows: E4B's and E2B's own test-passing solutions to the 299 row problems, run again here) and
#                                   prose, question and story rows. No agent rows, no on-policy samples, no untested code. Before the fit:
#                                   lcb_own.py check, the smoke test of every coding row (it must print DATA_CHECK_OK or nothing is fitted).
#                                   Then lcbteacher on the 60 held coding rows. Output gall_e4b_okfit.pt.
#   TEACH=bridge CORR=okfit         D4 with that corrector, taught only on answers that pass (ON_VERIFY=1)        -> arm onl_d4v
#   TEACH=direct                    C (E4B's distribution), taught only on answers that pass (the filter's control) -> arm onl_cv
#   STAGE=fit TEACH=control, then TEACH=control CORR=okfit (209g, user: "start the new fit on both the 4x corrector and self-corrector"):
#                                   the 4x self-corrector fitted on the same rows, and S4 taught with it                -> arm onl_s4v
# 209l (user 11-11: "yes, try option 1 first"): ERASEDIR=<folder of the job repository with erase.pt> ERASE_SHA=.. (TEACH=bridge only, code12): the
#                                   direction stub_dir.py found is erased from E4B's states before the cross-model corrector reads them
#                                   (online.py ON_ERASE), in its fit, in the teacher check after the fit and wherever it teaches.
# 209n (user 11-11: "so how can we remove it from e2b?"): VERIFY_HARD=1 (code13): on an item where no sample of the student passes, E4B's verified
#                                   solution is learned as text (cross-entropy), not toward the teacher's distribution on it.
# 209p (user 11-11: "cant we use negative reinforcement to tell the model not to do the stub?"): STUB_NEG=<weight> (code14): a sample of the student
#                                   that is a placeholder answer has its placeholder tokens made less likely (a bounded unlikelihood loss).
# 209r (user 11-11: "add a judge in the middle of that layer"): FB_OWNDIR=<folder with xl_rows_fit.jsonl> FB_SHA=.. (code15): on an item where no
#                                   sample passes, the fallback text is the judged, direct solution of the problem from that file (with VERIFY_HARD=1:
#                                   the hybrid copying judged text).
# In the teaching loop (online.py, ON_VERIFY=1) each of the student's samples is run on the problem's tests; it learns on the passing ones and,
# when none passes, on E4B's verified solution of that problem. Everything else is xq_lcb.sh: the same models, stream, dose, evaluation and gate.
#   v2/infra/colab/colab_run.sh -s xqvD1 -g G4 -o /content/out -d kaggle_g4agent/runs/xqvD1 kaggle_g4agent/colab/xq_lcbv.sh ON_SEED=1 TEACH=bridge CORR=okfit CORR_SHA=.. MD5S=online:..,xlearn:..,lcb_rows:..,xcode:..,xbridge:..,translator:..,lcb_own:..
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${ON_SEED:?}" "${TEACH:?}" "${MD5S:?}"; CORR=${CORR:-none}; N=${LCB_N:-598}; STAGE=${STAGE:-teach}
case "$STAGE:$TEACH:$CORR" in fit:bridge:none|fit:control:none) ARM=none; : "${FIT_W:?STAGE=fit needs FIT_W}" "${LCB_FIT_SHA:?STAGE=fit needs LCB_FIT_SHA}";; teach:bridge:okfit) ARM=onl_d4v; : "${CORR_SHA:?CORR=okfit needs CORR_SHA}";;
  teach:control:okfit) ARM=onl_s4v; : "${CORR_SHA:?CORR=okfit needs CORR_SHA}";; teach:direct:none) ARM=onl_cv;;
  teach:bridge:okfit2) ARM=onl_d4w; : "${CORR_SHA:?CORR=okfit2 needs CORR_SHA}";; teach:control:okfit2) ARM=onl_s4w; : "${CORR_SHA:?CORR=okfit2 needs CORR_SHA}";;
  *) echo "unknown stage and teacher $STAGE:$TEACH:$CORR"; exit 1;; esac   # 209g: control = the self-corrector (fed E2B's own states), fitted and taught the same way; 209h: okfit2 = correctors fitted with verified-token targets (FIT_HARD)
C=/content/onl_code; D=/content/onl; O=/content/out; mkdir -p $C $D/out $O
CODEDIR=${CODEDIR:-code10}   # 209h: code11 = code10 with online.py's ON_FIT_HARD
OWNDIR=${OWNDIR:-own209e}; OUTDIR=${OUTDIR:-out209e}   # 209f: the job repository's folders of the candidate rows and of the fitted corrector (209e's by default)
PRI=(); case "$CORR" in okfit|okfit2) PRI+=(--include "on/$OUTDIR/*");; esac; [ "$STAGE" = fit ] && PRI+=(--include "on/$OWNDIR/*"); [ -n "${ERASEDIR:-}" ] && PRI+=(--include "on/$ERASEDIR/erase.pt"); [ -n "${FB_OWNDIR:-}" ] && PRI+=(--include "on/$FB_OWNDIR/xl_rows_fit.jsonl")   # 209e: the corrector fitted on verified solutions; the candidate rows of its fit
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/s3_*" --include "xq/mix_*" --include "xq/xq_questions.jsonl" --include "on/$CODEDIR/*" --include "on/out/*" --include "on/out4x/*" --include "on/models/*" \
  --include "xq/learn/xl_rows.jsonl" --include "xq/learn/xl_base.json" --include "xq/learn/xl_rows_short.jsonl" --include "xq/learn/xl_short.json" "${PRI[@]}" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
cp /content/inx/on/$CODEDIR/*.py $C/ || exit 1
for kv in ${MD5S//,/ }; do f=${kv%%:*}; [ "$(md5sum $C/$f.py | cut -c1-10)" = "${kv##*:}" ] && echo "md5 ok $f.py" || { echo "$f.py is not the reviewed $CODEDIR version"; exit 1; }; done
for f in online xlearn lcb_rows xcode xbridge translator lcb_own; do case ",$MD5S" in *",$f:"*) ;; *) echo "no md5 given for $f.py"; exit 1;; esac; done
check() { [ "$(sha256sum $1 | cut -d' ' -f1)" = "$2" ] && echo "sha ok $(basename $1)" || { echo "SHA MISMATCH $1"; exit 1; }; }
# the data: one pristine merge (/content/xb, read by the evaluation's off-task rows) and a copy for the stream ($D), to which the lcb rows are appended
python $C/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/ || exit 1
check /content/xb/s3_ids.npy 84cf0f5b1ff41cb1c12e64079a51fb25a90bba1a15aa2de2d669aa290edb9043
check /content/xb/s3_calls.jsonl 2df5f431e5dbcd8468ffb1fce5798d7cbb10a0d3422e8405477223ef604bd8cd
cp /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl $D/ || exit 1
cp /content/inx/on/out/gall_e4b.pt /content/inx/on/out/gall_e2b.pt $D/out/ || exit 1
check $D/out/gall_e4b.pt 476ea42f4b0a2be7213c32da112f4135c9af1d76ba69b7ec35086d91e5d260b2
check $D/out/gall_e2b.pt eada45ff8c8f78599c324c8a6239b6eb890a250182817b9c873be54b2eeb4f54
if [ "$TEACH:$CORR" = bridge:okfit ] || [ "$TEACH:$CORR" = bridge:okfit2 ]; then cp /content/inx/on/$OUTDIR/gall_e4b_okfit.pt $D/out/gall_e4b.pt || exit 1; check $D/out/gall_e4b.pt $CORR_SHA; fi    # 209e: the 4x cross-model corrector fitted on verified solutions only
if [ "$TEACH:$CORR" = control:okfit ] || [ "$TEACH:$CORR" = control:okfit2 ]; then cp /content/inx/on/$OUTDIR/gall_e2b_okfit.pt $D/out/gall_e2b.pt || exit 1; check $D/out/gall_e2b.pt $CORR_SHA; fi   # 209g: the 4x self-corrector fitted on the same rows
GE=""; if [ -n "${ERASEDIR:-}" ]; then   # 209l: the erased direction, for the cross-model corrector only
  [ "$TEACH" = bridge ] || { echo "ERASEDIR is for TEACH=bridge"; exit 1; }; : "${ERASE_SHA:?ERASEDIR needs ERASE_SHA}"
  cp /content/inx/on/$ERASEDIR/erase.pt $D/out/erase.pt || exit 1; check $D/out/erase.pt $ERASE_SHA; GE="ON_ERASE=$D/out/erase.pt"; fi
FBE=""; if [ -n "${FB_OWNDIR:-}" ]; then : "${FB_SHA:?FB_OWNDIR needs FB_SHA}"; check /content/inx/on/$FB_OWNDIR/xl_rows_fit.jsonl $FB_SHA; FBE="ON_FALLBACK_ROWS=/content/inx/on/$FB_OWNDIR/xl_rows_fit.jsonl"; fi   # 209r
used_erase() { [ -z "$GE" ] || grep -q "erased from the corrector's input" $1 || { echo "ERASEDIR was given and $(basename $1) did not use it"; exit 1; }; }
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
# 206's environment (xq_online.sh), with the coding stream; samples up to 512 tokens
BASE="E4B=/content/me4b_qat E2B=/content/me2b_qat TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=512 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
V="ON_VERIFY=1 ${VERIFY_HARD:+ON_VERIFY_HARD=1} ${STUB_NEG:+ON_STUB_NEG=$STUB_NEG} $FBE XC_DATA=/content/lcb XB_DIR=/content/xbv"   # 209e: the teaching loop runs the student's samples on the problems' tests (xlearn's pools and tests)
s=$ON_SEED; unset ON_SEED; tag=$([ "$s" = 1 ] || echo _s$s); se=$([ "$s" = 1 ] || echo ON_SEED=$s); n=onl_${TEACH}_keep_replay_lcbv$tag; ns=onl_${TEACH}_keep_replay_lcbv_small$tag; cd $C   # as xq_online.sh: seed 1 has no ON_SEED at all
try() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; env $BASE "$@" > $O/$lg.log 2>&1 && return 0
        echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 20; cp $D/out/*_partial.json $O/ 2> /dev/null; exit 1; }
# the coding study's evaluation environment (xq_learn.sh STAGE=e2b), on this session's E2B; E4B lends only its tokenizer and config
xl_setup() { rm -rf $1; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/
  cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/; }
xl() { local lg=$1 dir=$2; shift 2; echo "$(date +%H:%M) $lg"
  env XB_DIR=$dir XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=$s XB_SIZE=0.03 XB_HARM=0.5 \
      XL_EXTRA_ARMS=onl_d4,onl_s4,onl_c,onl_sft,onl_d4f,onl_s4f,onl_d4v,onl_cv,onl_s4v,onl_d4w,onl_s4w PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" > $O/$lg.log 2>&1 && return 0
  echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 25; exit 1; }
xl_setup /content/xl_onl_small || { echo "evaluation setup failed"; exit 1; }
xl lcb_rows /content/xl_onl_small python lcb_rows.py $D; grep -E "lcb rows|decode back|LCB_ROWS_OK" $O/lcb_rows.log | cut -c1-300
grep -q LCB_ROWS_OK $O/lcb_rows.log || { echo "no lcb rows"; exit 1; }
if [ "$STAGE" = fit ]; then   # 209e: the verified rows, the smoke test of the data, then the fit and the teacher check; no teaching here
  src=$([ "$TEACH" = bridge ] && echo e4b || echo e2b)   # 209g: bridge fits the cross-model corrector (gall_e4b), control the self-corrector (gall_e2b)
  xl fit_rows /content/xl_onl_small LCB_FIT=/content/inx/on/$OWNDIR/xl_rows_fit.jsonl LCB_FIT_SHA=$LCB_FIT_SHA python lcb_own.py rows $D; grep -E "left out|FIT_ROWS_OK" $O/fit_rows.log | cut -c1-400
  grep -q FIT_ROWS_OK $O/fit_rows.log || { echo "no fit rows"; exit 1; }
  echo "$(date +%H:%M) data_check"   # the smoke test: nothing is fitted unless every coding row passes it
  env XB_DIR=/content/xl_onl_small XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=$s FIT_W=$FIT_W python lcb_own.py check $D > $O/data_check.log 2>&1; rc=$?
  grep -vE "Warning|warn" $O/data_check.log | tail -n 40 | cut -c1-600; cp $D/out/lcb_data_check.json $O/ 2> /dev/null
  [ $rc = 0 ] && grep -q DATA_CHECK_OK $O/data_check.log || { echo "DATA_CHECK_FAILED: nothing is fitted"; exit 1; }
  N0=$(python -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['split'] == 'train'))" /content/xb/s3_calls.jsonl) || exit 1; [ "$N0" = 3295 ] || { echo "the 1x fitting pool has $N0 rows, not 3295"; exit 1; }
  FT="TR_SRC_DEPTHS=0.5,0.65 ON_TRMIX=1 ON_TR_SIZE=0 ON_CAL_POOL=$N0 ON_DIR=$D ON_TR_W=$FIT_W ${FIT_HARD:+ON_FIT_HARD=$FIT_HARD} $GE"   # 199's fit settings; the calibration rows (state spreads only) are the 1x pool's
  try lcbteacher_small_$src $G ON_SMALL=1 ON_ONL_TEACH=$TEACH python online.py lcbteacher; rm -f $D/out/lcb_teacher_$TEACH.json   # the check's own small pass, on the 1x corrector still in $D/out, before the fit
  try fit_small_$src $FT ON_SMALL=1 python online.py train gall_$src; grep -E "translator gall|step 0 |step 19 |erased from" $O/fit_small_$src.log | cut -c1-330; used_erase $O/fit_small_$src.log
  try fit_$src $FT ON_STEPS=2400 python online.py train gall_$src; grep -E "translator gall|step 2399 |erased from" $O/fit_$src.log | cut -c1-400; used_erase $O/fit_$src.log
  [ -z "${FIT_HARD:-}" ] || grep "verified-token targets so far" $O/fit_$src.log | tail -n 1 | grep -qE "so far: [1-9]" || { echo "FIT_HARD was asked for and the fit did not use it"; exit 1; }   # review 209h: the last line must count at least one position
  [ -z "${FIT_HARD:-}" ] || grep "verified-token targets so far" $O/fit_$src.log | tail -n 1 | cut -c1-200
  cp $D/out/gall_$src.pt $O/gall_${src}_okfit.pt || { echo "no fitted corrector"; exit 1; }; sha256sum $O/gall_${src}_okfit.pt
  try lcbteacher_$src $G $GE ON_ONL_TEACH=$TEACH python online.py lcbteacher; grep "lcb teacher" $O/lcbteacher_$src.log | cut -c1-400; used_erase $O/lcbteacher_$src.log
  cp $D/out/lcb_teacher_$TEACH.json $O/ || exit 1
  echo "$(date +%H:%M) XQC_FIT_DONE"; exit 0
fi
# small passes: four stream items with this teacher, then the evaluation on a handful of problems with the small adapter (the whole path, before the long run)
try onl_small_${TEACH}_lcbv$tag $G $GE $V $se ON_SMALL=1 ON_STREAM=lcb ON_ONL_TEACH=$TEACH ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_lcbv_small$tag python online.py online
grep -E "lcb stream|online eval|verify" $O/onl_small_${TEACH}_lcbv$tag.log | head -n 6 | cut -c1-400
grep -q "^.* verify: last" $O/onl_small_${TEACH}_lcbv$tag.log || { echo "the small pass did not verify any sample"; exit 1; }; used_erase $O/onl_small_${TEACH}_lcbv$tag.log
[ -z "${VERIFY_HARD:-}" ] || grep -q "as text (cross-entropy) so far" $O/onl_small_${TEACH}_lcbv$tag.log || { echo "VERIFY_HARD was asked for and online.py does not know it (CODEDIR?)"; exit 1; }   # review 209n: before the long run
[ -z "$FBE" ] || grep -q "judged fallback text:" $O/onl_small_${TEACH}_lcbv$tag.log || { echo "FB_OWNDIR was given and online.py does not know it (CODEDIR?)"; exit 1; }
[ -z "${STUB_NEG:-}" ] || grep -q "placeholder samples made less likely so far" $O/onl_small_${TEACH}_lcbv$tag.log || { echo "STUB_NEG was asked for and online.py does not know it (CODEDIR?)"; exit 1; }
cp -r $D/out/lora_$ns /content/xl_onl_small/out/xl_$ARM || { echo "no small adapter"; ls $D/out; exit 1; }
xl eval_small /content/xl_onl_small XB_SMALL=1 python xlearn.py evaluate; grep -E "evaluating|passed|EVAL_DONE" $O/eval_small.log | tail -n 4 | cut -c1-300
grep -q EVAL_DONE $O/eval_small.log || { echo "small evaluation did not finish"; exit 1; }
# the run
try train_$n $G $GE $V $se ON_STREAM=lcb ON_ONL_N=$N ON_ONL_TEACH=$TEACH ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 ON_BEH_TAG=_lcbv$tag python online.py online
grep -E "online eval" $O/train_$n.log | tail -n 1 | cut -c1-500; grep "verify: last" $O/train_$n.log | tail -n 1 | cut -c1-300
cp $D/out/online_${TEACH}_keep_replay_lcbv$tag.json $O/ && cp -r $D/out/lora_$n $O/ || { echo "training outputs missing"; ls $D/out; exit 1; }   # saved before anything else runs
[ -z "${VERIFY_HARD:-}" ] || grep "as text (cross-entropy) so far" $O/train_$n.log | tail -n 1 | grep -qE "so far: [1-9]" || { echo "VERIFY_HARD was asked for and the run did not use it"; exit 1; }
[ -z "${STUB_NEG:-}" ] || grep "placeholder samples made less likely so far" $O/train_$n.log | tail -n 1 | grep -oE "placeholder samples made less likely so far: [0-9]+ \([0-9]+ tokens\)"   # review 209p: printed, not required (a student may sample no placeholder)
# the coding study's evaluation
xl_setup /content/xl_onl || { echo "evaluation setup failed"; exit 1; }
cp -r $D/out/lora_$n /content/xl_onl/out/xl_$ARM || exit 1
xl eval /content/xl_onl python xlearn.py evaluate; grep -E "evaluating|passed|re-scored" $O/eval.log | tail -n 5 | cut -c1-300
cp /content/xl_onl/out/xl_eval_s$s.json $O/xl_eval_lcb_${ARM}_s$s.json && cp /content/xl_onl/out/xl_gens_*.jsonl $O/ || { echo "evaluation outputs missing"; ls /content/xl_onl/out; exit 1; }
echo "$(date +%H:%M) XQC_EVAL_DONE"
# 206's agent gate on the coding-taught LoRA (what it does to agent decisions); last, so a failure here loses nothing above
try gate_$n $G $GE ON_LORA_NAME=lora_$n python online.py lora gate
tail -n 2 $O/gate_$n.log | head -n 1 | cut -c1-800
cp $D/out/gate_lora_$n.json $D/out/gate_lora_${n}_points.json $O/ || { echo "gate outputs missing"; ls $D/out; exit 1; }
echo "$(date +%H:%M) XQC_DONE"
