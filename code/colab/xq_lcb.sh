#!/bin/bash
# Addendum 209 (user 11-10: "can we try Dx4 on contest coding? That seems like the most imporant next tests we run ... before anything else"):
# the online teachers of addendum 206 on contest coding. One teacher and one seed per session, on a Colab G4:
#   TEACH=bridge  CORR=4x    D4: base E2B corrected by 199's 4x cross-model corrector (fed E4B's states)      -> arm onl_d4
#   TEACH=control CORR=4x    S4: base E2B corrected by 199's 4x self-corrector (fed E2B's own states)        -> arm onl_s4
#   TEACH=direct             C:  E4B's own distribution                                                      -> arm onl_c
#   TEACH=sft                copying: cross-entropy on E4B's verified solution (no sampling)                 -> arm onl_sft
# Everything else is 206's recipe unchanged (xq_online.sh: rank-16 LoRA, lr 3e-4, two samples per item, keep 1.0 on one untaught row per item,
# replay every 4th item, the 4090's dequantized models and the same correctors, sha256-checked), except: the stream is the coding study's 299
# row problems (the prompt; E4B's shortest verified solution as the logged reply), 598 items = two seeded passes (the coding study's dose of
# two epochs, and 206's stream length); a sample is at most 512 tokens (xlearn's on-policy arms; 206: 256) and ends at its closing fence.
# Then the coding study's own evaluation (xlearn.py evaluate, unchanged but for accepting the adapter's name): greedy answers of E2B alone and
# with the LoRA on the 299 row problems and the 188 held problems, hidden tests, the held cross-entropy and the off-task rows; and 206's
# 240-point agent gate for what coding teaching does to agent decisions. Small passes of the rows, the stream and the evaluation come first.
# Amendment 209a (user 11-10, after seed 1: "we never fitted this Dx4 to the contest problems, we have to do that first / train it on that x4
# like we did for the other teacher"): STAGE=fit fits a 4x corrector (2,400 steps, 199's settings: no size term, do-no-harm 0.5, noise, depth
# drop, the 1x calibration rows) on 206's fitting rows plus the contest rows: the 299 row problems with E4B's verified solution (kind lcb) and
# E2B's own samples at those prompts (onpolicy_lcb), as 206's correctors saw agent replies and E2B's own samples at agent points.
# TEACH=bridge fits the cross-model corrector (gall_e4b), TEACH=control the self-corrector (gall_e2b); then lcbteacher reads the fitted
# corrector as a teacher on the 60 held coding rows; outputs gall_<src>_lcbfit.pt. CORR=fit (with CORR_SHA) teaches with such a corrector,
# uploaded by the launcher to on/out209/: arms onl_d4f (bridge) and onl_s4f (control).
#   v2/infra/colab/colab_run.sh -s xqcD1 -g G4 -o /content/out -d kaggle_g4agent/runs/xqcD1 kaggle_g4agent/colab/xq_lcb.sh ON_SEED=1 TEACH=bridge CORR=4x MD5S=online:..,xlearn:..,lcb_rows:..,xcode:..,xbridge:..,translator:..
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${ON_SEED:?}" "${TEACH:?}" "${MD5S:?}"; CORR=${CORR:-none}; N=${LCB_N:-598}; STAGE=${STAGE:-teach}
case "$TEACH:$CORR" in bridge:4x) ARM=onl_d4;; control:4x) ARM=onl_s4;; direct:none) ARM=onl_c;; sft:none) ARM=onl_sft;; bridge:fit) ARM=onl_d4f;; control:fit) ARM=onl_s4f;; bridge:none|control:none) ARM=none;; *) echo "unknown teacher $TEACH:$CORR"; exit 1;; esac
case "$STAGE:$TEACH:$CORR" in fit:bridge:none|fit:control:none) ;; fit:*|*:bridge:none|*:control:none) echo "unknown stage $STAGE for $TEACH:$CORR"; exit 1;; teach:*) ;; *) echo "unknown stage $STAGE"; exit 1;; esac   # 209a: a corrector teacher always names its corrector; the fit stage names none
[ "$CORR" = fit ] && : "${CORR_SHA:?CORR=fit needs CORR_SHA}"
C=/content/onl_code; D=/content/onl; O=/content/out; mkdir -p $C $D/out $O
PRI=(); [ "${PRIOR_SFT:-0}" = 1 ] && PRI=(--include "xq/prior_e2b/s$ON_SEED/xl_sft/*"); [ "$CORR" = fit ] && PRI+=(--include "on/out209/*")   # 209a: the correctors fitted with the contest rows   # the coding study's offline SFT adapter of this seed (186), as a reference arm in this session's evaluation
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/s3_*" --include "xq/mix_*" --include "xq/xq_questions.jsonl" --include "on/code9/*" --include "on/out/*" --include "on/out4x/*" --include "on/models/*" \
  --include "xq/learn/xl_rows.jsonl" --include "xq/learn/xl_base.json" --include "xq/learn/xl_rows_short.jsonl" --include "xq/learn/xl_short.json" "${PRI[@]}" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
cp /content/inx/on/code9/*.py $C/ || exit 1
for kv in ${MD5S//,/ }; do f=${kv%%:*}; [ "$(md5sum $C/$f.py | cut -c1-10)" = "${kv##*:}" ] && echo "md5 ok $f.py" || { echo "$f.py is not the reviewed code9 version"; exit 1; }; done
for f in online xlearn lcb_rows xcode xbridge translator; do case ",$MD5S" in *",$f:"*) ;; *) echo "no md5 given for $f.py"; exit 1;; esac; done
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
if [ "$TEACH:$CORR" = bridge:4x ]; then cp /content/inx/on/out4x/gall_e4b.pt $D/out/gall_e4b.pt || exit 1; check $D/out/gall_e4b.pt 80c7ccdf00d4de4a58dc06229a17707024ab86e61a9968c806ab4a8d5a555bb9; fi    # 201, 206: D4's teacher
if [ "$TEACH:$CORR" = control:4x ]; then cp /content/inx/on/out4x/gall_e2b.pt $D/out/gall_e2b.pt || exit 1; check $D/out/gall_e2b.pt 07a097db27592545a495dee95e4f858d587c692a1455e97779f33e89e761e792; fi   # 206: S4's teacher
if [ "$TEACH:$CORR" = bridge:fit ]; then cp /content/inx/on/out209/gall_e4b_lcbfit.pt $D/out/gall_e4b.pt || exit 1; check $D/out/gall_e4b.pt $CORR_SHA; fi    # 209a: the 4x cross-model corrector fitted with the contest rows
if [ "$TEACH:$CORR" = control:fit ]; then cp /content/inx/on/out209/gall_e2b_lcbfit.pt $D/out/gall_e2b.pt || exit 1; check $D/out/gall_e2b.pt $CORR_SHA; fi   # 209a: the 4x self-corrector fitted with the contest rows
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
# 206's environment (xq_online.sh), with the coding stream; samples up to 512 tokens
BASE="E4B=/content/me4b_qat E2B=/content/me2b_qat TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=512 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
s=$ON_SEED; unset ON_SEED; tag=$([ "$s" = 1 ] || echo _s$s); se=$([ "$s" = 1 ] || echo ON_SEED=$s); n=onl_${TEACH}_keep_replay_lcb$tag; ns=onl_${TEACH}_keep_replay_lcb_small$tag; cd $C   # as xq_online.sh: seed 1 has no ON_SEED at all
try() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; env $BASE "$@" > $O/$lg.log 2>&1 && return 0
        echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 20; cp $D/out/*_partial.json $O/ 2> /dev/null; exit 1; }
# the coding study's evaluation environment (xq_learn.sh STAGE=e2b), on this session's E2B; E4B lends only its tokenizer and config
xl_setup() { rm -rf $1; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/
  cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/; }
xl() { local lg=$1 dir=$2; shift 2; echo "$(date +%H:%M) $lg"
  env XB_DIR=$dir XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=$s XB_SIZE=0.03 XB_HARM=0.5 \
      XL_EXTRA_ARMS=onl_d4,onl_s4,onl_c,onl_sft,onl_d4f,onl_s4f PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" > $O/$lg.log 2>&1 && return 0
  echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 25; exit 1; }
xl_setup /content/xl_onl_small || { echo "evaluation setup failed"; exit 1; }
xl lcb_rows /content/xl_onl_small python lcb_rows.py $D; grep -E "lcb rows|decode back|LCB_ROWS_OK" $O/lcb_rows.log | cut -c1-300
grep -q LCB_ROWS_OK $O/lcb_rows.log || { echo "no lcb rows"; exit 1; }
if [ "$STAGE" = fit ]; then   # 209a: fit the corrector with the contest rows, then read it as a teacher on the held coding rows; no teaching here
  src=$([ "$TEACH" = bridge ] && echo e4b || echo e2b)
  N0=$(python -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['split'] == 'train'))" /content/xb/s3_calls.jsonl) || exit 1; [ "$N0" = 3295 ] || { echo "the 1x fitting pool has $N0 rows, not 3295"; exit 1; }
  FT="TR_SRC_DEPTHS=0.5,0.65 ON_TRMIX=1 ON_TR_SIZE=0 ON_CAL_POOL=$N0 ON_DIR=$D ON_TR_W=${FIT_W:?STAGE=fit needs FIT_W}"   # 199's fit settings; the calibration rows are the 1x pool's (the lcb rows come after it)
  try lcbteacher_small_$src $G ON_SMALL=1 ON_ONL_TEACH=$TEACH python online.py lcbteacher; rm -f $D/out/lcb_teacher_$TEACH.json   # review 209a: the check's own small pass, on the 1x corrector still in $D/out, before the long fit
  try fit_small_$src $FT ON_SMALL=1 python online.py train gall_$src; grep -E "translator gall|step 0 |step 19 " $O/fit_small_$src.log | cut -c1-330
  try fit_$src $FT ON_STEPS=2400 python online.py train gall_$src; grep -E "translator gall|step 2399 " $O/fit_$src.log | cut -c1-400
  cp $D/out/gall_$src.pt $O/gall_${src}_lcbfit.pt || { echo "no fitted corrector"; exit 1; }; sha256sum $O/gall_${src}_lcbfit.pt
  try lcbteacher_$src $G ON_ONL_TEACH=$TEACH python online.py lcbteacher; grep "lcb teacher" $O/lcbteacher_$src.log | cut -c1-400
  cp $D/out/lcb_teacher_$TEACH.json $O/ || exit 1
  echo "$(date +%H:%M) XQC_FIT_DONE"; exit 0
fi
# small passes: four stream items with this teacher, then the evaluation on a handful of problems with the small adapter (the whole path, before the long run)
try onl_small_${TEACH}_lcb$tag $G $se ON_SMALL=1 ON_STREAM=lcb ON_ONL_TEACH=$TEACH ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=_lcb_small$tag python online.py online
grep -E "lcb stream|online eval" $O/onl_small_${TEACH}_lcb$tag.log | head -n 3 | cut -c1-400
cp -r $D/out/lora_$ns /content/xl_onl_small/out/xl_$ARM || { echo "no small adapter"; ls $D/out; exit 1; }
if [ "${PRIOR_SFT:-0}" = 1 ]; then cp -r /content/inx/xq/prior_e2b/s$s/xl_sft /content/xl_onl_small/out/xl_sft || { echo "PRIOR_SFT=1 but no stored SFT adapter for seed $s"; exit 1; }; fi   # review 209: the rank-64 reference adapter is loaded beside the rank-16 one in the small pass too
xl eval_small /content/xl_onl_small XB_SMALL=1 python xlearn.py evaluate; grep -E "evaluating|passed|EVAL_DONE" $O/eval_small.log | tail -n 4 | cut -c1-300
grep -q EVAL_DONE $O/eval_small.log || { echo "small evaluation did not finish"; exit 1; }
# the run
try train_$n $G $se ON_STREAM=lcb ON_ONL_N=$N ON_ONL_TEACH=$TEACH ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 ON_BEH_TAG=_lcb$tag python online.py online
grep -E "online eval" $O/train_$n.log | tail -n 1 | cut -c1-500
cp $D/out/online_${TEACH}_keep_replay_lcb$tag.json $O/ && cp -r $D/out/lora_$n $O/ || { echo "training outputs missing"; ls $D/out; exit 1; }   # saved before anything else runs
# the coding study's evaluation
xl_setup /content/xl_onl || { echo "evaluation setup failed"; exit 1; }
cp -r $D/out/lora_$n /content/xl_onl/out/xl_$ARM || exit 1
if [ "${PRIOR_SFT:-0}" = 1 ]; then cp -r /content/inx/xq/prior_e2b/s$s/xl_sft /content/xl_onl/out/xl_sft || exit 1   # a reference, not an arm: trained offline at rank 64 on E2B dequantized on Colab (186), evaluated here on the 4090's dequantized E2B
  echo "reference: the coding study's offline SFT adapter, seed $s (key sft in the evaluation)"; fi
xl eval /content/xl_onl python xlearn.py evaluate; grep -E "evaluating|passed|re-scored" $O/eval.log | tail -n 5 | cut -c1-300
cp /content/xl_onl/out/xl_eval_s$s.json $O/xl_eval_lcb_${ARM}_s$s.json && cp /content/xl_onl/out/xl_gens_*.jsonl $O/ || { echo "evaluation outputs missing"; ls /content/xl_onl/out; exit 1; }
echo "$(date +%H:%M) XQC_EVAL_DONE"
# 206's agent gate on the coding-taught LoRA (what it does to agent decisions); last, so a failure here loses nothing above
try gate_$n $G ON_LORA_NAME=lora_$n python online.py lora gate
tail -n 2 $O/gate_$n.log | head -n 1 | cut -c1-800
cp $D/out/gate_lora_$n.json $D/out/gate_lora_${n}_points.json $O/ || { echo "gate outputs missing"; ls $D/out; exit 1; }
echo "$(date +%H:%M) XQC_DONE"
