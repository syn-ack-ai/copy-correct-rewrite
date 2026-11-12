#!/bin/bash
# Amendment 209o (user 11-11: "we will have to figure out how to use the corrector to correct the teacher model as well, if its doing something
# bad"): the teacher's own corrector, fitted on the clean text. The same fit as xq_lcbv.sh STAGE=fit (the same rows, weights, steps, smoke
# test, verified-token targets) with E4B as the model that is written into: TR_STUDENT=e4b TR_TEACHERS=e4b, the corrector gall_e4b reads
# E4B's own states (50% and 65% depth) and writes into E4B (its 35% layer and every later one). Where E4B's top token is the verified text's
# next token the target is E4B's own distribution (nothing to learn); where it is not, the target is that token. Then, in the same session,
# stub_probe.py reads the stub's probability after the prompt and the signature under E4B and under E4B with its corrector, on the 404
# LeetCode-format problems of 209l (on/erase209l/probe_texts.json). No student is taught here.
#   v2/infra/colab/colab_run.sh -s xqyTfix -g G4 -o /content/out -d kaggle_g4agent/runs/xqyTfix kaggle_g4agent/colab/xq_tfix.sh FIT_W=.. LCB_FIT_SHA=.. PROBE_MD5=.. TEXTS_SHA=.. MD5S=online:..,xlearn:..,lcb_rows:..,xcode:..,xbridge:..,translator:..,lcb_own:..
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${MD5S:?}" "${FIT_W:?}" "${LCB_FIT_SHA:?}" "${PROBE_MD5:?}" "${TEXTS_SHA:?}"
C=/content/onl_code; D=/content/onl; O=/content/out; P=/content/probe; mkdir -p $C $D/out $O $P
CODEDIR=${CODEDIR:-code13}; OWNDIR=${OWNDIR:-own209i}; STEPS=${STEPS:-2400}
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/s3_*" --include "xq/mix_*" --include "xq/xq_questions.jsonl" --include "on/$CODEDIR/*" --include "on/models/*" --include "on/$OWNDIR/*" --include "on/erase209l/stub_probe.py" --include "on/erase209l/probe_texts.json" \
  --include "xq/learn/xl_rows.jsonl" --include "xq/learn/xl_base.json" --include "xq/learn/xl_rows_short.jsonl" --include "xq/learn/xl_short.json" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
cp /content/inx/on/$CODEDIR/*.py $C/ && cp /content/inx/on/erase209l/stub_probe.py $C/ && cp /content/inx/on/erase209l/probe_texts.json $P/ || exit 1
for kv in ${MD5S//,/ }; do f=${kv%%:*}; [ "$(md5sum $C/$f.py | cut -c1-10)" = "${kv##*:}" ] && echo "md5 ok $f.py" || { echo "$f.py is not the reviewed $CODEDIR version"; exit 1; }; done
for f in online xlearn lcb_rows xcode xbridge translator lcb_own; do case ",$MD5S" in *",$f:"*) ;; *) echo "no md5 given for $f.py"; exit 1;; esac; done
[ "$(md5sum $C/stub_probe.py | cut -c1-10)" = "$PROBE_MD5" ] && echo "md5 ok stub_probe.py" || { echo "stub_probe.py is not the reviewed version"; exit 1; }
check() { [ "$(sha256sum $1 | cut -d' ' -f1)" = "$2" ] && echo "sha ok $(basename $1)" || { echo "SHA MISMATCH $1"; exit 1; }; }
check $P/probe_texts.json $TEXTS_SHA
python $C/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/ || exit 1
check /content/xb/s3_ids.npy 84cf0f5b1ff41cb1c12e64079a51fb25a90bba1a15aa2de2d669aa290edb9043
check /content/xb/s3_calls.jsonl 2df5f431e5dbcd8468ffb1fce5798d7cbb10a0d3422e8405477223ef604bd8cd
cp /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl $D/ || exit 1
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
# E4B is both the model written into and the model whose distribution is the target; samples as in the teaching environment
BASE="E4B=/content/me4b_qat E2B=/content/me2b_qat TR_STUDENT=e4b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=512 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cd $C
try() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; env $BASE "$@" > $O/$lg.log 2>&1 && return 0
        echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 20; exit 1; }
# the coding study's evaluation environment (prompts, tests, the tokenizer), as xq_lcbv.sh; E2B's path is its tokenizer's only
xl_setup() { rm -rf $1; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/
  cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/; }
xl() { local lg=$1 dir=$2; shift 2; echo "$(date +%H:%M) $lg"
  env XB_DIR=$dir XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=1 XB_SIZE=0.03 XB_HARM=0.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" > $O/$lg.log 2>&1 && return 0
  echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 25; exit 1; }
xl_setup /content/xl_onl_small || { echo "evaluation setup failed"; exit 1; }
xl lcb_rows /content/xl_onl_small python lcb_rows.py $D; grep -E "lcb rows|decode back|LCB_ROWS_OK" $O/lcb_rows.log | cut -c1-300
grep -q LCB_ROWS_OK $O/lcb_rows.log || { echo "no lcb rows"; exit 1; }
xl fit_rows /content/xl_onl_small LCB_FIT=/content/inx/on/$OWNDIR/xl_rows_fit.jsonl LCB_FIT_SHA=$LCB_FIT_SHA python lcb_own.py rows $D; grep -E "left out|FIT_ROWS_OK" $O/fit_rows.log | cut -c1-400
grep -q FIT_ROWS_OK $O/fit_rows.log || { echo "no fit rows"; exit 1; }
echo "$(date +%H:%M) data_check"   # the smoke test: nothing is fitted unless every coding row passes it
env XB_DIR=/content/xl_onl_small XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=1 FIT_W=$FIT_W python lcb_own.py check $D > $O/data_check.log 2>&1; rc=$?
grep -vE "Warning|warn" $O/data_check.log | tail -n 12 | cut -c1-400; cp $D/out/lcb_data_check.json $O/ 2> /dev/null
[ $rc = 0 ] && grep -q DATA_CHECK_OK $O/data_check.log || { echo "DATA_CHECK_FAILED: nothing is fitted"; exit 1; }
N0=$(python -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['split'] == 'train'))" /content/xb/s3_calls.jsonl) || exit 1; [ "$N0" = 3295 ] || { echo "the 1x fitting pool has $N0 rows, not 3295"; exit 1; }
FT="TR_SRC_DEPTHS=0.5,0.65 ON_TRMIX=1 ON_TR_SIZE=0 ON_CAL_POOL=$N0 ON_DIR=$D ON_TR_W=$FIT_W ON_FIT_HARD=1"   # 209i's fit settings
try fit_small_e4b $FT ON_SMALL=1 python online.py train gall_e4b; grep -E "translator gall|step 0 |step 19 |peak memory" $O/fit_small_e4b.log | cut -c1-330
try fit_e4b $FT ON_STEPS=$STEPS python online.py train gall_e4b; grep -E "translator gall|step $((STEPS - 1)) " $O/fit_e4b.log | cut -c1-400
grep "verified-token targets so far" $O/fit_e4b.log | tail -n 1 | grep -qE "so far: [1-9]" || { echo "the fit did not use the verified-token targets"; exit 1; }
grep "verified-token targets so far" $O/fit_e4b.log | tail -n 1 | cut -c1-200
cp $D/out/gall_e4b.pt $O/gall_e4b_self_okfit.pt || { echo "no fitted corrector"; exit 1; }; sha256sum $O/gall_e4b_self_okfit.pt
# the stub under E4B and under E4B with its own corrector (stub_probe.py: the model written into is listed as "E2B alone", here E4B; "self teacher gall_e4b" is E4B corrected)
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D ON_STREAM=lcb TR_CONTROL=gall_e4b PROBE_DIR=$P PROBE_CORR= PROBE_CTRL=gall_e4b PROBE_LORAS="   # no samples are drawn (none of these problems is a row problem)
X=/content/xl_probe; xl_setup $X || exit 1
xl probe_rows $X PROBE_DIR=$P python stub_probe.py rows; grep -E "left out|PROBE_ROWS_OK" $O/probe_rows.log | cut -c1-400; cp $P/probe_ids.json $O/ || exit 1
try probe_small $G PROBE_SMALL=1 python stub_probe.py probe; grep -E "corrector|student|samples|STUB_PROBE_DONE" $O/probe_small.log | cut -c1-300; cp $P/stub_probe_small.json $O/ 2> /dev/null
try probe $G python stub_probe.py probe; grep -E "corrector|student|samples|STUB_PROBE_DONE" $O/probe.log | cut -c1-300
cp $P/stub_probe.json $O/ || { echo "no probe output"; exit 1; }
python - $O/stub_probe.json <<'PY' || { echo "the probe's check failed"; exit 1; }   # review 209o: the corrector must change the numbers, and the two copies of E4B must agree
import json, sys
P = json.load(open(sys.argv[1]))["problems"]; d = lambda a, b: max(abs(x - y) for p in P for x, y in zip(p["writers"][a]["logp"], p["writers"][b]["logp"]))
live, same = d("self teacher gall_e4b", "E2B alone"), d("E4B", "E2B alone"); print("the corrector changes a stub token's log-probability by at most", round(live, 4), "| the two copies of E4B differ by at most", round(same, 5))
sys.exit(0 if live > 1e-3 and same < 1e-3 else 1)
PY
echo "$(date +%H:%M) XQC_TFIX_DONE"
