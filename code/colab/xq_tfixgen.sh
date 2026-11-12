#!/bin/bash
# Amendment 209q: what the corrected teacher writes (transfer/tfix_gen.py) on a Colab G4. E4B, with the corrector of 209o on (on/out209o/
# gall_e4b_self_okfit.pt: its own states read, itself written into), answers held problems greedily, two passes per token and no cache; the
# answers are run on the problems' tests. No training. The setup is xq_tfix.sh's without the fit. From on/$GENDIR: tfix_gen.py, gen_items.json.
#   v2/infra/colab/colab_run.sh -s xqyTgen -g G4 -o /content/out -d kaggle_g4agent/runs/xqyTgen kaggle_g4agent/colab/xq_tfixgen.sh GEN_MD5=.. ITEMS_SHA=.. CORR_SHA=.. MD5S=online:..,xlearn:..,xcode:..,xbridge:..,translator:..
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${MD5S:?}" "${GEN_MD5:?}" "${ITEMS_SHA:?}" "${CORR_SHA:?}"
C=/content/onl_code; D=/content/onl; O=/content/out; P=$O/gen; mkdir -p $C $D/out $O $P   # review 209q: the answers are written under /content/out, so a session that dies half way still returns them
CODEDIR=${CODEDIR:-code13}; GENDIR=${GENDIR:-gen209q}
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/s3_*" --include "xq/mix_*" --include "xq/xq_questions.jsonl" --include "on/$CODEDIR/*" --include "on/models/*" --include "on/out209o/gall_e4b_self_okfit.pt" --include "on/$GENDIR/*" \
  --include "xq/learn/xl_rows.jsonl" --include "xq/learn/xl_base.json" --include "xq/learn/xl_rows_short.jsonl" --include "xq/learn/xl_short.json" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
cp /content/inx/on/$CODEDIR/*.py $C/ && cp /content/inx/on/$GENDIR/tfix_gen.py $C/ && cp /content/inx/on/$GENDIR/gen_items.json $P/ || exit 1
for kv in ${MD5S//,/ }; do f=${kv%%:*}; [ "$(md5sum $C/$f.py | cut -c1-10)" = "${kv##*:}" ] && echo "md5 ok $f.py" || { echo "$f.py is not the reviewed $CODEDIR version"; exit 1; }; done
for f in online xlearn xcode xbridge translator; do case ",$MD5S" in *",$f:"*) ;; *) echo "no md5 given for $f.py"; exit 1;; esac; done
[ "$(md5sum $C/tfix_gen.py | cut -c1-10)" = "$GEN_MD5" ] && echo "md5 ok tfix_gen.py" || { echo "tfix_gen.py is not the reviewed version"; exit 1; }
check() { [ "$(sha256sum $1 | cut -d' ' -f1)" = "$2" ] && echo "sha ok $(basename $1)" || { echo "SHA MISMATCH $1"; exit 1; }; }
check $P/gen_items.json $ITEMS_SHA
python $C/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/ || exit 1
cp /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl $D/ || exit 1
cp /content/inx/on/out209o/gall_e4b_self_okfit.pt $D/out/gall_e4b.pt || exit 1; check $D/out/gall_e4b.pt $CORR_SHA   # 209o's corrector for E4B itself
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
BASE="E4B=/content/me4b_qat E2B=/content/me2b_qat TR_STUDENT=e4b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D ON_STREAM=lcb TR_CONTROL=gall_e4b GEN_DIR=$P"
cd $C
try() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; env $BASE "$@" > $O/$lg.log 2>&1 && return 0
        echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 20; exit 1; }
X=/content/xl_gen; rm -rf $X; mkdir -p $X/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $X/
cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $X/ || exit 1
xl() { local lg=$1; shift; echo "$(date +%H:%M) $lg"
  env XB_DIR=$X XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=1 XB_SIZE=0.03 XB_HARM=0.5 GEN_DIR=$P "$@" > $O/$lg.log 2>&1 && return 0
  echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 25; exit 1; }
xl gen_rows python tfix_gen.py rows; grep GEN_ROWS_OK $O/gen_rows.log | cut -c1-200
try gen_small $G GEN_SMALL=1 python tfix_gen.py gen; grep -E "corrector|answered|GEN_DONE" $O/gen_small.log | cut -c1-200
xl score_small GEN_SMALL=1 python tfix_gen.py score; grep GEN_SCORED $O/score_small.log | cut -c1-300
try gen $G python tfix_gen.py gen; grep -E "GEN_DONE" $O/gen.log | cut -c1-200; cp $P/gen_out*.jsonl $O/ || exit 1   # GEN_ALONE=1 (the control: the corrector off) writes gen_out_alone.jsonl
xl score python tfix_gen.py score; grep GEN_SCORED $O/score.log | cut -c1-500
cp $P/gen_scored*.jsonl $O/ || { echo "no scored output"; exit 1; }
echo "$(date +%H:%M) XQC_TGEN_DONE"
