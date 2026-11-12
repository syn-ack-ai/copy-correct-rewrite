#!/bin/bash
# Amendment 209l (user 11-11: "is there a way of using the probe to 'fix' this bad behavour?"; "yes, try option 1 first"): find the direction in
# E4B's states that says "the stub comes next" (transfer/stub_dir.py find), so that it can be erased from what the cross-model corrector reads.
# Forward passes and 12-token continuations of E4B on a Colab G4; no training. The setup is xq_stubprobe.sh's (the same models, data merge and
# code10, sha256- and md5-checked). From on/$ERASEDIR: stub_dir.py, stub_probe.py (its rows step) and probe_texts.json (stub_dir.py texts).
# The whitening statistics are those of 209g's fitted cross-model corrector (on/out209g; every G4 refit has the same ones).
# Outputs: erase.pt, stub_dir.json (and the small pass's), probe_ids.json.
#   v2/infra/colab/colab_run.sh -s xqlDir -g G4 -o /content/out -d kaggle_g4agent/runs/xqlDir kaggle_g4agent/colab/xq_erase.sh MD5S=online:..,xlearn:..,xcode:..,xbridge:..,translator:.. DIR_MD5=.. PROBE_MD5=.. TEXTS_SHA=.. STATS_SHA=..
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${MD5S:?}" "${DIR_MD5:?}" "${PROBE_MD5:?}" "${TEXTS_SHA:?}" "${STATS_SHA:?}"
C=/content/onl_code; D=/content/onl; O=/content/out; P=/content/probe; mkdir -p $C $D/out $O $P
ERASEDIR=${ERASEDIR:-erase209l}; CODEDIR=${CODEDIR:-code10}
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/s3_*" --include "xq/mix_*" --include "xq/xq_questions.jsonl" --include "on/$CODEDIR/*" --include "on/out/*" --include "on/out209g/gall_e4b_okfit.pt" --include "on/models/*" --include "on/$ERASEDIR/*" \
  --include "xq/learn/xl_rows.jsonl" --include "xq/learn/xl_base.json" --include "xq/learn/xl_rows_short.jsonl" --include "xq/learn/xl_short.json" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
cp /content/inx/on/$CODEDIR/*.py $C/ && cp /content/inx/on/$ERASEDIR/stub_dir.py /content/inx/on/$ERASEDIR/stub_probe.py $C/ && cp /content/inx/on/$ERASEDIR/probe_texts.json $P/ || exit 1
for kv in ${MD5S//,/ }; do f=${kv%%:*}; [ "$(md5sum $C/$f.py | cut -c1-10)" = "${kv##*:}" ] && echo "md5 ok $f.py" || { echo "$f.py is not the reviewed version"; exit 1; }; done
for f in online xlearn xcode xbridge translator; do case ",$MD5S" in *",$f:"*) ;; *) echo "no md5 given for $f.py"; exit 1;; esac; done
[ "$(md5sum $C/stub_dir.py | cut -c1-10)" = "$DIR_MD5" ] && echo "md5 ok stub_dir.py" || { echo "stub_dir.py is not the reviewed version"; exit 1; }
[ "$(md5sum $C/stub_probe.py | cut -c1-10)" = "$PROBE_MD5" ] && echo "md5 ok stub_probe.py" || { echo "stub_probe.py is not the reviewed version"; exit 1; }
check() { [ "$(sha256sum $1 | cut -d' ' -f1)" = "$2" ] && echo "sha ok $(basename $1)" || { echo "SHA MISMATCH $1"; exit 1; }; }
check $P/probe_texts.json $TEXTS_SHA
python $C/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/ || exit 1
check /content/xb/s3_ids.npy 84cf0f5b1ff41cb1c12e64079a51fb25a90bba1a15aa2de2d669aa290edb9043
check /content/xb/s3_calls.jsonl 2df5f431e5dbcd8468ffb1fce5798d7cbb10a0d3422e8405477223ef604bd8cd
cp /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl $D/ || exit 1
cp /content/inx/on/out209g/gall_e4b_okfit.pt $D/out/gall_e4b_stats.pt || exit 1
check $D/out/gall_e4b_stats.pt $STATS_SHA                                 # 209g's fitted cross-model corrector: only its whitening statistics are read
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
# the teaching environment of xq_lcbv.sh (the corrector's source layers: TR_SRC_DEPTHS)
BASE="E4B=/content/me4b_qat E2B=/content/me2b_qat TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=512 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D ON_STREAM=lcb PROBE_DIR=$P DIR_STATS=$D/out/gall_e4b_stats.pt"
cd $C
try() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; env $BASE "$@" > $O/$lg.log 2>&1 && return 0
        echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 25; exit 1; }
# the prompts, as the coding study builds them (the evaluation environment; no model is loaded)
X=/content/xl_probe; rm -rf $X; mkdir -p $X/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $X/
cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $X/ || exit 1
echo "$(date +%H:%M) probe_rows"
env XB_DIR=$X XB_TEACHER=/content/me4b_qat XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XL_NO_KEPT_POOL=1 XC_DATA=/content/lcb XB_SEED=1 XB_SIZE=0.03 XB_HARM=0.5 PROBE_DIR=$P python stub_probe.py rows > $O/probe_rows.log 2>&1 \
  || { echo "probe_rows FAILED"; grep -vE "Warning|warn" $O/probe_rows.log | tail -n 25; exit 1; }
grep -E "left out|PROBE_ROWS_OK" $O/probe_rows.log | cut -c1-400; cp $P/probe_ids.json $O/ || exit 1
try dir_small $G PROBE_SMALL=1 python stub_dir.py find; grep -E "labels|direction|cause|STUB_DIR_DONE" $O/dir_small.log | cut -c1-500
cp $P/stub_dir_small.json $O/ || exit 1
try dir $G python stub_dir.py find; grep -E "labels|direction|cause|STUB_DIR_DONE" $O/dir.log | cut -c1-900
cp $P/stub_dir.json $P/erase.pt $O/ || { echo "no direction output"; exit 1; }; sha256sum $O/erase.pt
echo "$(date +%H:%M) XQC_DIR_DONE"
