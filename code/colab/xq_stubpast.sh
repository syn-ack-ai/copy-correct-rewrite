#!/bin/bash
# Amendment 209m: what follows the stub (transfer/stub_past.py) on a Colab G4. E4B, E2B and E2B with a student's adapter each continue their own
# stored stub answers past the closing fence; a second fenced block is run on the problem's tests. No training. The setup is xq_lcbfix.sh's.
# Inputs from on/$PASTDIR: stub_past.py and past_items.jsonl; adapters from on/probe209h/loras. Output past_out.jsonl.
#   v2/infra/colab/colab_run.sh -s xqyPast -g G4 -o /content/out -d kaggle_g4agent/runs/xqyPast kaggle_g4agent/colab/xq_stubpast.sh MD5S=xlearn:..,xcode:..,xbridge:.. PAST_MD5=.. ITEMS_SHA=.. LORAS=d4v_s1
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${MD5S:?}" "${PAST_MD5:?}" "${ITEMS_SHA:?}" "${LORAS:?}"
PASTDIR=${PASTDIR:-past209m}; C=/content/onl_code; O=/content/out; X=/content/xl_past; mkdir -p $C $O $X/out
hf download SkyPanther/g4agent-job --repo-type dataset --include "on/code10/*" --include "on/models/*" --include "on/$PASTDIR/*" --include "on/probe209h/loras/*" \
  --include "xq/learn/xl_rows.jsonl" --include "xq/learn/xl_base.json" --include "xq/learn/xl_rows_short.jsonl" --include "xq/learn/xl_short.json" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
cp /content/inx/on/code10/*.py $C/ && cp /content/inx/on/$PASTDIR/stub_past.py $C/ && cp /content/inx/on/$PASTDIR/past_items.jsonl $X/ || exit 1
for kv in ${MD5S//,/ }; do f=${kv%%:*}; [ "$(md5sum $C/$f.py | cut -c1-10)" = "${kv##*:}" ] && echo "md5 ok $f.py" || { echo "$f.py is not the reviewed code10 version"; exit 1; }; done
for f in xlearn xcode xbridge; do case ",$MD5S" in *",$f:"*) ;; *) echo "no md5 given for $f.py"; exit 1;; esac; done
[ "$(md5sum $C/stub_past.py | cut -c1-10)" = "$PAST_MD5" ] && echo "md5 ok stub_past.py" || { echo "stub_past.py is not the reviewed version"; exit 1; }
check() { [ "$(sha256sum $1 | cut -d' ' -f1)" = "$2" ] && echo "sha ok $(basename $1)" || { echo "SHA MISMATCH $1"; exit 1; }; }
check $X/past_items.jsonl $ITEMS_SHA
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $X/ || exit 1
PL=""; for a in ${LORAS//,/ }; do d=/content/inx/on/probe209h/loras/$a; [ -f $d/adapter_model.safetensors ] && [ -f $d/adapter_config.json ] || { echo "adapter $a missing"; exit 1; }; sha256sum $d/adapter_model.safetensors; PL="$PL,$a=$d"; done; PL=${PL#,}
cd $C
run() { local lg=$1; shift; echo "$(date +%H:%M) $lg"
  env XB_DIR=$X XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/me2b_qat "XB_SBOS=<bos>" XC_DATA=/content/lcb STUB_LORAS=$PL PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" > $O/$lg.log 2>&1 && return 0
  echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 25; exit 1; }
run past_small STUB_SMALL=1 python stub_past.py $X/past_items.jsonl $O/past_out_small.jsonl; grep -E "stub answers|STUB_PAST_DONE" $O/past_small.log | cut -c1-300
run past python stub_past.py $X/past_items.jsonl $O/past_out.jsonl; grep -E "stub answers|STUB_PAST_DONE" $O/past.log | cut -c1-300
[ -s $O/past_out.jsonl ] || { echo "no output"; exit 1; }
echo "$(date +%H:%M) XQC_PAST_DONE"
