#!/bin/bash
# Addendum 178 (user 11-07: "self-bridge does seem slightly better though I think"; "yes, queue it after 175"): more problems for the same LoRAs.
# The 172/174q forms trained only on agent rows and questions, never on LiveCodeBench, so every LiveCodeBench easy/medium problem is a fair test
# item. For seed XB_SEED: Qwen alone, the control LoRA, the bridge LoRA (outputs-only graft objective, rank 64), plain SFT (seq_d1000) and SFT +
# KL to self (seqself_d1000), one greedy attempt, the bridge down, on the files XC_FILES (seeds 2-3: test.jsonl-test4.jsonl, to pool with their
# 188-problem results of 174q; seed 1: all six files). Seed 1 has no outputs-only LoRAs or seqself yet: they are trained here first with the
# recipes of 169c/174q. Small pass first.
#   v2/infra/colab/colab_run.sh -s xqwN -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_wide.sh XB_SEED=N [XC_FILES=...]
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
B=/content/bench/transfer; O=/content/out; mkdir -p $B $O; S=${XB_SEED:?}
[ -n "$XC_FILES" ] || { [ $S = 1 ] && XC_FILES=test.jsonl,test2.jsonl,test3.jsonl,test4.jsonl,test5.jsonl,test6.jsonl || XC_FILES=test.jsonl,test2.jsonl,test3.jsonl,test4.jsonl; }; export XC_FILES
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/*" --include "xq/*.py" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/code/*" \
  --include "xq/x1/s$S/xb_*" --include "xq/g/s$S/xlora_*_w0/*" --include "xq/d/s$S/*" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
cp /content/inx/s3/*.py /content/inx/xq/*.py /content/inx/xq/code/*.py $B/
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 \
  --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
if [ ! -f /content/me4b_qat/config.json ]; then
  hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --local-dir /content/me4b_qat_ct > /dev/null || { echo "QAT download failed"; exit 1; }
  uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
  python $B/dequant.py /content/me4b_qat_ct /content/me4b_qat > $O/dequant_e4b.log 2>&1 && grep DEQUANT_OK $O/dequant_e4b.log || { echo "dequantize FAILED"; exit 1; }
fi
[ -f /content/mqwen/config.json ] || hf download Qwen/Qwen3-1.7B --local-dir /content/mqwen > /dev/null || { echo "Qwen download failed"; exit 1; }
[ -f /content/xb/s3_ids.npy ] || python $B/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/
export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/mqwen XC_DATA=/content/lcb XB_SIZE=0.03 XB_HARM=0.5 XB_GSTATE_W=0 XB_GONP=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
run() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; "$@" > $O/xw_$lg.log 2>&1 || { echo "$lg FAILED"; tail -n 25 $O/xw_$lg.log; exit 1; }
        grep -E "step 999|passed|done|self-check|problems" $O/xw_$lg.log | tail -n 3 | cut -c1-300; }
setup() { rm -rf $1/out; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/
          cp /content/inx/xq/x1/s$S/xb_*.pt $1/out/ || return 1; cp -r /content/inx/xq/g/s$S/xlora_*_w0 /content/inx/xq/d/s$S/xlora_* $1/out/ 2>/dev/null; export XB_DIR=$1; }
missing() {   # train the forms seed S lacks (seed 1): outputs-only graft-objective LoRAs toward the control and bridge translators, and seqself
  for t in control bridge; do [ -d $XB_DIR/out/xlora_${t}_w0 ] || run lora_$t$1 python xbridge.py graft lora $t; done
  [ -d $XB_DIR/out/xlora_seqself_d1000 ] || run seqself$1 env XB_LSTEPS=${LSTEPS:-1000} XB_LRANK=64 XB_LLAYERS=span XB_LTAG=_d1000 python xbridge.py lora seqself
}
export XB_SMALL=1 XB_SEED=$S; setup /content/xw_small || exit 1; missing _small; run fast_small env XC_ONLY=qwen,lora_c python xcode.py fast; unset XB_SMALL
setup /content/xw_s$S || exit 1; missing ""; cp -r $XB_DIR/out/xlora_*_w0 $XB_DIR/out/xlora_seqself_d1000 $O/ 2>/dev/null
run fast env XC_ONLY=qwen,lora_c,lora_b,lora_sd,lora_self python xcode.py fast
run eval python xcode.py evaluate; cp $XB_DIR/out/xcode_results.json $O/xw_results_s$S.json; mkdir -p $O/gens; cp $XB_DIR/out/gens_*.jsonl $O/gens/
python - $XB_DIR/out/xcode_results.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); print("problems", d["problems"], "passed", json.dumps(d["passed"])); print("paired", json.dumps(d["paired over all problems (a only, b only, exact p)"]))
EOF
echo "$(date +%H:%M) XQW_DONE"
