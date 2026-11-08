#!/bin/bash
# Addendum 174q (user 11-07: "can we test if the translator is actually a missing piece we need sooner?"): for each seed in SEEDS, a rank-64 LoRA
# trained like the dose-matched baselines (MLP at layers 9-27, 1,000 steps x 2 rows, the same 70/30 agent/question rows) with SFT on E4B's
# reply + 1.0 x KL(original Qwen || student) over the written span: no translator (xbridge.py lora seqself). Then LiveCodeBench (the 188
# problems of addendum 172, one greedy attempt, the bridge down): Qwen alone, the control LoRA (the translator path), the bridge LoRA, plain
# SFT at the same dose, and the new LoRA. Small pass first.
#   v2/infra/colab/colab_run.sh -s xqs -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_selftest.sh [SEEDS="2 3"]
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
SEEDS=${SEEDS:-2,3}; B=/content/bench/transfer; O=/content/out; mkdir -p $B $O
INC=(--include "s3/*" --include "xq/*.py" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/code/*")
for s in ${SEEDS//,/ }; do INC+=(--include "xq/g/s$s/xlora_*_w0/*" --include "xq/d/s$s/xlora_seq_d1000/*"); done
hf download SkyPanther/g4agent-job --repo-type dataset "${INC[@]}" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
cp /content/inx/s3/*.py /content/inx/xq/*.py /content/inx/xq/code/*.py $B/
[ -f /content/lcb/test6.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 \
  --include test5.jsonl --include test6.jsonl --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
if [ ! -f /content/me4b_qat/config.json ]; then
  hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --local-dir /content/me4b_qat_ct > /dev/null || { echo "QAT download failed"; exit 1; }
  uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
  python $B/dequant.py /content/me4b_qat_ct /content/me4b_qat > $O/dequant_e4b.log 2>&1 && grep DEQUANT_OK $O/dequant_e4b.log || { echo "dequantize FAILED"; exit 1; }
fi
[ -f /content/mqwen/config.json ] || hf download Qwen/Qwen3-1.7B --local-dir /content/mqwen > /dev/null || { echo "Qwen download failed"; exit 1; }
[ -f /content/xb/s3_ids.npy ] || python $B/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/
export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/mqwen XC_DATA=/content/lcb XB_SIZE=0.03 XB_HARM=0.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
run() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; "$@" > $O/xs_$lg.log 2>&1 || { echo "$lg FAILED"; tail -n 25 $O/xs_$lg.log; exit 1; }
        grep -E "lora seqself step|passed|done|self-check" $O/xs_$lg.log | tail -n 8 | cut -c1-300; }
for s in ${SEEDS//,/ }; do
  export XB_SEED=$s; d=/content/xs_s$s; rm -rf $d/out; mkdir -p $d/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $d/
  cp -r /content/inx/xq/g/s$s/xlora_*_w0 /content/inx/xq/d/s$s/xlora_seq_d1000 $d/out/ || exit 1; export XB_DIR=$d
  run s${s}_small env XB_SMALL=1 XB_LRANK=64 XB_LLAYERS=span XB_LTAG=_small python xbridge.py lora seqself; rm -rf $d/out/xlora_seqself_small
  run s${s}_train env XB_LSTEPS=1000 XB_LRANK=64 XB_LLAYERS=span XB_LTAG=_d1000 python xbridge.py lora seqself; cp -r $d/out/xlora_seqself_d1000 $O/xlora_seqself_d1000_s$s
  run s${s}_fast env XC_ONLY=qwen,lora_c,lora_b,lora_sd,lora_self python xcode.py fast
  run s${s}_eval python xcode.py evaluate; cp $d/out/xcode_results.json $O/xs_results_s$s.json; mkdir -p $O/gens_s$s; cp $d/out/gens_*.jsonl $O/gens_s$s/
  python - $d/out/xcode_results.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); print("passed", json.dumps(d["passed"])); print("paired", json.dumps(d["paired over all problems (a only, b only, exact p)"]))
EOF
done
echo "$(date +%H:%M) XQS_DONE"
