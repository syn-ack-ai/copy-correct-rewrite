#!/bin/bash
# The graft's bars on identical rows (user 11-07: "if the graft and smaller model work independently of the bridge, that in itself is what we
# publish"): for seed XB_SEED, the saved graft-screen forms (HF xq/g/s{SEED}/: xgraft_*_w*.pt, xlora_*_w*/) and addendum 168b's step-2 LoRAs
# (xq/x1/s{SEED}/: seq, uld, bridge, control) scored together with no bridge (xbridge.py geval): held agent / questions / fresh questions,
# the rows the forms trained on, KL from Qwen alone on untaught prose / code / stories.
#   v2/infra/colab/colab_run.sh -s xqgb -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_gbase.sh XB_SEED=1
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
B=/content/bench/transfer; O=/content/out; mkdir -p $B $O
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/*" --include "xq/*.py" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/x1/s${XB_SEED:?}/*" --include "xq/g/s${XB_SEED}/*" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
cp /content/inx/s3/*.py /content/inx/xq/*.py $B/
if [ ! -f /content/me4b_qat/config.json ]; then
  hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --local-dir /content/me4b_qat_ct > /dev/null || { echo "QAT download failed"; exit 1; }
  uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
  python $B/dequant.py /content/me4b_qat_ct /content/me4b_qat > $O/dequant_e4b.log 2>&1 && grep DEQUANT_OK $O/dequant_e4b.log || { echo "dequantize FAILED"; exit 1; }
fi
[ -f /content/mqwen/config.json ] || hf download Qwen/Qwen3-1.7B --local-dir /content/mqwen > /dev/null || { echo "Qwen download failed"; exit 1; }
[ -f /content/xb/s3_ids.npy ] || python $B/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/
export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/mqwen XB_SIZE=0.03 XB_HARM=0.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
d=/content/xgb_s$XB_SEED; mkdir -p $d/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $d/
cp -r /content/inx/xq/x1/s$XB_SEED/* /content/inx/xq/g/s$XB_SEED/* $d/out/; ls $d/out
echo "$(date +%H:%M) eval"; XB_DIR=$d python xbridge.py geval > $O/xgb_eval.log 2>&1 || { echo "eval FAILED"; tail -n 20 $O/xgb_eval.log; exit 1; }
cp $d/out/xgraft_eval_s$XB_SEED.json $O/xgb_eval_s$XB_SEED.json; grep -E "graft eval" $O/xgb_eval.log | cut -c1-1200
echo "$(date +%H:%M) XQGB_DONE"
