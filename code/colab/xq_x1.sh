#!/bin/bash
# Report X1 (reports/Why LoRA loses bridge gains.md): retention accounting on the saved addendum 168b artifacts (per seed: translators at 0.03:0.5
# and the bridge / control LoRAs, from the HF dataset xq/x1/s{1,2,3}/). No training. Small pass on seed 1 first.
#   v2/infra/colab/colab_run.sh -s xqx1 -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_x1.sh
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
B=/content/bench/transfer; O=/content/out; mkdir -p $B $O
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/*" --include "xq/*" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
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
run() {  # seed, small?
  d=/content/x1_s$1$2; mkdir -p $d/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $d/; cp -r /content/inx/xq/x1/s$1/* $d/out/
  if [ -n "$2" ]; then export XB_SMALL=1; else unset XB_SMALL; fi
  echo "$(date +%H:%M) X1 seed $1$2"; XB_DIR=$d XB_SEED=$1 python xbridge.py x1 > $O/x1_s$1$2.log 2>&1 || { echo "X1 seed $1 FAILED"; tail -n 20 $O/x1_s$1$2.log; return 1; }
  cp $d/out/x1_s$1.json $O/x1_s$1$2.json; tail -n 2 $O/x1_s$1$2.log | head -n 1 | cut -c1-2500
}
run 1 _small || exit 1
for s in 1 2 3; do run $s || exit 1; done
echo "$(date +%H:%M) X1_DONE"
