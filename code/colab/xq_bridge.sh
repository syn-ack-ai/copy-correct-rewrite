#!/bin/bash
# Addendum 166: the cross-tokenizer bridge, Gemma 4 E4B (4-bit QAT, dequantized; teacher) -> Qwen3-1.7B (student), transfer/xbridge.py.
# Small pass first (20 steps each, the suite on 3 rows per kind; SMALL_ONLY=1 stops), then the bridge and control translators (XB_STEPS, 600)
# and the genericity suite. Inputs from the private HF dataset (s3/: agent rows, dequant.py; xq/: xbridge.py, mix_merge.py, the mix, the
# plain questions).
#   v2/infra/colab/colab_run.sh -s xq -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_bridge.sh [XB_SIZE=..] [SMALL_ONLY=1]
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
for d in /content/xb /content/xbsm; do
  [ -f $d/s3_ids.npy ] || python $B/mix_merge.py /content/inx/s3 /content/inx/xq $d || { echo "merge failed"; exit 1; }
  cp /content/inx/xq/xq_questions.jsonl $d/
done
export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/mqwen PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "configuration: XB_SIZE=${XB_SIZE:-0.1} XB_HARM=${XB_HARM:-0.5} XB_STEPS=${XB_STEPS:-600}"
cd $B
phase() {
  if [ $1 = small ]; then export XB_SMALL=1 XB_DIR=/content/xbsm; T=_small; else unset XB_SMALL; export XB_DIR=/content/xb; T=""; fi
  for s in bridge control; do
    echo "$(date +%H:%M) $1: train $s"; python xbridge.py train $s > $O/xb_train_$s$T.log 2>&1 || { echo "train FAILED"; tail -n 25 $O/xb_train_$s$T.log; return 1; }
    grep -E "teacher layers|rows|step [0-9]+ " $O/xb_train_$s$T.log | awk 'NR <= 3 || NR % 4 == 0' | tail -n 7 | cut -c1-300
  done
  echo "$(date +%H:%M) $1: suite"; python xbridge.py suite > $O/xb_suite$T.log 2>&1 || { echo "suite FAILED"; tail -n 25 $O/xb_suite$T.log; return 1; }
  cp $XB_DIR/out/xb_suite.json $O/xb_suite$T.json; cp $XB_DIR/out/xb_bridge.pt $O/xb_bridge$T.pt; cp $XB_DIR/out/xb_control.pt $O/xb_control$T.pt
  tail -n 2 $O/xb_suite$T.log | head -n 1 | cut -c1-3000
}
echo "$(date +%H:%M) small test"; phase small || exit 1
[ "${SMALL_ONLY:-0}" = 1 ] && { echo "SMALL_DONE"; exit 0; }
echo "$(date +%H:%M) full run"; phase full || exit 1
echo "$(date +%H:%M) XQ_DONE"
