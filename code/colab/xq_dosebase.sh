#!/bin/bash
# Dose-matched baselines for bar 3 (user 11-07: "yes, add the dose-matched baselines after the replication"): for seed XB_SEED, two plain
# distillation LoRAs in the same form and dose as the rank-64 graft-objective LoRA (rank 64, MLP at layers 9-27, 1,000 steps x 2 rows, the
# same 70/30 agent/question rows): seq_d1000 (SFT on E4B's reply tokens) and seqdense_d1000 (next-token loss over the whole written span, no
# teacher); then the eval with that seed's graft-objective forms (HF xq/g/s{SEED}) and step-2 LoRAs (xq/x1/s{SEED}) on identical rows.
#   v2/infra/colab/colab_run.sh -s xqdN -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_dosebase.sh XB_SEED=N
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
d=/content/xdb_s$XB_SEED; mkdir -p $d/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $d/
cp -r /content/inx/xq/x1/s$XB_SEED/* /content/inx/xq/g/s$XB_SEED/* $d/out/; export XB_DIR=$d
for a in seq seqdense; do   # small pass first, then the dose-matched run
  echo "$(date +%H:%M) $a small"; XB_SMALL=1 XB_LRANK=64 XB_LLAYERS=span XB_LTAG=_small python xbridge.py lora $a > $O/xdb_${a}_small.log 2>&1 || { echo "$a small FAILED"; tail -n 20 $O/xdb_${a}_small.log; exit 1; }
  rm -rf $d/out/xlora_${a}_small
  echo "$(date +%H:%M) $a 1000 steps"; XB_LSTEPS=1000 XB_LRANK=64 XB_LLAYERS=span XB_LTAG=_d1000 python xbridge.py lora $a > $O/xdb_$a.log 2>&1 || { echo "$a FAILED"; tail -n 20 $O/xdb_$a.log; exit 1; }
  grep -E "lora $a step" $O/xdb_$a.log | tail -n 1 | cut -c1-200; cp -r $d/out/xlora_${a}_d1000 $O/
done
echo "$(date +%H:%M) eval"; python xbridge.py geval > $O/xdb_eval.log 2>&1 || { echo "eval FAILED"; tail -n 20 $O/xdb_eval.log; exit 1; }
cp $d/out/xgraft_eval_s$XB_SEED.json $O/xdb_eval_s$XB_SEED.json; grep -E "graft eval held agent" $O/xdb_eval.log | cut -c1-1500
echo "$(date +%H:%M) XQD_DONE"
