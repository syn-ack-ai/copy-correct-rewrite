#!/bin/bash
# Addendum 176 (user 11-07: "if the translator actually does add something, then a better trained translator might work a lot better"; "yes,
# queue the curve if 175 shows a gain"; widened, 17:20: "yes, widen 176 with both translators", run whatever 175 shows): a training-length curve for
# the control translator (Qwen's own states: the self-bridge) and the bridge translator (E4B's states), TRS="control bridge". For seed XB_SEED
# each 600-step translator of 168b is continued in rounds of 300 steps with 175's recipe (lr 1e-4, agent and question rows clean, fresh rows per
# round via XB_ROUND, no graft); after 1,200, 2,400 and 4,800 total steps a rank-64 LoRA is trained toward it (the outputs-only graft objective,
# 1,000 steps, fixed rows, as 172's LoRAs). Scored with Qwen alone and the original control and bridge LoRAs (600 steps) on the 188 LiveCodeBench problems of
# 172 (one greedy attempt, the bridge down) and on X1's held rows and untaught prose (geval). Small pass first.
#   v2/infra/colab/colab_run.sh -s xqcvN -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_curve.sh XB_SEED=N
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
B=/content/bench/transfer; O=/content/out; mkdir -p $B $O
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/*" --include "xq/*.py" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/code/*" \
  --include "xq/x1/s${XB_SEED:?}/xb_*" --include "xq/g/s$XB_SEED/xlora_control_w0/*" --include "xq/g/s$XB_SEED/xlora_bridge_w0/*" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
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
export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/mqwen XC_DATA=/content/lcb XB_SIZE=0.03 XB_HARM=0.5 XB_GSTATE_W=0 XB_GONP=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
run() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; "$@" > $O/xc_$lg.log 2>&1 || { echo "$lg FAILED"; tail -n 25 $O/xc_$lg.log; exit 1; }
        grep -E "(control|bridge) step (19|299) |lora (control|bridge) step 999|passed|done|self-check|graft eval held agent" $O/xc_$lg.log | tail -n 3 | cut -c1-300; }
setup() { rm -rf $1/out; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/
          cp /content/inx/xq/x1/s$XB_SEED/xb_*.pt $1/out/ && cp -r /content/inx/xq/g/s$XB_SEED/xlora_control_w0 /content/inx/xq/g/s$XB_SEED/xlora_bridge_w0 $1/out/ || return 1; export XB_DIR=$1; }
curve() {   # $1 = log suffix, $2 = rounds at which a LoRA is made (total steps = 600 + 300 x round), $3 = last round; every translator in TRS
  for t in ${TRS:-control bridge}; do
    prev=xb_$t
    for k in $(seq 1 $3); do
      run ${t}_k$k$1 env XB_ROUND=$k XB_CLEAN=1 XB_TR_INIT=$prev XB_TR_OUT=xb_${t}_k$k XB_STEPS=${STEPS:-300} XB_LR=1e-4 python xbridge.py train $t
      [ $prev != xb_$t ] && rm -f $XB_DIR/out/$prev.pt; prev=xb_${t}_k$k
      if [[ " $2 " == *" $k "* ]]; then
        L=$((600 + 300 * k)); cp $XB_DIR/out/$prev.pt $XB_DIR/out/xb_${t}_L$L.pt
        run lora_${t}_L$L$1 env XB_TRNAME=xb_${t}_L$L XB_GTAG=_L$L python xbridge.py graft lora $t
      fi
    done
    rm -f $XB_DIR/out/$prev.pt
  done
}
export XB_SMALL=1 STEPS=20; setup /content/xcv_small || exit 1; curve _small "2" 2; unset XB_SMALL STEPS
setup /content/xcv_s$XB_SEED || exit 1; curve "" "2 6 14" 14
cp $XB_DIR/out/xb_*_L*.pt $O/; cp -r $XB_DIR/out/xlora_*_w0_L* $O/
run fast env XC_ONLY=qwen,lora_c,lora_c_L1200,lora_c_L2400,lora_c_L4800,lora_b,lora_b_L1200,lora_b_L2400,lora_b_L4800 python xcode.py fast
run eval python xcode.py evaluate; cp $XB_DIR/out/xcode_results.json $O/xcv_code_results_s$XB_SEED.json; mkdir -p $O/gens; cp $XB_DIR/out/gens_*.jsonl $O/gens/
python - $XB_DIR/out/xcode_results.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); print("passed", json.dumps(d["passed"])); print("paired", json.dumps(d["paired over all problems (a only, b only, exact p)"]))
EOF
run geval python xbridge.py geval; cp $XB_DIR/out/xgraft_eval_s$XB_SEED.json $O/xcv_geval_s$XB_SEED.json
echo "$(date +%H:%M) XQCV_DONE"
