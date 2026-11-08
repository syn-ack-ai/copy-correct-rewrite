#!/bin/bash
# Addendum 184 on the 4090 (user 11-08: "que the translator training on the 4090"): after the inputs (PREP184_DONE) and the RSI pilot
# (RSI4090_DONE), with the GPU free twice: per seed 1-3, the keep rows (build_keep), the cross-fitted selective translator (fit_xk; arm xk only on
# XK_PASS), the arms sft_s and sft_c_s (anchors, retrained here: this machine's numbers are compared with each other only), sft_k, sft_ck, xk,
# rb and rb_c (184b), then the evaluation (fixed / new / kept, the fragile set, off-task). Qwen3-1.7B; no E4B weights (XB_TEACHER_TOK_ONLY: its tokenizer and config
# only); every hidden test in the bubblewrap sandbox. A small pass first (every step, xk trained whatever its gate). Log ~/x184/logs/x184.log;
# X184_DONE (or X184_HALTED).
X=~/x184; B=$X/bench; LG=$X/logs; L=$LG/x184.log; P=~/toyq/.pt/bin/python; mkdir -p $LG $X/out; log() { echo "$(date +%H:%M) $*" >> $L; }
room() { local a=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$a" -ge ${1:-2} ] || { log "only ${a} GB free: X184_HALTED"; exit 1; }; }   # review 11-08: 2 GB per seed
log "waiting for the inputs (PREP184_DONE) and the RSI pilot (RSI4090_DONE)"
until grep -q PREP184_DONE $LG/prep.log 2>/dev/null; do grep -qE "FAILED|stopping" $LG/prep.log 2>/dev/null && { log "prep failed: X184_HALTED"; exit 1; }; sleep 120; done
for f in xlearn.py xbridge.py xcode.py; do [ -f $B/$f ] || { log "no $B/$f: X184_HALTED"; exit 1; }; done; [ -f $X/inx/xl_fragile.json ] || { log "no fragile set: X184_HALTED"; exit 1; }
until grep -q RSI4090_DONE ~/rsi4090/logs/rsi.log 2>/dev/null || ! pgrep -f "bash rsi4090_run.sh" > /dev/null; do sleep 300; done   # review 11-08: the pilot can end without its marker
grep -q RSI4090_DONE ~/rsi4090/logs/rsi.log 2>/dev/null && log "RSI pilot done" || log "the RSI pilot runner ended without RSI4090_DONE (see ~/rsi4090/logs/rsi.log); going on once the GPU is free"
free=0; until [ $free -ge 2 ]; do   # no other GPU job (online test, xlearn, vLLM) and under 2 GB used, twice in a row
  if ! pgrep -f "bin/python[0-9.]* (online|xlearn)\.py" > /dev/null && ! pgrep -f "[v]llm serve" > /dev/null && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
GT=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/6cd26aa*/ | head -1); [ -f $GT/tokenizer.json ] || { log "no Gemma tokenizer: X184_HALTED"; exit 1; }
export XB_TEACHER=$GT XB_TEACHER_TOK_ONLY=1 XB_STUDENT=$HOME/rsi4090/mqwen XC_DATA=$X/lcb XB_SIZE=0.03 XB_HARM=0.5 XC_SANDBOX=bwrap PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $B || exit 1; log "start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1) | free disk $(df -h --output=avail / | tail -1)"
setup() {   # $1 folder, $2 seed: a clean out/ with the seed's control translator, the shared rows, baselines, generations, short rows, fragile set
  rm -rf $1/out; mkdir -p $1/out; ln -sf ~/rsi4090/xb/s3_ids.npy ~/rsi4090/xb/s3_calls.jsonl ~/rsi4090/xb/xq_questions.jsonl $1/
  cp $X/inx/xq/x1/s$2/xb_control.pt $1/out/ && cp $X/inx/xq/learn/xl_rows.jsonl $X/inx/xq/learn/xl_base.json $X/inx/xq/learn/xl_rows_short.jsonl $X/inx/xq/learn/xl_short.json $X/inx/xq/learn/xl_build_gens.jsonl $X/inx/xq/learn/xl_rows_rb.jsonl $X/inx/xq/learn/xl_rb.json $X/inx/xl_fragile.json $1/ || return 1
  export XB_DIR=$1 XB_SEED=$2; }
run() { local lg=$1; shift; log "$lg"; $P "$@" > $LG/$lg.log 2>&1 || { log "$lg FAILED: $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
        grep -E "keep rows:|cross-fit gate|^XK_|passed: trained|TRAIN_DONE" $LG/$lg.log | tail -n 2 | cut -c1-900 >> $L; }
ARMS="sft_s sft_c_s sft_k sft_ck rb rb_c"; ok=1; room 3   # 184b: rb (the anchor) and rb_c
export XB_SMALL=1; setup $X/small 1 || { log "small setup failed: X184_HALTED"; exit 1; }
run small_keep xlearn.py build_keep || ok=0
[ $ok = 1 ] && { run small_xk xlearn.py fit_xk || ok=0; }
for a in $ARMS xk; do [ $ok = 1 ] && { run small_train_$a xlearn.py train $a || ok=0; }; done
[ $ok = 1 ] && { run small_eval xlearn.py evaluate || ok=0; }
[ $ok = 1 ] || { log "small pass failed: X184_HALTED"; exit 1; }
unset XB_SMALL; rm -rf $X/small; log "small pass ok"
for s in 1 2 3; do
  room; setup $X/s$s $s || { log "seed $s: setup failed"; continue; }
  run keep_s$s xlearn.py build_keep || continue
  A="$ARMS"
  if run xk_s$s xlearn.py fit_xk && grep -q "^XK_PASS" $LG/xk_s$s.log; then A="$A xk"; else log "seed $s: arm xk skipped (XK_FAIL or the fit failed)"; fi
  cp $X/s$s/xl_xk.json $X/out/xl_xk_s$s.json 2>/dev/null
  nt=0; for a in $A; do run train_${a}_s$s xlearn.py train $a && nt=$((nt + 1)) || log "seed $s: arm $a failed; the others are evaluated"; done   # review 11-08
  rm -f $X/s$s/out/xb_control_xk0.pt $X/s$s/out/xb_control_xk1.pt   # disk (review 11-08): only arm xk reads them; xl_xk.json keeps the folds and the gate
  [ $nt -gt 0 ] || { log "seed $s: no arm trained, no evaluation"; continue; }
  run eval_s$s xlearn.py evaluate && cp $X/s$s/out/xl_eval_s$s.json $X/out/xl_eval_x184_s$s.json && mkdir -p $X/out/gens_s$s && cp $X/s$s/out/xl_gens_*.jsonl $X/out/gens_s$s/ 2>/dev/null; log "seed $s done"
  rm -f $X/s$s/out/xb_control.pt $X/inx/xq/x1/s$s/xb_control.pt $X/s$s/out/xl_*/adapter_model.safetensors   # disk (184b): HF input copies and the adapter weights (seeded: re-trainable in minutes; configs and train.json stay)
done
log "X184_DONE"
