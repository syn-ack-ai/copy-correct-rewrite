#!/bin/bash
# Addendum 174 (transfer/xlearn.py): does the bridge add anything on top of plain SFT on E4B's text? Counted three ways: fixed (trained problems
# Qwen failed, now solved), new (held problems never trained on), kept (nothing it solved is lost; plus off-task knowledge).
#   STAGE=build: E4B's verified solutions on older LiveCodeBench (test.jsonl-test4.jsonl) = the rows, and the baselines (Qwen alone, E4B) on
#                both pools; outputs xl_rows.jsonl, xl_base.json (uploaded to HF xq/learn/ by the launcher).
#   STAGE=arms XB_SEED=N: the four arms (sft, sft_b, sft_c, kl_b; rank-64 LoRA, same rows, order and dose) with seed N's translators, then the
#                evaluation (both pools + off-task rows).
# Small pass first in both stages.   v2/infra/colab/colab_run.sh -s xqlB -g G4 -o /content/out -d LOCAL kaggle_g4agent/colab/xq_learn.sh STAGE=build
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
B=/content/bench/transfer; O=/content/out; mkdir -p $B $O
INC=(--include "s3/*" --include "xq/*.py" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/code/*")
[ "$STAGE" = arms ] && INC+=(--include "xq/x1/s${XB_SEED:?}/xb_*" --include "xq/learn/*")
[ "$STAGE" = regrade ] && INC+=(--include "xq/prior/s${XB_SEED:?}/*" --include "xq/regrade/*")   # 194: this seed's stored adapters and the lost lists
[ "$STAGE" = cf ] && INC+=(--include "xq/x1/s${XB_SEED:?}/xb_*" --include "xq/learn/*" --include "xq/prior/s${XB_SEED}/*")   # 181: the 179 adapters of this seed, when uploaded
[ "$STAGE" = short ] && INC+=(--include "xq/learn/*")   # 179a: the short-rows build reads the rows, baselines and the build's generations
[ "$STAGE" = build ] && INC+=(--include "xq/x1/s1/xb_*")   # the gate check uses seed 1's translators
[ "$STAGE" = e2b ] && INC+=(--include "xq/learn/*" --include "xq/prior_e2b/s${XB_SEED:?}/xl_sft/*")   # 186: rows, baselines, short rows, and the runbook build's plans (xl_rb_gens.jsonl)
[ "$STAGE" = two ] && INC+=(--include "xq/learn/*" --include "xq/prior_e2b/s${XB_SEED:?}/xl_sft/*")   # 187: rows, baselines, short rows, E2B's own answers; the E2B sft anchor from 186
[ "$STAGE" = rbuild ] && INC+=(--include "xq/learn/*")   # 182: the runbook rows read the rows, baselines, the build's generations and the short rows
[ "$STAGE" = rb ] && INC+=(--include "xq/learn/*" --include "xq/prior/s${XB_SEED:?}/xl_sft/*")   # 182: the runbook rows and this seed's 179 sft (the anchor); no translators (review 11-08)
[ "$STAGE" = rb ] && [[ "${XL_ARMS:-rb}" == *_c* ]] && INC+=(--include "xq/x1/s${XB_SEED}/xb_control.pt")   # 184c: rb_c needs the seed's control translator
hf download SkyPanther/g4agent-job --repo-type dataset "${INC[@]}" --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
cp /content/inx/s3/*.py /content/inx/xq/*.py /content/inx/xq/code/*.py $B/
[ -f /content/lcb/test4.jsonl ] || hf download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "test*.jsonl" --local-dir /content/lcb > /dev/null || { echo "LiveCodeBench download failed"; exit 1; }
ls /content/lcb
if [ "$STAGE" = e2b ]; then   # 186: E2B is the student; E4B only lends its tokenizer and config (XB_TEACHER_TOK_ONLY)
  [ -f /content/me4b_tok/tokenizer.json ] || hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --include "*.json" --include "tokenizer*" --local-dir /content/me4b_tok > /dev/null || { echo "E4B tokenizer download failed"; exit 1; }
  if [ ! -f /content/me2b_qat/config.json ]; then
    hf download google/gemma-4-E2B-it-qat-w4a16-ct --revision 971342c08f607aa7779983f6b5289778b5d271a7 --local-dir /content/me2b_qat_ct > /dev/null || { echo "E2B QAT download failed"; exit 1; }
    uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
    python $B/dequant.py /content/me2b_qat_ct /content/me2b_qat > $O/dequant_e2b.log 2>&1 && grep DEQUANT_OK $O/dequant_e2b.log || { echo "E2B dequantize FAILED"; tail -n 20 $O/dequant_e2b.log; exit 1; }
  fi
elif [ "$STAGE" = regrade ]; then   # 194: the Qwen student only; E4B lends its tokenizer (XB_TEACHER_TOK_ONLY)
  [ -f /content/me4b_tok/tokenizer.json ] || hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --include "*.json" --include "tokenizer*" --local-dir /content/me4b_tok > /dev/null || { echo "E4B tokenizer download failed"; exit 1; }
elif [ ! -f /content/me4b_qat/config.json ]; then   # (187: STAGE=two also loads E2B, below)
  hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --local-dir /content/me4b_qat_ct > /dev/null || { echo "QAT download failed"; exit 1; }
  uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
  python $B/dequant.py /content/me4b_qat_ct /content/me4b_qat > $O/dequant_e4b.log 2>&1 && grep DEQUANT_OK $O/dequant_e4b.log || { echo "dequantize FAILED"; exit 1; }
fi
if [ "$STAGE" = two ] && [ ! -f /content/me2b_qat/config.json ]; then   # 187: both Gemma models, full weights (each is the other's teacher)
  hf download google/gemma-4-E2B-it-qat-w4a16-ct --revision 971342c08f607aa7779983f6b5289778b5d271a7 --local-dir /content/me2b_qat_ct > /dev/null || { echo "E2B QAT download failed"; exit 1; }
  uv pip install -q --python /opt/e4b/bin/python compressed-tensors || exit 1
  python $B/dequant.py /content/me2b_qat_ct /content/me2b_qat > $O/dequant_e2b.log 2>&1 && grep DEQUANT_OK $O/dequant_e2b.log || { echo "E2B dequantize FAILED"; tail -n 20 $O/dequant_e2b.log; exit 1; }
fi
[ "$STAGE" = e2b ] || [ "$STAGE" = two ] || [ -f /content/mqwen/config.json ] || hf download Qwen/Qwen3-1.7B --local-dir /content/mqwen > /dev/null || { echo "Qwen download failed"; exit 1; }
[ -f /content/xb/s3_ids.npy ] || python $B/mix_merge.py /content/inx/s3 /content/inx/xq /content/xb || { echo "merge failed"; exit 1; }
cp /content/inx/xq/xq_questions.jsonl /content/xb/
export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/mqwen XC_DATA=/content/lcb XB_SIZE=0.03 XB_HARM=0.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
if [ "$STAGE" = two ]; then   # 187: XL_DIR=e2b: E4B teaches E2B; XL_DIR=e4b: E2B teaches E4B (rows = E2B's own passing answers)
  case "${XL_DIR:?}" in e2b) export XB_TEACHER=/content/me4b_qat XB_STUDENT=/content/me2b_qat ;; e4b) export XB_TEACHER=/content/me2b_qat XB_STUDENT=/content/me4b_qat XL_ROWS_FROM=/content/inx/xq/learn/xl_gens_e2b_alone.jsonl XL_ROWS_PASS=/content/inx/xq/learn/xl_eval_e2b_alone.json ;; *) echo "XL_DIR must be e2b or e4b"; exit 1 ;; esac
  export XB_SBOS="<bos>" XL_NO_KEPT_POOL=1
fi
[ "$STAGE" = regrade ] && export XB_TEACHER=/content/me4b_tok XB_TEACHER_TOK_ONLY=1 XC_MAXNEW=${XL_REGRADE_MAXNEW:-1536} XL_REGRADE=/content/inx/xq/regrade/lost.json XL_REGRADE_DIR=/content/inx/xq/prior/s$XB_SEED   # 194
[ "$STAGE" = e2b ] && export XB_TEACHER=/content/me4b_tok XB_TEACHER_TOK_ONLY=1 XB_STUDENT=/content/me2b_qat XB_SBOS="<bos>" XL_NO_KEPT_POOL=1 XL_RB_PLANS=/content/inx/xq/learn/xl_rb_gens.jsonl   # 186
run() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; "$@" > $O/xl_$lg.log 2>&1 || { echo "$lg FAILED"; tail -n 25 $O/xl_$lg.log; exit 1; }
        grep -E "pools|solved|Qwen alone|E4B greedy|arm |step [0-9]+ of|passed|self-check|short rows|refit|rewrite|plans|runbook|first plan" $O/xl_$lg.log | tail -n 6 | cut -c1-300; }
setup() {   # $1 = folder; a clean out/ with the data links (and the seed's translators, the shared rows and baselines for the arms)
  rm -rf $1/out; mkdir -p $1/out; ln -sf /content/xb/s3_ids.npy /content/xb/s3_calls.jsonl /content/xb/xq_questions.jsonl $1/; export XB_DIR=$1
  if [ "$STAGE" = arms ] || [ "$STAGE" = cf ]; then cp /content/inx/xq/x1/s$XB_SEED/xb_*.pt $1/out/ && cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json $1/ || return 1
    cp /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/ 2>/dev/null; fi   # 179a, when built
  if [ "$STAGE" = rb ]; then cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_rb.jsonl /content/inx/xq/learn/xl_rb.json $1/ || return 1
    cp /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/ 2>/dev/null; cp /content/inx/xq/x1/s$XB_SEED/xb_control.pt $1/out/ 2>/dev/null; true; fi   # 182: sft and rb need no translators; 184c: rb_c the control
  if [ "$STAGE" = two ]; then cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/ || return 1; fi   # 187
  if [ "$STAGE" = e2b ]; then cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/ || return 1; fi   # 186: not the Qwen runbook rows
  if [ "$STAGE" = rbuild ]; then cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json $1/ || return 1
    cp /content/inx/xq/learn/xl_build_gens.jsonl /content/inx/xq/learn/xl_rows_short.jsonl /content/inx/xq/learn/xl_short.json $1/ 2>/dev/null; true; fi
  if [ "$STAGE" = short ]; then cp /content/inx/xq/learn/xl_rows.jsonl /content/inx/xq/learn/xl_base.json $1/ && cp /content/inx/xq/learn/xl_build_gens.jsonl $1/ 2>/dev/null; true; fi
  if [ "$STAGE" = build ]; then cp /content/inx/xq/x1/s1/xb_*.pt $1/out/ || return 1; fi
}
if [ "$STAGE" = build ]; then
  export XB_SEED=0 XB_SMALL=1; setup /content/xl_small && run build_small python xlearn.py build || exit 1
  run precheck_small python xlearn.py precheck || exit 1
  unset XB_SMALL; setup /content/xl_build && run build python xlearn.py build || exit 1
  run precheck python xlearn.py precheck || exit 1; d=/content/xl_build; cp $d/xl_build_gens.jsonl $d/xl_precheck.json $O/
  python - $d <<'EOF'   # 174b: the gate decision goes into the baselines file; the arms sessions choose their arms from it
import json, sys
d = sys.argv[1]; b = json.load(open(f"{d}/xl_base.json")); g = json.load(open(f"{d}/xl_precheck.json"))["gate"]; b["gate"] = g; json.dump(b, open(f"{d}/xl_base.json", "w")); print(g)
EOF
  cp $d/xl_rows.jsonl $d/xl_base.json $O/
  grep -q '"gate": "GATE_PASS"' $d/xl_precheck.json && echo "GATE_PASS: arms sft, sft_b, sft_c, kl_b, sdft" || echo "GATE_FAIL: the bridge does not beat its control on these rows; arms sft, sft_self, sft_c, sdft (174b, 174c)"
  grep -E "precheck" $O/xl_precheck.log | cut -c1-400; python - <<'EOF'
import json, collections
import glob
rows = [json.loads(l) for l in open(glob.glob("/content/xl_build/xl_rows.jsonl")[0])]
print("rows", len(rows), dict(collections.Counter((r["difficulty"], r["source"]) for r in rows)))
EOF
  echo "$(date +%H:%M) XQL_BUILD_DONE"
elif [ "$STAGE" = arms ]; then
  ARMS=$(python -c "import json; print('sft sft_b sft_c kl_b sdft' if json.load(open('/content/inx/xq/learn/xl_base.json')).get('gate', 'GATE_PASS') == 'GATE_PASS' else 'sft sft_self sft_c sdft')"); [ -n "$XL_ARMS" ] && ARMS="${XL_ARMS//,/ }"; echo "arms: $ARMS (gate in xl_base.json; XL_ARMS overrides, 179)"
  setup /content/xl_s${XB_SEED}_small || exit 1; export XB_SMALL=1
  for a in $ARMS; do run train_${a}_small python xlearn.py train $a || exit 1; done
  run eval_small python xlearn.py evaluate || exit 1
  unset XB_SMALL; setup /content/xl_s$XB_SEED || exit 1
  for a in $ARMS; do run train_$a python xlearn.py train $a || exit 1; cp -r $XB_DIR/out/xl_$a $O/xl_${a}_s$XB_SEED; done
  run eval python xlearn.py evaluate || exit 1
  cp $XB_DIR/out/xl_eval_s$XB_SEED.json $XB_DIR/out/xl_gens_*.jsonl $O/; tail -n 2 $O/xl_eval.log | head -n 1 | cut -c1-3000
  echo "$(date +%H:%M) XQL_ARMS_DONE"
elif [ "$STAGE" = short ]; then   # 179a: shortest-correct rows; run with colab_run -w (the write token) so they can be uploaded to HF xq/learn/
  export XB_SEED=0 XB_SMALL=1; setup /content/xl_short_small && run short_small python xlearn.py build_short || exit 1; unset XB_SMALL
  setup /content/xl_short || exit 1; run short python xlearn.py build_short || exit 1; d=/content/xl_short; cp $d/xl_rows_short.jsonl $d/xl_short.json $O/; cat $d/xl_short.json; echo
  mkdir -p /content/up_short && cp $d/xl_rows_short.jsonl $d/xl_short.json /content/up_short/ && hf upload SkyPanther/g4agent-job /content/up_short xq/learn --repo-type dataset > /dev/null && echo "uploaded to xq/learn" || { echo "upload FAILED"; exit 1; }
  echo "$(date +%H:%M) XQL_SHORT_DONE"
elif [ "$STAGE" = cf ]; then   # 181: refit the control translator on the rows, gate it, the rewrite rows, arms opd_cf and rw_cf, then the evaluation
  ARMS="${XL_ARMS:-opd_cf,mix_cf,rw_cf}"; ARMS="${ARMS//,/ }"; echo "arms: $ARMS (+ this seed's 179 adapters when present)"
  setup /content/xl_cf${XB_SEED}_small || exit 1; export XB_SMALL=1
  run fit_cf_small python xlearn.py fit_cf || exit 1; run rewrite_small python xlearn.py rewrite || exit 1
  for a in $ARMS; do run train_${a}_small python xlearn.py train $a || exit 1; done
  run eval_small python xlearn.py evaluate || exit 1
  unset XB_SMALL; setup /content/xl_cf$XB_SEED || exit 1; P=/content/inx/xq/prior/s$XB_SEED
  for a in sft opd_c mix_c; do [ -f $P/xl_$a/adapter_model.safetensors ] && cp -r $P/xl_$a $XB_DIR/out/xl_$a && echo "prior adapter: $a (179 seed $XB_SEED)"; done
  [ -d $XB_DIR/out/xl_sft ] || ARMS="sft $ARMS"   # no prior sft: train the anchor here
  run fit_cf python xlearn.py fit_cf || exit 1; cp $XB_DIR/xl_cf_gate.json $O/; cp $XB_DIR/out/xb_control_cf.pt $O/xb_control_cf_s$XB_SEED.pt
  python -c "import json; d = json.load(open('$XB_DIR/xl_cf_gate.json')); print({k: v for k, v in d.items() if k not in ('first50 (ce, alone, harm, gate)', 'last50')})" | cut -c1-1500
  if ! grep -q '"gate": "CF_PASS"' $XB_DIR/xl_cf_gate.json; then echo "CF_FAIL: the refit translator is not clearly better on the rows or not better than Qwen alone on held; no arms"; echo "$(date +%H:%M) XQL_CF_DONE"; exit 0; fi
  run rewrite python xlearn.py rewrite || exit 1; cp $XB_DIR/xl_rows_rw.jsonl $XB_DIR/xl_rw.json $XB_DIR/xl_rw_gens.jsonl $O/; cat $XB_DIR/xl_rw.json; echo
  NRWR=$(python -c "import json; print(json.load(open('$XB_DIR/xl_rw.json'))['rows'])"); [ "$NRWR" -gt 0 ] || { ARMS="${ARMS//rw_cf/}"; echo "no passing rewrites: rw_cf skipped"; }   # review 11-07
  for a in $ARMS; do run train_$a python xlearn.py train $a || exit 1; cp -r $XB_DIR/out/xl_$a $O/xl_${a}_s$XB_SEED; done
  run eval python xlearn.py evaluate || exit 1
  cp $XB_DIR/out/xl_eval_s$XB_SEED.json $O/xl_eval_cf_s$XB_SEED.json; cp $XB_DIR/out/xl_gens_*.jsonl $O/; tail -n 2 $O/xl_eval.log | head -n 1 | cut -c1-3000
  echo "$(date +%H:%M) XQL_CF_DONE"
elif [ "$STAGE" = rbuild ]; then   # 182: the runbook rows, built once (no seed); run with colab_run -w so they can be uploaded to HF xq/learn/
  export XB_SEED=0 XB_SMALL=1; setup /content/xl_rb_small && run runbook_small python xlearn.py runbook || exit 1; cat /content/xl_rb_small/xl_rb.json; echo; unset XB_SMALL
  export XL_RB_PROBE=${XL_RB_PROBE:-24}; setup /content/xl_rb_probe && run runbook_probe python xlearn.py runbook || exit 1; unset XL_RB_PROBE; cp /content/xl_rb_probe/xl_rb_probe.json $O/; cat /content/xl_rb_probe/xl_rb_probe.json | cut -c1-4000; echo
  OKP=$(python -c "import json; d = json.load(open('/content/xl_rb_probe/xl_rb_probe.json')); print(int(d['plans passing the screen'] >= ${XL_RB_PROBE_MIN:-0.6} * d['probe rows']))")   # review 11-08: the yield check
  [ "$OKP" = 1 ] || { echo "RB_PROBE_FAIL: too few plans pass the screen on the probe; no full build (182)"; echo "$(date +%H:%M) XQL_RB_BUILD_DONE"; exit 0; }
  setup /content/xl_rb || exit 1; run runbook python xlearn.py runbook || exit 1; d=/content/xl_rb; cp $d/xl_rows_rb.jsonl $d/xl_rb.json $d/xl_rb_gens.jsonl $O/; cat $d/xl_rb.json; echo
  NRBR=$(python -c "import json; print(json.load(open('$d/xl_rb.json'))['rows'])"); echo "runbook rows: $NRBR"
  [ "$NRBR" -ge "${XL_RB_MIN:-100}" ] || { echo "RB_FEW: fewer than ${XL_RB_MIN:-100} runbook rows; not uploaded, no seed sessions (182)"; echo "$(date +%H:%M) XQL_RB_BUILD_DONE"; exit 0; }
  mkdir -p /content/up_rb && cp $d/xl_rows_rb.jsonl $d/xl_rb.json $d/xl_rb_gens.jsonl /content/up_rb/ && hf upload SkyPanther/g4agent-job /content/up_rb xq/learn --repo-type dataset > /dev/null && echo "RB_UPLOADED to xq/learn" || { echo "upload FAILED"; exit 1; }
  echo "$(date +%H:%M) XQL_RB_BUILD_DONE"
elif [ "$STAGE" = rb ]; then   # 182: arm rb on the runbook rows (184c: XL_ARMS=rb_c, XL_TAG=rbc); this seed's 179 sft adapter is the in-session anchor (trained here if missing); then the evaluation
  ARMS="${XL_ARMS:-rb}"; ARMS="${ARMS//,/ }"; TAG="${XL_TAG:-rb}"; echo "arms: $ARMS (tag $TAG)"
  setup /content/xl_rb${XB_SEED}_small || exit 1; export XB_SMALL=1
  for a in $ARMS; do run train_${a}_small python xlearn.py train $a || exit 1; done; run eval_small python xlearn.py evaluate || exit 1
  unset XB_SMALL; setup /content/xl_rb$XB_SEED || exit 1; P=/content/inx/xq/prior/s$XB_SEED
  if [ -f $P/xl_sft/adapter_model.safetensors ]; then cp -r $P/xl_sft $XB_DIR/out/xl_sft && echo "prior adapter: sft (179 seed $XB_SEED)"
  else run train_sft python xlearn.py train sft || exit 1; cp -r $XB_DIR/out/xl_sft $O/xl_sft_s$XB_SEED; fi
  for a in $ARMS; do run train_$a python xlearn.py train $a || exit 1; cp -r $XB_DIR/out/xl_$a $O/xl_${a}_s$XB_SEED; done
  run eval python xlearn.py evaluate || exit 1
  cp $XB_DIR/out/xl_eval_s$XB_SEED.json $O/xl_eval_${TAG}_s$XB_SEED.json; cp $XB_DIR/out/xl_gens_*.jsonl $O/; tail -n 2 $O/xl_eval.log | head -n 1 | cut -c1-3000
  echo "$(date +%H:%M) XQL_RB_DONE"
elif [ "$STAGE" = e2b ]; then   # 186 (user 11-08: "replicate with e2b"): Gemma 4 E2B as the coding student, arms sft, sft_s, rb (E2B re-solves with the Qwen build's plans), then the drift from the original E2B
  : "${XB_SEED:?}"; ARMS="${XL_ARMS:-sft sft_s rb}"; ARMS="${ARMS//,/ }"; echo "E2B arms: $ARMS (seed $XB_SEED)"   # review 11-08: the seed is required
  setup /content/xl_e2b_small || exit 1; export XB_SMALL=1
  case " $ARMS " in *" rb "*) run runbook_small python xlearn.py runbook || exit 1 ;; esac   # 188: the runbook only when arm rb runs
  for a in $ARMS; do run train_${a}_small python xlearn.py train $a || exit 1; done
  run eval_small python xlearn.py evaluate || exit 1
  unset XB_SMALL; setup /content/xl_e2b || exit 1
  case " $ARMS " in *" rb "*)
    run runbook python xlearn.py runbook || exit 1; cp $XB_DIR/xl_rows_rb.jsonl $XB_DIR/xl_rb.json $XB_DIR/xl_rb_gens.jsonl $O/; cat $XB_DIR/xl_rb.json; echo
    NRBR=$(python -c "import json; print(json.load(open('$XB_DIR/xl_rb.json'))['rows'])"); [ "$NRBR" -gt 0 ] || { ARMS="${ARMS//rb/}"; echo "no E2B runbook rows: rb skipped"; } ;;
  esac
  P=/content/inx/xq/prior_e2b/s$XB_SEED; case " $ARMS " in *" sft "*) ;; *) [ -f $P/xl_sft/adapter_model.safetensors ] && cp -r $P/xl_sft $XB_DIR/out/xl_sft && echo "prior adapter: sft (186 E2B seed $XB_SEED, the in-session anchor)" ;; esac   # 188
  for a in $ARMS; do run train_$a python xlearn.py train $a || exit 1; cp -r $XB_DIR/out/xl_$a $O/xl_${a}_e2b_s$XB_SEED; done
  run eval python xlearn.py evaluate || exit 1
  cp $XB_DIR/out/xl_eval_s$XB_SEED.json $O/xl_eval_e2b_s$XB_SEED.json; cp $XB_DIR/out/xl_gens_*.jsonl $O/; tail -n 2 $O/xl_eval.log | head -n 1 | cut -c1-3000
  python - $XB_DIR/rows_untaught.json <<'PY' > $O/xl_untaught_rows.log 2>&1 || echo "untaught rows FAILED"
import json, sys, xbridge as XB
from transformers import AutoTokenizer
import os
rows = XB.rows_all(AutoTokenizer.from_pretrained(os.environ["XB_TEACHER"]))
out = [{"kind": r["kind"], "ctx": r["ctx"], "reply": r["reply"]} for r in rows if r["split"] == "held" and r["kind"] in ("prose", "code", "story")]
json.dump(out, open(sys.argv[1], "w")); print({k: sum(1 for r in out if r["kind"] == k) for k in ("prose", "code", "story")})
PY
  DRIFT_DIR=$XB_DIR/out python drift.py $XB_STUDENT $XB_DIR/rows_untaught.json /content $O/drift_e2b_s$XB_SEED.json $(ls -d $XB_DIR/out/xl_*/ | xargs -n1 basename | sed 's/^xl_//') > $O/xl_drift.log 2>&1 && echo "drift done" || { echo "drift FAILED (the results above stand)"; tail -n 5 $O/xl_drift.log; }   # review 11-08: not fatal
  echo "$(date +%H:%M) XQL_E2B_DONE"
elif [ "$STAGE" = two ]; then   # 187 (user 11-08: two-way E4B <-> E2B, "including the 600 trained correctors for both"): fit both correctors for this student, then the anchor arms, the evaluation and the drift
  : "${XB_SEED:?}"
  if [ "$XL_DIR" = e2b ]; then ARMS="${XL_ARMS:-sft_self_s sft_c_s sft_b_s}"; else ARMS="${XL_ARMS:-sft sft_self sft_c sft_b}"; fi; ARMS="${ARMS//,/ }"
  echo "two-way, student ${XL_DIR}: teacher $XB_TEACHER, student $XB_STUDENT; arms $ARMS (seed $XB_SEED)"
  fit() { run translator_$1$2 python xbridge.py train $1 || exit 1; grep -E "^.*(calibration|step [0-9]+ of|TRAIN_DONE)" $O/xl_translator_$1$2.log | tail -n 2 | cut -c1-300; }
  setup /content/xl_two_small || exit 1; export XB_SMALL=1
  [ "$XL_DIR" = e4b ] && { run rows_small python xlearn.py build_from_gens || exit 1; }
  fit control _small; fit bridge _small
  for a in $ARMS; do run train_${a}_small python xlearn.py train $a || exit 1; done
  run eval_small python xlearn.py evaluate || exit 1
  unset XB_SMALL; setup /content/xl_two || exit 1
  [ "$XL_DIR" = e4b ] && { run rows python xlearn.py build_from_gens || exit 1; cp $XB_DIR/xl_rows.jsonl $O/xl_rows_from_e2b.jsonl; cp $XB_DIR/xl_base.json $O/xl_base_from_e2b.json; }
  fit control ""; fit bridge ""
  P=/content/inx/xq/prior_e2b/s$XB_SEED
  if [ "$XL_DIR" = e2b ] && [ -f $P/xl_sft/adapter_model.safetensors ]; then cp -r $P/xl_sft $XB_DIR/out/xl_sft && echo "prior adapter: sft (186 E2B seed $XB_SEED, the in-session anchor)"
  elif [ "$XL_DIR" = e2b ]; then ARMS="sft $ARMS"; fi
  for a in $ARMS; do run train_$a python xlearn.py train $a || exit 1; cp -r $XB_DIR/out/xl_$a $O/xl_${a}_${XL_DIR}_s$XB_SEED; done
  run eval python xlearn.py evaluate || exit 1
  cp $XB_DIR/out/xl_eval_s$XB_SEED.json $O/xl_eval_two_${XL_DIR}_s$XB_SEED.json; cp $XB_DIR/out/xl_gens_*.jsonl $O/; tail -n 2 $O/xl_eval.log | head -n 1 | cut -c1-3000
  python - $XB_DIR/rows_untaught.json <<'PY' > $O/xl_untaught_rows.log 2>&1 || echo "untaught rows FAILED"
import json, sys, os, xbridge as XB
from transformers import AutoTokenizer
rows = XB.rows_all(AutoTokenizer.from_pretrained(os.environ["XB_TEACHER"]))
out = [{"kind": r["kind"], "ctx": r["ctx"], "reply": r["reply"]} for r in rows if r["split"] == "held" and r["kind"] in ("prose", "code", "story")]
json.dump(out, open(sys.argv[1], "w")); print({k: sum(1 for r in out if r["kind"] == k) for k in ("prose", "code", "story")})
PY
  DRIFT_DIR=$XB_DIR/out python drift.py $XB_STUDENT $XB_DIR/rows_untaught.json /content $O/drift_two_${XL_DIR}_s$XB_SEED.json $(ls -d $XB_DIR/out/xl_*/ | xargs -n1 basename | sed 's/^xl_//') > $O/xl_drift.log 2>&1 && echo "drift done" || { echo "drift FAILED (the results above stand)"; tail -n 5 $O/xl_drift.log; }
  cp $XB_DIR/out/xb_control.pt $O/xb_control_${XL_DIR}_s$XB_SEED.pt; cp $XB_DIR/out/xb_bridge.pt $O/xb_bridge_${XL_DIR}_s$XB_SEED.pt   # the fitted correctors (for later use)
  echo "$(date +%H:%M) XQL_TWO_DONE"
elif [ "$STAGE" = regrade ]; then   # 194 (user 11-08, from the Fable review): the problems each stored Qwen adapter of this seed lost, regenerated with XC_MAXNEW tokens; no training
  : "${XB_SEED:?}"; ls /content/inx/xq/prior/s$XB_SEED | tr '\n' ' '; echo
  setup /content/xl_rg_small || exit 1; export XB_SMALL=1; run regrade_small python xlearn.py regrade || exit 1
  unset XB_SMALL; setup /content/xl_rg || exit 1; run regrade python xlearn.py regrade || exit 1
  cp $XB_DIR/out/xl_regrade_s$XB_SEED.json $XB_DIR/out/xl_regrade_gens_s$XB_SEED.jsonl $O/
  python -c "import json; d = json.load(open('$O/xl_regrade_s$XB_SEED.json')); print({a: (v['lost'], v['recovered'], v['longer than 768 tokens']) for a, v in d['arms'].items()})"
  echo "$(date +%H:%M) XQL_REGRADE_DONE"
else
  echo "STAGE must be build, arms, short, cf, rbuild, rb, e2b or two"; exit 1
fi
