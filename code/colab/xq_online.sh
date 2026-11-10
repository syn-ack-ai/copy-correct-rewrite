#!/bin/bash
# Addendum 189c (user 11-08: "yes, move C seeds 2-3 to Colab"): online setup C (E4B's own distribution + keep + replay), one seed per
# session, with the 4090 runner's settings (dreamer/online_rep2_4090.sh, arm direct N) on a Colab A100 40 GB (review 189c: G4 at ~9.5 CU/h could not cover two ~4 h runs). Everything that can be checked is
# checked before training:
#   code: on/code7/*.py = bench/archetypes/transfer/*.py at commit time (online.py md5 dc8a9137fe, the 4090's code7);
#   inputs: s3 rows + the 165 mix merged by mix_merge.py, sha256 equal to the 4090's data2 files;
#   correctors: on/out/gall_e4b.pt, gall_e2b.pt, sha256 equal to the 4090's data3/out;
#   models: the 4090's dequantized E4B and E2B (made with an older library stack), uploaded from the Mac's sha256-equal copies to
#           on/models/; a sha256 mismatch stops the session (review 189c: a fresh dequant on Colab's stack may not give the same weights).
# Then the small pass, the 600-item run and the 240-point gate, as on the 4090. Outputs in /content/out (downloaded to runs/NAME).
#   v2/infra/colab/colab_run.sh -s xqoC2 -g A100 -o /content/out -d kaggle_g4agent/runs/xqoC2 kaggle_g4agent/colab/xq_online.sh ON_SEED=2
while [[ $1 == *=* ]]; do export "$1"; shift; done
source /content/colab_env.sh || exit 1
: "${ON_SEED:?}"; TEACH=${TEACH:-direct}; CODE=${CODE:-code7}; CORR=${CORR:-1x}   # addendum 201: CORR=4x uses 199's 4x cross corrector (on/out4x) as D's teacher   # 189d/197a: the teacher (direct = C, bridge = D, gated = 197) and the code folder (code8 = code7 + gated)
case "$TEACH:$CODE:$CORR" in direct:code7:1x|bridge:code7:1x|gated:code8:1x|bridge:code7:4x|sft:code7:1x|sftoff:code7:1x|bridge:code7:imit1|control:code7:4xself|control:code7:1x|blend:code7:1x|control:code7:4x) ;; *) echo "TEACH=$TEACH with CODE=$CODE and CORR=$CORR is not a planned run"; exit 1;; esac   # 192b: sft = online SFT, sftoff = offline SFT
X4=(); [ "$CORR" = 4x ] && X4=(--include "on/out4x/*")   # addendum 201: the 4x corrector only when asked (no glob, no empty word)
case "$CORR" in imit1|4xself) X4=(--include "on/out204/*"); : "${CORR_SHA:?addendum 204 needs CORR_SHA}";; esac   # addendum 204: the chosen teacher, sha256 from the launcher
C=/content/onl_code; D=/content/onl; O=/content/out; mkdir -p $C $D/out $O
hf download SkyPanther/g4agent-job --repo-type dataset --include "s3/s3_*" --include "xq/mix_*" --include "on/code7/*" --include "on/code8/*" --include "on/out/*" "${X4[@]}" --include "on/models/*" \
  --force-download --local-dir /content/inx > /dev/null || { echo "inputs download failed"; exit 1; }
cp /content/inx/on/code7/*.py $C/ || exit 1; [ "$CODE" = code8 ] && { cp /content/inx/on/code8/online.py $C/online.py || exit 1; }
WANT=$([ "$CODE" = code8 ] && echo 5ad4956e39 || echo dc8a9137fe)
[ "$(md5sum $C/online.py | cut -c1-10)" = $WANT ] || { echo "online.py is not the 4090's $CODE version"; exit 1; }
python $C/mix_merge.py /content/inx/s3 /content/inx/xq $D || { echo "merge failed"; exit 1; }
cp /content/inx/on/out/gall_e4b.pt /content/inx/on/out/gall_e2b.pt $D/out/ || exit 1
check() { [ "$(sha256sum $1 | cut -d' ' -f1)" = "$2" ] && echo "sha ok $(basename $1)" || { echo "SHA MISMATCH $1"; exit 1; }; }
check $D/s3_ids.npy 84cf0f5b1ff41cb1c12e64079a51fb25a90bba1a15aa2de2d669aa290edb9043
check $D/s3_calls.jsonl 2df5f431e5dbcd8468ffb1fce5798d7cbb10a0d3422e8405477223ef604bd8cd
check $D/out/gall_e4b.pt 476ea42f4b0a2be7213c32da112f4135c9af1d76ba69b7ec35086d91e5d260b2
check $D/out/gall_e2b.pt eada45ff8c8f78599c324c8a6239b6eb890a250182817b9c873be54b2eeb4f54
if [ "$CORR" = 4x ] && [ "$TEACH" = bridge ]; then cp /content/inx/on/out4x/gall_e4b.pt $D/out/gall_e4b.pt || exit 1; check $D/out/gall_e4b.pt 80c7ccdf00d4de4a58dc06229a17707024ab86e61a9968c806ab4a8d5a555bb9; fi   # addendum 201
if [ "$CORR" = 4x ] && [ "$TEACH" = control ]; then cp /content/inx/on/out4x/gall_e2b.pt $D/out/gall_e2b.pt || exit 1; check $D/out/gall_e2b.pt 07a097db27592545a495dee95e4f858d587c692a1455e97779f33e89e761e792; fi   # addendum 206: 199's 4x self-corrector as the control teacher
if [ "$CORR" = imit1 ]; then cp /content/inx/on/out204/gall_e4b.pt $D/out/gall_e4b.pt || exit 1; check $D/out/gall_e4b.pt $CORR_SHA; fi   # addendum 204: 200's beta-1 imitable cross corrector as D's teacher
if [ "$CORR" = 4xself ]; then cp /content/inx/on/out204/gall_e2b.pt $D/out/gall_e2b.pt || exit 1; check $D/out/gall_e2b.pt $CORR_SHA; fi   # addendum 204: 199's 4x self-corrector as the control teacher
rm -rf /content/me4b_qat /content/me2b_qat; mv /content/inx/on/models/e4b_qat_bf16 /content/me4b_qat && mv /content/inx/on/models/e2b_qat_bf16 /content/me2b_qat || { echo "models missing"; exit 1; }
check /content/me4b_qat/model.safetensors 14b754d4a9c0adda6f7d83f585dd54e2b0e3258d916a87d28ebd7def82b77729
check /content/me2b_qat/model.safetensors fff95a792cfcef727383cde950b6750692dbf87561f13217152d718ea91041b0
# the 4090 runner's environment (online_rep2_4090.sh), with this session's model paths
BASE="E4B=/content/me4b_qat E2B=/content/me2b_qat TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="TR_SRC_DEPTHS=0.5,0.65 ON_DIR=$D ON_BRIDGE_TR=gall_e4b TR_CONTROL=gall_e2b"
s=$ON_SEED; unset ON_SEED; tag=$([ "$s" = 1 ] || echo _s$s);   # review 189d: seed 1 must run with no ON_SEED at all (offset 0), like the 4090
 se=$([ "$s" = 1 ] || echo ON_SEED=$s); n=onl_${TEACH}_keep_replay$tag; sm=$([ "$TEACH" = gated ] && echo _small197 || echo _small189); cd $C   # as arm(): seed 1 has no tag and no ON_SEED
try() { local lg=$1; shift; echo "$(date +%H:%M) $lg"; env $BASE "$@" > $O/$lg.log 2>&1 && return 0
        echo "$lg FAILED"; grep -vE "Warning|warn" $O/$lg.log | tail -n 20; cp $D/out/*_partial.json $O/ 2> /dev/null; exit 1; }   # review: keep the partial curve
if [ "$TEACH" = sftoff ]; then   # amendment 192b: 192's offline SFT (cross-entropy on E4B's logged replies, the 600 stream items, about two passes), as online_rep3_4090.sh
  try lora_small_sft_ce $G ON_SMALL=1 ON_LORA_LOSS=ce ON_LORA_POOL=stream ON_LORA_OUT=lora_sft_ce_small python online.py lora train
  try train_sft_ce $G ON_LORA_LOSS=ce ON_LORA_POOL=stream ON_LORA_OUT=lora_sft_ce python online.py lora train
  cp -r $D/out/lora_sft_ce $O/ || { echo "training outputs missing"; ls $D/out; exit 1; }
  try gate_sft_ce $G ON_LORA_NAME=lora_sft_ce python online.py lora gate
  tail -n 2 $O/gate_sft_ce.log | head -n 1 | cut -c1-800
  cp $D/out/gate_lora_sft_ce.json $D/out/gate_lora_sft_ce_points.json $O/ || { echo "gate outputs missing"; ls $D/out; exit 1; }
  echo "$(date +%H:%M) XQO_DONE"; exit 0
fi
try onl_small_${TEACH}$tag $G $se ON_SMALL=1 ON_ONL_TEACH=$TEACH ON_ONL_KEEP=1.0 ON_ONL_REPLAY=2 ON_BEH_TAG=$sm$tag python online.py online
try train_$n $G $se ON_ONL_TEACH=$TEACH ON_ONL_KEEP=1.0 ON_ONL_REPLAY=4 ON_BEH_TAG=$tag python online.py online
grep -E "online eval" $O/train_$n.log | tail -n 1 | cut -c1-500
cp $D/out/online_${TEACH}_keep_replay$tag.json $O/ && cp -r $D/out/lora_$n $O/ || { echo "training outputs missing"; ls $D/out; exit 1; }   # review: saved before the gate
try gate_$n $G ON_LORA_NAME=lora_$n python online.py lora gate
tail -n 2 $O/gate_$n.log | head -n 1 | cut -c1-800
cp $D/out/gate_lora_$n.json $D/out/gate_lora_${n}_points.json $O/ || { echo "gate outputs missing"; ls $D/out; exit 1; }
echo "$(date +%H:%M) XQO_DONE"
