#!/bin/bash
# Addendum 177 on the 4090 (user 11-07 22:20: "dont run the rsi on the colab, use the 4090 with the smaller models that fit there"): the
# four-arm Qwen3-1.7B pilot (S:U1 ReST and S:U5 on-policy self-bridge on own wins; T:U1 and T:U5 with E4B's verified rows added; 3 rounds;
# seed 1), the settings of colab/xq_rsi.sh. Starts after the inputs (PREP_DONE) and the online arm D (ONLINE_BKR_DONE, which runs after resdec; addendum 183), and each arm only when
# the GPU is free (< 2 GB used: no other job, no Strata server). Every hidden test runs in a bubblewrap sandbox (XC_SANDBOX=bwrap: no network,
# read-only root, home hidden). Small pass first for every arm (2 rounds), then the full run arm after arm, one process per arm; summary at
# the end. Log ~/rsi4090/logs/rsi.log; RSI4090_DONE.
D=~/rsi4090; B=$D/bench; LG=$D/logs; L=$LG/rsi.log; P=~/toyq/.pt/bin/python; mkdir -p $LG; log() { echo "$(date +%H:%M) $*" >> $L; }
ARMS=${ARMS:-S:U1,S:U5,T:U1,T:U5}; ROUNDS=${ROUNDS:-3}; export XB_SEED=${XB_SEED:-1}
log "waiting for the inputs and online arm D (after resdec; addendum 183)"
until grep -q PREP_DONE $D/prep.log 2>/dev/null; do grep -q "FAILED\|stopping" $D/prep.log 2>/dev/null && { log "prep failed: stop"; exit 1; }; sleep 60; done
until grep -q ONLINE_BKR_DONE ~/toyq/behave/logs/online_bkr.log 2>/dev/null; do sleep 120; done; log "resdec and online arm D done"
until grep -q ONLINE_REP3_DONE ~/toyq/behave/logs/online_rep.log 2>/dev/null; do sleep 120; done; log "online D seeds 2-3, controls, C seeds 2-3 and the SFT arms done (addenda 189, 189b, 192)"   # user 11-08: "queue D seeds 2-3 and the controls after the gate"; "queue C seeds 2-3 after D seed 3"
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 120; done; }
GT=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/6cd26aa*/ | head -1); [ -f $GT/tokenizer.json ] || { log "no Gemma tokenizer"; exit 1; }
export XB_STUDENT=$D/mqwen XB_TEACHER=$GT XC_DATA=$D/lcb XB_SIZE=0.03 XB_HARM=0.5 RSI_NAME=qwen RSI_BATCH=${RSI_BATCH:-64} RSI_TROWS=$D/inx/xq/learn/xl_rows.jsonl \
  XC_SANDBOX=bwrap PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $B
run() { local lg=$1; shift; log "$lg"; "$@" > $LG/rsi_qwen_$lg.log 2>&1; local rc=$?
        log "$lg rc $rc: $(grep -E 'RSI |merged round|decision rules|final attempt' $LG/rsi_qwen_$lg.log | tail -n 4 | tr '\n' ' ' | cut -c1-900)"; return $rc; }
setup() { rm -rf $1/out; mkdir -p $1/out; ln -sf $D/xb/s3_ids.npy $D/xb/s3_calls.jsonl $D/xb/xq_questions.jsonl $1/; export XB_DIR=$1; }
wait_gpu; export XB_SMALL=1; unset RSI_ROUNDS; setup $D/small                                             # small pass: every arm, 2 rounds
for a in ${ARMS//,/ }; do run small_${a/:/-} $P rsi.py loop $a || { log "small $a FAILED"; exit 1; }
  grep -q "merged round 1" $LG/rsi_qwen_small_${a/:/-}.log || { log "small $a: no update ran in round 1"; exit 1; }; done
run small_summary $P rsi.py summary || exit 1
unset XB_SMALL; export RSI_ROUNDS=$ROUNDS; setup $D/s$XB_SEED; FAIL=0                                      # the full run, arm after arm
for a in ${ARMS//,/ }; do wait_gpu; run ${a/:/-} $P rsi.py loop $a || FAIL=1; log "disk free $(df -h --output=avail / | tail -1)"; done
run summary $P rsi.py summary; log "RSI4090_DONE (any arm failed: $FAIL)"
