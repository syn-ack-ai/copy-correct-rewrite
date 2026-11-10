#!/bin/bash
# Studio (11-05, addendum 128, S4a): the plain arm of the live bridged rollouts. QAT E4B (google/gemma-4-E4B-it-qat-w4a16-ct, rev 6cd26aa,
# the weights Kaggle serves) on the 4090's vLLM (port 8000; never the user's 8080), tunnelled to the box, running the same held-out gap tasks
# as the bridge and control arms (box tmp/s4/live40.txt, train_s1 pool), submissions/s1t1024, G4_TIME_SCALE=3, one attempt each.
# Waits for the 4090's trivia queue (run_g2l.sh). Log runs/s4/plain.log, PLAIN_DONE (or a line ending in HALTED).
# LORA=DIR (on dreamer) serves that PEFT adapter as "s4b" (--lora-modules) and runs SUB (default submissions/s1t1024; an adapter arm needs a
# submission whose agent.yaml says "adapter: s4b" with the folder in adapters/s4b, as Kaggle reads it).
#   nohup kaggle_g4agent/dreamer/s4_plain.sh [IDS=live40.txt] [NAME=s4_plain] [LORA=DIR SUB=submissions/s1t1024_s4b] &
for x in "$@"; do export "$x"; done
MREPO=${MREPO:-google/gemma-4-E4B-it-qat-w4a16-ct}; MREV=${MREV:-6cd26aaa2357fb2bad8c51699a7558a4d1a965bb}   # 196: the model to serve (default E4B, as before)
K=~/systemone/kaggle_g4agent; X=$K/runs/s4; Q=$X/plain.log; SP=18333; BP=18033; PP=8133; IDS=${IDS:-live40.txt}; NAME=${NAME:-s4_plain}; SUB=${SUB:-submissions/s1t1024}; LM=""; [ -n "$LORA" ] && LM="--lora-modules s4b=$LORA"; mkdir -p $X
log() { echo "$(date +%H:%M) $*" >> $Q; }
dr() { ssh -o ControlPath=none -o ServerAliveInterval=30 dreamer "$@" < /dev/null; }
bx() { ssh -o ControlPath=none -o ServerAliveInterval=30 box "$@" < /dev/null; }
down() {  # only the vLLM on port 8000 (never the one on 8080)
  dr 'pgrep -f "[v]llm serve.*--port 8000" | xargs -r kill; sleep 15; pgrep -f "[v]llm serve.*--port 8000" | xargs -r kill -9; true'
  pkill -f "[t]unnel_4090.sh $SP $BP"; pkill -f "[s]sh -f -N.*-R $BP:127.0.0.1:$SP box"; pkill -f "[s]sh -N.*-L $SP:127.0.0.1:8000 dreamer"
  log "vLLM stopped; 4090 memory in use: $(dr 'nvidia-smi --query-gpu=memory.used --format=csv,noheader')"; }
log "start: ${MREPO##*/} ${LORA:+with LoRA $LORA }on $IDS as $NAME, $SUB"
until dr 'grep -qE "G2L_DONE|FAILED|nothing to do|did not start" ~/toyq/transfer/logs/run_g2l.log 2>/dev/null'; do sleep 60; done
dr 'pgrep -f "transfer/(store|translator|pair).py" > /dev/null' && { log "a trivia job still runs on the 4090: HALTED"; exit 1; }
dr 'pgrep -f "[v]llm serve.*--port 8000" > /dev/null' && { log "a vLLM already runs on port 8000: HALTED"; exit 1; }
M=$(dr "~/g4agent/swe/bin/python -c \"from huggingface_hub import snapshot_download as s; print(s('$MREPO', revision='$MREV'))\" 2>/dev/null | tail -n 1")
[ -n "$M" ] && dr "test -f $M/config.json" || { log "$MREPO download failed: HALTED"; exit 1; }
log "$MREPO at $M"
trap down EXIT
dr "cd ~/g4agent && (nohup ./serve_model.sh $M 8000 0.90 128 --limit-mm-per-prompt '{\"image\":0,\"audio\":0}' --enforce-eager $LM > vllm_s4_plain.log 2>&1 < /dev/null &)"
for i in $(seq 1 60); do dr 'curl -sf 127.0.0.1:8000/health > /dev/null' && break; sleep 15; done
dr 'curl -sf 127.0.0.1:8000/health > /dev/null' || { log "vLLM not up: HALTED"; dr 'tail -n 20 ~/g4agent/vllm_s4_plain.log' >> $Q; exit 1; }
( nohup $K/dreamer/tunnel_4090.sh $SP $BP > $X/tunnel_plain.out 2>&1 < /dev/null & )
for i in $(seq 1 20); do curl -sf 127.0.0.1:$SP/health > /dev/null && bx "curl -sf 127.0.0.1:$BP/health > /dev/null" && break; sleep 6; done
bx "curl -sf 127.0.0.1:$BP/health > /dev/null" || { log "tunnel not up: HALTED"; exit 1; }
log "serving; rollouts start"
bx "cd ~/g4agent && rm -rf rollouts/$NAME && G4_TIME_SCALE=3 ONLY=\$HOME/g4agent/tmp/s4/$IDS ./rollout_train.sh \$HOME/g4agent/train_s1 http://127.0.0.1:$BP/v1 $NAME $SUB 3 3 $PP > rollouts_$NAME.out 2>&1"; rc=$?
dr 'curl -sf 127.0.0.1:8000/health > /dev/null' || { log "$NAME: vLLM not healthy after the rollouts: HALTED"; exit 1; }   # review 196: no result line from a broken arm
[ $rc = 0 ] || { log "$NAME: rollouts exited $rc (box or the ssh link failed): HALTED"; exit 1; }
log "$NAME done: $(bx "cat ~/g4agent/rollouts/$NAME/res_*/task_results.jsonl 2>/dev/null | grep -c '\"resolved\": true'")/$(bx "cat ~/g4agent/rollouts/$NAME/res_*/task_results.jsonl 2>/dev/null | wc -l") resolved"
log "PLAIN_DONE"
