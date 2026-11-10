#!/bin/bash
# Addendum 199 (user 11-09: "yes, pre-register and queue the corrector scaling check"): the online correctors (no-penalty gall, the D and
# self-corrector teachers) fitted 4x longer (2,400 steps instead of 600) on the same mix plus build_mix_extra.py's extra training rows
# (code, stories; every held row unchanged), cross (gall_e4b, reads E4B) and self (gall_e2b, reads E2B), with the 1x calibration rows
# (ON_CAL_POOL); then the agent rung and the genericity suite for the 4x pair and, with the same code, for the 1x pair. After 198
# (ONLINE_REP2_DONE) and the review flag code10/READY (skipped if the flag is missing at 10:30); always ends with exactly "HH:MM SCALE199_DONE",
# which 192 (online_rep3b_4090.sh) waits for. Forward passes and corrector fits only; about 7.5 h. Log ~/toyq/behave/logs/online_rep.log.
#   (cd ~/toyq/behave && exec setsid nohup bash scale199_4090.sh) > /dev/null 2>&1 < /dev/null &
B=~/toyq/behave; LG=$B/logs; L=$LG/online_rep.log; P=~/toyq/.pt/bin/python; log() { echo "$(date +%H:%M) $*" >> $L; }
done_line() { grep -qE '^[0-9]{2}:[0-9]{2} SCALE199_DONE$' $L 2>/dev/null; }
fin() { log "$1"; done_line || log "SCALE199_DONE"; exit 0; }
trap 'done_line || { log "199 runner ended unexpectedly"; log "SCALE199_DONE"; }' EXIT   # review 199a: a killed or failed runner still releases 192
BASE="E4B=$HOME/toyq/models/e4b_qat_bf16 E2B=$HOME/toyq/models/e2b_qat_bf16 TR_STUDENT=e2b TR_TEACHERS=e4b TR_PLE_CPU=1 TR_ALL_KV=0 ON_REPLY=1 ON_OPD_K=2 ON_OPD_GEN=256 ON_NGATE=240 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
D3=$B/data3; D4=$B/data4; D1=$B/data4_1x; DS=$B/data4_small
try() { local lg=$1; shift; timeout ${TO:-1h} env $BASE "$@" > $LG/$lg.log 2>&1 &&   # review 199a: every step has a time limit (fits 5 h)
   { log "$lg ok"; return 0; }; log "$lg FAILED (or over ${TO:-1h}): $(grep -vE 'Warning|warn' $LG/$lg.log | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
log "addendum 199: corrector scaling check queued after 198 (ONLINE_REP2_DONE), before 192"
until grep -qE '^[0-9]{2}:[0-9]{2} ONLINE_REP2_DONE$' $L 2>/dev/null; do sleep 120; done
dl=$(date -d "2026-11-09 10:30" +%s)   # review 199a: a fixed date
until [ -f $B/code10/READY ]; do [ "$(date +%s)" -gt "$dl" ] && fin "199 not run: code10/READY missing at 10:30"; sleep 60; done
cd $B/code10 || fin "199 not run: no code10"
free=0; until [ $free -ge 2 ]; do                                       # as 192: no process on the GPU, no vLLM, under 2 GB used, twice in a row
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && ! pgrep -f "[v]llm serve" > /dev/null && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 2000 ]; then free=$((free + 1)); else free=0; fi; sleep 60
done
log "199 start: GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
[ -f $D4/out/gall_e4b.pt ] && fin "199 not run: $D4/out/gall_e4b.pt exists already"
$P append_extra.py $D3 $B/mix_extra $D4 >> $LG/scale199_prep.log 2>&1 || fin "199 not run: append_extra failed"
N0=$($P -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['split'] == 'train'))" $D3/s3_calls.jsonl) || fin "199 not run: no 1x row count"
mkdir -p $D1/out $DS/out && ln -sf $D3/s3_ids.npy $D3/s3_calls.jsonl $D1/ && ln -sf $D4/s3_ids.npy $D4/s3_calls.jsonl $DS/ && cp $D3/out/gall_e4b.pt $D3/out/gall_e2b.pt $D1/out/ || fin "199 not run: eval dirs"
log "199 prep: $(tail -n 1 $LG/scale199_prep.log | cut -c1-200) | 1x training rows (calibration pool) $N0"
GT="TR_SRC_DEPTHS=0.5,0.65 ON_TRMIX=1 ON_TR_SIZE=0 ON_CAL_POOL=$N0"; EV="TR_SRC_DEPTHS=0.5,0.65"
# 1. small pass in its own folder: both fits (20 steps), the rung and the suite (small)
for s in e4b e2b; do try s199_small_tr_$s $GT ON_DIR=$DS ON_SMALL=1 $P online.py train gall_$s || fin "199 small pass failed (fit $s)"; done
try s199_small_rung $EV ON_DIR=$DS ON_SMALL=1 $P online.py rung gall && try s199_small_mix $EV ON_DIR=$DS ON_SMALL=1 $P online.py mixgate gall || fin "199 small pass failed (evals)"
# 2. the 1x pair with this code (the baseline)
try s199_rung_1x $EV ON_DIR=$D1 $P online.py rung gall; try s199_mix_1x $EV ON_DIR=$D1 $P online.py mixgate gall
# 3. the 4x fits; the rung after the cross fit (bridge against alone only) for an early read
TO=5h try s199_tr_e4b $GT ON_DIR=$D4 ON_STEPS=2400 $P online.py train gall_e4b || fin "199: the 4x cross fit failed"
try s199_rung_4x_cross $EV ON_DIR=$D4 $P online.py rung gall && cp $D4/out/gate_rung_gall.json $D4/out/gate_rung_gall_cross_only.json
TO=5h try s199_tr_e2b $GT ON_DIR=$D4 ON_STEPS=2400 $P online.py train gall_e2b || fin "199: the 4x self fit failed"
try s199_rung_4x $EV ON_DIR=$D4 $P online.py rung gall; try s199_mix_4x $EV ON_DIR=$D4 $P online.py mixgate gall
log "199 result: $($P - $D1/out $D4/out $D3/out <<'EOF' 2>&1 | cut -c1-3000
import json, sys, torch
a, b, c = sys.argv[1:4]; out = {}
for tag, d in (("1x", a), ("4x", b)):
    try:
        r = json.load(open(f"{d}/gate_rung_gall.json"))["whole reply, cross-entropy"]; out[f"rung {tag} (alone, cross, self, self - cross)"] = [r["alone"], r["bridge"], r.get("control"), round(r["control"] - r["bridge"], 4) if r.get("control") is not None else None]
    except Exception as e:
        out[f"rung {tag}"] = repr(e)[:80]
    try:
        m = json.load(open(f"{d}/gate_mix_gall.json"))["per kind"]; out[f"suite {tag} ce (alone, cross, self)"] = {k: [m[k][x]["ce"] for x in ("alone", "bridge", "control")] for k in m}
    except Exception as e:
        out[f"suite {tag}"] = repr(e)[:80]
try:
    s1 = torch.load(f"{c}/gall_e4b.pt", weights_only=False, map_location="cpu")["state"]; s4 = torch.load(f"{b}/gall_e4b.pt", weights_only=False, map_location="cpu")["state"]
    out["calibration check: max |mx 1x - mx 4x| (cross)"] = float(abs(torch.as_tensor(s1["mx"]) - torch.as_tensor(s4["mx"])).max())
except Exception as e:
    out["calibration check"] = repr(e)[:80]
print(json.dumps(out))
EOF
)"
fin "199 done"
