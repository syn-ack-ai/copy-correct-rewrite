#!/bin/bash
# Addendum 184 inputs on the 4090 (runs at once; network and CPU only, no GPU). From HF: xq/learn/* (rows, baselines, the build's
# generations, the short rows) and each seed's control translator (xq/x1/s1-3/xb_control.pt). LiveCodeBench train files (test.jsonl-test4.jsonl,
# the pinned revision as on Colab) one at a time: only the problems the evaluation reads (the 299 row problems and the train-pool problems the
# build's Qwen run passed; every difficulty, unlike ~/rsi4090/lcb, which is easy and medium only), then the raw file is deleted. The held files
# (test5, test6, easy and medium) are the ~/rsi4090/lcb copies. Checks: every needed train id found; the held pool equals the baselines' list.
# Stops if fewer than 4 GB stay free. Log ~/x184/logs/prep.log; PREP184_DONE.
X=~/x184; HF=~/toyq/.pt/bin/hf; P=~/toyq/.pt/bin/python; L=$X/logs/prep.log; mkdir -p $X/lcb $X/raw $X/inx $X/logs $X/out $X/bench; log() { echo "$(date +%H:%M) $*" >> $L; }
room() { local a=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$a" -ge 4 ] || { log "only ${a} GB free: stopping"; exit 1; }; }
: > $L; room; log "prep: inputs from HF"   # review 11-08: a fresh log, so an old FAILED line never halts the runner
$HF download SkyPanther/g4agent-job --repo-type dataset --include "xq/learn/*" --include "xq/x1/s1/xb_control.pt" --include "xq/x1/s2/xb_control.pt" --include "xq/x1/s3/xb_control.pt" --local-dir $X/inx > /dev/null 2>&1 || { log "inputs download FAILED"; exit 1; }
for s in 1 2 3; do [ -f $X/inx/xq/x1/s$s/xb_control.pt ] || { log "no control translator for seed $s: FAILED"; exit 1; }; done
$P - $X/inx/xq/learn/xl_base.json $X/lcb/need_ids.json <<'PY' >> $L 2>&1 || { log "ids FAILED"; exit 1; }
import json, sys
b = json.load(open(sys.argv[1])); need = sorted(set(b["rows"]) | {i for i in b["train"] if b["qwen"].get(i)}); json.dump(need, open(sys.argv[2], "w")); print("train ids needed:", len(need))
PY
for f in test test2 test3 test4; do
  room; $HF download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "$f.jsonl" --local-dir $X/raw > /dev/null 2>&1 || { log "download $f FAILED"; exit 1; }
  $P - $X/raw/$f.jsonl $X/lcb/$f.jsonl.part $X/lcb/need_ids.json <<'PY' >> $L 2>&1 || { log "filter $f FAILED"; exit 1; }
import json, sys
need = set(json.load(open(sys.argv[3]))); n = k = 0
with open(sys.argv[2], "w") as fh:
    for line in open(sys.argv[1]):
        n += 1
        if json.loads(line)["question_id"] in need:
            fh.write(line); k += 1
print(sys.argv[1].split("/")[-1], "kept", k, "of", n)
PY
  mv $X/lcb/$f.jsonl.part $X/lcb/$f.jsonl; rm -f $X/raw/$f.jsonl
done
rm -rf $X/raw; ln -sf ~/rsi4090/lcb/test5.jsonl ~/rsi4090/lcb/test6.jsonl $X/lcb/
$P - $X <<'PY' >> $L 2>&1 || { log "check FAILED"; exit 1; }
import json, sys
X = sys.argv[1]; b = json.load(open(f"{X}/inx/xq/learn/xl_base.json")); need = set(json.load(open(f"{X}/lcb/need_ids.json")))
got = {json.loads(l)["question_id"] for f in ("test", "test2", "test3", "test4") for l in open(f"{X}/lcb/{f}.jsonl")}
held = []
for f in ("test5", "test6"):
    for l in open(f"{X}/lcb/{f}.jsonl"):
        r = json.loads(l)
        if r["difficulty"] in ("easy", "medium") and r["question_id"] not in {h for h, _ in held}:
            held.append((r["question_id"], (r["contest_date"][:10], r["question_id"])))
held = [h for h, _ in sorted(held, key=lambda x: x[1])]
assert need <= got, f"missing train ids: {sorted(need - got)[:10]}"
assert held == b["held"], f"held pool differs: {len(held)} vs {len(b['held'])}"
print("check ok: train ids", len(need), "found; held pool", len(held), "matches")
PY
cp $X/bench/x184_fragile.json $X/inx/xl_fragile.json || { log "no fragile set in $X/bench: FAILED"; exit 1; }
for f in xlearn.py xbridge.py xcode.py; do [ -f $X/bench/$f ] || { log "no $X/bench/$f: FAILED"; exit 1; }; done
log "lcb $(du -sh $X/lcb | cut -f1) | inputs $(du -sh $X/inx | cut -f1) | free $(df -h --output=avail / | tail -1)"; log "PREP184_DONE"
