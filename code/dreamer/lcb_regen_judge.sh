#!/bin/bash
# Amendment 209i: judge one resampling batch. For each regenerated row the three shortest versions that pass their tests and the rules of
# form are read by DeepSeek: directness first (two votes), then length for those direct in both (two votes), then cleanness with 209g's
# text for those of reasonable length in both (two votes). The votes are merged into one file per batch (verdicts2_TAG.jsonl), in which a
# version is clean only if all six votes say so.   lcb_regen_judge.sh TAG FIX_OUT.jsonl     (TAG r1: the rambling batch; r2: the overlong batch)
# Then, once: lcb_regen_judge.sh final   -> xl_rows_fit.jsonl (lcb_fit_rows.py final over both batches) and the sanity check.
K=~/systemone/kaggle_g4agent; W=$K/runs/xql/own209i; J="python3 $K/dreamer/judge_rows.py"; F="python3 $K/dreamer/lcb_fit_rows.py"; cd $W || exit 1
if [ "$1" = final ]; then
  python3 - <<'EOF' || exit 1
import json, collections
rows = [json.loads(l) for l in open("xl_rows_pool.jsonl")]
def votes(f):
    V = collections.defaultdict(dict)
    for v in map(json.loads, open(f)):
        if v.get("verdict"): V[v["rid"]][v["vote"]] = v["verdict"]
    return V
D, L = votes("verdicts1.jsonl"), votes("verdicts_len.jsonl"); n = collections.Counter()
with open("verdicts_first.jsonl", "w") as fh:                          # two synthetic votes per row: not_clean when both votes of a reading flag it
    for r in rows:
        ramb = len(D[r["rid"]]) >= 2 and all(x == "not_clean" for x in D[r["rid"]].values()); long_ = len(L.get(r["rid"], {})) >= 2 and all(x == "not_clean" for x in L[r["rid"]].values())
        assert len(D[r["rid"]]) >= 2 and (ramb or len(L.get(r["rid"], {})) >= 2), r["rid"]
        v = "not_clean" if (ramb or long_) else "clean"; n["rambling" if ramb else "overlong" if long_ else "kept"] += 1
        for k in (0, 1):
            fh.write(json.dumps({"rid": r["rid"], "vote": k, "verdict": v, "issues": [], "reason": "rambling in both votes" if ramb else "overlong in both votes" if long_ else ""}) + "\n")
print("first reading, combined:", dict(n))
EOF
  args=""; for t in r1 r2 best; do [ -s verdicts2_$t.jsonl ] && args="$args $t verdicts2_$t.jsonl"; done   # best: the judge's pick of 16 (lcb_best16.py)
  $F final . verdicts_first.jsonl $args | tail -40; exit 0
fi
TAG=$1; FX=$2; [ -s "$FX" ] || { echo "no fix_out at $FX"; exit 1; }
$F pool2 . $FX $TAG | cut -c1-500 || exit 1
python3 - $TAG <<'EOF' || exit 1
import json, sys, collections
t = sys.argv[1]; p = f"xl_rows_pool2_{t}.jsonl"; rows = [json.loads(l) for l in open(p)]; by = collections.defaultdict(list)
for r in rows: by[r["of"]].append(r)
keep = [r for v in by.values() for r in sorted(v, key=lambda r: len(r["reply"]))[:3]]
with open(p, "w") as fh:
    for r in keep: fh.write(json.dumps(r) + "\n")
print(f"batch {t}: {len(rows)} passing versions of {len(by)} rows; the three shortest per row: {len(keep)}")
EOF
stage() {   # FLAG OUT IN: judge IN with FLAG twice (errors retried), print the count clean in both votes, write the rows that are to NEXT
  local flag=$1 out=$2 in=$3 next=$4; rm -f $out
  $J run $in --out $out $flag --votes 2 --cap 3 --workers 24 | tail -1 | cut -c1-160; $J run $in --out $out $flag --votes 2 --cap 3 --workers 12 > /dev/null
  python3 - $in $out $next <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]; V = collections.defaultdict(dict)
for v in map(json.loads, open(sys.argv[2])):
    if v.get("verdict"): V[v["rid"]][v["vote"]] = v["verdict"]
miss = [r["rid"] for r in rows if len(V[r["rid"]]) < 2]; assert not miss, f"{len(miss)} versions without both votes"
ok = [r for r in rows if all(x == "clean" for x in V[r["rid"]].values())]
with open(sys.argv[3], "w") as fh:
    for r in ok: fh.write(json.dumps(r) + "\n")
print(f"   {sys.argv[2]}: {len(ok)} of {len(rows)} versions pass in both votes ({len({r['of'] for r in ok})} rows)")
EOF
}
stage --direct  vd_$TAG.jsonl xl_rows_pool2_$TAG.jsonl pass_d_$TAG.jsonl || exit 1
stage --concise vc_$TAG.jsonl pass_d_$TAG.jsonl        pass_c_$TAG.jsonl || exit 1
stage ""        vk_$TAG.jsonl pass_c_$TAG.jsonl        pass_k_$TAG.jsonl || exit 1
python3 - $TAG <<'EOF'
import json, sys, collections
t = sys.argv[1]; V = collections.defaultdict(list); cost = 0.0
for f in (f"vd_{t}.jsonl", f"vc_{t}.jsonl", f"vk_{t}.jsonl"):
    for v in map(json.loads, open(f)):
        cost += float(v.get("cost") or 0)
        if v.get("verdict"): V[v["rid"]].append(v)
with open(f"verdicts2_{t}.jsonl", "w") as fh:
    for rid, vs in V.items():
        for k, v in enumerate(vs): fh.write(json.dumps({**v, "vote": k}) + "\n")
acc = [rid for rid, vs in V.items() if len(vs) == 6 and all(v["verdict"] == "clean" for v in vs)]
print(f"batch {t}: versions accepted in all six votes: {len(acc)} | rows with an accepted version: {len({r.split('#')[0] for r in acc})} | judge cost ${cost:.3f}")
EOF
