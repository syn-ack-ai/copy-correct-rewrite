"""Addendum 208: the self-corrector students' LoRAs live on the same 80 tasks as 196, 202 and 207: attempts present, episodes solved, tasks
solved at least once, error endings, and paired comparisons by task on the solve rate (solves / attempts), exact sign tests. Runs on box in
~/g4agent/rollouts (stdin)."""
import json, glob
from math import comb
ARMS = {"plain": ["x196_plain_a", "x196_plain_b", "x207_plain_c", "x207_plain_d"], "null": ["x207_N_a", "x207_N_b"],
        "D": ["x196_D_a", "x196_D_b", "x207_D_c", "x207_D_d"], "C": ["x196_C_a", "x196_C_b"], "D4x": ["x202_D4x_a", "x202_D4x_b"],
        "S": ["x208_S_a", "x208_S_c"], "S4x": ["x208_S4_a", "x208_S4_c"],
        "D + D4x": ["x196_D_a", "x196_D_b", "x207_D_c", "x207_D_d", "x202_D4x_a", "x202_D4x_b"], "S + S4x": ["x208_S_a", "x208_S_c", "x208_S4_a", "x208_S4_c"]}
def load(n):
    d = {}
    for f in glob.glob(n + "/res_*/task_results.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l); d[r.get("task_id") or r.get("id") or r.get("instance_id")] = r
    return d
def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
R = {a: [d for d in (load(n) for n in ns) if len(d) == 80] for a, ns in ARMS.items()}   # only complete attempts (80 task results)
ids = sorted(set().union(*(set(d) for ds in R.values() for d in ds))); rate = {}
for a, ds in R.items():
    if not ds:
        print(f"{a}: no complete attempt"); continue
    ep = [r for d in ds for r in d.values()]; err = [str(r.get("error") or "") for r in ep]; kinds = {}
    for e in err:
        if e:
            k = "timeout" if "timeout" in e.lower() else e[:30]   # review 207: context-window errors end in "Retried: 5 times"
            kinds[k] = kinds.get(k, 0) + 1   # 208: was inside the comment in x207_pertask.py, so the kinds printed empty (the totals were right)
    rate[a] = {t: sum(bool(d.get(t, {}).get("resolved")) for d in ds) / len(ds) for t in ids}
    print(f"{a}: attempts {len(ds)} | solved {sum(bool(r.get('resolved')) for r in ep)}/{len(ep)} | tasks solved at least once {sum(v > 0 for v in rate[a].values())} | error endings {sum(bool(e) for e in err)} {kinds}")
for x, y in (("S", "null"), ("S4x", "null"), ("S + S4x", "null"), ("D", "S"), ("D4x", "S4x"), ("D + D4x", "S + S4x"), ("S", "C"), ("S4x", "C"), ("S", "plain"), ("S4x", "plain"), ("D", "null"), ("D4x", "null")):
    if x in rate and y in rate:
        w = sum(1 for t in ids if rate[x][t] > rate[y][t]); l = sum(1 for t in ids if rate[y][t] > rate[x][t]); print(f"{x} vs {y}: higher solve rate on {w} tasks, lower on {l}, p={sign(w, l):.3f}")
