"""Addendum 202: per-task comparison of the 4x-corrector D LoRA (x202_D4x_a/b) with 196's arms (plain, D, C) on the same 80 tasks: solves per
task over two attempts, exact sign tests. Runs on box in ~/g4agent/rollouts (stdin)."""
import json, glob
from math import comb
R = {}
for n in ("x202_D4x_a", "x202_D4x_b", "x196_plain_a", "x196_plain_b", "x196_D_a", "x196_D_b", "x196_C_a", "x196_C_b"):
    d = {}
    for f in glob.glob(n + "/res_*/task_results.jsonl"):
        for l in open(f):
            r = json.loads(l); d[r.get("task_id") or r.get("id") or r.get("instance_id")] = bool(r.get("resolved"))
    R[n] = d
ids = sorted(set.union(*(set(d) for d in R.values())))
arm = {a: {t: R[f"{p}_a"].get(t, False) + R[f"{p}_b"].get(t, False) for t in ids} for a, p in (("D4x", "x202_D4x"), ("plain", "x196_plain"), ("D", "x196_D"), ("C", "x196_C"))}
def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
print("tasks", len(ids), "| resolved of", 2 * len(ids), {a: sum(v.values()) for a, v in arm.items()}, "| tasks solved at least once", {a: sum(x > 0 for x in v.values()) for a, v in arm.items()})
for y in ("plain", "D", "C"):
    w = sum(1 for t in ids if arm["D4x"][t] > arm[y][t]); l = sum(1 for t in ids if arm[y][t] > arm["D4x"][t]); print(f"D4x vs {y}: more {w}, fewer {l}, p={sign(w, l):.3f}")
