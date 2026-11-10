"""The live test's table and by-task comparisons from data/live/live80_attempts.json (made by live_export.py on the rollout host).
Per arm: attempts, episodes solved, tasks solved at least once, error endings (time-outs), model calls per task. Comparisons are by task on
the solve rate (solves / attempts), with exact two-sided sign tests, as in the pre-registered analysis (addenda 207, 208).
  python live_table.py [JSON]"""
import json, sys
from math import comb
from pathlib import Path
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data/live/live80_attempts.json"))["arms"]
def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
rate = {}; ids = sorted({t for a in D.values() for d in a.values() for t in d})
print(f"{'arm':24s} attempts  solved      tasks  errors (time-outs)  calls/task  solved per attempt")
for arm, att in D.items():
    ep = [r for d in att.values() for r in d.values()]; rate[arm] = {t: sum(d[t]["resolved"] for d in att.values()) / len(att) for t in ids}
    print(f"{arm:24s} {len(att):8d}  {sum(r['resolved'] for r in ep):3d} of {len(ep):3d}  {sum(v > 0 for v in rate[arm].values()):5d}  {sum(bool(r['error']) for r in ep):6d} ({sum(r['error'] == 'timeout' for r in ep):3d})"
          f"        {sum(r['calls'] for r in ep) / len(ep):6.1f}      {', '.join(str(sum(r['resolved'] for r in d.values())) for d in att.values())}")
def pool(name, arms):
    arms = [a for a in arms if a in D]
    if arms:
        att = [d for a in arms for d in D[a].values()]; rate[name] = {t: sum(d[t]["resolved"] for d in att) / len(att) for t in ids}
pool("D + D4", ["D LoRA (4090 seed 1)", "D4 LoRA (A100 seed 1)"]); pool("S + S4", ["S LoRA (4090 seed 1)", "S4 LoRA (G4 seed 1)"])
pool("D, both adapters", ["D LoRA (4090 seed 1)", "D LoRA (A100 seed 1)"]); pool("C, both adapters", ["C LoRA (4090 seed 1)", "C LoRA (A100 seed 1)"])
print("\nby task: higher solve rate on / lower on (exact sign test)")
for x, y in (("D LoRA (4090 seed 1)", "null adapter"), ("D LoRA (4090 seed 1)", "plain E2B"), ("D LoRA (4090 seed 1)", "C LoRA (4090 seed 1)"), ("D4 LoRA (A100 seed 1)", "null adapter"),
             ("D4 LoRA (A100 seed 1)", "D LoRA (4090 seed 1)"), ("C LoRA (4090 seed 1)", "null adapter"), ("null adapter", "plain E2B"), ("D LoRA (A100 seed 1)", "C LoRA (A100 seed 1)"),
             ("D, both adapters", "C, both adapters"), ("S LoRA (4090 seed 1)", "null adapter"), ("S4 LoRA (G4 seed 1)", "null adapter"), ("S + S4", "null adapter"),
             ("D LoRA (4090 seed 1)", "S LoRA (4090 seed 1)"), ("D4 LoRA (A100 seed 1)", "S4 LoRA (G4 seed 1)"), ("D + D4", "S + S4")):
    if x in rate and y in rate:
        w = sum(rate[x][t] > rate[y][t] for t in ids); l = sum(rate[y][t] > rate[x][t] for t in ids); print(f"  {x} vs {y}: {w} vs {l}, p = {sign(w, l):.4f}")
