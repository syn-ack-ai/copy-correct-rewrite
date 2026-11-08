"""Graft replication summary (addenda 169c and the dose-matched baselines): per seed, from RUNS/xqd{1,2,3}/xdb_eval_s{N}.json (every form and
baseline on identical rows): held agent cross-entropy for every arm, the bridge-form retention of each pair (90% bootstrap), and the gap of
the bridge forms to the dose-matched plain-distillation LoRAs; then means over seeds.   python graft_agg.py RUNS_DIR"""
import json, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]); rows = {}
for s in (1, 2, 3):
    f = R / f"xqd{s}" / f"xdb_eval_s{s}.json"
    if not f.exists():
        print("seed", s, "missing"); continue
    d = json.load(open(f))["sets"]
    for name in ("held agent", "train agent", "held question_fresh"):
        if name not in d:
            continue
        r = d[name]; m = r["means"]
        print(f"seed {s} {name}: live lead {r['live lead']}; " + " | ".join(f"{k} {v:.3f}" for k, v in sorted(m.items())))
        for pr, v in r["pairs"].items():
            print(f"    {pr}: lead {v['lead']:+.4f} retention {v['retention [mean, 5%, 95%]']}")
            rows.setdefault((name, pr), []).append(v["retention [mean, 5%, 95%]"][0])
        if r.get("p(true)"):
            print("    p(true): " + " | ".join(f"{k} {v:.3f}" for k, v in sorted(r["p(true)"].items())))
        for b in ("step2 seq_d1000 LoRA", "step2 seqdense_d1000 LoRA", "step2 seq LoRA"):
            if b in m:
                for fm in ("graft_bridge_w0", "lora_bridge_w0", "lora_bridge_w1"):
                    if fm in m:
                        rows.setdefault((name, f"{fm} minus {b}"), []).append(m[fm] - m[b])
print("\nmeans over seeds:")
for (name, k), v in sorted(rows.items()):
    v = [x for x in v if x is not None]
    if v:
        print(f"  {name} | {k}: mean {np.mean(v):+.4f} over {len(v)} seeds {[round(x, 4) for x in v]}")
