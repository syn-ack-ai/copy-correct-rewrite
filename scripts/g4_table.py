"""Score the addendum-206 G4 runs on the 4090 partition (52 differing / 168 agreeing decisions).
usage: g4score.py [seed ...]   (default: every seed that has files)"""
import json, os, sys
from math import comb
_H = os.path.dirname(os.path.abspath(__file__))   # the data folder: $G4_DATA, or data/online beside scripts/ (the public release), or the project's own path
O = os.path.join(os.environ.get("G4_DATA") or next(p for p in (os.path.join(_H, "..", "data", "online"), os.path.expanduser("~/systemone/papers/kaggle_bridge/data/online")) if os.path.isdir(p)), "")
PK = "decision points (matches the teacher, probability of its choice, student alone differs)"
def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
al = json.load(open(O + "gate_lora_onl_bridge_keep_replay_points.json"))[PK]["E4B alone"]
dif = [i for i, x in enumerate(al) if x[2]]; same = [i for i, x in enumerate(al) if not x[2]]
ARMS = [("S", "g4_", "control_keep_replay"), ("D", "g4_", "bridge_keep_replay"), ("S4", "g4_4x_", "control_keep_replay"),
        ("D4", "g4_4x_", "bridge_keep_replay"), ("DI", "g4_imit1_", "bridge_keep_replay"), ("C", "g4_", "direct_keep_replay")]
seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
L = {}; R = {}; SO = {s: 0 if s == 1 else 1000 * s for s in range(1, 10)}
for sd in seeds:
    sx = "" if sd == 1 else f"_s{sd}"
    for k, pre, a in ARMS:
        fp = O + f"{pre}gate_lora_onl_{a}{sx}_points.json"
        if not os.path.exists(fp): continue
        p = json.load(open(fp))[PK]; on = json.load(open(O + f"{pre}online_{a}{sx}.json")); c = on.get("held curve (bridge down)") or on.get("held curve"); X = p["LoRA"]
        g = [x["gain over base"] for x in on["prequential windows (before learning each item)"]]; f = sum(X[i][0] for i in dif); b = sum(1 - X[i][0] for i in same)
        gg = json.load(open(O + f"{pre}gate_lora_onl_{a}{sx}.json")); ce = [v for kk, v in gg.items() if kk.startswith("cross-entropy to the 31B along")][0]
        assert gg["lora"]["seed offset"] == on["seed offset"] == SO[sd], (k, sd, gg["lora"]["seed offset"], on["seed offset"])
        L[(k, sd)] = X; R[(k, sd)] = dict(f=f, b=b, net=f - b, held=c[-1]["held ce (on)"], preq=sum(g) / len(g))
        print(f"s{sd} {k:3s} fixed {f:2d} broken {b:2d} net {f - b:+3d} | held CE {c[-1]['held ce (on)']:.4f} drift {c[-1]['KL from base on untaught prose / code / story']:.4f} qdrift {c[-1].get('KL from base on held questions (never kept)'):.4f} | preq {sum(g) / len(g):.4f} | gate CE {ce.get('LoRA')} | flips {sum(1 for i in range(220) if p['E4B alone'][i][0] != al[i][0])} | teacher {on['teacher']} items {on['items']}")
def pair(a, b, sd):
    if (a, sd) not in L or (b, sd) not in L: return
    A, B = L[(a, sd)], L[(b, sd)]; w = sum(1 for i in range(220) if A[i][0] and not B[i][0]); l = sum(1 for i in range(220) if B[i][0] and not A[i][0])
    w5 = sum(1 for i in dif if A[i][0] and not B[i][0]); l5 = sum(1 for i in dif if B[i][0] and not A[i][0])
    print(f"  s{sd} {a} vs {b}: all {w} vs {l} p={sign(w, l):.3f} | differing {w5} vs {l5} | held CE diff {R[(a, sd)]['held'] - R[(b, sd)]['held']:+.4f} | net diff {R[(a, sd)]['net'] - R[(b, sd)]['net']:+d}")
for sd in seeds:
    for a, b in [("D", "S"), ("D4", "S4"), ("S4", "S"), ("D4", "D"), ("DI", "D"), ("DI", "S"), ("D", "C"), ("S", "C"), ("D4", "C"), ("S4", "C"), ("DI", "C")]: pair(a, b, sd)
for k, _, _ in ARMS:
    v = [R[(k, sd)] for sd in seeds if (k, sd) in R]
    if len(v) > 1: print(f"mean {k:3s} n={len(v)} net {sum(x['net'] for x in v) / len(v):+.1f} ({', '.join('%+d' % x['net'] for x in v)}) held {sum(x['held'] for x in v) / len(v):.4f} preq {sum(x['preq'] for x in v) / len(v):.4f}")
