"""Addendum 206, amendment 206c: the G4 online table read row by row (added after seeds 1-2 at the user's request; the rule for the
three-seed reading was fixed before any seed-3 session started).
1. Closeness, paired on the gate's 240 held rows (cross-entropy to E4B along its reply; the same rows for every student): mean difference
   with 95% bootstrap intervals over rows, over episodes (118) and over repositories (17), and the number of repositories where the first
   student is closer. data/online/gate_rows_groups.json holds each row's episode and repository index.
2. Speed: the first checkpoint at which a student's held CE is at or below the plain teacher's (C) student at item 600, same seed;
   prequential gain per 100-item window.
usage: g4_rows.py [seed ...]   (default 1-5; missing runs are skipped)"""
import json, os, random, sys
_H = os.path.dirname(os.path.abspath(__file__))   # the data folder: $G4_DATA, or data/online beside scripts/ (the public release), or the project's own path
O = os.path.join(os.environ.get("G4_DATA") or next(p for p in (os.path.join(_H, "..", "data", "online"), os.path.expanduser("~/systemone/papers/kaggle_bridge/data/online")) if os.path.isdir(p)), ""); random.seed(0)
G = json.load(open(O + "gate_rows_groups.json"))
A = {"S": ("g4_", "control_keep_replay"), "D": ("g4_", "bridge_keep_replay"), "S4": ("g4_4x_", "control_keep_replay"), "D4": ("g4_4x_", "bridge_keep_replay"),
     "DI": ("g4_imit1_", "bridge_keep_replay"), "C": ("g4_", "direct_keep_replay")}
seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
def path(kind, k, sd):
    pre, a = A[k]; sx = "" if sd == 1 else f"_s{sd}"
    return O + (f"{pre}gate_lora_onl_{a}{sx}_points.json" if kind == "rows" else f"{pre}online_{a}{sx}.json")
def has(k, sd): return os.path.exists(path("rows", k, sd)) and os.path.exists(path("on", k, sd))
def rows(k, sd): return json.load(open(path("rows", k, sd)))["cross-entropy per row"]["LoRA"]
def online(k, sd):
    o = json.load(open(path("on", k, sd)))
    return [w["gain over base"] for w in o["prequential windows (before learning each item)"]], [(x["item"], x["held ce (on)"]) for x in (o.get("held curve (bridge down)") or o.get("held curve"))]
def cboot(d, grp, B=4000):
    by = {}
    for v, g in zip(d, grp): by.setdefault(g, []).append(v)
    ks = list(by); out = []
    for _ in range(B):
        s = n = 0
        for g in (random.choice(ks) for _ in ks): s += sum(by[g]); n += len(by[g])
        out.append(s / n)
    out.sort(); return out[int(.025 * B)], out[int(.975 * B) - 1]
def rep(lab, x, y):
    d = [a - b for a, b in zip(x, y)]; m = sum(d) / len(d); by = {}
    for v, g in zip(d, G["repo"]): by.setdefault(g, []).append(v)
    (l1, h1), (l2, h2), (l3, h3) = cboot(d, list(range(len(d)))), cboot(d, G["episode"]), cboot(d, G["repo"])
    print(f"  {lab:30s} {m:+.4f} | rows [{l1:+.4f}, {h1:+.4f}] | episodes [{l2:+.4f}, {h2:+.4f}] | repos [{l3:+.4f}, {h3:+.4f}] | closer in {sum(1 for g in by if sum(by[g]) < 0)} of {len(by)} repos")
    return m
PAIRS = (("D", "S"), ("D4", "S4"), ("D", "C"), ("S", "C"), ("D4", "D"), ("S4", "S"), ("DI", "D"))
print("closeness on the gate's 240 held rows: first minus second (negative: the first student is closer to E4B)")
signs = {}
for sd in seeds:
    for a, b in PAIRS:
        if has(a, sd) and has(b, sd): signs.setdefault((a, b), []).append(rep(f"seed {sd} {a} vs {b}", rows(a, sd), rows(b, sd)))
for a, b in PAIRS:
    sds = [sd for sd in seeds if has(a, sd) and has(b, sd)]
    if len(sds) > 1:
        rep(f"seeds {sds} mean {a} vs {b}", [sum(v) / len(v) for v in zip(*(rows(a, sd) for sd in sds))], [sum(v) / len(v) for v in zip(*(rows(b, sd) for sd in sds))])
cs = [(a, b, sd) for a, b in (("D", "S"), ("D4", "S4")) for sd in seeds if has(a, sd) and has(b, sd)]
if cs:
    neg = sum(1 for a, b in (("D", "S"), ("D4", "S4")) for m in signs.get((a, b), []) if m < 0)
    rep(f"cross vs self, {len(cs)} pairs mean", [sum(v) / len(v) for v in zip(*(rows(a, sd) for a, b, sd in cs))], [sum(v) / len(v) for v in zip(*(rows(b, sd) for a, b, sd in cs))])
    from math import comb
    k_ = min(neg, len(cs) - neg); print(f"  cross closer in {neg} of {len(cs)} pairs (exact two-sided sign test p = {min(1.0, 2 * sum(comb(len(cs), i) for i in range(k_ + 1)) / 2 ** len(cs)):.3f})")
print("\nspeed: prequential gain per 100-item window; first checkpoint at or below C's held CE at item 600 (same seed)")
for sd in seeds:
    cf = online("C", sd)[1][-1][1] if has("C", sd) else None
    for k in ("C", "S", "D", "DI", "S4", "D4"):
        if not has(k, sd): continue
        g, c = online(k, sd); first = next((i for i, v in c if v <= cf), None) if cf is not None and k != "C" else None
        print(f"  seed {sd} {k:2s} windows {' '.join('%.3f' % v for v in g)} | mean {sum(g) / len(g):.4f} | held CE at 50/100/150/600: {' '.join('%.4f' % dict(c)[i] for i in (50, 100, 150, 600))}" + (f" | matches C's final {cf:.4f} at item {first}" if first is not None else ""))
