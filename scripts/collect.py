"""Collect every number the paper uses from the run outputs into data/results.json (standard library only).
  python collect.py RUNS_DIR OUT_JSON
RUNS_DIR = kaggle_g4agent/runs: xqlB (rows, baselines), xqlO1-3 (179), xqlCF1-3 (181), xqlRB0-3 (182), xqlRBC1-3 (184c, when done),
xqlS1-3 (174), xqw1-3 (178), xqd1-3 (169d); the online test files are read from data/online/."""
import json, statistics, sys
from pathlib import Path

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); ONL = OUT.parent / "online"
base = json.load(open(R / "xqlB/xl_base.json")); rows = set(base["rows"]); held = base["held"]


def counts(pp, arm, rws=None):
    rws = rows if rws is None else rws   # 187: the E2B-teaches-E4B direction trains on E2B's own passing problems
    q, r = pp["qwen"], pp[arm]; trained = [i for i in q if i in rws]; kp = [i for i in q if i not in rws and i not in held and q[i]]
    return {"fixed": sum(1 for i in trained if not q[i] and r[i]), "fixed of": sum(1 for i in trained if not q[i]),
            "new": sum(1 for i in held if not q[i] and r[i]), "new of": sum(1 for i in held if not q[i]),
            "held solved": sum(1 for i in held if r[i]), "held of": len(held),
            "lost held": sum(1 for i in held if q[i] and not r[i]), "lost train pool": sum(1 for i in trained + kp if q[i] and not r[i]),
            "qwen held": sum(1 for i in held if q[i]), "qwen train pool": sum(1 for i in trained + kp if q[i])}


def family(files_by_seed, rws=None):
    out = {"per seed": {}, "offtask": {}, "e4b held CE": {}}
    for s, files in files_by_seed.items():
        pp, off, ce = {}, {}, {}
        for f in files:
            if not f.exists():
                continue
            d = json.load(open(f))
            for a, v in d["per problem"].items():
                if a in pp:
                    assert a not in ("qwen", "sft") or pp[a] == v, f"{a} differs across sessions in {f}"
                pp.setdefault(a, v)
            for part, v in d["off-task (kept knowledge)"].items():
                for m, arms in v.items():
                    if m != "rows":
                        for a, x in arms.items():
                            off.setdefault(f"{part} | {m}", {}).setdefault(a, x)
            for a, x in d["held: cross-entropy of E4B's passing solutions (lower = closer to E4B)"].items():
                if a != "rows":
                    ce.setdefault(a, x)
        if pp:
            out["per seed"][s] = {a: counts(pp, a, rws) for a in pp if a != "qwen"}; out["offtask"][s] = off; out["e4b held CE"][s] = ce
    arms = sorted({a for v in out["per seed"].values() for a in v})
    if not arms:
        out["pooled"] = {}; return out
    out["pooled"] = {a: {k: sum(out["per seed"][s][a][k] for s in out["per seed"] if a in out["per seed"][s]) for k in next(iter(out["per seed"].values()))["sft"]}
                     | {"seeds": sum(1 for s in out["per seed"] if a in out["per seed"][s])} for a in arms}
    return out


res = {"main": family({s: [R / f"xqlO{s}/xl_eval_s{s}.json", R / f"xqlCF{s}/xl_eval_cf_s{s}.json", R / f"xqlRB{s}/xl_eval_rb_s{s}.json", R / f"xqlRBC{s}/xl_eval_rbc_s{s}.json", R / f"xqlSS{s}/xl_eval_sfself_s{s}.json"] for s in (1, 2, 3)}),
       "f174": family({s: [R / f"xqlS{s}/xl_eval_s{s}.json"] for s in (1, 2, 3)})}
E2B = family({s: [R / f"xqlE2B{s}/xl_eval_e2b_s{s}.json"] for s in (1, 2, 3)})   # 186: Gemma 4 E2B as the student ("qwen" = the student alone)
if E2B["per seed"]:
    res["e2b"] = E2B; res["e2b drift"] = {}
    for s in (1, 2, 3):
        f = R / f"xqlE2B{s}/drift_e2b_s{s}.json"
        if f.exists():
            res["e2b drift"].update(json.load(open(f))["summary"])
    res["e2b runbook build"] = {s: json.load(open(R / f"xqlE2B{s}/xl_rb.json")) for s in (1, 2, 3) if (R / f"xqlE2B{s}/xl_rb.json").exists()}



def fit_summary(log):
    """a corrector fit log (xbridge.py train): mean cross-entropy with and without the corrector on its own fitting batches over the last 100
    steps (the 25-step progress lines from step 500), and the mean size of its change there"""
    import re
    b, a, z = [], [], []
    for line in open(log, errors="replace"):
        m = re.search(r"step (\d+) cross-entropy \(bridged\) ([\d.]+) \(alone ([\d.]+) \) change size ([\d.]+)", line)
        if m and int(m.group(1)) >= 500:
            b.append(float(m.group(2))); a.append(float(m.group(3))); z.append(float(m.group(4)))
    return {"with": round(statistics.mean(b), 4), "alone": round(statistics.mean(a), 4), "change size": round(statistics.mean(z), 4), "lines": len(b)} if b else None


# 187: two-way E4B <-> E2B with each seed's freshly fitted 600-step correctors; Qwen's per-seed correctors (same recipe: size 0.03, harm 0.5) for comparison
QF = {}
for s, d_ in ((1, "xqr2"), (2, "xqr2"), (3, "xqr3")):
    for kind in ("control", "bridge"):
        f = R / d_ / f"xs_train_{kind}_s0.03_h0.5_r{s}.log"
        if f.exists():
            QF[f"{kind}|{s}"] = fit_summary(f)
res["qwen corrector fit"] = QF
for d_, pre in (("e2b", "xqlTW2b"), ("e4b", "xqlTW4b")):
    rws = None
    if d_ == "e4b":   # rows = E2B's own passing answers (build_from_gens), the same for every seed
        bf = [R / f"{pre}{s}/xl_base_from_e2b.json" for s in (1, 2, 3) if (R / f"{pre}{s}/xl_base_from_e2b.json").exists()]
        rws = set(json.load(open(bf[0]))["rows"]) if bf else None
    if d_ == "e4b" and rws is None:
        continue
    fam = family({s: [R / f"{pre}{s}/xl_eval_two_{d_}_s{s}.json"] for s in (1, 2, 3)}, rws)
    if not fam["per seed"]:
        continue
    res[f"two {d_}"] = fam; res[f"two {d_} drift"] = {}; res[f"two {d_} fit"] = {}
    if rws is not None:
        res[f"two {d_}"]["rows"] = len(rws)
    for s in (1, 2, 3):
        f = R / f"{pre}{s}/drift_two_{d_}_s{s}.json"
        if f.exists():
            res[f"two {d_} drift"].update(json.load(open(f))["summary"])
        for kind in ("control", "bridge"):
            f = R / f"{pre}{s}/xl_translator_{kind}.log"
            if f.exists():
                res[f"two {d_} fit"][f"{kind}|{s}"] = fit_summary(f)

# 178: wide LiveCodeBench, the agent-row LoRAs
W = {}
for s in (1, 2, 3):
    d = json.load(open(R / f"xqw{s}/xw_results_s{s}.json")); W[s] = {"problems": d["problems"], "passed": {k: v[0] for k, v in d["passed"].items()}, "pairs": d["paired over all problems (a only, b only, exact p)"]}
res["wide"] = W

# fragile set: how often each problem Qwen alone solves is lost (8 arms x 3 seeds), and why
arms8 = ["sft", "sft_s", "sft_c_s", "opd_c", "mix_c", "opd_cf", "rw_cf", "mix_cf"]; PP, TX = {}, {}
for s in (1, 2, 3):
    for f, d_ in ((R / f"xqlO{s}/xl_eval_s{s}.json", R / f"xqlO{s}"), (R / f"xqlCF{s}/xl_eval_cf_s{s}.json", R / f"xqlCF{s}")):
        for a, v in json.load(open(f))["per problem"].items():
            if (a, s) not in PP:
                PP[(a, s)] = v
                g = d_ / f"xl_gens_{a}.jsonl"
                if g.exists():
                    TX[(a, s)] = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(g)}
q = PP[("qwen", 1)]; qs = [i for i in q if q[i]]; cnt = {i: sum(1 for a in arms8 for s in (1, 2, 3) if not PP[(a, s)][i]) for i in qs}
lost_len, unf, own_len = [], 0, []
for a in arms8:
    for s in (1, 2, 3):
        for i in qs:
            if not PP[(a, s)][i]:
                t = TX[(a, s)].get(i, ""); lost_len.append(len(t)); unf += "```" not in t; own_len.append(len(TX[("qwen", 1)].get(i, "")))
qt = TX[("qwen", 1)]
res["fragile"] = {"histogram (lost in k of 24 runs)": {k: sum(1 for v in cnt.values() if v == k) for k in range(25)}, "qwen solved": len(qs),
                  "lost output chars": lost_len, "qwen passing chars": [len(qt[i]) for i in qs if i in qt], "qwen own chars on the lost cases": own_len, "lost unfenced": unf, "lost cases": len(lost_len),
                  "fragile (>= 6) trained rows": sum(1 for i in qs if cnt[i] >= 6 and i in rows), "fragile (>= 6)": sum(1 for i in qs if cnt[i] >= 6)}

# output length per arm (seed 1 generations on all 495 problems)
res["lengths"] = {a: statistics.median(len(t) for t in TX[(a, 1)].values()) for a in ["qwen"] + arms8 if (a, 1) in TX}
for a, f in (("rb", R / "xqlRB1/xl_gens_rb.jsonl"), ("rb_c", R / "xqlRBC1/xl_gens_rb_c.jsonl")):
    if f.exists():
        res["lengths"][a] = statistics.median(len(json.loads(l)["text"]) for l in open(f))

# teachers of the on-policy arms: CE of E4B's held passing solutions (181 gate)
res["teachers"] = {s: json.load(open(R / f"xqlCF{s}/xl_cf_gate.json"))["held, E4B's passing solutions"]["means (alone, old control, refit)"] for s in (1, 2, 3)}
res["runbook build"] = json.load(open(R / "xqlRB0/xl_rb.json"))
res["rewrite build"] = {s: json.load(open(R / f"xqlCF{s}/xl_rw.json")) for s in (1, 2, 3)}

# 169d: agent replies (E4B -> Qwen), the corrector-trained forms against dose-matched SFT
A = {}
for s in (1, 2, 3):
    d = json.load(open(R / f"xqd{s}/xdb_eval_s{s}.json")); A[s] = {k: v["means"] for k, v in d["sets"].items() if "means" in v}
res["agent169d"] = A

# online test (170)
ONLA = [a for a in ("direct", "bridge", "direct_keep_replay", "bridge_keep_replay") if (ONL / f"online_{a}.json").exists()]   # 183: D = bridge_keep_replay
res["online"] = {a: {"run": json.load(open(ONL / f"online_{a}.json")), "gate": json.load(open(ONL / f"gate_lora_onl_{a}.json"))} for a in ONLA}
# 189: per-decision gate output (re-gated, identical totals): fixed / broken / net per setup, exact sign tests of D against each
PK = "decision points (matches the teacher, probability of its choice, student alone differs)"
PTS = {a: json.load(open(ONL / f"gate_lora_onl_{a}_points.json"))[PK] for a in ONLA if (ONL / f"gate_lora_onl_{a}_points.json").exists()}
if PTS:
    from math import comb
    def _sign(w, l):
        n = w + l
        return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
    al = next(iter(PTS.values()))["E4B alone"]; assert all(v["E4B alone"] == al for v in PTS.values()); dif = [i for i, x in enumerate(al) if x[2]]; same = [i for i, x in enumerate(al) if not x[2]]
    G = {a: {"fixed": sum(v["LoRA"][i][0] for i in dif), "of differing": len(dif), "broken": sum(1 - v["LoRA"][i][0] for i in same), "of agreeing": len(same),
             "matches": sum(x[0] for x in v["LoRA"]), "points": len(al)} for a, v in PTS.items()}
    for g in G.values():
        g["net"] = g["fixed"] - g["broken"]
    if "bridge_keep_replay" in PTS:
        Dl = PTS["bridge_keep_replay"]["LoRA"]
        for a, v in PTS.items():
            if a != "bridge_keep_replay":
                X_ = v["LoRA"]; w52 = sum(1 for i in dif if Dl[i][0] and not X_[i][0]); l52 = sum(1 for i in dif if X_[i][0] and not Dl[i][0])
                w = sum(1 for i in range(len(Dl)) if Dl[i][0] and not X_[i][0]); l = sum(1 for i in range(len(Dl)) if X_[i][0] and not Dl[i][0])
                G[a]["D vs this"] = {"differing: D only, this only, p": [w52, l52, round(_sign(w52, l52), 4)], "all: D only, this only, p": [w, l, round(_sign(w, l), 4)]}
    res["online gate points"] = G
for f in sorted(OUT.parent.glob("drift*.json")):   # review D1: drift from the original student (scripts/drift.py); every drift file of the paper
    if f.exists():
        res.setdefault("drift", {}).update(json.load(open(f))["summary"])
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT, "| main arms:", sorted(res["main"]["pooled"]), "| 174 arms:", sorted(res["f174"]["pooled"]))
