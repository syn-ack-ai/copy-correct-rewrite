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

# 190: the multi-task control (E2B; sft_mt_s, sft_self_mt_s; each session re-scores 186's sft)
MT = family({s: [R / f"xqlMT{s}/xl_eval_e2b_s{s}.json"] for s in (1, 2, 3)})
if MT["per seed"]:
    res["mt e2b"] = MT; res["mt e2b drift"] = {}
    for s in (1, 2, 3):
        f = R / f"xqlMT{s}/drift_e2b_s{s}.json"
        if f.exists():
            res["mt e2b drift"].update({k: v for k, v in json.load(open(f))["summary"].items() if k.split("|")[0] in ("sft_mt_s", "sft_self_mt_s")})

# 188: the matched-loss test (E2B; sft_es stops at the running CE 0.24, sft_lr at a tenth of the lr; each session re-scores 186's sft)
ML = family({s: [R / f"xqlML{s}/xl_eval_e2b_s{s}.json"] for s in (1, 2, 3)})
if ML["per seed"]:
    res["ml e2b"] = ML; res["ml e2b drift"] = {}
    for s in (1, 2, 3):
        f = R / f"xqlML{s}/drift_e2b_s{s}.json"
        if f.exists():
            res["ml e2b drift"].update({k: v for k, v in json.load(open(f))["summary"].items() if k.split("|")[0] in ("sft_es", "sft_lr")})

# 193: the KL strength sweep (E2B; sft_self_s at lambda 0.3 and 3; lambda 1 is 187's sft_self_s; each session re-scores 186's sft)
KL = family({s: [R / f"xqlKL{s}/xl_eval_e2b_s{s}.json"] for s in (1, 2, 3)})
if KL["per seed"]:
    res["kl e2b"] = KL; res["kl e2b drift"] = {}
    for s in (1, 2, 3):
        f = R / f"xqlKL{s}/drift_e2b_s{s}.json"
        if f.exists():
            res["kl e2b drift"].update({k: v for k, v in json.load(open(f))["summary"].items() if k.split("|")[0] in ("sft_self_s_l03", "sft_self_s_l3")})

# 194a: the capped lost answers of the 11 Qwen recipes, continued by 768 tokens (pooled over seeds)
RG = {s: json.load(open(R / f"xqlRG{s}/xl_regrade_s{s}.json")) for s in (1, 2, 3) if (R / f"xqlRG{s}/xl_regrade_s{s}.json").exists()}
if RG:
    res["regrade"] = {a: {k: sum(RG[s]["arms"][a][k] for s in RG) for k in ("lost", "capped", "closed", "recovered")} for a in RG[min(RG)]["arms"]}

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

# 181: the corrector teachers run live (greedy, the translator on Qwen's own states, no E4B), from the rewrite stage of each seed
res["teachers live"] = {s: dict(zip(("refit", "old", "alone"), json.load(open(R / f"xqlCF{s}/xl_rw.json"))["teacher greedy, held (refit, old, Qwen alone)"])) for s in (1, 2, 3) if (R / f"xqlCF{s}/xl_rw.json").exists()}

# online test (170)
ONLA = [a for a in ("direct", "bridge", "direct_keep_replay", "bridge_keep_replay", "bridge_keep_replay_s2", "control_keep_replay", "blend_keep_replay") if (ONL / f"online_{a}.json").exists()]   # 183: D = bridge_keep_replay; 189: D seed 2, the self-corrector control
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
    if "bridge_keep_replay_s2" in PTS:   # 189: D seed 2 against the other setups too
        D2 = PTS["bridge_keep_replay_s2"]["LoRA"]
        for a, v in PTS.items():
            if a not in ("bridge_keep_replay", "bridge_keep_replay_s2"):
                X_ = v["LoRA"]; w = sum(1 for i in range(len(D2)) if D2[i][0] and not X_[i][0]); l = sum(1 for i in range(len(D2)) if X_[i][0] and not D2[i][0])
                G[a]["D seed 2 vs this"] = {"all: D only, this only, p": [w, l, round(_sign(w, l), 4)]}
    res["online gate points"] = G
# 206 / 206c: the online teachers on one GPU type (G4), every seed present: the table, the paired row-level and per-decision tests of the
# cross corrector against the self-corrector, and the speed measure (first checkpoint at or below the plain teacher's final held CE)
G4A = {"C": ("g4_", "direct_keep_replay"), "S": ("g4_", "control_keep_replay"), "D": ("g4_", "bridge_keep_replay"), "DI": ("g4_imit1_", "bridge_keep_replay"),
       "S4": ("g4_4x_", "control_keep_replay"), "D4": ("g4_4x_", "bridge_keep_replay")}
if (ONL / "gate_rows_groups.json").exists() and (ONL / "gate_lora_onl_bridge_keep_replay_points.json").exists():
    import random
    from math import comb
    def _sg(w, l):
        n = w + l
        return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
    GRP = json.load(open(ONL / "gate_rows_groups.json"))["repo"]; al4 = json.load(open(ONL / "gate_lora_onl_bridge_keep_replay_points.json"))[PK]["E4B alone"]   # the 4090 partition
    dif4 = [i for i, x in enumerate(al4) if x[2]]; same4 = [i for i, x in enumerate(al4) if not x[2]]; T = {}; ROWS = {}; DEC = {}
    for k, (pre, a) in G4A.items():
        for sd in (1, 2, 3):
            sx = "" if sd == 1 else f"_s{sd}"; fp, fo, fg = (ONL / f"{pre}gate_lora_onl_{a}{sx}_points.json", ONL / f"{pre}online_{a}{sx}.json", ONL / f"{pre}gate_lora_onl_{a}{sx}.json")
            if not (fp.exists() and fo.exists() and fg.exists()):
                continue
            P = json.load(open(fp)); on = json.load(open(fo)); c = on.get("held curve (bridge down)") or on.get("held curve"); X = P[PK]["LoRA"]
            g = [w["gain over base"] for w in on["prequential windows (before learning each item)"]]; f = sum(X[i][0] for i in dif4); b = sum(1 - X[i][0] for i in same4)
            ROWS[(k, sd)] = P["cross-entropy per row"]["LoRA"]; DEC[(k, sd)] = [x[0] for x in X]
            T.setdefault(k, {})[str(sd)] = {"held ce": c[-1]["held ce (on)"], "gain": round(sum(g) / len(g), 4), "windows": g, "fixed": f, "broken": b, "net": f - b,
                                             "drift": c[-1]["KL from base on untaught prose / code / story"], "drift, mean of last six": round(sum(e["KL from base on untaught prose / code / story"] for e in c[-6:]) / 6, 4),
                                             "gate ce": round(sum(ROWS[(k, sd)]) / len(ROWS[(k, sd)]), 4), "curve": [[e["item"], e["held ce (on)"]] for e in c],
                                             "hardware flips": sum(1 for i in range(len(al4)) if P[PK]["E4B alone"][i][0] != al4[i][0])}
    for sd in ("1", "2", "3"):   # speed: the first checkpoint at or below C's held CE at item 600, same seed
        if sd in T.get("C", {}):
            for k in T:
                if k != "C" and sd in T[k]:
                    T[k][sd]["item reaching C's final"] = next((i for i, v in T[k][sd]["curve"] if v <= T["C"][sd]["held ce"]), None)
    def _boot(d, B=4000, seed=0):
        rnd = random.Random(seed); by = {}
        for v, gk in zip(d, GRP):
            by.setdefault(gk, []).append(v)
        ks = sorted(by); out = []
        for _ in range(B):
            s_ = n_ = 0
            for gk in (rnd.choice(ks) for _ in ks):
                s_ += sum(by[gk]); n_ += len(by[gk])
            out.append(s_ / n_)
        out.sort(); return [round(sum(d) / len(d), 4), round(out[int(.025 * B)], 4), round(out[int(.975 * B) - 1], 4), sum(1 for gk in by if sum(by[gk]) < 0), len(by)]
    def _avg(keys):
        return [sum(v) / len(v) for v in zip(*(ROWS[k] for k in keys))]
    PR = {}
    for a, b in (("D", "S"), ("D4", "S4"), ("D", "C"), ("S", "C"), ("D4", "C"), ("S4", "C"), ("DI", "C"), ("D4", "D"), ("S4", "S"), ("DI", "D")):
        sds = [sd for sd in (1, 2, 3) if (a, sd) in ROWS and (b, sd) in ROWS]
        for sd in sds:
            A_, B_ = DEC[(a, sd)], DEC[(b, sd)]; w = sum(1 for i in range(len(A_)) if A_[i] and not B_[i]); l = sum(1 for i in range(len(A_)) if B_[i] and not A_[i])
            PR.setdefault(f"{a} vs {b}", {})[str(sd)] = {"rows: mean, lo, hi, repos closer, repos": _boot([x - y for x, y in zip(ROWS[(a, sd)], ROWS[(b, sd)])]), "decisions: first only, second only, p": [w, l, round(_sg(w, l), 4)]}
        if sds:
            PR[f"{a} vs {b}"]["all seeds"] = {"seeds": sds, "rows: mean, lo, hi, repos closer, repos": _boot([x - y for x, y in zip(_avg([(a, sd) for sd in sds]), _avg([(b, sd) for sd in sds]))])}
    cs = [(a, b, sd) for a, b in (("D", "S"), ("D4", "S4")) for sd in (1, 2, 3) if (a, sd) in ROWS and (b, sd) in ROWS]
    if cs:   # 206c: the cross corrector against the self-corrector over every pair present
        d = [sum(DEC[(a, sd)][i] for a, b, sd in cs) - sum(DEC[(b, sd)][i] for a, b, sd in cs) for i in range(len(al4))]; w = sum(v > 0 for v in d); l = sum(v < 0 for v in d)
        means = [PR[f"{a} vs {b}"][str(sd)]["rows: mean, lo, hi, repos closer, repos"][0] for a, b, sd in cs]
        PR["cross vs self"] = {"pairs": len(cs), "pairs with the cross student closer": sum(m < 0 for m in means), "pair means": means,
                               "rows: mean, lo, hi, repos closer, repos": _boot([x - y for x, y in zip(_avg([(a, sd) for a, b, sd in cs]), _avg([(b, sd) for a, b, sd in cs]))]),
                               "decisions: more cross students match, more self students match, p": [w, l, round(_sg(w, l), 4)], "net differences": [T[a][str(sd)]["net"] - T[b][str(sd)]["net"] for a, b, sd in cs]}
    res["online g4"] = {"table": T, "pairs": PR, "partition": {"differing": len(dif4), "agreeing": len(same4)}}
for f in sorted(OUT.parent.glob("drift*.json")):   # review D1: drift from the original student (scripts/drift.py); every drift file of the paper
    if f.exists():
        res.setdefault("drift", {}).update(json.load(open(f))["summary"])
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT, "| main arms:", sorted(res["main"]["pooled"]), "| 174 arms:", sorted(res["f174"]["pooled"]))
