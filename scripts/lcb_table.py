"""Addendum 209: the online teachers on contest coding, scored from the sessions' outputs in papers/kaggle_bridge/data/lcb (or $LCB_DATA):
xl_eval_lcb_<arm>_s<seed>.json (xlearn.evaluate: per-problem passes for E2B alone, key "qwen", and the arm), online_<teacher>_keep_replay_lcb[_sN].json
(the stream's curve) and gate_lora_onl_<teacher>_keep_replay_lcb[_sN]_points.json (206's agent gate, scored on the 4090 partition as g4_table.py).
Per arm and seed: held solved (of 188), new and kept, fixed (row problems E2B alone fails), lost (row and held), the paired test against E2B alone;
between arms of a seed and pooled over seeds: exact sign tests on discordant held problems (and on row problems). Standard library only.
usage: lcb_table.py [seed ...]   (default: every seed that has files)"""
import json, os, sys
from math import comb
_H = os.path.dirname(os.path.abspath(__file__))
_D0 = os.path.join(_H, "..", "..", "papers", "kaggle_bridge", "data", "lcb")
D = os.path.join(os.environ.get("LCB_DATA") or (_D0 if os.path.isdir(_D0) else os.path.join(_H, "..", "data", "lcb")), "")   # the second form: the public repository's layout (scripts/ beside data/)
ONL = os.path.join(D, "..", "online", "")
# (name, evaluation arm, teacher, folder of its stream and gate files). 209a's arms with the correctors fitted with the contest rows (D4f, S4f) write
# stream and gate files under the names of 209's unfitted ones, so theirs are stored in fit/.
# 209e's arms (D4v, Cv: verified data and teaching only on answers that pass) write theirs as ..._lcbv. 209h's arms (D4w, S4w: the corrector fitted with verified-token targets) use the same names, stored in h/.
# 209i's arms (D4x, S4x: both correctors fitted with those targets on the regenerated rows) write under 209h's names too, stored in i/.
# 209l's arm (D4e: 209g's fit with the stub direction erased from the corrector's input) writes under D4v's names, stored in l/.
ARMS = [("D4x", "onl_d4w", "bridge", "i/", "lcbv"), ("S4x", "onl_s4w", "control", "i/", "lcbv"), ("D4r", "onl_d4w", "bridge", "r/", "lcbv"), ("D4n", "onl_d4w", "bridge", "n/", "lcbv"), ("D4p", "onl_d4w", "bridge", "p/", "lcbv"), ("D4e", "onl_d4v", "bridge", "l/", "lcbv"), ("D4xe", "onl_d4w", "bridge", "lx/", "lcbv"), ("D4w", "onl_d4w", "bridge", "h/", "lcbv"), ("S4w", "onl_s4w", "control", "h/", "lcbv"), ("D4v", "onl_d4v", "bridge", "", "lcbv"), ("S4v", "onl_s4v", "control", "", "lcbv"), ("Cv", "onl_cv", "direct", "", "lcbv"), ("D4f", "onl_d4f", "bridge", "fit/", "lcb"), ("S4f", "onl_s4f", "control", "fit/", "lcb"),
        ("D4", "onl_d4", "bridge", "", "lcb"), ("S4", "onl_s4", "control", "", "lcb"), ("C", "onl_c", "direct", "", "lcb"), ("copy", "onl_sft", "sft", "", "lcb")]
PK = "decision points (matches the teacher, probability of its choice, student alone differs)"


def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    base = json.load(open(os.path.join(D, "xl_base_ids.json"))); held, rows = base["held"], base["rows"]
    al = json.load(open(ONL + "gate_lora_onl_bridge_keep_replay_points.json"))[PK]["E4B alone"]; dif = [i for i, x in enumerate(al) if x[2]]; same = [i for i, x in enumerate(al) if not x[2]]
    R, Q, T = {}, {}, {}
    for sd in seeds:
        sx = "" if sd == 1 else f"_s{sd}"
        for k, arm, teach, sub, st in ARMS:
            f = D + (sub if sub in ("h/", "i/", "l/", "lx/", "n/", "p/", "r/") else "") + f"xl_eval_lcb_{arm}_s{sd}.json"   # 209h's and 209i's arms share an evaluation name: each in its folder
            if not os.path.exists(f):
                continue
            d = json.load(open(f)); res = d["per problem"]; q, r = res["qwen"], res[arm]; R[(k, sd)] = r; Q[(k, sd)] = q
            assert set(held) <= set(q) and set(rows) <= set(q) and d["seed"] == sd, (k, sd)
            hs, new, kept = sum(r[i] for i in held), sum(1 for i in held if r[i] and not q[i]), sum(1 for i in held if r[i] and q[i])
            fixed, fail = sum(1 for i in rows if r[i] and not q[i]), sum(1 for i in rows if not q[i])
            lost_r, lost_h = sum(1 for i in rows if q[i] and not r[i]), sum(1 for i in held if q[i] and not r[i])
            w, l = new, lost_h; ce = d.get("held: cross-entropy of E4B's passing solutions (lower = closer to E4B)", {}); ot = d.get("off-task (kept knowledge)", {})
            line = f"s{sd} {k:5s} held {hs:3d} (alone {sum(q[i] for i in held)}; new {new}, kept {kept}) vs alone {w} vs {l} p={sign(w, l):.4f} | fixed {fixed} of {fail} | lost {lost_r + lost_h} (row {lost_r}, held {lost_h})"
            line += f" | E4B's held solutions CE {ce.get(arm)} (alone {ce.get('qwen')}) | prose CE {ot.get('held prose (untaught)', {}).get('cross-entropy', {}).get(arm)} (alone {ot.get('held prose (untaught)', {}).get('cross-entropy', {}).get('qwen')})"
            fo = D + sub + f"online_{teach}_keep_replay_{st}{sx}.json"
            if os.path.exists(fo):
                on = json.load(open(fo)); c = on["held curve (bridge down)"]; g = [x["gain over base"] for x in on["prequential windows (before learning each item)"]]
                if on.get("verify"):
                    v = on["verify"]; nf = v["items learned on E4B's solution (no sample passed)"]; line += f" | verify: {v['samples passing']} of {v['samples']} samples pass, {nf} of {v['learned items (with replays)']} items on E4B's solution"
                line += f" | stream: held CE {c[0]['held ce (base)']} -> {c[-1]['held ce (on)']} (lowest {min(x['held ce (on)'] for x in c)}), teacher {c[0].get('held ce (teacher)')}, gain first pass {sum(g[:3]) / 3:.4f}, drift {c[-1]['KL from base on untaught prose / code / story']:.4f}"
            fg = D + sub + f"gate_lora_onl_{teach}_keep_replay_{st}{sx}_points.json"
            if os.path.exists(fg):
                X = json.load(open(fg))[PK]["LoRA"]; fx, br = sum(X[i][0] for i in dif), sum(1 - X[i][0] for i in same); line += f" | agent gate fixed {fx} broken {br} net {fx - br:+d}"
            print(line)
            if "sft" in res and k == "copy":                              # the coding study's offline SFT adapter, a reference in the copying session
                r2 = res["sft"]; T[sd] = r2
                print(f"s{sd} ref   held {sum(r2[i] for i in held):3d} (offline SFT, rank 64, no keep) vs alone {sum(1 for i in held if r2[i] and not q[i])} vs {sum(1 for i in held if q[i] and not r2[i])} | fixed {sum(1 for i in rows if r2[i] and not q[i])} | lost {sum(1 for i in rows + held if q[i] and not r2[i])}")
        qs = [(k, Q[(k, sd)]) for k, *_ in ARMS if (k, sd) in Q]
        for k, q in qs[1:]:                                              # E2B alone must answer the same in every session of a seed
            n = sum(1 for i in rows + held if q[i] != qs[0][1][i]); print(f"  s{sd} E2B alone, session {k} vs session {qs[0][0]}: {n} of {len(rows) + len(held)} results differ")
    pairs = [("D4r", "D4n"), ("D4r", "D4x"), ("D4r", "Cv"), ("D4r", "copy"), ("D4n", "D4x"), ("D4n", "Cv"), ("D4n", "copy"), ("D4p", "D4x"), ("D4p", "Cv"), ("S4x", "S4v"), ("D4e", "D4v"), ("D4e", "Cv"), ("D4xe", "D4x"), ("D4xe", "Cv"), ("D4e", "D4x"), ("D4x", "Cv"), ("D4x", "S4x"), ("D4x", "copy"), ("D4x", "C"), ("D4x", "D4v"), ("D4x", "D4w"), ("S4x", "Cv"), ("S4x", "copy"), ("S4x", "S4v"), ("D4w", "Cv"), ("D4w", "D4v"), ("D4w", "copy"), ("D4w", "S4w"), ("D4v", "S4v"), ("S4v", "Cv"), ("S4v", "copy"), ("D4v", "Cv"), ("D4v", "C"), ("D4v", "copy"), ("Cv", "C"), ("Cv", "copy"), ("D4v", "D4f"), ("D4f", "C"), ("D4f", "copy"), ("D4f", "S4f"), ("S4f", "C"), ("S4f", "copy"), ("C", "copy"), ("D4f", "D4"), ("D4", "C"), ("D4", "copy"), ("D4", "S4"), ("S4", "C"), ("S4", "copy")]
    for a, b in pairs:
        tot = {"held": [0, 0], "row": [0, 0]}; n = 0
        for sd in seeds:
            if (a, sd) not in R or (b, sd) not in R:
                continue
            n += 1
            for name, ids in (("held", held), ("row", rows)):
                w = sum(1 for i in ids if R[(a, sd)][i] and not R[(b, sd)][i]); l = sum(1 for i in ids if R[(b, sd)][i] and not R[(a, sd)][i]); tot[name][0] += w; tot[name][1] += l
                if name == "held":
                    print(f"  s{sd} {a} vs {b}: held {w} vs {l} p={sign(w, l):.3f} (solved {sum(R[(a, sd)][i] for i in held)} vs {sum(R[(b, sd)][i] for i in held)})", end="")
                else:
                    print(f" | row problems {w} vs {l} p={sign(w, l):.3f}")
        if n > 1:
            print(f"  pooled over {n} seeds {a} vs {b}: held {tot['held'][0]} vs {tot['held'][1]} p={sign(*tot['held']):.4f} | row problems {tot['row'][0]} vs {tot['row'][1]} p={sign(*tot['row']):.4f}")
    for k, *_ in ARMS:                                                   # each arm against E2B alone, pooled
        sds = [sd for sd in seeds if (k, sd) in R]
        if len(sds) > 1:
            w = sum(1 for sd in sds for i in held if R[(k, sd)][i] and not Q[(k, sd)][i]); l = sum(1 for sd in sds for i in held if Q[(k, sd)][i] and not R[(k, sd)][i])
            print(f"  pooled over {len(sds)} seeds {k} vs E2B alone: held {w} vs {l} p={sign(w, l):.4f} | held solved per seed {[sum(R[(k, sd)][i] for i in held) for sd in sds]}")


if __name__ == "__main__":
    main()
