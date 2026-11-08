"""Addendum 182: pool arm rb over seeds and pair it with the stored arms of 179 (sft_s, sft_c_s) and 181 (rw_cf, mix_cf, opd_cf), whose
per-problem results come from earlier sessions. The cross-session pairing is used only for seeds where this session's sft reproduces the
stored sft per-problem results exactly (181 reproduced 179 exactly); otherwise rb is paired with sft (in-session) only.
  python rb_pair.py RUNS_DIR [ARM PREFIX TAG]   (RUNS_DIR holds xqlB/xl_base.json, xqlO{s}/xl_eval_s{s}.json, xqlCF{s}/xl_eval_cf_s{s}.json,
  xqlRB{s}/xl_eval_rb_s{s}.json; 184c: ARM rb_c, PREFIX xqlRBC, TAG rbc pairs rb_c with the stored arms including rb)"""
import json, sys
from math import comb
from pathlib import Path


def p2(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


ARM, PRE, TAG = (sys.argv[2:5] + ["rb", "xqlRB", "rb"][len(sys.argv[2:5]):]) if len(sys.argv) > 2 else ("rb", "xqlRB", "rb")
R = Path(sys.argv[1]); base = json.load(open(R / "xqlB/xl_base.json")); rows = set(base["rows"]); held = base["held"]
STORED = ("sft_s", "sft_c_s", "rw_cf", "mix_cf", "opd_cf") + (("rb",) if ARM != "rb" else ()); P, T, ok_seeds = {}, {}, []
for s in (1, 2, 3):
    f = R / f"{PRE}{s}/xl_eval_{TAG}_s{s}.json"
    if not f.exists():
        print(f"seed {s}: no {ARM} result yet"); continue
    d = json.load(open(f)); pp = d["per problem"]; q = pp["qwen"]
    st = {}
    for g in (R / f"xqlO{s}/xl_eval_s{s}.json", R / f"xqlCF{s}/xl_eval_cf_s{s}.json", R / f"xqlRB{s}/xl_eval_rb_s{s}.json"):
        if g.exists():
            st.update({a: v for a, v in json.load(open(g))["per problem"].items() if a not in st})
    same = st.get("sft") == pp["sft"] and st.get("qwen") == q; ok_seeds += [s] if same else []
    print(f"seed {s}: in-session sft and Qwen reproduce the stored per-problem results: {same}" + ("" if same else f" (sft differs on {sum(1 for i in pp['sft'] if st.get('sft', {}).get(i) != pp['sft'][i])} problems)"))
    trained = [i for i in q if i in rows]; kp = [i for i in q if i not in rows and i not in held and q[i]]
    arms = {"sft": pp["sft"], ARM: pp[ARM], **({a: st[a] for a in STORED if a in st} if same else {})}
    for a, r_ in arms.items():
        t = T.setdefault(a, [0, 0, 0, 0, 0])
        t[0] += sum(1 for i in trained if not q[i] and r_[i]); t[1] += sum(1 for i in held if not q[i] and r_[i])
        t[2] += sum(1 for i in held if r_[i]); t[3] += sum(1 for i in held if q[i] and not r_[i]); t[4] += sum(1 for i in trained + kp if q[i] and not r_[i])
    for b in arms:
        if b == ARM:
            continue
        for part, ids in (("trained", trained), ("held", held)):
            x = P.setdefault((b, part), [0, 0]); x[0] += sum(1 for i in ids if arms[ARM][i] and not arms[b][i]); x[1] += sum(1 for i in ids if arms[b][i] and not arms[ARM][i])
print("\narm       fixed  new  held solved  lost held  lost trained   (pooled over the seeds read; stored arms only over seeds", ok_seeds, ")")
for a, t in T.items():
    print(f"{a:9s} {t[0]:5d} {t[1]:4d} {t[2]:12d} {t[3]:10d} {t[4]:13d}")
print(f"\npaired, {ARM} only vs other only:")
for (b, part), (x, y) in P.items():
    print(f"  {ARM} vs {b:8s} {part:8s} {x:4d} vs {y:4d}  p = {p2(x, y):.3g}")
