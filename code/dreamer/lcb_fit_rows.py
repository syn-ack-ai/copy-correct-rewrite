"""Amendment 209f: the candidate rows of the corrector's fit, built from verified answers already stored, with an audit by rule of what the
tests cannot see. A candidate is a solution to one of the coding study's 299 row problems that passed its tests in an earlier session:
  E4B   its shortest verified solution (runs/xqlShort/xl_rows_short.jsonl), then the build's passing greedy answer and samples (runs/xqlB/xl_build_gens.jsonl)
  E2B   its own greedy answer where it solves the problem alone (209's evaluations: the same text in every session, passing in all six)
Not used: the solutions E2B wrote with a plan in the prompt (the runbook rows): 209 of 277 repeat the plan as numbered comments and 11 name it.
A candidate is left out, with the reason counted, if
  placeholder   a comment line starts with an ellipsis ("# ...")
  stub          a def whose body is only comments, a pass or a docstring
  aloud         a comment thinks aloud or refers to its instructions (wait, re-read, actually, let's, let me, I will, I think, however, refine,
                reconsider, mistake, not correct, typo, the prompt, the plan, a hint, alternative approach, interpretation A/B)
  comments      half or more of its non-empty lines are comments
  fence, parse  not exactly one closing fence at the end; the code does not parse
  near          it is 90% or more the same text as a row already kept for the problem
At most CAPN = 2 rows per author and problem, in the order above. Writes OUT/xl_rows_fit.jsonl (id, ctx, reply, author, source, difficulty) and
OUT/xl_fit.json (counts, sha256), then re-reads the file and prints the audit: every rule again on the written rows (all must be 0).
usage: lcb_fit_rows.py OUT_DIR      Standard library only. The session's lcb_own.py runs every row on its tests again before the fit."""
import ast, collections, difflib, hashlib, json, os, re, sys
_H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(_H, "..", "runs", ""); DL = os.path.join(_H, "..", "..", "papers", "kaggle_bridge", "data", "lcb", "")
PLACE = re.compile(r"^\s*#\s*\.\.\.", re.M); CAPN = 2
ALOUD = re.compile(r"(?i)\b(wait|re-?read|re-?verify|hmm+|actually|oops|mistake|not correct|incorrect assumption|correction|let'?s|let us|let me|i will|i'll|i think|i am|however|refine|reconsider|"
                   r"typo|the prompt|plan|hint|as suggested|as requested|alternative approach|interpretation [ab])\b")
EVALS = ("xl_eval_lcb_onl_d4f_s2.json", "xl_eval_lcb_onl_c_s2.json", "xl_eval_lcb_onl_sft_s2.json", "xl_eval_lcb_onl_c_s1.json", "xl_eval_lcb_onl_sft_s1.json", "xl_eval_lcb_onl_d4_s1.json")


def code(t):
    return t.rstrip()[:-3]


def sim(a, b):
    """share of text two rows have in common (the measure is not symmetric: the larger of both orders)"""
    return max(difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, b, a).ratio())


def reason(t, soft=True):
    """why a text is not a clean row (None = clean); soft=False: only the rules of form (fence, placeholder, parse, stub)"""
    if not t.rstrip().endswith("```") or t.count("```") != 1:
        return "fence"
    c = code(t); L = [l.strip() for l in c.split("\n") if l.strip()]; com = [l for l in L if l.startswith("#")]
    if PLACE.search(c):
        return "placeholder"
    try:
        tree = ast.parse(c)
    except SyntaxError:
        return "parse"
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and all(isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)) for b in f.body):
            return "stub"
    if not soft:
        return None
    if any(ALOUD.search(l) for l in com) or any(ALOUD.search(l.split("#", 1)[1]) for l in L if "#" in l and not l.startswith("#") and "'" not in l.split("#", 1)[0][-1:] and '"' not in l.split("#", 1)[0][-1:]):
        return "aloud"
    if len(com) >= 0.5 * max(1, len(L)):
        return "comments"
    return None


def candidates():
    """(author, id) -> [(source, text)] in order of preference, and the short rows (prompt, difficulty) by id"""
    base = json.load(open(DL + "xl_base_ids.json")); rows = base["rows"]; rowset = set(rows)
    short = {r["id"]: r for r in (json.loads(l) for l in open(R + "xqlShort/xl_rows_short.jsonl"))}; assert set(short) == rowset
    cut = lambda t: t[:t.index("```") + 3]  # noqa: E731
    cand = collections.defaultdict(list)
    for i in rows:
        cand[("e4b", i)].append(("shortest verified", short[i]["reply"]))
    for g in (json.loads(l) for l in open(R + "xqlB/xl_build_gens.jsonl")):
        if g["model"] == "e4b" and g["pool"] == "train" and g["pass"] and "```" in g["text"] and g["id"] in rowset:
            cand[("e4b", g["id"])].append(("build " + g["kind"], cut(g["text"])))
    finals = [json.load(open(DL + f))["per problem"]["qwen"] for f in EVALS]
    for g in (json.loads(l) for l in open(R + "xqcDf2/xl_gens_qwen.jsonl")):
        if g["id"] in rowset and all(f[g["id"]] for f in finals) and "```" in g["text"]:
            cand[("e2b", g["id"])].append(("greedy", cut(g["text"])))
    return cand, short, rows, set(base["held"])


def pool(out_dir):
    """209g: every candidate that passes the rules of form, near-copies left out, no cap and no word lists: the rows a model judge reads
    (judge_rows.py). 'soft' records the word-list rule that 209f would have applied. Plus calibration rows whose answer is known."""
    os.makedirs(out_dir, exist_ok=True); cand, short, rows, held = candidates(); out, why, kept = [], collections.Counter(), collections.defaultdict(list)
    for au in ("e4b", "e2b"):
        for i in rows:
            for k, (src, t) in enumerate(cand.get((au, i), [])):
                w = reason(t, soft=False)
                if w is None and any(sim(code(t), code(o)) >= 0.9 for o in kept[i]):
                    w = "near"
                if w:
                    why[f"{au}: {w}"] += 1; continue
                kept[i].append(t); out.append({"rid": f"{au}:{i}:{k}", "id": i, "ctx": short[i]["ctx"], "reply": t, "author": au, "source": src, "difficulty": short[i]["difficulty"], "soft": reason(t)})
    old = [json.loads(l) for l in open(R + "xql/own209e/xl_rows_fit.jsonl")]   # calibration: 209e's rows that name a plan or hold the stub's shape are not clean
    cal = [r for r in old if r["source"].startswith("with a plan") and re.search(r"(?im)^\s*#.*\bplan\b", r["reply"])] + [r for r in old if reason(r["reply"], soft=False) == "stub"]
    for k, r in enumerate(cal):
        out.append({"rid": f"calib:{r['author']}:{r['id']}:{k}", "id": r["id"], "ctx": r["ctx"], "reply": r["reply"], "author": r["author"], "source": r["source"], "difficulty": r["difficulty"], "soft": reason(r["reply"]), "expect": "not_clean"})
    with open(os.path.join(out_dir, "xl_rows_pool.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    n = collections.Counter(r["author"] for r in out if not r.get("expect"))
    print(json.dumps({"pool rows": sum(n.values()), "by author": dict(n), "problems": len({r["id"] for r in out if not r.get("expect")}), "left out by form": dict(why),
                      "the word lists would leave out": dict(collections.Counter(r["soft"] for r in out if not r.get("expect") and r["soft"])), "calibration rows (expected not clean)": len(cal)}))


def _votes(path, nv=2):
    """rid -> 'clean' (every vote), 'not_clean' (any vote says so) or None (votes missing); and the votes"""
    V = collections.defaultdict(dict)
    for v in map(json.loads, open(path)):
        if v.get("verdict"):
            V[v["rid"]][v["vote"]] = v
    return {r: ("clean" if all(x["verdict"] == "clean" for x in vs.values()) else "not_clean") if len(vs) >= nv else None for r, vs in V.items()}, V


def fixitems(out_dir, verdicts):
    """209g: the pool rows the judge did not call clean in every vote, with its notes, for their author model to rewrite (transfer/lcb_fix.py)"""
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "xl_rows_pool.jsonl")) if not json.loads(l).get("expect")]; res, V = _votes(verdicts); out = []
    miss = [r["rid"] for r in rows if res.get(r["rid"]) is None]; assert not miss, f"{len(miss)} rows without both votes: {miss[:5]}"
    for r in rows:
        if res[r["rid"]] == "not_clean":
            notes = "; ".join(sorted({(", ".join(v["issues"]) + ": " if v["issues"] else "") + v["reason"] for v in V[r["rid"]].values() if v["verdict"] == "not_clean"}))
            out.append({"rid": r["rid"], "id": r["id"], "author": r["author"], "reply": r["reply"], "notes": notes[:600]})
    p = os.path.join(out_dir, "fix_items.jsonl")
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"pool rows": len(rows), "clean in every vote": sum(1 for r in rows if res[r["rid"]] == "clean"), "to rewrite": len(out), "by author": dict(collections.Counter(r["author"] for r in out)), "sha": hashlib.sha256(open(p, "rb").read()).hexdigest()}))


def pool2(out_dir, fix_out, tag="r1"):
    """209g: the author models' new versions (round TAG) that pass their tests, end with their fence and pass the rules of form: the judge reads them"""
    rows = {r["rid"]: r for r in (json.loads(l) for l in open(os.path.join(out_dir, "xl_rows_pool.jsonl")))}; out, why = [], collections.Counter(); seen = set()
    fx = [json.loads(l) for l in open(fix_out)]; fx.sort(key=lambda r: (r["rid"], r["kind"] != "greedy"))   # a greedy version first; samples in the order drawn
    for k, f in enumerate(fx):
        o = rows[f["rid"]]; t = f["text"]
        w = "fails its tests" if not f["pass"] else "no fence" if not f["fenced"] else reason(t, soft=False) or ("the same text again" if (f["rid"], t) in seen or t == o["reply"] else None)
        if w:
            why[f"{f['author']}: {w}"] += 1; continue
        seen.add((f["rid"], t)); lab = {"continued": ", continued by its author from before the first flagged line", "sampled again": ", sampled again by its author (a fresh answer to the plain prompt)"}.get(f["kind"], ", rewritten by its author after review")
        out.append({"rid": f"{f['rid']}#{tag}.{k}", "of": f["rid"], "id": o["id"], "ctx": o["ctx"], "reply": t, "author": o["author"], "source": o["source"] + lab, "difficulty": o["difficulty"], "soft": reason(t)})
    with open(os.path.join(out_dir, f"xl_rows_pool2_{tag}.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"round": tag, "new versions": len(fx), "of rows": len({f["rid"] for f in fx}), "to the judge": len(out), "rows with a version for the judge": len({r["of"] for r in out}), "left out": dict(why)}))


def _fixed(out_dir, rounds):
    """rounds: [(tag, verdict file)] -> {original rid: its shortest version that is clean in every vote, over all rounds} (the agent rows' rule:
    the judges only accept, the shortest accepted sample is used), and per round the rows read"""
    fixed, per = {}, {}
    for tag, vf in rounds:
        res2, V2 = _votes(vf); p2 = [json.loads(l) for l in open(os.path.join(out_dir, f"xl_rows_pool2_{tag}.jsonl"))]
        miss = [r["rid"] for r in p2 if res2.get(r["rid"]) is None]; assert not miss, f"round {tag}: {len(miss)} versions without both votes: {miss[:5]}"
        for r in p2:
            if res2[r["rid"]] == "clean" and (r["of"] not in fixed or len(r["reply"]) < len(fixed[r["of"]]["reply"])):
                fixed[r["of"]] = r
        per[tag] = (p2, res2, V2)
    return fixed, per


def nextround(out_dir, nxt, *rv):
    """209g, another round (user 11-11: "after the resample have deepseek judge them again to verify and have it be fixed if its still not good";
    "i belive we have a resample script and a judge thst pics the best sample": the agent rows' repair_rounds.sh): every flagged row with no
    accepted version after the rounds read so far (rv: TAG VERDICTS pairs) is sampled again, with more samples and more variety: once from
    its cut and once as a fresh answer to the plain prompt (an empty prefix). Writes cont_items_NXT.jsonl (two items per row)."""
    fixed, _ = _fixed(out_dir, list(zip(rv[0::2], rv[1::2]))); first = [json.loads(l) for l in open(os.path.join(out_dir, "cont_items.jsonl"))]; out = []
    for it in first:
        if it["rid"] not in fixed:
            base = {k: it[k] for k in ("rid", "id", "author", "reply", "notes")}
            out += [{**base, "prefix": it["prefix"]}] + ([{**base, "prefix": ""}] if it["prefix"] else [])
    p = os.path.join(out_dir, f"cont_items_{nxt}.jsonl")
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    n = len({r["rid"] for r in out})
    print(json.dumps({"flagged rows": len(first), "with an accepted version so far": len(first) - n, "share of the flagged rows fixed so far": round((len(first) - n) / max(1, len(first)), 3), "sampled again in round " + nxt: n, "items": len(out),
                      "rows by author": dict(collections.Counter(a_ for a_, _ in {(r["author"], r["rid"]) for r in out})), "sha": hashlib.sha256(open(p, "rb").read()).hexdigest()}))


def final(out_dir, verdicts, *rv):
    """209g: the fit's rows: pool rows clean in every vote, and for a row that was not, its first new version that is (rounds in order;
    rv: TAG VERDICTS pairs); near-copies left out; at most CAPN per author and problem"""
    keep_split = os.environ.get("FIT_KEEP_SPLIT") == "1"   # 209i: a row only one vote flagged is kept as it is (only rows every vote flags are regenerated or left out)
    pool_ = [json.loads(l) for l in open(os.path.join(out_dir, "xl_rows_pool.jsonl")) if not json.loads(l).get("expect")]; res, V1 = _votes(verdicts)
    miss = [r["rid"] for r in pool_ if res.get(r["rid"]) is None]; assert not miss, f"{len(miss)} pool rows without both votes: {miss[:5]}"   # review 209g: a missing vote is never read as not clean
    if keep_split:
        res = {k_: ("clean" if v_ == "clean" or any(x["verdict"] == "clean" for x in V1[k_].values()) else v_) for k_, v_ in res.items()}
    fixed, _per = _fixed(out_dir, list(zip(rv[0::2], rv[1::2])))
    out, why, kept, n, new = [], collections.Counter(), collections.defaultdict(list), collections.Counter(), []   # new: the row's text was replaced in this round
    for r in pool_:                                                      # the pool's order: E4B's rows first, each problem's rows in order of preference
        use = r if res.get(r["rid"]) == "clean" else fixed.get(r["rid"])
        if use is None:
            why[f"{r['author']}: not clean, no clean new version"] += 1; continue
        if n[(r["author"], r["id"])] >= CAPN:
            why[f"{r['author']}: over the cap"] += 1; continue
        if reason(use["reply"], soft=False) or any(sim(code(use["reply"]), code(o)) >= 0.9 for o in kept[r["id"]]):
            why[f"{r['author']}: a near-copy or a fault of form"] += 1; continue
        kept[r["id"]].append(use["reply"]); n[(r["author"], r["id"])] += 1; new.append(use["reply"] != r["reply"])
        out.append({"id": r["id"], "ctx": r["ctx"], "reply": use["reply"], "author": r["author"], "source": use["source"], "difficulty": r["difficulty"]})
    p = os.path.join(out_dir, "xl_rows_fit.jsonl")
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest(); nb = collections.Counter(r["author"] for r in out)
    cs = lambda t: sum(l.strip().startswith("#") for l in code(t).split("\n")) / max(1, sum(1 for l in code(t).split("\n") if l.strip()))  # noqa: E731
    st = {"rows": len(out), "by author": dict(nb), "rows whose text was replaced in this round": dict(collections.Counter(r["author"] for r, w in zip(out, new) if w)), "of those, rewritten": dict(collections.Counter(r["author"] for r, w in zip(out, new) if w and "rewritten by its author" in r["source"])), "problems": len({r["id"] for r in out}),
          "problems by author": {a: len({r["id"] for r in out if r["author"] == a}) for a in nb}, "left out": dict(why), "rows the word lists of 209f would still leave out": dict(collections.Counter(reason(r["reply"]) for r in out if reason(r["reply"]))),
          "comment-line share (mean): rows kept as they were / replaced": [round(sum(cs(r["reply"]) for r, w in zip(out, new) if not w) / max(1, new.count(False)), 3), round(sum(cs(r["reply"]) for r, w in zip(out, new) if w) / max(1, new.count(True)), 3)],
          "reply chars (median): kept / replaced": [(sorted(len(r["reply"]) for r, w in zip(out, new) if not w) or [0])[new.count(False) // 2], (sorted(len(r["reply"]) for r, w in zip(out, new) if w) or [0])[new.count(True) // 2]],
          "cap per author and problem": CAPN, "sha": sha}
    json.dump(st, open(os.path.join(out_dir, "xl_fit.json"), "w"), indent=1); print(json.dumps(st, indent=1))


def sanity(out_dir, nshow="3", seed="11"):
    """209g, before any fit (user 11-11: "you sanity check the results before we restart the refit"): the final file read back and checked row by
    row, the continued rows against their cuts, and NSHOW rows of each kind printed to be read"""
    import random
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "xl_rows_fit.jsonl"))]; base = json.load(open(DL + "xl_base_ids.json")); rowset, held = set(base["rows"]), set(base["held"])
    short = {r["id"]: r for r in (json.loads(l) for l in open(R + "xqlShort/xl_rows_short.jsonl"))}; pool_ = [json.loads(l) for l in open(os.path.join(out_dir, "xl_rows_pool.jsonl"))]
    orig, osrc = collections.defaultdict(list), collections.defaultdict(list)
    for r in pool_:
        orig[(r["author"], r["id"])].append(r["reply"]); osrc[(r["author"], r["id"])].append(r["source"])
    def this_round(r):
        """'' for a row kept as it was; else the label this round added to its pool row's source (209i's pool rows may carry 209g's own labels)"""
        if r["reply"] in orig[(r["author"], r["id"])]:
            return ""
        m = [o for o in osrc[(r["author"], r["id"])] if r["source"].startswith(o) and len(r["source"]) > len(o)]
        return r["source"][len(max(m, key=len)):] if m else None
    items = {}
    for f in sorted(os.listdir(out_dir)):                                # every round's cut items: (author, id, prefix)
        if f.startswith("cont_items") and f.endswith(".jsonl"):
            for it in (json.loads(l) for l in open(os.path.join(out_dir, f))):
                items.setdefault((it["author"], it["id"]), []).append(it["prefix"])
    bad = collections.defaultdict(list); seen = set(); n = collections.Counter(); per = collections.Counter()
    for k, r in enumerate(rows):
        lab = this_round(r); cont = lab != ""; rew = bool(lab) and "rewritten by its author" in lab; n[(r["author"], "rewritten" if rew else "completed again" if cont else "kept as it was")] += 1; per[(r["author"], r["id"])] += 1   # 209i: a rewrite needs no kept prefix
        if lab is None: bad["a replaced row without a label of this round"].append(k); lab = ""
        if set(r) != {"id", "ctx", "reply", "author", "source", "difficulty"}: bad["fields"].append(k)
        if r["id"] not in rowset or r["id"] in held: bad["not a row problem, or a held problem"].append(k)
        if r["ctx"] != short[r["id"]]["ctx"]: bad["the prompt is not the problem's prompt"].append(k)
        if reason(r["reply"], soft=False): bad["a fault of form (" + str(reason(r["reply"], soft=False)) + ")"].append(k)
        if (r["id"], r["reply"]) in seen: bad["the same text twice"].append(k)
        seen.add((r["id"], r["reply"]))
        if cont:
            pre = [x for x in items.get((r["author"], r["id"]), []) if x and r["reply"].startswith(x)]
            if not pre and "sampled again" not in lab and not rew: bad["a continued row does not start with a kept prefix"].append(k)
            if r["reply"] in orig[(r["author"], r["id"])]: bad["a continued row is the original text"].append(k)
            L = [l.strip() for l in code(r["reply"]).split("\n") if l.strip().startswith(("def ", "class "))]
            if len(L) != len(set(L)): bad["a def or class line written twice"].append(k)
    if max(per.values()) > CAPN: bad["over the cap"].append(max(per.values()))
    cs = lambda t: sum(l.strip().startswith("#") for l in code(t).split("\n")) / max(1, sum(1 for l in code(t).split("\n") if l.strip()))  # noqa: E731
    kinds = {"kept as it was": [r for r in rows if this_round(r) == ""], "completed again": [r for r in rows if this_round(r) and "rewritten by its author" not in this_round(r)],
             "rewritten": [r for r in rows if this_round(r) and "rewritten by its author" in this_round(r)]}
    print(json.dumps({"rows": len(rows), "by author and kind": {f"{a} / {k_}": c for (a, k_), c in sorted(n.items())}, "problems": len({r["id"] for r in rows}), "rows per author and problem (count of pairs)": dict(collections.Counter(per.values())),
                      "comment-line share (mean)": {k_: round(sum(cs(r["reply"]) for r in v) / max(1, len(v)), 3) for k_, v in kinds.items()}, "rows with no comment line": {k_: sum(1 for r in v if cs(r["reply"]) == 0) for k_, v in kinds.items()},
                      "chars (median)": {k_: (sorted(len(r["reply"]) for r in v) or [0])[len(v) // 2] for k_, v in kinds.items()}, "the word lists of 209f would still flag": dict(collections.Counter(reason(r["reply"]) for r in rows if reason(r["reply"]))),
                      "failures": {k_: len(v) for k_, v in bad.items()}}, indent=1))
    rng = random.Random(int(seed))
    for k_, v in kinds.items():
        for r in rng.sample(v, min(int(nshow), len(v))):
            pre = max([x for x in items.get((r["author"], r["id"]), []) if r["reply"].startswith(x)], key=len, default="") if k_ == "completed again" else ""
            print(f"\n===== {k_} | {r['author']} | {r['id']} | {r['source']}" + (f" | kept prefix: {pre.count(chr(10))} lines" if pre else ""))
            body = r["reply"]; print(body[:len(pre)] + ("<<<<< CUT: the author continues here >>>>>\n" if pre else "") + body[len(pre):len(pre) + 1500])
    print("\nSANITY_OK" if not bad else "\nSANITY_FAIL " + json.dumps({k_: v[:6] for k_, v in bad.items()}))


def main():
    if sys.argv[1] in ("pool", "fixitems", "pool2", "nextround", "final", "sanity"):
        return {"pool": pool, "fixitems": fixitems, "pool2": pool2, "nextround": nextround, "final": final, "sanity": sanity}[sys.argv[1]](*sys.argv[2:])
    out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
    base = json.load(open(DL + "xl_base_ids.json")); rows = base["rows"]; rowset = set(rows); held = set(base["held"])
    short = {r["id"]: r for r in (json.loads(l) for l in open(R + "xqlShort/xl_rows_short.jsonl"))}; assert set(short) == rowset
    cut = lambda t: t[:t.index("```") + 3]  # noqa: E731
    cand = collections.defaultdict(list)                                # (author, id) -> [(source, text)] in order of preference
    for i in rows:
        cand[("e4b", i)].append(("shortest verified", short[i]["reply"]))
    for g in (json.loads(l) for l in open(R + "xqlB/xl_build_gens.jsonl")):
        if g["model"] == "e4b" and g["pool"] == "train" and g["pass"] and "```" in g["text"] and g["id"] in rowset:
            cand[("e4b", g["id"])].append(("build " + g["kind"], cut(g["text"])))
    finals = [json.load(open(DL + f))["per problem"]["qwen"] for f in EVALS]
    for g in (json.loads(l) for l in open(R + "xqcDf2/xl_gens_qwen.jsonl")):
        if g["id"] in rowset and all(f[g["id"]] for f in finals) and "```" in g["text"]:
            cand[("e2b", g["id"])].append(("greedy", cut(g["text"])))
    out, why, kept = [], collections.Counter(), collections.defaultdict(list)
    for au in ("e4b", "e2b"):                                           # E4B's rows first, so a near-copy by E2B of E4B's text is the one left out
        for i in rows:
            n = 0
            for src, t in cand.get((au, i), []):
                if n >= CAPN:
                    break
                w = reason(t)
                if w is None and any(sim(code(t), code(o)) >= 0.9 for o in kept[i]):
                    w = "near"
                if w:
                    why[f"{au}: {w}"] += 1; continue
                kept[i].append(t); n += 1; out.append({"id": i, "ctx": short[i]["ctx"], "reply": t, "author": au, "source": src, "difficulty": short[i]["difficulty"]})
    p = os.path.join(out_dir, "xl_rows_fit.jsonl")
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest(); back = [json.loads(l) for l in open(p)]
    audit = collections.Counter(reason(r["reply"]) or "clean" for r in back); nb = collections.Counter(r["author"] for r in back); byp = collections.Counter(r["id"] for r in back)
    dup = sum(1 for i in byp for a in range(len(kept[i])) for b in range(a + 1, len(kept[i])) if sim(code(kept[i][a]), code(kept[i][b])) >= 0.9)
    assert set(audit) == {"clean"}, audit
    assert dup == 0, dup
    assert not {r["id"] for r in back} & held and all(r["id"] in rowset and r["ctx"] == short[r["id"]]["ctx"] for r in back)
    cs = lambda t: sum(l.strip().startswith("#") for l in code(t).split("\n")) / max(1, sum(1 for l in code(t).split("\n") if l.strip()))  # noqa: E731
    st = {"rows": len(back), "by author": dict(nb), "problems": len(byp), "problems by author": {a: len({r["id"] for r in back if r["author"] == a}) for a in nb}, "sources": dict(collections.Counter(r["author"] + ": " + r["source"] for r in back)),
          "left out": dict(why), "audit of the written rows": dict(audit), "near-copies among the written rows": dup, "comment-line share (mean)": {a: round(sum(cs(r["reply"]) for r in back if r["author"] == a) / nb[a], 3) for a in nb},
          "reply chars (median, max)": [sorted(len(r["reply"]) for r in back)[len(back) // 2], max(len(r["reply"]) for r in back)], "by difficulty": dict(collections.Counter(r["difficulty"] for r in back)), "cap per author and problem": CAPN, "sha": sha}
    json.dump(st, open(os.path.join(out_dir, "xl_fit.json"), "w"), indent=1); print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
