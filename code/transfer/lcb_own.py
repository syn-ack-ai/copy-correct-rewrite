"""Amendment 209e (user 11-10: "clean up the data that we train the corrector on", "make sure we use clean, working code, verified both either
from e4b or e2b", "do a smoke test/sanity check of the data before we train on it"): the corrector's coding rows as verified solutions only.
  rows ON_DIR   appends kind "fit_lcb" (split train) to ON_DIR (after lcb_rows.py) from LCB_FIT (xl_rows_fit.jsonl, sha256 LCB_FIT_SHA): solutions
                to the coding study's row problems that passed their tests in an earlier session, at most two per author and problem.
                E4B's: its shortest verified solution and the build's passing greedy answer or samples. E2B's: its own greedy answer where it
                solves the problem alone, and the solution it wrote with a plan in the prompt (the runbook rows; the row's context is the
                plain prompt). A row is left out if a comment line starts with an ellipsis ("# ...": a placeholder), if it is longer than
                the fit reads, or if it does not pass its problem's tests here: every row is run before it is written.
  check ON_DIR  the smoke test before any fit: every fit_lcb row and every row of the teaching stream (kind lcb, train) is read back FROM THE
                TOKENS the fit trains on and must (1) decode to its prompt, as "<bos>" + xlearn's prompt, (2) end with its closing fence,
                (3) pass every test of its problem in this session (a failure is tried once more alone), (4) parse, with no stub (a def
                whose body is only comments) and, for fit_lcb, no placeholder comment, (5) fit the fit's lengths (no cut: reply <= 1024
                tokens, context + reply <= 6144), (6) belong to a row problem, never a held one, (7) not repeat another row. FIT_W (the
                fit's draw weights) may name only fit_lcb, prose, question and story. Writes ON_DIR/out/lcb_data_check.json, prints a
                summary and two rows per author to read, and prints DATA_CHECK_OK only when nothing fails (exit 1 otherwise).
Model-written text only; nothing is added to it.
  XB_DIR=... XB_STUDENT=... XC_DATA=... LCB_FIT=... LCB_FIT_SHA=... python lcb_own.py rows ON_DIR ; FIT_W=... python lcb_own.py check ON_DIR"""
import ast, collections, hashlib, json, os, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xlearn as XL  # noqa: E402  (prompts, pools and tests exactly as the coding study builds them; no model is loaded)
XC = XL.XC
PLACE = re.compile(r"^\s*#\s*\.\.\.", re.M); ALLOWED = {"fit_lcb", "prose", "question", "story"}; RMAX, CAP = 1024, 6144   # online.py's cuts


def is_stub(rep):
    L = [l for l in rep.rstrip()[:-3].split("\n") if l.strip() and not l.strip().startswith("#")]
    return bool(L) and L[-1].strip().startswith("def ") and L[-1].rstrip().endswith(":")


def passes(probs, texts):
    """every test passes; a failure in the parallel run is tried once more alone (no load near the time limit)"""
    ok = XL.score(probs, texts); return [o or bool(XC.run_problem(p, t)) for p, t, o in zip(probs, texts, ok)]


def rows():
    from transformers import AutoTokenizer
    D = Path(sys.argv[2]); tok = AutoTokenizer.from_pretrained(os.environ["XB_STUDENT"]); base = json.load(open(XL.BASE)); XC.selfcheck()
    fit_f = Path(os.environ["LCB_FIT"]); assert hashlib.sha256(fit_f.read_bytes()).hexdigest() == os.environ["LCB_FIT_SHA"], "the candidate rows are not the checked file"
    tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; rowset = set(base["rows"]); held = set(base["held"])
    cand = [(r["author"], r) for r in (json.loads(l) for l in open(fit_f))]; out_c = collections.Counter(); keep = []; seen = set()
    for au, r in cand:
        i, rep = r["id"], r["reply"]
        assert au in ("e4b", "e2b") and i in rowset and i not in held and r["ctx"] == byid[i]["prompt"] and rep.endswith("```") and rep.count("```") == 1, (au, i)
        if PLACE.search(rep) or is_stub(rep):
            out_c[au + ": a placeholder comment or a stub"] += 1; continue
        if (i, rep) in seen:
            out_c[au + ": the same text as another row"] += 1; continue
        nc, nr = len(tok("<bos>" + r["ctx"], add_special_tokens=False)["input_ids"]), len(tok(rep, add_special_tokens=False)["input_ids"])
        if nr > RMAX or nc + nr > CAP:
            out_c[au + ": longer than the fit reads"] += 1; continue
        seen.add((i, rep)); keep.append((au, r))
    ok = passes([byid[r["id"]] for _, r in keep], [r["reply"] for _, r in keep])
    for (au, r), o in zip(keep, ok):
        if not o:
            out_c[au + ": does not pass its tests in this session"] += 1; print("left out (does not pass here):", au, r["id"])
    keep = [k for k, o in zip(keep, ok) if o]
    assert len(keep) >= 0.9 * len(cand), f"only {len(keep)} of {len(cand)} candidate rows are left: the tests or the rows are not what they were"   # review 209e
    ids = np.load(D / "s3_ids.npy"); assert ids.dtype == np.int32; old = [json.loads(l) for l in open(D / "s3_calls.jsonl")]
    assert any(r.get("kind") == "lcb" for r in old) and not any(r.get("kind") == "fit_lcb" for r in old), "ON_DIR needs the lcb rows and no fit_lcb rows yet"
    new, out, off = [], [], len(ids)
    for au, r in keep:
        c = tok("<bos>" + r["ctx"], add_special_tokens=False)["input_ids"]; r_ = tok(r["reply"], add_special_tokens=False)["input_ids"]
        assert c[0] == tok.bos_token_id and len(r_) > 0 and tok.bos_token_id not in c[1:] + r_ and tok.pad_token_id not in c + r_, r["id"]
        out.append({"kind": "fit_lcb", "split": "train", "off": off, "ctx_len": len(c), "reply_len": len(r_), "action": "text", "run": r["id"], "repo": byid[r["id"]]["platform"], "id": r["id"],
                    "difficulty": byid[r["id"]]["difficulty"], "author": au})
        new += c + r_; off += len(c) + len(r_)
    np.save(D / "s3_ids.npy", np.concatenate([ids, np.asarray(new, dtype=np.int32)]))
    with open(D / "s3_calls.jsonl", "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    chk = np.load(D / "s3_ids.npy", mmap_mode="r"); assert len(chk) == off and (chk[:len(ids)] == ids).all()
    n = collections.Counter(r["author"] for r in out); print("left out:", dict(out_c))
    print("FIT_ROWS_OK", len(out), "rows:", dict(n), "| problems", len({r["id"] for r in out}), "| tokens appended", off - len(ids), "| rows before, unchanged:", len(old), flush=True)


def check():
    from transformers import AutoTokenizer
    D = Path(sys.argv[2]); tok = AutoTokenizer.from_pretrained(os.environ["XB_STUDENT"]); base = json.load(open(XL.BASE)); XC.selfcheck()
    tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; rowset = set(base["rows"]); held = set(base["held"])
    ids = np.load(D / "s3_ids.npy", mmap_mode="r"); allr = [json.loads(l) for l in open(D / "s3_calls.jsonl")]
    fit = [r for r in allr if r.get("kind") == "fit_lcb"]; stream = [r for r in allr if r.get("kind") == "lcb" and r["split"] == "train"]
    bad = collections.defaultdict(list); W = dict((k, float(v)) for k, v in (kv.split(":") for kv in os.environ["FIT_W"].split(",")))
    if set(W) - ALLOWED or "fit_lcb" not in W:
        bad["the fit's weights name a kind that is not allowed"] += sorted(set(W) - ALLOWED) or ["fit_lcb is missing"]
    kinds = collections.Counter(r.get("kind", "agent") for r in allr if r["split"] == "train")
    for k in W:
        if not kinds.get(k):
            bad["a kind in the fit's weights has no training rows"].append(k)
    if not fit or any(r["split"] != "train" for r in fit):
        bad["no fit_lcb rows, or one outside the training split"].append(len(fit))
    if len(stream) != len(rowset):
        bad["the teaching stream does not have one row per row problem"].append(len(stream))
    todo, seen = [], set()
    for name, rs in (("fit_lcb", fit), ("lcb stream", stream)):
        for r in rs:
            i = r["id"]; ctx = tok.decode(np.asarray(ids[r["off"]:r["off"] + r["ctx_len"]]).tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            rep = tok.decode(np.asarray(ids[r["off"] + r["ctx_len"]:r["off"] + r["ctx_len"] + r["reply_len"]]).tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if i not in rowset or i in held:
                bad["not a row problem, or a held problem"].append((name, i)); continue
            if ctx != "<bos>" + byid[i]["prompt"]:
                bad["the context does not decode to the problem's prompt"].append((name, i))
            if not rep.rstrip().endswith("```") or rep.count("```") != 1:
                bad["no single closing fence at the end"].append((name, i))
            if r["reply_len"] > RMAX or r["ctx_len"] + r["reply_len"] > CAP:
                bad["longer than the fit reads (it would be cut)"].append((name, i, r["ctx_len"], r["reply_len"]))
            if is_stub(rep):
                bad["a stub (a def whose body is only comments)"].append((name, i))
            try:
                ast.parse(rep.rstrip()[:-3])
            except SyntaxError:
                bad["does not parse"].append((name, i))
            if name == "fit_lcb":
                if PLACE.search(rep):
                    bad["a placeholder comment in a fit row"].append((name, i))
                if (i, rep) in seen:
                    bad["a fit row repeats another"].append((name, i))
                seen.add((i, rep))
            todo.append((name, r, rep))
    uniq = {}                                                            # each distinct (problem, text) is run once
    for name, r, rep in todo:
        uniq.setdefault((r["id"], rep), []).append(name)
    keys = list(uniq); ok = passes([byid[i] for i, _ in keys], [t for _, t in keys])
    for (i, _), o in zip(keys, ok):
        if not o:
            bad["does not pass its tests"].append((uniq[(i, _)][0], i))
    au = collections.Counter(r.get("author") for r in fit); L = sorted(r["reply_len"] for r in fit)
    cm = lambda t: sum(l.strip().startswith("#") for l in t.split("\n")) / max(1, sum(1 for l in t.split("\n") if l.strip()))  # noqa: E731
    rep_of = {id(r): rep for _, r, rep in todo}
    out = {"fit_lcb rows": len(fit), "by author": dict(au), "problems with a fit row": len({r["id"] for r in fit}), "problems with a row of both authors": sum(1 for v in collections.Counter(r["id"] for r in fit).values() if v > 1),
           "teaching-stream rows": len(stream), "distinct texts run on their tests": len(keys), "passing": int(sum(ok)), "reply tokens (median, max)": [L[len(L) // 2], L[-1]] if L else None,
           "comment-line share by author": {a: round(float(np.mean([cm(rep_of[id(r)]) for r in fit if r.get("author") == a])), 3) for a in au},
           "by difficulty": dict(collections.Counter(r["difficulty"] for r in fit)), "training rows by kind in ON_DIR": dict(kinds), "fit weights": W, "failures": {k: v for k, v in bad.items()}}
    json.dump(out, open(D / "out" / "lcb_data_check.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "failures"}))
    for a in sorted(au):                                                 # two rows per author, to read
        for r in [r for r in fit if r.get("author") == a][:2]:
            print(f"--- {a} {r['id']} ({r['reply_len']} tokens)\n" + rep_of[id(r)][:500])
    if bad:
        for k, v in bad.items():
            print("DATA_CHECK_FAIL", k, len(v), v[:8])
        sys.exit(1)
    print("DATA_CHECK_OK", len(fit), "fit rows and", len(stream), "stream rows: every one decodes to its prompt, ends with its fence, passes its tests, parses, is no stub", flush=True)


if __name__ == "__main__":
    {"rows": rows, "check": check}[sys.argv[1]]()
