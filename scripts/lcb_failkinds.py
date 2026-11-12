"""209/209a: how the answers of the contest-coding evaluation fail, per arm, from the sessions' stored greedy answers
(runs/<session>/xl_gens_<arm>.jsonl: id, text, pass). Kinds, for a failing answer:
  unfinished  the answer has no closing fence (it used the 768 new tokens): "repeating" if one line shape (digits masked) occurs 6+ times,
              "mostly comments" if half or more of its lines are comments, else "other";
  stub        finished, and its last code line is a def with no body (only comments follow, e.g. "# ... implementation ...");
  syntax      finished, not a stub, does not parse;
  wrong       finished and parses: wrong output, an error at run time or too slow (the stored result does not say which).
Also: what the stubs of one arm cost it (lost problems that are stubs; what the other arms do on those problems).
usage: lcb_failkinds.py NAME=session/arm ...   e.g.  alone=xqcDf2/qwen D4f_s2=xqcDf2/onl_d4f C_s2=xqcC2/onl_c copy_s2=xqcF2/onl_sft
       the first name is the reference (E2B alone); the second is the arm whose stubs are costed. Standard library only."""
import ast, collections, json, os, re, sys
_H = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_H, "..", "runs", ""); BASE = os.path.join(_H, "..", "..", "papers", "kaggle_bridge", "data", "lcb", "xl_base_ids.json")
if not os.path.exists(BASE):                                             # the public repository's layout: scripts/ beside data/ (answers gzipped in data/generations)
    BASE = os.path.join(_H, "..", "data", "lcb", "xl_base_ids.json")


def gens(s, arm):
    """the stored greedy answers of a session's arm: runs/<session>/xl_gens_<arm>.jsonl, or data/generations/<session>/xl_gens_<arm>.jsonl.gz"""
    import gzip
    f = RUNS + f"{s}/xl_gens_{arm}.jsonl"
    if os.path.exists(f):
        return [json.loads(l) for l in open(f)]
    return [json.loads(l) for l in gzip.open(os.path.join(_H, "..", "data", "generations", s, f"xl_gens_{arm}.jsonl.gz"), "rt")]
KINDS = ["pass", "wrong", "syntax", "stub", "unfinished, repeating", "unfinished, mostly comments", "unfinished, other"]


def code_of(r):
    t = r["text"].rstrip(); fin = t.endswith("```"); return (t[:-3] if fin else t), fin


def stub(r):
    c, fin = code_of(r); L = [l for l in c.split("\n") if l.strip() and not l.strip().startswith("#")]
    return fin and bool(L) and L[-1].strip().startswith("def ") and L[-1].rstrip().endswith(":")


def kind(r):
    if r["pass"]:
        return "pass"
    c, fin = code_of(r); lines = [l.strip() for l in c.split("\n") if l.strip()]
    if not fin:
        shp = collections.Counter(re.sub(r"\d+", "N", l) for l in lines if len(l) > 8); top = shp.most_common(1)[0][1] if shp else 0
        return "unfinished, repeating" if top >= 6 else ("unfinished, mostly comments" if sum(l.startswith("#") for l in lines) >= 0.5 * max(1, len(lines)) else "unfinished, other")
    if stub(r):
        return "stub"
    try:
        ast.parse(c)
    except SyntaxError:
        return "syntax"
    return "wrong"


def main():
    base = json.load(open(BASE)); held, rows = base["held"], base["rows"]; G = {}
    for a in sys.argv[1:]:
        n, p = a.split("="); s, arm = p.split("/"); G[n] = {r["id"]: r for r in gens(s, arm)}
        assert set(held) <= set(G[n]) and set(rows) <= set(G[n]), n
    names = list(G)
    for scope, ids in (("held problems", held), ("row problems", rows)):
        print(f"== {scope} ({len(ids)})")
        for n in names:
            c = collections.Counter(kind(G[n][i]) for i in ids); print(f"  {n:16s}", " | ".join(f"{k} {c[k]}" for k in KINDS))
    if len(names) > 1:
        q, d = G[names[0]], G[names[1]]
        for scope, ids in (("held", held), ("row", rows)):
            S = [i for i in ids if stub(d[i])]; lost = [i for i in ids if q[i]["pass"] and not d[i]["pass"]]; ns = [i for i in ids if not stub(d[i])]
            print(f"== {names[1]}, {scope}: {len(S)} stubs | lost {len(lost)} (kinds {dict(collections.Counter(kind(d[i]) for i in lost))})")
            print("   on its stub problems, passes:", ", ".join(f"{n} {sum(G[n][i]['pass'] for i in S)}" for n in names), f"| {names[0]} also a stub there: {sum(stub(q[i]) for i in S)}")
            print(f"   on the other {len(ns)} problems, passes:", ", ".join(f"{n} {sum(G[n][i]['pass'] for i in ns)}" for n in names))


if __name__ == "__main__":
    main()
