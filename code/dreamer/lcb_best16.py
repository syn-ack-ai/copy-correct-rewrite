"""Amendment 209i, best of 16 (user 11-11: "cant we run a resample of 16x and then have a judge decide which is best? we have a script for
that already i belive": the agent rows' repair rounds, 8 + 8 samples and a judge that selects). For the coding rows:
  items   OUT_DIR SHARDS      the flagged rows with no accepted version after the first batches get 12 more samples from their own author
                              model: FIX_NC=6 continuations from the cut and 6 fresh answers (16 with the 4 they have); written in SHARDS files
  groups  OUT_DIR FIX_OUT...  per such row, its passing versions over all batches (tests, closing fence, the rules of form; the original
                              included; the author's own rewrites too, lcb_fix.py FIX_MODE=rewrite), the eight shortest, each with how it
                              was written, for the judge to pick from (judge_rows.py --best, two votes)
  picks   OUT_DIR PICKS       the pick per row: both votes must call their pick acceptable; if they pick different versions, the shorter
                              one. Writes xl_rows_pool2_best.jsonl (one version per row) and verdicts2_best.jsonl for lcb_fit_rows.py final.
Standard library only."""
import collections, hashlib, json, os, sys
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H)
import lcb_fit_rows as FR


def _votes(path):
    V = collections.defaultdict(dict)
    for v in map(json.loads, open(path)):
        if v.get("verdict"):
            V[v["rid"]][v["vote"]] = v
    return V


def unresolved(D):
    """flagged rows (rambling or overlong in both votes of the first reading) with no version accepted in all six votes so far"""
    cuts = {}
    for f in ("cont_items.jsonl", "cont_items_len.jsonl"):
        for it in (json.loads(l) for l in open(os.path.join(D, f))):
            if it["prefix"] or it["rid"] not in cuts:
                cuts.setdefault(it["rid"], it) if not it["prefix"] else cuts.__setitem__(it["rid"], it)
    acc = set()
    for t in ("r1", "r2"):
        p = os.path.join(D, f"verdicts2_{t}.jsonl")
        if os.path.exists(p):
            V = collections.defaultdict(list)
            for v in map(json.loads, open(p)):
                V[v["rid"]].append(v["verdict"])
            acc |= {rid.split("#")[0] for rid, vs in V.items() if len(vs) == 6 and all(x == "clean" for x in vs)}
    return {rid: it for rid, it in cuts.items() if rid not in acc}, acc


def items(D, shards):
    un, acc = unresolved(D); out = []
    for rid, it in sorted(un.items()):
        base = {k: it[k] for k in ("rid", "id", "author", "reply", "notes")}
        out.append({**base, "prefix": it["prefix"]}); out.append({**base, "prefix": ""})
    n = int(shards); out.sort(key=lambda r: (r["author"], r["rid"], r["prefix"] == ""))
    by = collections.defaultdict(list)
    for k, r in enumerate(out):
        by[k % n].append(r)
    for k in range(n):
        d = os.path.join(D, "..", f"fix209k{k}"); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "fix_items.jsonl"), "w") as fh:
            for r in by[k]:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps({"flagged rows": len(un) + len(acc), "already accepted": len(acc), "to resample": len(un), "by author": dict(collections.Counter(it["author"] for it in un.values())), "items": len(out), "shards": {k: len(v) for k, v in by.items()}}))


def groups(D, *fix_outs):
    un, _ = unresolved(D); pool = {r["rid"]: r for r in (json.loads(l) for l in open(os.path.join(D, "xl_rows_pool.jsonl")))}
    V = collections.defaultdict(dict); n = collections.Counter()      # rid -> {text: how it was written}
    KIND = {"continued": "continued", "sampled again": "sampled again", "greedy": "rewritten", "sample": "rewritten"}   # lcb_fix.py's kinds: FIX_MODE=continue, then FIX_MODE=rewrite
    for f in fix_outs:
        for x in (json.loads(l) for l in open(f)):
            if x["rid"] in un:
                n["versions"] += 1
                if x["pass"] and x["fenced"] and FR.reason(x["text"], soft=False) is None:
                    V[x["rid"]].setdefault(x["text"], KIND[x["kind"]])
    out = []
    for rid in sorted(un):
        c = sorted(set(V[rid]) | {pool[rid]["reply"]}, key=len)[:8]    # the original is a candidate too; the eight shortest
        out.append({"rid": rid, "id": pool[rid]["id"], "ctx": pool[rid]["ctx"], "cands": c, "kinds": ["original" if t == pool[rid]["reply"] else V[rid][t] for t in c], "orig": pool[rid]["reply"], "n passing": len(V[rid])})
    with open(os.path.join(D, "best_groups.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    k = sorted(r["n passing"] for r in out)
    print(json.dumps({"rows": len(out), "versions read": n["versions"], "passing versions per row (median, min, max)": [k[len(k) // 2], k[0], k[-1]] if k else None, "rows with no passing new version": sum(1 for r in out if r["n passing"] == 0),
                      "candidates by kind": dict(collections.Counter(x for r in out for x in r["kinds"]))}))


def picks(D, pfile):
    G = {r["rid"]: r for r in (json.loads(l) for l in open(os.path.join(D, "best_groups.jsonl")))}; pool = {r["rid"]: r for r in (json.loads(l) for l in open(os.path.join(D, "xl_rows_pool.jsonl")))}
    V = _votes(pfile); out, st = [], collections.Counter(); delta = []
    for rid, g in G.items():
        vs = V.get(rid, {}); assert len(vs) >= 2, f"no two votes for {rid}"
        if not all(v["acceptable"] for v in vs.values()):
            st["the best is not acceptable to both votes: left out"] += 1; continue
        ch = sorted({g["cands"][v["best"] - 1] for v in vs.values()}, key=len)[0]; st["same pick" if len({v["best"] for v in vs.values()}) == 1 else "different picks: the shorter"] += 1
        kind = g["kinds"][g["cands"].index(ch)]; st["the pick is " + kind] += 1; o = pool[rid]; delta.append(len(ch) / len(g["orig"]))   # lcb_fit_rows.py sanity reads these labels
        lab = {"original": ", kept: the judge's pick among the original, the resamples and the rewrites", "continued": ", the judge's pick: continued from the cut by its author",
               "sampled again": ", the judge's pick: sampled again by its author", "rewritten": ", the judge's pick: rewritten by its author (told the kind of fault and the line)"}[kind]
        out.append({"rid": f"{rid}#best", "of": rid, "id": o["id"], "ctx": o["ctx"], "reply": ch, "author": o["author"], "source": o["source"] + lab, "difficulty": o["difficulty"], "soft": None})
    with open(os.path.join(D, "xl_rows_pool2_best.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(D, "verdicts2_best.jsonl"), "w") as fh:
        for r in out:
            for k in (0, 1):
                fh.write(json.dumps({"rid": r["rid"], "vote": k, "verdict": "clean", "issues": [], "reason": "the judge's pick, acceptable in both votes"}) + "\n")
    d = sorted(delta)
    print(json.dumps({"rows judged": len(G), "picked": len(out), **dict(st), "length of the pick against the original (median ratio)": round(d[len(d) // 2], 2) if d else None}))


if __name__ == "__main__":
    {"items": items, "groups": groups, "picks": picks}[sys.argv[1]](*sys.argv[2:])
