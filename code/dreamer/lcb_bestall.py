"""Addendum 177c, the judge of the copied text (user 11-11: "we need to use a better/larger judge though"; "use the openrouter deepseek as
the judge"). For the loop's fallback text (the solution copied on a problem where no own attempt passes), a larger judge than 209g-209r's
(DeepSeek V4 Pro 0813 on OpenRouter, in place of DeepSeek V4.1 Flash 1017) picks, per training problem, the best of the problem's verified
solutions. The judge selects; it never writes or edits text. Every candidate passed the problem's tests in the sessions that wrote it.
  groups  OWN_DIR STREAM_ROWS OUT_DIR   per problem of STREAM_ROWS (E4B's verified rows, xl_rows.jsonl): E4B's stream row and the distinct
                                        verified texts of OWN_DIR (xl_rows_pool.jsonl: E4B's and E2B's own passing solutions;
                                        xl_rows_pool2_*.jsonl: their authors' passing rewrites and resamples; xl_rows_fit.jsonl), each
                                        ending with its one fence and passing the rules of form; the stream row and the seven shortest
                                        others, shortest first -> OUT_DIR/all_groups.jsonl
  calib   OWN_DIR OUT_DIR OLD_FIT GENS  groups with one planted fault among the real candidates -> OUT_DIR/calib_groups.jsonl:
                                        (a) a row of OLD_FIT (209e's fit rows) with the stub's shape or a comment that names a plan (209g's
                                        calibration rows), (b) a pool row the word list calls thinking aloud and both votes of 209i's first
                                        reading called rambling, (c) a placeholder answer of untrained E2B on the problem (GENS: stored greedy
                                        answers, xl_gens_qwen.jsonl; lcb_failkinds.py's rule). The planted text is put where its length puts it.
  calibreport OUT_DIR VOTES             votes that pick the planted text, by kind of fault (the gate: no vote picks a stub or plan row as
                                        acceptable, and at most 2 of the thinking-aloud votes pick the planted text as acceptable)
  picks   OUT_DIR VOTES                 per problem: both votes must call their pick acceptable; different picks: the shorter. Writes
                                        OUT_DIR/xl_rows_best.jsonl (id, ctx, reply, author, source, difficulty: one row per problem, the
                                        format of xl_rows_fit.jsonl) and prints what was picked.
judge:  python3 judge_rows.py run OUT_DIR/all_groups.jsonl --out OUT_DIR/all_votes.jsonl --best --votes 2 --model deepseek/deepseek-v4-pro-0813 --cap 4
Standard library only."""
import collections, glob, hashlib, json, os, sys
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H)
import lcb_fit_rows as FR

NC = 8                                                                   # candidates per group: the stream row and the 7 shortest others


def _rows(f):
    return [json.loads(l) for l in open(f)]


def _ok(t):
    return FR.reason(t, soft=False) is None                              # one closing fence at the end, no placeholder, parses, no stub


def _cands(own, stream):
    """problem id -> {text: (author, kind)} over every verified text held, and the stream rows by id"""
    S = {r["id"]: r for r in _rows(stream)}; assert len(S) == len(_rows(stream)), "one stream row per problem"
    C = collections.defaultdict(dict)
    for f, kind in [(os.path.join(own, "xl_rows_pool.jsonl"), "own solution (a row of the pool: as first written, or its author's continuation from 209g)")] + [(f, "its author's rewrite or resample") for f in sorted(glob.glob(os.path.join(own, "xl_rows_pool2_*.jsonl")))] \
            + [(os.path.join(own, "xl_rows_fit.jsonl"), "fit row")]:
        for r in _rows(f):
            if r.get("expect") or r["id"] not in S or not _ok(r["reply"]):   # 209g's planted calibration rows are in the pool file: never candidates
                continue
            C[r["id"]].setdefault(r["reply"], (r["author"], kind))
    return C, S


def groups(own, stream, out):
    C, S = _cands(own, stream); os.makedirs(out, exist_ok=True); G = []; n = collections.Counter()
    for i in sorted(S):
        s = S[i]["reply"]; n["stream rows that fail a rule of form (kept as a candidate, counted)"] += not _ok(s)
        others = sorted((t for t in C[i] if t != s), key=len)[:NC - 1]; c = sorted([s] + others, key=len)
        kinds = ["E4B's stream row" if t == s else f"{C[i][t][0]}: {C[i][t][1]}" for t in c]
        G.append({"rid": f"all:{i}", "id": i, "ctx": S[i]["ctx"], "cands": c, "kinds": kinds, "authors": ["e4b" if t == s else C[i][t][0] for t in c], "stream": c.index(s) + 1, "difficulty": S[i]["difficulty"], "n verified": len(C[i] | {s: 0})})
        n["groups"] += 1; n["groups with only the stream row"] += len(c) == 1
    with open(os.path.join(out, "all_groups.jsonl"), "w") as fh:
        for g in G:
            fh.write(json.dumps(g) + "\n")
    k = sorted(len(g["cands"]) for g in G); ch = sorted(sum(len(t) for t in g["cands"]) + len(g["ctx"]) for g in G)
    print(json.dumps({**dict(n), "candidates per group (median, min, max)": [k[len(k) // 2], k[0], k[-1]], "characters per request (median, max)": [ch[len(ch) // 2], ch[-1]],
                      "sha256 all_groups.jsonl": hashlib.sha256(open(os.path.join(out, "all_groups.jsonl"), "rb").read()).hexdigest()}))


def calib(own, out, old_fit, gens):
    import re
    import lcb_failkinds as FK
    G = {g["id"]: g for g in _rows(os.path.join(out, "all_groups.jsonl"))}; P = []
    for r in _rows(gens):                                                # (c) the model's own placeholder answers, as stored
        if r["id"] in G and FK.kind(r) == "stub":
            P.append(({"id": r["id"], "reply": r["text"]}, "stub"))
    for r in _rows(old_fit):                                             # (a) 209g's calibration rule on 209e's rows
        if r["source"].startswith("with a plan") and re.search(r"(?im)^\s*#.*\bplan\b", r["reply"]):
            P.append((r, "names a plan"))
        elif FR.reason(r["reply"], soft=False) == "stub":
            P.append((r, "stub"))
    V = collections.defaultdict(dict)
    for v in _rows(os.path.join(own, "verdicts1.jsonl")):
        if v.get("verdict"):
            V[v["rid"]][v["vote"]] = v["verdict"]
    for r in _rows(os.path.join(own, "xl_rows_pool.jsonl")):             # (b) thinking aloud by the word list and by both votes of 209i's first reading
        if not r.get("expect") and FR.reason(r["reply"]) == "aloud" and len(V.get(r["rid"], {})) >= 2 and all(x == "not_clean" for x in V[r["rid"]].values()):
            P.append((r, "thinking aloud"))
    out_rows, seen, n = [], set(), collections.Counter()
    for r, kind in P:
        g = G.get(r["id"])
        if g is None or (r["id"], kind) in seen:
            continue
        real = [t for t in g["cands"] if t != r["reply"]]
        if kind == "thinking aloud":                                     # the real candidates must include one the first reading did not flag: the fit rows are those
            real = [t for t, k_ in zip(g["cands"], g["kinds"]) if t != r["reply"] and (k_.endswith("rewrite or resample") or k_.endswith("fit row"))]
        if not real:
            n[f"{kind}: no other candidate"] += 1; continue
        seen.add((r["id"], kind)); c = sorted(real[:NC - 1] + [r["reply"]], key=len)
        out_rows.append({"rid": f"calib:{kind.replace(' ', '_')}:{r['id']}", "id": r["id"], "ctx": g["ctx"], "cands": c, "planted": c.index(r["reply"]) + 1, "plant": kind}); n[kind] += 1
    per = {"stub": 12, "names a plan": 12, "thinking aloud": 16}; keep = []
    for kind, m in per.items():                                          # a fixed, seeded choice per kind
        ks = sorted((x for x in out_rows if x["plant"] == kind), key=lambda x: hashlib.sha256(x["rid"].encode()).hexdigest())[:m]; keep += ks
    with open(os.path.join(out, "calib_groups.jsonl"), "w") as fh:
        for g in keep:
            fh.write(json.dumps(g) + "\n")
    print(json.dumps({"planted groups available": dict(n), "used": dict(collections.Counter(g["plant"] for g in keep)), "planted text is the shortest candidate in": sum(g["planted"] == 1 for g in keep)}))


def _votes(f):
    V = collections.defaultdict(dict)
    for v in _rows(f):
        if v.get("verdict") == "picked":
            V[v["rid"]][v["vote"]] = v
    return V


def calibreport(out, votes):
    G = {g["rid"]: g for g in _rows(os.path.join(out, "calib_groups.jsonl"))}; V = _votes(votes); st = collections.defaultdict(collections.Counter); cost = sum(float(v.get("cost") or 0) for v in _rows(votes))
    for rid, g in G.items():
        vs = V.get(rid, {}); st[g["plant"]]["groups"] += 1; st[g["plant"]]["votes missing"] += 2 - min(2, len(vs))
        for v in vs.values():
            st[g["plant"]]["votes"] += 1
            if v["best"] == g["planted"]:
                st[g["plant"]]["picked the planted text"] += 1; st[g["plant"]]["... and called it acceptable"] += bool(v["acceptable"])
        st[g["plant"]]["groups where the picks would put the planted text in the file"] += len(vs) >= 2 and all(v["acceptable"] for v in vs.values()) and min(v["best"] for v in vs.values()) == g["planted"]
    hard = sum(st[k]["... and called it acceptable"] for k in ("stub", "names a plan")); soft = st["thinking aloud"]["... and called it acceptable"]; miss = sum(c["votes missing"] for c in st.values())
    ok = hard == 0 and soft <= 2 and miss == 0 and all(st[k]["groups"] >= 8 for k in ("stub", "names a plan", "thinking aloud"))   # review 177c: an empty or thin set never passes
    print(json.dumps({k: dict(c) for k, c in st.items()})); print(json.dumps({"models": dict(collections.Counter(v.get("model") for v in _rows(votes) if v.get("verdict"))), "cost": round(cost, 4)}))
    print("CALIB_OK" if ok else f"CALIB_FAILED (stub or plan picks accepted {hard}, thinking-aloud picks accepted {soft} of at most 2, votes missing {miss})")
    sys.exit(0 if ok else 1)


def picks(out, votes):
    G = {g["rid"]: g for g in _rows(os.path.join(out, "all_groups.jsonl"))}; V = _votes(votes); rows, st = [], collections.Counter(); ratio = []; cost = sum(float(v.get("cost") or 0) for v in _rows(votes))
    for rid, g in sorted(G.items()):
        vs = V.get(rid, {}); assert len(vs) >= 2, f"no two votes for {rid}"
        if not all(v["acceptable"] for v in vs.values()):
            st["no pick acceptable to both votes: no row (the loop keeps E4B's stream row)"] += 1; continue
        k = min(v["best"] for v in vs.values()) - 1; ch = g["cands"][k]; st["same pick" if len({v["best"] for v in vs.values()}) == 1 else "different picks: the shorter"] += 1   # review 177c: cands are sorted by length, so the lower number is the shorter text (a tie in length is decided by the order sent, not by a set's order)
        assert _ok(ch) or k + 1 == g["stream"], rid
        if not _ok(ch):                                                  # a stream row that fails a rule of form is never written
            st["the pick is a stream row that fails a rule of form: no row"] += 1; continue
        st["the pick is the stream row" if k + 1 == g["stream"] else f"the pick is by {g['authors'][k]}, not the stream row"] += 1; st["the pick is the shortest candidate"] += k == 0
        ratio.append(len(ch) / len(g["cands"][g["stream"] - 1]))
        rows.append({"id": g["id"], "ctx": g["ctx"], "reply": ch, "author": g["authors"][k], "source": g["kinds"][k] + "; picked by the larger judge among the problem's verified solutions (both votes acceptable)", "difficulty": g["difficulty"]})
    p = os.path.join(out, "xl_rows_best.jsonl")
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    d = sorted(ratio); L = sorted(len(r["reply"]) for r in rows)
    print(json.dumps({"problems judged": len(G), "rows written": len(rows), **dict(st), "length of the pick against the stream row (median ratio)": round(d[len(d) // 2], 2) if d else None, "median characters": L[len(L) // 2] if L else None,
                      "models": dict(collections.Counter(v.get("model") for v in _rows(votes) if v.get("verdict"))), "cost": round(cost, 4), "sha256 xl_rows_best.jsonl": hashlib.sha256(open(p, "rb").read()).hexdigest()}))


if __name__ == "__main__":
    {"groups": groups, "calib": calib, "calibreport": calibreport, "picks": picks}[sys.argv[1]](*sys.argv[2:])
