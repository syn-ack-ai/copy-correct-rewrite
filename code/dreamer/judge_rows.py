"""A second reader for the corrector's coding rows (amendment 209g; user 11-10: "cant we use deepseek or a different model to review them?").
Every row is a solution that already passes its problem's tests; the judge reads what tests cannot see: is the text clean training material?
Two independent votes per row (the project's rule for a model judge); a row is kept only if every vote says clean.
Sends only the problem statement and the model-written solution: no system prompt of ours, no names, no paths. OpenRouter key from the
repository's .env or ~/.config/openrouter/key (never printed), data collection denied, every call's cost logged, new calls stop at --cap dollars.
  python3 judge_rows.py run POOL.jsonl --out VERDICTS.jsonl [--votes 2] [--cap 3] [--workers 12] [--limit N] [--model M]
  python3 judge_rows.py run FIX_ITEMS.jsonl --out CUTS.jsonl --locate      (the first bad line of each flagged row, two votes; rows need ctx)
  python3 judge_rows.py run POOL.jsonl --out VERDICTS_DIRECT.jsonl --direct     (209i: written directly or rambling; "clean" = direct; the first rambling line)
  python3 judge_rows.py report POOL.jsonl VERDICTS.jsonl
POOL rows: rid, id, ctx (the problem prompt), reply (the solution with its closing fence), author; optional "expect" (clean | not_clean:
calibration rows whose answer is known) and "soft" (the word-list rule that would leave the row out, for the comparison in the report)."""
import argparse, collections, concurrent.futures as cf, json, os, re, threading, time, urllib.error, urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"; MODEL = "deepseek/deepseek-v4.1-flash-1017"
ISSUES = ("unfinished", "thinking_aloud", "refers_to_instructions", "leftover_code", "hardcoded", "other")
SYSTEM = """You review training data for a code model. You are given a programming problem and a Python solution that is already known to pass all hidden tests. Do not judge correctness or efficiency. Decide only whether the solution's text is clean training material.

A clean solution is complete, direct code. Its comments, if any, say what the code does or why (for example "# Read the input", "# dp[i] is the best sum ending at i", a short statement of the approach, numbered steps that describe the code below them).

Mark it not_clean if any of these holds:
- unfinished: a placeholder or stub (a function whose body is only comments, pass or "..."), a TODO, a part left to be written.
- thinking_aloud: comments in which the writer reasons its way to the answer instead of describing the code: second thoughts ("Wait", "Actually", "Let's re-read the problem", "Let's reconsider"), doubts or self-corrections, an approach that is discussed and then dropped, the examples traced by hand line after line, long restatements of the problem statement.
- refers_to_instructions: comments that mention a prompt, a plan, a hint, instructions, a user, or what the problem statement "seems to" have got wrong.
- leftover_code: a definition that is written twice or never used, abandoned earlier attempts, debug output.
- hardcoded: the sample inputs are special-cased instead of the problem being solved.
- other: anything else that should not be imitated (say what).

Ordinary explanatory comments are fine even when there are many of them, and so are phrases such as "we need" or "let's use a set" when they simply describe the code. Judge the text as a whole: one borderline phrase in otherwise plain code is clean. Small things are not reasons by themselves: an unused import, one commented-out line of set-up, a defensive try/except around reading input, or a comment that points to the problem's statement or constraints while explaining the code.

Answer with JSON only: {"verdict": "clean" or "not_clean", "issues": [zero or more of "unfinished", "thinking_aloud", "refers_to_instructions", "leftover_code", "hardcoded", "other"], "reason": "one short sentence"}"""


def api_key():
    """the OpenRouter key, never printed: the environment's OPENROUTER_KEY, else OPENROUTER_KEY in the repository's .env (git-ignored), else ~/.config/openrouter/key"""
    k = os.environ.get("OPENROUTER_KEY"); env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    if not k and os.path.exists(env):
        for l in open(env):
            m = re.match(r"^\s*OPENROUTER_KEY\s*=\s*(.*?)\s*$", l)
            if m:
                k = m.group(1).strip().strip("\"'")
    return k or open(os.path.expanduser("~/.config/openrouter/key")).read().strip()


DIRECT = """You review training data for a code model. You are given a programming problem and a Python solution with numbered lines. The solution is already known to pass all hidden tests. Do not judge correctness. Decide only whether it is written directly or rambles.

Direct: code is the bulk of the answer. Comments are short: a line or two that states the approach, and brief notes at steps that are not obvious.

Rambling, if any of these holds:
- a comment block of more than about six lines in a row that restates the problem, explains the plan in prose, analyses the constraints or walks through examples;
- comments on most lines that only repeat what the code says;
- more comment lines than code lines overall;
- a docstring or comment that reasons about the task instead of describing the code.

Answer with JSON only: {{"verdict": "clean" if the solution is direct, "not_clean" if it rambles, "issues": ["rambling"] or [], "first_bad_line": <the number of the first line of the first rambling block, or null if direct>, "reason": "one short sentence"}}"""

CONCISE = """You review training data for a code model. You are given a programming problem and a Python solution that is already known to pass all hidden tests. Do not judge correctness, and do not judge the comments. Decide only whether the code is unnecessarily long for this problem.

Unnecessarily long: the code is much longer than the problem needs, by roughly half or more, because of any of these: the same steps written out several times instead of once; long defensive handling of inputs that the problem's format rules out (several try/except blocks, repeated checks for empty or malformed input); helper functions, classes, variables or conversions that add nothing; a first approach or a brute-force version left in beside the real one; results built in several passes where one would do.

Not unnecessarily long: a solution whose length comes from the problem itself; one short guard around reading input; ordinary clear code that could be golfed a little.

Answer with JSON only: {{"verdict": "clean" if the length is reasonable, "not_clean" if it is unnecessarily long, "issues": ["overlong"] or [], "reason": "one short sentence naming what could go"}}"""

BEST = """You choose training data for a code model. You are given a programming problem and several Python solutions to it, numbered. Every one already passes all hidden tests, so do not judge correctness. Pick the one that is the best example to learn from.

The best example is complete and direct: the code is the bulk of it and is no longer than the problem needs; comments are short and say what the code does. Prefer, in this order: no thinking aloud in comments (second thoughts, doubts, self-corrections, examples traced by hand, long restatements of the problem), no reference to a prompt, plan, hint or instructions, no leftover or duplicated code, no unnecessary length (repeated steps, long defensive input handling the format rules out, needless helpers or passes), then the shorter one. Between two versions that are otherwise equal, prefer the one with a few short comments over one with none.

Then say whether your pick is acceptable as it stands: not acceptable if it still has a placeholder or stub, thinking aloud, a reference to instructions, or is clearly rambling or bloated.

Answer with JSON only: {{"best": <the number of your pick>, "acceptable": true or false, "reason": "one short sentence"}}"""


def prompt_best(r):
    return "PROBLEM (as given to the model):\n" + r["ctx"].rstrip() + "".join(f"\n\n===== SOLUTION {i + 1} =====\n```python\n" + c.rstrip() for i, c in enumerate(r["cands"]))


LOCATE = """You review training data for a code model. You are given a programming problem and a Python solution with numbered lines. A reviewer found this in the solution's text: {notes}

Find where the trouble begins: the first line that should not be imitated (the first comment that thinks aloud, doubts, corrects itself or refers to instructions; the first line of leftover, unused or abandoned code; the first line of a part that is unfinished). The lines before it must be fine to keep as they are; a solution will be continued from just before that line. If the trouble is spread through the whole solution from its first lines, answer with the first line of the function body or of the main code.

Answer with JSON only: {{"first_bad_line": <line number>, "reason": "one short sentence"}}"""


def prompt_locate(r):
    code = r["reply"].rstrip()[:-3].rstrip("\n").split("\n")
    return "PROBLEM (as given to the model):\n" + r["ctx"].rstrip() + "\n\nSOLUTION (numbered lines):\n" + "\n".join(f"{i + 1:3d}| {l}" for i, l in enumerate(code))


def prompt(r):
    return "PROBLEM (as given to the model):\n" + r["ctx"].rstrip() + "\n\nSOLUTION (passes all tests):\n```python\n" + r["reply"].rstrip()


def run(a):
    k = api_key()
    rows = [json.loads(l) for l in open(a.pool)][: a.limit or None]
    done = {(v["rid"], v["vote"]) for v in map(json.loads, open(a.out)) if v.get("verdict")} if os.path.exists(a.out) else set()   # an error is retried on the next run
    todo = [(r, v) for r in rows for v in range(a.votes) if (r["rid"], v) not in done]
    lock = threading.Lock(); spent = [0.0]; out = open(a.out, "a"); stop = []

    def one(job):
        r, vote = job
        with lock:
            if spent[0] >= a.cap:
                return "cap"
            if stop:                                                    # review 209g: an authorisation or payment error ends the run at once
                return "stopped: " + stop[0]
        loc = getattr(a, "locate", False)                               # 209g: locate = the first bad line of a flagged row (the cut for its author's continuation)
        drc = getattr(a, "direct", False)                               # 209i: direct = is the answer written directly or does it ramble (and where the rambling starts)
        bst = getattr(a, "best", False)                                 # 209i: best = pick the best of a row's passing versions (rows hold "cands"), and say whether it is acceptable
        con = getattr(a, "concise", False)                              # 209j: concise = is the code unnecessarily long for the problem
        msgs = [{"role": "system", "content": LOCATE.format(notes=r.get("notes", ""))}, {"role": "user", "content": prompt_locate(r)}] if loc else [{"role": "system", "content": BEST.format()}, {"role": "user", "content": prompt_best(r)}] if bst else [{"role": "system", "content": DIRECT.format()}, {"role": "user", "content": prompt_locate(r)}] if drc else [{"role": "system", "content": CONCISE.format()}, {"role": "user", "content": prompt(r)}] if con else [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(r)}]
        body = {"model": a.model, "messages": msgs, "max_tokens": 6000, "response_format": {"type": "json_object"},
                "reasoning": {"effort": a.effort}, "usage": {"include": True}, "provider": {"allow_fallbacks": True, "data_collection": "deny"}}
        err = ""
        for t in range(3):
            try:
                rq = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
                x = json.load(urllib.request.urlopen(rq, timeout=300)); txt = x["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", txt, re.S); v = json.loads(m.group(0)) if m else {}; u = x.get("usage") or {}
                verdict = v.get("verdict") if v.get("verdict") in ("clean", "not_clean") else None
                if loc:
                    nl = len(r["reply"].rstrip()[:-3].rstrip("\n").split("\n")); fb = v.get("first_bad_line")
                    verdict = "located" if isinstance(fb, int) and 1 <= fb <= nl else None
                if bst:
                    verdict = "picked" if isinstance(v.get("best"), int) and 1 <= v["best"] <= len(r["cands"]) and isinstance(v.get("acceptable"), bool) else None
                if drc and verdict == "not_clean":                       # a rambling verdict needs the line where the rambling starts
                    nl = len(r["reply"].rstrip()[:-3].rstrip("\n").split("\n")); fb = v.get("first_bad_line")
                    verdict = verdict if isinstance(fb, int) and 1 <= fb <= nl else None
                rec = {"rid": r["rid"], "vote": vote, "verdict": verdict, **({"first_bad_line": v.get("first_bad_line")} if (loc or drc) else {}), **({"best": v.get("best"), "acceptable": v.get("acceptable")} if bst else {}), "issues": [i for i in (v.get("issues") or []) if i in ISSUES + ("rambling", "overlong")], "reason": str(v.get("reason") or "")[:300],
                       **({} if verdict else {"raw": txt[:400], "finish": x["choices"][0].get("finish_reason")}), "model": x.get("model"), "provider": x.get("provider"), "cost": u.get("cost"),
                       "tokens": [u.get("prompt_tokens"), u.get("completion_tokens")]}
                with lock:
                    spent[0] += float(u.get("cost") or 0)
                if not verdict and t < 2:
                    body["reasoning"] = {"effort": "low"}; continue
                with lock:
                    out.write(json.dumps(rec) + "\n"); out.flush()
                return verdict or "no verdict"
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:200].replace(k, "<key>")
                if e.code in (401, 402, 403):
                    with lock:
                        stop.append(f"HTTP {e.code}")
                    break
                time.sleep(5 * (t + 1))
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]; time.sleep(5 * (t + 1))
        with lock:
            out.write(json.dumps({"rid": r["rid"], "vote": vote, "error": err}) + "\n"); out.flush()
        return "error"
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = collections.Counter(ex.map(one, todo))
    print(json.dumps({"rows": len(rows), "calls": dict(res), "already done": len(done), "spent $": round(spent[0], 4)}))


def votes_of(path):
    V = collections.defaultdict(dict)
    for v in map(json.loads, open(path)):
        if v.get("verdict"):
            V[v["rid"]][v["vote"]] = v
    return V


def report(a):
    rows = [json.loads(l) for l in open(a.pool)]; V = votes_of(a.verdicts); n = collections.Counter(); iss = collections.Counter(); byau = collections.defaultdict(collections.Counter); soft = collections.Counter(); cal = collections.Counter()
    for r in rows:
        vs = V.get(r["rid"], {})
        if len(vs) < a.votes:
            n["votes missing"] += 1; continue
        k = "clean (every vote)" if all(v["verdict"] == "clean" for v in vs.values()) else ("not clean (every vote)" if all(v["verdict"] == "not_clean" for v in vs.values()) else "split")
        if r.get("expect"):                                             # review 209g: calibration rows are counted only in their own table
            cal[(r["expect"], k)] += 1; continue
        n[k] += 1; byau[r.get("author")][k] += 1; soft[(r.get("soft") or "passes the word lists", k)] += 1
        for v in vs.values():
            for i in v["issues"]:
                iss[i] += 1
    cost = sum(float(v.get("cost") or 0) for v in map(json.loads, open(a.verdicts)))
    print(json.dumps({"rows": sum(1 for r in rows if not r.get("expect")), "verdicts": dict(n), "by author": {k: dict(v) for k, v in byau.items()}, "issues named (over votes)": dict(iss),
                      "word-list rule against the judge": {f"{s} / {k}": c for (s, k), c in sorted(soft.items())}, "calibration (expected / judged)": {f"{e} / {k}": c for (e, k), c in sorted(cal.items())}, "cost $": round(cost, 4)}, indent=1))
    for k_ in ("not_clean",):
        ex = [(r, V[r["rid"]]) for r in rows if not r.get("expect") and len(V.get(r["rid"], {})) >= a.votes and all(v["verdict"] == k_ for v in V[r["rid"]].values())][:8]
        for r, vs in ex:
            print(f"  not clean: {r['rid']} | {sorted({i for v in vs.values() for i in v['issues']})} | {list(vs.values())[0]['reason'][:160]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("pool"); r.add_argument("--out", required=True); r.add_argument("--votes", type=int, default=2); r.add_argument("--cap", type=float, default=3.0)
    r.add_argument("--workers", type=int, default=12); r.add_argument("--limit", type=int); r.add_argument("--model", default=MODEL); r.add_argument("--effort", default="low")
    r.add_argument("--best", action="store_true", help="209i: rows hold 'cands' (passing versions of one row): pick the best and say whether it is acceptable")
    r.add_argument("--concise", action="store_true", help="209j: judge whether the code is unnecessarily long (verdict clean = reasonable length)")
    r.add_argument("--direct", action="store_true", help="209i: judge direct against rambling (verdict clean = direct), with the first rambling line")
    r.add_argument("--locate", action="store_true", help="POOL rows are flagged rows with 'notes' (fix_items.jsonl): ask for the first bad line instead of a verdict")
    p = sp.add_parser("report"); p.add_argument("pool"); p.add_argument("verdicts"); p.add_argument("--votes", type=int, default=2)
    a = ap.parse_args(); {"run": run, "report": report}[a.cmd](a)
