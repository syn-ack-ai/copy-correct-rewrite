"""Addendum 156 loop gate on live rollouts (box rollouts/NAME): replies cut at the token cap (finish_reason "length"), episodes with more than
one submit_patch call, episodes ending in an error (time, call or turn limit), resolved. Episodes are grouped by their first user message.
  python3 live_loops.py ROLLOUTS_DIR NAME [NAME ...]"""
import glob, hashlib, json, os, sys
d = sys.argv[1]
for n in sys.argv[2:]:
    calls = cap = 0; subs = {}
    for f in glob.glob(f"{d}/{n}/calls.jsonl"):
        for line in open(f):
            try:
                j = json.loads(line)
            except Exception:
                continue
            msgs = j["request"]["messages"]; u = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ep = hashlib.md5(json.dumps(u).encode()).hexdigest(); ch = (j.get("response") or {}).get("choices") or [{}]
            calls += 1; cap += ch[0].get("finish_reason") == "length"
            tc = (ch[0].get("message") or {}).get("tool_calls") or []
            subs[ep] = subs.get(ep, 0) + sum(t["function"]["name"] == "submit_patch" for t in tc)
    res = [json.loads(l) for f in glob.glob(f"{d}/{n}/res_*/task_results.jsonl") for l in open(f) if l.strip()]
    err = sum(bool(r.get("error")) for r in res); rv = sum(bool(r.get("resolved")) for r in res)
    print(f"{n}: resolved {rv}/{len(res)} | replies at the cap {cap}/{calls} ({cap / max(1, calls):.1%}) | episodes submitting 2+ times {sum(v >= 2 for v in subs.values())}/{len(subs)} | episodes ending in an error {err}/{len(res)}")
