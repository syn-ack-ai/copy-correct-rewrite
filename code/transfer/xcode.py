"""Addendum 172 (user 11-07: "can you try our best recipe with the graft and lora 64, on the qwen and e4b pair and see if qwen can do tougher
coding tasks?"; "Did it solve any that it could only solve with the larger model bridge up?"; chose a LiveCodeBench subset). Function-level
Python problems with hidden tests (LiveCodeBench code_generation_lite, files test5 + test6: problems released Sep 2024 - Apr 2025, easy and
medium), one greedy attempt per problem, the same plain prompt for every model (the bridge needs one shared text; no chat template).
Arms: qwen (Qwen3-1.7B alone), e4b (Gemma 4 E4B alone), graft_b / graft_c (Qwen + the outputs-only graft toward the bridged / control-bridged
Qwen, seed 2), lora_b / lora_c (Qwen + the rank-64 outputs-only LoRAs, seed 2), plain distillation on E4B's text, seed 2: lora_seq (step 2,
300 steps), lora_sd (seq_d1000) and lora_dd (seqdense_d1000) (172a: the dose-matched baselines), and, on the gap set (E4B passes, Qwen alone
fails), bridge / control (Qwen with the live translator fed E4B's states / Qwen's own, re-tokenized and re-aligned at every generated token,
no cache, as in training) and alone_live (Qwen alone on the same token-by-token path: the matched baseline for the live arms).
Review 11-07 (a549): functional problems run the solution as a module with empty stdin and captured stdout, only the returned value is
compared; exact integer comparison; LiveCodeBench's own import string; every test; the gap set fixed once and reused; a scorer self-check.
  python xcode.py fast | live | evaluate      (env: XB_TEACHER XB_STUDENT XB_DIR XC_DATA [XB_SMALL XC_N XC_MAXNEW])"""
import base64, hashlib, io, json, math, os, pickle, subprocess, sys, tempfile, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xbridge as XB  # noqa: E402

DEV = XB.DEV; OD = XB.OD; SMALL = XB.SMALL; DATA = Path(os.environ.get("XC_DATA", "."))
NMAX = 4 if SMALL else int(os.environ.get("XC_N", "1000")); MAXNEW = 384 if SMALL else int(os.environ.get("XC_MAXNEW", "768")); BATCH = 16
PROMPT = "Solve the following programming problem in Python 3.\n\n{content}\n\n{io}\n\nWrite the complete solution in one Python code block.\n\n```python\n"
IO_STDIN = "Read the input from standard input and write the answer to standard output."
IO_FUNC = "Your solution must complete this code (define class Solution):\n{starter}"
HEADER = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\n"
          "from copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
          "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\n"
          "import re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
          "import itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n")   # LiveCodeBench's import_string
ADAPTERS = (("lora_b", "xlora_bridge_w0"), ("lora_c", "xlora_control_w0"), ("lora_seq", "xlora_seq"), ("lora_sd", "xlora_seq_d1000"), ("lora_dd", "xlora_seqdense_d1000"), ("lora_self", "xlora_seqself_d1000"), ("lora_c_long", "xlora_control_w0_long"), ("lora_b_long", "xlora_bridge_w0_long"),
            ("lora_c_L1200", "xlora_control_w0_L1200"), ("lora_c_L2400", "xlora_control_w0_L2400"), ("lora_c_L4800", "xlora_control_w0_L4800"),
            ("lora_b_L1200", "xlora_bridge_w0_L1200"), ("lora_b_L2400", "xlora_bridge_w0_L2400"), ("lora_b_L4800", "xlora_bridge_w0_L4800"))   # 176
ONLY = set(a for a in os.environ.get("XC_ONLY", "").split(",") if a)   # 174q: run only these arms (empty = all)
want = lambda a: not ONLY or a in ONLY  # noqa: E731


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


class _NoClasses(pickle.Unpickler):
    def find_class(self, module, name):                                  # the payload is a pickled str: refuse any class (no code from the data)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def _tests(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(_NoClasses(io.BytesIO(zlib.decompress(base64.b64decode(s.encode("utf-8"))))).load())


def problems():
    out, seen = [], set()
    for f in os.environ.get("XC_FILES", "test5.jsonl,test6.jsonl").split(","):   # 178: XC_FILES picks the LiveCodeBench files (default: 172's two)
        for line in open(DATA / f):
            r = json.loads(line)
            if r["difficulty"] not in ("easy", "medium") or r["question_id"] in seen:
                continue
            seen.add(r["question_id"]); tests = json.loads(r["public_test_cases"]) + _tests(r["private_test_cases"]); meta = json.loads(r["metadata"] or "{}")
            func = meta.get("func_name"); io_ = IO_FUNC.format(starter=r["starter_code"].rstrip()) if func else IO_STDIN
            out.append({"id": r["question_id"], "platform": r["platform"], "difficulty": r["difficulty"], "date": r["contest_date"][:10], "func": func,
                        "prompt": PROMPT.format(content=r["question_content"].strip(), io=io_), "tests": tests})
    out.sort(key=lambda p: (p["date"], p["id"])); return out[:NMAX]


def code_of(text):
    return text.split("```")[0]


def stop_ids(model, tok):
    e = model.generation_config.eos_token_id; ids = set(e if isinstance(e, (list, tuple)) else [e]) | {tok.eos_token_id}
    for t in ("<|endoftext|>", "<|im_end|>", "<end_of_turn>", "<eos>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def gen_hf(model, tok, prompts, bos=""):
    """greedy, batched, left-padded; a row stops at its closing ``` or an end token"""
    tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; outs = []; eos = stop_ids(model, tok)
    for a in range(0, len(prompts), BATCH):
        enc = tok([bos + p for p in prompts[a:a + BATCH]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        y = model.generate(**enc, do_sample=False, max_new_tokens=MAXNEW, pad_token_id=pad, eos_token_id=eos, use_cache=True, stop_strings=["```"], tokenizer=tok)
        outs += [tok.decode(t[enc["input_ids"].shape[1]:], skip_special_tokens=True) for t in y]
        log("generated", a + len(prompts[a:a + BATCH]), "of", len(prompts))
    return outs


def align(B, prompt, text):
    """as xbridge.encode, with the reply = what was generated: Gemma ids (with BOS), Qwen ids, for each Qwen position the last Gemma token
    ending at or before it, the written span's start (never Qwen's first token nor positions aligned only to E4B's BOS)"""
    eg = B.tg("<bos>" + text, return_offsets_mapping=True, add_special_tokens=False); gids = eg["input_ids"]
    gend = np.maximum.accumulate(np.array([e for _, e in eg["offset_mapping"]]) - len("<bos>"))
    eq = B.tq(XB.SBOS + text, return_offsets_mapping=True, add_special_tokens=False); qids = eq["input_ids"]   # 186: the student's BOS (none for Qwen)
    qs = np.array([s_ for s_, _ in eq["offset_mapping"]]) - len(XB.SBOS); qe = np.array([e for _, e in eq["offset_mapping"]]) - len(XB.SBOS)
    j = np.clip(np.searchsorted(gend, qe, side="right") - 1, 0, len(gids) - 1); rs = np.nonzero(qs >= len(prompt))[0]
    rstart = int(rs[0]) if len(rs) else len(qids); s0 = max(1, int(np.argmax(j > 0)), rstart - XB.SPAN)
    return gids, qids, j, s0


@torch.no_grad()
def gen_bridged(B, T, src, prompt, eos):
    """greedy, one token at a time; every step re-tokenizes the text, runs E4B (src 'bridge') or Qwen alone (src 'control') for the source
    states and writes the translator over the span, as in training; src 'alone' = the same path with no translator"""
    gen = []
    for _ in range(MAXNEW):
        text = prompt + B.tq.decode(gen, skip_special_tokens=True); gids, qids, j, s0 = align(B, prompt, text); n = len(qids); x = torch.tensor([qids], device=DEV)
        if src == "bridge":
            B.gT.on = True; B.mT(input_ids=torch.tensor([gids], device=DEV), logits_to_keep=1, use_cache=False); oT = B.gT.take()
            S = torch.cat([oT[l][torch.as_tensor(j, device=DEV)] for l in B.TL], -1); del oT
        elif src == "control":
            B.gS.on = True; B.mS(input_ids=x, logits_to_keep=1, use_cache=False); oS = B.gS.take(); S = torch.cat([oS[l] for l in B.SL], -1); del oS
        if src != "alone":
            B.write(T, S, s0, n)
        try:
            t = int(B.mS(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1].argmax())
        finally:
            B.P.slot.clear()
        if t in eos:
            break
        gen.append(t)
        if "```" in B.tq.decode(gen[-6:], skip_special_tokens=True):
            break
    return B.tq.decode(gen, skip_special_tokens=True)


def save(arm, probs, outs):
    with open(OD / f"gens_{arm}.jsonl", "w") as fh:
        for p, o in zip(probs, outs):
            fh.write(json.dumps({"id": p["id"], "text": o}) + "\n")
    if outs:
        log(arm, "first generation starts:", json.dumps(outs[0][:160]), "| without a closing fence:", sum(1 for o in outs if "```" not in o), "of", len(outs))


def fast():
    from peft import PeftModel
    probs = problems(); prompts = [p["prompt"] for p in probs]; log("problems", len(probs), "| max new tokens", MAXNEW)
    B = XB.Bridge("bridge"); t0 = time.time(); gl = [B.WL] + B.sites
    if want("e4b"):
        save("e4b", probs, gen_hf(B.mT, B.tg, prompts, bos="<bos>")); log("e4b done", round(time.time() - t0), "s")
    save("qwen", probs, gen_hf(B.mS, B.tq, prompts)); log("qwen done", round(time.time() - t0), "s")
    for nm, f in (("graft_b", "xgraft_bridge_w0"), ("graft_c", "xgraft_control_w0")):
        if not want(nm):
            continue
        ck = torch.load(OD / f"{f}.pt", weights_only=False, map_location=DEV); assert ck["layers"] == gl, (ck["layers"], gl)
        G = XB.QGraft(B.ds, len(gl), ck["width"]).to(DEV); G.load_state_dict(ck["state"]); H = XB.GraftHooks(B.mS, gl, G.eval()); H.on = True
        try:
            save(nm, probs, gen_hf(B.mS, B.tq, prompts))
        finally:
            H.on = False
            for h in H.hs:
                h.remove()
        log(nm, "done", round(time.time() - t0), "s")
    pm = None
    for nm, f in ADAPTERS:
        d = OD / f
        if not want(nm):
            continue
        if not d.is_dir():
            log("missing", d, "- arm", nm, "skipped"); continue
        if pm is None:
            pm = PeftModel.from_pretrained(B.mS, str(d), adapter_name=nm)
        else:
            pm.load_adapter(str(d), adapter_name=nm)
        pm.set_adapter(nm); pm.eval(); save(nm, probs, gen_hf(pm, B.tq, prompts)); log(nm, "done", round(time.time() - t0), "s")
    print("FAST_DONE", flush=True)


def live():
    probs = problems(); byid = {p["id"]: p for p in probs}
    ids = [p["id"] for p in probs[:2]] if SMALL else json.load(open(OD / "gap_ids.json")); ps = [byid[i] for i in ids if i in byid]
    B = XB.Bridge("bridge"); T = {"bridge": XB.load_tr(B, "bridge")[0], "control": XB.load_tr(B, "control")[0], "alone": None}; eos = set(stop_ids(B.mS, B.tq)); t0 = time.time()
    log("live arms on the gap set:", len(ps), "problems")
    for arm in ("bridge", "control", "alone"):
        outs = []
        for k, p in enumerate(ps):
            outs.append(gen_bridged(B, T[arm], arm, p["prompt"], eos)); save(arm if arm != "alone" else "alone_live", ps[:len(outs)], outs)   # partial saves
            log(arm, k + 1, "of", len(ps), round(time.time() - t0), "s")
    print("LIVE_DONE", flush=True)


# the scorer. Each test runs in a fresh interpreter (-s, PYTHONHASHSEED=0), with CPU 8 s, memory 2 GB, output file ≤ 64 MB, wall 12 s, a
# temporary directory as cwd and a bare environment (no tokens or keys in it; files on the VM stay readable and the network stays open: a
# disposable Colab VM is the sandbox). Stdin problems run as __main__. Functional problems: the arguments are read first, the solution runs
# as a module named "solution" with empty stdin and its prints captured, and only the returned value is printed (as JSON).
BOOT = """import resource, runpy, sys, io, json, contextlib
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30)); resource.setrlimit(resource.RLIMIT_CPU, (8, 8)); resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
try:
    resource.setrlimit(resource.RLIMIT_STACK, (512 << 20, resource.getrlimit(resource.RLIMIT_STACK)[1]))
except Exception:
    pass
FUNC = {func!r}
sys.argv = ["sol.py"]
if FUNC is None:
    runpy.run_path("sol.py", run_name="__main__")
else:
    _args = [json.loads(l) for l in sys.stdin.read().splitlines() if l.strip()]; sys.stdin = io.StringIO(""); _out = sys.stdout
    with contextlib.redirect_stdout(io.StringIO()):
        _ns = runpy.run_path("sol.py", run_name="solution"); _res = getattr(_ns["Solution"](), FUNC)(*_args)
    _out.write(json.dumps(_res, default=lambda o: o.item() if hasattr(o, "item") else (list(o) if hasattr(o, "__iter__") else str(o))) + "\\n")
"""
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp", "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

SANDBOX = os.environ.get("XC_SANDBOX", "")   # 181 (user 11-07: the RSI pilot on the 4090): "bwrap" runs every test in a bubblewrap sandbox: no network, a read-only root, home hidden but the interpreter


def _cmd(td):
    """the test command; with XC_SANDBOX=bwrap the model's code cannot reach the network or write outside its temporary folder"""
    if SANDBOX != "bwrap":
        return [sys.executable, "-s", "boot.py"]
    home = os.path.expanduser("~"); keep = sorted({os.path.realpath(sys.prefix), os.path.dirname(os.path.realpath(sys.base_prefix))})   # the parent: uv links versions inside it
    c = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", home]
    for k in keep:
        c += ["--ro-bind", k, k]
    if os.path.realpath(sys.prefix) != sys.prefix:
        c += ["--ro-bind", sys.prefix, sys.prefix]
    return c + ["--bind", td, td, "--chdir", td, "--unshare-all", "--die-with-parent", "--new-session", sys.executable, "-s", "boot.py"]
FLOATY = set(".eE")


def _same(out, exp, func):
    if func:
        try:
            a, b = json.loads(out), json.loads(exp)
        except Exception:
            return out.strip() == exp.strip()
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
        if num(a) and num(b) and (isinstance(a, float) or isinstance(b, float)):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return a == b
    ta, tb = out.split(), exp.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if not (FLOATY & set(x + y)):                                     # review 11-07: integers and words compare exactly
            return False
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True


def run_problem(p, text):
    """True when every test passes (public first)"""
    code = code_of(text)
    if not code.strip():
        return False
    try:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "sol.py").write_text(HEADER + code); (Path(td) / "boot.py").write_text(BOOT.format(func=p["func"])); of = Path(td) / "out.txt"
            for t in p["tests"]:
                with open(of, "wb") as fo:
                    try:
                        r = subprocess.run(_cmd(td), input=t["input"].encode("utf-8"), stdout=fo, stderr=subprocess.DEVNULL, timeout=12, cwd=td, env=ENV)
                    except subprocess.TimeoutExpired:
                        return False
                if r.returncode != 0 or not _same(of.read_bytes()[:64 << 20].decode("utf-8", "replace"), t["output"], p["func"]):
                    return False
        return True
    except Exception:
        return False


def selfcheck():
    """the scorer must pass known-right answers and fail known-wrong ones before any arm is scored"""
    f = {"func": "add", "tests": [{"input": "1\n2", "output": "3"}, {"input": "[1,2]\n[3]", "output": "[1, 2, 3]"}]}
    s = {"func": None, "tests": [{"input": "3 4\n", "output": "7\n"}, {"input": "998244352 1\n", "output": "998244353"}]}
    ok_f = "class Solution:\n    def add(self, a, b):\n        print('debug')\n        return a + b\nprint(Solution().add(1, 2))\nif __name__ == '__main__':\n    input()\n"
    ok_s = "a, b = map(int, input().split())\nprint(a + b)\n"
    checks = {"functional right (with prints)": run_problem(f, ok_f), "functional wrong": not run_problem(f, ok_f.replace("a + b", "b + a")),
              "stdin right": run_problem(s, ok_s), "stdin off by one": not run_problem(s, ok_s.replace("a + b", "a + b - 1")),
              "math names": run_problem(s, "x = gcd(4, 6) + inf * 0 if False else 0\na, b = map(int, input().split())\nprint(bisect.bisect_left([a + b], 0) + a + b)\n")}
    log("scorer self-check:", checks); assert all(checks.values()), "scorer self-check FAILED"


def mcnemar(a, b, ids):
    """exact two-sided sign test on the discordant problems: (a only, b only, p)"""
    x = sum(1 for i in ids if a.get(i) and not b.get(i)); y = sum(1 for i in ids if b.get(i) and not a.get(i)); n = x + y
    if n == 0:
        return [x, y, 1.0]
    p = sum(math.comb(n, k) for k in range(0, min(x, y) + 1)) / 2 ** n; return [x, y, round(min(1.0, 2 * p), 4)]


def evaluate():
    selfcheck(); probs = {p["id"]: p for p in problems()}; res = {}; nth = min(16, os.cpu_count() or 4)
    for f in sorted(OD.glob("gens_*.jsonl")):
        arm = f.stem[len("gens_"):]; h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]; cf = OD / f"scored_{arm}.json"
        if cf.exists() and json.load(open(cf))["hash"] == h:                # review 11-07: an arm is scored once; reruns reuse it
            res[arm] = json.load(open(cf))["pass"]; log(arm, "passed", sum(res[arm].values()), "of", len(res[arm]), "(cached)"); continue
        rows = [json.loads(l) for l in open(f)]
        with ThreadPoolExecutor(nth) as ex:
            ok = list(ex.map(lambda r: run_problem(probs[r["id"]], r["text"]), rows))
        res[arm] = {r["id"]: bool(o) for r, o in zip(rows, ok)}; json.dump({"hash": h, "pass": res[arm]}, open(cf, "w")); log(arm, "passed", sum(ok), "of", len(ok))
    ids = list(probs); out = {"problems": len(probs), "passed": {a: [sum(v.values()), len(v)] for a, v in res.items()}, "by platform and difficulty": {}}
    for a in res:
        by = {}
        for i, o in res[a].items():
            k = f"{probs[i]['platform']} {probs[i]['difficulty']}"; by.setdefault(k, [0, 0]); by[k][0] += o; by[k][1] += 1
        out["by platform and difficulty"][a] = by
    pairs = [("graft_b", "qwen"), ("lora_b", "qwen"), ("graft_c", "qwen"), ("lora_c", "qwen"), ("lora_seq", "qwen"), ("lora_sd", "qwen"), ("lora_dd", "qwen"),
             ("graft_b", "graft_c"), ("lora_b", "lora_c"), ("lora_b", "lora_sd"), ("lora_b", "lora_dd"), ("graft_b", "lora_sd"), ("e4b", "qwen"),
             ("lora_self", "qwen"), ("lora_c", "lora_self"), ("lora_self", "lora_sd"), ("lora_c", "lora_sd"),
             ("lora_c_long", "lora_c"), ("lora_b_long", "lora_b"), ("lora_c_long", "lora_b_long"), ("lora_c_long", "qwen"), ("lora_b_long", "qwen"),
             ("lora_c_L1200", "lora_c"), ("lora_c_L2400", "lora_c"), ("lora_c_L4800", "lora_c"), ("lora_c_L4800", "lora_c_L2400"), ("lora_c_L2400", "lora_c_L1200"),
             ("lora_c_L1200", "qwen"), ("lora_c_L2400", "qwen"), ("lora_c_L4800", "qwen"),
             ("lora_b_L1200", "lora_b"), ("lora_b_L2400", "lora_b"), ("lora_b_L4800", "lora_b"), ("lora_b_L4800", "lora_b_L2400"), ("lora_b_L2400", "lora_b_L1200"),
             ("lora_b_L1200", "qwen"), ("lora_b_L2400", "qwen"), ("lora_b_L4800", "qwen"),
             ("lora_b_L1200", "lora_c_L1200"), ("lora_b_L2400", "lora_c_L2400"), ("lora_b_L4800", "lora_c_L4800")]
    out["paired over all problems (a only, b only, exact p)"] = {f"{a} vs {b}": mcnemar(res[a], res[b], ids) for a, b in pairs if a in res and b in res}
    if "e4b" in res and "qwen" in res:
        now = [i for i in ids if res["e4b"].get(i) and not res["qwen"].get(i)]
        if not (OD / "gap_ids.json").exists():
            json.dump(now, open(OD / "gap_ids.json", "w"))                 # fixed once, from the first evaluation; the live arms ran on it
        gap = json.load(open(OD / "gap_ids.json"))
        out["gap set (E4B passes, Qwen alone fails; fixed at the first evaluation)"] = len(gap); out["gap set recomputed now"] = len(now)
        out["Qwen passes, E4B fails"] = sum(1 for i in ids if res["qwen"].get(i) and not res["e4b"].get(i))
        out["solved on the gap set"] = {a: [sum(1 for i in gap if res[a].get(i)), sum(1 for i in gap if i in res[a])] for a in res if a != "e4b"}
        out["lost where Qwen alone passes"] = {a: sum(1 for i in ids if res["qwen"].get(i) and i in res[a] and not res[a][i]) for a in res if a not in ("e4b", "qwen", "bridge", "control", "alone_live")}
        if "bridge" in res:
            out["gap set, paired (a only, b only, exact p)"] = {f"{a} vs {b}": mcnemar(res[a], res[b], gap) for a, b in (("bridge", "control"), ("bridge", "alone_live"), ("control", "alone_live")) if a in res and b in res}
            only = [i for i in gap if res["bridge"].get(i) and not res.get("control", {}).get(i)]
            out["gap problems the live bridge solves and the live control does not"] = only
            out["of those, solved with the bridge down"] = {a: [i for i in only if res[a].get(i)] for a in ("graft_b", "graft_c", "lora_b", "lora_c", "lora_seq", "lora_sd", "lora_dd") if a in res}
        out["per gap problem"] = {i: {a: res[a].get(i) for a in res} for i in gap}
    json.dump(out, open(OD / "xcode_results.json", "w"), indent=1); log(json.dumps({k: v for k, v in out.items() if k != "per gap problem"})); print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    {"fast": fast, "live": live, "evaluate": evaluate}[sys.argv[1]]()
