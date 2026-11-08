"""Addendum 174 (user 11-07: "we should test on 3 things really... how many problems it could not solve before, but now can, how many new
problems it solves and not losing any current knowledge"; training pool: older LiveCodeBench). Does the bridge add anything on top of plain
SFT on the teacher's text? Qwen3-1.7B learns from Gemma 4 E4B's own verified solutions to older LiveCodeBench problems (test.jsonl-test4.jsonl,
May 2023 - Sep 2024, every difficulty), and is tested on those (fixed) and on newer problems it never trained on (test5 + test6, easy and
medium, Sep 2024 - Apr 2025: new), and on what it already solved (kept), plus off-task knowledge (held agent replies, fresh questions,
untaught prose). Rank-64 LoRA on the MLPs of layers 9-27, the same rows, order and dose in every arm of a seed:
  sft    cross-entropy on E4B's solution
  sft_b  + LAMBDA x KL(bridged Qwen || student) over the written span (the live translator fed E4B's states)
  sft_c  + LAMBDA x KL(control-bridged Qwen || student) (the same translator design fed Qwen's own states)
  kl_b   the bridge KL alone (the dense recipe of 169, on these rows)
  sft_self  + LAMBDA x KL(original Qwen || student) over the span: no translator (174b, run instead of sft_b and kl_b when the gate fails;
            user 11-07: "do we still need the translator for the control?")
  sdft   the in-context self-teacher (174c; Shenfeld et al. 2026, arXiv 2601.19897): the student samples its own answer (T 1.0, at most SD_MAXNEW
         tokens); the teacher, an EMA of the student's LoRA (alpha SD_EMA), reads the same problem with E4B's verified solution in context ("This
         is an example for a response to the question: ... Now answer with a response of your own."); per-token KL(teacher || student) over the
         vocabulary on the sampled tokens, the first SD_MASK masked; no cross-entropy on E4B's text
  opd_b  on-policy through the bridge (179; user 11-07: "is there anyway we could make offpolicy training on policy using the translators?"): the
         student samples as in sdft; E4B reads the same text (prefill only), the bridge translator writes its states into the base student (the
         LoRA off), and that bridged distribution on every sampled token is the target: per-token KL(target || student), the first SD_MASK masked
  opd_c  the same with the control translator (the student's own states): an on-policy self-bridge, E4B's text only ever used to fit it
  mix_c  179b (user 11-07: "yes, add the hybrid arm to 179"; text + translator to make it on-policy): every row gives both the cross-entropy
         on E4B's verified solution (the text, with its gradient) and opd_c's on-policy term on the student's own sample (weight 1 each)
  opd_cf 181 (user 11-07: "I thought we were going to use the qwen + translator + e4b text (off policy) and pass it through the translator to make
         it onpolicy"): opd_c, but the control translator is first refit on these rows (fit_cf: the seed's control translator, fitted on agent
         rows, fine-tuned XL_CF_STEPS clean steps at XL_CF_LR on E4B's verified solutions), so E4B's contest solutions reach the teacher
  rw_cf  181 ("yes, build opd_cf and the rewrite arm too"): cross-entropy (sft) on the rewrite rows: the refit control translator live, Qwen
         writes each row problem's solution itself (greedy + XL_NRW samples, a second round of XL_NRW samples where none passed; KV-cached,
         gen_cf), and a row is the greedy rewrite if it passes every hidden test, else the first passing sample; problems with no passing
         rewrite have no row; steps matched to sft's (the rows cycle)
  mix_cf 181 (user 11-07: "yes, add mix_cf to 181"): mix_c with the refit translator: E4B's text copied, plus the on-policy term toward the
         refit-corrected base on the student's own samples
  rb     182 (user 11-08: "yes, build the runbook arm"; after Terminal-RSI, arXiv 2610.22192, recursive self-rewrite): cross-entropy (sft) on the
         runbook rows (runbook, built once for every seed): E4B reads each row problem with its own verified solution and writes a plan in plain
         English (no code; a leakage screen); Qwen alone solves the problem with the plan in its prompt (greedy, then XL_NRB samples where it failed, then
         XL_NRB more where none passed); a row is the greedy solution if it passes every hidden test, else the first passing sample, and it is
         trained WITHOUT the plan (the original prompt): Qwen's own text, E4B's know-how. Steps matched to sft's (the rows cycle)
  sft_k  184 (user 11-08: "can we also try creating a translator that was trained on the coding problems this agent is likely to lose"; "que
         the translator training on the 4090"): sft_s, but on the row problems Qwen alone already solves the row is Qwen's own passing greedy
         solution (build_keep; row-level do-no-harm: a correct answer is never overwritten with E4B's longer one)
  sft_ck sft_c_s on the keep rows (the old control translator's KL over the span)
  xk     sft_c on the keep rows with the cross-fitted selective translator (fit_xk): two translators, each fitted on one half of the row
         problems (E4B's shortest solutions where Qwen fails: learn; Qwen's own passing solutions where it succeeds: keep, KL to Qwen alone over
         the whole reply + a gate penalty; the fragile set oversampled), and each row's KL target comes from the one that never saw it
  sft_es / sft_lr  188 (user 11-08: "queue the matched-loss test"): sft that stops once the running training CE (50 steps) reaches XL_STOP_CE
         (default 0.24, Qwen's final level on E4B's text), and sft at a tenth of the learning rate for the full run: is the Gemma student's damage
         to untaught text the over-fitting of familiar text?
  sft_mt_s / sft_self_mt_s  190 (user 11-08: "run the k on prose also and see what happens... in that case both prose and agentic work/solved
         should be similar to the correctors"): sft_s / sft_self_s plus, each step, MT_N rows of the corrector's own fitting text (split train,
         kinds and weights XB.W: agent replies, prose, code, questions, stories) with cross-entropy on the reply: the multi-task baseline for the
         corrector anchor (does the anchor only carry that text?)
  sft_b_s  187 (user 11-08: two-way E4B <-> E2B, "including the 600 trained correctors for both"): sft_b on the shortest rows (the cross-model
         corrector's KL over the span), the same rows and dose as sft_s / sft_c_s / sft_self_s
  sft_self_s  185 (user 11-08: "yes, run the SFT + KL-to-original baseline on colab"): sft_self on the shortest rows: the same rows and dose as
         sft_s / sft_c_s, with 1.0 x KL(original Qwen || student) over the written span instead of the corrector's target (no translator)
  rb_c   184b (user 11-08: "yes, add rb_c to 184"): sft_c on the runbook rows (rb + the old control translator's KL over the span): does the
         self-bridge term undo rb's drift on untaught text (Qwen's own short rows sharpen it) and keep rb's coding gains?
Rows are model-written only: E4B's greedy solution when it passes every hidden test, else (easy and medium) the first of XL_NSAMP samples
(T 0.8) that does.
  python xlearn.py build | train ARM | evaluate | fit_cf | rewrite | runbook | build_keep | fit_xk   (env: XB_TEACHER XB_STUDENT XB_DIR XB_SEED XC_DATA [XB_SMALL XL_EPOCHS XL_LAMBDA])"""
import gc, hashlib, json, math, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xbridge as XB  # noqa: E402
import xcode as XC  # noqa: E402

DEV, OD, SMALL, SEED = XB.DEV, XB.OD, XB.SMALL, XB.SEED
TRAIN_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"); HELD_FILES = ("test5.jsonl", "test6.jsonl")
NTRAIN, NHELD = (6, 4) if SMALL else (10 ** 6, 10 ** 6); NSAMP = 2 if SMALL else int(os.environ.get("XL_NSAMP", "4")); TEMP = 0.8
EPOCHS = float(os.environ.get("XL_EPOCHS", "2")); LAM = float(os.environ.get("XL_LAMBDA", "1.0")); RANK = 64; LR = 3e-4
SD_EMA, SD_MASK, SD_MAXNEW = 0.02, 3, 32 if SMALL else int(os.environ.get("XL_SD_MAXNEW", "512"))
ARMS = ("sft", "sft_b", "sft_c", "kl_b", "sft_self", "sdft", "opd_b", "opd_c", "sft_s", "sft_c_s", "mix_c", "opd_cf", "rw_cf", "mix_cf", "rb", "sft_k", "sft_ck", "xk", "rb_c", "sft_self_s", "sft_b_s", "sft_es", "sft_lr", "sft_mt_s", "sft_self_mt_s", "sft_self_s_l03", "sft_self_s_l3"); ROWS = OD.parent / "xl_rows.jsonl"; BASE = OD.parent / "xl_base.json"
ROWS_SHORT = OD.parent / "xl_rows_short.jsonl"; SHORT = OD.parent / "xl_short.json"   # 179a: shortest-correct rows (sft_s, sft_c_s = sft, sft_c on them)
NSHORT = 2 if SMALL else int(os.environ.get("XL_NSHORT", "4"))   # shared by every seed's folder
MT_N = 4 if not SMALL else 1; MT_W = float(os.environ.get("XL_MT_W", "1.0"))   # 190a: mix rows per step: 4 x 299 = about 1,200 rows, the corrector's own exposure (600 steps x 2); weight of their mean
CFN = "xb_control_cf"; CF_GATE = OD.parent / "xl_cf_gate.json"; ROWS_RW = OD.parent / "xl_rows_rw.jsonl"; RW = OD.parent / "xl_rw.json"   # 181: per seed folder
CF_STEPS = 20 if SMALL else int(os.environ.get("XL_CF_STEPS", "600")); CF_LR = float(os.environ.get("XL_CF_LR", "1e-4")); CF_MIN = float(os.environ.get("XL_CF_MIN", "0.05"))
NRW = 2 if SMALL else int(os.environ.get("XL_NRW", "4")); RW_ROWS = int(os.environ.get("XL_RW_ROWS", "48"))   # rewrite: samples per round; generated rows per batch
ROWS_RB = OD.parent / "xl_rows_rb.jsonl"; RB = OD.parent / "xl_rb.json"   # 182: runbook rows, built once, shared by every seed's folder
NRB = 2 if SMALL else int(os.environ.get("XL_NRB", "4")); RB_MAXNEW = 64 if SMALL else int(os.environ.get("XL_RB_MAXNEW", "320")); RB_TRY = 2 if SMALL else 3   # samples per round; plan tokens; plan attempts
RB_HEAD = "Solve the following programming problem in Python 3.\n\n"; RB_TAIL = "\n\nWrite the complete solution in one Python code block.\n\n```python\n"
RB_ASK = ("Here is a programming problem and a correct Python 3 solution.\n\n{problem}\n\nSolution:\n```python\n{code}```\n\n"
          "Explain how to solve this problem as a short plan for another programmer: at most 8 numbered steps in plain English, covering the key "
          "idea, the algorithm, the edge cases and the input and output format. Do not write code.\n\nPlan:\n1.")
RB_USE = "\n\nA plan for the solution:\n{plan}\n\nWrite the complete solution in one Python code block, following the plan.\n\n```python\n"
RB_OVER = 0.10   # leakage screen: the most of the solution's code-token 4-grams (comments and strings removed) a plan may contain
RB_PROBE = int(os.environ.get("XL_RB_PROBE", "0"))   # review 11-08: > 0 = plans only, on the first RB_PROBE rows (the yield check before the full build)
RB_PLANS = os.environ.get("XL_RB_PLANS", "")   # 186: reuse the accepted plans of an earlier build (xl_rb_gens.jsonl) instead of asking E4B again
ROWS_K = OD.parent / "xl_rows_k.jsonl"; KJ = OD.parent / "xl_k.json"; FRAG = OD.parent / "xl_fragile.json"   # 184: keep rows; the fragile set (from 179/181)
XKN = "xb_control_xk"; XKJ = OD.parent / "xl_xk.json"; XK_KW, XK_GW, XK_FRW = 1.0, 0.1, 3   # 184: keep-row KL weight, gate penalty, fragile oversampling
XK_MIN = float(os.environ.get("XL_XK_MIN", "0.05"))   # 184: the cross-fit translator's least lead over the old control on unseen learn rows (nats)


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


def load(files, diffs, cap):
    out, seen = [], set()
    for f in files:
        for line in open(XC.DATA / f):
            r = json.loads(line)
            if r["difficulty"] not in diffs or r["question_id"] in seen:
                continue
            seen.add(r["question_id"]); tests = json.loads(r["public_test_cases"]) + XC._tests(r["private_test_cases"]); meta = json.loads(r["metadata"] or "{}")
            func = meta.get("func_name"); io_ = XC.IO_FUNC.format(starter=r["starter_code"].rstrip()) if func else XC.IO_STDIN
            out.append({"id": r["question_id"], "platform": r["platform"], "difficulty": r["difficulty"], "date": r["contest_date"][:10], "func": func,
                        "prompt": XC.PROMPT.format(content=r["question_content"].strip(), io=io_), "tests": tests})
    out.sort(key=lambda p: (p["date"], p["id"])); return out[:cap]


def pools():
    tr = load(TRAIN_FILES, ("easy", "medium", "hard"), NTRAIN); he = load(HELD_FILES, ("easy", "medium"), NHELD)
    assert not {p["id"] for p in tr} & {p["id"] for p in he}, "train and held pools overlap"; return tr, he


def score(probs, texts):
    with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
        return list(ex.map(lambda pt: XC.run_problem(pt[0], pt[1]), zip(probs, texts)))


@torch.no_grad()
def gen_sample(model, tok, prompts, n, bos=""):
    """n samples per prompt at temperature TEMP (top-p 0.95), batched; same stopping as XC.gen_hf"""
    tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; eos = XC.stop_ids(model, tok); outs = []; step = max(1, XC.BATCH // n)
    for a in range(0, len(prompts), step):
        ch = prompts[a:a + step]; enc = tok([bos + p for p in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        y = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=0.95, num_return_sequences=n, max_new_tokens=XC.MAXNEW, pad_token_id=pad, eos_token_id=eos,
                           use_cache=True, stop_strings=["```"], tokenizer=tok)
        dec = [tok.decode(t[enc["input_ids"].shape[1]:], skip_special_tokens=True) for t in y]; outs += [dec[i * n:(i + 1) * n] for i in range(len(ch))]
        log("sampled", a + len(ch), "of", len(prompts))
    return outs


def build():
    """E4B's verified solutions on the train pool (rows), and the baselines: Qwen alone and E4B greedy on both pools. Every generation is kept
    (xl_build_gens.jsonl). Review 11-07: samples scored in parallel; a row needs E4B's own closing fence (nothing appended by hand)"""
    XC.selfcheck(); torch.manual_seed(174); tr, he = pools(); B = XB.Bridge("bridge"); t0 = time.time(); log("pools: train", len(tr), "held", len(he)); G = []
    fenced = lambda t: "```" in t  # noqa: E731
    g = XC.gen_hf(B.mT, B.tg, [p["prompt"] for p in tr], bos="<bos>"); ok = score(tr, g); log("E4B greedy on the train pool:", sum(ok), "of", len(tr), round(time.time() - t0), "s")
    G += [{"model": "e4b", "kind": "greedy", "pool": "train", "id": p["id"], "text": t, "pass": o} for p, t, o in zip(tr, g, ok)]
    sol = {tr[i]["id"]: (g[i], "greedy") for i in range(len(tr)) if ok[i] and fenced(g[i])}
    need = [i for i in range(len(tr)) if tr[i]["id"] not in sol and tr[i]["difficulty"] != "hard"]   # samples for easy and medium only (hard: rarely solved, costly)
    ss = gen_sample(B.mT, B.tg, [tr[i]["prompt"] for i in need], NSAMP, bos="<bos>"); flat = [(i, t) for i, row in zip(need, ss) for t in row]
    so = score([tr[i] for i, _ in flat], [t for _, t in flat])
    for (i, t), o in zip(flat, so):
        G.append({"model": "e4b", "kind": "sample", "pool": "train", "id": tr[i]["id"], "text": t, "pass": o})
        if o and fenced(t) and tr[i]["id"] not in sol:
            sol[tr[i]["id"]] = (t, "sample")
    log("E4B solved (greedy or a sample, fenced):", len(sol), "of", len(tr), "| passing but unfenced, not used:", sum(1 for x in G if x["pass"] and not fenced(x["text"])), round(time.time() - t0), "s")
    with open(ROWS, "w") as fh:                                          # model-written rows: E4B's code and its own closing fence
        for p in tr:
            if p["id"] in sol:
                t = sol[p["id"]][0]; fh.write(json.dumps({"id": p["id"], "ctx": p["prompt"], "reply": t[:t.index("```") + 3], "source": sol[p["id"]][1], "difficulty": p["difficulty"]}) + "\n")
    ge = XC.gen_hf(B.mT, B.tg, [p["prompt"] for p in he], bos="<bos>"); qt = XC.gen_hf(B.mS, B.tq, [p["prompt"] for p in tr]); qh = XC.gen_hf(B.mS, B.tq, [p["prompt"] for p in he])
    oe, oqt, oqh = score(he, ge), score(tr, qt), score(he, qh)
    G += [{"model": "e4b", "kind": "greedy", "pool": "held", "id": p["id"], "text": t, "pass": o} for p, t, o in zip(he, ge, oe)]
    G += [{"model": "qwen", "kind": "greedy", "pool": pl, "id": p["id"], "text": t, "pass": o} for pl, ps, ts, os_ in (("train", tr, qt, oqt), ("held", he, qh, oqh)) for p, t, o in zip(ps, ts, os_)]
    with open(ROWS.parent / "xl_build_gens.jsonl", "w") as fh:
        for x in G:
            fh.write(json.dumps(x) + "\n")
    base = {"e4b greedy": {p["id"]: o for p, o in zip(tr + he, ok + oe)}, "qwen": {p["id"]: o for p, o in zip(tr + he, oqt + oqh)},
            "rows": sorted(sol), "row source": {k: v[1] for k, v in sol.items()}, "held": [p["id"] for p in he], "train": [p["id"] for p in tr],
            "e4b held solutions": {p["id"]: t[:t.index("```") + 3] for p, t, o in zip(he, ge, oe) if o and fenced(t)}, "rows sha": hashlib.sha256(ROWS.read_bytes()).hexdigest()}
    json.dump(base, open(BASE, "w")); log("Qwen alone: train pool", sum(oqt), "of", len(tr), "| held", sum(oqh), "of", len(he), "| E4B held", sum(oe), round(time.time() - t0), "s"); print("BUILD_DONE", flush=True)


def build_short():
    """179a (user 11-07: "yes, add the shortest-correct rows fix too"): for every row problem, NSHORT more E4B samples (T 0.8); among all of E4B's
    passing, fenced attempts (the row itself, the build's greedy and samples when xl_build_gens.jsonl is present, and these) keep the shortest in
    Qwen tokens. Model-written only: the text up to its own first fence. Writes xl_rows_short.jsonl and xl_short.json (sha, lengths)"""
    XC.selfcheck(); torch.manual_seed(1790); B = XB.Bridge("bridge"); t0 = time.time(); rows = [json.loads(l) for l in open(ROWS)]; tr = {p["id"]: p for p in pools()[0]}
    rows = [r for r in rows if r["id"] in tr]; ql = lambda t: len(B.tq(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    cut = lambda t: t[:t.index("```") + 3] if "```" in t else None  # noqa: E731
    cands = {r["id"]: {r["reply"]} for r in rows}; G = ROWS.parent / "xl_build_gens.jsonl"
    if G.exists():
        for l in open(G):
            g = json.loads(l)
            if g["model"] == "e4b" and g["pool"] == "train" and g["pass"] and g["id"] in cands and cut(g["text"]):
                cands[g["id"]].add(cut(g["text"]))
    ids = [r["id"] for r in rows]; ss = gen_sample(B.mT, B.tg, [tr[i]["prompt"] for i in ids], NSHORT, bos="<bos>"); flat = [(i, t) for i, row in zip(ids, ss) for t in row if cut(t)]
    ok = score([tr[i] for i, _ in flat], [t for _, t in flat])
    for (i, t), o in zip(flat, ok):
        if o:
            cands[i].add(cut(t))
    out, L0, L1 = [], [], []
    for r in rows:
        best = min(cands[r["id"]], key=ql); L0.append(ql(r["reply"])); L1.append(ql(best)); out.append({**r, "reply": best, "source": r["source"] + ("" if best == r["reply"] else "+short")})
    with open(ROWS_SHORT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    info = {"sha": hashlib.sha256(ROWS_SHORT.read_bytes()).hexdigest(), "rows": len(out), "changed": sum(1 for a, b in zip(L0, L1) if b < a), "candidates per row (mean)": round(float(np.mean([len(cands[i]) for i in ids])), 2),
            "median Qwen tokens, original": float(np.median(L0)), "median Qwen tokens, shortest": float(np.median(L1)), "over 600 tokens, original": sum(1 for x in L0 if x > 600), "over 600 tokens, shortest": sum(1 for x in L1 if x > 600)}
    json.dump(info, open(SHORT, "w")); log("short rows:", json.dumps(info), round(time.time() - t0), "s"); print("SHORT_DONE", flush=True)


@torch.no_grad()
def precheck():
    """the gate (review 11-07): before the arms, does the live bridge beat the live control at predicting E4B's verified solutions on these problems?
    Cross-entropy of the solution tokens for Qwen alone, the live bridge and the live control (this seed's translators), on the rows and on E4B's
    passing held solutions; per-row values, the lead (control - bridge) with a paired bootstrap 90% interval. GATE_PASS when the rows' lead is
    >= 0.02 nats and its lower bound > 0"""
    B = XB.Bridge("bridge"); Tb, Tc = XB.load_tr(B, "bridge")[0], XB.load_tr(B, "control")[0]; base = json.load(open(BASE)); t0 = time.time()
    rows = [json.loads(l) for l in open(ROWS)]; held = {p["id"]: p for p in pools()[1]}
    sets = {"rows (trained)": [{"ctx": r["ctx"], "reply": r["reply"]} for r in rows],
            "held, E4B's passing solutions": [{"ctx": held[i]["prompt"], "reply": t} for i, t in base["e4b held solutions"].items() if i in held]}
    out = {}
    for name, rs in sets.items():
        C = {"alone": [], "bridge": [], "control": []}
        for r in rs:
            e = XB.encode(B.tg, B.tq, r)
            if e is None:
                continue
            gids, qids, j, rstart, s0 = e; n = len(qids); y = torch.tensor(qids[rstart:], device=DEV); kr = n - rstart + 1; o = B.look(gids, qids, j, rstart)
            C["alone"].append(float(-o["la"].gather(-1, y[:, None])[:, 0].mean()))
            for a, T, src in (("bridge", Tb, o["T"]), ("control", Tc, o["S"])):
                B.write(T, src, s0, n)
                try:
                    L = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1)
                finally:
                    B.P.slot.clear()
                C[a].append(float(-L.gather(-1, y[:, None])[:, 0].mean()))
            del o
        A = {a: np.array(v) for a, v in C.items()}; d_ = A["control"] - A["bridge"]; rr = np.random.default_rng(0)
        if len(d_) == 0:
            out[name] = {"rows": 0}; continue
        bs = [d_[rr.integers(len(d_), size=len(d_))].mean() for _ in range(1000)]
        out[name] = {"rows": len(d_), "means": {a: round(float(v.mean()), 4) for a, v in A.items()}, "lead (control - bridge) [mean, 5%, 95%]": [round(float(d_.mean()), 4), round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)]}
        log("precheck", name, json.dumps(out[name]), round(time.time() - t0), "s")
    ld = out["rows (trained)"].get("lead (control - bridge) [mean, 5%, 95%]", [0.0, 0.0, 0.0]); out["gate"] = "GATE_PASS" if ld[0] >= 0.02 and ld[1] > 0 else "GATE_FAIL"
    json.dump(out, open(ROWS.parent / "xl_precheck.json", "w"), indent=1); print(out["gate"], flush=True)


def _lora_targets(m, lay):
    """186: the MLP projections of the chosen layers, for a Qwen (model.layers) or a Gemma (model.language_model.layers) student"""
    pre = "model.language_model.layers" if hasattr(m.model, "language_model") else "model.layers"
    return [f"{pre}.{l}.mlp.{q}_proj" for l in lay for q in ("gate", "up", "down")]


def _drop_teacher(B):
    """review 11-07: the control path never runs E4B; free its memory (fit_cf, rewrite)"""
    del B.mT; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _boot(d_):
    """mean and paired bootstrap 90% interval"""
    rr = np.random.default_rng(0); bs = [d_[rr.integers(len(d_), size=len(d_))].mean() for _ in range(1000)]
    return [round(float(d_.mean()), 4), round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)]


@torch.no_grad()
def _ce3(B, Ts, r):
    """cross-entropy of the reply tokens for Qwen alone, then with each control translator of Ts written over the span (Qwen's own states)"""
    e = XB.encode(B.tg, B.tq, r)
    if e is None:
        return None
    gids, qids, j, rstart, s0 = e; n = len(qids); y = torch.tensor(qids[rstart:], device=DEV); o = B.look(gids, qids, j, rstart, teacher=False)
    out = [float(-o["la"].gather(-1, y[:, None])[:, 0].mean())]
    for T in Ts:
        B.write(T, o["S"], s0, n)
        try:
            L = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
        finally:
            B.P.slot.clear()
        out.append(float(-L.gather(-1, y[:, None])[:, 0].mean()))
    del o; return out


def fit_cf():
    """181: the seed's control translator (fitted on agent rows) fine-tuned on E4B's verified solutions (the rows): CF_STEPS steps at CF_LR
    (warmup 30, cosine to 0.1), clean (no noise or depth dropout: 175's recipe), loss as xbridge.train (cross-entropy on the reply + XB_SIZE x
    the change's size + XB_HARM x do-no-harm where Qwen alone is already right). Then the gate: cross-entropy of E4B's solution tokens for Qwen
    alone, the old control and the refit control, on the rows and on E4B's passing held solutions (never trained on). CF_PASS when the refit
    beats the old control on the rows by >= CF_MIN nats (5% bound > 0) and beats Qwen alone on the held solutions (5% bound > 0)"""
    B = XB.Bridge("control"); _drop_teacher(B); t0 = time.time(); rows = [json.loads(l) for l in open(ROWS)]; base = json.load(open(BASE))
    assert hashlib.sha256(ROWS.read_bytes()).hexdigest() == base["rows sha"], "rows and baselines do not match"
    rows = rows[:3] if SMALL else rows
    enc = [XB.encode(B.tg, B.tq, {"ctx": r["ctx"], "reply": r["reply"]}) for r in rows]; assert all(e is not None for e in enc), "a row failed to encode"
    ck = torch.load(OD / "xb_control.pt", weights_only=False, map_location=DEV); T0 = XB.load_tr(B, "control")[0]; T = XB.load_tr(B, "control")[0]; T.train()
    for p_ in T0.parameters():
        p_.requires_grad_(False)
    rng = np.random.default_rng(1800 + SEED); torch.manual_seed(1801 + SEED)
    order = np.concatenate([rng.permutation(len(enc)) for _ in range(math.ceil(CF_STEPS * XB.ACC / len(enc)) + 1)])
    opt = torch.optim.AdamW(T.parameters(), lr=CF_LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / CF_STEPS)))  # noqa: E731
    log("refit control translator: rows", len(enc), "| steps", CF_STEPS, "x", XB.ACC, "| lr", CF_LR, "| size", XB.SIZE, "harm", XB.HARM, "| clean (no noise, no depth dropout)"); hist = []
    for step in range(CF_STEPS):
        for g_ in opt.param_groups:
            g_["lr"] = CF_LR * sched(step)
        opt.zero_grad(set_to_none=True); vals = []
        for a_ in range(XB.ACC):
            gids, qids, j, rstart, s0 = enc[int(order[step * XB.ACC + a_])]; n = len(qids); o = B.look(gids, qids, j, rstart, teacher=False)
            d, g, aw = B.write(T, o["S"], s0, n, h=o["E"])
            try:
                lsm = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
            finally:
                B.P.slot.clear()
            y = torch.tensor(qids[rstart:], device=DEV); ce = -lsm.gather(-1, y[:, None])[:, 0].mean(); loss = ce + XB.SIZE * ((d ** 2).mean() + aw); hm = torch.zeros((), device=DEV)
            ag = o["la"].argmax(-1) == y                                 # Qwen alone already right: keep its own distribution there
            if XB.HARM > 0 and ag.any():
                sv, si = o["la"][ag].topk(XB.TOPK, -1); hm = XB.tail_ce(lsm[ag], si, sv); loss = loss + XB.HARM * hm
            (loss / XB.ACC).backward(); vals.append((float(ce.detach()), float(-o["la"].gather(-1, y[:, None])[:, 0].mean()), float(hm.detach()), float(g.mean().detach())))
            del o, lsm, d, g, aw, loss
        torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step(); hist.append(np.mean(vals, 0))
        if step % 50 == 0 or step == CF_STEPS - 1:
            h_ = np.mean(hist[-50:], 0); log("refit step", step, "of", CF_STEPS, "| cross-entropy (refit, live)", round(float(h_[0]), 4), "(alone", round(float(h_[1]), 4), ") harm", round(float(h_[2]), 4), "gate", round(float(h_[3]), 4), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "loss is not finite"
    T.eval(); ck["state"] = {k: v.detach().clone() for k, v in T.state_dict().items()}; ck["steps"] = ck["steps"] + CF_STEPS
    ck["refit"] = {"rows": str(ROWS), "rows sha": base["rows sha"], "steps": CF_STEPS, "lr": CF_LR, "clean": True}; torch.save(ck, OD / f"{CFN}.pt")
    held = {p["id"]: p for p in pools()[1]}; hs = [{"ctx": held[i]["prompt"], "reply": t} for i, t in base["e4b held solutions"].items() if i in held]
    sets = {"rows (trained)": [{"ctx": r["ctx"], "reply": r["reply"]} for r in rows], "held, E4B's passing solutions": hs[:3] if SMALL else hs}
    out = {"steps": CF_STEPS, "lr": CF_LR, "min lead": CF_MIN, "first50 (ce, alone, harm, gate)": [float(v) for v in np.mean(hist[:50], 0)], "last50": [float(v) for v in np.mean(hist[-50:], 0)]}
    for name, rs in sets.items():
        V = np.array([v for v in (_ce3(B, (T0, T), r) for r in rs) if v is not None])
        out[name] = {"rows": int(len(V))} if len(V) == 0 else {"rows": int(len(V)), "means (alone, old control, refit)": [round(float(x), 4) for x in V.mean(0)],
                                                               "old - refit": _boot(V[:, 1] - V[:, 2]), "alone - refit": _boot(V[:, 0] - V[:, 2])}
    r_ = out["rows (trained)"].get("old - refit", [0.0, 0.0, 0.0]); h_ = out["held, E4B's passing solutions"].get("alone - refit", [0.0, 0.0, 0.0])
    out["gate"] = "CF_PASS" if r_[0] >= CF_MIN and r_[1] > 0 and h_[1] > 0 else "CF_FAIL"
    json.dump(out, open(CF_GATE, "w"), indent=1); log("refit gate:", json.dumps(out), round(time.time() - t0), "s"); print(out["gate"], flush=True)


class _Cap:
    """the chosen Qwen layers' outputs for the whole batch while .on (xbridge.Grab keeps batch row 0 only)"""
    def __init__(s, m, ls):
        s.on, s.out = False, {}; s.hs = [XB.layers(m)[l].register_forward_hook(s.make(l)) for l in ls]

    def make(s, l):
        def hook(_m, _i, out):
            if s.on:
                s.out[l] = (out[0] if isinstance(out, tuple) else out).detach().float()
            return out
        return hook

    def close(s):
        for h in s.hs:
            h.remove()


@torch.no_grad()
def gen_cf(B, T, prompts, n, seed, maxnew=None, with_greedy=True):
    """a control translator T live (Qwen's own states), KV-cached: per token two passes of the same Qwen, one alone (its source states at the new
    position; cache A) and one with T written there (cache B; the next-token logits). The same computation as xcode.gen_bridged up to numerics:
    the gall translator acts per position and every generated position lies in the written span (s0 onward, xcode.align on the prompt). Left
    padding with position ids from the attention mask, as generate. Per prompt: [greedy (with_greedy), n samples (T TEMP, top-p 0.95)]; a row
    stops at an end token or its closing ```"""
    from transformers import DynamicCache
    maxnew = maxnew or XC.MAXNEW; eos = set(XC.stop_ids(B.mS, B.tq)); pad = B.tq.pad_token_id if B.tq.pad_token_id is not None else B.tq.eos_token_id
    cap = _Cap(B.mS, B.SL); gen_ = torch.Generator(device=DEV).manual_seed(seed); outs = []; k0 = 0 if with_greedy else 1; per = max(1, RW_ROWS // (n + 1 - k0))   # review: batch by rows
    try:
        for a in range(0, len(prompts), per):
            ch = prompts[a:a + per]; P = [B.tq(XB.SBOS + p, add_special_tokens=False)["input_ids"] for p in ch]; S0 = [XC.align(B, p, p)[3] for p in ch]
            rows_ = [(i, k) for i in range(len(ch)) for k in range(k0, n + 1)]; R = len(rows_); L = max(len(x) for x in P)
            ids = torch.full((R, L), pad, dtype=torch.long); am = torch.zeros((R, L), dtype=torch.long); m = torch.zeros((R, L), dtype=torch.bool)
            for r_, (i, _) in enumerate(rows_):
                q = P[i]; ids[r_, L - len(q):] = torch.tensor(q); am[r_, L - len(q):] = 1; m[r_, L - len(q) + S0[i]:] = True
            x_in, am, m = ids.to(DEV), am.to(DEV), m.to(DEV); pos = (am.cumsum(-1) - 1).clamp_min(0); cA, cB = DynamicCache(), DynamicCache()
            greedy = torch.tensor([k == 0 for _, k in rows_], device=DEV); gen = [[] for _ in range(R)]; done = [False] * R
            for t in range(maxnew):
                cap.on = True; B.mS(input_ids=x_in, attention_mask=am, position_ids=pos, past_key_values=cA, use_cache=True, logits_to_keep=1); cap.on = False
                S = torch.cat([cap.out[l] for l in B.SL], -1); cap.out = {}

                def fn(hh, S=S):
                    new, dl, _, _, _ = T.gapply(S, hh); B.P.slot["d"] = dl; return new
                B.P.slot["mask"] = m if t == 0 else torch.ones((R, 1), dtype=torch.bool, device=DEV); B.P.slot["fn"] = fn
                try:
                    lg = B.mS(input_ids=x_in, attention_mask=am, position_ids=pos, past_key_values=cB, use_cache=True, logits_to_keep=1).logits[:, -1].float()
                finally:
                    B.P.slot.clear()
                pr = torch.softmax(lg / TEMP, -1); sp, si = pr.sort(-1, descending=True); cs = sp.cumsum(-1); sp[(cs - sp) > 0.95] = 0; sp = sp / sp.sum(-1, keepdim=True)
                nxt = torch.where(greedy, lg.argmax(-1), si.gather(-1, torch.multinomial(sp, 1, generator=gen_))[:, 0]); nl = nxt.tolist(); del lg, pr, sp, si, cs, S
                for r_ in range(R):
                    if done[r_]:
                        continue
                    if nl[r_] in eos:
                        done[r_] = True; continue
                    gen[r_].append(nl[r_])
                    if "```" in B.tq.decode(gen[r_][-6:], skip_special_tokens=True):
                        done[r_] = True
                if all(done):
                    break
                x_in = nxt[:, None]; am = torch.cat([am, torch.ones((R, 1), dtype=am.dtype, device=DEV)], 1); pos = pos[:, -1:] + 1
            dec = [B.tq.decode(g, skip_special_tokens=True) for g in gen]; outs += [[dec[r_] for r_, (i, _) in enumerate(rows_) if i == i_] for i_ in range(len(ch))]
            del cA, cB; log("generated (translator live)", a + len(ch), "of", len(prompts), "| rows per prompt", n + 1 - k0)
    finally:
        cap.close()
    return outs


def _cf_selfcheck(B, T, prompts, k=48):
    """the cached generator against xcode.gen_bridged (no cache, re-tokenized each step), greedy, k tokens"""
    old = XC.MAXNEW; XC.MAXNEW = k; eos = set(XC.stop_ids(B.mS, B.tq))
    try:
        ref = [XC.gen_bridged(B, T, "control", p, eos) for p in prompts]
    finally:
        XC.MAXNEW = old
    got = [g[0] for g in gen_cf(B, T, prompts, 0, 0, maxnew=k)]; pre = [len(os.path.commonprefix([a, b])) for a, b in zip(ref, got)]
    log("self-check, cached vs uncached greedy:", pre, "common characters of", [len(a) for a in ref])
    assert max(pre) >= min(32, max(len(a) for a in ref)), "the cached generator disagrees with xcode.gen_bridged"
    p = prompts[0]; _, qids, _, s0 = XC.align(B, p, p); x = torch.tensor([qids], device=DEV); n = len(qids)   # review: the translator must be live
    with torch.no_grad():
        B.gS.on = True; l0 = B.mS(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1].float(); oS = B.gS.take(); B.write(T, torch.cat([oS[l] for l in B.SL], -1), s0, n)
        try:
            l1 = B.mS(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        finally:
            B.P.slot.clear()
    dl = float((l1 - l0).abs().max()); log("self-check, the translator changes the next-token logits by (max abs)", round(dl, 4)); assert dl > 1e-3, "the translator is not live"


def rewrite():
    """181, rw_cf's rows: the refit control translator live, Qwen writes each row problem's solution (greedy + NRW samples; a second round of
    NRW samples where none passed). A row is the greedy rewrite if it passes every hidden test (fenced, cut at its own first ```), else the
    first such sample; no row where none passed. Also: the teachers' own greedy solve rates (old and refit control, live) on the rows'
    problems and the held pool, and how far the rewrite rows are from E4B's (Qwen alone's cross-entropy of both, text similarity, length).
    Every generation is kept (xl_rw_gens.jsonl). Model-written only"""
    import difflib
    XC.selfcheck(); B = XB.Bridge("control"); _drop_teacher(B); t0 = time.time(); base = json.load(open(BASE)); rows = [json.loads(l) for l in open(ROWS)]
    assert hashlib.sha256(ROWS.read_bytes()).hexdigest() == base["rows sha"], "rows and baselines do not match"
    trp = {p["id"]: p for p in load(TRAIN_FILES, ("easy", "medium", "hard"), 10 ** 6)}; he = pools()[1]
    rows = [r for r in rows if r["id"] in trp]; rows = rows[:3] if SMALL else rows; assert all(r["ctx"] == trp[r["id"]]["prompt"] for r in rows), "row prompts changed"
    Tc, Tf = XB.load_tr(B, "control")[0], XB.load_tr(B, "control", name=CFN)[0]; _cf_selfcheck(B, Tf, [r["ctx"] for r in rows[:2]])
    cut = lambda t: t[:t.index("```") + 3] if "```" in t else None  # noqa: E731
    G, best = [], {}
    for rnd in (1, 2):
        todo = [r for r in rows if r["id"] not in best]
        if not todo:
            break
        gs = gen_cf(B, Tf, [r["ctx"] for r in todo], NRW, 1803 + 10 * SEED + rnd, with_greedy=rnd == 1)
        flat = [(r, "greedy" if (rnd == 1 and k == 0) else "sample", t) for r, g in zip(todo, gs) for k, t in enumerate(g)]
        ok = score([trp[r["id"]] for r, _, _ in flat], [t for _, _, t in flat])
        for (r, kind, t), o in zip(flat, ok):                           # in order: a problem's greedy first, then its samples
            G.append({"pool": "train", "round": rnd, "teacher": "refit", "id": r["id"], "kind": kind, "text": t, "pass": o})
            if o and cut(t) and r["id"] not in best:
                best[r["id"]] = (cut(t), kind)
        log(f"rewrite round {rnd}: problems {len(todo)} | passing rewrites so far {len(best)} of {len(rows)}", round(time.time() - t0), "s")
    gc = gen_cf(B, Tc, [r["ctx"] for r in rows], 0, 0); oc = score([trp[r["id"]] for r in rows], [g[0] for g in gc])
    G += [{"pool": "train", "round": 0, "teacher": "old", "id": r["id"], "kind": "greedy", "text": g[0], "pass": o} for r, g, o in zip(rows, gc, oc)]
    hp = [p["prompt"] for p in he]; teach = {}
    for nm, T in (("refit", Tf), ("old", Tc)):
        g = [x[0] for x in gen_cf(B, T, hp, 0, 0)]; o = score(he, g); teach[nm] = sum(o)
        G += [{"pool": "held", "round": 0, "teacher": nm, "id": p["id"], "kind": "greedy", "text": t, "pass": oo} for p, t, oo in zip(he, g, o)]
    if SMALL and not best:                                               # the small pass checks the plumbing only: E4B's own rows stand in
        best = {r["id"]: (r["reply"], "small fallback, E4B row") for r in rows}
    q = base["qwen"]; ids = [r["id"] for r in rows]; e4 = {r["id"]: r["reply"] for r in rows}; ctx = {r["id"]: r["ctx"] for r in rows}
    ql = lambda t: len(B.tq(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    D_ = [(_ce3(B, (), {"ctx": ctx[i], "reply": t}), _ce3(B, (), {"ctx": ctx[i], "reply": e4[i]}), difflib.SequenceMatcher(None, t, e4[i]).ratio(), t == e4[i], ql(t), ql(e4[i])) for i, (t, _) in best.items()]
    D_ = [d for d in D_ if d[0] and d[1]]
    with open(ROWS_RW, "w") as fh:
        for r in rows:
            if r["id"] in best:
                fh.write(json.dumps({"id": r["id"], "ctx": r["ctx"], "reply": best[r["id"]][0], "source": "rw " + best[r["id"]][1], "difficulty": r["difficulty"]}) + "\n")
    with open(ROWS_RW.parent / "xl_rw_gens.jsonl", "w") as fh:
        for x in G:
            fh.write(json.dumps(x) + "\n")
    gr = {x["id"]: x["pass"] for x in G if x["teacher"] == "refit" and x["kind"] == "greedy" and x["pool"] == "train"}
    info = {"sha": hashlib.sha256(ROWS_RW.read_bytes()).hexdigest(), "rows": len(best), "row problems": len(rows), "rows by kind": {k: sum(1 for v in best.values() if v[1] == k) for k in sorted({v[1] for v in best.values()})},
            "covered where Qwen alone fails": [sum(1 for i in ids if i in best and not q.get(i)), sum(1 for i in ids if not q.get(i))],
            "teacher greedy, row problems (refit, old, Qwen alone)": [sum(gr.values()), sum(oc), sum(1 for i in ids if q.get(i))],
            "teacher greedy, held (refit, old, Qwen alone)": [teach["refit"], teach["old"], sum(1 for p in he if q.get(p["id"]))], "held problems": len(he),
            "Qwen alone cross-entropy, median (rewrite rows, E4B rows)": [round(float(np.median([d[0][0] for d in D_])), 4), round(float(np.median([d[1][0] for d in D_])), 4)] if D_ else None,
            "similarity to E4B's row, median (difflib)": round(float(np.median([d[2] for d in D_])), 4) if D_ else None, "identical to E4B's row": sum(1 for d in D_ if d[3]),
            "median Qwen tokens (rewrite rows, E4B rows)": [float(np.median([d[4] for d in D_])), float(np.median([d[5] for d in D_]))] if D_ else None}
    json.dump(info, open(RW, "w"), indent=1); log("rewrite rows:", json.dumps(info), round(time.time() - t0), "s"); print("RW_DONE", flush=True)


_CODE = re.compile(r"^(?:\d+[.)]\s*)?(?:def |class |import |from \S+ import |return\b|print\(|for \w+ in .+:$|while .+:$|if .+:$|elif .+:$|else:$)")


_STEP = re.compile(r"^(?:\*\*)?(?:step\s*)?(\d+)[.):]", re.I)        # "1.", "2)", "**3.**", "Step 4:"


def _plan(t):
    """182: the numbered plan that opens a runbook generation (the prompt ends with "1."): the steps and their continuation lines, up to a fence,
    a step numbered above 10, or an unindented paragraph after a blank line that is neither a step nor a bullet (review 11-08: markdown lists
    with blank lines between a step and its indented or bulleted details are kept)"""
    out, blank = [], False
    for ln in ("1." + t.split("```")[0]).split("\n"):
        s = ln.strip()
        if not s:
            blank = True; continue
        m = _STEP.match(s)
        if m and int(m.group(1)) > 10:
            break
        if m or (out and (not blank or ln[:1] in (" ", "\t") or s[:1] in "-*•")):
            out.append(s); blank = False
        else:
            break
    return "\n".join(out)


def _g4(s):
    t = re.findall(r"[A-Za-z_]\w*|\d+|[^\w\s]", s); return {tuple(t[i:i + 4]) for i in range(len(t) - 3)}


def _bare(code):
    """review 11-08: the code without comments and string contents (E4B's comments are English; counting them made the screen measure prose)"""
    import io, tokenize
    skip = {"COMMENT", "STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END", "NL", "NEWLINE", "INDENT", "DEDENT", "ENDMARKER", "ENCODING"}
    try:
        return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if tokenize.tok_name[t.type] not in skip)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return "\n".join(l.split("#")[0] for l in code.split("\n"))


def _screen(plan, code):
    """182 leakage screen: (passes, reason, overlap). At least 2 steps and 60 characters, no fence, no line that reads as code, no inline code span
    over 20 characters with "=", "(" or "[", and at most RB_OVER of the solution's code-token 4-grams (comments and strings removed) in the plan"""
    lines = plan.split("\n"); g = _g4(_bare(code)); ov = len(g & _g4(plan)) / max(1, len(g))
    if sum(1 for l in lines if _STEP.match(l)) < 2 or len(plan) < 60:
        return False, "too short", ov
    if "```" in plan:
        return False, "fence", ov
    if any(_CODE.match(l) for l in lines):
        return False, "code line", ov
    if any(len(x) > 20 and any(c in x for c in "=([") for x in re.findall(r"`([^`\n]+)`", plan)):
        return False, "inline code", ov
    return (True, "ok", ov) if ov <= RB_OVER else (False, "code overlap", ov)


def runbook():
    """182, rb's rows (built once; every seed trains on them). For each row problem E4B reads the problem with its own verified solution (the row)
    and writes a plan (RB_ASK; greedy, then one sample per attempt, up to RB_TRY attempts, until one passes the leakage screen _screen). Qwen alone
    then solves the problem with the plan in its prompt (RB_USE): greedy, then NRB samples where it failed, then NRB more where none passed. A row is
    the greedy solution if it passes every hidden test (fenced, cut at its own first ```), else the first passing sample, stored with the ORIGINAL
    prompt (no plan); no row where none passed. Also: Qwen's greedy solve rate with the plan against Qwen alone, and how far the rows are from
    E4B's (Qwen alone's cross-entropy of both, text similarity, length) and from Qwen alone's own greedy. Every generation is kept
    (xl_rb_gens.jsonl). Model-written only: E4B's plans, Qwen's code"""
    import difflib
    from collections import Counter
    XC.selfcheck(); B = XB.Bridge("control"); t0 = time.time(); base = json.load(open(BASE)); rows = [json.loads(l) for l in open(ROWS)]
    assert hashlib.sha256(ROWS.read_bytes()).hexdigest() == base["rows sha"], "rows and baselines do not match"
    trp = {p["id"]: p for p in load(TRAIN_FILES, ("easy", "medium", "hard"), 10 ** 6)}
    rows = [r for r in rows if r["id"] in trp]; rows = rows[:3] if SMALL else rows[:RB_PROBE] if RB_PROBE else rows; assert all(r["ctx"] == trp[r["id"]]["prompt"] for r in rows), "row prompts changed"
    assert all(r["ctx"].startswith(RB_HEAD) and r["ctx"].endswith(RB_TAIL) and r["reply"].endswith("```") for r in rows), "prompt or reply format changed"
    cut = lambda t: t[:t.index("```") + 3] if "```" in t else None  # noqa: E731
    ask = {r["id"]: RB_ASK.format(problem=r["ctx"][len(RB_HEAD):-len(RB_TAIL)], code=r["reply"][:-3]) for r in rows}
    G, plan, why = [], {}, {}; old = XC.MAXNEW; XC.MAXNEW = RB_MAXNEW
    if RB_PLANS:                                                         # 186: E4B's accepted plans from an earlier build (another student, the same plans; no E4B here)
        for l in open(RB_PLANS):
            g = json.loads(l)
            if g.get("stage") == "plan" and g.get("screen") == "ok" and g["id"] in ask:
                plan[g["id"]] = g["plan"]; G.append({**g, "reused from": RB_PLANS})
        why = {r["id"]: "no plan in the reused build" for r in rows if r["id"] not in plan}; log("plans reused:", len(plan), "of", len(rows), "from", RB_PLANS)
    try:
        for k in range(0 if RB_PLANS else RB_TRY):                                          # attempt 0: E4B greedy; later attempts: one sample each, for the plans that failed
            todo = [r for r in rows if r["id"] not in plan]
            if not todo:
                break
            ps = [ask[r["id"]] for r in todo]
            if k == 0:
                gs = XC.gen_hf(B.mT, B.tg, ps, bos="<bos>")
            else:
                torch.manual_seed(1820 + k); gs = [g[0] for g in gen_sample(B.mT, B.tg, ps, 1, bos="<bos>")]
            for r, t in zip(todo, gs):
                p_ = _plan(t); ok, reason, ov = _screen(p_, r["reply"][:-3])
                G.append({"stage": "plan", "id": r["id"], "attempt": k, "text": t, "plan": p_, "screen": reason, "overlap": round(ov, 4)})
                if ok:
                    plan[r["id"]] = p_
                why[r["id"]] = reason
            log(f"plans, attempt {k}: {len(plan)} of {len(rows)} pass the screen", round(time.time() - t0), "s")
    finally:
        XC.MAXNEW = old
    if RB_PROBE:                                                         # review 11-08: the yield check (plans only), before the full build
        last = {x["id"]: x for x in G}; info = {"probe rows": len(rows), "plans passing the screen": len(plan), "last screen result": dict(Counter(why.values())),
                                                 "overlap, median (all last attempts)": round(float(np.median([x["overlap"] for x in last.values()])), 4) if last else None,
                                                 "plans": {i: x["plan"] for i, x in list(last.items())[:6]}}
        json.dump(info, open(OD.parent / "xl_rb_probe.json", "w"), indent=1); log("probe:", json.dumps(info)[:3000]); print("RB_PROBE_DONE", flush=True); return
    _drop_teacher(B)
    if plan:
        i0 = next(iter(plan)); log("first plan (", i0, "):", plan[i0].replace("\n", " | ")[:600])
    up = {i: trp[i]["prompt"][:-len(RB_TAIL)] + RB_USE.format(plan=p_) for i, p_ in plan.items()}; best = {}
    for rnd in (1, 2, 3):                                                # 1: greedy; 2: NRB samples where it failed; 3: NRB more where none passed
        todo = [r for r in rows if r["id"] in plan and r["id"] not in best]
        if not todo:
            break
        ps = [up[r["id"]] for r in todo]; torch.manual_seed(1822 + rnd)
        gs = [[g] for g in XC.gen_hf(B.mS, B.tq, ps, bos=XB.SBOS)] if rnd == 1 else gen_sample(B.mS, B.tq, ps, NRB, bos=XB.SBOS)
        flat = [(r, "greedy" if rnd == 1 else "sample", t) for r, g in zip(todo, gs) for t in g]
        ok = score([trp[r["id"]] for r, _, _ in flat], [t for _, _, t in flat])
        for (r, kind, t), o in zip(flat, ok):                           # in order: a problem's greedy first, then its samples
            G.append({"stage": "solve", "round": rnd, "id": r["id"], "kind": kind, "text": t, "pass": o})
            if o and cut(t) and r["id"] not in best:
                best[r["id"]] = (cut(t), kind)
        log(f"runbook round {rnd}: problems {len(todo)} | passing solutions so far {len(best)} of {len(rows)}", round(time.time() - t0), "s")
    if SMALL and not best:                                               # review 11-08 (186): the small pass checks the plumbing only; E4B's rows stand in
        best = {r["id"]: (r["reply"], "small fallback, E4B row") for r in rows}
    q = base["qwen"]; ids = [r["id"] for r in rows]; e4 = {r["id"]: r["reply"] for r in rows}; ctx = {r["id"]: r["ctx"] for r in rows}   # q: the build's Qwen run (for another student, its own alone run is in xl_eval)
    ql = lambda t: len(B.tq(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    qg, sh, BG = {}, {}, ROWS.parent / "xl_build_gens.jsonl"            # Qwen alone's own greedy (the build's generations) and the shortest E4B rows, when present
    if BG.exists():
        for l in open(BG):
            g = json.loads(l)
            if g["model"] == "qwen" and g["kind"] == "greedy" and g["pool"] == "train" and g["id"] in e4:
                qg[g["id"]] = cut(g["text"]) or g["text"]
    if ROWS_SHORT.exists():
        for l in open(ROWS_SHORT):
            r_ = json.loads(l); sh[r_["id"]] = r_["reply"]
    sim = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()  # noqa: E731
    D_ = [(i, _ce3(B, (), {"ctx": ctx[i], "reply": t}), _ce3(B, (), {"ctx": ctx[i], "reply": e4[i]}), sim(t, e4[i]), t == e4[i], ql(t), ql(e4[i]),
           sim(t, qg[i]) if i in qg else None, t == qg.get(i), ql(sh[i]) if i in sh else None) for i, (t, _) in best.items()]
    D_ = [d for d in D_ if d[1] and d[2]]; med = lambda v: round(float(np.median(v)), 4) if len(v) else None  # noqa: E731
    with open(ROWS_RB, "w") as fh:
        for r in rows:
            if r["id"] in best:
                fh.write(json.dumps({"id": r["id"], "ctx": r["ctx"], "reply": best[r["id"]][0], "source": "rb " + best[r["id"]][1], "difficulty": r["difficulty"]}) + "\n")
    with open(ROWS_RB.parent / "xl_rb_gens.jsonl", "w") as fh:
        for x in G:
            fh.write(json.dumps(x) + "\n")
    gp = {x["id"]: x["pass"] for x in G if x["stage"] == "solve" and x["kind"] == "greedy"}; acc = [x for x in G if x["stage"] == "plan" and x["screen"] == "ok"]
    info = {"sha": hashlib.sha256(ROWS_RB.read_bytes()).hexdigest(), "rows": len(best), "row problems": len(rows),
            "rows by kind": dict(Counter(v[1] for v in best.values())), "plans passing the screen": len(plan), "plans by attempt": dict(Counter(x["attempt"] for x in acc)),
            "no plan: last screen result": dict(Counter(why[i] for i in ids if i not in plan)), "plan characters, median": med([len(p_) for p_ in plan.values()]),
            "plan steps, median": med([sum(1 for l in p_.split("\n") if _STEP.match(l)) for p_ in plan.values()]),
            "plan code-token 4-gram overlap, median (accepted)": med([x["overlap"] for x in acc]),
            "covered where Qwen alone fails": [sum(1 for i in ids if i in best and not q.get(i)), sum(1 for i in ids if not q.get(i))],
            "Qwen greedy on row problems (with the plan, alone)": [sum(1 for v in gp.values() if v), sum(1 for i in ids if q.get(i))],
            "Qwen greedy on the planned problems (with the plan, alone)": [sum(1 for v in gp.values() if v), sum(1 for i in plan if q.get(i))],
            "Qwen greedy with the plan where Qwen alone fails": [sum(1 for i in ids if gp.get(i) and not q.get(i)), sum(1 for i in ids if not q.get(i))],
            "Qwen alone cross-entropy, median (runbook rows, E4B rows)": [med([d[1][0] for d in D_]), med([d[2][0] for d in D_])],
            "similarity to E4B's row, median (difflib)": med([d[3] for d in D_]), "identical to E4B's row": sum(1 for d in D_ if d[4]),
            "similarity to Qwen alone's greedy, median (problems Qwen alone passes, fails)": [med([d[7] for d in D_ if d[7] is not None and q.get(d[0])]), med([d[7] for d in D_ if d[7] is not None and not q.get(d[0])])],
            "identical to Qwen alone's greedy": sum(1 for d in D_ if d[8]),
            "median Qwen tokens (runbook rows, E4B rows, shortest E4B rows)": [med([d[5] for d in D_]), med([d[6] for d in D_]), med([d[9] for d in D_ if d[9] is not None])]}
    json.dump(info, open(RB, "w"), indent=1); log("runbook rows:", json.dumps(info), round(time.time() - t0), "s"); print("RB_DONE", flush=True)


def build_from_gens():
    """187, direction E2B -> E4B: the rows are another model's own greedy passing answers (XL_ROWS_FROM: an E2B session's xl_gens_qwen.jsonl, the
    student alone, deterministic) on the build's row problems, fenced and cut at their own first ```, with the original prompts. Writes
    xl_rows.jsonl and xl_base.json over the session's copies (rows, row source and rows sha replaced; everything else kept). Model-written only"""
    base = json.load(open(BASE)); want = set(base["rows"]); trp = {p["id"]: p for p in load(TRAIN_FILES, ("easy", "medium", "hard"), 10 ** 6)}
    cut = lambda t: t[:t.index("```") + 3] if "```" in t else None  # noqa: E731
    got = {}; ok = None
    if os.environ.get("XL_ROWS_PASS"):                                   # 187a: the re-scored pass flags (the evaluation's per-problem results), not the first-pass ones
        ok = json.load(open(os.environ["XL_ROWS_PASS"]))["per problem"]["qwen"]
    for l in open(os.environ["XL_ROWS_FROM"]):
        g = json.loads(l)
        if g["id"] in want and (ok.get(g["id"]) if ok is not None else g["pass"]) and cut(g["text"]):
            got[g["id"]] = cut(g["text"])
    order = [i for i in base["rows"] if i in got]; assert order, "no passing answers to build rows from"
    with open(ROWS, "w") as fh:
        for i in order:
            fh.write(json.dumps({"id": i, "ctx": trp[i]["prompt"], "reply": got[i], "source": "greedy", "difficulty": trp[i]["difficulty"]}) + "\n")
    nb = {**base, "rows": order, "row source": {i: "greedy" for i in order}, "rows sha": hashlib.sha256(ROWS.read_bytes()).hexdigest(), "rows from": os.environ["XL_ROWS_FROM"],
          "rows of the build (E4B)": len(want)}
    json.dump(nb, open(BASE, "w")); log("rows from", os.environ["XL_ROWS_FROM"], ":", len(order), "of", len(want), "row problems"); print("ROWS_FROM_DONE", flush=True)


def build_keep():
    """184, the keep rows (deterministic, no model): for each row problem, Qwen alone's own passing greedy solution from the build
    (xl_build_gens.jsonl; fenced, cut at its own first ```) where the build's Qwen run passed, else the shortest E4B row (sft_s's). A problem
    Qwen passed whose greedy text has no closing fence keeps the E4B row (source "e4b short, qwen passed"; not used by fit_xk). Writes
    xl_rows_k.jsonl and xl_k.json"""
    base = json.load(open(BASE)); q = base["qwen"]; rows = [json.loads(l) for l in open(ROWS_SHORT)]; assert {r["id"] for r in rows} == set(base["rows"]), "short rows do not match"
    assert hashlib.sha256(ROWS_SHORT.read_bytes()).hexdigest() == json.load(open(SHORT))["sha"], "short rows changed"
    cut = lambda t: t[:t.index("```") + 3] if "```" in t else None  # noqa: E731
    own = {}
    for l in open(ROWS.parent / "xl_build_gens.jsonl"):
        g = json.loads(l)
        if g["model"] == "qwen" and g["kind"] == "greedy" and g["pool"] == "train" and g["pass"] and cut(g["text"]):
            own[g["id"]] = cut(g["text"])
    out = []
    for r in rows:
        if q.get(r["id"]) and r["id"] in own:
            out.append({**r, "reply": own[r["id"]], "source": "qwen own"})
        else:
            out.append({**r, "source": "e4b short, qwen passed" if q.get(r["id"]) else "e4b short"})
    with open(ROWS_K, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    info = {"sha": hashlib.sha256(ROWS_K.read_bytes()).hexdigest(), "rows": len(out), "by source": dict(Counter(r["source"] for r in out)),
            "median characters by source": {k: float(np.median([len(r["reply"]) for r in out if r["source"] == k])) for k in sorted({r["source"] for r in out})}}
    json.dump(info, open(KJ, "w"), indent=1); log("keep rows:", json.dumps(info)); print("KEEP_DONE", flush=True)


@torch.no_grad()
def _kl3(B, Ts, r):
    """KL(Qwen alone || Qwen with each control translator of Ts written over the span), mean over the reply tokens"""
    e = XB.encode(B.tg, B.tq, r)
    if e is None:
        return None
    gids, qids, j, rstart, s0 = e; n = len(qids); o = B.look(gids, qids, j, rstart, teacher=False); out = []
    for T in Ts:
        B.write(T, o["S"], s0, n)
        try:
            L = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
        finally:
            B.P.slot.clear()
        out.append(float((o["la"].exp() * (o["la"] - L)).sum(-1).mean()))
    del o; return out


def fit_xk():
    """184, the cross-fitted selective translator: the row problems split in two folds (stratified by keep / learn, seed 1840 + SEED); for each
    fold f, the seed's control translator (xb_control.pt) is fine-tuned CF_STEPS clean steps at CF_LR (warmup 30, cosine, as fit_cf) on fold f's
    rows only: learn rows (E4B's shortest solution where Qwen alone fails): cross-entropy + size + do-no-harm (as fit_cf); keep rows (Qwen's own
    passing solution where it succeeds): XK_KW x KL(Qwen alone || corrected) over the whole reply + size + XK_GW x the mean gate, the fragile set
    (xl_fragile.json "train", lost in >= 6 of 24 earlier runs) oversampled XK_FRW x. Saved as xb_control_xk0 / xb_control_xk1; arm xk takes each
    row's target from the translator fitted on the OTHER fold (never on that row). Gate on the unseen fold: learn rows, cross-entropy (alone, old
    control, cross-fit); keep rows, KL from Qwen alone (old, cross-fit); and E4B's passing held solutions. XK_PASS when the cross-fit beats the
    old control on unseen learn rows by >= XK_MIN nats (5% bound > 0) and is not more intrusive on unseen keep rows (KL cross-fit - old: 95% bound
    < 0.02)"""
    B = XB.Bridge("control")
    if B.mT is not None:
        _drop_teacher(B)
    t0 = time.time(); base = json.load(open(BASE)); kj = json.load(open(KJ)); assert hashlib.sha256(ROWS_K.read_bytes()).hexdigest() == kj["sha"], "keep rows changed"
    rows = [json.loads(l) for l in open(ROWS_K)]; rows = rows[:16] if SMALL else rows; fr = set(json.load(open(FRAG))["train"]) if FRAG.exists() else set()
    keep = [r for r in rows if r["source"] == "qwen own"]; learn = [r for r in rows if r["source"] == "e4b short"]; rng = np.random.default_rng(1840 + SEED); fold = {}
    other = [r for r in rows if r["source"] not in ("qwen own", "e4b short")]   # in a fold too (arm xk needs an unseen teacher for every row), not fitted on
    for grp in (keep, learn, other):
        for k, i in enumerate(rng.permutation(len(grp))):
            fold[grp[i]["id"]] = k % 2
    enc = {r["id"]: XB.encode(B.tg, B.tq, {"ctx": r["ctx"], "reply": r["reply"]}) for r in keep + learn}; assert all(e is not None for e in enc.values()), "a row failed to encode"
    kind_ = {r["id"]: "keep" for r in keep} | {r["id"]: "learn" for r in learn}
    T0 = XB.load_tr(B, "control")[0]
    for p_ in T0.parameters():
        p_.requires_grad_(False)
    info = {"folds": {f: {"keep": sum(1 for r in keep if fold[r["id"]] == f), "learn": sum(1 for r in learn if fold[r["id"]] == f), "fragile keep": sum(1 for r in keep if fold[r["id"]] == f and r["id"] in fr)} for f in (0, 1)},
            "steps": CF_STEPS, "lr": CF_LR, "keep KL weight": XK_KW, "gate penalty": XK_GW, "fragile oversampling": XK_FRW}
    log("cross-fit translators:", json.dumps(info))
    for f in (0, 1):
        items = [i for i in fold if fold[i] == f and i in kind_ for _ in range(XK_FRW if (i in fr and kind_[i] == "keep") else 1)]
        if not items:
            log("fold", f, "has no rows (small pass)"); items = list(kind_)
        ck = torch.load(OD / "xb_control.pt", weights_only=False, map_location=DEV); T = XB.load_tr(B, "control")[0]; T.train()
        rr = np.random.default_rng(1841 + 10 * SEED + f); torch.manual_seed(1842 + 10 * SEED + f)
        order = np.concatenate([rr.permutation(len(items)) for _ in range(math.ceil(CF_STEPS * XB.ACC / len(items)) + 1)])
        opt = torch.optim.AdamW(T.parameters(), lr=CF_LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / CF_STEPS)))  # noqa: E731
        hist = {"learn": [], "keep": []}
        for step in range(CF_STEPS):
            for g_ in opt.param_groups:
                g_["lr"] = CF_LR * sched(step)
            opt.zero_grad(set_to_none=True)
            for a_ in range(XB.ACC):
                i = items[int(order[step * XB.ACC + a_])]; gids, qids, j, rstart, s0 = enc[i]; n = len(qids); o = B.look(gids, qids, j, rstart, teacher=False)
                d, g, aw = B.write(T, o["S"], s0, n, h=o["E"])
                try:
                    lsm = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
                finally:
                    B.P.slot.clear()
                y = torch.tensor(qids[rstart:], device=DEV); size = XB.SIZE * ((d ** 2).mean() + aw)
                if kind_[i] == "learn":                                    # as fit_cf
                    ce = -lsm.gather(-1, y[:, None])[:, 0].mean(); loss = ce + size; ag = o["la"].argmax(-1) == y
                    if XB.HARM > 0 and ag.any():
                        sv, si = o["la"][ag].topk(XB.TOPK, -1); loss = loss + XB.HARM * XB.tail_ce(lsm[ag], si, sv)
                    hist["learn"].append((float(ce.detach()), float(g.mean().detach())))
                else:                                                      # keep: Qwen alone's own distribution over the whole reply, the gate closed
                    kl = (o["la"].exp() * (o["la"] - lsm)).sum(-1).mean(); loss = XK_KW * kl + size + XK_GW * g.mean()
                    hist["keep"].append((float(kl.detach()), float(g.mean().detach())))
                (loss / XB.ACC).backward(); del o, lsm, d, g, aw, loss
            torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step()
            if step % 100 == 0 or step == CF_STEPS - 1:
                m_ = {k: [round(float(x), 4) for x in np.mean(v[-100:], 0)] if v else None for k, v in hist.items()}
                log(f"xk fold {f} step {step} of {CF_STEPS} | learn (ce, gate) {m_['learn']} | keep (kl, gate) {m_['keep']}", round(time.time() - t0), "s")
        T.eval(); ck["state"] = {k: v.detach().clone() for k, v in T.state_dict().items()}; ck["steps"] = ck["steps"] + CF_STEPS
        ck["xk"] = {"fold": f, "rows sha": kj["sha"], "steps": CF_STEPS, "lr": CF_LR, "keep KL weight": XK_KW, "gate penalty": XK_GW, "fragile oversampling": XK_FRW}; torch.save(ck, OD / f"{XKN}{f}.pt"); del T, opt
    TX = [XB.load_tr(B, "control", name=f"{XKN}{f}")[0] for f in (0, 1)]; byid = {r["id"]: r for r in keep + learn}
    Lr, Kr, Fr = [], [], []
    for i, f in fold.items():                                            # each row against the translator fitted on the other fold
        if i not in kind_:
            continue
        r = {"ctx": byid[i]["ctx"], "reply": byid[i]["reply"]}; T_ = TX[1 - f]
        if kind_[i] == "learn":
            v = _ce3(B, (T0, T_), r)
            if v:
                Lr.append(v)
        else:
            v = _kl3(B, (T0, T_), r)
            if v:
                Kr.append(v); Fr += [v] if i in fr else []
    held = {p["id"]: p for p in pools()[1]}; hs = [{"ctx": held[i]["prompt"], "reply": t} for i, t in base["e4b held solutions"].items() if i in held]; hs = hs[:3] if SMALL else hs
    H = np.array([v for v in (_ce3(B, (T0, TX[0], TX[1]), r) for r in hs) if v is not None])
    L_, K_, F_ = np.array(Lr), np.array(Kr), np.array(Fr)
    info["unseen learn rows (alone, old, cross-fit)"] = {"rows": len(L_), "means": [round(float(x), 4) for x in L_.mean(0)], "old - cross-fit": _boot(L_[:, 1] - L_[:, 2]), "alone - cross-fit": _boot(L_[:, 0] - L_[:, 2])} if len(L_) else {"rows": 0}
    info["unseen keep rows, KL from Qwen alone (old, cross-fit)"] = {"rows": len(K_), "means": [round(float(x), 4) for x in K_.mean(0)], "cross-fit - old": _boot(K_[:, 1] - K_[:, 0])} if len(K_) else {"rows": 0}
    info["unseen fragile keep rows, KL (old, cross-fit)"] = {"rows": len(F_), "means": [round(float(x), 4) for x in F_.mean(0)]} if len(F_) else {"rows": 0}
    info["held, E4B's passing solutions (alone, old, xk0, xk1)"] = {"rows": len(H), "means": [round(float(x), 4) for x in H.mean(0)]} if len(H) else {"rows": 0}
    lo = info["unseen learn rows (alone, old, cross-fit)"].get("old - cross-fit", [0.0, 0.0, 0.0]); ko = info["unseen keep rows, KL from Qwen alone (old, cross-fit)"].get("cross-fit - old", [0.0, 0.0, 1.0])
    info["gate"] = "XK_PASS" if lo[0] >= XK_MIN and lo[1] > 0 and ko[2] < 0.02 else "XK_FAIL"; info["fold"] = fold
    json.dump(info, open(XKJ, "w"), indent=1); log("cross-fit gate:", json.dumps({k: v for k, v in info.items() if k != "fold"}), round(time.time() - t0), "s"); print(info["gate"], flush=True)


def train_arm(arm):
    from peft import LoraConfig, get_peft_model
    assert arm in ARMS; kind = {"sft_s": "sft", "sft_c_s": "sft_c", "opd_cf": "opd_c", "rw_cf": "sft", "mix_cf": "mix_c", "rb": "sft", "sft_k": "sft", "sft_ck": "sft_c", "xk": "sft_c", "rb_c": "sft_c", "sft_self_s": "sft_self", "sft_b_s": "sft_b", "sft_es": "sft", "sft_lr": "sft", "sft_mt_s": "sft", "sft_self_mt_s": "sft_self", "sft_self_s_l03": "sft_self", "sft_self_s_l3": "sft_self"}.get(arm, arm)   # 179a short rows; 181 refit translator, rewrite rows; 182 runbook rows; 184 keep rows
    RP = ROWS_SHORT if arm in ("sft_s", "sft_c_s", "sft_self_s", "sft_b_s", "sft_mt_s", "sft_self_mt_s", "sft_self_s_l03", "sft_self_s_l3") else ROWS_RW if arm == "rw_cf" else ROWS_RB if arm in ("rb", "rb_c") else ROWS_K if arm in ("sft_k", "sft_ck", "xk") else ROWS; sub = RP in (ROWS_RW, ROWS_RB)
    B = XB.Bridge("bridge"); t0 = time.time(); rows = [json.loads(l) for l in open(RP)]; base = json.load(open(BASE))
    want = base["rows sha"] if RP == ROWS else json.load(open({ROWS_SHORT: SHORT, ROWS_RW: RW, ROWS_RB: RB, ROWS_K: KJ}[RP]))["sha"]
    same = ({r["id"] for r in rows} <= set(base["rows"]) and len(rows) > 0) if sub else {r["id"] for r in rows} == set(base["rows"])   # 181, 182: rewrite and runbook rows cover a subset
    assert same and hashlib.sha256(RP.read_bytes()).hexdigest() == want, "rows and baselines do not match"
    enc = [e for e in (XB.encode(B.tg, B.tq, {"ctx": r["ctx"], "reply": r["reply"]}) for r in rows) if e is not None and e[3] - 1 >= e[4]]
    assert len(enc) == len(rows), f"{len(rows) - len(enc)} rows failed to encode"
    src = {"sft_b": "b", "kl_b": "b", "sft_c": "c", "sft_self": "self", "opd_b": "b", "opd_c": "c", "mix_c": "c"}.get(kind); onp = kind in ("sdft", "opd_b", "opd_c", "mix_c")
    T, tsel = {}, None                                                   # 184: only the translator the arm uses (the 4090 has no bridge translators)
    if src == "b":
        T["b"] = XB.load_tr(B, "bridge")[0]
    if src == "c" and arm != "xk":
        T["c"] = XB.load_tr(B, "control", name=CFN if arm in ("opd_cf", "mix_cf") else None)[0]
    if arm == "xk":                                                      # 184: each row's target from the cross-fit translator that never saw it
        fd = json.load(open(XKJ))["fold"]; assert SMALL or all(r["id"] in fd for r in rows), "a row has no fold"
        TX_ = [XB.load_tr(B, "control", name=f"{XKN}{f}")[0] for f in (0, 1)]; tsel = [TX_[1 - fd.get(r["id"], 0)] for r in rows]
    MT = arm in ("sft_mt_s", "sft_self_mt_s"); mtl = []; nclip = 0                   # 190: + the corrector's own fitting text, directly (its split "train", its kind weights XB.W)
    if MT:
        mrows = [r for r in XB.rows_all(B.tg) if r["split"] == "train" and r["kind"] in XB.W]; mkinds = [k for k in XB.W if any(r["kind"] == k for r in mrows)]
        menc = {k: [e for e in (XB.encode(B.tg, B.tq, r) for r in mrows if r["kind"] == k) if e is not None and e[3] - 1 >= e[4]] for k in mkinds}
        mkinds = [k for k in mkinds if menc[k]]; mpw = np.array([XB.W[k] for k in mkinds]); mpw /= mpw.sum(); mrng = np.random.default_rng(190 + SEED)   # its own stream: the coding order is unchanged
        log("arm", arm, "| mix rows (the corrector's fitting split):", {k: len(menc[k]) for k in mkinds}, "| weights", dict(zip(mkinds, mpw.round(3).tolist())), "|", MT_N, "per step, weight", MT_W)
    dose = len(base["rows"]) if sub else len(enc)                        # 181, 182: rw_cf and rb take sft's steps over their fewer rows (they cycle)
    steps = 4 if SMALL else max(1, math.ceil(EPOCHS * dose / XB.ACC)); rng = np.random.default_rng(174 + SEED)
    order = np.concatenate([rng.permutation(len(enc)) for _ in range(math.ceil(steps * XB.ACC / len(enc)) + 1)])   # the same order for every arm of a seed
    lay = [B.WL] + B.sites; torch.manual_seed(161 + SEED)
    pm = get_peft_model(B.mS, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, target_modules=_lora_targets(B.mS, lay), bias="none"))
    params = [p_ for n_, p_ in B.mS.named_parameters() if "lora_" in n_]
    for p_ in params:
        p_.data = p_.data.float()
    if kind == "sdft":                                                   # the teacher's weights: an EMA of the student's LoRA, starting equal
        pm.add_adapter("ema", LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, target_modules=_lora_targets(B.mS, lay), bias="none"))
        named = dict(B.mS.named_parameters()); ema = {}
        for n_, p_ in named.items():
            if "lora_" in n_ and ".default." in n_:
                e_ = named[n_.replace(".default.", ".ema.")]; e_.data = p_.data.detach().clone().float(); e_.requires_grad_(False); ema[n_] = (p_, e_)
        pm.set_adapter("default")
        tctx = lambda r: (r["ctx"][:-len("```python\n")] + "This is an example for a response to the question:\n\n```python\n" + r["reply"][:-3] + "```\n\n"  # noqa: E731
                          "Now answer with a response of your own.\n\n```python\n")
        assert all(r["ctx"].endswith("```python\n") and r["reply"].endswith("```") for r in rows)
    if onp:
        eos_ = set(XC.stop_ids(B.mS, B.tq)); pad_ = B.tq.pad_token_id if B.tq.pad_token_id is not None else B.tq.eos_token_id; n_skip = 0
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / steps)))  # noqa: E731
    lam_ = {"sft_self_s_l03": 0.3, "sft_self_s_l3": 3.0}.get(arm, LAM)   # 193: the KL strength sweep (KL to the original student)
    log("arm", arm, "| rows", len(enc), "of", len(rows), "| steps", steps, "(epochs", EPOCHS, ") | lambda", lam_, "| layers", lay, "rank", RANK); hist = []
    STOP = float(os.environ.get("XL_STOP_CE", "0.24")) if arm == "sft_es" else None; LRX = 0.1 if arm == "sft_lr" else 1.0; ran = steps   # 188
    for step in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = LR * LRX * sched(step)
        opt.zero_grad(set_to_none=True); vals = []
        if onp:                                                          # 174c / 179: on-policy samples; the teacher: sdft's EMA in context, or the bridged / control-bridged base
            rs_ = [rows[int(order[step * XB.ACC + a_])] for a_ in range(XB.ACC)]; B.tq.padding_side = "left"; pm.eval()
            with torch.no_grad():
                en_ = B.tq([XB.SBOS + r["ctx"] for r in rs_], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
                yy = pm.generate(**en_, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=SD_MAXNEW, pad_token_id=pad_, eos_token_id=sorted(eos_), stop_strings=["```"], tokenizer=B.tq)[:, en_["input_ids"].shape[1]:].tolist()
            pm.train()
            for r, ys in zip(rs_, yy):
                k_ = next((i for i, t in enumerate(ys) if t in eos_ or t == pad_), len(ys)); ys = ys[:k_]
                ex_ = XB.encode(B.tg, B.tq, {"ctx": r["ctx"], "reply": r["reply"]}); ce_r = 0.0
                if ex_ is not None and kind == "mix_c":                  # 179b: the text's cross-entropy, with its gradient (kept even if the sample is skipped)
                    _, q_, _, rs0, _ = ex_; lg_ = B.mS(input_ids=torch.tensor([q_], device=DEV), logits_to_keep=len(q_) - rs0 + 1, use_cache=False).logits[0, :-1].float()
                    ce_t = -Fn.log_softmax(lg_, -1).gather(-1, torch.tensor(q_[rs0:], device=DEV)[:, None])[:, 0].mean(); (ce_t / XB.ACC).backward(); ce_r = float(ce_t.detach()); del lg_, ce_t
                elif ex_ is not None:                                    # logged only: the student's cross-entropy on E4B's solution
                    with torch.no_grad():
                        _, q_, _, rs0, _ = ex_; lg_ = B.mS(input_ids=torch.tensor([q_], device=DEV), logits_to_keep=len(q_) - rs0 + 1, use_cache=False).logits[0, :-1].float()
                        ce_r = float(-Fn.log_softmax(lg_, -1).gather(-1, torch.tensor(q_[rs0:], device=DEV)[:, None])[:, 0].mean()); del lg_
                if len(ys) <= SD_MASK + 1:
                    n_skip += 1; vals.append((ce_r, 0.0, 0.0, 0.0, 0.0)); continue
                if kind == "sdft":
                    ps_ = B.tq(XB.SBOS + r["ctx"], add_special_tokens=False)["input_ids"]; pt_ = B.tq(XB.SBOS + tctx(r), add_special_tokens=False)["input_ids"]; L_ = len(ys)
                    with torch.no_grad():
                        pm.set_adapter("ema"); lt = Fn.log_softmax(B.mS(input_ids=torch.tensor([pt_ + ys], device=DEV), logits_to_keep=L_ + 1, use_cache=False).logits[0, :-1].float(), -1); pm.set_adapter("default")
                    ls = Fn.log_softmax(B.mS(input_ids=torch.tensor([ps_ + ys], device=DEV), logits_to_keep=L_ + 1, use_cache=False).logits[0, :-1].float(), -1)
                else:                                                    # 179: the sample as text, aligned for both tokenizers as in training (xbridge.encode)
                    eo = XB.encode(B.tg, B.tq, {"ctx": r["ctx"], "reply": B.tq.decode(ys, skip_special_tokens=True)})
                    if eo is None or len(eo[1]) - eo[3] <= SD_MASK + 1:
                        n_skip += 1; vals.append((ce_r, 0.0, 0.0, 0.0, 0.0)); continue
                    g2, q2, j2, r2, s2 = eo; n2 = len(q2); x2 = torch.tensor([q2], device=DEV); ys = q2[r2:]
                    with pm.disable_adapter(), torch.no_grad():
                        o = B.look(g2, q2, j2, r2, teacher=src == "b"); B.write(T[src], o["T"] if src == "b" else o["S"], s2, n2)
                        try:
                            lt = Fn.log_softmax(B.mS(input_ids=x2, logits_to_keep=n2 - r2 + 1, use_cache=False).logits[0, :-1].float(), -1)
                        finally:
                            B.P.slot.clear()
                        del o
                    ls = Fn.log_softmax(B.mS(input_ids=x2, logits_to_keep=n2 - r2 + 1, use_cache=False).logits[0, :-1].float(), -1)
                klp = (lt.exp() * (lt - ls)).sum(-1)[SD_MASK:]; kl = klp.mean(); yt = torch.tensor(ys, device=DEV)
                (kl / XB.ACC).backward(); vals.append((ce_r, float(kl.detach()), 0.0, float(kl.detach()), float(-lt.gather(-1, yt[:, None])[:, 0].mean()))); del lt, ls, klp, kl
        for a_ in range(0 if onp else XB.ACC):
            k_ = int(order[step * XB.ACC + a_]); gids, qids, j, rstart, s0 = enc[k_]; n = len(qids); x = torch.tensor([qids], device=DEV); y = torch.tensor(qids[rstart:], device=DEV); lt = None
            if src == "self":                                            # 174b: the original Qwen itself (the LoRA off, no translator)
                with pm.disable_adapter(), torch.no_grad():
                    lt = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
            elif src:                                                    # the target: Qwen with the translator written over the span, the LoRA off
                with pm.disable_adapter(), torch.no_grad():
                    o = B.look(gids, qids, j, rstart, teacher=src == "b"); B.write(tsel[k_] if tsel else T[src], o["T"] if src == "b" else o["S"], s0, n)
                    try:
                        lt = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)   # positions s0..n-1
                    finally:
                        B.P.slot.clear()
                    del o
            ls = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
            ce = -ls[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean(); klp = (lt.exp() * (lt - ls)).sum(-1) if lt is not None else torch.zeros(n - s0, device=DEV); kl = klp.mean()
            ex = (float(klp[:rstart - 1 - s0].mean()) if rstart - 1 > s0 else 0.0, float(klp[rstart - 1 - s0:].mean()), float(-lt[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean()) if lt is not None else 0.0)
            loss = {"sft": ce, "sft_b": ce + lam_ * kl, "sft_c": ce + lam_ * kl, "kl_b": kl, "sft_self": ce + lam_ * kl}[kind]
            (loss / XB.ACC).backward(); vals.append((float(ce.detach()), float(kl.detach())) + ex); del ls, lt, loss, klp
        if MT:                                                           # 190: MT_N rows of the corrector's fitting text, cross-entropy on the reply, weight MT_W
            for _ in range(MT_N):
                k = mkinds[int(mrng.choice(len(mkinds), p=mpw))]; gm, qm, jm, rm, sm = menc[k][int(mrng.integers(len(menc[k])))]
                xm = torch.tensor([qm], device=DEV); ym = torch.tensor(qm[rm:], device=DEV)
                lm = Fn.log_softmax(B.mS(input_ids=xm, logits_to_keep=len(qm) - rm + 1, use_cache=False).logits[0, :-1].float(), -1)
                cem = -lm.gather(-1, ym[:, None])[:, 0].mean(); (MT_W * cem / MT_N).backward();   # 190a: the mix rows' mean, weight MT_W
                mtl.append(float(cem.detach())); del lm, cem
        gn_ = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); nclip += gn_ > 1.0; opt.step(); hist.append(np.mean(vals, 0))   # 190a: count clipped steps (the mix rows share the clip)
        if kind == "sdft":
            with torch.no_grad():
                for p_, e_ in ema.values():
                    e_.mul_(1 - SD_EMA).add_(p_.detach(), alpha=SD_EMA)
        if STOP is not None and step >= 49 and float(np.mean([h[0] for h in hist[-50:]])) <= STOP:   # 188: matched final loss (50-step mean, as Qwen's last50 0.24; a 10-step mean dips by chance)
            ran = step + 1; log(arm, "stops at step", step, "of", steps, "| running cross-entropy (50 steps)", round(float(np.mean([h[0] for h in hist[-50:]])), 4), "<=", STOP); break
        if step % 50 == 0 or step == steps - 1:
            h_ = np.mean(hist[-50:], 0); log(arm, "step", step, "of", steps, "| cross-entropy on E4B's solution", round(float(h_[0]), 4), "| KL to the target", round(float(h_[1]), 4), "(prompt", round(float(h_[2]), 4), "reply", round(float(h_[3]), 4), ") | target's own cross-entropy", round(float(h_[4]), 4), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "loss is not finite"
    d = OD / f"xl_{arm}"; pm.save_pretrained(str(d), selected_adapters=["default"])
    if onp:
        log(arm, ": samples too short to train on", n_skip)
    json.dump({"arm": arm, "seed": SEED, "rows": len(enc), "steps": steps, "steps run": ran, "lr scale": LRX, "mix rows ce first / last 200": [float(np.mean(mtl[:200])), float(np.mean(mtl[-200:]))] if mtl else None, "share of steps clipped": nclip / max(1, len(hist)), "stop ce": STOP, "last50 ce at stop": float(np.mean([h[0] for h in hist[-50:]])), "epochs": EPOCHS, "lambda": lam_, "rank": RANK, "layers": lay, "first50": [float(v) for v in np.mean(hist[:50], 0)],
               "last50": [float(v) for v in np.mean(hist[-50:], 0)]}, open(d / "train.json", "w"))
    if torch.cuda.is_available():
        log(arm, "peak GPU memory", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "GB")   # review 11-08 (the 24 GB 4090)
    print("TRAIN_DONE", arm, flush=True)


@torch.no_grad()
def offtask(B, pm, names):
    """next-token cross-entropy on the reply for Qwen alone and each adapter: X1's held agent rows, fresh questions (and p(true)), untaught prose
    (the same rows as xbridge.recon_eval)"""
    rows = XB.rows_all(B.tg); rng = np.random.default_rng(171); sets = {}
    lq = {a: sorted({B.tq.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}
    for k in ("agent", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; picked = [hk[i] for i in rng.permutation(len(hk))[:XB.NX1H if k != "question_fresh" else 2 * XB.NX1H]]
        if k != "question":
            sets[f"held {k}"] = picked
    hk = [r for r in rows if r["kind"] == "prose" and r["split"] == "held"]; sets["held prose (untaught)"] = [hk[i] for i in rng.permutation(len(hk))[:3 if SMALL else 30]]
    out = {}
    for name, rs in sets.items():
        C = {a: [] for a in ["qwen"] + names}; P = {a: [] for a in C}
        for r in rs:
            e = XB.encode(B.tg, B.tq, r)
            if e is None:
                continue
            gids, qids, j, rstart, s0 = e; n = len(qids); x = torch.tensor([qids], device=DEV); y = torch.tensor(qids[rstart:], device=DEV)
            for a in C:
                if a == "qwen":
                    with pm.disable_adapter():
                        lg = B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float()
                else:
                    pm.set_adapter(a); lg = B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float()
                L = Fn.log_softmax(lg, -1); C[a].append(float(-L.gather(-1, y[:, None])[:, 0].mean()))
                if r["kind"].startswith("question"):
                    p_ = L[0].exp(); raw = {q: float(sum(p_[t_] for t_ in ts)) for q, ts in lq.items()}; ms = sum(raw.values()); P[a].append(raw[r["true"]] / ms if ms else 0.0)
        out[name] = {"rows": len(C["qwen"]), "cross-entropy": {a: round(float(np.mean(v)), 4) for a, v in C.items()}}
        if any(P.values()):
            out[name]["p(true)"] = {a: round(float(np.mean(v)), 4) for a, v in P.items() if v}
        out[name]["per row"] = C
    return out


def regrade():
    """194 (user 11-08, from the Fable review: "a 1,536-cap re-grade of the lost set"): each stored adapter of this seed regenerates the problems
    it lost (solved by the student alone, failed by the adapter, both greedy at 768 new tokens) with XC_MAXNEW new tokens, greedy, and each answer
    is scored alone. Lost lists: XL_REGRADE (json {seed: {arm: [ids]}}); adapters: XL_REGRADE_DIR/xl_ARM. No training."""
    from peft import PeftModel
    XC.selfcheck(); tr, he = pools(); byid = {p["id"]: p for p in tr + he}; B = XB.Bridge("bridge"); t0 = time.time()
    lost = json.load(open(os.environ["XL_REGRADE"]))[str(SEED)]; pdir = Path(os.environ["XL_REGRADE_DIR"])
    out = {"seed": SEED, "max new tokens": XC.MAXNEW, "arms": {}}; pm = None
    with open(OD / f"xl_regrade_gens_s{SEED}.jsonl", "w") as fh:
        for k, (a, lid) in enumerate(lost.items()):
            if SMALL and k >= 2:
                break
            lid = [p_["id"] for p_ in (tr + he)[:2]] if SMALL else [i for i in lid if i in byid]   # the small pass: two problems of its small pools
            assert SMALL or len(lid) == len(lost[a]), f"{a}: lost problems missing from the pools"
            if pm is None:
                pm = PeftModel.from_pretrained(B.mS, str(pdir / f"xl_{a}"), adapter_name=a)
            else:
                pm.load_adapter(str(pdir / f"xl_{a}"), adapter_name=a)
            pm.set_adapter(a); pm.eval(); g = XC.gen_hf(pm, B.tq, [byid[i]["prompt"] for i in lid], bos=XB.SBOS) if lid else []
            ok = [bool(XC.run_problem(byid[i], t)) for i, t in zip(lid, g)]; nt = [len(B.tq(t, add_special_tokens=False)["input_ids"]) for t in g]
            for i, t, o in zip(lid, g, ok):
                fh.write(json.dumps({"id": i, "arm": a, "text": t, "pass": o}) + "\n")
            out["arms"][a] = {"lost": len(lid), "recovered": int(sum(ok)), "recovered ids": [i for i, o in zip(lid, ok) if o], "new tokens": nt,
                              "longer than 768 tokens": int(sum(1 for n in nt if n > 768)), "recovered and longer than 768": int(sum(1 for n, o in zip(nt, ok) if o and n > 768))}
            log(a, "lost", len(lid), "| recovered with", XC.MAXNEW, "tokens:", int(sum(ok)), "| answers over 768 tokens:", out["arms"][a]["longer than 768 tokens"], round(time.time() - t0), "s")
    json.dump(out, open(OD / f"xl_regrade_s{SEED}.json", "w"), indent=1); print("REGRADE_DONE", flush=True)


def evaluate():
    """generation on the problems where a count can move: the trained ones, the held ones, and the untrained train-pool problems Qwen passed (for
    kept); review 11-07: problems where an arm and Qwen alone disagree are re-scored one at a time (no parallel load near the time limit)"""
    from peft import PeftModel
    XC.selfcheck(); tr, he = pools(); base = json.load(open(BASE)); rows = set(base["rows"]); src = base["row source"]; B = XB.Bridge("bridge"); t0 = time.time()
    trained = [p["id"] for p in tr if p["id"] in rows]; held = [p["id"] for p in he]
    kept_pool = [] if os.environ.get("XL_NO_KEPT_POOL") == "1" else [p["id"] for p in tr if p["id"] not in rows and base["qwen"].get(p["id"])]   # 186: another student has no build run of its own
    byid = {p["id"]: p for p in tr + he}; ev = [byid[i] for i in trained + kept_pool + held]; ids = [p["id"] for p in ev]
    names = [a for a in ARMS if (OD / f"xl_{a}").is_dir()]; pm = None
    for a in names:
        if pm is None:
            pm = PeftModel.from_pretrained(B.mS, str(OD / f"xl_{a}"), adapter_name=a)
        else:
            pm.load_adapter(str(OD / f"xl_{a}"), adapter_name=a)
    pm.eval(); res, txt = {}, {}; prompts = [p["prompt"] for p in ev]; log("evaluating", len(ev), "problems: trained", len(trained), "| untrained Qwen passed", len(kept_pool), "| held", len(held))

    def run(a, g):
        txt[a] = dict(zip(ids, g)); res[a] = dict(zip(ids, score(ev, g)))
        with open(OD / f"xl_gens_{a}.jsonl", "w") as fh:
            for i, t in zip(ids, g):
                fh.write(json.dumps({"id": i, "text": t, "pass": res[a][i]}) + "\n")
    with pm.disable_adapter():                                           # the student alone in this session (key "qwen"), the baseline of every count
        run("qwen", XC.gen_hf(pm, B.tq, prompts, bos=XB.SBOS))
    for a in names:
        pm.set_adapter(a); run(a, XC.gen_hf(pm, B.tq, prompts, bos=XB.SBOS)); log(a, "passed: trained", sum(res[a][i] for i in trained), "| held", sum(res[a][i] for i in held), round(time.time() - t0), "s")
    serial = {}
    for a in names:                                                      # re-score the disagreements one at a time
        for i in ids:
            if res[a][i] != res["qwen"][i]:
                for b in (a, "qwen"):
                    if (b, i) not in serial:
                        serial[(b, i)] = XC.run_problem(byid[i], txt[b][i]); res[b][i] = serial[(b, i)]
    log("re-scored serially:", len(serial), "problem results")
    q = res["qwen"]; gap = [i for i in held if base["e4b greedy"].get(i) and not q[i]]; rwids = {json.loads(l)["id"] for l in open(ROWS_RW)} if ROWS_RW.exists() else set()
    rbids = {json.loads(l)["id"] for l in open(ROWS_RB)} if ROWS_RB.exists() and ("rb" in names or "rb_c" in names) else set()
    fr_ = json.load(open(FRAG)) if FRAG.exists() else None
    out = {"seed": SEED, "problems": {"trained": len(trained), "untrained, Qwen passed": len(kept_pool), "held": len(held), "gap (held, E4B passes, Qwen fails)": len(gap)},
           "Qwen alone passes": {"trained": sum(q[i] for i in trained), "held": sum(q[i] for i in held)},
           "agreement with the build's Qwen run": round(float(np.mean([q[i] == base["qwen"][i] for i in ids])), 4), "arms": {}}
    for a in names:
        r_ = res[a]; fx = lambda c: [sum(1 for i in trained if c(i) and not q[i] and r_[i]), sum(1 for i in trained if c(i) and not q[i])]  # noqa: E731
        out["arms"][a] = {
            "1 fixed: trained problems Qwen alone failed, now passes": fx(lambda i: True),
            "1 by row source (greedy / sample)": {k: fx(lambda i, k=k: src[i] == k) for k in ("greedy", "sample")},
            "1 by difficulty": {k: fx(lambda i, k=k: byid[i]["difficulty"] == k) for k in ("easy", "medium", "hard")},
            "2 new: held problems Qwen alone failed, now passes": [sum(1 for i in held if not q[i] and r_[i]), sum(1 for i in held if not q[i])],
            "2b of those on the gap set (E4B passes)": [sum(1 for i in gap if r_[i]), len(gap)],
            "3 kept, held: Qwen alone passed, still passes": [sum(1 for i in held if q[i] and r_[i]), sum(1 for i in held if q[i])],
            "3 kept, train pool (trained + untrained)": [sum(1 for i in trained + kept_pool if q[i] and r_[i]), sum(1 for i in trained + kept_pool if q[i])],
            "paired vs Qwen alone (arm only, Qwen only, exact p)": {"trained": XC.mcnemar(r_, q, trained), "held": XC.mcnemar(r_, q, held)}}
        if rwids:                                                        # review 11-07 (181): rw_cf trains only the problems with a passing rewrite
            out["arms"][a]["1 fixed, by rewrite row (has one / none)"] = [fx(lambda i: i in rwids), fx(lambda i: i not in rwids)]
        if fr_:                                                          # 184: the fragile set (Qwen passed, lost in >= 6 of 24 earlier runs), lost / Qwen passed
            out["arms"][a]["3c lost on the fragile set (train pool, held)"] = [[sum(1 for i in fr_[k] if q.get(i) and not r_.get(i, True)), sum(1 for i in fr_[k] if q.get(i))] for k in ("train", "held")]
        if rbids:                                                        # 182: rb trains only the problems with a passing runbook solution
            out["arms"][a]["1 fixed, by runbook row (has one / none)"] = [fx(lambda i: i in rbids), fx(lambda i: i not in rbids)]
    pairs = [("sft_self_s_l03", "sft"), ("sft_self_s_l3", "sft"), ("sft_self_s_l3", "sft_self_s_l03"), ("sft_mt_s", "sft"), ("sft_self_mt_s", "sft"), ("sft_self_mt_s", "sft_mt_s"), ("sft_b", "sft_c"), ("sft_b", "sft"), ("sft_c", "sft"), ("kl_b", "sft"), ("sft_c", "sft_self"), ("sft_self", "sft"), ("sdft", "sft"), ("sft_c", "sdft"), ("sdft", "sft_self"),
             ("opd_b", "opd_c"), ("opd_b", "sft"), ("opd_c", "sft"), ("opd_b", "sdft"), ("opd_c", "sdft"), ("opd_b", "sft_b"), ("opd_c", "sft_c"), ("mix_c", "sft"), ("mix_c", "opd_c"), ("mix_c", "sft_c"), ("sft_s", "sft"), ("sft_c_s", "sft_c"),
             ("opd_cf", "opd_c"), ("opd_cf", "sft"), ("opd_cf", "mix_c"), ("rw_cf", "sft"), ("rw_cf", "opd_cf"), ("rw_cf", "opd_c"), ("mix_cf", "mix_c"), ("mix_cf", "sft"), ("mix_cf", "opd_cf"),
             ("rb", "sft"), ("rb", "sft_s"), ("rb", "rw_cf"), ("rb", "mix_cf"), ("rb", "opd_cf"),
             ("sft_c_s", "sft_s"), ("sft_k", "sft_s"), ("sft_ck", "sft_c_s"), ("sft_ck", "sft_k"), ("xk", "sft_ck"), ("xk", "sft_k"), ("xk", "sft_s"), ("rb_c", "rb"), ("rb_c", "sft_c_s"), ("rb_c", "sft_s"), ("rb_c", "sft_ck"), ("sft_self_s", "sft_s"), ("sft_self_s", "sft_c_s"), ("sft_b_s", "sft_c_s"), ("sft_b_s", "sft_s"), ("sft_b_s", "sft_self_s"), ("sft_c", "sft_self"), ("sft_b", "sft_self")]
    out["paired between arms (a only, b only, exact p)"] = {f"{a} vs {b}": {"trained": XC.mcnemar(res[a], res[b], trained), "held": XC.mcnemar(res[a], res[b], held)} for a, b in pairs if a in res and b in res}
    hs = [(i, t) for i, t in base["e4b held solutions"].items() if i in byid]; C = {a: [] for a in ["qwen"] + names}   # review: a continuous held measure
    with torch.no_grad():
        for i, t in hs:
            e = XB.encode(B.tg, B.tq, {"ctx": byid[i]["prompt"], "reply": t})
            if e is None:
                continue
            _, qids, _, rstart, _ = e; n = len(qids); x = torch.tensor([qids], device=DEV); y = torch.tensor(qids[rstart:], device=DEV)
            for a in C:
                if a == "qwen":
                    with pm.disable_adapter():
                        lg = B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float()
                else:
                    pm.set_adapter(a); lg = B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float()
                C[a].append(float(-Fn.log_softmax(lg, -1).gather(-1, y[:, None])[:, 0].mean()))
    out["held: cross-entropy of E4B's passing solutions (lower = closer to E4B)"] = {"rows": len(C["qwen"]), **{a: round(float(np.mean(v)), 4) for a, v in C.items() if v}}
    ot = offtask(B, pm, names); out["off-task (kept knowledge)"] = {k: {kk: vv for kk, vv in v.items() if kk != "per row"} for k, v in ot.items()}
    json.dump({**out, "per problem": res, "held CE per row": C, "off-task per row": {k: v["per row"] for k, v in ot.items()}}, open(OD / f"xl_eval_s{SEED}.json", "w"))
    log(json.dumps(out)); print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    {"build": build, "build_short": build_short, "precheck": precheck, "train": lambda: train_arm(sys.argv[2]), "evaluate": evaluate, "regrade": regrade, "fit_cf": fit_cf, "rewrite": rewrite, "runbook": runbook, "build_keep": build_keep, "fit_xk": fit_xk, "build_from_gens": build_from_gens}[sys.argv[1]]()
