"""Amendment 209l (user 11-11: "if possible we should try to somehow fix the bad habbits of thea teacher also? ... is there a way of using the
probe to 'fix' this bad behavour?"; "yes, try option 1 first"): the direction in E4B's states that says "the stub comes next", found so that it
can be erased from what the cross-model corrector reads (translator.Tr.set_erase; the corrector is then fitted again on the erased states).
  texts  (the Mac, no model; standard library)  PROBE_DIR/probe_texts.json from E4B's stored greedy answers (GENS, runs/xqlB/xl_build_gens.jsonl):
         every LeetCode-format problem whose answer has a one-line signature gets the same text, "from typing import List\\n\\nclass Solution:\\n
         <its signature>\\n        # ... implementation\\n```" (stub_probe.py's format); split = held (the 188 held problems) or fit (the train pool)
  rows   is stub_probe.py rows with this PROBE_DIR (prompt ids, text ids, the first body token)
  find   (the teaching environment, online.py's models and hooks) For each problem E4B reads the prompt and the text up to the signature. The
         states of its source layers (X.SRC, the corrector's input) at the last position are whitened with the corrector's own statistics
         (DIR_STATS, a fitted cross-model corrector). The label is what E4B does next: its greedy continuation of 12 tokens starts the stub (a
         comment that begins with "...", or a bare pass or ...). The direction is the difference of the two class means on the fit problems,
         unit length; erasing it sets the component along it to the no-stub class's mean, after which the two fit means are equal. Checked on
         the held problems: the AUC of the direction, and of a ridge probe fitted on the fit problems before and after the erasure (what a
         linear reader that uses the covariance still finds). Then the cause is tested on E4B
         itself: with each source layer's part of the direction erased in E4B's own residual stream, how many held stub problems still
         continue with the stub, against a random direction erased the same way.
  Output PROBE_DIR/erase.pt {U (1 x din), a (1), mx, sx, layers} and PROBE_DIR/stub_dir.json (the small pass: *_small)."""
import json, os, re, sys, time
from pathlib import Path
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
PD = Path(os.environ["PROBE_DIR"]); SMALL = os.environ.get("PROBE_SMALL") == "1"
SIG = re.compile(r"^class Solution:\n(?:    #[^\n]*\n|[ \t]*\n)*(    def \w+\(self[^\n]*\)[^\n]*:)[ \t]*\n", re.M)
BODY = "        # ... implementation\n```"


def is_stub(text):
    """the continuation starts the stub: its first non-empty line is a comment that begins with an ellipsis, or a bare pass or ..."""
    first = next((l.strip() for l in text.split("\n") if l.strip()), "")
    return bool(re.match(r"#\s*(\.\.\.|…)", first)) or first in ("pass", "...")


def texts():
    gens, base = sys.argv[2], json.load(open(sys.argv[3])); held = set(base["held"]); out = []; n = {"not LeetCode format or no one-line signature": 0}
    def stub_answer(r):                                                  # dreamer/lcb_failkinds.py's rule: finished, and the last code line is a def with no body
        t = r["text"].rstrip(); fin = t.endswith("```"); L = [l for l in (t[:-3] if fin else t).split("\n") if l.strip() and not l.strip().startswith("#")]
        return fin and bool(L) and L[-1].strip().startswith("def ") and L[-1].rstrip().endswith(":")
    for r in map(json.loads, open(gens)):
        if r["model"] != "e4b" or r["kind"] != "greedy":
            continue
        m = SIG.search(r["text"])
        if not m:
            n["not LeetCode format or no one-line signature"] += 1; continue
        pre = "from typing import List\n\nclass Solution:\n" + m.group(1) + "\n"
        out.append({"id": r["id"], "split": "held" if r["id"] in held else "fit", "ref_stub": False, "alone_stub": False, "alone_pass": False, "e4b_stub": stub_answer(r), "text": pre + BODY, "body_char": len(pre)})
    PD.mkdir(parents=True, exist_ok=True); json.dump(out, open(PD / "probe_texts.json", "w"))
    c = {}
    for o in out:
        k = f"{o['split']}, E4B's own greedy answer {'a stub' if o['e4b_stub'] else 'not a stub'}"; c[k] = c.get(k, 0) + 1
    print(json.dumps({"problems": len(out), **c, **n}))


def auc(s, y):
    p = [a for a, b in zip(s, y) if b]; q = [a for a, b in zip(s, y) if not b]
    return float("nan") if not p or not q else sum((a > b) + 0.5 * (a == b) for a in p for b in q) / (len(p) * len(q))


def find():
    import torch
    import online as ON, translator as X
    items = json.load(open(PD / "probe_ids.json")); DEV = X.DEV; t0 = time.time()
    if SMALL:
        items = [o for o in items if o["e4b_stub"] and o["split"] == "fit"][:4] + [o for o in items if not o["e4b_stub"] and o["split"] == "fit"][:4] + [o for o in items if o["e4b_stub"] and o["split"] == "held"][:3] + [o for o in items if not o["e4b_stub"] and o["split"] == "held"][:3]
    mT, mS, gT, gS = ON.load_models(); mT.eval(); del mS, gS; tok = X.tok_(); lay = X.SRC[ON.TT]; W = X.WIDTH[ON.TT]
    ck = torch.load(os.environ["DIR_STATS"], weights_only=False, map_location="cpu"); mx, sx = ck["state"]["mx"].float().to(DEV), ck["state"]["sx"].float().to(DEV)
    assert mx.shape[0] == W * len(lay) == ck["din"] and ck.get("src depths") == os.environ.get("TR_SRC_DEPTHS"), "the statistics are not this configuration's corrector's"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def nxt(o):
        """E4B's greedy continuation (12 tokens) after the prompt and the text up to the signature"""
        x = torch.tensor([o["ctx"] + o["rep"][:o["body_tok"]]], device=DEV); n = x.shape[1]
        g = mT.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=12, use_cache=True, pad_token_id=pad)
        return tok.decode(g[0, n:], skip_special_tokens=True)

    @torch.no_grad()
    def state(o):
        x = torch.tensor([o["ctx"] + o["rep"][:o["body_tok"]]], device=DEV); n = x.shape[1]
        gT.sl = slice(n - 1, n); gT.out = {}; mT(input_ids=x, logits_to_keep=1, use_cache=False); gT.sl = None
        src = torch.cat([gT.out[l] for l in lay], -1)[0]; gT.out = {}; assert src.shape == mx.shape, src.shape
        return ((src - mx) / sx).cpu()

    Z, y, cont = [], [], []
    for k, o in enumerate(items):
        Z.append(state(o)); c = nxt(o); cont.append(c); y.append(is_stub(c))
        if k % 50 == 0:
            ON.log("states", k, "of", len(items), round(time.time() - t0), "s")
    Z = torch.stack(Z); y = torch.tensor(y); fit = torch.tensor([o["split"] == "fit" for o in items]); he = ~fit
    agree = sum(int(a == o["e4b_stub"]) for a, o in zip(y.tolist(), items))
    ON.log("labels: the continuation starts the stub on", int(y.sum()), "of", len(items), "| fit", int((y & fit).sum()), "of", int(fit.sum()), "| held", int((y & he).sum()), "of", int(he.sum()), "| agrees with E4B's stored greedy answer on", agree)
    if SMALL and (int((y & fit).sum()) < 1 or int((~y & fit).sum()) < 1):   # the small pass only exercises the path: with one class missing among its few problems it takes E4B's stored labels
        y = torch.tensor([o["e4b_stub"] for o in items]); ON.log("small pass: a class is missing among the continuations; the stored labels are used")
    assert SMALL or (int((y & fit).sum()) >= 10 and int((~y & fit).sum()) >= 10), "too few fit problems of one class"

    def dom(Zf, yf):
        d = Zf[yf].mean(0) - Zf[~yf].mean(0); u = d / d.norm(); return u, float(Zf[~yf].mean(0) @ u), float(d.norm())

    u, a, gap = dom(Z[fit], y[fit]); proj = Z @ u; Ze = Z - (proj - a)[:, None] * u
    rest = float((Ze[fit][y[fit]].mean(0) - Ze[fit][~y[fit]].mean(0)).norm())

    def ridge_auc(A):
        """review 209l: a dual ridge probe (labels centred) fitted on the fit problems of A, read on the held ones: what is left for a linear reader
        that uses the covariance (a second difference of means would be rounding error: the erasure makes the fit means equal)"""
        m = A[fit].mean(0); Zc = (A[fit] - m).double(); t = (y[fit].double() - y[fit].double().mean()); Kk = Zc @ Zc.T
        w = Zc.T @ torch.linalg.solve(Kk + 0.1 * Kk.diagonal().mean() * torch.eye(len(Kk), dtype=torch.float64), t)
        return round(auc(((A[he] - m).double() @ w).tolist(), y[he].tolist()), 3)
    hy = y[he].tolist(); res = {"problems": len(items), "fit problems (stub next / not)": [int((y & fit).sum()), int((~y & fit).sum())], "held problems (stub next / not)": [int((y & he).sum()), int((~y & he).sum())],
           "labels agree with E4B's stored greedy answer": agree, "source layers": lay, "width per layer": W,
           "class-mean gap along the direction (whitened units)": round(gap, 3), "share of the direction's energy per source layer": [round(float((u[k * W:(k + 1) * W] ** 2).sum()), 3) for k in range(len(lay))],
           "projection, fit (mean stub next / not; sd of not)": [round(float(proj[fit][y[fit]].mean()), 3), round(a, 3), round(float(proj[fit][~y[fit]].std()), 3)],
           "AUC of the direction: fit, held": [round(auc(proj[fit].tolist(), y[fit].tolist()), 3), round(auc(proj[he].tolist(), hy), 3)],
           "after the erasure: class-mean gap left on the fit problems": round(rest, 5), "held AUC of a ridge probe fitted on the fit problems: before, after the erasure": [ridge_auc(Z), ridge_auc(Ze)],
           "continuations": [{"id": o["id"], "split": o["split"], "stub next": bool(b), "e4b_stub": o["e4b_stub"], "text": c} for o, b, c in zip(items, y.tolist(), cont)]}
    tag = "_small" if SMALL else ""
    torch.save({"U": u[None].clone(), "a": torch.tensor([a]), "mx": mx.cpu(), "sx": sx.cpu(), "layers": lay}, PD / f"erase{tag}.pt"); json.dump(res, open(PD / f"stub_dir{tag}.json", "w"))
    ON.log("direction:", {k_: v_ for k_, v_ in res.items() if k_ != "continuations"})
    # the cause, on E4B itself: each source layer's part of the direction erased in its own residual stream (all positions)
    L = mT.model.language_model.layers; cur = {"dirs": None}; g = torch.Generator().manual_seed(209); r = torch.randn(u.shape, generator=g); r = r / r.norm()

    def parts(v):                                                        # per layer: the unit part of v, and the no-stub fit mean of the projection on it
        out = []
        for k in range(len(lay)):
            vk = v[k * W:(k + 1) * W]; vk = vk / vk.norm(); out.append((vk.to(DEV), float((Z[fit][~y[fit]][:, k * W:(k + 1) * W] @ vk).mean()), mx[k * W:(k + 1) * W], sx[k * W:(k + 1) * W]))
        return out

    def mk(k):
        def hook(_m, _i, out):
            if cur["dirs"] is None:
                return out
            vk, ak, m_, s_ = cur["dirs"][k]; h = out[0] if isinstance(out, tuple) else out; hf = h.float()
            hn = (hf - s_ * ((((hf - m_) / s_) @ vk - ak)[..., None] * vk)).to(h.dtype)
            return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn
        return hook

    hs = [L[l].register_forward_hook(mk(k)) for k, l in enumerate(lay)]; hi = [o for o in items if o["split"] == "held"]; lab = {o["id"]: b for o, b in zip(items, y.tolist())}; cause = {}
    try:
        for name, v in (("the stub direction erased", u), ("a random direction erased", r)):
            cur["dirs"] = parts(v); c2 = [nxt(o) for o in hi]; cur["dirs"] = None
            cause[name] = {"held problems that continued with the stub: still do": sum(int(is_stub(c)) for o, c in zip(hi, c2) if lab[o["id"]]), "of": sum(int(lab[o["id"]]) for o in hi),
                           "held problems that did not: now do": sum(int(is_stub(c)) for o, c in zip(hi, c2) if not lab[o["id"]]), "of those": sum(int(not lab[o["id"]]) for o in hi),
                           "continuations of the former": [{"id": o["id"], "text": c} for o, c in zip(hi, c2) if lab[o["id"]]]}
            ON.log("cause:", name, {k_: v_ for k_, v_ in cause[name].items() if not k_.startswith("continuations")})
        chk = nxt(hi[0]); assert chk == cont[items.index(hi[0])], "the hooks are off and E4B does not answer as before"
    finally:
        for h in hs:
            h.remove()
    res["E4B itself"] = cause; json.dump(res, open(PD / f"stub_dir{tag}.json", "w")); print("STUB_DIR_DONE", len(items), "problems", flush=True)


if __name__ == "__main__":
    {"texts": texts, "find": find}[sys.argv[1]]()
