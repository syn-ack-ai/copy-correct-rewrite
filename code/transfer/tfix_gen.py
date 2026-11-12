"""Amendment 209q (2026-11-11): what the corrected teacher writes. 209o's corrector (E4B's own states read, E4B written into, fitted on the
clean text) lowered the stub's probability on new problems; here E4B answers held problems with that corrector on, and the answers are run
on the tests. Generation with the corrector needs two passes per token (E4B's own states first, then the pass with the correction written,
as online.control_lp), so it is a plain greedy loop without a cache, one problem at a time.
  rows   (the evaluation's environment, no model)  GEN_DIR/gen_items.json [{id, was}] -> GEN_DIR/gen_ids.json (adds ctx: "<bos>" + the prompt's ids)
  gen    (the teaching environment with TR_STUDENT=e4b TR_TEACHERS=e4b, TR_CONTROL=the corrector)  -> GEN_DIR/gen_out.jsonl {id, was, text, steps,
         seconds}; the first GEN_CHECK problems are also answered with the corrector off, to see that the loop alone writes E4B's stored answer
  score  (the evaluation's environment)  runs every answer on its problem's tests -> GEN_DIR/gen_scored.jsonl (adds pass, fenced)
GEN_SMALL=1: two problems, 24 tokens."""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
GD = Path(os.environ["GEN_DIR"]); SMALL = os.environ.get("GEN_SMALL") == "1"; NEW = 24 if SMALL else int(os.environ.get("GEN_NEW", "768")); NCHK = int(os.environ.get("GEN_CHECK", "3"))
ALONE = os.environ.get("GEN_ALONE") == "1"   # 209q's control: the same loop with the corrector off on every problem (the stored answers were written with a cache; the loop's own text differs from them)
TAG = ("_alone" if ALONE else "") + ("_small" if SMALL else "")


def rows():
    from transformers import AutoTokenizer
    import xlearn as XL
    tok = AutoTokenizer.from_pretrained(os.environ["XB_STUDENT"]); tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; out = []
    for it in json.load(open(GD / "gen_items.json")):
        c = tok("<bos>" + byid[it["id"]]["prompt"], add_special_tokens=False)["input_ids"]; assert c[0] == tok.bos_token_id and tok.bos_token_id not in c[1:], it["id"]
        out.append({**it, "ctx": c})
    json.dump(out, open(GD / "gen_ids.json", "w")); print("GEN_ROWS_OK", len(out), "problems | longest prompt", max(len(o["ctx"]) for o in out), "tokens", flush=True)


def gen():
    import torch
    import online as ON, translator as X
    assert ON.REPLY and ON.ST == ON.TT == "e4b", "the teacher's own corrector: TR_STUDENT=e4b TR_TEACHERS=e4b ON_REPLY=1"
    items = json.load(open(GD / "gen_ids.json")); items = items[:2] if SMALL else items; DEV = X.DEV
    mT, mS, gT, gS = ON.load_models(); del mT, gT; mS.eval(); tok = X.tok_()
    eos = set(ON.eos_of(mS)) | {tok.eos_token_id}                         # the end tokens of the stored answers' generation (xcode.stop_ids)
    for t_ in ("<|endoftext|>", "<|im_end|>", "<end_of_turn>", "<eos>"):
        i_ = tok.convert_tokens_to_ids(t_)
        if isinstance(i_, int) and i_ != tok.unk_token_id:
            eos.add(i_)
    eos.discard(None)
    Tc = ON.control_setup(); P = X.Patch(mS, Tc.sites); g = ON.Grab(mS, X.SRC[ON.ST]); ON.log("corrector", os.environ["TR_CONTROL"], "| problems", len(items), "| new tokens at most", NEW)

    @torch.no_grad()
    def answer(ctx, corrected):
        x = torch.tensor([ctx], device=DEV); c = len(ctx); gen_ = []; s0 = max(1, c - ON.SPAN)
        for _ in range(NEW):
            n_ = x.shape[1]
            if corrected:                                                # E4B's own source states over [s0, n), then the pass with the correction written (control_lp's two passes)
                g.sl = slice(s0, n_); g.out = {}; mS(input_ids=x, logits_to_keep=1, use_cache=False); g.sl = None
                src = torch.cat([g.out[l] for l in X.SRC[ON.ST]], -1); g.out = {}; ON.write(P, Tc, src, s0, n_, n_)
            try:
                t = int(mS(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1].argmax())
            finally:
                P.slot.clear()
            if t in eos:
                break
            gen_.append(t); x = torch.cat([x, torch.tensor([[t]], device=DEV)], 1)
            if "```" in tok.decode(gen_[-6:], skip_special_tokens=True):
                break
        return tok.decode(gen_, skip_special_tokens=True), len(gen_)

    tag = TAG; out = []
    with open(GD / f"gen_out{tag}.jsonl", "w") as fh:
        for k, o in enumerate(items):
            t0 = time.time(); text, n = answer(o["ctx"], not ALONE); r = {"id": o["id"], "was": o["was"], "text": text, "steps": n, "seconds": round(time.time() - t0, 1), "corrector": not ALONE}
            if k < NCHK and not ALONE:                                   # the loop with the corrector off: E4B alone, to set beside its stored answer
                r["alone text"], r["alone steps"] = answer(o["ctx"], False)
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if k % 5 == 0:
                ON.log("answered", k + 1, "of", len(items), "| last:", n, "tokens in", r["seconds"], "s")
    print("GEN_DONE", len(out), "answers", flush=True)


def score():
    import xlearn as XL
    XC = XL.XC; tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; XC.selfcheck(); tag = TAG
    R = [json.loads(l) for l in open(GD / f"gen_out{tag}.jsonl")]
    ok = XL.score([byid[r["id"]] for r in R], [r["text"] for r in R]); ok = [o or bool(XC.run_problem(byid[r["id"]], r["text"])) for r, o in zip(R, ok)]   # a failure is tried once more alone
    alone = {r["id"]: bool(XC.run_problem(byid[r["id"]], r["alone text"])) for r in R if "alone text" in r}
    with open(GD / f"gen_scored{tag}.jsonl", "w") as fh:
        for r, o in zip(R, ok):
            fh.write(json.dumps({**r, "pass": bool(o), "fenced": "```" in r["text"], **({"alone pass": alone[r["id"]]} if r["id"] in alone else {})}) + "\n")
    by = {}
    for r, o in zip(R, ok):
        b = by.setdefault(r["was"], [0, 0, 0]); b[0] += 1; b[1] += bool(o); b[2] += "```" in r["text"]
    print("GEN_SCORED", "(the corrector off)" if ALONE else "", {k: {"problems": v[0], "pass": v[1], "finished": v[2]} for k, v in by.items()}, "| the loop alone passes", sum(alone.values()), "of", len(alone), flush=True)


if __name__ == "__main__":
    {"rows": rows, "gen": gen, "score": score}[sys.argv[1]]()
