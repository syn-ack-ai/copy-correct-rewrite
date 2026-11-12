"""Amendment 209m (2026-11-11): what follows the stub? 209l's direction session showed E4B closing the block after "# ... implementation ..." and
at once opening a new one; the evaluation stops at the first closing fence and so never sees it. Here each writer is given the problem's
prompt and its own stored stub answer (to the closing fence) and goes on greedily, up to STUB_NEW new tokens, with no stop at a fence. If a
second fenced block follows, its code is run on the problem's tests. Writers: e4b and e2b (the models alone) and, for STUB_LORAS=name=dir,..,
E2B with that student's adapter. No training.
  XB_DIR=... XB_TEACHER=E4B XB_STUDENT=E2B XC_DATA=... python stub_past.py ITEMS.jsonl OUT.jsonl      (STUB_SMALL=1: two items per writer)
ITEMS: {writer, id, text} (text = the stored answer, ending with its fence). OUT adds: more (the continuation), block (the second block's code or
None), closed (it has its own closing fence), pass."""
import json, os, re, sys
from pathlib import Path
import torch

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xlearn as XL  # noqa: E402
XC, XB = XL.XC, XL.XB
SMALL = os.environ.get("STUB_SMALL") == "1"; NEW = int(os.environ.get("STUB_NEW", "768"))
OPEN = re.compile(r"```[^\n`]*\n"); REUSE = re.compile(r"(?:python|py)3?[ \t]*\n")   # review 209m: any info string opens a block; a continuation that starts with "python\n" reuses the stub's closing fence as its opening one


def second_block(more):
    """the first fenced block of the continuation: (code, closed) or (None, False)"""
    m = REUSE.match(more) or OPEN.search(more)
    if not m:
        return None, False
    rest = more[m.end():]; k = rest.find("```")
    return (rest[:k], True) if k >= 0 else (rest, False)


@torch.no_grad()
def go_on(model, tok, prompts, texts, bos="<bos>"):
    """greedy, batched, with no stop at a fence; the input is the prompt's ids followed by the stored answer's ids, tokenized apart as when the
    answer was generated (lcb_fix.cont_sample), left-padded"""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; eos = XC.stop_ids(model, tok); outs = []; B = max(1, XC.BATCH // 2)
    for a in range(0, len(prompts), B):
        seqs = [tok(bos + p, add_special_tokens=False)["input_ids"] + tok(x, add_special_tokens=False)["input_ids"] for p, x in zip(prompts[a:a + B], texts[a:a + B])]; L = max(map(len, seqs))
        ids = torch.tensor([[pad] * (L - len(q)) + q for q in seqs], device=XC.DEV); am = torch.tensor([[0] * (L - len(q)) + [1] * len(q) for q in seqs], device=XC.DEV)
        y = model.generate(input_ids=ids, attention_mask=am, do_sample=False, max_new_tokens=NEW, pad_token_id=pad, eos_token_id=eos, use_cache=True)
        outs += [tok.decode(t[L:], skip_special_tokens=True) for t in y]; XL.log("continued", a + len(seqs), "of", len(prompts))
    return outs


def main():
    from transformers import AutoTokenizer
    items = [json.loads(l) for l in open(sys.argv[1])]; tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; XC.selfcheck(); out = []
    loras = dict(a.split("=") for a in os.environ.get("STUB_LORAS", "").split(",") if a)
    assert all(it["id"] in byid and it["text"].rstrip().endswith("```") and it["writer"] in ("e4b", "e2b", *loras) for it in items), "an item is not a problem of the pools, does not end with its fence, or its writer has no adapter in STUB_LORAS"
    for w in ["e4b", "e2b", *loras]:
        its = [it for it in items if it["writer"] == w][:2 if SMALL else None]
        if not its:
            continue
        path = os.environ["XB_TEACHER"] if w == "e4b" else os.environ["XB_STUDENT"]; m = XB.load(path); tok = AutoTokenizer.from_pretrained(path)
        if w in loras:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, loras[w]); m.eval()
        more = go_on(m, tok, [byid[it["id"]]["prompt"] for it in its], [it["text"].rstrip() for it in its]); blocks = [second_block(t) for t in more]
        ok = XL.score([byid[it["id"]] for it in its], [(b or "") + "```" for b, _ in blocks])
        ok = [o or bool(XC.run_problem(byid[it["id"]], (b or "") + "```")) for it, (b, _), o in zip(its, blocks, ok)]   # a failure is tried once more alone (as lcb_fix.py)
        for it, t, (b, c), o in zip(its, more, blocks, ok):
            out.append({"writer": w, "id": it["id"], "more": t, "block": b, "closed": c, "pass": bool(o) and b is not None})
        mine = [r for r in out if r["writer"] == w]
        XL.log(w, "stub answers", len(mine), "| a second block follows", sum(r["block"] is not None for r in mine), "| it is closed", sum(r["closed"] for r in mine), "| it passes the tests", sum(r["pass"] for r in mine))
        del m; torch.cuda.empty_cache()
    with open(sys.argv[2], "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("STUB_PAST_DONE", len(out), "answers", flush=True)


if __name__ == "__main__":
    main()
