"""Amendment 209g (user 11-10: "have Deepseek review each of the tasks, if they look bad have the correct model fix it, and then have deepseek
review it again"): the author model rewrites its own solution that the judge did not call clean. Model-written text only: E4B rewrites
E4B's rows, E2B rewrites E2B's. For each item (rid, id, author, reply, notes = the judge's issues and reasons) the model is given the problem,
its own passing solution and the notes, and writes the solution again (a greedy answer and FIX_NS samples, at most 768 new tokens, to the
closing fence). Every rewrite is run on the problem's tests here; the Mac then sends the passing ones to the judge again.
  XB_DIR=... XB_TEACHER=E4B XB_STUDENT=E2B XC_DATA=... python lcb_fix.py ITEMS.jsonl OUT.jsonl      (FIX_SMALL=1: two items per author)
FIX_MODE=continue (user 11-11: "I thought we had a script that let a model pick up where it left off at a bad turn and complete from there?";
the agent rows' fork scripts and xlearn.gen_continue do that): nothing is asked of the model. An item also holds "prefix", its own solution up
to just before the first line the judge flagged; the author model continues from there under the plain problem prompt, FIX_NC samples at
temperature 0.8 (a greedy continuation of its own greedy answer would write the same text again). The text stays the model's own, written to
the prompt the row is filed under; the tests and the judge only select."""
import json, os, sys
from pathlib import Path
import torch

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xlearn as XL  # noqa: E402
XC, XB = XL.XC, XL.XB
SMALL = os.environ.get("FIX_SMALL") == "1"; NS = 1 if SMALL else int(os.environ.get("FIX_NS", "2"))
MODE = os.environ.get("FIX_MODE", "rewrite"); NC = 2 if SMALL else int(os.environ.get("FIX_NC", "4")); assert MODE in ("rewrite", "continue")
TEMP = float(os.environ.get("FIX_TEMP", XL.TEMP))   # later rounds sample with more variety (1.0), as the agent rows' repair rounds do; an empty prefix is a fresh answer to the plain prompt
FIX = ("Here is a programming problem and a correct Python 3 solution.\n\n{problem}\n\nSolution:\n```python\n{code}```\n\n"
       "A reviewer of the solution's text noted: {notes}\n\n"
       "Rewrite the solution. Keep the same algorithm and the same behaviour. Keep only comments that say what the code does. Remove second thoughts, reasoning about the task, "
       "examples traced by hand, references to instructions, and code that is unused or written twice.\n\nWrite the complete rewritten solution in one Python code block.\n\n```python\n")


@torch.no_grad()
def cont_sample(model, tok, prompts, prefixes, n, bos):
    """n sampled continuations per row: the input is the prompt's ids followed by the kept prefix's ids (tokenized apart, as xlearn.gen_continue),
    left-padded; temperature and top-p as xlearn.gen_sample; the same stopping (the closing ``` or an end token)"""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; eos = XC.stop_ids(model, tok); outs = []; step = max(1, XC.BATCH // n)
    for a in range(0, len(prompts), step):
        pre = [tok(x, add_special_tokens=False)["input_ids"] if x else [] for x in prefixes[a:a + step]]
        seqs = [tok(bos + p, add_special_tokens=False)["input_ids"] + x for p, x in zip(prompts[a:a + step], pre)]; L = max(map(len, seqs))
        ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=XC.DEV); am = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=XC.DEV)
        y = model.generate(input_ids=ids, attention_mask=am, do_sample=True, temperature=TEMP, top_p=0.95, num_return_sequences=n, max_new_tokens=XC.MAXNEW, pad_token_id=pad, eos_token_id=eos,
                           use_cache=True, stop_strings=["```"], tokenizer=tok)
        dec = [tok.decode(r[L:], skip_special_tokens=True) for r in y]; outs += [dec[i * n:(i + 1) * n] for i in range(len(seqs))]
        XL.log("continued", a + len(seqs), "of", len(prompts))
    return outs


def main():
    from transformers import AutoTokenizer
    items = [json.loads(l) for l in open(sys.argv[1])]; tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; XC.selfcheck(); out = []
    assert all(it["author"] in ("e4b", "e2b") and it["id"] in byid and it["reply"].endswith("```") for it in items)
    assert MODE == "rewrite" or all(it["reply"].startswith(it["prefix"]) and len(it["prefix"]) < len(it["reply"]) and (not it["prefix"] or it["prefix"].endswith("\n")) for it in items), "a prefix is not the start of its row"
    for au, path in (("e4b", os.environ["XB_TEACHER"]), ("e2b", os.environ["XB_STUDENT"])):
        its = [it for it in items if it["author"] == au][:2 if SMALL else None]
        if not its:
            continue
        m = XB.load(path); tok = AutoTokenizer.from_pretrained(path); prompts = []
        if MODE == "continue":                                           # the plain prompt, the kept prefix, the model's own continuation
            torch.manual_seed(209); ss = cont_sample(m, tok, [byid[it["id"]]["prompt"] for it in its], [it["prefix"] for it in its], NC, "<bos>")
            flat = [(it, "continued" if it["prefix"] else "sampled again", it["prefix"] + t) for it, row in zip(its, ss) for t in row]
        else:
            for it in its:
                p = byid[it["id"]]["prompt"]; assert p.startswith(XL.RB_HEAD) and p.endswith(XL.RB_TAIL), it["id"]
                prompts.append(FIX.format(problem=p[len(XL.RB_HEAD):-len(XL.RB_TAIL)], code=it["reply"][:-3], notes=it["notes"]))
            g = XC.gen_hf(m, tok, prompts, bos="<bos>"); torch.manual_seed(209); ss = XL.gen_sample(m, tok, prompts, NS, bos="<bos>")
            flat = [(it, "greedy", t) for it, t in zip(its, g)] + [(it, "sample", t) for it, row in zip(its, ss) for t in row]
        ok = XL.score([byid[it["id"]] for it, _, _ in flat], [t for _, _, t in flat])
        ok = [o or bool(XC.run_problem(byid[it["id"]], t)) for (it, _, t), o in zip(flat, ok)]   # a failure is tried once more alone
        for (it, kind, t), o in zip(flat, ok):
            out.append({"rid": it["rid"], "id": it["id"], "author": au, "kind": kind, "text": t[:t.index("```") + 3] if "```" in t else t, "fenced": "```" in t, "pass": bool(o)})
        XL.log(au, "items", len(its), "| rewrites", len(flat), "| passing", int(sum(ok)), "| items with a passing fenced rewrite", len({it["rid"] for (it, _, t), o in zip(flat, ok) if o and "```" in t}))
        del m; torch.cuda.empty_cache()
    with open(sys.argv[2], "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("FIX_DONE", len(out), "rewrites of", len({r["rid"] for r in out}), "rows", flush=True)


if __name__ == "__main__":
    main()
