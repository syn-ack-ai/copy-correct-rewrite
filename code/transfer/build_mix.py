"""Addendum 165, fix 2 (user 11-07: "do all 5 fixes first"): the mixed rows a generic translator trains and is tested on, in online.py's format
(token ids + rows with off, ctx_len, reply_len, split, kind). The agent rows stay in their own file; mix_merge.py appends these after them.
Kinds (all local, no download; [redacted]):
  prose          novels / grreal / grfiction passages (shape_store.train_texts: the fresh set's texts left out), BOS + 511 tokens, context 48
  story          the original and confirmation story texts (no fresh-set text), BOS + text, context 24
  code           Python standard-library source (this machine's CPython 3.12), 80-line chunks, BOS + up to 511 tokens, context 48
  question       the pattern question (TRIPLE, chat template, thinking off) on the training items, all 6 answer orders; reply = the true letter;
                 split by item (every 5th item held)
  question_fresh the same question on the asked test's fresh items (never in any translator's training), one answer order per item, held only
Held split for prose, story, code: every 7th text. Writes mix/mix_ids.npy (int32) and mix/mix_rows.jsonl (offsets into mix_ids).
  ~/.venvs/forget/bin/python transfer/build_mix.py"""
import glob, json, random, sys, sysconfig
from itertools import permutations
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import translator as X  # noqa: E402
from shape_store import train_texts, train_items  # noqa: E402
from shape_translator import TRIPLE  # noqa: E402

MAXT = 512


def fresh_items():
    """(anchor, true, lure, foil) for the fresh set, built as shape_store.train_items builds the training items"""
    import fresh
    out = []
    for its, ex in ((fresh.A, fresh.A2), (fresh.B, fresh.B2)):
        for it in its:
            out.append((it[2], it[3], ex[it[0]][0], ex[it[0]][1]))
    return out


def main():
    tok = X.tok_(); bos = tok.bos_token_id; ids, rows = [], []

    def add(tt, ctx_len, kind, split, **kw):
        rows.append({"off": len(ids), "ctx_len": int(ctx_len), "reply_len": int(len(tt) - ctx_len), "split": split, "kind": kind, "action": "text", **kw}); ids.extend(tt)

    st, pr = train_texts()
    titems = train_items(); heldq = {t for i, it in enumerate(titems) if i % 5 == 0 for t in it}   # review 11-07: no held question's text in the stories
    st = [t for t in st if t not in heldq]
    for i, t in enumerate(pr):
        tt = [bos] + tok.encode(t, add_special_tokens=False)[:MAXT - 1]
        if len(tt) >= 96:
            add(tt, 48, "prose", "held" if i % 7 == 0 else "train")
    rs = random.Random(165); st = list(st); rs.shuffle(st)
    for i, t in enumerate(st[:420]):
        tt = [bos] + tok.encode(t, add_special_tokens=False)[:MAXT - 1]
        if len(tt) >= 48:
            add(tt, 24, "story", "held" if i % 7 == 0 else "train")
    files = sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/*.py")); nc = 0
    for f in files:
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        for a in range(0, len(lines), 400):                              # up to a few chunks per module, spread through the file
            chunk = "\n".join(lines[a:a + 80]); tt = [bos] + tok.encode(chunk, add_special_tokens=False)[:MAXT - 1]
            if len(tt) >= 128 and nc < 420:
                add(tt, 48, "code", "held" if nc % 7 == 0 else "train", src=Path(f).name); nc += 1

    def q(item, order, kind, split, it_i):
        an, tr_, lu, fo = item; opts = {"true": tr_, "lure": lu, "foil": fo}
        text = TRIPLE.format(anchor=an, **{k: opts[o] for k, o in zip("abc", order)}); letter = "ABC"[order.index("true")]
        p = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        tt = tok.encode(p, add_special_tokens=False); add(tt + [tok.encode(letter, add_special_tokens=False)[0]], len(tt), kind, split, true=letter, item=it_i)
    for i, it in enumerate(titems):
        for order in permutations(["true", "lure", "foil"]):
            q(it, order, "question", "held" if i % 5 == 0 else "train", i)
    orders = list(permutations(["true", "lure", "foil"]))
    for i, it in enumerate(fresh_items()):
        q(it, orders[i % 6], "question_fresh", "held", i)
    out = HERE / "mix"; out.mkdir(exist_ok=True); np.save(out / "mix_ids.npy", np.asarray(ids, np.int32))
    with open(out / "mix_rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print("tokens", len(ids), "rows", len(rows), dict(Counter((r["kind"], r["split"]) for r in rows)))
    print("mean reply tokens", {k: round(float(np.mean([r["reply_len"] for r in rows if r["kind"] == k])), 1) for k in sorted({r["kind"] for r in rows})})


if __name__ == "__main__":
    main()
