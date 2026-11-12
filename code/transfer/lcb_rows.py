"""Addendum 209: the coding study's rows in online.py's data format, appended to ON_DIR as kind "lcb", so that online.py's stream (ON_STREAM=lcb)
can teach contest coding with the same teachers and recipe as the agent stream.
  train: the 299 row problems with E4B's shortest verified solution (xl_rows_short.jsonl of 179a: the prompt, then E4B's own text up to its
         closing fence), the rows the coding study's sft_s copies;
  held:  the held problems E4B's greedy answer solves (xl_base.json "e4b held solutions"), with xlearn's prompt for the problem.
A row is the prompt exactly as xcode.gen_hf sends it ("<bos>" + prompt, no other special token) followed by the solution's own tokens, so a
sample drawn from the context starts where an evaluated answer starts. Model-written text only; nothing is added to it.
  XB_DIR=... XB_TEACHER=... XB_STUDENT=... XC_DATA=... python lcb_rows.py ON_DIR      (XB_DIR holds xl_rows_short.jsonl, xl_short.json, xl_base.json)"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xlearn as XL  # noqa: E402  (prompts and pools exactly as the coding study builds them; no model is loaded)


def main():
    from transformers import AutoTokenizer
    D = Path(sys.argv[1]); tok = AutoTokenizer.from_pretrained(os.environ["XB_STUDENT"]); base = json.load(open(XL.BASE))
    assert hashlib.sha256(XL.ROWS_SHORT.read_bytes()).hexdigest() == json.load(open(XL.SHORT))["sha"], "short rows and xl_short.json do not match"
    rows = [json.loads(l) for l in open(XL.ROWS_SHORT)]; assert {r["id"] for r in rows} == set(base["rows"]) and len(rows) == len(base["rows"]), "short rows are not the build's row problems"
    tr, he = XL.pools(); byid = {p["id"]: p for p in tr + he}; held = [i for i in base["held"] if i in base["e4b held solutions"]]
    assert all(r["ctx"] == byid[r["id"]]["prompt"] for r in rows), "a row's prompt is not xlearn's prompt for its problem"
    assert all(r["reply"].endswith("```") for r in rows) and all(base["e4b held solutions"][i].endswith("```") for i in held), "a solution without its closing fence"
    assert not {r["id"] for r in rows} & set(base["held"]), "row problems and held problems overlap"
    ids = np.load(D / "s3_ids.npy"); assert ids.dtype == np.int32; old = [json.loads(l) for l in open(D / "s3_calls.jsonl")]
    assert not any(r.get("kind") == "lcb" for r in old), "ON_DIR already holds lcb rows"
    new, out, off = [], [], len(ids)
    for split, items in (("train", [(r["id"], r["ctx"], r["reply"]) for r in rows]), ("held", [(i, byid[i]["prompt"], base["e4b held solutions"][i]) for i in held])):
        for i, ctx, rep in items:
            c = tok("<bos>" + ctx, add_special_tokens=False)["input_ids"]; r_ = tok(rep, add_special_tokens=False)["input_ids"]
            assert c[0] == tok.bos_token_id and len(r_) > 0 and tok.bos_token_id not in c[1:] + r_ and tok.pad_token_id not in c + r_, i
            out.append({"kind": "lcb", "split": split, "off": off, "ctx_len": len(c), "reply_len": len(r_), "action": "text", "run": i, "repo": byid[i]["platform"], "id": i, "difficulty": byid[i]["difficulty"]})
            new += c + r_; off += len(c) + len(r_)
    np.save(D / "s3_ids.npy", np.concatenate([ids, np.asarray(new, dtype=np.int32)]))
    with open(D / "s3_calls.jsonl", "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    chk = np.load(D / "s3_ids.npy", mmap_mode="r"); assert len(chk) == off and (chk[:len(ids)] == ids).all()
    bad = 0                                                              # every train row decodes back to its text (reported, not fatal: a tokenizer may normalise a rare character)
    for r0, r in zip(out, rows):
        back = tok.decode(chk[r0["off"]:r0["off"] + r0["ctx_len"] + r0["reply_len"]].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        bad += back != "<bos>" + r["ctx"] + r["reply"]
    print("rows that do not decode back to their exact text:", bad, "of", len(rows)); assert bad <= len(rows) // 10, "too many rows change when decoded"
    q = lambda v: [int(x) for x in np.percentile(v, [50, 90, 100])]  # noqa: E731
    for split in ("train", "held"):
        rs = [r for r in out if r["split"] == split]
        print("lcb rows", split, len(rs), "| context tokens (median, p90, max)", q([r["ctx_len"] for r in rs]), "| solution tokens", q([r["reply_len"] for r in rs]),
              "| solutions over 512 tokens", sum(r["reply_len"] > 512 for r in rs))
    print("LCB_ROWS_OK", len(out), "rows,", off - len(ids), "tokens appended; agent and mix rows unchanged:", len(old))


if __name__ == "__main__":
    main()
