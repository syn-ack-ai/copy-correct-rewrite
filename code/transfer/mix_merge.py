"""Addendum 165: one ON_DIR holding the agent rows (unchanged, first) and the mixed rows of build_mix.py after them (offsets shifted), so the
behaviour arms see exactly the agent data they saw before (online.data() keeps agent rows unless ON_TRMIX=1) and the translator can mix kinds.
  python mix_merge.py AGENT_DIR MIX_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np

a, m, o = (Path(p) for p in sys.argv[1:4]); o.mkdir(parents=True, exist_ok=True); (o / "out").mkdir(exist_ok=True)
ia = np.load(a / "s3_ids.npy", mmap_mode="r"); im = np.load(m / "mix_ids.npy"); assert ia.dtype == np.int32 and im.dtype == np.int32, (ia.dtype, im.dtype)
np.save(o / "s3_ids.npy", np.concatenate([np.asarray(ia), im])); base = len(ia); n = 0
with open(o / "s3_calls.jsonl", "w") as fh:
    for line in open(a / "s3_calls.jsonl"):
        r = json.loads(line); assert "kind" not in r; fh.write(json.dumps(r) + "\n"); n += 1
    for line in open(m / "mix_rows.jsonl"):
        r = json.loads(line); r["off"] += base; fh.write(json.dumps(r) + "\n"); n += 1
chk = np.load(o / "s3_ids.npy", mmap_mode="r"); assert (chk[:base] == ia).all() and (chk[base:] == im).all()
print("merged", base, "+", len(im), "tokens;", n, "rows ->", o)
