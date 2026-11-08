"""Review item D1 (11-08): how far each coding recipe moved the student on text it never trained on, measured against the student itself.
Per held untaught row (prose: novel passages, code: CPython stdlib, story; never in any arm's training rows), over the reply tokens:
KL(original student || trained student) and the change in cross-entropy (trained - original). Lower-than-original CE is a change too: this
separates "did not move" from "moved toward lower loss". One forward of the original student per row, then each adapter in turn.
  python drift.py STUDENT_DIR ROWS_JSON RUNS_DIR OUT_JSON [ARM ...]
186: env XB_SBOS ("<bos>" for a Gemma student), DRIFT_DIR (adapters as DRIFT_DIR/xl_ARM, seed XB_SEED: one session's own folder), CUDA when present."""
import os
import json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

QW, ROWS, RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
WHERE = {"sft": "xqlO", "sft_s": "xqlO", "sft_c_s": "xqlO", "opd_c": "xqlO", "mix_c": "xqlO", "opd_cf": "xqlCF", "rw_cf": "xqlCF", "mix_cf": "xqlCF", "rb": "xqlRB", "rb_c": "xqlRBC", "sft_self_s": "xqlSS"}
ARMS = sys.argv[5:] or list(WHERE)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"); MAXT = 640; SBOS = os.environ.get("XB_SBOS", "")
tok = AutoTokenizer.from_pretrained(QW); base = AutoModelForCausalLM.from_pretrained(QW, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
DD = os.environ.get("DRIFT_DIR")
adapters = [(a, int(os.environ.get("XB_SEED", "1")), Path(DD) / f"xl_{a}") for a in ARMS] if DD else [(a, s, RUNS / f"{WHERE[a]}{s}/xl_{a}_s{s}") for a in ARMS for s in (1, 2, 3)]
adapters = [x for x in adapters if (x[2] / "adapter_model.safetensors").exists()]; print("adapters:", [f"{a}_s{s}" for a, s, _ in adapters], flush=True)
pm = None
for a, s, p in adapters:
    if pm is None:
        pm = PeftModel.from_pretrained(base, str(p), adapter_name=f"{a}_s{s}")
    else:
        pm.load_adapter(str(p), adapter_name=f"{a}_s{s}")
pm.eval(); rows = json.load(open(ROWS)); res = {}; t0 = time.time()
with torch.no_grad():
    for k, r in enumerate(rows):
        c = tok(SBOS + r["ctx"], add_special_tokens=False)["input_ids"]; y = tok(r["reply"], add_special_tokens=False)["input_ids"]
        ids = (c + y)[:MAXT]; n0 = len(c)
        if n0 < 1 or len(ids) - n0 < 4:
            continue
        x = torch.tensor([ids], device=DEV); yt = x[0, n0:]
        with pm.disable_adapter():
            lb = F.log_softmax(pm(input_ids=x).logits[0, n0 - 1:-1].float(), -1)
        ceb = float(-lb.gather(-1, yt[:, None]).mean()); hb = float(-(lb.exp() * lb).sum(-1).mean())   # review: entropy, so "sharpens" is measured
        for a, s, _ in adapters:
            pm.set_adapter(f"{a}_s{s}"); la = F.log_softmax(pm(input_ids=x).logits[0, n0 - 1:-1].float(), -1)
            kl = float((lb.exp() * (lb - la)).sum(-1).mean()); cea = float(-la.gather(-1, yt[:, None]).mean()); ha = float(-(la.exp() * la).sum(-1).mean())
            d = res.setdefault(f"{a}|{s}|{r['kind']}", {"kl": [], "dce": [], "ce": [], "ce base": [], "entropy": [], "entropy base": []})
            d["kl"].append(kl); d["dce"].append(cea - ceb); d["ce"].append(cea); d["ce base"].append(ceb); d["entropy"].append(ha); d["entropy base"].append(hb)
        if k % 10 == 0:
            print(f"row {k + 1} of {len(rows)} | {time.time() - t0:.0f} s", flush=True)
summ = {key: {m: round(sum(v) / len(v), 5) for m, v in d.items()} | {"rows": len(d["kl"])} for key, d in res.items()}
json.dump({"summary": summ, "per row": res}, open(OUT, "w")); print("wrote", OUT, f"{time.time() - t0:.0f} s")
