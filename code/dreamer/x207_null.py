"""Addendum 207: the null adapter. A copy of an online LoRA (setup D, seed 1) with every lora_B tensor set to zero, so the served model's
outputs are plain E2B's while the request goes through the same adapter path (vLLM --lora-modules, the s1t1024_s4b submission).
  python x207_null.py SRC_ADAPTER_DIR DST_DIR"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(src / "adapter_config.json")); sd = load_file(str(src / "adapter_model.safetensors"))
kb = [k for k in sd if "lora_B" in k]; ka = [k for k in sd if "lora_A" in k]
assert len(kb) == len(ka) == len(cfg["target_modules"]) and len(kb) + len(ka) == len(sd), (len(kb), len(ka), len(cfg["target_modules"]), len(sd))
assert all(float(sd[k].abs().sum()) > 0 for k in kb), "the source adapter already has an all-zero lora_B"
out = {k: (torch.zeros_like(v) if "lora_B" in k else v.clone()) for k, v in sd.items()}
save_file(out, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"}); shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
chk = load_file(str(dst / "adapter_model.safetensors"))
assert set(chk) == set(sd) and all(float(chk[k].abs().sum()) == 0 for k in kb) and all(torch.equal(chk[k], sd[k]) for k in ka)
assert all(chk[k].dtype == sd[k].dtype and chk[k].shape == sd[k].shape for k in sd)
print("NULL_OK", len(kb), "lora_B tensors zeroed,", len(ka), "lora_A kept, rank", cfg["r"], "->", dst)
