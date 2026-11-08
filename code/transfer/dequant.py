"""Decompress a compressed-tensors QAT w4a16 checkpoint to plain bf16 (addendum 121): the exact weight values vLLM serves, in a folder that
loads without compressed-tensors. Keeps only dense tensors (drops weight_packed / weight_scale / weight_shape / zero_point), removes
quantization_config, then reloads the folder as a plain model and checks every tensor equals the decompressed one.
  ~/g4agent/swe/bin/python transfer/dequant.py QAT_DIR OUT_DIR"""
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

src, dst = sys.argv[1], Path(sys.argv[2]); t = time.time()
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map={"": "cpu"}, quantization_config=CompressedTensorsConfig(run_compressed=False))
QK = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "zero_point", "weight_g_idx")
sd = m.state_dict(); dense = {k: v.contiguous() for k, v in sd.items() if not k.split(".")[-1] in QK}
dropped = sorted({k.split(".")[-1] for k in sd} - {k.split(".")[-1] for k in dense})
lin = [k for k in dense if k.endswith("_proj.weight")]
assert lin and all(dense[k].dtype == torch.bfloat16 and dense[k].dim() == 2 for k in lin), "projection weights are not dense bf16"
print("dense tensors", len(dense), "of", len(sd), "| dropped kinds", dropped, "| projection weights", len(lin), flush=True)
# m.save_pretrained fails here (11-05: it tries to revert the quantizer's weight conversion), so the folder is written by hand: the source's
# config without quantization_config, its generation config and tokenizer, and one safetensors file of the dense tensors.
import shutil
from safetensors.torch import save_file
dst.mkdir(parents=True, exist_ok=True); c = json.load(open(Path(src) / "config.json")); had = c.pop("quantization_config", None) is not None
json.dump(c, open(dst / "config.json", "w"), indent=2)
for f in ("generation_config.json", "chat_template.jinja"):
    if (Path(src) / f).exists():
        shutil.copy(Path(src) / f, dst / f)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
seen, out = {}, {}
for k, v in dense.items():   # tied tensors (lm_head = embeddings) are stored once; safetensors refuses shared storage
    key = (v.data_ptr(), tuple(v.shape))
    if key in seen:
        continue
    seen[key] = k; out[k] = v.clone()
save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"}); del m, out
p = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.bfloat16, device_map={"": "cpu"}); ps = p.state_dict()
missing = [k for k in dense if k not in ps]; extra = [k for k in ps if k not in dense and not k.endswith("lm_head.weight")]
diff = max(float((ps[k].float() - dense[k].float()).abs().max()) for k in dense if k in ps)
assert not missing and diff == 0.0, f"reload differs: missing {missing[:5]}, max abs diff {diff}"
print("DEQUANT_OK in", round(time.time() - t), "s | quantization_config removed:", had, "| reload: max abs diff", diff, "| extra keys", extra[:3], flush=True)
