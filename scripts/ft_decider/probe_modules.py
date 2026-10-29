"""Liệt kê module Linear thực tế của decider-2b (Qwen3_5ForCausalLM) trên thiết bị meta — để chọn đích LoRA. Không nạp trọng số."""
import json, collections, sys
from pathlib import Path
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa: E402
folder = snapshot_download("Mapika/decider-2b", revision="ff99089b6810691bd79134ec6febb5d9afa6e619")
cfg = AutoConfig.from_pretrained(folder)
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg)
cnt = collections.Counter(); shapes = {}
for n, mod in m.named_modules():
    if isinstance(mod, torch.nn.Linear):
        parts = n.split(".")
        key = ".".join(p for p in parts if not p.isdigit() and p not in ("model", "layers"))
        cnt[key] += 1; shapes[key] = tuple(mod.weight.shape)
res = {"class": type(m).__name__, "linear_modules": {k: {"count": v, "shape_out_in": shapes[k]} for k, v in sorted(cnt.items())}}
import transformers.models.qwen3_5.modeling_qwen3_5 as q
res["fla_fast_path"] = {k: bool(getattr(q, k, None)) for k in ("is_fast_path_available", "causal_conv1d_fn", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")}
(RESULTS / "ft_decider" / "modules.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res))
