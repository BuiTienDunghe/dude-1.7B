"""Tỉ lệ số token (bộ khôi phục) của bản có dấu so với bản không dấu, trên state tiếng Việt dev_v1: để chọn cỡ đoạn đầu vào
sao cho đầu ra (có dấu) không vượt max_length 256. Không chạy model."""
import json, sys
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, load_json, read_jsonl  # noqa: E402
cfg = load_json(CONFIG / "restorers.json")
toks = {"vit5": PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(cfg["vit5"]["repo"], "tokenizer.json", revision=cfg["vit5"]["revision"]),
                                        eos_token="</s>", pad_token="<pad>", unk_token="<unk>"),
        "small": AutoTokenizer.from_pretrained(cfg["small"]["repo"], revision=cfg["small"]["revision"])}
dev = {r["id"]: r for r in read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")}
pairs = [(r["state"], dev[r["id"][:-7]]["state"]) for r in dev.values() if r["lang"] == "vi" and r["variant"] == "nodiac"]
for n, t in toks.items():
    rat = np.array([len(t(o)["input_ids"]) / len(t(nd)["input_ids"]) for nd, o in pairs])
    unk = sum(t.unk_token_id in t(o)["input_ids"] for _, o in pairs)
    print(n, f"ratio p50 {np.median(rat):.3f} p99 {np.percentile(rat, 99):.3f} max {rat.max():.3f}; states with <unk> in diacritized form: {unk}/{len(pairs)}")
