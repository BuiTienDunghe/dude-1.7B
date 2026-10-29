"""vit5: transformers 5.19 không nạp được T5Tokenizer từ repo (TypeError khi dựng Unigram từ 'vocab').
Kiểm cách thay: nạp tokenizer.json của chính repo (PreTrainedTokenizerFast) và so id với sentencepiece spiece.model (+ </s>)
trên mọi chuỗi state/câu hỏi/lựa chọn tiếng Việt (có dấu và không dấu) của dev_v1. Ra: results/restore/tok_check_vit5.json"""
import json, sys
from pathlib import Path
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, load_json, read_jsonl  # noqa: E402
c = load_json(CONFIG / "restorers.json")["vit5"]
fast = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(c["repo"], "tokenizer.json", revision=c["revision"]),
                               eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(c["repo"], "spiece.model", revision=c["revision"]))
texts = []
for r in read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl"):
    if r["lang"] == "vi":
        texts += [r["state"], r["question"], *r["options"]]
texts = sorted(set(texts))
n_diff, ex = 0, []
n_diff_stripped = sum(fast(t.strip())["input_ids"] != sp.encode(t.strip()) + [1] for t in texts)
n_edge_ws = sum(t != t.strip() for t in texts)
for t in texts:
    a = fast(t)["input_ids"]; b = sp.encode(t) + [1]
    if a != b:
        n_diff += 1
        if len(ex) < 5: ex.append({"text": t[:200], "fast": a[:20], "sp": b[:20]})
rt = fast.decode(fast("Toi yeu Viet Nam")["input_ids"], skip_special_tokens=True)
res = {"n_texts": len(texts), "n_diff": n_diff, "n_diff_after_strip_edges": n_diff_stripped, "n_texts_with_edge_whitespace": n_edge_ws, "examples": ex, "roundtrip": rt, "vocab_size": len(fast)}
(RESULTS / "restore" / "tok_check_vit5.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("n_texts", "n_diff", "n_diff_after_strip_edges", "n_texts_with_edge_whitespace", "roundtrip", "vocab_size")}, ensure_ascii=False))
