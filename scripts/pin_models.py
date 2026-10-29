"""Ghim commit sha của hai model, kiểm tra hai model dùng chung tokenizer/chat template,
và kiểm tra mỗi chữ cái A..J là đúng một token ngay tại vị trí trả lời của prompt."""
import hashlib
import json

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from common import CONFIG, LETTERS, MODELS_FILE, load_json, render_prompt

NAMES = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
api = HfApi()
pins = {n: api.model_info(n).sha for n in NAMES}
print(pins)

frames = load_json(CONFIG / "prompt_frames.json")
demo = {
    "en": {"lang": "en", "state": "The man turned on the faucet.", "question": "What is the more plausible effect?",
           "options": [f"option {i}" for i in range(10)]},
    "vi": {"lang": "vi", "state": "Người đàn ông bật vòi nước.", "question": "Kết quả nào hợp lý hơn?",
           "options": [f"lựa chọn {i}" for i in range(10)]},
}
fingerprints = {}
for name, rev in pins.items():
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    vocab = json.dumps(sorted(tok.get_vocab().items()), ensure_ascii=False)
    fingerprints[name] = (hashlib.sha256(vocab.encode()).hexdigest(), hashlib.sha256(tok.chat_template.encode()).hexdigest())
    print(name, "vocab", len(tok), "fingerprint", fingerprints[name])
    for lang, row in demo.items():
        p = render_prompt(row, tok, frames)
        assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(p[-80:])
        base = tok.encode(p, add_special_tokens=False)
        for L in LETTERS:
            ids = tok.encode(L, add_special_tokens=False)
            full = tok.encode(p + L, add_special_tokens=False)
            assert len(ids) == 1, (L, ids)
            assert full[:-1] == base and full[-1] == ids[0], (L, full[-3:], base[-3:])
        if name == NAMES[0]:
            print(f"--- prompt {lang} ---\n{p}<END>")
    print(name, "letters A..J: single token at answer boundary OK", [tok.convert_tokens_to_ids(L) for L in LETTERS])

assert len(set(fingerprints.values())) == 1, "hai model khác tokenizer/chat template"
print("tokenizer + chat template identical across models")
MODELS_FILE.write_text(json.dumps(pins, indent=2), encoding="utf-8")
