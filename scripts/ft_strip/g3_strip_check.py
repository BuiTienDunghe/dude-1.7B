"""Cổng G3: kiểm bước bỏ dấu khi nạp dữ liệu train (p = 0,30; một nửa chỉ bỏ ở state).
- tỉ lệ bỏ dấu thực tế theo epoch và theo hai chế độ, trên mọi dòng tiếng Việt của train_v0;
- 0 dòng không phải tiếng Việt bị đổi (áp đúng quy tắc của train_lora.maybe_strip lên mọi dòng);
- đ/Đ và ký tự Unicode dạng tổ hợp: không còn dấu kết hợp (Mn), không còn đ/Đ, không còn chữ Việt có dấu trong phần bị bỏ;
- 5 ví dụ trước/sau, mỗi trường tối đa 200 ký tự.
Ra: results/ft_strip/g3_strip_check.json
"""
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, apply_strip, read_jsonl, strip_diacritics, strip_decision  # noqa: E402

SEED, P, FRAC, EPOCHS = 1, 0.30, 0.5, 2
train = read_jsonl(DATA / "train_v0.jsonl")
cut = lambda s: s[:200]


def maybe_strip(row, epoch):   # đúng quy tắc của scripts/train_lora.py: maybe_strip
    if P <= 0 or row["text_lang"] != "vi":
        return row, None
    mode = strip_decision(SEED, epoch, row["id"], P, FRAC)
    return apply_strip(row, mode), mode


def has_marks(s):
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s)) or "đ" in s or "Đ" in s


res = {"by_epoch": [], "non_vi_changed": 0, "vi_rows": sum(r["text_lang"] == "vi" for r in train),
       "text_lang": dict(Counter(r["text_lang"] for r in train))}
examples, bad = [], []
for epoch in range(EPOCHS):
    c = Counter()
    for r in train:
        s, m = maybe_strip(r, epoch)
        if r["text_lang"] != "vi":
            if s != r or m is not None:
                res["non_vi_changed"] += 1
            continue
        c[m or "none"] += 1
        if m:
            fields = [s["state"]] + ([s["question"]] + s["options"] if m == "all" else [])
            if any(has_marks(f) for f in fields):
                bad.append(r["id"])
            if m == "state_only":   # câu hỏi và lựa chọn phải giữ nguyên
                assert s["question"] == r["question"] and s["options"] == r["options"]
            assert s["target"] == r["target"] and len(s["options"]) == len(r["options"]) and s["id"] == r["id"]
            if epoch == 0 and len(examples) < 5 and (len(examples) < 3 or m == "state_only"):
                examples.append({"id": r["id"], "mode": m, "state_before": cut(r["state"]), "state_after": cut(s["state"]),
                                 "question_before": cut(r["question"]), "question_after": cut(s["question"]),
                                 "options_before": cut(" | ".join(r["options"])), "options_after": cut(" | ".join(s["options"]))})
    n = res["vi_rows"]
    res["by_epoch"].append({"epoch": epoch, "none": c["none"], "state_only": c["state_only"], "all": c["all"],
                            "rate_stripped": (c["state_only"] + c["all"]) / n, "rate_state_only": c["state_only"] / n, "rate_all": c["all"] / n})
res["rows_with_remaining_marks_after_strip"] = len(bad)
# Unicode: đ/Đ, dạng dựng sẵn (NFC) và dạng tổ hợp (NFD), ký tự IPA có dấu kết hợp
tests = {"Đà Nẵng đẹp": "Da Nang dep", unicodedata.normalize("NFD", "Tiếng Việt có dấu"): "Tieng Viet co dau",
         unicodedata.normalize("NFC", "Tiếng Việt có dấu"): "Tieng Viet co dau", "Ưu điểm ĐỊA LÝ": "Uu diem DIA LY", "[lə fʁɑ̃sɛ]": "[lə fʁɑsɛ]"}
res["unicode_tests"] = {k: {"got": strip_diacritics(k), "expected": v, "ok": strip_diacritics(k) == v} for k, v in tests.items()}
res["train_vi_rows_with_decomposed_marks"] = sum(1 for r in train if r["text_lang"] == "vi" and r["state"] != unicodedata.normalize("NFC", r["state"]))
res["examples"] = examples
(RESULTS / "ft_strip" / "g3_strip_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=1))
for e in examples:
    print(json.dumps(e, ensure_ascii=False))
