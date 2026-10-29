"""Module recast dùng chung cho train_v1, lấy NGUYÊN VĂN từ scripts/build_train_v0.py (file đó chạy ngay khi import và ghi đè data/train_v0.jsonl,
nên không import được): ts_row, pl_row, cites_other_options và các bộ lọc của finalize (2..10 lựa chọn, target là phân phối, state không rỗng,
lựa chọn không trích chữ cái lựa chọn khác, state <= max_state_tokens token Qwen3 (v0: 1024; mặc định mới 2048), khử rò rỉ). verify_recast_v0.py chứng minh hai hàm ts_row/pl_row ở đây
dựng lại đúng từng dòng của train_v0/val_v0 từ nguồn.
Khác v0 (chỉ thêm, không đổi phép lọc cũ): finalize nhận nhiều chỉ mục; ngoài state còn kiểm câu hỏi và từng lựa chọn có >= 4 từ.
QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import json
import re

from common import LETTERS, load_json, CONFIG
from decontam import normalize

MAX_OPTIONS = len(LETTERS)
# Trần độ dài state (token Qwen3): tham số của make_finalize, mặc định 2.048 từ 28/10 (audit 27/10: ViFactCheck trên dev có trung vị 886 token).
# v0, phần A, B, C dựng với 1.024: các nơi gọi cũ truyền max_state_tokens=1024 tường minh để dựng lại đúng như cũ.
DEFAULT_MAX_STATE_TOKENS = 2048
fmt = load_json(CONFIG / "train_format.json")

LETTER_REF = [re.compile(p) for p in (
    r"\b[A-J]\b\s*(?:,|and|&|or)\s*\b[A-J]\b",
    r"\([a-jA-J1-9]\)\s*(?:,|and|&|or)\s*\(?[a-jA-J1-9]\)",
    r"(?i)\b(?:options?|answers?|choices?|statements?)\s*\(?[A-J1-9]\)?(?:\s|$|,|\.)",
)]


def cites_other_options(options):
    return any(p.search(o) for o in options for p in LETTER_REF)


def make_finalize(tok, indexes, drops, max_state_tokens=DEFAULT_MAX_STATE_TOKENS):
    """indexes: [(tên, decontam.Index)]. Trả về finalize(row) -> row | None, đếm lý do loại vào drops (Counter).
    max_state_tokens: trần độ dài state (mặc định 2.048; v0/C dựng với 1.024)."""
    def finalize(row):
        if not (2 <= len(row["options"]) <= MAX_OPTIONS):
            drops[f"options_not_2_to_{MAX_OPTIONS}"] += 1; return None
        if abs(sum(row["target"]) - 1) > 1e-4 or min(row["target"]) < 0:
            drops["target_not_distribution"] += 1; return None
        if not row["state"].strip():
            drops["empty_state"] += 1; return None
        if cites_other_options(row["options"]):
            drops["options_cite_other_options"] += 1; return None
        n = len(tok.encode(row["state"], add_special_tokens=False))
        if n > max_state_tokens:
            drops[f"state_over_{max_state_tokens}_tokens"] += 1; return None
        row["n_state_tokens"] = n
        units = [("state", row["state"])] + [(f, t) for f, t in [("question", row["question"] or "")] + [("option", o) for o in row["options"]]
                                              if len(normalize(t).split()) >= 4]
        for name, ix in indexes:
            for field, text in units:
                hit = ix.check(text)
                if hit:
                    drops[f"decontam:{name}:{field}:{hit[0]}:{hit[1]}"] += 1; return None
        return row
    return finalize


def ts_row(x):
    src = x["source"]
    if src.startswith("multilingual/"):
        text_lang = "vi" if re.search(r"(?:[-/_])vi$", src) else "multi"
    else:
        text_lang = "en"
    lang = "vi" if text_lang == "vi" else "en"   # khung prompt chỉ có en/vi
    if x["kind"] == "noul":
        p = float(x["target"][0])
        options, target = list(fmt["noul_options"][lang]), [p, 1 - p]
    else:
        options, target = list(x["options"]), [float(t) for t in x["target"]]
    return {"id": f"ts:{x['id']}", "source": src, "dataset": "tasksource", "lang": lang, "text_lang": text_lang,
            "kind": x["kind"], "state": x["state"], "question": x["question"], "options": options, "target": target,
            "group_keys": [f"g:{x['group_id']}"], "license_use": x["license_use"], "variant": x["variant"]}


def pl_row(x, ds_name, n):
    state = json.loads(x["state"]); qs = json.loads(x["questions"]); gold = json.loads(x["gold"])
    (qid, q), = qs.items()
    labels = fmt["pholaya_state_labels"]
    text = next(iter(state.values())) if len(state) == 1 else "\n".join(f"{labels[k]}: {v}" for k, v in state.items())
    probs = gold[qid]["probabilities"]
    if q["type"] == "noul":
        p = float(probs["true"])
        assert abs(p + float(probs["false"]) - 1) < 1e-6
        options, target = list(fmt["noul_options"]["vi"]), [p, 1 - p]
    else:
        keys = list(q["criteria"])
        assert set(keys) == set(probs), (keys, probs)
        options, target = [q["criteria"][k] for k in keys], [float(probs[k]) for k in keys]
    task = "aspect" if qid.startswith("aspect_") else qid
    return {"id": f"pl:{ds_name}:{n}", "source": f"pholaya/{ds_name}/{task}", "dataset": f"pholaya/{ds_name}",
            "lang": "vi", "text_lang": "vi", "kind": q["type"], "state": text, "question": q["instructions"],
            "options": options, "target": target, "group_keys": [f"f:{normalize(v)}" for v in state.values()],
            "license_use": "research-only (PhoLaya-base, UIT corpora)", "variant": qid}
