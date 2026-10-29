"""Danh mục mọi chuỗi tiếng Việt DO AGENT VIẾT mà benchmark (bench 1.1) dùng, kèm bản tiếng Anh, giá trị nhãn gốc và trạng thái gốc.

Dùng chung cho tools/review_app/app.py (màn hình "Mẫu và nhãn"), scripts/human_edits/build_revised.py và release_check.py.
Trạng thái gốc: "đã duyệt" nếu mục trong config có "_status" bắt đầu bằng "người làm dự án đã duyệt"; còn lại "chưa duyệt".
Chuỗi lấy nguyên từ nguồn (câu hỏi và lựa chọn Belebele, lựa chọn XCOPA, state, nhận định) KHÔNG nằm ở đây: chúng là "từ nguồn".
Các mục ViHSD trong config/bench_v1_questions.json không dùng ở bench 1.1 nên không được liệt kê.
"""
import json
from pathlib import Path

APPROVED_PREFIX = "người làm dự án đã duyệt"


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _approved(obj):
    return isinstance(obj, dict) and str(obj.get("_status", "")).startswith(APPROVED_PREFIX)


def inventory(config_dir="config"):
    c = Path(config_dir)
    tr, q, fmt = _load(c / "translations.json"), _load(c / "bench_v1_questions.json"), _load(c / "train_format.json")
    out = []

    def add(key, group, vi, en, label, approved, used_in, src_file, note=""):
        out.append({"key": key, "group": group, "vi": vi, "en": en, "label": label, "base_status": "đã duyệt" if approved else "chưa duyệt",
                    "used_in": used_in, "file": src_file, "note": note})

    for k, used in (("sib200", "SIB-200"), ("massive", "MASSIVE"), ("xcopa_cause", "XCOPA (question = cause)"), ("xcopa_effect", "XCOPA (question = effect)")):
        v = tr["questions"][k]
        add(f"translations.questions.{k}", "Câu hỏi mẫu", v["vi"], v["en"], "—", _approved(v), used, "config/translations.json")
    for k, used in (("factcheck_choice", "ViFactCheck, ViWikiFC · choice"), ("factcheck_noul", "ViFactCheck, ViWikiFC · noul")):
        add(f"bench_v1_questions.{k}.question", "Câu hỏi mẫu", q[k]["question"], "— (khối bản địa chỉ có tiếng Việt)", "{claim} = nhận định từ nguồn",
            _approved(q[k]), used, "config/bench_v1_questions.json")
    for k, v in sorted(tr["sib200_topics"].items()):
        add(f"translations.sib200_topics.{k}", "Tên chủ đề SIB-200", v["vi"], v["en"], f"category = {k}", _approved(v), "SIB-200", "config/translations.json")
    for k, v in sorted(tr["massive_intents"].items()):
        add(f"translations.massive_intents.{k}", "Mô tả ý định MASSIVE", v["vi"], v["en"], f"intent = {k}", _approved(v), "MASSIVE", "config/translations.json")
    fc_orig = {"supported": "ViFactCheck labels = 0; ViWikiFC gold_label = Supports", "refuted": "ViFactCheck labels = 1; ViWikiFC gold_label = Refutes",
               "nei": "ViFactCheck labels = 2; ViWikiFC gold_label = Not_Enough_Information"}
    for lab, v in q["factcheck_choice"]["options"].items():
        add(f"bench_v1_questions.factcheck_choice.options.{lab}", "Lựa chọn kiểm chứng", v, "—", f"{lab} ({fc_orig[lab]})",
            _approved(q["factcheck_choice"]), "ViFactCheck, ViWikiFC · choice", "config/bench_v1_questions.json")
    for j, (vi, en) in enumerate(zip(fmt["noul_options"]["vi"], fmt["noul_options"]["en"])):
        add(f"train_format.noul_options.vi.{j}", "Có/Không", vi, en, "đúng khi nhãn gốc = supported" if j == 0 else "đúng khi nhãn gốc = refuted hoặc nei",
            _approved(fmt), "ViFactCheck, ViWikiFC · noul (lựa chọn cố định [Có, Không])", "config/train_format.json",
            "Cũng là lựa chọn noul tiếng Việt của train_v0.")
    assert len({x["key"] for x in out}) == len(out)
    return out


GROUPS = ["Câu hỏi mẫu", "Tên chủ đề SIB-200", "Mô tả ý định MASSIVE", "Lựa chọn kiểm chứng", "Có/Không"]
FROM_SOURCE_NOTE = ("Từ nguồn (không cần duyệt ở màn hình này; sửa từng câu ở màn hình \"Từng câu\"): câu hỏi và 4 lựa chọn Belebele, "
                    "2 lựa chọn XCOPA, mọi state, và nhận định chèn vào câu hỏi kiểm chứng.")
