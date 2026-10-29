"""Quyết định người dùng (chat 20/10): nhãn kiểm chứng trong bảng thuật ngữ v2 THEO BENCH. Lấy đúng chuỗi người dùng đã duyệt trong
config/bench_v1_questions.json (factcheck_choice.options; chỉ đọc config):
  SUPPORTS -> options.supported, REFUTES -> options.refuted, NOT ENOUGH INFO -> options.nei.
Đổi 3 nhãn và mọi câu hỏi trích các nhãn đó trong ngoặc kép (thay đúng chuỗi trong ngoặc kép, giữ nguyên phần còn lại của câu).
Sao lưu bản trước: full_prep/bang_thuat_ngu_v2_phan1_truoc_20_10.json (chỉ lần đầu). Ghi nhật ký: full_prep/doi_nhan_kiem_chung.json.
QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "results/train_v1/translated_sonnet/full_prep"
cfg = json.load(open(ROOT / "config/bench_v1_questions.json", encoding="utf-8"))["factcheck_choice"]
assert "duyệt" in cfg.get("_status", ""), cfg.get("_status")
NEW = {"SUPPORTS": cfg["options"]["supported"], "REFUTES": cfg["options"]["refuted"], "NOT ENOUGH INFO": cfg["options"]["nei"]}
bk = F / "bang_thuat_ngu_v2_phan1_truoc_20_10.json"
if not bk.exists():
    shutil.copy2(F / "bang_thuat_ngu_v2_phan1.json", bk)
p1 = json.load(open(bk, encoding="utf-8"))
OLD = {k: p1[k]["vi"] for k in NEW}
log = []
for k, v in p1.items():
    before = v["vi"]
    if k in NEW:
        v["vi"] = NEW[k]
    elif any(f'"{lab}"' in k for lab in NEW):
        for lab in NEW:
            v["vi"] = v["vi"].replace(f'"{OLD[lab]}"', f'"{NEW[lab]}"')
        assert all(f'"{NEW[lab]}"' in v["vi"] for lab in NEW if f'"{lab}"' in k), k
    if v["vi"] != before:
        v["ghi_chu"] = f"20/10: nhãn kiểm chứng theo bench (người dùng quyết). Trước: {before}"
        log.append({"khoa": k, "truoc": before, "sau": v["vi"]})
(F / "bang_thuat_ngu_v2_phan1.json").write_text(json.dumps(p1, ensure_ascii=False, indent=1), encoding="utf-8")
(F / "doi_nhan_kiem_chung.json").write_text(json.dumps({"nguon": "config/bench_v1_questions.json factcheck_choice.options", "nhan_moi": NEW, "doi": log},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"nhan_moi": NEW, "so_muc_doi": len(log)}, ensure_ascii=False))
