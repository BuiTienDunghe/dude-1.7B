"""Giai đoạn 0 train_v1: thêm (append) 25 đề xuất chuỗi tiếng Việt CHỜ DUYỆT vào config/vi_proposals.jsonl cho phần A:
nhãn ý định vi-tourism-admin-intent (8, nguồn là mã tiếng Anh), nhãn chủ đề VNTC (10, nguồn không dấu), mẫu state + tên môn EXAMS (1 + 6, nguồn tiếng Anh).
Câu hỏi KHÔNG đề xuất mới: dùng lại câu hỏi đã có của bench (translations.json: massive.vi, sib200.vi). Chạy một lần; từ chối nếu id đã có."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "config/vi_proposals.jsonl"
T = "2026-10-17T15:47:00+07:00"
WHO = "agent (phiên Decision model tiếng Việt, Opus 5.5; Giai đoạn 0 train_v1)"
INTENT = {  # mã nhãn -> (đề xuất, nghĩa tiếng Anh theo mẫu câu của nguồn)
    "ask_admin_change": ("hỏi một đơn vị hành chính có bị sáp nhập, chia tách hay đổi địa giới không",
                         "ask whether an administrative unit was merged, split or had its boundaries changed"),
    "ask_location_mapping": ("hỏi một đơn vị hành chính hiện nằm ở đâu, thuộc đơn vị nào", "ask where an administrative unit is now located / which unit it belongs to"),
    "ask_new_name": ("hỏi tên mới của một đơn vị hành chính sau sáp nhập", "ask the new name of an administrative unit after the merger"),
    "ask_old_name": ("hỏi tên cũ hoặc các đơn vị cũ hợp thành một đơn vị hành chính", "ask the old name, or the former units that make up an administrative unit"),
    "compare_before_after_merge": ("so sánh một đơn vị hành chính trước và sau sáp nhập", "compare an administrative unit before and after the merger"),
    "recommend_tourist_spot": ("gợi ý địa điểm du lịch nên đến", "recommend tourist spots to visit"),
    "search_tourist_address": ("tìm địa chỉ hoặc đường đến một địa điểm du lịch", "find the address of / directions to a tourist spot"),
    "search_tourist_by_old_or_new": ("hỏi một địa điểm du lịch thuộc đơn vị hành chính cũ hoặc mới nào",
                                     "ask which old or new administrative unit a tourist spot belongs to"),
}
VNTC = {"Chinh tri Xa hoi": "chính trị - xã hội", "Doi song": "đời sống", "Khoa hoc": "khoa học", "Kinh doanh": "kinh doanh", "Phap luat": "pháp luật",
        "Suc khoe": "sức khỏe", "The gioi": "thế giới", "The thao": "thể thao", "Van hoa": "văn hóa", "Vi tinh": "vi tính"}
SUBJ = {"Physics": "Vật lý", "Biology": "Sinh học", "History": "Lịch sử", "Chemistry": "Hóa học", "Geography": "Địa lý", "Citizenship": "Giáo dục công dân"}
new = []
for k, (vi, en) in INTENT.items():
    new.append({"id": f"tv1-intent-{k}", "khoa": f"train_v1.intent_tourism.label.{k}", "gia_tri_de_xuat": vi, "ban_tieng_anh": f"{k}: {en}",
                "ly_do": "Lựa chọn (nhãn ý định) cho dữ liệu TRAIN phần A, nguồn phamluan/vi-tourism-admin-intent: nguồn chỉ có mã nhãn tiếng Anh. Viết theo kiểu lựa chọn "
                         "MASSIVE tiếng Việt của bench (cụm động từ, chữ thường), mô tả đúng phạm vi mẫu câu của nguồn (đã đọc 4 câu mẫu mỗi nhãn). Câu hỏi dùng lại "
                         "'Người dùng muốn trợ lý làm gì?' của bench (translations.json questions.massive.vi), không đề xuất mới. Không vào benchmark."})
for k, vi in VNTC.items():
    new.append({"id": f"tv1-vntc-{k.lower().replace(' ', '_')}", "khoa": f"train_v1.vntc.label.{k}", "gia_tri_de_xuat": vi, "ban_tieng_anh": f"{k} (nhãn gốc không dấu)",
                "ly_do": "Lựa chọn (chủ đề) cho dữ liệu TRAIN phần A, nguồn kornwtp/VNTC-10Topics: nhãn gốc là tên chuyên mục báo viết không dấu; đề xuất thêm dấu, chữ "
                         "thường như nhãn SIB-200 của bench. Câu hỏi dùng lại 'Văn bản này thuộc chủ đề nào?' (translations.json questions.sib200.vi). Không vào benchmark."})
new.append({"id": "tv1-exams-state", "khoa": "train_v1.exams.state_template", "gia_tri_de_xuat": "Môn {mon}, lớp {lop}",
            "ban_tieng_anh": "Subject {subject}, grade {grade}",
            "ly_do": "State cho dữ liệu TRAIN phần A, nguồn mhardalov/exams (crosslingual_vi, không kèm đoạn văn): đề thi không có văn bản đọc, nên state ghi môn và lớp "
                     "(trường info của nguồn); câu hỏi = câu hỏi đề thi nguyên văn, lựa chọn = đáp án nguyên văn. {mon} lấy từ các đề xuất tv1-exams-subj-*. Không vào benchmark."})
for k, vi in SUBJ.items():
    new.append({"id": f"tv1-exams-subj-{k.lower()}", "khoa": f"train_v1.exams.subject.{k}", "gia_tri_de_xuat": vi, "ban_tieng_anh": k,
                "ly_do": "Tên môn (trường info.subject của EXAMS là tiếng Anh) cho mẫu state tv1-exams-state. Không vào benchmark."})
have = {json.loads(l)["id"] for l in P.read_text(encoding="utf-8").splitlines() if l.strip()}
assert not ({x["id"] for x in new} & have), "đề xuất đã có, không thêm lần hai"
assert P.read_bytes().endswith(b"\n")
with open(P, "a", encoding="utf-8", newline="\n") as f:
    for x in new:
        f.write(json.dumps({**x, "nguoi_de_xuat": WHO, "thoi_diem": T}, ensure_ascii=False) + "\n")
print(len(new), "đề xuất đã thêm")
