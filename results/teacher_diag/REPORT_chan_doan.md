# Chẩn đoán teacher: Quyet-Medium chấm prompt train của C (28/10/2026)

Theo `results/BRIEF_chan_doan_teacher.md`, phần B. Người dùng duyệt "cho chạy chẩn đoán" lúc khoảng 01:15.

**Phạm vi:**
- Không train. Không đụng test.
- Không có prompt dev hay test nào làm đầu vào teacher.
- Không có chữ tiếng Việt mới. Không commit, không xóa file.

## Kết luận, áp đúng bảng đã chốt trước trên phần VI

| Chỉ số (VI, 15.028 dòng) | Giá trị | Ngưỡng |
|---|---|---|
| **A** (argmax teacher trùng nhãn) | **0,811** | ≥ 0,95 → làm mềm nhãn; ≤ 0,90 → thông tin khác |
| **S** (trung bình 1 − p_teacher(nhãn) trên dòng trùng) | **0,138** | ≤ 0,05 → làm mềm nhãn; ≥ 0,10 → thông tin khác |
| D90 (bất đồng mà teacher tự tin > 0,9) | 0,016 | |
| Entropy chuẩn hóa trung bình | 0,424 | |

**→ "Teacher mang thông tin khác nhãn đáng kể": 2a đáng train. Vì A thấp (≤ 0,90), phải soát mù trước, để biết nhãn sai hay teacher sai.**

- Với đích 0,6·teacher + 0,4·nhãn, trên dòng trùng nhau đích chưng cất lệch khỏi nhãn khoảng 0,6·S ≈ 0,083.
- Theo đúng bảng của brief, worker chỉ báo số và áp bảng, không tự quyết bước tiếp.

## Cách chạy

- **Đầu vào:** `results/train_v1/components/C.jsonl` (sha `005ca3d5…`), các dòng `kind ∈ {choice, noul}`: **29.524 dòng** (24.480 choice, 5.044 noul). Đúng văn bản trong file, bản có dấu.
- **Teacher:** `chinhnc/Quyet-1.0-Medium` (revision `bd509d64…`), adapter đã kiểm ở bước E (`scripts/baselines_v1_2/adapters.py`, mapping text).
  - Chạy trong `.venv-baselines`, offline, **batch 1**. Adapter chấm từng câu, nên không có phép kiểm gom batch.
  - `probs` theo đúng thứ tự `options` của dòng C; câu noul ánh xạ Có/Không qua `noul_map` của adapter.
- **Bị loại:** 1 dòng bị từ chối (đọc hiểu; adapter báo hai lựa chọn trùng nhau); 0 dòng bị cắt.
- **Script:** `scripts/teacher_diag/run_quyet_C.py` (chấm), `scripts/teacher_diag/analyze_quyet_C.py` (phân tích, mẫu soát).
- **Thời gian GPU:** 3.316 giây (03:11 → 04:06; trước đó hàng đợi chờ Ollama nhả GPU sau eval đêm của dự án khác). Phần A (FLA) dùng 243 giây. **Tổng brief: 3.559 giây (59 phút)**, dưới trần 90 phút.
- **Nhiệt độ cao nhất:** 78°C ở phần B, 60°C ở phần A. Không lượt nào vượt 85°C.

## Theo ngôn ngữ và kiểu câu

| Nhóm | n | A | S | D90 |
|---|---|---|---|---|
| **VI** | 15.028 | **0,811** | **0,138** | 0,016 |
| EN | 14.154 | 0,688 | 0,210 | 0,022 |
| multi | 342 | 0,675 | 0,225 | 0,018 |
| choice | 24.480 | 0,744 | 0,174 | 0,019 |
| noul | 5.044 | 0,781 | 0,156 | 0,018 |
| Toàn bộ | 29.524 | 0,750 | 0,171 | 0,019 |

## Phần VI theo nhóm dạng bài

| Nhóm (theo brief) | n | A | S | D90 | Entropy |
|---|---|---|---|---|---|
| Đọc hiểu (vsmrc, race, quail, cosmos_qa, dream, exams) | 3.197 | 0,854 | 0,085 | 0,032 | 0,272 |
| Kiểm chứng/NLI (vitaminc, nli_fever, health_fact, vianli, vinli) | 4.276 | 0,751 | 0,199 | 0,007 | 0,606 |
| Ý định (tourism, multidogo, banking77) | 1.991 | 0,754 | 0,145 | 0,012 | 0,344 |
| Nhân quả (e-CARE, headline_cause) | 985 | 0,719 | 0,218 | 0,014 | 0,648 |
| Chủ đề (vntc, vsfc/topic) | 1.676 | 0,888 | 0,089 | 0,027 | 0,222 |
| Còn lại | 2.903 | 0,877 | 0,118 | 0,012 | 0,420 |

**Theo dataset** (≥ 100 dòng, gộp mọi ngôn ngữ của dataset đó), bảng đủ ở `chan_doan.json`.

Thấp nhất:

| Dataset | A | S |
|---|---|---|
| conll2003/ner_tags | 0,40 | |
| math_qa | 0,47 | |
| health_fact | 0,52 | 0,37 |
| headline_cause | 0,63 | |
| quail | 0,65 | |
| vitaminc | 0,68 | 0,26 |

Cao nhất:

| Dataset | A |
|---|---|
| vsmrc/mrc | 0,97 |
| vsfc/sentiment | 0,97 |
| clinc_oos | 0,94 |
| dream | 0,93 |
| visfd/complaint | 0,92 |

**Ghi chú để đọc số:**
- A thấp có thể do **nhãn sai** hoặc do **teacher sai**. Số không phân biệt được hai trường hợp này.
- Nhóm kiểm chứng/NLI và nhân quả có S cao và D90 rất thấp: khi bất đồng với nhãn, teacher hiếm khi tự tin > 0,9.
- health_fact và vitaminc là hai nguồn audit 27/10 đã nêu (health_fact không có văn bản để đọc; vitaminc dùng chữ NEI lệch dev).

## Mẫu để điều phối soát mù

- **File:** `results/teacher_diag/mau_soat_mu.jsonl`.
- **Cách lấy:** 60 dòng VI teacher bất đồng với nhãn mà tự tin > 0,9, chia đều **10 dòng mỗi nhóm** trong 6 nhóm, seed 20261028, thứ tự đã xáo.
- **Kho để lấy** (số dòng VI bất đồng tự tin): đọc hiểu 101, chủ đề 45, còn lại 36, kiểm chứng 28, ý định 23, nhân quả 14.
- **Các cột:** `ma`, `id`, `nhom`, `dataset`, `kind`, `state`, `question`, `options` (có chữ cái). Thêm **hai cột tách riêng `nhan_goc` và `dap_an_teacher`** (kèm `p_teacher`) để che khi soát.

## File

- **Kết quả:** `results/teacher_diag/quyet_C.jsonl` (29.524 dòng: id, dataset/source, text_lang, kind, k, gold_argmax, gold_target, probs, pred, agree, p_gold, p_max, entropy_norm, cờ).
- **Manifest:** `manifest_quyet_C.json` (sha đầu vào, revision Quyet, thời gian).
- **Khác:** `chan_doan.json`, `mau_soat_mu.jsonl`.
- **Log:** `results/G0/logs/30_teacher_quyet_C_0.txt`, `results/G0/logs/temp.log`.
