# Ghi chú điều phối: bước 0 của kế hoạch phân tích xu hướng (giao ngày 26/10)

Brief: `results/BRIEF_phan_tich_xu_huong.md`. Người dùng duyệt (chat 26/10): "chuẩn bị phép thử đi".

**Phạm vi và ràng buộc:**
- Chỉ dùng **dev 1.3** (`data/bench_v1_3/dev_v1_3.jsonl`, sha c6a9a75d…). Không đụng test.
- Chưa chấm, chưa train. Không commit, không xóa file, không ghi `data/`.
- Mức dùng: cửa sổ 5 giờ từ 88% thì không mở việc mới, 92% dừng hẳn; không dùng mức dùng thêm.
- Ghi `PLAN_decisions`. Báo Quản lý (`SendMessage to "Quản lý"`) khi xong, rồi **DỪNG**.

## Việc

1. **P3 "bỏ văn bản"** (dựng bằng script):
   - nguồn: Belebele, ViFactCheck, ViWikiFC;
   - xóa state; giữ câu hỏi, lựa chọn và đáp án;
   - các bản: VI có dấu, VI không dấu, EN nếu có;
   - ghi mức đoán mò của từng nguồn.
2. **P4 "văn bản không liên quan"** (ViFactCheck, ViWikiFC):
   - mỗi nhận định ghép với state của một câu KHÁC (seed cố định, khác bài; khác chủ đề nếu lọc được bằng script);
   - đáp án = nhãn "không đủ thông tin" của bench;
   - bỏ những cặp mà state mới vô tình chứa thông tin của nhận định: kiểm trùng từ khóa hoặc tên riêng.
3. **P1 "đổi chữ nhãn" và P5 "đổi câu hỏi":** chỉ SOẠN chữ, không tự dùng.
   - **P1:** với mỗi nhãn cố định trong dev, soạn 2 cách diễn đạt cùng nghĩa, khác chữ rõ ràng; không trùng chữ đang dùng trong train_v1 (A/B/C) hay bench. Các nhãn gồm:
     - nhãn kiểm chứng của ViFactCheck và ViWikiFC;
     - tên chủ đề SIB-200;
     - mô tả ý định MASSIVE;
     - Có/Không của câu noul.
   - **P5:** mỗi nguồn 1 câu hỏi khác, cùng nghĩa với câu hỏi mẫu hiện tại.
   - Ghi thành đề xuất loại mẫu vào `config/vi_proposals.jsonl` (append), khóa dạng `probe_v1.P1.<nguồn>.<nhãn>.v1` / `.v2` và `probe_v1.P5.<nguồn>`.
   - **Người dùng tự duyệt**, không tự duyệt.
   - Dùng Opus (tự làm hoặc subagent opus) để soạn.
   - Báo số đề xuất và in bảng chữ đề xuất.
4. **Viết sẵn, chưa chạy:**
   - script dựng file P1/P5 từ giá trị đã duyệt (`gia_tri_moi` mới nhất trong `edits_log`);
   - script chấm một model trên dev + P1–P5 (batch 1, model ngoài chạy offline);
   - script lát cắt theo mục 2 của brief: nguồn gốc so với nguồn dịch, VI/không dấu/EN, kiểu câu choice/noul, số lựa chọn (2 / 3–4 / ≥5), độ dài văn bản, câu đã sửa ở 1.2, cờ `mo_ho`; kèm CI bootstrap và McNemar;
   - ma trận chuyển đổi M1 → M2 theo nguồn;
   - đo trùng lỗi với decider-2b và Quyet;
   - Brier/ECE.

   Thử trên preds đã có của M1 (`results/bench_v1_3_trial/eval/<model>/preds.jsonl`) cho các lát không cần chấm mới.
5. **Kiểm `train_lora.py`** có lưu adapter ở 25/50/75/100% số bước không. Nếu chưa thì thêm tùy chọn, mặc định TẮT. Kiểm bằng chạy giả rất ngắn hoặc đọc code; không train thật.
6. **Mọi file phép thử** để ở `results/probe_v1/`. Manifest ghi rõ: đây là bộ phân tích, KHÔNG phải benchmark, KHÔNG BAO GIỜ dùng để train.
7. **Kiểm P3/P4 không trùng train_v1** (`components/A.jsonl`, `B.jsonl`, `C.jsonl`).

## Báo cáo `results/probe_v1/REPORT_chuan_bi.md`

- số câu mỗi phép thử theo nguồn;
- mức đoán mò;
- ví dụ 3 dòng mỗi phép thử, cắt 200 ký tự;
- danh sách đề xuất chờ duyệt;
- kết quả kiểm script train.

## Bối cảnh hữu ích

- **Dòng dev:** các trường `id`, `pair_id`, `source`, `lang` (vi/en), `variant` (orig/nodiac), `kind` (choice/noul/topic/intent…), `state`, `question`, `options`, `target` (chỉ số đáp án), `mo_ho` (tùy chọn).
- **ViFactCheck / ViWikiFC:** chỉ có VI (không có EN). Câu hỏi choice dạng "...Nhận định: {claim}"; lựa chọn "Được văn bản xác nhận / Bị văn bản bác bỏ / Văn bản không đủ thông tin" (`config/bench_v1_questions.json` → `factcheck_choice`). Câu noul hỏi "Văn bản có xác nhận phát biểu sau không? Phát biểu: {claim}" với lựa chọn Có/Không.
- **Preds M1:** `results/bench_v1_3_trial/eval/Qwen3-1.7B-fullvocab-s1 (dự án)/preds.jsonl`; decider-2b-text và Quyet-1.0-Medium nằm cùng chỗ. Tóm tắt điểm ở `scores_v1_3_trial.json`.
- **Câu đã sửa ở 1.2:** xem `data/bench_v1_3/manifest_dev_v1_3.json` (provenance) hoặc `results/dev_v1_2/`.
- **Bộ chấm:** `scripts/eval_decider_v1.py`, `scripts/eval_base.py`; adapter cho model ngoài ở `scripts/baselines_v1_2/`. Venv WSL: `bash setup/run.sh <script>`.
