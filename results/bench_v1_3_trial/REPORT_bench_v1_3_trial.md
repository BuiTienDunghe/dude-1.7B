# Dựng thử bench 1.3 (khu tạm, CHƯA đóng băng)

Ngày 2026-10-17. Khu tạm `results/bench_v1_3_trial/`. Không ghi `data/`, không đóng băng, không chạy model nào trên test, không train. Chưa chấm (chờ điều phối kiểm báo cáo này).

**1.3 = 1.2 + đúng các thay đổi sau:**
- **A. Quyết định của người dùng ngày 17/10, trên kết quả soát nhãn mù.** Nguồn: `results/label_review_v1_2/quyet_dinh_28_mo_ho.json`, theo lời người dùng "loại câu 25 và 28, còn lại mơ hồ".
  - Loại 2 cặp ViWikiFC.
  - Gắn cờ mơ hồ cho 26 cặp.
  - Áp đề xuất `lr-07c4a55b9f82`: người dùng sửa rồi duyệt trên công cụ lúc 03:31.
- **B. Lát MLQA riêng: TẠM DỪNG** (mục 1).

## 1. MLQA: tạm dừng, chưa có bản xem xét của người dùng

Quyết định ngày 14/10 (PLAN_decisions, dòng "Chốt phạm vi bench 1.2"): MLQA vào 1.3 "kèm bản xem xét riêng của người dùng trên 30 câu đã có". Đã tìm trong repo và log.

**Có:**
- `results/review/mlqa_test_review.html` và `results/review/mlqa_test_review_sheet.csv`, tạo ngày 14/10 lúc 06:24.
- Đây là 30 câu MLQA test (T-01 … T-30) do agent soạn để người dùng đọc. Mỗi câu có: pair_id, đáp án tiếng Việt, ghi chú máy (ví dụ "nhiễu do LLM chọn; Opus: NHAN_NGUON_NGHI_SAI").

**Không có:**
- Hai cột `verdict` và `ghi_chu` dành cho người dùng **trống ở cả 30/30 dòng**.
- Lớp ghi đè (`data/human_edits/edits_log.jsonl`) và `results/human_review/label_decisions_remote.jsonl` không có quyết định nào cho các câu MLQA. Bản ghi MLQA duy nhất trong lớp ghi đè là việc duyệt chuỗi mẫu câu hỏi `pilot.mlqa.question` (ủy quyền theo agent, 14/10), không phải xem xét 30 câu.
- Thư mục Downloads không có file xem xét nào của người dùng.

Theo brief: **dừng phần MLQA, không tự bịa.** Test 1.3 vì vậy hiện trùng hoàn toàn test 1.2. Khi người dùng có bản xem xét (điền cột verdict/ghi_chu, hoặc gửi trong chat), làm tiếp:
- test: 197 câu ứng viên + `mlqa_test_fixes.jsonl`;
- dev: split validation, cùng script recast;
- các phép kiểm: trùng ViWikiFC, dev↔test, nguồn train_v0, ghim revision, seed.

## 2. Cách dựng và file

**Lớp ghi đè:**
- Dựng từ lớp ghi đè **1.790 bản ghi** (`scripts/bench_v1_3/run_build_v1_3_trial.sh`). Lệnh dựng giống 1.2: `build_revised.py --trial --nfc --provenance --test_labels …`, có chốt kiểm dev↔test.
- Thêm 2 bản ghi so với lúc đóng băng 1.2 (1.786 lúc cập nhật dev): loại 2 cặp ViWikiFC.
  - Do phiên này ghi qua `POST /api/item`, lý do "mơ hồ nên loại", ghi chú dẫn nguyên lời người dùng.
  - id `26293909…` và `d4c4d484…`.
- Bản ghi duyệt lr-07c4 do người dùng ghi trên công cụ lúc 03:31.

**Cờ mơ hồ** (`scripts/bench_v1_3/make_v1_3_trial.py`):
- Trường `"mo_ho": true` trên mọi dòng (EN, VI, VI không dấu) của 26 cặp. Dòng khác không thêm trường (vắng = không mơ hồ), để giữ nguyên byte so với 1.2.
- Câu mơ hồ **vẫn tính trong điểm chính**; báo cáo có thêm lát "bỏ câu mơ hồ".

| File | Dòng | sha256 |
|---|---|---|
| `dev_v1_3_trial.jsonl` | 3.847 | `c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e` |
| `test_v1_3_trial.jsonl` | 4.086 | `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da` |

**Test 1.3 TRÙNG TỪNG BYTE test 1.2 đã đóng băng** (cùng sha `2bdbd210…`): script assert điều này. Lát MLQA của test đang tạm dừng.

## 3. Thay đổi so với 1.2

### Dev (29 cặp)

| Nguồn | Gắn cờ mơ hồ | Loại câu | Sửa nội dung | Cặp đổi |
|---|---|---|---|---|
| SIB-200 | 22 | 0 | 0 | 22 |
| ViWikiFC | 4 | 2 | 0 | 6 |
| Belebele | 0 | 0 | 1 | 1 |
| MASSIVE, XCOPA, ViFactCheck | 0 | 0 | 0 | 0 |
| **Tổng** | **26** | **2** | **1** | **29** |

- **Loại:**
  - `viwikifc:dev:uit_167_11_191_1_32~3810fb04` (câu 25);
  - `viwikifc:dev:uit_484_30_8_3_32~cc7b8660` (câu 28).
  - Nguồn này chỉ có tiếng Việt, nên mỗi cặp bớt 2 dòng. Dev 3.851 → 3.847 dòng.
- **Sửa nội dung:** belebele `…/Airline_alliances#1`, lựa chọn C: "Hãng hàng không có thể cung cấp thêm nhiều quyền lợi" → "Hãng hàng không có thể đang bổ sung thêm quyền lợi mới". Dấu cách cuối có sẵn từ nguồn, kể cả bản EN.
  - Dòng VI và VI không dấu đổi; `vi_revised_fields` thêm `option:2`; `n_tokens` tính lại. Dòng EN không đổi.
- **Gắn cờ mơ hồ:** 26 cặp, chỉ thêm trường `mo_ho`, nội dung không đổi:
  - SIB-200 1158, 1233, 1256, 153, 155, 1689, 1774, 1821, 1883, 1915, 1944, 200, 227, 229, 325, 339, 566, 567, 650, 663, 667, 769;
  - ViWikiFC uit_1014_58_57_1_12, uit_1657_115_2_1_32, uit_24_1_108_1_22, uit_481_28_22_1_21.
- **Không có thay đổi nào khác**: script so từng dòng với `data/bench_v1_2/dev_v1_2.jsonl`.

### Test

Không đổi: trùng từng byte test 1.2.

## 4. Số câu từng lát (số cặp; mỗi cặp = VI có dấu + VI không dấu, cộng EN nếu nguồn có)

| Nguồn | Dev: chính | Dev: bỏ câu mơ hồ | Dev: mơ hồ | Test: chính |
|---|---|---|---|---|
| Belebele | 294 | 294 | 0 | 267 |
| SIB-200 | 299 | 277 | 22 | 119 |
| MASSIVE | 294 | 294 | 0 | 294 |
| XCOPA | 100 | 100 | 0 | 300 |
| ViFactCheck | 158 | 158 | 0 | 291 |
| ViWikiFC | 285 | 281 | 4 | 282 |
| **Tổng** | **1.430** | **1.404** | **26** | **1.553** |
| MLQA (lát riêng) | tạm dừng | — | — | tạm dừng |

Số test theo nguồn đếm từ bản dựng (manifest_v1_3_trial.json); tổng 1.553 cặp, như 1.2.

## 5. Kiểm

| Phép kiểm | Kết quả |
|---|---|
| Test 1.3 trùng byte test 1.2 | ĐẠT (`2bdbd210…`) |
| Rò rỉ dev↔test (cách build_bench_v1_1: khớp chính xác, MinHash ≥ 0,8, 13-gram / câu ngắn nằm trọn) | 0 / 0 |
| Rò rỉ train_v0 + val_v0 | test 2 (như 1.1 và 1.2), dev 0 |
| Bản không dấu sinh từ bản VI hiện hành | 0 lệch (dev 1.430, test 1.553 dòng) |
| Provenance đủ trên mọi dòng, id duy nhất | ĐẠT |
| Đáp án ngược nguồn | dev 10, test 13 (như 1.2) |
| `release_check` | mã 0, 0 đề xuất chờ |

## 6. Kế hoạch chấm (chưa chạy; chờ điều phối kiểm)

- **Chỉ dev.** Dòng không đổi byte, sau khi bỏ riêng trường `mo_ho`, dùng lại preds dev 1.2.
- **Dòng cần chấm lại: 2**, là VI và VI không dấu của Airline_alliances#1 (`rescore/rescore_dev.jsonl`). 4 dòng của 2 cặp loại bị bỏ khỏi điểm.
- **Model:**
  - 7 model trong `scores_v1_2.json`;
  - Qwen3.5-2B gốc (eval_base);
  - Quyet-1.0-Medium (adapter bước E, `.venv-baselines`, offline).
  - Batch 1.
- **Bảng so 1.2 với 1.3:** điểm chính và lát bỏ câu mơ hồ. Lát MLQA chờ người dùng.

## 7. Điểm dev 1.2 so với 1.3 (chỉ dev; batch 1)

Preds dev 1.3: 3845 dòng nội dung không đổi dùng lại preds dev 1.2; 2 dòng chấm lại (VI và VI không dấu của Airline_alliances#1) bằng cả 9 model; 4 dòng của 2 cặp bị loại không còn trong điểm. Accuracy (%) trung bình đều theo nguồn (6 nguồn VI, 4 nguồn EN); cột cuối là trung bình trên mọi câu của 1.3.

### VI có dấu

| Model | 1.2 | 1.3 chính | 1.3 − 1.2 | 1.3 bỏ câu mơ hồ | 1.3 trên mọi câu |
|---|---|---|---|---|---|
| decider-2b-text | 86,3 | 86,5 | +0,2 | 87,5 | 86,9 |
| decider-2b-keyed | 84,9 | 85,1 | +0,2 | 86,1 | 85,6 |
| decider-ft-ep2-text | 84,9 | 85,1 | +0,2 | 86,1 | 85,8 |
| decider-ft-ep2-keyed | 84,5 | 84,7 | +0,2 | 85,7 | 85,5 |
| Qwen3-1.7B-fullvocab-s1 (dự án) | 79,7 | 79,9 | +0,1 | 80,7 | 80,8 |
| Qwen3-1.7B | 70,6 | 70,6 | +0,1 | 71,4 | 70,7 |
| strip-s1 | 80,5 | 80,5 | +0,1 | 81,5 | 81,3 |
| Qwen3.5-2B | 70,6 | 70,6 | +0,1 | 71,3 | 70,4 |
| Quyet-1.0-Medium | 89,4 | 89,6 | +0,2 | 90,6 | 89,2 |

### VI không dấu

| Model | 1.2 | 1.3 chính | 1.3 − 1.2 | 1.3 bỏ câu mơ hồ | 1.3 trên mọi câu |
|---|---|---|---|---|---|
| decider-2b-text | 49,6 | 49,6 | -0,0 | 50,1 | 48,7 |
| decider-2b-keyed | 48,5 | 48,5 | -0,0 | 49,1 | 47,8 |
| decider-ft-ep2-text | 53,7 | 53,8 | +0,1 | 54,2 | 52,2 |
| decider-ft-ep2-keyed | 53,6 | 53,6 | +0,1 | 54,1 | 51,7 |
| Qwen3-1.7B-fullvocab-s1 (dự án) | 62,3 | 62,4 | +0,1 | 63,1 | 62,8 |
| Qwen3-1.7B | 48,5 | 48,5 | -0,0 | 49,0 | 48,3 |
| strip-s1 | 62,5 | 62,5 | +0,0 | 63,3 | 63,1 |
| Qwen3.5-2B | 39,0 | 39,0 | -0,1 | 39,2 | 36,6 |
| Quyet-1.0-Medium | 64,5 | 64,6 | +0,1 | 65,2 | 65,0 |

### EN

| Model | 1.2 | 1.3 chính | 1.3 − 1.2 | 1.3 bỏ câu mơ hồ | 1.3 trên mọi câu |
|---|---|---|---|---|---|
| decider-2b-text | 90,3 | 90,3 | +0,0 | 91,7 | 90,0 |
| decider-2b-keyed | 89,8 | 89,8 | +0,0 | 91,2 | 89,2 |
| decider-ft-ep2-text | 89,8 | 89,8 | +0,0 | 91,2 | 89,4 |
| decider-ft-ep2-keyed | 89,4 | 89,4 | +0,0 | 90,8 | 89,1 |
| Qwen3-1.7B-fullvocab-s1 (dự án) | 88,0 | 88,0 | +0,0 | 89,6 | 87,3 |
| Qwen3-1.7B | 84,6 | 84,6 | +0,0 | 85,9 | 84,3 |
| strip-s1 | 88,3 | 88,3 | +0,0 | 89,9 | 87,5 |
| Qwen3.5-2B | 84,5 | 84,5 | +0,0 | 85,8 | 84,6 |
| Quyet-1.0-Medium | 94,5 | 94,5 | +0,0 | 96,0 | 93,3 |

### Câu mơ hồ (26 cặp): số dòng model chọn đúng đáp án chuẩn

| Model | VI có dấu | VI không dấu | EN |
|---|---|---|---|
| decider-2b-text | 5/26 | 2/26 | 3/22 |
| decider-2b-keyed | 5/26 | 1/26 | 3/22 |
| decider-ft-ep2-text | 4/26 | 3/26 | 3/22 |
| decider-ft-ep2-keyed | 5/26 | 2/26 | 3/22 |
| Qwen3-1.7B-fullvocab-s1 (dự án) | 8/26 | 4/26 | 0/22 |
| Qwen3-1.7B | 7/26 | 4/26 | 3/22 |
| strip-s1 | 6/26 | 4/26 | 1/22 |
| Qwen3.5-2B | 8/26 | 5/26 | 4/22 |
| Quyet-1.0-Medium | 4/26 | 6/26 | 1/22 |

### Theo nguồn, VI có dấu: 1.2 → 1.3 (chỉ ViWikiFC và Belebele có dòng đổi)

| Model | belebele | sib200 | massive | xcopa | vifactcheck | viwikifc |
|---|---|---|---|---|---|---|
| decider-2b-text | 87,8 → 88,1 | 82,3 → 82,3 | 96,3 → 96,3 | 86,0 → 86,0 | 83,5 → 83,5 | 82,2 → 82,8 |
| decider-2b-keyed | 88,4 → 88,8 | 81,3 → 81,3 | 92,2 → 92,2 | 83,0 → 83,0 | 82,9 → 82,9 | 81,9 → 82,5 |
| decider-ft-ep2-text | 87,8 → 88,1 | 81,6 → 81,6 | 94,2 → 94,2 | 83,0 → 83,0 | 80,4 → 80,4 | 82,6 → 83,2 |
| decider-ft-ep2-keyed | 86,7 → 87,1 | 82,3 → 82,3 | 93,2 → 93,2 | 82,0 → 82,0 | 79,7 → 79,7 | 83,3 → 83,9 |
| Qwen3-1.7B-fullvocab-s1 (dự án) | 82,3 → 82,7 | 82,6 → 82,6 | 86,7 → 86,7 | 78,0 → 78,0 | 72,8 → 72,8 | 76,0 → 76,5 |
| Qwen3-1.7B | 75,2 → 75,2 | 75,6 → 75,6 | 75,2 → 75,2 | 80,0 → 80,0 | 57,6 → 57,6 | 59,9 → 60,4 |
| strip-s1 | 81,6 → 82,0 | 82,3 → 82,3 | 89,1 → 89,1 | 80,0 → 80,0 | 73,4 → 73,4 | 76,3 → 76,5 |
| Qwen3.5-2B | 77,9 → 78,2 | 76,3 → 76,3 | 63,9 → 63,9 | 73,0 → 73,0 | 70,3 → 70,3 | 62,0 → 62,1 |
| Quyet-1.0-Medium | 93,9 → 94,2 | 83,6 → 83,6 | 96,6 → 96,6 | 93,0 → 93,0 | 88,6 → 88,6 | 80,8 → 81,4 |

Lát MLQA: tạm dừng (mục 1). Chưa đóng băng.

## 8. Đóng băng (2026-10-17)

Người dùng: "chọn 1, đóng băng 1.3 trước". MLQA dời sang 1.4. Đã dựng lại từ lớp ghi đè (1.790 bản ghi), trùng sha bản thử, và đóng băng vào `data/bench_v1_3/`:
- `dev_v1_3.jsonl`: 3.847 dòng, sha `c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e`;
- `test_v1.3.jsonl`: chỉ đọc, sha `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da`, trùng từng byte test 1.2;
- kèm `FROZEN_test_v1.3.sha256`, `manifest_dev_v1_3.json`, `manifest_test_v1.3.json` (chỉ đọc), `leakage_v1_3.json`.

`data/bench_v1_2/` không đổi (script kiểm sha trước và sau). Code: `scripts/bench_v1_3/run_rebuild_v1_3.sh`, `freeze_v1_3.py`.
