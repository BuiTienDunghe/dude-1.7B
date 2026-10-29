# Dựng thử bench 1.2 (khu tạm, CHƯA đóng băng)

Bản 1.2 = bench 1.1 sau khi người soát: áp lớp ghi đè `data/human_edits` (đề xuất đã duyệt, sửa câu, loại câu, đổi đáp án dev), quyết định nhãn test (`results/human_review/label_decisions_remote.jsonl`, chỉ dòng split = test), và chuẩn hóa NFC mọi trường tiếng Việt (bản không dấu sinh lại từ bản NFC). MLQA để bản 1.3, Global-PIQA không làm. Không train; không chạy model nào trên test.

- Dựng: `bash scripts/human_edits/run_trial_v1_2.sh` (gọi `build_revised.py --trial --nfc --test_labels …`). Kiểm đồng nhất với lớp ghi đè rỗng: dev trùng byte, test trùng hash 1.1.
- Ra: `dev_v1_2_trial.jsonl` (3854 dòng, sha `0915baef16b4…`), `test_v1_2_trial.jsonl` (4086 dòng, sha `96f990b7a13c…`). Không chmod, không FROZEN, không ghi `data/`. dev/test 1.1 giữ nguyên hash: {'dev_v1_1.jsonl': True, 'test_v1.1.jsonl': True}.
- Nhãn test đã áp: {'loai': 39, 'doi_nhan': 13} (dòng `giu` không đổi gì). Chuỗi mẫu đã sửa: translations.massive_intents.audio_volume_other, translations.massive_intents.takeaway_query. Dòng tiếng Anh không NFC: 0 (bản Anh giữ nguyên).
- `release_check.py`: mã thoát 2. Chuỗi tiếng Việt do agent viết: 78 — chưa duyệt 63, đã duyệt 13, đã sửa 2. / Đề xuất đang chờ trong C:\Users\dungbt06\Fine-tune\config\vi_proposals.jsonl: 0 (chuỗi mẫu 0, câu dev 0, câu test 0; không chặn phát hành; chưa vào benchmark). / Câu đã có chỉnh sửa của người: dev 338, test 301 (test chỉ đổi khi dựng bản 1.2 theo yêu cầu).

## Tóm tắt

**Dựng.** Đạt: kiểm đồng nhất (lớp ghi đè rỗng → trùng byte với 1.1), dev/test 1.1 không đổi hash. 1.2 thử có dev 1.463 → 1.433 cặp (loại 30), test 1.592 → 1.553 cặp (loại 39); đổi ít nhất một thứ ở 461 cặp dev và 498 cặp test (bảng mục 1). `release_check.py` vẫn TỪ CHỐI vì còn 63 chuỗi mẫu đã soát OK nhưng chưa bấm duyệt (không tự duyệt).

**Điểm dev, cả bộ** (tất cả nguồn; 1.1 → 1.2; trong ngoặc: chênh và điểm 1.1 chỉ trên các cặp còn lại):

| Model | EN | VI có dấu | VI không dấu |
|---|---|---|---|
| decider-2b-text | 88.9 → 90.0 (+1.1; còn lại 89.7) | 84.5 → 86.5 (+2.0; còn lại 85.8) | 47.4 → 48.8 (+1.4; còn lại 47.9) |
| decider-2b-keyed | 88.0 → 89.2 (+1.2; còn lại 88.9) | 83.2 → 85.3 (+2.2; còn lại 84.4) | 46.7 → 48.2 (+1.5; còn lại 47.2) |
| decider-ft-ep2-text | 88.0 → 89.4 (+1.4; còn lại 88.9) | 83.5 → 85.4 (+1.9; còn lại 84.8) | 51.3 → 52.1 (+0.9; còn lại 51.8) |
| decider-ft-ep2-keyed | 87.6 → 89.1 (+1.5; còn lại 88.6) | 83.5 → 85.2 (+1.7; còn lại 84.7) | 50.8 → 52.1 (+1.3; còn lại 51.3) |
| Qwen3-1.7B-fullvocab-s1 | 85.9 → 87.3 (+1.4; còn lại 86.8) | 78.5 → 80.2 (+1.7; còn lại 79.6) | 61.6 → 62.7 (+1.1; còn lại 62.1) |
| Qwen3-1.7B | 83.0 → 84.3 (+1.3; còn lại 83.8) | 67.2 → 70.0 (+2.8; còn lại 68.2) | 47.0 → 47.9 (+1.0; còn lại 47.6) |
| strip-s1 | 86.1 → 87.6 (+1.5; còn lại 87.0) | 79.2 → 81.2 (+1.9; còn lại 80.3) | 62.2 → 63.5 (+1.3; còn lại 62.8) |

- **Mọi model đều tăng**, nhiều nhất ở VI có dấu (+1,7 đến +2,8). Phần lớn mức tăng (khoảng một nửa đến hai phần ba ở VI có dấu, gần hết ở EN) đến từ việc loại câu: so cột "còn lại" với 1.1. Phần còn lại đến từ sửa dịch và sửa nhãn. Riêng Qwen3-1.7B gốc thì phần do sửa lớn hơn (VI: +1,0 do loại, +1,8 do sửa). EN cũng tăng vì đổi đáp án và loại câu áp cho cả ba biến thể.
- **Thứ hạng model không đổi**: decider-2b vẫn dẫn ở VI có dấu, adapter đối chứng và strip vẫn dẫn ở VI không dấu, Qwen3-1.7B gốc vẫn thấp nhất ở VI.
- **33 ô (model × nguồn × biến thể) nhảy quá 2 điểm** (mục 4). Gần như đều ở Belebele (sửa đoạn văn; lớn nhất Qwen3-1.7B gốc VI +7,2 với 19 câu từ sai thành đúng) và MASSIVE (đổi đáp án, loại câu, sửa state). Một ô giảm: decider-ft (text và keyed) ở XCOPA VI −3,0, do 3 câu sửa state từ đúng thành sai. **Không ô nào nhảy chỉ vì NFC**: các cặp "chỉ đổi do NFC" không xuất hiện trong danh sách câu gây nhảy.

## 1. Thay đổi theo nguồn và loại

Cách đếm: một cặp (EN + VI + VI không dấu cùng `pair_id`) được đếm ở **mọi** loại nó thuộc (ví dụ sửa state và đổi đáp án). "Chỉ đổi do NFC" là cặp đổi byte nhưng chữ sau NFC không đổi. "Loại câu" không tính thêm loại nào khác. Cột "do chuỗi mẫu": cặp có lựa chọn đổi vì 2 chuỗi mẫu MASSIVE đã sửa.

### dev

| Nguồn | Số cặp 1.1 | đổi đáp án | sửa câu hỏi | sửa lựa chọn | sửa state | loại câu | chỉ đổi do NFC | Cặp có đổi | trong đó do chuỗi mẫu |
|---|---|---|---|---|---|---|---|---|---|
| Belebele | 300 | 2 | 14 | 18 | 185 | 6 | 0 | 197 | 0 |
| MASSIVE | 300 | 2 | 0 | 77 | 24 | 5 | 51 | 151 | 77 |
| SIB-200 | 300 | 1 | 0 | 0 | 61 | 1 | 0 | 63 | 0 |
| ViFactCheck | 163 | 1 | 3 | 0 | 0 | 5 | 3 | 12 | 0 |
| ViWikiFC | 300 | 4 | 7 | 0 | 0 | 13 | 0 | 23 | 0 |
| XCOPA | 100 | 0 | 0 | 6 | 11 | 0 | 0 | 15 | 0 |
| **Tổng** | 1463 | 10 | 24 | 101 | 281 | 30 | 54 | 461 | 77 |

### test

| Nguồn | Số cặp 1.1 | đổi đáp án | sửa câu hỏi | sửa lựa chọn | sửa state | loại câu | chỉ đổi do NFC | Cặp có đổi | trong đó do chuỗi mẫu |
|---|---|---|---|---|---|---|---|---|---|
| Belebele | 272 | 1 | 13 | 15 | 159 | 5 | 0 | 169 | 0 |
| MASSIVE | 300 | 1 | 0 | 95 | 22 | 6 | 46 | 163 | 95 |
| SIB-200 | 120 | 0 | 0 | 0 | 33 | 1 | 0 | 34 | 0 |
| ViFactCheck | 300 | 5 | 11 | 0 | 1 | 9 | 15 | 40 | 0 |
| ViWikiFC | 300 | 6 | 9 | 0 | 2 | 18 | 1 | 33 | 0 |
| XCOPA | 300 | 0 | 0 | 40 | 24 | 0 | 0 | 59 | 0 |
| **Tổng** | 1592 | 13 | 33 | 150 | 241 | 39 | 62 | 498 | 95 |

## 2. Câu test đã xem (ghi trong manifest)

- `test_ids_nguoi_da_xem`: **317** cặp test đã hiện cho người: có đề xuất theo câu (251), có quyết định nhãn từ xa (71), hoặc nằm trong danh sách vấn đề nhãn (label_issues, 70).
- `test_ids_agent_da_doc`: **1592** cặp, tức toàn bộ test 1.1, vì lượt kiểm tra cuối Opus đã đọc hết.

## 3. Điểm DEV 1.2 so với 1.1

Dòng dev trùng byte với 1.1 dùng lại preds; 800 dòng đổi byte (kể cả chỉ do NFC) được chấm lại, cùng điều kiện như lần chấm 1.1 (`scripts/human_edits/rescore_dev_v1_2.sh`); dòng bị loại bỏ khỏi điểm 1.2. Độ đúng tính theo đáp án của từng bản (1.2 dùng đáp án đã sửa). "1.1 trên cặp còn lại" tách tác động của việc loại câu khỏi tác động của việc sửa.

### decider-2b-text

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 87.3 | 88.8 | 88.8 | +1.4 |
| Belebele | vi | 300 → 294 | 83.0 | 84.0 | 86.7 | **+3.7** |
| Belebele | vi không dấu | 300 → 294 | 49.0 | 50.0 | 52.4 | **+3.4** |
| MASSIVE | en | 300 → 295 | 95.0 | 95.9 | 96.6 | +1.6 |
| MASSIVE | vi | 300 → 295 | 93.3 | 94.9 | 96.6 | **+3.3** |
| MASSIVE | vi không dấu | 300 → 295 | 53.7 | 54.2 | 54.9 | +1.2 |
| SIB-200 | en | 300 → 299 | 83.3 | 83.6 | 83.9 | +0.6 |
| SIB-200 | vi | 300 → 299 | 82.3 | 82.6 | 82.3 | -0.1 |
| SIB-200 | vi không dấu | 300 → 299 | 36.0 | 36.1 | 36.1 | +0.1 |
| ViFactCheck | vi | 163 → 158 | 82.8 | 84.2 | 83.5 | +0.7 |
| ViFactCheck | vi không dấu | 163 → 158 | 52.8 | 53.2 | 54.4 | +1.7 |
| ViWikiFC | vi | 300 → 287 | 79.7 | 82.2 | 82.6 | **+2.9** |
| ViWikiFC | vi không dấu | 300 → 287 | 46.0 | 46.7 | 47.0 | +1.0 |
| XCOPA | en | 100 → 100 | 92.0 | 92.0 | 92.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 86.0 | 86.0 | 85.0 | -1.0 |
| XCOPA | vi không dấu | 100 → 100 | 53.0 | 53.0 | 54.0 | +1.0 |
| Tất cả | en | 1000 → 988 | 88.9 | 89.7 | 90.0 | +1.1 |
| Tất cả | vi | 1463 → 1433 | 84.5 | 85.8 | 86.5 | +2.0 |
| Tất cả | vi không dấu | 1463 → 1433 | 47.4 | 47.9 | 48.8 | +1.4 |

### decider-2b-keyed

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 86.7 | 88.1 | 88.1 | +1.4 |
| Belebele | vi | 300 → 294 | 83.3 | 84.4 | 87.4 | **+4.1** |
| Belebele | vi không dấu | 300 → 294 | 49.0 | 50.0 | 52.4 | **+3.4** |
| MASSIVE | en | 300 → 295 | 92.7 | 93.9 | 94.6 | +1.9 |
| MASSIVE | vi | 300 → 295 | 89.0 | 90.5 | 92.2 | **+3.2** |
| MASSIVE | vi không dấu | 300 → 295 | 51.7 | 52.2 | 52.9 | +1.2 |
| SIB-200 | en | 300 → 299 | 83.0 | 83.3 | 83.6 | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.3 | 81.6 | 81.3 | -0.1 |
| SIB-200 | vi không dấu | 300 → 299 | 36.0 | 36.1 | 36.8 | +0.8 |
| ViFactCheck | vi | 163 → 158 | 83.4 | 84.8 | 84.2 | +0.7 |
| ViFactCheck | vi không dấu | 163 → 158 | 53.4 | 53.8 | 54.4 | +1.1 |
| ViWikiFC | vi | 300 → 287 | 79.3 | 81.9 | 82.2 | **+2.9** |
| ViWikiFC | vi không dấu | 300 → 287 | 44.7 | 45.6 | 46.0 | +1.3 |
| XCOPA | en | 100 → 100 | 93.0 | 93.0 | 93.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 82.0 | 82.0 | 82.0 | +0.0 |
| XCOPA | vi không dấu | 100 → 100 | 52.0 | 52.0 | 52.0 | +0.0 |
| Tất cả | en | 1000 → 988 | 88.0 | 88.9 | 89.2 | +1.2 |
| Tất cả | vi | 1463 → 1433 | 83.2 | 84.4 | 85.3 | +2.2 |
| Tất cả | vi không dấu | 1463 → 1433 | 46.7 | 47.2 | 48.2 | +1.5 |

### decider-ft-ep2-text

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 87.0 | 88.4 | 89.1 | **+2.1** |
| Belebele | vi | 300 → 294 | 82.0 | 83.0 | 86.4 | **+4.4** |
| Belebele | vi không dấu | 300 → 294 | 45.0 | 45.2 | 46.9 | +1.9 |
| MASSIVE | en | 300 → 295 | 94.0 | 95.3 | 95.9 | +1.9 |
| MASSIVE | vi | 300 → 295 | 91.3 | 92.9 | 94.6 | **+3.2** |
| MASSIVE | vi không dấu | 300 → 295 | 62.7 | 63.1 | 64.4 | +1.7 |
| SIB-200 | en | 300 → 299 | 81.7 | 81.9 | 82.3 | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.0 | 81.3 | 81.3 | +0.3 |
| SIB-200 | vi không dấu | 300 → 299 | 36.3 | 36.5 | 35.8 | -0.5 |
| ViFactCheck | vi | 163 → 158 | 81.0 | 82.9 | 82.3 | +1.3 |
| ViFactCheck | vi không dấu | 163 → 158 | 63.2 | 64.6 | 63.3 | +0.1 |
| ViWikiFC | vi | 300 → 287 | 81.0 | 83.3 | 82.6 | +1.6 |
| ViWikiFC | vi không dấu | 300 → 287 | 51.7 | 53.0 | 52.6 | +0.9 |
| XCOPA | en | 100 → 100 | 92.0 | 92.0 | 92.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 84.0 | 84.0 | 81.0 | **-3.0** |
| XCOPA | vi không dấu | 100 → 100 | 60.0 | 60.0 | 61.0 | +1.0 |
| Tất cả | en | 1000 → 988 | 88.0 | 88.9 | 89.4 | +1.4 |
| Tất cả | vi | 1463 → 1433 | 83.5 | 84.8 | 85.4 | +1.9 |
| Tất cả | vi không dấu | 1463 → 1433 | 51.3 | 51.8 | 52.1 | +0.9 |

### decider-ft-ep2-keyed

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 86.7 | 88.1 | 88.8 | **+2.1** |
| Belebele | vi | 300 → 294 | 82.0 | 83.0 | 85.4 | **+3.4** |
| Belebele | vi không dấu | 300 → 294 | 42.7 | 42.9 | 46.9 | **+4.3** |
| MASSIVE | en | 300 → 295 | 93.7 | 95.3 | 95.9 | **+2.3** |
| MASSIVE | vi | 300 → 295 | 91.0 | 92.5 | 94.2 | **+3.2** |
| MASSIVE | vi không dấu | 300 → 295 | 62.7 | 63.4 | 63.7 | +1.1 |
| SIB-200 | en | 300 → 299 | 81.3 | 81.6 | 81.9 | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.7 | 81.9 | 82.3 | +0.6 |
| SIB-200 | vi không dấu | 300 → 299 | 34.3 | 34.4 | 34.1 | -0.2 |
| ViFactCheck | vi | 163 → 158 | 79.8 | 81.6 | 81.0 | +1.3 |
| ViFactCheck | vi không dấu | 163 → 158 | 63.8 | 64.6 | 63.9 | +0.1 |
| ViWikiFC | vi | 300 → 287 | 81.0 | 83.3 | 82.6 | +1.6 |
| ViWikiFC | vi không dấu | 300 → 287 | 53.0 | 54.4 | 54.0 | +1.0 |
| XCOPA | en | 100 → 100 | 91.0 | 91.0 | 91.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 84.0 | 84.0 | 81.0 | **-3.0** |
| XCOPA | vi không dấu | 100 → 100 | 61.0 | 61.0 | 62.0 | +1.0 |
| Tất cả | en | 1000 → 988 | 87.6 | 88.6 | 89.1 | +1.5 |
| Tất cả | vi | 1463 → 1433 | 83.5 | 84.7 | 85.2 | +1.7 |
| Tất cả | vi không dấu | 1463 → 1433 | 50.8 | 51.3 | 52.1 | +1.3 |

### Qwen3-1.7B-fullvocab-s1

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 84.7 | 86.1 | 86.7 | **+2.1** |
| Belebele | vi | 300 → 294 | 77.3 | 78.6 | 80.6 | **+3.3** |
| Belebele | vi không dấu | 300 → 294 | 58.3 | 58.5 | 58.5 | +0.2 |
| MASSIVE | en | 300 → 295 | 88.0 | 89.5 | 90.2 | **+2.2** |
| MASSIVE | vi | 300 → 295 | 83.7 | 84.7 | 86.4 | **+2.8** |
| MASSIVE | vi không dấu | 300 → 295 | 63.3 | 64.4 | 66.1 | **+2.8** |
| SIB-200 | en | 300 → 299 | 83.3 | 83.6 | 83.9 | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.7 | 81.9 | 82.6 | +0.9 |
| SIB-200 | vi không dấu | 300 → 299 | 58.3 | 58.5 | 60.2 | +1.9 |
| ViFactCheck | vi | 163 → 158 | 70.6 | 72.2 | 71.5 | +1.0 |
| ViFactCheck | vi không dấu | 163 → 158 | 63.8 | 65.2 | 64.6 | +0.8 |
| ViWikiFC | vi | 300 → 287 | 75.3 | 77.0 | 76.3 | +1.0 |
| ViWikiFC | vi không dấu | 300 → 287 | 67.0 | 67.6 | 66.9 | -0.1 |
| XCOPA | en | 100 → 100 | 91.0 | 91.0 | 91.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 79.0 | 79.0 | 78.0 | -1.0 |
| XCOPA | vi không dấu | 100 → 100 | 56.0 | 56.0 | 57.0 | +1.0 |
| Tất cả | en | 1000 → 988 | 85.9 | 86.8 | 87.3 | +1.4 |
| Tất cả | vi | 1463 → 1433 | 78.5 | 79.6 | 80.2 | +1.7 |
| Tất cả | vi không dấu | 1463 → 1433 | 61.6 | 62.1 | 62.7 | +1.1 |

### Qwen3-1.7B

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 79.0 | 79.9 | 80.6 | +1.6 |
| Belebele | vi | 300 → 294 | 67.3 | 68.4 | 74.5 | **+7.2** |
| Belebele | vi không dấu | 300 → 294 | 46.0 | 46.9 | 48.3 | **+2.3** |
| MASSIVE | en | 300 → 295 | 88.3 | 89.8 | 90.5 | **+2.2** |
| MASSIVE | vi | 300 → 295 | 70.7 | 71.9 | 73.6 | **+2.9** |
| MASSIVE | vi không dấu | 300 → 295 | 43.0 | 43.7 | 44.7 | +1.7 |
| SIB-200 | en | 300 → 299 | 80.7 | 80.9 | 81.3 | +0.6 |
| SIB-200 | vi | 300 → 299 | 74.0 | 74.2 | 75.9 | +1.9 |
| SIB-200 | vi không dấu | 300 → 299 | 51.3 | 51.5 | 51.2 | -0.2 |
| ViFactCheck | vi | 163 → 158 | 55.8 | 57.0 | 56.3 | +0.5 |
| ViFactCheck | vi không dấu | 163 → 158 | 46.0 | 46.8 | 46.2 | +0.2 |
| ViWikiFC | vi | 300 → 287 | 59.0 | 60.3 | 60.3 | +1.3 |
| ViWikiFC | vi không dấu | 300 → 287 | 46.7 | 47.4 | 47.0 | +0.4 |
| XCOPA | en | 100 → 100 | 86.0 | 86.0 | 86.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 79.0 | 79.0 | 78.0 | -1.0 |
| XCOPA | vi không dấu | 100 → 100 | 51.0 | 51.0 | 52.0 | +1.0 |
| Tất cả | en | 1000 → 988 | 83.0 | 83.8 | 84.3 | +1.3 |
| Tất cả | vi | 1463 → 1433 | 67.2 | 68.2 | 70.0 | +2.8 |
| Tất cả | vi không dấu | 1463 → 1433 | 47.0 | 47.6 | 47.9 | +1.0 |

### strip-s1

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 84.3 | 85.7 | 86.4 | **+2.1** |
| Belebele | vi | 300 → 294 | 78.3 | 79.3 | 81.0 | **+2.6** |
| Belebele | vi không dấu | 300 → 294 | 60.7 | 60.9 | 63.6 | **+2.9** |
| MASSIVE | en | 300 → 295 | 89.0 | 90.5 | 91.2 | **+2.2** |
| MASSIVE | vi | 300 → 295 | 86.0 | 87.1 | 89.5 | **+3.5** |
| MASSIVE | vi không dấu | 300 → 295 | 65.0 | 66.1 | 67.1 | **+2.1** |
| SIB-200 | en | 300 → 299 | 83.0 | 83.3 | 83.6 | +0.6 |
| SIB-200 | vi | 300 → 299 | 80.7 | 80.9 | 82.3 | +1.6 |
| SIB-200 | vi không dấu | 300 → 299 | 59.0 | 59.2 | 60.2 | +1.2 |
| ViFactCheck | vi | 163 → 158 | 71.2 | 72.8 | 72.2 | +1.0 |
| ViFactCheck | vi không dấu | 163 → 158 | 62.0 | 63.9 | 63.3 | +1.3 |
| ViWikiFC | vi | 300 → 287 | 76.0 | 77.7 | 77.0 | +1.0 |
| ViWikiFC | vi không dấu | 300 → 287 | 66.0 | 66.6 | 65.2 | -0.8 |
| XCOPA | en | 100 → 100 | 92.0 | 92.0 | 92.0 | +0.0 |
| XCOPA | vi | 100 → 100 | 80.0 | 80.0 | 80.0 | +0.0 |
| XCOPA | vi không dấu | 100 → 100 | 57.0 | 57.0 | 58.0 | +1.0 |
| Tất cả | en | 1000 → 988 | 86.1 | 87.0 | 87.6 | +1.5 |
| Tất cả | vi | 1463 → 1433 | 79.2 | 80.3 | 81.2 | +1.9 |
| Tất cả | vi không dấu | 1463 → 1433 | 62.2 | 62.8 | 63.5 | +1.3 |

## 4. Các ô nhảy hơn 2 điểm và câu gây ra

- **decider-2b-text · Belebele · vi|orig**: 83.0 → 86.7 (1.1 trên cặp còn lại 84.0). Bị loại 6 dòng (đúng ở 1.1: 2). Đổi đúng/sai 14 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: đúng → sai (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Solar_System/Moon#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#2`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/Nadal_beats_Raonic_to_reach_Australian_Open_semis#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/US_President_Trump_announces_troop_withdrawal_from_Syria#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Goma#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Horse_riding#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
- **decider-2b-text · Belebele · vi|nodiac**: 49.0 → 52.4 (1.1 trên cặp còn lại 50.0). Bị loại 6 dòng (đúng ở 1.1: 0). Đổi đúng/sai 23 cặp:
  - `belebele:https://en.wikibooks.org/wiki/Blended_Learning_in_K-12/Blended_Learning_in_Grades_9-12#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/European_History/Revolution_in_France#2`: đúng → sai (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/General_Biology/Genetics/Mutation#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Islam/Early_Islam#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: đúng → sai (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Managing_Groups_and_Teams/How_Do_You_Build_High-performing_Virtual_Teams%3F#1`: đúng → sai (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Solar_System/Moon#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/8.0_magnitude_earthquake_occurs_near_Tonga#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Phone_threat_closes_tunnels,_snarls_traffic_in_Baltimore,_Maryland#2`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/President_of_Costa_Rica_contracts_H1N1_swine_flu#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Sealing_ships_trapped_in_ice_off_coast_of_Newfoundland#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/US_President_Trump_announces_troop_withdrawal_from_Syria#1`: đúng → sai (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Culture_shock#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Staycation#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Thimphu#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Travel_agencies#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Travel_photography/Film#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Vancouver#1`: sai → đúng (sửa state)
- **decider-2b-text · MASSIVE · vi|orig**: 93.3 → 96.6 (1.1 trên cặp còn lại 94.9). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 5 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3183`: sai → đúng (sửa state)
- **decider-2b-text · ViWikiFC · vi|orig**: 79.7 → 82.6 (1.1 trên cặp còn lại 82.2). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 5 cặp:
  - `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: sai → đúng (đổi đáp án)
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: đúng → sai (đổi đáp án)
  - `viwikifc:dev:uit_121_8_1_2_12~ef519358`: đúng → sai (sửa câu hỏi)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: sai → đúng (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: sai → đúng (đổi đáp án)
- **decider-2b-keyed · Belebele · vi|orig**: 83.3 → 87.4 (1.1 trên cặp còn lại 84.4). Bị loại 6 dòng (đúng ở 1.1: 2). Đổi đúng/sai 17 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: đúng → sai (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Solar_System/Moon#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Doctor_to_be_charged_after_mother%27s_death_in_Ohio#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/EBay_to_acquire_Skype#2`: sai → đúng (sửa state, sửa câu hỏi, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#2`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/Hollywood_sign_modified_to_read_%27Hollyweed%27#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Nadal_beats_Raonic_to_reach_Australian_Open_semis#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/US_President_Trump_announces_troop_withdrawal_from_Syria#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Past_events/Wikimania_2018_Cape_Town_Guidebook#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Southern_Africa#2`: sai → đúng (sửa state, sửa lựa chọn)
- **decider-2b-keyed · Belebele · vi|nodiac**: 49.0 → 52.4 (1.1 trên cặp còn lại 50.0). Bị loại 6 dòng (đúng ở 1.1: 0). Đổi đúng/sai 19 cặp:
  - `belebele:https://en.wikibooks.org/wiki/Blended_Learning_in_K-12/Blended_Learning_in_Grades_9-12#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/European_History/Revolution_in_France#2`: đúng → sai (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/General_Biology/Genetics/Mutation#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: đúng → sai (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Solar_System/Moon#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Solar_System/Moon#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Electronic_voting_disputed_in_France#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Hollywood_sign_modified_to_read_%27Hollyweed%27#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Nadal_beats_Raonic_to_reach_Australian_Open_semis#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Sealing_ships_trapped_in_ice_off_coast_of_Newfoundland#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Streymoy#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Travel_agencies#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Travel_photography/Film#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Vancouver#1`: sai → đúng (sửa state)
- **decider-2b-keyed · MASSIVE · vi|orig**: 89.0 → 92.2 (1.1 trên cặp còn lại 90.5). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 5 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3183`: sai → đúng (sửa state)
- **decider-2b-keyed · ViWikiFC · vi|orig**: 79.3 → 82.2 (1.1 trên cặp còn lại 81.9). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 5 cặp:
  - `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: sai → đúng (đổi đáp án)
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: đúng → sai (đổi đáp án)
  - `viwikifc:dev:uit_121_8_1_2_12~ef519358`: đúng → sai (sửa câu hỏi)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: sai → đúng (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: sai → đúng (đổi đáp án)
- **decider-ft-ep2-text · Belebele · en|orig**: 87.0 → 89.1 (1.1 trên cặp còn lại 88.4). Bị loại 6 dòng (đúng ở 1.1: 1). Đổi đúng/sai 2 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
- **decider-ft-ep2-text · Belebele · vi|orig**: 82.0 → 86.4 (1.1 trên cặp còn lại 83.0). Bị loại 6 dòng (đúng ở 1.1: 2). Đổi đúng/sai 16 cặp:
  - `belebele:https://en.wikibooks.org/wiki/Communication_Theory/Nonverbal_Communication#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/Modern_Photography/Composition#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Transport/History#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Burning_debris_from_satellites_spotted_over_several_US_cities#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#2`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/US_President_Trump_announces_troop_withdrawal_from_Syria#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Casablanca#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Grand_Canyon#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Vancouver#1`: sai → đúng (sửa state)
- **decider-ft-ep2-text · MASSIVE · vi|orig**: 91.3 → 94.6 (1.1 trên cặp còn lại 92.9). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 5 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3183`: sai → đúng (sửa state)
- **decider-ft-ep2-text · XCOPA · vi|orig**: 84.0 → 81.0 (1.1 trên cặp còn lại 84.0). Bị loại 0 dòng (đúng ở 1.1: 0). Đổi đúng/sai 3 cặp:
  - `xcopa:19`: đúng → sai (sửa state)
  - `xcopa:33`: đúng → sai (sửa state)
  - `xcopa:79`: đúng → sai (sửa state)
- **decider-ft-ep2-keyed · Belebele · en|orig**: 86.7 → 88.8 (1.1 trên cặp còn lại 88.1). Bị loại 6 dòng (đúng ở 1.1: 1). Đổi đúng/sai 2 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
- **decider-ft-ep2-keyed · Belebele · vi|orig**: 82.0 → 85.4 (1.1 trên cặp còn lại 83.0). Bị loại 6 dòng (đúng ở 1.1: 2). Đổi đúng/sai 19 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Modern_Photography/Composition#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Transport/History#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Doctor_to_be_charged_after_mother%27s_death_in_Ohio#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#2`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/Guinness_turns_250#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Nadal_beats_Raonic_to_reach_Australian_Open_semis#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Saturn%27s_moon_Titan_hosts_liquid_lakes_and_rivers#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/US_President_Trump_announces_troop_withdrawal_from_Syria#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Casablanca#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Cinque_Terre#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Grand_Canyon#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Vancouver#1`: sai → đúng (sửa state)
- **decider-ft-ep2-keyed · Belebele · vi|nodiac**: 42.7 → 46.9 (1.1 trên cặp còn lại 42.9). Bị loại 6 dòng (đúng ở 1.1: 2). Đổi đúng/sai 16 cặp:
  - `belebele:https://en.wikibooks.org/wiki/Animal_Behavior/Population_Genetics#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Animal_Behavior/Population_Genetics#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/Managing_Groups_and_Teams/How_Do_You_Build_High-performing_Virtual_Teams%3F#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/New_Zealand_History/Polynesian_Settlement#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Transport/History#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Transport/History#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Asia/Taiwan#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/EBay_to_acquire_Skype#2`: sai → đúng (sửa state, sửa câu hỏi, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/India%27s_flag_lands_on_Moon#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Khyber_Pakhtunkhwa_Provincial_Assembly_votes_to_merge_with_Pakistani_Federally_Administered_Tribal_Areas#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Microsoft_and_Yahoo_team_up_to_make_IM_clients_compatible#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/National_Hockey_League_news:_March_22,_2008#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Volunteer_travel#2`: đúng → sai (sửa state)
- **decider-ft-ep2-keyed · MASSIVE · en|orig**: 93.7 → 95.9 (1.1 trên cặp còn lại 95.3). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 2 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
- **decider-ft-ep2-keyed · MASSIVE · vi|orig**: 91.0 → 94.2 (1.1 trên cặp còn lại 92.5). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 5 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3183`: sai → đúng (sửa state)
- **decider-ft-ep2-keyed · XCOPA · vi|orig**: 84.0 → 81.0 (1.1 trên cặp còn lại 84.0). Bị loại 0 dòng (đúng ở 1.1: 0). Đổi đúng/sai 3 cặp:
  - `xcopa:19`: đúng → sai (sửa state)
  - `xcopa:33`: đúng → sai (sửa state)
  - `xcopa:79`: đúng → sai (sửa state)
- **Qwen3-1.7B-fullvocab-s1 · Belebele · en|orig**: 84.7 → 86.7 (1.1 trên cặp còn lại 86.1). Bị loại 6 dòng (đúng ở 1.1: 1). Đổi đúng/sai 2 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
- **Qwen3-1.7B-fullvocab-s1 · Belebele · vi|orig**: 77.3 → 80.6 (1.1 trên cặp còn lại 78.6). Bị loại 6 dòng (đúng ở 1.1: 1). Đổi đúng/sai 18 cặp:
  - `belebele:https://en.wikibooks.org/wiki/European_History/Revolution_in_France#2`: đúng → sai (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Big_Cats/Tiger#1`: đúng → sai (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Burning_debris_from_satellites_spotted_over_several_US_cities#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#2`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/Hollywood_sign_modified_to_read_%27Hollyweed%27#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/US_President_Trump_announces_troop_withdrawal_from_Syria#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Buffalo-Pittsburgh_Highway#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Goma#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Hostels#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Past_events/Wikimania_2018_Cape_Town_Guidebook#1`: sai → đúng (sửa state)
- **Qwen3-1.7B-fullvocab-s1 · MASSIVE · en|orig**: 88.0 → 90.2 (1.1 trên cặp còn lại 89.5). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 2 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
- **Qwen3-1.7B-fullvocab-s1 · MASSIVE · vi|orig**: 83.7 → 86.4 (1.1 trên cặp còn lại 84.7). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 7 cặp:
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:1416`: sai → đúng (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3183`: sai → đúng (sửa state)
  - `massive:3637`: đúng → sai (sửa state)
  - `massive:6642`: sai → đúng (sửa state, sửa lựa chọn)
- **Qwen3-1.7B-fullvocab-s1 · MASSIVE · vi|nodiac**: 63.3 → 66.1 (1.1 trên cặp còn lại 64.4). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 7 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:11693`: sai → đúng (sửa lựa chọn)
  - `massive:15207`: sai → đúng (sửa state)
  - `massive:7474`: sai → đúng (sửa state)
  - `massive:9863`: đúng → sai (sửa lựa chọn)
- **Qwen3-1.7B · Belebele · vi|orig**: 67.3 → 74.5 (1.1 trên cặp còn lại 68.4). Bị loại 6 dòng (đúng ở 1.1: 1). Đổi đúng/sai 20 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Asia/Taiwan#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Big_Cats/Lion#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Big_Cats/Tiger#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Burning_debris_from_satellites_spotted_over_several_US_cities#1`: đúng → sai (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/China_promises_cut_in_carbon_dioxide_emissions#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Electronic_voting_disputed_in_France#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Hollywood_sign_modified_to_read_%27Hollyweed%27#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Casablanca#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Hitchhiking_boats#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Hostels#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Past_events/Wikimania_2018_Cape_Town_Guidebook#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Vancouver#2`: sai → đúng (sửa state)
- **Qwen3-1.7B · Belebele · vi|nodiac**: 46.0 → 48.3 (1.1 trên cặp còn lại 46.9). Bị loại 6 dòng (đúng ở 1.1: 0). Đổi đúng/sai 14 cặp:
  - `belebele:https://en.wikibooks.org/wiki/American_Revolution/The_Declaration_of_Independence#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Blended_Learning_in_K-12/Blended_Learning_in_Grades_9-12#1`: sai → đúng (sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/Cognitive_Psychology_and_Cognitive_Neuroscience/Problem_Solving_from_an_Evolutionary_Perspective#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: đúng → sai (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Transport/History#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/World_War_II/Fall_of_East_Africa_and_War_in_North_Africa#1`: đúng → sai (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Dark_matter_lacks_extra_gravitational_force#2`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/West_Indies_wins_the_third_match_of_the_cricket_series_against_India#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Grand_Canyon#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Tornado_safety#2`: sai → đúng (sửa state)
- **Qwen3-1.7B · MASSIVE · en|orig**: 88.3 → 90.5 (1.1 trên cặp còn lại 89.8). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 2 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
- **Qwen3-1.7B · MASSIVE · vi|orig**: 70.7 → 73.6 (1.1 trên cặp còn lại 71.9). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 7 cặp:
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:2851`: sai → đúng (sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state)
  - `massive:3637`: đúng → sai (sửa state)
  - `massive:4694`: sai → đúng (sửa state, sửa lựa chọn)
- **strip-s1 · Belebele · en|orig**: 84.3 → 86.4 (1.1 trên cặp còn lại 85.7). Bị loại 6 dòng (đúng ở 1.1: 1). Đổi đúng/sai 2 cặp:
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: sai → đúng (đổi đáp án, sửa state)
- **strip-s1 · Belebele · vi|orig**: 78.3 → 81.0 (1.1 trên cặp còn lại 79.3). Bị loại 6 dòng (đúng ở 1.1: 2). Đổi đúng/sai 11 cặp:
  - `belebele:https://en.wikibooks.org/wiki/European_History/Revolution_in_France#2`: đúng → sai (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:World_War_II/Battle_of_Britain#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Burning_debris_from_satellites_spotted_over_several_US_cities#1`: sai → đúng (sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/Explorers_find_shipwreck_thought_to_be_massive_WWII_battleship_Musashi#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#2`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikinews.org/wiki/Senate_questions_Roberts_in_confirmation_hearings#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Canyoning#2`: đúng → sai (sửa state, sửa câu hỏi, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Travel_photography/Film#2`: đúng → sai (sửa state)
- **strip-s1 · Belebele · vi|nodiac**: 60.7 → 63.6 (1.1 trên cặp còn lại 60.9). Bị loại 6 dòng (đúng ở 1.1: 3). Đổi đúng/sai 26 cặp:
  - `belebele:https://en.wikibooks.org/wiki/Cognitive_Psychology_and_Cognitive_Neuroscience/Problem_Solving_from_an_Evolutionary_Perspective#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Biology/Nature_of_Science#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: sai → đúng (đổi đáp án)
  - `belebele:https://en.wikibooks.org/wiki/Lucid_Dreaming/Introduction#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Lucid_Dreaming/Introduction#2`: đúng → sai (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikibooks.org/wiki/Transport/History#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:Kings_and_Queens_of_England/The_Anglo-Saxons#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikibooks.org/wiki/Wikijunior:United_States_Charters_of_Freedom/Constitution#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/2009_Shri_Shyam_Bhajan_Hindu_religious_festival_held_in_Hyderabad#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`: sai → đúng (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/France_issues_pollution_alert_as_ship_splits_off_Basque_coast#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Guinness_turns_250#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Khyber_Pakhtunkhwa_Provincial_Assembly_votes_to_merge_with_Pakistani_Federally_Administered_Tribal_Areas#2`: đúng → sai (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Phone_threat_closes_tunnels,_snarls_traffic_in_Baltimore,_Maryland#2`: đúng → sai (sửa state, sửa câu hỏi)
  - `belebele:https://en.wikinews.org/wiki/UK_government_outlines_plans_for_rail_transport#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikinews.org/wiki/Wikimedia_Foundation_addresses_controversial_content_conflict#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Airline_alliances#2`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Canyoning#2`: sai → đúng (sửa state, sửa câu hỏi, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Eye_care#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Hostels#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Past_events/Wikimania_2018_Cape_Town_Guidebook#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/South_Pole#1`: sai → đúng (sửa state, sửa lựa chọn)
  - `belebele:https://en.wikivoyage.org/wiki/Streymoy#1`: sai → đúng (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Travel_photography/Film#1`: đúng → sai (sửa state)
  - `belebele:https://en.wikivoyage.org/wiki/Vancouver#1`: đúng → sai (sửa state)
  - … còn 1 cặp (đủ danh sách ở `scores_v1_2_trial.json`).
- **strip-s1 · MASSIVE · en|orig**: 89.0 → 91.2 (1.1 trên cặp còn lại 90.5). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 2 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
- **strip-s1 · MASSIVE · vi|orig**: 86.0 → 89.5 (1.1 trên cặp còn lại 87.1). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 7 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:15294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3183`: sai → đúng (sửa state)
  - `massive:6642`: sai → đúng (sửa state, sửa lựa chọn)
- **strip-s1 · MASSIVE · vi|nodiac**: 65.0 → 67.1 (1.1 trên cặp còn lại 66.1). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 7 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án)
  - `massive:10457`: sai → đúng (đổi đáp án)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:1234`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state)
  - `massive:3296`: sai → đúng (sửa lựa chọn)
  - `massive:6642`: đúng → sai (sửa state, sửa lựa chọn)

## 5. File

- `dev_v1_2_trial.jsonl`, `test_v1_2_trial.jsonl`, `changes_v1_2_trial.jsonl` (loại thay đổi từng cặp), `build_report_v1_2_trial.json`, `manifest_v1_2_trial.json`.
- `rescore/rescore_dev.jsonl` (dòng dev đổi byte), `eval_rescore/` (preds chấm lại + log), `eval/<model>/preds.jsonl` (preds dev 1.2 đã gộp), `scores_v1_2_trial.json`.
- Code: `scripts/human_edits/build_revised.py` (thêm `--trial`, `--nfc`, `--test_labels`), `run_trial_v1_2.sh`, `rescore_dev_v1_2.sh`, `report_trial_v1_2.py`.

Chưa đóng băng gì. Khi người làm dự án bảo đóng băng: chạy `build_revised.py` không có `--trial` (ghi test mới chỉ đọc + FROZEN) và giữ test 1.1 để đối chiếu.
