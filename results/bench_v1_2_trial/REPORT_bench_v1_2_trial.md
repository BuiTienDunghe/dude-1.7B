# Dựng thử bench 1.2 (khu tạm, CHƯA đóng băng)

Bản 1.2 = bench 1.1 sau khi người soát: áp lớp ghi đè `data/human_edits` (đề xuất đã duyệt, sửa câu, loại câu, đổi đáp án dev), quyết định nhãn test (`results/human_review/label_decisions_remote.jsonl`, chỉ dòng split = test), và chuẩn hóa NFC mọi trường tiếng Việt (bản không dấu sinh lại từ bản NFC). MLQA để bản 1.3, Global-PIQA không làm. Không train; không chạy model nào trên test.

- Dựng: `bash scripts/human_edits/run_trial_v1_2.sh` (gọi `build_revised.py --trial --nfc --test_labels …`). Kiểm đồng nhất với lớp ghi đè rỗng: dev trùng byte, test trùng hash 1.1.
- Ra: `dev_v1_2_trial.jsonl` (3854 dòng, sha `0cccd357779f…`), `test_v1_2_trial.jsonl` (4086 dòng, sha `077323ee9772…`). Không chmod, không FROZEN, không ghi `data/`. dev/test 1.1 giữ nguyên hash: {'dev_v1_1.jsonl': True, 'test_v1.1.jsonl': True}.
- Nhãn test đã áp: {'loai': 39, 'doi_nhan': 13} (dòng `giu` không đổi gì). Chuỗi mẫu đã sửa: bench_v1_questions.factcheck_choice.options.nei, translations.massive_intents.audio_volume_other, translations.massive_intents.lists_query, translations.massive_intents.music_query, translations.massive_intents.news_query, translations.massive_intents.social_post, translations.massive_intents.takeaway_query, translations.massive_intents.transport_query, translations.massive_intents.transport_traffic. Dòng tiếng Anh không NFC: 0 (bản Anh giữ nguyên).
- `release_check.py`: mã thoát 0. Chuỗi tiếng Việt do agent viết: 78 — chưa duyệt 0, đã duyệt 69, đã sửa 9. / Đề xuất đang chờ trong C:\Users\dungbt06\Fine-tune\config\vi_proposals.jsonl: 0 (chuỗi mẫu 0, câu dev 0, câu test 0; không chặn phát hành; chưa vào benchmark). / Câu đã có chỉnh sửa của người: dev 338, test 301 (test chỉ đổi khi dựng bản 1.2 theo yêu cầu).

## Tóm tắt số liệu

| Model | EN 1.1 → 1.2 (còn lại) | VI có dấu 1.1 → 1.2 (còn lại) | VI không dấu 1.1 → 1.2 (còn lại) |
|---|---|---|---|
| decider-2b-text | 88.9 → 90.0 (89.7) | 84.5 → 86.4 (85.8) | 47.4 → 48.6 (47.9) |
| decider-2b-keyed | 88.0 → 89.2 (88.9) | 83.2 → 85.1 (84.4) | 46.7 → 47.7 (47.2) |
| decider-ft-ep2-text | 88.0 → 89.4 (88.9) | 83.5 → 85.1 (84.8) | 51.3 → 52.4 (51.8) |
| decider-ft-ep2-keyed | 87.6 → 89.1 (88.6) | 83.5 → 85.1 (84.7) | 50.8 → 51.8 (51.3) |
| Qwen3-1.7B-fullvocab-s1 | 85.9 → 87.3 (86.8) | 78.5 → 80.3 (79.6) | 61.6 → 62.7 (62.1) |
| Qwen3-1.7B | 83.0 → 84.3 (83.8) | 67.2 → 70.4 (68.2) | 47.0 → 48.0 (47.6) |
| strip-s1 | 86.1 → 87.6 (87.0) | 79.2 → 81.1 (80.3) | 62.2 → 62.9 (62.8) |

Số ô (model × nguồn × biến thể) nhảy quá 2 điểm so với 1.1: 45 (mục 4).

## 1. Thay đổi theo nguồn và loại

Cách đếm: một cặp (EN + VI + VI không dấu cùng `pair_id`) được đếm ở **mọi** loại nó thuộc (ví dụ sửa state và đổi đáp án). "Chỉ đổi do NFC" là cặp đổi byte nhưng chữ sau NFC không đổi. "Loại câu" không tính thêm loại nào khác. Cột "do chuỗi mẫu": cặp có lựa chọn đổi vì 2 chuỗi mẫu MASSIVE đã sửa.

### dev

| Nguồn | Số cặp 1.1 | đổi đáp án | sửa câu hỏi | sửa lựa chọn | sửa state | loại câu | chỉ đổi do NFC | Cặp có đổi | trong đó do chuỗi mẫu |
|---|---|---|---|---|---|---|---|---|---|
| Belebele | 300 | 2 | 14 | 18 | 185 | 6 | 0 | 197 | 0 |
| MASSIVE | 300 | 2 | 0 | 228 | 24 | 5 | 14 | 251 | 228 |
| SIB-200 | 300 | 1 | 0 | 0 | 61 | 1 | 0 | 63 | 0 |
| ViFactCheck | 163 | 1 | 3 | 80 | 0 | 5 | 1 | 87 | 80 |
| ViWikiFC | 300 | 4 | 7 | 162 | 0 | 13 | 0 | 180 | 162 |
| XCOPA | 100 | 0 | 0 | 6 | 11 | 0 | 0 | 15 | 0 |
| **Tổng** | 1463 | 10 | 24 | 494 | 281 | 30 | 15 | 793 | 470 |

### test

| Nguồn | Số cặp 1.1 | đổi đáp án | sửa câu hỏi | sửa lựa chọn | sửa state | loại câu | chỉ đổi do NFC | Cặp có đổi | trong đó do chuỗi mẫu |
|---|---|---|---|---|---|---|---|---|---|
| Belebele | 272 | 1 | 13 | 15 | 159 | 5 | 0 | 169 | 0 |
| MASSIVE | 300 | 1 | 0 | 240 | 22 | 6 | 15 | 269 | 240 |
| SIB-200 | 120 | 0 | 0 | 0 | 33 | 1 | 0 | 34 | 0 |
| ViFactCheck | 300 | 5 | 11 | 135 | 1 | 9 | 6 | 160 | 135 |
| ViWikiFC | 300 | 6 | 9 | 131 | 2 | 18 | 1 | 160 | 131 |
| XCOPA | 300 | 0 | 0 | 40 | 24 | 0 | 0 | 59 | 0 |
| **Tổng** | 1592 | 13 | 33 | 561 | 241 | 39 | 22 | 851 | 506 |

### Theo chuỗi mẫu đã sửa (số cặp có lựa chọn hoặc câu hỏi đổi vì chuỗi đó)

| Khóa chuỗi mẫu | dev | test |
|---|---|---|
| `bench_v1_questions.factcheck_choice.options.nei` | 242 | 266 |
| `translations.massive_intents.audio_volume_other` | 36 | 47 |
| `translations.massive_intents.lists_query` | 51 | 41 |
| `translations.massive_intents.music_query` | 46 | 51 |
| `translations.massive_intents.news_query` | 52 | 52 |
| `translations.massive_intents.social_post` | 57 | 52 |
| `translations.massive_intents.takeaway_query` | 46 | 59 |
| `translations.massive_intents.transport_query` | 54 | 44 |
| `translations.massive_intents.transport_traffic` | 46 | 45 |

## A. Nguồn gốc văn bản và đáp án (provenance)

Mỗi dòng có thêm `gold_provenance`; dòng tiếng Việt có thêm `vi_provenance` và `vi_revised_fields`. NFC và đổi chuỗi mẫu không tính là sửa. Số dòng VI có dấu (bản không dấu giống hệt):

| Tập · nguồn | source | revised | sửa state | sửa câu hỏi | sửa lựa chọn |
|---|---|---|---|---|---|
| dev · belebele | 104 | 190 | 185 | 14 | 18 |
| dev · massive | 271 | 24 | 24 | 0 | 0 |
| dev · sib200 | 238 | 61 | 61 | 0 | 0 |
| dev · vifactcheck | 155 | 3 | 0 | 3 | 0 |
| dev · viwikifc | 280 | 7 | 0 | 7 | 0 |
| dev · xcopa | 85 | 15 | 11 | 0 | 6 |
| test · belebele | 104 | 163 | 159 | 13 | 21 |
| test · massive | 272 | 22 | 22 | 0 | 0 |
| test · sib200 | 86 | 33 | 33 | 0 | 0 |
| test · vifactcheck | 280 | 11 | 1 | 11 | 0 |
| test · viwikifc | 271 | 11 | 2 | 9 | 0 |
| test · xcopa | 241 | 59 | 24 | 0 | 43 |

Đáp án đi ngược nguồn (`gold_provenance = overridden`): 23 cặp (dev 10, test 13):

- dev `belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2`: 0 → 2
- dev `belebele:https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation#1`: 3 → 0
- dev `sib200:242`: 4 → 6
- dev `massive:10058`: 6 → 2
- dev `massive:10457`: 0 → 9
- dev `vifactcheck:dev:4688-18935994`: 0 → 1
- dev `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: 1 → 0
- dev `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: 1 → 0
- dev `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: 0 → 1
- dev `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: 1 → 0
- test `belebele:test:https://en.wikinews.org/wiki/Iranian_passenger_jet%27s_wheel_catches_fire#2`: 2 → 3
- test `massive:test:14412`: 5 → 6
- test `vifactcheck:test:1023-18936279`: 0 → 1
- test `vifactcheck:test:1108-19464299`: 1 → 0
- test `vifactcheck:test:2317-18855494`: 0 → 1
- test `vifactcheck:test:3924-19928721`: 1 → 2
- test `vifactcheck:test:6719-18427936`: 0 → 1
- test `viwikifc:test:uit_1015_58_61_7_21~5cf738de`: 1 → 0
- test `viwikifc:test:uit_1656_114_100_3_12~40f2e4a7`: 0 → 1
- test `viwikifc:test:uit_459_27_139_1_32~4e9470dd`: 2 → 1
- test `viwikifc:test:uit_506_32_26_2_31~e5b0ddb3`: 1 → 0
- test `viwikifc:test:uit_625_37_55_1_11~3afdedf3`: 0 → 1
- test `viwikifc:test:uit_798_40_23_2_31~7b411c24`: 1 → 0

Quy trình (ghi trong manifest): đề xuất do LLM (Opus, Fable) viết, người làm dự án duyệt. Cách duyệt theo lời người làm dự án: "đọc từng câu với đề xuất đổi đáp án và nhãn; đề xuất dịch mà Opus đồng ý thì duyệt theo lô sau khi đọc mẫu, nếu cần thiết thì gọi fable ra". Số liệu từ log (manifest `provenance.so_lieu_tu_log`): 545 bản ghi duyệt đề xuất do người làm dự án bấm từ xa; 132 bản ghi duyệt đề xuất mang ghi chú ủy quyền; 66 bản ghi chuỗi mẫu từ Excel; 143 quyết định nhãn từ xa (đều mang ghi chú ủy quyền). Log không tách được các nhóm 5 trên trang / lô 350 / 6 trong chat / 182 đề xuất kiểm tra cuối; các số đó do điều phối cung cấp và được ghi riêng.

## B. Câu dev đúng → sai chỉ vì sửa chữ

Cặp dev có `vi_provenance = revised`, `gold_provenance = source`, không bị loại, mà model chuyển đúng (1.1) → sai (1.2 thử) ở VI có dấu. Đếm trên 5 model riêng (decider-2b text, decider-ft-ep2 text, đối chứng, Qwen3-1.7B gốc, strip; bản keyed chỉ để tham khảo). Nhóm chính (từ 2 model trở lên, đúng tiêu chí brief): **5** cặp; nhóm phụ (1 model, đưa thêm để đọc): 22 cặp; tổng 27. Theo nguồn: belebele 14, massive 2, sib200 4, vifactcheck 1, viwikifc 1, xcopa 5. 9 cặp có đoạn văn dùng chung (hoàn tác sẽ kéo theo câu cùng đoạn); sửa tay không qua đề xuất: 0.

Nhóm chính: `belebele:https://en.wikibooks.org/wiki/European_History/Revolution_in_France#2` (2), `massive:3637` (2), `sib200:1117` (2), `viwikifc:dev:uit_121_8_1_2_32~3ad67559` (2), `xcopa:19` (4).

Dữ liệu: `results/bench_v1_2_trial/dung_sai_items.json`. Quyết định ghi qua công cụ (màn "Đúng→sai", `POST /api/dung_sai` {pair_id, action: giu_moi | hoan_tac, note}); "hoàn tác" gỡ đúng các đề xuất đã áp (mới nhất trước), đánh dấu từ chối, câu cùng đoạn về bản cũ theo; có sửa chồng lên thì báo lỗi, không ghi.

## C. Kỹ thuật

- Bỏ 3 trường provenance thì bản dựng này trùng hash bản thử trước: dev `c12b380ae819…`, test `d17b4c893f97…`.
- Bản không dấu sinh từ bản VI hiện hành (strip_diacritics trên state, câu hỏi, lựa chọn; cùng đáp án): khớp {'dev': 1433, 'test': 1553} dòng, trong đó revised {'dev': 300, 'test': 299}; không khớp: 0.
- Rò rỉ train_v0 + val_v0 (3 phép như lần dựng 1.1: khớp chính xác, MinHash ≥ 0,8, 13-gram / câu ngắn nằm trọn; chỉ mục là state của bench, quét state của train):

| Bản · tập | Số lần trùng theo nguồn và phép |
|---|---|
| 1.1 · dev | 0 |
| 1.1 · test | vifactcheck|vi|contain13: 2 |
| 1.2 · dev | 0 |
| 1.2 · test | vifactcheck|vi|contain13: 2 |

## 2. Câu test đã xem (ghi trong manifest)

- `test_ids_nguoi_da_xem`: **317** cặp test đã hiện cho người: có đề xuất theo câu (251), có quyết định nhãn từ xa (71), hoặc nằm trong danh sách vấn đề nhãn (label_issues, 70).
- `test_ids_agent_da_doc`: **1592** cặp, tức toàn bộ test 1.1, vì lượt kiểm tra cuối Opus đã đọc hết.

## 3. Điểm DEV 1.2 so với 1.1

Dòng dev trùng byte với 1.1 dùng lại preds; 1507 dòng đổi byte (kể cả chỉ do NFC) được chấm lại, cùng điều kiện như lần chấm 1.1 (`scripts/human_edits/rescore_dev_v1_2.sh`); dòng bị loại bỏ khỏi điểm 1.2. Độ đúng tính theo đáp án của từng bản (1.2 dùng đáp án đã sửa). "1.1 trên cặp còn lại" tách tác động của việc loại câu khỏi tác động của việc sửa.

### decider-2b-text

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 87.3 | 88.8 | 88.8 | 88.3 (103) | +1.4 |
| Belebele | vi | 300 → 294 | 83.0 | 84.0 | 86.7 | 87.4 (103) | **+3.7** |
| Belebele | vi không dấu | 300 → 294 | 49.0 | 50.0 | 52.4 | 53.4 (103) | **+3.4** |
| MASSIVE | en | 300 → 295 | 95.0 | 95.9 | 96.6 | 97.0 (269) | +1.6 |
| MASSIVE | vi | 300 → 295 | 93.3 | 94.9 | 96.3 | 96.3 (269) | **+2.9** |
| MASSIVE | vi không dấu | 300 → 295 | 53.7 | 54.2 | 53.9 | 53.9 (269) | +0.2 |
| SIB-200 | en | 300 → 299 | 83.3 | 83.6 | 83.9 | 83.1 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 82.3 | 82.6 | 82.3 | 81.0 (237) | -0.1 |
| SIB-200 | vi không dấu | 300 → 299 | 36.0 | 36.1 | 36.1 | 35.9 (237) | +0.1 |
| ViFactCheck | vi | 163 → 158 | 82.8 | 84.2 | 83.5 | 84.4 (154) | +0.7 |
| ViFactCheck | vi không dấu | 163 → 158 | 52.8 | 53.2 | 53.2 | 53.9 (154) | +0.4 |
| ViWikiFC | vi | 300 → 287 | 79.7 | 82.2 | 82.2 | 82.7 (277) | **+2.6** |
| ViWikiFC | vi không dấu | 300 → 287 | 46.0 | 46.7 | 48.1 | 47.3 (277) | **+2.1** |
| XCOPA | en | 100 → 100 | 92.0 | 92.0 | 92.0 | 91.8 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 86.0 | 86.0 | 85.0 | 85.9 (85) | -1.0 |
| XCOPA | vi không dấu | 100 → 100 | 53.0 | 53.0 | 54.0 | 57.6 (85) | +1.0 |
| Tất cả | en | 1000 → 988 | 88.9 | 89.7 | 90.0 | 90.3 (694) | +1.1 |
| Tất cả | vi | 1463 → 1433 | 84.5 | 85.8 | 86.4 | 86.5 (1125) | +1.9 |
| Tất cả | vi không dấu | 1463 → 1433 | 47.4 | 47.9 | 48.6 | 48.7 (1125) | +1.3 |

### decider-2b-keyed

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 86.7 | 88.1 | 88.1 | 88.3 (103) | +1.4 |
| Belebele | vi | 300 → 294 | 83.3 | 84.4 | 87.4 | 87.4 (103) | **+4.1** |
| Belebele | vi không dấu | 300 → 294 | 49.0 | 50.0 | 52.4 | 55.3 (103) | **+3.4** |
| MASSIVE | en | 300 → 295 | 92.7 | 93.9 | 94.6 | 94.8 (269) | +1.9 |
| MASSIVE | vi | 300 → 295 | 89.0 | 90.5 | 92.2 | 92.2 (269) | **+3.2** |
| MASSIVE | vi không dấu | 300 → 295 | 51.7 | 52.2 | 51.2 | 51.3 (269) | -0.5 |
| SIB-200 | en | 300 → 299 | 83.0 | 83.3 | 83.6 | 83.1 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.3 | 81.6 | 81.3 | 79.7 (237) | -0.1 |
| SIB-200 | vi không dấu | 300 → 299 | 36.0 | 36.1 | 36.8 | 36.7 (237) | +0.8 |
| ViFactCheck | vi | 163 → 158 | 83.4 | 84.8 | 82.9 | 83.8 (154) | -0.5 |
| ViFactCheck | vi không dấu | 163 → 158 | 53.4 | 53.8 | 51.3 | 51.9 (154) | **-2.1** |
| ViWikiFC | vi | 300 → 287 | 79.3 | 81.9 | 81.9 | 82.7 (277) | **+2.5** |
| ViWikiFC | vi không dấu | 300 → 287 | 44.7 | 45.6 | 47.4 | 46.6 (277) | **+2.7** |
| XCOPA | en | 100 → 100 | 93.0 | 93.0 | 93.0 | 91.8 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 82.0 | 82.0 | 82.0 | 82.4 (85) | +0.0 |
| XCOPA | vi không dấu | 100 → 100 | 52.0 | 52.0 | 52.0 | 55.3 (85) | +0.0 |
| Tất cả | en | 1000 → 988 | 88.0 | 88.9 | 89.2 | 89.5 (694) | +1.2 |
| Tất cả | vi | 1463 → 1433 | 83.2 | 84.4 | 85.1 | 84.9 (1125) | +2.0 |
| Tất cả | vi không dấu | 1463 → 1433 | 46.7 | 47.2 | 47.7 | 47.8 (1125) | +1.0 |

### decider-ft-ep2-text

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 87.0 | 88.4 | 89.1 | 93.2 (103) | **+2.1** |
| Belebele | vi | 300 → 294 | 82.0 | 83.0 | 86.4 | 88.3 (103) | **+4.4** |
| Belebele | vi không dấu | 300 → 294 | 45.0 | 45.2 | 46.9 | 52.4 (103) | +1.9 |
| MASSIVE | en | 300 → 295 | 94.0 | 95.3 | 95.9 | 95.9 (269) | +1.9 |
| MASSIVE | vi | 300 → 295 | 91.3 | 92.9 | 94.2 | 94.4 (269) | **+2.9** |
| MASSIVE | vi không dấu | 300 → 295 | 62.7 | 63.1 | 64.7 | 65.4 (269) | **+2.1** |
| SIB-200 | en | 300 → 299 | 81.7 | 81.9 | 82.3 | 80.6 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.0 | 81.3 | 81.3 | 79.3 (237) | +0.3 |
| SIB-200 | vi không dấu | 300 → 299 | 36.3 | 36.5 | 35.8 | 35.9 (237) | -0.5 |
| ViFactCheck | vi | 163 → 158 | 81.0 | 82.9 | 80.4 | 81.8 (154) | -0.6 |
| ViFactCheck | vi không dấu | 163 → 158 | 63.2 | 64.6 | 60.8 | 62.3 (154) | **-2.4** |
| ViWikiFC | vi | 300 → 287 | 81.0 | 83.3 | 82.6 | 83.4 (277) | +1.6 |
| ViWikiFC | vi không dấu | 300 → 287 | 51.7 | 53.0 | 55.1 | 54.9 (277) | **+3.4** |
| XCOPA | en | 100 → 100 | 92.0 | 92.0 | 92.0 | 90.6 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 84.0 | 84.0 | 81.0 | 83.5 (85) | **-3.0** |
| XCOPA | vi không dấu | 100 → 100 | 60.0 | 60.0 | 61.0 | 62.4 (85) | +1.0 |
| Tất cả | en | 1000 → 988 | 88.0 | 88.9 | 89.4 | 89.6 (694) | +1.4 |
| Tất cả | vi | 1463 → 1433 | 83.5 | 84.8 | 85.1 | 85.4 (1125) | +1.6 |
| Tất cả | vi không dấu | 1463 → 1433 | 51.3 | 51.8 | 52.4 | 54.8 (1125) | +1.1 |

### decider-ft-ep2-keyed

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 86.7 | 88.1 | 88.8 | 91.3 (103) | **+2.1** |
| Belebele | vi | 300 → 294 | 82.0 | 83.0 | 85.4 | 88.3 (103) | **+3.4** |
| Belebele | vi không dấu | 300 → 294 | 42.7 | 42.9 | 46.9 | 48.5 (103) | **+4.3** |
| MASSIVE | en | 300 → 295 | 93.7 | 95.3 | 95.9 | 95.9 (269) | **+2.3** |
| MASSIVE | vi | 300 → 295 | 91.0 | 92.5 | 93.6 | 93.7 (269) | **+2.6** |
| MASSIVE | vi không dấu | 300 → 295 | 62.7 | 63.4 | 62.7 | 63.6 (269) | +0.0 |
| SIB-200 | en | 300 → 299 | 81.3 | 81.6 | 81.9 | 81.0 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.7 | 81.9 | 82.3 | 80.2 (237) | +0.6 |
| SIB-200 | vi không dấu | 300 → 299 | 34.3 | 34.4 | 34.1 | 34.6 (237) | -0.2 |
| ViFactCheck | vi | 163 → 158 | 79.8 | 81.6 | 79.7 | 81.2 (154) | -0.0 |
| ViFactCheck | vi không dấu | 163 → 158 | 63.8 | 64.6 | 62.0 | 63.6 (154) | -1.8 |
| ViWikiFC | vi | 300 → 287 | 81.0 | 83.3 | 83.3 | 84.1 (277) | **+2.3** |
| ViWikiFC | vi không dấu | 300 → 287 | 53.0 | 54.4 | 55.1 | 54.9 (277) | **+2.1** |
| XCOPA | en | 100 → 100 | 91.0 | 91.0 | 91.0 | 89.4 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 84.0 | 84.0 | 81.0 | 83.5 (85) | **-3.0** |
| XCOPA | vi không dấu | 100 → 100 | 61.0 | 61.0 | 62.0 | 63.5 (85) | +1.0 |
| Tất cả | en | 1000 → 988 | 87.6 | 88.6 | 89.1 | 89.3 (694) | +1.5 |
| Tất cả | vi | 1463 → 1433 | 83.5 | 84.7 | 85.1 | 85.5 (1125) | +1.6 |
| Tất cả | vi không dấu | 1463 → 1433 | 50.8 | 51.3 | 51.8 | 54.0 (1125) | +1.1 |

### Qwen3-1.7B-fullvocab-s1

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 84.7 | 86.1 | 86.7 | 89.3 (103) | **+2.1** |
| Belebele | vi | 300 → 294 | 77.3 | 78.6 | 80.6 | 79.6 (103) | **+3.3** |
| Belebele | vi không dấu | 300 → 294 | 58.3 | 58.5 | 58.5 | 66.0 (103) | +0.2 |
| MASSIVE | en | 300 → 295 | 88.0 | 89.5 | 90.2 | 90.0 (269) | **+2.2** |
| MASSIVE | vi | 300 → 295 | 83.7 | 84.7 | 86.8 | 87.4 (269) | **+3.1** |
| MASSIVE | vi không dấu | 300 → 295 | 63.3 | 64.4 | 66.4 | 66.9 (269) | **+3.1** |
| SIB-200 | en | 300 → 299 | 83.3 | 83.6 | 83.9 | 83.5 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 81.7 | 81.9 | 82.6 | 81.4 (237) | +0.9 |
| SIB-200 | vi không dấu | 300 → 299 | 58.3 | 58.5 | 60.2 | 58.6 (237) | +1.9 |
| ViFactCheck | vi | 163 → 158 | 70.6 | 72.2 | 72.8 | 73.4 (154) | **+2.2** |
| ViFactCheck | vi không dấu | 163 → 158 | 63.8 | 65.2 | 65.2 | 66.2 (154) | +1.4 |
| ViWikiFC | vi | 300 → 287 | 75.3 | 77.0 | 76.0 | 77.3 (277) | +0.6 |
| ViWikiFC | vi không dấu | 300 → 287 | 67.0 | 67.6 | 66.2 | 66.4 (277) | -0.8 |
| XCOPA | en | 100 → 100 | 91.0 | 91.0 | 91.0 | 92.9 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 79.0 | 79.0 | 78.0 | 78.8 (85) | -1.0 |
| XCOPA | vi không dấu | 100 → 100 | 56.0 | 56.0 | 57.0 | 56.5 (85) | +1.0 |
| Tất cả | en | 1000 → 988 | 85.9 | 86.8 | 87.3 | 88.0 (694) | +1.4 |
| Tất cả | vi | 1463 → 1433 | 78.5 | 79.6 | 80.3 | 80.4 (1125) | +1.9 |
| Tất cả | vi không dấu | 1463 → 1433 | 61.6 | 62.1 | 62.7 | 64.1 (1125) | +1.1 |

### Qwen3-1.7B

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 79.0 | 79.9 | 80.6 | 85.4 (103) | +1.6 |
| Belebele | vi | 300 → 294 | 67.3 | 68.4 | 74.5 | 73.8 (103) | **+7.2** |
| Belebele | vi không dấu | 300 → 294 | 46.0 | 46.9 | 48.3 | 53.4 (103) | **+2.3** |
| MASSIVE | en | 300 → 295 | 88.3 | 89.8 | 90.5 | 90.3 (269) | **+2.2** |
| MASSIVE | vi | 300 → 295 | 70.7 | 71.9 | 75.3 | 75.1 (269) | **+4.6** |
| MASSIVE | vi không dấu | 300 → 295 | 43.0 | 43.7 | 45.1 | 45.0 (269) | **+2.1** |
| SIB-200 | en | 300 → 299 | 80.7 | 80.9 | 81.3 | 81.0 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 74.0 | 74.2 | 75.9 | 75.9 (237) | +1.9 |
| SIB-200 | vi không dấu | 300 → 299 | 51.3 | 51.5 | 51.2 | 50.6 (237) | -0.2 |
| ViFactCheck | vi | 163 → 158 | 55.8 | 57.0 | 57.6 | 59.1 (154) | +1.8 |
| ViFactCheck | vi không dấu | 163 → 158 | 46.0 | 46.8 | 44.9 | 46.1 (154) | -1.1 |
| ViWikiFC | vi | 300 → 287 | 59.0 | 60.3 | 59.9 | 60.3 (277) | +0.9 |
| ViWikiFC | vi không dấu | 300 → 287 | 46.7 | 47.4 | 47.7 | 47.7 (277) | +1.1 |
| XCOPA | en | 100 → 100 | 86.0 | 86.0 | 86.0 | 85.9 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 79.0 | 79.0 | 78.0 | 78.8 (85) | -1.0 |
| XCOPA | vi không dấu | 100 → 100 | 51.0 | 51.0 | 52.0 | 49.4 (85) | +1.0 |
| Tất cả | en | 1000 → 988 | 83.0 | 83.8 | 84.3 | 85.9 (694) | +1.3 |
| Tất cả | vi | 1463 → 1433 | 67.2 | 68.2 | 70.4 | 69.6 (1125) | +3.2 |
| Tất cả | vi không dấu | 1463 → 1433 | 47.0 | 47.6 | 48.0 | 48.1 (1125) | +1.1 |

### strip-s1

| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |
|---|---|---|---|---|---|---|---|
| Belebele | en | 300 → 294 | 84.3 | 85.7 | 86.4 | 86.4 (103) | **+2.1** |
| Belebele | vi | 300 → 294 | 78.3 | 79.3 | 81.0 | 78.6 (103) | **+2.6** |
| Belebele | vi không dấu | 300 → 294 | 60.7 | 60.9 | 63.6 | 68.9 (103) | **+2.9** |
| MASSIVE | en | 300 → 295 | 89.0 | 90.5 | 91.2 | 91.1 (269) | **+2.2** |
| MASSIVE | vi | 300 → 295 | 86.0 | 87.1 | 89.2 | 88.8 (269) | **+3.2** |
| MASSIVE | vi không dấu | 300 → 295 | 65.0 | 66.1 | 67.5 | 67.7 (269) | **+2.5** |
| SIB-200 | en | 300 → 299 | 83.0 | 83.3 | 83.6 | 83.1 (237) | +0.6 |
| SIB-200 | vi | 300 → 299 | 80.7 | 80.9 | 82.3 | 80.2 (237) | +1.6 |
| SIB-200 | vi không dấu | 300 → 299 | 59.0 | 59.2 | 60.2 | 58.2 (237) | +1.2 |
| ViFactCheck | vi | 163 → 158 | 71.2 | 72.8 | 73.4 | 74.0 (154) | **+2.3** |
| ViFactCheck | vi không dấu | 163 → 158 | 62.0 | 63.9 | 62.7 | 63.6 (154) | +0.7 |
| ViWikiFC | vi | 300 → 287 | 76.0 | 77.7 | 76.3 | 77.6 (277) | +0.3 |
| ViWikiFC | vi không dấu | 300 → 287 | 66.0 | 66.6 | 62.0 | 62.5 (277) | **-4.0** |
| XCOPA | en | 100 → 100 | 92.0 | 92.0 | 92.0 | 94.1 (85) | +0.0 |
| XCOPA | vi | 100 → 100 | 80.0 | 80.0 | 80.0 | 81.2 (85) | +0.0 |
| XCOPA | vi không dấu | 100 → 100 | 57.0 | 57.0 | 58.0 | 58.8 (85) | +1.0 |
| Tất cả | en | 1000 → 988 | 86.1 | 87.0 | 87.6 | 88.0 (694) | +1.5 |
| Tất cả | vi | 1463 → 1433 | 79.2 | 80.3 | 81.1 | 80.7 (1125) | +1.9 |
| Tất cả | vi không dấu | 1463 → 1433 | 62.2 | 62.8 | 62.9 | 63.3 (1125) | +0.7 |

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
- **decider-2b-text · MASSIVE · vi|orig**: 93.3 → 96.3 (1.1 trên cặp còn lại 94.9). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 8 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:12006`: đúng → sai (sửa lựa chọn)
  - `massive:12740`: sai → đúng (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3404`: đúng → sai (sửa lựa chọn)
- **decider-2b-text · ViWikiFC · vi|orig**: 79.7 → 82.2 (1.1 trên cặp còn lại 82.2). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 8 cặp:
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: sai → đúng (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_191_12_107_1_12~6682f2fa`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_29_2_15_2_22~a2cb69c5`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_436_27_79_2_31~7a5287ad`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_631_37_78_1_22~bbdee907`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_635_37_96_1_12~82a26480`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: sai → đúng (đổi đáp án)
- **decider-2b-text · ViWikiFC · vi|nodiac**: 46.0 → 48.1 (1.1 trên cặp còn lại 46.7). Bị loại 13 dòng (đúng ở 1.1: 4). Đổi đúng/sai 12 cặp:
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_112_6_5_2_11~b2ff6a60`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: đúng → sai (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_187_12_67_1_12~029fbb3e`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_199_13_27_1_22~0458dacc`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_2581_154_297_1_22~c73f5673`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_530_33_78_2_11~1008f201`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_845_44_52_6_11~32808231`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_862_44_156_1_31~8aaf5903`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: sai → đúng (đổi đáp án)
  - `viwikifc:dev:uit_968_55_24_1_21~d0dbaa9c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_969_55_34_3_32~809a6240`: sai → đúng (sửa lựa chọn)
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
- **decider-2b-keyed · MASSIVE · vi|orig**: 89.0 → 92.2 (1.1 trên cặp còn lại 90.5). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 9 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:11745`: sai → đúng (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2955`: sai → đúng (sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3404`: đúng → sai (sửa lựa chọn)
  - `massive:9447`: đúng → sai (sửa lựa chọn)
- **decider-2b-keyed · ViFactCheck · vi|nodiac**: 53.4 → 51.3 (1.1 trên cặp còn lại 53.8). Bị loại 5 dòng (đúng ở 1.1: 2). Đổi đúng/sai 8 cặp:
  - `vifactcheck:dev:1204-19157247`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:4243-18811997`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:4688-18935994`: sai → đúng (đổi đáp án)
  - `vifactcheck:dev:5295-19153545`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:6368-18853998`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:6629-19807224`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:6853-18813865`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:749-19049365`: đúng → sai (sửa lựa chọn)
- **decider-2b-keyed · ViWikiFC · vi|orig**: 79.3 → 81.9 (1.1 trên cặp còn lại 81.9). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 8 cặp:
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_121_8_1_2_12~ef519358`: đúng → sai (sửa câu hỏi, sửa lựa chọn)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: sai → đúng (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_191_12_107_1_12~6682f2fa`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_29_2_15_2_22~a2cb69c5`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_436_27_79_2_31~7a5287ad`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_631_37_78_1_22~bbdee907`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: sai → đúng (đổi đáp án)
- **decider-2b-keyed · ViWikiFC · vi|nodiac**: 44.7 → 47.4 (1.1 trên cặp còn lại 45.6). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 13 cặp:
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: đúng → sai (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_189_12_101_1_21~36d8861a`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_189_12_101_1_31~cb5d4d62`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_2731_163_26_1_31~977b229b`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_356_22_28_1_11~18e9da5c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_530_33_78_3_21~ce8ef37c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_818_41_96_6_11~27d58d7c`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_840_44_41_1_32~9ce1c7db`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: sai → đúng (đổi đáp án)
  - `viwikifc:dev:uit_967_55_15_3_21~cec03e8c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_968_55_24_1_21~d0dbaa9c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_969_55_34_3_32~809a6240`: sai → đúng (sửa lựa chọn)
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
- **decider-ft-ep2-text · MASSIVE · vi|orig**: 91.3 → 94.2 (1.1 trên cặp còn lại 92.9). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 10 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10605`: đúng → sai (sửa lựa chọn)
  - `massive:11045`: đúng → sai (sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:12408`: sai → đúng (sửa lựa chọn)
  - `massive:12740`: sai → đúng (sửa lựa chọn)
  - `massive:15993`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
- **decider-ft-ep2-text · MASSIVE · vi|nodiac**: 62.7 → 64.7 (1.1 trên cặp còn lại 63.1). Bị loại 5 dòng (đúng ở 1.1: 2). Đổi đúng/sai 21 cặp:
  - `massive:10186`: đúng → sai (sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10697`: đúng → sai (sửa lựa chọn)
  - `massive:11045`: đúng → sai (sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:12134`: đúng → sai (sửa lựa chọn)
  - `massive:12258`: sai → đúng (sửa lựa chọn)
  - `massive:12318`: sai → đúng (sửa lựa chọn)
  - `massive:12408`: sai → đúng (sửa lựa chọn)
  - `massive:12674`: sai → đúng (sửa lựa chọn)
  - `massive:12740`: sai → đúng (sửa lựa chọn)
  - `massive:15471`: sai → đúng (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2946`: đúng → sai (sửa lựa chọn)
  - `massive:3296`: sai → đúng (sửa lựa chọn)
  - `massive:4287`: đúng → sai (sửa lựa chọn)
  - `massive:53`: sai → đúng (sửa lựa chọn)
  - `massive:5621`: đúng → sai (sửa lựa chọn)
  - `massive:851`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:8752`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:9673`: sai → đúng (sửa lựa chọn)
- **decider-ft-ep2-text · ViFactCheck · vi|nodiac**: 63.2 → 60.8 (1.1 trên cặp còn lại 64.6). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 16 cặp:
  - `vifactcheck:dev:1993-18846436`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:2157-18744758`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:3830-19623871`: đúng → sai (sửa câu hỏi, sửa lựa chọn)
  - `vifactcheck:dev:3857-19902180`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:3870-19085185`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:4147-18971485`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:4286-19818074`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:436-19764650`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:4469-19076065`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:4688-18935994`: đúng → sai (đổi đáp án)
  - `vifactcheck:dev:5386-18795734`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:5524-18971485`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:5587-18808851`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:6853-18813865`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:7222-19463449`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:749-19049365`: đúng → sai (sửa lựa chọn)
- **decider-ft-ep2-text · ViWikiFC · vi|nodiac**: 51.7 → 55.1 (1.1 trên cặp còn lại 53.0). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 32 cặp:
  - `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_166_11_127_2_11~8bc2cb59`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_186_12_63_2_11~146dd377`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_199_13_27_1_22~0458dacc`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_243_16_42_1_32~c1304e0b`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_248_16_79_1_22~aa662aad`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_24_1_108_1_22~6b63a2a8`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_2581_154_297_1_22~c73f5673`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_25_1_120_1_31~6e5ac66e`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_268_18_86_1_32~71559017`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_269_18_92_1_11~3fd4feb6`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_2733_163_55_1_31~e280c28a`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_430_27_56_1_31~02412d9c`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_436_27_79_2_31~7a5287ad`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_495_30_32_1_11~b286fb87`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_495_30_32_1_21~9d66d23c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_499_31_25_3_12~d311c6fa`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_52_3_94_1_31~14fe0cd8`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_530_33_78_3_21~ce8ef37c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_546_34_2_3_32~d861eb59`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_547_34_4_2_21~033995e2`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_54_4_6_5_32~4e41099a`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_632_37_78_3_21~91800aa2`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_790_39_147_4_21~7ce9ad28`: sai → đúng (sửa lựa chọn)
  - … còn 7 cặp (đủ danh sách ở `scores_v1_2_trial.json`).
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
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
- **decider-ft-ep2-keyed · MASSIVE · vi|orig**: 91.0 → 93.6 (1.1 trên cặp còn lại 92.5). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 11 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10605`: đúng → sai (sửa lựa chọn)
  - `massive:11045`: đúng → sai (sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:12408`: sai → đúng (sửa lựa chọn)
  - `massive:12740`: sai → đúng (sửa lựa chọn)
  - `massive:15993`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:9743`: đúng → sai (sửa lựa chọn)
- **decider-ft-ep2-keyed · ViWikiFC · vi|orig**: 81.0 → 83.3 (1.1 trên cặp còn lại 83.3). Bị loại 13 dòng (đúng ở 1.1: 4). Đổi đúng/sai 8 cặp:
  - `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: đúng → sai (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_2731_163_26_1_31~977b229b`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_356_22_28_1_22~524c540e`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_815_41_76_2_21~2aaf7419`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_858_44_115_1_12~507d541a`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: đúng → sai (đổi đáp án)
- **decider-ft-ep2-keyed · ViWikiFC · vi|nodiac**: 53.0 → 55.1 (1.1 trên cặp còn lại 54.4). Bị loại 13 dòng (đúng ở 1.1: 3). Đổi đúng/sai 30 cặp:
  - `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_113_6_12_2_31~4174e918`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_166_11_127_2_11~8bc2cb59`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_189_12_101_1_31~cb5d4d62`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_191_12_107_1_12~6682f2fa`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_199_13_27_1_22~0458dacc`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_2581_154_297_1_22~c73f5673`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_25_1_120_1_31~6e5ac66e`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_268_18_86_1_32~71559017`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_269_18_92_1_11~3fd4feb6`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_2733_163_55_1_12~8368fa8d`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_2733_163_55_1_31~e280c28a`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_430_27_56_1_31~02412d9c`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_436_27_79_2_31~7a5287ad`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_495_30_32_1_11~b286fb87`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_52_3_94_1_31~14fe0cd8`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_530_33_78_3_21~ce8ef37c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_546_34_2_3_32~d861eb59`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_54_4_6_5_32~4e41099a`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_621_37_41_1_12~0396c7e4`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_632_37_78_3_21~91800aa2`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_788_39_137_3_22~5b3de970`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_790_39_147_4_21~7ce9ad28`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_815_41_76_2_12~80b68f10`: sai → đúng (sửa lựa chọn)
  - … còn 5 cặp (đủ danh sách ở `scores_v1_2_trial.json`).
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
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
- **Qwen3-1.7B-fullvocab-s1 · MASSIVE · vi|orig**: 83.7 → 86.8 (1.1 trên cặp còn lại 84.7). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 12 cặp:
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:1177`: sai → đúng (sửa lựa chọn)
  - `massive:12006`: đúng → sai (sửa lựa chọn)
  - `massive:12258`: sai → đúng (sửa lựa chọn)
  - `massive:12895`: sai → đúng (sửa lựa chọn)
  - `massive:1416`: sai → đúng (sửa lựa chọn)
  - `massive:15559`: đúng → sai (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3637`: đúng → sai (sửa state)
  - `massive:6642`: sai → đúng (sửa state, sửa lựa chọn)
- **Qwen3-1.7B-fullvocab-s1 · MASSIVE · vi|nodiac**: 63.3 → 66.4 (1.1 trên cặp còn lại 64.4). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 16 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:11693`: sai → đúng (sửa lựa chọn)
  - `massive:12134`: sai → đúng (sửa lựa chọn)
  - `massive:12258`: sai → đúng (sửa lựa chọn)
  - `massive:12303`: sai → đúng (sửa lựa chọn)
  - `massive:12740`: sai → đúng (sửa lựa chọn)
  - `massive:13643`: đúng → sai (sửa lựa chọn)
  - `massive:15559`: sai → đúng (sửa lựa chọn)
  - `massive:15993`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:2211`: sai → đúng (sửa lựa chọn)
  - `massive:2946`: sai → đúng (sửa lựa chọn)
  - `massive:4008`: đúng → sai (sửa lựa chọn)
  - `massive:8752`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:9863`: đúng → sai (sửa lựa chọn)
- **Qwen3-1.7B-fullvocab-s1 · ViFactCheck · vi|orig**: 70.6 → 72.8 (1.1 trên cặp còn lại 72.2). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 3 cặp:
  - `vifactcheck:dev:350-18427001`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:4688-18935994`: đúng → sai (đổi đáp án)
  - `vifactcheck:dev:6386-18610100`: sai → đúng (sửa lựa chọn)
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
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
- **Qwen3-1.7B · MASSIVE · vi|orig**: 70.7 → 75.3 (1.1 trên cặp còn lại 71.9). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 16 cặp:
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10697`: đúng → sai (sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:11876`: sai → đúng (sửa lựa chọn)
  - `massive:12055`: sai → đúng (sửa lựa chọn)
  - `massive:12895`: sai → đúng (sửa lựa chọn)
  - `massive:15867`: sai → đúng (sửa lựa chọn)
  - `massive:17009`: sai → đúng (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2668`: đúng → sai (sửa lựa chọn)
  - `massive:2851`: sai → đúng (sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3637`: đúng → sai (sửa state)
  - `massive:4129`: sai → đúng (sửa lựa chọn)
  - `massive:4694`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:6883`: sai → đúng (sửa lựa chọn)
- **Qwen3-1.7B · MASSIVE · vi|nodiac**: 43.0 → 45.1 (1.1 trên cặp còn lại 43.7). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 26 cặp:
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10697`: đúng → sai (sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:113`: sai → đúng (sửa lựa chọn)
  - `massive:12006`: đúng → sai (sửa lựa chọn)
  - `massive:12125`: đúng → sai (sửa lựa chọn)
  - `massive:12258`: sai → đúng (sửa lựa chọn)
  - `massive:12303`: đúng → sai (sửa lựa chọn)
  - `massive:12318`: sai → đúng (sửa lựa chọn)
  - `massive:12408`: sai → đúng (sửa lựa chọn)
  - `massive:12674`: sai → đúng (sửa lựa chọn)
  - `massive:12740`: sai → đúng (sửa lựa chọn)
  - `massive:12895`: đúng → sai (sửa lựa chọn)
  - `massive:15294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:15455`: đúng → sai (sửa lựa chọn)
  - `massive:15471`: sai → đúng (sửa lựa chọn)
  - `massive:15993`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:17046`: đúng → sai (sửa lựa chọn)
  - `massive:2161`: đúng → sai (sửa lựa chọn)
  - `massive:3637`: sai → đúng (sửa state)
  - `massive:3803`: sai → đúng (sửa lựa chọn)
  - `massive:5184`: sai → đúng (sửa lựa chọn)
  - `massive:6136`: sai → đúng (sửa lựa chọn)
  - `massive:8132`: sai → đúng (sửa lựa chọn)
  - `massive:8752`: đúng → sai (sửa state, sửa lựa chọn)
  - … còn 1 cặp (đủ danh sách ở `scores_v1_2_trial.json`).
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
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
- **strip-s1 · MASSIVE · vi|orig**: 86.0 → 89.2 (1.1 trên cặp còn lại 87.1). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 14 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10506`: đúng → sai (sửa lựa chọn)
  - `massive:11045`: đúng → sai (sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:12006`: đúng → sai (sửa lựa chọn)
  - `massive:12408`: đúng → sai (sửa lựa chọn)
  - `massive:12895`: sai → đúng (sửa lựa chọn)
  - `massive:15294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:1621`: sai → đúng (sửa lựa chọn)
  - `massive:2161`: sai → đúng (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:3183`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:6642`: sai → đúng (sửa state, sửa lựa chọn)
- **strip-s1 · MASSIVE · vi|nodiac**: 65.0 → 67.5 (1.1 trên cặp còn lại 66.1). Bị loại 5 dòng (đúng ở 1.1: 0). Đổi đúng/sai 28 cặp:
  - `massive:10058`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:10457`: sai → đúng (đổi đáp án, sửa lựa chọn)
  - `massive:11294`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:11876`: đúng → sai (sửa lựa chọn)
  - `massive:11980`: sai → đúng (sửa lựa chọn)
  - `massive:12107`: sai → đúng (sửa lựa chọn)
  - `massive:12125`: đúng → sai (sửa lựa chọn)
  - `massive:12134`: sai → đúng (sửa lựa chọn)
  - `massive:12258`: sai → đúng (sửa lựa chọn)
  - `massive:12303`: sai → đúng (sửa lựa chọn)
  - `massive:1234`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:12408`: sai → đúng (sửa lựa chọn)
  - `massive:12895`: sai → đúng (sửa lựa chọn)
  - `massive:1365`: sai → đúng (sửa lựa chọn)
  - `massive:15319`: đúng → sai (sửa lựa chọn)
  - `massive:15471`: sai → đúng (sửa lựa chọn)
  - `massive:15812`: đúng → sai (sửa lựa chọn)
  - `massive:15993`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:2161`: đúng → sai (sửa lựa chọn)
  - `massive:2499`: sai → đúng (sửa state, sửa lựa chọn)
  - `massive:2946`: đúng → sai (sửa lựa chọn)
  - `massive:3296`: sai → đúng (sửa lựa chọn)
  - `massive:4008`: đúng → sai (sửa lựa chọn)
  - `massive:6642`: đúng → sai (sửa state, sửa lựa chọn)
  - `massive:756`: đúng → sai (sửa lựa chọn)
  - … còn 3 cặp (đủ danh sách ở `scores_v1_2_trial.json`).
- **strip-s1 · ViFactCheck · vi|orig**: 71.2 → 73.4 (1.1 trên cặp còn lại 72.8). Bị loại 5 dòng (đúng ở 1.1: 1). Đổi đúng/sai 5 cặp:
  - `vifactcheck:dev:1807-18621446`: đúng → sai (sửa lựa chọn)
  - `vifactcheck:dev:2680-18808499`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:436-19764650`: sai → đúng (sửa lựa chọn)
  - `vifactcheck:dev:4688-18935994`: đúng → sai (đổi đáp án)
  - `vifactcheck:dev:6853-18813865`: sai → đúng (sửa lựa chọn)
- **strip-s1 · ViWikiFC · vi|nodiac**: 66.0 → 62.0 (1.1 trên cặp còn lại 66.6). Bị loại 13 dòng (đúng ở 1.1: 7). Đổi đúng/sai 19 cặp:
  - `viwikifc:dev:uit_1036_61_18_2_31~b50aa4b1`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1100_70_59_1_12~416a3b72`: đúng → sai (đổi đáp án, sửa lựa chọn)
  - `viwikifc:dev:uit_1829_125_54_3_12~fb78d9c6`: đúng → sai (đổi đáp án, sửa câu hỏi)
  - `viwikifc:dev:uit_189_12_102_1_21~5b2e8728`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_191_12_107_1_12~6682f2fa`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_1922_130_50_1_31~18ec69b3`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_2581_154_297_1_22~c73f5673`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_269_18_86_3_22~9ed8aebf`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_2818_175_34_1_31~1c712305`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_30_2_18_1_21~af2f901b`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_430_27_56_1_31~02412d9c`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_499_31_25_3_12~d311c6fa`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_530_33_78_3_22~acdc7991`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_54_4_6_5_32~4e41099a`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_803_40_54_3_31~67724e1c`: sai → đúng (sửa lựa chọn)
  - `viwikifc:dev:uit_840_44_40_1_21~6a7d2379`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_845_44_52_6_11~32808231`: đúng → sai (sửa lựa chọn)
  - `viwikifc:dev:uit_91_5_74_3_31~d4bf3885`: đúng → sai (đổi đáp án)
  - `viwikifc:dev:uit_966_55_15_1_31~f3e88b85`: đúng → sai (sửa lựa chọn)

## 5. So với lần dựng thử trước (chỉ khác ở chuỗi mẫu)

Lần trước: dev `0915baef16b4…`, test `96f990b7a13c…`, 1584 bản ghi lớp ghi đè, chuỗi mẫu đã sửa: 2. Lần này: dev `0cccd357779f…`, test `077323ee9772…`, 1650 bản ghi, chuỗi mẫu đã sửa: 9. Giữa hai lần chỉ có thêm bản ghi chuỗi mẫu (duyệt + sửa), nên chênh điểm dưới đây chỉ do chuỗi mẫu.

| Model | Nguồn | biến thể | 1.2 lần trước | 1.2 lần này | chênh |
|---|---|---|---|---|---|
| decider-2b-text | MASSIVE | en orig | 96.6 | 96.6 | +0.0 |
| decider-2b-text | MASSIVE | vi orig | 96.6 | 96.3 | -0.3 |
| decider-2b-text | MASSIVE | vi nodiac | 54.9 | 53.9 | -1.0 |
| decider-2b-text | ViFactCheck | vi orig | 83.5 | 83.5 | +0.0 |
| decider-2b-text | ViFactCheck | vi nodiac | 54.4 | 53.2 | -1.3 |
| decider-2b-text | ViWikiFC | vi orig | 82.6 | 82.2 | -0.3 |
| decider-2b-text | ViWikiFC | vi nodiac | 47.0 | 48.1 | +1.0 |
| decider-2b-keyed | MASSIVE | en orig | 94.6 | 94.6 | +0.0 |
| decider-2b-keyed | MASSIVE | vi orig | 92.2 | 92.2 | +0.0 |
| decider-2b-keyed | MASSIVE | vi nodiac | 52.9 | 51.2 | -1.7 |
| decider-2b-keyed | ViFactCheck | vi orig | 84.2 | 82.9 | -1.3 |
| decider-2b-keyed | ViFactCheck | vi nodiac | 54.4 | 51.3 | -3.2 |
| decider-2b-keyed | ViWikiFC | vi orig | 82.2 | 81.9 | -0.3 |
| decider-2b-keyed | ViWikiFC | vi nodiac | 46.0 | 47.4 | +1.4 |
| decider-ft-ep2-text | MASSIVE | en orig | 95.9 | 95.9 | +0.0 |
| decider-ft-ep2-text | MASSIVE | vi orig | 94.6 | 94.2 | -0.3 |
| decider-ft-ep2-text | MASSIVE | vi nodiac | 64.4 | 64.7 | +0.3 |
| decider-ft-ep2-text | ViFactCheck | vi orig | 82.3 | 80.4 | -1.9 |
| decider-ft-ep2-text | ViFactCheck | vi nodiac | 63.3 | 60.8 | -2.5 |
| decider-ft-ep2-text | ViWikiFC | vi orig | 82.6 | 82.6 | +0.0 |
| decider-ft-ep2-text | ViWikiFC | vi nodiac | 52.6 | 55.1 | +2.4 |
| decider-ft-ep2-keyed | MASSIVE | en orig | 95.9 | 95.9 | +0.0 |
| decider-ft-ep2-keyed | MASSIVE | vi orig | 94.2 | 93.6 | -0.7 |
| decider-ft-ep2-keyed | MASSIVE | vi nodiac | 63.7 | 62.7 | -1.0 |
| decider-ft-ep2-keyed | ViFactCheck | vi orig | 81.0 | 79.7 | -1.3 |
| decider-ft-ep2-keyed | ViFactCheck | vi nodiac | 63.9 | 62.0 | -1.9 |
| decider-ft-ep2-keyed | ViWikiFC | vi orig | 82.6 | 83.3 | +0.7 |
| decider-ft-ep2-keyed | ViWikiFC | vi nodiac | 54.0 | 55.1 | +1.0 |
| Qwen3-1.7B-fullvocab-s1 | MASSIVE | en orig | 90.2 | 90.2 | +0.0 |
| Qwen3-1.7B-fullvocab-s1 | MASSIVE | vi orig | 86.4 | 86.8 | +0.3 |
| Qwen3-1.7B-fullvocab-s1 | MASSIVE | vi nodiac | 66.1 | 66.4 | +0.3 |
| Qwen3-1.7B-fullvocab-s1 | ViFactCheck | vi orig | 71.5 | 72.8 | +1.3 |
| Qwen3-1.7B-fullvocab-s1 | ViFactCheck | vi nodiac | 64.6 | 65.2 | +0.6 |
| Qwen3-1.7B-fullvocab-s1 | ViWikiFC | vi orig | 76.3 | 76.0 | -0.3 |
| Qwen3-1.7B-fullvocab-s1 | ViWikiFC | vi nodiac | 66.9 | 66.2 | -0.7 |
| Qwen3-1.7B | MASSIVE | en orig | 90.5 | 90.5 | +0.0 |
| Qwen3-1.7B | MASSIVE | vi orig | 73.6 | 75.3 | +1.7 |
| Qwen3-1.7B | MASSIVE | vi nodiac | 44.7 | 45.1 | +0.3 |
| Qwen3-1.7B | ViFactCheck | vi orig | 56.3 | 57.6 | +1.3 |
| Qwen3-1.7B | ViFactCheck | vi nodiac | 46.2 | 44.9 | -1.3 |
| Qwen3-1.7B | ViWikiFC | vi orig | 60.3 | 59.9 | -0.3 |
| Qwen3-1.7B | ViWikiFC | vi nodiac | 47.0 | 47.7 | +0.7 |
| strip-s1 | MASSIVE | en orig | 91.2 | 91.2 | +0.0 |
| strip-s1 | MASSIVE | vi orig | 89.5 | 89.2 | -0.3 |
| strip-s1 | MASSIVE | vi nodiac | 67.1 | 67.5 | +0.3 |
| strip-s1 | ViFactCheck | vi orig | 72.2 | 73.4 | +1.3 |
| strip-s1 | ViFactCheck | vi nodiac | 63.3 | 62.7 | -0.6 |
| strip-s1 | ViWikiFC | vi orig | 77.0 | 76.3 | -0.7 |
| strip-s1 | ViWikiFC | vi nodiac | 65.2 | 62.0 | -3.1 |

## 6. File

- `dev_v1_2_trial.jsonl`, `test_v1_2_trial.jsonl`, `changes_v1_2_trial.jsonl` (loại thay đổi từng cặp), `build_report_v1_2_trial.json`, `manifest_v1_2_trial.json`.
- `rescore/rescore_dev.jsonl` (dòng dev đổi byte), `eval_rescore/` (preds chấm lại + log), `eval/<model>/preds.jsonl` (preds dev 1.2 đã gộp), `scores_v1_2_trial.json`.
- Code: `scripts/human_edits/build_revised.py` (thêm `--trial`, `--nfc`, `--test_labels`), `run_trial_v1_2.sh`, `rescore_dev_v1_2.sh`, `report_trial_v1_2.py`.

Chưa đóng băng gì. Khi người làm dự án bảo đóng băng: chạy `build_revised.py` không có `--trial` (ghi test mới chỉ đọc + FROZEN) và giữ test 1.1 để đối chiếu.
