# Bench 1.2 (đã đóng băng)

Bản 1.2 = bench 1.1 sau khi người soát: áp lớp ghi đè `data/human_edits` (đề xuất đã duyệt, sửa câu, loại câu, đổi đáp án dev, bản dịch người dùng gửi), quyết định nhãn test (`results/human_review/label_decisions_remote.jsonl`, dòng mới nhất theo cặp), chuẩn hóa NFC mọi trường tiếng Việt (bản không dấu sinh lại từ bản đã sửa). MLQA để bản 1.3, Global-PIQA không làm. Không train; không chạy model nào trên test.

## Tóm tắt

- **Đóng băng** ở `data/bench_v1_2/` lúc 2026-10-16T15:37:56+07:00: `test_v1.2.jsonl` (4086 dòng, chỉ đọc, sha `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da`), `FROZEN_test_v1.2.sha256`, `manifest_test_v1.2.json` (chỉ đọc); `dev_v1_2.jsonl` (3851 dòng, sha `ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b`), `manifest_dev_v1_2.json`, `leakage_v1_2.json`. Bản 1.1 giữ nguyên ở `data/bench_v1_1/`.
- **Cập nhật dev 2026-10-16** (test KHÔNG đổi, vẫn `2bdbd2109a32…`): Người dùng chốt trong chat 16/10: 'sửa Tiger#1 và sib200:339: Tay đua này thành Lenz'. Tiger#1: câu hỏi dev 1.2 mất phủ định so với EN 'would not be considered' (hoàn tác ở mục B đã gỡ cả sửa câu hỏi), nên đáp án D 'Leo trèo' sai với câu hỏi VI; sửa thành 'Điểm nào sau đây không được coi là thế mạnh lớn nhất của loài hổ?' (c-ffc24b885010), đoạn văn giữ bản đã hoàn tác. sib200:339: state 'Tay đua này' -> 'Lenz' (c-fd9d420c49d9). Cặp đổi: `belebele:https://en.wikibooks.org/wiki/Wikijunior:Big_Cats/Tiger#1`, `sib200:339`. Dev sha `de9bb4aa2886…` → `ef88be1625c8…`; các dòng đổi đã chấm lại bằng 7 model.
- Dựng từ 1786 bản ghi lớp ghi đè và 144 quyết định nhãn. `release_check`: mã 0, 0 đề xuất chờ.
- Cặp đổi so với 1.1: dev 792, test 850. Đáp án ngược nguồn: dev 10, test 13.
- Rò rỉ: dev↔test 0 / 0; train_v0 + val_v0 với test 2, với dev 0 (như 1.1). Một cặp dev (`massive:4129`) bị loại để giữ dev↔test = 0 (mục C).
- Người dùng chưa xác nhận câu nào qua nút "Đồng ý (đã đọc)": `user_confirmations.jsonl` chưa có; đã xác nhận 0/93 câu bị loại hoặc đổi đáp án.

Điểm dev (%), trung bình trên mọi câu; "chỉ dòng source" = cặp có văn bản tiếng Việt và đáp án giữ như nguồn:

| Model | EN: 1.1 / 1.2 / 1.2 source | VI có dấu: 1.1 / 1.2 / 1.2 source | VI không dấu: 1.1 / 1.2 / 1.2 source |
|---|---|---|---|
| decider-2b-text | 88.9 / 90.0 / 90.3 | 84.5 / 86.7 / 86.5 | 47.4 / 48.7 / 48.6 |
| decider-2b-keyed | 88.0 / 89.2 / 89.5 | 83.2 / 85.4 / 84.9 | 46.7 / 47.8 / 47.8 |
| decider-ft-ep2-text | 88.0 / 89.4 / 89.6 | 83.5 / 85.6 / 85.4 | 51.3 / 52.2 / 54.6 |
| decider-ft-ep2-keyed | 87.6 / 89.1 / 89.3 | 83.5 / 85.3 / 85.4 | 50.8 / 51.7 / 53.8 |
| Qwen3-1.7B-fullvocab-s1 | 85.9 / 87.3 / 88.0 | 78.5 / 80.7 / 80.4 | 61.6 / 62.7 / 64.0 |
| Qwen3-1.7B | 83.0 / 84.3 / 85.9 | 67.2 / 70.6 / 69.6 | 47.0 / 48.3 / 48.2 |
| strip-s1 | 86.1 / 87.5 / 88.0 | 79.2 / 81.2 / 80.7 | 62.2 / 63.1 / 63.5 |

Số câu: EN 1000 → 987 (source 694); VI 1463 → 1432 (source 1125). Bảng theo nguồn ở mục 5. Không kết luận hướng đi.

## 1. So với bản dựng thử cuối

Bản thử: dev `0cccd357779f…`, test `077323ee9772…` (1650 bản ghi). Bản hiện hành (test trùng từng byte bản đóng băng) dựng từ 1786 bản ghi. Mọi cặp khác bản thử đều thuộc một loại dưới đây (không có cặp không giải thích được):

| Loại khác biệt | dev | test |
|---|---|---|
| chuỗi mẫu transport_query | 54 | 44 |
| hoàn tác ở màn Đúng→sai (mục B) | 2 | 0 |
| loại vì trùng gần với test | 1 | 0 |
| bản dịch / yêu cầu người dùng gửi trong chat | 14 | 0 |
| quyết định mới / đổi ý trên trang | 1 | 3 |

- Bản dịch / yêu cầu người dùng gửi trong chat (dev): `belebele:https://en.wikibooks.org/wiki/Wikijunior:Big_Cats/Tiger#1`, `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#1`, `belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee#2`, `belebele:https://en.wikivoyage.org/wiki/Buffalo-Pittsburgh_Highway#1`, `belebele:https://en.wikivoyage.org/wiki/Buffalo-Pittsburgh_Highway#2`, `belebele:https://en.wikivoyage.org/wiki/Canyoning#2`, `belebele:https://en.wikivoyage.org/wiki/Goma#1`, `belebele:https://en.wikivoyage.org/wiki/Goma#2`, `belebele:https://en.wikivoyage.org/wiki/Grand_Canyon#1`, `belebele:https://en.wikivoyage.org/wiki/Grand_Canyon#2`, `sib200:339`, `sib200:810`, `xcopa:22`, `xcopa:81`.
- Quyết định trên trang: dev `sib200:339`; test `belebele:test:https://en.wikivoyage.org/wiki/Running#1`, `belebele:test:https://en.wikivoyage.org/wiki/Running#2`, `xcopa:test:1`.
- Hoàn tác ở mục B: `xcopa:33` (gỡ c-0af1682b067b; kéo theo 0 câu); `belebele:https://en.wikibooks.org/wiki/Wikijunior:Big_Cats/Tiger#1` (gỡ c-29000e76f10b, c-217b74f12c60; kéo theo 0 câu). Giữ bản mới: 19 câu.
- Quyết định nhãn mới sau bản thử: massive:test:5823 loai (không đổi dữ liệu vì cặp đã bị loại sẵn).
- `transport_query`: "tra cứu tuyến đường hoặc giờ tàu xe" → "tra cứu tuyến đường hoặc lịch trình".
- Dòng dev có nội dung khác bản thử: 138, đã chấm lại bằng 7 model (`scripts/human_edits/rescore_dev_v1_2_final.sh`); 3713 dòng còn lại dùng preds của bản thử.

## 2. Thay đổi so với 1.1 theo nguồn và loại

Một cặp (EN + VI + VI không dấu) được đếm ở **mọi** loại nó thuộc. "Chỉ đổi do NFC" là cặp đổi byte nhưng chữ sau NFC không đổi.

### dev

| Nguồn | Số cặp 1.1 | đổi đáp án | sửa câu hỏi | sửa lựa chọn | sửa state | loại câu | chỉ đổi do NFC | Cặp có đổi |
|---|---|---|---|---|---|---|---|---|
| Belebele | 300 | 2 | 19 | 25 | 184 | 6 | 0 | 197 |
| MASSIVE | 300 | 2 | 0 | 227 | 24 | 6 | 14 | 251 |
| SIB-200 | 300 | 1 | 0 | 0 | 61 | 1 | 0 | 63 |
| ViFactCheck | 163 | 1 | 3 | 80 | 0 | 5 | 1 | 87 |
| ViWikiFC | 300 | 4 | 7 | 162 | 0 | 13 | 0 | 180 |
| XCOPA | 100 | 0 | 0 | 7 | 11 | 0 | 0 | 14 |
| **Tổng** | 1463 | 10 | 29 | 501 | 280 | 31 | 15 | 792 |

### test

| Nguồn | Số cặp 1.1 | đổi đáp án | sửa câu hỏi | sửa lựa chọn | sửa state | loại câu | chỉ đổi do NFC | Cặp có đổi |
|---|---|---|---|---|---|---|---|---|
| Belebele | 272 | 1 | 13 | 15 | 159 | 5 | 0 | 169 |
| MASSIVE | 300 | 1 | 0 | 240 | 22 | 6 | 15 | 269 |
| SIB-200 | 120 | 0 | 0 | 0 | 33 | 1 | 0 | 34 |
| ViFactCheck | 300 | 5 | 11 | 135 | 1 | 9 | 6 | 160 |
| ViWikiFC | 300 | 6 | 9 | 131 | 2 | 18 | 1 | 160 |
| XCOPA | 300 | 0 | 0 | 40 | 23 | 0 | 0 | 58 |
| **Tổng** | 1592 | 13 | 33 | 561 | 240 | 39 | 22 | 850 |

## 3. Nguồn gốc văn bản và đáp án (A)

`gold_provenance` trên mọi dòng; `vi_provenance` + `vi_revised_fields` trên dòng tiếng Việt (NFC và chuỗi mẫu không tính là sửa). Số dòng VI có dấu:

| Tập · nguồn | source | revised | sửa state | sửa câu hỏi | sửa lựa chọn |
|---|---|---|---|---|---|
| dev · belebele | 104 | 190 | 184 | 19 | 40 |
| dev · massive | 270 | 24 | 24 | 0 | 0 |
| dev · sib200 | 238 | 61 | 61 | 0 | 0 |
| dev · vifactcheck | 155 | 3 | 0 | 3 | 0 |
| dev · viwikifc | 280 | 7 | 0 | 7 | 0 |
| dev · xcopa | 86 | 14 | 11 | 0 | 7 |
| test · belebele | 104 | 163 | 159 | 13 | 21 |
| test · massive | 272 | 22 | 22 | 0 | 0 |
| test · sib200 | 86 | 33 | 33 | 0 | 0 |
| test · vifactcheck | 280 | 11 | 1 | 11 | 0 |
| test · viwikifc | 271 | 11 | 2 | 9 | 0 |
| test · xcopa | 242 | 58 | 23 | 0 | 43 |

Đáp án đi ngược nguồn: 23 cặp:

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

Quy trình: Đề xuất do LLM (Opus, Fable) viết, người làm dự án duyệt. Cách duyệt theo lời người làm dự án: "đọc từng câu với đề xuất đổi đáp án và nhãn; đề xuất dịch mà Opus đồng ý thì duyệt theo lô sau khi đọc mẫu, nếu cần thiết thì gọi fable ra".

Số liệu thật từ log (manifest `provenance.so_lieu_tu_log`):

- Đề xuất: 732 bản ghi; người dùng bấm từ xa 555; mang ghi chú ủy quyền 132; điều phối ghi theo lời người dùng trong chat 43; khác (theo lựa chọn của người dùng trên trang) 2.
- Đề xuất đang áp: 715, theo người viết: agent (phiên Decision model tiếng Việt, Opus 5.5) 2, agent_flores_final_check 2, agent_opus 307, agent_opus (soát chuỗi mẫu, phiên Decision model tiếng Việt) 2, agent_opus_extra 35, agent_opus_final_check 185, agent_verify 137, dieu_phoi 10, nguoi_lam_du_an 35.
- Chuỗi mẫu: Excel 66, từ xa 2, theo chat 1. Đúng→sai: giữ bản mới 19, hoàn tác 2.
- Quyết định nhãn: 144 bản ghi (mang ghi chú ủy quyền 143); mới nhất theo cặp: dev|doi_nhan 10, dev|giu 32, dev|loai 30, test|doi_nhan 13, test|giu 19, test|loai 39.
- Số do điều phối cung cấp (log không tách được): người dùng tự bấm "5 trên trang, 1 lô 350, 6 trong chat, Excel chuỗi mẫu"; theo ủy quyền / theo agent "132 đề xuất, 141 nhãn, 182 đề xuất kiểm tra cuối, và các điều chỉnh an toàn".

## 4. Kỹ thuật (C)

- Bản không dấu sinh từ bản VI hiện hành (strip_diacritics trên state, câu hỏi, lựa chọn; cùng đáp án): kiểm {'dev': 1432, 'test': 1553} dòng, trong đó revised {'dev': 299, 'test': 298}; không khớp: 0.
- Rò rỉ, đúng cách build_bench_v1_1 (khớp chính xác, MinHash ≥ 0,8, 13-gram / câu ngắn nằm trọn): dev↔test 0 / 0; train_v0 + val_v0: test 2 (contain13|test:vifactcheck|train_source=pholaya/vinli/relation: 1, contain13|test:vifactcheck|train_source=pholaya/vinli/entailment: 1), dev 0.
- Lần dựng đầu phát hiện 1 trùng dev↔test (câu ngắn nằm trọn): test `massive:test:1483` "tôi đã đặt báo thức nào" (sau khi sửa "lời nhắc" → "báo thức" cho khớp EN "alarms") nằm trọn trong dev `massive:4129` "tôi đã đặt báo thức nào ngay bây giờ". Hai câu MASSIVE khác nhau, bản EN không trùng. Điều phối chọn loại cặp dev (test giữ nguyên); dev MASSIVE còn 294 cặp (1.1: 300; 5 cặp khác đã bị loại từ trước trong lớp ghi đè). `build_revised.py` giờ có chốt kiểm dev↔test (bản thử ghi và cảnh báo, bản thật chặn).

## 5. Điểm DEV theo nguồn

Độ đúng theo đáp án của từng bản. "1.2 chỉ dòng source": cặp có `vi_provenance = source` và `gold_provenance = source` (EN tính theo cặp). Ô có |1.2 − 1.1| > 2 điểm được tô đậm.

### decider-2b-text

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 87.3 | 88.8 | 88.3 | +1.4 |
| Belebele | VI có dấu | 300 → 294 (103) | 83.0 | 87.8 | 87.4 | **+4.8** |
| Belebele | VI không dấu | 300 → 294 (103) | 49.0 | 53.1 | 53.4 | **+4.1** |
| MASSIVE | EN | 300 → 294 (268) | 95.0 | 96.6 | 97.0 | +1.6 |
| MASSIVE | VI có dấu | 300 → 294 (268) | 93.3 | 96.3 | 96.3 | **+2.9** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 53.7 | 53.7 | 53.7 | +0.1 |
| SIB-200 | EN | 300 → 299 (237) | 83.3 | 83.9 | 83.1 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 82.3 | 82.3 | 81.0 | -0.1 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 36.0 | 35.8 | 35.9 | -0.2 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 82.8 | 83.5 | 84.4 | +0.7 |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 52.8 | 53.2 | 53.9 | +0.4 |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 79.7 | 82.2 | 82.7 | **+2.6** |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 46.0 | 48.1 | 47.3 | **+2.1** |
| XCOPA | EN | 100 → 100 (86) | 92.0 | 92.0 | 91.9 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 86.0 | 86.0 | 86.0 | +0.0 |
| XCOPA | VI không dấu | 100 → 100 (86) | 53.0 | 54.0 | 57.0 | +1.0 |
| Tất cả | EN | 1000 → 987 (694) | 88.9 | 90.0 | 90.3 | +1.1 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 84.5 | 86.7 | 86.5 | +2.2 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 47.4 | 48.7 | 48.6 | +1.3 |

### decider-2b-keyed

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 86.7 | 88.1 | 88.3 | +1.4 |
| Belebele | VI có dấu | 300 → 294 (103) | 83.3 | 88.4 | 87.4 | **+5.1** |
| Belebele | VI không dấu | 300 → 294 (103) | 49.0 | 52.7 | 55.3 | **+3.7** |
| MASSIVE | EN | 300 → 294 (268) | 92.7 | 94.6 | 94.8 | +1.9 |
| MASSIVE | VI có dấu | 300 → 294 (268) | 89.0 | 92.2 | 92.2 | **+3.2** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 51.7 | 51.4 | 51.5 | -0.3 |
| SIB-200 | EN | 300 → 299 (237) | 83.0 | 83.6 | 83.1 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 81.3 | 81.3 | 79.7 | -0.1 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 36.0 | 36.5 | 36.7 | +0.5 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 83.4 | 82.9 | 83.8 | -0.5 |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 53.4 | 51.3 | 51.9 | **-2.1** |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 79.3 | 81.9 | 82.7 | **+2.5** |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 44.7 | 47.4 | 46.6 | **+2.7** |
| XCOPA | EN | 100 → 100 (86) | 93.0 | 93.0 | 91.9 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 82.0 | 83.0 | 82.6 | +1.0 |
| XCOPA | VI không dấu | 100 → 100 (86) | 52.0 | 52.0 | 54.7 | +0.0 |
| Tất cả | EN | 1000 → 987 (694) | 88.0 | 89.2 | 89.5 | +1.2 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 83.2 | 85.4 | 84.9 | +2.2 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 46.7 | 47.8 | 47.8 | +1.1 |

### decider-ft-ep2-text

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 87.0 | 89.1 | 93.2 | **+2.1** |
| Belebele | VI có dấu | 300 → 294 (103) | 82.0 | 87.8 | 88.3 | **+5.8** |
| Belebele | VI không dấu | 300 → 294 (103) | 45.0 | 46.9 | 52.4 | +1.9 |
| MASSIVE | EN | 300 → 294 (268) | 94.0 | 95.9 | 95.9 | +1.9 |
| MASSIVE | VI có dấu | 300 → 294 (268) | 91.3 | 94.2 | 94.4 | **+2.9** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 62.7 | 64.3 | 64.9 | +1.6 |
| SIB-200 | EN | 300 → 299 (237) | 81.7 | 82.3 | 80.6 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 81.0 | 81.6 | 79.3 | +0.6 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 36.3 | 35.5 | 35.9 | -0.9 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 81.0 | 80.4 | 81.8 | -0.6 |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 63.2 | 60.8 | 62.3 | **-2.4** |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 81.0 | 82.6 | 83.4 | +1.6 |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 51.7 | 55.1 | 54.9 | **+3.4** |
| XCOPA | EN | 100 → 100 (86) | 92.0 | 92.0 | 90.7 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 84.0 | 83.0 | 83.7 | -1.0 |
| XCOPA | VI không dấu | 100 → 100 (86) | 60.0 | 60.0 | 61.6 | +0.0 |
| Tất cả | EN | 1000 → 987 (694) | 88.0 | 89.4 | 89.6 | +1.4 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 83.5 | 85.6 | 85.4 | +2.1 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 51.3 | 52.2 | 54.6 | +0.9 |

### decider-ft-ep2-keyed

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 86.7 | 88.8 | 91.3 | **+2.1** |
| Belebele | VI có dấu | 300 → 294 (103) | 82.0 | 86.7 | 88.3 | **+4.7** |
| Belebele | VI không dấu | 300 → 294 (103) | 42.7 | 47.3 | 48.5 | **+4.6** |
| MASSIVE | EN | 300 → 294 (268) | 93.7 | 95.9 | 95.9 | **+2.3** |
| MASSIVE | VI có dấu | 300 → 294 (268) | 91.0 | 93.2 | 93.3 | **+2.2** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 62.7 | 62.2 | 63.1 | -0.4 |
| SIB-200 | EN | 300 → 299 (237) | 81.3 | 81.9 | 81.0 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 81.7 | 82.3 | 80.2 | +0.6 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 34.3 | 33.8 | 34.6 | -0.6 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 79.8 | 79.7 | 81.2 | -0.0 |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 63.8 | 62.0 | 63.6 | -1.8 |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 81.0 | 83.3 | 84.1 | **+2.3** |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 53.0 | 55.1 | 54.9 | **+2.1** |
| XCOPA | EN | 100 → 100 (86) | 91.0 | 91.0 | 89.5 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 84.0 | 82.0 | 83.7 | **-2.0** |
| XCOPA | VI không dấu | 100 → 100 (86) | 61.0 | 61.0 | 62.8 | +0.0 |
| Tất cả | EN | 1000 → 987 (694) | 87.6 | 89.1 | 89.3 | +1.5 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 83.5 | 85.3 | 85.4 | +1.9 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 50.8 | 51.7 | 53.8 | +0.9 |

### Qwen3-1.7B-fullvocab-s1

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 84.7 | 86.7 | 89.3 | **+2.1** |
| Belebele | VI có dấu | 300 → 294 (103) | 77.3 | 82.3 | 79.6 | **+5.0** |
| Belebele | VI không dấu | 300 → 294 (103) | 58.3 | 58.8 | 66.0 | +0.5 |
| MASSIVE | EN | 300 → 294 (268) | 88.0 | 90.1 | 89.9 | **+2.1** |
| MASSIVE | VI có dấu | 300 → 294 (268) | 83.7 | 86.7 | 87.3 | **+3.1** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 63.3 | 66.3 | 66.4 | **+3.0** |
| SIB-200 | EN | 300 → 299 (237) | 83.3 | 83.9 | 83.5 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 81.7 | 82.6 | 81.4 | +0.9 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 58.3 | 60.2 | 58.6 | +1.9 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 70.6 | 72.8 | 73.4 | **+2.2** |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 63.8 | 65.2 | 66.2 | +1.4 |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 75.3 | 76.0 | 77.3 | +0.6 |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 67.0 | 66.2 | 66.4 | -0.8 |
| XCOPA | EN | 100 → 100 (86) | 91.0 | 91.0 | 93.0 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 79.0 | 78.0 | 79.1 | -1.0 |
| XCOPA | VI không dấu | 100 → 100 (86) | 56.0 | 57.0 | 57.0 | +1.0 |
| Tất cả | EN | 1000 → 987 (694) | 85.9 | 87.3 | 88.0 | +1.4 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 78.5 | 80.7 | 80.4 | +2.2 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 61.6 | 62.7 | 64.0 | +1.1 |

### Qwen3-1.7B

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 79.0 | 80.6 | 85.4 | +1.6 |
| Belebele | VI có dấu | 300 → 294 (103) | 67.3 | 75.2 | 73.8 | **+7.8** |
| Belebele | VI không dấu | 300 → 294 (103) | 46.0 | 49.3 | 53.4 | **+3.3** |
| MASSIVE | EN | 300 → 294 (268) | 88.3 | 90.5 | 90.3 | **+2.1** |
| MASSIVE | VI có dấu | 300 → 294 (268) | 70.7 | 75.2 | 75.0 | **+4.5** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 43.0 | 45.6 | 45.5 | **+2.6** |
| SIB-200 | EN | 300 → 299 (237) | 80.7 | 81.3 | 81.0 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 74.0 | 75.6 | 75.9 | +1.6 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 51.3 | 51.2 | 50.6 | -0.2 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 55.8 | 57.6 | 59.1 | +1.8 |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 46.0 | 44.9 | 46.1 | -1.1 |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 59.0 | 59.9 | 60.3 | +0.9 |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 46.7 | 47.7 | 47.7 | +1.1 |
| XCOPA | EN | 100 → 100 (86) | 86.0 | 86.0 | 86.0 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 79.0 | 80.0 | 79.1 | +1.0 |
| XCOPA | VI không dấu | 100 → 100 (86) | 51.0 | 52.0 | 48.8 | +1.0 |
| Tất cả | EN | 1000 → 987 (694) | 83.0 | 84.3 | 85.9 | +1.3 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 67.2 | 70.6 | 69.6 | +3.4 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 47.0 | 48.3 | 48.2 | +1.4 |

### strip-s1

| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |
|---|---|---|---|---|---|---|
| Belebele | EN | 300 → 294 (103) | 84.3 | 86.4 | 86.4 | **+2.1** |
| Belebele | VI có dấu | 300 → 294 (103) | 78.3 | 81.6 | 78.6 | **+3.3** |
| Belebele | VI không dấu | 300 → 294 (103) | 60.7 | 64.3 | 68.9 | **+3.6** |
| MASSIVE | EN | 300 → 294 (268) | 89.0 | 91.2 | 91.0 | **+2.2** |
| MASSIVE | VI có dấu | 300 → 294 (268) | 86.0 | 89.1 | 88.8 | **+3.1** |
| MASSIVE | VI không dấu | 300 → 294 (268) | 65.0 | 68.0 | 68.3 | **+3.0** |
| SIB-200 | EN | 300 → 299 (237) | 83.0 | 83.6 | 83.1 | +0.6 |
| SIB-200 | VI có dấu | 300 → 299 (237) | 80.7 | 82.3 | 80.2 | +1.6 |
| SIB-200 | VI không dấu | 300 → 299 (237) | 59.0 | 60.2 | 58.2 | +1.2 |
| ViFactCheck | VI có dấu | 163 → 158 (154) | 71.2 | 73.4 | 74.0 | **+2.3** |
| ViFactCheck | VI không dấu | 163 → 158 (154) | 62.0 | 62.7 | 63.6 | +0.7 |
| ViWikiFC | VI có dấu | 300 → 287 (277) | 76.0 | 76.3 | 77.6 | +0.3 |
| ViWikiFC | VI không dấu | 300 → 287 (277) | 66.0 | 62.0 | 62.5 | **-4.0** |
| XCOPA | EN | 100 → 100 (86) | 92.0 | 92.0 | 94.2 | +0.0 |
| XCOPA | VI có dấu | 100 → 100 (86) | 80.0 | 80.0 | 81.4 | +0.0 |
| XCOPA | VI không dấu | 100 → 100 (86) | 57.0 | 58.0 | 59.3 | +1.0 |
| Tất cả | EN | 1000 → 987 (694) | 86.1 | 87.5 | 88.0 | +1.4 |
| Tất cả | VI có dấu | 1463 → 1432 (1125) | 79.2 | 81.2 | 80.7 | +2.0 |
| Tất cả | VI không dấu | 1463 → 1432 (1125) | 62.2 | 63.1 | 63.5 | +0.9 |

## 6. So với điểm bản thử (toàn bộ dev, VI có dấu / không dấu)

| Model | VI có dấu: thử → cuối | VI không dấu: thử → cuối | EN: thử → cuối |
|---|---|---|---|
| decider-2b-text | 86.4 → 86.7 | 48.6 → 48.7 | 90.0 → 90.0 |
| decider-2b-keyed | 85.1 → 85.4 | 47.7 → 47.8 | 89.2 → 89.2 |
| decider-ft-ep2-text | 85.1 → 85.6 | 52.4 → 52.2 | 89.4 → 89.4 |
| decider-ft-ep2-keyed | 85.1 → 85.3 | 51.8 → 51.7 | 89.1 → 89.1 |
| Qwen3-1.7B-fullvocab-s1 | 80.3 → 80.7 | 62.7 → 62.7 | 87.3 → 87.3 |
| Qwen3-1.7B | 70.4 → 70.6 | 48.0 → 48.3 | 84.3 → 84.3 |
| strip-s1 | 81.1 → 81.2 | 62.9 → 63.1 | 87.6 → 87.5 |

Ô (model × nguồn × biến thể) nhảy quá 2 điểm so với 1.1: 43 (danh sách ở `scores_v1_2.json`).

## 7. Câu test đã xem

- `test_ids_nguoi_da_xem` (manifest test): 317 cặp (vi_proposals 251, label_decisions_remote 71, label_issues 70).
- `test_ids_agent_da_doc`: toàn bộ cặp test 1.1 (lượt kiểm tra cuối Opus đã đọc hết: verify + reread 40 lô).

## 8. File

- `data/bench_v1_2/`: `dev_v1_2.jsonl`, `test_v1.2.jsonl` (chỉ đọc), `FROZEN_test_v1.2.sha256`, `manifest_dev_v1_2.json`, `manifest_test_v1.2.json` (chỉ đọc), `leakage_v1_2.json`.
- `results/bench_v1_2_stage/`: bản dựng, `changes_v1_2_trial.jsonl`, `build_report_v1_2_trial.json`, `check_v1_2.json`, `diff_vs_trial_v1_2.json`, `rescore/`, `eval_rescore/`.
- `results/dev_v1_2/`: `eval/<model>/preds.jsonl` (preds dev 1.2 đã gộp), `scores_v1_2.json`, báo cáo này.
- Code: `scripts/human_edits/run_build_v1_2.sh`, `check_stage_v1_2.py`, `diff_stage_vs_trial_v1_2.py`, `rescore_dev_v1_2_final.sh`, `freeze_v1_2.py`, `report_v1_2.py`, `drop_dev_pair_via_tool.py`; `build_revised.py` (thêm chốt dev↔test).
