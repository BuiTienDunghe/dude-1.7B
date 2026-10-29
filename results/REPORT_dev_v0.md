# Đo hai model gốc Qwen3 trên dev_v0 (Anh–Việt)

Ngày 2026-10-12. Không train gì. Số liệu dưới đây lấy từ `results/run1`; `results/run2` chạy lại độc lập và **giống hệt từng byte**: cả `preds.jsonl`, `metrics.json` và `report.md` đều trùng SHA-256.

## Thiết lập

- **Model** (revision ghim trong `config/models.json`):
  - `Qwen/Qwen3-0.6B@c1899de2…`
  - `Qwen/Qwen3-1.7B@70d244cc…`
  - Hai model dùng chung tokenizer và chat template (đã so vocab và hash template).
- **Cách chấm:** chat template với `enable_thinking=False`, mỗi câu một forward pass. Lấy logit của token kế tiếp tại các chữ A..(k), rồi softmax trên đúng k chữ đó.
  - Đã kiểm tra A..J là một token ngay tại ranh giới trả lời: `encode(prompt+L)[:-1] == encode(prompt)`.
  - Chạy bf16, sdpa, batch 1, `torch.use_deterministic_algorithms(True)`.
- **Phần cứng / phần mềm:** RTX 5060 Ti 16 GB (sm_120), driver 591.86, WSL2 Ubuntu 26.04, Python 3.12.15, torch 2.11.0+cu128, transformers 5.19.0. Danh sách gói đầy đủ ở `requirements.lock.txt`.
- **Tập đo:** 1000 cặp, tức 2000 câu: Belebele 300, SIB-200 300, MASSIVE 300, XCOPA 100. Seed 20261012. Chi tiết ở `data/manifest_dev_v0.json`.

## Kết quả

Accuracy tính bằng %, kèm [CI 95% bootstrap]: 2000 lần lấy mẫu lại theo state. Với Belebele, state là đoạn văn (link); với các nguồn khác, mỗi câu là một state. Trung bình đều được tính bằng cách lấy mẫu lại trong từng nguồn rồi lấy trung bình của mỗi lần lặp.

Brier là tổng đa lớp Σ(p−y)², nên đoán đều cho đúng 1−1/k.

| Nguồn | k | Đoán đều | Nhãn phổ biến¹ | 0.6B en | 0.6B vi | 1.7B en | 1.7B vi |
|---|---|---|---|---|---|---|---|
| Belebele | 4 | 25.0 | 27.0 (chữ) | 56.3 [50.7, 62.0] | 46.3 [40.5, 52.2] | 79.0 [74.3, 83.7] | 67.3 [61.8, 72.8] |
| SIB-200 | 7 | 14.3 | 18.0 (lớp) | 70.0 [64.7, 75.0] | 70.7 [65.7, 75.7] | 80.7 [76.3, 84.7] | 74.0 [68.7, 78.7] |
| MASSIVE | 10 | 10.0 | 10.5 (lớp) | 78.0 [73.0, 82.3] | 64.0 [58.7, 69.3] | 88.0 [84.3, 91.3] | 71.3 [66.0, 76.7] |
| XCOPA | 2 | 50.0 | 51.0 (chữ) | 73.0 [64.0, 81.0] | 61.0 [51.0, 71.0] | 86.0 [79.0, 93.0] | 79.0 [71.0, 87.0] |
| **Trung bình đều** | | 24.8 | | **69.3** [66.2, 72.5] | **60.5** [57.0, 64.1] | **83.4** [81.0, 85.8] | **72.9** [69.9, 75.8] |

| Brier | 0.6B en | 0.6B vi | 1.7B en | 1.7B vi | Đoán đều |
|---|---|---|---|---|---|
| Belebele | 0.695 | 0.748 | 0.397 | 0.564 | 0.750 |
| SIB-200 | 0.512 | 0.462 | 0.366 | 0.481 | 0.857 |
| MASSIVE | 0.356 | 0.518 | 0.229 | 0.508 | 0.900 |
| XCOPA | 0.395 | 0.499 | 0.249 | 0.344 | 0.500 |
| Trung bình đều | 0.490 | 0.557 | 0.310 | 0.474 | 0.752 |

¹ Có hai loại mốc "nhãn phổ biến":
- **(lớp)**: luôn chọn lớp gốc phổ biến nhất trong dev, là "khoa học/công nghệ" với SIB-200 và một ý định với MASSIVE. Với MASSIVE, nếu nhãn đó không nằm trong 10 lựa chọn của câu thì tính như đoán đều.
- **(chữ)**: luôn chọn chữ cái đúng phổ biến nhất. Dùng cho Belebele và XCOPA vì hai bộ này không có lớp.

Lấy mẫu cân theo nhãn nên cả hai mốc đều sát mức đoán đều.

### Mức tụt Anh → Việt (điểm %, ghép theo pair_id, McNemar chính xác)

| Nguồn | 0.6B tụt [CI] | p | 1.7B tụt [CI] | p |
|---|---|---|---|---|
| Belebele | 10.0 [4.4, 15.9] | 0.001 | 11.7 [6.3, 16.9] | 1.6e-05 |
| SIB-200 | −0.7 [−6.3, 5.0] | 0.90 | 6.7 [1.3, 12.0] | 0.015 |
| MASSIVE | 14.0 [9.0, 19.0] | 2.8e-07 | 16.7 [12.0, 21.3] | 3.0e-11 |
| XCOPA | 12.0 [3.0, 21.0] | 0.023 | 7.0 [−3.0, 16.0] | 0.21 |
| Trung bình đều² | 8.8 [5.5, 12.0] | 1.3e-07 | 10.5 [7.0, 13.6] | 5.5e-15 |

### 1.7B trừ 0.6B (điểm %, cùng câu, McNemar chính xác)

| Nguồn | en [CI] | p | **vi** [CI] | p |
|---|---|---|---|---|
| Belebele | +22.7 [17.2, 28.6] | 7e-13 | **+21.0** [15.1, 27.1] | 1e-10 |
| SIB-200 | +10.7 [5.0, 16.3] | 3e-04 | **+3.3** [−2.0, 8.7] | 0.28 |
| MASSIVE | +10.0 [5.3, 14.3] | 2e-05 | **+7.3** [1.7, 13.0] | 0.017 |
| XCOPA | +13.0 [3.0, 23.0] | 0.019 | **+18.0** [6.0, 29.0] | 0.005 |
| Trung bình đều² | +14.1 [10.9, 17.5] | 8e-21 | **+12.4** [8.6, 16.2] | 1.5e-11 |

² Ở hàng trung bình đều, mức tụt hoặc hiệu là trung bình của bốn nguồn. Riêng p là McNemar gộp tất cả các cặp bất đồng.

### Tốc độ (bf16, sdpa, đo trên phần cứng nêu ở trên)

| | Trung vị độ trễ batch 1 | p90 batch 1 | Số câu/giây ở batch 16 | Batch 16 cùng argmax với batch 1 |
|---|---|---|---|---|
| Qwen3-0.6B | 32.2 ms | 36.9–37.3 ms | **118–151** (dao động giữa hai lượt) | 98.1% |
| Qwen3-1.7B | 40.6–40.8 ms | 51.2–53.0 ms | 56.4–56.5 | 98.7% |

- Độ trễ batch 1 đo trên toàn bộ 2000 câu, sau 10 câu warmup. Prompt có trung vị 134 token.
- Batch 16 dùng left padding và sắp câu theo độ dài. Thời gian đo có tính cả bước padding trên CPU. Số của 0.6B dao động mạnh vì model nhỏ bị giới hạn bởi CPU và chi phí khởi chạy kernel.

## Nhận xét: model nào tốt hơn cho tiếng Việt

**Qwen3-1.7B tốt hơn rõ rệt.** Trên tiếng Việt, trung bình đều theo nguồn cao hơn **+12.4 điểm** (72.9 so với 60.5; CI [8.6, 16.2]; p≈1.5e-11).

- Khoảng cách lớn ở các bài cần suy luận: đọc hiểu (Belebele) +21 điểm, nhân quả (XCOPA) +18 điểm.
- MASSIVE chỉ hơn +7 điểm.
- SIB-200 không có khác biệt có ý nghĩa (+3.3, p=0.28).
- 1.7B cũng có Brier tốt hơn ở tiếng Việt (0.474 so với 0.557).

Cái giá là batch 1 chậm hơn khoảng 1.26 lần, và thông lượng khi gom batch thấp hơn khoảng 2–2.7 lần.

Cả hai model đều tụt khoảng 9–10 điểm từ Anh sang Việt. 1.7B không thu hẹp khoảng cách này. Phần hơn của 0.6B ở XCOPA tiếng Việt thực ra phần lớn là do thiên vị chữ A (mục 1 bên dưới), nên con số 61% đó không đáng tin.

## Những điều bất thường

1. **0.6B thiên vị chữ A ở XCOPA tiếng Việt.** Model chọn A ở 80/100 câu.
   - Khi đáp án đúng là A: đúng 46/51 câu (90%).
   - Khi đáp án đúng là B: đúng 15/49 câu (31%).
   - Vì vậy accuracy 61% chủ yếu đến từ vị trí, không phản ánh năng lực suy luận. Ở Belebele, 0.6B nghiêng về A với tiếng Việt (118/300, đúng là 81) và về D với tiếng Anh (110/300, đúng là 77).
2. **1.7B dồn câu sai về chữ A ở MASSIVE tiếng Việt.** Model chọn A 81 lần trong khi chỉ 27 câu có đáp án A. Khoảng 2/3 số câu sai tiếng Việt rơi vào A. Ở tiếng Anh không có hiện tượng này (28 so với 27).
3. **1.7B đôi khi muốn trả lời bằng văn xuôi thay vì chữ cái.**
   - 18/2000 câu (0.9%) có dưới 50% xác suất dồn vào các chữ cái. Token đứng đầu là `The`, hoặc byte của chữ `Đ` (kiểu "Đáp án…").
   - Thấp nhất là 0.003, ở một câu Belebele tiếng Anh. Với những câu này, "quyết định" sau khi chuẩn hóa lại trên k chữ gần như là nhiễu.
   - 0.6B không bị: mức thấp nhất là 0.76.
   - Khối xác suất trên các chữ cái của từng câu được lưu ở trường `letter_mass` trong `preds.jsonl`.
4. **Gom batch làm đổi khoảng 1.3–1.9% quyết định.** Đó là ảnh hưởng của bf16 kết hợp left padding. Số chính thức vì thế lấy từ batch 1. Nếu sau này suy luận theo batch thì sẽ không tái tạo đúng từng câu.
5. **0.6B không tụt Anh→Việt ở SIB-200 (−0.7, p=0.90), trong khi 1.7B tụt 6.7 (p=0.015).** Có thể do 1.7B dùng ngữ cảnh tiếng Anh tốt hơn hẳn, còn bài phân loại chủ đề ở tiếng Việt đã chạm trần tương tự cho cả hai. Đây chỉ là phỏng đoán, tôi chưa kiểm chứng.
6. **Lần chấm đầu tiên bị treo khoảng 11 phút.** Tiến trình chờ futex, gần như không dùng CPU và không in ra gì. Tôi đã dừng nó. Năm lần chạy sau đó (một lần debug và bốn lần chính thức) đều không bị treo. Tôi chưa rõ nguyên nhân.
7. **Đặc điểm của dữ liệu nguồn:**
   - MASSIVE validation chỉ có 59/60 ý định: `audio_volume_other` không có ở validation. Ý định này chỉ có thể xuất hiện làm nhãn nhiễu.
   - `massive:5667` "tap start and then select settings" mang nhãn `alarm_query`, có vẻ là nhãn sai từ nguồn.
   - Bản Việt của Belebele `Hostels#1` dịch lệch: "allow them in private rooms" thành "có phòng dành riêng cho trẻ em".
   - Một số đoạn Belebele có dấu `"` thừa ở đầu, do lỗi từ file nguồn.
8. **AmazonScience/massive gốc không tải được.** `datasets` 5.1 từ chối loading script. Tôi dùng mirror `mteb/amazon_massive_intent` đúng như anh/chị đã dặn. Card của mirror ghi license apache-2.0, còn bản gốc là cc-by-4.0.

## Kiểm tra trước khi báo xong

- [x] Đã in 5 cặp Anh–Việt mỗi nguồn: `results/check_dev_v0.txt`. Tôi đã đọc và các cặp khớp nghĩa.
- [x] Mốc đoán đều bằng đúng 1/k ở từng nguồn: 25.0, 14.3, 10.0, 50.0. Brier đoán đều bằng đúng 1−1/k.
- [x] Chạy lại lần hai ra cùng con số: `compare_runs.py` báo IDENTICAL. Dựng lại `dev_v0.jsonl` và `reserved_test.json` cũng ra đúng SHA-256 cũ.
- [x] Không câu nào thuộc phần để dành cho test. `check_dev_v0.py` kiểm tra bằng assert theo link Belebele, index_id SIB, id MASSIVE, split XCOPA.

## Cần anh/chị duyệt tay

1. **`config/translations.json` (`reviewed: false`)**, gồm:
   - câu hỏi SIB, MASSIVE, XCOPA;
   - tên 7 chủ đề;
   - mô tả 60 ý định.
   Tôi tự viết cả phần **vi lẫn en** của mô tả ý định và tên chủ đề, nên phần tiếng Anh cũng cần duyệt.
2. **`config/prompt_frames.json`**: câu hướng dẫn và các nhãn khung "Văn bản / Câu hỏi / Các lựa chọn", cả bản Việt và bản Anh.
3. **Nguồn COPA tiếng Anh:**
   - Lấy từ `aps/super_glue` config `copa`, split validation, `@3de24cf…`. Khớp 100/100 theo idx, cùng loại câu hỏi và cùng nhãn.
   - Card trên HF ghi license là "other". Trang COPA gốc (asgordon.github.io/copa.html) ghi **BSD 2-Clause**.
   - Anh/chị xác nhận có chấp nhận chuỗi license này không.
4. **Các câu XCOPA có `changed=True` (idx 14, 21, 22):** tôi xem thấy vẫn khớp nghĩa, và đã gắn cờ trong trường `flags`.
5. **Cách chia Belebele:** 488 đoạn văn chia thành 244 cho dev và 244 để dành test (danh sách ở `data/reserved_test.json`). Nửa dev có 445 câu, đã dùng 300. 145 câu còn lại được ghi là "unused_not_reserved", có thể dùng cho dev_v1.
6. **Hai lỗi nguồn ở mục 7 phần bất thường** (`massive:5667`, Belebele `Hostels#1`): quyết định giữ hay loại.

## Danh sách file

| File | Nội dung |
|---|---|
| `data/dev_v0.jsonl` | 2000 dòng gồm id, pair_id, source, lang, kind, state, question, options, target, split_origin, n_tokens, cùng cluster, gold_label, flags |
| `data/manifest_dev_v0.json` | revision dataset, license, seed, SHA-256, thống kê, phần để dành, phần chưa dùng, ghi chú |
| `data/reserved_test.json` | danh sách các phần để dành cho test |
| `config/translations.json`, `config/prompt_frames.json`, `config/models.json` | bản dịch, khung prompt, revision model |
| `results/run{1,2}/<model>/preds.jsonl` | dự đoán thô từng câu: probs, pred, correct, letter_mass, top_token |
| `results/run{1,2}/<model>/timing.json` | tốc độ, phần cứng, phiên bản phần mềm |
| `results/run{1,2}/metrics.json`, `report.md` | toàn bộ chỉ số, kể cả phân bố chữ cái theo từng nguồn và ngôn ngữ |
| `scripts/` | `inspect_schemas.py`, `check_pairing.py`, `check_copa.py` (in schema, kiểm tra khóa ghép); `build_dev_v0.py`; `check_dev_v0.py`; `eval_base.py`; `report.py`; `compare_runs.py`; `inspect_anomalies.py` |
| `setup/` | `install_env.sh` (venv `.venv` bằng uv), `run.sh`, `run_all.sh` |
