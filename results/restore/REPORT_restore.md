# Khôi phục dấu trước khi chấm (phần A)

Ngày 2026-10-13. Không train gì. Không ghi gì vào `data/bench_v1`. Không model nào chạy trên test 1.0 và không mở file test 1.0. Mọi số đo trên dev_v1 (SHA-256 khớp manifest, `89455863…`), batch 1. Bảng đầy đủ ở `tables_restore.md`, số máy đọc được ở `metrics_restore.json`.

## Trả lời câu hỏi

**Đặt bộ khôi phục dấu trước model, điểm lát "VI không dấu" hồi lại được bao nhiêu so với bản có dấu?** Lấy khối song song (4 nguồn), kết quả chính là đầu ra thô của bộ khôi phục:

| Model | có dấu | không dấu | khôi phục vit5 | còn thiếu so với có dấu | khôi phục small | còn thiếu so với có dấu |
|---|---|---|---|---|---|---|
| adapter đối chứng | 79,8 | 58,8 | 74,5 (hồi 75% khoảng cách) | −5,3 [−7,8; −2,8] | 69,0 (49%) | −10,8 [−13,4; −8,2] |
| adapter bỏ dấu (strip) | 80,8 | 60,8 | 74,9 (70%) | −5,9 [−8,4; −3,4] | 70,2 (47%) | −10,6 [−13,3; −7,9] |
| decider-2b | 86,0 | 49,2 | 81,9 (89%) | −4,1 [−6,3; −2,0] | 74,2 (68%) | −11,8 [−14,7; −8,9] |

Khối bản địa (3 nguồn): với vit5, cả ba model còn thiếu 1,6–2,5 điểm so với bản có dấu. Với small, mức thiếu là 2,8–8,8 điểm.

**Cách đọc đã định trước.** Đo khoảng cách của decider-2b giữa bản có dấu và bản khôi phục, ở khối song song, trên đầu ra thô. Áp riêng cho từng bộ:

| Bộ khôi phục | Khoảng cách | Vùng |
|---|---|---|
| **vit5** | **4,1** [2,0; 6,3] | **≤ 5: lợi thế không dấu của model dự án không còn khi so theo pipeline** |
| **small** | **11,8** [8,9; 14,7] | **> 10: khả năng chịu không dấu của chính model vẫn có giá trị** |

Hai bộ rơi vào hai vùng khác nhau. Tôi báo cả hai, không gộp và không chọn bộ nào. Cả hai bộ được chấm và báo đầy đủ; không có bước nào chọn bộ "tốt nhất" theo điểm dev_v1.

---

## 1. Đã chạy gì

### 1.1 Bộ khôi phục

| | vit5 | small |
|---|---|---|
| Repo @ revision (ghim ở `config/restorers.json`) | nrl-ai/vn-diacritic-vit5-base @ `30ea5a9e4a0b9436e18915fd4dbb5876eaee7325` | nrl-ai/vn-diacritic-small @ `379ae2d32b4fd3c232c2f579682954d1d035c9c0` |
| Kiến trúc | T5ForConditionalGeneration, khoảng 220M | MBartForConditionalGeneration (BARTpho-syllable), khoảng 115M |
| Giấy phép (thẻ YAML và thân card) | apache-2.0 | apache-2.0 |
| trust_remote_code | **không cần** | **không cần** |
| Dữ liệu train: YAML `datasets:` (nguyên văn) | `hirine/wikipedia-vietnamese-1M296K-dataset`, `tmnam20/Vietnamese-News-dedup` | như vit5 |
| Dữ liệu train: thân card | 594.945 cặp; trích: "register-balanced mix of Vietnamese Wikipedia (CC-BY-SA-4.0) and Vietnamese news" | 500.000 cặp, cùng hỗn hợp |
| Cổng chất lượng của tác giả | "passed" | "did not pass" (business_55 word_accuracy 0,9444 < 0,96) |

Có một chỗ sai nhỏ trên card của small: phần License ghi BARTpho là của "VietAI", nhưng repo gốc `vinai/bartpho-syllable-base` thuộc VinAI.

**Không bộ nào cần thêm model ngoài.** Brief cho phép thêm tối đa một model. Ứng viên `Toshiiiii1/Vietnamese_diacritics_restoration_5th` có thẻ apache-2.0 nhưng card trống, không có cách dùng hay dữ liệu train, nên tôi không thêm.

**Tokenizer của vit5.** transformers 5.19 không dựng được `T5Tokenizer` từ repo này (TypeError ở 'vocab'). Tôi nạp `tokenizer.json` của chính repo bằng `PreTrainedTokenizerFast`.
- Đã kiểm trên 7.254 chuỗi tiếng Việt của dev_v1: sau khi bỏ khoảng trắng đầu/cuối, id trùng 100% với sentencepiece `spiece.model` (+ `</s>`). Kết quả ở `tok_check_vit5.json`.
- Mọi chuỗi đưa vào model đều đã bỏ khoảng trắng đầu/cuối; khoảng trắng gốc được giữ lại bên ngoài.

### 1.2 Cách khôi phục (`scripts/restore/restore_dev.py`)

- **Đầu vào:** 1.763 dòng tiếng Việt không dấu toàn bộ của dev_v1. Khôi phục cả state, câu hỏi và từng lựa chọn. Kết quả ghi ở `results/restore/rows/`.
- **Cách gọi theo card:** `AutoModelForSeq2SeqLM`, fp32, greedy, `max_length=256`, batch 1 (mỗi đoạn một lần gọi).
- **Tách văn bản:** tách theo xuống dòng, rồi theo ranh giới câu (sau . ! ? … và khoảng trắng), rồi gom các câu liền nhau thành đoạn tối đa 160 token.
  - Lý do chọn 160: trên state dev_v1, tỉ lệ token bản có dấu / bản không dấu có p99 = 1,5 (`tok_ratio.py`), nên đầu ra khoảng 240 token, dưới 256. Con số này chọn theo thống kê tokenizer, không theo điểm.
  - Câu dài hơn 160 token thì tách tại khoảng trắng: 19 câu (vit5), 21 câu (small).
  - Không cắt bỏ phần nào. Các đoạn được ghép lại bằng đúng khoảng trắng gốc.
- **Log từng lần gọi:** đầu vào, đầu ra, có chạm max_length không. File `chunks_<bộ>.jsonl`.

### 1.3 Bản "đã sửa", dùng làm kiểm độ nhạy (`scripts/restore/make_fixed.py`)

Dựng được trong pipeline thật vì lúc suy luận vẫn có đầu vào không dấu. Quy tắc áp riêng cho state, câu hỏi và từng lựa chọn:

1. Tách bản không dấu N theo khoảng trắng; giữ nguyên mọi khoảng trắng của N.
2. Tách đầu ra thô R theo khoảng trắng.
3. Khóa của một từ là các ký tự chữ-số của nó. Với từ của R thì lấy khóa sau khi bỏ dấu. Căn hai dãy khóa bằng difflib.
4. **Theo từ:** từ của N nằm trong khối khớp thì lấy dấu từ từ tương ứng của R, **theo từng ký tự**. Ký tự không khớp, thường là dấu câu R làm rơi như ngoặc kép cong, giữ nguyên ký tự của N. Nhờ đó dấu câu bị rơi được giữ lại.
5. Từ không khớp (R bỏ sót, lặp, bịa, hoặc bị cắt ở max_length) thì giữ nguyên từ không dấu.
6. Kiểm bằng `assert strip(F) == N`, khớp cả khoảng trắng. Bất biến đúng 100% ở cả hai bộ.

Tỉ lệ từ phải giữ nguyên không dấu, theo nguồn:

| Bộ | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC | ViHSD |
|---|---|---|---|---|---|---|---|
| vit5 | 0,37% | 0,03% | 0,03% | 0,00% | 0,56% | 0,25% | 0,34% |
| small | 10,62% | 2,14% | 1,32% | 0,33% | 17,81% | 7,41% | 1,27% |

### 1.4 Chấm

- **Cổng G1 (trước mọi lượt chấm):** chấm lại adapter đối chứng trên dev_v1 cho preds trùng từng byte với bản đã lưu (sha `1468dcca…`). **Đạt.**
- **Ba model × (2 bộ thô + 2 bộ đã sửa) = 12 lượt chấm.** Đường chấm không đổi, batch 1:
  - adapter đối chứng (`results/ft_loss`) và adapter bỏ dấu (`results/ft_strip`): `scripts/eval_base.py`;
  - decider-2b: `scripts/bench/eval_decider_v1.py`, câu choice ánh xạ "text" như ở phần 5a.
- Số có dấu và không dấu lấy từ preds đã lưu. Đã kiểm: khớp đúng các bảng của strip_v0 (79,8 / 58,8 / 66,2 / 53,9 của đối chứng).
- decider-2b có 4 (vit5) và 2 (small) dòng dài quá 1.536 token. Giống phần 5a, giới hạn đã được nâng lên 32.768 token nên không dòng nào bị cắt.

## 2. Số đo

### 2.1 Accuracy (%, trung bình đều theo nguồn)

| Khối | Model | có dấu | không dấu | khôi phục vit5 | khôi phục small | đã sửa vit5 | đã sửa small |
|---|---|---|---|---|---|---|---|
| song song (4 nguồn) | đối chứng | 79,8 | 58,8 | 74,5 | 69,0 | 74,6 | 69,5 |
| song song (4 nguồn) | strip | 80,8 | 60,8 | 74,9 | 70,2 | 74,8 | 70,2 |
| song song (4 nguồn) | decider-2b | 86,0 | 49,2 | 81,9 | 74,2 | 81,8 | 75,8 |
| song song · MASSIVE + XCOPA | đối chứng | 80,0 | 59,3 | 73,0 | 67,7 | 72,8 | 68,2 |
| song song · MASSIVE + XCOPA | strip | 82,2 | 61,8 | 74,0 | 68,8 | 73,7 | 69,8 |
| song song · MASSIVE + XCOPA | decider-2b | 89,3 | 55,8 | 83,8 | 76,8 | 83,8 | 77,5 |
| bản địa (3 nguồn) | đối chứng | 66,2 | 53,9 | 63,7 | 57,4 | 63,5 | 56,9 |
| bản địa (3 nguồn) | strip | 66,3 | 56,9 | 64,5 | 63,5 | 64,4 | 64,3 |
| bản địa (3 nguồn) | decider-2b | 66,1 | 42,9 | 64,5 | 60,7 | 64,4 | 63,9 |

CI 95% (bootstrap theo state) và chênh lệch theo cặp kèm McNemar ở `tables_restore.md`: theo khối ở mục 1–2, theo nguồn ở mục 5 và 5b.

### 2.2 Cách đọc định trước và các lát đối chiếu (decider-2b: có dấu − khôi phục, điểm)

| Khối | khôi phục vit5 | khôi phục small | đã sửa vit5 (kiểm độ nhạy) | đã sửa small (kiểm độ nhạy) |
|---|---|---|---|---|
| **song song (4 nguồn), lát định trước** | **4,1 [2,0; 6,3] → ≤ 5** | **11,8 [8,9; 14,7] → > 10** | 4,2 [2,1; 6,5] → ≤ 5 | 10,2 [7,5; 12,9] → > 10 |
| song song · chỉ MASSIVE + XCOPA | 5,5 [1,5; 9,3] → ở giữa | 12,5 [8,0; 17,3] → > 10 | 5,5 [1,5; 9,5] → ở giữa | 11,8 [7,2; 16,3] → > 10 |
| bản địa (3 nguồn) | 1,6 [−0,6; 3,8] → ≤ 5 | 5,4 [2,5; 8,5] → ở giữa | 1,7 [−0,5; 3,9] → ≤ 5 | 2,2 [−0,5; 5,0] → ≤ 5 |

### 2.3 So theo pipeline: model dự án trừ decider-2b sau khôi phục (cùng câu)

| Khối | strip không dấu − decider + vit5 | strip + vit5 − decider + vit5 | strip không dấu − decider + small | strip + small − decider + small |
|---|---|---|---|---|
| song song (4 nguồn) | −21,1 [−25,0; −17,1] | −7,0 [−10,0; −3,9] | −13,3 [−17,6; −9,3] | −3,9 [−7,1; −0,9] |
| song song · MASSIVE + XCOPA | −22,0 [−28,5; −15,5] | −9,8 [−15,2; −4,7] | −15,0 [−21,5; −8,2] | −8,0 [−13,2; −2,7] |
| bản địa (3 nguồn) | −7,6 [−12,0; −3,1] | 0,0 [−4,2; +4,1] | −3,8 [−8,5; +0,9] | +2,8 [−0,7; +6,4] |

Adapter đối chứng cho kết quả cùng chiều (`tables_restore.md`, mục 4).

### 2.4 Chất lượng khôi phục so với bản có dấu gốc

So trên state, sau NFC, phân biệt hoa/thường, bỏ qua khoảng trắng.

| Bộ | Nguồn | âm tiết đúng | âm tiết có dấu đúng | state khớp hoàn toàn | câu hỏi khớp | mọi lựa chọn khớp |
|---|---|---|---|---|---|---|
| vit5 | Belebele / SIB-200 / MASSIVE / XCOPA | 97,3 / 97,8 / 90,0 / 92,9 | 97,3 / 97,7 / 89,2 / 93,0 | 23,0 / 65,0 / 52,7 / 68,0 | 82,0 / 100 / 100 / 100 | 58,7 / 100 / 4,0 / 47,0 |
| vit5 | ViFactCheck / ViWikiFC / ViHSD | 93,7 / 96,9 / 73,2 | 93,6 / 97,0 / 73,0 | 0,6 / 37,7 / 8,3 | 63,2 / 70,3 / 100 | 100 / 100 / 52,0 |
| vit5 | tất cả | **94,5** | 94,5 | 35,7 | 88,5 | 65,5 |
| small | Belebele / SIB-200 / MASSIVE / XCOPA | 82,6 / 96,6 / 87,5 / 91,6 | 82,4 / 96,5 / 87,4 / 91,7 | 5,3 / 50,3 / 45,0 / 61,0 | 69,7 / 100 / 100 / 100 | 45,3 / 0,0 / 3,7 / 36,0 |
| small | ViFactCheck / ViWikiFC / ViHSD | 77,9 / 84,9 / 68,5 | 77,8 / 85,0 / 68,9 | 0,0 / 16,3 / 8,7 | 42,3 / 51,3 / 48,0 | 52,1 / 57,0 / 0,0 |
| small | tất cả | **80,8** | 80,7 | 24,8 | 72,4 | 24,9 |

Bản đã sửa: âm tiết đúng 96,8% (vit5) và 83,6% (small).

### 2.5 Bất biến: bỏ dấu bản khôi phục phải ra lại đúng bản không dấu (chỉ cho lệch khoảng trắng)

| Bộ | Dòng vi phạm | state / câu hỏi / lựa chọn | Đoạn chạm max_length | Chỉ mất hoặc thêm ký tự ngoài chữ-số | Có đổi chữ-số |
|---|---|---|---|---|---|
| vit5 | **372 / 1.763 (21,1%)** | 315 / 20 / 57 | 6 dòng | 195 | 171 |
| small | **1.198 / 1.763 (68,0%)** | 782 / 93 / 477 | 0 | 38 | 1.160 |

Vi phạm theo nguồn:
- vit5: ViFactCheck 95, Belebele 78, ViHSD 67, ViWikiFC 58, MASSIVE 53, SIB-200 21, XCOPA 0.
- small: SIB-200 300, Belebele 257, ViWikiFC 197, ViFactCheck 163, MASSIVE 159, ViHSD 114, XCOPA 8.

**Ký tự bị rơi:**
- vit5: phần "chỉ mất ký tự ngoài chữ-số" chủ yếu là ngoặc kép cong “ (207 lần) và ” (202). Hai ký tự này không có trong bảng từ của ViT5: 290/1.763 state có dấu của dev_v1 chứa ký tự ngoài bảng từ.
- small: phần lớn là chữ cái (a ×10.020, n ×9.318, …), tức rơi hoặc bịa cả từ.

Năm ví dụ, mỗi trường cắt 200 ký tự, in ở `rows/check_vit5.log` và `rows/check_small.log`. Ví dụ đổi chữ của vit5, ngoài ViHSD: '¾' thành '34'; 'tngam' thành 'suu'; rơi một chữ 'a'.

### 2.6 Lỗi có hệ thống ở lựa chọn (đầu ra thô)

| Bộ | Nguồn | Dòng có lựa chọn bị đổi chữ-số | Hai lựa chọn trùng nhau | Lựa chọn đúng mất từ |
|---|---|---|---|---|
| vit5 | MASSIVE / Belebele / còn lại | 47 / 6 / 0 | 0 | 0 |
| small | SIB-200 / MASSIVE / Belebele / XCOPA / bản địa | **300 / 145** / 25 / 7 / 0 | 0 | 40 / 11 / 7 / 4 / 0 |

- small biến lựa chọn "sức khỏe" của SIB-200 thành "sức" ở cả 300/300 dòng; bản đã sửa thành "sức khoe".
- Ở MASSIVE, small sinh "lạc đà" thay cho "lạc đề" (61 lần) và "sự kiện khơi lịch" thay cho "xóa sự kiện khỏi lịch" (56 lần).
- **SIB-200 với small: toàn bộ phần tụt nằm ở 40 câu có đáp án "sức khỏe".**

| decider-2b, SIB-200 | có dấu | khôi phục small |
|---|---|---|
| 40 câu đáp án "sức khỏe" | 75% | 5% |
| 260 câu còn lại | 83,5% | 83,8% |

- Hai adapter cũng vậy: 90% thành 48% (đối chứng) và 82% thành 50% (strip) ở 40 câu này. Bản đã sửa ("sức khoe") còn tệ hơn: 20% và 12%.
- Với vit5, lựa chọn SIB không bị đổi; decider-2b giữ 75% ở 40 câu này.

### 2.7 Độ trễ của bộ khôi phục

Đo ở batch 1, cho state + câu hỏi + lựa chọn của một câu; fp32, greedy, theo cách dùng trên card.

| State (token Qwen) | dòng | vit5 p50 / p90 (ms) | small p50 / p90 (ms) |
|---|---|---|---|
| ≤ 128 | 1.319 | 943 / 1.651 | 578 / 894 |
| 129–512 | 308 | 2.487 / 3.927 | 1.304 / 1.967 |
| 513–1.024 | 95 | 8.391 / 10.663 | 3.891 / 4.912 |
| > 1.024 | 41 | 13.591 / 18.642 | 6.615 / 8.859 |
| tất cả | 1.763 | 1.052 / 3.563 (trung bình 1.988) | 639 / 1.857 (trung bình 1.034) |

Đặt cạnh độ trễ của model quyết định đã đo trước đây (batch 1, bf16):
- Qwen3-1.7B + adapter: p50 29 / 50 / 181 ms ở prompt 128 / 512 / 2.048 token.
- Trung vị trên dev_v1: adapter đối chứng 41 ms, decider-2b 61 ms.

Như vậy bộ khôi phục chậm hơn model quyết định khoảng 20–75 lần. Tổng thời gian khôi phục 1.763 dòng là 58 phút (vit5) và 30 phút (small).

## 3. Diễn giải

Mọi điểm trong mục này là cách đọc số, không phải số đo.

1. **Theo cách đọc định trước, hai bộ cho hai kết luận ngược nhau.**
   - Với vit5 (4,1 → ≤ 5), lợi thế "chịu được không dấu" của model dự án so với decider-2b không còn khi so theo pipeline.
   - Với small (11,8 → > 10), khả năng chịu không dấu của chính model vẫn có giá trị.
   - Bản đã sửa không đổi vùng ở lát định trước: vit5 4,2, small 10,2. Nhưng small 10,2 chỉ vừa qua mốc 10, và CI [7,5; 12,9] chứa 10.
   - **Bản đã sửa có đổi vùng ở một lát đối chiếu:** khối bản địa với small, từ "ở giữa" (5,4) sang "≤ 5" (2,2). Các lát đối chiếu khác không đổi vùng.
2. **Một phần mức tụt của small đến từ lỗi có hệ thống ở lựa chọn, không phải ở việc hiểu văn bản.**
   - Ở SIB-200, decider-2b mất 9,0 điểm, toàn bộ nằm ở 40 câu "sức khỏe". 9,0 điểm này đóng góp khoảng 2,3 điểm vào khoảng cách 11,8 (trung bình đều trên 4 nguồn).
   - Nếu không có lỗi này, khoảng cách của small sẽ khoảng 9,5, tức rơi vào vùng "ở giữa". Đây là ước lượng tính tay, không phải số đo, và không đổi cách đọc. Ngưỡng giữ nguyên, small vẫn ở trong báo cáo.
   - Bản đã sửa không gỡ được lỗi này, vì từ "khoe" vẫn không dấu. Mức tụt khác của small (rơi từ, bịa từ ở Belebele, ViFactCheck, ViWikiFC) được bản đã sửa gỡ một phần: decider-2b ở bản địa tăng từ 60,7 lên 63,9.
3. **Nếu chỉ dùng MASSIVE + XCOPA (hai nguồn ít rủi ro trùng nhất), vit5 chuyển từ "≤ 5" sang "ở giữa" (5,5 [1,5; 9,3]); small vẫn > 10.**
   - Ngưỡng không đổi. Đây chỉ là ghi chú: kết luận "≤ 5" của vit5 phụ thuộc một phần vào Belebele và SIB-200. Khoảng cách của decider-2b ở hai nguồn này là 4,0 và 1,3.
   - Độ chính xác khôi phục cũng cao hơn ở Belebele/SIB (97,3/97,8% âm tiết đúng) so với MASSIVE/XCOPA (90,0/92,9%).
4. **So thẳng theo pipeline, decider-2b + bộ khôi phục hơn model dự án ở khối song song với cả hai bộ.**
   - decider + vit5 81,9 so với strip trên không dấu 60,8 (−21,1). Kể cả decider + small 74,2 vẫn hơn strip trên không dấu (−13,3).
   - Khi cả hai cùng dùng bộ khôi phục, decider vẫn hơn: −7,0 (vit5), −3,9 (small).
   - Ở khối bản địa, strip + vit5 và decider + vit5 ngang nhau (0,0 [−4,2; +4,1]).
   - Điều này đúng ngay cả với small, tức là cả khi cách đọc định trước nói "vẫn có giá trị". Cách đọc định trước đo mức tụt của decider so với chính nó, không so trực tiếp hai pipeline. Hai góc nhìn này được báo song song; tôi không thay ngưỡng.
5. **decider-2b hồi lại nhiều hơn model dự án:** 89% so với 70–75% khoảng cách với vit5.
   - Model dự án vốn đã chịu được không dấu một phần, nên phần còn lại để hồi ít hơn.
   - Văn bản khôi phục gần với văn bản có dấu mà decider được train.
6. **Cái giá của pipeline là độ trễ.** Ở state ≤ 128 token, bộ khôi phục mất khoảng 0,6–0,9 s mỗi câu, so với khoảng 30–60 ms của model quyết định. Số này đo ở fp32 và gọi tuần tự từng đoạn; chưa thử bf16 hay chạy theo lô.

## 4. Điều chưa chắc

- **Cả hai kết luận chính đều dựa trên ước lượng điểm.** Theo quy tắc định trước, vùng được xếp theo ước lượng điểm. Nhưng CI của cả hai đều chứa ranh giới: vit5 4,1 [2,0; 6,3] chứa 5; small 11,8 [8,9; 14,7] chứa 10.
- **Rủi ro trùng với dữ liệu train của bộ khôi phục (Wikipedia tiếng Việt + tin tức).**
  - ViWikiFC (Wikipedia) và ViFactCheck (tin tức) có thể có điểm "cao giả". vit5 hồi gần hết ở hai nguồn này: decider 81,0 thành 81,6 (ViFactCheck) và 76,7 thành 74,3 (ViWikiFC).
  - **Belebele và SIB-200 là văn bản FLORES-200 lấy từ Wikinews, Wikijunior và Wikivoyage, cùng họ Wikipedia và tin tức.** Khối song song đầy đủ vì vậy cũng mang rủi ro này. Lát MASSIVE + XCOPA (mục 2.2) là phép kiểm cho rủi ro đó: với vit5, vùng chuyển từ ≤ 5 sang "ở giữa".
  - Chưa kiểm trực tiếp xem văn bản dev_v1 có nằm trong hai bộ train của bộ khôi phục không.
- **ViHSD** (mạng xã hội, viết tắt, tiếng lóng) là nơi cả hai bộ khôi phục kém nhất (73,2% và 68,5% âm tiết đúng), và là nguồn có ít rủi ro trùng nhất. ViHSD và MASSIVE là hai nguồn đáng tin nhất để đọc mức hồi thật. Ở ViHSD, decider-2b sau vit5 đạt 37,7 so với 40,7 bản có dấu; strip đạt 49,0 so với 53,0.
- **Mỗi cấu hình chỉ chạy một lần**, greedy, một cách tách đoạn (160 token). Cách tách và cách giải mã khác (beam search) có thể đổi kết quả. Không thử, vì brief yêu cầu dùng như card.
- **Độ trễ** đo ở fp32, gọi tuần tự. Pipeline thật có thể nhanh hơn nhiều với bf16 và chạy theo lô; chưa đo.
- decider-2b chỉ chấm ở thứ tự lựa chọn thông thường, như ở phần 5a.
- Card của vit5 khuyên dùng gói `nom-vn`, có thêm lớp luật và fallback bằng LLM. Tôi chỉ dùng model trần theo "Quick start", nên kết quả có thể thấp hơn pipeline đầy đủ của tác giả.

## 5. File

- `rows/`:
  - `dev_v1_vi_restored_{vit5,small}.jsonl` (đầu ra thô) và `…_{vit5,small}_fixed.jsonl` (bản đã sửa);
  - `log_*.jsonl` (thời gian từng dòng), `chunks_*.jsonl` (đầu vào/đầu ra từng lần gọi), `meta_*.json`;
  - `quality_*.json`, `fixed_stats_*.json`, `check_*.log`.
- `eval/{vit5,small,vit5_fixed,small_fixed}/{control-s1,strip-s1,decider-2b-text,decider-2b-keyed}/`: preds.
- `gates/g1/`: cổng G1. `pipeline.log`: nhật ký.
- `tables_restore.md`, `metrics_restore.json`: bảng đầy đủ.
- `tok_check_vit5.json`: kiểm tokenizer.
- Code: `scripts/restore/{restore_dev,check_restore,make_fixed,report_restore,tok_check_vit5,tok_ratio}.py`, `setup/restore_*.sh`, `config/restorers.json`.
