# Chấm các decision model khác trên DEV 1.2 (bước E)

Ngày 2026-10-16. Chỉ inference trên `data/bench_v1_2/dev_v1_2.jsonl` (sha `ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b`, đã kiểm trước mỗi lượt chấm). Không đọc, không chạy model nào trên test; không train.

- **Bảng đầy đủ:** `results/baselines_v1_2/tables_baselines_v1_2.md` (accuracy theo nguồn, CI, Brier, so với dự án theo nguồn, đảo thứ tự, độ trễ, trạng thái kiểm adapter). Số liệu thô: `metrics_baselines_v1_2.json`.
- **Model theo danh sách gốc của brief:**
  - Qwen/Qwen3.5-2B gốc: chấm lại trên dev 1.2.
  - Mapika/decider-2b: dùng lại preds dev 1.2; chấm thêm thứ tự đảo.
  - mstrasser/Jeff-Qwen3.5-2B: **bị chặn** (mục 5).
  - chinhnc/Quyet-1.0-Tiny, -Small, -Medium.
  - convaiinnovations/laya-multilingual, vanty120/laya-vi.
  - Không chấm Qwen3.5-2B-Base.
- **Mốc dự án:** Qwen3-1.7B gốc và Qwen3-1.7B adapter full_vocab (gọi tắt là "dự án"), dùng preds dev 1.2 đã có, chấm thêm thứ tự đảo. Qwen3-0.6B ft chưa có preds trên dev 1.2 nên không đưa vào.
- **Cách tính:**
  - Accuracy là **trung bình đều theo nguồn**: 6 nguồn cho tiếng Việt (gồm ViFactCheck, ViWikiFC), 4 nguồn song song cho tiếng Anh. Con số này khác số "trung bình trên mọi câu" trong REPORT_bench_v1_2.
  - CI 95% là bootstrap theo state.
  - "Chỉ dòng source" là các cặp có văn bản tiếng Việt và đáp án giữ như nguồn (vi_provenance = source và gold_provenance = source).

## Tóm tắt số liệu (chỉ model qua đủ kiểm adapter; không kết luận hướng đi)

| Model | Tham số | VI có dấu [CI] | VI có dấu, chỉ source | VI không dấu | EN [CI] | EN − VI | Nhất quán khi đảo (VI) | Brier VI | Độ trễ trung vị batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| **Quyet-1.0-Medium** | 4,66 B | **89,4** [87,8; 91,1] | 88,8 | 64,5 | **94,5** [93,1; 95,7] | 2,7 | 97,9 | 0,166 | 87,5 ms |
| **decider-2b (text)** | 1,88 B | **86,3** [84,4; 88,2] | 86,3 | 49,6 | 90,3 [88,4; 92,3] | 2,3 | 93,9 | 0,209 | 52,3 ms |
| decider-2b (keyed) | 1,88 B | 84,9 [82,9; 86,9] | 84,7 | 48,5 | 89,8 [87,9; 91,8] | 3,6 | 92,7 | 0,217 | 54,3 ms |
| **Qwen3-1.7B full_vocab (dự án)** | 2,03 B + LoRA | **79,7** [77,4; 82,0] | 79,7 | **62,3** | 88,0 [85,7; 90,0] | 5,5 | 90,4 | 0,294 | 32,9 ms |
| Qwen3-1.7B gốc | 2,03 B | 70,6 [68,1; 73,2] | 70,5 | 48,5 | 84,6 [82,2; 86,9] | 8,1 | 70,0 | 0,543 | 32,3 ms |
| Qwen3.5-2B gốc | 2,27 B trong repo (chỉ nạp phần văn bản) | 70,6 [68,1; 73,1] | 70,9 | 39,0 | 84,5 [81,8; 86,9] | 11,7 | 71,3 | 0,411 | 57,7 ms |

Ghi chú về cách tính hai cột cuối:
- **Nhất quán khi đảo:** tính trên câu KHÔNG phải Có/Không (noul) cho **mọi** model, kể cả mốc dự án, để cùng một tập câu.
- **Độ trễ:** trung vị batch 1 trên toàn bộ 3.851 dòng dev. Riêng decider-2b đo ở lượt đảo trên 3.445 dòng không phải noul: khác tập dòng nhưng cùng cách đo.
- McNemar và Brier theo từng nguồn cho cả ba biến thể có trong `tables_baselines_v1_2.md`.

Model nhỏ hơn (Quyet-1.0-Tiny, -Small, laya-multilingual, laya-vi): **chưa qua kiểm câu hiển nhiên**, không vào bảng chính. Số chỉ để tham khảo ở bảng phụ (mục 3b). Jeff-Qwen3.5-2B: **bị chặn** (mục 5).

## 1. Bảng cùng cỡ và ba câu hỏi

Hiệu = model − dự án, tính bằng điểm %. Cặp câu ghép trên cùng câu. p là McNemar chính xác gộp. Với decider-2b dùng ánh xạ "text" (điểm cao hơn, có lợi cho model ngoài).

| Model | Tham số | VI có dấu | EN | VI − dự án [CI] (p) | EN − dự án [CI] (p) | VI bỏ MASSIVE − dự án | Độ trễ |
|---|---|---|---|---|---|---|---|
| decider-2b (text) | 1,88 B | 86,3 | 90,3 | **+6,6** [+4,4; +8,9] (5e-09) | **+2,4** [+0,2; +4,6] (0,017) | +6,0 [+3,3; +8,6] | 52,3 ms |
| Jeff-Qwen3.5-2B | 2,21 B | bị chặn | bị chặn | — | — | — | — |
| Qwen3.5-2B gốc | 2,27 B | 70,6 | 84,5 | −9,2 [−12,2; −6,3] (3e-14) | −3,5 [−6,3; −0,8] (0,032) | −6,4 [−9,9; −3,0] | 57,7 ms |
| Qwen3-1.7B gốc | 2,03 B | 70,6 | 84,6 | −9,2 [−11,5; −6,7] (2e-18) | −3,4 [−5,7; −1,3] (0,004) | −8,7 [−11,3; −6,1] | 32,3 ms |
| **Qwen3-1.7B full_vocab (dự án)** | 2,03 B + LoRA | 79,7 | 88,0 | — | — | — | 32,9 ms |

**1. Tiếng Việt: dự án kém decider-2b 6,6 điểm** [4,4; 8,9]. Bỏ MASSIVE vẫn kém 6,0 [3,3; 8,6].
- Theo nguồn (decider − dự án, VI có dấu):
  - Belebele +5,4 (p=0,036);
  - MASSIVE +9,5 (6e-08);
  - ViFactCheck +10,8 (0,006);
  - ViWikiFC +6,3 (0,011);
  - XCOPA +8,0 (0,10);
  - SIB-200 −0,3 (1,0).
- Dự án hơn Qwen3.5-2B gốc và Qwen3-1.7B gốc, mỗi model 9,2 điểm.
- **Tiếng Việt không dấu thì ngược lại:** dự án 62,3, decider-2b 49,6, tức dự án hơn 12,7 [9,2; 16,1] (4e-18).

**2. Tiếng Anh: dự án kém decider-2b 2,4 điểm** [0,2; 4,6] (p=0,017).
- Bỏ MASSIVE còn 1,0 [−1,7; +3,8], không phân biệt được.
- decider-2b tụt từ Anh sang Việt 2,3 điểm, dự án tụt 5,5.

**3. Qwen3.5-2B gốc so với Qwen3-1.7B gốc** (Qwen3.5 − Qwen3, ghép cặp):
- **VI có dấu: −0,0** [−3,1; +3,2] (p=0,88).
  - Theo nguồn: MASSIVE −11,2 (p=0,001), ViFactCheck +12,7 (p=0,01).
  - Các nguồn khác không có ý nghĩa thống kê.
  - Bỏ MASSIVE: +2,2 [−1,3; +5,7].
- **EN: −0,1** [−3,0; +2,9] (p=0,87). Bỏ MASSIVE: +1,6 [−2,1; +5,2].
- **VI không dấu: −9,4** [−13,9; −5,0] (p=8e-10). Kém rõ ở SIB-200 (−21,4) và MASSIVE (−28,2).
- Qwen3.5-2B gốc là bản đã post-train. decider-2b fine-tune từ Qwen3.5-2B-**Base**, repo đó không chấm.

## 2. Nhóm lớn hơn: Quyet-1.0-Medium (4,66 B, Qwen3.5-4B + LoRA đã merge)

- VI có dấu **89,4** [87,8; 91,1], tức hơn dự án **9,7** [+7,3; +12,2] (p=1e-15).
  - Bỏ MASSIVE vẫn +9,7.
  - Theo nguồn:
    - Belebele +11,6;
    - XCOPA +15,0;
    - ViFactCheck +15,8;
    - MASSIVE +9,9;
    - ViWikiFC +4,9 (p=0,10);
    - SIB-200 +1,0 (p=0,70).
- EN **94,5**: hơn dự án 6,5 [+4,4; +8,6]. Riêng XCOPA EN 100,0 và Belebele EN 97,3.
- VI không dấu 64,5: hơn dự án 2,2 [−1,2; +5,5] (p=0,16), không phân biệt được.
- Nhất quán khi đảo 97,9 (VI). Độ trễ 87,5 ms, gấp khoảng 2,7 lần dự án.
- Card không nêu dữ liệu huấn luyện (mục 6).
- Hai cách ánh xạ cho **cùng một prompt**: gói ẩn khóa khi có mô tả, và đã kiểm xác suất trùng nhau. Vì vậy chỉ có một bộ số.

## 3. Nhóm nhỏ hơn: chưa qua kiểm câu hiển nhiên, không báo điểm

Theo brief, model không qua kiểm thì không báo điểm như một kết quả. Bảng dưới chỉ ghi trạng thái kiểm.

| Model | Ví dụ README | Câu hiển nhiên sai (text / keyed) | Khớp predict() công khai |
|---|---|---|---|
| Quyet-1.0-Tiny (183 M) | đạt (chỉ kiểm hình dạng; card không nêu số) | 6 / 2 | có |
| Quyet-1.0-Small (328 M) | đạt (chỉ kiểm hình dạng) | 6 / 2 | có |
| laya-vi (322 M) | đạt (khớp số card trong ±0,03) | 2 / 3 | có |
| laya-multilingual (322 M) | **KHÔNG đạt** | 12 / 12 | có |

Câu hiển nhiên: 40 câu, mỗi câu chấm ở 2 thứ tự.

Chi tiết từng model:
- **Quyet Tiny và Small**:
  - Ánh xạ keyed chỉ sai câu Có/Không số 7, ở cả hai ngôn ngữ và cả hai thứ tự. Câu này cần so sánh số: "Minh cao hơn Hùng?" khi Hùng 1m80, Minh 1m60.
  - Ánh xạ text sai thêm 4 câu chọn đáp án, mỗi câu sai ở một thứ tự. Khi lựa chọn chỉ là một từ trần, xác suất gần như chia đều và lệch theo vị trí.
- **laya-vi**: sai câu Có/Không số 4 (tiếng Việt) và số 6 (tiếng Anh), nhất quán ở cả hai thứ tự.
- **laya-multilingual**:
  - Sai mọi câu Có/Không có đáp án "Có", ở cả hai ngôn ngữ, tức gần như luôn trả lời "Không".
  - Ví dụ README của card ("billing") cũng không khớp: model chọn "sales" (0,58), kể cả khi chạy fp32. Vậy nguyên nhân không phải bf16.
  - Gói `laya` tự ghi revision "đã duyệt" của repo này là `e4e9ddf2…`, khác revision dự án đã ghim ngày 12/10 (`1720e3e3…`). Không tải revision khác.

**Vì sao nhiều khả năng đây là lỗi của model, không phải của adapter:**
- Với mọi câu hiển nhiên, ở cả hai thứ tự và hai ánh xạ, xác suất của adapter **khớp hàm `predict()` công khai của chính gói đến 4 chữ số** (lệch tối đa < 5e-5, 80 lượt gọi mỗi model). Phép đối chiếu này gồm cả cách gán câu Có/Không theo chữ của lựa chọn.
- Adapter vì vậy đi đúng đường của tác giả. Các câu sai là do model trả lời sai.
- Số trên dev của 4 model này **có trong** `eval/` và `metrics_baselines_v1_2.json` (gắn `tin_cay: false`) nhưng không đưa vào bảng. Có công bố hay không là việc của người làm dự án.

### 3b. Bảng phụ: CHƯA QUA KIỂM CÂU HIỂN NHIÊN – chỉ tham khảo

Người làm dự án đồng ý đưa số của bốn model này vào báo cáo nội bộ (16/10), dưới dạng bảng phụ. Các số này không thuộc bảng chính.
- Dùng cách ánh xạ cho VI có dấu cao hơn (quy tắc brief).
- Câu hiển nhiên bị sai ghi ở cột cuối.
- Adapter khớp predict() công khai của gói đến 4 chữ số, nên số phản ánh đúng model.

| Model | Ánh xạ dùng | VI có dấu [CI] | VI không dấu | EN [CI] | Brier VI / EN | Nhất quán khi đảo (VI / VI không dấu / EN) | Độ trễ | Câu hiển nhiên sai |
|---|---|---|---|---|---|---|---|---|
| Quyet-1.0-Small (328 M) | keyed (text: 55,4) | 66,6 [63,9; 69,3] | 44,5 | 76,7 [73,7; 79,7] | 0,459 / 0,323 | 81,6 / 69,2 / 90,5 | 15,7 ms | keyed: Có/Không số 7 (VI, EN; hai thứ tự). text: thêm câu chọn đáp án 1 (VI, EN), 2 (EN), 5 (EN) |
| Quyet-1.0-Tiny (183 M) | keyed (text: 55,7) | 65,4 [62,8; 68,1] | 42,6 | 73,3 [70,2; 76,3] | 0,475 / 0,365 | 85,2 / 71,1 / 87,8 | 13,5 ms | keyed: Có/Không số 7 (VI, EN; hai thứ tự). text: thêm câu chọn đáp án 2 (VI), 5 (EN), 6 (VI, EN) |
| laya-vi (322 M) | text (keyed: 54,1) | 55,5 [52,8; 58,2] | 40,4 | 62,5 [59,3; 65,7] | 0,584 / 0,488 | 82,4 / 65,7 / 81,4 | 16,2 ms | Có/Không số 4 (VI) và số 6 (EN), hai thứ tự. keyed: thêm câu chọn đáp án 5 (VI, một thứ tự) |
| laya-multilingual (322 M) | keyed (text: 51,6) | 52,3 [49,8; 54,9] | 33,8 | 63,6 [60,4; 66,9] | 0,644 / 0,487 | 69,8 / 40,2 / 74,6 | 16,5 ms | Mọi câu Có/Không có đáp án "Có" (0, 2, 4, 6, 8; VI và EN) và câu 7 (EN); câu chọn đáp án 5 (VI). **Ví dụ README trượt cả ở fp32** ("sales" thay vì "billing"). **Gói laya tự ghi revision "đã duyệt" `e4e9ddf2…`, khác pin `1720e3e3…` của dự án.** |

So với dự án (VI có dấu 79,7; EN 88,0), cả bốn model nhỏ hơn đều thấp hơn rõ ở mọi biến thể.

## 4. Dự án đứng ở đâu (nêu số, không kết luận hướng đi)

- **Cùng cỡ** (VI có dấu):
  - Dưới decider-2b 6,6 điểm, kể cả khi bỏ MASSIVE.
  - Trên Qwen3.5-2B gốc và Qwen3-1.7B gốc 9,2 điểm.
  - Ở tiếng Anh, khoảng cách với decider-2b là 2,4, và còn 1,0 (không có ý nghĩa) khi bỏ MASSIVE.
  - Ở VI không dấu, dự án đứng đầu nhóm cùng cỡ.
- **Lớn hơn:** dưới Quyet-1.0-Medium 9,7 điểm (VI có dấu) và 6,5 (EN). Ở VI không dấu, chênh +2,2 nhưng CI chứa 0.
- **Nhỏ hơn:** không có model nào qua đủ kiểm (mục 3). Theo bảng phụ (chỉ tham khảo), cả bốn model đều thấp hơn dự án: VI có dấu 52–67, EN 63–77.

## 5. Jeff-Qwen3.5-2B: bị chặn

- Đã tải trọng số ở revision ghim `568a137a…` (4,45 GB, safetensors, không có mã).
- Nhưng phần chạy model (template prompt và head `readout.safetensors`) nằm ở repo GitHub `firelex/jeff`, không có trong repo HF. README bảo `git clone` rồi `uv sync`.
- Repo GitHub này ngoài các nguồn brief nêu, nên **không tải, không chạy**.
- Tự viết lại phần đọc đầu ra thì dễ sai, ngược nguyên tắc "thà đánh giá đối thủ quá cao".
- Theo brief: dừng ở model này, không thay bằng model khác. Muốn chấm Jeff thì cần người làm dự án cho phép tải và rà mã `firelex/jeff` ở một commit ghim.

## 6. Khả năng model đã thấy dữ liệu (theo model card / file cấu hình; không suy đoán thành kết luận)

| Model | Card / cấu hình nói gì về dữ liệu train | MASSIVE | Belebele | SIB-200 | COPA/XCOPA | ViFactCheck / ViWikiFC |
|---|---|---|---|---|---|---|
| decider-2b | Khoảng 95 bộ dữ liệu công khai; danh sách đầy đủ nằm trên GitHub, không có trong repo HF (xem REPORT_baselines 12/10) | **Có thể đã thấy; card mâu thuẫn** (`massive-en-US` ghi "trained", nhưng cũng được liệt kê là held-out) | không nêu | không nêu | không nêu | không rõ |
| Jeff-Qwen3.5-2B | v1.1+ thêm train split của MASSIVE và CLINC150; README nói kết quả trên hai bộ này "không còn zero-shot" | **Đã thấy** (locale nào: không rõ) | không rõ | không rõ | không rõ | không rõ |
| Qwen3.5-2B gốc | không mô tả | không rõ | không rõ | không rõ | không rõ | không rõ |
| Quyet-1.0 Tiny / Small / Medium | card không nêu dữ liệu (`datasets: null`); "English; also tuned for Vietnamese"; Tiny distil từ Medium | không rõ | không rõ | không rõ | không rõ | không rõ |
| laya-multilingual | AG News, BoolQ trong mix train; MASSIVE và XNLI chỉ thấy trong bảng đánh giá, không ghi là train hay held-out | không rõ | không rõ | không rõ | không rõ | không rõ |
| laya-vi | Dữ liệu tiếng Việt tổng hợp hoàn toàn (10 miền nghiệp vụ) + 15% replay tiếng Anh (nguồn không rõ); fine-tune từ laya-multilingual | không rõ (MASSIVE vi-VN chỉ được nêu là tập kiểm) | không rõ | không rõ | không rõ | không rõ |
| Dự án (full_vocab) | Pretrain Qwen3: không rõ. Fine-tune trên train_v0, đã loại MASSIVE, SLURP, IntentGrasp, COPA, XCOPA, Belebele, SIB-200; rò rỉ train_v0 + val_v0 với dev 1.2 = 0 | fine-tune: không | không | không | không | fine-tune: không trùng (kiểm rò rỉ = 0) |

## 7. Kiểm adapter

- **Qwen3.5-2B gốc:**
  - Chấm bằng đúng `scripts/eval_base.py` trong `.venv` chính, không sửa code chấm.
  - Chấm trên bản sao dev giống hệt, chỉ để trống `n_tokens` (đếm bằng tokenizer Qwen3). A..J là một token: đây là phép assert có sẵn trong eval_base.
  - Câu hiển nhiên: 20/20 đúng ở cả hai thứ tự.
- **decider-2b:**
  - Dùng adapter `scripts/bench/eval_decider_v1.py` đã kiểm ở bước baseline và ở dev 1.1 (assert sha từng file trong gói, nhiệt độ phát hành, chạy eager).
  - Thứ tự đảo chấm trên câu không phải Có/Không (3445 dòng). Câu Có/Không dùng đường yes/no gốc, đầu vào không phụ thuộc thứ tự.
- **Quyet và Laya (adapter mới, `scripts/baselines_v1_2/`):**
  - Hai cách ánh xạ:
    - "text": mỗi lựa chọn là một nhãn mang nguyên văn nội dung;
    - "keyed": khóa A, B, … và nội dung nằm ở mô tả.
  - Câu Có/Không dùng kiểu "noul" gốc của model. Xác suất gán theo chữ của lựa chọn (Có/Co/Yes, Không/Khong/No).
  - Nhiệt độ: đúng bản phát hành (assert khớp file cấu hình trong snapshot).
  - Xác suất chưa làm tròn: Quyet dùng `raw_probabilities(strict=True)`; Laya dùng đường nội bộ giống `calibrate.records_from_labeled`.
  - Đối chiếu `predict()` công khai: khớp đến 4 chữ số ở cả 5 model (mục 3).
  - Mọi câu dev: xác suất đủ k phần tử, tổng bằng 1 (assert trong lúc chạy).
- **Tái lập:** sau khi chấm, chạy lại một mình 4 lượt: Qwen3.5-2B (thường), full_vocab (đảo), decider-2b text và keyed (đảo). Dự đoán và xác suất trùng 100% với lượt đầu. Độ trễ trong bảng lấy từ các lượt chạy lại này, khi GPU không làm việc khác.

## 8. Cắt cụt và từ chối

- **decider-2b:** 5 câu vượt 1.536 token (mức cắt mặc định). Đã nâng giới hạn lên 32.768, nên 0 câu bị cắt.
- **Qwen3.5-2B và mốc dự án:** eval_base không cắt.
- **Quyet:** chế độ `strict` (báo lỗi thay vì cắt) cho 0 câu bị cắt, 0 câu bị từ chối.
- **Laya:**
  - 101 câu dev vượt giới hạn mặc định (1.024 token, phần đề + lựa chọn 256).
  - Adapter nâng lên 8.192 / 512 (card laya-multilingual nói đọc được tới 8.192). Ở giới hạn đã nâng, 0 câu bị cắt; không lựa chọn nào vượt 48 token, mức cắt cứng của gói.
  - Laya-vi được fine-tune ở 1.024, nên phần trên 1.024 nằm ngoài phân phối huấn luyện. Ghi chú này chỉ để biết vì laya-vi không có trong bảng.
- Không câu nào bị loại khỏi điểm ở các model trong bảng.

## 9. Bảo mật code bên thứ ba

Chi tiết: `results/baselines_v1_2/notes_code_review.md`.
- `quyet` 1.0.2 được đọc toàn bộ.
- `laya` 0.4.6: đọc toàn bộ đường nạp và chấm, quét regex cả 44 file.
- Không có telemetry, không gọi mạng ngoài lệnh tải model từ HF Hub, không subprocess, exec, pickle, `trust_remote_code`.
- Laya có các tính năng gửi dữ liệu ra ngoài (`integrations`, `mcp`, `serve`), nhưng chỉ chạy khi người dùng chủ động bật; adapter không dùng.
- Laya có thể ghi lại `tokenizer_config.json` trong thư mục nạp, nên adapter nạp từ một bản sao, không đụng HF cache.
- Mọi lượt chấm chạy với `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1` sau khi đã tải model; runner assert hai biến này.
- Gói cài vào `.venv-baselines` bằng `--no-deps`, từ wheel đã kiểm sha256 với PyPI. torch / transformers dùng lại bản ghim của `.venv` chính qua file `.pth` chỉ đọc; `.venv` chính không đổi.

## 10. Ứng viên soát nhãn (chỉ lập danh sách, không sửa bench, không ghi đề xuất)

File: `results/baselines_v1_2/ung_vien_soat_nhan.jsonl`.

**Tiêu chí:** ít nhất 2/3 số model đáng tin cùng chọn một đáp án khác đáp án chuẩn. Có 9 model bỏ phiếu:
- 7 preds của scores_v1_2.json: decider-2b text và keyed, decider-ft-ep2 text và keyed, full_vocab, Qwen3-1.7B gốc, strip;
- Qwen3.5-2B gốc;
- Quyet-1.0-Medium.

**Kết quả: 365 dòng, thuộc 279 cặp.**
- Theo biến thể:
  - VI có dấu 114 dòng (20 dòng thuộc cặp revised);
  - EN 58 dòng;
  - VI không dấu 193 dòng.
- Số cặp có dòng VI có dấu hoặc EN là ứng viên: 134.
- Tỉ lệ đồng thuận: 9/9 có 41 dòng; 8/9 có 86; 7/9 có 84; 6/9 có 154.

**Cảnh báo: 9 người bỏ phiếu KHÔNG độc lập.**
- Có 4 model họ decider (decider-2b text và keyed, decider-ft-ep2 text và keyed) và 3 model họ Qwen3-1.7B (gốc, full_vocab, strip).
- Hai họ này chiếm 7/9 phiếu. Nhiều ứng viên vì vậy có thể chỉ là điểm mù chung của hai họ, không phải nhãn sai.
- Phép thử độ nhạy: điều phối tính, tôi đã tính lại và khớp. Nếu đổi đáp án chuẩn của MỌI dòng ứng viên theo đa số, VI có dấu (trung bình đều) thay đổi:
  - Quyet-1.0-Medium chỉ +1,5 (EN +0,3);
  - decider-2b text +6,4 (EN +4,2);
  - dự án +6,4 (EN +3,9).
- Tức là ở phần lớn ứng viên, Quyet-Medium đang chọn đúng đáp án chuẩn. Danh sách này chỉ là đầu vào cho soát nhãn mù, không phải danh sách nhãn sai.

Dòng VI không dấu chiếm nhiều vì các model yếu ở bản không dấu. Ứng viên chỉ ở bản không dấu nhiều khả năng phản ánh độ khó, không phải nhãn sai. Mỗi dòng ghi:
- pair_id, nguồn, biến thể;
- đáp án chuẩn, đáp án đa số;
- tỉ lệ đồng thuận và danh sách model chọn đáp án đa số.

## 11. Những gì tôi không chắc

- **Quyet-1.0-Medium:** card không nêu dữ liệu huấn luyện. Điểm rất cao ở XCOPA EN (100) và Belebele EN (97,3) chưa loại trừ được khả năng đã thấy dữ liệu.
- **decider-2b và MASSIVE:** decider-2b có thể đã thấy MASSIVE, nên các con số "bỏ MASSIVE" thận trọng hơn.
- **Prompt khác nhau giữa các model:** decider-2b và Quyet-Medium dùng prompt tiếng Anh cố định của tác giả với nội dung tiếng Việt, còn dự án dùng khung prompt tiếng Việt. Mỗi bên dùng đúng format của mình.
- **Độ trễ không so trực tiếp được.** Các model Qwen3.5 chạy không có `flash-linear-attention`, Laya chạy autocast bf16 theo mặc định của gói. Độ trễ đo trên toàn bộ lời gọi của từng adapter.
- **Model nhỏ hơn:** cách xử lý theo đúng chữ brief. Bằng chứng ở mục 3 cho thấy lỗi nằm ở model, nên người làm dự án có thể muốn xem số của bốn model này.
- **Adapter của dự án mới có 1 seed.**

## 12. Ghim, môi trường và file

**Revision:**
- Qwen3.5-2B `15852e8c…`, decider-2b `ff99089b…`;
- Quyet-1.0-Tiny `fba8ca92…`, Small `f007dfeb…`, Medium `bd509d64…`;
- laya-multilingual `1720e3e3…`, laya-vi `860b2b5f…`, Jeff `568a137a…`;
- đầy đủ ở `results/baselines/pins_models.json` và `results/baselines_v1_2/downloads.json`.

**Thư viện:**
- torch 2.11.0+cu128, transformers 5.19.0, safetensors 0.8.0, huggingface_hub 2.2.0, numpy 2.5.4 (bản ghim của `.venv` chính);
- quyet 1.0.2, laya 0.4.6 (`.venv-baselines`).

**Phần cứng và cách chạy:** RTX 5060 Ti 16 GB, WSL2, batch 1, bf16. Mỗi lượt chấm có giới hạn thời gian (20–180 phút).

| File | Nội dung |
|---|---|
| `results/baselines_v1_2/eval/{normal,reverse}/<model>/` | preds từng câu và `timing.json` (model ngoài; lượt đảo của mốc dự án) |
| `results/baselines_v1_2/eval/rerun/` | lượt chạy lại một mình để đo độ trễ và kiểm tái lập |
| `results/baselines_v1_2/checks/`, `obvious/` | ví dụ README, câu hiển nhiên, đối chiếu predict() |
| `results/baselines_v1_2/notes_code_review.md`, `notes_model_api.md` | rà mã gói và ghi chú API (subagent Sonnet viết; các điểm then chốt tôi đã kiểm lại bằng grep và phép đối chiếu predict()) |
| `results/baselines_v1_2/tables_baselines_v1_2.md`, `metrics_baselines_v1_2.json`, `ung_vien_soat_nhan.jsonl` | bảng, số liệu thô, ứng viên soát nhãn |
| `scripts/baselines_v1_2/*.py`, `setup/baselines_v1_2_*.sh` | code (adapter mới đặt riêng; không sửa code chấm của dự án) |
