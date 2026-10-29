# Báo cáo tổng hợp: decision model và benchmark tiếng Việt

Cập nhật: 2026-10-13. File này gom toàn bộ các lượt chạy từ đầu dự án tới phần 5a: làm gì, chạy thế nào, ra số gì, kiểm tra gì, và còn chờ duyệt gì. Mỗi phần trỏ tới báo cáo chi tiết riêng của nó.

---

## Mục lục

0. Bối cảnh, môi trường, quy tắc chung
1. Bước 1: dựng tập đo `dev_v0`
2. Bước 2: đo hai model gốc trên `dev_v0`
3. Bước 3: fine-tune thử Qwen3-0.6B và Qwen3-1.7B (2 seed)
4. Bước 3b: sửa công thức loss (full_vocab)
5. Bước 4: chọn model
6. Bước 4b: chấm model của người khác (Qwen3.5-2B, decider-2b)
7. Bước 4c: kiểm cuDNN SDPA trên Blackwell
8. Phần 5a: benchmark lõi mở (test 1.0 đóng băng) và `dev_v1`
9. Sự cố vận hành và cách đã xử lý
10. Việc còn chờ người duyệt
11. Danh mục file

---

## 0. Bối cảnh, môi trường, quy tắc chung

**Mục tiêu dự án.** Có một *decision model* nhỏ dùng tốt cho tiếng Việt, kèm một benchmark tiếng Việt đủ tin cậy để chứng minh điều đó.
- Decision model nhận một văn bản (state) và một câu hỏi có sẵn lựa chọn, rồi trả về phân phối xác suất trên các lựa chọn; không sinh văn bản.
- Có ba loại câu hỏi:
  - `choice`: chọn một trong các lựa chọn;
  - `noul`: đúng/sai;
  - `score`: chọn một mức trên thang có thứ tự.
- **Cách chấm**:
  1. Dựng prompt qua chat template, tắt chế độ thinking.
  2. Đánh chữ A, B, C… cho các lựa chọn.
  3. Chạy một forward pass.
  4. Đọc logit của token kế tiếp tại các chữ cái đó và softmax trên đúng k lựa chọn.

**Lượt dọn dẹp trước đó.** Trước khi bắt đầu, người dùng yêu cầu xóa toàn bộ dự án làm dở lần trước.
- Đã xóa trong thư mục `C:\Users\dungbt06\Fine-tune`: `.venv`, `scripts/`, `test_gpu.py`, `data/`, `results/`.
- Đã xóa trong WSL Ubuntu:
  - cache `~/.cache/uv` (khoảng 7 GB) và `~/.cache/huggingface`;
  - công cụ `uv` và thư mục cấu hình của nó;
  - `.nv`, `.zshrc`;
  - dòng `. "$HOME/.local/bin/env"` trong `.bashrc` và `.profile`. Bước này người dùng tự chạy, vì lệnh xóa của agent bị chế độ tự động chặn.
- Sau đó `uv` được cài lại để dựng môi trường mới.

**Môi trường** (dùng cho mọi lượt chạy):

| Mục | Giá trị |
|---|---|
| Phần cứng | NVIDIA GeForce RTX 5060 Ti 16 GB (Blackwell, sm_120), driver 591.86, RAM 32 GB |
| Hệ điều hành | Windows 11 + WSL2 Ubuntu 26.04; thư mục dự án `/mnt/c/Users/dungbt06/Fine-tune` |
| Python | 3.12.15, venv `.venv` trong thư mục dự án, tạo bằng uv |
| Thư viện | torch 2.11.0+cu128, cuDNN 9.19, transformers 5.19.0, datasets 5.1.0, peft 0.21.2, datasketch 2.0.0; danh sách đầy đủ ở `requirements.lock.txt` |
| Model | `Qwen/Qwen3-0.6B@c1899de2`, `Qwen/Qwen3-1.7B@70d244cc`, `Qwen/Qwen3.5-2B@15852e8c` (ghim ở `config/models.json`); `Mapika/decider-2b@ff99089b` |
| Seed dữ liệu | 20261012 |

**Quy tắc áp dụng xuyên suốt** (theo PLAN.md, mục 10):
- Không đụng phần để dành cho test.
- Không đưa nguồn của dev vào train.
- Không chỉnh model theo điểm test.
- Chỉ có một module dựng prompt (`scripts/common.py: render_prompt`), dùng chung cho train và chấm.
- Ghim revision của mọi dataset và model; seed cố định.
- Số chính thức lấy ở batch 1; mỗi lượt chấm có giới hạn thời gian.
- In dữ liệu tối đa 5 dòng, mỗi trường cắt còn 200 ký tự; log train ghi ra file.

---

## 1. Bước 1: dựng tập đo `dev_v0`

Báo cáo chi tiết: `results/REPORT_dev_v0.md`. Dữ liệu: `data/dev_v0.jsonl` (SHA-256 `f7b12704…866f`), `data/manifest_dev_v0.json`, `data/reserved_test.json`.

**Thành phần:** 1.000 cặp Anh–Việt (2.000 câu), toàn bộ là câu hỏi `choice`, nhãn người, bản dịch do người làm.

| Nguồn (revision) | Cặp | Lấy từ | Lựa chọn | Khóa ghép Anh–Việt |
|---|---|---|---|---|
| facebook/belebele `7899cdfa` | 300 | Chia theo đoạn văn: 244 đoạn cho dev, 244 đoạn để dành cho test; dev dùng 300/445 câu | 4 | (link, question_number) |
| Davlan/sib200 `38977a66` | 300 | 99 validation + 201 lấy từ train, cân nhãn | 7 chủ đề | index_id |
| mteb/amazon_massive_intent `940fd47a` | 300 | validation, cân trên 59 ý định | 10 (1 đúng + 9 nhiễu theo seed) | id |
| cambridgeltl/xcopa `042f7895` (vi) + aps/super_glue copa `3de24cf8` (en) | 100 | validation; ghép theo idx (100/100 khớp loại câu hỏi và nhãn) | 2 | idx |

**Chi tiết dựng:**
- Repo gốc AmazonScience/massive bị datasets từ chối vì dùng loading script, nên dùng bản mirror mteb.
- Câu hỏi, tên 7 chủ đề và 60 mô tả ý định (bản Anh và bản Việt) do agent viết, nằm ở `config/translations.json` (**chưa được người duyệt**).
- Khung prompt nằm ở `config/prompt_frames.json` (**chưa được người duyệt**).
- Thứ tự lựa chọn được xáo theo từng cặp; hai câu cùng `pair_id` có cùng thứ tự và cùng bộ nhãn nhiễu.
- Phần để dành cho test, ghi trong `reserved_test.json`: 244 đoạn Belebele, và split test của SIB-200, MASSIVE, XCOPA.
- Chạy dựng lại lần hai cho ra đúng cùng SHA-256.

---

## 2. Bước 2: đo hai model gốc trên `dev_v0`

Báo cáo chi tiết: `results/REPORT_dev_v0.md`; dự đoán thô ở `results/run1/`. `run2` là lượt chạy lại, trùng từng byte với `run1`.

**Accuracy (%), trung bình đều theo nguồn:**

| | Tiếng Anh | Tiếng Việt | Tụt Anh→Việt |
|---|---|---|---|
| Qwen3-0.6B | 69,3 [66,2; 72,5] | 60,5 [57,0; 64,1] | 8,8 |
| Qwen3-1.7B | 83,4 [81,0; 85,8] | 72,9 [69,9; 75,8] | 10,5 |

- **So hai model trên tiếng Việt:** 1.7B hơn 0.6B 12,4 điểm [8,6; 16,2] (McNemar p=1,5e-11). Khoảng cách lớn nhất ở Belebele (+21) và XCOPA (+18).
- **Tốc độ:**
  - Độ trễ batch 1: 0.6B 32 ms, 1.7B 41 ms.
  - Ở batch 16: 0.6B xử lý 118–151 câu/giây, 1.7B 56 câu/giây.
  - Gom batch làm đổi 1,3–1,9% quyết định so với batch 1.
- **Bất thường:**
  - 0.6B thiên vị chữ A: ở XCOPA tiếng Việt chọn A 80/100 câu.
  - 1.7B dồn câu sai về A ở MASSIVE tiếng Việt.
  - 1.7B có 18/2.000 câu mà tổng xác suất trên các chữ cái (`letter_mass`) dưới 0,5.
  - Lần chấm đầu tiên treo 11 phút không rõ lý do; từ đó mọi lần chấm đều có timeout.

---

## 3. Bước 3: fine-tune thử Qwen3-0.6B và Qwen3-1.7B

Báo cáo chi tiết: `results/ft/REPORT_ft_v0.md`; bảng ở `results/ft/report_ft.md`.

**Tập train `train_v0`** (`data/manifest_train_v0.json`): 20.000 câu train và 1.000 câu val nội bộ.
- **Nguồn:**
  - 16.000 câu từ `tasksource/tasksource-jev-typed-decisions@bf1e4a87` (nhãn người).
  - 4.000 câu tiếng Việt từ `LakoreAI/PhoLaya-base@034d4273`, gồm ViNLI, VSFC, ViSFD (chỉ cho nghiên cứu).
- **Theo loại câu hỏi:** choice 15.162, noul 3.375, score 1.463. Theo ngôn ngữ của khung prompt: en 15.998, vi 4.002.
- **Nguồn bị loại:** 26 nguồn tasksource có liên quan tới dev, xác định qua `sources.yaml`: IntentGrasp, SLURP, MASSIVE, SuperGLUE COPA, Balanced-COPA và 21 nguồn XCOPA. Phần MASSIVE trong PhoLaya cũng bị loại.
- **Chống rò rỉ:**
  - Lọc bằng ba phép: trùng nguyên văn, gần trùng (MinHash ≥ 0,8) và chứa đoạn, đối chiếu với 44.050 văn bản của dev và test.
  - Đã loại 105 ứng viên; số trùng còn lại trên tập cuối là **0**.
- **Giới hạn 5% mỗi nguồn:** tính theo tác vụ, theo quyết định của người dùng.
- **Chia val nội bộ:** theo văn bản gốc.

**Công thức train** (giống nhau cho cả hai model):
- LoRA r=16, α=32, dropout 0,05 trên q, k, v, o, gate, up, down.
- 2 epoch, batch hiệu dụng 32 (micro 8 × 4), lr 2e-4, warmup 3% rồi cosine, bf16, gradient checkpointing.
- Xáo lựa chọn mỗi lần nạp câu; lấy checkpoint cuối epoch 2.
- Loss bản đầu: CE trên softmax của riêng k chữ cái (`letters_k`).

**Seed:** seed 1 chạy theo quy tắc của plan. Seed 2 do phiên điều phối giao chạy thêm để đo nhiễu (một lần train dài hơn 1 giờ nên quy tắc không bắt buộc).

**Kết quả** (accuracy %, trung bình đều):

| | vi | en | Thời gian train | VRAM đỉnh | Val loss (đầu → epoch 1 → epoch 2) |
|---|---|---|---|---|---|
| 0.6B gốc | 60,5 | 69,3 | | | |
| 0.6B ft s1 / s2 | 76,7 / 73,5 | 80,2 / 80,3 | 61,7 / 61,8 phút | 5,1 GB | 2,713 → 0,815 → 0,832 / 2,713 → 0,806 → 0,840 |
| 1.7B gốc | 72,9 | 83,4 | | | |
| 1.7B ft s1 / s2 | 79,8 / 80,8 | 87,2 / 86,0 | 106,1 / 106,3 phút | 10,2 GB | 5,495 → 0,771 → 0,761 / 5,495 → 0,765 → 0,773 |

- **1.7B − 0.6B sau fine-tune, tiếng Việt:** +5,2 điểm, trung bình 4 cặp seed (dao động +3,2 đến +7,3). Riêng Belebele +11,3.
- **Nhiễu giữa hai seed:** 0.6B lệch 3,2 điểm, gần như toàn bộ do XCOPA; 1.7B lệch 0,9 điểm.
- **Mức tụt Anh→Việt hẹp lại:**
  - 0.6B: 8,8 → 3,5 (s1) và 6,8 (s2);
  - 1.7B: 10,5 → 7,3 (s1) và 5,3 (s2).
- **Thiên vị vị trí gần như hết:** tỉ lệ nhất quán khi đảo thứ tự lựa chọn (tiếng Việt) tăng từ 62,9 lên khoảng 86 ở 0.6B, và từ 73,9 lên 88–90 ở 1.7B.
- **Bất thường:** `letter_mass` của 1.7B sụp ở cả hai seed (1.919 và 1.865 trên 2.000 câu dưới 0,5). Nguyên nhân: loss chỉ phụ thuộc vào hiệu giữa các logit chữ cái, nên toàn bộ logit co lại. Đã xử lý ở bước 3b.
- **Temperature fit trên val:** model gốc có T = 13,6 và 18,6, và Brier trên dev xấu đi vì val và dev khác phân phối. Sau fine-tune T = 1,41–1,55.

**Độ trễ batch 1 theo độ dài prompt** (model đã fine-tune, LoRA merge vào bf16):

| Độ dài | 0.6B | 1.7B |
|---|---|---|
| 128 token | 28 ms | 29 ms |
| 512 token | 27 ms | 50 ms |
| 2.048 token | 85 ms | 181 ms |

Ở batch 16 trên dev_v0, 0.6B xử lý được 138 câu/giây, 1.7B 65 câu/giây.

---

## 4. Bước 3b: sửa công thức loss

Báo cáo chi tiết: `results/ft_loss/REPORT_loss_fix.md`; bảng ở `results/ft_loss/tables_loss_fix.md`.

- **Loss mới (`--loss full_vocab`):** cross-entropy trên toàn từ vựng, target mềm đặt trên k token chữ cái. Công thức này bằng loss cũ trừ log `letter_mass`.
- **Kiểm đẳng thức bằng số** (32 câu): sai lệch lớn nhất 1,9e-6 trên model gốc và 4,8e-7 trên adapter cũ.
- **Cổng chặn:** chấm lại adapter cũ cho kết quả trùng từng byte với bản đã lưu.
- **Lần train kiểm chứng:** Qwen3-1.7B seed 1, mọi thứ khác giữ nguyên như bước 3; mất 110,3 phút. `letter_mass` trên 200 câu val đạt 0,996 ở bước 125 và khoảng 0,999 đến hết, nên không chạm cổng dừng ở 20% số bước.

| Tiêu chí | Adapter mới | Adapter cũ | Kết quả |
|---|---|---|---|
| `letter_mass` < 0,5 (trên 2.000 câu); trung bình | **0**; 0,999 | 1.919; 0,165 | ĐẠT |
| Accuracy tiếng Việt | 79,8 | 79,8 (chênh −0,1; p=1) | ĐẠT |
| Sinh greedy 1 token ra chữ cái hợp lệ | 100/100 | 100/100 | ĐẠT; tiêu chí này không phân biệt được hai adapter |

- Tiếng Anh: 86,4 so với 87,2 (p=0,58).
- Brier tiếng Việt: 0,304 → 0,290 sau khi áp T = 1,56.
- Phân phối toàn từ vựng trên 5 câu thử: entropy 0,04–0,69 (adapter cũ 6–11), logit lớn nhất 26–28 (adapter cũ 7–11).
- **Quyết định** (đã ghi trong PLAN_decisions.md): lần train chính thức dùng loss mới.

---

## 5. Bước 4: chọn model

Người làm dự án chốt chọn **Qwen/Qwen3-1.7B**. Căn cứ:
- 1.7B hơn 0.6B 5,2 điểm trên tiếng Việt và 11,3 điểm trên Belebele.
- Với prompt ngắn ở batch 1, hai model nhanh như nhau.
- Kiểu dùng mặc định là từng yêu cầu một.

---

## 6. Bước 4b: chấm model của người khác trên `dev_v0`

Báo cáo chi tiết: `results/baselines/REPORT_baselines.md`. Phạm vi đã thu hẹp còn hai model; Quyet, Laya và Jeff để sau.

| Model | vi | en | Tụt Anh→Việt | Nhất quán vi | Độ trễ batch 1 |
|---|---|---|---|---|---|
| decider-2b (cách ánh xạ "text") | **86,0** | **89,2** | 3,2 | 93,3 | 48,5 ms¹ |
| Dự án: Qwen3-1.7B full_vocab | 79,8 | 86,4 | 6,7 | 88,0 | 31,1 ms |
| Qwen3-1.7B gốc | 72,9 | 83,4 | 10,5 | 73,9 | 31,2 ms |
| Qwen3.5-2B gốc | 70,1 | 82,6 | 12,5 | 68,0 | 54,4 ms¹ |

¹ Không cài flash-linear-attention, nên số này không so trực tiếp được với Qwen3.

- **Dự án so với decider-2b:**
  - Tiếng Việt: kém 6,2 [3,5; 9,0]. Bỏ MASSIVE (nguồn decider-2b có thể đã thấy khi train) thì kém 4,4 [1,1; 7,8].
  - Tiếng Anh: kém 2,8; bỏ MASSIVE thì kém 1,2 (không có ý nghĩa thống kê).
- **Qwen3.5-2B gốc so với Qwen3-1.7B gốc:** tiếng Việt −2,8, tiếng Anh −0,8; cả hai đều không có ý nghĩa thống kê.
- **Kiểm adapter:**
  - Ví dụ README của decider-2b khớp (0,926 và 0,703 so với 0,93 và 0,70).
  - 10 câu hiển nhiên đúng ở cả hai thứ tự lựa chọn.
  - Xác suất cộng bằng 1, đủ số phần tử; không câu nào bị cắt cụt.
  - Không model nào cần `trust_remote_code`.
  - Mã `decider/` đã rà: không gọi mạng ngoài Hugging Face, không ghi file.

---

## 7. Bước 4c: kiểm cuDNN SDPA trên Blackwell

Báo cáo chi tiết: `results/cudnn_check/REPORT_cudnn_check.md`.

**Không có bằng chứng cuDNN SDPA làm sai kết quả của dự án**, vì với cấu hình của dự án backend này không được gọi.
- Đường chấm (batch 1, không mask) dùng flash attention.
- Đường train (có padding và mask) dùng memory-efficient attention.
- Chấm lại adapter full_vocab với cuDNN tắt cho kết quả trùng từng byte với bản cũ.
- Đường train, mặc định so với tắt cuDNN: chênh lệch bằng 0 ở logit, loss và chuẩn gradient.
- So với chế độ chỉ dùng math (mốc nhiễu bf16): logit lệch tối đa 0,25, loss lệch 0,016, chuẩn gradient lệch 4,3%.

---

## 8. Phần 5a: benchmark lõi mở và `dev_v1`

Báo cáo chi tiết: `results/dev_v1/REPORT_dev_v1.md`; bảng ở `results/dev_v1/tables_dev_v1.md`.

**Tập test 1.0:** đã đóng băng (SHA-256 `d92d31f6…0d47`, file chỉ đọc). **Chưa model nào chạy trên tập này.**

| Nguồn | Câu test (en / vi / vi không dấu) | Loại câu hỏi |
|---|---|---|
| Belebele | 272 / 272 / 272 | choice |
| SIB-200 | 120 / 120 / 120 | choice |
| MASSIVE | 300 / 300 / 300 | choice |
| XCOPA (+ COPA tiếng Anh) | 300 / 300 / 300 | choice |
| ViFactCheck (tranthaihoa, MIT) | — / 300 / 300 | choice 3 hoặc noul |
| ViWikiFC (High-Will, CC BY 4.0) | — / 300 / 300 | choice 3 hoặc noul |
| ViHSD (uitnlp, MIT, bộ có cổng truy cập) | — / 300 / 300 | score 3 mức hoặc noul |

**Thiếu so với quy mô trong brief:** SIB-200 chỉ còn 120/204 câu và Belebele 272/300. Lý do là văn bản FLORES của hai bộ trùng với dev_v0, nên phải loại để đạt 0 trùng giữa test và dev.

**`dev_v1`** (4.526 dòng):
- Khối song song là nguyên dev_v0, cộng bản không dấu cho các câu tiếng Việt.
- Khối bản địa lấy từ split dev: ViFactCheck 163 câu (dev và test dùng chung bài báo), ViWikiFC 300, ViHSD 300. Mỗi câu có thêm bản không dấu.

**Chống rò rỉ:**
- Test và dev_v1: 0 trùng ở cả hai chiều.
- train_v0 trùng 3 câu với test và 1 câu với dev_v1: câu ViNLI nằm trong bài báo ViFactCheck, và bình luận VSFC/ViSFD trùng hoặc chứa bình luận ViHSD. Theo brief, không sửa, chỉ đánh dấu.

**Cổng kiểm tra:**
- Đường chấm của dự án tái lập dev_v0: 2.000/2.000 dòng trùng.
- decider-2b tái lập bước baseline: xác suất trùng 2.000/2.000.
- Adapter decider cho noul/score: lệch so với `system_one` của gói tối đa 4,9e-5.
- Câu hiển nhiên noul/score qua cả hai phía; không có gán nhầm có hệ thống.
- Không câu nào bị cắt cụt.

**Kết quả trên dev_v1** (accuracy %):

| Lát | Qwen3-1.7B gốc | Dự án (full_vocab) | decider-2b | decider − dự án (p) |
|---|---|---|---|---|
| Song song, tiếng Anh | 82,9 | 85,5 | **88,6** | +3,1 (0,005) |
| Song song, tiếng Việt có dấu | 71,7 | 79,9 | **86,0** | +6,1 (9e-7) |
| Song song, tiếng Việt **không dấu** | 47,4 | **59,4** | 48,4 | −11,0 (5e-9) |
| Bản địa, có dấu | 53,7 | 65,3 | 63,4 | −1,8 (0,39) |
| Bản địa, **không dấu** | 46,9 | **52,3** | 41,2 | −11,1 (3e-8) |
| ViHSD, có dấu | 46,7 | **50,7** | 40,7 | −10,0 (0,01) |
| Noul bản địa, có dấu | 70,0 | **73,8** | 65,6 | −8,3 (0,002) |

**Ba câu hỏi của brief:**
1. **decider-2b tụt ở đâu:** so với mức 86,0% của chính nó, nó tụt mạnh nhất ở văn bản không dấu (48,4%; SIB-200 rơi từ 82,3 xuống 36,0). Nó cũng yếu ở ViHSD (40,7%, thấp hơn cả mốc nhãn phổ biến) và ở câu noul của khối bản địa.
2. **Dự án ngang hoặc hơn ở đâu:** hơn ở mọi lát không dấu, ViHSD và noul bản địa; ngang ở khối bản địa có dấu nói chung, ViWikiFC, SIB-200 và score.
3. **Còn quá ít mẫu:** state dài hơn 1.024 token (41 câu), ViFactCheck dev (163 câu), XCOPA (100 câu), score (144 câu).

---

## 9. Sự cố vận hành và cách đã xử lý

| Sự cố | Ảnh hưởng | Xử lý |
|---|---|---|
| Lần chấm đầu treo 11 phút | Không ảnh hưởng số liệu | Mọi lượt chấm và train có timeout từ đó; không tái diễn |
| Train 0.6B lần đầu chậm (4,5 giây/bước) | Mất vài phút | Sắp 32 câu của mỗi batch theo độ dài trước khi chia micro-batch; công thức train không đổi về mặt toán học |
| Gõ sai tham số timeout ("18000bash") | 1.7B khởi động chậm khoảng 30 phút | Sửa script; không chỉnh file script khi nó đang chạy |
| Lệnh `pkill` tự giết chính shell gọi nó | Một lần debug bị dừng | Dùng mẫu `[p]ython…` |
| Trích dẫn trong PowerShell làm hỏng lệnh bash (`sed`, `python -c`) | Lệnh thất bại | Viết lệnh vào file script |
| `REPORT_loss_fix.md` ghi đè `report_loss_fix.md` (Windows không phân biệt hoa/thường) | Mất bảng, đã dựng lại trùng từng byte | Đặt tên file bảng khác hẳn tên báo cáo |
| Adapter decider trên dev_v1 đưa nhầm dòng dev_v0 (`kind` cũ) sang đường score | 180/2.000 quyết định sai ở lượt đầu | Cổng tái lập baseline phát hiện; đã sửa và chạy lại trước khi lập bảng |
| Assert `noul`/`score` sai định dạng (vectơ rộng 255) | Lượt decider dừng | Đọc đúng như `systemone.assemble` của gói |
| ViHSD có cổng truy cập | Tạm dừng nguồn này | Người dùng tự xin quyền; đưa vào trước khi đóng băng test |

---

## 10. Việc còn chờ người duyệt

1. `config/translations.json` và `config/prompt_frames.json`: đang dùng như hiện tại, chưa duyệt.
2. `config/train_format.json` (cách viết noul thành Có/Không, nhãn trường của PhoLaya): chưa duyệt.
3. `config/bench_v1_questions.json` (câu hỏi cho ba nguồn bản địa): chưa duyệt.
4. `results/dev_v1/review_native_dev20.md`: 20 dòng dev mỗi nguồn bản địa, gồm bản không dấu. ViHSD có thể chứa ngôn từ xúc phạm.
5. Phần thiếu của test 1.0 (SIB-200 120/204, Belebele 272/300) và của dev ViFactCheck (163/300): chấp nhận như hiện tại, hay sửa ở bản 1.1.
6. Ánh xạ nhãn số của ViHSD (0 sạch, 1 xúc phạm, 2 thù ghét), suy ra từ bài báo gốc.
7. Bốn câu train_v0 trùng với ViFactCheck và ViHSD: đã đánh dấu, không sửa.
8. Bước 6: có train chính thức hay không, và train để sửa cái gì. Người làm dự án quyết sau khi xem kết quả dev_v1.

---

## 11. Danh mục file

| Nhóm | File |
|---|---|
| Kế hoạch | `C:\Users\dungbt06\Downloads\PLAN.md`, `PLAN_decisions.md` |
| Bước 1–2 | `data/dev_v0.jsonl`, `data/manifest_dev_v0.json`, `data/reserved_test.json`, `results/REPORT_dev_v0.md`, `results/run1/`, `results/run2/` |
| Bước 3 | `data/train_v0.jsonl`, `data/val_v0.jsonl`, `data/manifest_train_v0.json`, `results/ft/REPORT_ft_v0.md`, `results/ft/report_ft.md`, `results/ft/latency_ft.json`, `results/ft/train/` |
| Bước 3b | `results/ft_loss/REPORT_loss_fix.md`, `results/ft_loss/tables_loss_fix.md`, `results/ft_loss/train/` |
| Bước 4b | `results/baselines/REPORT_baselines.md`, `results/baselines/tables_baselines.md` |
| Bước 4c | `results/cudnn_check/REPORT_cudnn_check.md` |
| Phần 5a | `data/bench_v1/` (test 1.0 chỉ đọc, dev_v1, manifest, leakage), `results/dev_v1/REPORT_dev_v1.md`, `results/dev_v1/tables_dev_v1.md` |
| Code | `scripts/` (common.py, eval_base.py, train_lora.py, build_*.py, report_*.py, decontam.py, stats.py, baselines/, bench/, cudnn_check/), `setup/` (run.sh, các pipeline) |
| Cấu hình | `config/models.json`, `translations.json`, `prompt_frames.json`, `train_format.json`, `bench_v1_questions.json` |
