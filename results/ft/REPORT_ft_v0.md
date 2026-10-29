# Bước 3: fine-tune thử Qwen3-0.6B và Qwen3-1.7B, đo lại trên dev_v0

Ngày 2026-10-12, cập nhật sau khi có seed 2.

- Bảng đầy đủ (mọi nguồn, mọi seed, CI, phân bố chữ cái) nằm ở `results/ft/report_ft.md`; số liệu thô ở `results/ft/metrics_ft.json`.
- Bản chỉ có seed 1 được giữ ở `report_ft_s1only.md` và `metrics_ft_s1only.json`. Mọi số của seed 1 trong bản mới trùng với bản đó.

**Hai file cấu hình chưa được người duyệt.** Người làm dự án đã chấp nhận dùng `config/prompt_frames.json` và `config/translations.json` như hiện tại (`PLAN_decisions.md`). `config/train_format.json` (cách viết noul thành Có/Không, nhãn trường của PhoLaya) cũng chưa ai duyệt.

**Seed 2 được chạy thêm, ngoài quy tắc 1 giờ.** Một lần train mất 62 phút (0.6B) và 106 phút (1.7B), nên theo quy tắc chỉ bắt buộc 1 seed. Seed 2 do phiên điều phối giao chạy thêm để đo nhiễu giữa các seed trước khi chọn model (xem `PLAN_decisions.md`). Công thức giữ nguyên, chỉ đổi seed.

## Kết luận ngắn

| Tiếng Việt, trung bình đều theo nguồn | Seed 1 | Seed 2 | Trung bình |
|---|---|---|---|
| 0.6B sau fine-tune | 76,7 | 73,5 | 75,1 |
| 1.7B sau fine-tune | 79,8 | 80,8 | 80,3 |
| 1.7B − 0.6B (cùng seed) | +3,2 [0,5; 5,7] | +7,3 [4,1; 10,5] | |
| 1.7B − 0.6B (trung bình 4 cặp seed, kể cả cặp chéo) | | | **+5,2** (dao động +3,2 đến +7,3) |

- **Sau fine-tune, 0.6B kém 1.7B khoảng 5 điểm trên tiếng Việt** (trung bình 4 cặp seed). Trước fine-tune khoảng cách là 12,4 điểm. Mọi cặp seed đều có p < 1e-4.
- **Trên tiếng Anh**, khoảng cách là +6,3 (dao động +5,7 đến +7,0).
- **Nhiễu giữa seed của 0.6B lớn ở tiếng Việt:** trung bình đều chênh 3,2 điểm, gần như toàn bộ do XCOPA (87 so với 72; p=0,0007 trên cùng 100 câu). Ba nguồn còn lại chênh không quá 1,7 điểm. 1.7B ổn định hơn: chênh 0,9 điểm.
- **Belebele (đọc hiểu đoạn dài) là chỗ 0.6B kém rõ và ổn định nhất:** +11,3 điểm (dao động +9,3 đến +13,3 qua 4 cặp seed).
- **Độ trễ batch 1 phụ thuộc mạnh vào độ dài prompt** (bảng ở mục Tốc độ):
  - Ở khoảng 128 token, hai model gần như bằng nhau (28 so với 29 ms).
  - Ở 512 token, 0.6B nhanh hơn 1,8 lần; ở 2.048 token, nhanh hơn 2,1 lần.
  - Ở batch 16 trên dev_v0, 0.6B nhanh hơn 2,1 lần (138 so với 65 câu/giây).
- **Fine-tune gần như xóa thiên vị vị trí ở cả hai model và cả hai seed.**
- **`letter_mass` của 1.7B sụp ở cả hai seed.** Lỗi này không làm sai accuracy và Brier, nhưng là lỗi của công thức loss mà lần train chính thức phải sửa (mục bất thường 1).

## Khuyến nghị theo kiểu dùng (chưa chốt; quyết định là của người làm dự án)

Mốc ở mục 7 của PLAN.md: 0.6B kém dưới 5 điểm tiếng Việt thì chọn 0.6B; trên 10 điểm thì chọn 1.7B; ở giữa thì quyết theo kiểu dùng.

Mức kém đo được là **+5,2 điểm** (trung bình 4 cặp seed; dao động +3,2 đến +7,3), tức nằm sát cận dưới của vùng "ở giữa". **Kiểu dùng mặc định hiện là từng yêu cầu một** (`PLAN_decisions.md`), vì chưa có ứng dụng cụ thể.

1. **Từng yêu cầu một (batch 1), kiểu mặc định: nghiêng về 1.7B.**
   - Accuracy tiếng Việt: 1.7B hơn 5,2 điểm (+3,2 đến +7,3).
   - Độ trễ p50:

     | Độ dài prompt | 0.6B | 1.7B |
     |---|---|---|
     | 128 token | 28 ms | 29 ms |
     | 512 token | 27 ms | 50 ms |
     | 2.048 token | 85 ms | 181 ms |

   - Prompt của dev_v0 có trung vị 134 token. Với yêu cầu ngắn cỡ đó, 0.6B **không nhanh hơn**, nên nhường 5 điểm accuracy là không đổi lại được gì.
   - 0.6B chỉ có lợi khi prompt dài và ngân sách độ trễ thật chặt: ở 2.048 token, cả hai model đều dưới 0,2 giây.
2. **Hàng loạt: cân nhắc, nghiêng về 0.6B nếu chi phí GPU là ràng buộc chính.**
   - Accuracy tiếng Việt kém 5,2 điểm; thông lượng ở batch 16 cao gấp 2,1 lần (138 so với 65 câu/giây).
   - Train cũng rẻ bằng khoảng một nửa (62 so với 106 phút; VRAM đỉnh 5,1 so với 10,2 GB).
   - Nếu 5 điểm là quá nhiều cho ứng dụng thì vẫn chọn 1.7B.
3. **Văn bản dài: chọn 1.7B.**
   - Belebele tiếng Việt: 1.7B hơn **11,3 điểm** (từ +9,3 đến +13,3, mọi cặp seed có p ≤ 0,002). Mức này vượt mốc 10 điểm.
   - Độ trễ ở 2.048 token: 85 ms (0.6B) so với 181 ms (1.7B); VRAM đỉnh khi chấm: 1,4 so với 3,6 GB.
   - Lưu ý: 2.048 token vượt mức state ≤ 1.024 token lúc train, và Belebele chỉ dài khoảng 250 token. Accuracy ở 2.048 token **chưa được đo**, đây chỉ là số tốc độ.

## Kết quả chính (thứ tự lựa chọn thường, batch 1)

Accuracy %, trung bình đều theo nguồn, [CI 95% bootstrap theo state]:

| | 0.6B gốc | 0.6B ft s1 | 0.6B ft s2 | 1.7B gốc | 1.7B ft s1 | 1.7B ft s2 |
|---|---|---|---|---|---|---|
| Tiếng Việt | 60,5 [57,0; 64,0] | **76,7** [74,1; 79,3] | **73,5** [70,4; 76,6] | 72,9 [69,9; 75,9] | **79,8** [77,0; 82,6] | **80,8** [77,9; 83,4] |
| Tiếng Anh | 69,3 [66,2; 72,5] | 80,2 [77,5; 83,0] | 80,3 [77,6; 82,9] | 83,4 [81,0; 85,8] | 87,2 [85,0; 89,2] | 86,0 [83,8; 88,3] |
| Brier vi (T=1) | 0,557 | 0,368 | 0,402 | 0,474 | 0,296 | 0,290 |
| Brier en (T=1) | 0,490 | 0,296 | 0,289 | 0,310 | 0,205 | 0,218 |

Accuracy tiếng Việt theo nguồn:

| | 0.6B gốc | 0.6B ft s1 | 0.6B ft s2 | 1.7B gốc | 1.7B ft s1 | 1.7B ft s2 |
|---|---|---|---|---|---|---|
| Belebele | 46,3 | 62,7 | 64,3 | 67,3 | 73,7 | 76,0 |
| SIB-200 | 70,7 | 78,7 | 78,7 | 74,0 | 81,3 | 84,3 |
| MASSIVE | 64,0 | 78,3 | 79,0 | 71,3 | 84,3 | 84,7 |
| XCOPA | 61,0 | 87,0 | 72,0 | 79,0 | 80,0 | 78,0 |

**So với chính model đó trước fine-tune** (McNemar trên cùng câu):

| | Tiếng Việt [CI], p | Tiếng Anh, p |
|---|---|---|
| 0.6B s1 | +16,2 [12,9; 19,4], p=5e-23 | +10,8, p=3e-15 |
| 0.6B s2 | +13,0 [8,9; 16,9], p=9e-18 | +11,0, p=4e-14 |
| 1.7B s1 | +6,9 [4,3; 9,6], p=3e-10 | +3,8, p=0,003 |
| 1.7B s2 | +7,8 [4,9; 10,7], p=1e-12 | +2,6, p=0,009 |

**1.7B − 0.6B trên cùng câu** (tiếng Việt / tiếng Anh, McNemar):

| Cặp | Tiếng Việt [CI], p | Tiếng Anh, p |
|---|---|---|
| Model gốc | +12,4 [8,7; 16,4] | +14,1 |
| s1 − s1 | +3,2 [0,5; 5,7], p=8e-5 | +7,0, p=5e-6 |
| s2 − s2 | +7,3 [4,0; 10,5], p=4e-8 | +5,7, p=4e-6 |
| 1.7B s1 − 0.6B s2 | +6,3 [3,2; 9,4], p=8e-6 | +6,8 |
| 1.7B s2 − 0.6B s1 | +4,1 [1,2; 6,7], p=6e-7 | +5,8 |
| **Trung bình 4 cặp** | **+5,2** (Belebele +11,3; SIB +4,2; MASSIVE +5,8; XCOPA −0,5) | **+6,3** |

**Chênh giữa hai seed (s1 − s2, cùng câu):**
- 0.6B: tiếng Việt +3,2 [0,7; 5,9], trong đó XCOPA +15,0 (p=0,0007), các nguồn khác từ −1,7 đến 0; tiếng Anh −0,2.
- 1.7B: tiếng Việt −0,9 [−3,2; 1,3]; tiếng Anh +1,2.

**Mức tụt Anh → Việt** (ghép theo pair_id): fine-tune làm khoảng cách hẹp lại ở cả hai model và cả hai seed.

| | Gốc | ft s1 | ft s2 |
|---|---|---|---|
| 0.6B | 8,8 | 3,5 | 6,8 |
| 1.7B | 10,5 | 7,3 | 5,3 |

## Thiên vị vị trí: chấm thêm một lượt với thứ tự lựa chọn đảo ngược

Trung bình đều theo nguồn, tiếng Việt:

| | acc TB hai lượt | Nhất quán (cùng nội dung đáp án ở cả hai lượt) | Chênh acc giữa các vị trí đáp án đúng (max−min) |
|---|---|---|---|
| 0.6B gốc | 62,2 | 62,9 | 31,7 |
| 0.6B ft s1 | 76,1 | **86,6** | 9,1 |
| 0.6B ft s2 | 72,9 | **85,2** | 15,3 |
| 1.7B gốc | 72,3 | 73,9 | 22,1 |
| 1.7B ft s1 | 79,9 | **90,0** | 11,7 |
| 1.7B ft s2 | 80,2 | **87,5** | 10,7 |

- Fine-tune có tác dụng đúng như kỳ vọng: tỉ lệ nhất quán tăng rõ, và accuracy hai lượt gần như bằng nhau.
- Ví dụ rõ nhất là XCOPA tiếng Việt của 0.6B:
  - Trước fine-tune: chọn A ở 80/100 câu; đúng 91% khi đáp án là A, 36% khi đáp án là B.
  - Seed 1: đúng 86% khi đáp án là A, 83% khi là B.
  - Seed 2: đúng 64% khi đáp án là A, 79% khi là B, tức thiên lệch đảo sang chữ B. Đây là một phần lý do XCOPA của seed 2 thấp hơn.
- Chênh giữa các vị trí vẫn còn 7–15 điểm. Mỗi ô vị trí chỉ có khoảng 30–75 câu, nên phần chênh này có thể vẫn là nhiễu; tôi chưa kiểm định.

## Tốc độ

**Độ trễ batch 1 theo độ dài prompt** (`scripts/bench_latency.py`, `results/ft/latency_ft.json`):
- Đo trên model đã fine-tune seed 1, LoRA merge vào bf16, đúng đường chấm của eval, khi GPU không làm việc gì khác.
- Mỗi mức: 10 lượt warmup và 60 lượt đo. Chạy hai vòng 0.6B → 1.7B → 0.6B → 1.7B, mỗi model nạp riêng.

| Prompt (token thực tế) | 0.6B p50 (vòng 1 / 2) | 0.6B p90 | 1.7B p50 (vòng 1 / 2) | 1.7B p90 | 1.7B ÷ 0.6B |
|---|---|---|---|---|---|
| 128 | 28,9 / 27,4 ms | 30,0 / 28,7 | 30,1 / 28,0 ms | 32,8 / 29,1 | 1,03× |
| 512 | 27,9 / 26,9 ms | 29,1 / 28,1 | 50,1 / 49,5 ms | 51,5 / 49,8 | 1,8× |
| 2.048 | 84,9 / 84,9 ms | 85,2 / 85,4 | 181,4 / 181,6 ms | 181,9 / 181,8 | 2,1× |
| VRAM đỉnh khi chấm ở 2.048 | 1,41 GB | | 3,56 GB | | |

- **Nguồn prompt:**
  - 128 token: một câu thật của dev_v0 (tiếng Việt), dùng nguyên câu.
  - 512 và 2.048 token: dev_v0 không có câu nào dài tới mức đó, nên state là các đoạn Belebele tiếng Việt của dev_v0 nối lại rồi cắt theo token; câu hỏi và lựa chọn lấy từ một câu Belebele.
  - Hai model dùng cùng văn bản ở mỗi mức.
- **Độ trôi giữa hai vòng:** dưới 2 ms.
- **Vì sao 0.6B ở 128 và 512 token gần như bằng nhau:** ở mức đó, thời gian chủ yếu là chi phí cố định cho mỗi lượt chạy (khởi chạy kernel), không phụ thuộc kích thước model. Đó cũng là lý do hai model bằng nhau ở 128 token.

**Thông lượng và thời gian train:**

| | Batch 1 trên dev_v0 (ms, trung vị) | Batch 16 trên dev_v0 (câu/giây) | Thời gian train | VRAM đỉnh khi train (allocated / reserved) |
|---|---|---|---|---|
| 0.6B ft s1 / s2 | 26,9 / 27,9 | 138 / 138 | 61,7 / 61,8 phút | 5,1 / 6,5 GB |
| 1.7B ft s1 / s2 | 30,8 / 31,5 | 65 / 65 | 106,1 / 106,3 phút | 10,2–10,3 / 12,3–12,4 GB |

## Kiểm tra trước khi báo xong

- [x] **Model gốc chạy lại khớp bước 2:** `preds.jsonl` của cả hai model trùng **từng byte** với `results/run1`.
- [x] **Số dòng trùng giữa train/val và dev cùng phần để dành test bằng 0.**
  - Kiểm bằng ba phép: trùng nguyên văn, gần trùng (MinHash Jaccard ≥ 0,8), và train chứa một đoạn của văn bản dev/test.
  - Văn bản đối chiếu gồm 44.050 văn bản: state của dev_v0, phần để dành test, và mọi split của Belebele, SIB-200, MASSIVE, XCOPA, COPA.
  - Trong lúc lấy mẫu đã loại 105 ứng viên; kiểm lại trên tập cuối được 0.
- [x] **Val loss giảm so với lúc đầu ở cả hai model và cả hai seed:**
  - 0.6B: 2,713 → 0,832 (s1) và 0,840 (s2).
  - 1.7B: 5,495 → 0,761 (s1) và 0,773 (s2).
- [x] **Prompt lúc train và lúc chấm** dựng bằng cùng một hàm `common.render_prompt`.
- [x] **Mỗi lượt chấm và lượt đo có giới hạn thời gian.** Không lượt nào bị treo.

## Những điều bất thường

1. **`letter_mass` sụp ở 1.7B, và lặp lại ở seed 2.** Tiêu chí "về gần 0 câu dưới 0,5" không đạt.

   | | Số câu < 0,5 | TB vi / en |
   |---|---|---|
   | 0.6B gốc | 0 | 0,98 / 1,00 |
   | 0.6B s1 / s2 | 6 / 11 | khoảng 0,93 |
   | 1.7B gốc | 18 | khoảng 0,98 |
   | 1.7B s1 | **1.919/2.000** | 0,12 / 0,16 |
   | 1.7B s2 | **1.865/2.000** | 0,17 / 0,23 |

   - **Chẩn đoán** (`scripts/probe_full_softmax.py`, 5 câu dev, seed 1):
     - Token đứng đầu của 1.7B sau fine-tune vẫn là chữ cái đúng thứ tự, nhưng **logit toàn vocab bị co lại**: logit lớn nhất 35–51 → 7–11; độ lệch chuẩn khoảng 4,1 → khoảng 0,95.
     - Entropy trên toàn vocab tăng từ gần 0 lên 6–11 nat. Phần xác suất còn lại trải đều ra hàng chục nghìn token.
     - Nó không dồn sang biến thể của chữ cái: " G", "Ｇ" và các biến thể khác chỉ có khoảng 0,1–0,2% mỗi token.
     - 0.6B cũng co nhưng ít hơn (logit lớn nhất khoảng 20), và xác suất vẫn dồn vào A–J.
   - **Nguyên nhân:**
     - Loss chỉ là softmax trên k chữ cái. Loss này chỉ phụ thuộc vào **hiệu** giữa logit của các chữ cái, nên không có gì giữ mức logit chữ cái so với phần còn lại của vocab.
     - Hiện tượng lặp lại ở seed 2, nên đây là đặc tính của công thức, không phải ngẫu nhiên. Lý do 1.7B co mạnh hơn 0.6B thì tôi chưa giải thích được.
   - **Ảnh hưởng:**
     - Accuracy, Brier, thiên vị vị trí và temperature đều chỉ dùng softmax trên k chữ cái, nên **không bị sai**.
     - Ai đọc xác suất trên toàn vocab hoặc sinh văn bản tự do sẽ nhận được phân phối gần như phẳng.
     - Logit nhỏ làm sai số làm tròn bf16 chiếm tỉ lệ lớn hơn trong hiệu giữa các chữ cái. Tôi chưa đo ảnh hưởng của điều này.
   - **Đề xuất cho lần train chính thức (chưa áp dụng):** thêm số hạng λ·(−log Σ_{chữ cái của câu} p_vocab), hoặc cross-entropy trên toàn vocab với target đặt vào token chữ cái; theo dõi `letter_mass` trên val nội bộ trong lúc train.
2. **XCOPA tiếng Việt: việc 0.6B hơn 1.7B ở seed 1 không lặp lại ở seed 2.**
   - Seed 1: 0.6B 87 so với 1.7B 80. Seed 2: 72 so với 78.
   - Riêng 0.6B dao động 15 điểm giữa hai seed trên cùng 100 câu; 1.7B chỉ dao động 2 điểm.
   - Kết luận: bất thường ở seed 1 phần lớn là nhiễu của 0.6B trên một bộ 2 lựa chọn, 100 câu. XCOPA tiếng Việt không nên dùng một mình để so hai model.
3. **Val loss tăng nhẹ ở epoch 2** ở 0.6B (cả hai seed: 0,815 → 0,832 và 0,806 → 0,840) và ở 1.7B seed 2 (0,765 → 0,773). Theo plan, tôi vẫn lấy checkpoint cuối epoch 2.
4. **Temperature fit trên val làm Brier của model gốc xấu đi trên dev.**
   - T của hai model gốc là 13,6 và 18,6; Brier tiếng Việt xấu đi: 0,557 → 0,693 và 0,474 → 0,585.
   - Val (tasksource cùng PhoLaya, nhiều noul/score/câu ghép) và dev (choice đọc hiểu/phân loại) **khác phân phối**.
   - Sau fine-tune, T là 1,41–1,55, và Brier trên dev tốt lên một chút ở cả bốn adapter.
5. **Độ trễ của model gốc ở bước này (27,5 và 31,2 ms) thấp hơn bước 2 (32 và 41 ms)** trên cùng phần cứng. Số tuyệt đối dao động giữa các lần chạy; so sánh nên dựa vào số đo trong cùng một phiên.
6. **Sự cố vận hành, không ảnh hưởng kết quả:**
   - Lần chạy 0.6B đầu tiên chậm (4,5 giây/bước) nên tôi dừng. Sau đó tôi sắp 32 câu của mỗi batch hiệu dụng theo độ dài trước khi chia micro-batch; gradient vẫn là trung bình trên đúng 32 câu đó.
   - Lần khởi động 1.7B đầu tiên lỗi vì tôi gõ sai tham số timeout; tôi đã chạy lại.

## Những gì tôi không chắc

- **Chỉ có 2 seed.** Ước lượng nhiễu giữa seed từ 2 điểm là rất thô. Đặc biệt với 0.6B, chênh lệch 3,2 điểm giữa hai seed đến từ một nguồn 100 câu.
- **Ngôn ngữ của dòng tasksource** được suy ra từ tên nguồn. 403 dòng train là ngôn ngữ khác Anh/Việt nhưng dùng khung prompt tiếng Anh.
- **Đã loại 30 dòng có lựa chọn dẫn chiếu tới lựa chọn khác** (kiểu "A and B"). Câu `score` vẫn được xáo theo yêu cầu, dù đây là thang có thứ tự.
- **Lọc trùng khá bảo thủ:** 97/105 ứng viên bị loại là do chứa nguyên một câu lệnh MASSIVE từ 4 từ trở lên. Có thể đã loại cả câu không thật sự trùng.
- **Nguồn đã thấy:** ViNLI, VSFC, ViSFD nằm trong train_v0. Nếu chúng có mặt trong khối B của benchmark thì phải gắn cờ "nguồn đã thấy" (PLAN.md mục 8).
- **License của train_v0:** 4.000 dòng chỉ dùng cho nghiên cứu (PhoLaya) và 1.606 dòng non-commercial (tasksource). Chỉ dùng nội bộ, đã ghi trong manifest.
- **Độ trễ ở 512 và 2.048 token** đo bằng văn bản Belebele nối lại. Một văn bản tự nhiên dài cùng số token sẽ cho tốc độ như nhau, nhưng accuracy ở độ dài đó chưa được đo.

## Dữ liệu train_v0 (`data/manifest_train_v0.json`)

- **Số dòng:** 20.000 train và 1.000 val; seed dữ liệu 20261012.
  - tasksource `@bf1e4a87`: 16.000 train + 800 val.
  - PhoLaya `@034d4273` (ViNLI, VSFC, ViSFD): 4.000 train + 200 val.
- **Theo loại câu hỏi:** choice 15.162, noul 3.375, score 1.463. Khung prompt: en 15.998, vi 4.002.
- **Giới hạn 5% tính theo tác vụ:** nguồn lớn nhất là 6 tác vụ PhoLaya, mỗi tác vụ khoảng 3,3%. Tính theo bộ dữ liệu thì mỗi bộ PhoLaya chiếm khoảng 6,7%.
- **Đã loại 26 nguồn tasksource (theo `sources.yaml`):** IntentGrasp (chứa câu lệnh MASSIVE/SLURP), slurp/intent, multilingual/massive, super_glue/copa, balanced-copa, và 21 nguồn multilingual/xcopa. Không có nguồn nào trỏ tới FLORES, Belebele hay SIB-200.
- **Val chia theo văn bản gốc:** union-find trên group_id, state đã chuẩn hóa và từng trường văn bản của PhoLaya. Không có văn bản nào nằm ở cả hai phía.

## Công thức train (giống nhau cho cả hai model và cả hai seed)

- **LoRA:** r=16, α=32, dropout 0,05 trên q, k, v, o, gate, up, down. Số tham số train: 0.6B 10,1 M; 1.7B 17,4 M.
- **Lịch học:** 2 epoch; batch hiệu dụng 32 (micro 8 × 4); lr 2e-4, warmup 3% rồi cosine giảm về 0; grad clip 1,0. Learning rate không phải chỉnh.
- **Độ chính xác số:** trọng số bf16 với autocast bf16; tham số LoRA ở fp32. Gradient checkpointing bật.
- **Loss:** cross-entropy giữa softmax trên k chữ cái ở vị trí trả lời và phân phối target. Xáo lại thứ tự lựa chọn mỗi lần nạp câu.
- **Checkpoint:** lấy cuối epoch 2. Không dùng dev_v0 để chọn checkpoint hay hyperparameter.
- **Temperature:** fit trên val nội bộ.
- **Lúc chấm:** LoRA được merge vào trọng số bf16 rồi chấm bằng đúng đường chấm của model gốc, ở batch 1.

## File

| File | Nội dung |
|---|---|
| `data/train_v0.jsonl`, `data/val_v0.jsonl`, `data/manifest_train_v0.json` | tập train/val nội bộ và manifest |
| `results/ft/train/<model>-s{1,2}/` | adapter, `train_log.jsonl`, `summary.json` |
| `results/ft/eval/{normal,reverse,val}/<tên>/preds.jsonl` | dự đoán thô từng câu (xác suất, logit chữ cái, letter_mass) |
| `results/base_rerun/` | model gốc chấm lại, trùng từng byte với `results/run1/` |
| `results/ft/temp/*.json` | temperature fit trên val |
| `results/ft/latency_ft.json` | độ trễ batch 1 theo độ dài prompt |
| `results/ft/report_ft.md`, `metrics_ft.json` (cùng bản `_s1only`) | bảng đầy đủ và số liệu |
| `scripts/build_train_v0.py`, `decontam.py`, `train_lora.py`, `fit_temperature.py`, `report_ft.py`, `stats.py`, `probe_full_softmax.py`, `bench_latency.py` | code |
| `setup/train_one.sh`, `pipeline_ft.sh`, `pipeline_ft2.sh`, `pipeline_ft3.sh` | cách chạy |
