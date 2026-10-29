# Bước 0: sửa lỗi theo audit (28/10/2026)

Theo `results/BRIEF_buoc0_G0.md`; người dùng duyệt "duyệt 0 + G0" ngày 28/10.

**Phạm vi:**
- **Không train.** Chỉ một lượt forward + backward để đo VRAM, không optimizer step, không lưu adapter.
- Không đụng test, không thêm chữ tiếng Việt mới, không sửa `config/` hay `data/bench_v1_3/`, không xóa file, không commit.
- Ghi `data/proxy_v1/` là việc brief cho phép.

## Tóm tắt

| Mục | Kết quả |
|---|---|
| `shuffled()` chỉ xáo câu choice | Đã sửa. Trên 1.938 bước của C: **13.095/13.095 câu noul/score giữ nguyên thứ tự**. Thứ tự batch không đổi (`ids_sha` trùng bản cũ) |
| Loss mặc định | `full_vocab` |
| Chia micro-batch theo ngân sách token | `--token_budget N`, mặc định 0 = đường cũ. Đo VRAM: câu 2.048 token với ngân sách 16.384 cần 8,37 GB |
| Trần state | Thành tham số, mặc định **2.048**. Các nơi dựng v0/A/B/C cũ truyền 1.024 tường minh. **Chưa dựng lại dữ liệu nào** |
| `scripts/check_label_strings.py` | Đã viết. Trên C: bắt được "Văn bản không đủ thông tin" (vitaminc, 593 lần) và "không đủ thông tin" (quail, 354 lần), lệch nhãn dev "…để kết luận". train_v0: 0 lệch |
| Tập chọn model `data/proxy_v1` | 769 cặp, 2.077 dòng. Khử trùng 2 chiều với dev, test 1.3, train_v0, val_v0 và C: **0 chạm** |
| Chấm lại C trên dev | **Trùng từng byte** với lần chấm cũ (sha `044EBE45…`) |

## 1. `scripts/train_lora.py`

**`shuffled()`**
- Chỉ áp hoán vị cho câu `kind == "choice"`. Câu `noul` (Có/Không) và `score` (thang có thứ tự) giữ nguyên thứ tự lựa chọn và target. Đây là quyết định 12/10, thực thi sau audit 27/10.
- Hoán vị vẫn được rút cho **mọi** câu, nên dòng số ngẫu nhiên tiêu y như cũ. Thứ tự batch và hoán vị của câu choice giống hệt các lần train cũ cùng seed.

Kiểm bằng chạy giả `--dump_batches`, không nạp model:
- **So với bản chạy giả cũ** (C, `strip_p 0.3`): `ids_sha` và số câu bỏ dấu trùng ở cả 3 bước. `tokens_sha` khác ở cả 3 bước, đúng như dự kiến, vì 13 câu noul/score trong 3 bước đó nay không bị xáo.
- **Toàn bộ 1.938 bước của C** (`results/G0/buoc0/dump_C_micro8/batch_dump_kiem_loai_cau.json`):

  | Loại câu | Số lượt | Kết quả |
  |---|---|---|
  | noul/score | 13.095 | 13.095 giữ nguyên thứ tự (so theo bản bỏ dấu) |
  | choice | 48.921 | 41.002 đổi thứ tự; phần còn lại rơi đúng hoán vị đồng nhất |

**Loss mặc định:** `--loss` mặc định là `full_vocab`. Các lần train trước đều truyền `full_vocab` tường minh nên không bị ảnh hưởng.

**`--token_budget N`:**
- Chia batch hiệu dụng 32 câu (đã sắp theo độ dài) thành các nhóm, mỗi nhóm có (số câu × độ dài dài nhất làm tròn lên bội 64) ≤ N, giống ft_decider. Gradient vẫn là Σ loss / 32.
- Mặc định 0 giữ đúng đường cũ (micro-batch 8 × accum 4).

| Cách chia, trên C | Số micro-batch | Nhóm lớn nhất (token sau pad) |
|---|---|---|
| micro-batch 8 (cũ) | 7.752 | **21.504** |
| `--token_budget 16384` | **3.793** | 16.384 |

**`--vram_probe T`:** một lượt forward + backward trên nhóm câu giả dài T token, lấy state dài nhất của file train lặp lại rồi cắt. Không optimizer step. Kết quả với Qwen3-1.7B, LoRA r16, gradient checkpointing (`results/G0/buoc0/vram/`):

| Ngân sách | Số câu 2.048 token | Token sau pad | Đỉnh VRAM (allocated / reserved) |
|---|---|---|---|
| 8.192 | 4 | 8.192 | 5,85 / 6,07 GB |
| **16.384** | 8 | 16.384 | **8,37 / 8,61 GB** |
| 32.768 | 16 | 32.768 | 13,41 / 13,73 GB (GPU 15,9 GB) |

Chưa gồm trạng thái AdamW của 17,4 triệu tham số LoRA (khoảng 0,14 GB).

**Đề xuất cho C′:** `--token_budget 16384`. Mức này còn dư khoảng 7 GB, và giữ được micro-batch 8 câu khi câu dài tới 2.048 token. Mức 32.768 vừa nhưng sát trần.

**Thêm:** `--train_file` / `--val_file` (26/10) và `--save_fracs` (26/10) giữ nguyên.

## 2. Trần độ dài state

| File | Thay đổi |
|---|---|
| `scripts/train_v1/recast_v0.py` | `make_finalize(..., max_state_tokens=2048)`; khóa lý do loại thành `state_over_<N>_tokens` |
| `build_stage0.py`, `update_stage0_v2.py`, `decontam_vi.py` | Truyền `max_state_tokens=1024` tường minh, để dựng lại v0/A/B/C đúng như cũ |
| `scripts/build_train_v0.py` | Thêm `--max_state_tokens` (mặc định 2.048). Muốn dựng lại đúng train_v0/val_v0 thì truyền 1.024 |

**Chưa dựng lại dữ liệu nào.** C′ là bước sau.

## 3. Kiểm chữ nhãn: `scripts/check_label_strings.py`

**Cách làm:**
- Bộ chữ chuẩn lấy **từ dev 1.3**, không từ config: nhãn kiểm chứng, Có/Không, 7 chủ đề SIB-200, 60 ý định MASSIVE, tách theo biến thể (VI có dấu, không dấu, EN).
- Mỗi lựa chọn trong file train được xếp vào một trong ba loại:
  - "trùng" nhãn dev;
  - "gần trùng mà lệch": chuỗi ngắn ≥ 2 từ nằm trong chuỗi dài ≤ 10 từ, hoặc giống nhau ≥ 0,9 khi độ dài lệch ≤ 20%;
  - không liên quan.
- `--strict` thoát mã 1 nếu có chuỗi lệch.

**Kết quả trên `results/train_v1/components/C.jsonl`** (`results/G0/buoc0/check_label_strings_C.json`): 10.037 lần trùng nhãn dev; 4.050 lần gần trùng mà lệch (104 chuỗi). Các chuỗi lệch lớn:

| Nguồn train | Chữ trong train | Nhãn dev gần nhất | Số lần |
|---|---|---|---|
| vitaminc | "Văn bản không đủ thông tin" | "Văn bản không đủ thông tin để kết luận" | **593** |
| quail | "không đủ thông tin" | "Văn bản không đủ thông tin để kết luận" | **354** |
| vntc | "khoa học" | "khoa học/công nghệ" | 1.000 |
| vntc | "chính trị - xã hội" | "chính trị" | 1.000 |
| vi-tourism-admin-intent | "gợi ý địa điểm du lịch nên đến" (và 2 nhãn khác) | "du lịch" (SIB) | 1.000 mỗi nhãn |

Phần còn lại là các câu ngắn tình cờ chứa một chủ đề, ví dụ "Họ là khách du lịch." trong cosmos_qa.

Độ phủ: C có đúng chữ 2/3 nhãn kiểm chứng VI có dấu (thiếu nhãn NEI), 4/7 chủ đề SIB VI, 0/60 ý định MASSIVE, và 0 nhãn ở bản không dấu.

**Trên `data/train_v0.jsonl`:** 3.003 lần trùng, **0 lệch**.

**Quy tắc mới** đã thêm vào docstring của 7 script dựng dữ liệu: `build_train_v0.py`, `recast_v0.py`, `build_stage0.py`, `update_stage0_v2.py`, `rebuild_a_approved.py`, `decontam_vi.py`, `apply_bench_labels.py`. Nội dung: chữ nhãn, lựa chọn cố định và câu hỏi mẫu lấy từ dev 1.3 hoặc lớp ghi đè đã duyệt, **không** lấy từ `config/bench_v1_questions.json` hay `config/translations.json`; kiểm bằng `check_label_strings.py` trước khi dùng.

## 4. Tập chọn model `data/proxy_v1/`

Dùng để chọn checkpoint và fit ngưỡng noul. **Không bao giờ train trên tập này.** ViFactCheck và ViWikiFC là dữ liệu UIT, **chỉ dùng nội bộ**.

**Script:** `scripts/proxy_v1/build_proxy_v1.py`, seed lấy mẫu 20261012 + 2810. Đọc file nguồn gốc đã tải ở `results/train_v1/raw_refs/`.

**Đầu ra:** `proxy_v1.jsonl` (sha `b003b0f6…`), `manifest_proxy_v1.json`.

| Nguồn | Lấy từ | Ứng viên | Cặp | Dòng (VI có dấu / không dấu / EN) |
|---|---|---|---|---|
| ViFactCheck | split train | 5.062 | 150 (choice 78, noul 72) | 150 / 150 / — |
| ViWikiFC | split train | 16.738 | 150 (choice 72, noul 78) | 150 / 150 / — |
| SIB-200 | split train trừ 200 câu dev 1.3 (split validation đã dùng hết cho dev) | 501 | 150 | 150 / 150 / 150 |
| MASSIVE | split train | 11.514 | 150 | 150 / 150 / 150 |
| XCOPA | 200 câu split test gốc **không** vào test 1.3 (XCOPA tiếng Việt không có split train); EN lấy từ COPA test theo idx, như bench | 200 | 150 | 150 / 150 / 150 |
| Belebele | câu có đoạn văn ngoài dev/test 1.3 (250 câu); 212 bị loại vì chung 13-gram (đoạn văn FLORES trùng với SIB và dev) | 250 | **19** | 19 / 19 / 19 |

**Khung và chữ:** đúng khung bench. Mọi chữ nhãn và câu hỏi mẫu **lấy từ dev 1.3**:
- tiền tố câu hỏi kiểm chứng;
- 3 nhãn kiểm chứng; Có/Không;
- 7 chủ đề SIB, ghép VI–EN theo vị trí;
- 60 ý định MASSIVE: khóa ↔ chữ EN suy ra từ nhãn đúng và danh sách nhiễu trong dev, chữ VI theo vị trí;
- câu hỏi XCOPA.

Không có chữ tiếng Việt mới. Cách dựng giống `build_bench_v1.py`:
- kiểm chứng: choice hoặc noul theo seed từng dòng;
- MASSIVE: 1 đúng + 9 nhiễu;
- lấy mẫu cân bằng nhãn;
- mã cặp `proxy:<nguồn>:<khóa>`;
- mỗi đoạn văn kiểm chứng chỉ lấy một nhận định.

**Khử trùng:**
- Phạm vi: dev 1.3, test 1.3 (chỉ dùng để dò, không in nội dung), `data/train_v0.jsonl`, `data/val_v0.jsonl`, `results/train_v1/components/C.jsonl`.
- **Chiều thuận** dùng `scripts/decontam.py`: trùng chính xác, MinHash ≥ 0,8, chung 13-gram, câu 4–12 từ nằm trọn trong ứng viên.
- **Chiều ngược:** câu ứng viên 4–12 từ nằm trọn trong văn bản tham chiếu; câu dưới 4 từ chỉ xét trùng chính xác.
- Câu hỏi hay lựa chọn lặp ≥ 3 lần trong một file coi là khung, không tính.
- Ứng viên chạm thì bỏ. Kiểm lại toàn tập sau cùng: **0 chạm**.
- Số bị loại theo lý do ghi trong manifest. Ví dụ ViFactCheck: 226 trùng chính xác, 119 chứa câu ngắn, 27 chung 13-gram, 2 gần trùng.

## 5. Kiểm sau khi sửa

- **Chạy giả trên C** (mục 1): 1.938 bước; câu noul/score không bị xáo.
- **Chấm lại bản C** (adapter cuối, seed 1) trên dev 1.3 bằng `eval_base.py`. `eval_base.py` có thêm tùy chọn cho G0 (`--revision`, `--frame`, `--chat_template_from`), mặc định giữ nguyên.
  - Kết quả: `results/G0/rescore_C/C-s1/preds.jsonl` **trùng từng byte** với `results/probe_v1/eval/dev/C-s1/preds.jsonl`.
  - sha256 `044EBE45C6913028E4B3B624A521CFF8093454B61C6A8ED550359406FD479CA8`; chạy lúc 00:42, 336 giây.

## File

- **Sửa:** `scripts/train_lora.py`, `scripts/eval_base.py`, `scripts/build_train_v0.py`, `scripts/train_v1/{recast_v0,build_stage0,update_stage0_v2,decontam_vi,rebuild_a_approved,apply_bench_labels}.py`.
- **Mới:** `scripts/check_label_strings.py`, `scripts/proxy_v1/build_proxy_v1.py`.
- **Kết quả:** `results/G0/buoc0/` (dump, VRAM, kiểm chữ nhãn), `results/G0/rescore_C/`, `data/proxy_v1/`.
