# BRIEF: Bước 0 (sửa lỗi) và G0 (đo, không train) — 28/10/2026

**Người dùng đã duyệt (chat 28/10): "duyệt 0 + G0".** Bối cảnh:
- `results/huong_di_2710/TONG_HOP_huong_di.md`;
- `results/audit_2710/TONG_HOP_dieu_phoi.md`.

Mục tiêu dự án đã đổi thành "model tiếng Việt tốt nhất trong ≤ 40 ms". Mốc mới: VI có dấu > 86,5, không dấu ≥ 61, EN ≥ 87.

**Ngân sách GPU:** khoảng 1,5 giờ, tối đa 2 giờ. Vượt 2,5 giờ thì dừng và báo.

## Quy tắc (giữ nguyên các quy tắc cũ)

- **KHÔNG train.**
  - Được làm một lượt forward + backward trên batch dài nhất để đo VRAM.
  - Không optimizer step, không lưu adapter.
- **Không đụng test.** Chỉ dùng dev 1.3, sha `c6a9a75d…`.
- **Không viết chữ tiếng Việt mới.** Không sửa `config/` hay `data/bench_v1_3/`, không xóa file, không commit.
- **Cài đặt chỉ trong venv mới `.venv-g0`.** Không đụng `.venv` hay `.venv-baselines`.
- **Ghim revision cho mọi model tải về.** Ghi sha vào manifest.
- **Mức dùng Claude:** từ 88% (5 giờ hoặc tuần) thì không bắt đầu việc mới, đến 92% thì dừng.
- **Các con số dưới đây đã chốt trước khi thấy kết quả.** Không đổi cách đọc sau khi có số.

## Bước 0: sửa lỗi

1. **`scripts/train_lora.py`:**
   - `shuffled()` chỉ xáo câu `kind == "choice"`; giữ nguyên thứ tự câu `noul` và `score`. Đây là quyết định 12/10, kèm audit 27/10.
   - Đổi mặc định `--loss` thành `full_vocab`.
   - Thêm cách chia micro-batch theo ngân sách token (như `--token_budget` của ft_decider), để dòng dài tới 2.048 token không làm tràn 16 GB.
   - Đo VRAM trên batch dài nhất có thể có, giả lập bằng dòng 2.048 token.
2. **Trần độ dài state:**
   - Cho trần 1.024 token trong `scripts/build_train_v0.py:36` và `scripts/train_v1/recast_v0.py:12` thành tham số, mặc định **2.048**.
   - **Không dựng lại dữ liệu** nào lúc này. C′ là bước sau.
3. **Kiểm chữ nhãn: viết `scripts/check_label_strings.py`.**
   - Script so tập chuỗi lựa chọn có nhãn cố định (kiểm chứng, Có/Không, SIB, MASSIVE) giữa một file train bất kỳ và dev 1.3, theo từng nguồn hoặc khung.
   - Chạy trên `C.jsonl` và báo cáo. Kết quả phải cho thấy lệch "Văn bản không đủ thông tin" (C) so với "…để kết luận" (dev).
   - Từ nay, mọi bộ dựng dữ liệu phải lấy chữ nhãn **từ dev 1.3 hoặc lớp ghi đè đã duyệt, không lấy từ `config/bench_v1_questions.json`**. Ghi quy tắc này vào docstring các script dựng.
4. **Tập chọn model riêng ("proxy").** Dùng để chọn checkpoint và fit ngưỡng noul; **không bao giờ train trên tập này**.
   - Lấy từ phần **không thuộc** dev hay test 1.3 của chính 6 nguồn bench: các split train/validation còn lại của ViFactCheck, ViWikiFC, SIB-200, MASSIVE, XCOPA.
   - Belebele không có split train, nên bỏ hoặc ghi rõ cách xử lý.
   - Dùng đúng khung bench và chữ đã duyệt. Có VI có dấu, VI không dấu, và EN nếu nguồn song song.
   - Cỡ: khoảng 150 câu mỗi nguồn VI.
   - Kiểm khử trùng bằng `scripts/decontam.py`, gồm trùng chính xác, MinHash ≥ 0,8 và 13-gram:
     - với dev và test 1.3: **0 chạm**;
     - với `train_v0` và `C.jsonl`: **0 chạm**.
   - Ghi `data/proxy_v1/` kèm manifest và sha.
   - Đánh dấu dữ liệu UIT là chỉ dùng nội bộ.
5. **Kiểm sau khi sửa:**
   - Chạy giả `--dump_batches` trên C: báo số bước, và xác nhận dòng noul/score không bị xáo.
   - Chấm lại bản C (adapter cuối, seed 1) trên dev bằng `eval_base.py`: **preds phải trùng từng byte với lần chấm cũ**, vì code chấm không đổi.

## G0: đo (không train)

### Chuẩn bị

- **Tải, ghim revision:**
  - `Qwen/Qwen3.5-2B-Base`, khoảng 4,5 GB;
  - `Qwen/Qwen3.5-4B` **revision `851bf6e806ef`**, đúng nền của Quyet-Medium, khoảng 9,3 GB.
  - Chỉ nạp phần văn bản.
- **`.venv-g0`:** sao từ lock của `.venv`, thêm `flash-linear-attention` (ghim phiên bản). Nếu không cài được hoặc không chạy được trên sm_120:
  - ghi lỗi;
  - đo không có FLA;
  - **không vá thư viện**.
- **Nếu tokenizer bản Base không có chat template:** dùng template của bản post-train cùng họ (cùng vocab) và ghi rõ đã làm vậy. Assert chữ A–J là 1 token, như `eval_base.py` đang làm.

### G0b: chấm trên dev 1.3

- **Model:** Qwen3.5-2B-Base và Qwen3.5-4B (revision trên).
- **Cách chấm:** khung dự án (`render_prompt`), batch 1, toàn bộ dev (VI có dấu, VI không dấu, EN).
- **Báo cáo:**
  - điểm theo nguồn và trung bình đều;
  - CI theo cụm;
  - so với Qwen3-1.7B gốc (M0), Qwen3.5-2B gốc (70,6), M1, C, decider-2b và Quyet-Medium (preds có sẵn).

### G0c: khung prompt tiếng Anh

- **Model:** Qwen3-1.7B gốc, Qwen3.5-2B gốc, Qwen3.5-2B-Base.
- **Dữ liệu:** các dòng VI có dấu của dev.
- **Cách làm:** chấm bằng khung `en` của `config/prompt_frames.json` (chỉ dẫn và nhãn khung bằng tiếng Anh; nội dung, câu hỏi và lựa chọn giữ tiếng Việt), rồi so với khung `vi`.
- **Không có chữ mới.**

### G0a: độ trễ, đo cùng một cách

- **Viết `scripts/latency_g0.py`.** Đường suy luận có CUDA graphs: torch.compile `mode="reduce-overhead"` hoặc bắt graph tay theo nhóm độ dài pad cố định. Chỉ một forward, `logits_to_keep=1`.
- **Cổng đúng:** trên toàn bộ dev, argmax trùng đường eager **100%**, xác suất lệch ≤ 1e-3. Không đạt thì báo, không dùng số đo.
- **Model:**
  - C-s1 đã merge (Qwen3-1.7B);
  - Qwen3.5-2B-Base, có FLA và không FLA;
  - Qwen3.5-4B, một lượt, chỉ để có số thật.
- **Báo cáo, cho cả eager và graphs:**
  - trung vị, p90, p95, p99;
  - **p90 theo nhóm độ dài prompt:** ≤ 128, 129–256, 257–512, 513–1.024, > 1.024 token.
- **GPU phải rảnh.**
  - Trước mỗi lượt đo, kiểm `nvidia-smi` xem có tiến trình nào khác dùng GPU, và kiểm mức dùng GPU trong 10 giây.
  - Có tiến trình khác thì đợi, thử lại mỗi 5 phút, ghi vào log.
  - Mỗi cấu hình đo **2 lần**; hai lần lệch > 10% thì đo lần 3.
  - Lưu `nvidia-smi` trước và sau.

## Cách đọc (chốt trước khi thấy số)

- **Nền 2B.**
  - **"Đáng cân nhắc đổi nền"** nếu cả hai điều sau đều đúng:
    - Qwen3.5-2B-Base hơn Qwen3-1.7B gốc **≥ 3 điểm** VI có dấu, và CI theo cặp không chứa 0;
    - p90 có CUDA graphs của 2B-Base **≤ p90 của C + 5 ms**.
  - Thiếu một trong hai điều kiện thì **giữ 1.7B**.
  - Quyết định cuối do người dùng.
- **Điểm của Quyet đến từ đâu.**
  - Qwen3.5-4B gốc **≥ 85** VI có dấu → lợi thế của Quyet chủ yếu do cỡ.
  - **≤ 75** → chủ yếu do công thức fine-tune.
  - Ở giữa → trộn lẫn.
- **Khung prompt.**
  - Khung EN hơn khung VI **≥ 2 điểm**, CI không chứa 0, ở **ít nhất 2/3 model** → ghi "khung VI có thể làm mất điểm", để người dùng quyết khi dựng C′.
  - Ngoài trường hợp đó → giữ khung VI.
- **Độ trễ:** chỉ báo số, không kết luận về ngưỡng 40 ms. Định nghĩa ngưỡng do người dùng quyết sau G0.

## Đầu ra

- `results/G0/REPORT_G0.md`: kết luận trước, rồi bảng.
- `results/G0/`: preds, log, manifest (revision, sha, phiên bản thư viện).
- Báo cáo bước 0: `results/G0/REPORT_buoc0.md`.
- **Xong thì dừng và báo điều phối. Không sang bước 2a.**
