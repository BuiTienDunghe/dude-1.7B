# dude-1.7B: model quyết định tiếng Việt chạy dưới 40 ms

Model nhỏ dựa trên **Qwen3-1.7B + LoRA**. Đầu vào là một văn bản và một câu hỏi có sẵn các lựa chọn. Đầu ra là xác suất cho từng lựa chọn, không sinh văn bản. Repo đi kèm một **benchmark tiếng Việt** dùng để chấm mọi model trong cùng điều kiện.

> **Repo riêng tư, chỉ dùng nội bộ.** Chỉ có code, cấu hình nhỏ, báo cáo và thông số. Không có dữ liệu, trọng số hay test (lý do ở mục 4).

## 1. Mục tiêu (đặt ngày 27/10/2026)

| Chỉ tiêu | Ngưỡng | Hiện tại (v0.3-2b, TB 2 seed) |
|---|---|---|
| VI có dấu, dev 1.3, trung bình 6 nguồn | > 86,5 (vượt decider-2b) | 84,38 — chưa đạt |
| VI không dấu | ≥ 61 | 62,85 — đạt |
| Tiếng Anh | ≥ 87 | 88,68 — đạt |
| Độ trễ, trung vị batch 1, GPU 16 GB | ≤ 40 ms | khoảng 32 ms — đạt |

## 2. Các phiên bản

Xem **[RELEASES.md](RELEASES.md)** hoặc tab *Releases* trên GitHub. Mỗi phiên bản ghi đủ: model nền, dữ liệu (số dòng, sha), cấu hình LoRA và tối ưu, thời gian và VRAM, điểm theo từng nguồn, phép thử, độ trễ, luật đạt hoặc trượt, quyết định, sha của adapter.

| Tag | Model | VI có dấu | Không dấu | EN | Trạng thái |
|---|---|---|---|---|---|
| `v0.1-m1` | M1, train_v0 (12/10) | 79,88 | 62,37 | 87,95 | mốc tham chiếu |
| `v0.2-c` | C, thêm dữ liệu VI và dữ liệu dịch (26/10) | 81,65 | 63,97 | 88,20 | mốc tham chiếu |
| `v0.3-2b` | 2b, dữ liệu đúng khung bench, 2 seed (28–29/10) | 84,38 | 62,85 | 88,68 | chưa nhận: T3b trượt |
| `bench-v1.3` | Benchmark 1.3 (17/10) | — | — | — | đóng băng; test chưa mở |

Tiếp theo là 2b′ (hướng 1), rồi hướng 2 (MASSIVE, Belebele, XCOPA). Xem `results/TONG_KET_DU_AN_29_10.html`.

## 3. Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `scripts/` | Code Python: dựng bench, dựng dữ liệu, train (`train_lora.py`), chấm (`eval_base.py`, `scripts/probe_v1/score_probe.py`), khử trùng (`decontam.py`), phân tích |
| `setup/` | Script chạy trong WSL: hàng đợi GPU, lính gác nhiệt (`g0_temp_guard_v2.sh`), chạy train từng bậc |
| `tools/review_app/` | Công cụ hiệu đính và cổng duyệt chuỗi tiếng Việt, chỉ chạy ở 127.0.0.1 |
| `config/` | Khung câu hỏi, bản dịch nhãn, danh sách model |
| `results/**/*.md` | Báo cáo từng bước: brief, báo cáo worker, phân tích của Fable, kiểm toán |
| `PLAN_decisions.md` | Nhật ký mọi quyết định, có ngày |
| `TONG_QUAN_DU_AN.md`, `results/TONG_KET_DU_AN_29_10.html` | Tổng quan và tổng kết |
| `MANIFEST_SHA.md` | sha256 của dữ liệu, bench và adapter không đưa lên repo |
| `release_notes/` | Ghi chú từng phiên bản, cũng là nội dung của các GitHub Release |

Đường dẫn trong báo cáo trỏ vào thư mục làm việc gốc `C:\Users\dungbt06\Fine-tune`. Nhiều file nhắc tới trong báo cáo không có trong repo.

## 4. Không có trong repo, và vì sao

- **Dữ liệu train và bench** (`data/`, `results/**/*.jsonl`): có dữ liệu UIT (ViANLI, ViWikiFC, ViFactCheck…) chỉ dùng nội bộ, và có phần bản dịch do người dùng tự làm. **Test bench được khóa bằng mã băm**: lọt ra ngoài thì bench mất giá trị.
- **Trọng số** (20 adapter, khoảng 1,3 GB): train trên dữ liệu chỉ dùng nội bộ. Mã băm nằm trong `MANIFEST_SHA.md`.
- **Báo cáo soát dịch toàn bench** (`results/translation_audit*`) và **ứng viên bench 1.4** (`results/bench_candidates/`): có chứa nội dung test, hoặc nội dung sẽ thành test sau này.
- **File kết quả từng câu, checkpoint, log, tài liệu của bên thứ ba.**

Trước khi push, `scripts/xuat_github/kiem_xuat.py` kiểm ba thứ:
- không chuỗi riêng nào của test (mọi bản 1.0–1.3) nằm trong repo;
- không có chuỗi giống khóa hay token;
- không file nào quá 10 MB.

Repo được dựng lại bằng `scripts/xuat_github/xuat_dude.py`.

## 5. Luật làm việc của dự án

- Không chạy model trên test. Chỉ chấm test một lần ở cuối, theo giao thức ghi trước.
- Mỗi lần train cần người làm dự án duyệt. Trước đó có phân tích trước train và Fable phản biện. Luật đọc kết quả được ghi trước.
- Agent không tự viết chữ tiếng Việt mới vào bench, config hay dữ liệu train. Chữ mới phải qua cổng duyệt.
- Ứng viên nào cách nhau dưới 2 điểm thì phải chạy 2 seed mới so được.

Chưa chọn giấy phép (license).
