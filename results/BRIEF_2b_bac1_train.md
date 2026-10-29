# BRIEF 2b bậc 1: train 2b-s1 và chấm (28/10/2026)

**Người dùng duyệt** (chat 28/10, khoảng 10:50): "được, đồng ý duyệt, bạn chủ động quyết nhưng nhớ để ý ngưỡng tôi nhắc từ trước".

Thiết kế và luật đọc: `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md` mục 6 và **mục 9**. Mục 9 thay các luật tương ứng ở mục 6.

## 0. Ngưỡng dừng (bắt buộc)

- Nhiệt độ GPU > **85°C** → dừng train ngay. Giữ lính gác nhiệt chạy nền suốt lần train.
- Mức dùng tuần > 90% hoặc Fable > 90% → dừng. Điều phối theo dõi và sẽ báo nếu gần ngưỡng.
- Không đụng test. Không sửa `config/`, `data/bench_v1_3/`. Không xóa, không commit.
- **Chỉ train đúng 1 lần (2b-s1).** Không chạy seed 2, không chạy lại C, không train gì khác.
- Ngân sách GPU của brief: tối đa **5 giờ**. Vượt thì dừng và báo.

## 1. Nhận GPU

- Chỉ bắt đầu khi điều phối báo **"GPU SẴN"** (đã nhờ bên Ún nhường).
- Kiểm GPU rảnh theo tiêu chí cũ: không có tiến trình, VRAM < 500 MiB, công suất < 40 W liên tục 10 giây.

## 2. Đo VRAM (vài phút)

Chạy `scripts/train_lora.py --vram_probe 1664`, cùng model và LoRA như dưới.
- reserved ≤ 14 GB → train theo đường micro-batch 8 như C (`--micro_batch 8`, `--token_budget` 0).
- reserved > 14 GB → `--token_budget 9216` (đỉnh bằng C). Gradient tương đương (Σ loss / 32).

Ghi lựa chọn và số đo vào manifest.

## 3. Train 2b-s1

- **Dữ liệu:** `results/train_2b/train_2b_v3.jsonl`. Kiểm sha `ea604a01…` trước khi chạy.
- **Cấu hình:** chép đúng `scripts/train_v1/run_train_C.sh`, chỉ đổi file train, thư mục đầu ra và phần chia batch ở mục 2:
  - Qwen/Qwen3-1.7B, seed 1;
  - LoRA như C (r16/α32);
  - lr 2e-4, 2 epoch, batch hiệu dụng 32, warm-up 3%;
  - `--loss full_vocab`, `strip_p` 0;
  - val = `val_v0` như C; cổng letter_mass theo cùng tỉ lệ bước như C;
  - `PYTHONHASHSEED=0` và các biến môi trường như C;
  - `--save_fracs 0.25,0.5,0.75`.
- **Trần state:** 2.048. Kiểm không dòng nào bị cắt.
- **Đầu ra:** `results/train_2b/train/2b-Qwen3-1.7B-s1/`, gồm `summary.json` (thời gian, VRAM đỉnh, nhiệt độ tối đa, số bước).
- Cổng letter_mass trượt thì dừng và báo, như với C.

## 4. Chấm (adapter cuối; không chấm test)

1. **2b-s1:** dùng `scripts/probe_v1/score_probe.py` như C-s1, `--files dev,p2,p3,p4,pa,pb,p1b,p5`. Tên `2b-s1`.
2. **proxy_v1 (chỉ báo):** C-s1 chưa từng chấm trên proxy. Chấm **cả C-s1 (adapter cũ) lẫn 2b-s1** trên `data/proxy_v1/proxy_v1.jsonl`, cùng script và cách chấm. Không có cách chấm sẵn thì dùng `eval_base.py` với file dữ liệu khác, giữ mọi tham số mặc định.
3. **`val_kc_v2.jsonl` (chỉ báo):** chấm 2b-s1. C-s1 cũng chấm nếu tốn dưới 5 phút.
4. Checkpoint 25/50/75%: không chấm, chỉ lưu.

## 5. Báo cáo `results/train_2b/REPORT_2b_bac1.md`

Tính bằng script, so với `results/probe_v1/eval/*/C-s1/preds.jsonl`. **Áp đúng bảng dưới, không tự quyết bước tiếp.**

| Mã | Định nghĩa (dev 1.3, VI có dấu, câu kiểm chứng choice của ViFactCheck + ViWikiFC, trừ khi ghi khác) | Ngưỡng | C-s1 |
|---|---|---|---|
| T1 | P4 có dấu: tỉ lệ chọn NEI, từng nguồn. Báo thêm theo nhãn gốc của nhận định | ≥ 85% cả hai nguồn | 62,5 / 50,3 |
| T2 | Recall NEI gộp hai nguồn (n = 75) | ≥ 0,50; vùng xám 0,42–0,50; < 0,42 là "không chuyển" | 0,347 |
| T3a | Precision NEI gộp: số câu đoán NEI đúng / số câu đoán NEI | ≥ 0,70 | 0,862 (25/29) |
| T3b | Recall Xác nhận + Bác bỏ của ViWikiFC (n = 120) | ≥ 0,78 | 0,875 |
| T4 | VI macro có dấu (6 nguồn) | ≥ 79,65 | 81,65 |
| G1 | Macro không dấu | ≥ 62,47 | 63,97 |
| G2 | Macro EN | ≥ 87,20 | 88,20 |
| G3 | Pa ViWikiFC choice: mức tụt so với dev | tụt ≤ 10 điểm | −16,8 |
| G4 | P5 kiểm chứng (choice và Có/Không, có dấu): mức đổi so với dev | không kém C quá 3 điểm | — |
| G5 | SIB-200 và Belebele có dấu, từng nguồn | không kém C quá 5 điểm | 79,9 / 84,0 |

**Chỉ báo:**
- recall Xác nhận gộp (T3 cũ, C 0,859);
- recall từng nhãn của từng nguồn;
- điểm từng nguồn (có dấu, không dấu);
- pb, p1b;
- proxy kiểm chứng của C-s1 và 2b-s1;
- `val_kc_v2`;
- val loss;
- thời gian, VRAM, nhiệt độ tối đa.

**Kèm:** bootstrap theo cụm 2.000 lần cho hiệu VI macro (2b-s1 − C-s1), và McNemar cho T2/T3b.

## 6. Kết thúc

- Xong chấm thì gửi **"GPU XONG"** sang session `local_ad0853e1-dbf5-4579-b3e3-3a3093b693f9`, kèm bảng T/G. Điều phối sẽ trả GPU cho bên Ún.
- Rồi **dừng**. Không sang bậc 2.
