# BRIEF 2b bậc 2: train 2b-s2 và chấm (29/10/2026)

**Người dùng duyệt** (chat 29/10, khoảng 01:20): "được, chạy 2b-s2 đi, nhớ giữ nhiệt độ dưới ngưỡng".

Luật đọc nằm ở `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md`:
- mục 9: các luật T/G;
- mục 10: chỉ chạy 2b-s2, chỉ báo a–d;
- mục 11: dự đoán và cách đọc, đã ghi trước khi chạy.

Brief bậc 1 (`results/BRIEF_2b_bac1_train.md`) vẫn áp dụng, trừ các điểm đổi dưới đây.

## 0. Ngưỡng dừng (bắt buộc)

- Nhiệt độ GPU > **85°C** → dừng train ngay. Giữ lính gác nhiệt chạy nền suốt lần train. Từ 83°C trở lên thì báo điều phối.
- Mức dùng tuần > 90% hoặc Fable > 90% → dừng. Điều phối theo dõi.
  - **Cập nhật 29/10, khoảng 02:00:** người dùng hạ ngưỡng xuống **35% usage tuần**. Điều phối áp ngưỡng này cho cả Fable. Ngưỡng nhiệt vẫn là 85°C.
  - Nếu điều phối nhắn "DỪNG USAGE": dừng an toàn, rồi báo lại.
- Không đụng test. Không sửa `config/` hay `data/bench_v1_3/`. Không xóa, không commit.
- **Chỉ train đúng 1 lần, 2b-s2.** Không chạy C-s2. Không train gì khác.
- Ngân sách GPU tối đa **5 giờ**. Vượt thì dừng và báo.

## 1. Nhận GPU

- Chỉ bắt đầu khi điều phối báo **"GPU SẴN"**.
- Kiểm GPU rảnh như bậc 1:
  - không có tiến trình nào;
  - VRAM < 500 MiB;
  - công suất < 40 W liên tục 10 giây.

## 2. Train 2b-s2

- **Dữ liệu:** `results/train_2b/train_2b_v3.jsonl`. Kiểm sha `ea604a01…` trước khi chạy.
- **Cấu hình:** chép đúng lần chạy 2b-s1 (`scripts/train_2b/run_2b_bac1.sh` và manifest của s1). **Chỉ đổi seed thành 2 và thư mục đầu ra.**
  - Giữ micro-batch 8 và `--token_budget 0` như s1. Không cần đo VRAM lại vì dữ liệu giống s1; đỉnh VRAM của s1 là 10,25 GB.
  - Biến môi trường giữ như s1, gồm cả `PYTHONHASHSEED=0`.
  - `--save_fracs 0.25,0.5,0.75`.
  - Trần state 2.048. Kiểm không dòng nào bị cắt.
- **Đầu ra:** `results/train_2b/train/2b-Qwen3-1.7B-s2/`, gồm `summary.json` ghi thời gian, VRAM đỉnh, nhiệt độ tối đa và số bước.
- Cổng letter_mass trượt thì dừng và báo.

## 3. Chấm (adapter cuối; không chấm test)

1. `scripts/probe_v1/score_probe.py --files dev,p2,p3,p4,pa,pb,p1b,p5`, đặt tên `2b-s2`.
2. `data/proxy_v1/proxy_v1.jsonl`, chấm 2b-s2 theo đúng cách đã chấm 2b-s1.
3. `val_kc_v2.jsonl`, chấm 2b-s2.
4. Các checkpoint 25/50/75% chỉ lưu, không chấm.

## 4. Báo cáo `results/train_2b/REPORT_2b_bac2.md`

Tính bằng script, từ các file preds sau:
- `results/probe_v1/eval/*/{C-s1,2b-s1,2b-s2}/preds.jsonl`;
- `results/train_2b/eval/{proxy,val_kc_v2}/`.

**Áp đúng luật ở mục 9–11. Không tự quyết bước tiếp.**

1. **Bảng T1–T4, G1–G5** (định nghĩa và ngưỡng như brief bậc 1, mục 5). Mỗi luật có ba cột:
   - 2b-s2 so với C-s1;
   - **trung bình 2b-s1 và 2b-s2** so với C-s1;
   - 2b-s1, để đối chiếu.
   
   Ghi rõ đạt hay trượt ở cả hai cột đầu.
2. **Biên theo số câu** cho T3b (sàn 94/120), T3a và G1, của s2 và của trung bình hai seed.
3. **Chỉ báo mục 10:**
   - (a) recall "Có" của câu Có/Không, theo từng nguồn, có dấu và không dấu;
   - (b) tỉ lệ đoán NEI trên ViWikiFC không dấu (s1: 91/161);
   - (c) recall Bác bỏ ViWikiFC có dấu (s1: 0,770; C: 0,878);
   - (d) |2b-s2 − 2b-s1| trên VI macro, macro không dấu, EN và từng nguồn kiểm chứng.
4. **Nhiễu seed:**
   - với từng nguồn và từng chế độ (có dấu, không dấu, EN): số câu s1 và s2 trả lời khác nhau, và hiệu điểm;
   - bootstrap theo cụm 2.000 lần cho hai hiệu (2b-s2 − C-s1) và (trung bình 2b − C-s1) trên VI macro;
   - McNemar s2 so với C-s1 cho T2 và T3b.
5. **Chỉ báo khác như bậc 1:**
   - pb, p1b;
   - proxy, có dấu và không dấu báo riêng;
   - val_kc_v2;
   - val loss;
   - thời gian, VRAM, nhiệt độ tối đa.

## 5. Kết thúc

- Chấm xong thì gửi **"GPU XONG"** sang session `local_ad0853e1-dbf5-4579-b3e3-3a3093b693f9`, kèm bảng T/G hai cột và bốn chỉ báo a–d. Điều phối sẽ trả GPU cho Ún.
- Sau đó **dừng**. Không chạy C-s2, không sửa dữ liệu.
