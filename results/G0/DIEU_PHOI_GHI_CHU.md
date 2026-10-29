# Ghi chú điều phối: bước 0 + G0 (worker, 28/10, khoảng 01:10)

> **02:50: G0 ĐÃ XONG.** Đã viết REPORT_buoc0.md và REPORT_G0.md, đã ghi PLAN_decisions, đã báo điều phối. GPU của G0: 6.495 giây.
> **Việc tiếp theo:** (A) FLA và (B) chẩn đoán teacher (mục "Việc giao thêm" bên dưới; brief `results/BRIEF_chan_doan_teacher.md`), ngân sách GPU riêng tối đa 1,5 giờ. Xong cả hai thì gửi "GPU XONG".
>
> **Hạ tầng dùng lại:**
> - hàng đợi `setup/g0_queue_c.sh` (có `gpu_idle` theo điều kiện công suất, `vram_ok`, kiểm nhiệt độ, lính gác tiến trình và khung 03:00–03:05 cho đo độ trễ);
> - `scripts/latency_g0.py` (có `--modes eager,eager_pad,graphs`);
> - lính gác nhiệt `setup/g0_temp_guard.sh` vẫn chạy;
> - tập con độ trễ `results/G0/rows/dev_lat_1000.jsonl`.
>
> **02:47, tiến độ (A)/(B):**
> - FLA chạy được sau khi có gcc (`results/G0/env/fla_check_sau_gcc.txt`: FLA_OK).
> - Hàng đợi D (`setup/g0_queue_d.sh '20 21 22' 0`, ngân sách GPU của brief tính từ 0) đang chạy:
>   - bước 20: kiểm đúng, chấm 2B-Base có FLA trên `results/G0/rows/dev_fla_600_ntok_null.jsonl` (600 dòng, 100 mỗi nguồn) → `results/G0/eval/fla/Qwen3.5-2B-Base_FLA`; phải so với `results/G0/eval/dev/Qwen3.5-2B-Base` (argmax 100%, lệch ≤ 1e-3). Không đạt thì KHÔNG dùng số độ trễ có FLA;
>   - bước 21: độ trễ 2B-Base có FLA trên tập con;
>   - bước 22: 4B có FLA, bỏ nếu (A) đã quá 1.500 giây.
> - Sau D: chạy hàng đợi E (`setup/g0_queue_e.sh 30 <giây GPU của A>`) = teacher: `scripts/teacher_diag/run_quyet_C.py` → `results/teacher_diag/quyet_C.jsonl` (chạy tiếp được nếu dừng giữa chừng).
> - Sau đó viết script phân tích: A, S, D90, entropy theo toàn bộ / text_lang / dataset ≥ 100 / nhóm dạng bài; mẫu 60 dòng VI bất đồng tự tin p > 0,9, seed 20261028, có cột tách cho nhãn và đáp án teacher, ghi `results/teacher_diag/mau_soat_mu.jsonl`.
> - Rồi viết `REPORT_chan_doan.md`, áp đúng bảng cách đọc của brief, và thêm mục "có FLA" vào REPORT_G0. Ghi PLAN_decisions, gửi "GPU XONG", DỪNG.
>
> **02:52:**
> - (A) xong: FLA chạy được, nhưng kiểm đúng KHÔNG đạt (argmax 591/600, lệch tối đa 0,062) → không đo độ trễ có FLA. Mục 4b đã thêm vào REPORT_G0. (A) dùng 243 giây GPU; đã báo điều phối.
> - (B): hàng đợi E (`g0_queue_e.sh 30 243`) đang đợi tới 03:06 rồi chạy `run_quyet_C.py`. Log từng lượt: `results/G0/logs/30_teacher_quyet_C_0.txt` (in tiến độ mỗi 1.000 dòng).
> - Khi xong: chạy `python scripts/teacher_diag/analyze_quyet_C.py` (ĐÃ VIẾT) → `chan_doan.json` và `mau_soat_mu.jsonl`; viết `results/teacher_diag/REPORT_chan_doan.md` (A/S/D90 theo toàn bộ, text_lang, dataset ≥ 100, nhóm VI; cách đọc áp đúng bảng của brief; manifest; nhiệt độ tối đa); ghi PLAN_decisions; gửi "GPU XONG"; DỪNG.
> - C có 29.524 dòng choice/noul (24.480 choice + 5.044 noul); tin gửi điều phối đã ghi nhầm 29.514.
>
> Với FLA: chạy trong `.venv-g0` (gcc đã có). Đầu tiên chạy lại phép kiểm kernel như trong `setup/g0_prepare.sh`.

Brief: `results/BRIEF_buoc0_G0.md`. Cách đọc đã chốt trước: `results/G0/CACH_DOC_chot_truoc.md`.

## Đã xong

**Bước 0**
- Báo cáo: `results/G0/REPORT_buoc0.md`.
- Chấm lại C trùng từng byte (sha `044EBE45…`).
- Đo VRAM: ngân sách 8.192 / 16.384 / 32.768 token → 5,85 / 8,37 / 13,41 GB.
- Kiểm chữ nhãn; `data/proxy_v1` sha `b003b0f6…`.

**G0c đã chấm** (dòng VI có dấu, macro; trong `results/G0/eval/g0c/`):

| Model | Khung VI | Khung EN | EN − VI [CI] |
|---|---|---|---|
| Qwen3-1.7B | 70,65 | 72,09 | +1,45 [−0,57; +3,43] |
| Qwen3.5-2B | 70,63 | 71,77 | +1,14 [−0,53; +2,79] |

Cả hai không đạt ngưỡng 2 điểm.

**FLA 0.6.0:** cài được trong `.venv-g0` nhưng KHÔNG chạy được. Lỗi: "Failed to find C compiler" (WSL không có gcc, sudo cần mật khẩu). Theo brief: ghi lỗi, không vá, đo không FLA. Mọi bước Qwen3.5 chạy trong `.venv`; `.venv-g0` không dùng.

**GPU đã dùng (hàng đợi A):** 1.242 giây.

**Revision:**
- Qwen3.5-2B-Base: `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (tải xong 01:04).
- Qwen3.5-4B @851bf6e806ef: đang tải. `setup/g0_prepare.sh` ghi `results/G0/env/models_g0.json` khi tải xong.

## Đang chạy hoặc còn lại

- **Hàng đợi B:** lần chạy hiện tại làm bước "3 8" (G0b 2B-Base, G0c 2B-Base khung EN), với `REVB=…`, `REV4=x`.
- **Lính gác nhiệt:** `setup/g0_temp_guard.sh` chạy nền. Vượt 85°C thì dừng, ghi `results/G0/logs/QUA_NHIET.flag`, đợi < 70°C; phải báo điều phối trước khi chạy lại.
- **Khi `models_g0.json` có:** chạy hàng đợi B cho các bước "4 9 11 12", với tham số `GPU_S` đã dùng; mỗi lần chạy, xem log để cộng dồn.
  - Bước 4: G0b 4B.
  - Bước 9: đo độ trễ C-s1.
  - Bước 11: đo độ trễ 2B-Base không FLA.
  - Bước 12: đo độ trễ 4B.
  - Lệnh: `wsl -e bash -lc "bash setup/g0_queue_b.sh '4 9 11 12' <GPU_S>"`.
  - Trước đó: kiểm nhiệt độ; các lượt đo độ trễ có lính gác tiến trình Windows và khung 03:00–03:05 (lượt đè lên thì bỏ, đo lại).
- **Sau đó:**
  - `python scripts/g0_analyze.py` → `results/G0/g0_tong_hop.json`;
  - viết `results/G0/REPORT_G0.md` (kết luận trước, bảng sau; đọc theo CACH_DOC_chot_truoc; ghi nhiệt độ tối đa mỗi lượt từ `logs/temp.log`);
  - ghi PLAN_decisions;
  - gửi "GPU XONG" cho Quản lý (`local_ad0853e1-dbf5-4579-b3e3-3a3093b693f9`), rồi DỪNG. Không sang bước 2a.

## Cập nhật khoảng 02:16

**G0b đã chấm xong:**
- 2B-Base: `results/G0/eval/dev/Qwen3.5-2B-Base`.
- 4B: `results/G0/eval/dev/Qwen3.5-4B`. Preds đủ 3.847 câu. Lượt đo thông lượng batch 16 của `eval_base` bị kẹt khoảng 25 phút (VRAM 16 GB, có lẽ tràn sang RAM qua WDDM); điều phối đồng ý dừng PID 4025 bằng SIGTERM lúc 01:57, nên 4B không có `timing.json`.

**G0c đã chấm đủ 3 model.**

**Điều kiện "GPU rảnh" đổi lúc 02:05** (điều phối chọn b): sau khi dừng PID 4025, chỉ số tải bị kẹt ở 100% dù công suất 27 W và VRAM 0. Điều kiện mới: không có tiến trình compute + VRAM < 500 MiB + công suất < 40 W liên tục 10 giây. Ghi vào REPORT_G0.

**Độ trễ C-s1 (bước 9):**
- Chốt so với 26/10: trung vị eager 31,80 ms so với 32,6 ms → trong 15%, đạt.
- Chỉ số tải về bình thường sau lượt đo (98% khi đo, 0% sau). Xung và công suất ghi trong `logs/9_C-s1_xung_0.txt`.
- Eager: trung vị 31,80, p90 47,27, p95 85,44, p99 135,81 ms.
- CUDA graphs **không qua cổng đúng**: argmax trùng 99,17% (32 câu lệch), xác suất lệch tối đa 0,194, 1.897 câu lệch > 1e-3. Theo quy tắc: không dùng số đo graphs, không nới cổng.
- Graphs còn chậm hơn eager ở câu dài vì pad tới nhóm độ dài: p90 nhóm 513–1.024 là 102 ms. Số tham khảo: trung vị 24,9, p90 49,1.

**GPU cộng dồn:** 4.930 giây. Đang chạy bước 11 (2B-Base không FLA), sau đó bước 12 (4B).

## Cập nhật 02:30 (điều phối 02:27)

Từ lượt đo độ trễ sau, dùng **tập con** `results/G0/rows/dev_lat_1000.jsonl`:
- 1.000 dòng, phân tầng theo nhóm độ dài `n_tokens` (token Qwen3), seed 20261012;
- theo nhóm: ≤128 → 247, 129–256 → 517, 257–512 → 155, 513–1.024 → 45, >1.024 → 36.

`latency_g0.py` nay ghi thêm `do_tre_tung_cau_lan_cuoi`.

**Cách chuyển:** `setup/g0_switch_to_c.sh` đợi "XONG 11" (2B-Base không FLA, toàn dev, đang chạy), rồi dừng hàng đợi B (bỏ bước 12, 4B toàn dev) và chạy `setup/g0_queue_c.sh "13 14"`:
- bước 13: C-s1 trên tập con, để so với toàn dev;
- bước 14: 4B trên tập con. Bỏ nếu GPU cộng dồn > 8.400 giây (140 phút).

Ngân sách G0 là 150 phút. Cổng đúng làm trên tập con.

**Trong báo cáo cần ghi:** số đo trên tập con; tập con gồm những dòng nào; với C-s1, tập con lệch bao nhiêu so với toàn dev.

## Cập nhật 02:36

**Bước 11 (2B-Base không FLA, toàn dev):** xong lúc 02:29, cộng dồn 5.774 giây. Sau đó hàng đợi C: bước 13 (C-s1 tập con) xong lúc 02:34, cộng dồn 6.050 giây. Bước 14 (4B tập con) bắt đầu 02:34.

**Chẩn đoán đường graphs, C-s1 trên tập con** (`latency_C-s1_tapcon.json`):

| Phép so | Câu lệch argmax | Lệch xác suất tối đa | Câu lệch > 1e-3 |
|---|---|---|---|
| Cổng gốc: eager không pad với graphs | 6 (99,4% trùng) | 0,123 | 490 |
| (i) eager không pad với eager có pad (sàn nhiễu theo hình dạng) | 3 | 0,069 | 84 |
| (ii) eager có pad với graphs (kiểm việc bắt graph) | **5** | **0,123** | **481** |

→ (ii) cũng lệch, nên theo khung điều phối: **graphs có lỗi riêng, giữ nhãn "THAM KHẢO, đường graphs chưa qua cổng đúng"**.

Câu lệch argmax rơi vào các nhóm T = 128, 192, 256, **không** tập trung ở câu pad nhiều (số token pad trung bình của câu lệch là 30, của toàn tập con là 43). Câu lệch > 1e-3 trải khắp các nhóm, nhiều nhất ở T = 128–256. Có thể do kernel cuBLAS / sdpa lúc bắt graph khác lúc chạy thường. Chưa sửa; ghi là việc để sau.

**Độ trễ tập con so với toàn dev, C-s1 eager:**

| Phân vị | Tập con | Toàn dev | Lệch |
|---|---|---|---|
| Trung vị | 30,06 | 31,80 | −5,5% |
| p90 | 45,80 | 47,27 | −3,1% |
| p95 | 84,94 | 85,44 | −0,6% |
| p99 | 134,38 | 135,81 | −1,1% |

Eager có pad: trung vị 28,86, p90 49,49. Graphs (tham khảo): trung vị 24,88, p90 49,15, p95 102,1.

**Sau bước 14:** `python scripts/g0_analyze.py` (script đọc `latency/latency_*.json`, kể cả tập con) → viết REPORT_G0 → ghi PLAN_decisions → báo điều phối "G0 xong" → chuyển sang (A) FLA và (B) teacher theo `results/BRIEF_chan_doan_teacher.md`. Gửi "GPU XONG" sau khi xong (A) và (B).

**Điều kiện "GPU rảnh" cho các hàng đợi sau:** dùng điều kiện công suất (`setup/g0_queue_b.sh`, hàm `gpu_idle`).

## Việc giao thêm (điều phối, khoảng 01:15)

**Làm SAU khi G0 xong.** Người dùng đã cài build-essential trong WSL (gcc 15.2) và duyệt "cho chạy chẩn đoán". Brief: `results/BRIEF_chan_doan_teacher.md`; đọc hết rồi làm đúng theo brief.

**(A) FLA**, khoảng 15 phút GPU:
1. Kiểm FLA chạy được (`.venv-g0`).
2. Kiểm đúng: argmax có FLA trùng 100% với không FLA.
3. Đo lại độ trễ 2B-Base có FLA, eager và CUDA graphs; còn thời gian thì đo thêm 4B.

**(B) Chẩn đoán teacher**, khoảng 1 giờ GPU:
- Quyet-Medium chấm khoảng 29.500 prompt choice/noul của C, chỉ bản có dấu, không đưa prompt dev hay test vào.
- Báo chỉ số A, S, D90 theo nhóm dạng bài.
- Xuất mẫu 60 dòng để điều phối soát mù.
- Cách đọc đã chốt trước trong brief.

**Quy tắc:**
- KHÔNG train.
- Nhiệt độ > 85°C thì dừng; mức dùng tuần > 90% thì dừng; không đo độ trễ trong 03:00–03:05.
- Ngân sách GPU cho brief này tối đa 1,5 giờ.
- Người dùng đang ngủ: có vấn đề thì báo điều phối.
- Xong (A) và (B) thì gửi "GPU XONG" rồi dừng.

**FLA trong G0 hiện tại:** điều phối dặn đo không FLA và ghi rõ trong REPORT_G0 rằng đây là **cận trên**.

## Quy tắc

Không train, không đụng test, không commit, không xóa, không ghi `config/` hay `data/bench_v1_3/`. Ngân sách GPU tối đa 2 giờ, 2,5 giờ thì dừng. Mức dùng 5 giờ hoặc tuần đạt 88% thì không bắt đầu việc mới, đạt 92% thì dừng; mức dùng tuần quá 90% thì dừng.
