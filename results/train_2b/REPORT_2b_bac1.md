# Báo cáo bậc 1 của 2b: train 2b-s1 (Qwen3-1.7B LoRA, seed 1)

Brief: `results/BRIEF_2b_bac1_train.md`. Người dùng duyệt ngày 28/10.

Mọi số dưới đây đều tính bằng script, không chép tay:
- Bảng T/G: `scripts/train_2b/bang_tg.py --m 2b-s1`, ghi ra `results/train_2b/bang_tg_2b-s1.json`.
- Mốc so sánh: `results/probe_v1/eval/*/C-s1/preds.jsonl`.
- Dev 1.3 (sha c6a9a75d…). Không chấm test. Không chấm checkpoint 25/50/75%, chỉ lưu (adapter_step494 / 987 / 1480).

**Áp đúng bảng dưới. Worker không quyết bước tiếp.**

## 1. Bảng T/G

| Mã | Định nghĩa | Ngưỡng | C-s1 | 2b-s1 | Kết quả |
|---|---|---|---|---|---|
| T1 | P4 có dấu: tỉ lệ chọn NEI (ViFactCheck / ViWikiFC) | ≥ 85% cả hai | 62,5 / 50,3 | **100,0 / 100,0** | đạt |
| T2 | Recall NEI gộp (n = 75) | ≥ 0,50 (vùng xám 0,42–0,50) | 0,347 | **0,800** | đạt |
| T3a | Precision NEI gộp | ≥ 0,70 | 0,862 (25/29) | **0,772 (61/79)** | đạt |
| T3b | Recall Xác nhận + Bác bỏ ViWikiFC (n = 120) | ≥ 0,78 | 0,875 | **0,800** | đạt |
| T4 | VI macro có dấu, 6 nguồn | ≥ 79,65 | 81,65 | **84,84** | đạt |
| G1 | Macro không dấu | ≥ 62,47 | 63,97 | **63,35** | đạt |
| G2 | Macro EN | ≥ 87,20 | 88,20 | **88,80** | đạt |
| G3 | Pa ViWikiFC choice: mức đổi so với dev (n = 161) | tụt ≤ 10 điểm | −16,8 [−23,9; −9,1] | **−4,3 [−9,2; +0,6]** | đạt |
| G4 | P5 kiểm chứng: mức đổi so với dev (n = 443) | ≥ mức của C − 3 = −5,3 | −2,3 | **−1,8** | đạt |
| G5 | SIB-200 / Belebele có dấu | ≥ C − 5 điểm | 79,9 / 84,0 | **81,9 / 83,0** | đạt |

**Kết quả: đạt cả 10 dòng (T1, T2, T3a, T3b, T4, G1–G5).**

**Thống kê kèm:**
- **T4:** hiệu VI macro (2b-s1 − C-s1) = **+3,19 điểm**. CI 95% từ bootstrap theo cụm 2.000 lần là **[+1,02; +5,34]**, rút theo từng nguồn rồi lấy trung bình đều; seed lấy từ `common.SEED`.
- **McNemar T2:** C đúng, 2b sai: 1; C sai, 2b đúng: 35; p < 0,0001.
- **McNemar T3b:** C đúng, 2b sai: 15; C sai, 2b đúng: 6; p = 0,078.
- **Cách tính CI ở G3:** bootstrap theo cụm 1.000 lần, như các lần trước.

## 2. Chỉ báo

### T1 theo nhãn gốc của nhận định (P4 có dấu, tỉ lệ chọn NEI, %)

| Nguồn \| nhãn gốc | C-s1 | 2b-s1 |
|---|---|---|
| ViFactCheck \| Xác nhận | 61,1 | 100,0 |
| ViFactCheck \| Bác bỏ | 50,0 | 100,0 |
| ViFactCheck \| NEI | 73,5 | 100,0 |
| ViWikiFC \| Xác nhận | 45,7 | 100,0 |
| ViWikiFC \| Bác bỏ | 58,1 | 100,0 |
| ViWikiFC \| NEI | 41,5 | 100,0 |

### Recall từng nhãn, dev có dấu, dòng choice

| Nguồn \| nhãn | C-s1 | 2b-s1 |
|---|---|---|
| ViFactCheck \| Xác nhận | 0,833 | 0,889 |
| ViFactCheck \| Bác bỏ | 0,857 | 0,929 |
| ViFactCheck \| NEI | 0,206 | 0,824 |
| ViWikiFC \| Xác nhận | 0,870 | 0,848 |
| ViWikiFC \| Bác bỏ | 0,878 | 0,770 |
| ViWikiFC \| NEI | 0,463 | 0,780 |

- **Recall Xác nhận gộp (T3 cũ):** C 0,859, 2b 0,859.
- **Số câu đoán NEI:** tăng từ 29 lên 79. Số câu đoán NEI mà sai tăng từ 4 lên 18.

### Điểm từng nguồn, dev (%)

| Nguồn | C có dấu | 2b có dấu | C không dấu | 2b không dấu |
|---|---|---|---|---|
| belebele | 84,0 | 83,0 | 59,5 | 61,6 |
| sib200 | 79,9 | 81,9 | 62,2 | 58,9 |
| massive | 88,1 | 89,5 | 74,1 | 71,1 |
| xcopa | 85,0 | 82,0 | 59,0 | 52,0 |
| vifactcheck | 72,2 | 89,9 | 66,5 | 76,6 |
| viwikifc | 80,7 | 82,8 | 62,5 | 60,0 |

### Các bộ khác

| Bộ | C-s1 | 2b-s1 |
|---|---|---|
| pb, acc (mức đổi so với dev), n = 593 | 87,0 (+3,0) | 88,7 (+3,0) |
| p1b, acc (mức đổi so với dev), n = 482 | 65,4 (+1,9) | 71,8 (+1,5) |
| proxy_v1, dòng kiểm chứng ViFactCheck + ViWikiFC (n = 600) | 71,2 | 74,0 |
| val_kc_v2 (n = 300) | 82,0 | 98,0 |

Cách chấm hai bộ cuối:
- **proxy_v1:** `eval_base.py`, tham số mặc định, file `data/proxy_v1/proxy_v1.jsonl`. Preds nằm ở `results/train_2b/eval/proxy/<tên>/`.
- **val_kc_v2:** chấm trên `results/train_2b/eval/_rows/val_kc_v2_eval.jsonl`. Preds nằm ở `results/train_2b/eval/val_kc_v2/<tên>/`.

### Train

| | C-s1 | 2b-s1 |
|---|---|---|
| val loss (val_v0) bắt đầu / epoch 1 / epoch 2 | 7,018 / 0,758 / 0,757 | 7,018 / 0,760 / 0,772 |
| Số bước | — | 1.974 (micro 8, effective 32, 2 epoch, 31.596 dòng) |
| Thời gian train (wall) | 10.625 giây | 12.612 giây |
| VRAM đỉnh | 10,25 GB | 10,25 GB |

**Chi tiết train 2b-s1:**
- **letter_mass:** bước 0 là 0,786 (42/200 dưới 0,5). Từ bước 127 trở đi luôn ≥ 0,997, 0/200 dưới 0,5. Cổng ở bước 254 đạt (0,9985).
- **Đo VRAM:** `--vram_probe 1752`, không phải 1664 như brief. Lý do: 1752 là độ dài prompt thật dài nhất; điều phối đã đồng ý. Kết quả: reserved 8,27 GB, allocated 7,64 GB, nên dùng micro_batch 8. Sai lệch này đã ghi trong `manifest_train.json`.
- **Loss train** ở bước 1970: 0,445.

**Nhiệt độ và GPU:**
- **Nhiệt độ GPU:** cao nhất **79°C** (11:13), lấy từ 538 lần đo trong `results/G0/logs/temp.log` khung 10:55–15:06. Lúc chấm cao nhất 78°C. Chưa lần nào ≥ 80°C. Không có cờ quá nhiệt.
- **Sửa số nhiệt độ đã báo trước:** tin nhắn trước tôi báo 81°C, nhưng số đó lấy trên cả file log, gồm cả trước lượt này.
- **GPU cộng dồn: 14.981 / 18.000 giây** (4,16 giờ), gồm:

| Bước | Giây GPU |
|---|---|
| Đo VRAM | 76 |
| Train | 12.679 |
| score_probe | 1.583 |
| proxy 2b-s1 / C-s1 | 218 / 220 |
| val_kc_v2 2b-s1 / C-s1 | 102 / 103 |

## 3. File

- Adapter, log, `summary.json`, `manifest_train.json`, `hang_doi.log`: `results/train_2b/train/2b-Qwen3-1.7B-s1/`.
- Preds probe: `results/probe_v1/eval/<phép>/2b-s1/preds.jsonl`.
- Log chấm: `results/probe_v1/eval/logs/2b-s1.txt`, `results/train_2b/eval/logs/`.
- Bảng: `results/train_2b/bang_tg_2b-s1.json`.
