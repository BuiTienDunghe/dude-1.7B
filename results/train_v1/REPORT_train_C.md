# Train thử bản C: số liệu trên dev 1.3

Ngày 26/10/2026.

**Phạm vi:**
- Train **thử**, không phải train chính thức (2 seed).
- Người dùng cho phép ngày 26/10 ("Từ 1.7B gốc"); điều phối ghi PLAN_decisions.
- Chỉ dùng dev 1.3; **không chạy gì trên test**. Không commit, không xóa file, không ghi `data/` hay `config/`.

**Báo cáo này chỉ trình bày số, không kết luận kịch bản K1–K9.** Điều phối và Fable sẽ đọc.

**Ký hiệu:**

| Mã | Model |
|---|---|
| M0 | Qwen3-1.7B gốc |
| M1 | model dự án (train_v0, full_vocab s1) |
| C@x% | adapter bản C ở x% số bước; C@100% = bản cuối |
| R1 | decider-2b, ánh xạ text |
| R2 | Quyet-1.0-Medium |

## 1. Lượt train

- **Công thức giống M1** (đối chiếu `results/ft_loss/train/Qwen3-1.7B-s1/summary.json`):
  - Qwen/Qwen3-1.7B @70d244cc; LoRA r16/α32/dropout 0,05 trên 7 lớp proj;
  - lr 2e-4, batch hiệu dụng 32 (micro 8 × accum 4), warmup 3% + cosine;
  - 2 epoch, seed 1, loss full_vocab;
  - strip_p 0, như M1.
- **Chỉ đổi:**
  - `--train_file results/train_v1/components/C.jsonl` (31.034 dòng, sha `005ca3d5…`);
  - thêm `--save_fracs 0.25,0.5,0.75`.
  - Val giữ `data/val_v0.jsonl` (sha `4377e710…`) như M1, để val_loss so được với nhau.
- **Cổng letter_mass 0,5 ở bước 250** (bước tuyệt đối, như M1), đo mỗi 125 bước. Tham số tương ứng: `--monitor_frac 0.0645 --gate_step_frac 0.129`.
- **Kết quả:**
  - 1.938 bước, warmup 58 bước.
  - Letter_mass ở bước 125 là 0,9975, ở bước 250 là 0,9977 → **qua cổng**.
  - val_loss: 7,018 (bước 0) → 0,758 (hết epoch 1) → **0,757** (hết epoch 2). M1 tương ứng: 0,756 → 0,776.
  - Thời gian 10.625 giây (khoảng 2,95 giờ). VRAM đỉnh 10,25 GB (dành sẵn 12,9 GB).
  - Adapter: bước 484, 969, 1454 và adapter cuối.
- **File** (trong `results/train_v1/train/C-Qwen3-1.7B-s1/`): `manifest_train.json` (lệnh, sha train/val/script), `train_log.jsonl`, `summary.json`, `stdout.txt`.

## 2. Chỉ tiêu đặt trước

Nguồn chỉ tiêu: `results/BRIEF_train_chinh_thuc.md` mục 2.

**Cách tính:**
- Điểm là **trung bình đều 6 nguồn** (macro), như bảng mục 1 của brief. Điểm trên mọi câu ghi trong ngoặc.
- CI là bootstrap 95% theo cụm, rút lại riêng từng nguồn, 1.000 lần, seed cố định cho từng phép so.
- Script: `scripts/train_v1/trend_C.py`; số trong `results/train_v1/trend_C.json`.

| Chỉ tiêu | M1 | C@100% | Chênh so với M1 [CI 95%] | Đạt? |
|---|---|---|---|---|
| VI có dấu ≥ 84,0 | 79,88 (80,84) | **81,65** (82,10) | +1,77 [−0,28; +3,66] | **Không**: dưới 84,0, và CI chứa 0 |
| VI không dấu ≥ 61,0 | 62,37 (62,80) | 63,97 (64,41) | +1,60 [−1,21; +4,30] | Đạt |
| EN ≥ 87,0 | 87,95 (87,34) | 88,20 (87,03) | +0,25 [−1,20; +1,71] | Đạt |
| Trung vị độ trễ batch 1 trên dev ≤ 40 ms | 32,6–32,7 (đo lại) | **32,6–32,7** (đo lại, GPU trống); lần đầu 42,1 khi GPU dùng chung với game | ≈ 0 | **Đạt** (xem dưới) |

**Theo nguồn, VI có dấu:**

| Nguồn | M1 | C@100% | Chênh [CI 95%] | p McNemar | Yêu cầu | Kết quả |
|---|---|---|---|---|---|---|
| ViFactCheck | 72,8 | 72,2 | −0,6 [−5,6; +3,9] | 1,0 | tăng, p < 0,05 | Không |
| MASSIVE | 86,7 | 88,1 | +1,4 [−2,4; +5,1] | 0,58 | tăng, p < 0,05 | Không |
| Belebele | 82,7 | 84,0 | +1,4 [−2,3; +5,1] | 0,58 | không tụt | Không tụt |
| ViWikiFC | 76,5 | 80,7 | +4,2 [+1,0; +8,3] | 0,050 | không tụt | Không tụt |
| XCOPA | 78,0 | 85,0 | +7,0 [−1,0; +14,0] | 0,12 | không tụt | Không tụt |
| SIB-200 | 82,6 | 79,9 | −2,7 [−6,0; +0,7] | 0,20 | (không có yêu cầu riêng) | — |

**Độ trễ: lần đầu đo khi GPU dùng chung với game; đã đo lại trên GPU trống.**

*Lần đầu, giữ nguyên số, KHÔNG dùng để so chỉ tiêu:*
- Đo cùng lúc chấm: M0 33,5 ms, M1 33,5, C@25% 34,4, C@75% 42,4, C@100% 42,1.
- Lúc chấm C@75% và C@100%, nvidia-smi phía Windows cho thấy "League of Legends.exe" đang dùng GPU (875 MiB, tải 27% khi không có tiến trình chấm nào). Nói cách khác, **GPU dùng chung với game**.
- Thông lượng batch 16 cũng giảm (M1 34,6 câu/giây, C@100% 29,4), dù C cùng kiến trúc và cùng độ dài prompt.

*Đo lại ngày 26/10, sau khi người dùng tắt game:*
- Chạy `scripts/train_v1/run_latency_C.sh`, xen kẽ M1, C, M1, C, rồi thêm C@25%.
- Trước lượt 1, GPU: 0 MiB, 0%, không ứng dụng nào. nvidia-smi trước mỗi lượt lưu ở `results/train_v1/do_tre_lai/nvidia_smi_lan*.txt`.
- Chỉ đo thời gian, bằng chính `eval_base.py`; ghi vào `results/train_v1/do_tre_lai/`, không thay preds.
- Preds của mỗi lượt trùng 3.847/3.847 với preds đã chấm của đúng model đó.

| Lượt | Model | Trung vị batch 1 (ms) | p90 (ms) | Thông lượng batch 16 (câu/giây) |
|---|---|---|---|---|
| 1 | M1 | 32,60 | 47,4 | 36,6 |
| 2 | C@100% | 32,68 | 47,0 | 36,6 |
| 3 | M1 | 32,74 | 47,2 | 36,6 |
| 4 | C@100% | 32,61 | 47,4 | 36,6 |
| 5 | C@25% | 32,42 | 46,9 | 36,5 |

Trong lúc đo, phía Windows có lúc hiện "PID 4, Insufficient Permissions". Đó là tiến trình System của Windows, ứng với chính lượt đo trong WSL; bảng tiến trình đầy đủ không có ứng dụng nào khác.

## 3. Đường xu hướng 4 mốc (dev 1.3, macro 6 nguồn)

| Mốc | Bước | VI có dấu | VI không dấu | EN | Chênh VI có dấu so với M1 [CI] | Độ trễ (ms) |
|---|---|---|---|---|---|---|
| M1 | — | 79,88 | 62,37 | 87,95 | — | 33,5 |
| C@25% | 484 | 81,07 | 61,35 | 88,30 | +1,19 [−0,99; +3,30] | 34,4 |
| C@50% | 969 | 80,17 | 64,98 | 88,20 | +0,29 [−1,73; +2,45] | — (thiếu timing.json) |
| C@75% | 1454 | 81,38 | 62,98 | 87,70 | +1,50 [−0,46; +3,54] | 42,4 (GPU dùng chung với game) |
| C@100% | 1938 | **81,65** | 63,97 | 88,20 | +1,77 [−0,28; +3,66] | 42,1 (GPU dùng chung với game); **đo lại 32,6–32,7** |

Cột độ trễ là số đo cùng lúc chấm. Số đo lại trên GPU trống (M1 32,6–32,7; C@100% 32,6–32,7; C@25% 32,4) ở mục 2.

**Theo nguồn, VI có dấu, qua các mốc:**

| Nguồn | M1 | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Belebele | 82,7 | 85,0 | 83,0 | 82,7 | 84,0 |
| SIB-200 | 82,6 | 84,6 | 79,9 | 80,3 | 79,9 |
| MASSIVE | 86,7 | 88,8 | 88,8 | 88,8 | 88,1 |
| XCOPA | 78,0 | 80,0 | 82,0 | 85,0 | 85,0 |
| ViFactCheck | 72,8 | 71,5 | 71,5 | 70,9 | 72,2 |
| ViWikiFC | 76,5 | 76,5 | 75,8 | 80,7 | 80,7 |

**Ghi chú về C@50%:**
- Khi đổi thứ tự chấm theo lệnh điều phối, tôi dừng script hàng đợi. Tiến trình con bị dừng theo, ở bước đo thông lượng (batch 16) chạy sau lượt chấm chính.
- Preds của lượt chấm chính (batch 1) đã ghi đủ 3.847/3.847 câu trước đó nên dùng được; chỉ thiếu `timing.json`.
- Theo quy tắc "đã chấm xong thì không chấm lại", tôi không chấm lại mốc này.

## 4. Lát cắt (mục 2 của brief)

Accuracy trên mọi câu, %. CI 95% bootstrap theo cụm; McNemar chính xác theo cặp câu.

| Lát | Nhóm | n | M0 | M1 | C@100% | C − M1 [CI] | p | R1 | R2 |
|---|---|---|---|---|---|---|---|---|---|
| Toàn dev | | 3.847 | 65,9 | 75,8 | **76,8** | +1,0 [−0,3; +2,2] | 0,10 | 73,5 | 81,3 |
| Ngôn ngữ | VI có dấu | 1.430 | 70,7 | 80,8 | **82,1** | +1,3 [−0,5; +3,0] | 0,17 | 86,9 | 89,2 |
| | VI không dấu | 1.430 | 48,3 | 62,8 | **64,4** | +1,6 [−0,7; +4,0] | 0,20 | 48,7 | 65,0 |
| | EN | 987 | 84,3 | 87,3 | **87,0** | −0,3 [−1,9; +1,4] | 0,81 | 90,0 | 93,3 |
| VI có dấu | nguồn gốc VI | 443 | 59,4 | 75,2 | **77,6** | +2,5 [−0,7; +5,6] | 0,14 | 83,1 | 84,0 |
| | nguồn dịch | 987 | 75,8 | 83,4 | **84,1** | +0,7 [−1,4; +2,7] | 0,56 | 88,5 | 91,6 |
| Kiểu câu | chọn một | 3.443 | 65,1 | 75,0 | **76,5** | +1,5 [+0,1; +2,6] | 0,024 | 72,7 | 81,9 |
| | Có/Không | 404 | 72,8 | 82,4 | **79,5** | −3,0 [−6,3; +0,5] | 0,11 | 79,7 | 75,5 |
| Số lựa chọn | 2 | 704 | 72,7 | 79,4 | **79,4** | +0,0 [−2,7; +2,8] | 1,0 | 78,7 | 79,5 |
| | 3–4 | 1.364 | 57,1 | 70,7 | **71,3** | +0,7 [−1,5; +2,9] | 0,58 | 69,1 | 78,8 |
| | ≥ 5 | 1.779 | 69,9 | 78,3 | **79,9** | +1,6 [+0,1; +3,3] | 0,057 | 74,7 | 83,8 |
| Độ dài văn bản | ngắn | 1.387 | 66,8 | 76,5 | **75,6** | −0,9 [−2,7; +0,9] | 0,40 | 72,4 | 81,2 |
| | vừa | 1.256 | 65,2 | 75,4 | **76,8** | +1,4 [−0,7; +3,4] | 0,19 | 74,2 | 80,2 |
| | dài | 1.204 | 65,5 | 75,4 | **78,2** | +2,7 [+0,3; +5,1] | 0,021 | 73,9 | 82,4 |
| Đã sửa ở 1.2 (dòng VI) | đã sửa | 598 | 62,0 | 70,7 | **71,4** | +0,7 [−2,5; +3,8] | 0,75 | 68,6 | 78,4 |
| | giữ nguyên | 2.262 | 58,8 | 72,1 | **73,7** | +1,6 [−0,0; +3,2] | 0,059 | 67,5 | 76,8 |
| Cờ mơ hồ | có | 74 | 18,9 | 16,2 | **13,5** | −2,7 [−9,6; +5,3] | 0,73 | 13,5 | 14,9 |
| | không | 3.773 | 66,8 | 77,0 | **78,0** | +1,1 [−0,2; +2,3] | 0,084 | 74,6 | 82,6 |

**Ma trận chuyển M1 → C@100% (toàn dev):**

| | → đúng | → sai |
|---|---|---|
| đúng | 2.676 | 240 |
| sai | 278 | 653 |

Theo nguồn: `phan_tich_C/C-s1_vs_M1.md`.

**Trùng lỗi với model tham chiếu:**

| Cặp | Jaccard tập câu sai | P(R sai \| M sai) | Tỉ lệ sai của M |
|---|---|---|---|
| M1 với decider | 0,382 | 0,579 | 24,2% |
| C với decider | 0,365 | 0,573 | 23,2% |
| M1 với Quyet | 0,350 | 0,460 | 24,2% |
| C với Quyet | 0,361 | 0,479 | 23,2% |

**Hiệu chỉnh (P7, toàn dev):**

| Model | Brier | ECE | Tự tin trung bình khi sai | Tự tin trung bình khi đúng |
|---|---|---|---|---|
| M1 | 0,350 | 0,087 | 0,699 | 0,891 |
| C@100% | 0,344 | 0,088 | 0,714 | 0,899 |

## 5. Phép thử (mục 3 và "Cập nhật 26/10" của brief)

### P2: đảo thứ tự lựa chọn

| Phạm vi | M1 | C@100% |
|---|---|---|
| Toàn dev: chọn cùng đáp án | 88,7% | 89,6% |
| Toàn dev: accuracy khi đảo | 74,9 | 76,2 |

Theo nguồn, VI có dấu (chọn cùng đáp án, M1 → C):

| Nguồn | M1 → C |
|---|---|
| Belebele | 90,1 → 90,8 |
| MASSIVE | 90,1 → 92,9 |
| SIB-200 | 92,6 → 92,6 |
| ViFactCheck choice | 93,8 → 91,2 |
| ViWikiFC choice | 93,8 → 96,9 |
| XCOPA | 82,0 → 90,0 |

### P3: bỏ văn bản

Accuracy %, kèm mốc "đoán mò / luôn chọn đáp án hay gặp nhất":

| Nhóm | Mốc | M1 | C@100% [CI] |
|---|---|---|---|
| Belebele EN | 25,0 / 26,9 | 45,6 | 43,5 [38,1; 49,6] |
| Belebele VI có dấu | 25,0 / 26,9 | 38,8 | 42,5 [36,4; 48,6] |
| Belebele VI không dấu | 25,0 / 26,9 | 31,3 | 29,9 [25,1; 34,9] |
| ViFactCheck choice, có dấu | 33,3 / 42,5 | 46,2 | 43,8 [32,9; 54,4] |
| ViFactCheck choice, không dấu | 33,3 / 42,5 | 40,0 | 48,8 [37,2; 60,0] |
| ViWikiFC choice, có dấu | 33,3 / 46,6 | 31,1 | 34,8 [27,0; 42,8] |
| ViWikiFC choice, không dấu | 33,3 / 46,6 | 34,8 | 35,4 [28,6; 44,3] |
| ViFactCheck Có/Không, có dấu / không dấu | 50,0 / 64,1 | 64,1 / 60,3 | 57,7 / 47,4 |
| ViWikiFC Có/Không, có dấu / không dấu | 50,0 / 62,1 | 60,5 / 58,1 | 62,9 / 53,2 |

**(d) Phân bố nhãn dự đoán ở P3 kiểm chứng choice** (bỏ văn bản):

| Nhóm | M1 | C@100% |
|---|---|---|
| ViFactCheck, có dấu | NEI 71,2 / bác bỏ 8,8 / xác nhận 20,0 | NEI 76,2 / bác bỏ 11,2 / xác nhận 12,5 |
| ViWikiFC, có dấu | NEI 72,7 / bác bỏ 17,4 / xác nhận 9,9 | NEI 63,4 / bác bỏ 28,6 / xác nhận 8,1 |
| ViFactCheck, không dấu | NEI 32,5 / bác bỏ 67,5 | NEI 47,5 / bác bỏ 52,5 |
| ViWikiFC, không dấu | NEI 42,2 / bác bỏ 57,8 | NEI 47,8 / bác bỏ 51,6 / xác nhận 0,6 |

Phân bố đáp án thật: ViFactCheck NEI 42,5 / bác bỏ 35,0 / xác nhận 22,5; ViWikiFC NEI 25,5 / bác bỏ 46,6 / xác nhận 28,0.

### P4: văn bản không liên quan, kiểu choice

Tỉ lệ chọn "không đủ thông tin". Mốc thiên vị: trên câu dev gốc của cùng nhận định, model chọn nhãn này bao nhiêu khi đáp án thật là nhãn khác.

| Nhóm | M1 (mốc thiên vị) | C@100% [CI] (mốc thiên vị) |
|---|---|---|
| ViFactCheck, có dấu | 91,2 (2,2) | **62,5** [52,5; 72,7] (2,2) |
| ViWikiFC, có dấu | 88,2 (5,0) | **50,3** [43,2; 57,4] (2,5) |
| ViFactCheck, không dấu | 93,8 (6,5) | 87,5 [80,2; 94,7] (8,7) |
| ViWikiFC, không dấu | 91,9 (12,5) | 89,4 [84,0; 94,2] (19,2) |

**P4 câu Có/Không bị loại khỏi kết luận** (điều phối chốt 26/10): M1 và C đều đạt 99–100%, không phân biệt được.

### P5 và P1′

- P5 đổi câu hỏi. P1′ dùng chính câu hỏi P5 và đổi thêm chữ nhãn; chỉ có ở kiểm chứng choice.
- Chênh tính theo cặp câu.
- Kiểm chứng gộp hai nguồn.

| Khung | M1: dev → P5 (P5 − dev) | C: dev → P5 (P5 − dev) | M1: P1′ (P1′ − P5) | C: P1′ (P1′ − P5) |
|---|---|---|---|---|
| Kiểm chứng choice, có dấu | 67,6 → 70,5 (+2,9 [−0,9; +6,6]) | 70,5 → 68,0 (−2,5 [−5,2; +0,4]) | 73,4 (+2,9 [−2,0; +7,9], p 0,32) | 72,6 (**+4,6** [+1,3; +8,1], p 0,019) |
| Kiểm chứng choice, không dấu | 53,5 → 53,1 (−0,4) | 56,4 → 52,7 (−3,7 [−9,4; +2,4]) | 60,2 (+7,0 [+0,4; +14,0]) | 58,1 (+5,4 [−0,9; +11,6]) |
| Kiểm chứng Có/Không, có dấu | 84,2 → 84,2 (0,0) | 86,1 → 84,2 (−2,0 [−4,6; 0,0]) | — | — |
| Kiểm chứng Có/Không, không dấu | 80,7 → 80,2 (−0,5) | 72,8 → 75,7 (+3,0 [−0,9; +6,5]) | — | — |
| SIB-200, có dấu | 82,6 → 82,9 (+0,3) | 79,9 → 80,9 (+1,0 [−1,0; +3,0]) | — | — |
| SIB-200, không dấu | 60,2 → 63,2 (+3,0) | 62,2 → 62,5 (+0,3) | — | — |
| MASSIVE, có dấu | 86,7 → 86,4 (−0,3) | 88,1 → 87,4 (−0,7 [−2,4; +0,7]) | — | — |
| MASSIVE, không dấu | 66,3 → 66,0 (−0,3) | 74,2 → 74,5 (+0,3) | — | — |

### Pa: lựa chọn bỏ dấu, văn bản và câu hỏi giữ dấu

Mỗi ô: dev → Pa (chênh [CI]); mốc là accuracy trên dòng không dấu toàn phần của cùng cặp.

| Nhóm (VI có dấu) | M1 | C@100% |
|---|---|---|
| Toàn bộ | 80,6 → 74,2 (−6,4 [−8,6; −4,1]); mốc 64,4 | 81,3 → 72,3 (**−9,0** [−11,6; −6,7]); mốc 66,3 |
| SIB-200 | 82,6 → 71,6 (−11,0 [−16,4; −6,0]) | 79,9 → 74,2 (−5,7 [−9,7; −1,3]) |
| MASSIVE | 86,7 → 78,6 (−8,2 [−12,6; −3,7]) | 88,1 → 75,5 (−12,6 [−17,3; −7,8]) |
| ViFactCheck choice | 60,0 → 57,5 (−2,5 [−11,5; +6,3]) | 57,5 → 57,5 (0,0 [−9,0; +10,0]) |
| ViWikiFC choice | 71,4 → 67,1 (−4,3 [−10,4; +1,2]) | 77,0 → 60,2 (**−16,8** [−24,1; −9,3]) |
| ViFactCheck Có/Không | 85,9 → 84,6 (−1,3 [−9,6; +5,5]) | 87,2 → 82,0 (−5,1 [−11,5; +1,4]) |
| ViWikiFC Có/Không | 83,1 → 83,9 (+0,8 [−3,1; +5,2]) | 85,5 → 79,0 (−6,5 [−12,1; −1,6]) |

### Pb: nhãn tiếng Anh, văn bản tiếng Việt

Mỗi ô: dev → Pb (chênh [CI]); mốc là accuracy trên dòng EN cùng cặp.

| Nhóm | M1 | C@100% |
|---|---|---|
| Toàn bộ | 84,7 → 86,2 (+1,5 [−1,0; +4,2]); mốc EN 87,0 | 84,0 → 87,0 (+3,0 [+0,7; +5,4]); mốc EN 87,7 |
| SIB-200 | 82,6 → 84,3 (+1,7) | 79,9 → 84,3 (+4,3 [+0,7; +8,0]) |
| MASSIVE | 86,7 → 88,1 (+1,4) | 88,1 → 89,8 (+1,7 [−1,0; +4,4]) |

### (c) Nhầm nhãn SIB-200 và MASSIVE

Tỉ lệ dự đoán / tỉ lệ đáp án, %. TVD là khoảng cách giữa phân bố dự đoán và phân bố đáp án.

| Tập | Model | TVD | health | sports | gợi ý địa điểm (MASSIVE) | TVD MASSIVE |
|---|---|---|---|---|---|---|
| dev VI có dấu | M1 | 0,074 | 14,7 / 13,4 | 14,1 / 13,4 | 1,7 / 1,7 | 0,088 |
| | C | 0,064 | 13,7 / 13,4 | 13,0 / 13,4 | 1,4 / 1,7 | 0,078 |
| dev VI không dấu | M1 | 0,141 | 14,4 / 13,4 | 7,4 / 13,4 | 0,0 / 1,7 | 0,211 |
| | C | 0,114 | 12,4 / 13,4 | 11,4 / 13,4 | 2,0 / 1,7 | 0,140 |
| Pa | M1 | 0,144 | 12,0 / 13,4 | 11,4 / 13,4 | 1,7 / 1,7 | 0,153 |
| | C | 0,110 | 11,0 / 13,4 | 14,7 / 13,4 | 1,4 / 1,7 | 0,146 |

## 6. Chưa làm

- ~~Đo lại độ trễ~~: đã đo lại trên GPU trống ngày 26/10 (mục 2).
- `timing.json` của C@50% (lý do ở mục 3). Độ trễ của C@25% và C@100% đo lại đều khoảng 32,5 ms, nên không chạy riêng cho C@50%.
- ~~Đọc lỗi bằng Opus~~: đã làm, xem `phan_tich_C/doc_loi_opus.md`.

## 7. File

**Kết quả:**
- Preds: `results/probe_v1/eval/<phép>/C-s1/`, và `results/probe_v1/eval/dev/C-s1-step{484,969,1454}/`.
- Phân tích: `results/train_v1/phan_tich_C/C-s1_vs_M1.{json,md}` (bảng đủ: theo nguồn, mọi phép thử, ma trận nhầm SIB đầy đủ).
- Chỉ tiêu và đường xu hướng: `results/train_v1/trend_C.json`.

**Script:**
- `scripts/train_v1/run_train_C.sh` (train và hàng đợi chấm ban đầu), `run_C_final.sh` (bản cuối), `run_latency_C.sh` (chưa chạy), `trend_C.py`;
- `scripts/probe_v1/analyze_probe.py`: seed riêng cho mỗi lần tính CI; đã chạy lại cả phân tích mốc.

**Lệnh phân tích:**

```bash
python scripts/probe_v1/analyze_probe.py --m1 "Qwen3-1.7B-fullvocab-s1 (dự án)" --m2 "C-s1" --m0 "Qwen3-1.7B" --r "decider-2b-text" --r "quyet-medium-text" --dev_eval results/probe_v1/eval/dev --out results/train_v1/phan_tich_C/C-s1_vs_M1.json
```

```bash
python scripts/train_v1/trend_C.py
```
