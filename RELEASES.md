# Các phiên bản

Mỗi mục dưới đây cũng là một GitHub Release (tag cùng tên). Mọi tag trỏ vào ảnh chụp code ngày 29/10/2026, vì dự án chưa có lịch sử git trước đó; thông số là của từng lần chạy gốc.



## v0.3-2b — 2b: dữ liệu đúng khung bench, 2 seed (28–29/10/2026)

> Trạng thái: **chưa nhận nguyên trạng.** Luật T3b, đặt trước khi train, trượt ở trung bình 2 seed. Đang sửa ở hướng 1 (2b′).
> Chỉ dùng nội bộ. Tag trỏ vào ảnh chụp code ngày 29/10.

### Cấu hình train
| Mục | Giá trị |
|---|---|
| Model nền | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Dữ liệu | `results/train_2b/train_2b_v3.jsonl`: 31.596 dòng, sha256 `ea604a0138c20f42b4beae67cdeb3fe02908c0fa96cd63b104c469cc54a11635` |
| Thành phần | C′ = C sửa theo kiểm toán: bỏ nguồn không văn bản, bỏ VNTC, bỏ 146 dòng teacher bất đồng tự tin; đưa vitaminc/nli_fever về khung bench (2d); 250 dòng ghép lạc đề → NEI (2e); tăng cường "từ chối" cho đọc hiểu (2h). Cộng **KC**: 3.000 câu kiểm chứng tự dựng từ Wikipedia tiếng Việt 20231101 (CC BY-SA), nhãn đến từ cách dựng, đọc mù 2 vòng |
| LoRA, tối ưu, loss | như v0.2: r 16 / α 32 / 0,05; lr 2e-4; batch 32 = 8 × 4; 2 epoch; warmup 3% rồi cosine; `full_vocab` |
| Số bước | 1.974 (warmup 59) |
| Trần state | 2.048 token |
| Seed | 1 và 2 |
| Thời gian | s1: 12.612 giây; s2: 12.621 giây. Khoảng 4,1–4,2 giờ GPU mỗi seed, tính cả chấm |
| VRAM, nhiệt | đỉnh 10,25 GB; nhiệt GPU tối đa 79°C (s1), 77°C (s2) |
| Val loss | s1: 0,760 → 0,772; s2: 0,759 → 0,761 |

### Điểm trên dev 1.3
| | VI có dấu | Không dấu | EN |
|---|---|---|---|
| **2b-s1** | **84,84** | 63,35 | 88,80 |
| **2b-s2** | **83,92** | 62,34 | 88,55 |
| **Trung bình 2 seed** | **84,38** | 62,85 | 88,68 |
| C (v0.2) | 81,65 | 63,97 | 88,20 |
| decider-2b-text (mốc cần vượt) | 86,50 | 49,64 | 90,33 |

- Trung bình 2 seed hơn C **+2,73**, CI bootstrap cụm [+0,90; +4,68]. Lệch giữa 2 seed: 0,92 điểm.

| VI có dấu theo nguồn | ViFactCheck | ViWikiFC | SIB-200 | MASSIVE | Belebele | XCOPA |
|---|---|---|---|---|---|---|
| 2b-s1 | 89,87 | 82,81 | 81,94 | 89,46 | 82,99 | 82,00 |
| 2b-s2 | 87,34 | 76,14 | 81,94 | 89,12 | 82,99 | 86,00 |
| C | 72,15 | 80,70 | 79,93 | 88,10 | 84,01 | 85,00 |
| decider-2b-text | 83,54 | 82,81 | 82,27 | 96,26 | 88,10 | 86,00 |

| Không dấu theo nguồn | ViFactCheck | ViWikiFC | SIB-200 | MASSIVE | Belebele | XCOPA |
|---|---|---|---|---|---|---|
| 2b-s1 | 76,58 | 60,00 | 58,86 | 71,09 | 61,56 | 52,00 |
| 2b-s2 | 80,38 | 58,95 | 52,51 | 70,41 | 56,80 | 55,00 |

| EN theo nguồn | SIB-200 | MASSIVE | Belebele | XCOPA |
|---|---|---|---|---|
| 2b-s1 | 82,94 | 93,54 | 85,71 | 93,00 |
| 2b-s2 | 83,95 | 93,20 | 85,03 | 92,00 |

### Luật đặt trước (PHAN_TICH_TRUOC_TRAIN_2b mục 9–11)
| | s1 | s2 | Trung bình 2 seed |
|---|---|---|---|
| Số luật đạt (T1–T4, G1–G5) | 10/10 | 7/10 | 9/10 |
| Luật trượt | — | T3a 0,679; T3b 84/120; G1 62,34 | **T3b 90/120, cần 94** |

- **Phép thử:** P4 choice có dấu 100 / 100 (s1), trong khi C chỉ 62,5. Pa −4,3 và P5 −1,8 (s1). Theo cách đo của báo cáo 2b, Pa của C là −16,8; cách đo này khác cách đo trong báo cáo C ở v0.2.
- **Lỗi chính:** câu Bác bỏ của ViWikiFC bị đoán thành "không đủ thông tin". Tỉ lệ nhận đúng câu Bác bỏ: C 0,878 → s1 0,770 → s2 0,608. Nguyên nhân là mẹo trùng từ do KC tạo ra: câu Bác bỏ trong KC chép gần nguyên văn bản, còn câu NEI thì ít trùng từ.
- **Mức tăng so với C đến gần như toàn bộ từ ViFactCheck.** Khoảng cách còn lại tới decider nằm ở MASSIVE, Belebele và XCOPA.
- **Độ trễ:** chưa đo riêng. Cùng kiến trúc và cùng cỡ LoRA với C.

### Trọng số
Không đưa lên repo vì có dữ liệu UIT chỉ dùng nội bộ.

| Adapter | Kích thước | sha256 |
|---|---|---|
| `results/train_2b/train/2b-Qwen3-1.7B-s1/adapter/adapter_model.safetensors` | 69,8 MB | `3398ef3f22e07a93f0b7753c2ceea1c3694e327ce22e56d9b7115a08c3a826bb` |
| `results/train_2b/train/2b-Qwen3-1.7B-s2/adapter/adapter_model.safetensors` | 69,8 MB | `2cfcb8bb5773a9682223a7aba05ab2611fef7249f49ad16bfaafe935cb8f74e7` |

### Tiếp theo
**2b′ (hướng 1):** 2b cộng 1.500 câu ViANLI có câu Bác bỏ ít trùng từ, và bỏ dấu tại chỗ. Dữ liệu V-e đã dựng xong: `results/train_2b_prime/train_2b_prime.jsonl`, 33.096 dòng, sha256 `c97bb08d…`. Dự đoán 84,6–85,4. Đang chờ duyệt train. Sẽ có release riêng sau khi train.

---

## v0.2-c — C: thêm dữ liệu tiếng Việt và dữ liệu dịch (26/10/2026)

> Trạng thái: **mốc tham chiếu**, chỉ dùng nội bộ. Bản kiểm toán ngày 27/10 dẫn tới việc sửa dữ liệu ở v0.3.
> Tag trỏ vào ảnh chụp code ngày 29/10. Thông số dưới đây là của lần chạy gốc.

### Cấu hình train
| Mục | Giá trị |
|---|---|
| Model nền | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Dữ liệu | `results/train_v1/components/C.jsonl`: 31.034 dòng, sha256 `005ca3d547aa2f9e2f356a8f75eb7f3f17addba1cc4d74e3a7b44a7b39c86e89` |
| Thành phần | train_v0, phần A (dữ liệu VI có sẵn, gồm 25 chuỗi mới người dùng đã duyệt), phần B (6.944 dòng EN dịch sang VI bằng Sonnet, có đọc lại), trần 5% mỗi nguồn, khử rò rỉ với bench |
| Val | `data/val_v0.jsonl`: 1.000 dòng |
| LoRA, tối ưu, loss | như v0.1: r 16 / α 32 / 0,05; lr 2e-4; batch 32 = 8 × 4; 2 epoch; warmup 3% rồi cosine; `full_vocab` |
| Số bước | 1.938 (warmup 58); lưu checkpoint ở 25/50/75% |
| Seed | 1 |
| Trần state | 1.024 token |
| Thời gian, VRAM | 10.625 giây (2,95 giờ); đỉnh 10,25 GB |
| Val loss | 7,018 → 0,758 → 0,757 |
| Cổng letter_mass | bước 250: 0,9977 (ngưỡng 0,5), đạt |

### Điểm trên dev 1.3
| | VI có dấu | Không dấu | EN |
|---|---|---|---|
| **C** | **81,65** | **63,97** | **88,20** |
| M1 (v0.1) | 79,88 | 62,37 | 87,95 |

Chênh C − M1 ở VI có dấu: +1,77, CI [−0,28; +3,66].

| Nguồn | ViFactCheck | ViWikiFC | SIB-200 | MASSIVE | Belebele | XCOPA |
|---|---|---|---|---|---|---|
| VI có dấu | 72,15 | 80,70 | 79,93 | 88,10 | 84,01 | 85,00 |
| Không dấu | 66,46 | 62,46 | 62,21 | 74,15 | 59,52 | 59,00 |
| EN | — | — | 82,94 | 92,52 | 83,33 | 94,00 |

- **Checkpoint 25/50/75%** (VI có dấu): 81,07 / 80,17 / 81,38.
- **Phép thử:**
  - P4 choice có dấu: 62,5 / 50,3, kém M1 (91,2 / 88,2). Đây là điểm yếu chính.
  - Pa (lựa chọn bỏ dấu): −9,0.
  - Pb (nhãn tiếng Anh): +3,0.
- **Độ trễ, batch 1:**
  - GPU trống, ngày 26/10: trung vị 32,4–32,7 ms;
  - G0, ngày 28/10, đường thường: trung vị 31,8, p90 47,3, p95 85,4 ms.
  - Đạt ≤ 40 ms tính theo trung vị.

### Vì sao dừng ở đây
Kiểm toán ngày 27/10 tìm ra ba lỗi dữ liệu:
- dữ liệu VI lệch khung bench;
- khoảng 1.109 dòng không có văn bản;
- NEI chỉ chiếm 12,8%, trong khi bench là 31%.

Kèm theo là lỗi quy trình: xáo thứ tự câu thang điểm, trần 1.024 token, mặc định loss. Nghiên cứu hướng đi cùng ngày đổi mục tiêu thành "model VI tốt nhất trong ≤ 40 ms, vượt decider-2b (86,5)".

### Trọng số
Adapter `results/train_v1/train/C-Qwen3-1.7B-s1/adapter/adapter_model.safetensors` (69,8 MB), sha256 `45259cdaee27152558e081b1a0160525d4039af2d8187b10f6698fd826fac412`. Không đưa lên repo vì có dữ liệu UIT chỉ dùng nội bộ.

---

## v0.1-m1 — M1: lần train đầu tiên của dự án (12/10/2026)

> Trạng thái: **mốc tham chiếu**, chỉ dùng nội bộ. Không đính kèm trọng số (xem cuối trang).
> Tag này trỏ vào ảnh chụp code ngày 29/10, vì dự án chưa có lịch sử git trước đó. Thông số dưới đây là của lần chạy gốc.

### Cấu hình train
| Mục | Giá trị |
|---|---|
| Model nền | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Dữ liệu | `data/train_v0.jsonl`: 20.000 dòng, sha256 `d10c9362d2cbade1b5e19790938f06c3a82011276d1da02491b6703e2e1fb466` |
| Val | `data/val_v0.jsonl`: 1.000 dòng, sha256 `4377e710…` |
| LoRA | r 16, α 32, dropout 0,05; q, k, v, o, gate, up, down; 17.432.576 tham số train |
| Tối ưu | lr 2e-4; batch hiệu dụng 32 = 8 × 4; 2 epoch; 1.250 bước; warmup 37 bước rồi cosine về 0; grad clip 1,0 |
| Loss | `full_vocab`: cross-entropy trên toàn từ vựng tại vị trí trả lời, đích mềm trên các token chữ cái |
| Độ chính xác số | trọng số bf16, autocast bf16, tham số LoRA fp32; gradient checkpointing |
| Seed | 1 |
| Trần state | 1.024 token (lúc đó chưa nâng lên 2.048) |
| Thời gian, VRAM | 6.619 giây (1,84 giờ); đỉnh 10,25 GB |
| Val loss | 7,018 → 0,756 (epoch 1) → 0,776 (epoch 2) |

### Điểm trên dev 1.3 (accuracy %, trung bình đều theo nguồn)
| | VI có dấu | Không dấu | EN |
|---|---|---|---|
| **M1** | **79,88** | **62,37** | **87,95** |
| M0, Qwen3-1.7B gốc | 70,64 | 48,45 | 84,59 |

| Nguồn | ViFactCheck | ViWikiFC | SIB-200 | MASSIVE | Belebele | XCOPA |
|---|---|---|---|---|---|---|
| VI có dấu | 72,78 | 76,49 | 82,61 | 86,73 | 82,65 | 78,00 |
| Không dấu | 65,19 | 66,32 | 60,20 | 66,33 | 59,18 | 57,00 |
| EN | — | — | 83,95 | 90,14 | 86,73 | 91,00 |

- **P4** (văn bản lạc đề → "không đủ thông tin", choice có dấu): 91,2 / 88,2.
- **Độ trễ:** trung vị batch 1 khoảng 32,6 ms, đo khi GPU trống ngày 26/10.

### Quyết định kèm theo
Chọn Qwen3-1.7B thay vì 0.6B: hơn 5,2 điểm VI (trung bình 4 cặp seed), tốc độ ngang nhau ở prompt ngắn batch 1.

### Trọng số
Adapter `results/ft_loss/train/Qwen3-1.7B-s1/adapter/adapter_model.safetensors` (69,8 MB), sha256 `4b3087b0d35a65a4a2191c46eb55c192eab17d01b5a032914a2d770765cbfd40`. Không đưa lên repo vì dữ liệu train có nguồn chỉ dùng nội bộ.

---

## bench-v1.3 — Benchmark tiếng Việt 1.3 (đóng băng 17/10/2026)

> Benchmark dùng để chấm mọi phiên bản model ở trên. **Repo không chứa dữ liệu bench**, chỉ có mã băm và cách dựng.
> **Test chưa chạy model nào.** Test chỉ được chấm một lần ở cuối, theo giao thức ghi trước.

### Thành phần dev 1.3
`data/bench_v1_3/dev_v1_3.jsonl`: 3.847 dòng, sha256 `c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e`.

| Nguồn | Loại | VI có dấu | VI không dấu | EN song song |
|---|---|---|---|---|
| ViFactCheck | kiểm chứng (choice + Có/Không) | 158 | 158 | — |
| ViWikiFC | kiểm chứng (choice + Có/Không) | 285 | 285 | — |
| SIB-200 | chủ đề | 299 | 299 | 299 |
| MASSIVE | ý định | 294 | 294 | 294 |
| Belebele | đọc hiểu | 294 | 294 | 294 |
| XCOPA | nhân quả | 100 | 100 | 100 |

- **Điểm** là accuracy, trung bình đều theo nguồn (macro). Một câu XCOPA đáng 0,167 điểm macro; ViFactCheck 0,105; ViWikiFC 0,058; các nguồn còn lại khoảng 0,057.
- **Cách chấm:** model chọn chữ cái đáp án. Điểm lấy từ logit các token chữ cái, không sinh văn bản.

### Lịch sử phiên bản
| Bản | Ngày | Thay đổi | sha256 test |
|---|---|---|---|
| 1.0 | 12–13/10 | Lõi mở: Belebele, MASSIVE, XCOPA, SIB-200 (song song) và ViFactCheck, ViWikiFC, ViHSD (bản địa). Mọi dòng VI có bản không dấu | `d92d31f6…0d47` |
| 1.1 | 13/10 | Bỏ ViHSD. Sửa câu chữ kiểm chứng và 8 mô tả ý định MASSIVE, có người làm dự án duyệt | `fbad818b…` |
| 1.2 | 16/10 | Người soát toàn bộ (Opus soát dịch và đáp án, người dùng dịch lại một số đoạn Belebele); 144 quyết định nhãn; NFC; dev sha `ef88be16…` | `2bdbd210…` |
| **1.3** | **17/10** | Loại 2 cặp, đánh dấu 26 cặp mơ hồ (đọc mù 134 cặp). MLQA dời sang 1.4. **Test giữ nguyên từng byte như 1.2** | `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da` |

**Kiểm khi đóng băng:** dev↔test 0 trùng; train_v0 chạm test 2 / dev 0; không dấu 0 lệch; `release_check` đạt.

### Điểm tham chiếu trên dev 1.3 (VI có dấu)
| Model | VI có dấu | Không dấu | Ghi chú |
|---|---|---|---|
| Quyet-1.0-Medium | 89,6 | — | 4,66B tham số, khoảng 88 ms: quá chậm cho mục tiêu |
| decider-2b-text | 86,50 | 49,64 | **mốc cần vượt** |
| 2b, TB 2 seed (v0.3) | 84,38 | 62,85 | |
| C (v0.2) | 81,65 | 63,97 | |
| M1 (v0.1) | 79,88 | 62,37 | |
| Qwen3-1.7B gốc | 70,64 | 48,45 | |

---

