# Báo cáo bậc 2 của 2b: train 2b-s2 (Qwen3-1.7B LoRA, seed 2)

- **Brief:** `results/BRIEF_2b_bac2_train.md`. Người dùng duyệt 29/10, khoảng 01:20.
- **Luật đọc:** `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md` mục 9–11, ghi trước khi chạy.
- **Cách tính:** mọi số tính bằng `scripts/train_2b/bang_tg_bac2.py`, ra `results/train_2b/bang_tg_bac2.json` và `.md`.
  - Script đã được chạy thử trước khi train, cho s2 := s1. Kết quả ra lại đúng mọi số của bậc 1.
- **Phạm vi:** dev 1.3. Không chấm test. Checkpoint 25/50/75% chỉ lưu, không chấm. Không chạy C-s2.
- **Cấu hình:** chép đúng lần chạy s1, chỉ đổi seed (1 → 2) và thư mục đầu ra. Script tự kiểm điều này trước khi train (`manifest_train.json`).
- **Train:**
  - bắt đầu 01:27:32, xong 04:59:34, exit 0;
  - cổng letter_mass ở bước 254: 0,9974, 0/200 dưới 0,5.
- **Chấm:** xong 05:31:40.
- **GPU:** cộng dồn 14.648 / 18.000 giây (4,07 giờ):

| Bước | Giây GPU |
|---|---|
| Train | 12.722 |
| score_probe | 1.603 |
| proxy | 219 |
| val_kc_v2 | 104 |

- **Nhiệt độ:** tối đa 77°C khi train, không có cờ quá nhiệt.

**Worker chỉ áp đúng bảng, không quyết bước tiếp.**

## 1. Bảng T1–T4, G1–G5 (so với C-s1)

| Mã | Ngưỡng | C-s1 | 2b-s2 | Kết quả s2 | TB 2 seed | Kết quả TB | 2b-s1 (đối chiếu) |
|---|---|---|---|---|---|---|---|
| T1 | ≥ 85% cả hai | 62,5 / 50,3 | 100,0 / 99,4 | đạt | 100,0 / 99,7 | đạt | 100,0 / 100,0 |
| T2 | ≥ 0,50 (xám 0,42–0,50) | 0,347 | 0,693 | đạt | 0,747 | đạt | 0,800 |
| T3a | ≥ 0,70 | 0,862 (25/29) | 0,679 (53/78) | trượt | 0,726 (57/78,5) | đạt | 0,772 (61/79) |
| T3b | ≥ 0,78 (94/120) | 0,875 (105/120) | 0,700 (84/120) | trượt | 0,750 (90/120) | trượt | 0,800 (96/120) |
| T4 | ≥ 79,65 | 81,65 | 83,92 | đạt | 84,38 | đạt | 84,84 |
| G1 | ≥ 62,47 | 63,97 | 62,34 | trượt | 62,85 | đạt | 63,35 |
| G2 | ≥ 87,20 | 88,20 | 88,54 | đạt | 88,67 | đạt | 88,80 |
| G3 | tụt ≤ 10 | -16,8 [-23,9; -9,1] | -3,7 [-8,5; 1,3] | đạt | -4,0 [-8,0; 0,0] | đạt | -4,3 [-9,2; 0,6] |
| G4 | ≥ C − 3 = -5,3 | -2,3 | 1,8 | đạt | 0,0 | đạt | -1,8 |
| G5 | ≥ C − 5 (74,9 / 79,0) | 79,9 / 84,0 | 81,9 / 83,0 | đạt | 81,9 / 83,0 | đạt | 81,9 / 83,0 |

T1 = ViFactCheck / ViWikiFC; G5 = SIB-200 / Belebele; G3 kèm CI bootstrap cụm 1.000 lần. T3a của TB = trung bình precision hai seed (dạng gộp: 0,726).

**Thống kê:**

- VI macro, 2b-s2 − C-s1: 2,27 điểm, CI 95% bootstrap cụm 2.000 lần [0,32; 4,32].
- VI macro, TB 2b − C-s1: 2,73 điểm, CI 95% bootstrap cụm 2.000 lần [0,90; 4,68].
- VI macro, 2b-s1 − C-s1: 3,19 điểm, CI 95% bootstrap cụm 2.000 lần [1,02; 5,34].
- McNemar T2, s2 so với C-s1: C đúng s2 sai 1, C sai s2 đúng 27, p = 0,0000.
- McNemar T3b, s2 so với C-s1: C đúng s2 sai 24, C sai s2 đúng 3, p = 0,0000.

**Biên theo số câu:**

| Luật | 2b-s2 | TB 2 seed |
|---|---|---|
| T3b (sàn 94/120) | 84,0 → biên -10,0 câu | 90,0 → biên -4,0 câu |
| T3a (< 0,70 khi) | mất 0 lần đoán NEI đúng, hoặc thêm 0 lần đoán NEI sai | mất 3, hoặc thêm 3 |
| G1 (sàn 62,47) | biên -0,13 điểm ≈ số câu không dấu phải sai thêm: belebele -2; sib200 -2; massive -2; xcopa 0; vifactcheck -1; viwikifc -2 | biên 0,38 điểm: belebele 7; sib200 7; massive 7; xcopa 3; vifactcheck 4; viwikifc 7 |

**Chỉ báo mục 10:**

(a) Recall "Có" của câu Có/Không:

| Nguồn \| chế độ | n | C-s1 | 2b-s1 | 2b-s2 | TB |
|---|---|---|---|---|---|
| vifactcheck có dấu | 28 | 1,000 | 0,821 | 0,821 | 0,821 |
| vifactcheck không dấu | 28 | 0,929 | 0,429 | 0,536 | 0,482 |
| viwikifc có dấu | 47 | 0,872 | 0,830 | 0,808 | 0,819 |
| viwikifc không dấu | 47 | 0,660 | 0,425 | 0,404 | 0,415 |

(b) Đoán NEI trên ViWikiFC không dấu (choice, gold NEI 41/161): C-s1 40, 2b-s1 91, 2b-s2 97 (/ 161; ngưỡng câu hỏi mục 10: > 50% = > 80,5).

(c) Recall Bác bỏ ViWikiFC có dấu (n = 74): C-s1 0,878, 2b-s1 0,770, 2b-s2 0,608, TB 0,689.

(d) 2b-s2 so với 2b-s1:

| Đo | 2b-s1 | 2b-s2 | s2 − s1 | \|Δ\| |
|---|---|---|---|---|
| VI_macro | 84,84 | 83,92 | -0,92 | 0,92 |
| khong_dau_macro | 63,35 | 62,34 | -1,01 | 1,01 |
| EN_macro | 88,80 | 88,54 | -0,25 | 0,25 |
| vifactcheck có dấu | 89,87 | 87,34 | -2,53 | 2,53 |
| vifactcheck không dấu | 76,58 | 80,38 | 3,80 | 3,80 |
| viwikifc có dấu | 82,81 | 76,14 | -6,67 | 6,67 |
| viwikifc không dấu | 60,00 | 58,95 | -1,05 | 1,05 |

**Nhiễu seed (dev, 2b-s1 so với 2b-s2):**

| Nguồn \| chế độ | n | số câu trả lời khác | 2b-s1 | 2b-s2 | s2 − s1 |
|---|---|---|---|---|---|
| belebele có dấu | 294 | 23 (7,8%) | 83,0 | 83,0 | 0,0 |
| sib200 có dấu | 299 | 22 (7,4%) | 81,9 | 81,9 | 0,0 |
| massive có dấu | 294 | 23 (7,8%) | 89,5 | 89,1 | -0,3 |
| xcopa có dấu | 100 | 12 (12,0%) | 82,0 | 86,0 | 4,0 |
| vifactcheck có dấu | 158 | 10 (6,3%) | 89,9 | 87,3 | -2,5 |
| viwikifc có dấu | 285 | 38 (13,3%) | 82,8 | 76,1 | -6,7 |
| belebele không dấu | 294 | 73 (24,8%) | 61,6 | 56,8 | -4,8 |
| sib200 không dấu | 299 | 95 (31,8%) | 58,9 | 52,5 | -6,4 |
| massive không dấu | 294 | 83 (28,2%) | 71,1 | 70,4 | -0,7 |
| xcopa không dấu | 100 | 29 (29,0%) | 52,0 | 55,0 | 3,0 |
| vifactcheck không dấu | 158 | 19 (12,0%) | 76,6 | 80,4 | 3,8 |
| viwikifc không dấu | 285 | 50 (17,5%) | 60,0 | 58,9 | -1,1 |
| belebele EN | 294 | 26 (8,8%) | 85,7 | 85,0 | -0,7 |
| sib200 EN | 299 | 17 (5,7%) | 82,9 | 83,9 | 1,0 |
| massive EN | 294 | 13 (4,4%) | 93,5 | 93,2 | -0,3 |
| xcopa EN | 100 | 3 (3,0%) | 93,0 | 92,0 | -1,0 |

**Chỉ báo khác:**

| Bộ | C-s1 | 2b-s1 | 2b-s2 | TB |
|---|---|---|---|---|
| pb, acc (mức đổi so với dev), n = 593 | 87,0 (3,0) | 88,7 (3,0) | 88,0 (2,5) | 88,4 (2,8) |
| p1b, acc (mức đổi so với dev), n = 482 | 65,4 (1,9) | 71,8 (1,5) | 69,3 (3,3) | 70,5 (2,4) |
| proxy kiểm chứng có dấu (n = 300) | 75,0 | 80,0 | 83,3 | — |
| proxy kiểm chứng không dấu (n = 300) | 67,3 | 68,0 | 69,3 | — |
| val_kc_v2 (n = 300) | 82,0 | 98,0 | 98,7 | — |

**Train:**

| | 2b-s1 | 2b-s2 |
|---|---|---|
| val loss bắt đầu / epoch 1 / epoch 2 | 7,018 / 0,760 / 0,772 | 7,018 / 0,759 / 0,761 |
| Số bước | 1974 | 1974 |
| Thời gian train (wall, giây) | 12612 | 12621 |
| VRAM đỉnh (GB) | 10,25 | 10,25 |
| Nhiệt độ tối đa khi train (°C; số lần đo) | 79 (460) | 77 (446) |
| GPU cộng dồn cả lượt (giây) | 14981 | 14648 |

**Ghi chú về biên:** số âm ở bảng biên nghĩa là đã trượt.
- **T3a của s2:** 53/78 = 0,679, đã dưới 0,70. Để đạt cần thêm 2 lần đoán NEI đúng (55/78 = 0,705), hoặc bớt 3 lần đoán NEI sai (53/75 = 0,707).
- **G1 của s2:** thiếu 0,13 điểm, tức khoảng 1–2 câu không dấu (xcopa: 1 câu = 0,167 điểm; ViFactCheck: 1 câu = 0,105 điểm).

## 2. Áp luật mục 9–11

**Kết quả từng luật, theo cột:**

| Cột | Đạt | Trượt |
|---|---|---|
| 2b-s2 so với C-s1 | 7/10 | **T3a** (0,679 < 0,70), **T3b** (84/120 < 94), **G1** (62,34 < 62,47) |
| TB 2 seed so với C-s1 | 9/10 | **T3b** (0,750 = 90/120 < 94) |

**Mục 11, "Cách đọc khi s2 trượt một luật":**
- "Quyết đi/dừng theo trung bình 2 seed của 2b so với C-s1."
- **T3b:** TB 2 seed cũng trượt. Mục 11 ghi: "Trung bình 2 seed cũng trượt: luật trượt thật. Báo người dùng trước khi làm bước tiếp."
- **T3a và G1:** s2 trượt nhưng TB 2 seed đạt (T3a 0,726; G1 62,85). Luật mục 11 coi là nhiễu seed khi "mức trượt nằm trong dải checkpoint của C"; dải đó là:

| Đo | Dải checkpoint của C | 2b-s2 |
|---|---|---|
| precision NEI | 0,76–0,89 | 0,679 (ngoài dải, thấp hơn) |
| XN + BB ViWikiFC | 95–106/120 | 84 (ngoài dải) |
| G1 (không dấu) | không có trong `g4_out.txt` | — |

Worker chỉ báo các số này; điều phối và người dùng đọc.

**Mục 10, các câu hỏi seed 2 phải trả lời:**
- **Bác bỏ ViWikiFC có tụt ≥ 8 điểm nữa không?** s2 = 0,608: thấp hơn C 27,0 điểm và thấp hơn s1 16,2 điểm. Lặp lại, và nặng hơn s1.
- **"NEI mặc định" khi không dấu có lặp lại không (> 50%)?** s2 đoán NEI 97/161 = 60,2% (s1 91/161 = 56,5%; gold 41; C 40). Lặp lại.
- **T2 / T3a / T3b có giữ không?**

| Luật | s2 | s1 | Giữ? |
|---|---|---|---|
| T2 | 0,693 | 0,800 | giữ (≥ 0,50) |
| T3a | 0,679 (53/78) | 0,772 | không giữ |
| T3b | 0,700 (84/120) | 0,800 | không giữ |

- **Macro lệch seed 1 bao nhiêu?** |Δ| VI macro = 0,92 (≤ 2), không dấu 1,01, EN 0,25.
  - Từng nguồn kiểm chứng: ViWikiFC có dấu 6,67; ViFactCheck có dấu 2,53; ViFactCheck không dấu 3,80; ViWikiFC không dấu 1,05.

**Các nhánh mục 11, chỉ ghi điều kiện:**
- |Δ| VI macro = 0,92 ≤ 2, nên điều kiện chạy C-s2 vì |Δ| > 2 (mục 10) không xảy ra.
- Nhánh "|Δ| ≤ 2 và giữ đủ luật" **không** thỏa: T3b trượt cả ở TB 2 seed.
- Hai chi phí đã ghi ở mục 10 đều lặp lại: Bác bỏ ViWikiFC tụt và NEI mặc định khi không dấu. Mục 11 ghi: "Hai chi phí lặp lại → lỗi có hệ thống, đúng mục tiêu sửa."

So với dự đoán ghi trước ở mục 11:
- VI macro 83,92, trong dải dự đoán 82,5–86.
- T3b là luật "dễ trượt nhất (khoảng 30%)", và đã trượt.
- G1 trượt (dự đoán khoảng 25–35%). T3a trượt (dự đoán khoảng 10%).
- Đạt lại đủ 10 luật: không (dự đoán khoảng 40–55%).

## 3. File

- **Adapter, log, summary.json, manifest:** `results/train_2b/train/2b-Qwen3-1.7B-s2/`.
  - `summary.json` có thêm khóa `nhiet_do_toi_da_c` (77°C), do `run_2b_bac2.sh` ghi sau train.
- **Preds:**
  - probe: `results/probe_v1/eval/<phép>/2b-s2/preds.jsonl`;
  - proxy và val_kc_v2: `results/train_2b/eval/{proxy,val_kc_v2}/2b-s2/`.
- **Log chấm:** `results/probe_v1/eval/logs/2b-s2.txt`, `results/train_2b/eval/logs/{proxy,valkc}_2b-s2.txt`.
