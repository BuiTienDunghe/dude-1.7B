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
