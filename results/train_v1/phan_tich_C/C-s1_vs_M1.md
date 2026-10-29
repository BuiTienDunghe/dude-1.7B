# Phân tích xu hướng: M1 = Qwen3-1.7B-fullvocab-s1 (dự án), M2 = C-s1, M0 = Qwen3-1.7B, R1 = decider-2b-text, R2 = quyet-medium-text

Phân tích trên dev 1.3 và bộ phân tích probe_v1; không phải điểm benchmark; không dùng để chọn checkpoint.

## Lát cắt (accuracy %, CI 95% bootstrap theo cụm; chênh M2 − M1 kèm McNemar)

| Lát | Nhóm | n | M1 | M2 | M0 | R1 | R2 | M2 − M1 | p McNemar |
|---|---|---|---|---|---|---|---|---|---|
| da_sua_o_1_2 (dòng VI) | da_sua | 598 | 70.7 [66.8; 74.5] | 71.4 [67.7; 75.1] | 62.0 [57.8; 66.4] | 68.6 [64.9; 72.0] | 78.4 [75.3; 81.6] | +0.7 [-2.5; 3.8] | 0.749 |
| da_sua_o_1_2 (dòng VI) | giu_nguyen | 2262 | 72.1 [70.1; 74.4] | 73.7 [71.7; 75.9] | 58.8 [56.5; 61.1] | 67.5 [65.7; 69.4] | 76.8 [74.9; 78.8] | +1.6 [-0.0; 3.2] | 0.0587 |
| do_dai_van_ban | dai | 1204 | 75.4 [72.4; 78.4] | 78.2 [75.3; 81.0] | 65.5 [62.4; 68.8] | 73.9 [71.2; 76.7] | 82.4 [79.7; 85.0] | +2.7 [0.3; 5.1] | 0.021 |
| do_dai_van_ban | ngan | 1387 | 76.5 [73.8; 79.3] | 75.6 [72.8; 78.4] | 66.8 [63.6; 69.8] | 72.4 [69.8; 74.8] | 81.2 [78.6; 83.7] | -0.9 [-2.7; 0.9] | 0.404 |
| do_dai_van_ban | vua | 1256 | 75.4 [72.4; 78.3] | 76.8 [73.9; 79.7] | 65.2 [62.2; 68.3] | 74.2 [71.6; 76.8] | 80.2 [77.5; 82.9] | +1.4 [-0.7; 3.4] | 0.193 |
| goc_vs_dich (VI có dấu) | nguon_dich | 987 | 83.4 [80.9; 85.6] | 84.1 [81.9; 86.3] | 75.8 [73.1; 78.3] | 88.5 [86.6; 90.5] | 91.6 [89.8; 93.2] | +0.7 [-1.4; 2.7] | 0.558 |
| goc_vs_dich (VI có dấu) | nguon_goc_VI | 443 | 75.2 [71.4; 79.0] | 77.6 [74.2; 81.2] | 59.4 [54.6; 63.7] | 83.1 [79.6; 86.5] | 84.0 [80.8; 87.0] | +2.5 [-0.7; 5.6] | 0.144 |
| kieu_cau | chon_mot | 3443 | 75.0 [73.2; 76.8] | 76.5 [74.6; 78.2] | 65.1 [63.0; 67.1] | 72.7 [71.2; 74.4] | 81.9 [80.3; 83.5] | +1.5 [0.1; 2.6] | 0.0237 |
| kieu_cau | noul | 404 | 82.4 [78.2; 86.6] | 79.5 [75.0; 84.4] | 72.8 [68.0; 78.0] | 79.7 [76.0; 84.0] | 75.5 [70.7; 80.6] | -3.0 [-6.3; 0.5] | 0.111 |
| mo_ho | co_co | 74 | 16.2 [6.7; 26.4] | 13.5 [6.5; 21.9] | 18.9 [8.6; 30.7] | 13.5 [4.0; 24.7] | 14.9 [5.5; 25.0] | -2.7 [-9.6; 5.3] | 0.727 |
| mo_ho | khong_co | 3773 | 77.0 [75.4; 78.6] | 78.0 [76.4; 79.5] | 66.8 [65.0; 68.5] | 74.6 [73.2; 76.1] | 82.6 [81.2; 84.0] | +1.1 [-0.2; 2.3] | 0.0841 |
| ngon_ngu | en|orig | 987 | 87.3 [85.2; 89.2] | 87.0 [84.9; 89.2] | 84.3 [81.8; 86.6] | 90.0 [88.1; 91.6] | 93.3 [91.8; 94.9] | -0.3 [-1.9; 1.4] | 0.807 |
| ngon_ngu | vi|nodiac | 1430 | 62.8 [60.4; 65.2] | 64.4 [62.0; 66.8] | 48.3 [45.8; 51.0] | 48.7 [46.3; 51.0] | 65.0 [62.6; 67.4] | +1.6 [-0.7; 4.0] | 0.203 |
| ngon_ngu | vi|orig | 1430 | 80.8 [78.6; 82.8] | 82.1 [80.1; 84.1] | 70.7 [68.3; 73.2] | 86.9 [85.0; 88.6] | 89.2 [87.6; 90.8] | +1.3 [-0.5; 3.0] | 0.168 |
| nguon | belebele | 882 | 76.2 [72.7; 79.6] | 75.6 [72.3; 79.0] | 68.4 [64.5; 72.1] | 76.6 [73.4; 79.7] | 85.5 [83.0; 87.8] | -0.6 [-2.9; 2.0] | 0.702 |
| nguon | massive | 882 | 81.1 [77.9; 84.1] | 84.9 [82.2; 87.6] | 70.4 [67.2; 73.6] | 82.2 [79.7; 84.6] | 90.7 [88.4; 92.9] | +3.9 [1.5; 6.1] | 0.0011 |
| nguon | sib200 | 897 | 75.6 [71.9; 79.3] | 75.0 [71.0; 78.7] | 69.3 [65.5; 73.0] | 67.3 [63.8; 70.7] | 77.0 [73.1; 80.6] | -0.6 [-2.8; 1.8] | 0.707 |
| nguon | vifactcheck | 316 | 69.0 [62.9; 74.8] | 69.3 [63.0; 75.3] | 51.3 [44.4; 57.8] | 68.3 [63.2; 73.3] | 76.9 [72.1; 81.6] | +0.3 [-4.1; 5.2] | 1 |
| nguon | viwikifc | 570 | 71.4 [67.0; 75.3] | 71.6 [67.8; 75.6] | 54.0 [49.1; 59.0] | 65.4 [61.7; 69.3] | 67.2 [62.8; 71.4] | +0.2 [-2.7; 3.3] | 1 |
| nguon | xcopa | 300 | 75.3 [70.0; 81.0] | 79.3 [74.3; 84.0] | 72.7 [67.3; 77.3] | 77.3 [72.3; 82.3] | 85.0 [81.7; 88.3] | +4.0 [-0.3; 8.3] | 0.0961 |
| nguon (VI có dấu) | belebele | 294 | 82.7 [78.4; 86.8] | 84.0 [79.8; 88.0] | 75.2 [70.1; 80.0] | 88.1 [84.5; 91.7] | 94.2 [91.4; 96.7] | +1.4 [-2.3; 5.1] | 0.585 |
| nguon (VI có dấu) | massive | 294 | 86.7 [82.7; 90.1] | 88.1 [84.4; 91.5] | 75.2 [70.4; 80.3] | 96.3 [93.9; 98.3] | 96.6 [94.6; 98.6] | +1.4 [-2.4; 5.1] | 0.585 |
| nguon (VI có dấu) | sib200 | 299 | 82.6 [77.9; 86.6] | 79.9 [75.2; 84.6] | 75.6 [70.6; 80.6] | 82.3 [77.6; 86.6] | 83.6 [79.3; 87.6] | -2.7 [-6.0; 0.7] | 0.201 |
| nguon (VI có dấu) | vifactcheck | 158 | 72.8 [65.6; 79.2] | 72.2 [65.6; 78.6] | 57.6 [49.6; 64.5] | 83.5 [77.2; 89.2] | 88.6 [83.4; 93.2] | -0.6 [-5.6; 3.9] | 1 |
| nguon (VI có dấu) | viwikifc | 285 | 76.5 [71.7; 80.8] | 80.7 [76.5; 85.0] | 60.4 [54.6; 66.1] | 82.8 [78.3; 87.0] | 81.4 [76.8; 85.8] | +4.2 [1.0; 8.3] | 0.0501 |
| nguon (VI có dấu) | xcopa | 100 | 78.0 [70.0; 86.0] | 85.0 [77.0; 92.0] | 80.0 [72.0; 88.0] | 86.0 [79.0; 92.0] | 93.0 [88.0; 97.0] | +7.0 [-1.0; 14.0] | 0.118 |
| so_lua_chon | 2 | 704 | 79.4 [76.2; 82.7] | 79.4 [76.1; 82.6] | 72.7 [69.4; 76.0] | 78.7 [75.3; 82.0] | 79.5 [76.0; 82.5] | +0.0 [-2.7; 2.8] | 1 |
| so_lua_chon | 3-4 | 1364 | 70.7 [67.8; 73.6] | 71.3 [68.5; 74.3] | 57.1 [53.8; 60.3] | 69.1 [66.5; 71.9] | 78.8 [76.3; 81.1] | +0.7 [-1.5; 2.9] | 0.58 |
| so_lua_chon | >=5 | 1779 | 78.3 [75.9; 80.8] | 79.9 [77.6; 82.2] | 69.9 [67.5; 72.5] | 74.7 [72.5; 76.7] | 83.8 [81.6; 86.0] | +1.6 [0.1; 3.3] | 0.0571 |
| tat_ca | tat_ca | 3847 | 75.8 [74.1; 77.4] | 76.8 [75.1; 78.3] | 65.9 [64.2; 67.7] | 73.5 [72.0; 74.9] | 81.3 [79.8; 82.7] | +1.0 [-0.3; 2.2] | 0.104 |

## Chuyển đổi M1 → M2

| Nguồn | đúng→đúng | đúng→sai | sai→đúng | sai→sai |
|---|---|---|---|---|
| belebele | 615 | 57 | 52 | 158 |
| massive | 680 | 35 | 69 | 98 |
| sib200 | 619 | 59 | 54 | 165 |
| vifactcheck | 192 | 26 | 27 | 71 |
| viwikifc | 360 | 47 | 48 | 115 |
| xcopa | 210 | 16 | 28 | 46 |
| tat_ca | 2676 | 240 | 278 | 653 |

## Trùng lỗi với model tham chiếu

| Cặp | Jaccard tập sai | P(R sai \| M sai) | tỉ lệ sai của R | tỉ lệ sai của M |
|---|---|---|---|---|
| M1_voi_R1 (decider-2b-text) | 0.382 | 0.579 | 0.265 | 0.242 |
| M2_voi_R1 (decider-2b-text) | 0.365 | 0.573 | 0.265 | 0.232 |
| M0_voi_R1 (decider-2b-text) | 0.379 | 0.488 | 0.265 | 0.341 |
| M1_voi_R2 (quyet-medium-text) | 0.350 | 0.460 | 0.187 | 0.242 |
| M2_voi_R2 (quyet-medium-text) | 0.361 | 0.479 | 0.187 | 0.232 |
| M0_voi_R2 (quyet-medium-text) | 0.319 | 0.375 | 0.187 | 0.341 |

## Hiệu chỉnh (toàn dev)

| Model | Brier | ECE | tự tin khi sai | tự tin khi đúng |
|---|---|---|---|---|
| M1 | 0.3499 | 0.0868 | 0.699 | 0.891 |
| M2 | 0.3441 | 0.0879 | 0.714 | 0.898 |
| M0 | 0.6250 | 0.2970 | 0.917 | 0.976 |
| R1 | 0.3469 | 0.0145 | 0.520 | 0.810 |
| R2 | 0.2698 | 0.0290 | 0.665 | 0.882 |

## Phép thử so theo cặp với dev (acc dev → acc phép thử, chênh, CI)

### p1b

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| khung:kiem_chung_choice|nodiac | 53.5 → 60.2 (+6.6 [1.2; 12.1]) | 56.4 → 58.1 (+1.7 [-5.1; 8.4]) | 31.1 → 41.1 (+10.0 [1.3; 18.2]) | 33.2 → 48.5 (+15.3 [5.8; 25.4]) | 49.4 → 57.7 (+8.3 [2.1; 14.3]) |
| khung:kiem_chung_choice|orig | 67.6 → 73.4 (+5.8 [1.2; 10.2]) | 70.5 → 72.6 (+2.1 [-1.2; 5.1]) | 41.9 → 52.3 (+10.4 [4.4; 16.5]) | 77.6 → 73.9 (-3.7 [-9.0; 1.3]) | 83.8 → 82.2 (-1.7 [-4.8; 1.3]) |
| tat_ca | 60.6 → 66.8 (+6.2 [2.9; 9.5]) | 63.5 → 65.3 (+1.9 [-1.8; 5.8]) | 36.5 → 46.7 (+10.2 [4.4; 15.9]) | 55.4 → 61.2 (+5.8 [-0.2; 11.9]) | 66.6 → 69.9 (+3.3 [0.0; 6.8]) |
| vifactcheck|vi|nodiac|chon_mot | 48.8 → 55.0 (+6.2 [-1.3; 14.1]) | 60.0 → 57.5 (-2.5 [-13.9; 7.2]) | 30.0 → 42.5 (+12.5 [2.6; 22.1]) | 33.8 → 45.0 (+11.2 [-4.5; 27.4]) | 56.2 → 63.7 (+7.5 [-3.6; 18.8]) |
| vifactcheck|vi|orig|chon_mot | 60.0 → 75.0 (+15.0 [6.2; 25.3]) | 57.5 → 66.2 (+8.8 [2.7; 15.0]) | 33.8 → 47.5 (+13.8 [3.6; 24.4]) | 76.2 → 73.8 (-2.5 [-9.3; 4.1]) | 87.5 → 86.2 (-1.2 [-6.1; 2.6]) |
| viwikifc|vi|nodiac|chon_mot | 55.9 → 62.7 (+6.8 [0.0; 14.0]) | 54.7 → 58.4 (+3.7 [-4.5; 11.5]) | 31.7 → 40.4 (+8.7 [-1.9; 19.6]) | 32.9 → 50.3 (+17.4 [5.7; 29.2]) | 46.0 → 54.7 (+8.7 [1.9; 15.8]) |
| viwikifc|vi|orig|chon_mot | 71.4 → 72.7 (+1.2 [-3.2; 6.0]) | 77.0 → 75.8 (-1.2 [-4.6; 2.0]) | 46.0 → 54.7 (+8.7 [1.8; 16.6]) | 78.3 → 73.9 (-4.3 [-11.2; 2.8]) | 82.0 → 80.1 (-1.9 [-6.0; 1.9]) |

### p5

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| khung:kiem_chung_choice|nodiac | 53.5 → 53.1 (-0.4 [-6.4; 5.2]) | 56.4 → 52.7 (-3.7 [-9.4; 2.4]) | 31.1 → 33.6 (+2.5 [-5.8; 10.8]) | 33.2 → 41.5 (+8.3 [0.9; 14.6]) | 49.4 → 55.2 (+5.8 [-0.8; 12.1]) |
| khung:kiem_chung_choice|orig | 67.6 → 70.5 (+2.9 [-0.9; 6.6]) | 70.5 → 68.0 (-2.5 [-5.2; 0.4]) | 41.9 → 38.6 (-3.3 [-10.3; 3.3]) | 77.6 → 75.1 (-2.5 [-6.5; 1.3]) | 83.8 → 82.6 (-1.2 [-3.4; 0.8]) |
| khung:kiem_chung_noul|nodiac | 80.7 → 80.2 (-0.5 [-3.5; 2.7]) | 72.8 → 75.7 (+3.0 [-0.9; 6.5]) | 65.3 → 73.3 (+7.9 [2.9; 13.0]) | 69.8 → 70.3 (+0.5 [-4.5; 5.5]) | 66.8 → 66.3 (-0.5 [-3.6; 2.9]) |
| khung:kiem_chung_noul|orig | 84.2 → 84.2 (+0.0 [-3.0; 3.1]) | 86.1 → 84.2 (-2.0 [-4.6; 0.0]) | 80.2 → 79.7 (-0.5 [-4.3; 3.4]) | 89.6 → 90.1 (+0.5 [-2.0; 3.0]) | 84.2 → 86.1 (+2.0 [0.0; 4.4]) |
| khung:massive|nodiac | 66.3 → 66.0 (-0.3 [-4.1; 2.7]) | 74.2 → 74.5 (+0.3 [-2.7; 3.4]) | 45.6 → 40.1 (-5.4 [-8.5; -2.4]) | 53.7 → 54.8 (+1.0 [-2.4; 4.1]) | 78.6 → 77.9 (-0.7 [-4.1; 2.4]) |
| khung:massive|orig | 86.7 → 86.4 (-0.3 [-2.4; 2.0]) | 88.1 → 87.4 (-0.7 [-2.4; 0.7]) | 75.2 → 73.1 (-2.0 [-6.1; 2.0]) | 96.3 → 95.9 (-0.3 [-1.4; 0.7]) | 96.6 → 96.9 (+0.3 [0.0; 1.0]) |
| khung:sib200|nodiac | 60.2 → 63.2 (+3.0 [-0.3; 6.3]) | 62.2 → 62.5 (+0.3 [-2.3; 3.0]) | 51.2 → 51.5 (+0.3 [-4.3; 5.0]) | 35.8 → 37.1 (+1.3 [-1.3; 4.0]) | 63.9 → 63.9 (+0.0 [-2.7; 2.7]) |
| khung:sib200|orig | 82.6 → 82.9 (+0.3 [-2.0; 2.7]) | 79.9 → 80.9 (+1.0 [-1.0; 3.0]) | 75.6 → 78.3 (+2.7 [-0.3; 5.7]) | 82.3 → 81.9 (-0.3 [-2.0; 1.7]) | 83.6 → 83.6 (+0.0 [-1.3; 1.3]) |
| massive|vi|nodiac|chon_mot | 66.3 → 66.0 (-0.3 [-4.1; 2.7]) | 74.2 → 74.5 (+0.3 [-2.7; 3.4]) | 45.6 → 40.1 (-5.4 [-8.5; -2.4]) | 53.7 → 54.8 (+1.0 [-2.4; 4.1]) | 78.6 → 77.9 (-0.7 [-4.1; 2.4]) |
| massive|vi|orig|chon_mot | 86.7 → 86.4 (-0.3 [-2.4; 2.0]) | 88.1 → 87.4 (-0.7 [-2.4; 0.7]) | 75.2 → 73.1 (-2.0 [-6.1; 2.0]) | 96.3 → 95.9 (-0.3 [-1.4; 0.7]) | 96.6 → 96.9 (+0.3 [0.0; 1.0]) |
| sib200|vi|nodiac|chon_mot | 60.2 → 63.2 (+3.0 [-0.3; 6.3]) | 62.2 → 62.5 (+0.3 [-2.3; 3.0]) | 51.2 → 51.5 (+0.3 [-4.3; 5.0]) | 35.8 → 37.1 (+1.3 [-1.3; 4.0]) | 63.9 → 63.9 (+0.0 [-2.7; 2.7]) |
| sib200|vi|orig|chon_mot | 82.6 → 82.9 (+0.3 [-2.0; 2.7]) | 79.9 → 80.9 (+1.0 [-1.0; 3.0]) | 75.6 → 78.3 (+2.7 [-0.3; 5.7]) | 82.3 → 81.9 (-0.3 [-2.0; 1.7]) | 83.6 → 83.6 (+0.0 [-1.3; 1.3]) |
| tat_ca | 72.5 → 73.1 (+0.6 [-0.6; 1.8]) | 73.8 → 73.3 (-0.5 [-1.6; 0.6]) | 58.1 → 58.1 (+0.0 [-1.8; 1.8]) | 66.8 → 67.8 (+1.0 [-0.3; 2.2]) | 76.3 → 77.0 (+0.6 [-0.4; 1.6]) |
| vifactcheck|vi|nodiac|chon_mot | 48.8 → 53.8 (+5.0 [-4.9; 14.1]) | 60.0 → 57.5 (-2.5 [-13.9; 9.1]) | 30.0 → 42.5 (+12.5 [-1.3; 26.0]) | 33.8 → 47.5 (+13.8 [2.5; 25.9]) | 56.2 → 61.3 (+5.0 [-4.8; 15.2]) |
| vifactcheck|vi|nodiac|noul | 82.0 → 84.6 (+2.6 [-3.8; 9.0]) | 73.1 → 79.5 (+6.4 [1.3; 12.2]) | 60.3 → 75.6 (+15.4 [7.6; 24.1]) | 73.1 → 70.5 (-2.6 [-11.4; 6.3]) | 74.4 → 74.4 (+0.0 [-4.9; 5.1]) |
| vifactcheck|vi|orig|chon_mot | 60.0 → 65.0 (+5.0 [-2.5; 13.6]) | 57.5 → 56.2 (-1.2 [-6.2; 4.0]) | 33.8 → 38.8 (+5.0 [-6.2; 17.1]) | 76.2 → 80.0 (+3.8 [-2.6; 12.2]) | 87.5 → 86.2 (-1.2 [-4.0; 0.0]) |
| vifactcheck|vi|orig|noul | 85.9 → 89.7 (+3.9 [-1.3; 9.6]) | 87.2 → 80.8 (-6.4 [-11.8; -1.3]) | 82.0 → 83.3 (+1.3 [-5.5; 8.0]) | 91.0 → 89.7 (-1.3 [-6.0; 2.7]) | 89.7 → 91.0 (+1.3 [0.0; 4.0]) |
| viwikifc|vi|nodiac|chon_mot | 55.9 → 52.8 (-3.1 [-10.3; 4.3]) | 54.7 → 50.3 (-4.3 [-11.1; 2.5]) | 31.7 → 29.2 (-2.5 [-11.8; 8.0]) | 32.9 → 38.5 (+5.6 [-2.6; 14.2]) | 46.0 → 52.2 (+6.2 [-1.2; 13.8]) |
| viwikifc|vi|nodiac|noul | 79.8 → 77.4 (-2.4 [-6.0; 0.8]) | 72.6 → 73.4 (+0.8 [-4.8; 6.2]) | 68.5 → 71.8 (+3.2 [-2.5; 9.8]) | 67.7 → 70.2 (+2.4 [-4.7; 8.9]) | 62.1 → 61.3 (-0.8 [-5.1; 3.2]) |
| viwikifc|vi|orig|chon_mot | 71.4 → 73.3 (+1.9 [-2.0; 5.8]) | 77.0 → 73.9 (-3.1 [-6.5; -0.6]) | 46.0 → 38.5 (-7.4 [-15.7; 0.6]) | 78.3 → 72.7 (-5.6 [-10.0; -1.3]) | 82.0 → 80.8 (-1.2 [-4.3; 1.9]) |
| viwikifc|vi|orig|noul | 83.1 → 80.7 (-2.4 [-5.8; 0.9]) | 85.5 → 86.3 (+0.8 [0.0; 2.5]) | 79.0 → 77.4 (-1.6 [-6.3; 2.5]) | 88.7 → 90.3 (+1.6 [-1.6; 5.4]) | 80.7 → 83.1 (+2.4 [-0.8; 6.1]) |

### pa

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| massive|vi|orig|chon_mot | 86.7 → 78.6 (-8.2 [-12.6; -3.7]) | 88.1 → 75.5 (-12.6 [-17.3; -7.8]) | 75.2 → 63.3 (-11.9 [-17.7; -6.8]) | 96.3 → 77.2 (-19.1 [-23.8; -14.6]) | 96.6 → 86.7 (-9.9 [-13.6; -6.5]) |
| sib200|vi|orig|chon_mot | 82.6 → 71.6 (-11.0 [-16.4; -6.0]) | 79.9 → 74.2 (-5.7 [-9.7; -1.3]) | 75.6 → 58.2 (-17.4 [-24.1; -11.0]) | 82.3 → 51.2 (-31.1 [-37.5; -25.1]) | 83.6 → 80.3 (-3.3 [-6.7; -0.3]) |
| tat_ca | 80.6 → 74.2 (-6.4 [-8.6; -4.1]) | 81.3 → 72.3 (-9.0 [-11.6; -6.7]) | 68.5 → 59.9 (-8.6 [-11.4; -5.8]) | 85.9 → 66.4 (-19.4 [-22.7; -16.4]) | 88.2 → 83.6 (-4.7 [-6.6; -2.8]) |
| vifactcheck|vi|orig|chon_mot | 60.0 → 57.5 (-2.5 [-11.5; 6.3]) | 57.5 → 57.5 (+0.0 [-9.0; 10.0]) | 33.8 → 45.0 (+11.2 [0.0; 23.1]) | 76.2 → 78.8 (+2.5 [-7.7; 12.8]) | 87.5 → 85.0 (-2.5 [-6.2; 0.0]) |
| vifactcheck|vi|orig|noul | 85.9 → 84.6 (-1.3 [-9.6; 5.5]) | 87.2 → 82.0 (-5.1 [-11.5; 1.4]) | 82.0 → 82.0 (+0.0 [-3.8; 3.9]) | — | — |
| viwikifc|vi|orig|chon_mot | 71.4 → 67.1 (-4.3 [-10.4; 1.2]) | 77.0 → 60.2 (-16.8 [-24.1; -9.3]) | 46.0 → 39.1 (-6.8 [-13.9; 0.0]) | 78.3 → 68.9 (-9.3 [-15.2; -3.4]) | 82.0 → 83.2 (+1.2 [-2.5; 4.5]) |
| viwikifc|vi|orig|noul | 83.1 → 83.9 (+0.8 [-3.1; 5.2]) | 85.5 → 79.0 (-6.5 [-12.1; -1.6]) | 79.0 → 79.0 (+0.0 [-2.4; 2.4]) | — | — |

### pb

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| massive|vi|orig|chon_mot | 86.7 → 88.1 (+1.4 [-2.0; 5.1]) | 88.1 → 89.8 (+1.7 [-1.0; 4.4]) | 75.2 → 78.6 (+3.4 [-1.0; 8.2]) | 96.3 → 95.6 (-0.7 [-2.4; 1.0]) | 96.6 → 96.9 (+0.3 [0.0; 1.0]) |
| sib200|vi|orig|chon_mot | 82.6 → 84.3 (+1.7 [-2.0; 5.7]) | 79.9 → 84.3 (+4.3 [0.7; 8.0]) | 75.6 → 77.6 (+2.0 [-2.3; 6.0]) | 82.3 → 82.9 (+0.7 [-2.3; 3.7]) | 83.6 → 85.3 (+1.7 [-1.0; 4.3]) |
| tat_ca | 84.7 → 86.2 (+1.5 [-1.0; 4.2]) | 84.0 → 87.0 (+3.0 [0.7; 5.4]) | 75.4 → 78.1 (+2.7 [-0.3; 6.1]) | 89.2 → 89.2 (+0.0 [-1.8; 1.7]) | 90.0 → 91.1 (+1.0 [-0.2; 2.4]) |

### p2

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| belebele|en|orig|chon_mot | cùng lựa chọn 90.5; acc 86.7 → 83.7 | cùng lựa chọn 91.5; acc 83.3 → 84.7 | cùng lựa chọn 81.6; acc 80.6 → 74.2 | — | — |
| belebele|vi|nodiac|chon_mot | cùng lựa chọn 79.9; acc 59.2 → 58.5 | cùng lựa chọn 81.6; acc 59.5 → 59.5 | cùng lựa chọn 60.2; acc 49.3 → 48.6 | — | — |
| belebele|vi|orig|chon_mot | cùng lựa chọn 90.1; acc 82.7 → 81.6 | cùng lựa chọn 90.8; acc 84.0 → 82.3 | cùng lựa chọn 76.2; acc 75.2 → 70.4 | — | — |
| massive|en|orig|chon_mot | cùng lựa chọn 92.5; acc 90.1 → 91.8 | cùng lựa chọn 95.2; acc 92.5 → 93.2 | cùng lựa chọn 90.8; acc 90.5 → 89.5 | — | — |
| massive|vi|nodiac|chon_mot | cùng lựa chọn 73.1; acc 66.3 → 65.6 | cùng lựa chọn 77.9; acc 74.2 → 69.7 | cùng lựa chọn 46.6; acc 45.6 → 43.5 | — | — |
| massive|vi|orig|chon_mot | cùng lựa chọn 90.1; acc 86.7 → 84.4 | cùng lựa chọn 92.9; acc 88.1 → 88.4 | cùng lựa chọn 68.4; acc 75.2 → 74.5 | — | — |
| sib200|en|orig|chon_mot | cùng lựa chọn 95.0; acc 84.0 → 83.6 | cùng lựa chọn 94.0; acc 82.9 → 82.9 | cùng lựa chọn 84.0; acc 81.3 → 82.6 | — | — |
| sib200|vi|nodiac|chon_mot | cùng lựa chọn 79.9; acc 60.2 → 60.9 | cùng lựa chọn 84.6; acc 62.2 → 63.9 | cùng lựa chọn 52.8; acc 51.2 → 52.5 | — | — |
| sib200|vi|orig|chon_mot | cùng lựa chọn 92.6; acc 82.6 → 82.9 | cùng lựa chọn 92.6; acc 79.9 → 78.9 | cùng lựa chọn 74.6; acc 75.6 → 74.6 | — | — |
| tat_ca | cùng lựa chọn 88.7; acc 75.8 → 74.9 | cùng lựa chọn 89.6; acc 76.8 → 76.2 | cùng lựa chọn 69.0; acc 65.9 → 63.0 | — | — |
| vifactcheck|vi|nodiac|chon_mot | cùng lựa chọn 92.5; acc 48.8 → 46.2 | cùng lựa chọn 83.8; acc 60.0 → 56.2 | cùng lựa chọn 56.2; acc 30.0 → 27.5 | — | — |
| vifactcheck|vi|nodiac|noul | cùng lựa chọn 94.9; acc 82.0 → 76.9 | cùng lựa chọn 93.6; acc 73.1 → 79.5 | cùng lựa chọn 84.6; acc 60.3 → 50.0 | — | — |
| vifactcheck|vi|orig|chon_mot | cùng lựa chọn 93.8; acc 60.0 → 60.0 | cùng lựa chọn 91.2; acc 57.5 → 61.3 | cùng lựa chọn 62.5; acc 33.8 → 33.8 | — | — |
| vifactcheck|vi|orig|noul | cùng lựa chọn 97.4; acc 85.9 → 85.9 | cùng lựa chọn 98.7; acc 87.2 → 88.5 | cùng lựa chọn 82.0; acc 82.0 → 64.1 | — | — |
| viwikifc|vi|nodiac|chon_mot | cùng lựa chọn 91.9; acc 55.9 → 50.3 | cùng lựa chọn 86.3; acc 54.7 → 49.7 | cùng lựa chọn 59.6; acc 31.7 → 32.9 | — | — |
| viwikifc|vi|nodiac|noul | cùng lựa chọn 96.8; acc 79.8 → 76.6 | cùng lựa chọn 98.4; acc 72.6 → 71.0 | cùng lựa chọn 69.3; acc 68.5 → 47.6 | — | — |
| viwikifc|vi|orig|chon_mot | cùng lựa chọn 93.8; acc 71.4 → 72.0 | cùng lựa chọn 96.9; acc 77.0 → 75.2 | cùng lựa chọn 55.9; acc 46.0 → 48.4 | — | — |
| viwikifc|vi|orig|noul | cùng lựa chọn 99.2; acc 83.1 → 82.3 | cùng lựa chọn 98.4; acc 85.5 → 87.1 | cùng lựa chọn 79.8; acc 79.0 → 66.9 | — | — |
| xcopa|en|orig|chon_mot | cùng lựa chọn 97.0; acc 91.0 → 90.0 | cùng lựa chọn 96.0; acc 94.0 → 92.0 | cùng lựa chọn 89.0; acc 86.0 → 85.0 | — | — |
| xcopa|vi|nodiac|chon_mot | cùng lựa chọn 74.0; acc 57.0 → 59.0 | cùng lựa chọn 62.0; acc 59.0 → 57.0 | cùng lựa chọn 8.0; acc 52.0 → 50.0 | — | — |
| xcopa|vi|orig|chon_mot | cùng lựa chọn 82.0; acc 78.0 → 78.0 | cùng lựa chọn 90.0; acc 85.0 → 79.0 | cùng lựa chọn 82.0; acc 80.0 → 72.0 | — | — |

### p3

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| belebele|en|orig|chon_mot | 45.6 [39.5; 51.6] (dev 86.7; đoán mò 25.0; hay gặp 26.9) | 43.5 [38.1; 49.6] (dev 83.3; đoán mò 25.0; hay gặp 26.9) | 42.9 [37.5; 48.2] (dev 80.6; đoán mò 25.0; hay gặp 26.9) | 51.4 [45.7; 58.0] (dev 88.8; đoán mò 25.0; hay gặp 26.9) | 58.8 [53.3; 64.5] (dev 97.3; đoán mò 25.0; hay gặp 26.9) |
| belebele|vi|nodiac|chon_mot | 31.3 [26.2; 36.3] (dev 59.2; đoán mò 25.0; hay gặp 26.9) | 29.9 [25.1; 34.9] (dev 59.5; đoán mò 25.0; hay gặp 26.9) | 23.5 [18.8; 28.8] (dev 49.3; đoán mò 25.0; hay gặp 26.9) | 28.2 [23.6; 32.8] (dev 53.1; đoán mò 25.0; hay gặp 26.9) | 30.3 [25.8; 34.8] (dev 65.0; đoán mò 25.0; hay gặp 26.9) |
| belebele|vi|orig|chon_mot | 38.8 [33.1; 44.7] (dev 82.7; đoán mò 25.0; hay gặp 26.9) | 42.5 [36.4; 48.6] (dev 84.0; đoán mò 25.0; hay gặp 26.9) | 35.4 [30.0; 41.3] (dev 75.2; đoán mò 25.0; hay gặp 26.9) | 42.2 [36.4; 48.0] (dev 88.1; đoán mò 25.0; hay gặp 26.9) | 51.4 [46.0; 56.9] (dev 94.2; đoán mò 25.0; hay gặp 26.9) |
| vifactcheck|vi|nodiac|chon_mot | 40.0 [29.6; 50.0] (dev 48.8; đoán mò 33.3; hay gặp 42.5) | 48.8 [37.2; 60.0] (dev 60.0; đoán mò 33.3; hay gặp 42.5) | 33.8 [23.2; 45.0] (dev 30.0; đoán mò 33.3; hay gặp 42.5) | 37.5 [28.2; 48.0] (dev 33.8; đoán mò 33.3; hay gặp 42.5) | 32.5 [22.5; 42.7] (dev 56.2; đoán mò 33.3; hay gặp 42.5) |
| vifactcheck|vi|nodiac|noul | 60.3 [48.8; 71.0] (dev 82.0; đoán mò 50.0; hay gặp 64.1) | 47.4 [35.4; 59.2] (dev 73.1; đoán mò 50.0; hay gặp 64.1) | 61.5 [51.2; 71.2] (dev 60.3; đoán mò 50.0; hay gặp 64.1) | 38.5 [27.8; 49.3] (dev 73.1; đoán mò 50.0; hay gặp 64.1) | 44.9 [33.8; 55.6] (dev 74.4; đoán mò 50.0; hay gặp 64.1) |
| vifactcheck|vi|orig|chon_mot | 46.2 [35.9; 56.6] (dev 60.0; đoán mò 33.3; hay gặp 42.5) | 43.8 [32.9; 54.4] (dev 57.5; đoán mò 33.3; hay gặp 42.5) | 27.5 [18.5; 36.7] (dev 33.8; đoán mò 33.3; hay gặp 42.5) | 41.2 [30.8; 51.3] (dev 76.2; đoán mò 33.3; hay gặp 42.5) | 38.8 [29.3; 48.8] (dev 87.5; đoán mò 33.3; hay gặp 42.5) |
| vifactcheck|vi|orig|noul | 64.1 [53.9; 73.8] (dev 85.9; đoán mò 50.0; hay gặp 64.1) | 57.7 [46.3; 68.8] (dev 87.2; đoán mò 50.0; hay gặp 64.1) | 64.1 [53.5; 74.7] (dev 82.0; đoán mò 50.0; hay gặp 64.1) | 62.8 [52.6; 73.6] (dev 91.0; đoán mò 50.0; hay gặp 64.1) | 48.7 [38.5; 59.0] (dev 89.7; đoán mò 50.0; hay gặp 64.1) |
| viwikifc|vi|nodiac|chon_mot | 34.8 [27.8; 42.6] (dev 55.9; đoán mò 33.3; hay gặp 46.6) | 35.4 [28.6; 44.3] (dev 54.7; đoán mò 33.3; hay gặp 46.6) | 24.8 [18.4; 31.6] (dev 31.7; đoán mò 33.3; hay gặp 46.6) | 24.8 [17.8; 31.8] (dev 32.9; đoán mò 33.3; hay gặp 46.6) | 34.2 [26.5; 41.9] (dev 46.0; đoán mò 33.3; hay gặp 46.6) |
| viwikifc|vi|nodiac|noul | 58.1 [50.0; 67.2] (dev 79.8; đoán mò 50.0; hay gặp 62.1) | 53.2 [44.9; 61.2] (dev 72.6; đoán mò 50.0; hay gặp 62.1) | 57.3 [48.8; 65.6] (dev 68.5; đoán mò 50.0; hay gặp 62.1) | 47.6 [38.6; 56.8] (dev 67.7; đoán mò 50.0; hay gặp 62.1) | 41.1 [32.8; 50.0] (dev 62.1; đoán mò 50.0; hay gặp 62.1) |
| viwikifc|vi|orig|chon_mot | 31.1 [24.1; 38.6] (dev 71.4; đoán mò 33.3; hay gặp 46.6) | 34.8 [27.0; 42.8] (dev 77.0; đoán mò 33.3; hay gặp 46.6) | 33.5 [26.2; 40.8] (dev 46.0; đoán mò 33.3; hay gặp 46.6) | 34.8 [27.9; 41.9] (dev 78.3; đoán mò 33.3; hay gặp 46.6) | 44.1 [37.3; 50.6] (dev 82.0; đoán mò 33.3; hay gặp 46.6) |
| viwikifc|vi|orig|noul | 60.5 [52.1; 68.6] (dev 83.1; đoán mò 50.0; hay gặp 62.1) | 62.9 [53.7; 71.3] (dev 85.5; đoán mò 50.0; hay gặp 62.1) | 63.7 [55.1; 72.7] (dev 79.0; đoán mò 50.0; hay gặp 62.1) | 56.5 [48.4; 64.5] (dev 88.7; đoán mò 50.0; hay gặp 62.1) | 53.2 [45.0; 61.3] (dev 80.7; đoán mò 50.0; hay gặp 62.1) |

### p4

| Nhóm | M1 | M2 | M0 | R1 | R2 |
|---|---|---|---|---|---|
| vifactcheck|vi|nodiac|chon_mot | 93.8 [88.0; 98.8] (mốc thiên vị 6.5) | 87.5 [80.2; 94.7] (mốc thiên vị 8.7) | 85.0 [76.6; 91.7] (mốc thiên vị 28.3) | 45.0 [34.6; 56.6] (mốc thiên vị 28.3) | 78.8 [69.4; 87.2] (mốc thiên vị 10.9) |
| vifactcheck|vi|nodiac|noul | 100.0 [100.0; 100.0] (mốc thiên vị 14.3) | 100.0 [100.0; 100.0] (mốc thiên vị 7.1) | 100.0 [100.0; 100.0] (mốc thiên vị 0.0) | 97.4 [93.3; 100.0] (mốc thiên vị 46.4) | 100.0 [100.0; 100.0] (mốc thiên vị 3.6) |
| vifactcheck|vi|orig|chon_mot | 91.2 [84.6; 97.4] (mốc thiên vị 2.2) | 62.5 [52.5; 72.7] (mốc thiên vị 2.2) | 87.5 [79.8; 93.7] (mốc thiên vị 0.0) | 88.8 [81.8; 95.0] (mốc thiên vị 8.7) | 97.5 [93.9; 100.0] (mốc thiên vị 6.5) |
| vifactcheck|vi|orig|noul | 100.0 [100.0; 100.0] (mốc thiên vị 7.1) | 100.0 [100.0; 100.0] (mốc thiên vị 0.0) | 100.0 [100.0; 100.0] (mốc thiên vị 0.0) | 100.0 [100.0; 100.0] (mốc thiên vị 7.1) | 97.4 [93.5; 100.0] (mốc thiên vị 0.0) |
| viwikifc|vi|nodiac|chon_mot | 91.9 [87.4; 96.0] (mốc thiên vị 12.5) | 89.4 [84.0; 94.2] (mốc thiên vị 19.2) | 55.9 [47.8; 64.1] (mốc thiên vị 29.2) | 55.3 [47.8; 62.7] (mốc thiên vị 35.8) | 41.0 [33.7; 48.1] (mốc thiên vị 5.8) |
| viwikifc|vi|nodiac|noul | 100.0 [100.0; 100.0] (mốc thiên vị 21.3) | 99.2 [97.4; 100.0] (mốc thiên vị 34.0) | 99.2 [97.4; 100.0] (mốc thiên vị 21.3) | 86.3 [79.3; 92.3] (mốc thiên vị 25.5) | 80.7 [73.6; 88.2] (mốc thiên vị 0.0) |
| viwikifc|vi|orig|chon_mot | 88.2 [82.9; 92.9] (mốc thiên vị 5.0) | 50.3 [43.2; 57.4] (mốc thiên vị 2.5) | 78.9 [72.7; 84.6] (mốc thiên vị 0.8) | 82.6 [76.3; 88.6] (mốc thiên vị 15.8) | 97.5 [94.9; 99.4] (mốc thiên vị 11.7) |
| viwikifc|vi|orig|noul | 100.0 [100.0; 100.0] (mốc thiên vị 12.8) | 100.0 [100.0; 100.0] (mốc thiên vị 12.8) | 100.0 [100.0; 100.0] (mốc thiên vị 12.8) | 100.0 [100.0; 100.0] (mốc thiên vị 8.5) | 82.3 [75.4; 89.3] (mốc thiên vị 0.0) |

## dev → P5 → P1′ theo nguồn (acc %; chênh P5 − dev và P1′ − P5 theo cặp câu, CI 95% bootstrap theo cụm, p McNemar)

P5 chỉ đổi câu hỏi; P1′ dùng chính câu hỏi P5 và đổi thêm chữ nhãn (chỉ kiểm chứng choice), nên P1′ − P5 là phần do đổi chữ nhãn. Dòng khung:* gộp hai nguồn kiểm chứng; dòng tat_ca của P5 gồm cả 4 khung, còn P1′ − P5 ở dòng đó chỉ tính trên câu kiểm chứng choice.

| Model | Nhóm | n | dev | P5 | P5 − dev | P1′ | P1′ − P5 |
|---|---|---|---|---|---|---|---|
| M1 | khung:kiem_chung_choice|nodiac | 241 | 53.5 | 53.1 | -0.4 [-6.4; 5.2], p 1 | 60.2 | +7.0 [0.4; 14.0], p 0.0568 |
| M1 | khung:kiem_chung_choice|orig | 241 | 67.6 | 70.5 | +2.9 [-0.9; 6.6], p 0.189 | 73.4 | +2.9 [-2.0; 7.9], p 0.324 |
| M1 | khung:kiem_chung_noul|nodiac | 202 | 80.7 | 80.2 | -0.5 [-3.5; 2.7], p 1 | — | — |
| M1 | khung:kiem_chung_noul|orig | 202 | 84.2 | 84.2 | +0.0 [-3.0; 3.1], p 1 | — | — |
| M1 | khung:massive|nodiac | 294 | 66.3 | 66.0 | -0.3 [-4.1; 2.7], p 1 | — | — |
| M1 | khung:massive|orig | 294 | 86.7 | 86.4 | -0.3 [-2.4; 2.0], p 1 | — | — |
| M1 | khung:sib200|nodiac | 299 | 60.2 | 63.2 | +3.0 [-0.3; 6.3], p 0.136 | — | — |
| M1 | khung:sib200|orig | 299 | 82.6 | 82.9 | +0.3 [-2.0; 2.7], p 1 | — | — |
| M1 | massive|vi|nodiac|chon_mot | 294 | 66.3 | 66.0 | -0.3 [-4.1; 2.7], p 1 | — | — |
| M1 | massive|vi|orig|chon_mot | 294 | 86.7 | 86.4 | -0.3 [-2.4; 2.0], p 1 | — | — |
| M1 | sib200|vi|nodiac|chon_mot | 299 | 60.2 | 63.2 | +3.0 [-0.3; 6.3], p 0.136 | — | — |
| M1 | sib200|vi|orig|chon_mot | 299 | 82.6 | 82.9 | +0.3 [-2.0; 2.7], p 1 | — | — |
| M1 | tat_ca | 2072 | 72.5 | 73.1 | +0.6 [-0.6; 1.8], p 0.359 | 66.8 | +5.0 [1.5; 8.7], p 0.0264 |
| M1 | vifactcheck|vi|nodiac|chon_mot | 80 | 48.8 | 53.8 | +5.0 [-4.9; 14.1], p 0.455 | 55.0 | +1.2 [-9.9; 11.7], p 1 |
| M1 | vifactcheck|vi|nodiac|noul | 78 | 82.0 | 84.6 | +2.6 [-3.8; 9.0], p 0.688 | — | — |
| M1 | vifactcheck|vi|orig|chon_mot | 80 | 60.0 | 65.0 | +5.0 [-2.5; 13.6], p 0.344 | 75.0 | +10.0 [1.3; 19.5], p 0.0574 |
| M1 | vifactcheck|vi|orig|noul | 78 | 85.9 | 89.7 | +3.9 [-1.3; 9.6], p 0.375 | — | — |
| M1 | viwikifc|vi|nodiac|chon_mot | 161 | 55.9 | 52.8 | -3.1 [-10.3; 4.3], p 0.5 | 62.7 | +9.9 [1.9; 17.9], p 0.0259 |
| M1 | viwikifc|vi|nodiac|noul | 124 | 79.8 | 77.4 | -2.4 [-6.0; 0.8], p 0.453 | — | — |
| M1 | viwikifc|vi|orig|chon_mot | 161 | 71.4 | 73.3 | +1.9 [-2.0; 5.8], p 0.549 | 72.7 | -0.6 [-6.6; 5.2], p 1 |
| M1 | viwikifc|vi|orig|noul | 124 | 83.1 | 80.7 | -2.4 [-5.8; 0.9], p 0.375 | — | — |
| M2 | khung:kiem_chung_choice|nodiac | 241 | 56.4 | 52.7 | -3.7 [-9.4; 2.4], p 0.314 | 58.1 | +5.4 [-0.9; 11.6], p 0.154 |
| M2 | khung:kiem_chung_choice|orig | 241 | 70.5 | 68.0 | -2.5 [-5.2; 0.4], p 0.146 | 72.6 | +4.6 [1.3; 8.1], p 0.0192 |
| M2 | khung:kiem_chung_noul|nodiac | 202 | 72.8 | 75.7 | +3.0 [-0.9; 6.5], p 0.21 | — | — |
| M2 | khung:kiem_chung_noul|orig | 202 | 86.1 | 84.2 | -2.0 [-4.6; 0.0], p 0.219 | — | — |
| M2 | khung:massive|nodiac | 294 | 74.2 | 74.5 | +0.3 [-2.7; 3.4], p 1 | — | — |
| M2 | khung:massive|orig | 294 | 88.1 | 87.4 | -0.7 [-2.4; 0.7], p 0.688 | — | — |
| M2 | khung:sib200|nodiac | 299 | 62.2 | 62.5 | +0.3 [-2.3; 3.0], p 1 | — | — |
| M2 | khung:sib200|orig | 299 | 79.9 | 80.9 | +1.0 [-1.0; 3.0], p 0.581 | — | — |
| M2 | massive|vi|nodiac|chon_mot | 294 | 74.2 | 74.5 | +0.3 [-2.7; 3.4], p 1 | — | — |
| M2 | massive|vi|orig|chon_mot | 294 | 88.1 | 87.4 | -0.7 [-2.4; 0.7], p 0.688 | — | — |
| M2 | sib200|vi|nodiac|chon_mot | 299 | 62.2 | 62.5 | +0.3 [-2.3; 3.0], p 1 | — | — |
| M2 | sib200|vi|orig|chon_mot | 299 | 79.9 | 80.9 | +1.0 [-1.0; 3.0], p 0.581 | — | — |
| M2 | tat_ca | 2072 | 73.8 | 73.3 | -0.5 [-1.6; 0.6], p 0.474 | 65.3 | +5.0 [1.9; 8.2], p 0.0149 |
| M2 | vifactcheck|vi|nodiac|chon_mot | 80 | 60.0 | 57.5 | -2.5 [-13.9; 9.1], p 0.839 | 57.5 | +0.0 [-11.1; 10.5], p 1 |
| M2 | vifactcheck|vi|nodiac|noul | 78 | 73.1 | 79.5 | +6.4 [1.3; 12.2], p 0.0625 | — | — |
| M2 | vifactcheck|vi|orig|chon_mot | 80 | 57.5 | 56.2 | -1.2 [-6.2; 4.0], p 1 | 66.2 | +10.0 [3.9; 16.7], p 0.0078 |
| M2 | vifactcheck|vi|orig|noul | 78 | 87.2 | 80.8 | -6.4 [-11.8; -1.3], p 0.0625 | — | — |
| M2 | viwikifc|vi|nodiac|chon_mot | 161 | 54.7 | 50.3 | -4.3 [-11.1; 2.5], p 0.337 | 58.4 | +8.1 [0.0; 16.2], p 0.0854 |
| M2 | viwikifc|vi|nodiac|noul | 124 | 72.6 | 73.4 | +0.8 [-4.8; 6.2], p 1 | — | — |
| M2 | viwikifc|vi|orig|chon_mot | 161 | 77.0 | 73.9 | -3.1 [-6.5; -0.6], p 0.125 | 75.8 | +1.9 [-1.7; 5.7], p 0.549 |
| M2 | viwikifc|vi|orig|noul | 124 | 85.5 | 86.3 | +0.8 [0.0; 2.5], p 1 | — | — |
| M0 | khung:kiem_chung_choice|nodiac | 241 | 31.1 | 33.6 | +2.5 [-5.8; 10.8], p 0.617 | 41.1 | +7.5 [-0.4; 14.9], p 0.079 |
| M0 | khung:kiem_chung_choice|orig | 241 | 41.9 | 38.6 | -3.3 [-10.3; 3.3], p 0.396 | 52.3 | +13.7 [7.2; 20.2], p 0.0001 |
| M0 | khung:kiem_chung_noul|nodiac | 202 | 65.3 | 73.3 | +7.9 [2.9; 13.0], p 0.0037 | — | — |
| M0 | khung:kiem_chung_noul|orig | 202 | 80.2 | 79.7 | -0.5 [-4.3; 3.4], p 1 | — | — |
| M0 | khung:massive|nodiac | 294 | 45.6 | 40.1 | -5.4 [-8.5; -2.4], p 0.0015 | — | — |
| M0 | khung:massive|orig | 294 | 75.2 | 73.1 | -2.0 [-6.1; 2.0], p 0.405 | — | — |
| M0 | khung:sib200|nodiac | 299 | 51.2 | 51.5 | +0.3 [-4.3; 5.0], p 1 | — | — |
| M0 | khung:sib200|orig | 299 | 75.6 | 78.3 | +2.7 [-0.3; 5.7], p 0.134 | — | — |
| M0 | massive|vi|nodiac|chon_mot | 294 | 45.6 | 40.1 | -5.4 [-8.5; -2.4], p 0.0015 | — | — |
| M0 | massive|vi|orig|chon_mot | 294 | 75.2 | 73.1 | -2.0 [-6.1; 2.0], p 0.405 | — | — |
| M0 | sib200|vi|nodiac|chon_mot | 299 | 51.2 | 51.5 | +0.3 [-4.3; 5.0], p 1 | — | — |
| M0 | sib200|vi|orig|chon_mot | 299 | 75.6 | 78.3 | +2.7 [-0.3; 5.7], p 0.134 | — | — |
| M0 | tat_ca | 2072 | 58.1 | 58.1 | +0.0 [-1.8; 1.8], p 1 | 46.7 | +10.6 [4.7; 16.3], p 0.0001 |
| M0 | vifactcheck|vi|nodiac|chon_mot | 80 | 30.0 | 42.5 | +12.5 [-1.3; 26.0], p 0.11 | 42.5 | +0.0 [-12.0; 11.5], p 1 |
| M0 | vifactcheck|vi|nodiac|noul | 78 | 60.3 | 75.6 | +15.4 [7.6; 24.1], p 0.0005 | — | — |
| M0 | vifactcheck|vi|orig|chon_mot | 80 | 33.8 | 38.8 | +5.0 [-6.2; 17.1], p 0.523 | 47.5 | +8.8 [0.0; 18.1], p 0.118 |
| M0 | vifactcheck|vi|orig|noul | 78 | 82.0 | 83.3 | +1.3 [-5.5; 8.0], p 1 | — | — |
| M0 | viwikifc|vi|nodiac|chon_mot | 161 | 31.7 | 29.2 | -2.5 [-11.8; 8.0], p 0.716 | 40.4 | +11.2 [1.2; 20.9], p 0.0414 |
| M0 | viwikifc|vi|nodiac|noul | 124 | 68.5 | 71.8 | +3.2 [-2.5; 9.8], p 0.455 | — | — |
| M0 | viwikifc|vi|orig|chon_mot | 161 | 46.0 | 38.5 | -7.4 [-15.7; 0.6], p 0.104 | 54.7 | +16.2 [7.7; 24.8], p 0.0007 |
| M0 | viwikifc|vi|orig|noul | 124 | 79.0 | 77.4 | -1.6 [-6.3; 2.5], p 0.727 | — | — |
| R1 | khung:kiem_chung_choice|nodiac | 241 | 33.2 | 41.5 | +8.3 [0.9; 14.6], p 0.0286 | 48.5 | +7.0 [-0.4; 14.9], p 0.118 |
| R1 | khung:kiem_chung_choice|orig | 241 | 77.6 | 75.1 | -2.5 [-6.5; 1.3], p 0.263 | 73.9 | -1.2 [-6.9; 4.4], p 0.771 |
| R1 | khung:kiem_chung_noul|nodiac | 202 | 69.8 | 70.3 | +0.5 [-4.5; 5.5], p 1 | — | — |
| R1 | khung:kiem_chung_noul|orig | 202 | 89.6 | 90.1 | +0.5 [-2.0; 3.0], p 1 | — | — |
| R1 | khung:massive|nodiac | 294 | 53.7 | 54.8 | +1.0 [-2.4; 4.1], p 0.69 | — | — |
| R1 | khung:massive|orig | 294 | 96.3 | 95.9 | -0.3 [-1.4; 0.7], p 1 | — | — |
| R1 | khung:sib200|nodiac | 299 | 35.8 | 37.1 | +1.3 [-1.3; 4.0], p 0.481 | — | — |
| R1 | khung:sib200|orig | 299 | 82.3 | 81.9 | -0.3 [-2.0; 1.7], p 1 | — | — |
| R1 | massive|vi|nodiac|chon_mot | 294 | 53.7 | 54.8 | +1.0 [-2.4; 4.1], p 0.69 | — | — |
| R1 | massive|vi|orig|chon_mot | 294 | 96.3 | 95.9 | -0.3 [-1.4; 0.7], p 1 | — | — |
| R1 | sib200|vi|nodiac|chon_mot | 299 | 35.8 | 37.1 | +1.3 [-1.3; 4.0], p 0.481 | — | — |
| R1 | sib200|vi|orig|chon_mot | 299 | 82.3 | 81.9 | -0.3 [-2.0; 1.7], p 1 | — | — |
| R1 | tat_ca | 2072 | 66.8 | 67.8 | +1.0 [-0.3; 2.2], p 0.141 | 61.2 | +2.9 [-2.1; 8.3], p 0.292 |
| R1 | vifactcheck|vi|nodiac|chon_mot | 80 | 33.8 | 47.5 | +13.8 [2.5; 25.9], p 0.0433 | 45.0 | -2.5 [-17.7; 11.5], p 0.868 |
| R1 | vifactcheck|vi|nodiac|noul | 78 | 73.1 | 70.5 | -2.6 [-11.4; 6.3], p 0.774 | — | — |
| R1 | vifactcheck|vi|orig|chon_mot | 80 | 76.2 | 80.0 | +3.8 [-2.6; 12.2], p 0.508 | 73.8 | -6.2 [-15.8; 3.6], p 0.302 |
| R1 | vifactcheck|vi|orig|noul | 78 | 91.0 | 89.7 | -1.3 [-6.0; 2.7], p 1 | — | — |
| R1 | viwikifc|vi|nodiac|chon_mot | 161 | 32.9 | 38.5 | +5.6 [-2.6; 14.2], p 0.262 | 50.3 | +11.8 [3.0; 21.1], p 0.0295 |
| R1 | viwikifc|vi|nodiac|noul | 124 | 67.7 | 70.2 | +2.4 [-4.7; 8.9], p 0.629 | — | — |
| R1 | viwikifc|vi|orig|chon_mot | 161 | 78.3 | 72.7 | -5.6 [-10.0; -1.3], p 0.0117 | 73.9 | +1.2 [-5.3; 8.0], p 0.86 |
| R1 | viwikifc|vi|orig|noul | 124 | 88.7 | 90.3 | +1.6 [-1.6; 5.4], p 0.625 | — | — |
| R2 | khung:kiem_chung_choice|nodiac | 241 | 49.4 | 55.2 | +5.8 [-0.8; 12.1], p 0.0595 | 57.7 | +2.5 [-2.4; 7.7], p 0.392 |
| R2 | khung:kiem_chung_choice|orig | 241 | 83.8 | 82.6 | -1.2 [-3.4; 0.8], p 0.453 | 82.2 | -0.4 [-2.9; 2.1], p 1 |
| R2 | khung:kiem_chung_noul|nodiac | 202 | 66.8 | 66.3 | -0.5 [-3.6; 2.9], p 1 | — | — |
| R2 | khung:kiem_chung_noul|orig | 202 | 84.2 | 86.1 | +2.0 [0.0; 4.4], p 0.219 | — | — |
| R2 | khung:massive|nodiac | 294 | 78.6 | 77.9 | -0.7 [-4.1; 2.4], p 0.839 | — | — |
| R2 | khung:massive|orig | 294 | 96.6 | 96.9 | +0.3 [0.0; 1.0], p 1 | — | — |
| R2 | khung:sib200|nodiac | 299 | 63.9 | 63.9 | +0.0 [-2.7; 2.7], p 1 | — | — |
| R2 | khung:sib200|orig | 299 | 83.6 | 83.6 | +0.0 [-1.3; 1.3], p 1 | — | — |
| R2 | massive|vi|nodiac|chon_mot | 294 | 78.6 | 77.9 | -0.7 [-4.1; 2.4], p 0.839 | — | — |
| R2 | massive|vi|orig|chon_mot | 294 | 96.6 | 96.9 | +0.3 [0.0; 1.0], p 1 | — | — |
| R2 | sib200|vi|nodiac|chon_mot | 299 | 63.9 | 63.9 | +0.0 [-2.7; 2.7], p 1 | — | — |
| R2 | sib200|vi|orig|chon_mot | 299 | 83.6 | 83.6 | +0.0 [-1.3; 1.3], p 1 | — | — |
| R2 | tat_ca | 2072 | 76.3 | 77.0 | +0.6 [-0.4; 1.6], p 0.271 | 69.9 | +1.0 [-1.9; 3.8], p 0.542 |
| R2 | vifactcheck|vi|nodiac|chon_mot | 80 | 56.2 | 61.3 | +5.0 [-4.8; 15.2], p 0.455 | 63.7 | +2.5 [-6.9; 12.7], p 0.79 |
| R2 | vifactcheck|vi|nodiac|noul | 78 | 74.4 | 74.4 | +0.0 [-4.9; 5.1], p 1 | — | — |
| R2 | vifactcheck|vi|orig|chon_mot | 80 | 87.5 | 86.2 | -1.2 [-4.0; 0.0], p 1 | 86.2 | +0.0 [-5.6; 4.9], p 1 |
| R2 | vifactcheck|vi|orig|noul | 78 | 89.7 | 91.0 | +1.3 [0.0; 4.0], p 1 | — | — |
| R2 | viwikifc|vi|nodiac|chon_mot | 161 | 46.0 | 52.2 | +6.2 [-1.2; 13.8], p 0.11 | 54.7 | +2.5 [-3.0; 8.3], p 0.503 |
| R2 | viwikifc|vi|nodiac|noul | 124 | 62.1 | 61.3 | -0.8 [-5.1; 3.2], p 1 | — | — |
| R2 | viwikifc|vi|orig|chon_mot | 161 | 82.0 | 80.8 | -1.2 [-4.3; 1.9], p 0.688 | 80.1 | -0.6 [-3.4; 1.8], p 1 |
| R2 | viwikifc|vi|orig|noul | 124 | 80.7 | 83.1 | +2.4 [-0.8; 6.1], p 0.375 | — | — |

## Nhầm nhãn (c): tỉ lệ dự đoán / tỉ lệ đáp án của nhãn theo dõi; TVD

| Model | Nguồn \| tập | TVD | nhãn theo dõi |
|---|---|---|---|
| M1 | sib200|dev|vi|orig | 0.074 | health: 14.7/13.4; sports: 14.1/13.4 |
| M1 | sib200|dev|vi|nodiac | 0.141 | health: 14.4/13.4; sports: 7.4/13.4 |
| M1 | sib200|dev|en|orig | 0.064 | health: 13.4/13.4; sports: 13.7/13.4 |
| M1 | sib200|pa | 0.144 | health: 12.0/13.4; sports: 11.4/13.4 |
| M1 | sib200|pb | 0.060 | health: 13.0/13.4; sports: 13.0/13.4 |
| M1 | massive|dev|vi|orig | 0.088 | recommendation_locations: 1.7/1.7 |
| M1 | massive|dev|vi|nodiac | 0.211 | recommendation_locations: 0.0/1.7 |
| M1 | massive|dev|en|orig | 0.071 | recommendation_locations: 1.4/1.7 |
| M1 | massive|pa | 0.153 | recommendation_locations: 1.7/1.7 |
| M1 | massive|pb | 0.095 | recommendation_locations: 1.0/1.7 |
| M2 | sib200|dev|vi|orig | 0.064 | health: 13.7/13.4; sports: 13.0/13.4 |
| M2 | sib200|dev|vi|nodiac | 0.114 | health: 12.4/13.4; sports: 11.4/13.4 |
| M2 | sib200|dev|en|orig | 0.070 | health: 12.7/13.4; sports: 13.4/13.4 |
| M2 | sib200|pa | 0.110 | health: 11.0/13.4; sports: 14.7/13.4 |
| M2 | sib200|pb | 0.060 | health: 12.0/13.4; sports: 13.4/13.4 |
| M2 | massive|dev|vi|orig | 0.078 | recommendation_locations: 1.4/1.7 |
| M2 | massive|dev|vi|nodiac | 0.140 | recommendation_locations: 2.0/1.7 |
| M2 | massive|dev|en|orig | 0.065 | recommendation_locations: 1.7/1.7 |
| M2 | massive|pa | 0.146 | recommendation_locations: 1.4/1.7 |
| M2 | massive|pb | 0.088 | recommendation_locations: 1.4/1.7 |
| M0 | sib200|dev|vi|orig | 0.080 | health: 13.0/13.4; sports: 11.7/13.4 |
| M0 | sib200|dev|vi|nodiac | 0.241 | health: 7.7/13.4; sports: 11.0/13.4 |
| M0 | sib200|dev|en|orig | 0.070 | health: 11.7/13.4; sports: 12.4/13.4 |
| M0 | sib200|pa | 0.171 | health: 11.0/13.4; sports: 8.7/13.4 |
| M0 | sib200|pb | 0.074 | health: 12.7/13.4; sports: 10.7/13.4 |
| M0 | massive|dev|vi|orig | 0.146 | recommendation_locations: 1.7/1.7 |
| M0 | massive|dev|vi|nodiac | 0.269 | recommendation_locations: 2.0/1.7 |
| M0 | massive|dev|en|orig | 0.085 | recommendation_locations: 1.0/1.7 |
| M0 | massive|pa | 0.194 | recommendation_locations: 2.4/1.7 |
| M0 | massive|pb | 0.143 | recommendation_locations: 1.0/1.7 |
| R1 | sib200|dev|vi|orig | 0.064 | health: 11.7/13.4; sports: 13.7/13.4 |
| R1 | sib200|dev|vi|nodiac | 0.378 | health: 9.7/13.4; sports: 51.2/13.4 |
| R1 | sib200|dev|en|orig | 0.107 | health: 9.7/13.4; sports: 14.7/13.4 |
| R1 | sib200|pa | 0.291 | health: 10.7/13.4; sports: 31.8/13.4 |
| R1 | sib200|pb | 0.087 | health: 10.7/13.4; sports: 13.7/13.4 |
| R1 | massive|dev|vi|orig | 0.031 | recommendation_locations: 1.4/1.7 |
| R1 | massive|dev|vi|nodiac | 0.347 | recommendation_locations: 0.7/1.7 |
| R1 | massive|dev|en|orig | 0.031 | recommendation_locations: 1.4/1.7 |
| R1 | massive|pa | 0.153 | recommendation_locations: 0.0/1.7 |
| R1 | massive|pb | 0.034 | recommendation_locations: 1.4/1.7 |
| R2 | sib200|dev|vi|orig | 0.080 | health: 13.0/13.4; sports: 13.4/13.4 |
| R2 | sib200|dev|vi|nodiac | 0.104 | health: 7.0/13.4; sports: 13.0/13.4 |
| R2 | sib200|dev|en|orig | 0.130 | health: 10.7/13.4; sports: 13.7/13.4 |
| R2 | sib200|pa | 0.127 | health: 11.0/13.4; sports: 13.4/13.4 |
| R2 | sib200|pb | 0.097 | health: 12.4/13.4; sports: 14.4/13.4 |
| R2 | massive|dev|vi|orig | 0.031 | recommendation_locations: 1.4/1.7 |
| R2 | massive|dev|vi|nodiac | 0.122 | recommendation_locations: 0.3/1.7 |
| R2 | massive|dev|en|orig | 0.027 | recommendation_locations: 1.7/1.7 |
| R2 | massive|pa | 0.088 | recommendation_locations: 1.0/1.7 |
| R2 | massive|pb | 0.027 | recommendation_locations: 1.7/1.7 |

## P3 kiểm chứng (d): phân bố nhãn dự đoán khi bỏ văn bản

| Model | Nhóm | đáp án | dự đoán P3 | dự đoán dev cùng câu |
|---|---|---|---|---|
| M1 | vifactcheck|vi|nodiac|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 32.5, refuted 67.5 | nei 12.5, refuted 33.8, supported 53.8 |
| M1 | vifactcheck|vi|nodiac|noul | co 35.9, khong 64.1 | co 16.7, khong 83.3 | co 43.6, khong 56.4 |
| M1 | vifactcheck|vi|orig|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 71.2, refuted 8.8, supported 20.0 | nei 8.8, refuted 47.5, supported 43.8 |
| M1 | vifactcheck|vi|orig|noul | co 35.9, khong 64.1 | co 10.3, khong 89.7 | co 44.9, khong 55.1 |
| M1 | viwikifc|vi|nodiac|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 42.2, refuted 57.8 | nei 18.6, refuted 45.3, supported 36.0 |
| M1 | viwikifc|vi|nodiac|noul | co 37.9, khong 62.1 | co 5.7, khong 94.3 | co 41.9, khong 58.1 |
| M1 | viwikifc|vi|orig|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 72.7, refuted 17.4, supported 9.9 | nei 14.3, refuted 50.3, supported 35.4 |
| M1 | viwikifc|vi|orig|noul | co 37.9, khong 62.1 | co 6.5, khong 93.5 | co 45.2, khong 54.8 |
| M2 | vifactcheck|vi|nodiac|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 47.5, refuted 52.5 | nei 20.0, refuted 50.0, supported 30.0 |
| M2 | vifactcheck|vi|nodiac|noul | co 35.9, khong 64.1 | co 55.1, khong 44.9 | co 57.7, khong 42.3 |
| M2 | vifactcheck|vi|orig|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 76.2, refuted 11.2, supported 12.5 | nei 10.0, refuted 55.0, supported 35.0 |
| M2 | vifactcheck|vi|orig|noul | co 35.9, khong 64.1 | co 14.1, khong 85.9 | co 48.7, khong 51.3 |
| M2 | viwikifc|vi|nodiac|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 47.8, refuted 51.5, supported 0.6 | nei 24.8, refuted 40.4, supported 34.8 |
| M2 | viwikifc|vi|nodiac|noul | co 37.9, khong 62.1 | co 46.0, khong 54.0 | co 39.5, khong 60.5 |
| M2 | viwikifc|vi|orig|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 63.3, refuted 28.6, supported 8.1 | nei 13.0, refuted 50.9, supported 36.0 |
| M2 | viwikifc|vi|orig|noul | co 37.9, khong 62.1 | co 16.9, khong 83.1 | co 42.7, khong 57.3 |
| M0 | vifactcheck|vi|nodiac|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 61.3, refuted 7.5, supported 31.2 | nei 25.0, refuted 10.0, supported 65.0 |
| M0 | vifactcheck|vi|nodiac|noul | co 35.9, khong 64.1 | co 35.9, khong 64.1 | co 75.6, khong 24.4 |
| M0 | vifactcheck|vi|orig|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 12.5, refuted 27.5, supported 60.0 | nei 2.5, refuted 16.2, supported 81.2 |
| M0 | vifactcheck|vi|orig|noul | co 35.9, khong 64.1 | co 23.1, khong 76.9 | co 53.8, khong 46.2 |
| M0 | viwikifc|vi|nodiac|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 66.5, refuted 5.0, supported 28.6 | nei 29.8, refuted 9.3, supported 60.9 |
| M0 | viwikifc|vi|nodiac|noul | co 37.9, khong 62.1 | co 43.5, khong 56.5 | co 53.2, khong 46.8 |
| M0 | viwikifc|vi|orig|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 5.6, refuted 36.0, supported 58.4 | nei 1.9, refuted 27.3, supported 70.8 |
| M0 | viwikifc|vi|orig|noul | co 37.9, khong 62.1 | co 17.7, khong 82.3 | co 49.2, khong 50.8 |
| R1 | vifactcheck|vi|nodiac|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 72.5, refuted 2.5, supported 25.0 | nei 31.2, refuted 2.5, supported 66.2 |
| R1 | vifactcheck|vi|nodiac|noul | co 35.9, khong 64.1 | co 94.9, khong 5.1 | co 29.5, khong 70.5 |
| R1 | vifactcheck|vi|orig|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 78.8, refuted 3.8, supported 17.5 | nei 33.8, refuted 36.2, supported 30.0 |
| R1 | vifactcheck|vi|orig|noul | co 35.9, khong 64.1 | co 55.1, khong 44.9 | co 39.7, khong 60.3 |
| R1 | viwikifc|vi|nodiac|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 81.4, refuted 1.2, supported 17.4 | nei 35.4, refuted 5.0, supported 59.6 |
| R1 | viwikifc|vi|nodiac|noul | co 37.9, khong 62.1 | co 87.1, khong 12.9 | co 50.8, khong 49.2 |
| R1 | viwikifc|vi|orig|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 60.9, refuted 19.2, supported 19.9 | nei 29.8, refuted 41.0, supported 29.2 |
| R1 | viwikifc|vi|orig|noul | co 37.9, khong 62.1 | co 44.4, khong 55.6 | co 42.7, khong 57.3 |
| R2 | vifactcheck|vi|nodiac|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 23.8, refuted 26.2, supported 50.0 | nei 18.8, refuted 27.5, supported 53.8 |
| R2 | vifactcheck|vi|nodiac|noul | co 35.9, khong 64.1 | co 83.3, khong 16.7 | co 59.0, khong 41.0 |
| R2 | vifactcheck|vi|orig|chon_mot | nei 42.5, refuted 35.0, supported 22.5 | nei 42.5, refuted 5.0, supported 52.5 | nei 38.8, refuted 32.5, supported 28.7 |
| R2 | vifactcheck|vi|orig|noul | co 35.9, khong 64.1 | co 84.6, khong 15.4 | co 46.2, khong 53.8 |
| R2 | viwikifc|vi|nodiac|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 17.4, refuted 27.3, supported 55.3 | nei 6.8, refuted 24.8, supported 68.3 |
| R2 | viwikifc|vi|nodiac|noul | co 37.9, khong 62.1 | co 64.5, khong 35.5 | co 75.8, khong 24.2 |
| R2 | viwikifc|vi|orig|chon_mot | nei 25.5, refuted 46.6, supported 28.0 | nei 31.1, refuted 17.4, supported 51.5 | nei 27.3, refuted 40.4, supported 32.3 |
| R2 | viwikifc|vi|orig|noul | co 37.9, khong 62.1 | co 68.5, khong 31.4 | co 57.3, khong 42.7 |

## Số câu bị loại (model ngoài)

```
{"M1": {"dev": {"so_dong": 3847, "bi_loai": 0}, "p1b": {"so_dong": 482, "bi_loai": 0}, "p3": {"so_dong": 1768, "bi_loai": 0}, "p4": {"so_dong": 886, "bi_loai": 0}, "p5": {"so_dong": 2072, "bi_loai": 0}, "pa": {"so_dong": 1036, "bi_loai": 0}, "pb": {"so_dong": 593, "bi_loai": 0}, "p2": {"so_dong": 3847, "bi_loai": 0}}, "M2": {"dev": {"so_dong": 3847, "bi_loai": 0}, "p1b": {"so_dong": 482, "bi_loai": 0}, "p3": {"so_dong": 1768, "bi_loai": 0}, "p4": {"so_dong": 886, "bi_loai": 0}, "p5": {"so_dong": 2072, "bi_loai": 0}, "pa": {"so_dong": 1036, "bi_loai": 0}, "pb": {"so_dong": 593, "bi_loai": 0}, "p2": {"so_dong": 3847, "bi_loai": 0}}, "M0": {"dev": {"so_dong": 3847, "bi_loai": 0}, "p1b": {"so_dong": 482, "bi_loai": 0}, "p3": {"so_dong": 1768, "bi_loai": 0}, "p4": {"so_dong": 886, "bi_loai": 0}, "p5": {"so_dong": 2072, "bi_loai": 0}, "pa": {"so_dong": 1036, "bi_loai": 0}, "pb": {"so_dong": 593, "bi_loai": 0}, "p2": {"so_dong": 3847, "bi_loai": 0}}, "R1": {"dev": {"so_dong": 3847, "bi_loai": 0}, "p1b": {"so_dong": 482, "bi_loai": 0}, "p3": {"so_dong": 1768, "bi_loai": 0}, "p4": {"so_dong": 886, "bi_loai": 0}, "p5": {"so_dong": 2072, "bi_loai": 0}, "pa": {"so_dong": 834, "bi_loai": 0}, "pb": {"so_dong": 593, "bi_loai": 0}}, "R2": {"dev": {"so_dong": 3847, "bi_loai": 0}, "p1b": {"so_dong": 482, "bi_loai": 0}, "p3": {"so_dong": 1768, "bi_loai": 0}, "p4": {"so_dong": 886, "bi_loai": 0}, "p5": {"so_dong": 2072, "bi_loai": 0}, "pa": {"so_dong": 834, "bi_loai": 0}, "pb": {"so_dong": 593, "bi_loai": 0}}}
```
