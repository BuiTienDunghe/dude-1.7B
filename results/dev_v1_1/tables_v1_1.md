dev_v1_1 SHA-256 `18ba27b4eb1fd5da…` (khớp manifest). Ánh xạ dùng cho decider: decider-2b: text (TB năm lát text 70.8 / keyed 70.1); decider-ft ep2: text (TB năm lát text 72.5 / keyed 72.4)

## 1. Năm lát chính — accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]

| Lát | n | decider-2b | adapter đối chứng | Qwen3-1.7B gốc | strip (Qwen3) | decider-ft ep2 |
|---|---|---|---|---|---|---|
| song song · EN | 1000 | 89.4 [87.5; 91.3] | 86.8 [84.4; 89.0] | 83.5 [80.9; 85.9] | 87.1 [84.7; 89.2] | 88.7 [86.5; 90.6] |
| song song · VI có dấu | 1000 | 86.2 [83.7; 88.4] | 80.4 [77.5; 83.1] | 72.8 [69.7; 75.7] | 81.2 [78.4; 83.9] | 84.6 [82.1; 87.1] |
| song song · VI không dấu | 1000 | 47.9 [44.6; 51.3] | 59.0 [55.8; 62.4] | 47.8 [44.2; 51.3] | 60.4 [57.0; 63.8] | 51.0 [47.7; 54.4] |
| bản địa · có dấu | 463 | 81.2 [77.5; 84.8] | 72.9 [68.8; 76.9] | 57.4 [52.9; 62.0] | 73.6 [69.3; 77.8] | 81.0 [77.4; 84.4] |
| bản địa · không dấu | 463 | 49.4 [44.8; 53.9] | 65.4 [61.3; 69.9] | 46.3 [41.3; 51.2] | 64.0 [59.5; 68.5] | 57.4 [52.8; 62.1] |

## 2. Chênh lệch theo cặp (điểm, [CI 95%], McNemar p)

| Lát | So sánh | Chênh |
|---|---|---|
| song song · EN | adapter đối chứng − decider-2b | -2.7 [-5.0; -0.5] (p=0.0059) |
| song song · EN | Qwen3-1.7B gốc − decider-2b | -5.9 [-8.8; -3.1] (p=3.3e-06) |
| song song · EN | strip (Qwen3) − decider-2b | -2.3 [-4.4; -0.4] (p=0.0051) |
| song song · EN | decider-ft ep2 − decider-2b | -0.8 [-2.2; +0.8] (p=0.27) |
| song song · EN | decider-2b − adapter đối chứng | +2.7 [+0.5; +5.0] (p=0.0059) |
| song song · EN | Qwen3-1.7B gốc − adapter đối chứng | -3.2 [-5.3; -1.1] (p=0.0055) |
| song song · EN | strip (Qwen3) − adapter đối chứng | +0.3 [-0.8; +1.5] (p=0.88) |
| song song · EN | decider-ft ep2 − adapter đối chứng | +1.9 [-0.4; +4.3] (p=0.055) |
| song song · VI có dấu | adapter đối chứng − decider-2b | -5.8 [-8.5; -3.0] (p=7.7e-06) |
| song song · VI có dấu | Qwen3-1.7B gốc − decider-2b | -13.4 [-16.5; -10.1] (p=8.3e-23) |
| song song · VI có dấu | strip (Qwen3) − decider-2b | -4.9 [-7.8; -2.0] (p=0.0001) |
| song song · VI có dấu | decider-ft ep2 − decider-2b | -1.6 [-3.5; +0.3] (p=0.1) |
| song song · VI có dấu | decider-2b − adapter đối chứng | +5.8 [+3.0; +8.5] (p=7.7e-06) |
| song song · VI có dấu | Qwen3-1.7B gốc − adapter đối chứng | -7.7 [-10.5; -4.8] (p=7.1e-12) |
| song song · VI có dấu | strip (Qwen3) − adapter đối chứng | +0.8 [-1.3; +2.8] (p=0.41) |
| song song · VI có dấu | decider-ft ep2 − adapter đối chứng | +4.2 [+1.4; +7.1] (p=0.0017) |
| song song · VI không dấu | adapter đối chứng − decider-2b | +11.1 [+7.1; +15.3] (p=1.6e-11) |
| song song · VI không dấu | Qwen3-1.7B gốc − decider-2b | -0.1 [-5.0; +4.9] (p=0.92) |
| song song · VI không dấu | strip (Qwen3) − decider-2b | +12.5 [+8.5; +16.7] (p=1.8e-13) |
| song song · VI không dấu | decider-ft ep2 − decider-2b | +3.1 [-0.4; +6.6] (p=0.15) |
| song song · VI không dấu | decider-2b − adapter đối chứng | -11.1 [-15.3; -7.1] (p=1.6e-11) |
| song song · VI không dấu | Qwen3-1.7B gốc − adapter đối chứng | -11.2 [-15.2; -7.1] (p=1.7e-12) |
| song song · VI không dấu | strip (Qwen3) − adapter đối chứng | +1.4 [-2.2; +4.8] (p=0.31) |
| song song · VI không dấu | decider-ft ep2 − adapter đối chứng | -8.0 [-12.3; -3.6] (p=1.3e-07) |
| bản địa · có dấu | adapter đối chứng − decider-2b | -8.3 [-12.8; -3.7] (p=0.00071) |
| bản địa · có dấu | Qwen3-1.7B gốc − decider-2b | -23.8 [-28.9; -18.8] (p=1.1e-19) |
| bản địa · có dấu | strip (Qwen3) − decider-2b | -7.7 [-12.4; -3.0] (p=0.0032) |
| bản địa · có dấu | decider-ft ep2 − decider-2b | -0.3 [-3.1; +2.6] (p=1) |
| bản địa · có dấu | decider-2b − adapter đối chứng | +8.3 [+3.7; +12.8] (p=0.00071) |
| bản địa · có dấu | Qwen3-1.7B gốc − adapter đối chứng | -15.5 [-20.1; -11.1] (p=1.9e-12) |
| bản địa · có dấu | strip (Qwen3) − adapter đối chứng | +0.6 [-2.1; +3.3] (p=0.74) |
| bản địa · có dấu | decider-ft ep2 − adapter đối chứng | +8.0 [+4.1; +11.9] (p=0.00015) |
| bản địa · không dấu | adapter đối chứng − decider-2b | +16.0 [+10.5; +21.8] (p=2.8e-09) |
| bản địa · không dấu | Qwen3-1.7B gốc − decider-2b | -3.0 [-8.4; +2.6] (p=0.5) |
| bản địa · không dấu | strip (Qwen3) − decider-2b | +14.6 [+8.6; +20.5] (p=5.9e-08) |
| bản địa · không dấu | decider-ft ep2 − decider-2b | +8.0 [+3.6; +12.8] (p=0.0051) |
| bản địa · không dấu | decider-2b − adapter đối chứng | -16.0 [-21.8; -10.5] (p=2.8e-09) |
| bản địa · không dấu | Qwen3-1.7B gốc − adapter đối chứng | -19.1 [-24.9; -13.5] (p=1.9e-12) |
| bản địa · không dấu | strip (Qwen3) − adapter đối chứng | -1.4 [-5.5; +2.3] (p=0.58) |
| bản địa · không dấu | decider-ft ep2 − adapter đối chứng | -8.0 [-13.3; -3.0] (p=0.00024) |

## 3. Theo nguồn — accuracy (%)

| Lát · nguồn | decider-2b | adapter đối chứng | Qwen3-1.7B gốc | strip (Qwen3) | decider-ft ep2 |
|---|---|---|---|---|---|
| song song · EN · belebele | 87.3 | 84.7 | 79.0 | 84.3 | 87.0 |
| song song · EN · massive | 95.0 | 88.0 | 88.3 | 89.0 | 94.0 |
| song song · EN · sib200 | 83.3 | 83.3 | 80.7 | 83.0 | 81.7 |
| song song · EN · xcopa | 92.0 | 91.0 | 86.0 | 92.0 | 92.0 |
| song song · VI có dấu · belebele | 83.0 | 77.3 | 67.3 | 78.3 | 82.0 |
| song song · VI có dấu · massive | 93.3 | 83.7 | 70.7 | 86.0 | 91.3 |
| song song · VI có dấu · sib200 | 82.3 | 81.7 | 74.0 | 80.7 | 81.0 |
| song song · VI có dấu · xcopa | 86.0 | 79.0 | 79.0 | 80.0 | 84.0 |
| song song · VI không dấu · belebele | 49.0 | 58.3 | 46.0 | 60.7 | 45.0 |
| song song · VI không dấu · massive | 53.7 | 63.3 | 43.0 | 65.0 | 62.7 |
| song song · VI không dấu · sib200 | 36.0 | 58.3 | 51.3 | 59.0 | 36.3 |
| song song · VI không dấu · xcopa | 53.0 | 56.0 | 51.0 | 57.0 | 60.0 |
| bản địa · có dấu · vifactcheck | 82.8 | 70.6 | 55.8 | 71.2 | 81.0 |
| bản địa · có dấu · viwikifc | 79.7 | 75.3 | 59.0 | 76.0 | 81.0 |
| bản địa · không dấu · vifactcheck | 52.8 | 63.8 | 46.0 | 62.0 | 63.2 |
| bản địa · không dấu · viwikifc | 46.0 | 67.0 | 46.7 | 66.0 | 51.7 |

## 4. Theo kiểu câu (khối bản địa; accuracy %, trung bình đều theo nguồn)

| Lát | n | decider-2b | adapter đối chứng | Qwen3-1.7B gốc | strip (Qwen3) | decider-ft ep2 |
|---|---|---|---|---|---|---|
| bản địa · choice · có dấu | 256 | 74.8 | 62.7 | 37.2 | 63.9 | 74.2 |
| bản địa · choice · không dấu | 256 | 32.9 | 51.6 | 31.1 | 52.8 | 44.0 |
| bản địa · noul · có dấu | 207 | 88.8 | 84.7 | 81.2 | 84.8 | 89.2 |
| bản địa · noul · không dấu | 207 | 69.2 | 81.6 | 64.9 | 77.0 | 73.6 |

## 5. Mức sàn (mục D) — dòng tiếng Việt có dấu, tính trên ĐÁP ÁN

| Nguồn · kiểu câu | n | số lựa chọn | ngẫu nhiên % | luôn chọn đáp án (chữ) phổ biến nhất % | đáp án đó | luôn chọn vị trí phổ biến nhất % | vị trí |
|---|---|---|---|---|---|---|---|
| belebele|reading_mc | 300 | {4: 300} | 25.0 | 0.7 | 4 | 27.0 | A |
| massive|intent | 300 | {10: 300} | 10.0 | 2.0 | tìm công thức nấu ăn | 13.0 | G |
| sib200|topic | 300 | {7: 300} | 14.3 | 18.0 | khoa học/công nghệ | 18.0 | C |
| vifactcheck|choice | 85 | {3: 85} | 33.3 | 43.5 | Văn bản không đủ thông tin | 40.0 | A |
| vifactcheck|noul | 78 | {2: 78} | 50.0 | 62.8 | Không | 62.8 | B |
| viwikifc|choice | 171 | {3: 171} | 33.3 | 46.2 | Bị văn bản bác bỏ | 35.1 | A |
| viwikifc|noul | 129 | {2: 129} | 50.0 | 61.2 | Không | 61.2 | B |
| xcopa|causal | 100 | {2: 100} | 50.0 | 1.0 | Nước chảy ra từ vòi. | 51.0 | A |

## 6. Câu noul của khối bản địa — tỉ lệ dự đoán "Có" và AUROC của P(Có)

| Biến thể | n | tỉ lệ đáp án "Có" | decider-2b: dự đoán Có % / AUROC / acc % | adapter đối chứng: dự đoán Có % / AUROC / acc % | Qwen3-1.7B gốc: dự đoán Có % / AUROC / acc % | strip (Qwen3): dự đoán Có % / AUROC / acc % | decider-ft ep2: dự đoán Có % / AUROC / acc % |
|---|---|---|---|---|---|---|---|
| có dấu | 207 | 38.2 | 42.5 / 0.936 / 87.9 | 45.4 / 0.940 / 84.1 | 51.7 / 0.868 / 80.7 | 47.8 / 0.939 / 84.5 | 43.0 / 0.952 / 89.4 |
| không dấu | 207 | 38.2 | 43.5 / 0.718 / 68.6 | 43.5 / 0.848 / 81.2 | 62.8 / 0.761 / 65.7 | 46.4 / 0.841 / 76.3 | 45.4 / 0.795 / 72.5 |

## 7. Đọc lại kết luận cũ trên bản 1.1 (mục F)

**strip_v0** (adapter bỏ dấu − adapter đối chứng; ngưỡng: không dấu tăng ≥ 5 ở song song và bản địa, CI không chứa 0; VI có dấu và EN giảm ≤ 2):

| Chỉ số | bản 1.0 | bản 1.1 |
|---|---|---|
| nodiac_parallel_gain | +2.0 | +1.4 [-2.1; +4.9] |
| nodiac_native_gain | +3.0 | -1.4 [-5.1; +2.3] |
| vi_orig_parallel_change | +1.1 | +0.8 [-1.3; +2.9] |
| vi_orig_native_change | +0.1 | +0.6 [-2.0; +3.4] |
| en_change | +0.4 | +0.3 [-0.8; +1.5] |
| Kết luận | KHÔNG ĐẠT | KHÔNG ĐẠT |

**ft_decider_v0** (decider-ft ep2 − decider-2b, cùng ánh xạ "text" như báo cáo gốc):

| Chỉ số | bản 1.0 | bản 1.1 |
|---|---|---|
| en | -0.7 | -0.8 |
| vi_orig_parallel | -1.8 | -1.6 [-3.6; +0.2] |
| native_orig | +4.8 | -0.3 [-3.3; +2.6] |
| nodiac_parallel | +1.4 | +3.1 |
| Kết luận | ĐÁNG THEO | CÒN LẠI: báo số, không kết luận |

**restore** (decider-2b: có dấu − khôi phục, khối song song):

| Bộ | bản 1.0, 4 nguồn (lát định trước) | bản 1.1 | chỉ Belebele + SIB + XCOPA (như nhau ở 1.0 và 1.1) |
|---|---|---|---|
| vit5 | 4.1 | không tính lại | 3.1 |
| small | 11.8 | không tính lại | 11.4 |

MASSIVE và kiểm chứng đổi chữ: bản khôi phục cũ không còn khớp; KHÔNG tính lại (không chạy vit5/small). Con số 3 nguồn giống hệt nhau ở 1.0 và 1.1 vì ba nguồn này không đổi; không so được với ngưỡng định cho 4 nguồn.
