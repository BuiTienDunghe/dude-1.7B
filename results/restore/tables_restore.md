dev_v1 SHA-256 khớp manifest: `894558631429840e…`. 1.763 câu tiếng Việt (1.000 song song, 763 bản địa).
Kết quả chính: **khôi phục <bộ>** = đầu ra thô của bộ khôi phục. **đã sửa <bộ>** = kiểm độ nhạy (chiếu về đầu vào không dấu).

## 1. Accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]

| Khối | Model | có dấu | không dấu | khôi phục vit5 | khôi phục small | đã sửa vit5 | đã sửa small |
|---|---|---|---|---|---|---|---|
| song song (4 nguồn) | control | 79.8 [76.9; 82.8] | 58.8 [55.6; 62.2] | 74.5 [71.4; 77.6] | 69.0 [65.9; 72.1] | 74.6 [71.7; 77.5] | 69.5 [66.4; 72.6] |
| song song (4 nguồn) | strip | 80.8 [78.1; 83.5] | 60.8 [57.5; 64.2] | 74.9 [71.8; 77.8] | 70.2 [67.1; 73.3] | 74.8 [71.8; 78.0] | 70.2 [67.2; 73.3] |
| song song (4 nguồn) | decider-2b | 86.0 [83.7; 88.5] | 49.2 [45.7; 52.6] | 81.9 [79.1; 84.6] | 74.2 [71.2; 77.1] | 81.8 [79.0; 84.4] | 75.8 [72.9; 78.8] |
| song song · chỉ MASSIVE + XCOPA | control | 80.0 [75.3; 84.3] | 59.3 [53.7; 65.0] | 73.0 [68.0; 77.8] | 67.7 [62.3; 72.8] | 72.8 [67.7; 77.7] | 68.2 [62.8; 73.3] |
| song song · chỉ MASSIVE + XCOPA | strip | 82.2 [77.7; 86.7] | 61.8 [56.0; 67.7] | 74.0 [69.0; 79.0] | 68.8 [63.3; 74.0] | 73.7 [68.3; 78.5] | 69.8 [64.3; 74.7] |
| song song · chỉ MASSIVE + XCOPA | decider-2b | 89.3 [85.3; 93.0] | 55.8 [50.3; 61.3] | 83.8 [79.3; 88.0] | 76.8 [72.0; 81.7] | 83.8 [79.0; 88.0] | 77.5 [72.5; 82.2] |
| bản địa (3 nguồn) | control | 66.2 [62.8; 69.2] | 53.9 [50.2; 57.1] | 63.7 [60.3; 66.9] | 57.4 [54.1; 60.8] | 63.5 [60.0; 66.7] | 56.9 [53.5; 60.3] |
| bản địa (3 nguồn) | strip | 66.3 [63.0; 69.7] | 56.9 [53.5; 60.6] | 64.5 [61.1; 67.9] | 63.5 [60.0; 67.0] | 64.4 [60.8; 67.9] | 64.3 [60.9; 67.5] |
| bản địa (3 nguồn) | decider-2b | 66.1 [62.9; 69.2] | 42.9 [39.3; 46.6] | 64.5 [61.4; 67.7] | 60.7 [57.3; 63.9] | 64.4 [61.2; 67.6] | 63.9 [60.9; 67.1] |

## 2. Chênh lệch theo cặp (điểm %, [CI 95%], McNemar p)

| Khối | Model | Biến thể | − không dấu | − có dấu |
|---|---|---|---|---|
| song song (4 nguồn) | control | khôi phục vit5 | +15.7 [+11.9; +19.5] (p=4.8e-20) | -5.3 [-7.8; -2.8] (p=2.5e-07) |
| song song (4 nguồn) | control | khôi phục small | +10.2 [+6.2; +14.0] (p=1.7e-08) | -10.8 [-13.4; -8.2] (p=6.7e-17) |
| song song (4 nguồn) | control | đã sửa vit5 | +15.8 [+12.3; +19.5] (p=1.7e-20) | -5.2 [-7.5; -2.7] (p=2.8e-07) |
| song song (4 nguồn) | control | đã sửa small | +10.7 [+6.8; +14.6] (p=2.5e-09) | -10.2 [-12.9; -7.7] (p=7.9e-16) |
| song song (4 nguồn) | strip | khôi phục vit5 | +14.1 [+10.5; +18.1] (p=5.6e-16) | -5.9 [-8.4; -3.4] (p=1.9e-09) |
| song song (4 nguồn) | strip | khôi phục small | +9.4 [+5.5; +13.5] (p=3.9e-07) | -10.6 [-13.3; -7.9] (p=4e-17) |
| song song (4 nguồn) | strip | đã sửa vit5 | +14.0 [+10.0; +17.9] (p=1.3e-15) | -6.0 [-8.4; -3.7] (p=2.6e-10) |
| song song (4 nguồn) | strip | đã sửa small | +9.4 [+5.5; +13.4] (p=2.5e-07) | -10.6 [-13.3; -8.0] (p=1.3e-17) |
| song song (4 nguồn) | decider-2b | khôi phục vit5 | +32.8 [+29.1; +36.4] (p=3.2e-71) | -4.1 [-6.3; -2.0] (p=2.8e-07) |
| song song (4 nguồn) | decider-2b | khôi phục small | +25.0 [+21.2; +28.9] (p=1.9e-39) | -11.8 [-14.7; -8.9] (p=2.2e-24) |
| song song (4 nguồn) | decider-2b | đã sửa vit5 | +32.6 [+28.7; +36.2] (p=1.1e-70) | -4.2 [-6.5; -2.1] (p=6e-08) |
| song song (4 nguồn) | decider-2b | đã sửa small | +26.7 [+22.9; +30.5] (p=1.4e-48) | -10.2 [-12.9; -7.5] (p=1.4e-18) |
| song song · chỉ MASSIVE + XCOPA | control | khôi phục vit5 | +13.7 [+7.2; +20.0] (p=4.7e-06) | -7.0 [-11.5; -2.3] (p=0.00031) |
| song song · chỉ MASSIVE + XCOPA | control | khôi phục small | +8.3 [+2.0; +15.2] (p=0.0084) | -12.3 [-16.8; -7.8] (p=4.5e-09) |
| song song · chỉ MASSIVE + XCOPA | control | đã sửa vit5 | +13.5 [+7.0; +19.5] (p=6.8e-06) | -7.2 [-11.7; -2.5] (p=0.00015) |
| song song · chỉ MASSIVE + XCOPA | control | đã sửa small | +8.8 [+2.3; +15.3] (p=0.007) | -11.8 [-16.2; -7.5] (p=7.6e-09) |
| song song · chỉ MASSIVE + XCOPA | strip | khôi phục vit5 | +12.2 [+5.7; +19.0] (p=0.00017) | -8.2 [-12.5; -3.7] (p=5.1e-06) |
| song song · chỉ MASSIVE + XCOPA | strip | khôi phục small | +7.0 [+0.3; +13.8] (p=0.085) | -13.3 [-18.2; -8.7] (p=5.8e-12) |
| song song · chỉ MASSIVE + XCOPA | strip | đã sửa vit5 | +11.8 [+5.3; +18.5] (p=0.0004) | -8.5 [-13.0; -4.0] (p=1.2e-06) |
| song song · chỉ MASSIVE + XCOPA | strip | đã sửa small | +8.0 [+1.2; +14.5] (p=0.022) | -12.3 [-16.8; -8.0] (p=3.9e-10) |
| song song · chỉ MASSIVE + XCOPA | decider-2b | khôi phục vit5 | +28.0 [+21.5; +34.3] (p=1.2e-23) | -5.5 [-9.3; -1.5] (p=7e-05) |
| song song · chỉ MASSIVE + XCOPA | decider-2b | khôi phục small | +21.0 [+14.7; +27.5] (p=4.2e-13) | -12.5 [-17.3; -8.0] (p=1.6e-11) |
| song song · chỉ MASSIVE + XCOPA | decider-2b | đã sửa vit5 | +28.0 [+21.5; +34.0] (p=1.2e-23) | -5.5 [-9.5; -1.5] (p=7e-05) |
| song song · chỉ MASSIVE + XCOPA | decider-2b | đã sửa small | +21.7 [+15.8; +27.8] (p=5.6e-16) | -11.8 [-16.3; -7.2] (p=5.7e-10) |
| bản địa (3 nguồn) | control | khôi phục vit5 | +9.8 [+6.3; +13.4] (p=2.1e-07) | -2.5 [-4.6; -0.4] (p=0.008) |
| bản địa (3 nguồn) | control | khôi phục small | +3.6 [-0.0; +7.5] (p=0.024) | -8.8 [-11.8; -5.5] (p=7.3e-08) |
| bản địa (3 nguồn) | control | đã sửa vit5 | +9.6 [+6.0; +13.2] (p=3.3e-07) | -2.7 [-4.9; -0.7] (p=0.0049) |
| bản địa (3 nguồn) | control | đã sửa small | +3.0 [-0.5; +6.7] (p=0.056) | -9.3 [-12.4; -6.5] (p=6.1e-10) |
| bản địa (3 nguồn) | strip | khôi phục vit5 | +7.6 [+3.6; +11.4] (p=0.00045) | -1.8 [-4.4; +0.7] (p=0.076) |
| bản địa (3 nguồn) | strip | khôi phục small | +6.6 [+2.0; +10.9] (p=0.0017) | -2.8 [-6.0; +0.5] (p=0.073) |
| bản địa (3 nguồn) | strip | đã sửa vit5 | +7.5 [+3.3; +11.2] (p=0.00066) | -1.9 [-4.4; +0.4] (p=0.04) |
| bản địa (3 nguồn) | strip | đã sửa small | +7.4 [+3.6; +11.1] (p=0.00078) | -2.0 [-5.0; +0.9] (p=0.053) |
| bản địa (3 nguồn) | decider-2b | khôi phục vit5 | +21.6 [+17.2; +26.1] (p=2.5e-21) | -1.6 [-3.8; +0.6] (p=0.1) |
| bản địa (3 nguồn) | decider-2b | khôi phục small | +17.7 [+13.7; +21.9] (p=2.2e-16) | -5.4 [-8.5; -2.5] (p=0.0006) |
| bản địa (3 nguồn) | decider-2b | đã sửa vit5 | +21.5 [+17.3; +25.8] (p=2.7e-21) | -1.7 [-3.9; +0.5] (p=0.076) |
| bản địa (3 nguồn) | decider-2b | đã sửa small | +21.0 [+16.6; +25.4] (p=2.1e-21) | -2.2 [-5.0; +0.5] (p=0.092) |

## 3. Cách đọc đã định trước — decider-2b: khoảng cách có dấu − khôi phục (điểm)

Áp cho khối song song đủ 4 nguồn và đầu ra thô. Các dòng khác (đã sửa; chỉ MASSIVE + XCOPA; bản địa) chỉ để đối chiếu.

| Khối | Biến thể | Khoảng cách [CI 95%] | Vùng |
|---|---|---|---|
| song song (4 nguồn) | khôi phục vit5 | 4.1 [2.0; 6.3] | ≤ 5: lợi thế không dấu không còn khi so theo pipeline |
| song song (4 nguồn) | khôi phục small | 11.8 [8.9; 14.7] | > 10: khả năng chịu không dấu của model vẫn có giá trị |
| song song (4 nguồn) | đã sửa vit5 | 4.2 [2.1; 6.5] | ≤ 5: lợi thế không dấu không còn khi so theo pipeline |
| song song (4 nguồn) | đã sửa small | 10.2 [7.5; 12.9] | > 10: khả năng chịu không dấu của model vẫn có giá trị |
| song song · chỉ MASSIVE + XCOPA | khôi phục vit5 | 5.5 [1.5; 9.3] | ở giữa (5–10): chỉ báo số, không kết luận |
| song song · chỉ MASSIVE + XCOPA | khôi phục small | 12.5 [8.0; 17.3] | > 10: khả năng chịu không dấu của model vẫn có giá trị |
| song song · chỉ MASSIVE + XCOPA | đã sửa vit5 | 5.5 [1.5; 9.5] | ở giữa (5–10): chỉ báo số, không kết luận |
| song song · chỉ MASSIVE + XCOPA | đã sửa small | 11.8 [7.2; 16.3] | > 10: khả năng chịu không dấu của model vẫn có giá trị |
| bản địa (3 nguồn) | khôi phục vit5 | 1.6 [-0.6; 3.8] | ≤ 5: lợi thế không dấu không còn khi so theo pipeline |
| bản địa (3 nguồn) | khôi phục small | 5.4 [2.5; 8.5] | ở giữa (5–10): chỉ báo số, không kết luận |
| bản địa (3 nguồn) | đã sửa vit5 | 1.7 [-0.5; 3.9] | ≤ 5: lợi thế không dấu không còn khi so theo pipeline |
| bản địa (3 nguồn) | đã sửa small | 2.2 [-0.5; 5.0] | ≤ 5: lợi thế không dấu không còn khi so theo pipeline |

## 4. So theo pipeline: model dự án − decider-2b sau khôi phục (cùng câu; điểm, [CI], p)

| Khối | So sánh | Chênh |
|---|---|---|
| song song (4 nguồn) | control không dấu − decider-2b khôi phục vit5 | -23.1 [-27.2; -19.0] (p=2.9e-35) |
| song song (4 nguồn) | control khôi phục vit5 − decider-2b khôi phục vit5 | -7.4 [-10.4; -4.4] (p=1.9e-07) |
| song song (4 nguồn) | strip không dấu − decider-2b khôi phục vit5 | -21.1 [-25.0; -17.1] (p=2.1e-30) |
| song song (4 nguồn) | strip khôi phục vit5 − decider-2b khôi phục vit5 | -7.0 [-10.0; -3.9] (p=6.2e-07) |
| song song (4 nguồn) | control không dấu − decider-2b khôi phục small | -15.3 [-19.3; -11.1] (p=5.5e-15) |
| song song (4 nguồn) | control khôi phục small − decider-2b khôi phục small | -5.2 [-8.3; -1.9] (p=0.00075) |
| song song (4 nguồn) | strip không dấu − decider-2b khôi phục small | -13.3 [-17.6; -9.3] (p=4.1e-11) |
| song song (4 nguồn) | strip khôi phục small − decider-2b khôi phục small | -3.9 [-7.1; -0.9] (p=0.0087) |
| song song (4 nguồn) | control không dấu − decider-2b đã sửa vit5 | -22.9 [-26.8; -18.9] (p=7.7e-35) |
| song song (4 nguồn) | control đã sửa vit5 − decider-2b đã sửa vit5 | -7.2 [-10.3; -4.0] (p=8.4e-07) |
| song song (4 nguồn) | strip không dấu − decider-2b đã sửa vit5 | -20.9 [-24.9; -16.8] (p=8.4e-30) |
| song song (4 nguồn) | strip đã sửa vit5 − decider-2b đã sửa vit5 | -6.9 [-10.1; -3.8] (p=1.2e-06) |
| song song (4 nguồn) | control không dấu − decider-2b đã sửa small | -17.0 [-21.1; -12.6] (p=4.9e-20) |
| song song (4 nguồn) | control đã sửa small − decider-2b đã sửa small | -6.3 [-9.4; -3.2] (p=1.6e-06) |
| song song (4 nguồn) | strip không dấu − decider-2b đã sửa small | -15.0 [-19.1; -10.9] (p=9.9e-16) |
| song song (4 nguồn) | strip đã sửa small − decider-2b đã sửa small | -5.6 [-8.7; -2.7] (p=8.3e-06) |
| song song · chỉ MASSIVE + XCOPA | control không dấu − decider-2b khôi phục vit5 | -24.5 [-31.3; -18.0] (p=4e-17) |
| song song · chỉ MASSIVE + XCOPA | control khôi phục vit5 − decider-2b khôi phục vit5 | -10.8 [-16.0; -5.8] (p=6.3e-08) |
| song song · chỉ MASSIVE + XCOPA | strip không dấu − decider-2b khôi phục vit5 | -22.0 [-28.5; -15.5] (p=3.5e-14) |
| song song · chỉ MASSIVE + XCOPA | strip khôi phục vit5 − decider-2b khôi phục vit5 | -9.8 [-15.2; -4.7] (p=6.9e-07) |
| song song · chỉ MASSIVE + XCOPA | control không dấu − decider-2b khôi phục small | -17.5 [-24.0; -11.0] (p=2.9e-09) |
| song song · chỉ MASSIVE + XCOPA | control khôi phục small − decider-2b khôi phục small | -9.2 [-14.5; -3.7] (p=9.8e-06) |
| song song · chỉ MASSIVE + XCOPA | strip không dấu − decider-2b khôi phục small | -15.0 [-21.5; -8.2] (p=1.4e-06) |
| song song · chỉ MASSIVE + XCOPA | strip khôi phục small − decider-2b khôi phục small | -8.0 [-13.2; -2.7] (p=7.1e-06) |
| song song · chỉ MASSIVE + XCOPA | control không dấu − decider-2b đã sửa vit5 | -24.5 [-31.5; -17.7] (p=4e-17) |
| song song · chỉ MASSIVE + XCOPA | control đã sửa vit5 − decider-2b đã sửa vit5 | -11.0 [-16.2; -5.8] (p=3.8e-08) |
| song song · chỉ MASSIVE + XCOPA | strip không dấu − decider-2b đã sửa vit5 | -22.0 [-28.2; -15.5] (p=3.5e-14) |
| song song · chỉ MASSIVE + XCOPA | strip đã sửa vit5 − decider-2b đã sửa vit5 | -10.2 [-15.5; -4.8] (p=1.7e-07) |
| song song · chỉ MASSIVE + XCOPA | control không dấu − decider-2b đã sửa small | -18.2 [-24.8; -11.3] (p=6e-11) |
| song song · chỉ MASSIVE + XCOPA | control đã sửa small − decider-2b đã sửa small | -9.3 [-15.2; -3.7] (p=5.7e-07) |
| song song · chỉ MASSIVE + XCOPA | strip không dấu − decider-2b đã sửa small | -15.7 [-22.5; -8.8] (p=4.9e-08) |
| song song · chỉ MASSIVE + XCOPA | strip đã sửa small − decider-2b đã sửa small | -7.7 [-13.0; -2.5] (p=7.1e-06) |
| bản địa (3 nguồn) | control không dấu − decider-2b khôi phục vit5 | -10.7 [-14.5; -6.9] (p=2.2e-06) |
| bản địa (3 nguồn) | control khôi phục vit5 − decider-2b khôi phục vit5 | -0.9 [-4.8; +2.9] (p=0.83) |
| bản địa (3 nguồn) | strip không dấu − decider-2b khôi phục vit5 | -7.6 [-12.0; -3.1] (p=0.019) |
| bản địa (3 nguồn) | strip khôi phục vit5 − decider-2b khôi phục vit5 | -0.0 [-4.2; +4.1] (p=0.43) |
| bản địa (3 nguồn) | control không dấu − decider-2b khôi phục small | -6.8 [-11.0; -2.6] (p=0.0021) |
| bản địa (3 nguồn) | control khôi phục small − decider-2b khôi phục small | -3.3 [-6.8; +0.3] (p=0.29) |
| bản địa (3 nguồn) | strip không dấu − decider-2b khôi phục small | -3.8 [-8.5; +0.9] (p=0.33) |
| bản địa (3 nguồn) | strip khôi phục small − decider-2b khôi phục small | +2.8 [-0.7; +6.4] (p=0.031) |
| bản địa (3 nguồn) | control không dấu − decider-2b đã sửa vit5 | -10.6 [-14.6; -6.6] (p=2.6e-06) |
| bản địa (3 nguồn) | control đã sửa vit5 − decider-2b đã sửa vit5 | -1.0 [-4.9; +2.9] (p=0.83) |
| bản địa (3 nguồn) | strip không dấu − decider-2b đã sửa vit5 | -7.5 [-12.2; -3.2] (p=0.022) |
| bản địa (3 nguồn) | strip đã sửa vit5 − decider-2b đã sửa vit5 | -0.1 [-4.2; +4.1] (p=0.47) |
| bản địa (3 nguồn) | control không dấu − decider-2b đã sửa small | -10.1 [-14.1; -6.1] (p=6.4e-06) |
| bản địa (3 nguồn) | control đã sửa small − decider-2b đã sửa small | -7.1 [-10.8; -3.1] (p=0.0052) |
| bản địa (3 nguồn) | strip không dấu − decider-2b đã sửa small | -7.0 [-11.7; -2.4] (p=0.031) |
| bản địa (3 nguồn) | strip đã sửa small − decider-2b đã sửa small | +0.4 [-3.2; +4.2] (p=0.42) |

## 5. Theo nguồn — accuracy (%)

| Nguồn | n | Model | có dấu | không dấu | khôi phục vit5 | khôi phục small | đã sửa vit5 | đã sửa small |
|---|---|---|---|---|---|---|---|---|
| belebele | 300 | control | 77.3 | 58.3 | 71.7 | 64.7 | 72.3 | 69.7 |
| belebele | 300 | strip | 78.3 | 60.7 | 72.7 | 66.7 | 73.0 | 72.0 |
| belebele | 300 | decider-2b | 83.0 | 49.0 | 79.0 | 69.7 | 79.0 | 75.0 |
| sib200 | 300 | control | 81.7 | 58.3 | 80.3 | 76.0 | 80.3 | 72.0 |
| sib200 | 300 | strip | 80.7 | 59.0 | 79.0 | 76.7 | 79.0 | 69.3 |
| sib200 | 300 | decider-2b | 82.3 | 36.0 | 81.0 | 73.3 | 80.3 | 73.3 |
| massive | 300 | control | 81.0 | 62.7 | 74.0 | 69.3 | 73.7 | 69.3 |
| massive | 300 | strip | 84.3 | 66.7 | 75.0 | 69.7 | 74.3 | 71.7 |
| massive | 300 | decider-2b | 92.7 | 58.7 | 85.7 | 79.7 | 85.7 | 82.0 |
| xcopa | 100 | control | 79.0 | 56.0 | 72.0 | 66.0 | 72.0 | 67.0 |
| xcopa | 100 | strip | 80.0 | 57.0 | 73.0 | 68.0 | 73.0 | 68.0 |
| xcopa | 100 | decider-2b | 86.0 | 53.0 | 82.0 | 74.0 | 82.0 | 73.0 |
| vifactcheck | 163 | control | 71.2 | 62.6 | 73.0 | 62.6 | 72.4 | 62.6 |
| vifactcheck | 163 | strip | 70.6 | 61.3 | 71.8 | 68.1 | 72.4 | 73.0 |
| vifactcheck | 163 | decider-2b | 81.0 | 52.8 | 81.6 | 73.0 | 81.6 | 79.8 |
| viwikifc | 300 | control | 76.7 | 64.7 | 75.3 | 72.3 | 75.3 | 71.3 |
| viwikifc | 300 | strip | 75.3 | 63.3 | 72.7 | 75.0 | 72.3 | 74.3 |
| viwikifc | 300 | decider-2b | 76.7 | 42.0 | 74.3 | 73.0 | 74.3 | 75.0 |
| vihsd | 300 | control | 50.7 | 34.3 | 42.7 | 37.3 | 42.7 | 36.7 |
| vihsd | 300 | strip | 53.0 | 46.0 | 49.0 | 47.3 | 48.3 | 45.7 |
| vihsd | 300 | decider-2b | 40.7 | 34.0 | 37.7 | 36.0 | 37.3 | 37.0 |

## 5b. Theo nguồn — chênh lệch theo cặp (điểm, [CI 95% bootstrap theo state], McNemar p)

| Nguồn | Model | Biến thể | − không dấu | − có dấu |
|---|---|---|---|---|
| belebele | control | khôi phục vit5 | +13.3 [+7.1; +19.2] (p=4.5e-05) | -5.7 [-9.2; -2.6] (p=0.0023) |
| belebele | control | khôi phục small | +6.3 [-0.7; +13.2] (p=0.09) | -12.7 [-17.2; -7.9] (p=4.5e-07) |
| belebele | control | đã sửa vit5 | +14.0 [+7.9; +20.2] (p=1.4e-05) | -5.0 [-8.4; -1.9] (p=0.0059) |
| belebele | control | đã sửa small | +11.3 [+5.4; +17.4] (p=0.00051) | -7.7 [-11.8; -3.6] (p=0.0011) |
| belebele | strip | khôi phục vit5 | +12.0 [+5.7; +18.3] (p=0.00022) | -5.7 [-9.1; -2.3] (p=0.0023) |
| belebele | strip | khôi phục small | +6.0 [-1.0; +12.7] (p=0.11) | -11.7 [-16.5; -6.8] (p=7.7e-06) |
| belebele | strip | đã sửa vit5 | +12.3 [+6.2; +18.9] (p=0.00016) | -5.3 [-8.7; -2.3] (p=0.0015) |
| belebele | strip | đã sửa small | +11.3 [+4.6; +17.7] (p=0.00067) | -6.3 [-10.8; -2.0] (p=0.0054) |
| belebele | decider-2b | khôi phục vit5 | +30.0 [+24.1; +35.8] (p=1.8e-17) | -4.0 [-7.2; -1.3] (p=0.012) |
| belebele | decider-2b | khôi phục small | +20.7 [+14.2; +27.4] (p=8.9e-09) | -13.3 [-18.3; -8.8] (p=1e-08) |
| belebele | decider-2b | đã sửa vit5 | +30.0 [+24.0; +35.7] (p=1.8e-17) | -4.0 [-7.2; -1.0] (p=0.012) |
| belebele | decider-2b | đã sửa small | +26.0 [+20.1; +32.0] (p=4.7e-14) | -8.0 [-12.0; -4.2] (p=7e-05) |
| sib200 | control | khôi phục vit5 | +22.0 [+16.3; +27.7] (p=2.4e-13) | -1.3 [-3.0; -0.3] (p=0.12) |
| sib200 | control | khôi phục small | +17.7 [+12.0; +23.3] (p=2e-08) | -5.7 [-9.3; -2.3] (p=0.0046) |
| sib200 | control | đã sửa vit5 | +22.0 [+16.3; +27.7] (p=2.4e-13) | -1.3 [-2.7; -0.3] (p=0.12) |
| sib200 | control | đã sửa small | +13.7 [+7.3; +20.0] (p=4.6e-05) | -9.7 [-14.0; -5.7] (p=9e-06) |
| sib200 | strip | khôi phục vit5 | +20.0 [+14.3; +25.3] (p=2.7e-11) | -1.7 [-3.3; -0.3] (p=0.062) |
| sib200 | strip | khôi phục small | +17.7 [+12.0; +23.0] (p=5.2e-09) | -4.0 [-7.3; -0.7] (p=0.029) |
| sib200 | strip | đã sửa vit5 | +20.0 [+14.7; +25.7] (p=2.7e-11) | -1.7 [-3.3; -0.3] (p=0.062) |
| sib200 | strip | đã sửa small | +10.3 [+4.7; +16.7] (p=0.0017) | -11.3 [-15.7; -7.3] (p=1.4e-07) |
| sib200 | decider-2b | khôi phục vit5 | +45.0 [+38.7; +51.0] (p=8.3e-34) | -1.3 [-3.0; +0.3] (p=0.22) |
| sib200 | decider-2b | khôi phục small | +37.3 [+30.3; +44.3] (p=4.2e-21) | -9.0 [-12.7; -5.3] (p=3.5e-06) |
| sib200 | decider-2b | đã sửa vit5 | +44.3 [+38.3; +50.0] (p=3e-33) | -2.0 [-3.7; -0.7] (p=0.031) |
| sib200 | decider-2b | đã sửa small | +37.3 [+30.3; +44.7] (p=4.2e-21) | -9.0 [-13.0; -5.3] (p=7.4e-06) |
| massive | control | khôi phục vit5 | +11.3 [+5.7; +17.0] (p=0.00022) | -7.0 [-11.0; -3.3] (p=0.0011) |
| massive | control | khôi phục small | +6.7 [+0.7; +13.0] (p=0.04) | -11.7 [-16.3; -7.0] (p=1.2e-06) |
| massive | control | đã sửa vit5 | +11.0 [+5.0; +16.7] (p=0.00032) | -7.3 [-11.7; -3.3] (p=0.00047) |
| massive | control | đã sửa small | +6.7 [+0.7; +12.7] (p=0.045) | -11.7 [-16.7; -7.3] (p=1.2e-06) |
| massive | strip | khôi phục vit5 | +8.3 [+2.7; +14.0] (p=0.0052) | -9.3 [-13.3; -5.7] (p=1.5e-05) |
| massive | strip | khôi phục small | +3.0 [-3.0; +9.0] (p=0.39) | -14.7 [-19.7; -10.3] (p=3.9e-10) |
| massive | strip | đã sửa vit5 | +7.7 [+1.7; +13.0] (p=0.012) | -10.0 [-14.3; -6.0] (p=2.8e-06) |
| massive | strip | đã sửa small | +5.0 [-1.0; +10.7] (p=0.12) | -12.7 [-17.7; -8.0] (p=3.2e-08) |
| massive | decider-2b | khôi phục vit5 | +27.0 [+21.7; +32.7] (p=2e-18) | -7.0 [-10.0; -4.0] (p=1.9e-05) |
| massive | decider-2b | khôi phục small | +21.0 [+14.7; +27.0] (p=4.2e-10) | -13.0 [-17.3; -9.0] (p=2.2e-10) |
| massive | decider-2b | đã sửa vit5 | +27.0 [+21.3; +33.0] (p=2e-18) | -7.0 [-10.3; -4.0] (p=1.9e-05) |
| massive | decider-2b | đã sửa small | +23.3 [+17.7; +29.3] (p=2.5e-13) | -10.7 [-15.0; -7.0] (p=1.9e-08) |
| xcopa | control | khôi phục vit5 | +16.0 [+4.0; +27.0] (p=0.011) | -7.0 [-16.0; +1.0] (p=0.17) |
| xcopa | control | khôi phục small | +10.0 [-2.0; +21.0] (p=0.13) | -13.0 [-21.0; -5.0] (p=0.0023) |
| xcopa | control | đã sửa vit5 | +16.0 [+4.0; +27.0] (p=0.011) | -7.0 [-15.0; +1.0] (p=0.17) |
| xcopa | control | đã sửa small | +11.0 [+0.0; +23.0] (p=0.09) | -12.0 [-20.0; -4.0] (p=0.0042) |
| xcopa | strip | khôi phục vit5 | +16.0 [+4.0; +27.0] (p=0.017) | -7.0 [-15.0; +1.0] (p=0.14) |
| xcopa | strip | khôi phục small | +11.0 [-1.0; +23.0] (p=0.099) | -12.0 [-20.0; -4.0] (p=0.0075) |
| xcopa | strip | đã sửa vit5 | +16.0 [+4.0; +28.0] (p=0.017) | -7.0 [-15.0; +1.0] (p=0.14) |
| xcopa | strip | đã sửa small | +11.0 [+0.0; +22.0] (p=0.099) | -12.0 [-21.0; -4.0] (p=0.0075) |
| xcopa | decider-2b | khôi phục vit5 | +29.0 [+18.0; +40.0] (p=2.4e-06) | -4.0 [-11.0; +3.0] (p=0.42) |
| xcopa | decider-2b | khôi phục small | +21.0 [+10.0; +32.0] (p=0.00051) | -12.0 [-20.0; -4.0] (p=0.012) |
| xcopa | decider-2b | đã sửa vit5 | +29.0 [+19.0; +40.0] (p=2.4e-06) | -4.0 [-11.0; +3.0] (p=0.42) |
| xcopa | decider-2b | đã sửa small | +20.0 [+9.0; +30.0] (p=0.00082) | -13.0 [-22.0; -5.0] (p=0.0072) |
| vifactcheck | control | khôi phục vit5 | +10.4 [+3.6; +17.2] (p=0.0076) | +1.8 [-1.3; +4.9] (p=0.45) |
| vifactcheck | control | khôi phục small | +0.0 [-8.2; +8.5] (p=1) | -8.6 [-15.6; -2.0] (p=0.02) |
| vifactcheck | control | đã sửa vit5 | +9.8 [+2.8; +17.6] (p=0.014) | +1.2 [-1.8; +4.3] (p=0.69) |
| vifactcheck | control | đã sửa small | +0.0 [-8.2; +8.4] (p=1) | -8.6 [-14.8; -2.9] (p=0.013) |
| vifactcheck | strip | khôi phục vit5 | +10.4 [+1.7; +19.0] (p=0.024) | +1.2 [-3.8; +6.3] (p=0.8) |
| vifactcheck | strip | khôi phục small | +6.7 [-2.7; +16.7] (p=0.2) | -2.5 [-10.2; +5.0] (p=0.62) |
| vifactcheck | strip | đã sửa vit5 | +11.0 [+1.9; +19.3] (p=0.015) | +1.8 [-2.3; +6.6] (p=0.55) |
| vifactcheck | strip | đã sửa small | +11.7 [+3.2; +19.9] (p=0.011) | +2.5 [-3.6; +8.8] (p=0.52) |
| vifactcheck | decider-2b | khôi phục vit5 | +28.8 [+19.5; +38.3] (p=3.8e-08) | +0.6 [-2.6; +4.3] (p=1) |
| vifactcheck | decider-2b | khôi phục small | +20.2 [+11.7; +29.0] (p=5.1e-05) | -8.0 [-14.6; -1.8] (p=0.019) |
| vifactcheck | decider-2b | đã sửa vit5 | +28.8 [+19.8; +37.8] (p=3.8e-08) | +0.6 [-3.1; +4.2] (p=1) |
| vifactcheck | decider-2b | đã sửa small | +27.0 [+17.2; +36.1] (p=6.2e-08) | -1.2 [-7.0; +4.3] (p=0.82) |
| viwikifc | control | khôi phục vit5 | +10.7 [+5.7; +15.8] (p=0.00063) | -1.3 [-3.8; +1.0] (p=0.39) |
| viwikifc | control | khôi phục small | +7.7 [+2.0; +13.4] (p=0.019) | -4.3 [-8.6; -0.3] (p=0.06) |
| viwikifc | control | đã sửa vit5 | +10.7 [+5.7; +16.0] (p=0.00063) | -1.3 [-3.9; +1.0] (p=0.39) |
| viwikifc | control | đã sửa small | +6.7 [+1.2; +12.2] (p=0.045) | -5.3 [-8.6; -2.3] (p=0.0025) |
| viwikifc | strip | khôi phục vit5 | +9.3 [+3.2; +15.8] (p=0.0056) | -2.7 [-5.5; +0.0] (p=0.077) |
| viwikifc | strip | khôi phục small | +11.7 [+5.9; +17.5] (p=0.00064) | -0.3 [-4.3; +3.7] (p=1) |
| viwikifc | strip | đã sửa vit5 | +9.0 [+2.8; +15.5] (p=0.0073) | -3.0 [-5.7; -0.6] (p=0.035) |
| viwikifc | strip | đã sửa small | +11.0 [+5.1; +16.9] (p=0.00081) | -1.0 [-4.6; +2.5] (p=0.68) |
| viwikifc | decider-2b | khôi phục vit5 | +32.3 [+25.4; +39.1] (p=7.2e-18) | -2.3 [-4.8; +0.0] (p=0.065) |
| viwikifc | decider-2b | khôi phục small | +31.0 [+24.2; +37.7] (p=1.2e-16) | -3.7 [-7.3; +0.0] (p=0.071) |
| viwikifc | decider-2b | đã sửa vit5 | +32.3 [+25.4; +39.3] (p=7.2e-18) | -2.3 [-4.9; +0.0] (p=0.065) |
| viwikifc | decider-2b | đã sửa small | +33.0 [+25.9; +39.9] (p=2.6e-19) | -1.7 [-4.9; +1.4] (p=0.42) |
| vihsd | control | khôi phục vit5 | +8.3 [+3.0; +14.0] (p=0.0073) | -8.0 [-13.0; -2.7] (p=0.0037) |
| vihsd | control | khôi phục small | +3.0 [-1.7; +7.7] (p=0.25) | -13.3 [-19.3; -7.7] (p=1.1e-05) |
| vihsd | control | đã sửa vit5 | +8.3 [+2.3; +13.9] (p=0.0073) | -8.0 [-13.3; -3.0] (p=0.0032) |
| vihsd | control | đã sửa small | +2.3 [-2.0; +6.7] (p=0.37) | -14.0 [-19.9; -8.3] (p=5e-06) |
| vihsd | strip | khôi phục vit5 | +3.0 [-2.7; +8.7] (p=0.38) | -4.0 [-9.0; +1.0] (p=0.16) |
| vihsd | strip | khôi phục small | +1.3 [-4.3; +7.3] (p=0.74) | -5.7 [-10.7; -0.3] (p=0.05) |
| vihsd | strip | đã sửa vit5 | +2.3 [-3.7; +8.3] (p=0.51) | -4.7 [-9.7; +0.3] (p=0.092) |
| vihsd | strip | đã sửa small | -0.3 [-6.4; +5.4] (p=1) | -7.3 [-12.3; -2.3] (p=0.0071) |
| vihsd | decider-2b | khôi phục vit5 | +3.7 [-2.0; +9.0] (p=0.22) | -3.0 [-7.6; +1.7] (p=0.27) |
| vihsd | decider-2b | khôi phục small | +2.0 [-3.0; +7.3] (p=0.54) | -4.7 [-9.7; +0.7] (p=0.092) |
| vihsd | decider-2b | đã sửa vit5 | +3.3 [-1.7; +8.3] (p=0.26) | -3.3 [-8.3; +1.3] (p=0.21) |
| vihsd | decider-2b | đã sửa small | +3.0 [-2.0; +8.4] (p=0.33) | -3.7 [-8.6; +1.7] (p=0.18) |

## 6. Chất lượng khôi phục so với bản có dấu gốc (state; NFC, phân biệt hoa/thường, bỏ qua khoảng trắng)

| Bộ | Nguồn | dòng | âm tiết đúng % | âm tiết có dấu đúng % | state khớp hoàn toàn % | câu hỏi khớp % | mọi lựa chọn khớp % | dòng vi phạm bất biến |
|---|---|---|---|---|---|---|---|---|
| vit5 | belebele | 300 | 97.3 | 97.3 | 23.0 | 82.0 | 58.7 | 78 |
| vit5 | sib200 | 300 | 97.8 | 97.7 | 65.0 | 100.0 | 100.0 | 21 |
| vit5 | massive | 300 | 90.0 | 89.2 | 52.7 | 100.0 | 4.0 | 53 |
| vit5 | xcopa | 100 | 92.9 | 93.0 | 68.0 | 100.0 | 47.0 | 0 |
| vit5 | vifactcheck | 163 | 93.7 | 93.6 | 0.6 | 63.2 | 100.0 | 95 |
| vit5 | viwikifc | 300 | 96.9 | 97.0 | 37.7 | 70.3 | 100.0 | 58 |
| vit5 | vihsd | 300 | 73.2 | 73.0 | 8.3 | 100.0 | 52.0 | 67 |
| vit5 | block:parallel | 1000 | 96.9 | 96.8 | 49.0 | 94.6 | 53.5 | 152 |
| vit5 | block:native | 763 | 93.7 | 93.6 | 18.2 | 80.5 | 81.1 | 220 |
| vit5 | all | 1763 | 94.5 | 94.5 | 35.7 | 88.5 | 65.5 | 372 |
| small | belebele | 300 | 82.6 | 82.4 | 5.3 | 69.7 | 45.3 | 257 |
| small | sib200 | 300 | 96.6 | 96.5 | 50.3 | 100.0 | 0.0 | 300 |
| small | massive | 300 | 87.5 | 87.4 | 45.0 | 100.0 | 3.7 | 159 |
| small | xcopa | 100 | 91.6 | 91.7 | 61.0 | 100.0 | 36.0 | 8 |
| small | vifactcheck | 163 | 77.9 | 77.8 | 0.0 | 42.3 | 52.1 | 163 |
| small | viwikifc | 300 | 84.9 | 85.0 | 16.3 | 51.3 | 57.0 | 197 |
| small | vihsd | 300 | 68.5 | 68.9 | 8.7 | 48.0 | 0.0 | 114 |
| small | block:parallel | 1000 | 85.7 | 85.5 | 36.3 | 90.9 | 18.3 | 724 |
| small | block:native | 763 | 79.0 | 79.0 | 9.8 | 48.1 | 33.6 | 474 |
| small | all | 1763 | 80.8 | 80.7 | 24.8 | 72.4 | 24.9 | 1198 |
| vit5_fixed | belebele | 300 | 97.7 | 97.5 | 27.0 | 85.3 | 60.3 | 0 |
| vit5_fixed | sib200 | 300 | 98.1 | 97.9 | 68.7 | 100.0 | 100.0 | 0 |
| vit5_fixed | massive | 300 | 90.1 | 89.2 | 53.3 | 100.0 | 4.7 | 0 |
| vit5_fixed | xcopa | 100 | 92.9 | 93.0 | 68.0 | 100.0 | 47.0 | 0 |
| vit5_fixed | vifactcheck | 163 | 97.4 | 97.3 | 2.5 | 65.0 | 100.0 | 0 |
| vit5_fixed | viwikifc | 300 | 97.3 | 97.0 | 43.3 | 71.3 | 100.0 | 0 |
| vit5_fixed | vihsd | 300 | 73.4 | 73.0 | 10.3 | 100.0 | 52.0 | 0 |
| vit5_fixed | block:parallel | 1000 | 97.2 | 97.0 | 51.5 | 95.6 | 54.2 | 0 |
| vit5_fixed | block:native | 763 | 96.6 | 96.5 | 21.6 | 81.3 | 81.1 | 0 |
| vit5_fixed | all | 1763 | 96.8 | 96.6 | 38.6 | 89.4 | 65.9 | 0 |
| small_fixed | belebele | 300 | 85.2 | 82.7 | 6.0 | 71.0 | 45.7 | 0 |
| small_fixed | sib200 | 300 | 96.8 | 96.5 | 53.3 | 100.0 | 0.0 | 0 |
| small_fixed | massive | 300 | 88.0 | 87.3 | 48.0 | 100.0 | 3.7 | 0 |
| small_fixed | xcopa | 100 | 91.7 | 91.8 | 61.0 | 100.0 | 36.0 | 0 |
| small_fixed | vifactcheck | 163 | 81.3 | 77.9 | 0.0 | 44.8 | 52.1 | 0 |
| small_fixed | viwikifc | 300 | 87.2 | 85.1 | 17.7 | 51.3 | 57.0 | 0 |
| small_fixed | vihsd | 300 | 69.8 | 69.2 | 10.7 | 48.0 | 0.0 | 0 |
| small_fixed | block:parallel | 1000 | 87.7 | 85.8 | 38.3 | 91.3 | 18.4 | 0 |
| small_fixed | block:native | 763 | 82.1 | 79.1 | 11.1 | 48.6 | 33.6 | 0 |
| small_fixed | all | 1763 | 83.6 | 80.9 | 26.5 | 72.8 | 25.0 | 0 |

## 7. Bất biến (bỏ dấu bản khôi phục = bản không dấu, chỉ cho lệch khoảng trắng)

| Bộ | dòng vi phạm | state | câu hỏi | lựa chọn | chạm max_length | chỉ mất/thêm ký tự ngoài chữ-số | đổi chữ-số | tách câu cưỡng bức | ký tự bị rơi nhiều nhất |
|---|---|---|---|---|---|---|---|---|---|
| vit5 | 372 | 315 | 20 | 57 | 6 | 195 | 171 | 19 | `n`×277 `a`×268 `o`×213 `“`×207 `”`×202 `i`×191 `h`×189 `u`×168 |
| small | 1198 | 782 | 93 | 477 | 0 | 38 | 1160 | 21 | `a`×10020 `n`×9318 `o`×8120 `h`×7669 `u`×6352 `i`×6167 `t`×5499 `c`×4933 |
| vit5_fixed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |  |
| small_fixed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 |  |

Dòng vi phạm theo nguồn: vit5: belebele 78, massive 53, sib200 21, vifactcheck 95, vihsd 67, viwikifc 58; small: belebele 257, massive 159, sib200 300, vifactcheck 163, vihsd 114, viwikifc 197, xcopa 8

## 7b. Lỗi ở lựa chọn sau khôi phục (đầu ra thô) và accuracy theo nhóm dòng

"đổi" = có ít nhất một lựa chọn mà chuỗi chữ-số (sau khi bỏ dấu) khác bản không dấu; "giữ" = mọi lựa chọn giữ nguyên chữ-số. "trùng" = hai lựa chọn trở thành giống hệt nhau (NFC, không phân biệt hoa/thường) dù bản không dấu khác nhau; "đúng mất từ" = lựa chọn đúng còn ít từ hơn bản không dấu (vd. "suc khoe" → "sức").

| Bộ | Nguồn | dòng | lựa chọn bị đổi | trùng | đúng mất từ |
|---|---|---|---|---|---|
| vit5 | belebele | 300 | 6 | 0 | 0 |
| vit5 | sib200 | 300 | 0 | 0 | 0 |
| vit5 | massive | 300 | 47 | 0 | 0 |
| vit5 | xcopa | 100 | 0 | 0 | 0 |
| vit5 | vifactcheck | 163 | 0 | 0 | 0 |
| vit5 | viwikifc | 300 | 0 | 0 | 0 |
| vit5 | vihsd | 300 | 0 | 0 | 0 |
| small | belebele | 300 | 25 | 0 | 7 |
| small | sib200 | 300 | 300 | 0 | 40 |
| small | massive | 300 | 145 | 0 | 11 |
| small | xcopa | 100 | 7 | 0 | 4 |
| small | vifactcheck | 163 | 0 | 0 | 0 |
| small | viwikifc | 300 | 0 | 0 | 0 |
| small | vihsd | 300 | 0 | 0 | 0 |

Accuracy (%) theo nhóm dòng, cùng các câu đó ở mọi biến thể:

| Bộ | Nguồn | Model | nhóm | n | có dấu | không dấu | khôi phục (thô) | đã sửa |
|---|---|---|---|---|---|---|---|---|
| vit5 | belebele | control | đổi | 6 | 66.7 | 100.0 | 50.0 | 66.7 |
| vit5 | belebele | control | giữ | 294 | 77.6 | 57.5 | 72.1 | 72.4 |
| vit5 | belebele | strip | đổi | 6 | 83.3 | 83.3 | 50.0 | 83.3 |
| vit5 | belebele | strip | giữ | 294 | 78.2 | 60.2 | 73.1 | 72.8 |
| vit5 | belebele | decider-2b | đổi | 6 | 66.7 | 50.0 | 50.0 | 33.3 |
| vit5 | belebele | decider-2b | giữ | 294 | 83.3 | 49.0 | 79.6 | 79.9 |
| vit5 | sib200 | control | giữ | 300 | 81.7 | 58.3 | 80.3 | 80.3 |
| vit5 | sib200 | strip | giữ | 300 | 80.7 | 59.0 | 79.0 | 79.0 |
| vit5 | sib200 | decider-2b | giữ | 300 | 82.3 | 36.0 | 81.0 | 80.3 |
| vit5 | massive | control | đổi | 47 | 78.7 | 55.3 | 74.5 | 72.3 |
| vit5 | massive | control | giữ | 253 | 81.4 | 64.0 | 73.9 | 73.9 |
| vit5 | massive | strip | đổi | 47 | 83.0 | 70.2 | 76.6 | 74.5 |
| vit5 | massive | strip | giữ | 253 | 84.6 | 66.0 | 74.7 | 74.3 |
| vit5 | massive | decider-2b | đổi | 47 | 91.5 | 53.2 | 83.0 | 83.0 |
| vit5 | massive | decider-2b | giữ | 253 | 92.9 | 59.7 | 86.2 | 86.2 |
| vit5 | xcopa | control | giữ | 100 | 79.0 | 56.0 | 72.0 | 72.0 |
| vit5 | xcopa | strip | giữ | 100 | 80.0 | 57.0 | 73.0 | 73.0 |
| vit5 | xcopa | decider-2b | giữ | 100 | 86.0 | 53.0 | 82.0 | 82.0 |
| vit5 | vifactcheck | control | giữ | 163 | 71.2 | 62.6 | 73.0 | 72.4 |
| vit5 | vifactcheck | strip | giữ | 163 | 70.6 | 61.3 | 71.8 | 72.4 |
| vit5 | vifactcheck | decider-2b | giữ | 163 | 81.0 | 52.8 | 81.6 | 81.6 |
| vit5 | viwikifc | control | giữ | 300 | 76.7 | 64.7 | 75.3 | 75.3 |
| vit5 | viwikifc | strip | giữ | 300 | 75.3 | 63.3 | 72.7 | 72.3 |
| vit5 | viwikifc | decider-2b | giữ | 300 | 76.7 | 42.0 | 74.3 | 74.3 |
| vit5 | vihsd | control | giữ | 300 | 50.7 | 34.3 | 42.7 | 42.7 |
| vit5 | vihsd | strip | giữ | 300 | 53.0 | 46.0 | 49.0 | 48.3 |
| vit5 | vihsd | decider-2b | giữ | 300 | 40.7 | 34.0 | 37.7 | 37.3 |
| small | belebele | control | đổi | 25 | 68.0 | 48.0 | 32.0 | 64.0 |
| small | belebele | control | giữ | 275 | 78.2 | 59.3 | 67.6 | 70.2 |
| small | belebele | strip | đổi | 25 | 64.0 | 48.0 | 44.0 | 68.0 |
| small | belebele | strip | giữ | 275 | 79.6 | 61.8 | 68.7 | 72.4 |
| small | belebele | decider-2b | đổi | 25 | 84.0 | 36.0 | 76.0 | 80.0 |
| small | belebele | decider-2b | giữ | 275 | 82.9 | 50.2 | 69.1 | 74.5 |
| small | sib200 | control | đổi | 300 | 81.7 | 58.3 | 76.0 | 72.0 |
| small | sib200 | strip | đổi | 300 | 80.7 | 59.0 | 76.7 | 69.3 |
| small | sib200 | decider-2b | đổi | 300 | 82.3 | 36.0 | 73.3 | 73.3 |
| small | massive | control | đổi | 145 | 80.7 | 68.3 | 70.3 | 69.0 |
| small | massive | control | giữ | 155 | 81.3 | 57.4 | 68.4 | 69.7 |
| small | massive | strip | đổi | 145 | 87.6 | 69.0 | 70.3 | 73.1 |
| small | massive | strip | giữ | 155 | 81.3 | 64.5 | 69.0 | 70.3 |
| small | massive | decider-2b | đổi | 145 | 94.5 | 61.4 | 79.3 | 83.4 |
| small | massive | decider-2b | giữ | 155 | 91.0 | 56.1 | 80.0 | 80.6 |
| small | xcopa | control | đổi | 7 | 100.0 | 42.9 | 71.4 | 85.7 |
| small | xcopa | control | giữ | 93 | 77.4 | 57.0 | 65.6 | 65.6 |
| small | xcopa | strip | đổi | 7 | 85.7 | 42.9 | 71.4 | 71.4 |
| small | xcopa | strip | giữ | 93 | 79.6 | 58.1 | 67.7 | 67.7 |
| small | xcopa | decider-2b | đổi | 7 | 85.7 | 28.6 | 100.0 | 85.7 |
| small | xcopa | decider-2b | giữ | 93 | 86.0 | 54.8 | 72.0 | 72.0 |
| small | vifactcheck | control | giữ | 163 | 71.2 | 62.6 | 62.6 | 62.6 |
| small | vifactcheck | strip | giữ | 163 | 70.6 | 61.3 | 68.1 | 73.0 |
| small | vifactcheck | decider-2b | giữ | 163 | 81.0 | 52.8 | 73.0 | 79.8 |
| small | viwikifc | control | giữ | 300 | 76.7 | 64.7 | 72.3 | 71.3 |
| small | viwikifc | strip | giữ | 300 | 75.3 | 63.3 | 75.0 | 74.3 |
| small | viwikifc | decider-2b | giữ | 300 | 76.7 | 42.0 | 73.0 | 75.0 |
| small | vihsd | control | giữ | 300 | 50.7 | 34.3 | 37.3 | 36.7 |
| small | vihsd | strip | giữ | 300 | 53.0 | 46.0 | 47.3 | 45.7 |
| small | vihsd | decider-2b | giữ | 300 | 40.7 | 34.0 | 36.0 | 37.0 |

## 8. Bản đã sửa — số từ (có chữ-số) phải giữ nguyên không dấu, theo nguồn

| Bộ | Nguồn | dòng bị sửa | từ | từ giữ không dấu | tỉ lệ % |
|---|---|---|---|---|---|
| vit5 | belebele | 78/300 | 45898 | 170 | 0.37 |
| vit5 | sib200 | 21/300 | 15457 | 5 | 0.03 |
| vit5 | massive | 53/300 | 19440 | 5 | 0.03 |
| vit5 | xcopa | 1/100 | 2696 | 0 | 0.00 |
| vit5 | vifactcheck | 96/163 | 108839 | 611 | 0.56 |
| vit5 | viwikifc | 60/300 | 41845 | 105 | 0.25 |
| vit5 | vihsd | 72/300 | 7846 | 27 | 0.34 |
| small | belebele | 257/300 | 45898 | 4876 | 10.62 |
| small | sib200 | 300/300 | 15457 | 331 | 2.14 |
| small | massive | 159/300 | 19440 | 256 | 1.32 |
| small | xcopa | 9/100 | 2696 | 9 | 0.33 |
| small | vifactcheck | 163/163 | 108839 | 19387 | 17.81 |
| small | viwikifc | 202/300 | 41845 | 3100 | 7.41 |
| small | vihsd | 120/300 | 7846 | 100 | 1.27 |

## 9. Độ trễ bộ khôi phục, batch 1 (state + câu hỏi + lựa chọn của một câu; fp32, greedy, theo thẻ model)

| Bộ | state (token Qwen) | dòng | p50 ms | p90 ms | TB ms | số lần gọi TB |
|---|---|---|---|---|---|---|
| vit5 | ≤128 | 1319 | 943 | 1651 | 948 | 7.3 |
| vit5 | 129–512 | 308 | 2487 | 3927 | 2766 | 6.3 |
| vit5 | 513–1024 | 95 | 8391 | 10663 | 8560 | 9.8 |
| vit5 | >1024 | 41 | 13591 | 18642 | 14353 | 14.8 |
| vit5 | all | 1763 | 1052 | 3563 | 1988 | 7.5 |
| small | ≤128 | 1319 | 578 | 894 | 559 | 7.3 |
| small | 129–512 | 308 | 1304 | 1967 | 1409 | 6.4 |
| small | 513–1024 | 95 | 3891 | 4912 | 3945 | 10.0 |
| small | >1024 | 41 | 6615 | 8859 | 6759 | 15.1 |
| small | all | 1763 | 639 | 1857 | 1034 | 7.5 |

Model quyết định (đo trước, batch 1, bf16): Qwen3-1.7B + adapter p50 29 / 50 / 181 ms ở prompt 128 / 512 / 2.048 token (`results/ft/latency_ft.json`); trung vị trên dev_v1: adapter đối chứng 41 ms, decider-2b 61 ms.
