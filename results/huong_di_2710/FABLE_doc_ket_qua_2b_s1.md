# Đọc kết quả bậc 1 của 2b (2b-s1 so với C-s1) — Fable 5.1, 28/10/2026

Chỉ đọc. Mọi số tự tính lại bằng script CPU trên preds và dữ liệu có sẵn, đặt ở `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2b\` (tên script ghi cạnh từng số; mục 9). Không chạy model, không đọc `test_v1.3.jsonl`, không sửa file dự án ngoài file này.

Đối tượng: `results/train_2b/REPORT_2b_bac1.md`, `bang_tg_2b-s1.json`; preds `results/probe_v1/eval/{dev,p4,pa,p5,pb,p1b}/{2b-s1,C-s1}/`; `results/train_2b/eval/{proxy,val_kc_v2}/`; `data/bench_v1_3/dev_v1_3.jsonl`; `results/train_2b/train_2b_v3.jsonl`. Luật đọc: `PHAN_TICH_TRUOC_TRAIN_2b.md` §6, §9; phản biện trước: `PHAN_BIEN_2b_fable.md`.

## Kết luận trước

1. **Mọi số trong REPORT khớp** (mục 1). Nhưng bảng T/G không cho thấy điều quan trọng nhất: **trên bản không dấu, 2b đã chuyển sang "NEI mặc định"**. ViWikiFC không dấu (choice, n = 161): 2b đoán NEI **91/161 = 57%** dòng (gold 25%), precision 0,40; recall Xác nhận **0,65 → 0,30**, Bác bỏ 0,55 → 0,45. Noul không dấu: recall "Có" của ViFactCheck **0,90 → 0,45**. G1 đạt chỉ vì ViFactCheck không dấu +10,1 bù cho SIB −3,3, MASSIVE −3,1, XCOPA −7,0, ViWikiFC −2,5 (Belebele +2,0); bỏ phần bù ấy thì macro không dấu còn 61,67 < 62,47. Bản có dấu có cùng độ nghiêng nhưng nhẹ: ViWikiFC đoán NEI 30% dòng (gold 25%), precision NEI 0,67. (`g5_nodiac_shortcut.py`)
2. **+3,19 macro VI là thật trên dev, nhưng 92% đến từ một nguồn, một nhãn**: ViFactCheck +17,7 góp +2,95; năm nguồn còn lại cộng lại +0,25. Trong ViFactCheck, toàn bộ là NEI (choice: 7 → 28/34; McNemar 21 so 0). NEI đúng tăng ở **mọi** mức bao phủ, kể cả bao phủ ≥ 0,9 (2 → 9/12), và precision NEI trên ViFactCheck là 28/30 = 0,93 → không phải mẹo trùng từ; đó là đọc được kiểu NEI "chèn tuổi/năm" mà KC N-thêm nhắm đúng. (`g1_dev_breakdown.py`)
3. **Proxy (train split, ngoài mẫu) cho cùng chiều nhưng nhỏ hơn; số 74,0 so 71,2 trong REPORT bị pha nodiac.** Có dấu, choice + noul: proxy **+5,0 [−0,7; +10,7]** so với dev +7,7 [3,4; 12,0]; ViFactCheck proxy **+8,7 [1,3; 16,0]** so với dev +17,7 [10,8; 25,0]; ViWikiFC +1,3 so với +2,1. CI chồng nhau: không mâu thuẫn, nhưng ViFactCheck trên dev nằm ở đầu lạc quan. Một phần vì **cơ cấu KC được chỉnh theo phân bố dev** (§9.7: N-thêm "giống dev về độ phủ"; D7 so với tỉ lệ NEI và "Có" của dev; phản biện của tôi cũng cấp số bao phủ dev ViWikiFC). Không có rò rỉ theo dòng (0/443 nhận định dev nằm trong train; 24/241 dòng có ≥ 1 8-gram chung với KC, tối đa 6% văn bản, mức tăng trên các dòng ấy bằng phần còn lại). (`g2_proxy_probes.py`, `g3_leakage.py`, `g5`)
4. **Bác bỏ ViWikiFC 0,878 → 0,770 là đúng chi phí đã dự báo**: 13 dòng mất (10 sang NEI, 3 sang Xác nhận), 9/13 ở bao phủ < 0,7, 0 dòng ở ≥ 0,9; proxy lặp lại (0,955 → 0,727, n = 22). Mốc ngoài (decider-2b) cũng chỉ 0,78–0,80 → một phần là sàn nhãn ViWikiFC; phần riêng của 2b là ranh NEI/Bác bỏ mỏng: đổi câu hỏi (P5) thì NEI ViWikiFC 78 → 61, Bác bỏ 77 → 82 (C chỉ đổi −7 ở NEI). Thêm một chi phí **không có cổng**: recall "Có" noul tụt ở mọi nơi (dev ViFactCheck 0,966 → 0,828, ViWikiFC 0,872 → 0,809; proxy 0,885 → 0,731, 0,963 → 0,741). (`g1`, `g2`, `g4_models_agg.py`)
5. **Nguồn không kiểm chứng: không có tác dụng phụ trên bản có dấu** (4 nguồn gộp +0,4 [−1,7; +2,5], p = 0,76; proxy +3,4 [0,4; 6,6]). Bản không dấu −2,0 [−4,7; +0,6] (p = 0,15), cùng chiều âm ở 3/4 nguồn: nhỏ, chưa tách khỏi nhiễu, nhưng cùng cơ chế với điểm 1. XCOPA −3 (7 so 4) và Belebele −1 (18 so 15) là nhiễu. (`g1`, `g4`, `g5`)
6. **Khuyến nghị: chạy 2b-s2 trước (≈ 4 h), hoãn C-s2, chưa sang 2c.** Nhiễu seed chưa từng được đo (0,79 là SD giữa checkpoint trong một lần chạy C), và NEI của C dao động 0,23–0,48 giữa các checkpoint → seed 2 của 2b có thông tin thật (Bác bỏ và NEI-mặc-định có lặp không). C-s2 không đổi được kết luận nào: cổng T4 bậc 2 chỉ trượt nếu 2b-s2 − C-s2 < −5,2. Ghi là sai lệch có ngày so với dạng 2 × 2 đã đăng ký; chạy C-s2 sau nếu 2b-s2 lệch s1 quá 2 điểm hoặc người dùng muốn giữ nguyên dạng. (mục 8)

**Nói với người dùng:** trên test 1.3 (cùng cách dựng với dev) kỳ vọng giữ gần đủ, **+2 đến +3,5**, vì hành vi NEI mới sống qua đổi chữ nhãn (p1b), đổi câu hỏi (P5) và bỏ dấu lựa chọn (Pa). Trên dữ liệu ngoài bench, ViFactCheck giữ khoảng một nửa đến hai phần ba, ViWikiFC ≈ 0, bốn nguồn khác 0 đến +3 → **+1,5 đến +3**. Đổi lại: Bác bỏ ViWikiFC −10,8 (p = 0,096, lặp trên proxy), "Có" noul −6 đến −14, và trên văn bản không dấu model chọn NEI thay cho đọc.

## 1. Số đã kiểm khớp với REPORT

| Mục REPORT | Số REPORT | Tự tính | Script |
|---|---|---|---|
| T1 P4 có dấu NEI | 100,0 / 100,0; theo nhãn gốc 6 ô đều 100 | khớp; p(NEI) trung bình 0,999 / 0,998 | `g1` §10 |
| T2 recall NEI gộp | 0,347 → 0,800; McNemar 1 / 35 | khớp (26 → 60/75) | `g1` §1, §4 |
| T3a precision NEI | 25/29 → 61/79 | khớp | `g1` §1 |
| T3b XN+BB ViWikiFC | 0,875 → 0,800; McNemar 15 / 6, p = 0,078 | khớp (mất 2 XN + 13 BB, được 1 + 5) | `g1` §4 |
| T4 | 81,65 → 84,84; CI [1,02; 5,34] | khớp từng nguồn; tổng đóng góp +3,20 (làm tròn) | `g1` §8 |
| G1 / G2 | 63,97 → 63,35; 88,20 → 88,80 | khớp từng nguồn | `g1` §8 |
| G3 Pa ViWikiFC | −16,8 → −4,3 | khớp; Bác bỏ dưới Pa: C 88 → 43, 2b 77 → 63,5 | `g2` probes |
| G4 P5 kiểm chứng | −2,3 → −1,8 | khớp theo từng ô | `g2` probes |
| Recall từng nguồn / nhãn | bảng chỉ báo | khớp cả 12 ô | `g1` §1 |
| pb, p1b | 87,0 (+3,0) → 88,7 (+3,0); 65,4 (+1,9) → 71,8 (+1,5) | khớp | `g2` probes |
| proxy_v1 kiểm chứng | 71,2 → 74,0 (n = 600) | khớp **nhưng** gồm 300 dòng nodiac (+0,7); có dấu riêng 75,0 → 80,0 | `g2` |
| val_kc_v2 | 82,0 → 98,0 | khớp; lỗi còn lại của 2b: 2/23 Bác bỏ KC-dài, 3/21 Xác nhận KC-dài | `g2` |

Không có số sai. Các mục dưới là về cách đọc.

## 2. ViFactCheck +17,7: đọc thật, không phải mẹo trùng từ — nhưng là một kiểu NEI

**Phân rã (`g1` §1–§3, có dấu, choice n = 80, noul n = 78):**
- Choice 57,5 → 87,5 (+30); noul 87,2 → 92,3 (+5,1). Gộp 72,2 → 89,9.
- Ma trận nhầm choice: NEI gold 34: C đoán sup 10 / ref 17 / nei 7; 2b đoán 2 / 4 / **28**. Xác nhận 15 → 16/18, Bác bỏ 24 → 26/28. Không mất dòng Xác nhận nào, mất 1 Bác bỏ (sang NEI, bao phủ 0,88).
- Precision NEI riêng ViFactCheck: C 7/8, 2b **28/30 = 0,93**. Ở bao phủ ≥ 0,9 (47 dòng, 12 NEI), 2b đoán NEI 9 lần, đúng cả 9.

**NEI đúng theo mức bao phủ (cover = tỉ lệ từ của nhận định có trong văn bản):**

| Bao phủ | n NEI | C đúng | 2b đúng |
|---|---|---|---|
| < 0,7 | 5 | 2 | 5 |
| 0,7–0,9 | 17 | 3 | 14 |
| ≥ 0,9 | 12 | 2 | 9 |

Tăng ở cả ba mức, nhiều nhất ở mức cao: **mẹo bao phủ không giải thích được**. Tỉ số P(đoán NEI | cover < 0,9) / P(đoán NEI | cover ≥ 0,9) trên hai nguồn gộp: C 4,2, **2b 2,4**, gold 1,6 — 2b bám bao phủ *ít hơn* C (C hầu như chỉ đoán NEI khi bao phủ thấp), gần gold hơn.

**Cơ chế (`g4` §4, `g5` §2):** các dòng NEI bao phủ ≥ 0,9 mà 2b mới đúng đều có dạng "chèn chi tiết không kiểm được": "già làng … **80 tuổi**", "BS … **sinh năm 1997**", "PGS.TS … **55 tuổi**", "Ông … **60 tuổi**", "18 giờ ngày 16/6". Đây đúng là kiểu N-thêm của KC (chèn cụm thời gian/nơi chốn không có trong văn bản). Kiểm mẹo "nhận định có số/tuổi/năm không có trong văn bản → NEI": ở dev, 13 dòng **Bác bỏ** ViFactCheck mang cờ này, 2b đoán Bác bỏ 11, NEI 2 — không bị cờ kéo; cờ chỉ có ở 15% NEI ViFactCheck. Vậy 2b học được kiểu NEI "thêm chi tiết" đúng như ViFactCheck dựng, không phải phản xạ theo số.

**P4 theo nhãn gốc:** 18/18, 28/28, 34/34 (ViFactCheck) và 46/46, 74/74, 41/41 (ViWikiFC), p(NEI) 0,999. Đạt điều kiện "không phụ thuộc nhãn gốc"; nhưng mức bão hoà cho thấy đó là luật cứng "lạc đề → NEI" đã học thẳng từ 2e + N-lạc đề, như đã ghi trước: T1 là sàn.

**Proxy ViFactCheck (train split; `g2`, `g4` §3):** choice có dấu 65,4 → 76,9 (+11,5), noul 81,9 → 87,5; NEI choice 0,21 → **0,54** (n = 24) so với dev 0,21 → 0,82 (n = 34). Hai lý do cho khoảng cách:
- cơ cấu nhãn: proxy 31/23/24 (NEI 31%), dev 18/28/34 (NEI 42,5%); nếu tái trọng số recall dev theo cơ cấu proxy, gain choice từ +30 còn +23;
- kiểu NEI: NEI proxy đa dạng hơn (diễn đạt lại, câu hỏi "hay", suy đoán), ít kiểu "chèn tuổi/năm" hơn; 2b sai 11/24, phần lớn ở các dòng ấy. Với n = 24 thì khác biệt gain (+33 so với +62 điểm recall) ≈ 1,8 SE — gợi ý, chưa chắc.

## 3. Bác bỏ ViWikiFC 0,878 → 0,770: đúng chi phí đã dự báo, ở bao phủ thấp

(`g1` §2, §4, §5; `g2`; `g4` §1)

| Bao phủ | n Bác bỏ | C đúng | 2b đúng |
|---|---|---|---|
| < 0,7 | 40 | 33 | 28 |
| 0,7–0,9 | 29 | 27 | 24 |
| ≥ 0,9 | 5 | 5 | 5 |

- 13 dòng mất: 10 sang NEI, 3 sang Xác nhận; bao phủ 0,35–0,84, **9/13 dưới 0,7, không dòng nào ≥ 0,9**. 5 dòng được (C từng đoán Xác nhận 4, NEI 1). McNemar 13 / 5, p = 0,096.
- Nhiều dòng mất có p(NEI) của 2b 0,92–0,97: nhận định diễn đạt lại một mâu thuẫn ("Hoàng Trần Ích là … doanh nhân", "54 dân tộc … tôn thờ các vua chúa") — đúng loại "Bác bỏ ở độ phủ thấp" mà dữ liệu 2b không có (phản biện 2.4).
- **Proxy lặp lại:** Bác bỏ ViWikiFC 0,955 → 0,727 (−3 ở < 0,7, −2 ở 0,7–0,9, 0 ở ≥ 0,9); XN+BB ViWikiFC 0,896 → 0,771.
- **Không phải toàn bộ là lỗi 2b:** decider-2b (mốc ngoài, text/keyed) cũng ở Bác bỏ ViWikiFC 0,78–0,80, precision NEI 0,68–0,71, recall NEI 0,69 — gần giống hồ sơ của 2b (0,77 / 0,77 / 0,80). Một phần là sàn nhãn của ViWikiFC.
- **Phần riêng của 2b là ranh mỏng:** P5 (đổi câu hỏi) làm ViWikiFC NEI 78,0 → 61,0 và Bác bỏ 77,0 → 82,4; C chỉ đổi NEI 46 → 39, Bác bỏ không đổi. Tổng P5 ViWikiFC −1,9 nhìn ổn, nhưng là hai dịch chuyển ngược chiều.

## 4. Rò rỉ và học thuộc trong khung

**Rò rỉ theo dòng: không có** (`g3_leakage.py`, `g5` §3).
- 0/443 nhận định dev kiểm chứng nằm nguyên trong state/question của 15.637 dòng train VI.
- 8-gram: 24/241 dòng dev choice có ≥ 1 8-gram chung với state KC, mức cao nhất 6% số 8-gram của văn bản; nội dung là tên cơ quan ("hội các quốc gia đông nam á asean", "bí thư thường trực thành ủy hà nội"). Acc trên 24 dòng ấy: C 0,625 → 2b 0,792; phần còn lại 0,714 → 0,825 — mức tăng bằng nhau, không có lợi riêng.
- Khung câu hỏi, chữ lựa chọn KC/2d/2e giống dev đúng như thiết kế (D2), thứ tự lựa chọn cân 540/547/558.

**Học thuộc khung: không** (`g2` probes).
- p1b (đổi chữ nhãn thành "Chưa đủ căn cứ / Nhận định đúng / sai"): 2b ViFactCheck 85,0 (dev 87,5), ViWikiFC 77,6 (79,5). Gain so với C vẫn còn dưới chữ nhãn khác: 66,2 → 85,0 và 75,8 → 77,6.
- Pa −4,3 (C −16,8); Bác bỏ ViWikiFC dưới Pa 77 → 63,5 (C 88 → 43).
- P5: ViFactCheck 87,5 = dev; ViWikiFC −1,9 nhưng với dịch chuyển NEI/Bác bỏ ở mục 3.
- pb (nhãn tiếng Anh, chỉ sib/massive): +3,0 cả hai. val_kc_v2 98% (C 82%): trong phân bố dựng, chỉ cho biết công thức được học gần hết.

**Phụ thuộc phân bố dev (không phải rò rỉ, nhưng cần ghi):** cơ cấu NEI của KC được chọn vì N-thêm "giống dev về độ phủ" (§9.7), D7 so với tỉ lệ NEI 31,1% và "Có" 37,1% của dev, và `PHAN_BIEN_2b_fable.md` 2.1 cấp số bao phủ dev ViWikiFC (59 / 54% < 0,7). Dev vì vậy không còn là ước lượng hoàn toàn độc lập cho **công thức dữ liệu**; proxy (train split, cùng cách dựng, chỉ dùng để đặt ngưỡng D6′/D10) là ước lượng sạch hơn. Đó là lý do nên đọc ViFactCheck +17,7 ở đầu lạc quan, với proxy +8,7 ở đầu dè dặt.

**Vì sao proxy "74,0 so với 71,2" nhỏ:** (i) 300/600 dòng là nodiac (+0,7); có dấu riêng +5,0 [−0,7; +10,7] so với dev +7,7 [3,4; 12,0]; (ii) cơ cấu nhãn ít NEI hơn; (iii) kiểu NEI đa dạng hơn (mục 2); (iv) văn bản ViWikiFC proxy dài hơn (trung vị 113 so với 77 từ). Trên 6 nguồn có dấu, macro proxy lại **+4,2** (SIB-200 +8,0, p = 0,002; Belebele n = 19 nhiễu) — proxy không nói gain tổng bị thổi phồng; nó nói gain **phân bố khác**: ít ở ViFactCheck, nhiều hơn ở SIB.

## 5. Nguồn không kiểm chứng

(`g1` §8–§9, `g4` §2, `g5` §5)

| Nhóm | Hiệu 2b − C | CI 95% cụm | McNemar |
|---|---|---|---|
| 4 nguồn không KC, có dấu, dev (n = 987) | +0,4 | [−1,7; +2,5] | 48 / 52, p = 0,76 |
| 4 nguồn không KC, có dấu, proxy (n = 469) | +3,4 | [+0,4; +6,6] | SIB 1 / 13 |
| 4 nguồn không KC, không dấu, dev (n = 987) | −2,0 | [−4,7; +0,6] | 96 / 76, p = 0,15 |
| EN 4 nguồn (n = 987) | +0,6 | — | 27 / 36, p = 0,31 |

- XCOPA có dấu 85 → 82: 7 so 4 trên 100 câu, p = 0,55; 11 câu đổi chiều không có mẫu chung (có câu cả hai model đều rất tự tin theo hai hướng ngược nhau). Nhiễu. Belebele 84 → 83: 18 / 15, p = 0,73. Nhiễu.
- Không dấu: SIB −3,3 (34 / 24), MASSIVE −3,1 (23 / 14), XCOPA −7 (15 / 8; cả hai mức 59 và 52 đều sát sàn ngẫu nhiên 50), Belebele +2,0. Từng nguồn là nhiễu; gộp −2,0 cùng chiều ở 3/4 nguồn, p = 0,15: **có thể là tác dụng phụ nhỏ của việc bỏ VNTC và thêm 587 dòng KC không dấu**, chưa chứng minh được với một seed. Đọc chung với mục 6: nodiac là chế độ 2b yếu nhất.

## 6. Hai chi phí không có cổng

(`g5` §1, `g1` §7, `g2`)

**(a) Không dấu: NEI mặc định.** Dev choice không dấu:

| Nguồn | Model | Đoán NEI / n | Precision NEI | Recall XN / BB / NEI |
|---|---|---|---|---|
| ViFactCheck | C | 16/80 | 0,75 | 0,72 / 0,82 / 0,35 |
| ViFactCheck | 2b | 39/80 | 0,74 | 0,78 / **0,61** / 0,85 |
| ViWikiFC | C | 40/161 | 0,42 | 0,65 / 0,55 / 0,41 |
| ViWikiFC | 2b | **91/161** | 0,40 | **0,30** / 0,45 / 0,88 |

Gộp: recall NEI 0,40 → 0,85, precision 0,52 → 0,50, 130 lần đoán NEI cho 75 NEI gold. Noul không dấu: "Có" ViFactCheck 0,90 → 0,45, ViWikiFC 0,66 → 0,43. Khi đọc kém (mất dấu), 2b không còn phân biệt mà rơi về NEI. T3a/T3b chỉ đo bản có dấu nên không bắt.

**(b) Noul "Có" tụt ở bản có dấu, cả dev và proxy:** ViFactCheck 0,966 → 0,828 (28 → 24/29), ViWikiFC 0,872 → 0,809; proxy 0,885 → 0,731, 0,963 → 0,741. Noul gộp vẫn tăng (87,2 → 92,3; 85,5 → 87,1) nhờ NEI và Bác bỏ, nên bảng chỉ báo không lộ. Đây là mặt "Không" của cùng độ nghiêng NEI.

## 7. Trả lời năm câu hỏi

1. **ViFactCheck +17,7:** đọc thật. Toàn bộ là NEI; tăng ở mọi mức bao phủ, nhất là ≥ 0,9; precision NEI 0,93; không bám cờ số thiếu; không phụ thuộc chữ nhãn (p1b) hay câu hỏi (P5). Nhưng là **một kiểu NEI** (chèn chi tiết) mà KC N-thêm nhắm đúng; proxy cho +8,7 [1,3; 16,0], tức khoảng một nửa. P4 theo nhãn gốc 100% ở 6 ô: đạt, bão hoà, là sàn.
2. **Bác bỏ ViWikiFC:** đúng chi phí dự báo ở phản biện 2.1/2.4: 9/13 dòng mất ở bao phủ < 0,7, 0 ở ≥ 0,9, proxy lặp lại. Một phần là sàn nhãn (decider-2b cũng 0,78–0,80); phần riêng của 2b là ranh NEI/Bác bỏ đổi theo câu hỏi (P5).
3. **Rò rỉ:** không theo dòng. Học thuộc khung: không (p1b, Pa, P5 giữ gain). Có **phụ thuộc phân bố dev** ở khâu thiết kế KC; proxy là ước lượng sạch hơn và nhỏ hơn. Pa −4,3 / P5 −1,8 / pb / p1b đều không cho dấu hiệu ngược.
4. **Nguồn không kiểm chứng:** có dấu: nhiễu (gộp +0,4; proxy +3,4). Không dấu: −2,0 gộp, cùng chiều, p = 0,15 — tác dụng phụ nhỏ có thể có, cùng cơ chế với NEI mặc định; cần seed 2 để kết luận.
5. **Khuyến nghị:** mục 8.

## 8. Khuyến nghị

**Nhiễu seed chưa từng được đo.** Số 0,79 ở §5 là SD giữa checkpoint 50/75/100% của một lần chạy C (`PHAN_TICH_TRUOC_TRAIN.md` dòng 74), không phải giữa seed. Trong một lần chạy C: macro 80,17–81,65; **recall NEI 0,23–0,48**; ViFactCheck 70,9–72,2 (`g4` §1). Hành vi NEI/Bác bỏ dao động mạnh ngay trong một lần train, trong khi ViFactCheck của C ổn định thấp. 2b-s1 ở 84,84 cao hơn checkpoint tốt nhất của C 3,2 điểm.

**Cổng T4 bậc 2 không thể đổi kết luận.** Trung bình 2 seed mỗi bên < C − 1,0 đòi 2b-s2 − C-s2 < −5,2, tức lệch 8,4 điểm so với seed 1; dải checkpoint của C chỉ 1,5. C-s2 vì vậy chỉ trả lời câu "SD seed của C là bao nhiêu" cho câu kết luận cuối, không trả lời câu đi/dừng.

**Đề nghị:**
1. **Chạy 2b-s2 trước** (≈ 3,5 h train + 0,5 h chấm, gồm dev, P4, Pa, P5, p1b, proxy có dấu, val_kc). Câu hỏi của seed 2: (a) Bác bỏ ViWikiFC có tụt lại ≥ 8 điểm không; (b) NEI mặc định trên nodiac có lặp không (đoán NEI ViWikiFC nodiac > 50% dòng); (c) T2 / T3a / T3b có giữ không; (d) macro lệch s1 bao nhiêu. Ghi thêm hai **chỉ báo** (không phải cổng mới, vì đặt sau khi thấy số): recall "Có" noul theo nguồn; tỉ lệ đoán NEI trên ViWikiFC không dấu.
2. **Hoãn C-s2, quyết sau 2b-s2.** Chạy C-s2 nếu 2b-s2 lệch s1 quá 2 điểm macro (nhiễu seed lớn hơn tưởng) hoặc người dùng muốn giữ nguyên dạng 2 × 2 đã đăng ký cho báo cáo cuối. Đây là **sai lệch có ngày** so với §6 bậc 2; ghi vào phân tích trước khi bấm train, kèm lý do ở trên.
3. **Chưa sang 2c trước khi có 2b-s2.** 2c đổi thêm một biến; lỗi còn lại của 2b (Bác bỏ diễn đạt lại ở bao phủ thấp, NEI mặc định khi đọc kém) không phải lỗi thiếu tiếng Anh nói chung. Nếu 2c là NLI tiếng Anh có mâu thuẫn ở trùng từ thấp thì đúng là ứng viên cho lỗi Bác bỏ — xét sau bậc 2, với 2b (2 seed) làm mốc.
4. **Không đụng test.** Chưa có gì cần tới test; test chỉ dùng một lần ở cuối.

**Nói với người dùng về +3,19:**
- Trên test 1.3 (cùng cách dựng với dev): kỳ vọng **+1,5 đến +3,5**, với giả định tỉ lệ NEI của test gần dev — gain gần như toàn là NEI nên tỉ lệ ấy quyết định độ lớn, và `manifest_test_v1.3.json` chỉ ghi số dòng theo nguồn × kind (ViFactCheck choice 135 / noul 156, ViWikiFC 131 / 151), không ghi cơ cấu nhãn; không mở test để kiểm. Hành vi mới sống qua p1b, P5, Pa; NEI McNemar 35 so 1. CI item [1,02; 5,34] chưa gồm nhiễu seed (chưa đo).
- Trên dữ liệu ngoài bench (proxy): ViFactCheck giữ một nửa đến hai phần ba (+8,7 so với +17,7), ViWikiFC ≈ 0 (NEI tăng bù Bác bỏ tụt), bốn nguồn khác 0 đến +3 → macro **+1,5 đến +3**.
- 92% của +3,19 là một nguồn, một nhãn. Câu đúng là: "2b sửa chắc lỗi NEI đã nhắm; trả giá bằng Bác bỏ ViWikiFC −10,8 (p = 0,096, lặp trên proxy), 'Có' noul −6 đến −14, và trên văn bản không dấu model chọn NEI thay cho đọc (ViWikiFC không dấu: 57% dòng, gold 25%)."

## 9. Script và file đã dùng

Thư mục: `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2b\` (kết quả chạy lưu cạnh: `g1_out.txt`, `g2_out*.txt`, `g3_out.txt`, `g4_out.txt`, `g5_out.txt`).

| Script | Việc | Đọc |
|---|---|---|
| `g1_dev_breakdown.py` | ma trận nhầm; acc theo nhãn × bao phủ (cover, max_ov); tỉ lệ đoán NEI theo bao phủ; dòng đổi chiều; dòng Bác bỏ ViWikiFC mất; noul; điểm nguồn × biến thể + McNemar; XCOPA 100 câu; P4 theo nhãn gốc | `dev_v1_3.jsonl`, `eval/{dev,p4}/{C-s1,2b-s1}/preds.jsonl`, `p4_van_ban_khong_lien_quan.jsonl` |
| `g2_proxy_probes.py` | proxy theo nguồn × kind × biến thể × nhãn × bao phủ; so phân bố bao phủ proxy/dev; val_kc_v2 theo nguồn × kind × nhãn; Pa / P5 / pb / p1b theo nguồn × kind × nhãn | `data/proxy_v1/proxy_v1.jsonl`, `results/train_2b/eval/{proxy,val_kc_v2}/*/preds.jsonl`, `_rows/val_kc_v2_eval.jsonl`, `probe_v1/{pa,p5,pb,p1b}*.jsonl` + preds |
| `g3_leakage.py` | nhận định dev nằm trong train; 8-gram văn bản dev so với state train VI và KC; khung câu hỏi / lựa chọn / thứ tự | `train_2b_v3.jsonl`, dev |
| `g4_models_agg.py` | mọi model có preds dev (C checkpoint, base, M1, decider-2b); McNemar gộp 4 nguồn; tái trọng số ViFactCheck theo cơ cấu proxy; ví dụ NEI dev / proxy; khung proxy so dev | `eval/dev/*/preds.jsonl`, proxy |
| `g5_nodiac_shortcut.py` | ma trận nhầm không dấu; mẹo "số thiếu → NEI" trên dev / proxy / KC / 2d; dòng dev trùng 8-gram với KC và acc; câu hỏi P5; bootstrap cụm CI cho hiệu dev / proxy theo nhóm | như trên |
| `kcf.py` | hàm đo trùng từ chép từ `scripts/train_2b/kc_util.py` (normalize, tách câu, max_ov, cover) | — |

Phép đo bao phủ dùng cùng normalize của `decontam.py` (NFC, lower, bỏ ký tự không phải chữ/số). Không đọc `test_v1.3.jsonl`. Không chạy model.
