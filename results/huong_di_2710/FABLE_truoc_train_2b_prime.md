# 2b′ trước khi train: có đáng 8 giờ GPU không, và đọc thế nào — Fable 5.1, 29/10/2026

Chỉ CPU, không chạy model, không đọc test, không sửa file dự án ngoài file này. Mọi số dưới đây tự tính lại từ `train_2b_v3.jsonl`, `tam/lan1/` và `tam/lan2/` của 2b′, dev 1.3, proxy_v1 và preds dev của C-s1 / 2b-s1 / 2b-s2 / decider (script ở mục 6; `h1_out.txt`, `h2_out.txt` cạnh script). "cover" = tỉ lệ từ của nhận định có trong **cả** state; "max_ov" = với câu khớp nhất (phép đo của D11). Số D11 tái lập đúng: v3 0,733, lần 1 0,725, lần 2 0,603.

## Tóm tắt

- **Khuyên: train 2b′, nhưng ở dạng biến thể V-e** (đổi cách chọn 300 dòng 2d bỏ dấu ở C2(b) sang hash mù trên mọi dòng 2d; khối ViANLI, val, noul giữ nguyên từng byte). Nếu điều phối không muốn dựng lại thì **lần 1 nguyên trạng vẫn đáng chạy**; chênh giữa hai bản là bậc hai. Không chạy lần 2.
- **D11(a) gần như không đổi vì hai lẽ số học, không phải vì ViANLI vô dụng.** (i) Khối ≤ 5% (428 dòng R+N có dấu) không thể kéo probe fit trên 1.611 dòng KC+2d+2e từ 0,733 xuống 0,681 trừ khi tự nó tương quan ngược (đúng là điều lần 2 làm, probe riêng ViANLI = 1,000). (ii) **C2(b) tự triệt tiêu P2 trên khối có dấu:** 300 dòng 2d bỏ dấu được chọn đúng từ nhóm Xác nhận / Bác bỏ max_ov < 0,7, nên khối có dấu mất 117 dòng Bác bỏ cover < 0,7 của 2d trong khi ViANLI thêm 135 → **Bác bỏ cover < 0,7 có dấu chỉ từ 215 lên 233 (+18)**; P(Bác bỏ | cover < 0,5) đứng ở 12%, P(NEI | cover [0,5; 0,7)) còn tăng 38 → 47%. V-e đưa số này lên 296.
- **Probe trên khối gộp không phải proxy đúng cho cái model học.** Probe fit riêng KC xếp 57/74 Bác bỏ dev vào phía NEI (đúng hành vi 2b: 19/20 lần đoán NEI); probe fit trên cả khối v3 (KC+2d+2e) chỉ xếp 19/74. Tức model đã học luật của khối con nhất quán (KC), không học "trung bình" của khối. Proxy nên dùng: đếm dòng "sửa" ở vùng mẹo, probe riêng khối mới ≈ đa số (lần 1: 0,63 so với 0,60; lần 2: 1,00), và P(nhãn | dải) tính **cả hai chế độ**.
- **Đòn bẩy lớn nhất của 2b′ là noul, mà D11 không nhìn thấy:** v3 có 9 dòng "Có" ở cover < 0,9 (2–5% mỗi dải), 2b′ có 238 (27–28% ở hai dải thấp; dev 24–40%). Ô noul "Có" là ô 2b mất 7,5 câu so với C (≈ 0,6 macro) và là điểm cuối **ít nhiễu seed nhất** (|s2 − s1| = 1 câu).
- **Kỳ vọng sửa lại:** dev VI có dấu TB 2 seed **84,6–85,4 (trung tâm 84,9)** so với 2b 84,38; Bác bỏ ViWikiFC 0,72–0,80 (2b 0,689); "Có" noul 0,85–0,90 (2b 0,820); NEI ViFactCheck 0,76–0,82 (2b 0,809, rủi ro mất 1–2 câu); không dấu 63,5–65 (2b 62,85). **86,5 ngoài tầm** (P < 2%).
- **Công suất:** nhiễu seed 1 lần chạy σ ≈ 0,6–0,8 macro → 2 seed so với 2 seed phát hiện được **≥ 1,5–2,0 điểm** (80%), không phát hiện +0,5. Vì vậy luật đi/dừng phải dựa vào (a) sàn "không tệ hơn 2b", (b) hai điểm cuối ít nhiễu (recall "Có" noul, recall NEI ViFactCheck), (c) T3b trên TB 2 seed như cổng đã đăng ký; **không** dựa vào hiệu macro. Khẳng định "> 86,5" bằng TB 2 seed cần TB ≥ 87,2.
- **Hướng 1 kết thúc sau lần chạy này dù kết quả thế nào** (trừ lỗi khung lặp lại được). Phần còn lại tới 86,5 (decider hơn TB 2b ròng +2,1 macro; riêng ba nguồn MASSIVE/Belebele/XCOPA là +2,65, bù lại 2b hơn decider +1,05 ở ViFactCheck) là của hướng 2; 2b′ chỉ quyết **nền** cho hướng 2 là 2b hay 2b′. Nên bắt đầu phần CPU của hướng 2 ngay, song song với train.
- **Đọc mù:** đáng làm, nhưng thu gọn còn 60 dòng (30 Bác bỏ + 30 NEI của ViANLI, bản `tam/lan1/`, sha khớp manifest). Chỉ dừng train nếu ≥ 8/30 Bác bỏ bị đọc thành NEI hoặc Xác nhận, hoặc ≥ 3/60 dòng mà tiền đề ghép làm nhãn sai.

## 1. Lần 1 nguyên trạng: điều gì thực sự đổi so với v3

**1a. P(nhãn | dải cover), dòng choice (S/R/N, %).** "Có dấu" là tập D11 dùng; "cả hai" gồm cả dòng không dấu, là cái model thấy.

| Khối | [0; 0,5) | [0,5; 0,7) | [0,7; 0,9) | ≥ 0,9 |
|---|---|---|---|---|
| v3 có dấu | n=645: 11/12/**76** | n=479: 34/28/38 | n=501: 47/21/32 | n=1.053: 57/32/11 |
| 2b′ lần 1 có dấu | n=781: 18/**12**/70 | n=513: 26/26/**47** | n=615: 40/27/33 | n=1.126: 54/34/12 |
| 2b′ lần 1 cả hai | n=919: 20/16/64 | n=781: 31/30/39 | n=758: 40/27/33 | n=1.394: 52/35/13 |
| V-e có dấu (mục 2) | n=808: 19/15/66 | n=573: 31/30/38 | n=562: 39/27/34 | n=1.088: 52/36/12 |
| 2b′ lần 2 có dấu (bị bác) | n=886: 16/29/56 | n=419: 32/24/43 | n=578: 43/17/40 | n=1.161: 52/29/19 |
| dev ViWikiFC | n=31: 23/26/52 | n=69: 29/46/25 | n=51: 28/57/16 | n=10: 50/50/0 |
| proxy 2 nguồn | n=12: 25/8/67 | n=30: 33/40/27 | n=48: 31/29/40 | n=60: 48/30/22 |

Ở khối có dấu, lần 1 chỉ đổi cột Xác nhận (11 → 18% ở dải thấp: entailment của ViANLI là nhãn trùng từ **thấp nhất**, trung vị cover 0,50 so với Bác bỏ 0,69 và NEI 0,60). Bác bỏ ở dải thấp không nhúc nhích, và dải [0,5; 0,7) còn "NEI hơn" v3. Lý do ở bảng sau.

**1b. Bác bỏ choice có cover < 0,7, theo nguồn và chế độ**

| | 2d có dấu | ViANLI có dấu | KC | **Tổng có dấu** | 2d không dấu | ViANLI không dấu | **Tổng cả hai** |
|---|---|---|---|---|---|---|---|
| v3 | 215 | — | 0 | **215** | 0 | — | 215 |
| 2b′ lần 1 | 98 | 135 | 0 | **233** | 111 | 38 | 382 |
| V-e | 161 | 135 | 0 | **296** | 54 | 38 | 388 |

C2(b) bỏ dấu 169 Xác nhận + 131 Bác bỏ của 2d, tất cả max_ov < 0,7 (nhóm ưu tiên có 561 dòng). Đó là đề xuất P4 của tôi, và nó mâu thuẫn với P2 trên khối có dấu: dòng 2d trùng thấp vừa là nguồn Bác bỏ-trùng-thấp duy nhất của v3, vừa bị chuyển sang không dấu. Phần không dấu hưởng (0 → 149 Bác bỏ trùng thấp), phần có dấu — nơi tính 86,5 — gần như không.

**1c. Noul: P("Có" | dải cover), %** — thay đổi lớn nhất của 2b′

| | [0; 0,5) | [0,5; 0,7) | [0,7; 0,9) | ≥ 0,9 |
|---|---|---|---|---|
| v3 có dấu | n=128: **2** | n=101: **5** | n=125: **2** | n=791: 55 |
| 2b′ lần 1 có dấu | n=289: **28** | n=268: **27** | n=250: 13 | n=868: 51 |
| riêng ViANLI (cả hai) | n=205: 48 | n=212: 40 | n=163: 29 | n=95: 12 |
| dev 2 nguồn | n=21: 24 | n=53: 40 | n=67: 34 | n=61: 43 |
| proxy 2 nguồn | n=11: 18 | n=25: 40 | n=71: 31 | n=43: 44 |

Dải [0,7; 0,9) vẫn thiếu "Có" (13% so với dev 34%); hai dải thấp đã về mức dev. 9 câu "Có" dev mà 2b tự tin sai (p(Có) < 0,1) có max_ov 0,29–0,54, đúng hai dải vừa được sửa.

**1d. Dòng "sửa" so với dòng "củng cố" mẹo bao phủ** (cả hai chế độ): v3 có 215 Bác bỏ + 234 Xác nhận choice cover < 0,7 và 9 "Có" noul cover < 0,9 (458) so với 744 NEI cover < 0,7 + 396 "Không" cover < 0,9 (1.140), tỉ số **0,40**. 2b′ lần 1: 382 + 425 + 238 = 1.045 so với 893 + 747 = 1.640, tỉ số **0,64**. Khối ViANLI còn có ≈ 200 tiền đề dùng ≥ 2 lần với nhãn khác nhau (manifest: 211 dùng ≥ 2 lần; ép đọc nội dung), probe chỉ đọc nhận định 0,37 < đa số 0,40.

**1e. Probe gộp có đo đúng cái model học không? Không.** Áp chéo:

| Probe fit trên | Bác bỏ dev ViWikiFC bị xếp phía NEI (n=74) | Bác bỏ ViANLI phía NEI (n=258) | acc R/N trên ViANLI |
|---|---|---|---|
| riêng KC (acc 0,87) | **0,77** = 57/74, khớp 19/20 lần 2b đoán NEI | 0,68 | — |
| cả khối v3 (0,733) | 0,26 | 0,34 | 0,575 (đa số 0,60) |
| cả khối 2b′ lần 1 (0,725) | 0,27 | — | 0,579 |

Hành vi của 2b khớp luật của **riêng KC**, không khớp probe cả khối (2d nhiễu nhãn, model không fit theo). Vậy "acc probe trên khối gộp ≤ proxy + 5" đo sai thứ: nó bị KC chi phối và không thể đạt bằng một khối ≤ 5% trừ khi khối đó tương quan ngược. Thêm nữa, mốc proxy 0,631 (n=190) không ổn: cùng probe trên dev kiểm chứng cho 0,683 (n=303), riêng choice proxy 0,495 — tức dev/proxy thật cũng có một phần tách được bằng bề mặt. Ở riêng ViANLI, bề mặt gần vô dụng (0,63 so với đa số 0,60; worker 0,65 do chia fold khác) — đúng tính chất cần: 176/258 Bác bỏ ViANLI nằm trong "vùng NEI" của luật KC, là phản ví dụ sạch.

**Hai rủi ro còn lại, không đo được bằng CPU.** (i) **Định dạng:** state ViANLI là 3 câu báo ngắn (trung vị 121 token) khác hẳn đoạn Wikipedia / bài báo dài của dev; model có thể học luật có điều kiện "state kiểu ViANLI thì đọc nội dung, state dài thì dùng bao phủ". 2d (1.160 dòng, 67% Bác bỏ trùng thấp, câu 8–12 chữ dịch máy) đã không ngăn được mẹo ở 2b — bằng chứng rằng chỉ "có phản ví dụ" là chưa đủ. Đây là lý do chính kỳ vọng có đầu dưới ≈ 0. Dấu hiệu nhận ra sau s1: `val_vianli` cao (≥ 0,9) mà "Có" noul dev không đổi. (ii) **Nhãn ViANLI:** điều phối đọc 40 contradiction thấy 6 đáng ngờ + 5 phi lý; đọc mù ở mục 5.

**1f. Kỳ vọng sửa lại** (dev, TB 2 seed; dải chủ quan từ số câu ở trên, 1 câu ViFactCheck = 0,105 macro, ViWikiFC = 0,058)

| Điểm cuối | C-s1 | 2b-s1 / s2 / TB | decider | Kỳ vọng 2b′ TB | Ghi chú |
|---|---|---|---|---|---|
| VI macro có dấu | 81,65 | 84,84 / 83,92 / 84,38 | 86,50 | **84,6–85,4 (84,9)** | +0,2…+1,0; P(TB ≥ 86,5) ≈ 1–4% ngay cả ở đầu lạc quan 85,5 |
| "Có" noul gộp (n=75) | 0,920 | 0,827 / 0,813 / 0,820 | 0,920 | **0,85–0,90** | +2…+5 câu ≈ +0,2…+0,45 macro; đòn bẩy chính |
| Bác bỏ ViWikiFC choice (n=74) | 0,878 | 0,770 / 0,608 / 0,689 | 0,797 | **0,72–0,80** | +2…+8 câu ≈ +0,1…+0,45; yếu hơn tôi ước lúc đầu |
| Xác nhận ViWikiFC choice (n=46) | 0,870 | 0,848 / 0,848 | 0,826 | 0,85–0,89 | +0…+2 câu |
| NEI ViFactCheck choice (n=34) | 0,206 | 0,824 / 0,794 / 0,809 | 0,676 | **0,76–0,82** | P(NEI \| cover ≥ 0,7) không đổi (33/12%) → mất ≤ 2 câu; đây là "cái KC mua" (+20,5 câu ≈ +2,15 macro) |
| NEI ViWikiFC choice (n=41) | 0,463 | 0,780 / 0,610 / 0,695 | 0,707 | 0,62–0,72 | có thể mất 1–3 câu ở cover thấp |
| T3b XN+BB ViWikiFC (/120) | 105 | 96 / 84 / 90 | 97 | 90–97 | |
| Không dấu macro (G1) | 63,97 | 63,35 / 62,34 / 62,85 | 49,64 | 63,5–65,0 | 1.514 dòng không dấu, đủ ba nhãn, thay 587 KC |
| Đoán NEI ViWikiFC không dấu (/161, gold 41) | 40 | 91 / 97 | 57 | 65–80 | |
| EN macro | 88,20 | 88,80 / 88,54 | 90,33 | 88,3–89,0 | không đổi |

So với ước lượng cũ 85,4–86,1: mốc cũ là s1 (84,84) và gồm P1 (+0,2–0,4, nay bỏ); phần Bác bỏ hạ vì 1b; phần noul nâng vì 1c. Probe D11 tự nó không đổi ước lượng; hai sự thật tìm ra khi kiểm nó thì có.

## 2. Có biến thể tối thiểu tốt hơn không?

Mô phỏng bằng lọc dòng, không ghi file; D11 và bảng tính lại cùng code. "Sửa / củng cố" tính trên choice có dấu.

| Biến thể | Trong luật? | D11(a) có dấu | Probe riêng ViANLI | NEI choice (D7 ≥ 30%) | Bác bỏ / Xác nhận / NEI cover < 0,7 có dấu | Nhận xét |
|---|---|---|---|---|---|---|
| Lần 1 nguyên trạng | có | 0,725 | 0,63 (đa số 0,60) | 34,3% | 233 / 273 / 788 | mẹo có dấu gần như không đổi (1b) |
| **V-e: C2(b) chọn 300 dòng 2d theo hash mù trên mọi dòng 2d** (bỏ ưu tiên S/R max_ov < 0,7) | sai lệch có ngày với brief C2(b); đúng nguyên tắc chung "chọn theo hash, không theo nhãn"; không chạm ViANLI, KC, trần 1.580 | 0,724 | 0,63 | 34,3% | **296 / 336 / 749** | +63 Bác bỏ, +63 Xác nhận trùng thấp có dấu; hai dải thấp 19/15/66 và 31/30/38 (lần 1: 18/12/70 và 26/26/47). **Cái KC mua không chạm:** P(NEI \| cover ≥ 0,7) = 34/12%, bằng lần 1 (33/12%). Giá trả ở phía không dấu: 2d Bác bỏ cover < 0,7 không dấu 111 → 54; không dấu còn 2d S/R/N 165/84/51 + ViANLI 102/131/80, vẫn đủ ba nhãn → kỳ vọng G1 ở 1f lấy đầu dưới. Dựng lại vài phút CPU, D1/D3 tự chạy lại cho 300 id mới (logic bù theo hash khi bản không dấu chạm khử trùng đã có trong `build_2b_prime.py`); khối ViANLI, val, mẫu đọc mù **không đổi** |
| V-e′: không bỏ dấu 2d nào | bỏ hẳn C2(b) | 0,704 | 0,63 | 34,3% | 350 / 398 / 788 | mạnh hơn cho có dấu nhưng không dấu chỉ còn ViANLI + KC + ngoài khung; đi ngược P4 nhiều hơn |
| V-a: bỏ KC N-lạc đề + N-cùng bài (224 dòng, 136 choice) | **không** ("Không đổi KC"; sẽ là sai lệch KC thứ hai sau mục 9.7) | 0,715 | 0,63 | 31,9% | 233 / 273 / 686 | hai dải thấp 20/14/66 và 27/27/46: chỉ hạ P(NEI \| cover < 0,5) 70 → 66%, không thêm Bác bỏ nào; N-lạc đề là kiểu dạy T1 (P4 → NEI) cùng 2e; NEI ViFactCheck ở cover ≥ 0,7 (N-thêm, N-ghép; 32/12%) không chạm nên "cái KC mua" an toàn, nhưng NEI ViWikiFC cover thấp (52% gold) lộ. Lợi nhỏ, chi phí luật lớn |
| V-a2: bỏ ½ N-bỏ bằng + N-lạc đề + N-cùng bài (292 dòng) | không | 0,717 | 0,63 | 31,1% | 233 / 273 / 663 | hai dải thấp 20/14/66 và 28/28/43; như V-a, sát sàn D7; N-bỏ bằng (360 dòng, 50% cover < 0,7) là kiểu NEI lớn nhất của KC nên rủi ro T2 lớn hơn V-a |
| Bỏ bớt KC Bác bỏ (R-số/R-tên) | không | — | — | — | — | không thêm Bác bỏ trùng thấp nào, chỉ hạ prior Bác bỏ; mất recall Bác bỏ ViFactCheck (2b 0,875 ở dải cao). Không xét |
| V-d: lần 2 "nhẹ" (Bác bỏ lấy đều trong max_ov < 0,75, NEI lấy đều trong > 0,4; xấp xỉ trên tiền đề đơn) | lần sửa duy nhất đã dùng; cần điều phối cho phép lại | — | **0,69** (đa số 0,60) | 34,3% | ≈ +100 Bác bỏ trùng thấp | probe riêng ViANLI lên +9 điểm trên đa số: bắt đầu tương quan ngược; cùng lỗi lần 2 ở mức nhẹ. Không khuyên |
| Đổi cơ cấu ViANLI (thêm 80 dòng Bác bỏ tới trần 1.580) | có | ≈ | ≈ | ≈ | +≈ 45 | bậc ba; không đáng dựng lại |

**Khuyên một biến thể: V-e.** Nó sửa đúng chỗ tự triệt tiêu ở 1b, không đụng khối ViANLI (nên mẫu đọc mù và val còn giá trị), không đụng KC, không tốn GPU. Nếu điều phối không muốn ghi sai lệch brief C2(b), chạy lần 1 nguyên trạng; chênh V-e so với lần 1 (+63 dòng Bác bỏ có dấu trùng thấp, +27%) là bậc hai so với rủi ro định dạng ở 1e. **Cổng D11(a) khi dựng lại V-e:** giữ số làm chỉ báo, thay cổng đạt/trượt bằng (i) probe riêng ViANLI ≤ đa số + 10 điểm (chống tương quan ngược; lần 1 đạt, lần 2 trượt), (ii) báo bảng 1a và 1b cả hai chế độ. Lý do ghi ở 1e: cổng cũ không thể đạt bằng khối ≤ 5%.

## 3. Công suất / MDE

**Nhiễu seed trên dev** (2b-s1 so với 2b-s2; `h2`):
- VI macro |Δ| = 0,92. σ của **một** lần chạy: 0,65 (|Δ|/√2), 0,82 (ước lượng E|Δ|), 0,61 (mô hình lật ngẫu nhiên từ số câu đổi đúng/sai theo nguồn) → **σ₁ ≈ 0,6–0,8**.
- ViWikiFC có dấu: 37 câu đổi đúng/sai, hiệu ròng −19 câu (−6,7 điểm); nếu lật ngẫu nhiên SD chỉ 2,1 điểm → z = −3,1. **Nhiễu seed ở ViWikiFC có chiều**, là độ mạnh của mẹo thay đổi theo seed, không phải lật ngẫu nhiên. Mọi điểm cuối Bác bỏ/NEI ViWikiFC vì thế nhiễu lớn: σ₁ recall Bác bỏ 0,115, NEI ViWikiFC 0,121, T2 0,075, T3a 0,066, T3b 8,5 câu.
- Điểm cuối **ít nhiễu**: "Có" noul gộp σ₁ 0,010 (1 câu), NEI ViFactCheck 0,021 (1 câu), acc ViFactCheck 1,8 điểm, Xác nhận ViWikiFC 0 câu.
- Proxy (n nhỏ) lặp lại cả ba cơ chế: Bác bỏ ViWikiFC choice 21/22 (C) → 16 / 14 (s1 / s2); "Có" noul 49/53 → 39 / 44; NEI ViFactCheck 5/24 → 13 / 15.

**MDE** (một phía 5%, công suất 80% = 2,49σ; 50% = 1,645σ):

| So sánh | SD của hiệu | MDE 80% | MDE 50% |
|---|---|---|---|
| TB 2 seed 2b′ so với TB 2 seed 2b (84,38) | σ₁ = 0,6–0,8 | **1,5–2,0** | 1,0–1,35 |
| 1 seed 2b′ so với TB 2 seed 2b | σ₁·√1,5 = 0,75–1,0 | 1,9–2,5 | 1,2–1,65 |
| TB 2 seed so với mốc cố định 86,5 | σ₁/√2 = 0,43–0,58 | cần TB ≥ **87,2–87,5** để khẳng định > 86,5 | P(TB ≥ 86,5 \| μ = 85,0) ≤ 0,5%; μ = 85,5: 1–4% |
| recall "Có" noul TB (n=75) | 0,010 | 0,025 ≈ 2 câu | |
| recall NEI ViFactCheck TB (n=34) | 0,021 | 0,053 ≈ 2 câu | |
| recall Bác bỏ ViWikiFC TB (n=74) | 0,115 | 0,29 | ngưỡng cố định: P(TB ≥ 0,78 \| không đổi) ≈ 13%, P(TB ≥ 0,82 \| không đổi) ≈ 5% |
| T3b TB (/120) | 0,071 | 21 câu | P(TB ≥ 94 \| không đổi, TB 2b = 90) ≈ 25% |

Nhiễu mẫu dev (bootstrap cụm, cho khả năng chuyển sang test; không phải nhiễu seed): TB 2b − C-s1 = 2,70 ± 0,94; decider − TB 2b = 2,12 ± 0,99 [0,13; 3,97].

**Kết luận:** 2 seed của 2b′ **không** phân biệt được +0,5 so với 2b, **không** khẳng định được 86,5, nhưng **có** đọc được: (a) 2b′ không tệ hơn 2b quá 1,5 điểm; (b) ô noul "Có" và ô NEI ViFactCheck đổi ≥ 2 câu; (c) với Bác bỏ ViWikiFC chỉ đọc được hiệu lớn (≥ +0,13) hoặc dùng ngưỡng với tỉ lệ đạt giả đã ghi. Vì hiệu kỳ vọng nằm dưới MDE của macro, luật ở mục 4 dựa vào (a)–(c), và 2 seed là **đủ để quyết nền cho hướng 2**, không đủ để tuyên bố mức tăng.

## 4. Luật đọc đăng ký trước cho 2b′

Mốc so sánh là **TB 2 seed của 2b** (84,38; các ô ở bảng 1f), không phải s1. Mọi recall đo ở argmax (trước ngưỡng P1). Dòng P1 báo riêng, không dùng để quyết. T1 (≥ 85% cả hai), letter_mass, G2 (EN ≥ 87,20), G5 giữ như cũ. Ngưỡng dưới chọn để P(đạt giả | không đổi) ≈ 5–15% ở ô nhiễu và ≈ 2 câu ở ô ít nhiễu.

**Bậc 1 — seed 1 chỉ để DỪNG, không để kết luận tốt** (s1 tốt có thể là may seed: 2b-s1 hơn s2 6,7 điểm ViWikiFC trong khi proxy ngược chiều):

| Mã | Luật trên 2b′-s1 | Nếu trượt |
|---|---|---|
| S1 | VI macro ≥ 83,4 (= 84,38 − 1,0; SD của s1 − TB 2b là 0,75–1,0 → P(dừng \| không đổi) ≈ 11–16%) | dừng, không chạy s2; nền hướng 2 = 2b |
| S2 | recall NEI ViFactCheck choice ≥ 0,70 (24/34; 2b 0,81; C 0,21) | dừng: mất cái KC mua |
| S3 | recall "Có" noul gộp ≥ 0,80 (60/75; 2b 0,820) | dừng: cơ chế noul không hoạt động (σ₁ = 1 câu nên −2 câu là thật) |
| S4 | acc ViFactCheck có dấu ≥ 85,0 (134/158; 2b 88,6 − 2σ₁, σ₁ = 1,8) | dừng |
| S5 (chỉ báo) | `val_vianli` ≥ 0,85 **và** "Có" noul gộp < 0,84 → ghi "học có điều kiện theo định dạng" (1e); không dừng một mình, nhưng là lý do kết thúc hướng 1 dù TB sau đó đạt sàn | |
| Đạt S1–S4 | **chạy s2 bất kể s1 cao bao nhiêu** | |

**Bậc 2 — TB 2 seed quyết:**

| Mã | Luật (TB 2 seed) | Vai |
|---|---|---|
| A1 | VI macro ≥ 84,38 (không tệ hơn 2b); vùng xám [83,9; 84,38) | sàn; **không** đòi "+0,5" vì dưới MDE |
| A2 | recall "Có" noul gộp ≥ 0,867 (65/75; +3,5 câu so với 2b, trong khi nhiễu seed ở ô này là 1 câu) | bằng chứng cơ chế noul; C 0,920 |
| A3 | T3b XN+BB ViWikiFC ≥ 94/120 (cổng đã trượt ở 2b; với XN 39/46 cần BB ≥ 55/74 = 0,74). Chỉ báo kèm: recall Bác bỏ ViWikiFC ≥ 0,78 và **cả hai seed ≥ 0,70** (2b-s2 0,608) | cơ chế Bác bỏ; P(đạt giả) ≈ 25% nên chỉ đọc cùng A1–A2 |
| A4 | recall NEI ViFactCheck ≥ 0,75 (25,5/34; nhiễu seed 1 câu) **và** acc ViFactCheck ≥ 86,0 (2b 88,6; σ_TB 1,3) | bảo vệ "cái KC mua"; P(trượt \| không đổi) ≈ 2% |
| A4′ (chỉ báo) | T2 recall NEI gộp ≥ 0,64 và T3a precision NEI gộp ≥ 0,63 (= TB 2b − 2σ_TB; 2b 0,746 / 0,726). Ở mức cổng cũ 0,70 hai luật này bác nhầm một model **không đổi** 19% / 29% vì mang nhiễu NEI ViWikiFC (σ₁ 0,121); và 2b′ được thiết kế để kéo NEI ở dải trùng thấp xuống nên T2 dự kiến giảm nhẹ | chỉ báo: trượt thì ghi, không bác |
| A5 | G1 không dấu ≥ 62,85 (sàn); kỳ vọng ≥ 63,9; đoán NEI ViWikiFC không dấu ≤ 72/161 (45%; 2b 58%, C 25%) | P4; chỉ cột độ bền, 0 điểm cho 86,5 |
| A6 | EN ≥ 87,20; \|s2 − s1\| VI macro ≤ 2 (nếu > 2: báo, không chạy thêm seed) | |

**Quyết định:**
- **A1 + A2 + A4 đạt** (A3 có hay không) → 2b′ là **nền mới**; ghi A3 đạt/trượt làm kết quả cơ chế Bác bỏ; đóng hướng 1; hướng 2 dựng trên dữ liệu 2b′.
- **A1 vùng xám, A2 + A4 đạt, A5 đạt** → 2b′ "tương đương 2b, noul sửa được, Bác bỏ chưa" → vẫn lấy 2b′ làm nền (không tệ hơn, bền hơn khi không dấu), đóng hướng 1.
- **A1 < 83,9 hoặc A4 trượt** → bác 2b′; nền = 2b; đóng hướng 1.
- **A2 trượt mà A1 đạt** → noul không sửa được bằng dữ liệu kiểu này; nền = cái nào cao hơn TB; đóng hướng 1; **không** thử 2b″.
- **Dừng hướng 1, sang hướng 2:** vô điều kiện sau lần chạy này. Lý do: trần còn lại ở kiểm chứng ≤ ≈ 1,5 macro và dưới MDE; khoảng cách tới 86,5 nằm ở MASSIVE (+1,13 so với decider), Belebele (+0,85), XCOPA (+0,67). Ngoại lệ duy nhất: lỗi khung / chuỗi (T1, letter_mass, D1) lặp lại được → sửa và chạy lại 1 seed. **Không** đọc "> 86,5" từ dev TB 2 seed trừ khi TB ≥ 87,2; tuyên bố 86,5 để cuối hướng 2, với giao thức test đăng ký riêng.
- **Test:** không chấm test cho 2b′ (giữ cho mốc cuối).

**Tỉ lệ lỗi của bộ luật (tự tính từ σ ở mục 3):** bậc 1 dừng nhầm khi 2b′ thật ra bằng 2b: S1 11–16% + S2 ≈ 2% + S3 ≈ 5% + S4 ≈ 2–3% → **≈ 20–25%**; khi 2b′ tốt hơn +0,5: ≈ 10%. Bậc 2: A1 < 83,9 bác nhầm 16–21% dưới giả thuyết "bằng 2b" (vô hại: hai model bằng nhau thì nền nào cũng được), 2–4% khi 2b′ = 84,9; A4 ≈ 2%. A5 sàn 62,85 = TB 2b nên là đồng xu dưới giả thuyết không đổi; chấp nhận vì dữ liệu không dấu đổi lớn (587 → 1.514 dòng, đủ ba nhãn) và A5 chỉ là cột độ bền, không quyết nền.

## 5. Đọc mù `mau_doc_vianli.jsonl` (bản `tam/lan1/`)

Kiểm cơ học: 110 dòng = 30/30/30 ViANLI (69 choice, 41 noul, 10 không dấu) + 20 dòng C2(c); sha khớp `manifest_2b_prime.json` lần 1; nhãn ở cột riêng. Tiền đề ghép: 87/1.472 dòng có tiền đề ghép trùng từ với giả thuyết ≥ tiền đề thật (XN 37 / BB 22 / NEI 15), chỉ 5 dòng ≥ 0,5; 28/1.500 state không tách được đúng 3 câu (vô hại). Rủi ro "tiền đề ghép vô tình xác nhận / bác bỏ" nhỏ nhưng chưa ai đọc.

**Đáng làm, thu gọn:** chỉ 60 dòng ViANLI (30 Bác bỏ + 30 NEI); bỏ 30 Xác nhận và 20 dòng C2(c) (kiểm chuỗi đã làm cơ học, D2). Lý do: nhãn là **lỗi duy nhất mà D1–D12 không nhìn thấy**, và nằm đúng ranh Bác bỏ / NEI đang sửa; chi phí ≈ 60 × 150 token đọc + vài nghìn token trả lời, nhỏ so với 4–8 giờ GPU. Phải đọc mù thật (session mới, không thấy cột `nhan`), xếp mỗi dòng vào XN / BB / NEI và cờ "nhãn đúng nhờ/tại tiền đề ghép". Người dùng quyết vì chạm quota (quy tắc không dùng thêm usage; Opus là model được chỉ định cho việc đọc kiểm).

**Kết quả đổi quyết định:**
- Bác bỏ bị đọc thành NEI hoặc XN **≥ 8/30** (≥ 27%; dev ≈ 12%, điều phối đọc 15%) → không train lần 1 / V-e: khối dạy "Bác bỏ trông như NEI" ở đúng ranh cần sửa; không có cách lọc cơ học → sang hướng 2 trên 2b.
- NEI bị đọc thành XN/BB ≥ 8/30 → như trên (nhãn neutral của ViANLI là thứ giữ NEI ở dải trùng cao).
- ≥ 3/60 dòng mà người đọc chỉ ra tiền đề ghép làm nhãn sai → sửa bộ lọc ghép (CPU), dựng lại, rồi train.
- Khớp ≥ 23/30 ở cả hai nhãn và ≤ 2 lỗi ghép → train như mục 2. Khớp 20–22/30 → train, ghi vào giới hạn, hạ đầu trên kỳ vọng 0,3.

## 6. Script (thư mục `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2bp\`; chạy `python -X utf8 -I <script>`)

| Script | Việc | Đầu ra |
|---|---|---|
| `kcu.py` | nạp đúng hàm đo của `scripts/train_2b/kc_util.py` (không cần datasketch), `feats` của `phan_tich_bacbo.py`, `is_fc` / `dia` của `build_2b_prime.py` | — |
| `h1_block_tables.py` | bảng P(nhãn \| dải) max_ov / cover, choice và noul, có dấu / cả hai, v3 / lần 1 / lần 2 / dev / proxy; Bác bỏ trùng thấp theo nguồn × chế độ; tái lập D11 (0,733 / 0,725 / 0,603) và áp chéo (KC, v3, lần 1 → ViANLI, dev, proxy); dòng sửa / củng cố; mô phỏng V-e, V-e′, V-a, V-a2, V-a3, V-e+V-a3, V-d (xấp xỉ pool); tiền đề ghép liên quan; kiểm mẫu đọc mù | `h1_out.json`, `h1_out.txt` |
| `h2_power.py` | macro và điểm từng nguồn của C-s1 / 2b-s1 / 2b-s2 / decider; nhiễu seed theo nguồn (lật, z so với lật ngẫu nhiên); 19 điểm cuối theo seed (định nghĩa như `bang_tg_bac2.py`); σ₁ và MDE cho macro và từng điểm cuối; bootstrap cụm dev; điểm cuối trên proxy | `h2_out.json`, `h2_out.txt` |

Không đọc `test_v1.3.jsonl`, không chạy model, không GPU, không tải gì, không ghi file dữ liệu. Không ghi chuỗi tiếng Việt mới vào dữ liệu hay config.
