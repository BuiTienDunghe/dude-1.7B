# Phân tích lỗi Bác bỏ ViWikiFC theo dòng (29/10)

- **Yêu cầu:** người dùng, sau REPORT_2b_bac2.md.
- **Phạm vi:** chỉ CPU, chỉ đọc preds đã có. Không chấm model, không đọc test. Dữ liệu UIT chỉ dùng nội bộ.
- **Script:** `scripts/train_2b/phan_tich_bacbo.py`.
- **Đầu ra:**
  - `results/train_2b/tam/phan_tich_bacbo.json` (số liệu);
  - `results/train_2b/tam/bacbo_dong.jsonl` (từng dòng: đặc trưng, đoán và xác suất của C-s1, 2b-s1, 2b-s2).
- **Dòng phân tích:** dev 1.3, ViWikiFC, choice có dấu, gold Bác bỏ, n = 74. Phần đọc tay ở mục 4 là của một người đọc (worker), không đọc mù.

## 1. Tóm tắt

1. **Lỗi là nhầm Bác bỏ thành NEI, không phải mất khả năng nhận ra "không được xác nhận".**
   - Choice có dấu: 2b-s2 đoán NEI 20/74 câu Bác bỏ (C 3; s1 14), đoán Xác nhận 9 (C 6).
   - Cùng các nhận định đó ở dạng Có/Không ("Văn bản có xác nhận…không?"): C 1,000, s1 0,971, s2 0,971. Dạng này gộp Bác bỏ và NEI vào "Không", nên không bị ảnh hưởng.
   - Xác nhận choice có dấu không đổi: C 0,870, s1 và s2 0,848.
2. **Lỗi tập trung ở câu có độ trùng từ thấp.** Acc của s2 theo max_ov:

| max_ov | n | acc s2 |
|---|---|---|
| < 0,5 | 12 | 0,333 |
| 0,5–0,7 | 33 | 0,576 |
| 0,7–0,9 | 26 | 0,769 |

   Ở C, các mức tương ứng là 0,750 / 0,848 / 0,962.
3. **Nguyên nhân dữ liệu:**
   - Trong KC, Bác bỏ chỉ có 2 kiểu dựng: **R-so** (đổi số) và **R-ten** (đổi tên riêng). Cả hai đều chép gần nguyên câu (max_ov trung vị 0,88–0,94).
   - NEI của KC (trừ N-them) có độ trùng thấp (trung vị 0,13–0,52).
   - Vì vậy "không được xác nhận + trùng từ thấp" gần như luôn là NEI trong KC.
   - Bác bỏ của dev phần lớn là kiểu khác, xem mục 4: lượng từ hoặc độc quyền, phủ định, thay danh từ hay động từ chung, đảo vai, suy luận số hoặc thời gian. Chúng trùng từ thấp hơn (trung vị 0,667; 61% dưới 0,7). Chỉ 10,8% có số lạ và 2,7% có tên lạ.
4. **Probe bề mặt xác nhận điều này** (mục 3).
   - Một logistic regression chỉ dùng đặc trưng trùng từ, phủ định, số, tên phân biệt được KC Bác bỏ với KC NEI 88,5%. Trọng số lớn nhất là độ phủ token.
   - Áp lên dev, probe xếp **57/74** câu Bác bỏ vào phía "NEI của KC".
   - Ở phía đó s2 chỉ đúng 0,544 (s1 0,719); ở phía còn lại đúng 0,824 (s1 0,941).
   - **19/20** lần s2 đoán NEI cho câu Bác bỏ rơi vào phía NEI.
   - Câu Xác nhận cũng phần lớn ở phía NEI (39/46) nhưng gần như không bị đoán NEI (1/46). Model vẫn nhận ra quan hệ xác nhận; chỉ ranh giới Bác bỏ / NEI dời theo độ trùng từ.
5. **Nhiễu seed lớn trên đúng tập này.**
   - C đúng nhưng mất ở cả hai seed: 11 câu.
   - Chỉ mất ở s2: 11 câu. Chỉ mất ở s1: 2 câu.
   - ViWikiFC có dấu là nguồn có tỉ lệ trả lời khác nhau giữa hai seed cao nhất (13,3%).
   - Nghĩa là chi phí có hệ thống (11 câu mất ở cả hai seed), và độ lớn dao động mạnh theo seed.

## 2. Số theo nhóm (choice có dấu, ViWikiFC)

| Gold | n | C-s1 | 2b-s1 | 2b-s2 | Đoán của s2 (XN / BB / NEI) |
|---|---|---|---|---|---|
| Bác bỏ | 74 | 0,878 | 0,770 | 0,608 | 9 / 45 / 20 |
| Xác nhận | 46 | 0,870 | 0,848 | 0,848 | 40 / 5 / 1 |
| NEI | 41 | 0,463 | 0,780 | 0,610 | 10 / 5 / 26 |

**Xác suất trung bình trên câu Bác bỏ** (thứ tự XN / BB / NEI):

| Mô hình | XN | BB | NEI |
|---|---|---|---|
| C | 0,115 | 0,769 | 0,116 |
| s1 | 0,063 | 0,713 | 0,224 |
| s2 | 0,118 | 0,624 | 0,258 |

**Mẫu đúng/sai C → s1 → s2** trên 74 câu Bác bỏ (Đ = đúng, S = sai):

| Mẫu | ĐĐĐ | ĐĐS | ĐSS | ĐSĐ | SĐS | SĐĐ | SSS |
|---|---|---|---|---|---|---|---|
| Số câu | 41 | 11 | 11 | 2 | 3 | 2 | 4 |

- 11 câu mất ở cả hai seed: cả s1 và s2 đều đoán NEI 8 câu, Xác nhận 3 câu.

**Theo đặc trưng** (acc C / s1 / s2):

| Đặc trưng | n | C | s1 | s2 |
|---|---|---|---|---|
| Nhận định có phủ định | 23 | 0,870 | 0,870 | 0,739 |
| Nhận định không có phủ định | 51 | 0,882 | 0,725 | 0,549 |
| Có số không có trong văn bản | 8 | 1,000 | 0,750 | 0,250 |
| Không có số lạ | 66 | 0,864 | 0,773 | 0,652 |

Nhóm có số lạ chỉ có n = 8, nên không đọc được nhiều từ nhóm này.

**Không dấu** (cùng 74 câu), có cùng chiều và nặng hơn:

| | C | s1 | s2 |
|---|---|---|---|
| Acc | 0,554 | 0,446 | 0,351 |
| Số câu đoán NEI | 19 | 39 | 40 |

## 3. So đặc trưng dòng Bác bỏ: dev và train_2b_v3 (có dấu)

| Nhóm | n | max_ov trung vị | max_ov < 0,7 | Phủ định trong nhận định | Số lạ | Tên lạ | Độ dài trung vị |
|---|---|---|---|---|---|---|---|
| **Dev ViWikiFC** | 74 | **0,667** | **60,8%** | 31,1% | 10,8% | 2,7% | 20 |
| KC ngắn R-so | 230 | 0,933 | 29,6% | 10,0% | 69,1% | 0,0% | 30 |
| KC ngắn R-ten | 170 | 0,875 | 30,0% | 7,1% | 1,2% | 74,1% | 30 |
| KC dài R-so | 137 | 0,938 | 28,5% | 10,9% | 38,7% | 0,0% | 31 |
| KC dài R-ten | 100 | 0,905 | 31,0% | 17,0% | 0,0% | 49,0% | 35 |
| nli_fever | 143 | 0,545 | 72,7% | 27,3% | 9,1% | 4,9% | 8 |
| vitaminc | 183 | 0,571 | 72,7% | 1,6% | 37,2% | 19,7% | 12 |

**KC NEI theo kiểu**, để đối chiếu:

| Kiểu | max_ov trung vị | max_ov < 0,7 |
|---|---|---|
| N-bobang | 0,29–0,33 | ≥ 98% |
| N-ghep | 0,50–0,52 | ≈ 90% |
| N-cungbai | 0,24–0,26 | 100% |
| N-lacde | 0,13–0,16 | 100% |
| N-them | 0,88–0,90 | ≈ 2% |

Trong KC, độ trùng thấp là dấu hiệu NEI gần như tuyệt đối. Bác bỏ trong train có độ trùng thấp chỉ đến từ nli_fever và vitaminc (326 dòng, câu ngắn, dịch). Còn KC Bác bỏ, 637 dòng, đều trùng cao.

## 4. Đọc tay 24 câu C đúng nhưng ít nhất một seed 2b sai

| Kiểu bác bỏ | Số câu | Cách bác bỏ |
|---|---|---|
| Lượng từ hoặc độc quyền | 6 | dùng "chỉ", "độc nhất", "mới", "một góc" (văn bản nói "phần lớn"), "chỉ một phần… 1.000 km" |
| Phủ định | 5 | phủ định tường minh hoặc ngầm: "không đa dạng", "không thấy ngọn núi nào", "chưa bao giờ ra đời" (văn bản nói đã bắt đầu soạn), "không có nhiều ý nghĩa" |
| Thay danh từ hoặc động từ chung | 4 | thay danh từ / động từ thường, không phải tên hay số: "doanh nhân" ↔ "đại tướng", "tranh chấp về đảo" ↔ "về vùng đặc quyền kinh tế", "truyền bá văn hóa" ↔ "buôn bán" |
| Đảo vai hoặc sự kiện ngược | 3 | "cướp lại" ↔ "trao trả", "Liễu Thăng diệt Lê Lợi", "thoát chết" ↔ bị giết |
| Suy luận số hoặc thời gian | 3 | "1 thập kỷ" ↔ "hơn 20 năm", "đầu thế kỷ 20" ↔ "năm 1988", 1428 ↔ 1427 |
| Nhãn đáng ngờ | 3 | đọc giống NEI hơn: nhận định thêm hệ quả hay chi tiết mà văn bản không nói |

- **Gần kiểu R-so hoặc R-ten của KC: chỉ 1/24 câu** (1428 ↔ 1427). s1 làm đúng câu này, s2 đoán NEI.
- 5 kiểu bác bỏ đầu không có trong KC Bác bỏ. Phủ định ở dạng câu dài cũng gần như vắng.
- Các kiểu này trùng từ thấp hơn, nên rơi vào vùng mà KC gán NEI.
- Ba câu nhãn đáng ngờ chiếm 3/74 ≈ 4 điểm recall. Dù bỏ ba câu này, mức tụt vẫn còn.

Mã từng câu nằm trong `results/train_2b/tam/bacbo_dong.jsonl`: lọc gold = refuted, dung_C-s1 = true, và ít nhất một seed sai.

## 5. Gợi ý cho bước sửa dữ liệu

Phần này chỉ để tham khảo. Điều phối và người dùng quyết; worker không quyết bước tiếp.

Mục 11 của PHAN_TICH_TRUOC_TRAIN_2b đã ghi hướng "sửa dữ liệu cho lỗi Bác bỏ ở độ trùng từ thấp". Số ở trên gợi ý ba thứ cần sửa:

1. **Thêm kiểu Bác bỏ cho KC**, chọn đúng các kiểu đang thiếu:
   - lượng từ hoặc độc quyền ("chỉ", "duy nhất", "đầu tiên", "phần lớn" ↔ "một phần");
   - phủ định chèn vào câu dài;
   - động từ ngược nghĩa hoặc đảo vai chủ thể / đối tượng;
   - thay danh từ chung chỉ thuộc tính hay nghề nghiệp;
   - đổi mốc thời gian dưới dạng diễn đạt khác ("thập kỷ", "thế kỷ").
2. **Phá tương quan "trùng thấp ↔ NEI"** trong KC:
   - có Bác bỏ dựng trên câu đã diễn đạt lại, trùng từ khoảng 0,4–0,7;
   - ngược lại, có NEI trùng cao như N-them, nhưng không chỉ một kiểu.

   Đo lại bằng chính probe ở mục 3. Mục tiêu: probe bề mặt trên KC không còn tách được Bác bỏ / NEI tốt hơn nhiều so với tỉ lệ đa số.
3. **Không dấu:** cùng cơ chế, nặng hơn (s2 đoán NEI 40/74). Cần đọc riêng khi sửa.

**Giới hạn của phân tích:**
- n = 74. Đọc tay 24 câu do một người, không đọc mù.
- Probe chỉ cho thấy tương quan bề mặt, không chứng minh nhân quả.
- Nhiễu seed trên tập này lớn (11 câu chỉ mất ở s2). So sánh nào trên ViWikiFC Bác bỏ cũng nên dùng trung bình ≥ 2 seed.
