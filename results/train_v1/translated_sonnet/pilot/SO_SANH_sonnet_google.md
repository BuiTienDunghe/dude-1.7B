# So sánh 200 dòng dịch thử: Sonnet và Google Translate (agent ngoài)

Ngày 2026-10-19. Yêu cầu của người dùng (chat 19/10): "so thử 200 dòng đó với 200 dòng của agent xem cái nào tốt hơn".

Phạm vi đọc bản ngoài:
- Chỉ đọc `results/train_v1/translated/ban_dich.jsonl` để lấy đúng 200 id của mẫu. Bản trích lưu ở `so_sanh/google_200.jsonl`, chỉ để so sánh.
- Không dùng bản đó để train hay làm nháp; không sửa gì trong thư mục `translated/`.

Không train, không commit, không xóa file.

## Kết luận

**Bản Sonnet (dịch + đọc lại) tốt hơn rõ rệt.**

| Chỉ số (200 dòng, chấm mù) | Sonnet | Google |
|---|---|---|
| Thắng | **149 (74%)** | 5 (2%) |
| Ngang | 46 (23%) | 46 (23%) |
| Dòng có lỗi đổi đáp án | **0** | **18 (9%)** |
| Dòng có lỗi đổi nghĩa (kể cả lỗi nhỏ) | 3 (2%) | 107 (54%) |
| Độ tự nhiên trung bình (1–5) | 4,42 | 3,48 |

- Google có 18 dòng mà người đọc chỉ có bản tiếng Việt sẽ không còn chọn được đáp án gốc. Dùng 18 dòng này để train là dạy nhãn sai.
- Bản Google cũng dịch sai nhãn "None of the other options" ở mọi chỗ trong mẫu (mục 1).

## 1. Kiểm khách quan (không dùng người chấm)

| | Sonnet | Google |
|---|---|---|
| `kiem_ban_dich.py` | ĐẠT, 0 lỗi, 2 cảnh báo (1 thiếu số do lỗi gõ "I'11" của bản gốc, 1 lựa chọn ngắn) | ĐẠT, 0 lỗi, 4 cảnh báo (2 thiếu số, 2 lựa chọn lệch độ dài) |
| Chuỗi lặp (56 chuỗi của bảng thuật ngữ có trong mẫu): số chuỗi dịch hơn 1 kiểu | 0 | 0 |
| "None of the other options ." (2 lần trong mẫu) | "Không lựa chọn nào khác đúng." (đúng) | "Không có lựa chọn nào khác." (**sai nghĩa**) |
| "None of the above choices ." (4 lần) | "Không lựa chọn nào ở trên đúng." | "Không có lựa chọn nào ở trên." (**sai nghĩa**) |

- Cả hai bản đều nhất quán ở chuỗi lặp; agent ngoài cũng dùng một bảng thuật ngữ. Kiểm định dạng không phân biệt được hai bản; khác biệt nằm ở nghĩa.
- Ngoài ra người chấm phát hiện một ký tự ẩn (zero-width) trong một dòng của bản Google (c015). `kiem_ban_dich.py` hiện không bắt loại lỗi này; nên thêm phép kiểm trước khi nhận bản dịch đầy đủ.

## 2. Chấm mù theo cặp (Opus)

**Cách làm** (`scripts/train_v1/so_sanh_prep.py`, `so_sanh_report.py`):
- Mỗi dòng đưa cho người chấm: bản EN, bản X, bản Y, và đáp án đúng.
- X/Y xáo theo seed 20261019, không ghi nguồn: Sonnet là X ở 100 dòng, là Y ở 100 dòng. Bảng giải mã `cham_map.json` người chấm không đọc.
- 5 subagent Opus, mỗi lô 40 dòng; kiểm mức dùng trước mỗi lô.
- Mỗi bản được chấm:
  - (a) có lỗi đổi nghĩa không, ở chỗ nào;
  - (b) lỗi đó có làm đổi đáp án không;
  - (c) độ tự nhiên 1–5.

  Sau đó người chấm kết luận bản nào tốt hơn hoặc ngang.
- Đầu vào và kết quả: `so_sanh/cham/lo_{k}_in.jsonl`, `lo_{k}_out.jsonl`; tổng hợp ở `so_sanh/tong_hop.json`.

**Theo dạng bài:**

| Dạng bài | n | Sonnet thắng | Google thắng | Ngang | Đổi nghĩa: Sonnet | Đổi nghĩa: Google | Đổi đáp án: Sonnet | Đổi đáp án: Google | Tự nhiên TB: Sonnet | Tự nhiên TB: Google |
|---|---|---|---|---|---|---|---|---|---|---|
| Đọc hiểu | 40 | 39 (98%) | 0 (0%) | 1 (2%) | 0 (0%) | 34 (85%) | 0 (0%) | 4 (10%) | 4,58 | 3,10 |
| Ý định | 40 | 32 (80%) | 1 (2%) | 7 (18%) | 0 (0%) | 13 (32%) | 0 (0%) | 1 (2%) | 4,42 | 3,38 |
| Kiểm chứng | 40 | 21 (52%) | 0 (0%) | 19 (48%) | 0 (0%) | 14 (35%) | 0 (0%) | 4 (10%) | 4,40 | 3,92 |
| Yes/no | 40 | 35 (88%) | 1 (2%) | 4 (10%) | 1 (2%) | 32 (80%) | 0 (0%) | 7 (18%) | 4,33 | 3,23 |
| Nhân quả | 40 | 22 (55%) | 3 (8%) | 15 (38%) | 2 (5%) | 14 (35%) | 0 (0%) | 2 (5%) | 4,38 | 3,80 |
| **Tổng** | **200** | **149 (74%)** | **5 (2%)** | **46 (23%)** | **3 (2%)** | **107 (54%)** | **0 (0%)** | **18 (9%)** | **4,42** | **3,48** |

- **Vị trí X/Y không thiên lệch:** Sonnet thắng 75/100 dòng khi là X và 74/100 dòng khi là Y.
- **Độ tự nhiên:** 192/200 dòng Sonnet đạt 4–5 điểm; Google 91/200 dòng.
- **Lỗi đổi nghĩa** được người chấm tính cả lỗi nhỏ (một từ lệch nghĩa, tên mất dấu), nên tỷ lệ 54% của Google gồm nhiều lỗi vặt. Chỉ số quan trọng là **đổi đáp án**: 9% so với 0%.
- **Lỗi của Google tập trung ở:**
  - dịch cả tên riêng: tên phim, sách, ban nhạc, sản phẩm, như "Foxcatcher" → "Đớp cáo", "for King and Country" → "dành cho nhà vua và đất nước";
  - hiểu sai nghĩa theo ngữ cảnh;
  - bỏ chữ trong câu hỏi Có/Không;
  - nhãn "None of …".
- **Dòng Google thắng hoặc Sonnet có lỗi:**
  - **Google thắng 5 dòng**, đều không phải lỗi đổi đáp án của Sonnet:
    - Sonnet giữ kiểu viết hoa từng chữ của tiêu đề tin tiếng Anh (c042);
    - Google dùng tên Việt hóa quen thuộc ("Thanh Hải-Tây Tạng", "Mỹ") ở c046, c062;
    - "maternal woman" (c183): đây chính là chỗ Sonnet 2 bỏ sót, đã nêu trong `REPORT_pilot.md`;
    - một nhãn ý định Google dịch sát hơn (c200).
  - **3 dòng Sonnet có lỗi đổi nghĩa** (đều không đổi đáp án): c116 ("natural-born citizen"), c120 (mất chủ thể "he"; Google cũng mất), c183.
- **Lỗi chung của cả hai bản:** đổi tiền tố "Turn 0" thành "Lượt 0" ở một số dòng hội thoại. Người chấm trừ điểm tự nhiên, không tính đổi nghĩa. Khi dịch đầy đủ nên thêm "Turn N:" vào bảng thuật ngữ như tiền tố giữ nguyên.

**Hạn chế của phép chấm:**
- Người chấm là Opus, cùng họ Claude với người dịch Sonnet, nên có thể thiên vị văn phong cùng họ.
- Ba điều giảm bớt lo ngại này:
  - chấm mù và vị trí X/Y không lệch;
  - kết luận dựa chủ yếu vào lỗi đổi đáp án, mỗi lỗi đều chỉ ra chỗ cụ thể (xem mục 3);
  - lỗi "None of the other options" của Google đã thấy ở phần kiểm khách quan, không cần người chấm.
- 200 dòng là mẫu nhỏ: khoảng tin cậy của 18/200 vào khoảng 5–14%.

## 3. 10 ví dụ khác biệt rõ nhất

Xếp theo độ chênh: đổi đáp án, rồi đổi nghĩa, rồi độ tự nhiên. Nhận xét là lời người chấm (nói theo X/Y), kèm giải mã.

1. **c119, ý định (multidogo):** Google biến tên người dùng "bala" thành "nhé", nên câu không còn là cung cấp thông tin. **Đổi đáp án.** Sonnet giữ tên. (X = Sonnet)
2. **c021, yes/no (boolq_passage):** Google dịch "Ivy League school" thành "trường liên minh thường niên"; người đọc không còn nối câu hỏi với văn bản. **Đổi đáp án.** (X = Google)
3. **c028, yes/no (boolq):** Google dịch tên ban nhạc "for King and Country" thành "dành cho nhà vua và đất nước", câu hỏi vô nghĩa. **Đổi đáp án.** (X = Sonnet)
4. **c038, đọc hiểu (dream):** lựa chọn đúng "It's burnt" bị Google dịch "Nó bị cháy rồi", lệch với "bị bỏng" trong hội thoại, nên lựa chọn "bị thương" gần như ngang đúng. **Đổi đáp án.** (X = Sonnet)
5. **c052, yes/no (sharc):** Google dịch "Is it 25 years after you got the loan?" thành "Có phải 25 năm sau bạn mới nhận được khoản vay không?", nên không còn suy ra được "Có". **Đổi đáp án.** (X = Sonnet)
6. **c081, đọc hiểu (dream):** Google dịch "pie" thành "bánh" chung chung, nên "bánh ngọt" (cake) cũng thành ngang đúng; thêm nhiều lỗi khác trong hội thoại. **Đổi đáp án.** (X = Google)
7. **c083, yes/no (boolq_passage):** Google dịch tên sách "Before I Fall" thành "trước khi tôi ngã", câu hỏi vô nghĩa; "Notes" thành "Trái phiếu". **Đổi đáp án.** (X = Google)
8. **c097, đọc hiểu (cosmos_qa):** Google dịch "cat genie" thành "thần đèn mèo", "litter" thành "rác"; lựa chọn đúng "just for them" (cho lũ mèo) thành "dành riêng cho họ" (cho người). **Đổi đáp án.** (X = Sonnet)
9. **c172, yes/no (boolq_passage):** Google bỏ mất "nói dối" trong câu hỏi về "lie bumps", nên dễ trả lời Đúng, ngược đáp án Sai. **Đổi đáp án.** (X = Sonnet)
10. **c004, kiểm chứng (nli_fever):** Google dịch tên phim "Foxcatcher" thành "Đớp cáo" ở text_A nhưng giữ nguyên ở text_B, nên "trung lập" trở nên hợp lý. **Đổi đáp án.** (X = Sonnet)

Danh sách đủ 18 dòng Google đổi đáp án và nhận xét từng dòng: `so_sanh/tong_hop.json`.

## 4. Mức dùng

| Mốc | Thời điểm | % tuần | % 5 giờ |
|---|---|---|---|
| Sau khi xong 200 dòng Sonnet | 16:02:09 | 31 | 3 |
| Sau khi chấm so sánh | 16:11:46 | 32 | 4 |

- Lượt chấm dùng 5 subagent Opus, tổng 601.442 token subagent (`token_subagent.jsonl`, vai `opus_cham`), thời gian khoảng 10 phút.
- Cả việc (dịch thử + so sánh) dùng từ 31% → 32% tuần và 0% → 4% cửa sổ 5 giờ.

## 5. File

Tất cả nằm trong `results/train_v1/translated_sonnet/pilot/`.

| File | Nội dung |
|---|---|
| `so_sanh.html` | **cho người dùng đọc**: 30 dòng (6 mỗi dạng bài, seed 20261017), 3 cột EN \| Sonnet \| Google, đáp án tô xanh có dấu ✓, nhận xét người chấm và nguồn |
| `so_sanh/kiem_khach_quan.json` | kiểm khách quan |
| `so_sanh/kiem_sonnet.json`, `so_sanh/kiem_google.json` | kết quả `kiem_ban_dich.py` |
| `so_sanh/cham/`, `so_sanh/cham_map.json` | chấm mù và bảng giải mã |
| `so_sanh/tong_hop.json`, `so_sanh/bang_so.md` | tổng hợp |
| `so_sanh/google_200.jsonl` | 200 dòng trích từ bản ngoài, chỉ để so sánh |

## 6. Bổ sung (19/10): nhận xét của một agent khác và kiểm khách quan v2

**Nhận xét của agent khác trên `so_sanh.html`** (30 dòng, người dùng chuyển lại):
- Khớp với kết quả chấm mù. Agent đó chấm Google hơn ở 2/30 câu (nhãn `atm_support`; tiêu đề viết hoa từng chữ). Hai câu này cũng nằm trong 5 dòng Google thắng ở lượt chấm mù (c200, c042).
- Đã kiểm trực tiếp trong file và thấy đúng:
  - Google đổi "Kirin 820" thành "Snapdragon 820";
  - Google đổi "Li Ming" thành "Lý Minh" ở câu hỏi nhưng giữ "Li Ming" trong bài;
  - Google dịch "sister" lúc "chị gái" lúc "em gái" trong cùng một bài;
  - Sonnet phân biệt "nhạc công" với "nhạc sĩ sáng tác".

**Ba điểm tôi không đồng ý hoàn toàn với agent kia:**
- **"sister":** "em gái" của Sonnet cũng không có căn cứ hơn "chị gái" của Google. Tiếng Việt buộc phải chọn, nên chỉ chấm được tính nhất quán trong cùng bài: Google sai, Sonnet đúng.
- **"Turn 1" → "Lượt 1":** agent kia coi là lỗi nhỏ. Hướng dẫn dịch của dự án yêu cầu giữ nguyên tiền tố, nên đây là vi phạm quy tắc, dù không đổi nghĩa.
- **"Do I my notes…":** đã kiểm, bản gốc đúng là "Do I\nmy notes…", thiếu động từ. Đây là lỗi của dữ liệu nguồn, không tính cho bản dịch nào.

**Kiểm khách quan v2.** `kiem_ban_dich.py` nay báo lỗi khi có ký tự ẩn, và cảnh báo khi tiền tố cần giữ nguyên bị đổi. Bản trước lưu ở `to_translate/kiem_ban_dich_v1.py`.

| Bản | Kết luận | Lỗi | Cảnh báo |
|---|---|---|---|
| Sonnet | ĐẠT | 0 | 2 (như trước) |
| Google | **KHÔNG ĐẠT** | **8 dòng có ký tự ẩn U+200B** | 20, trong đó **16 dòng đổi tiền tố**, ví dụ "Q:" → "Hỏi:" |

- 200 dòng gốc của mẫu không có ký tự ẩn nào, vậy ký tự ẩn do quy trình dịch của bản Google thêm vào.
- Kết quả đầy đủ: `so_sanh/kiem_google_v2.json`.
