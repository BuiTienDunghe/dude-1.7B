# Soát lại 150 chữ đề xuất P1/P5 (probe_v1) sau khi người dùng duyệt

Ngày 26/10/2026. Người soát: Fable (chỉ đọc repo; không sửa config, data, không chạy model). Nguồn đọc: `config/vi_proposals.jsonl` (150 dòng `pv1-*`), `results/probe_v1/fable_soat/quyet_dinh_nguoi_dung.json`, `results/probe_v1/duyet_150_chu_phep_thu.html` (trang duyệt), `results/BRIEF_phan_tich_xu_huong.md`, `results/probe_v1/REPORT_chuan_bi.md`, `scripts/probe_v1/build_p1_p5.py`, dev 1.3 (chỉ đếm chuỗi nhãn và câu hỏi), và `results/train_v1/components/C.jsonl` (chỉ đếm xem chuỗi nhãn của dev có nằm trong dữ liệu sắp train không). Không đọc test.

## 1. Kết luận

1. **128 mục "chưa chọn" nhiều khả năng là người dùng dừng duyệt giữa chừng, không phải 128 lần cân nhắc rồi bỏ.** Trang duyệt xếp thứ tự: P5 (#1–6), nhãn kiểm chứng và Có/Không (#7–16), chủ đề SIB-200 (#17–30), ý định MASSIVE (#31–150). Mọi mục có quyết định đều nằm ở #1–22; mọi mục chưa chọn là #23–150, liền một mạch từ dòng "thể thao" trở đi. Theo quy ước người dùng đã nêu (chưa chọn = bỏ) thì vẫn tính là bỏ, nhưng khuyến nghị dưới đây phụ thuộc vào việc hỏi lại người dùng **một câu**: "Anh/chị dừng ở dòng 22 hay cố ý bỏ qua 128 dòng sau?" Câu trả lời quyết định việc tiếp theo là duyệt lại 128 mục hay chỉ duyệt một trang mới khoảng 5–12 mục.

2. **Về chữ: 20 mục bị bỏ rõ ràng có 11 mục có vấn đề thật** (9 mục lỗi về chữ: sai nghĩa, dịch sát chữ Anh, liệt kê thay cho tên chủ đề; 2 mục lỗi về khung câu hỏi); 9 mục còn lại chữ không có lỗi rõ, nhiều khả năng bị bỏ theo cả nhóm. **128 mục chưa chọn có 6 mục nên sửa** (đều là MASSIVE), 122 mục đạt yêu cầu cùng nghĩa, khác chữ, tự nhiên. **2 mục được duyệt không có lỗi chặn**, chỉ có một lưu ý cách đọc.

3. **Hệ quả lớn nhất không nằm ở chữ mà ở cổng kiểm định đã chốt trước.** Brief định nghĩa K1 "tiến bộ thật" cần "P1 và P5 tụt ≤ 2 điểm", và điều kiện lên train chính thức là "K1 và không K2". Với 0 mục P1 được duyệt, theo đúng định nghĩa của kế hoạch thì **không thể tuyên bố K1 và không thể loại K2**: cổng trở thành không quyết định được, không chỉ "yếu đi". Thêm nữa, `build_p1_p5.py` chỉ dựng một nguồn khi **đủ mọi nhãn** của nguồn đó, nên thiếu một nhãn là mất cả nguồn.

4. **Đếm chuỗi nhãn trong dữ liệu sắp train (C.jsonl) cho thấy K2 "học thuộc chữ nhãn" chỉ có thể xảy ra ở nhãn kiểm chứng choice**, đúng nhóm người dùng đã bỏ. Chữ nhãn SIB-200 và MASSIVE của dev gần như không có trong C, nên 120 mục MASSIVE (80% khối lượng duyệt) lại là phần ít giá trị nhất cho K2. Vì vậy bộ tối thiểu chỉ cần **3 mục duyệt mới** (3 nhãn kiểm chứng choice, một cách diễn đạt), **khuyến nghị 5** (thêm 2 câu hỏi P5 cho SIB và MASSIVE), cộng 2 mục đã duyệt; thêm 7 mục SIB nếu muốn.

## 2. Các mục có vấn đề và chữ thay thế

Quy ước cột "Vấn đề": **Nghĩa** = lệch nghĩa so với chữ đang dùng; **Tự nhiên** = tiếng Việt gượng hoặc dịch sát chữ Anh; **Nhầm nhãn** = có thể bị hiểu sang nhãn khác trong cùng bộ; **Khung** = đổi khung câu hỏi hoặc lệch với câu hỏi giữ nguyên. "Lý do người dùng bỏ (đoán)" là phỏng đoán có căn cứ, không phải lời người dùng.

### 2a. 20 mục người dùng bỏ rõ ràng

Lưu ý chung cho 6 mục nhãn kiểm chứng choice: P1 **giữ nguyên câu hỏi**, mà câu hỏi dev nêu đích danh ba nhãn bằng chữ cũ ("được văn bản xác nhận, bị văn bản bác bỏ, hay văn bản không đủ thông tin để kết luận"), rồi lựa chọn lại dùng chữ mới. Model phải tự ánh xạ "xác nhận" trong câu hỏi sang "Đúng theo văn bản" trong lựa chọn. Đây là một chi phí không liên quan đến học thuộc, M1 và M2 cùng chịu, nhưng làm P1 nhiễu hơn. Đây có thể là lý do người dùng bỏ cả nhóm. Cách sửa đề xuất ở mục 3.3.

| # trang | id | Chữ đang dùng | Chữ đề xuất | Vấn đề | Lý do người dùng bỏ (đoán) | Chữ thay thế đề xuất |
|---|---|---|---|---|---|---|
| 7 | `pv1-P1-kiem_chung-supported-v1` | Được văn bản xác nhận | Đúng theo văn bản | Khung (lệch với câu hỏi giữ nguyên); chữ thì đúng nghĩa | Lựa chọn không còn "trả lời" câu hỏi đang nêu ba nhãn bằng chữ cũ | Dùng bộ nhãn ở mục 3.3 ("Nhận định đúng") kèm câu hỏi P5 đã duyệt; nếu giữ câu hỏi cũ: "Nhận định này đúng theo văn bản" |
| 8 | `pv1-P1-kiem_chung-supported-v2` | Được văn bản xác nhận | Văn bản ủng hộ nhận định này | **Tự nhiên**: "ủng hộ" là dịch sát "support"; tiếng Việt dùng "ủng hộ" cho người, phong trào, ý kiến, không dùng cho văn bản với nhận định | Dịch sát chữ Anh | "Văn bản khẳng định nhận định này" |
| 9 | `pv1-P1-kiem_chung-refuted-v1` | Bị văn bản bác bỏ | Sai theo văn bản | Khung (như #7) | Như #7 | "Nhận định này sai theo văn bản" |
| 10 | `pv1-P1-kiem_chung-refuted-v2` | Bị văn bản bác bỏ | Văn bản phủ nhận nhận định này | Không có lỗi rõ | Bỏ theo nhóm | Giữ được |
| 11 | `pv1-P1-kiem_chung-nei-v1` | Văn bản không đủ thông tin để kết luận | Không xác định được từ văn bản | Khung; hơi cộc, thiếu chủ ngữ ("không xác định được" cái gì) | Như #7, cộng cảm giác câu chưa trọn | "Văn bản không cho biết đủ để kết luận" hoặc "Chưa đủ căn cứ để đánh giá" (chữ của câu hỏi P5 đã duyệt) |
| 12 | `pv1-P1-kiem_chung-nei-v2` | Văn bản không đủ thông tin để kết luận | Văn bản chưa đủ căn cứ để đánh giá | Hơi lệch: "văn bản chưa đủ căn cứ" nghe như văn bản thiếu cơ sở, chứ không phải người đọc thiếu căn cứ | Như #7 | "Không đủ căn cứ trong văn bản để đánh giá" |
| 13 | `pv1-P1-kiem_chung_noul-co-v1` | Có | Có xác nhận | Không sai nghĩa; nhưng với câu hỏi dạng "có … không?", "Có/Không" là câu trả lời duy nhất tự nhiên; mọi chữ khác đều gượng | Thấy đổi Có/Không là vô nghĩa | **Bỏ cả 4 mục P1 noul** (lý do ở mục 3.2) |
| 14 | `pv1-P1-kiem_chung_noul-co-v2` | Có | Đúng vậy | **Nghĩa**: "Đúng vậy" khẳng định một phát biểu, không phải trả lời "có xác nhận không" | Như trên | Bỏ |
| 15 | `pv1-P1-kiem_chung_noul-khong-v1` | Không | Không xác nhận | Như #13 | Như trên | Bỏ |
| 16 | `pv1-P1-kiem_chung_noul-khong-v2` | Không | Không phải vậy | **Nghĩa**: "Không phải vậy" = phát biểu sai (bác bỏ), còn "Không" ở đây gồm cả bác bỏ lẫn không đủ thông tin. Chính ghi chú của worker đã cảnh báo điều này khi từ chối "Sai", rồi v2 lại mắc đúng lỗi đó | Sai nghĩa | Bỏ |
| 17 | `pv1-P1-sib200-science_technology-v1` | khoa học/công nghệ | khoa học và kỹ thuật | Không có lỗi rõ ("khoa học kỹ thuật" là cụm quen) | Bỏ theo nhóm, hoặc thấy "kỹ thuật" hẹp hơn "công nghệ" | "khoa học kỹ thuật" |
| 18 | `pv1-P1-sib200-science_technology-v2` | khoa học/công nghệ | nghiên cứu, phát minh và thiết bị | **Tự nhiên**: là bản liệt kê, không phải tên chủ đề; "thiết bị" không gợi công nghệ | Không phải tên chủ đề | "công nghệ và nghiên cứu khoa học" (vẫn chung gốc từ; với nhãn một chữ khó tránh) |
| 19 | `pv1-P1-sib200-travel-v1` | du lịch | lữ hành | **Tự nhiên**: từ ngành nghề ("công ty lữ hành"), không dùng làm tên chủ đề bài viết | Từ nghe lạ | "tham quan và du ngoạn" |
| 20 | `pv1-P1-sib200-travel-v2` | du lịch | chuyến đi và tham quan | Không có lỗi rõ | Bỏ theo nhóm | Giữ được |
| 21 | `pv1-P1-sib200-politics-v1` | chính trị | chính quyền và nhà nước | **Nghĩa**: "chính quyền và nhà nước" là bộ máy, hẹp hơn "chính trị" (gồm bầu cử, đảng phái, quan hệ quốc tế) | Lệch nghĩa | "chính trường và bầu cử" |
| 22 | `pv1-P1-sib200-politics-v2` | chính trị | chính phủ, bầu cử, ngoại giao | Liệt kê, không phải tên chủ đề; nghĩa thì gần đủ | Không phải tên chủ đề | "chính trường, bầu cử và ngoại giao" |
| 1 | `pv1-P5-sib200` | Văn bản này thuộc chủ đề nào? | Đoạn văn trên nói về lĩnh vực gì? | **Nghĩa**: "lĩnh vực" (ngành, mảng) hẹp hơn "chủ đề"; "trên" phụ thuộc cách bày prompt | Lệch nghĩa | "Đoạn văn này nói về chủ đề gì?" |
| 2 | `pv1-P5-massive` | Người dùng muốn trợ lý làm gì? | Người dùng đang nhờ trợ lý việc gì? | Hơi lệch: "nhờ" là xin giúp, gượng với câu lệnh kiểu "tắt đèn" và với chào hỏi, kể chuyện cười | "nhờ" không khớp mọi ý định | "Câu nói này yêu cầu trợ lý làm việc gì?" |
| 3 | `pv1-P5-xcopa_cause` | Nguyên nhân nào hợp lý hơn? | Điều gì nhiều khả năng đã dẫn đến chuyện này? | **Khung**: bỏ mất thế so sánh "hơn" giữa hai lựa chọn; "chuyện này" văn nói | Đổi khung câu hỏi | "Điều nào có khả năng là nguyên nhân hơn?" |
| 4 | `pv1-P5-xcopa_effect` | Kết quả nào hợp lý hơn? | Chuyện gì nhiều khả năng xảy ra tiếp theo? | **Khung**: như #3 | Như #3 | "Điều nào có khả năng xảy ra sau đó hơn?" |

Tóm lại nhóm này: **11 mục có vấn đề thật** gồm 9 mục lỗi về chữ (#8, #11 nhẹ, #12 nhẹ, #14, #16, #18, #19, #21, #1; #22 và #2 cùng dạng nhưng nhẹ, xếp vào nhóm "không rõ") và 2 mục lỗi về khung (#3, #4); **9 mục không có lỗi rõ** (#7, #9, #10, #13, #15, #17, #20, #22, #2). Một tín hiệu đáng để ý nhưng chưa đủ kết luận: **người dùng duyệt đúng hai câu P5 của nguồn tiếng Việt gốc (kiểm chứng) và bỏ cả bốn câu P5 của nguồn dịch** (SIB, MASSIVE, XCOPA). Có thể người dùng chỉ coi phép thử trên nguồn gốc là đáng làm.

### 2b. 128 mục chưa chọn: 6 mục nên sửa, 122 mục đạt

| # trang | id | Chữ đang dùng | Chữ đề xuất | Vấn đề | Chữ thay thế đề xuất |
|---|---|---|---|---|---|
| 60 | `pv1-P1-massive-email_addcontact-v2` | thêm liên hệ email | bổ sung người liên lạc qua thư điện tử | **Nhầm nhãn**: đọc thành "liên lạc với người qua email", dễ lẫn với `email_sendemail` | "lưu thêm một người vào danh bạ email" |
| 135 | `pv1-P1-massive-takeaway_order-v1` | đặt đồ ăn mang về | gọi món giao tận nơi | **Nghĩa**: giao tận nơi là giao hàng, không phải mang về | "gọi đồ ăn để mang đi" |
| 148 | `pv1-P1-massive-transport_ticket-v2` | đặt vé tàu xe | giữ chỗ cho chuyến đi | **Nghĩa**: quá rộng, có thể là khách sạn, taxi; dễ lẫn `transport_taxi` | "mua vé cho chuyến tàu hoặc chuyến xe" |
| 41 | `pv1-P1-massive-audio_volume_other-v1` | các điều chỉnh âm lượng khác | điều chỉnh âm thanh theo cách khác | **Nghĩa**: mất chữ "âm lượng", thành chỉnh âm thanh nói chung (có thể là bass, treble) | "chỉnh độ to nhỏ theo cách khác" |
| 92 | `pv1-P1-massive-lists_createoradd-v2` | tạo danh sách hoặc thêm vào danh sách | bổ sung việc cần làm hoặc lập danh mục mới | **Nghĩa**: "việc cần làm" thu hẹp về to-do; "danh mục" nghiêng về mục lục, phân loại | "ghi thêm một mục hoặc lập một danh sách mới" |
| 52 | `pv1-P1-massive-cooking_query-v2` | trả lời câu hỏi về nấu ăn | giải thích một điều về bếp núc | **Tự nhiên**: "bếp núc" là chuyện nội trợ nói chung, lệch khỏi nấu ăn | "giải đáp thắc mắc khi nấu ăn" |

Những mục chưa chọn còn lại: 8 mục SIB (sports, health, entertainment, geography) và 114 mục MASSIVE đều cùng nghĩa, khác chữ, tiếng Việt tự nhiên và không gây nhầm sang nhãn khác. Vài chỗ văn nói ("cho loa kêu nhỏ lại", "vặn to tiếng", "bật đài") là chấp nhận được với MASSIVE vì câu lệnh gốc cũng là văn nói. Riêng SIB: "y tế" (health v1) là nhãn tự nhiên nhất cả bộ; "nghệ thuật và biểu diễn" (entertainment v1) hơi lệch về nghệ thuật nhưng không lẫn với sáu nhãn còn lại; "núi non, sông biển, khí hậu" (geography v2) có "khí hậu" hơi kéo về khoa học, mức nhẹ.

### 2c. 2 mục đã duyệt

| id | Chữ đã duyệt | Nhận xét |
|---|---|---|
| `pv1-P5-kiem_chung_choice` | Đối chiếu với nội dung đã cho, nhận định dưới đây đúng, sai, hay chưa đủ căn cứ để đánh giá? Nhận định: {claim} | Không có lỗi chặn. Hai lưu ý: (a) chữ trong câu hỏi (đúng / sai / chưa đủ căn cứ) không còn khớp chữ của lựa chọn (xác nhận / bác bỏ / không đủ thông tin), model phải ánh xạ; (b) "đúng/sai" là khung đúng-sai ngoài đời, còn lựa chọn là khung "văn bản nói gì". Nhận định đúng ngoài đời nhưng văn bản không nói tới có thể bị kéo về "đúng". P4 sẽ bắt hiện tượng này; chỉ cần ghi nhận khi đọc kết quả. |
| `pv1-P5-kiem_chung_noul` | Nội dung văn bản có cho thấy điều sau đây là đúng không? Phát biểu: {claim} | Tốt. "cho thấy … là đúng" tương đương "xác nhận"; "Không" vẫn gồm cả bác bỏ và không đủ thông tin. |

## 3. Hệ quả cho phép đo

### 3.1. K2 có thể xảy ra ở đâu: đếm chuỗi nhãn của dev trong C.jsonl

"Học thuộc chữ nhãn" chỉ xảy ra được với chữ model đã gặp khi train. Đếm trực tiếp trong `results/train_v1/components/C.jsonl` (31.034 dòng):

| Nhóm nhãn dev | Có trong C? | Chi tiết |
|---|---|---|
| Kiểm chứng choice: "Được văn bản xác nhận", "Bị văn bản bác bỏ", "Văn bản không đủ thông tin" | **Có, ~600 lần mỗi nhãn** | 593 dòng vitaminc (phần B, dịch máy) dùng đúng ba chuỗi này làm lựa chọn (649 dòng có chứa chuỗi, số còn lại nằm trong câu hỏi); C dùng "Văn bản không đủ thông tin" (không có "để kết luận"), chuỗi dev là bản dài hơn. Câu hỏi trong C khác dev ("Hãy chọn nhãn mô tả văn bản đúng nhất" v.v.) |
| Có/Không (noul) | Có, 3.006 dòng | Nhưng là cặp từ phổ quát, câu hỏi khác; không thể nói là "học thuộc" |
| 7 chủ đề SIB-200 | **Chỉ 2 trùng đúng chữ**: "sức khỏe", "thể thao" | 1.000 dòng VNTC dùng **cùng câu hỏi** "Văn bản này thuộc chủ đề nào?" với bộ 10 nhãn khác (chính trị - xã hội, đời sống, khoa học, kinh doanh, pháp luật, sức khỏe, thế giới, thể thao, văn hóa, vi tính) |
| 60 ý định MASSIVE | **Không** (0 lần cho mọi mô tả; chỉ "pha cà phê" 7, "đặt báo thức" 2 trong văn bản thường) | 1.000 dòng tourism-intent dùng **cùng câu hỏi** "Người dùng muốn trợ lý làm gì?" với 8 nhãn khác; một nhãn gần: "gợi ý địa điểm du lịch nên đến" so với dev "gợi ý địa điểm" |
| Câu hỏi XCOPA | Không | — |

Hệ quả:
- **P1 kiểm chứng choice là phép thử K2 duy nhất có cơ sở từ dữ liệu train.** Đây chính là 6 mục người dùng bỏ.
- **P1 SIB và P1 MASSIVE đo "khớp từ giữa văn bản và nhãn" hơn là "học thuộc từ train".** Vẫn có ích để biết model hiểu hay khớp từ, nhưng không phân biệt M2 với M1 về K2. 120 mục MASSIVE là 80% khối lượng duyệt mà giá trị K2 thấp nhất.
- **P5 SIB và P5 MASSIVE lại có giá trị hơn tưởng**, vì hai câu hỏi này nằm 1.000 lần mỗi câu trong C với bộ nhãn khác. Model có thể học "câu hỏi này đi với bộ nhãn VNTC/tourism".
- P5 XCOPA giá trị thấp: câu hỏi không có trong C, XCOPA không có nhãn cố định.

### 3.2. Nếu giữ nguyên quyết định (0 P1, 2 P5 kiểm chứng)

- **Mất khả năng phát hiện K2 theo đúng định nghĩa đã chốt** ("P1 tụt mạnh hơn M1"). Còn lại chỉ dấu hiệu gián tiếp: điểm tăng dồn vào nguồn có nhãn cố định; P3 bỏ văn bản ở kiểm chứng (nếu M2 thiên về một nhãn khi không có văn bản thì đó là thiên vị nhãn học từ vitaminc); P4; và P5 kiểm chứng (đã duyệt). Gián tiếp đủ để **nghi** K2, không đủ để **loại** K2.
- **Không tuyên bố được K1**, vì K1 yêu cầu "P1 và P5 tụt ≤ 2 điểm". Điều kiện lên train chính thức (K1 và không K2) do đó không quyết định được. Cần hoặc duyệt bộ tối thiểu, hoặc **sửa brief** để bỏ P1 khỏi định nghĩa K1/K2 và thay bằng các dấu hiệu gián tiếp ở trên (phải ghi vào PLAN_decisions trước khi train, đúng tinh thần "chốt trước khi thấy số").
- P1 noul: nên bỏ hẳn, độc lập với quyết định người dùng. Lý do: decider và Quyet trả lời noul bằng đường có/không riêng nên không đo được ở hai model tham chiếu (REPORT_chuan_bi đã ghi); Có/Không là câu trả lời duy nhất tự nhiên cho câu "có … không?"; và hai chữ v2 sai nghĩa.
- P4 noul không bị ảnh hưởng. P2, P3, P4, P6, P7 giữ nguyên.

### 3.3. Bộ tối thiểu để vẫn đo được

| Ưu tiên | Mục | Số mục cần duyệt mới | Lý do |
|---|---|---|---|
| **Bắt buộc** | P1 kiểm chứng choice, **một** cách diễn đạt (3 nhãn) | 3 | Nhóm duy nhất có chữ nhãn nằm ~600 lần trong C |
| **Bắt buộc** | P5 kiểm chứng choice và noul | 0 (đã duyệt) | — |
| Nên | P5 SIB, P5 MASSIVE | 2 | Hai câu hỏi nằm 1.000 lần mỗi câu trong C với bộ nhãn khác |
| Tùy chọn | P1 SIB, một cách diễn đạt (7 nhãn) | 7 | 2 nhãn trùng VNTC; đo khớp từ |
| Bỏ | P1 noul (4), P1 MASSIVE (120), P5 XCOPA (2) | 0 | Lý do ở 3.1, 3.2 |

Tổng: **tối thiểu 3, khuyến nghị 5 mục duyệt mới** (12 nếu lấy cả SIB) thay cho 150.

**Cách dựng P1 kiểm chứng sạch hơn (đề xuất, chưa làm):** thay vì giữ câu hỏi cũ rồi đổi lựa chọn (gây lệch khung như mục 2a), dựng biến thể dùng **câu hỏi P5 đã duyệt** kèm lựa chọn lặp lại đúng chữ của câu hỏi đó:

- câu hỏi: "Đối chiếu với nội dung đã cho, nhận định dưới đây đúng, sai, hay chưa đủ căn cứ để đánh giá? Nhận định: {claim}" (đã duyệt);
- lựa chọn: **"Nhận định đúng" / "Nhận định sai" / "Chưa đủ căn cứ để đánh giá"**.

Ba chữ này gần như không cần duyệt thêm vì người dùng đã duyệt chính các từ đó trong câu hỏi. Khi đó có ba bậc trên cùng một câu: dev → P5 (chỉ đổi câu hỏi) → P1′ (đổi cả câu hỏi và nhãn); chênh **P1′ − P5** tách riêng phần "đổi chữ nhãn", không còn nhiễu do câu hỏi và lựa chọn lệch khung. Cần sửa nhỏ `build_p1_p5.py` (hiện P1 luôn giữ câu hỏi dev) và thêm một dòng vào manifest. Nếu không muốn sửa script, dùng chữ ở cột "Chữ thay thế" của mục 2a với câu hỏi cũ.

Chữ thay thế cho P5 SIB và MASSIVE (mục 2a, #1, #2): "Đoạn văn này nói về chủ đề gì?" và "Câu nói này yêu cầu trợ lý làm việc gì?".

### 3.4. Cách đạt cùng mục đích mà không cần duyệt chữ mới (bổ sung, không thay thế)

| Cách | Cần duyệt | Đo được gì | Nhiễu đi kèm |
|---|---|---|---|
| **Lựa chọn không dấu, văn bản và câu hỏi giữ dấu** (strip_diacritics, phương pháp đã dùng trong dự án) | 0 | C có **0** chuỗi nhãn không dấu, nên model học thuộc chữ có dấu sẽ gặp chuỗi lạ; model hiểu nghĩa đọc được | Đọc không dấu vốn yếu hơn (M1: 62,8% so với 80,8%); phải so với dòng không dấu toàn phần đã có trong dev để tách phần nhiễu này |
| **Nhãn tiếng Anh** (lấy từ dòng EN cùng cặp trong dev) cho SIB và MASSIVE | 0 | Văn bản tiếng Việt, nhãn tiếng Anh: model có hiểu nghĩa nhãn hay chỉ khớp chữ Việt | Lẫn với năng lực tiếng Anh; dùng điểm EN của M1 làm mốc |
| **Ma trận nhầm nhãn trên dev** (chỉ phân tích, không dựng file) | 0 | M2 có dồn dự đoán vào "sức khỏe"/"thể thao" ở SIB (hai nhãn trùng VNTC trong C) và "gợi ý địa điểm" ở MASSIVE (gần nhãn tourism) so với M1 không | Chỉ bắt được K2 kiểu "kéo về nhãn đã gặp", không bắt được học thuộc toàn bộ |
| **P3 ở kiểm chứng đọc theo phân bố nhãn** | 0 | Khi bỏ văn bản, M2 có nghiêng về một nhãn theo kiểu vitaminc không | Đã nằm trong P3, chỉ cần thêm cách đọc |
| Chữ đã duyệt sẵn trong `translations.json` | 0 | Hai mô tả MASSIVE người dùng đã duyệt 13/10 khác chữ dev: `transport_query` "tra cứu lộ trình hoặc giờ tàu xe", `transport_traffic` "cho biết tình hình kẹt xe trên đường" | Chỉ 2 trong 60 nhãn, không đủ dựng P1 MASSIVE (script cần đủ mọi nhãn) |

Hai cách đầu có thể thêm vào `build_p1_p5.py` như biến thể P1 không cần giá trị duyệt; `analyze_probe.py` đã so theo `orig_id` nên không cần đổi.

## 4. Khuyến nghị cho người dùng

1. **Trả lời một câu:** dừng ở dòng 22, hay cố ý bỏ 128 dòng sau? Nếu dừng, không cần duyệt lại 128 dòng; chỉ cần một trang mới 5–12 mục.
2. **Không duyệt lại nguyên văn 20 mục đã bỏ.** 11 mục có vấn đề thật; 9 mục còn lại chữ không sai nhưng nên thay bằng chữ ở mục 2a và 3.3 để tránh lệch khung với câu hỏi.
3. **Duyệt tối thiểu 3, khuyến nghị 5 mục mới:** 3 nhãn kiểm chứng choice ("Nhận định đúng" / "Nhận định sai" / "Chưa đủ căn cứ để đánh giá", đi kèm câu hỏi P5 đã duyệt) và 2 câu hỏi P5 SIB, MASSIVE ("Đoạn văn này nói về chủ đề gì?", "Câu nói này yêu cầu trợ lý làm việc gì?"). Tùy chọn thêm 7 nhãn SIB một cách diễn đạt (dùng v1 hiện có cho sports, health, entertainment, geography; thay chữ cho science_technology, travel, politics như mục 2a).
4. **Bỏ hẳn P1 noul, P1 MASSIVE, P5 XCOPA** với lý do ở mục 3; ghi vào PLAN_decisions trước khi train rằng K1/K2 được đo bằng P1 kiểm chứng (+ SIB nếu có), P5 bốn nguồn, và các dấu hiệu bổ sung không cần duyệt ở mục 3.4.
5. Nếu người dùng vẫn muốn giữ nguyên quyết định hiện tại (0 P1), thì **phải sửa định nghĩa K1 và điều kiện lên train chính thức trong brief trước khi train**, vì theo chữ hiện tại cổng không quyết định được.

Mọi đề xuất chữ mới trong báo cáo này chưa được ghi vào `config/vi_proposals.jsonl`; nếu người dùng đồng ý thì worker ghi đề xuất theo quy trình cũ để người dùng duyệt trên trang.
