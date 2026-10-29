# Kế hoạch phân tích xu hướng model sau khi train (đặt trước khi train)

Ngày 26/10/2026, điều phối viết. **Đây là đề xuất, chưa làm gì.**

Mục đích: khi train xong không chỉ hỏi "điểm có tăng không", mà trả lời được **vì sao tăng** và **model có thật sự hiểu tiếng Việt hơn không**. Mọi phép đo, kịch bản và cách đọc đều được chốt **trước** khi thấy kết quả, để không ai đọc số theo ý mình sau đó.

## Cập nhật 26/10 (chốt trước khi chấm hay train; thay cho các chỗ tương ứng bên dưới)

Người dùng duyệt 150 chữ P1/P5: 2 duyệt, 148 bỏ. Fable 5.1 soát lại (`results/probe_v1/fable_soat/REPORT_fable.md`) và thấy nguy cơ học thuộc chữ nhãn tập trung ở **nhãn kiểm chứng**: ba chữ nhãn của dev nằm 593 lần mỗi chữ trong bản C. Chữ nhãn SIB-200 và MASSIVE gần như không có trong C. Người dùng giao điều phối và Fable chủ động. Bộ phép thử chốt lại như sau:

- **P1′ (thay P1):** chỉ cho kiểm chứng choice. Dùng câu hỏi P5 kiểm chứng đã duyệt, kèm 3 lựa chọn "Nhận định đúng" / "Nhận định sai" / "Chưa đủ căn cứ để đánh giá" (đề xuất `pv1b-P1-*`, chờ người dùng duyệt). Chênh P1′ − P5 là phần do đổi chữ nhãn.
- **P5:** chỉ 4 khung: kiểm chứng choice, kiểm chứng noul (2 câu này đã duyệt), SIB-200 "Đoạn văn này nói về chủ đề gì?", MASSIVE "Câu nói này yêu cầu trợ lý làm việc gì?" (`pv1b-P5-*`, chờ duyệt).
- **Bỏ:** P1 cho câu Có/Không, P1 MASSIVE, P1 SIB-200, P5 XCOPA.
- **Thêm, không cần duyệt chữ:**
  - (a) lựa chọn bỏ dấu nhưng văn bản và câu hỏi giữ dấu;
  - (b) nhãn tiếng Anh lấy từ dòng EN cùng cặp, cho SIB-200 và MASSIVE;
  - (c) ma trận nhầm nhãn: M2 có dồn dự đoán về "sức khỏe"/"thể thao" ở SIB-200, hay "gợi ý địa điểm" ở MASSIVE không;
  - (d) đọc P3 kiểm chứng theo phân bố nhãn: khi bỏ văn bản, M2 có nghiêng về một nhãn không.
- **Định nghĩa lại K1, K2:**
  - **K1** cần: P1′ − P5 tụt ≤ 2 điểm ở kiểm chứng; P5 cả 4 khung tụt ≤ 2 điểm; (a) và (b) không tụt nhiều hơn M1; (c) không dồn nhãn.
  - **K2** nếu bất kỳ điều sau đúng: P1′ − P5 tụt quá 2 điểm và tụt mạnh hơn M1; (a) tụt mạnh hơn M1; (c) dồn nhãn rõ rệt.

Toàn bộ chỉ chạy trên **dev 1.3**. Test 1.3 không được đụng tới cho đến bước chấm cuối cùng.

## 1. Model được so sánh

| Mã | Model | Vai trò |
|---|---|---|
| M0 | Qwen3-1.7B gốc | Sàn: model chưa học gì |
| **M1** | **Model dự án hiện tại** (train_v0) | **Mốc chính để so** |
| M2 | Bản C mới (và A, B nếu chạy) | Đối tượng phân tích |
| R | decider-2b, Quyet-Medium | Tham chiếu bên ngoài, đã có sẵn kết quả |

M0, M1 và R được chấm mọi phép thử **trước khi train**, để có sẵn mốc.

## 2. Lát cắt điểm (tính cho mọi model)

Mọi lát đều kèm khoảng tin cậy bootstrap. Riêng chênh M2 − M1 tính thêm kiểm định McNemar trên từng cặp câu.

| Lát | Để thấy gì |
|---|---|
| Theo nguồn (6 nguồn) | Tăng hay giảm ở đâu |
| **Nguồn tiếng Việt gốc** (ViFactCheck, ViWikiFC) **so với nguồn dịch** (Belebele, SIB-200, MASSIVE, XCOPA) | Hiểu tiếng Việt thật, hay chỉ quen văn phong dịch |
| VI có dấu / VI không dấu / EN | Có quên tiếng Anh hay không dấu không |
| Theo kiểu câu: chọn một (choice) / Có-Không (noul) | Tăng do một kiểu câu hay đều |
| Theo số lựa chọn (2 / 3–4 / ≥ 5) | Có mạnh lên ở câu nhiều lựa chọn không |
| Theo độ dài văn bản (ngắn / vừa / dài) | Có đọc được văn bản dài không |
| Câu đã sửa ở 1.2 so với câu giữ nguyên | Có nhạy với cách viết của người soát không |
| Có / không có cờ "mơ hồ" | Model mới có "đoán đúng ý người gán" ở câu mơ hồ không |

## 3. Bộ phép thử chẩn đoán (dựng trước, đóng băng, chỉ trên dev)

| Mã | Phép thử | Cách dựng | Model hiểu thật thì | Model dùng mẹo thì |
|---|---|---|---|---|
| **P1** | **Đổi chữ nhãn** | Thay chữ của nhãn cố định bằng từ cùng nghĩa (ví dụ "Được văn bản xác nhận" → "Đúng theo văn bản"; tên chủ đề SIB, mô tả ý định MASSIVE, "Có/Không" → "Đúng/Sai"). Mỗi nhãn 2 cách diễn đạt khác. **Bạn duyệt danh sách chữ** | Điểm gần như không đổi | Điểm tụt mạnh, vì model học thuộc chữ chứ không học nghĩa |
| **P2** | **Đảo thứ tự lựa chọn** | Đã có sẵn | Vẫn chọn cùng đáp án | Đổi đáp án theo vị trí |
| **P3** | **Bỏ văn bản** (chỉ giữ câu hỏi và lựa chọn) | Xóa state ở Belebele, ViFactCheck, ViWikiFC | Điểm rơi về gần mức đoán mò | Điểm vẫn cao, tức là trả lời theo mẹo ở câu hỏi/lựa chọn, không đọc văn bản |
| **P4** | **Thay bằng văn bản không liên quan** (kiểm chứng) | Ở ViFactCheck/ViWikiFC, ghép nhận định với một đoạn văn của câu khác; đáp án đúng thành "không đủ thông tin" | Chọn "không đủ thông tin" | Vẫn chọn "xác nhận"/"bác bỏ" theo kiến thức sẵn có hoặc theo mẹo |
| **P5** | **Đổi câu hỏi** | Mỗi nguồn thay câu hỏi mẫu bằng 1 câu khác cùng nghĩa. **Bạn duyệt** | Không đổi | Tụt, tức là phụ thuộc đúng câu hỏi lúc train |
| P6 | Tiếng Việt không dấu | Đã có sẵn | Giữ hoặc tăng | Tụt, tức là quên kỹ năng không dấu |
| P7 | Hiệu chỉnh độ tự tin | Brier, ECE, độ tự tin trung bình khi sai | Tự tin đúng mức | Tự tin cao cả khi sai |

P3 và P4 dựng bằng script, không thêm chữ tiếng Việt mới. P1 và P5 có chữ tiếng Việt mới, nên đi qua đề xuất để bạn duyệt như mọi lần. Các file này là **bộ phân tích**, không phải benchmark, và không bao giờ dùng để train.

## 4. Theo dõi trong lúc train (phải bật trước khi train)

- **Lưu adapter ở 4 mốc:** 25%, 50%, 75% và 100% số bước. Chấm từng mốc trên dev (điểm chính, không dấu, EN, P1) để vẽ đường xu hướng.
- **Ma trận chuyển đổi M1 → M2** cho từng nguồn: câu sai thành đúng, câu đúng thành sai.
- **Đọc lỗi bằng Opus, mù** (không biết model nào):
  - 50 câu M2 mới làm đúng: đúng vì lý do đúng hay vì mẹo?
  - 50 câu M2 mới làm sai (trước đó đúng): lỗi loại gì?
- **Model mới giống ai hơn:** đo mức trùng lỗi của M2 với decider-2b và Quyet. Xem model đang học kỹ năng chung hay học kiểu riêng của dữ liệu train.

## 5. Kịch bản kết quả và cách đọc (chốt trước)

| # | Dấu hiệu | Kết luận | Làm gì tiếp |
|---|---|---|---|
| **K1. Tiến bộ thật** | VI có dấu tăng ở **cả** nguồn gốc lẫn nguồn dịch; P1 và P5 tụt ≤ 2 điểm; P3 gần mức đoán mò; P4 đúng nhiều hơn M1; không dấu và EN không tụt | Model hiểu tiếng Việt tốt hơn | Sang train chính thức 2 seed |
| **K2. Học thuộc chữ nhãn** | Tăng chủ yếu ở nguồn có nhãn cố định (ViFactCheck, ViWikiFC, SIB, MASSIVE); P1 tụt mạnh hơn M1 | Điểm tăng là ảo | Đa dạng hóa cách diễn đạt nhãn trong dữ liệu train rồi train lại; không lên train chính thức |
| **K3. Học văn phong dịch** | Tăng ở 4 nguồn dịch, nguồn gốc gần như đứng yên | Quen kiểu câu dịch, chưa hiểu tiếng Việt tự nhiên hơn | So A với B; tăng tỉ lệ dữ liệu gốc, giảm phần dịch |
| **K4. Dùng mẹo, không đọc văn bản** | P3 vẫn cao hơn đoán mò rõ, hoặc tăng so với M1; P4 vẫn chọn "xác nhận/bác bỏ" | Model đoán từ câu hỏi/lựa chọn | Rà dữ liệu train tìm mẹo (lựa chọn đúng thường dài hơn…), lọc rồi train lại |
| **K5. Quên** | EN tụt quá 1 điểm hoặc không dấu tụt quá 1,4 điểm (dưới ngưỡng đã đặt) | Đánh đổi chưa chấp nhận được | Chỉnh tỉ lệ trộn, thêm dữ liệu không dấu |
| **K6. Tự tin sai** | Accuracy tăng nhưng Brier/ECE xấu đi, tự tin cao khi sai | Khó dùng thật | Hiệu chỉnh nhiệt độ (temperature) trên một nửa dev, đo trên nửa còn lại |
| **K7. Thiên vị vị trí** | Nhất quán khi đảo thứ tự giảm | Học theo vị trí đáp án | Kiểm phân bố vị trí đáp án đúng trong dữ liệu train |
| **K8. Quá khớp theo thời gian** | Đường xu hướng đạt đỉnh ở 50–75% rồi đi xuống | Train quá lâu | Giảm số epoch ở train chính thức |
| **K9. Không đổi** | Mọi lát chênh trong nhiễu | Dữ liệu thêm không giúp | Xem lại: model nền quá nhỏ, cần cách train khác, hoặc cần dữ liệu khác loại |

Các kịch bản có thể xảy ra cùng lúc, ví dụ K1 cộng K6. Báo cáo sẽ nêu đủ.

**Điều kiện để lên train chính thức:** đạt chỉ tiêu ở brief train (VI ≥ 84 v.v.), **và** rơi vào K1, **và** không rơi vào K2, K4 hay K5.

## 6. Việc cần làm trước khi train (nếu bạn duyệt)

1. Worker dựng P3 và P4 bằng script, và soạn danh sách chữ cho P1 và P5 → **bạn duyệt**.
2. Chấm M0, M1 (và decider-2b, Quyet nếu cần) trên mọi phép thử, để có mốc.
3. Kiểm script train có lưu adapter ở 4 mốc được không; nếu chưa thì thêm.
4. Ghi toàn bộ kế hoạch này vào PLAN_decisions trước khi train.

Chi phí ước tính: vài giờ GPU để chấm mốc (mỗi phép thử chỉ là một phần của dev), mức dùng Claude thấp. Riêng việc Opus đọc lỗi khoảng 100 câu mỗi model.
