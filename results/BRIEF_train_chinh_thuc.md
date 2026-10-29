# Brief: train chính thức (bước 6), bản đề xuất chờ duyệt

Ngày 2026-10-17. Người viết: điều phối. **Đây là đề xuất. Chưa train, chưa dựng dữ liệu, chưa giao việc.** Sau khi người dùng duyệt, việc đầu tiên được giao chỉ là Giai đoạn 0. Mọi số đo dựa trên dev 1.3 (`data/bench_v1_3/dev_v1_3.jsonl`, sha `c6a9a75d…`). Test 1.3 (sha `2bdbd210…`) không được chạy model nào cho tới Giai đoạn 3.

## 0. Quyết định của người dùng (chat 17/10)

- **(a)** ĐỒNG Ý: được đọc văn bản test 1.3 để khử rò rỉ. Không chạy model.
- **(b)** NỘI BỘ: model chỉ dùng nội bộ, giữ mọi loại giấy phép, kể cả UIT. Ghi rõ giấy phép từng dòng trong manifest.
- **(c)** Phía dự án chỉ **tìm và chuẩn bị dữ liệu**: chọn dòng, recast, khử rò rỉ phía tiếng Anh, xuất file cần dịch. **Việc dịch do một agent khác người dùng cử.** Không dịch, không dùng model dịch tại máy. Khi nhận bản dịch về: kiểm định dạng, khử rò rỉ lại phía tiếng Việt, rồi mới dùng.
- **(f)** Công bố: tính sau, khi model đủ mạnh.
- **(d)** Train thử cả 3 bản **A, B, C**.
- **(e)** Train chính thức **2 seed**. **Không** có nhánh Qwen3.5-2B; chốt Qwen3-1.7B.

## 1. Chẩn đoán: model dự án thua ở đâu

Model dự án hiện tại là Qwen3-1.7B + LoRA full_vocab, seed 1, train trên train_v0. Bảng dưới là VI có dấu trên dev 1.3, so với decider-2b text (decider-2b dựng từ Qwen3.5-2B-Base).

| Nguồn | Dự án | decider-2b | Chênh | Dạng bài |
|---|---|---|---|---|
| ViFactCheck | 72,8 | 83,5 | **−10,8** | kiểm chứng |
| MASSIVE | 86,7 | 96,3 | **−9,5** | ý định |
| XCOPA | 78,0 | 86,0 | −8,0 | nhân quả |
| ViWikiFC | 76,0 | 82,2 | −6,3 | kiểm chứng trên Wikipedia |
| Belebele | 82,3 | 87,8 | −5,4 | đọc hiểu |
| SIB-200 | 82,6 | 82,3 | ngang | chủ đề |
| **Trung bình đều** | **79,9** | **86,5** | −6,6 | |
| VI không dấu | **62,4** | 49,6 | **+12,7** | điểm mạnh cần giữ |
| EN | 88,0 | 90,3 | −2,4 | |
| Độ trễ batch 1 | **33 ms** | 52 ms | | |

Ghi chú: số theo nguồn lấy từ dev 1.2 (bảng E). Dev 1.3 chỉ khác 1.2 ở 2 cặp ViWikiFC bị loại và 1 câu Belebele; điểm chính lệch không quá 0,2.

**Điều quyết định hướng đi:** khi train thêm decider-2b trên đúng train_v0, ta được `decider-ft-ep2`. Model này đạt 84,9 trên dev 1.2, **thấp hơn** decider-2b gốc (86,3). Vậy thêm dữ liệu kiểu train_v0 không thu hẹp được khoảng cách. Thứ còn thiếu là **dạng bài**:
- train_v0 phần tiếng Việt chỉ có NLI, cảm xúc/khía cạnh, chủ đề hẹp và độc hại (UIT qua PhoLaya);
- không có đọc hiểu, ý định, kiểm chứng hay nhân quả bằng tiếng Việt.

Vì vậy brief đề xuất **bổ sung theo dạng bài còn thiếu**, không đơn thuần tăng số dòng.

## 2. Mục tiêu đặt trước, chốt trước khi chạy

Tất cả tính trên dev 1.3, batch 1, cùng cách tính của `scores_v1_3_trial.json`: trung bình đều theo nguồn, CI bootstrap theo state, McNemar theo cặp.

| Loại | Chỉ tiêu |
|---|---|
| **Chính** | VI có dấu (điểm chính) **≥ 84,0**, và CI của chênh so với model hiện tại (79,9) không chứa 0 |
| Giữ: không dấu | VI không dấu ≥ 61,0 (hiện 62,4) |
| Giữ: tiếng Anh | EN ≥ 87,0 (hiện 88,0) |
| Giữ: tốc độ | Trung vị độ trễ batch 1 trên dev 1.3 ≤ 40 ms. Kiểu dùng là từng yêu cầu một, nên đây là ràng buộc cứng |
| Theo nguồn | ViFactCheck và MASSIVE phải tăng (McNemar p < 0,05). Belebele và ViWikiFC: không tụt. **XCOPA: chỉ yêu cầu không tụt** (xem mục 3.3) |
| Báo cáo kèm | Lát "bỏ câu mơ hồ"; Brier; nhất quán khi đảo thứ tự; bảng cờ "nguồn đã thấy khi train" (PLAN mục 8) |

Model không đạt chỉ tiêu chính thì không lên Giai đoạn 2. Khi đó báo lại, không tự đổi chỉ tiêu.

## 3. Dữ liệu

### 3.1 Quy tắc cứng

- **Không dùng bất kỳ split nào** (kể cả train và validation) của: Belebele/FLORES, SIB-200, MASSIVE (và IntentGrasp, SLURP vì chứa câu MASSIVE), XCOPA/COPA/Balanced-COPA, ViFactCheck, ViWikiFC, MLQA (sắp vào 1.4), Global-PIQA. Mọi bản sao của chúng cũng loại.
- Giữ nguyên **16.000 dòng tiếng Anh của train_v0** làm phần cố định, để so được với các lần trước. Chỉ phần tiếng Việt thay đổi.
- Mỗi nguồn không quá 5% tổng dòng, như train_v0.
- Khử rò rỉ bằng `scripts/decontam.py` (trùng, MinHash ≥ 0,8, chứa 13-gram) với **toàn bộ văn bản dev 1.3 và test 1.3**, cùng mọi split của các nguồn bench. Bước này phải *đọc văn bản* test 1.3, không chạy model. Cần người dùng cho phép (mục 8a).
  - Lưu ý: train_v0 hiện trùng 2 câu với test 1.3 (leakage_v1_3.json). Bản train mới phải bằng 0.

### 3.2 Thành phần đề xuất, theo dạng bài còn thiếu

Nguồn tiếng Việt lấy từ khảo sát `results/train_v1_survey/REPORT_sources.md` (13/10). Số dòng là đề xuất, chốt ở Giai đoạn 0.

| Dạng bài (nguồn bench bị thua) | Bản địa mở | Dịch máy từ tasksource (EN→VI) | Rủi ro |
|---|---|---|---|
| Đọc hiểu (Belebele) | VSMRC 2.000; EXAMS-vi 500 | race, quail, cosmos_qa, dream: 2.000 | VSMRC trùng ViWikiFC cao, phải khử chặt |
| Ý định (MASSIVE) | vi-tourism-admin-intent 1.000 (sinh từ mẫu câu, miền hẹp) | banking77, clinc_oos, multidogo: 1.500 | Không dùng MASSIVE, SLURP, IntentGrasp |
| Kiểm chứng (ViFactCheck, ViWikiFC) | (aiMy144 loại vì không rõ giấy phép) | vitaminc, nli_fever, health_fact: 2.000 | Nguồn tin tức tiếng Việt trùng ViFactCheck cao, nên không dùng |
| Yes/no (noul) | ViANLI entailment 1.000 | boolq, sharc: 800 | ViANLI có thẻ giấy phép mâu thuẫn |
| Chủ đề (SIB-200, đang ngang) | VNTC 1.000 | | Tin cũ 2007, rủi ro vừa |
| NLI, cảm xúc, khía cạnh, độc hại | giữ 4.000 dòng VI của train_v0 | | UIT research-only |

### 3.3 Nhân quả (XCOPA)

Kho tasksource đã khảo sát chỉ có nguồn nhân quả thuộc họ XCOPA, đều bị loại cứng. Hiện **không có đòn bẩy train cho XCOPA**, nên chỉ tiêu XCOPA là "không tụt". Giai đoạn 0 sẽ tìm một nguồn nhân quả không thuộc họ COPA. Nếu tìm được, báo lại để người dùng quyết có thêm vào không.

### 3.4 Dịch máy cho dữ liệu train

- **Mặc định:** model dịch chạy tại máy, giấy phép cho dùng đầu ra để train: Qwen3-4B hoặc Hy-MT2-1.8B (Apache-2.0, theo khảo sát).
- **Opus chỉ dùng để soát mẫu** khoảng 100 dòng dịch, đo tỷ lệ lỗi.
- **Không đề xuất dùng Opus để dịch hàng chục nghìn dòng train**, vì:
  - tốn nhiều mức dùng, trong khi quy tắc là không dùng mức dùng thêm;
  - điều khoản của Anthropic giới hạn việc dùng đầu ra của Claude để huấn luyện model.
  - Quy tắc "dùng Opus để dịch, kiểm tra" của người dùng áp cho **văn bản benchmark**. Cần người dùng xác nhận nó không áp cho dữ liệu train (mục 8c).
- Trước khi dịch hàng loạt: chạy thử 50 dòng để đo tốc độ thật. Tốc độ trong khảo sát chỉ là ước lượng, chưa đo.

### 3.5 Giấy phép (ảnh hưởng tới việc công bố model)

train_v0 hiện có:
- 9.324 dòng commercial;
- 5.070 dòng không ghi rõ;
- 4.000 dòng research-only (UIT);
- 1.606 dòng non-commercial.

Bản train mới phải theo **một** chính sách do người dùng chọn (mục 8b):
- **(i) Chỉ dùng nội bộ:** giữ mọi loại giấy phép.
- **(ii) Công bố được:** chỉ giữ commercial và các giấy phép mở. Bỏ UIT, trừ khi có thỏa thuận với nhóm UIT. Bỏ non-commercial. Quyết riêng về dòng "không ghi rõ".

## 4. Công thức train

Giữ đúng công thức của model hiện tại, để mọi thay đổi chỉ đến từ dữ liệu:
- **Model:** Qwen/Qwen3-1.7B @ `70d244cc`. Model nền chốt theo kiểu dùng từng yêu cầu một.
- **LoRA:** r16, α32, dropout 0,05, gắn vào mọi lớp chiếu attention và MLP.
- **Tối ưu:** lr 2e-4, batch hiệu dụng 32, warmup rồi cosine, 2 epoch.
- **Loss:** full_vocab.
- **Cổng kiểm ở bước 250:** letter_mass. Prompt dùng chung module với lúc chấm.
- **Nhánh tùy chọn, chỉ khi người dùng bật:**
  - Qwen3.5-2B với cùng dữ liệu, khoảng 5–7 giờ. Chậm hơn 1.7B khoảng 1,8 lần nên khó đạt 40 ms.
  - Bỏ dấu ngẫu nhiên: strip_v0 trước đây không đạt, nên không đưa vào mặc định.

## 5. Các giai đoạn, mỗi giai đoạn dừng chờ người dùng

| Giai đoạn | Việc | Ra | Thời gian ước tính |
|---|---|---|---|
| **0. Dựng dữ liệu** (không train) | Tải và ghim revision các nguồn; recast bằng đúng script; dịch thử 50 dòng rồi dịch hàng loạt; Opus soát 100 dòng; khử rò rỉ; manifest | train_v1 + manifest + báo cáo khử rò rỉ | Nửa ngày đến 1 ngày; dịch 1–3 giờ GPU (chưa đo) |
| **1. Train thử** | 1 seed, khoảng 30.000 dòng mỗi bản: **A** = train_v0 + bản địa mở; **B** = train_v0 + dịch máy; **C** = A + B. Mốc so là model hiện tại, không train lại | Bảng so trên dev 1.3 theo mục 2 | Khoảng 2,8 giờ mỗi bản, tổng khoảng 8,5 giờ |
| **2. Train chính thức** | Bản thắng ở giai đoạn 1, 2 seed; cân nhắc tăng lên khoảng 60.000 dòng nếu giai đoạn 1 cho thấy còn tăng theo dữ liệu | 2 adapter + báo cáo dev | Khoảng 2,8 giờ (30.000 dòng) đến 5,5 giờ (60.000 dòng) mỗi seed |
| **3. Chấm test 1.3** | **Đúng một lần**, batch 1, model đã chốt và các model so sánh. Chỉ chạy khi người dùng cho phép riêng | Số chính thức | Dưới 1 giờ |

Cách ước thời gian: lần train hiện tại mất 6.619 giây cho 20.000 dòng × 2 epoch, tức khoảng 0,165 giây mỗi dòng mỗi epoch. Lần đó dùng 10,2 GB bộ nhớ GPU với micro-batch 8. Dòng đọc hiểu dài có thể buộc giảm micro-batch xuống 4, khi đó chậm hơn.

**Mức dùng Claude:** chủ yếu là viết script, theo dõi và Opus soát 100 dòng. Ước khoảng vài phần trăm hạn mức tuần cho mỗi giai đoạn; chưa đo. Ngưỡng giữ nguyên: từ 88% không mở việc mới, 92% dừng hẳn, không dùng mức dùng thêm, Fable < 90%.

## 6. Ràng buộc chung

- Không chạy model trên test trước Giai đoạn 3. Không chỉnh model, prompt hay hyperparameter theo điểm test.
- Ghim revision mọi dataset và model; seed cố định; chạy lại phải ra cùng con số.
- Không commit hay xóa file nếu không được bảo. Mọi bước ghi một dòng vào PLAN_decisions.
- Nguồn nào không tải được hoặc không làm được như mô tả: dừng và báo, không tự thay nguồn khác.

## 7. Rủi ro đã biết

- Dữ liệu dịch máy không phải văn bản bản địa. Có thể giúp kỹ năng nhưng làm câu chữ kém tự nhiên. Bản A so với B sẽ cho biết.
- Nguồn bản địa mở phần lớn có nhãn do máy sinh (VSMRC, intent). Sẽ gắn cờ để tách khi phân tích.
- Nguồn Wikipedia và tin tức dễ trùng ViWikiFC và ViFactCheck, nên khử rò rỉ phải chặt. Nếu bị loại quá nhiều thì báo lại.
- Chỉ tiêu 84,0 có thể không đạt chỉ bằng dữ liệu; decider-2b có model nền khác. Khi đó dừng và báo, cân nhắc nhánh Qwen3.5-2B.

## 8. Cần người dùng quyết trước Giai đoạn 0

- **(a)** Cho phép *đọc văn bản* test 1.3 để khử rò rỉ (không chạy model)?
- **(b)** Chính sách giấy phép: (i) chỉ dùng nội bộ, hay (ii) công bố được (bỏ UIT và non-commercial)? Câu này gắn với **(f)**.
- **(c)** Dịch dữ liệu train bằng model tại máy, Opus chỉ soát mẫu: đồng ý?
- **(d)** Train thử cả 3 bản A, B, C, hay chỉ A và B?
- **(e)** Giai đoạn 2 chạy 2 seed? Có bật nhánh Qwen3.5-2B không?
- **(f)** Có định công bố model (bước 8 của PLAN) không?
