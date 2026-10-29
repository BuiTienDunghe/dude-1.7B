# Soát nhãn mù 134 cặp dev 1.2

Ngày 2026-10-16. Dev: `data/bench_v1_2/dev_v1_2.jsonl` (sha `ef88be16…`, đã kiểm). Không sửa dev, test, bench hay config. Riêng một đề xuất sửa chữ được ghi vào `config/vi_proposals.jsonl` ở trạng thái CHỜ DUYỆT. Không chạy gì trên test, không train.

## Tóm tắt

- **Phạm vi:** 134 cặp. Đây là các cặp trong `results/baselines_v1_2/ung_vien_soat_nhan.jsonl` có ít nhất một dòng VI có dấu hoặc EN là ứng viên; ứng viên chỉ ở bản không dấu thì bỏ. Đếm lại đúng 134.
  - Theo nguồn: SIB-200 43, ViWikiFC 31, Belebele 21, ViFactCheck 17, MASSIVE 11, XCOPA 11.
- **Kết quả:**
  - nhóm 1 (đáp án chuẩn đúng): **105**;
  - nhóm 2 (đáp án chuẩn sai): **0**;
  - nhóm 3 (bản dịch VI đổi nghĩa so với EN): **1**;
  - nhóm 4 (mơ hồ): **28**;
  - nhóm 5 (không đủ căn cứ): 0.
- **Không tìm thấy nhãn sai nào có căn cứ.**
  - Phần lớn ứng viên là chỗ các model chọn sai: 105/134.
  - Phần còn lại là câu mơ hồ, chủ yếu ở SIB-200 (22) và ViWikiFC (6).
- **Quyet-1.0-Medium dẫn đầu không phải nhờ nhãn sai:**
  - Ở 105 cặp có đáp án chuẩn đúng (nhóm 1), Quyet chọn đúng đáp án chuẩn 52/105 ở VI có dấu.
  - Trong cùng 105 cặp, decider-2b đúng 24, dự án 16.
  - Nếu bỏ 28 cặp mơ hồ, thứ hạng các model không đổi; mọi model tăng khoảng 1 điểm.

## 1. Cách làm (mù)

**Người soát:** 5 subagent Opus, mỗi agent một lô 26–27 cặp, chạy cục bộ trong phiên. Dữ liệu UIT không gửi ra dịch vụ nào khác. Agent chỉ dùng Read/Write, không mạng, không chạy code.

**Bước A (mù):**
- Người soát nhận câu VI (bản dev 1.2 hiện hành), câu EN nếu có, và một đoạn mô tả trung tính về dataset.
- Ví dụ mô tả: "SIB-200: nhãn theo chủ đề của BÀI gốc".
- KHÔNG có đáp án chuẩn, phiếu model, đáp án đa số hay tên model.
- Người soát tự chọn đáp án cho VI và EN độc lập, nêu lý do và các đáp án khác cũng bảo vệ được.

**Bước B:**
- File đáp án chuẩn của từng lô chỉ được tạo SAU khi lô đó xong bước A (script kiểm điều này).
- Người soát đọc đáp án chuẩn rồi xếp mỗi cặp vào đúng một nhóm, kèm bằng chứng và đề xuất.

**Kiểm chất lượng:** 15 cặp ngẫu nhiên (seed 20261016) được giao cho một người soát Opus thứ hai, làm lại A–B độc lập.

**Tiêu chuẩn thận trọng, cần biết khi đọc số "0 nhóm 2":**
- Hướng dẫn bước B dặn: ranh giới mờ (chủ đề SIB-200 của câu lẻ, ranh giới "không đủ thông tin" ở kiểm chứng) thì xếp nhóm 4 hoặc 1. Chỉ xếp nhóm 2 khi đáp án chuẩn rõ ràng mâu thuẫn với văn bản.
- Tiêu chuẩn này cố ý nghiêng về "không đổi nhãn", để tránh sửa bench cho hợp model.

## 2. Số cặp theo nhóm × nguồn

| Nguồn | Số cặp | Nhóm 1 | Nhóm 2 | Nhóm 3 | Nhóm 4 | Nhóm 5 | Trả lời mù VI trùng đáp án chuẩn |
|---|---|---|---|---|---|---|---|
| SIB-200 | 43 | 21 | 0 | 0 | 22 | 0 | 18/43 |
| ViWikiFC | 31 | 25 | 0 | 0 | 6 | 0 | 20/31 |
| Belebele | 21 | 20 | 0 | 1 | 0 | 0 | 19/21 |
| ViFactCheck | 17 | 17 | 0 | 0 | 0 | 0 | 16/17 |
| MASSIVE | 11 | 11 | 0 | 0 | 0 | 0 | 10/11 |
| XCOPA | 11 | 11 | 0 | 0 | 0 | 0 | 11/11 |
| **Tổng** | **134** | **105** | **0** | **1** | **28** | **0** | **94/134 (70%)** |

- **Trả lời mù trùng đáp án chuẩn:** VI 94/134 (70%), EN 58/86 (67%). Mọi khác biệt VI/EN của người soát đều không làm đổi đáp án, trừ cặp nhóm 3.
- **Nơi rơi vào của 40 cặp trả lời mù khác đáp án chuẩn:** 28 vào nhóm 4, 12 vào nhóm 1.
  - **12 cặp nhóm 1 là nơi người soát đổi ý sau khi thấy đáp án chuẩn**, mỗi lô 1, 3, 4, 5 có 3 cặp.
  - Ở 11/12 cặp, ngay lúc trả lời mù họ đã ghi đáp án chuẩn là "cũng bảo vệ được". Không cặp nào được trả lời mù với độ tin cậy cao (9 trung bình, 3 thấp).
  - Vì vậy đây chủ yếu là chốt một chỗ mơ hồ theo đáp án chuẩn, không phải lật một nhận định chắc chắn.
  - Danh sách và lý do có trong `review.jsonl`, trường `ghi_chu`. Ví dụ ở belebele `…IPC_Alpine_World_Championships#1`: đoạn văn phân biệt "failed to finish" với "failed to rank".
  - Ở lô 5, người soát nói rõ đã dựa vào quy ước nhãn của chính dataset: thêm từ đánh giá thì nhãn là "không đủ thông tin".
- **Các cặp đã sửa ở bản 1.2 nằm trong phạm vi:**
  - Có 4 cặp dự án đã đổi nhãn ở 1.2 (vifactcheck 4688; viwikifc uit_1036_2, uit_1100, uit_1829). Cả 4 đều trùng trả lời mù và vào nhóm 1.
  - Có 25 cặp đã sửa chữ VI ở 1.2: 22 vào nhóm 1, 2 vào nhóm 4, 1 vào nhóm 3.

## 3. Kiểm chất lượng (15 cặp, người soát thứ hai)

| | Đồng ý |
|---|---|
| Trả lời mù VI | 14/15 |
| Trả lời mù EN | 10/10 |
| Nhóm | 13/15 |
| Gộp "giữ" (nhóm 1, 5) và "đổi hay loại" (nhóm 2, 3, 4) | 13/15 |

Ba chỗ bất đồng:
- **massive:15812 và vifactcheck 7363:** người thứ nhất xếp nhóm 1, người thứ hai xếp nhóm 4. Cả hai người đều trả lời mù giống nhau (A), khác đáp án chuẩn.
- **viwikifc uit_863:** cả hai người đều xếp nhóm 1, nhưng trả lời mù khác nhau (A và B). Người thứ hai ghi cặp này sát ranh giới nhóm 4.

Như vậy, không ai thấy nhãn sai ở cả 15 cặp. Bất đồng chỉ nằm giữa "đúng" và "mơ hồ" ở ranh giới kiểm chứng và MASSIVE.

## 4. Quyet có dẫn đầu thật không

Không có cặp nào ở nhóm 2, nên không so được giữa nhóm 1 và nhóm 2. Thay vào đó, bảng dưới đếm số lần mỗi model chọn đúng đáp án chuẩn trong từng nhóm, ở dòng VI có dấu và dòng EN.

| Model | Nhóm 1, VI (105) | Nhóm 1, EN (63) | Nhóm 4, VI (28) | Nhóm 4, EN (22) | Nhóm 3, VI / EN (1) |
|---|---|---|---|---|---|
| Quyet-1.0-Medium | **52** | **44** | 4 | 1 | 0 / 1 |
| Qwen3.5-2B gốc | 36 | 29 | 9 | 4 | 0 / 0 |
| Qwen3-1.7B gốc | 25 | 30 | 7 | 3 | 0 / 1 |
| decider-2b (text) | 24 | 25 | 5 | 3 | 0 / 0 |
| decider-2b (keyed) | 21 | 24 | 5 | 3 | 0 / 0 |
| dự án (full_vocab) | 16 | 24 | 8 | 0 | 0 / 0 |

- Ứng viên được chọn vì đa số model (7 trong 9 phiếu thuộc họ decider và họ Qwen3-1.7B) chọn khác đáp án chuẩn. Vì vậy các model này đúng ít ở đây là do cách chọn ứng viên.
- Điểm đáng chú ý: ở 105 cặp mà người soát xác nhận đáp án chuẩn ĐÚNG, Quyet chọn đúng gần một nửa. Lợi thế của Quyet trong bảng E không đến từ nhãn sai.
- Ở 28 cặp mơ hồ, Quyet hiếm khi chọn đáp án chuẩn (4/28). Bỏ các cặp này thì Quyet tăng nhẹ, giống các model khác (mục 5).

## 5. Độ nhạy (chỉ tính, không sửa bench)

- Áp nhóm 2: không có cặp nào nên không đổi nhãn.
- Bỏ nhóm 4: bỏ 28 cặp, tức cả ba dòng của mỗi cặp.
- Áp nhóm 3 (sửa chữ) cần chấm lại, nên không tính ở đây.
- Accuracy là trung bình đều theo nguồn, đơn vị %.

| Model | VI có dấu: bây giờ → bỏ nhóm 4 | VI không dấu | EN |
|---|---|---|---|
| Quyet-1.0-Medium | 89,4 → 90,6 | 64,5 → 65,2 | 94,5 → 96,0 |
| decider-2b (text) | 86,3 → 87,4 | 49,6 → 50,1 | 90,3 → 91,7 |
| decider-2b (keyed) | 84,9 → 86,0 | 48,5 → 49,1 | 89,8 → 91,2 |
| dự án (full_vocab) | 79,7 → 80,6 | 62,3 → 63,0 | 88,0 → 89,6 |
| Qwen3-1.7B gốc | 70,6 → 71,4 | 48,5 → 49,0 | 84,6 → 85,9 |
| Qwen3.5-2B gốc | 70,6 → 71,3 | 39,0 → 39,2 | 84,5 → 85,8 |

Mọi model tăng 0,2–1,6 điểm; thứ hạng và khoảng cách giữa các model gần như không đổi.

## 6. Đề xuất (chờ người dùng quyết; không áp gì)

- **Nhóm 3, sửa chữ:** 1 đề xuất, đã ghi vào `config/vi_proposals.jsonl` với id `lr-07c4a55b9f82`, ở trạng thái CHỜ DUYỆT.
  - Cặp belebele `…/Airline_alliances#1`, lựa chọn C (option:2): "có thể cung cấp thêm nhiều quyền lợi" thành "có thể đang bổ sung thêm quyền lợi mới".
  - Lý do: EN là "might be adding additional privileges". Bản VI hiện làm C gần ngang đáp án chuẩn B.
  - `gia_tri_cu` xuất hiện đúng 1 lần ở cả bản dev 1.2 lẫn giá trị hiện hành của công cụ. release_check vẫn ĐẠT, báo 1 đề xuất đang chờ.
- **Nhóm 4, loại câu:** 28 cặp, chỉ ghi trong `review.jsonl`, không áp.
  - SIB-200 (22): 1158, 1233, 1256, 153, 155, 1689, 1774, 1821, 1883, 1915, 1944, 200, 227, 229, 325, 339, 566, 567, 650, 663, 667, 769.
  - ViWikiFC (6): uit_1014_58_57_1_12, uit_1657_115_2_1_32, uit_167_11_191_1_32, uit_24_1_108_1_22, uit_481_28_22_1_21, uit_484_30_8_3_32.
- **Nhóm 2, đổi nhãn:** không có.
- Mọi thay đổi được người dùng chấp nhận sẽ vào bản dev mới (1.2.1 hoặc 1.3). Bản 1.2 giữ nguyên.

## 7. Ước tỷ lệ nhãn sai còn lại của dev và các kiểu lỗi lặp lại

**Ước tỷ lệ:**
- **Nhãn sai có căn cứ:** 0/134 ứng viên. Cận trên 95% theo "quy tắc ba" là khoảng 3/134 ≈ 2,2% trong nhóm ứng viên.
- **Ngoài nhóm ứng viên:** tức 1.298 cặp dev VI còn lại, nơi đa số model đồng ý với đáp án chuẩn hoặc chia phiếu. Phép soát này KHÔNG đo được nhãn sai mà model cũng sai theo. Muốn ước phần này cần một mẫu ngẫu nhiên ngoài ứng viên do người soát.
- **Câu mơ hồ:** ít nhất 28/1.432 cặp dev (2,0%). Riêng SIB-200 có ít nhất 22/299 (7,4%), và đây mới chỉ là phần nằm trong ứng viên.
- **Các lần soát trước:** đã sửa 10 đáp án dev ở 1.2. 4 trong số đó nằm trong phạm vi lần này và được trả lời mù xác nhận.

**Các kiểu lặp lại, dùng cho lần soát test bằng người (KHÔNG chạy model trên test):**
1. **SIB-200, nhãn theo bài nhưng câu lẻ:**
   - Câu về hiện tượng tự nhiên, thiên văn có nhãn "địa lý" nhưng đọc như "khoa học".
   - Câu kiểu cẩm nang du lịch có nhãn "giải trí" nhưng đọc như "du lịch".
   - Giải phẫu cơ thể không nhất quán: tim và hệ thần kinh là "khoa học", mắt là "sức khỏe".
   - Các câu cùng bài có thể mang nhãn khác nhau (1880 du lịch, 1883 chính trị).
2. **ViWikiFC, ranh giới "không đủ thông tin":**
   - Nhận định thêm từ đánh giá ("to lớn", "gay gắt", "độc đáo") hoặc thêm quan hệ nhân quả ("nhờ vào", "do… nên").
   - Nhận định đổi một từ ("sản vật" thành "cống phẩm").
   - Quy ước của dataset không luôn nhất quán: hai mục cùng kiểu thêm nhân quả mang nhãn khác nhau (uit_1014 và uit_187).
3. **Belebele, sắc thái của lựa chọn khi dịch:** dịch sai thì, hoặc dịch sai mức chắc chắn ("might be adding" thành "có thể cung cấp thêm"), làm một lựa chọn gây nhiễu thành bảo vệ được.
4. **MASSIVE:**
   - Nhãn chung cho yêu cầu ngoài phạm vi (massive:15812, đáp án D) có mô tả không nói rõ đó là nhãn chung.
   - Vài câu không có lựa chọn nào khớp hẳn (8132, 16588), người soát phải chọn lựa chọn gần nhất.
5. **Tên riêng được địa phương hóa** (Hà Nội/LA, Mai/Jane) không làm đổi đáp án. Bản dịch "Lê" viết thường dễ đọc nhầm (massive:16244).

## 8. File

| File | Nội dung |
|---|---|
| `results/label_review_v1_2/review.jsonl` | mỗi cặp một dòng: id, nguồn, trả lời mù kèm lý do, đáp án chuẩn, nhóm, bằng chứng, đề xuất, ghi chú |
| `results/label_review_v1_2/summary.json` | số liệu của báo cáo này |
| `results/label_review_v1_2/blind/`, `A/`, `gold/`, `B/` | gói mù, trả lời mù, đáp án chuẩn (tạo sau bước A), phân loại |
| `results/label_review_v1_2/qc/` | kiểm chất lượng: blind, A, gold, B của người soát thứ hai |
| `results/label_review_v1_2/pairs.json` | danh sách cặp theo lô và mẫu kiểm chất lượng (seed 20261016) |
| `scripts/label_review_v1_2/prep_blind.py`, `analyze.py` | code |
