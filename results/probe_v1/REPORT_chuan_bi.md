# probe_v1: chuẩn bị bộ phép thử chẩn đoán (bước 0 của kế hoạch phân tích xu hướng)

Ngày 26/10/2026. Theo `results/BRIEF_phan_tich_xu_huong.md`. Người dùng đã duyệt kế hoạch.

**Đây là bộ phân tích, KHÔNG phải benchmark, và KHÔNG BAO GIỜ được dùng để train** (kể cả để chọn checkpoint). Điều này cũng ghi trong `manifest_probe_v1.json`.

**Phạm vi đã giữ:**
- Chỉ dùng dev 1.3 (sha `c6a9a75d…`, script kiểm sha trước khi chạy). Không đọc test.
- **Chưa chấm** model nào, **chưa train**.
- Không commit, không xóa file, không ghi `data/`. Chỉ append vào `config/vi_proposals.jsonl`.

## Tóm tắt

| Phép thử | Trạng thái | Số câu |
|---|---|---|
| P3 bỏ văn bản | **Đã dựng**, đóng băng bằng sha | 1.768 |
| P4 văn bản không liên quan | **Đã dựng**, đóng băng bằng sha | 886 (443 nhận định × 2 bản) |
| P1 đổi chữ nhãn | Đã soạn chữ: **144 đề xuất chờ người dùng duyệt**. Script dựng file đã viết và chạy thử bằng chữ chưa duyệt (ở thư mục nháp) | sẽ có 2.072 mỗi bản v1/v2 |
| P5 đổi câu hỏi | Đã soạn chữ: **6 đề xuất chờ duyệt**; script đã viết và chạy thử | sẽ có 2.272 |
| P2 đảo thứ tự, P6 không dấu | Đã có sẵn: dev chấm với `--reverse`, và các dòng không dấu trong dev | — |

**Kiểm script train:** `train_lora.py` trước đây chỉ lưu adapter ở cuối train. Đã thêm `--save_fracs`, **mặc định tắt**. Đã chạy giả (chỉ đường dữ liệu, không nạp model): bật hay tắt cờ thì thứ tự batch giống hệt byte.

**Trùng train_v1:** P3/P4 không trùng A/B/C. Chỉ có các lựa chọn dưới 4 từ trùng chữ, ví dụ "fear", "Hoa Kỳ" (chi tiết ở mục 4).

## 1. Số câu theo nguồn và mức đoán mò

### P3: bỏ văn bản

`p3_bo_van_ban.jsonl`, sha `6dbfff7c…`.

**Cách dựng:**
- `state = ""`: khung prompt vẫn in nhãn "Văn bản:" rồi để trống, nên **không thêm chữ tiếng Việt mới**.
- Câu hỏi, lựa chọn và đáp án giữ nguyên.
- Lấy mọi biến thể có trong dev: Belebele có VI có dấu, VI không dấu và EN; ViFactCheck và ViWikiFC chỉ có tiếng Việt.

| Nguồn \| ngôn ngữ \| biến thể \| kiểu | Số câu | Đoán mò (1/k) | Luôn chọn đáp án hay gặp nhất | Luôn chọn vị trí hay gặp nhất |
|---|---|---|---|---|
| belebele \| en \| orig | 294 | 25,0% | — (không có nhãn cố định) | 26,9% (A) |
| belebele \| vi \| orig | 294 | 25,0% | — | 26,9% (A) |
| belebele \| vi \| nodiac | 294 | 25,0% | — | 26,9% (A) |
| vifactcheck \| choice (orig, nodiac, mỗi bản) | 80 | 33,3% | 42,5% (không đủ thông tin) | 41,3% |
| vifactcheck \| noul (mỗi bản) | 78 | 50,0% | 64,1% (Không) | 64,1% |
| viwikifc \| choice (mỗi bản) | 161 | 33,3% | 46,6% (bác bỏ) | 37,3% |
| viwikifc \| noul (mỗi bản) | 124 | 50,0% | 62,1% (Không) | 62,1% |

**Tổng P3:** Belebele 882, ViFactCheck 316, ViWikiFC 570, cộng 1.768 câu.

**Cách đọc:** "gần mức đoán mò" phải so với **cả hai mốc**. Ở câu noul, model luôn trả lời "Không" đã được khoảng 62–64%.

### P4: văn bản không liên quan

`p4_van_ban_khong_lien_quan.jsonl`, sha `c10f394e…`.

**Ghép cặp:**
- Mỗi nhận định (bản có dấu) được ghép với state của một câu khác **cùng nguồn, khác bài**, seed 20261012:
  - ViFactCheck: khác cluster, tức khác bài báo;
  - ViWikiFC: khác mã bài Wikipedia (trường thứ 3 của id `uit_*`) và khác cluster.
- Bản không dấu dùng đúng cặp đó, lấy state không dấu của dòng dev tương ứng.
- Ứng viên được xáo theo seed và lấy ứng viên đầu tiên qua bộ lọc. Ngưỡng lọc chốt trước khi xem kết quả, dựa trên phân bố đo trên dev.

**Bộ lọc, bỏ ứng viên khi:**
- Jaccard bigram giữa hai state > 0,02 (để lọc cùng chủ đề; với hai state ngẫu nhiên, 95% có Jaccard ≤ 0,022);
- có cụm chữ hoa (tên riêng) của nhận định xuất hiện trong state mới;
- có số từ 2 chữ số trở lên của nhận định xuất hiện trong state mới;
- hơn 10% bigram của nhận định có trong state mới (nhận định với state của chính nó: trung vị 50%).

**Kết quả lọc:**
- Ứng viên bị loại: tên riêng 42, Jaccard 39, số 23, bigram 8.
- Không nhận định nào hết ứng viên, nên **không bỏ câu nào**.
- 221 đoạn văn khác nhau được dùng làm state mới (một đoạn dùng nhiều nhất 5 lần).

**Đáp án:**
- choice: lựa chọn "Văn bản không đủ thông tin để kết luận". Vị trí lấy theo từng dòng, vì lựa chọn đã bị xáo.
- noul: **"Không"**. Đây là diễn giải của worker: câu noul không có nhãn "không đủ thông tin", và văn bản không liên quan thì không xác nhận phát biểu. **Điều phối có thể bác** điểm này. Nếu bác, chỉ cần bỏ các dòng noul.

| Nguồn \| kiểu | Mỗi bản (orig / nodiac) | Đoán mò | Nhãn gốc của nhận định |
|---|---|---|---|
| vifactcheck \| choice | 80 | 33,3% | Cả hai kiểu gộp: bác bỏ 60, xác nhận 47, không đủ thông tin 51 |
| vifactcheck \| noul | 78 | 50,0% | |
| viwikifc \| choice | 161 | 33,3% | Cả hai kiểu gộp: bác bỏ 108, xác nhận 93, không đủ thông tin 84 |
| viwikifc \| noul | 124 | 50,0% | |

**Tổng P4:** ViFactCheck 316, ViWikiFC 570, cộng 886 câu.

**Cách đọc:** ở P4 đáp án luôn cùng một nhãn, nên model chỉ cần nghiêng về "không đủ thông tin"/"Không" là đã đạt điểm cao. Vì vậy `analyze_probe.py` ghi thêm một mốc: trên **các câu dev gốc của cùng nhận định**, model chọn nhãn đó bao nhiêu lần khi đáp án thật là nhãn khác. P4 chỉ có ý nghĩa khi so với mốc này.

**Đã đọc tay một vài cặp.** Phần lớn rõ ràng không liên quan, ví dụ nhận định về học phí ghép với bài về ông Thaksin.

Có cặp ở **vùng xám chủ đề**:
- Nhận định về Trung Quốc và quần đảo Nam Sa ghép với đoạn về tỉnh Palawan.
- Nhận định về Trung Quốc sau khi Liên Xô sụp đổ ghép với đoạn niên biểu năm 1969 có "xung đột biên giới Trung-Xô".

State mới ở các cặp này không chứa thông tin để kết luận về nhận định, nên đáp án vẫn đúng. Tuy vậy, model dựa vào kiến thức sẵn có có thể bị hút sang đáp án khác. Đó đúng là điều P4 muốn đo.

### P1 và P5: chờ duyệt chữ

**Số câu dự kiến** (đếm từ lần dựng thử bằng chữ chưa duyệt, ở thư mục nháp, chưa ghi vào `results/probe_v1`):

| | ViFactCheck | ViWikiFC | SIB-200 | MASSIVE | XCOPA | Tổng |
|---|---|---|---|---|---|---|
| P1 v1 | 316 | 570 | 598 | 588 | — | 2.072 |
| P1 v2 | 316 | 570 | 598 | 588 | — | 2.072 |
| P5 | 316 | 570 | 598 | 588 | 200 | 2.272 |

Mọi phép trong bảng tính trên VI có dấu và VI không dấu.

**Mức đoán mò:**
- Giống dev: ViFactCheck/ViWikiFC choice 1/3, noul 1/2, SIB-200 1/7, MASSIVE 1/10, XCOPA 1/2.
- Đáp án và thứ tự lựa chọn giữ như dev. P1 và P5 được so theo cặp với cùng câu ở dev (trường `orig_id`).

**Không áp dụng:**
- Belebele không có P5, vì câu hỏi Belebele là câu riêng từng câu, không có câu hỏi mẫu.
- EN không có P1/P5. Đây là phép thử chữ tiếng Việt.

## 2. Ví dụ (3 dòng mỗi phép thử, cắt 200 ký tự)

**P3**

| id | state | câu hỏi | lựa chọn (đáp án in đậm) |
|---|---|---|---|
| `belebele:…Wikijunior:Biology/Cells#1:en:P3` | (trống) | According to the passage, what did Robert Hooke do before other biologists? | Use a microscope to view an organism; **Observe dead cells**; Observe small squares; Discover the smallest functional units of an organism |
| `vifactcheck:dev:3830-19623871:vi:P3` | (trống) | Dựa vào văn bản, nhận định sau được văn bản xác nhận, bị văn bản bác bỏ, hay văn bản không đủ thông tin để kết luận? Nhận định: BS Lê Văn Dũng sinh năm 1997, Phó khoa Phẫu thuật - Gây mê hồi sức cho b… | **Văn bản không đủ thông tin để kết luận**; Bị văn bản bác bỏ; Được văn bản xác nhận |
| `viwikifc:dev:uit_189_12_102_1_21~5b2e8728:vi:nodiac:P3` | (trống) | Dua vao van ban, nhan dinh sau duoc van ban xac nhan, bi van ban bac bo, hay van ban khong du thong tin de ket luan? Nhan dinh: Cac le hoi truyen thong duoc to chuc o Nghe An thi khong da dang ve so l… | Duoc van ban xac nhan; Van ban khong du thong tin de ket luan; **Bi van ban bac bo** |

**P4**

| id | state | câu hỏi | lựa chọn (đáp án in đậm) |
|---|---|---|---|
| `vifactcheck:dev:2143-19412154:vi:P4` | Sau 16 năm sống lưu vong, ông Thaksin sẵn sàng về chấp hành án tù ở Thái Lan, để có thể dành phần đời còn lại cùng gia đình. "Tôi đã bị giam cầm 16 năm trong nhà tù lớn bởi họ ngăn tôi được ở cạnh gia… | Dựa vào văn bản, nhận định sau được văn bản xác nhận, bị văn bản bác bỏ, hay văn bản không đủ thông tin để kết luận? Nhận định: Cử trì đề nghị điều chỉnh lộ trình thu học phí theo hướng tăng mạnh rồi … | Được văn bản xác nhận; Bị văn bản bác bỏ; **Văn bản không đủ thông tin để kết luận** |
| `viwikifc:dev:uit_2735_163_62_3_22~677ca463:vi:P4` | Nhiều tác giả, Phan Huy Chú và dòng họ Phan Huy. Sở Văn hóa thông tin Hà Sơn Bình xuất bản, 1983. Trong đó, chủ yếu tham khảo 4 bài viết, là:-Phan Huy Lê, "Về dòng họ Phan Huy ở Sài Sơn". | Dựa vào văn bản, nhận định sau được văn bản xác nhận, bị văn bản bác bỏ, hay văn bản không đủ thông tin để kết luận? Nhận định: Đảng cộng sản chuyển vai trò cho tư nhân trong nền kinh tế thị trường xã… | **Văn bản không đủ thông tin để kết luận**; Bị văn bản bác bỏ; Được văn bản xác nhận |
| `viwikifc:dev:uit_547_34_4_2_11~cb2dabf9:vi:P4` | Lực lượng Bảo vệ Lăng Chủ tịch Hồ Chí Minh: thành lập năm 1976VPA có số lượng khoảng 450.000 người, còn tổng lực lượng, bao gồm cả bán quân sự và dân quân tự vệ, có thể lên khoảng 5.000.000 người. Năm… | Văn bản có xác nhận phát biểu sau không? Phát biểu: Châu Á là nơi tồn tại ba nước có nền văn minh lớn. | Có; **Không** |

**P1 v1** (dựng thử bằng chữ chưa duyệt, chỉ để xem dạng)

| id | state | câu hỏi | lựa chọn (đáp án in đậm) |
|---|---|---|---|
| `vifactcheck:dev:2143-19412154:vi:P1v1` | (PLO)- Theo Luật Giáo dục đại học, dù có xét tuyển đại học bằng điểm học bạ hay không, các trường phổ thông vẫn phải có biện pháp đánh giá đúng kết quả học tập của người học. Bộ GD&ĐT vừa có phúc đáp … | (giữ nguyên câu hỏi dev) … Nhận định: Cử trì đề nghị điều chỉnh lộ trình thu học phí theo hướng tăng mạnh rồi … | Đúng theo văn bản; **Sai theo văn bản**; Không xác định được từ văn bản |
| `sib200:1624:vi:P1v1` | 20% lượng nước đổ từ các con sông trên hành tinh này ra đại dương đều bắt nguồn từ Amazon. | Văn bản này thuộc chủ đề nào? | lữ hành; các môn thi đấu; khoa học và kỹ thuật; chính quyền và nhà nước; y tế; **vùng đất và địa hình**; nghệ thuật và biểu diễn |
| `massive:12768:vi:P1v1` | bao nhiêu tiền một vé xe lửa đến hồ chí minh | Người dùng muốn trợ lý làm gì? | cho biết tỷ giá ngoại tệ; cài một báo thức mới; đề xuất hoạt động nên tham gia; làm một ly cà phê; **tìm đường đi hoặc giờ chạy của xe, tàu**; báo ngày tháng hoặc thời gian; hủy một báo thức; … |

**P5** (dựng thử bằng chữ chưa duyệt)

| id | state | câu hỏi | lựa chọn (đáp án in đậm) |
|---|---|---|---|
| `viwikifc:dev:uit_1830_125_64_1_12~af4343a6:vi:P5` | Hận thù: "Các phong trào quần chúng có thể tăng lên và lan rộng mà không có Thiên Chúa, nhưng không bao giờ không có niềm tin vào một con quỷ ". "Ma quỷ lý tưởng" là một người nước ngoài. | Nội dung văn bản có cho thấy điều sau đây là đúng không? Phát biểu: Phong trào quần chúng có thể tự nhân lên. | **Có**; Không |
| `xcopa:35:vi:P5` | Cảnh sát đã khám xét xe của bọn tội phạm. | Điều gì nhiều khả năng đã dẫn đến chuyện này? | **Họ đang tìm kiếm ma túy bất hợp pháp.**; Họ đang cố khiến chúng thú tội. |
| `massive:12768:vi:P5` | bao nhiêu tiền một vé xe lửa đến hồ chí minh | Người dùng đang nhờ trợ lý việc gì? | trả lời câu hỏi về tỷ giá tiền tệ; đặt báo thức; gợi ý sự kiện; pha cà phê; **tra cứu tuyến đường hoặc lịch trình**; … |

## 3. Đề xuất chờ duyệt

**150 đề xuất loại mẫu** (khóa), append vào `config/vi_proposals.jsonl` (751 → 901 dòng):
- **P1:** 144 đề xuất = (3 nhãn kiểm chứng + 2 Có/Không + 7 chủ đề SIB + 60 ý định MASSIVE) × 2 cách diễn đạt.
- **P5:** 6 đề xuất.

**Không tự duyệt.** Người dùng duyệt, sửa hoặc bỏ trên công cụ duyệt như với các chuỗi `tv1-*`.

**Cách soạn:**
- Opus (worker) tự soạn.
- Chuỗi gốc lấy từ dev 1.3, không lấy từ config. Ví dụ: dev dùng "Văn bản không đủ thông tin **để kết luận**", và 8 mô tả MASSIVE trong dev khác `translations.json` hiện tại.

**Kiểm bằng script** (`propose_p1_p5.py`, 0 lỗi):
- Không đề xuất nào trùng (sau chuẩn hóa, cả có dấu và bỏ dấu) với câu hỏi hay lựa chọn của train_v1 A/B/C, dev 1.3, `translations.json`, `bench_v1_questions.json`, `train_format.json` hay bảng thuật ngữ v2.
- Trong cùng một nguồn, chuỗi gốc, v1 và v2 của mọi nhãn đôi một khác nhau, kể cả khi bỏ dấu.
- Tự kiểm chỉ mục: mọi chuỗi gốc đều bị bắt là "đã dùng".

**Lựa chọn thiết kế cần người dùng biết:**
- **Khóa dùng chung cho hai nguồn kiểm chứng.** ViFactCheck và ViWikiFC dùng chung một bộ nhãn, nên khóa là `probe_v1.P1.kiem_chung.*` cho choice và `probe_v1.P1.kiem_chung_noul.*` cho Có/Không, thay vì tách theo nguồn.
- **Không dùng "Đúng/Sai" cho Có/Không** (khác ví dụ trong brief). Với câu hỏi "Văn bản có xác nhận…?", "Không" gồm cả *bác bỏ* lẫn *không đủ thông tin*, còn "Sai" chỉ có nghĩa *bác bỏ*. Vì vậy đề xuất là "Có xác nhận/Không xác nhận" (v1) và "Đúng vậy/Không phải vậy" (v2).
- **P1 chỉ đổi lựa chọn, giữ nguyên câu hỏi.** Câu hỏi choice của kiểm chứng vẫn nêu ba nhãn bằng chữ gốc.
- **P5 chỉ đổi câu hỏi, giữ nguyên lựa chọn.** Câu P5 kiểm chứng không nhắc lại chữ của nhãn.
- **Hệ quả cho decider và Quyet ở P1 noul.** Hai model này trả lời câu noul bằng đường có/không riêng, không nhìn chữ lựa chọn, nên P1 noul không đo được gì ở chúng. Script chấm bỏ các dòng đó khi chấm R và ghi lại trong log chạy.

Bảng đầy đủ ở cuối báo cáo (phụ lục A). J là độ giống chữ với chuỗi gốc (Jaccard theo từ). J cao nhất là 0,5, ở các nhãn khó đổi chữ như "chơi trò chơi" → "mở một trò chơi".

## 4. Kiểm trùng train_v1 (P3, P4)

Kết quả ghi trong `manifest_probe_v1.json` → `trung_train_v1`. So với `components/A.jsonl` (24.349 dòng), `B.jsonl` (26.638) và `C.jsonl` (31.034).

**Trùng nguyên văn** (state, câu hỏi, lựa chọn; sau chuẩn hóa; bỏ chuỗi cố định lặp ≥ 3 lần trong dev):
- Chuỗi từ 4 từ trở lên: **0** ở cả A, B và C.
- Chuỗi dưới 4 từ: có trùng (C có 72 chuỗi khác nhau), đều là lựa chọn ngắn. Thường gặp nhất là "fear" (52 lần, từ emotion), "ba", "books", "Hoa Kỳ", "Trung Quốc".
- Chuỗi ngắn như vậy không phải rò rỉ, và đúng với quy tắc khử rò rỉ đã dùng (phép "chứa đoạn" chỉ xét từ 4 từ).

**Chứa 10-gram** (nhận định, và câu hỏi Belebele, nằm trong state train):
- **1 nhận định** (`vifactcheck:dev:255-19570487`, ở cả P3 và P4) chung cụm "cơ quan cảnh sát điều tra công an tp hcm" với dòng `vntc:train:33383`.
- Đây là cụm báo chí thông dụng, không phải cùng tin. Kết luận: không phải rò rỉ.

## 5. Kiểm script train

**Trước khi sửa:** `scripts/train_lora.py` chỉ lưu `<out_dir>/adapter/` ở cuối train, và `adapter_gate_step<N>` khi cổng letter_mass dừng sớm. **Không** lưu ở 25/50/75%.

**Đã thêm `--save_fracs`** (ví dụ `--save_fracs 0.25,0.5,0.75`):
- **Mặc định trống, tức tắt.** Không truyền cờ thì hành vi y như cũ.
- Bật thì lưu `<out_dir>/adapter_step<N>/` ở các mốc, ghi sự kiện `save_adapter` vào `train_log.jsonl` và `save_steps` vào `summary.json`.
- Mốc 100% chính là `adapter/` cuối train, nên không lưu lặp.
- Lưu adapter không tiêu RNG và không đổi thứ tự batch.

**Chạy giả** (`--dump_batches 3`, chỉ đường dữ liệu, không nạp model, không train):
- Với train_v0, `total_steps = 1250`. Cờ `--save_fracs 0.25,0.5,0.75,1.0` cho `save_steps = [312, 625, 938]`; mốc 1.0 bị bỏ đúng như thiết kế.
- `batch_dump.jsonl` khi bật cờ và khi tắt cờ **giống hệt từng byte**.

**Một điểm khác nhận thấy, ngoài phạm vi, chưa sửa:** `train_lora.py:73` đọc cố định `data/train_v0.jsonl`, chưa có tùy chọn chọn file train. Muốn train trên `results/train_v1/components/C.jsonl` thì cần thêm tùy chọn đó, hoặc dùng script khác.

## 6. Script (trong `scripts/probe_v1/`)

| Script | Việc | Trạng thái |
|---|---|---|
| `build_p3_p4.py` | Dựng P3, P4, manifest, kiểm trùng train_v1 | **Đã chạy**. Chạy lại cho cùng sha |
| `probe_strings.py` | Bảng chữ gốc và đề xuất P1/P5 (dùng chung) | — |
| `propose_p1_p5.py` | Kiểm và in bảng; `--ghi` để append đề xuất. Từ chối ghi lặp nếu đã có khóa `probe_v1.*` | **Đã chạy `--ghi` một lần** |
| `build_p1_p5.py` | Dựng P1 v1/v2 và P5 từ chữ **đã duyệt** | Viết sẵn, **chưa chạy thật** (xem chi tiết dưới bảng) |
| `score_probe.py` | Chấm một model (M0, M1, M2 + `--adapter`, decider-2b, quyet-medium) trên dev, P1–P5, batch 1; Quyet chạy offline. Chỉ gọi đúng các đường chấm đã có | Viết sẵn, **chưa chạy**; chỉ chạy `--dry` |
| `analyze_probe.py` | Lát cắt, M2 − M1, ma trận chuyển, trùng lỗi, Brier/ECE, đọc phép thử | Đã thử trên preds có sẵn |

**`build_p1_p5.py`:**
- Lấy `gia_tri_moi` mới nhất mỗi khóa trong `edits_log`, chỉ khi hành động là duyet hoặc sua_roi_duyet.
- Một nguồn chỉ được dựng khi đủ mọi nhãn của nguồn đó.
- Chế độ `--thu_de_xuat --out <nháp>` đã thử ở thư mục nháp. Chế độ này có rào chặn, không được ghi vào `results/probe_v1`.

**`score_probe.py`, các kiểm trước khi chấm:**
- sha của dev;
- sha của file phép thử khớp manifest;
- tên file và id không chứa "test".

**`analyze_probe.py`, cách tính:**
- **Lát cắt** theo đúng mục 2 của brief:
  - theo nguồn; nguồn gốc VI so với nguồn dịch; VI có dấu / VI không dấu / EN; choice so với noul;
  - số lựa chọn 2 / 3–4 / ≥5;
  - độ dài văn bản: tam phân riêng từng nguồn theo token state VI của cặp, ngưỡng ghi trong file kết quả;
  - đã sửa ở 1.2 so với giữ nguyên; có hay không cờ mơ hồ.
- **Khoảng tin cậy:** bootstrap 95% theo cụm (cluster của dev), 1.000 lần, seed cố định.
- **Chênh M2 − M1:** CI theo cặp câu, kèm McNemar chính xác.
- **Phép thử:**
  - P1/P5 so theo cặp với dev;
  - P2 đo tỉ lệ chọn cùng đáp án khi đảo thứ tự;
  - P3 so với hai mốc đoán mò;
  - P4 kèm mốc thiên vị (mục 1).

**Thử trên preds có sẵn** (`results/bench_v1_3_trial/eval`):
- Cài đặt: M1 = model dự án; M2 tạm thay bằng `strip-s1` chỉ để thử cơ chế; M0 = Qwen3-1.7B; R = decider-2b-text và Quyet-1.0-Medium.
- Kết quả: `phan_tich/THU_preds_co_san_M2_la_strip-s1.{json,md}`, chạy 1,4 giây.
- Số khớp với điểm đã báo: decider-2b-text VI có dấu 86,9%, khớp `scores_v1_3_trial.json` (86,85).
- Ví dụ đọc được ngay từ M1 (dev 1.3):

  | Chỉ số | Giá trị |
  |---|---|
  | Toàn dev | 75,8% [74,2; 77,4] |
  | VI có dấu | 80,8% |
  | VI không dấu | 62,8% |
  | Nguồn gốc VI / nguồn dịch (VI có dấu) | 75,2% / 83,4% |
  | Brier / ECE | 0,350 / 0,087 |
  | Trùng lỗi với decider: Jaccard / P(decider sai \| M1 sai) | 0,38 / 0,58 |

- Các nhánh phép thử (P1, P2, P3, P4, P5) **chưa có preds thật**. Đã chạy thử bằng **preds giả** đặt trong thư mục nháp (không nằm trong `results/`) để chắc code chạy hết các nhánh. Số giả không được báo.

## 7. Việc tiếp theo (chờ quyết định)

1. Người dùng duyệt 150 đề xuất `probe_v1.*`, rồi chạy `python scripts/probe_v1/build_p1_p5.py`.
2. Điều phối chốt diễn giải P4 noul: đáp án "Không" (giữ), hoặc bỏ các dòng noul.
3. Chấm mốc M0, M1, decider-2b, Quyet trên dev và P1–P5 bằng `score_probe.py`. Việc này **cần lệnh**; chưa chạy.
   **Chưa kiểm:** decider (`decide()`) và adapter Quyet xử lý P3 với state rỗng ra sao. Khi chấm, xem số câu `excluded`/`rejected` ở P3 của các model R. Với model M thì đã rõ: `render_prompt` in nhãn văn bản rồi để trống.
4. Khi train: bật `--save_fracs 0.25,0.5,0.75`. Riêng việc chọn file train cho `train_lora.py` cần quyết định riêng (mục 5).

Mức dùng lúc viết báo cáo: cửa sổ 5 giờ 4%, tuần 15%, mức dùng thêm 0 USD.

## Phụ lục A. Bảng đề xuất P1/P5 (chờ duyệt)

| Khóa | Chuỗi gốc (dev 1.3) | v1 | v2 |
|---|---|---|---|
| `probe_v1.P1.kiem_chung.supported.v1/v2` | Được văn bản xác nhận | Đúng theo văn bản (J=0.29) | Văn bản ủng hộ nhận định này (J=0.33) |
| `probe_v1.P1.kiem_chung.refuted.v1/v2` | Bị văn bản bác bỏ | Sai theo văn bản (J=0.29) | Văn bản phủ nhận nhận định này (J=0.22) |
| `probe_v1.P1.kiem_chung.nei.v1/v2` | Văn bản không đủ thông tin để kết luận | Không xác định được từ văn bản (J=0.23) | Văn bản chưa đủ căn cứ để đánh giá (J=0.29) |
| `probe_v1.P1.kiem_chung_noul.co.v1/v2` | Có | Có xác nhận (J=0.33) | Đúng vậy (J=0.00) |
| `probe_v1.P1.kiem_chung_noul.khong.v1/v2` | Không | Không xác nhận (J=0.33) | Không phải vậy (J=0.33) |
| `probe_v1.P1.sib200.science_technology.v1/v2` | khoa học/công nghệ | khoa học và kỹ thuật (J=0.29) | nghiên cứu, phát minh và thiết bị (J=0.00) |
| `probe_v1.P1.sib200.travel.v1/v2` | du lịch | lữ hành (J=0.00) | chuyến đi và tham quan (J=0.00) |
| `probe_v1.P1.sib200.politics.v1/v2` | chính trị | chính quyền và nhà nước (J=0.17) | chính phủ, bầu cử, ngoại giao (J=0.14) |
| `probe_v1.P1.sib200.sports.v1/v2` | thể thao | các môn thi đấu (J=0.00) | vận động viên và giải đấu (J=0.00) |
| `probe_v1.P1.sib200.health.v1/v2` | sức khỏe | y tế (J=0.00) | bệnh tật và cách chữa (J=0.00) |
| `probe_v1.P1.sib200.entertainment.v1/v2` | giải trí | nghệ thuật và biểu diễn (J=0.00) | phim ảnh, âm nhạc, lễ hội (J=0.00) |
| `probe_v1.P1.sib200.geography.v1/v2` | địa lý | vùng đất và địa hình (J=0.17) | núi non, sông biển, khí hậu (J=0.00) |
| `probe_v1.P1.massive.alarm_query.v1/v2` | xem các báo thức đã đặt | kiểm tra báo thức hiện có (J=0.20) | liệt kê những giờ báo thức đang bật (J=0.17) |
| `probe_v1.P1.massive.alarm_remove.v1/v2` | xóa báo thức | hủy một báo thức (J=0.40) | bỏ giờ báo thức đã cài (J=0.29) |
| `probe_v1.P1.massive.alarm_set.v1/v2` | đặt báo thức | cài một báo thức mới (J=0.33) | hẹn giờ chuông reo (J=0.00) |
| `probe_v1.P1.massive.audio_volume_down.v1/v2` | giảm âm lượng | vặn nhỏ tiếng (J=0.00) | cho loa kêu nhỏ lại (J=0.00) |
| `probe_v1.P1.massive.audio_volume_mute.v1/v2` | tắt tiếng | tắt âm thanh (J=0.25) | cho loa im lặng (J=0.00) |
| `probe_v1.P1.massive.audio_volume_other.v1/v2` | các điều chỉnh âm lượng khác | điều chỉnh âm thanh theo cách khác (J=0.44) | thay đổi độ lớn của tiếng (không phải tăng, giảm hay tắt) (J=0.00) |
| `probe_v1.P1.massive.audio_volume_up.v1/v2` | tăng âm lượng | vặn to tiếng (J=0.00) | cho loa kêu to hơn (J=0.00) |
| `probe_v1.P1.massive.calendar_query.v1/v2` | xem lịch | kiểm tra lịch hẹn (J=0.20) | cho biết lịch có những việc gì (J=0.12) |
| `probe_v1.P1.massive.calendar_remove.v1/v2` | xóa sự kiện khỏi lịch | hủy một sự kiện trong lịch (J=0.38) | bỏ một cuộc hẹn đã ghi trong lịch (J=0.08) |
| `probe_v1.P1.massive.calendar_set.v1/v2` | thêm sự kiện hoặc lời nhắc vào lịch | ghi lịch hẹn hoặc đặt lời nhắc (J=0.36) | lên lịch một việc hoặc nhắc việc (J=0.27) |
| `probe_v1.P1.massive.cooking_query.v1/v2` | trả lời câu hỏi về nấu ăn | giải đáp thắc mắc về nấu nướng (J=0.17) | giải thích một điều về bếp núc (J=0.08) |
| `probe_v1.P1.massive.cooking_recipe.v1/v2` | tìm công thức nấu ăn | tra cách chế biến một món (J=0.00) | đưa ra cách làm một món ăn (J=0.09) |
| `probe_v1.P1.massive.datetime_convert.v1/v2` | đổi giờ giữa các múi giờ | quy đổi thời gian sang múi giờ khác (J=0.30) | tính giờ ở một nơi có múi giờ khác (J=0.18) |
| `probe_v1.P1.massive.datetime_query.v1/v2` | cho biết ngày hoặc giờ | báo ngày tháng hoặc thời gian (J=0.22) | trả lời hôm nay ngày mấy, bây giờ mấy giờ (J=0.18) |
| `probe_v1.P1.massive.email_addcontact.v1/v2` | thêm liên hệ email | lưu một địa chỉ email vào danh bạ (J=0.09) | bổ sung người liên lạc qua thư điện tử (J=0.08) |
| `probe_v1.P1.massive.email_query.v1/v2` | kiểm tra email | xem hộp thư (J=0.00) | đọc thư điện tử mới nhận (J=0.00) |
| `probe_v1.P1.massive.email_querycontact.v1/v2` | tra cứu thông tin liên hệ | tìm số điện thoại hoặc địa chỉ của một người (J=0.00) | xem thông tin một người trong danh bạ (J=0.17) |
| `probe_v1.P1.massive.email_sendemail.v1/v2` | gửi email | gửi thư điện tử (J=0.20) | viết thư điện tử cho ai đó (J=0.00) |
| `probe_v1.P1.massive.general_greet.v1/v2` | chào hỏi | đáp lời chào (J=0.25) | chào lại và hỏi thăm (J=0.40) |
| `probe_v1.P1.massive.general_joke.v1/v2` | kể chuyện cười | nói một câu đùa (J=0.00) | kể chuyện vui cho người dùng nghe (J=0.25) |
| `probe_v1.P1.massive.general_quirky.v1/v2` | trò chuyện phiếm hoặc đáp một câu nói kỳ quặc, lạc đề | tán gẫu hoặc đáp lại câu nói ngoài lề, khác thường (J=0.21) | nói chuyện vu vơ hoặc phản hồi lời nói lạ, không liên quan (J=0.14) |
| `probe_v1.P1.massive.iot_cleaning.v1/v2` | bật robot hút bụi | cho máy hút bụi tự động chạy (J=0.22) | khởi động robot dọn nhà (J=0.12) |
| `probe_v1.P1.massive.iot_coffee.v1/v2` | pha cà phê | làm một ly cà phê (J=0.33) | cho máy pha cà phê chạy (J=0.50) |
| `probe_v1.P1.massive.iot_hue_lightchange.v1/v2` | đổi màu đèn | chuyển màu ánh sáng của đèn (J=0.29) | thay đèn sang màu khác (J=0.33) |
| `probe_v1.P1.massive.iot_hue_lightdim.v1/v2` | giảm độ sáng đèn | làm đèn tối bớt (J=0.14) | chỉnh đèn mờ đi (J=0.14) |
| `probe_v1.P1.massive.iot_hue_lightoff.v1/v2` | tắt đèn | cho đèn ngừng sáng (J=0.20) | tắt hệ thống chiếu sáng (J=0.17) |
| `probe_v1.P1.massive.iot_hue_lighton.v1/v2` | bật đèn | mở đèn (J=0.33) | cho đèn bắt đầu sáng (J=0.17) |
| `probe_v1.P1.massive.iot_hue_lightup.v1/v2` | tăng độ sáng đèn | làm đèn sáng hơn (J=0.33) | chỉnh đèn sáng rõ lên (J=0.29) |
| `probe_v1.P1.massive.iot_wemo_off.v1/v2` | tắt ổ cắm thông minh | ngắt ổ điện thông minh (J=0.43) | tắt nguồn ổ cắm điều khiển từ xa (J=0.30) |
| `probe_v1.P1.massive.iot_wemo_on.v1/v2` | bật ổ cắm thông minh | mở ổ điện thông minh (J=0.43) | cấp điện cho ổ cắm điều khiển từ xa (J=0.17) |
| `probe_v1.P1.massive.lists_createoradd.v1/v2` | tạo danh sách hoặc thêm vào danh sách | lập danh sách mới hoặc ghi thêm mục vào danh sách (J=0.50) | bổ sung việc cần làm hoặc lập danh mục mới (J=0.14) |
| `probe_v1.P1.massive.lists_remove.v1/v2` | xóa mục khỏi danh sách hoặc xóa danh sách | bỏ một mục trong danh sách hoặc hủy cả danh sách (J=0.36) | gạch bớt mục hoặc bỏ hẳn một danh mục (J=0.27) |
| `probe_v1.P1.massive.lists_query.v1/v2` | đọc danh sách thành tiếng | đọc to các mục trong danh sách (J=0.33) | cho biết danh mục gồm những gì (J=0.09) |
| `probe_v1.P1.massive.music_dislikeness.v1/v2` | ghi nhận người dùng không thích một bài hát hoặc thể loại nhạc | lưu lại rằng người dùng không ưa một bài hoặc dòng nhạc (J=0.39) | đánh dấu bài hát hay loại nhạc người dùng ghét (J=0.35) |
| `probe_v1.P1.massive.music_likeness.v1/v2` | ghi nhận người dùng thích một bài hát hoặc thể loại nhạc | lưu lại rằng người dùng ưa một bài hoặc dòng nhạc (J=0.35) | đánh dấu bài hát hay loại nhạc người dùng yêu thích (J=0.44) |
| `probe_v1.P1.massive.music_settings.v1/v2` | đổi chế độ phát nhạc (lặp lại, phát ngẫu nhiên) | chỉnh cách phát nhạc (phát lặp, trộn bài) (J=0.23) | cài kiểu nghe nhạc như lặp bài hay xáo trộn (J=0.12) |
| `probe_v1.P1.massive.music_query.v1/v2` | cho biết nhạc gì đang phát | nói tên bài hát đang mở (J=0.09) | cho biết đang nghe bài gì (J=0.50) |
| `probe_v1.P1.massive.play_audiobook.v1/v2` | phát sách nói | mở một cuốn sách audio (J=0.14) | bật truyện audio (J=0.00) |
| `probe_v1.P1.massive.play_game.v1/v2` | chơi trò chơi | mở một trò chơi (J=0.50) | chơi game (J=0.33) |
| `probe_v1.P1.massive.play_music.v1/v2` | phát nhạc | mở nhạc (J=0.33) | bật một bài hát (J=0.00) |
| `probe_v1.P1.massive.play_podcasts.v1/v2` | phát podcast | mở podcast (J=0.33) | nghe một chương trình podcast (J=0.17) |
| `probe_v1.P1.massive.play_radio.v1/v2` | mở radio | bật đài (J=0.00) | nghe đài phát thanh (J=0.00) |
| `probe_v1.P1.massive.qa_currency.v1/v2` | trả lời câu hỏi về tỷ giá tiền tệ | cho biết tỷ giá ngoại tệ (J=0.25) | quy đổi giữa các loại tiền (J=0.07) |
| `probe_v1.P1.massive.qa_definition.v1/v2` | giải nghĩa một từ hoặc thuật ngữ | định nghĩa một từ (J=0.38) | cho biết ý nghĩa của một từ ngữ (J=0.36) |
| `probe_v1.P1.massive.qa_factoid.v1/v2` | trả lời một câu hỏi kiến thức | giải đáp câu hỏi về sự thật, kiến thức chung (J=0.31) | cho biết một thông tin thực tế (J=0.08) |
| `probe_v1.P1.massive.qa_maths.v1/v2` | tính toán | làm một phép tính (J=0.20) | giải bài toán số học (J=0.17) |
| `probe_v1.P1.massive.qa_stock.v1/v2` | cung cấp thông tin chứng khoán | báo giá cổ phiếu (J=0.00) | cập nhật diễn biến thị trường cổ phiếu (J=0.00) |
| `probe_v1.P1.massive.recommendation_events.v1/v2` | gợi ý sự kiện | đề xuất hoạt động nên tham gia (J=0.00) | giới thiệu chương trình, lễ hội sắp diễn ra (J=0.00) |
| `probe_v1.P1.massive.recommendation_locations.v1/v2` | gợi ý địa điểm | đề xuất nơi nên đến (J=0.00) | giới thiệu quán xá, chỗ đi chơi (J=0.00) |
| `probe_v1.P1.massive.recommendation_movies.v1/v2` | gợi ý phim | đề xuất phim nên xem (J=0.14) | giới thiệu bộ phim hay (J=0.14) |
| `probe_v1.P1.massive.social_query.v1/v2` | xem cập nhật trên mạng xã hội | kiểm tra thông báo mạng xã hội (J=0.27) | đọc bài mới đăng trên các trang mạng (J=0.15) |
| `probe_v1.P1.massive.social_post.v1/v2` | đăng bài lên mạng xã hội hoặc gửi khiếu nại | đăng tin lên trang mạng hoặc phản ánh với công ty (J=0.24) | viết bài lên mạng xã hội hoặc gửi lời phàn nàn (J=0.50) |
| `probe_v1.P1.massive.takeaway_order.v1/v2` | đặt đồ ăn mang về | gọi món giao tận nơi (J=0.00) | mua đồ ăn để mang đi (J=0.38) |
| `probe_v1.P1.massive.takeaway_query.v1/v2` | hỏi về đơn đồ ăn mang về hoặc việc giao đồ ăn | kiểm tra tình trạng đơn đồ ăn đã đặt (J=0.20) | hỏi đồ ăn đặt khi nào giao tới (J=0.31) |
| `probe_v1.P1.massive.news_query.v1/v2` | cung cấp tin tức | đọc tin mới (J=0.17) | cho biết thời sự hôm nay (J=0.00) |
| `probe_v1.P1.massive.transport_query.v1/v2` | tra cứu tuyến đường hoặc lịch trình | tìm đường đi hoặc giờ chạy của xe, tàu (J=0.14) | xem lộ trình hoặc thời gian biểu đi lại (J=0.14) |
| `probe_v1.P1.massive.transport_traffic.v1/v2` | cho biết tình hình giao thông trên đường | báo tình trạng kẹt xe (J=0.08) | cho biết đường có đông xe không (J=0.25) |
| `probe_v1.P1.massive.transport_taxi.v1/v2` | đặt taxi | gọi taxi (J=0.33) | gọi xe đưa đi (J=0.00) |
| `probe_v1.P1.massive.transport_ticket.v1/v2` | đặt vé tàu xe | mua vé tàu hoặc vé xe (J=0.50) | giữ chỗ cho chuyến đi (J=0.00) |
| `probe_v1.P1.massive.weather_query.v1/v2` | cho biết thời tiết | dự báo thời tiết (J=0.33) | cho biết trời nắng hay mưa (J=0.25) |

| Khóa | Câu hỏi gốc (dev 1.3) | Đề xuất |
|---|---|---|
| `probe_v1.P5.sib200` | Văn bản này thuộc chủ đề nào? | Đoạn văn trên nói về lĩnh vực gì? |
| `probe_v1.P5.massive` | Người dùng muốn trợ lý làm gì? | Người dùng đang nhờ trợ lý việc gì? |
| `probe_v1.P5.xcopa_cause` | Nguyên nhân nào hợp lý hơn? | Điều gì nhiều khả năng đã dẫn đến chuyện này? |
| `probe_v1.P5.xcopa_effect` | Kết quả nào hợp lý hơn? | Chuyện gì nhiều khả năng xảy ra tiếp theo? |
| `probe_v1.P5.kiem_chung_choice` | Dựa vào văn bản, nhận định sau được văn bản xác nhận, bị văn bản bác bỏ, hay văn bản không đủ thông tin để kết luận? Nhận định: {claim} | Đối chiếu với nội dung đã cho, nhận định dưới đây đúng, sai, hay chưa đủ căn cứ để đánh giá? Nhận định: {claim} |
| `probe_v1.P5.kiem_chung_noul` | Văn bản có xác nhận phát biểu sau không? Phát biểu: {claim} | Nội dung văn bản có cho thấy điều sau đây là đúng không? Phát biểu: {claim} |
