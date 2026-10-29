# Báo cáo sáng 14/10

Cập nhật lần cuối: 06:15 (sẽ cập nhật khi có kết quả mới).

Trong đêm: không train, không chấm model nào trên test, không sửa file benchmark. dev 1.1 và test 1.1 vẫn đúng hash. Chưa tốn đồng nào của mức dùng thêm.

## 1. Đã xong trong đêm

### Soát dịch và đáp án toàn bộ bench 1.1 (dev + test, 3.055 câu)

Dùng Opus quét lại toàn bộ, thêm một lượt Opus soi lại các câu bị gắn cờ.

- **Kết quả theo câu:**

  | Kết luận | Số câu |
  |---|---|
  | OK | 2.466 |
  | Dịch chưa chuẩn | 185 |
  | Dịch sai | 184 |
  | Lỗi chính tả | 79 |
  | Mơ hồ | 88 |
  | Nhãn gốc chắc sai | 13 |
  | Nhãn gốc có thể sai | 39 |

- **487 đề xuất sửa** đang chờ bạn duyệt ở trang "Chờ duyệt" của công cụ (http://127.0.0.1:8765). Trong đó:
  - 483 đề xuất theo câu (211 ở test, 38 đề xuất đổi đáp án);
  - 4 đề xuất mẫu câu hỏi.
- **Báo cáo:** `results/translation_audit/REPORT_translation_audit.md`. Danh sách câu mơ hồ: `merged/mo_ho.csv`.

### Khảo sát benchmark bổ sung (Fable 5.1)

- **Đề xuất của Fable:** `results/bench_candidates/FABLE_proposal.md`.
- **Hai bộ giấy phép mở đã chạy thử:** MLQA vi và Global-PIQA vi. Báo cáo ở `REPORT_pilot.md` và `REPORT_pilot2.md`.

### Khu tạm cho bản 1.2 (chưa đưa vào benchmark)

Báo cáo: `results/bench_candidates/stage_v1_2/REPORT_stage_v1_2.md`.

- **MLQA test:** 197 câu, mỗi câu có bản EN, VI có dấu và VI không dấu.
  - Đáp án là span do người gán của MLQA; 3 lựa chọn nhiễu do Opus chọn, một lượt Opus khác kiểm.
  - Điều phối đã kiểm độc lập: đạt.
  - Mẹo đoán theo vị trí câu chỉ còn 36–38%; đoán đều là 25%.
- **Global-PIQA lô 2:** 50 cặp. 6 câu chờ quyết nhãn; 17 đề xuất sửa chưa áp.
- **File đọc kiểm:** `results/review/pilot2_review.html` (20 câu MLQA, 20 câu Global-PIQA).

## 2. Đã xong lúc khoảng 06:30 (dừng ở đây; Fable dừng ở 80%)

- **Ý kiến thứ hai của Opus cho 487 đề xuất** (`results/translation_audit/second_opinion/REPORT_second_opinion.md`):

  | Kết luận | Số đề xuất |
  |---|---|
  | Đồng ý | 460 |
  | Đồng ý nhưng có chuỗi tốt hơn | 22 |
  | Cần người xem | 3 |
  | Bác | 2 |

  - Có 26 câu mang từ 2 đề xuất trở lên. Các đề xuất cùng câu nên duyệt cùng lúc.
- **Ý kiến thứ hai của Fable cho 141 câu nghi sai nhãn hoặc mơ hồ** (`results/translation_audit/label_second/REPORT_label_second.md`):

  | Kết luận của Fable | Số câu |
  |---|---|
  | Nhãn sai chắc | 25 (gồm cả 13 câu lượt trước đã cho là chắc sai) |
  | Nhãn có thể sai | 30 |
  | Câu hỏng, nên loại | 14 |
  | Mơ hồ | 50 |
  | Nhãn đúng | 22 |

  - Ở test, Fable đề nghị đổi nhãn hoặc loại 51 câu.
- **Soát 197 câu MLQA test:**
  - 150 câu OK;
  - 47 đề xuất sửa, chưa áp (`stage_v1_2/mlqa_test_fixes.jsonl`);
  - 10 câu mơ hồ hoặc nghi sai nhãn.

  File đọc kiểm 30 câu: `results/review/mlqa_test_review.html`.

## 2b. Duyệt từ xa

Trang duyệt riêng tư trên claude.ai: https://claude.ai/artifact/3uPXD6zscihQ6bonrJmKkT (chỉ tài khoản của bạn mở được).
- **Nội dung:** 134 đề xuất cần xem và 141 câu nhãn, cộng tab duyệt hàng loạt cho các đề xuất mà Opus đồng ý và không có cờ.
- **Chép về máy:** quyết định chép về công cụ qua `results/remote_review/sync_to_tool.py`, đã thử trên bản sao.

## 3. Chờ bạn quyết

1. Duyệt 487 đề xuất ở trang "Chờ duyệt" và 63 chuỗi mẫu đã soát OK ở trang "Mẫu và nhãn".
2. 13 câu nhãn gốc chắc sai: đổi nhãn hay loại khỏi bench? (danh sách trong `REPORT_translation_audit.md`)
3. Có làm lớp nghiên cứu giấy phép phi thương mại (XNLI, RTP-LX, SEAHORSE) không?
4. Có nhận nội dung độc hại của RTP-LX cho loại câu chấm điểm không?
5. Có chuẩn hóa Unicode NFC không? Hiện có các dòng VI chưa chuẩn:

   | Nguồn | dev | test |
   |---|---|---|
   | MASSIVE | 71/300 | 76/300 |
   | ViFactCheck | 4/163 | 16/300 |
   | ViWikiFC | 0 | 1/300 |

6. Có đưa MLQA (197 câu, có cờ "nhiễu do LLM chọn") và/hoặc Global-PIQA lô 2 (khoảng 44 câu sau khi quyết 6 nhãn) vào bản 1.2 không?
   - Với Global-PIQA: phía EN chưa có đoạn văn (state rỗng), cần chọn cách xử lý.
7. Đọc `results/review/pilot2_review.html`.

## 4. Kiểm tra cuối (chiều 14/10)

- Theo ủy quyền của bạn, các câu chưa bấm được quyết theo agent:
  - đề xuất theo ý kiến thứ hai của Opus;
  - nhãn theo đề nghị của Fable.
- Opus đọc lại toàn bộ 3.055 cặp sau khi áp: 617 cặp bị đổi và 2.438 cặp không đổi.
  - Không cặp nào sai đáp án sau thay đổi.
  - 63/65 câu bị loại được xác nhận là loại hợp lý.
- Điều chỉnh theo hướng an toàn (đều đã loại):
  - 2 câu đổi nhãn chuyển thành loại: `vifactcheck:test:3461`, `viwikifc:dev:uit_29`.
  - Loại thêm 2 câu mơ hồ: `sib200:238`, `massive:test:5823`.
  - 2 câu Opus muốn giữ lại nhưng vẫn loại: `vifactcheck:test:3254`, `viwikifc:test:uit_169`.
- Đã áp vào công cụ trên máy:
  - 628 quyết định cùng 2 câu loại thêm;
  - nhãn ở test ghi tại `results/human_review/label_decisions_remote.jsonl`.
- **190 đề xuất sửa chữ MỚI** (93 ở test) đang chờ duyệt.
  - Opus xem lần hai: đồng ý 189, có chuỗi tốt hơn 1.
  - Trên trang có 184 đề xuất duyệt hàng loạt được; 6 đề xuất cần bạn xem, trong đó 5 sửa lại câu bạn đã tự quyết.
- Lỗi chỉ do Unicode tổ hợp: 143 chỗ, để bước chuẩn hóa NFC khi dựng 1.2 xử lý.
- Lỗi gõ trong phát biểu gốc của ViFactCheck/ViWikiFC: chưa sửa hết, vì các agent áp quy tắc không đều. Cần một lượt riêng nếu bạn muốn sửa.
