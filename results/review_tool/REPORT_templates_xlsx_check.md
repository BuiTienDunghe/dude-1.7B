# Soát file `templates_review_fully_checked.xlsx` (Sonnet 5.5, 15/10/2026)

Tác giả: agent Sonnet 5.5. Agent không có quyền ghi file nên phiên điều phối lưu lại nội dung báo cáo.

- **File kiểm:** bản người dùng tải lên (`.claude/uploads/…/2494a4f4-templates_review_fully_checked.xlsx`).
- **Bản gốc để đối chiếu:** `export/templates_review.xlsx`.
- **Phạm vi đếm tác động:** chỉ dev (`dev_v1_1.jsonl`).

## Tóm tắt

1. **Toàn vẹn file.** File có 78 dòng, đủ khóa, đúng thứ tự: 65 OK, 13 SUA, không dòng trống.
   - 13 chuỗi sửa đều không rỗng, đúng NFC, giữ nguyên `{claim}`.
   - Cột chỉ đọc "trạng thái hiện tại" bị ghi đè ở 69/78 ô, bằng kết quả dự kiến chứ không phải trạng thái thật. Các cột "khóa", "tiếng Việt hiện tại" và "gốc" không đổi.
   - File được lưu bằng openpyxl (script), không phải bằng Excel.
2. **Phát hiện nặng nhất.** 6 trong 13 dòng SUA đảo ngược chuỗi đã duyệt ngày 13/10: factcheck_noul.question, general_quirky, transport_query, transport_traffic, options.nei, options.supported. Script nhập sẽ ghi đè các chuỗi này mà không cảnh báo.
3. **Chất lượng 13 dòng SUA.** Nên áp 6, nên giữ chuỗi hiện tại 4, nên dùng chuỗi khác 3.
4. **Tác động nếu áp cả 13 dòng.** 720/1.463 câu dev có dấu (49,2%) bị đổi, trong đó có 100% câu ViFactCheck/ViWikiFC (463/463).
5. **Chạy thử nhập (dry run).** Kế hoạch: duyệt 58, sửa 13, bỏ qua 7, lỗi 0. Không ghi gì (hash trước và sau giống nhau). Script không đọc cột trạng thái nên không bị đánh lừa.

## Bảng 13 dòng SUA

Ký hiệu cột "Kết luận":
- **ÁP**: áp chuỗi người điền đề xuất.
- **GIỮ**: giữ chuỗi hiện tại.
- **KHÁC**: dùng một chuỗi khác (ghi ở cột "Chuỗi nên dùng").

| # | Khóa | Hiện tại | Người điền đề xuất | Kết luận | Chuỗi nên dùng |
|---|---|---|---|---|---|
| 1 | factcheck_noul.question | Văn bản có xác nhận phát biểu sau không? Phát biểu: {claim} | …có ủng hộ phát biểu… | GIỮ | giữ nguyên |
| 2 | audio_volume_other | các điều chỉnh âm lượng khác | các thao tác điều chỉnh âm lượng khác | ÁP | như đề xuất |
| 3 | email_addcontact | thêm liên hệ email | thêm một liên hệ vào danh bạ | ÁP | như đề xuất |
| 4 | general_quirky | trò chuyện phiếm hoặc đáp một câu nói kỳ quặc, lạc đề | trò chuyện xã giao hoặc đáp lại lời nói lạc đề hay khác thường | GIỮ | giữ nguyên ("xã giao" dễ lẫn với "chào hỏi") |
| 5 | lists_query | xem nội dung danh sách | đọc danh sách thành tiếng | KHÁC | xem hoặc đọc nội dung danh sách |
| 6 | music_query | cho biết bài nhạc đang phát | cho biết nhạc gì đang phát | ÁP | như đề xuất |
| 7 | news_query | đọc tin tức | cung cấp tin tức | KHÁC | cho biết tin tức |
| 8 | social_post | đăng lên mạng xã hội hoặc gửi phản ánh | đăng bài lên mạng xã hội hoặc gửi khiếu nại | ÁP | như đề xuất |
| 9 | takeaway_query | hỏi về đơn đồ ăn mang về hoặc việc giao đồ ăn | hỏi về đơn đồ ăn mang đi hoặc dịch vụ giao đồ ăn | KHÁC | hỏi về đơn đồ ăn mang về hoặc dịch vụ giao đồ ăn |
| 10 | transport_query | tra cứu lộ trình hoặc giờ tàu xe | tra cứu tuyến đường hoặc lịch trình | GIỮ | giữ nguyên ("lịch trình" lẫn với "xem lịch") |
| 11 | transport_traffic | cho biết tình hình kẹt xe trên đường | cho biết tình hình giao thông trên đường | ÁP | như đề xuất (đảo ngược chuỗi đã duyệt) |
| 12 | options.nei | Văn bản không đủ thông tin | Văn bản không đủ thông tin để kết luận | ÁP | như đề xuất (đảo ngược chuỗi đã duyệt) |
| 13 | options.supported | Được văn bản xác nhận | Được văn bản ủng hộ | GIỮ | giữ nguyên |

Lý do chính cho từng nhóm:
- **Bộ ba "xác nhận / ủng hộ"** (dòng 1, dòng 13, và factcheck_choice.question):
  - Người điền đổi 2 chuỗi sang "ủng hộ" nhưng để câu hỏi 3 lựa chọn là OK với "xác nhận". Nếu áp, câu hỏi và lựa chọn trong cùng một câu sẽ dùng hai thuật ngữ khác nhau.
  - "Ủng hộ" nghiêng về nghĩa "tán thành" và làm mờ ranh giới với "không đủ thông tin".
  - Cặp "xác nhận / bác bỏ" là cặp đối lập tự nhiên trong kiểm chứng.
  - Quy tắc: ba chuỗi cùng đổi hoặc cùng giữ.
- **lists_query:** câu thật trong dev là tra cứu ("what are my lists"), không phải đọc to.
- **news_query:** dùng "cho biết …" để nhất quán với các ý định thời tiết, ngày giờ và giao thông.
- **takeaway_query:** takeaway_order vẫn dùng "mang về", nên cặp họ hàng này không nên lệch từ.
- **transport_query:** câu thật trong dev là "next train", "train time", nên "giờ tàu xe" đúng hơn.

## Dòng OK đáng xem lại

- **factcheck_choice.question:** ghi chú bảo nên đổi sang "ủng hộ" nhưng quyết định là OK, tự mâu thuẫn. Nếu giữ "xác nhận" thì OK là đúng, nhưng cần xóa ghi chú.
- **qa_factoid:** chính ghi chú nói "hơi rộng". Có thể đổi thành "trả lời một câu hỏi về thông tin thực tế"; ưu tiên thấp.
- **transport_ticket** ("đặt vé tàu xe"): người điền lập luận ở dòng transport_query mâu thuẫn với chuỗi này. Chỉ để cân nhắc.
- **Ưu tiên thấp:** qa_currency (nên dùng "tỷ giá hối đoái"), datetime_convert (nên dùng "quy đổi giờ"), qa_maths (chuỗi hiện tại khá chung).

## Tác động trên dev (câu có dấu)

| Chuỗi | Số câu chứa chuỗi | Là đáp án đúng |
|---|---|---|
| factcheck_noul.question | 207 (mọi câu Có/Không) | — |
| options.supported | 256 (mọi câu 3 lựa chọn) | 65 |
| options.nei | 256 | 82 |
| Mỗi ý định MASSIVE bị đổi | 36–61 | 0–5 |

- MASSIVE: 257/300 câu chứa ít nhất một chuỗi bị đổi.
- SIB-200, XCOPA và Belebele không bị ảnh hưởng.
- Nếu áp theo khuyến nghị ở trên, khối kiểm chứng chỉ còn chịu thay đổi của `nei`, cộng 9 ý định MASSIVE.

## Chạy thử nhập (dry run)

- Lệnh: `import_templates_review.py --dry_run`, thoát mã 0.
- Kế hoạch: duyệt 58, sửa 13, bỏ qua 7 (các dòng đã duyệt và vẫn OK), trống 0, lỗi 0.
- Script chỉ đọc 4 cột: khóa, tiếng Việt sửa, quyết định, ghi chú. Trạng thái lấy từ edits_log.
- Hạn chế: chế độ dry run thoát trước bước kiểm tra công cụ đang chạy. Script cũng không cảnh báo khi "sua" ghi đè chuỗi đã duyệt.

## Khuyến nghị thao tác

Không nhập nguyên file. Trước khi nhập, chỉnh file như sau:
- Dòng 1, 4, 10, 13: đổi về OK và xóa chuỗi sửa.
- Dòng 5, 7, 9: thay chuỗi sửa bằng chuỗi ở cột "Chuỗi nên dùng".
- Giữ nguyên 6 dòng còn lại (2, 3, 6, 8, 11, 12).

Riêng dòng 11 và 12 đảo ngược chuỗi đã duyệt, cần người làm dự án xác nhận. Nên thêm vào script nhập một cảnh báo khi "sua" ghi đè chuỗi đã duyệt.
