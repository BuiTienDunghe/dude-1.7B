# Hướng dẫn đọc lỗi (đọc mù)

Bạn đọc một lô câu hỏi trắc nghiệm tiếng Việt. Mỗi câu có: văn bản, câu hỏi, các lựa chọn, **đáp án đúng** (nhãn của bộ dữ liệu) và **đáp án một model đã chọn**. Bạn không biết đó là model nào, và không cần đoán.

## Quy tắc

- Chỉ mở **đúng hai file**: file hướng dẫn này và file lô được giao. **Không mở file nào khác** trong thư mục, đặc biệt không mở `khoa.json` hay lô khác.
- Đọc từng câu thật kỹ. Tự xét xem văn bản có đủ căn cứ cho đáp án đúng không.
- Không chạy code, không tìm trên mạng.

## Phân loại mỗi câu

Chọn đúng **một** mã `loai` cho mỗi câu.

**Nếu model chọn ĐÚNG đáp án:**

| Mã | Khi nào dùng |
|---|---|
| `dung_ly_do_dung` | Phải đọc và hiểu văn bản mới chọn được; không có lối tắt rõ ràng |
| `dung_meo_lua_chon` | Đoán được chỉ từ các lựa chọn hoặc câu hỏi, không cần văn bản. Ví dụ: lựa chọn đúng dài hơn hay cụ thể hơn hẳn, lặp lại chữ của câu hỏi, các lựa chọn khác vô lý |
| `dung_kien_thuc_san_co` | Trả lời được bằng kiến thức phổ thông, không cần văn bản |
| `dung_khop_be_mat` | Lựa chọn đúng trùng từ khóa với văn bản; chỉ khớp chữ cũng đủ chọn đúng |
| `dung_nhan_dang_ngo` | Nhãn "đúng" của bộ dữ liệu có vẻ sai hoặc mơ hồ; model trùng nhãn nhưng nhãn đáng ngờ |

**Nếu model chọn SAI:**

| Mã | Khi nào dùng |
|---|---|
| `sai_khong_doc_van_ban` | Lựa chọn của model không dựa trên văn bản, ví dụ theo kiến thức ngoài hoặc theo lựa chọn "nghe hợp lý" |
| `sai_hieu_sai_phu_dinh` | Hiểu sai phủ định hoặc câu hỏi kiểu "không đúng", "ngoại trừ" |
| `sai_nham_nhan_kiem_chung` | Kiểm chứng: nhầm giữa xác nhận / bác bỏ / không đủ thông tin. Ví dụ: chọn "bác bỏ" khi văn bản chỉ không nói tới |
| `sai_nham_nhan_gan_nghia` | Chủ đề hoặc ý định gần nghĩa nhau, ranh giới giữa hai nhãn mờ |
| `sai_suy_luan` | Có dựa vào văn bản nhưng suy luận sai: số liệu, quan hệ nhân quả, ai làm gì |
| `sai_mo_ho_hoac_nhan_nghi_sai` | Câu mơ hồ, hoặc nhãn bộ dữ liệu có vẻ sai; lựa chọn của model bảo vệ được |
| `sai_khac` | Không thuộc các loại trên; ghi rõ lý do trong `ghi_chu` |

## Kết quả

Ghi file JSONL (mỗi câu một dòng, UTF-8) vào **đúng đường dẫn được giao**. Mỗi dòng:

```json
{"ma": "X001", "loai": "<một mã ở trên>", "van_ban_du_can_cu": "co|khong|mot_phan", "tu_tin": "cao|vua|thap", "ghi_chu": "<tiếng Việt, tối đa 40 từ>"}
```

Ý nghĩa các trường:
- `van_ban_du_can_cu`: văn bản có đủ căn cứ để đi đến đáp án đúng không.
- `tu_tin`: mức chắc chắn của bạn về phân loại.

Đủ số dòng bằng số câu trong lô. Xong thì trả lời ngắn: số dòng đã ghi, và đếm theo `loai`.
