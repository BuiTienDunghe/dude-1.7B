# Ý kiến thứ hai cho đề xuất sửa MỚI (Opus, độc lập)

Các đề xuất dưới đây do một lượt Opus kiểm tra cuối viết ra. Việc của bạn là cho ý kiến thứ hai để người làm dự án duyệt nhanh. Bạn không sửa file nào ngoài file ra của mình.

## Đầu vào
Mỗi dòng của `in_XX.jsonl` là một đề xuất:
- `truong`, `gia_tri_cu` → `gia_tri_de_xuat`, `ly_do`, `giai_thich`;
- `vi_hien_hanh`: dòng tiếng Việt HIỆN HÀNH (đã áp các sửa trước), `target` đếm từ 0;
- `en`: bản Anh (null với nguồn gốc Việt).

## Kết luận (một trong bốn)
- `DONG_Y`: đề xuất đúng, nên áp.
- `DONG_Y_NHUNG_CO_CHUOI_TOT_HON`: nên sửa, nhưng chuỗi khác tốt hơn. Ghi vào `chuoi_thay_the`, cùng phạm vi `gia_tri_cu`.
- `BAC_GIU_NGUYEN`: bản hiện hành đã ổn, hoặc đề xuất sai hay gây lỗi mới.
- `CAN_NGUOI_XEM`: phụ thuộc lựa chọn của người, ví dụ đổi nhãn, hay sửa phát biểu gốc của nguồn bản địa.

## Kiểm thêm
- Sau khi áp, đáp án `target` còn đúng và duy nhất không.
- Chuỗi đề xuất có lỗi chính tả không, có đúng NFC không.
- Câu Belebele dùng chung đoạn văn #1/#2: chuỗi sửa phải hợp với cả hai câu hỏi.
- Phát biểu (claim) của ViFactCheck/ViWikiFC là do người gán gốc viết: sửa lỗi gõ là chấp nhận được nếu không đổi nghĩa và không đổi nhãn.

## Đầu ra
Ghi `out_XX.jsonl` bằng công cụ Write (hoặc script Python nếu cần giữ đúng byte), mỗi đề xuất một dòng, đúng thứ tự:

```json
{"id": "...", "ket_luan": "...", "chuoi_thay_the": "", "ly_do": "tiếng Việt ≤ 2 câu", "gold_sau_sua": "DUNG_DUY_NHAT|KHONG_DUY_NHAT|SAI", "loi_moi": "", "do_tin_cay": 0.9}
```

Chỉ đọc file vào của bạn và file này. Đọc lại file ra để kiểm: JSON hợp lệ, đủ dòng. Cuối cùng trả lời ngắn: số đề xuất theo `ket_luan`, và liệt kê các đề xuất không phải `DONG_Y`.
