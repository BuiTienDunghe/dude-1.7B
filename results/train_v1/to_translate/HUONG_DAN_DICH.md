# Hướng dẫn dịch: dữ liệu train_v1, phần B (Anh → Việt)

File cần dịch: `part_b_to_translate.jsonl` (cùng thư mục). Mỗi dòng là một câu hỏi trắc nghiệm dùng để **huấn luyện** model ra quyết định tiếng Việt. Dữ liệu chỉ dùng nội bộ.

## Định dạng

Mỗi dòng vào:

```json
{"id": "...", "state": "văn bản", "question": "câu hỏi", "options": ["lựa chọn 1", "lựa chọn 2", ...]}
```

Trả về **một file jsonl**, mỗi dòng đúng bốn trường đó, cùng `id`, đã dịch sang tiếng Việt:

```json
{"id": "<giữ nguyên>", "state": "<bản dịch>", "question": "<bản dịch>", "options": ["<bản dịch lựa chọn 1>", "<bản dịch lựa chọn 2>", ...]}
```

- Giữ nguyên `id`, không thêm hay bỏ dòng. Thứ tự dòng không bắt buộc, nhưng nên giữ.
- `options` rỗng (`[]`) nghĩa là câu hỏi Có/Không: lựa chọn cố định do dự án thêm sau. Trả về `[]`.
- Mã hóa UTF-8, một đối tượng JSON trên một dòng.

## Quy tắc dịch

1. **Giữ nguyên số lựa chọn và thứ tự lựa chọn.** Lựa chọn thứ k của bản dịch phải là bản dịch của lựa chọn thứ k của bản gốc. Không gộp, không tách, không đảo.
2. **Không đổi nghĩa.** Đáp án đúng được xác định theo bản gốc; bản dịch phải giữ đúng quan hệ giữa văn bản, câu hỏi và từng lựa chọn (lựa chọn sai vẫn phải sai, lựa chọn đúng vẫn phải đúng).
3. **Không thêm, không bớt thông tin.** Không giải thích, không chú thích, không trả lời câu hỏi.
4. **Tên riêng và số giữ nguyên** (tên người, địa danh, tổ chức, sản phẩm, mã số, ngày tháng, đơn vị đo). Có thể chuyển dấu thập phân theo kiểu Việt (3.5 → 3,5) nếu nhất quán.
5. **Tiếng Việt có dấu chuẩn, Unicode NFC.** Không viết không dấu, không trộn kiểu gõ dấu. Khoảng 65 dòng gốc tiếng Anh chưa ở dạng NFC (ký tự tổ hợp trong tên riêng); bản dịch vẫn phải chuẩn hóa NFC toàn bộ.
6. **Trả về đúng định dạng và id** như trên.

## Lưu ý theo loại dữ liệu

- **Câu hỏi mẫu lặp lại** (ví dụ "Which supplied option best answers the question?"): dịch **nhất quán**, cùng một câu tiếng Anh thì cùng một câu tiếng Việt ở mọi dòng. Danh sách chuỗi lặp từ 5 lần trở lên ở `chuoi_lap_lai.json`.
- **Nhãn lựa chọn cố định** (ví dụ `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`, `True` / `False`, `entailment` / `neutral` / `contradiction`, `Yes` / `No` / `Irrelevant` / `Clarification needed`): dịch nhất quán như trên.
- **Nhãn dạng mã** (ví dụ `card_arrival`, `changeseatassignment`, `orderburgerintent`): dịch thành cụm từ tiếng Việt ngắn mô tả đúng ý định (ví dụ `card_arrival` → "hỏi khi nào thẻ được giao tới"). Các nhãn khác nhau phải ra bản dịch khác nhau.
- **Hội thoại** có tiền tố `text_A:`, `text_B:`, `Turn 0:`, `M:`, `W:`, `Passage A:`, `###`: giữ nguyên các tiền tố và ký hiệu định dạng, chỉ dịch phần lời.
- **Nhân quả (e-CARE, HeadlineCause).**
  - e-CARE: state gồm một câu sự việc và câu hỏi đi kèm, như "What was the cause of this?" hay "What happened as a result?". Dịch cả hai; giữ đúng chiều nguyên nhân / kết quả.
  - HeadlineCause: state là hai tiêu đề tin có tiền tố `A:` và `B:`, giữ nguyên tiền tố. Nhãn như "first headline caused the second" phải giữ đúng thứ tự "thứ nhất / thứ hai", vì đổi chiều là đổi đáp án.
- Chỗ trống `_` trong câu điền khuyết: giữ nguyên ký hiệu `_` ở vị trí tương ứng.
- Câu sai chính tả hoặc lủng củng trong bản gốc: dịch theo nghĩa rõ nhất, không sửa nội dung.

## Kiểm trước khi gửi

Chạy (chỉ cần Python 3.8+, không cần thư viện ngoài):

```bash
python kiem_ban_dich.py part_b_to_translate.jsonl <bản_dịch>.jsonl --report ket_qua_kiem.json
```

Script kiểm: đủ id, không thừa, không lặp; đủ trường và đúng kiểu; số lựa chọn khớp; không chuỗi rỗng; NFC; không còn nguyên văn tiếng Anh; lựa chọn trong một dòng không trùng nhau. Script cũng cảnh báo (không chặn): chuỗi lặp dịch không nhất quán, độ dài lệch bất thường, số bị mất. Kết luận phải là `ĐẠT`.

Sau khi nhận bản dịch, dự án sẽ kiểm lại bằng chính script này, khử rò rỉ lần nữa phía tiếng Việt, rồi mới dùng để train.
