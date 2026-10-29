# Chỉ dẫn cho người ĐỌC LẠI độc lập (vai "Sonnet 2"): dịch đầy đủ phần B

Bạn **không phải** người dịch. Bạn soát bản dịch Anh → Việt của dữ liệu huấn luyện trắc nghiệm. Lời gọi cho bạn biết **các lô** cần soát. Mỗi lô 40 dòng.

## Được đọc / ghi

- **Chỉ đọc:**
  - `C:\Users\dungbt06\Fine-tune\results\train_v1\to_translate\HUONG_DAN_DICH_v2.md`;
  - `C:\Users\dungbt06\Fine-tune\results\train_v1\to_translate\bang_thuat_ngu_v2.json`;
  - `...\translated_sonnet\full\lo\lo_<NNN>_en.jsonl` (bản Anh);
  - `...\full\lo\lo_<NNN>_vi_s1.jsonl` (bản dịch cần soát).
- **Chỉ ghi** `lo_<NNN>_vi_s2.jsonl` và `lo_<NNN>_sua.jsonl` cho đúng các lô được giao (cùng thư mục `...\full\lo\`). Script phụ BẮT BUỘC để trong thư mục con RIÊNG của scratchpad tên `doclai_<lô đầu tiên>` (ví dụ `doclai_033`), không dùng tên file chung với agent khác. Script phụ để trong thư mục scratchpad.
- **Không đọc:** `results\train_v1\translated\`, file nhãn/target (`part_b_meta.jsonl`, `part_b_en.jsonl`).
- **Không sửa** file `_s1`.

## Soát từng lô

Đối chiếu **từng dòng, từng trường** (state, question, từng lựa chọn) với bản Anh:
- nghĩa có bị đổi, thêm, bớt không;
- quan hệ văn bản ↔ câu hỏi ↔ từng lựa chọn có giữ không: lựa chọn đúng/sai theo bản Anh phải vẫn đúng/sai theo bản Việt; không để hai lựa chọn thành cùng nghĩa;
- tên riêng (người, tác phẩm, sản phẩm giữ gốc; tên nước dùng tên Việt quen), số, chuỗi không rõ nghĩa (giữ nguyên), tiền tố (`Turn N:`, `text_A:`, `Q:`, `A:` … giữ nguyên; tiền tố khác theo `tien_to`), thứ tự và số lựa chọn;
- đúng bảng thuật ngữ (câu hỏi mẫu, nhãn, nhãn trích trong ngoặc kép);
- các quy tắc "Cách viết" của hướng dẫn: tiêu đề viết hoa kiểu Việt, chị/em và xưng hô nhất quán, thuật ngữ NLI nhất quán giữa text_A và text_B, không dịch nghĩa đen từ đa nghĩa;
- tiếng Việt tự nhiên: chỉ sửa khi thật sự vụng hoặc sai, không sửa theo sở thích.

Có lỗi thì sửa; không lỗi thì giữ nguyên.

## Ghi

Mỗi lô ghi 2 file (UTF-8, `ensure_ascii=False`, NFC, không ký tự ẩn; nên dùng script Python):

a) `lo_<NNN>_vi_s2.jsonl`: đủ dòng như `_s1`, mỗi dòng `{"id","state","question","options"}`, là bản **sau sửa**. Dòng không sửa thì chép nguyên.

b) `lo_<NNN>_sua.jsonl`: mỗi chỗ sửa một dòng:
```
{"id", "truong": "state" | "question" | "option:<chỉ số từ 0>", "truoc", "sau", "loai_loi", "ly_do"}
```
- `loai_loi` ∈ {doi_nghia, quan_he_lua_chon, ten_rieng_so, tien_to_dinh_dang, thu_tu_lua_chon, thuat_ngu, them_bot_noi_dung, tu_nhien_ngu_phap, khac}.
- Không sửa gì thì ghi file rỗng.

## Kiểm rồi mới sang lô sau

```
python C:\Users\dungbt06\Fine-tune\results\train_v1\to_translate\kiem_ban_dich.py <lô_en> <lô_vi_s2>
```
- Phải **ĐẠT**.
- Tự kiểm: mọi chỗ khác nhau giữa `_s1` và `_s2` đều có mục trong `_sua`, và ngược lại.
- Lô nào đã có `_vi_s2` và `_sua` hợp lệ (từ lần chạy trước) thì bỏ qua.

## Trả lời

Đúng **một dòng**:

```
xong <các lô>; dòng sửa <số>; mục sửa theo loại <loai:số,...>; kiểm ĐẠT
```

## Lưu ý v2.1 (kiểm giống hệt bản gốc)

Nếu bản dịch đã THÊM chữ không có trong bản gốc chỉ để tránh lỗi "còn nguyên văn" (ví dụ thêm "tên" trước một tên người, khi dòng gốc chỉ gồm tên/số/chuỗi lạ), hãy bỏ chữ thêm đó và trả về đúng chuỗi gốc (loai_loi them_bot_noi_dung). Script kiểm v2.1 chỉ cảnh báo cho trường hợp giữ nguyên này.
