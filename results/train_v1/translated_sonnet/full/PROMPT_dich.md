# Chỉ dẫn cho người DỊCH (vai "Sonnet 1"): dịch đầy đủ phần B

Bạn dịch Anh → Việt dữ liệu huấn luyện trắc nghiệm. Lời gọi cho bạn biết **các lô** cần làm (ví dụ 001, 002, 003, 004). Mỗi lô 40 dòng.

## Được đọc / ghi

- **Chỉ đọc:**
  - `C:\Users\dungbt06\Fine-tune\results\train_v1\to_translate\HUONG_DAN_DICH_v2.md` (quy tắc; làm đúng mọi mục);
  - `C:\Users\dungbt06\Fine-tune\results\train_v1\to_translate\bang_thuat_ngu_v2.json` (bảng thuật ngữ bắt buộc: `thuat_ngu` và `tien_to`);
  - các file đầu vào `C:\Users\dungbt06\Fine-tune\results\train_v1\translated_sonnet\full\lo\lo_<NNN>_en.jsonl`.
- **Chỉ ghi** `C:\Users\dungbt06\Fine-tune\results\train_v1\translated_sonnet\full\lo\lo_<NNN>_vi_s1.jsonl` cho đúng các lô được giao. Script phụ BẮT BUỘC để trong thư mục con RIÊNG của scratchpad tên `dich_<lô đầu tiên>` (ví dụ `dich_033`), không dùng tên file chung với agent khác. Script phụ (nếu cần) để trong thư mục scratchpad của bạn.
- **Không đọc:** thư mục `results\train_v1\translated\`, các file nhãn/target (`part_b_meta.jsonl`, `part_b_en.jsonl`), các file `_s2` / `_sua` của người đọc lại.
- **Không sửa** file nào khác.

## Làm từng lô, lần lượt

1. Đọc lô `_en`. Dịch state, question và từng lựa chọn.
   - Chuỗi trùng **nguyên văn** một khóa `thuat_ngu` thì dùng đúng `vi`.
   - Nhãn trích trong ngoặc kép bên trong câu hỏi dùng đúng bản dịch của nhãn đó.
   - Tiền tố đầu dòng: theo `tien_to` và mục "Tiền tố và định dạng" của hướng dẫn. `Turn N:`, `text_A:`, `Q:`, `A:` … luôn giữ nguyên.
   - `options` rỗng thì trả `[]`; question rỗng thì trả `""`.
   - Lựa chọn đúng vẫn đúng, lựa chọn sai vẫn sai.
2. Ghi `lo_<NNN>_vi_s1.jsonl`:
   - đủ 40 dòng (hoặc đúng số dòng của lô cuối);
   - mỗi dòng `{"id","state","question","options"}`, cùng id và thứ tự với đầu vào;
   - UTF-8, `ensure_ascii=False`, chuẩn hóa NFC, không ký tự ẩn.
   - Nên ghi bằng một script Python nhỏ để tránh lỗi JSON. Nên dùng chính bảng thuật ngữ trong script, để chuỗi trùng khóa lấy đúng giá trị.
3. Kiểm:
   ```
   python C:\Users\dungbt06\Fine-tune\results\train_v1\to_translate\kiem_ban_dich.py <lô_en> <lô_vi_s1>
   ```
   - Phải **ĐẠT**; có LỖI thì sửa rồi kiểm lại.
   - Xem từng cảnh báo: cảnh báo tiền tố hay mất số thì sửa nếu đúng là lỗi của bạn. Cảnh báo do lỗi gõ của bản gốc (ví dụ "I'11") thì giữ.
4. Sang lô tiếp theo. Nếu một lô đã có file `_vi_s1` qua kiểm ĐẠT (từ lần chạy trước) thì bỏ qua lô đó.

## Trả lời

Đúng **một dòng**:

```
xong <các lô>; lỗi 0; cảnh báo <số>; ghi chú <ngắn nhất có thể, hoặc "không">
```

## Lưu ý v2.1 (kiểm giống hệt bản gốc)

Dòng/lựa chọn chỉ gồm tên riêng, số hoặc chuỗi lạ thì GIỮ NGUYÊN; script kiểm chỉ cảnh báo (không còn báo lỗi) cho trường hợp này. KHÔNG thêm chữ (ví dụ "tên") chỉ để qua kiểm.
