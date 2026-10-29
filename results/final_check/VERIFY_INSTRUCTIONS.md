# Đọc lại bench 1.1 SAU KHI áp thay đổi (Opus): kiểm tra cuối

Người làm dự án đã đồng ý các thay đổi: sửa dịch hoặc chính tả, đổi nhãn, loại câu. Việc của bạn là đọc lại kết quả **sau khi áp** để chắc không có gì sai: không có lỗi mới, đáp án đúng và duy nhất, bản Việt khớp bản Anh. Bạn KHÔNG sửa file nào ngoài file ra của mình. Bạn KHÔNG ghi chuỗi tiếng Việt nào vào benchmark; chuỗi bạn đề nghị chỉ nằm trong file ra để người duyệt.

## Đầu vào
Mỗi dòng của `in_XX.jsonl` là một cặp câu, với các trường sau.
- `loai_thay_doi`: `SUA` (đã đổi chữ hoặc nhãn) hoặc `LOAI_KHOI_BENCH` (cặp bị loại).
- `en`: bản Anh gốc, có ở khối song song belebele/sib200/massive/xcopa; `null` với nguồn gốc Việt vifactcheck/viwikifc.
- `vi_truoc`, `vi_sau`: bản Việt trước và sau. `target` là chỉ số đáp án, đếm từ 0. `vi_sau` là `null` nếu cặp bị loại.
- `de_xuat_da_ap`: các sửa chữ đã áp (`cu` → `chuoi_ap`), kèm giải thích của agent và ý kiến Opus lượt 2.
- `quyet_dinh_nhan`, `y_kien_nhan`: quyết định nhãn (giữ / đổi / loại), kèm ý kiến lượt 1 và Fable.
- Một đoạn văn Belebele dùng chung cho #1 và #2. Sửa đoạn văn ở một câu thì câu kia cũng đổi. Hãy kiểm cả câu "chỉ đổi vì dùng chung đoạn văn".

## Kiểm gì (với `SUA`)
1. **Khớp nghĩa:** `vi_sau` khớp nghĩa với `en` (nếu có). Các chỗ đã sửa đúng và tự nhiên. Không có lỗi mới: chính tả, sót chữ, lặp chữ, dấu câu, NFC, chữ tiếng Anh sót lại.
2. **Đáp án:** `vi_sau.options[vi_sau.target]` là đáp án đúng và DUY NHẤT với `vi_sau.state` + `vi_sau.question`. Với khối song song, đáp án phải khớp đáp án của `en`.
3. **Đổi nhãn:** nếu nhãn bị đổi, kiểm đáp án mới có đúng không, dựa vào văn bản.
4. **Chỗ chưa sửa:** nếu còn lỗi dịch hay lỗi chính tả RÕ RÀNG chưa sửa trong `vi_sau`, ghi lại; chỉ lỗi làm sai nghĩa hoặc sai chính tả, không ghi chuyện văn phong.

## Kiểm gì (với `LOAI_KHOI_BENCH`)
Câu này có đáng loại không? Loại là hợp lý khi câu mơ hồ, có nhiều đáp án, hoặc câu hỏng. Nếu bạn thấy có một đáp án đúng duy nhất rõ ràng (nhãn gốc hoặc nhãn khác), ghi `GIU_LAI` kèm đáp án.

## Đầu ra
Ghi `out_XX.jsonl` (cùng số với file vào) bằng công cụ Write. Mỗi cặp một dòng JSON, đủ mọi cặp theo đúng thứ tự:

```json
{"split": "dev|test", "pair_id": "...", "ket_luan": "OK|CAN_SUA_THEM|DAP_AN_SAI|MO_HO|LOAI_HOP_LY|GIU_LAI",
 "van_de": "tiếng Việt, ≤ 2 câu; rỗng nếu OK",
 "sua": [{"truong": "state|question|option:<k>", "gia_tri_cu": "chuỗi con xuất hiện ĐÚNG 1 lần trong vi_sau", "gia_tri_de_xuat": "...", "ly_do": "loi_dich|chinh_ta"}],
 "dap_an_dung": null, "do_tin_cay": 0.9}
```

- `OK`: không còn vấn đề.
- `CAN_SUA_THEM`: còn lỗi chữ; ghi `sua`, mỗi mục là một chỗ, `gia_tri_cu` ngắn nhất đủ để duy nhất.
- `DAP_AN_SAI`: đáp án sau thay đổi sai; ghi `dap_an_dung`.
- `MO_HO`: không có một đáp án duy nhất sau thay đổi.
- `LOAI_HOP_LY` / `GIU_LAI`: dùng cho cặp bị loại; với `GIU_LAI` thì ghi `dap_an_dung`.
- `sua` rỗng `[]` nếu không cần.

Chỉ đọc hai file: file vào của bạn và file này. Ghi xong thì đọc lại file ra, kiểm mỗi dòng là JSON hợp lệ, đủ số cặp, và mỗi `gia_tri_cu` xuất hiện đúng 1 lần trong trường tương ứng của `vi_sau`. Cuối cùng trả lời ngắn: số cặp theo `ket_luan`, và liệt kê các cặp không OK.
