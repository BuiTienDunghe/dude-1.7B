# Đọc mù mẫu 2b, vòng 1 (điều phối, 28/10/2026)

**Cách làm:**
- Mẫu `results/train_2b/mau_doc.jsonl`, 310 dòng.
- 3 agent Opus, chia xen kẽ theo số dòng mod 3. Che `nhan` và mọi cột `nhan_*`; cấm mở thư mục `train_2b`.
- Quy ước ViWikiFC/FEVER: đúng một phần cộng chi tiết không kiểm được thì là NEI.
- Script chấm: `cham_doc2b.py` trong scratchpad của điều phối.
- Luật chốt trước (brief mục 7c): **lỗi rõ** = chọn khác nhãn, độ chắc "cao", không gắn cờ. Mỗi kiểu 30 dòng: ≤ 2 lỗi rõ thì đạt; 3–4 thì xem; ≥ 5 thì sửa. Riêng 2h (20 dòng): ≤ 1 thì đạt.

## Kết quả

| Kiểu | n | Trùng nhãn | Lỗi rõ | Kết luận | Cờ đáng chú ý |
|---|---|---|---|---|---|
| KC S-nguyên | 30 | 30 | 0 | đạt | loi_cau 1 |
| KC S-lược | 30 | 30 | 0 | đạt | **loi_cau 6** (câu cụt) |
| KC R-số | 30 | 29 | 1 | đạt | — |
| KC R-tên | 30 | 29 | 1 | đạt | loi_cau 2 |
| KC N-cùng bài | 30 | 30 | 0 | đạt | — |
| KC N-lạc đề | 30 | 30 | 0 | đạt | lac_khung 1, loi_cau 1 |
| 2d vitaminc/nli_fever khung bench (dịch) | 30 | 18 | 1 | đạt | mo_ho 9 |
| 2f câu hỏi lên ô câu hỏi | 30 | 28 | 0 | đạt | mo_ho 2 |
| 2g quail đổi chữ NEI | 30 | 24 | 0 | đạt | mo_ho 6, loi_cau 3 |
| 2h thêm lựa chọn NEI | 20 | 20 | 0 | đạt | — |
| 2h thay lựa chọn → NEI | 20 | 20 | 0 | đạt | lac_khung 20 (đúng thiết kế) |

Đã kiểm thêm: cả 51 dòng 2h thay lựa chọn đều có chuỗi NEI dev trong lựa chọn, và chuỗi đó là đáp án.

## Lỗi cụ thể (sửa trong lần dựng KC v2)

1. **Lược tạo câu cụt:** 8/76 dòng có lược bị gắn `loi_cau`.
   - Bỏ mệnh đề: "…như Dương Hộ, Đỗ Dự và Trương Hoa."
   - Bỏ ngoặc: cắt mất tên hợp chất "Bis(…)", hoặc để lại câu cụt "nay thuộc."
2. **Mảnh trích dẫn thư mục bị dùng làm nhận định:** "Joel Krieger, ed. Oxford University Press Inc. 2001.", "Editorial Alfaguara…".
3. **R-tên:**
   - **M146:** đổi "Việt Nam" → "Gia Lâm" thành câu vẫn đúng ("sản xuất bởi Gia Lâm"). Giá trị thay lấy từ chính phần câu gốc đã bị lược.
   - **M138:** thay bằng tên năm can chi ("các đại thần Mậu Dần").
   - **M297:** câu vỡ.
4. **R-số, M082:** đổi niên hiệu "Chiêu Thống thứ 2 (1788)" → "thứ 3 (1788)". Người đọc chọn NEI. Chấp nhận được: câu vẫn mâu thuẫn với văn bản.
5. **2d:** nhiễu nhãn của nli_fever, ví dụ M169: nhận định rõ ràng được xác nhận nhưng nhãn FEVER là NEI.
   - 12/30 dòng người đọc chọn khác nhãn, phần lớn với độ chắc "vừa" hoặc "thấp".
   - Hướng lệch phân tán: bác bỏ → NEI 4, bác bỏ → xác nhận 2, NEI → xác nhận 3, xác nhận → NEI 3.
   - Đây là đầu vào cho quyết định tỉ lệ 2d ở bước phân tích trước train.

---

# Đọc mù vòng 2: KC v2 (điều phối, 28/10/2026)

**Cách làm:**
- Mẫu `results/train_2b/mau_doc_v2.jsonl`, 220 dòng. Cùng cách che và quy ước như vòng 1; người đọc được dặn coi các định dạng ngày/số tương đương là như nhau.
- Script: `cham_doc2b_v2.py` trong scratchpad của điều phối.
- Luật chốt trước (brief, cuối mục 7c): lỗi rõ ≤ 7% của n (tối thiểu 1) thì đạt; ≤ 15% thì xem; > 15% thì bỏ hoặc sửa. Nhóm có n < 10 chỉ báo.

**Kết quả:**

| Nhóm | n | Trùng nhãn | Lỗi rõ | Kết luận | Cờ |
|---|---|---|---|---|---|
| N-bỏ bằng | 34 | 33 | 1 (V098) | đạt | **van_ban_gay 6**, mo_ho 1 |
| N-thêm | 34 | 34 | 0 | đạt | mo_ho 1, loi_cau 1 |
| N-ghép | 31 | 28 | 0 | đạt | loi_cau 3, mo_ho 3 |
| S-ghép | 33 | 33 | 0 | đạt | loi_cau 1, mo_ho 2 |
| R-ghép | 37 | 37 | 0 | đạt | — |
| S-lược (dựng lại) | 23 | 23 | 0 | đạt | loi_cau 1 (vòng 1: 6/30) |
| R-tên (dựng lại, không ghép) | 14 | 14 | 0 | đạt | loi_cau 1, mo_ho 2 |
| Có biến thể định dạng (nhóm chồng) | 46 | 46 | 0 | đạt | lac_khung 7 (đều N-lạc đề, đúng thiết kế) |
| S-nguyên / R-số / N-lạc đề | 3 / 4 / 7 | 3 / 4 / 7 | 0 | chỉ báo | — |

**Lỗi và cách xử lý** (lọc trong `train_2b_v2_loc.jsonl`, không dựng lại):
- **V098:** câu bị xóa có câu trùng ở chỗ khác trong bài, nên nhận định vẫn nằm nguyên trong state. Lọc theo bao phủ ≥ 0,9; trên pool đếm được 3 dòng như vậy.
- **"Văn bản gãy" ở N-bỏ bằng (6/34):** câu còn lại mở đầu bằng từ chỉ ngược ("Loại đèn này", "Một khác biệt khác"). Đây có thể là dấu hiệu riêng của nhãn NEI, nên lọc bằng danh sách từ nối mở rộng.
- **Một state còn sót cú pháp wiki "]]":** lọc mọi state có "[[" hoặc "]]".
- **V088:** giá trị suy ra được bằng phép trừ trên văn bản, người đọc gắn cờ mơ hồ. Đây là rủi ro "suy ra gián tiếp" đã biết của NEI dựng; giữ.
