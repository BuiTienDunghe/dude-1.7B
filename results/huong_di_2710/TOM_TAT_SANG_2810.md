# Tóm tắt buổi sáng 28/10/2026: kết quả đêm qua và việc cần bạn duyệt

## Kết luận nhanh

1. **Giữ Qwen3-1.7B.** Qwen3.5-4B chưa train đã đạt 84,1 điểm VI, cao hơn bản C đã train (81,65), nhưng chạy quá chậm: trung vị 82 ms, p90 138 ms. 4B bị loại dù hiểu ngưỡng 40 ms theo cách nào. Qwen3.5-2B-Base cũng không hơn 1.7B.
2. **Bỏ lần train 2a (chưng cất trên dữ liệu C).** Chẩn đoán cho thấy tác động quá nhỏ để đo được. Ngoài ra Quyet nhận diện "không đủ thông tin" kém ở các câu cùng chủ đề, nên tin Quyet ở bài kiểm chứng là sai hướng. Kết luận này đến sau khi Fable phản biện và tôi tự kiểm lại số. Bỏ bước này tiết kiệm khoảng 6–9 giờ GPU.
3. **Bước tiếp nên là 2b: sửa dữ liệu cho đúng khung bench.**
   - Bỏ dữ liệu hỏng: 3 nguồn không có văn bản và 146 dòng nhãn đáng ngờ.
   - Sửa chữ nhãn.
   - Dựng bằng chương trình (không viết chữ tiếng Việt mới) các câu kiểm chứng đúng khung bench, có "không đủ thông tin" cả loại cùng chủ đề lẫn loại lạc đề.
   - Thêm cách dạy "từ chối trả lời" học từ decider. Với decider, cách này đưa điểm câu lạc đề từ 0,68 lên 0,83 mà không giảm điểm chung.

## Đêm qua đã làm (không train, không đụng test)

| Việc | Kết quả |
|---|---|
| Bước 0: sửa lỗi | Đã sửa: xáo thứ tự lựa chọn, loss mặc định, trần độ dài 2.048, script kiểm chữ nhãn. Đã dựng tập chọn model riêng (769 cặp, 0 trùng). Chấm lại C trùng từng byte với lần trước |
| G0: đo | Xem bảng dưới. Khung prompt tiếng Việt giữ nguyên, vì khung tiếng Anh không hơn đủ ngưỡng |
| FLA | Bạn đã cài gcc, kernel chạy được, nhưng kết quả chỉ trùng 98,5% với khi không dùng kernel. Bỏ phần đo này, vì 2B đã bị loại bằng điểm |
| Chẩn đoán teacher | Quyet đồng ý với 81% nhãn VI. Soát mù 60 dòng. Fable phản biện rồi chốt: không chạy 2a |
| Học từ decider | Chưng cất không tăng độ chính xác của họ. Cách dạy "từ chối" của họ đánh đúng lỗi P4 của ta |

**Độ trễ** (batch 1, đo theo cách thường, ms):

| Model | Trung vị | p90 | p95 |
|---|---|---|---|
| C (1.7B) | 31,8 | 47,3 | 85,4 |
| 2B-Base | 53,2 | 70,3 | 117,0 |
| 4B | 82,0 | 138,0 | 263,8 |

**Tài nguyên đêm qua:**
- GPU: khoảng 2 giờ 50 phút, nhiệt độ cao nhất 78°C.
- Mức dùng tuần: 49%. Fable: khoảng 37%.
- Bên Ún đã nhường GPU từ 00:40 đến 04:10 và đã được báo dùng lại.

## Việc cần bạn duyệt

1. **Hướng đi:** đồng ý bỏ 2a, chuyển sang 2b?
2. **Dựng dữ liệu 2b** (chỉ dùng CPU, chưa train):
   - tải Wikipedia tiếng Việt (bản đóng gói sẵn trên Hugging Face, khoảng 1–2 GB, giấy phép CC BY-SA, chỉ dùng nội bộ);
   - worker dựng C′ cộng khoảng 5.000 dòng VI bằng chương trình;
   - kiểm khử trùng với dev và test, đọc mẫu.
3. **Định nghĩa "≤ 40 ms":** bản C hiện chỉ đạt ở trung vị. Có ba lựa chọn:
   - (a) tính theo trung vị;
   - (b) tính theo p90: cần cắt văn bản dài khi suy luận, hoặc sửa đường CUDA graphs;
   - (c) tính theo p95: khó đạt.
4. **Sau khi dữ liệu 2b xong**, tôi sẽ làm phân tích trước train như lần này, rồi mới xin duyệt train. Dự kiến 3 lần train: C′ seed 1 và 2, cộng C seed 2 làm mốc; tổng khoảng 8 giờ GPU.

## File chi tiết

- `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN.md`: phân tích đầy đủ, mục 10 là kết luận cuối.
- `results/huong_di_2710/PHAN_BIEN_phan_tich_fable.md`: phản biện của Fable.
- `results/G0/REPORT_G0.md`, `results/teacher_diag/REPORT_chan_doan.md`, `results/G0/REPORT_buoc0.md`: báo cáo của worker.
- `results/huong_di_2710/HOC_TU_DECIDER.md`: bài học từ decider.
