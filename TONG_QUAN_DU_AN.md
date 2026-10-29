# Tổng quan dự án: decision model và benchmark tiếng Việt

Cập nhật 17/10/2026 (điều phối). Chi tiết từng quyết định ở `PLAN_decisions.md`, kế hoạch gốc ở `Downloads/PLAN.md`.

## 1. Dự án làm gì

Hai mục tiêu:
1. **Một decision model nhỏ dùng tốt cho tiếng Việt.** Model nhận một văn bản và một câu hỏi có sẵn các lựa chọn, trả về xác suất cho từng lựa chọn, không sinh văn bản.
2. **Một benchmark tiếng Việt đáng tin** để chứng minh điều đó. Benchmark neo vào nhãn do người gán, song song Anh–Việt, và chấm mọi model trong cùng điều kiện.

Kiểu dùng đã chốt: **từng yêu cầu một**, nên tốc độ (độ trễ batch 1) là ràng buộc cứng.

## 2. Đã đi qua những gì

| Ngày | Chặng | Kết quả |
|---|---|---|
| 11–12/10 | Bước 1–3: dựng tập đo nhỏ, đo model gốc, train thử Qwen3-0.6B và Qwen3-1.7B | 1.7B hơn 0.6B 5,2 điểm tiếng Việt, mà với câu ngắn chạy nhanh như nhau |
| 12/10 | Sửa loss (full_vocab); **chọn Qwen3-1.7B**; đo baseline đầu tiên (Qwen3.5-2B gốc, decider-2b) | Model dự án hiện tại ra đời từ đây |
| 13/10 | Bench 1.0 rồi 1.1 (test đóng băng); train thử bỏ dấu (không đạt); train thử trên decider-2b; khảo sát nguồn train tiếng Việt | Biết model mạnh ở tiếng Việt không dấu, yếu ở đọc hiểu và kiểm chứng |
| 14–16/10 | **Soát toàn bộ bản dịch và nhãn**: Opus quét và soi lại; bạn duyệt trên công cụ và trang từ xa; chuỗi mẫu qua Excel | **Bench 1.2**: hơn 700 sửa, 23 đáp án đổi theo bằng chứng, chuẩn hóa NFC; test đóng băng |
| 16/10 | **Bước E: chấm các decision model khác** trên dev 1.2 | Quyet-Medium dẫn đầu, decider-2b thứ hai, dự án thứ ba; dự án nhất ở không dấu và tốc độ |
| 16–17/10 | Soát nhãn mù 134 cặp nghi ngờ | Không có nhãn sai; 28 câu mơ hồ (bạn loại 2, gắn cờ 26); 1 sửa dịch |
| 17/10 | **Bench 1.3 đóng băng** (MLQA dời sang 1.4); dọn khoảng 16 GB ổ đĩa | Test 1.3 trùng từng byte test 1.2 |
| 17/10 | **Brief train chính thức**, bạn chốt (a)–(f); **Giai đoạn 0** (chuẩn bị dữ liệu) xong | Phần A sẵn sàng; phần B chờ dịch |

## 3. Hiện đang có gì

### Benchmark 1.3 (`data/bench_v1_3/`, đã đóng băng)
- **Nguồn tính điểm chính:** Belebele, SIB-200, MASSIVE, XCOPA (song song Anh–Việt); ViFactCheck, ViWikiFC (tiếng Việt bản địa).
- **Kích thước:** dev 1.430 cặp VI, test 1.553 cặp. Mỗi câu có bản không dấu; các nguồn song song có thêm bản tiếng Anh.
- **Lát phụ:** "bỏ câu mơ hồ" (26 câu). MLQA sẽ là lát riêng ở bản 1.4.
- **Quy tắc:** test chưa từng được chạy model. Chỉ chấm test đúng một lần, ở cuối.

### Model và vị trí so với model khác (dev 1.3, VI có dấu)

| Model | Cỡ | VI có dấu | VI không dấu | EN | Độ trễ |
|---|---|---|---|---|---|
| Quyet-1.0-Medium | 4,7B | 89,6 | 64,6 | 94,5 | 88 ms |
| decider-2b | 1,9B | 86,5 | 49,6 | 90,3 | 52 ms |
| **Dự án (Qwen3-1.7B + LoRA)** | 2,0B | **79,9** | **62,4** | 88,0 | **33 ms** |
| Qwen3-1.7B gốc | 2,0B | 70,6 | 48,5 | 84,6 | 32 ms |
| Qwen3.5-2B gốc | 2,3B | 70,6 | 39,0 | 84,5 | 58 ms |

Model dự án thua nhiều nhất ở ViFactCheck (−10,8), MASSIVE (−9,5), XCOPA (−8,0) và Belebele (−5,4). Lý do: dữ liệu train hiện tại không có các dạng bài đó bằng tiếng Việt.

### Công cụ
- **Công cụ duyệt** tại http://127.0.0.1:8765, chỉ chạy trên máy. Mọi chuỗi tiếng Việt đi qua đề xuất và bạn duyệt.
- **Trang duyệt từ xa** (artifact riêng tư) và các trang HTML gửi qua chat.

## 4. Đang ở đâu: Bước 6, train chính thức

Kế hoạch ở `results/BRIEF_train_chinh_thuc.md`, bạn đã chốt:
- Chỉ Qwen3-1.7B, giữ nguyên công thức, chỉ đổi dữ liệu.
- Model dùng nội bộ; dữ liệu do agent khác của bạn dịch.
- Train thử 3 bản A, B, C, rồi train chính thức 2 seed.
- Mục tiêu (dev 1.3): VI có dấu ≥ 84,0, không dấu ≥ 61, EN ≥ 87, độ trễ ≤ 40 ms.

**Giai đoạn 0 (chuẩn bị dữ liệu) đã xong:**

| Phần | Nội dung | Trạng thái |
|---|---|---|
| Nền | train_v0 (20.000 dòng, 16.000 EN + 4.000 VI) bỏ 8 dòng trùng bench | Sẵn sàng |
| A: tiếng Việt bản địa | 4.736 dòng: đọc hiểu (VSMRC, EXAMS), ý định, yes/no (ViANLI), chủ đề (VNTC) | Dựng xong; chờ bạn duyệt 25 tên nhãn tiếng Việt |
| B: dịch từ tiếng Anh | 5.944 dòng: đọc hiểu, ý định, kiểm chứng, yes/no; khoảng 950 nghìn token cần dịch | File, hướng dẫn và script kiểm đã có ở `results/train_v1/to_translate/`; **chưa giao dịch** |

Mọi dòng mới đã được đối chiếu với toàn bộ dev, test và các nguồn bench: 0 dòng trùng.

## 5. Tiếp theo sẽ như thế nào

```
[Bạn chốt 5 điểm ở mục 6] → worker cập nhật file → tôi kiểm
   → bạn giao file cho agent dịch → dịch xong → tôi kiểm bản dịch (định dạng, trùng bench)
   → BÁO BẠN (chưa train)
   → bạn cho phép → Giai đoạn 1: train thử A, B, C (khoảng 8 giờ GPU)
   → so trên dev 1.3, chọn bản tốt nhất → BÁO BẠN
   → Giai đoạn 2: train chính thức 2 seed (khoảng 6–11 giờ GPU) → BÁO BẠN
   → Giai đoạn 3: chấm test 1.3 ĐÚNG MỘT LẦN (cần bạn cho phép riêng)
   → Bước 7: bảng so sánh chính thức với các model khác
   → Bước 8: công bố (tính sau, khi model đủ mạnh)
```

Việc làm song song, khi bạn muốn:
- **Bench 1.4:** thêm MLQA thành lát riêng, khi bạn xem xong 30 câu ở `results/review/mlqa_test_review.html`.
- **Phần 5b của benchmark** (khối dịch máy có kiểm, lớp chỉ cho nghiên cứu): chưa bắt đầu.

## 6. Các lựa chọn đang chờ bạn

### Ngay bây giờ, trước khi giao dịch

| # | Câu hỏi | Lựa chọn | Đề xuất |
|---|---|---|---|
| 1 | VSMRC (đọc hiểu tiếng Việt) | giữ 1.236 dòng / nâng lên 2.000 (phải tách nguồn để lách giới hạn 5%) | **Giữ 1.236** |
| 2 | Ý định (phần dịch) | giữ 1.144 / đủ 1.500 (phải viết cách dựng câu mới) | **Giữ 1.144** |
| 3 | 39 dòng tiếng Anh cũ trùng một phần với nguồn bench (1 dòng trùng nguyên văn test) | giữ / bỏ | **Bỏ cả 39** (chỉ 0,2%) |
| 4 | Dữ liệu nhân quả để giúp XCOPA: e-CARE (MIT), HeadlineCause (CC0) | không dùng / thêm khoảng 500 dòng mỗi nguồn vào phần dịch | **Thêm** |
| 5 | 25 tên nhãn tiếng Việt mới (`tv1-*`) | bạn tự duyệt tại 127.0.0.1:8765/proposals / "duyệt theo agent" | Tùy bạn |

### Sau này

| Lúc nào | Lựa chọn |
|---|---|
| Sau Giai đoạn 1, nếu không bản nào đạt 84,0 | Thêm dữ liệu theo dạng bài còn yếu; tăng lên khoảng 60.000 dòng; hoặc chấp nhận mục tiêu thấp hơn. Không tự đổi chỉ tiêu |
| Khi muốn công bố model | Cần thỏa thuận với nhóm UIT, hoặc train lại không có dữ liệu UIT và dữ liệu cấm thương mại |
| Dọn ổ đĩa mục B (khoảng 15 GB) | Bạn đã quyết giữ để so sánh; xem lại sau bước 7 |
| Bench 1.4 / phần 5b | Khi nào làm, làm đến đâu |

## 7. Quy tắc thường trực

- Không chạy model trên test trước Giai đoạn 3; không chỉnh model theo điểm test.
- Không train, commit hay xóa file khi bạn chưa bảo.
- Mọi chuỗi tiếng Việt vào benchmark hoặc config đều qua đề xuất và bạn duyệt. Việc dịch hay soát benchmark dùng Opus.
- Dữ liệu UIT chỉ dùng nội bộ.
- Không dùng mức dùng thêm. Từ 88% hạn mức không mở việc mới, 92% dừng hẳn, Fable < 90%.
