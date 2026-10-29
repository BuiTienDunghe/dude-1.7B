# Phản biện bản tổng hợp hướng đi (27/10/2026) — Fable 5.1

Chỉ đọc repo và ba báo cáo; 3 lượt web (mixture.py của decider, TFLOPS 5060 Ti, Liger). Không chạy gì.

## Kết luận

**Bản tổng hợp đứng được về hướng** (chưng cất + dữ liệu dựng bằng chương trình + tăng thông lượng, không đi 4B), nhưng **chưa duyệt được nguyên bản**. Ba sửa đổi quan trọng nhất:

1. **Tách bước 2.** Hiện bước 2 gộp bốn thay đổi (C′, 5k dòng mới, đích teacher, EN 50–100k) vào một lần train, lặp đúng lỗi audit đã nêu ("train C trước A và B nên không quy được nguyên nhân", `audit_2710/TONG_HOP_dieu_phoi.md:74`). Chèn **bước 2a**: C.jsonl giữ nguyên, chỉ thay `target` bằng đích KD, 2 seed mỗi bên (gồm **C seed 2**, hiện C chỉ có 1 seed), đúng thí nghiệm nhỏ nhất ở `PHUONG_PHAP_TRAIN_SUY_LUAN_fable.md:29`. Cổng "+2 so với C trung bình 2 seed" hiện so 2 seed KD với 1 seed C.
2. **Hạ giọng câu "khoảng cách là công thức".** Hai biến nhiễu Fable đã nêu (`DOI_THU_VA_MODEL_NEN_fable.md:79`) bị bỏ: decider xuất phát từ **Base**, còn 70,6 là bản **post-trained** (`REPORT_baselines_v1_2.md:12,71`); decider dùng khung tiếng Anh thuần, không chat template. Thêm vào G0 phép thử 15 phút của audit (chấm model gốc với khung EN trên dev VI, `audit_2710/PHUONG_PHAP_fable.md:129,187`), chấm Qwen3.5-2B-Base, và nâng Qwen3.5-4B gốc từ "tùy chọn" lên bắt buộc: đó là cách duy nhất tách điểm Quyet thành cỡ / công thức / nhiễm (`DOI_THU:81`).
3. **Nhánh decider-2b va với chỉ tiêu không dấu ≥ 61.** decider 49,6 so với dự án 62,3 (`REPORT_baselines_v1_2.md:24,26`); ft_decider chỉ nâng +1,4 (`ft_decider/REPORT_ft_decider_v0.md:16`) và hạ VI có dấu 86,3 → 84,9 (`audit_2710/PHUONG_PHAP_fable.md:161`). Bản tổng hợp vừa đặt mục tiêu "giữ lợi thế không dấu" vừa đề xuất nhánh xuất phát thấp hơn chỉ tiêu 12 điểm. Giữ nhánh này ở hàng phụ, thêm cổng không dấu, và nêu rõ phải đổi sang định dạng của decider (noul qua system_one, score tốn n forward, `REPORT_ft_decider_v0.md:43–47`).

## Từng vấn đề

### 1. Khẳng định thổi phồng hoặc chưa đúng

- **[vừa] "decider 0 dòng VI nên khoảng cách là công thức" (`TONG_HOP:18–21`).** Ngoài Base và khung EN ở trên: trung bình bằng nhau (70,6) che hồ sơ khác hẳn (MASSIVE −11,2, ViFactCheck +12,7, `REPORT_baselines_v1_2.md:66`); decider ghi massive-en-US "trained" (`baselines/meta/Mapika__decider-2b/README.md:372`), MASSIVE là bộ song song. Tôi đã kiểm `mixture.py`: không có belebele, xcopa, sib, flores, nên rò chéo ngôn ngữ chỉ ở MASSIVE và số "bỏ MASSIVE +6,0" là đủ. Sửa: viết "~5/8 điểm nhiều khả năng do công thức, Base và khung prompt chưa tách".
- **[nhẹ] "Không phải sửa code loss" (`TONG_HOP:25`).** Đúng về hàm loss: `train_lora.py:179–190` nhận target k chiều bất kỳ, `shuffled():93–95` hoán vị target theo options, train_v0 đã có 1.117 dòng target mềm. Nhưng cần **script mới** sinh target: chạy teacher, ánh xạ noul theo chữ lựa chọn (`scripts/baselines_v1_2/adapters.py:22–27`), xử lý dòng bị từ chối (Quyet "text" từ chối nhãn trùng, `adapters.py:58–59`; C chỉ có 1 dòng như vậy) và dòng dài bị strict chặn. Viết "không sửa loss, cần script sinh target".
- **[nhẹ, chiều ngược] "Chưa có số công khai cho bài k chữ cái" (`TONG_HOP:65,87`).** Có tiền lệ công khai đúng lớp bài: decider v11 train 20.100 dòng replay về **phân phối của chính v10** bằng KL trên các lựa chọn, và chọn nhánh đó thay nhánh nhãn cứng (`README.md:279,284,292–294`). Không có con số lợi ích, nhưng nên dẫn.
- **[nhẹ] Độ trễ 4B "≥ 60 ms" (`TONG_HOP:47`).** Cận dưới dựa trên 47 TFLOPS của flopper.io, trang này tự ghi NVIDIA không công bố. Kết luận vẫn đứng, thậm chí còn nhẹ: số đo thật của dự án (1.7B ≈ 1,1 TFLOP ở 391 token → 47 ms) cho hiệu suất ~50%, nên 4B thực tế ~100 ms. Nên neo vào số đo, không neo vào TFLOPS suy ra. "FP8 42–50 ms" và "1.7B p90 33–40 ms khi bật CUDA graphs" (`TONG_HOP:50`) đều dựa trên phép tách chi phí cố định mà chính Fable gọi là không sạch (`PHUONG_PHAP:84`); ghi là ước thô.
- **[vừa] "+1–3 chưng cất, +2–4 quy mô EN", cộng thành "84–87" (`TONG_HOP:39,65`).** Hai số đoán được cộng dồn. +2–4 suy từ decider (1,47M dòng, teacher 27B, RL), trong khi bằng chứng nội bộ ngược chiều: C@25% ≈ C@100%, decider-ft tụt khi thêm 20k dòng (`audit_2710/DU_LIEU_fable.md:171`). Ghi "chưa có bằng chứng, không cộng dồn"; kỳ vọng tổng nên là +2–4, không +3–7.
- **[vừa] "Mượn quy mô của người khác" (`TONG_HOP:22–26`) không khớp lộ trình.** KD chỉ chuyển hàm của teacher trên **prompt của ta**; lộ trình chỉ có 5k prompt VI mới, còn EN 50–100k đã có nhãn sẵn. Muốn "gán nhãn hàng trăm nghìn câu" phải có kho prompt VI lớn: dữ liệu dựng bằng chương trình cho nhãn miễn phí và 0 chữ mới, nên nên làm **bậc 5k → 20k** với `--save_fracs`, thay vì chốt 5k vì lý do bão hòa (bão hòa đo trên dữ liệu 16 câu hỏi lặp, không nói gì về dữ liệu đa dạng).
- **[nhẹ] Thời gian bước 2:** 70–135k dòng (tùy EN 50–100k là tổng hay cộng thêm) × 2 epoch ở ~45k dòng/giờ là 3–6 h mỗi seed, không phải "2–4,5 h".

### 2. Thứ tự lộ trình

- **[vừa] Tăng tốc 2× đặt trước G0.** Liger và bucket viết cho Qwen3; nếu G0 đổi nền sang Qwen3.5 thì nút thắt là kernel FLA, không phải Liger. Giữ phần sửa lỗi của bước 0, dời phần tăng tốc sau G0. Ước 2× cũng lạc quan: padding đã giảm một phần nhờ sắp 32 câu rồi chia micro-batch 8 (`train_lora.py:117–123,163–170`), nên lợi của bucket toàn tập chỉ là phần còn lại, chưa đo. Ghi 1,3–2× và đo 100 bước trước.
- **[vừa] Lọc dòng teacher–nhãn bất đồng nằm ở bước 3, sau train.** Quy tắc λ→0 khi argmax teacher ≠ nhãn (`PHUONG_PHAP:21`) bị bỏ; với λ = 0,6 cố định, teacher sai tự tin (>0,67) biến thành nhãn train. Lọc phải nằm lúc dựng target.
- **[nhẹ] Thiếu C seed 2** (2,95 h) làm mốc cho quy tắc 2 seed.

### 3. Phương án bỏ sót

- **Qwen3.5-2B-Base với định dạng dự án + KD từ Quyet** là ứng viên "mở lại nền" thật, không phải decider: cùng tokenizer với Quyet, giữ khung VI và lợi thế không dấu của pipeline hiện tại. Cần tải Base (chưa có trên máy) và chấm ở G0.
- **Tự nhất quán theo hoán vị** (`PHUONG_PHAP:73`), 0 teacher ngoài: đưa vào bước 3.
- **Teacher phụ decider:** yếu trên VI không dấu và nhiều dòng VI gốc; chỉ dùng trên EN hoặc nơi hai teacher đồng ý, không trung bình mù trên VI.

### 4. Rủi ro chưa nêu

- **[vừa] Teacher chỉ chạy trên prompt có dấu;** bỏ dấu lúc nạp (`maybe_strip`) giữ target nên vẫn đúng. Cấm chạy teacher trên bản không dấu (Quyet 64,5, decider 49,6).
- **[vừa] "Tập chọn model riêng" chưa định nghĩa.** Nếu dựng từ cùng bộ sinh với dữ liệu mới thì nó cùng phân phối với train, không phản ánh bench; ngưỡng noul +1,4 từng fit trên dev (`PHUONG_PHAP:69`). Cần đặc tả: nguồn không có trong train, khung bench.
- **[nhẹ] Mở rộng EN phải giữ danh sách loại** MASSIVE, SLURP, IntentGrasp, COPA, XCOPA, Belebele, SIB-200 (`REPORT_baselines_v1_2.md:162`).
- **[nhẹ] UIT:** teacher chạy cục bộ, không rò; nhưng mọi model train trên C/C′ vẫn "chỉ nội bộ" vì UIT (`REPORT_ft_decider_v0.md:3`), nên bàn "định vị so với Quyet" phải nói rõ chưa công bố được.
- **[nhẹ] Decontam** cần mọi split ViWikiFC/ViFactCheck trên đĩa (`DU_LIEU_KHONG_NHAN_fable.md:95`); cổng bước 1 mặc nhiên coi đã có. Liger có patch Qwen3 (bảng PyPI) nhưng là monkeypatch transformers, chưa xác nhận với 5.19: đóng khung thời gian thử.

### 5. Câu hỏi gửi người dùng (mục 6)

Q1–Q4 đúng, không thừa. Thiếu:
- duyệt **bước 2a** (KD trên C, 2 seed) và **C seed 2**;
- **đặc tả tập chọn model** là gì;
- chỉ tiêu **không dấu ≥ 61 có ràng buộc nhánh decider** không;
- danh sách tải/cài cần duyệt cho G0: Qwen3.5-2B-Base, Qwen3.5-4B gốc, `flash-linear-attention`, Liger (venv riêng); và từng lần chạy teacher trên train là một lần chạy model cần duyệt.
