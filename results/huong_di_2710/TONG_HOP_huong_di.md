# Hướng đi tối ưu trong ràng buộc một GPU 16 GB (27/10/2026)

Điều phối tổng hợp từ ba báo cáo nghiên cứu của Fable 5.1 và một báo cáo phản biện, đều nằm trong thư mục này:
- `DOI_THU_VA_MODEL_NEN_fable.md`: đối thủ đã train thế nào, giấy phép, model nền;
- `PHUONG_PHAP_TRAIN_SUY_LUAN_fable.md`: phương pháp train, độ trễ;
- `DU_LIEU_KHONG_NHAN_fable.md`: dữ liệu tiếng Việt không cần người gán nhãn;
- `PHAN_BIEN_fable.md`: phản biện bản nháp. Bản này đã sửa theo phản biện.

Bối cảnh chung nằm ở `BOI_CANH_chung.md`. Không agent nào chạy model, tải dữ liệu hay đụng test.

**Điều phối đã tự kiểm:**
- README decider-2b v11: "English only", RL 384 bước, và 20.100 dòng tự chưng cất bằng KL trên các lựa chọn.
- Qwen3.5-2B gốc 57,7 ms; decider-2b không dấu 49,6.
- Độ dài prompt dev 1.3: p50 165, p90 391, p95 862 token.
- `shuffled()` hoán vị nhãn cùng lựa chọn.
- P4 của Quyet đã đo ngày 26/10: chọn đúng "không đủ thông tin" **97,5%** ở VI có dấu, nhưng chỉ 41–79% ở VI không dấu.

## Kết luận

1. **Train như cũ thì đúng là không bằng Quyet.** Nhưng phần lớn khoảng cách nhiều khả năng đến từ **công thức train**, không phải từ cỡ model hay từ thiếu tiếng Việt.
   - decider-2b cùng cỡ với ta, **không có bộ tiếng Việt nào trong mixture công khai** (dữ liệu do teacher viết không ghi ngôn ngữ), khoảng 1,47 triệu dòng mỗi epoch, có teacher 27B viết dữ liệu và có RL. Nó đạt **86,5 VI**. Bỏ MASSIVE (bộ duy nhất nó có thể đã thấy) thì vẫn hơn model cũ của dự án +6,0.
   - Dự án dùng khoảng 62.000 lượt-dòng, đạt 81,65.
   - Có hai biến chưa tách được: decider xuất phát từ bản **Base** chưa post-train, và dùng khung prompt tiếng Anh. G0 sẽ đo hai biến này.
2. **Ta không tự tạo được quy mô đó** bằng người gán nhãn hay hạn mức Claude. Hướng khả quan nhất là **mượn sức của model mạnh hơn và tự dựng dữ liệu bằng chương trình**:
   - **Chưng cất (distillation) từ Quyet-Medium.** Quyet chấm phân phối A–J trên prompt train của ta, kết quả dùng làm đích mềm.
     - Teacher chỉ cần prompt, không cần nhãn.
     - Hàm loss không phải sửa. Chỉ cần thêm một script sinh đích.
     - decider v11 cũng làm đúng kiểu này (KL trên các lựa chọn) và giữ nhánh đó.
     - Quyet không có thói "bác bỏ" khi văn bản lạc đề (P4 97,5%), nên hợp làm teacher cho đúng điểm yếu của bản C.
     - Giấy phép Apache-2.0 nguyên bản, không cấm dùng đầu ra.
   - **Dữ liệu tiếng Việt dựng bằng chương trình**, đúng khung bench:
     - nguồn: Wikipedia tiếng Việt và văn bản web có giấy phép mở;
     - loại câu: kiểm chứng 3 nhãn (khoảng 35% "không đủ thông tin"), chọn câu tiếp theo, chọn tiêu đề, văn bản dài, bản không dấu;
     - nhãn đến từ cách dựng, nên **không có chữ tiếng Việt do agent viết** (trừ 2 câu hỏi), và không cần GPU;
     - làm theo bậc 5.000 rồi 20.000 dòng.
   - **Tăng tốc train 1,3–2×** bằng cách xếp batch theo độ dài, tắt gradient checkpointing ở câu ngắn, và dùng Liger kernel. Đo 100 bước trước khi tin.
3. **Nên đổi cách đặt mục tiêu.** Thay "bằng Quyet" bằng **"model tiếng Việt tốt nhất trong ngưỡng ≤ 40 ms trên một GPU phổ thông"**.
   - Quyet đạt 89,6, nhưng có 4,66B tham số và chạy 87,5 ms.
   - Mốc hợp lý: **vượt decider-2b (86,5)** ở VI có dấu, giữ không dấu ≥ 61 (decider chỉ 49,6).
   - Kỳ vọng tổng thực tế: **+2 đến +4** so với C, tức khoảng 84–86. **Chưa có bằng chứng chắc.** Các mức tăng không cộng dồn được.

## 1. Những phát hiện làm đổi bức tranh

1. **"Quyet train 120.000 bước" không so được.** Đó chỉ là nhãn checkpoint, không có batch, epoch hay số dòng. Nên so bằng lượt-dòng: dự án 62k, Jeff 285k, decider 1,47 triệu mỗi epoch, Quyet không rõ.
2. **Model 4B bị loại vì độ trễ.** Model 1.7B đo được 47 ms ở prompt p90 (391 token). 4B cần khoảng 2,6× phép tính, nên ước khoảng 100 ms. CUDA graphs không cứu được, vì phần lớn thời gian là tính toán chứ không phải khởi chạy.
3. **Ngay model hiện tại cũng chưa đạt 40 ms ở mọi mức.** Ước thô khi bật CUDA graphs:
   - trung vị khoảng 13–17 ms;
   - p90 khoảng 33–40 ms;
   - p95 (866 token, chủ yếu ViFactCheck) khoảng 60–85 ms.

   Cần định nghĩa "≤ 40 ms" là trung vị, p90 hay p95.
4. **Qwen3.5-2B chưa từng được đo công bằng.** Số 57,7 ms đo khi thiếu kernel `flash-linear-attention` và không có CUDA graphs. Về phép tính, model này ngang 1.7B.
5. **Không có bộ dữ liệu tiếng Việt có nhãn mới** nào vừa có giấy phép rõ vừa khớp các điểm yếu.
6. **Nhiễm của Quyet: không có bằng chứng công khai theo chiều nào.** Quyet đạt XCOPA EN 100/100, nhưng SIB-200 ngang dự án, và ở ViFactCheck tiếng Việt gốc vẫn hơn +15,8.

## 2. Lộ trình đề xuất (đã sửa theo phản biện)

| # | Việc | GPU | Chữ VI cần duyệt | Cổng hoặc kỳ vọng |
|---|---|---|---|---|
| **0** | **Sửa lỗi theo audit:** ngừng xáo câu thang điểm, đổi loss mặc định, nâng trần lên 2.048, lấy chữ nhãn từ dev; đặc tả tập chọn model riêng (mục 4) | 0 | 0 | — |
| **G0** | **Đo, không train:** (a) độ trễ 1.7B và Qwen3.5-2B có FLA và CUDA graphs, báo trung vị/p90/p95 theo nhóm độ dài; (b) chấm Qwen3.5-2B-**Base** và Qwen3.5-4B gốc trên dev; (c) chấm model gốc với khung prompt tiếng Anh trên dev VI (15 phút) | khoảng 2 h; **cần tải** Qwen3.5-2B-Base (~4,5 GB) và Qwen3.5-4B (~9 GB), ổ C còn trống 222 GB; **cài** flash-linear-attention trong venv riêng. Việc chấm 2B-Base đảo quyết định 12/10 "không chấm Qwen3.5-2B-Base". Lý do: đây là nền của decider-2b (86,5), và số 57,7 ms của 2B đo khi thiếu kernel | 0 | Tách điểm của Quyet thành phần do cỡ, do công thức và do khung prompt; chọn nền |
| **1** | **Tăng tốc train** trên nền đã chọn | khoảng 0,3 h đo | 0 | 1 epoch train_v0 cho val loss 0,756 ± 0,01; báo hệ số tăng tốc thật |
| **2a** | **Chưng cất trên C giữ nguyên dữ liệu.** Quyet chấm 31k prompt train (chỉ bản có dấu). Đích = 0,6 × teacher + 0,4 × nhãn; dòng nào teacher bất đồng với nhãn thì chỉ dùng nhãn và đánh dấu để soát. Câu thang điểm giữ nguyên đích cũ (teacher chỉ chạy trên câu chọn một và câu Có/Không). Train **2 seed**, **cộng C seed 2** làm mốc | teacher khoảng 1 h; 3 lần train, mỗi lần khoảng 1,5–3 h | 0 | VI ≥ +2 so với C (trung bình 2 seed mỗi bên); P4 ≥ M1; letter_mass ≥ 0,99 |
| **2b** | **Dữ liệu:** C′ theo audit, cộng 5.000 dòng VI dựng bằng chương trình (nếu có lợi, lên 20.000), Quyet chấm đích | 0 để dựng; teacher < 1 h; train 2 seed | 2 câu hỏi; đọc mẫu khoảng 180 dòng; **cần tải** Wikipedia vi và 1–2 shard FineWeb-2 | Khử trùng 0 chạm; probe chỉ đọc nhận định; NEI ≥ 30%; VI tăng so với 2a |
| **2c** | **Tăng tiếng Anh theo bậc 50k, rồi 100k** (giữ danh sách loại MASSIVE, SLURP, COPA, XCOPA, Belebele, SIB-200), kèm đích teacher | 3–6 h mỗi seed | 0 | VI ≥ +2 so với 2b |
| **3** | **Gần 0 GPU:** gộp 2 seed (merge thành ΔW rồi lấy trung bình), tự nhất quán theo hoán vị, ngưỡng noul fit trên tập chọn model | khoảng 0,3 h | 0 | +0 đến +1,5 |
| **Dừng** | Khoảng 06/11/2026: không vượt C ≥ 2 điểm thì hạ mục tiêu, hoặc chuyển sang hoàn tất benchmark | — | — | Bạn quyết |

Mỗi bước chỉ đổi một thứ, để biết điểm tăng hay giảm vì đâu (rút kinh nghiệm từ bản C). Mỗi lần chạy teacher trên dữ liệu train là một lần chạy model và **cần bạn duyệt**.

**Ứng viên nếu mở lại câu hỏi model nền** (quyết sau G0): **Qwen3.5-2B-Base, dùng định dạng của dự án, chưng cất từ Quyet.**
- Cùng tokenizer với Quyet.
- Giữ khung tiếng Việt và lợi thế không dấu.
- Chỉ đáng đổi nếu G0 cho thấy đạt độ trễ và có điểm nền tốt.

**Nhánh phụ: xuất phát từ decider-2b.** Không khuyến nghị làm nhánh chính, vì:
- không dấu chỉ 49,6, thấp hơn chỉ tiêu 61 tới 11 điểm;
- lần fine-tune decider ngày 13/10 chỉ +1,4 không dấu và làm VI có dấu tụt;
- phải đổi sang định dạng của decider.

## 3. Không làm

- **DPO/ORPO/GRPO:** đầu ra chỉ 1 token, cùng hướng với CE.
- **4B làm model chính:** vỡ độ trễ.
- **Sinh dữ liệu bằng LLM hoặc dịch thêm:** mọi dòng là chữ mới, phải duyệt 100%.
- **Packing, FP8 training, đổi engine:** chưa có FlashAttention-2 cho Blackwell; lợi nhỏ.
- **Gemma 4, Llama 3.2, Phi-4-mini, Sailor2, VyLinh:** yếu tiếng Việt, hoặc giấy phép hay cỡ không hợp.
- **decider-2b làm teacher cho phần tiếng Việt:** yếu ở không dấu và ở VI gốc. Chỉ dùng phụ cho phần tiếng Anh, hoặc ở chỗ nó đồng ý với Quyet.

## 4. Rủi ro và điều chưa biết

- **Mức tăng của chưng cất** cho bài toán dạng này chưa có con số công khai. decider có dùng nhưng không báo lợi ích.
- **Dùng Quyet làm teacher thì so với Quyet không còn độc lập:** model của ta sẽ thành "Quyet nén lại cho nhanh". Chưng cất chỉ trên prompt train nên không chép được đáp án dev.
- **Teacher chỉ chạy trên prompt có dấu.** Quyet yếu ở không dấu (P4 41–79%). Bản không dấu lấy đích từ bản có dấu tương ứng.
- **Tập chọn model riêng:**
  - phải dùng nguồn không có trong train và đúng khung bench;
  - không được dựng bằng chính bộ sinh của dữ liệu mới;
  - không lấy từ dev.

  Ngưỡng noul trước đây fit trên dev, nên phải fit lại trên tập này.
- **Tăng tốc phụ thuộc phần mềm:** Liger là bản vá transformers, chưa xác nhận với 5.19 và Blackwell. Đặt giới hạn thời gian cho việc thử.
- **Tải dữ liệu cần bạn duyệt.** Giấy phép tổng hợp không phủ bản quyền từng bài, nên chỉ dùng nội bộ. Mọi model train trên C/C′ vẫn chỉ dùng nội bộ vì có dữ liệu UIT.

## 5. Chỗ điều phối và Fable khác nhau

- **Quy mô hay khung dữ liệu.** Mảng dữ liệu nói "đòn bẩy không phải số dòng"; mảng đối thủ và phương pháp nói quy mô và độ đa dạng là chính. Điều phối cho rằng cả hai đúng: model bão hòa vì dữ liệu lặp (16 câu hỏi VI), cái thiếu là **đa dạng ở quy mô lớn**. Vì vậy dữ liệu dựng bằng chương trình đi theo bậc 5k rồi 20k, không chốt 5k.
- **Model 4B.** Mảng phương pháp xếp Qwen3-4B hạng 5, có điều kiện. Điều phối chỉ dùng 4B để đo (G0), không làm model chính.
- **Phản biện được chấp nhận toàn bộ:**
  - tách bước 2 thành 2a/2b/2c;
  - thêm C seed 2;
  - lọc dòng teacher bất đồng ngay lúc dựng đích;
  - thêm hai biến nhiễu (Base, khung tiếng Anh) vào G0;
  - hạ nhánh decider-2b xuống nhánh phụ;
  - không cộng dồn kỳ vọng;
  - tăng tốc 1,3–2×.
- **Điều phối thêm một điểm:** P4 của Quyet đã đo sẵn (97,5%), nên không cần đo lại trước khi dùng làm teacher.

## 6. Cần người dùng quyết

1. **"≤ 40 ms" nghĩa là gì:** trung vị, p90 hay p95? Nên quyết **sau G0**, khi có bảng đo. Ước thô: p95 của model hiện tại là 60–85 ms.
2. **Có dùng Quyet-Medium làm teacher không?** Hệ quả: so với Quyet không còn độc lập.
3. **Đổi mục tiêu thành "tốt nhất trong ≤ 40 ms"** với mốc vượt decider-2b (86,5) và không dấu ≥ 61?
4. **Duyệt bước 0 và G0.** G0 gồm tải Qwen3.5-2B-Base và Qwen3.5-4B, cài flash-linear-attention trong venv riêng. Việc chấm 2B-Base đảo quyết định 12/10. Kèm câu từ lần trước: áp dụng quy tắc 2 seed. Có mở lại model nền hay không thì quyết **sau G0**.
5. **Duyệt sau, chưa cần ngay:**
   - bước 2a (gồm C seed 2 và một lần chạy teacher);
   - bỏ quy tắc giữ cố định 16.000 dòng EN (cần cho 2c);
   - tải Wikipedia tiếng Việt và 1–2 shard FineWeb-2;
   - 2 câu hỏi tiếng Việt mới.
