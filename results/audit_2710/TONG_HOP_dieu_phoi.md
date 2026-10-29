# Kiểm toán toàn bộ dự án: báo cáo tổng hợp (27/10/2026)

Báo cáo này tổng hợp ba bản kiểm toán độc lập do Fable 5.1 viết, cộng phần điều phối tự đọc code và tự kiểm số:
- `PHUONG_PHAP_fable.md`: phương pháp và hàm chuyển LLM → decision model;
- `DU_LIEU_fable.md`: dữ liệu train;
- `DANH_GIA_CHIEN_LUOC_fable.md`: cách đánh giá và chiến lược.

Không bản nào chạy model, không đụng test và không sửa dữ liệu. Ký hiệu **[đã kiểm]** nghĩa là điều phối đã tự kiểm lại bằng code hoặc số liệu.

## Kết luận chung

**Dự án không có lỗi gốc ở hàm chuyển đổi hay ở cách đo.** Vấn đề nằm ở **công thức train**, gồm ba phần:
1. Quy mô train quá nhỏ.
2. Dữ liệu tiếng Việt dạy sai "khung bài": khác hẳn cách bench đặt câu hỏi, và có những dòng không có văn bản để đọc.
3. Vài lỗi nhỏ trong quy trình.

Cỡ model (1.7B) **chưa được chứng minh** là giới hạn, nhưng cũng chưa được loại trừ.

## 1. Thuật toán và hàm chuyển LLM → decision model: đúng và đạt chuẩn

**Hàm chuyển không có lỗi:**
- Dùng một hàm dựng prompt chung cho train và chấm [đã kiểm, `scripts/common.py:79–93`].
- Mỗi chữ cái A–J đúng một token và không làm đổi token đứng trước [đã kiểm, `eval_base.py:56,63–64`].
- Thinking tắt nhất quán; loss full_vocab đúng công thức; letter_mass 0,999.
- Cách làm này giống decider-2b và Quyet-Medium, đều đọc logit chữ cái ở token kế tiếp. Quyet đạt 89,6 bằng đúng cách này, nên **hàm chuyển không phải trần điểm**.

**Thuật toán train chưa tối ưu:**
- Ta chỉ train **1.250 bước** (bản C là 1.938). Quyet-Medium train **120.000 bước** cùng LoRA r16; Jeff full fine-tune 285.000 câu; decider dùng khoảng 95 bộ dữ liệu, cộng RL, LoRA r64. Ta nhỏ hơn đối thủ 1–2 bậc.
- Model đã khớp hết dữ liệu ngay từ mốc 25%: đường điểm theo các mốc phẳng, val loss không đổi giữa hai epoch. Vì vậy thêm 11.000 dòng chỉ được +1,8.
- **Không có tập chọn model phản ánh mục tiêu.** val_v0 cùng phân phối với train: bản C có val loss tốt hơn, nhưng lại hỏng ở P4.

**Lỗi nhỏ trong code:**
- Hàm `shuffled()` xáo cả câu Có/Không và câu thang điểm, dù PLAN 12/10 đã quyết **ngừng xáo câu thang điểm** [đã kiểm, `train_lora.py:93–95`, không kiểm loại câu]. Quyết định đó chưa từng được thực thi.
- Mặc định `--loss` vẫn là `letters_k`, loss đã biết là hỏng (các lần train đều tự truyền `full_vocab`, nên chưa gây hại).
- **Script train loại mọi văn bản dài hơn 1.024 token.** Văn bản tiếng Việt dài nhất trong train_v0 chỉ 220 token, trong khi ViFactCheck trên dev có độ dài trung vị **886 token** và 53/158 câu dài hơn 1.024 [đã kiểm].

## 2. Dữ liệu: đúng ở mức dòng, nhưng dạy sai khung bài

**Về mức dòng, dữ liệu đúng:**
- 0 dòng hỏng, 0 đáp án lệch, 0 rò rỉ sang dev.
- Nhãn người của PhoLaya sai rõ khoảng 3%.
- Bản dịch Sonnet không đổi nghĩa ở 30/30 cặp đọc kiểm.
- Lỗi recast tìm được: 21 dòng VSMRC có lựa chọn kiểu "Cả B và C". Khi xáo thứ tự lựa chọn, các dòng này thành nhãn sai.

**Vấn đề thật:**
- **Khoảng 1.109 dòng tiếng Việt không có văn bản để đọc:** health_fact chỉ có nhận định, boolq không có đoạn văn, EXAMS. Model học **phán đúng/sai bằng kiến thức sẵn có**, khớp đúng lỗi "chọn bác bỏ khi văn bản không liên quan" (bản C 110/241 lần, model hiện tại 26/241).
- **Khung kiểm chứng trong train khác hẳn bench.** 0 dòng dùng khung câu hỏi kiểm chứng của dev. 593 dòng vitaminc dùng chữ nhãn của bench nhưng đặt trong khung "hai văn bản + câu hỏi chung". Nhãn "không đủ thông tin" chỉ chiếm 12,8% (dev là 31,1%), và không có dòng nào dạng "văn bản lạc đề → không đủ thông tin".
- **Lệch chữ nhãn do lỗi quy trình của điều phối:** dữ liệu train dùng "Văn bản không đủ thông tin", còn dev dùng "**Văn bản không đủ thông tin để kết luận**" [đã kiểm]. Nguyên nhân: config trên đĩa và lớp ghi đè người dùng đã duyệt là hai nguồn khác nhau, và điều phối chỉ bảo lấy theo config. Kết quả là hai nhãn kia khớp từng chữ với dev, riêng nhãn "không đủ thông tin" thì không. Điều này có thể góp phần vào thói quen chọn "bác bỏ".
- **Khung đọc hiểu dịch khác Belebele:** câu hỏi thật nằm trong văn bản, còn ô câu hỏi chỉ là mệnh lệnh chung. VSMRC có mẹo "đáp án là lựa chọn dài nhất" (45%, Belebele chỉ 22%).
- **Câu hỏi bench dùng lại với bộ nhãn khác:** VNTC dùng câu hỏi của SIB-200, ý định du lịch dùng câu hỏi của MASSIVE.
- **Độ phủ thấp:** 4.002 dòng tiếng Việt của train_v0 chỉ có 16 câu hỏi khác nhau, gần như không có văn bản dài, và chỉ có 2–4 lựa chọn.

## 3. Cách đánh giá: đúng và công bằng, nhưng độ phân giải thấp

- Bench và cách chấm đúng: test chưa ai chạm, batch 1 cho mọi model, CI theo cụm, adapter của model ngoài khớp hàm predict() của chính gói đó. Không tìm thấy lỗi nào làm điểm dự án lệch quá 1 điểm.
- **Độ phân giải thấp:**
  - sai số chuẩn khoảng 1,2 điểm;
  - nhiễu do seed khoảng ±1;
  - một câu XCOPA nặng 0,17 điểm.

  Với 1 seed, cổng "≥ 84 và CI không chứa 0" sẽ đánh trượt một cải thiện thật 2–3 điểm. → Cần **2 seed** cho mọi lần quyết định.
- **Bảng độ trễ 33 / 52 / 88 ms không so được với nhau:**
  - decider và Quyet chấm khi tắt CUDA graphs, trong khi thẻ của decider ghi 3,2 ms khi bật;
  - 33 ms của ta là trung vị, bị prompt ngắn chi phối. p90 trên dev đã là 47 ms, và ViFactCheck có prompt dài khoảng 887 token.

  → **Quyết định chốt Qwen3-1.7B vì độ trễ (12/10 và 17/10) dựa trên số đo chưa so được, cần đo lại.**

## 4. Chiến lược và lịch sử

- **Phân bổ công sức:** khoảng 6 ngày cho benchmark, 2 ngày cho dữ liệu, dưới 1 ngày cho train. Phần benchmark làm tốt và có giá trị, nhưng model mới được thử rất ít.
- **Quyết định đáng xem lại:**
  - chốt Qwen3-1.7B dựa trên độ trễ chưa so được;
  - đặt cổng vượt độ phân giải của 1 seed;
  - train C trước A và B, nên không quy được nguyên nhân;
  - khoảng 5 triệu token Opus cho MLQA và Global-PIQA mà chưa vào bản nào.
- **Việc đúng, cần giữ:**
  - đóng băng test;
  - khử rò rỉ;
  - sửa loss sang full_vocab;
  - soát nhãn mù;
  - bộ phép thử probe, đã bắt được lỗi P4 mà điểm tổng không thấy.

## 5. Chỗ điều phối và Fable khác nhau, và cách điều phối chọn

Ba bản Fable thống nhất về chẩn đoán, chỉ khác về **thứ tự làm**:
- **Mảng phương pháp:** tăng quy mô dữ liệu tiếng Anh trước (50k → 200k dòng).
- **Mảng dữ liệu:** sửa khung dữ liệu tiếng Việt trước (C′).
- **Mảng chiến lược:** đo một ngày không train trước (G0), rồi đến C′, rồi tăng quy mô.

**Điều phối chọn thứ tự của mảng chiến lược.** G0 rẻ và có thể làm đổi cả quyết định model nền. C′ sửa các lỗi đã chỉ ra được nguyên nhân. Tăng quy mô tốn GPU nhất nên làm sau cùng, trên nền đã chọn.

## 6. Lộ trình đề xuất (chờ người dùng duyệt)

| Bước | Việc | Chi phí | Tiêu chí |
|---|---|---|---|
| **0. Sửa lỗi** (không train) | Ngừng xáo câu thang điểm và câu Có/Không; đổi mặc định loss sang full_vocab; nâng trần văn bản lên 2.048 token; sửa nguồn chữ nhãn (lấy theo dev, bản đã duyệt); dựng tập chọn model riêng; quy tắc 2 seed | Vài giờ worker | — |
| **G0. Đo** (không train) | Chấm Qwen3.5-2B-Base và Qwen3-4B gốc trên dev; đo lại độ trễ cho 1.7B, 2B, 4B và decider **cùng một cách**, có CUDA graphs, báo cả trung vị và p90 | Khoảng 2–3 giờ GPU | Quyết model nền: 4B dưới 40 ms (p90) thì cân nhắc |
| **1. C′** | Dữ liệu đúng khung bench: kiểm chứng đúng câu hỏi bench, nhãn "không đủ thông tin" khoảng 30%, thêm cặp văn bản lạc đề; bỏ khoảng 1.109 dòng không có văn bản; đưa câu hỏi đọc hiểu vào ô câu hỏi; bỏ 21 dòng VSMRC lỗi; bỏ hoặc đổi VNTC. **2 seed** | Khoảng 6–7 giờ GPU | P4 ≥ 85%; VI tăng so với C; không thói quen xấu |
| **2. Tăng quy mô** | Thêm tiếng Anh từ tasksource lên 50k rồi 100k dòng (đã khử rò rỉ, lọc giấy phép), giữ phần tiếng Việt của C′ | Khoảng 5–9 giờ GPU mỗi mức | VI tăng ≥ 2 điểm so với C′ (trung bình 2 seed) |
| **Điểm dừng, khoảng ngày 10** | Nếu không công thức nào vượt C ≥ 2 điểm: hạ mục tiêu (khoảng 82,5, kèm lợi thế không dấu và tốc độ), hoặc chuyển trọng tâm sang hoàn tất và công bố benchmark | — | Người dùng quyết |

## 7. Cần người dùng quyết

1. Duyệt **bước 0** (sửa lỗi) và **G0** (đo, không train).
2. Đồng ý **mở lại câu hỏi model nền**, nếu G0 cho thấy 2B hoặc 4B vẫn đạt độ trễ.
3. Đồng ý bỏ quy tắc **"giữ cố định 16.000 dòng tiếng Anh"** (đã duyệt 17/10), để làm được bước 2.
4. Đồng ý quy tắc **2 seed** cho mọi lần train dùng để quyết định.
5. **C′ có cần duyệt chữ tiếng Việt mới không:**
   - nếu chỉ dùng chữ của bench đã duyệt: không;
   - nếu đổi câu hỏi VNTC: 1 chuỗi;
   - nếu recast banking77 và clinc: khoảng 74 chuỗi, để sau.
