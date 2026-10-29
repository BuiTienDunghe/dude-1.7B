# Phản biện bản phân tích trước train (bản 2) — Fable 5.1, 28/10/2026

Chỉ đọc. Mọi số tự tính bằng script CPU trên file có sẵn (`phan_bien_kiem*.py`, scratchpad phiên này). Không chạy model, không đụng test.

## Kết luận trước

**"2a đáng train với đích đã sửa" không đứng được như một thí nghiệm riêng tốn 3 lần train.** Ba lý do: (i) thiết kế đã sửa chỉ dùng thông tin teacher ở khoảng 1% dòng VI và 0,08 khối lượng xác suất trên dòng đồng ý; (ii) soát mù không chứng minh "nhãn gốc sai", nó chứng minh "một LLM khác đồng ý với LLM teacher trên dòng dịch và dòng nhãn gộp"; (iii) ở kiểm chứng, nơi phân tích đặt kỳ vọng, teacher **yếu đúng ở NEI cùng chủ đề** (recall 0,29 trên vitaminc, cùng chữ nhãn với dev) nên không dạy được điều C cần. Phần lọc dữ liệu của 2a thì đúng, nhưng là vệ sinh bước 0, không phải thí nghiệm.

**Ba sửa đổi quan trọng nhất:**
1. **Tách 2a.** Lọc (bỏ 3 nguồn không văn bản, bỏ 146 dòng bất đồng tự tin ngoài đọc hiểu, soát lại dòng dịch) → đưa thẳng vào C′ như bước 0. Đích mềm 0,6·teacher + 0,4·nhãn → hoãn: kỳ vọng ≈ 0 (decider đo 0,002) và cổng hiện tại không phát hiện được +0,5–2.
2. **Bước tiếp theo là 2b lõi:** C′ đã lọc + tăng cường "từ chối trả lời" kiểu decider (chỉ code) + khung kiểm chứng theo bench (NEI ≥ 30%). Đích đo chính: P4 và **NEI recall** trên ViFactCheck/ViWikiFC dev (C hiện 0,21/0,44; Quyet 0,82/0,73); VI macro với CI làm điều kiện không tụt. proxy_v1 (300 cặp kiểm chứng, 94 nhãn NEI) làm phép đo phụ đúng chỗ giả thuyết.
3. **Sửa soát mù:** thêm đối chứng (30 dòng teacher *đồng ý*, 30 dòng bất đồng *không tự tin*), tách dòng dịch (`ts:`) với VI gốc, và hỏi "đúng theo quy ước bench" thay cho "đúng". Chi phí: chỉ token Opus.

## Số liệu đã kiểm

- Mục 1–2 (36/22/29/12/31/28; teacher sai tự tin 4/11/1/1/14/14): **khớp** preds dev có dấu.
- Mục 3: SE 1 model VI 1,10–1,19; SE hiệu theo cặp 0,99 (M1–C), 1,14 (C–Quyet); không dấu 1,37/1,36; EN 1,06–1,12/0,80 (bootstrap phân tầng nguồn, B = 2.000): **khớp**. Bảng xác suất qua cổng đúng với SD tổng 1,2–1,4.
- Mục 7, 8: A 0,811, S 0,138, D90 0,0164, bỏ 3 nguồn → 13.931 dòng, A 0,830; bảng soát mù 22/36/2; các hiệu G0: **khớp**.
- Sai: PHAN_TICH:223 "bỏ dòng khoảng 1,6% VI" → thật là **146/15.027 = 0,97%** (1,6% là D90 toàn VI, gồm 101 dòng đọc hiểu được giữ). PHAN_TICH:206 "28 câu chắc chắn và không gắn cờ" → là "không gắn cờ", gồm 12 câu độ chắc "vừa"; chỉ "cao" + không cờ = **16 câu: teacher 11, nhãn 5**.

## Các vấn đề

### 1. Thiết kế không dùng thông tin mà chẩn đoán gọi là "đáng kể" — cao
- VI bất đồng 18,9%, trong đó **17,3% không tự tin** → thiết kế giữ nhãn (PHAN_TICH:225), tức bỏ qua. Phần dùng thật: bỏ 146 dòng (0,97%) và làm mềm 0,6·S ≈ 0,08 trên dòng đồng ý — đúng cơ chế decider thấy không tăng điểm.
- Ngưỡng "A ≤ 0,90 → 2a đáng train" đo *độ bất đồng*, không đo *teacher tốt hơn nhãn*: sau khi bỏ 3 nguồn ai cũng công nhận là hỏng, A vẫn 0,830 và ngưỡng vẫn bật.

### 2. Bất đồng ở kiểm chứng đi ngược hướng P4; teacher yếu đúng chỗ khó — cao
- PHAN_TICH:216 nói nhãn gốc "lệch quy ước, khớp với lỗi P4". Dữ liệu: nli_fever có 115 nhãn "trung lập", teacher dự đoán 63; 66/115 dòng NEI bị đổi thành "mâu thuẫn/kéo theo". vinli/relation: 156 → 100. 5/10 dòng soát mù kiểm chứng là kiểu này và người soát theo teacher ở cả 5. Teacher **bớt NEI**, còn lỗi P4 của C là *thiếu* NEI.
- Ánh xạ noul không phải nguyên nhân: A ở noul (0,830) cao hơn choice (0,806); lệch Không→Có 350 so với Có→Không 160 là cùng thiên hướng "kéo theo" như ở nli_fever, không phải lỗi ánh xạ.
- NEI recall của teacher trên train: nli_fever 0,43; **vitaminc 0,29, precision 0,30**, với cùng chữ nhãn "Văn bản không đủ thông tin" như dev. Trên dev Quyet đạt 0,82/0,73 và P4 97,5%. Đọc: Quyet giỏi NEI "lạc đề", kém NEI "cùng chủ đề, thiếu dữ kiện". Bất đồng ở vitaminc là teacher sai, không phải nhãn sai. Đây là cách giải thích khác cho A = 0,811.
- Công bằng: tập 146 dòng bị bỏ chỉ có 7 dòng gold NEI (6 ở kiểm chứng); đích mềm trên dòng đồng ý còn tăng nhẹ khối lượng NEI (205 → 237 trong 1.191 dòng 3 nhãn). Thiết kế đã sửa không phá NEI, chỉ không giúp.
- Sửa: dạy NEI bằng dữ liệu dựng (abstain kiểu decider + cặp cùng chủ đề thiếu dữ kiện), không bằng teacher.

### 3. Soát mù thiếu đối chứng, lệch về dữ liệu dịch, người soát không độc lập — cao
- 41/60 dòng soát và 160/247 dòng D90 VI là dòng dịch bằng Sonnet. A trên VI dịch 0,729, VI gốc **0,876**. Trên chính các bộ tasksource bản tiếng Anh A cũng chỉ 0,688 dù Quyet đạt 94,5 EN trên dev → bất đồng là tính chất của *bộ nguồn* (health_fact 0,52, headline_cause 0,63, conll 0,40, math_qa 0,47), không phải tri thức mới của teacher. 10 dòng kiểm chứng soát mù gồm 7 nli_fever dịch; kết luận "nhãn gốc ở kiểm chứng/NLI thường sai" thực ra nói về một nguồn dịch, không nói gì về kiểm chứng VI gốc (vianli/vinli A 0,77–0,88).
- VNTC: 6/7 dòng "teacher đúng" có nhãn gốc "chính trị - xã hội" (mục báo gom chung), người soát gắn cờ mơ hồ 5/6 dòng đó. Đó là quy ước nguồn; teacher và người soát đều là LLM và cùng thích chủ đề cụ thể. "Trùng teacher" ≠ "nhãn sai".
- Không có mẫu đối chứng nên 36/60 chưa có mốc so.

### 4. Độ nhạy không đủ cho chính kỳ vọng — cao
- Kỳ vọng +0,5 đến +2 (PHAN_TICH:230) nhưng cổng bắt 50% ở +2, gần 0 ở +0,5. Ba lần train (6–9 giờ GPU) nhiều khả năng cho kết quả "không tách được".
- Sửa: đo ở chỗ giả thuyết dự đoán hiệu ứng lớn (P4, NEI recall: M1→C lệch 30 điểm, dễ bắt). proxy_v1 thêm 769 dòng VI có dấu, giảm SE do mẫu câu khoảng 20%; giá trị chính là 300 cặp kiểm chứng ngoài dev. Người dùng quyết vì proxy còn dùng chọn mốc.

### 5. G0: đọc đúng ngưỡng, nhưng bằng chứng về 2B-Base yếu — trung bình
- 2B-Base là model Base chấm bằng chat template mượn; decider dùng prompt thô và nâng 68,2 → 86,5. Zero-shot của model Base dự báo kém khả năng sau fine-tune; độ trễ 53 ms là cận trên không FLA. "Giữ 1.7B" vẫn hợp lý vì chi phí đổi nền; nên ghi là quyết định theo chi phí, không phải vì 2B-Base "không đạt".

### 6. Nhỏ
- Dòng noul suy từ MCQ (`noul-label-verification`, 538 dòng): ví dụ S001 (RACE, "cây thông kỳ lạ") nhãn "Không" đúng theo cách dựng, sai theo nghĩa thường. Lỗi ở bộ sinh (distractor được văn bản xác nhận); sửa ở bộ sinh, không bằng teacher.

## Trả lời 4 câu hỏi

1. **2a:** không đứng được. A = 0,811 giải thích được bằng bộ nguồn tasksource nhiễu và bản dịch, teacher yếu ở NEI cùng chủ đề, và quy ước nhãn gộp. Quy tắc bỏ dòng an toàn nhưng tác dụng ≈ 0.
2. **Thiết kế:** không tách được nguyên nhân vì cả ba thay đổi đều dưới ngưỡng phát hiện. Lọc là bước 0; đích mềm hoãn; λ khác không đổi bản chất (thông tin ≈ λ·S); đo bằng P4/NEI recall + proxy kiểm chứng.
3. **Số liệu:** mục 1–3, 7, 8 khớp; hai sai nhỏ ở trên.
4. **Bước tiếp theo:** **2b lõi, gộp phần lọc của 2a vào C′; không chạy 2a riêng.** Nếu đủ GPU: 2 seed mỗi bên; nếu không: C-s2 + 1 seed 2b, đọc P4/NEI trước. Rủi ro lối tắt "lạc đề → NEI" (HOC_TU_DECIDER mục 7.1) là thật — kết quả vitaminc của chính teacher cho thấy NEI cùng chủ đề mới là chỗ khó — nên cần cả cặp cùng chủ đề thiếu dữ kiện; phần này cần chữ Việt mới, người dùng duyệt. Đích mềm nếu muốn thử thì thêm sau, làm biến duy nhất trên 2b, đo bằng hiệu chỉnh (NLL, nhiệt độ), không bằng điểm.
