# Kiểm toán 27/10: đánh giá cách đo và chiến lược tổng thể

Ngày 27/10/2026. Người viết: Fable 5.1, mảng **đánh giá và chiến lược**. Hai mảng khác (phương pháp/hàm chuyển; dữ liệu train) do agent khác lo, ở đây chỉ dẫn chiếu, không làm trùng.

**Phạm vi và ràng buộc đã giữ:** chỉ đọc repo; không chạy model, không train; **không mở test 1.3** (không đọc file test lẫn `manifest_test_v1.3.json`; mọi số về test lấy từ `manifest_dev_v1_3.json` và các báo cáo đã dẫn); không sửa file nào ngoài file này. Số liệu lấy từ các báo cáo trong `results/` (ghi rõ nguồn), cộng một script đếm riêng chạy ngoài repo trên `data/bench_v1_3/dev_v1_3.jsonl`, `results/train_v1/components/C.jsonl` và `data/train_v0.jsonl` (`C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_strategy_counts_2710.py`). Hai thẻ model ngoài (decider-2b bản ghim `ff99089b`, Quyet-1.0-Medium) được đọc trên Hugging Face; không gửi dữ liệu dự án ra ngoài.

Mức chắc chắn ghi theo thang: **cao** (số đếm được, tái lập từ file), **vừa** (suy ra từ số đã có, còn một giả định), **thấp** (phán đoán).

---

## 0. Kết luận

1. **Cách đo về cơ bản đúng và công bằng; không thấy lỗi nào làm điểm của dự án thấp hay cao giả quá 1 điểm** (mức: cao). Test 1.3 chưa model nào chạm; dev đóng băng theo sha; batch 1 cho mọi model; CI bootstrap theo cụm, rút lại riêng từng nguồn để giữ cơ cấu trung bình đều; McNemar chính xác theo cặp; adapter model ngoài khớp `predict()` công khai đến 4 chữ số; decider được dùng cách ánh xạ có lợi hơn ("text"). Soát nhãn mù 134 cặp không tìm thấy nhãn sai nào; bỏ 26 câu mơ hồ thì mọi model cùng tăng khoảng 1 điểm, thứ hạng không đổi. Dòng dev bị người sửa (598 dòng VI) không nghiêng về dự án (70,7 ở dòng đã sửa so với 72,1 ở dòng giữ nguyên).

2. **Điểm yếu thật của phép đo là "độ phân giải", không phải độ công bằng** (mức: cao). Trung bình đều 6 nguồn trên 1.430 cặp VI có sai số chuẩn khoảng 1,2 điểm, tức khoảng tin cậy ±2,3 cho một model (khớp CI đã báo [77,4; 82,0]). Một câu XCOPA (n = 100) nặng 0,167 điểm macro; một câu ViFactCheck (n = 158) nặng 0,105. Nhiễu giữa hai seed của 1.7B khoảng ±1. Với sai số chuẩn của chênh theo cặp khoảng 1,0 điểm và độ lệch giữa seed khoảng 0,65 (hai seed chênh 0,9), một model có điểm *thật* đúng 84,0 sẽ cho số dưới 84,0 ở khoảng một nửa số lần train **một seed**; một cải thiện thật +2–3 điểm sẽ cho CI chứa 0 và bị kết luận "không đạt" dù có thật (một bước +4,1 có thật thì qua được). Cổng theo nguồn (ViFactCheck tăng với p < 0,05 trên n = 158) đòi +4–8 điểm trên nguồn đó, tùy số cặp bất đồng.

3. **Có ba chỗ lệch giữa train và bench mà cách đo không gây ra nhưng làm số của dự án thấp đi một cách có thể sửa được** (mức: cao, đếm trực tiếp trên C.jsonl):
   - **0 dòng** tiếng Việt trong C dùng hai khung câu hỏi kiểm chứng của dev; 593 dòng vitaminc dùng ba chữ nhãn của dev nhưng với khung câu hỏi khác, và chữ NEI là bản ngắn "Văn bản không đủ thông tin" trong khi dev dùng bản dài "…để kết luận" (dev: 241 dòng bản dài, 0 bản ngắn).
   - Hai khung câu hỏi mà train **có** dùng chung với dev ("Văn bản này thuộc chủ đề nào?", "Người dùng muốn trợ lý làm gì?") lại đi với **bộ nhãn khác hẳn**: 1.000 dòng VNTC với 10 nhãn (trùng dev đúng hai chữ "sức khỏe", "thể thao"), 1.000 dòng ý định du lịch với 8 nhãn. SIB-200 của bản C tụt −2,7 đúng ở hai nhãn trùng chữ.
   - Nguyên nhân gốc của lệch chữ NEI là **mất nguồn sự thật duy nhất**: `config/bench_v1_questions.json` trên đĩa vẫn ghi bản ngắn; bản dài chỉ nằm ở lớp ghi đè `data/human_edits/` (quyết định 15/10); phần B được dịch "theo bench" từ config nên mang chữ cũ. Đây là lỗi quy trình, không phải lỗi đo.

4. **Giới hạn chính của dự án không phải "thiếu dữ liệu tiếng Việt", cũng chưa chứng minh được là "model quá nhỏ"; nó là công thức và quy mô huấn luyện dạng quyết định** (mức: vừa). Bằng chứng: (a) decider-2b (1,9B, thẻ model ghi **chỉ tiếng Anh**, không có tiếng Việt) hơn dự án 6,6 điểm VI có dấu; (b) Qwen3.5-2B gốc và Qwen3-1.7B gốc **bằng nhau** trên VI (70,6 = 70,6), nên lợi thế của decider không đến từ model nền "giỏi tiếng Việt hơn"; (c) decider được huấn luyện giám sát trên khoảng 95 bộ dữ liệu quyết định công khai, thêm một giai đoạn RL hiệu chỉnh và LoRA rank 64 trên 42,7 nghìn dòng, trong khi dự án dùng LoRA rank 16, 2 epoch, 20–31 nghìn dòng; (d) thêm 11 nghìn dòng VI (gấp 3,75 lần phần VI) chỉ cho +1,77 với CI chứa 0, và đường 4 mốc 25/50/75/100 % của bản C (81,07 / 80,17 / 81,38 / 81,65) **phẳng từ mốc 25 %**, val loss epoch 1 → 2 là 0,758 → 0,757. Cả hai dấu hiệu nói "thêm cùng loại dữ liệu nữa không giúp"; chúng không nói "model hết sức", vì chưa có thí nghiệm nào đổi công thức (rank, epoch, lr, quy mô EN) hay đổi nền. Lưu ý cần giữ: Qwen3.5-2B-**Base** (nền thật của decider) chưa từng được chấm (người dùng từ chối 12/10), nên "dung lượng/nền" chưa bị loại trừ.

5. **Ràng buộc ≤ 40 ms, vốn là lý do khóa Qwen3-1.7B, đang dựa trên số đo không so được với nhau** (mức: vừa-cao). 33 / 52 / 88 ms của dự án / decider / Quyet là ba đường chạy khác nhau: decider chấm với `use_graphs=False` và không có kernel flash-linear-attention (thẻ model của chính decider ghi 3,2 ms p50 khi bật CUDA graphs + torch.compile, dù trên GPU khác); Quyet chạy qua gói `quyet` cũng không có kernel đó. Riêng 33 ms của 1.7B là số **trung vị trên toàn dev**, nơi 3/4 số câu là prompt ngắn 85–151 token; ở 128 token, 0.6B và 1.7B cùng 28–29 ms, tức phần lớn 33 ms là chi phí khởi chạy kernel, không phải tính toán. ViFactCheck có prompt p50 887 token, p90 1.386 token; theo đường độ trễ đã đo (512 token → 50 ms, 2.048 → 181 ms) các câu này chạy trên 40 ms rõ, và p90 toàn dev là 47 ms. Hệ quả: ngân sách 40 ms chưa loại được model lớn hơn, và cũng chưa được dự án đạt trên câu dài.

6. **Dự án đang dàn trải, nhưng theo một kiểu dễ hiểu: khoảng 6 ngày cho benchmark, khoảng 2 ngày chuẩn bị dữ liệu và dịch, dưới 1 ngày train** (mức: cao, theo PLAN_decisions). Từ 12 đến 17/10 là bench 1.0 → 1.3 (soát 3.055 câu, hơn 700 sửa, hai lần chạy thử MLQA/Global-PIQA tốn khoảng 5 triệu token Opus mà chưa vào bản nào, công cụ duyệt, trang duyệt từ xa, ý kiến thứ hai 10 agent, soát nhãn mù); 26/10 là bộ phép thử (150 chữ đề xuất, 148 bị bỏ; P1′ hóa ra không phân biệt được). Phía model: 3 lần train thử 12–13/10 (loss, strip, decider-ft), rồi 1 lần duy nhất ngày 26/10. Phần benchmark là mục tiêu số 2 của dự án và đã gần đến mức công bố được; phần model thì mới có đúng một thí nghiệm "đổi dữ liệu" và chưa có thí nghiệm "đổi công thức" nào.

7. **Mục tiêu 84 với 1.7B là khả thi về nguyên tắc (decider chứng minh lớp 2B làm được 86,5) nhưng không khả thi với công thức hiện tại, và không đo được bằng một seed** (mức: vừa). Ước lượng của tôi: xác suất chạm 84 trên dev trong 2 tuần nếu chỉ sửa dữ liệu C′ trên 1.7B: khoảng 30 %; nếu mở lại công thức/nền theo lộ trình mục 4: khoảng 50 %; số test có thể lệch số dev 1–2 điểm vì dev đã được dùng để chọn loss, nền, dữ liệu và cổng.

8. **Khuyến nghị số 1:** trước khi train thêm bất kỳ bản nào, chạy một **gói đo một ngày, không train** (mục 4, phương án G0): chấm Qwen3.5-2B-Base và Qwen3-4B gốc trên dev 1.3 bằng đúng `eval_base.py`; đo lại độ trễ cho 1.7B / 2B / 4B / decider trên **cùng một đường** (bật CUDA graphs, cài kernel FLA cho họ Qwen3.5), báo theo nguồn và p90, không chỉ trung vị; sửa nguồn sự thật chuỗi nhãn; và đổi quy tắc cổng thành **2 seed cho mọi bản thử**. Gói này trả lời câu hỏi lớn nhất còn mở (giới hạn ở nền hay ở công thức) với chi phí nhỏ hơn một lần train, và quyết định mọi phương án sau nó.

---

## 1. Câu 1: cách đo có đúng và công bằng không

### 1.1 Những gì đúng (mức: cao)

| Khía cạnh | Thực tế trong repo | Nguồn |
|---|---|---|
| Test | 1.553 cặp, đóng băng sha `2bdbd210…`, chưa model nào chạy; chỉ được *đọc văn bản* một lần để khử rò rỉ (quyết định (a) 17/10). 317 cặp test đã được người xem khi soát dịch; 13 nhãn test đổi theo soát của người | `manifest_dev_v1_3.json`, `REPORT_bench_v1_2.md` mục 7, `BRIEF_train_chinh_thuc.md` |
| Dev | 1.430 cặp VI (Belebele 294, SIB 299, MASSIVE 294, XCOPA 100, ViFactCheck 158, ViWikiFC 285), mỗi cặp có bản không dấu; 987 cặp có EN. NFC: 0 trường lệch (đếm lại) | script đếm |
| Điểm chính | Trung bình đều 6 nguồn (VI), 4 nguồn (EN). Macro so với micro lệch nhỏ và không nghiêng: dự án 79,9/80,8; decider 86,5/86,9; Quyet 89,6/89,2 | `REPORT_bench_v1_3_trial.md` mục 7 |
| CI | Bootstrap 1.000 lần theo cụm (cùng đoạn văn = cùng cụm), rút lại riêng từng nguồn để giữ cơ cấu macro; seed riêng mỗi phép so (đã sửa 26/10) | `trend_C.py`, `analyze_probe.py` |
| McNemar | Chính xác, nhị thức hai phía trên cặp câu bất đồng | `trend_C.py` |
| Batch 1 | Mọi model; lượt batch 16 chỉ để đo thông lượng | `eval_base.py`, `harness.py` |
| Adapter model ngoài | Ánh xạ "text" và "keyed", lấy cách cao hơn cho model ngoài; khớp `predict()` công khai đến 4 chữ số; nhiệt độ phát hành; không cắt câu (decider nâng cap 1.536 → 32.768; Quyet strict 0 câu bị loại) | `REPORT_baselines_v1_2.md` mục 7–8 |
| noul | Lựa chọn cố định [Có, Không], chấm bằng chữ cái A/B như mọi câu; decider/Quyet dùng đường yes/no gốc của chúng. Ngưỡng 0,5; chỉnh ngưỡng trên val chỉ cho +1,4 (CI chứa 0) | `REPORT_sensitivity.md` |
| Nhãn | Soát mù 134 cặp nghi ngờ: 0 nhãn sai, 28 mơ hồ (22 SIB); bỏ 26 cặp mơ hồ: mọi model +0,2…+1,6, thứ hạng không đổi | `REPORT_label_review.md` |

### 1.2 Những chỗ có thể làm số lệch, và lệch bao nhiêu

**(a) Khung câu hỏi và chữ nhãn: train khác bench** (mức: cao; đếm trực tiếp)

| Nguồn dev | Khung câu hỏi của dev | Trong C.jsonl (15.075 dòng VI) |
|---|---|---|
| ViFactCheck, ViWikiFC choice | "Dựa vào văn bản, nhận định sau được văn bản xác nhận, bị văn bản bác bỏ, hay…" | **0 dòng** dùng khung này. 593 dòng vitaminc dùng 3 chữ nhãn của dev với khung "Văn bản thứ nhất / thứ hai" + câu hỏi mẫu tasksource; nli_fever dùng nhãn NLI; health_fact không có văn bản |
| ViFactCheck, ViWikiFC noul | "Văn bản có xác nhận phát biểu sau không?" | **0 dòng** |
| SIB-200 | "Văn bản này thuộc chủ đề nào?" | 1.000 dòng VNTC, **cùng câu hỏi, 10 nhãn khác** (trùng chữ dev: "sức khỏe", "thể thao") |
| MASSIVE | "Người dùng muốn trợ lý làm gì?" | 1.000 dòng ý định du lịch, **cùng câu hỏi, 8 nhãn khác**; 0 trong 60 mô tả ý định MASSIVE có mặt |
| XCOPA | "Nguyên nhân/Kết quả nào hợp lý hơn?" | 0 dòng dùng khung; e-CARE + HeadlineCause 989 dòng khung khác → XCOPA +7,0 |
| Chữ NEI | "Văn bản không đủ thông tin **để kết luận**" (241 dòng) | 593 dòng mang bản **ngắn** "Văn bản không đủ thông tin", 0 dòng bản dài |

Hệ quả cho cách đọc số: benchmark **được phép** khác khung train (đó là mục đích "câu hỏi lạ"), nên đây không phải lỗi đo. Nhưng nó giải thích vì sao bản C tăng ở XCOPA/ViWikiFC mà đứng yên ở ViFactCheck và MASSIVE, và tụt ở SIB; và nó cho thấy dữ liệu train không được thiết kế theo bản đồ "nguồn bench ↔ dạng bài" như brief dự định. Chi tiết nguyên nhân trong dữ liệu (health_fact không văn bản, NEI 12,8 %, v.v.) đã có ở `results/train_v1/phan_tich_C/FABLE_doc_ket_qua.md` mục 3 và thuộc mảng dữ liệu.

Nguồn gốc lệch chữ NEI (mức: cao): `config/bench_v1_questions.json` ghi `"nei": "Văn bản không đủ thông tin"`; quyết định 15/10 (PLAN_decisions dòng "Chốt 78 chuỗi mẫu") áp bản dài qua lớp ghi đè `data/human_edits/`, không ghi ngược vào config; quyết định 20/10 dịch phần B "theo bench (config/bench_v1_questions.json)". Có **hai nguồn sự thật**, và phần dịch lấy nguồn cũ. Cũng vậy, `translations.json` và `bench_v1_questions.json` vẫn mang `_status: CHƯA ĐƯỢC NGƯỜI KIỂM` dù chuỗi đã được duyệt trên công cụ. Khuyến nghị: một script xuất "chuỗi hiệu lực" từ edits_log thành file đọc-chỉ, và mọi bước dựng train/probe đọc file đó.

**(b) Độ phân giải của dev** (mức: cao)

| Nguồn | n cặp | SE accuracy (điểm) tại mức M1 | 1 câu đổi = bao nhiêu điểm macro | Trọng số macro / micro |
|---|---|---|---|---|
| Belebele | 294 | 2,2 | 0,057 | 0,167 / 0,206 |
| SIB-200 | 299 | 2,2 | 0,056 | 0,167 / 0,209 |
| MASSIVE | 294 | 2,0 | 0,057 | 0,167 / 0,206 |
| **XCOPA** | **100** | **4,1** | **0,167** | 0,167 / 0,070 |
| **ViFactCheck** | **158** | **3,5** | **0,105** | 0,167 / 0,110 |
| ViWikiFC | 285 | 2,5 | 0,058 | 0,167 / 0,199 |
| **Macro VI** | 1.430 | **1,17 → ±2,3 (95 %)** | | |

- Hai nguồn nhỏ nhất chiếm 1/3 điểm macro nhưng chỉ 18 % số câu; XCOPA +7,0 của bản C là 7 câu (p = 0,12). Không thể nới XCOPA (split validation chỉ có 100 câu) và ViFactCheck dev bị giới hạn vì dùng chung bài báo với test. Vậy phải sống với ±2,3 bằng cách **tăng seed**, không phải tăng dev.
- Cổng đã chốt ("≥ 84,0 và CI chênh với 79,9 không chứa 0"): CI chênh một seed rộng khoảng ±2 (bản C: [−0,28; +3,66]). Một bước +4,1 có thật thì qua được (CI khoảng [+2,1; +6,1]); nhưng một cải thiện thật +2–3 cho CI chứa 0, và một model có điểm thật đúng 84,0 rơi dưới 84,0 ở khoảng nửa số seed. Cổng theo nguồn "ViFactCheck tăng, p < 0,05" trên n = 158 đòi số câu lật ròng tùy tổng số cặp bất đồng (0 đúng→sai / 6 sai→đúng đã cho p ≈ 0,03; bản C có 8 đúng→sai / 7 sai→đúng nên p = 1), tức khoảng +4–8 điểm trên nguồn. Hai cổng này được đặt trước khi có số (tốt), nhưng đặt vượt độ phân giải của phép đo (xấu). Hệ quả thực tế ở bản C: ViWikiFC +4,2 (p = 0,050) là mức tăng lớn nhất có thể kỳ vọng ở một nguồn, và nó vừa đủ chạm p = 0,05.

**(c) Độ trễ** (mức: vừa-cao)

| Số đang dùng | Cách đo | Vì sao không so được |
|---|---|---|
| Dự án 32,6–33,5 ms | `eval_base.py`, trung vị toàn dev, forward đơn thuần, không CUDA graphs | Trung vị bị prompt ngắn chi phối (SIB p50 129 token, MASSIVE 151, XCOPA 85). 0.6B và 1.7B cùng 28–29 ms ở 128 token (REPORT_ft_v0) → phần lớn là chi phí cố định mỗi lượt. p90 toàn dev 47 ms. ViFactCheck p50 887 / p90 1.386 token; theo đường 512 → 50 ms, 2.048 → 181 ms thì nguồn này chạy khoảng 60–120 ms |
| decider-2b 52 ms | `eval_decider_v1.py`: `use_graphs=False`, không flash-linear-attention, tính cả tách token | Thẻ model của decider ghi 3,2 ms p50 (CUDA graphs + torch.compile, GPU B300). Kiến trúc Qwen3.5 lai linear attention chạy bản fallback PyTorch nên chậm giả |
| Quyet-Medium 88 ms | gói `quyet`, không FLA | Như trên; Qwen3.5-4B |
| Qwen3.5-2B gốc 58 ms | `eval_base.py` | Không FLA |

Kết luận cho câu 1: **không có bằng chứng cách đo làm số dự án thấp hay cao giả** quá 1 điểm; bảng độ trễ là bảng duy nhất không công bằng, và nó bất lợi cho model ngoài, nhưng lại được dùng để **khóa** lựa chọn nền của dự án (mục 3).

**(d) Những điểm nhỏ hơn, chỉ cần ghi chú**

- decider-2b có thể đã thấy MASSIVE en-US (thẻ ghi "trained"); bỏ MASSIVE, dự án vẫn kém 6,0 [3,3; 8,6]; Quyet không công bố dữ liệu train. Dự án là model duy nhất có khử rò rỉ kiểm được. Khoảng cách có thể bị **thổi phồng nhẹ** theo hướng bất lợi cho dự án, không quá 1 điểm.
- "EN − VI" so 4 nguồn với 6 nguồn: với dự án, hai nguồn bản địa là hai nguồn yếu nhất (72,8; 76,5), nên 5,5 điểm "tụt ngôn ngữ" gồm cả phần "tụt theo nguồn". Không ảnh hưởng điểm chính.
- NFC: dev 0 lệch; C.jsonl 216 dòng VI (0,7 %) không NFC (PhoLaya cũ + ViANLI). Không đáng kể.
- Dev đã được dùng để chọn loss, nền, dữ liệu, cổng và kịch bản: điểm dev của dự án có xu hướng **lạc quan** nhẹ so với test; model ngoài không bị vậy. Đây là lý do số test (một lần) sẽ là số công bằng nhất.
- Train có 1.510 dòng `score` (5 %) nhưng bench 1.3 không còn câu score (ViHSD bỏ ở 1.1): phần này không đóng góp cho mục tiêu chính.
- Bộ phép thử probe_v1 tốt về ý đồ (P4 bắt được K4), nhưng P1′ không phân biệt được K2 (mọi model đều tăng khi đổi bộ nhãn), và P4 noul không phân biệt được model. Nên bỏ hai phép này khỏi định nghĩa K1/K2 chính thức trước lần chấm sau.

---

## 2. Câu 2: mục tiêu có thực tế không; giới hạn chính ở đâu

### 2.1 Khoảng cách cần vượt

| | VI có dấu (macro) | Cách 84,0 | Cách decider 86,5 | Cách Quyet 89,6 |
|---|---|---|---|---|
| M1 (train_v0, 20 k dòng, 4 k VI) | 79,9 | +4,1 | +6,6 | +9,7 |
| C (31 k dòng, 15 k VI) | 81,65 (+1,77 [−0,28; +3,66]) | +2,35 | +4,85 | +7,95 |

+4,1 điểm macro = **+24,6 điểm nguồn** gộp lại. Chỗ có thể lấy (chênh decider − dự án trên dev 1.2): ViFactCheck 10,8; MASSIVE 9,5; XCOPA 8,0; ViWikiFC 6,3; Belebele 5,4; SIB −0,3. Bản C lấy được: XCOPA +7,0, ViWikiFC +4,2, Belebele +1,4, MASSIVE +1,4; mất SIB −2,7, ViFactCheck −0,6.

### 2.2 Đường xu hướng: những gì ta có và không có

| Bằng chứng | Số | Đọc thế nào |
|---|---|---|
| Fine-tune lần đầu (train_v0) | 1.7B gốc 72,9 → 79,8/80,8 (2 seed, dev_v0); M0 70,6 → M1 79,9 (dev 1.3) | Bước nhảy lớn nhất đến từ "dạy định dạng". Hai seed chênh 0,9 |
| Gấp 1,5 lần dữ liệu, gấp 3,75 lần phần VI (bản C) | +1,77, CI chứa 0 | Bước nhỏ, chưa chắc có thật |
| Đường 4 mốc trong một lần train | 81,07 / 80,17 / 81,38 / 81,65 | **Phẳng từ 25 % số bước** (đã thấy 15,5 k dòng); không có dấu hiệu "còn tăng nếu thêm" |
| Val loss epoch 1 → 2 | 0,758 → 0,757 (M1: 0,756 → 0,776) | Bão hòa theo công thức hiện tại |
| decider + train_v0 (decider-ft-ep2) | 86,5 → 85,1 | Dữ liệu kiểu train_v0 **làm giảm** một model mạnh: giá trị của nó nằm ở định dạng, không ở nội dung |
| Qwen3.5-2B gốc so với Qwen3-1.7B gốc | 70,6 = 70,6 (VI); 84,5 ≈ 84,6 (EN) | Nền mới hơn không "giỏi tiếng Việt hơn" ở trạng thái gốc |
| decider-2b (chỉ tiếng Anh) | 86,5 VI | Năng lực quyết định học từ ~95 bộ tiếng Anh **chuyển sang tiếng Việt** được; lớp 2B đủ sức cho 86 |

Không có thí nghiệm nào trong repo đổi **công thức** (rank, epoch, lr, quy mô dòng tiếng Anh, full fine-tune) hay đổi **nền**. Vì vậy câu "giới hạn là dung lượng model" **chưa kiểm được**; câu "giới hạn là thiếu dữ liệu tiếng Việt" **bị bằng chứng phản lại** (decider không có tiếng Việt; thêm 11 k dòng VI gần như không đổi). Phần còn lại là **công thức + quy mô + thiết kế dữ liệu** (mức: vừa).

Một cách đọc tỉ lệ (mức: thấp, chỉ để định cỡ): dự án huấn luyện 31 k dòng, LoRA r16, 2 epoch; decider huấn luyện giám sát trên ~95 bộ (một mô tả cũ của cùng model ghi 942 k ví dụ / 64 bộ), rồi RL hiệu chỉnh, rồi LoRA r64. Khoảng cách quy mô cỡ 10–30 lần. Kho tasksource @ bf1e4a87 có 1,03 triệu dòng; với 0,165 giây/dòng/epoch đo được, 150 k dòng × 1 epoch là khoảng 7 giờ GPU: **nằm trong tầm tay** của một tuần, nhưng chưa ai thử.

### 2.3 Mục tiêu 84 và ràng buộc 40 ms

- **84 trên dev 1.3 với Qwen3-1.7B + LoRA r16 + sửa dữ liệu (C′):** Fable đã ước ở `FABLE_doc_ket_qua.md` mục 5: sửa P4/NEI có thể cho +1,4, SIB về mức cũ +0,45, cộng +1,77 hiện có → 83–84, "sát ngưỡng với nhiễu ±1". Tôi giữ ước lượng đó và đặt xác suất khoảng 30 %. Rủi ro lớn: điểm tăng do lặp đi lặp lại trên dev là một dạng khớp dev; test có thể thấp hơn 1–2 điểm.
- **84 nếu đổi công thức hoặc nền:** decider cho thấy 86 là có thật ở lớp 2B. Xác suất khoảng 50 % trong 2 tuần, với điều kiện gói đo G0 (mục 4) cho tín hiệu dương.
- **≤ 40 ms:** như mục 1.2(c), ràng buộc này hiện vừa **chưa được đo đúng** (trung vị thay cho theo nguồn/p90; đường chạy không tối ưu) vừa **không phân biệt được model** (phần cố định chiếm phần lớn). Trước khi dùng nó để loại Qwen3.5-2B hay Qwen3-4B, phải đo lại trên cùng đường chạy có CUDA graphs. Ước tính thô (mức: thấp): ở prompt 150 token trên RTX 5060 Ti, 4B dày đặc bf16 vào khoảng 40–45 ms không graphs và 20–25 ms có graphs; 1.7B có graphs khoảng 10–15 ms.
- **Kết luận:** mục tiêu 84 là **thực tế cho lớp model**, **không thực tế cho công thức hiện tại**, và **không đo được bằng một seed**. Nếu người dùng muốn giữ cả 84 lẫn 1.7B, phải chấp nhận đổi công thức (quy mô EN, rank, epoch) chứ không chỉ đổi phần tiếng Việt.

---

## 3. Câu 3: rà lịch sử quyết định

### 3.1 Dòng thời gian và phân bổ công

| Ngày | Phía benchmark | Phía model |
|---|---|---|
| 11–12/10 | dev_v0 (1.000 cặp) | Train thử 0.6B/1.7B × 2 seed; sửa loss full_vocab; **chốt 1.7B**; baseline decider |
| 13/10 | Bench 1.0 → 1.1 (bỏ ViHSD, sửa chữ kiểm chứng); khảo sát nguồn train | strip_v0 (không đạt); decider-ft (đổi kết luận sau 1.1); khôi phục dấu (hai vùng, không kết luận) |
| 14–16/10 | Soát dịch 3.055 câu (Sonnet → Opus → Opus); 483 + 190 đề xuất; công cụ duyệt + bảo mật; trang duyệt từ xa; ý kiến thứ hai 10 agent; MLQA/Global-PIQA 2 lần chạy thử + khu tạm (~5 triệu token Opus); bench 1.2 đóng băng; bước E chấm 9 model; soát nhãn mù 134 cặp | — |
| 17/10 | Bench 1.3 đóng băng; brief train; Giai đoạn 0 dựng dữ liệu | — |
| 18–20/10 | — | Duyệt 25 chuỗi; dịch 6.944 dòng bằng Sonnet (2 lượt), ~8 giờ |
| 26/10 | Bộ phép thử probe_v1 (150 chữ → 7 duyệt), chấm mốc 4 model | **Train thử C** (1 lần, 1 seed), đọc lỗi mù |

Tỉ lệ công: khoảng 6 ngày benchmark, khoảng 2 ngày chuẩn bị dữ liệu và dịch (17/10 Giai đoạn 0; 18–20/10 duyệt chuỗi và dịch), dưới 1 ngày train. Với mục tiêu số 2 (benchmark đáng tin), phần lớn công này **đúng chỗ** và đã tạo ra thứ gần công bố được nhất của dự án. Với mục tiêu số 1 (model), đến nay có đúng **một** thí nghiệm có chủ đích.

### 3.2 Quyết định đúng, nên giữ

- Đóng băng test trước khi train; chưa model nào chạm test; khử rò rỉ bốn phép với mọi split của mọi nguồn bench (mức: cao).
- Batch 1 làm số chính thức; một module dựng prompt dùng chung train/chấm; ghim revision; seed.
- Sửa loss full_vocab khi phát hiện letter_mass sụp (lỗi thật, sửa đúng, kiểm đẳng thức bằng số).
- Soát nhãn **mù**, ý kiến thứ hai độc lập, cổng duyệt chuỗi tiếng Việt do người quyết.
- Đặt chỉ tiêu và kịch bản đọc **trước** khi train; probe P4 phát hiện được một lỗi học thật (phán nhận định không cần văn bản).

### 3.3 Quyết định sai hướng hoặc tốn công không đóng góp vào mục tiêu chính

| # | Quyết định | Vì sao đáng xem lại | Mức |
|---|---|---|---|
| 1 | **Chốt Qwen3-1.7B ngày 12/10** trên dev_v0 (chỉ so với 0.6B), rồi **tái khẳng định 17/10** ("không cần train 2B") | Lúc chốt chưa có bench, chưa có decider/Quyet. Lúc tái khẳng định, lý do là bảng độ trễ không so được (1.2(c)); Qwen3.5-2B-Base chưa từng được chấm. Đây là quyết định có tác động lớn nhất mà bằng chứng mỏng nhất | cao |
| 2 | **Chỉ tiêu 84 và cổng CI** đặt mà không có thí nghiệm đường học (ví dụ train 25/50/100 % train_v0) | Cổng vượt độ phân giải một seed; brief ghi "có thể không đạt chỉ bằng dữ liệu" nhưng không có nhánh dự phòng được chuẩn bị | cao |
| 3 | **Train C trước A và B** (người dùng đã duyệt "ABC") | Không tách được phần bản địa và phần dịch; không quy được nguyên nhân tụt Có/Không không dấu (−7,9, p = 0,009). Tiết kiệm 5 giờ GPU, mất khả năng quy nhân quả | vừa |
| 4 | **Trần 5 %/nguồn (mức tác vụ) + "giữ nguyên 16.000 dòng EN"** | Hai quy tắc tốt cho khái quát hóa nhưng mâu thuẫn với mục tiêu "bổ sung theo dạng bài còn thiếu": VSMRC bị cắt 2.000 → 1.217; ý định chỉ 1.144 và không có nhãn kiểu MASSIVE; EN vẫn chiếm 50 % dòng và 56 % token state của C | vừa |
| 5 | **Cùng câu hỏi, khác bộ nhãn** (VNTC ↔ SIB; du lịch ↔ MASSIVE) và **khung kiểm chứng không có trong train** | Thiết kế dữ liệu không đối chiếu từng nguồn bench với "khung + bộ nhãn" của train (mục 1.2(a)). Thuộc mảng dữ liệu, nêu ở đây vì nó là hệ quả của việc không có bản đồ nguồn ↔ dạng bài khi duyệt Giai đoạn 0 | cao |
| 6 | **Công thức LoRA r16 / 2 epoch / lr 2e-4 giữ nguyên từ bước 3** | Chưa bao giờ là đối tượng thí nghiệm; decider dùng r64; val loss phẳng; không có lần nào thử full fine-tune, rank cao hơn, nhiều epoch hơn hay quy mô dòng lớn hơn | vừa |
| 7 | **MLQA và Global-PIQA**: 2 lần chạy thử + khu tạm 197 câu, ~5 triệu token Opus, soát 30 câu chờ người dùng | Chưa vào bản nào (1.4); người dùng chưa điền 30 câu. Không hại, nhưng là chi phí Claude lớn nhất không đóng góp cho cả hai mục tiêu tới giờ | cao |
| 8 | **strip_v0, decider-ft, khôi phục dấu** (13/10, 3 lần train/chấm) | strip không đạt; decider-ft đổi kết luận sau khi bench đổi; khôi phục cho hai vùng khác nhau. Cả ba đều là câu hỏi hợp lý, nhưng chúng tiêu hết "ngân sách train" của tuần đầu mà không thí nghiệm nào nhắm vào công thức chính | vừa |
| 9 | **150 chữ đề xuất P1/P5** cho người dùng duyệt | Người dùng dừng ở dòng 22; P1′ thay thế hóa ra không phân biệt được K2. Bộ phép thử vẫn có giá trị (P4, Pa), nhưng lẽ ra thiết kế tối thiểu từ đầu | vừa |
| 10 | **Hai nguồn sự thật cho chuỗi tiếng Việt** (config trên đĩa + lớp ghi đè) | Gây lệch chữ NEI trong 593 dòng train; `_status: CHƯA ĐƯỢC NGƯỞI KIỂM` còn nguyên trong config dù đã duyệt | cao |

Có **dàn trải** không? Có, nhưng không phải vì làm hai mục tiêu song song (hai mục tiêu bổ trợ nhau: không có bench 1.3 thì không ai thấy được +1,77 là nhiễu). Dàn trải nằm ở chỗ **mục tiêu model chỉ được tiếp cận bằng một đòn bẩy duy nhất (đổi phần tiếng Việt của dữ liệu)** trong khi các đòn bẩy rẻ hơn hoặc hứa hẹn hơn (đo đúng độ trễ, chấm nền khác, quy mô EN, công thức) chưa được chạm tới; và ở chỗ chi phí Claude dồn vào các lát bench phụ (MLQA, Global-PIQA) chưa dùng.

---

## 4. Câu 4: lộ trình 1–2 tuần

Giả định tài nguyên: một RTX 5060 Ti 16 GB; một lần train 31 k dòng × 2 epoch ≈ 3 giờ; chấm dev + probe ≈ 1,5 giờ; **2 seed ≈ 9 giờ** cho một bản. Hạn mức Claude theo quy tắc hiện hành (không dùng mức dùng thêm). Mọi cổng dưới đây dùng **trung bình 2 seed** và CI bootstrap theo cụm như hiện nay; ngưỡng "có thật" là chênh ≥ +2,5 với CI không chứa 0.

### G0. Gói đo một ngày, không train (bắt buộc trước mọi phương án) — **khuyến nghị số 1**

| Việc | Chi phí | Ra gì |
|---|---|---|
| Chấm Qwen3.5-2B-Base và Qwen3-4B **gốc** trên dev 1.3 bằng `eval_base.py` (cần ghim revision vào `config/models.json`; Qwen3.5 cần cài FLA vào venv riêng) | ≈ 1,5 giờ GPU | Biết nền nào có "sàn" VI cao hơn 70,6 trước khi tốn giờ train |
| Đo lại độ trễ **cùng một đường** cho 1.7B, Qwen3.5-2B, Qwen3-4B, decider: bật/tắt CUDA graphs (torch.compile "reduce-overhead" hoặc capture thủ công), có FLA; báo trung vị **theo nguồn** và p90, không chỉ trung vị toàn dev | ≈ 1 giờ GPU | Biết ngân sách 40 ms thật sự loại ai; biết dự án có đạt 40 ms ở ViFactCheck không |
| Sửa nguồn sự thật chuỗi: script xuất chuỗi hiệu lực từ `data/human_edits/edits_log.jsonl`; cập nhật `_status` trong config; kiểm C.jsonl/probe dùng chuỗi nào | 1–2 giờ, không GPU | Hết lệch "để kết luận" |
| Ghi vào PLAN_decisions: mọi bản thử từ nay **2 seed**; bỏ P1′ và P4-noul khỏi K1/K2 | 0 | Cổng đo được |

Tiêu chí dừng G0: không có; đây là phép đo. Rủi ro: cài FLA trên torch 2.11 + Blackwell + WSL có thể hỏng (cần kiểm cuDNN như đã làm ngày 12/10); người dùng đã nói "chốt 1.7B", nên G0 cần người dùng cho phép đo (không train) nền khác.

### Các phương án sau G0 (chọn theo kết quả G0; có thể ghép A với B hoặc C)

| Phương án | Nội dung | Chi phí (GPU / Claude) | Rủi ro | Tiêu chí dừng |
|---|---|---|---|---|
| **A. Giữ 84, giữ 1.7B, sửa dữ liệu C′** | Theo `FABLE_doc_ket_qua.md` §4 (mảng dữ liệu): bỏ health_fact, nâng NEI ≈ 30 %, tách VNTC khỏi khung SIB, thêm khung kiểm chứng của bench vào một phần dòng vitaminc; 2 seed | 9 giờ GPU / thấp (chữ mới cần duyệt: 1–2 chuỗi) | Khớp dev dần qua các vòng; chỉ chạm 84 ở mép nhiễu | Dừng A nếu trung bình 2 seed < 83,0 **hoặc** P4 NEI không về ≥ 85 % |
| **B. Đổi công thức, giữ 1.7B** | B1: quy mô EN từ tasksource lên 100–150 k dòng (bỏ trần 5 % cho phần EN, giữ phần VI của C′), 1 epoch; B2: LoRA r64/α128 (như decider) hoặc full fine-tune nếu vừa 16 GB; chạy B1 trước vì rẻ hơn về thiết kế | B1 ≈ 7–8 giờ GPU/seed; B2 ≈ 3–4 giờ/seed / thấp | Quên không dấu (phải theo dõi lát không dấu ≥ 61); giấy phép hỗn hợp (nội bộ nên được); dòng EN nhiều hơn nữa có thể không chuyển sang VI | Dừng B1 nếu seed 1 không hơn C ≥ +2,0 ở VI có dấu; dừng B2 nếu không hơn C ≥ +1,5 |
| **C. Đổi nền** | Cùng dữ liệu C′ (hoặc B1) trên Qwen3.5-2B-Base (nền của decider) và/hoặc Qwen3-4B (LoRA, micro-batch 4); chỉ chạy nếu G0 cho sàn gốc ≥ 73 hoặc độ trễ có graphs ≤ 40 ms theo nguồn | 2B ≈ 5–7 giờ/seed (brief đã ước); 4B ≈ 6–8 giờ/seed / thấp | Đổi nền là đảo quyết định 17/10; FLA/kernels; VRAM 4B | Dừng C nếu nền mới không hơn 1.7B cùng dữ liệu ≥ +2,0 **hoặc** vượt ngân sách độ trễ đã đo lại |
| **D. Đổi định nghĩa mục tiêu** | Chỉ tiêu mới: VI có dấu ≥ 82,5 (trung bình 2 seed, CI không chứa 0 so với 79,9), **đồng thời** không dấu ≥ 61 và dẫn đầu nhóm 2B ở không dấu (+12,7 so với decider), p90 độ trễ ≤ 60 ms; mô tả model là "nhanh, chịu được không dấu" | 0 GPU | Không phải mục tiêu đã đặt; cần người dùng chấp nhận | — |
| **E. Chuyển trọng tâm sang hoàn tất benchmark** | Chạm test 1.3 **một lần** cho 9 model đã có + bản dự án tốt nhất; viết báo cáo bench 1.3 (nguồn, quy trình soát, CI, bảng cờ "đã thấy"); bench 1.4 (MLQA) chỉ khi người dùng điền 30 câu | < 1 giờ GPU / vừa (viết) | Model dự án đứng thứ ba trong bảng của chính mình; công bố model lùi | — |

### Lịch đề xuất 2 tuần (một GPU)

| Ngày | Việc |
|---|---|
| 1 | **G0** trọn gói; quyết định nền và ngân sách độ trễ theo số đo mới; ghi PLAN_decisions |
| 2–3 | Dữ liệu C′ (mảng dữ liệu) + bộ probe sửa; train C′ **seed 1** trên 1.7B (3 giờ) + chấm; nếu G0 cho nền tốt hơn thì train C′ seed 1 trên nền đó thay vì 1.7B |
| 4–5 | **B1** (quy mô EN) seed 1 trên nền đã chọn (7–8 giờ) + chấm |
| 6 | So C′ với B1 trên dev + probe; chọn **một** công thức |
| 7–9 | Công thức đã chọn × **seed 2**; nếu cần, thử rank 64 (B2) seed 1 |
| 10 | Đọc theo cổng 2 seed. Đạt ≥ 84 và K1: sang train chính thức. 82,5–84: đề xuất D cho người dùng. < 82,5: đề xuất E, model làm mốc so sánh |
| 11–14 | Train chính thức (nếu đạt) hoặc hoàn tất bench (E); **chấm test một lần** ở cuối, cần phép riêng |

Điểm dừng toàn cục (ghi trước): nếu đến ngày 10 không công thức nào vượt C ≥ +2,0 (trung bình 2 seed), thì kết luận "giới hạn nằm ở công thức/nền ngoài tầm 2 tuần", chọn D hoặc E, không mở vòng dữ liệu thứ ba trên 1.7B.

---

## 5. Những điều tôi không chắc

- Thẻ model decider-2b đọc được trên web là bản ghim `ff99089b` (v11) nhưng trang chính đã sang v12 (có ảnh); số 3,2 ms là trên GPU B300, không phải RTX 5060 Ti. Tôi chỉ dùng nó để nói "đường chạy có graphs khác hẳn đường fallback", không dùng làm số so sánh.
- Ước tính độ trễ của 4B có graphs (20–25 ms) là tính tay từ băng thông và chi phí cố định, chưa đo. G0 là để đo.
- "Quy mô EN 100–150 k dòng" chưa có bằng chứng trực tiếp trong repo rằng nó chuyển sang VI; bằng chứng gián tiếp là decider (chỉ tiếng Anh) đạt 86,5 VI. decider-ft (thêm 20 k dòng vào model đã mạnh) giảm 1,4 là bằng chứng ngược nhưng là một tình huống khác (thêm vào model đã bão hòa).
- Xác suất 30 %/50 % là phán đoán, không phải tính toán.
- Tôi không đọc nội dung câu test, nên mọi nhận xét về "test lệch dev 1–2 điểm" là suy luận từ việc dev đã được dùng để chọn nhiều thứ, không từ số.

## 6. File đã dùng

Đọc: `PLAN_decisions.md`, `C:\Users\dungbt06\Downloads\PLAN.md`, `TONG_QUAN_DU_AN.md`, `results/BRIEF_train_chinh_thuc.md`, `results/BRIEF_phan_tich_xu_huong.md`, `results/BAO_CAO_TONG_HOP.md`, `results/baselines_v1_2/REPORT_baselines_v1_2.md`, `results/dev_v1_2/REPORT_bench_v1_2.md`, `results/bench_v1_3_trial/REPORT_bench_v1_3_trial.md`, `results/label_review_v1_2/REPORT_label_review.md`, `results/probe_v1/REPORT_moc.md`, `results/probe_v1/fable_soat/REPORT_fable.md`, `results/train_v1/REPORT_train_C.md`, `results/train_v1/REPORT_giai_doan_0.md`, `results/train_v1/phan_tich_C/FABLE_doc_ket_qua.md`, `results/train_v1/phan_tich_C/doc_loi_opus.md`, `results/train_v1/translated_sonnet/REPORT_dich_day_du.md`, `results/ft/REPORT_ft_v0.md`, `results/ft_decider/REPORT_ft_decider_v0.md`, `results/ft_strip/REPORT_strip_v0.md` (đầu), `results/restore/REPORT_restore.md` (đầu), `results/dev_v1_1/REPORT_sensitivity.md`, `results/dev_v1/REPORT_dev_v1.md` (đầu), `results/train_v1_survey/REPORT_sources.md` (đầu), `results/bench_candidates/REPORT_pilot.md` (đầu), `results/translation_audit/REPORT_translation_audit.md` (đầu), `results/review_tool/REPORT_review_tool.md` (đầu); `data/bench_v1_3/manifest_dev_v1_3.json`; `config/prompt_frames.json`, `config/bench_v1_questions.json`, `config/train_format.json`, `config/translations.json` (đầu), `data/manifest_train_v0.json` (đầu); script `scripts/eval_base.py`, `scripts/common.py`, `scripts/train_lora.py`, `scripts/bench/eval_decider_v1.py`, `scripts/baselines_v1_2/adapters.py`, `scripts/baselines_v1_2/harness.py`, `scripts/bench_v1_3/score_rows_ext.py`, `scripts/train_v1/trend_C.py`, `scripts/probe_v1/analyze_probe.py` (phần đầu).

Đếm (script riêng ngoài repo): `data/bench_v1_3/dev_v1_3.jsonl`, `results/train_v1/components/C.jsonl`, `data/train_v0.jsonl`.

Web: thẻ model [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) (trang hiện tại, v12) và [bản ghim ff99089b](https://huggingface.co/Mapika/decider-2b/raw/ff99089b6810691bd79134ec6febb5d9afa6e619/README.md) (v11), [chinhnc/Quyet-1.0-Medium](https://huggingface.co/chinhnc/Quyet-1.0-Medium), thẻ dataset [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions), [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).

Không mở: mọi file `test*` trong `data/bench_v1_*`, kể cả `manifest_test_v1.3.json`; số về test lấy từ `manifest_dev_v1_3.json` và các báo cáo đã dẫn.
