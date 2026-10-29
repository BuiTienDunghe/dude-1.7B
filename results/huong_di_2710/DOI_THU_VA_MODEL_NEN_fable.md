# Đối thủ và model nền: khảo cổ công thức, nhiễm, giấy phép, ứng viên (27/10/2026)

Mảng: công thức train thật của Quyet-1.0, decider-2b, Jeff-Qwen3.5-2B; rủi ro nhiễm; giấy phép; ứng viên model nền ≤ 4B. Chỉ đọc card, README, GitHub, metadata; không tải trọng số, không chạy gì. Số dev lấy từ `results/baselines_v1_2/`. Khoảng 35 lượt tìm/đọc web.

## 0. Kết luận

1. **Phần lớn khoảng cách +8 điểm đến từ công thức, không phải cỡ model.** decider-2b (1,88 B, nền Qwen3.5-2B-Base) được train **không có một dòng tiếng Việt** mà đạt 86,5 VI trên dev, hơn bản C (81,65) khoảng 5 điểm ở cỡ gần bằng **[có nguồn: README v11 "English only", lưu tại `results/baselines/meta/Mapika__decider-2b/README.md`; https://raw.githubusercontent.com/Mapika/decider/main/decider/data/mixture.py không có bộ tiếng Việt hay đa ngữ nào]**. Caveat: decider có thể đã thấy MASSIVE en-US; bỏ MASSIVE, decider vẫn hơn adapter cũ của dự án +6,0 [3,3; 8,6] **[có nguồn: REPORT_baselines_v1_2 mục 1]**. Vậy ít nhất ~5/8 điểm là do dữ liệu và quy mô train; phần còn lại (86,5 → 89,6) mới bị trộn giữa cỡ 4B và "tuned for Vietnamese" của Quyet **[suy luận]**.
2. **"Quyet-Medium train 120.000 bước" là con số thật nhưng không so được.** Nguồn duy nhất là `quyet_config.json` → `source.step: 120000`, một nhãn checkpoint đứng cạnh sha256 của adapter; không có batch size, epoch, độ dài chuỗi, số dòng **[có nguồn: https://huggingface.co/chinhnc/Quyet-1.0-Medium/tree/main; repo không có `adapter_config.json`, `training_args`, `trainer_state` hay log]**. Jeff cho thấy vì sao đơn vị "bước" gây lầm: 284.747 câu, batch 256, 1 epoch ≈ 1.100 bước, **ít bước hơn dự án** nhưng gấp 4,6 lần số lượt-dòng **[có nguồn: https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B]**. Khẳng định "nhỏ hơn đối thủ 1–2 bậc" trong audit không đứng được ở đơn vị bước; nên so bằng **lượt-dòng hoặc token**: dự án ≈ 62k lượt-dòng; Jeff 285k; decider 1,47M dòng / 455M token mỗi epoch; Quyet không biết.
3. **Nhiễm benchmark của Quyet: không có bằng chứng công khai theo chiều nào.** Card trống, 0 dataset, 0 thảo luận, không công bố điểm gì ngoài JevBench. Số trên dev khớp cả giả thuyết "đã thấy" lẫn "model mạnh" (mục 2).
4. **Giấy phép Quyet và decider đều là Apache-2.0 chuẩn, không điều khoản nào về đầu ra.** Distill từ xác suất của chúng không bị cấm; chỉ cần giữ NOTICE khi phân phối bản dẫn xuất (mục 3, trích nguyên văn).
5. **Model nền:** Qwen3.5-2B-Base là ứng viên mạnh nhất về bằng chứng tiếng Việt (86,5 qua decider). Nhưng **chưa cỡ nào của họ Qwen3.5 được đo dưới 40 ms** trên 5060 Ti: 2B gốc 57,7 ms, Quyet-4B 87,5 ms, cả hai đều không `flash-linear-attention` và không CUDA graphs. Biến quyết định cho cả 2B lẫn 4B là stack suy luận, và chỉ G0 trả lời được (mục 4–5).

## 1. Công thức train thật của đối thủ

| | Quyet-1.0-Medium | decider-2b v11 (bản dự án ghim) | Jeff-Qwen3.5-2B v1.2 |
|---|---|---|---|
| Nền | Qwen3.5-4B (post-trained), revision 851bf6e8 | Qwen3.5-2B-**Base** | Card ghi trọng số từ "Qwen3.5-2B"; model tree ghi nền Qwen3.5-2B-Base, cha fine-tune là Qwen3.5-2B |
| Cách | LoRA r16 đã merge; "letter-readout decision prompt" | SFT CE v1–v8 trên ~95 bộ công khai + dữ liệu teacher (Qwen3.5-27B cục bộ), 1,47M dòng / 455M token mỗi epoch; RL 384 bước (v8→v10, chỉ reward kết quả, không nhãn vàng); LoRA r64/α128, 2 epoch, 42.749 dòng, 1.676 bước × 65.536 token (v10→v11) | Full fine-tune, lr 1e-5, batch 256, 1 epoch |
| Dữ liệu | **Không nêu**; chỉ "English; also tuned for Vietnamese" | Teacher-written + công khai; **không có tiếng Việt** | 284.747 câu: bộ công khai recast, WinoGrande 40k, RAGTruth 15k, văn bản dài, ~50k câu tổng hợp từ teacher Qwen3.8-Flash-Next; **không nêu tiếng Việt**; v1.1+ thêm train split MASSIVE và CLINC150 |
| Bước / loss | `source.step` 120.000; loss không nêu | SFT: không nêu bước; loss CE; chọn checkpoint bằng val Choice NLL | ≈1.100 bước; CE trên chữ cái lựa chọn, một nhiệt độ hiệu chỉnh |
| RL / distil | Không nêu. Chỉ Tiny (183 M, mmBERT-small) ghi "distilled from Quyet-1.0-Medium"; Small là SEA-LION-ModernBERT-300M | RL có (v10); không dùng chữ "distillation" nhưng dữ liệu do teacher viết | Không RL; teacher sinh dữ liệu |
| Nguồn | https://huggingface.co/chinhnc/Quyet-1.0-Medium ; https://huggingface.co/chinhnc/Quyet-1.0-Tiny ; https://github.com/ncchinh/quyet ; https://pypi.org/project/quyet/ ; https://quyet.ai | README v11 cục bộ; https://github.com/Mapika/decider | https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B |

Ghi chú:
- Repo GitHub `ncchinh/quyet` chỉ là runtime (`src/quyet`, `tests`), không có script hay config train **[có nguồn: https://github.com/ncchinh/quyet]**. Profile HF `chinhnc` có 5 model, **0 dataset** **[có nguồn: https://huggingface.co/chinhnc]**. quyet.ai chỉ nêu JevBench (Large hạng 1, 71,39) **[có nguồn: https://quyet.ai]**.
- **Card HF decider-2b đã đổi:** commit 7614a6c ngày 20/10 là **v12** (text + ảnh, Noul T 0,05); bản dự án ghim ff99089b (12/10) là v11 **[có nguồn: https://huggingface.co/Mapika/decider-2b/commits/main]**. Card hiện tại mô tả bản khác (39,5k dòng text + 18k ảnh, không RL), nên công thức trên lấy từ README v11 lưu cục bộ. Jeff đã bị đánh dấu superseded bởi jeff-base v1.3 **[có nguồn: card Jeff]**.
- **Không kiểm được:** dữ liệu, số dòng, batch, epoch, loss của Quyet (mọi cỡ); tiếng Việt của Quyet từ đâu; bước và loss SFT của decider; tiếng Việt có trong pretrain Qwen3.5 không.

## 2. Rủi ro nhiễm của Quyet

Quyet-Medium trên dev, đếm từ `preds.jsonl` cục bộ (đúng / n): XCOPA EN **100/100**, VI 93/100; Belebele EN **286/294 = 97,3**, VI 276/294 = 93,9; MASSIVE EN 285/294, VI 284/294; SIB-200 EN và VI đều 250/299 = 83,6; ViFactCheck VI 140/158 = 88,6; ViWikiFC VI 232/287 = 80,8 **[có nguồn: `results/baselines_v1_2/eval/normal/quyet-medium-text/preds.jsonl`]**.

Hai cách đọc, đều **[suy luận]**:
- *Nghiêng về "đã thấy":* 100/100 COPA và 97,3 Belebele EN là mức trần; cả ba bộ gần trần (XCOPA, Belebele, MASSIVE) đều có bản tiếng Việt công khai trên HF, dễ lọt vào mix "tuned for Vietnamese"; `datasets: null` và 0 dataset công khai làm không loại trừ được.
- *Nghiêng về "model mạnh":* SIB-200 công khai y như vậy nhưng Quyet chỉ 83,6, bằng dự án (p=0,70); ViFactCheck là tiếng Việt gốc, không phải bộ dịch, Quyet vẫn hơn dự án +15,8; nhất quán khi đảo 97,9 loại được học thuộc vị trí chữ cái (không loại học thuộc nội dung); phép đổi nhãn theo đa số chỉ làm Quyet +1,5 trong khi model khác +6,4, khớp cả "giỏi hơn hẳn" lẫn "đã thấy đáp án".

Kết luận: **không có bằng chứng công khai**. Chỉ G0 (chấm Qwen3.5-4B gốc trên dev) mới tách phần do model nền khỏi phần do fine-tune: nếu 4B gốc đã ~95+ ở XCOPA EN thì 100 của Quyet không còn đáng ngờ **[suy luận]**.

## 3. Giấy phép: có được distill từ đầu ra không

**Quyet-1.0-Medium.** LICENSE là Apache-2.0 chuẩn (202 dòng), chỉ điền phụ lục "Copyright 2026 Chinh Nguyen"; không có câu nào về đầu ra model **[có nguồn: https://huggingface.co/chinhnc/Quyet-1.0-Medium/raw/main/LICENSE, đã đọc toàn văn]**. NOTICE nguyên văn **[có nguồn: https://huggingface.co/chinhnc/Quyet-1.0-Medium/raw/main/NOTICE]**:

> Quyet by Chinh Nguyen
> Copyright 2026 Chinh Nguyen
>
> Licensed under the Apache License, Version 2.0 (see LICENSE). Redistributions of this work or of derivative works must keep this NOTICE, including the credit line above.
>
> This work includes or derives from:
> - Qwen3.5 by Alibaba Cloud (Apache-2.0): https://huggingface.co/Qwen/Qwen3.5-4B (revision 851bf6e806ef)

Hai điều khoản Apache-2.0 liên quan, nguyên văn từ file LICENSE của Quyet:

> "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

> (d) If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License.

**decider-2b.** LICENSE Apache-2.0 chuẩn, phụ lục "Copyright 2026 Mark Marosi", không thêm gì **[có nguồn: https://raw.githubusercontent.com/Mapika/decider/main/LICENSE]**; card: "The weights are released under Apache 2.0, as the base model." **[có nguồn: card decider-2b]**.

**Đọc điều khoản [suy luận, không phải tư vấn pháp lý].** Apache-2.0 không đặt điều kiện nào lên đầu ra model; nghĩa vụ NOTICE chỉ phát sinh khi phân phối Derivative Works. Một model chỉ học từ **xác suất đầu ra** của Quyet không chứa trọng số Quyet và "remain separable", khó bị coi là Derivative Work; nhưng giữ dòng "Quyet by Chinh Nguyen" trong NOTICE và card không tốn gì và xóa mọi tranh cãi. Tác giả Quyet tự distill Tiny từ Medium, không có tín hiệu phản đối cách này. Dữ liệu train của decider có nguồn CC BY-SA, CC BY-NC, nhưng điều đó ràng buộc dữ liệu của họ, không ràng buộc đầu ra model. Đối chiếu: Llama 3.2 **có** điều khoản về đầu ra, "you shall also include "Llama" at the beginning of any such AI model name" khi dùng đầu ra để train model khác **[có nguồn: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct]**.

## 4. Ứng viên model nền ≤ ~4B

| Ứng viên | Cỡ | Giấy phép | Bằng chứng tiếng Việt | A–J 1 token | Nhận xét |
|---|---|---|---|---|---|
| **Qwen3.5-2B-Base / 2B** https://huggingface.co/Qwen/Qwen3.5-2B | 2,27 B repo (có vision), ~1,88 B text | Apache-2.0 | Dev: 2B gốc 70,6 VI, bằng Qwen3-1.7B gốc ở trung bình nhưng lệch mạnh theo nguồn (MASSIVE −11,2, ViFactCheck +12,7, không dấu −9,4); decider-2b (Base, train EN) **86,5**. Card (thinking): INCLUDE 55,4, Global PIQA 69,3, MMLU-ProX 52,3 | **Đã kiểm** (assert eval_base qua trên 2B) | Ứng viên chính. 57,7 ms khi không FLA |
| **Qwen3.5-4B(-Base)** https://huggingface.co/Qwen/Qwen3.5-4B | 4,66 B tổng, 4,21 B text | Apache-2.0 | Card: INCLUDE 71,0, Global PIQA 78,9, MMLU-ProX 71,5 (29 ngôn ngữ); MMMLU 76,1 nhưng MMMLU không có tiếng Việt; là nền của Quyet-Medium 89,6 | Đã kiểm gián tiếp: Quyet đọc chữ cái trên cùng vocab 248.320 | 9,3 GB bf16; "runs in bf16 on a 16 GB GPU" là câu về suy luận, **train LoRA 4B trên 16 GB chưa thử** (bản C 1.7B đỉnh 10,25 GB) |
| Qwen3-4B-Instruct-2507 https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | 4,0 B | Apache-2.0 | INCLUDE 60,1, MMLU-ProX 61,6, thấp hơn Qwen3.5-4B rõ | Cùng tokenizer họ Qwen3 [suy luận] | Attention chuẩn, không cần FLA |
| Gemma 4 E2B / E4B https://huggingface.co/google/gemma-4-E4B-it | 5,1 / 8 B tổng; 2,3 / 4,5 B hiệu dụng (PLE) | Apache-2.0 theo card; trang https://ai.google.dev/gemma/docs/gemma_4_license còn link "Terms of use", "Prohibited use" chưa đọc được | "35+ languages out-of-the-box, pre-trained on 140+"; tiếng Việt không được nêu; MMMLU E4B 76,6 (không có VI) | [suy luận] một token, vì Quyet-Large đọc chữ cái trên Gemma-4-31B-it | E4B ≈ 16 GB bf16 nếu không offload bảng PLE; vừa 16 GB để train hay không **chưa rõ** [suy luận] |
| Sailor2-1B https://arxiv.org/html/2502.12982v1 | 1 B (từ Qwen2.5-0.5B) | Apache-2.0 | Belebele-vi 36,8, XCOPA-vi 68,0 | [suy luận] | Loại vì yếu |
| Arcee-VyLinh https://huggingface.co/arcee-ai/Arcee-VyLinh | 3 B (Qwen2.5-3B) | Apache-2.0 | Chỉ m-ArenaHard VI dạng biểu đồ, không VMLU | [suy luận] | Thế hệ Qwen2.5, cũ hơn Qwen3.5 |
| Llama-3.2-3B; Phi-4-mini (https://ai.azure.com/catalog/models/Phi-4-mini-instruct) | 3,21 B; 3,8 B | Llama 3.2 Community (điều khoản đặt tên); MIT | Tiếng Việt **không** trong 8 / 23 ngôn ngữ chính thức | [suy luận] | Loại |

**Benchmark tiếng Việt công khai:** leaderboard VMLU không có model Qwen3/Qwen3.5 nào ≤ 5B; mục nhỏ tốt nhất là Vintern-3B (Qwen2.5-3B) 54,8 **[có nguồn: https://vmlu.ai/leaderboard]**. Không tìm được VMLU hay ViLLM-Eval cho Qwen3.5-4B, Qwen3-4B, Gemma 4. Vì vậy **dev của dự án là bằng chứng tiếng Việt tốt nhất hiện có**, và chỉ có số cho 1.7B và 2B.

**Độ trễ [suy luận từ số đã đo].** Ba điểm đo trên 5060 Ti: Qwen3-1.7B 32,3 ms (attention chuẩn), Qwen3.5-2B 57,7 ms (không FLA), Quyet-4B 87,5 ms (không FLA, prompt tiếng Anh cố định của gói). Trong họ Qwen3.5, 2,2× tham số text chỉ tốn 1,5× độ trễ, nên phần lớn chi phí là kernel GatedDeltaNet chạy không có FLA, không phải cỡ. Vì vậy **chưa cỡ nào của Qwen3.5 được chứng minh dưới 40 ms**, và 2B chỉ có nhiều dư địa hơn 4B. Với prompt ~900 token (ViFactCheck), 4B cần ~7–8 TFLOP mỗi lượt; GPU ước ~47–95 TFLOPS bf16 dense **[có nguồn: https://flopper.io/gpu/nvidia-geforce-rtx-5060-ti-16gb]** nên p90 của 4B nhiều khả năng vượt 40 ms kể cả khi có FLA. decider báo "4 ms per request with CUDA graphs" nhưng trên GH200 với prompt ngắn, không chuyển sang được.

## 5. +8 điểm đến từ cỡ hay công thức?

Bằng chứng công khai trả lời được **khoảng 5/8 điểm**: decider-2b cho thấy cùng cỡ ~2B, không tiếng Việt, đạt 86,5 VI nhờ ~1,47M dòng SFT đa dạng + RL + LoRA bổ sung; dự án với 31k dòng có tiếng Việt được 81,65. Qwen3.5-2B gốc và Qwen3-1.7B gốc cùng 70,6 trung bình, nên bước 1,7 → 2B tự thân không mua được gì ở trung bình; cái mua được là công thức **[có nguồn: REPORT_baselines_v1_2 mục 1]**. Hai biến chưa tách khi quy cho công thức: decider dùng nền **Base** (không post-train) và prompt thuần, không chat template **[suy luận]**.

Phần **~3 điểm còn lại** (86,5 → 89,6) không tách được bằng tài liệu: Quyet vừa 4B, vừa có tiếng Việt trong fine-tune (không rõ bao nhiêu), vừa có thể đã thấy benchmark. Chỉ GPU trả lời: (a) Qwen3.5-4B gốc trên dev, so với 70,6 của 2B gốc, đo phần do cỡ; (b) độ trễ 1.7B, 2B, 4B đo cùng cách, có FLA và CUDA graphs, cả trung vị và p90; (c) nếu còn ngân sách, chấm decider-4b (Qwen3.5-4B-Base, train EN, không RL) để có điểm "4B + công thức decider" làm đối chứng cho Quyet **[có nguồn: bảng họ decider trong README v11]**.

**Không kiểm được, cần ghi vào biên bản quyết định:** dữ liệu và quy mô thật của Quyet; Quyet có thấy XCOPA/Belebele không; tiếng Việt trong pretrain Qwen3.5; Gemma 4 có điều khoản dùng bổ sung ngoài Apache không; tokenizer Gemma và Qwen2.5 giữ A–J một token (chưa assert); train LoRA 4B có vừa 16 GB không.
