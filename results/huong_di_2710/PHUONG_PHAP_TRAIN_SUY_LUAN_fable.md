# Phương pháp train và suy luận: điểm VI trên mỗi giờ GPU (27/10/2026)

Người viết: Fable 5.1. Chỉ đọc repo, model card, bài báo; không chạy model, không tải trọng số hay dữ liệu; một script ngoài repo chỉ in phân vị `n_tokens` của dev (không in dòng nào). Mảng công thức đối thủ và nguồn dữ liệu do agent khác lo. Chi phí GPU tính theo hệ số đã đo: **0,165 s/dòng/epoch ≈ 21.000 dòng/giờ, VRAM đỉnh 10,25 GB**. Mọi mức tăng là kỳ vọng.

## 0. Kết luận

1. **Đòn bẩy rẻ nhất là thông lượng train.** Script hiện chỉ sắp độ dài *trong* một batch 32 câu rút ngẫu nhiên, bật gradient checkpointing dù còn 5,75 GB VRAM trống, dùng SDPA eager. Bucket độ dài toàn tập + tắt checkpointing ở bucket ngắn + Liger kernel ước **~2× số bước trong cùng số giờ** [suy luận, mục 1d]. Đây là hệ số nhân cho mọi phương án khác; chi phí vài giờ worker, ~15 phút GPU đo.
2. **Distillation phân phối chữ cái từ Quyet-Medium (+ decider-2b) là thay đổi thuật toán có điểm/giờ cao nhất.** Loss `full_vocab` đã nhận target mềm và `shuffled()` hoán vị target cùng options, nên chỉ cần **thay cột `target`** trong jsonl; teacher chạy một lần trên 31k prompt train, ~45 phút ở batch 1, dưới 15 phút nếu gom batch [suy luận từ 87,5 ms/câu]. Kỳ vọng +1–3 VI trên cùng dữ liệu [suy luận]; bằng chứng tổng quát mạnh, bằng chứng cho đúng bài này chưa có.
3. **DPO/ORPO/GRPO không đáng với đầu ra 1 token** [suy luận, mục 1b].
4. **4B không qua 40 ms ở p90, dù có CUDA graphs.** Đo trên dev: p90 độ dài prompt = 391 token, p95 = 866, ViFactCheck p50 = 985 (8,2% dev). Ở ≥ 400 token prefill 1.7B đã compute-bound; 4B cần ~2,6× FLOP → ~80–100 ms bf16 tại p90, ~45–55 ms nếu FP8 [suy luận, mục 2]. CUDA graphs chỉ sửa chi phí khởi chạy (~16–19 ms), tức đuôi prompt ngắn. 4B chỉ hợp lệ nếu cắt trần state khi suy luận hoặc đổi mục tiêu sang trung vị.
5. **Xếp hạng (mục 3):** (1) stack thông lượng → (2) KD trên prompt train sẵn có → (3) tăng EN theo bậc (audit #1) trên stack mới, kèm target KD → (4) việc ~0 GPU: soup 2 seed, ngưỡng noul trên tập proxy, lọc dòng bất đồng teacher–nhãn → (5) Qwen3-4B có điều kiện, sau G0.

## 1. Phương pháp train

### 1a. Knowledge distillation phân phối chữ cái

**Cơ chế, chi phí.** `row_losses` (train_lora.py:179–190) tính −Σ target_i·log p_vocab(chữ_i) với target k chiều bất kỳ; `shuffled()` hoán vị target cùng options → KD = tính phân phối teacher một lần theo thứ tự chuẩn, ghi vào `target`, **không đổi code loss** [suy luận từ code]. Teacher chỉ cần prompt. Quyet đã có gói và adapter chấm trong `results/baselines`, không cần tải; teacher không chịu ràng buộc 40 ms nên gom batch 16 có sắp độ dài (như phần batched của `eval_base.py`). **Vẫn cần người dùng duyệt vì là lần chạy model.**

**Bốn điểm phải xử lý.**
- *Prompt teacher khác prompt dự án:* KD chỉ chuyển phân phối trên k lựa chọn, không chuyển chuỗi token, nên teacher dùng prompt riêng của nó là bình thường. Cần ánh xạ noul: dự án [Có, Không]; Quyet A = mô tả đúng / B = sai; decider hàng [no, yes]. Dùng đúng `predict()` của gói như lúc chấm baseline.
- *Teacher sai hoặc nhiễm:* Quyet sai ~10% trên dev VI. Hinton et al. trộn CE với target mềm (nhiệt độ cao, nhân T²) và CE với nhãn thật trọng số thấp hơn [có nguồn: https://arxiv.org/abs/1503.02531]. Ở đây: target = λ·p_teacher + (1−λ)·onehot_gold, λ ≈ 0,5–0,7; nơi argmax teacher ≠ nhãn thì hạ λ về 0 và đánh dấu dòng (mục 1e) [suy luận]. "Student as an Inherent Denoiser of Noisy Teacher" (NeurIPS 2023) cho thấy student thường vượt nhãn teacher nhiễu giữa quá trình distill [có nguồn: https://arxiv.org/abs/2312.10185]. Trung bình phân phối Quyet + decider là cách giảm phương sai rẻ [suy luận]. Nếu teacher đã thấy nguồn bench, nhiễm chuyển gián tiếp qua hàm của teacher trên prompt train; mức rủi ro chuyển mảng công thức đối thủ.
- *Nhiệt độ:* lấy phân phối sau T theo kiểu của chính Quyet (1,33/1,60/1,43); so thêm bản T_KD = 2 trên tập proxy [suy luận].
- **Cấm** dùng prompt dev/test làm đầu vào teacher; chỉ prompt train và dữ liệu mới của mảng dữ liệu.

**Đòn bẩy KD mở ra:** train trên prompt tiếng Việt **không nhãn**. Chữ tiếng Việt mới phải qua duyệt; nếu tái dụng câu hỏi bench đã duyệt và văn bản có sẵn thì chi phí duyệt ≈ 0, nếu agent viết nhận định mới phải ghi số chuỗi.

**Giấy phép:** Quyet và Qwen3.5 đều Apache-2.0, yêu cầu giữ NOTICE, card không nói gì về distill [có nguồn: https://huggingface.co/chinhnc/Quyet-1.0-Medium]. Nên ghi nguồn teacher trong model card khi công bố [suy luận].

**Thí nghiệm nhỏ nhất:** C.jsonl với target KD (λ 0,6, teacher Quyet+decider) so với C gốc, **2 seed mỗi bên** (C mới có seed 1; audit đã yêu cầu seed 2). ~1 h teacher + 3 lần train (2,95 h, hoặc ~1,5 h sau 1d). Cổng: VI +≥2 trung bình 2 seed; P4 ≥ mức M1; letter_mass ≥ 0,99.

### 1b. Học sở thích / RL nhẹ

Với một token, DPO trên cặp (chữ đúng, chữ sai) là logistic trên hiệu logit có KL về model tham chiếu; GRPO với reward đúng/sai là REINFORCE trên một categorical k lớp: kỳ vọng gradient cùng hướng CE, phương sai cao hơn, thêm chi phí model tham chiếu [suy luận]. Bằng chứng công khai chỉ là một ca khách hàng (phân loại nhị phân, lỗi 4,7% → 1,3%) kèm cảnh báo DPO "khó làm đúng hơn SFT" [có nguồn: https://docs.openpipe.ai/features/dpo/overview]. Hành vi "văn bản lạc đề → không đủ thông tin" là CE trên dòng tổng hợp (audit #6), không cần RL. **Bỏ.**

### 1c. Model nền và cỡ LoRA

| Phương án | Chi phí/dòng so hiện tại | Dòng/giờ | Nhận xét |
|---|---|---|---|
| Qwen3-1.7B LoRA r16 (hiện tại) | 1× | ~21k | Cơ sở. r64/α128 gần như cùng giá; Biderman et al.: rank chỉ quan trọng khi dữ liệu lớn [có nguồn: https://arxiv.org/abs/2405.09673] → sau khi tăng dữ liệu. Full FT: AdamW fp32 ≈ 27 GB, không vừa; 8-bit Adam sát trần và LoRA khớp full FT ở dữ liệu nhỏ [suy luận + Biderman] → bỏ. |
| Qwen3.5-2B-Base LoRA bf16 | ~1,1× (FLOP ≈ 1.7B, vocab 248k) | ~19k | Nền của decider (86,5), là ứng viên "to hơn" duy nhất không vỡ độ trễ. Rủi ro: Gated DeltaNet cần kernel `flash-linear-attention`, card Qwen3-Next ghi là tùy chọn [có nguồn: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct]; dự án đo 57,7 ms khi thiếu kernel. |
| Qwen3-4B LoRA bf16 | ~2,6× (3,6B so 1,4B tham số ngoài embedding) | ~8k | C.jsonl 2 epoch ≈ 7,7 h; 8 GB trọng số, micro-batch 4 [suy luận]. Qwen3.5-4B thêm vision encoder và vocab 248k, không cần. QLoRA NF4 giữ chất lượng [có nguồn: https://arxiv.org/abs/2305.14314] nhưng thông lượng 0,7–0,8× [có nguồn: https://arxiv.org/pdf/2608.23816]; 4B bf16 đã vừa 16 GB → không dùng NF4. |

Quyết định nền chờ G0 (audit). **Thí nghiệm nhỏ nhất (sau G0):** Qwen3-4B LoRA r16 trên train_v0, 1 seed 1 epoch ≈ 2,6 h, so M1 cùng dữ liệu; cổng VI +≥4 (chi phí 2,6× nên lợi phải lớn hơn tăng dữ liệu cùng giờ).

### 1d. Tăng thông lượng train

Từ code [suy luận, `epoch_batches` train_lora.py:106–123]: 32 câu rút ngẫu nhiên rồi mới sắp; micro-batch chứa câu dài nhất pad tới max của 32 câu (p97 ≈ 700 token, p50 ≈ 66). `encode()` chạy hai lần mỗi dòng (CPU, nhỏ).

| Kỹ thuật | Ước tăng tốc | Bằng chứng, rủi ro |
|---|---|---|
| Bucket độ dài toàn tập, xáo trong bucket | 1,3–1,8× | [suy luận] giảm token pad ~40–60%; batch không i.i.d. (chuẩn thực hành); hash `--dump_batches` đổi. |
| Tắt gradient checkpointing ở bucket ≤ 512, micro-batch theo ngân sách token | 1,2–1,3× | Checkpointing tính lại forward ≈ +1/3 FLOP [suy luận]; giữ bật ở bucket dài. |
| Liger kernel (RMSNorm, RoPE, SwiGLU) | 1,1–1,25× | Qwen2 +25,5% thông lượng, −56,8% bộ nhớ [có nguồn: https://arxiv.org/abs/2410.10989]; fused linear-CE không áp dụng (loss 1 vị trí); cần kiểm sm_120 + transformers 5.19. |
| torch.compile forward | 1,1–1,3× | [suy luận]; pad lên bội 64 tránh recompile; graph break với peft. |
| Không khả thi hoặc không đáng trên máy này | — | Packing cần FlashAttention varlen [có nguồn: https://huggingface.co/docs/transformers/v5.14.0/padding_free] mà flash-attn 2 chưa có wheel sm_120, build nguồn crash [có nguồn: https://github.com/Dao-AILab/flash-attention/issues/2361]. FP8 torchao: 1.8B chỉ +18% ở seq 8k trên H100 [có nguồn: https://pytorch.org/blog/training-using-float8-fsdp2/], GEMM nhỏ ở đây khó bù [suy luận]. Unsloth hỗ trợ RTX 50 + WSL [có nguồn: https://docs.unsloth.ai/basics/training-llms-with-blackwell-rtx-50-series-and-unsloth] nhưng đòi viết lại vòng train tự có (loss 1 vị trí, monitor letter_mass, strip) → chỉ nếu Liger + bucket không đủ. |

**Tổng:** 1,5 × 1,25 × 1,15 ≈ **2,1×** → ~45k dòng/giờ [suy luận]. **Thí nghiệm nhỏ nhất:** đo 100 bước train_v0 trước/sau từng lớp (~15 phút GPU), rồi 1 epoch train_v0: val loss = 0,756 ± 0,01 so M1 epoch 1.

### 1e. Chọn, lọc dữ liệu

LESS chọn 5% theo ảnh hưởng gradient, "thường" vượt toàn tập, +2–5 so ngẫu nhiên [có nguồn: https://arxiv.org/abs/2402.04333], nhưng cần kho gradient (≈ một lượt forward/backward toàn tập) và tập đích few-shot không được lấy từ dev. Rẻ hơn, **miễn phí khi có 1a:** teacher tự tin ≠ nhãn → nghi nhãn sai (PhoLaya ~3%, 21 dòng VSMRC) → soát/bỏ; biên teacher nhỏ → câu khó, giữ; cả hai cùng tự tin → câu dễ, giảm [suy luận]. Curriculum: bằng chứng yếu, không ưu tiên. **Thí nghiệm nhỏ nhất:** đếm dòng bất đồng; nếu > 5%, soát mù 100 dòng (0 GPU).

### 1f. Gộp model

Soup 2 seed: 0 GPU thêm vì quy tắc 2 seed đã bắt buộc. Bằng chứng trái nhau: SeedLoRA nói adapter khác seed bổ trợ; AdapterSoup nói trung bình adapter khác seed cho kết quả kém [có nguồn: https://arxiv.org/abs/2302.07027]; LoRA ensembles (trung bình đầu ra) cải thiện nhất quán [có nguồn: https://arxiv.org/abs/2310.00035] nhưng nhân đôi độ trễ. Kỹ thuật: ma trận A khởi tạo ngẫu nhiên theo seed, nên **merge từng adapter thành ΔW rồi trung bình**, không trung bình A, B riêng [suy luận]. Không gộp được với decider (khác nền). **Thí nghiệm nhỏ nhất:** trung bình hai bản merge C seed 1, 2, chấm dev, 15 phút; kỳ vọng +0–1.

### 1g. Hiệu chỉnh, ngưỡng

Audit đã kết luận T chỉ đổi Brier; ngưỡng noul t* = 0,57 (+1,4) fit trên dev nên phải fit lại trên tập proxy. Chi phí ≈ 0. Thêm: dùng `letter_mass` và entropy làm cổng từ chối trong demo, không ảnh hưởng bench [suy luận].

### 1h. Ý tưởng khác

- **Tự nhất quán theo hoán vị:** target = trung bình dự đoán của chính model trên vài hoán vị lựa chọn, trộn với nhãn; giảm thiên vị vị trí (Zheng et al., audit 1.3) với 0 teacher ngoài, chi phí +vài forward không gradient mỗi dòng [suy luận]. Thử sau 1a.
- **Trần state khi suy luận** ≤ 600 token: cách rẻ duy nhất kéo p95 về gần 40 ms (mục 2); phải đo ViFactCheck tụt bao nhiêu; train với cùng trần để khớp.

## 2. Phía suy luận

**Số liệu.** RTX 5060 Ti: 448 GB/s [có nguồn: https://www.club386.com/nvidia-geforce-rtx-5060-ti-16gb-vs-rtx-4060-ti-8gb/]; NVIDIA không công bố TFLOPS theo độ chính xác cho GB206 [có nguồn: https://flopper.io/gpu/nvidia-geforce-rtx-5060-ti-16gb]. RTX 5090: bf16 tích lũy FP32 dense 209,5 TFLOPS, FP16 tích lũy FP16 419 (nửa tốc độ khi tích lũy FP32) [có nguồn: https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543]. Quy theo SM và xung (36/170 SM, ~2,57/2,41 GHz): 5060 Ti ≈ **47 TFLOPS bf16 (tích lũy FP32, cách PyTorch dùng), ≈ 95 TFLOPS FP8** [suy luận].

**Phân vị độ dài prompt dev 1.3** (3.847 dòng, trường `n_tokens`): p50 165, **p90 391**, p95 866, p99 1.510; ViFactCheck p50 985, p90 1.578 (316 dòng = 8,2%); 9,5% dòng > 400 token. Vậy p90 độ trễ 47 ms của bản C ứng với ~391 token, còn ViFactCheck nằm trong 8% đuôi.

**Chuỗi suy luận [toán học, suy luận].** FLOP prefill ≈ 2 × tham số ngoài embedding × token (lm_head 1 vị trí nhờ `logits_to_keep=1`).
- **Cận dưới không cần giả định hiệu suất:** Qwen3-4B (3,6B ngoài embedding) ở prompt p90 391 token = 2,8 TFLOP → ở **100% đỉnh bf16 47 TFLOPS vẫn ≥ 60 ms**; FP8 ở 100% đỉnh 95 ≥ 30 ms, hiệu suất thực 60–70% cho **42–50 ms**. ViFactCheck p50 (7,1 TFLOP) ≥ 150 ms bf16. Vậy 4B không qua 40 ms ở p90 dù CUDA graphs có xóa hết chi phí khởi chạy.
- Ước lượng thứ cấp cho 1.7B: ở 128 token 0.6B = 1.7B = 28–29 ms (REPORT_ft_v0), nhưng ở độ dài đó GEMM quá nhỏ để lấp GPU nên con số đó là sàn độ trễ kernel + khởi chạy, không tách sạch thành "chi phí cố định"; lấy thô 16–19 ms cố định thì 1.7B ở 391 token (1,1 TFLOP) còn ~30 ms tính → hiệu suất thực **~25–45 TFLOPS**, biên rộng.
- Bật CUDA graphs: 1.7B trung vị ≈ 13–17 ms, p90 ≈ **33–40 ms**, p95 (866 token, 2,4 TFLOP) ≈ 60–85 ms. Mục tiêu ≤ 40 ms chỉ đạt tới p90 ngay cả với 1.7B.
- Qwen3-4B trung vị (165 token, 1,2 TFLOP) ≈ 35–45 ms bf16, ≈ 20 ms FP8 → trung vị qua 40 ms chỉ với graphs + FP8.
- Qwen3.5-2B-Base: ~1,4–1,5B ngoài embedding → như 1.7B; DeltaNet giảm attention nhưng attention là phần nhỏ ở < 1.000 token.
- Quyet-Medium 87,5 ms trung vị eager khớp bức tranh.

**Từng kỹ thuật.**
- *CUDA graphs:* lợi lớn nhất khi batch 1 model nhỏ vì chi phí khởi chạy chiếm phần lớn [có nguồn: https://vllm.ai/blog/torch-compile; https://arxiv.org/abs/2410.00215]. Bài này **chỉ prefill một forward, không decode**, nên static KV cache không phải thứ giúp (`use_cache=False` được); thứ giúp là bắt graph forward trên các bucket độ dài pad cố định (torch.compile `reduce-overhead` hoặc graph tay), gather logit tại `len−1` như khi train; cổng: argmax trùng eager 100% [suy luận].
- *Lượng tử hóa:* FP8 mất −0,1 đến −0,7 MMLU-Pro ở model lớn [có nguồn: https://arxiv.org/abs/2411.02355]; INT4 AWQ mất 4,6 MMLU ở Llama3-8B so 0,46 của FP8 [có nguồn: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/benchmark.md]; chưa có số cho 2–4B và cho logit chữ cái → phải tự đo trên dev. W4A16 (AWQ/GPTQ/NVFP4 chỉ trọng số) **không giảm FLOP prefill** → vô ích cho p90; chỉ FP8 W8A8 có ích [suy luận].
- *Engine:* vLLM chạy trên sm_120 nhưng cần vá, không FA3 [có nguồn: https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492]; SGLang/TensorRT-LLM trên sm_120 thiếu bằng chứng; llama.cpp là int-quant. Với "một forward, đọc logit", transformers + torch.compile reduce-overhead đủ và giữ đúng đường chấm [suy luận]. *Prefix caching:* chỉ dẫn cố định vài chục token → < 5%, bỏ [suy luận].

**Kết luận mục 2:** cổng ≤ 40 ms chưa đạt ở p95 ngay với 1.7B; đòn bẩy là CUDA graphs (đuôi ngắn) và trần state (đuôi dài), không phải engine. G0 nên báo trung vị, p90, p95 và **p90 theo bucket độ dài** cho 1.7B, 2B-Base, 4B.

## 3. Xếp hạng theo điểm VI kỳ vọng trên giờ GPU

| # | Phương án | Giờ GPU | Kỳ vọng VI | Độ trễ | Thí nghiệm nhỏ nhất, cổng |
|---|---|---|---|---|---|
| 1 | Stack thông lượng: bucket toàn tập + tắt GC bucket ngắn + Liger | ~0,3 | 0 trực tiếp; nhân ~2× cho mọi phương án sau | không đổi | 100 bước đo thời gian; 1 epoch train_v0 val loss 0,756 ± 0,01 |
| 2 | KD từ Quyet+decider trên prompt train sẵn có, λ 0,6 | ~1 teacher + 3 × 2,95 (×0,5 sau #1) | +1 đến +3 [suy luận] | không đổi | C-KD so C, 2 seed/bên; VI +≥2, P4 ≥ M1, letter_mass ≥ 0,99 |
| 3 | Tăng EN theo bậc 50k/100k (audit #1) trên stack #1, kèm target KD | 2,5–5 h mỗi mức | +2 đến +4 [suy luận từ decider "English only" 86,5] | không đổi | `--save_fracs` lấy đường cong; VI +≥2 so C′, 2 seed |
| 4 | ~0 GPU: soup 2 seed, ngưỡng noul fit trên proxy, lọc dòng bất đồng teacher | ~0,3 | +0 đến +1,5 | không đổi | chấm dev 15 phút; soát mù 100 dòng |
| 5 | Qwen3-4B LoRA (sau G0) | 2,6 h/epoch train_v0; 7,7 h C | +3 đến +6 nếu đủ bước [suy luận từ Quyet 89,6 ở 120k bước] | **vỡ p90**; chỉ nếu cắt state hoặc mục tiêu theo trung vị + FP8 | G0 đo p90 theo bucket; rồi 1 epoch train_v0 so M1, VI +≥4 |

Không khuyến nghị: DPO/GRPO; QLoRA cho 4B trên 16 GB; packing (không có FA2 sm_120); FP8 training; đổi engine trước khi thử torch.compile reduce-overhead.

## 4. Chưa chắc

- Hiệu suất thực 25–45 TFLOPS của 1.7B suy từ một số đo p90 eager và phép tách chi phí cố định không sạch; kết luận về 4B không phụ thuộc số này (cận dưới ở 100% đỉnh). G0 thay thế.
- Mức tăng KD cho bài "k chữ cái" chưa có số công khai; +1–3 ước từ khoảng cách teacher–student 8 điểm.
- Liger, torch.compile trên sm_120 + transformers 5.19 chưa xác nhận; SeedLoRA và AdapterSoup trái nhau — thí nghiệm 15 phút rẻ hơn tranh luận.
