# Kiểm toán mảng PHƯƠNG PHÁP và HÀM CHUYỂN LLM → DECISION MODEL

Ngày 27/10/2026. Người kiểm: Fable 5.1. **Chỉ đọc repo**; không chạy model, không train, không đọc file test nào. Số đếm bổ sung lấy từ hai script nhỏ đặt ngoài repo (mục 6). Mảng dữ liệu và mảng đánh giá/chiến lược do hai agent khác phụ trách; ở đây chỉ nêu điểm giao nhau khi cần và chuyển giao rõ.

Ký hiệu: **M1** = Qwen3-1.7B + LoRA full_vocab seed 1 trên train_v0 (model dự án, VI có dấu 79,9 trên dev 1.3); **C** = cùng công thức trên C.jsonl 31.034 dòng (81,65); **R1** = decider-2b (86,5); **R2** = Quyet-1.0-Medium (89,6).

---

## 0. Kết luận

1. **Hàm chuyển LLM → decision model của dự án không có lỗi đúng/sai.** Prompt dựng bằng một hàm duy nhất cho cả train và chấm (`scripts/common.py:79–93`); chữ cái A–J mỗi chữ đúng một token và không làm đổi token đứng trước (assert ở `scripts/eval_base.py:56, 63–64`); không có BOS (`add_special_tokens=False` ở cả hai đường); thinking tắt nhất quán; loss full_vocab là cross-entropy toàn từ vựng với target mềm trên k token chữ cái (`scripts/train_lora.py:185–189`), đúng phương án decider dùng khi train trên nó và đúng cách làm thường gặp của "label-word" fine-tuning (LM-BFF, arXiv 2012.15723); letter_mass sau sửa 0,999 (0/2.000 câu dưới 0,5). Đường chấm với LoRA đã merge bf16 cho val loss trùng đường train chưa merge đến chữ số thứ ba (REPORT_loss_fix, mục "Val nội bộ theo loại câu hỏi"). Quyet-1.0-Medium dùng đúng thiết kế này (chữ cái A–J qua chat template, đọc logit token kế tiếp, T theo kiểu câu) và đạt 89,6, nên **bản thân thiết kế không phải trần**. Mức chắc chắn: cao.

2. **Lỗi/thiếu sót phương pháp quan trọng nhất không nằm ở hàm chuyển mà ở công thức train: ngân sách train nhỏ hơn mọi đối thủ 1–2 bậc, và không có tín hiệu chọn model nào tương quan với mục tiêu.**
   - M1: 1.250 bước × batch 32 (20.000 dòng × 2 epoch); C: 1.938 bước. Quyet-Medium: `source.step = 120000` với cùng LoRA r16 (`results/baselines/meta/chinhnc__Quyet-1.0-Medium/quyet_config.json`), tức gấp khoảng 60–100 lần số bước. decider-2b: SFT nhiều giai đoạn trên khoảng 95 bộ dữ liệu, rồi RL, rồi LoRA r64/α128 trên 42.749 dòng (README decider dòng 17, 21, 271–284). Jeff: full fine-tune 284.747 câu, batch 256, 1 epoch. Dự án đã train 40.000–62.000 lượt dòng; +11.000 dòng tiếng Việt (+50% dữ liệu) cho +1,8 điểm là điều **nhất quán với vị trí đáy của đường cong tăng theo dữ liệu**, không phải dấu hiệu "dữ liệu không còn tác dụng".
   - Val (val_v0) cùng phân phối với train; quy tắc cấm chọn theo dev; checkpoint luôn là cuối epoch 2; chỉ thử một learning rate. Val loss của C **tốt hơn** M1 (0,757 so với 0,776) trong lúc dev chỉ +1,8 và phép thử P4 sập (tỉ lệ chọn "không đủ thông tin" 91 → 62/50%). Công thức hiện không có cơ chế nào phát hiện "thói quen xấu" trước khi chấm dev.

3. **Ba điểm phải sửa trước lần train kế, chi phí gần bằng 0:** (a) quyết định "ngừng xáo thứ tự lựa chọn của câu score" (PLAN_decisions 12/10) **chưa được thực thi**: `train_lora.py:93–95` `shuffled()` xáo mọi kiểu câu, không kiểm `kind`; lần train C ngày 26/10 vẫn chạy script này. Tác động lên bench ≈ 0 (dev 1.3 không có câu score) nhưng khoảng 5% tín hiệu train (1.463/20.000; 1.510/31.034 dòng) bị mất tính thứ bậc. (b) `--loss` mặc định vẫn là `letters_k` (dòng 53), loss đã biết hỏng; mọi script chạy đều phải nhớ truyền `--loss full_vocab`. (c) Trần `MAX_STATE_TOKENS = 1024` (`build_train_v0.py:36`) cộng với nguồn VI ngắn làm **mọi state tiếng Việt trong train_v0 ≤ 220 token** (p99 137), trong khi state ViFactCheck của dev 1.3 có trung vị 847 token, 103/316 dòng dài hơn 1.024 token, 277/316 dài hơn 512 (script `len_check.py`). C nâng p99 VI lên 724 nhưng ViFactCheck không nhích (72,8 → 72,2), nên độ phủ độ dài là **điều kiện cần, chưa đủ**.

4. **Model nền:** không có bằng chứng Qwen3-1.7B là giới hạn chính ở cỡ ~2B; bằng chứng trực tiếp là ngân sách train. Qwen3.5-2B (bản post-train) chấm bằng đúng đường chữ cái của dự án cho VI có dấu 70,6, bằng Qwen3-1.7B gốc (70,6), chậm hơn (57,7 so với 32,3 ms) và kém hẳn ở không dấu (39,0 so với 48,5); decider-2b đi từ Qwen3.5-2B-**Base** (chưa được chấm, chủ dự án đã quyết không chấm) lên 86,3 nhờ công thức train. Ràng buộc giữ 1.7B là độ trễ ≤ 40 ms; nhưng 33 ms hiện đo ở chế độ eager, phần lớn là chi phí khởi chạy kernel (0.6B và 1.7B cùng 28–29 ms ở 128 token, REPORT_ft_v0), còn gói decider mặc định dùng CUDA graphs + torch.compile. "2B–4B có qua 40 ms khi dùng CUDA graphs không" là câu hỏi mở đáng kiểm một lần (chỉ inference), chưa phải kết luận.

5. **Khuyến nghị số 1:** đổi cách đặt bài toán train từ "giữ 16.000 dòng EN cố định, chỉ thêm phần VI" sang **tăng ngân sách train theo bậc** (trước hết phần tiếng Anh sẵn có của tasksource, không cần dịch: 100.000–200.000 dòng, 1 epoch, đo đường cong 50k/100k/200k trên dev 1.3), **kèm một tập chọn model (held-out proxy) khác nguồn bench nhưng cùng dạng bài**, để công thức có tín hiệu dừng/chọn và phát hiện thói quen xấu. Căn cứ: decider-2b chỉ train tiếng Anh ("English only") mà đạt 86,3 VI có dấu; fine-tune 16k EN + 4k VI đã nâng 1.7B từ 72,9 lên 79,8 VI trên dev_v0 (trên dev 1.3: 70,6 → 79,9). Chi phí: 200.000 dòng × 1 epoch × 0,165 giây ≈ 9 giờ GPU (hệ số 0,165 s/dòng/epoch lấy từ brief train chính thức, khớp 6.619 s của M1). **Lưu ý:** khuyến nghị này đảo một quy tắc của brief train chính thức đã được chủ dự án duyệt 17/10 (mục 3.1: "giữ nguyên 16.000 dòng tiếng Anh của train_v0 làm phần cố định, chỉ phần tiếng Việt thay đổi"), nên **cần chủ dự án duyệt lại rõ ràng** trước khi làm; phần dòng EN thêm vẫn phải qua `scripts/decontam.py` và lọc `license_use` như hiện hành (giao mảng dữ liệu chọn nguồn và khử rò rỉ).

---

## 1. Câu 1: Hàm chuyển LLM → decision model đã đúng và tốt chưa?

### 1.1 Cách dự án làm (bằng chứng dòng code)

| Thành phần | Code | Nhận xét |
|---|---|---|
| Dựng prompt | `common.py:79–86` `user_message`: `{instruction}\n\n{state_label}\n{state}\n\n{question_label} {question}\n\n{options_label}\n{A. opt}\n{B. opt}...`; `common.py:89–93` `render_prompt`: `apply_chat_template([...user...], add_generation_prompt=True, enable_thinking=False)` | Một hàm cho cả train và chấm (train_lora.py:88–90, eval_base.py:58). Khung theo `row["lang"]` (vi/en) từ `config/prompt_frames.json`. Với Qwen3 hybrid, `enable_thinking=False` thêm khối `<think>\n\n</think>\n\n` rỗng vào lượt assistant; model trả chữ cái ngay sau đó. |
| Chữ cái | `common.py:13` `LETTERS = "ABCDEFGHIJ"`; `train_lora.py:77–78`, `eval_base.py:55–56`: `convert_tokens_to_ids(L)` + assert mỗi chữ đúng 1 token | Chữ cái đứng sau `\n\n` nên token là `A` không có khoảng trắng; `eval_base.py:63–64` assert `encode(p + L)[:-1] == ids` (thêm chữ cái không đổi token trước). Không BOS (Qwen không dùng BOS; `add_special_tokens=False` ở cả hai đường). |
| Đọc logit | `eval_base.py:68–78`: `logits_to_keep=1`, lấy `logits[:, -1, :]`, `softmax(logits[letter_ids[:k]])`; ghi thêm `letter_mass = Σ softmax_toàn_vocab[letters[:k]]` và `top_token` | Chuẩn. Số chính thức ở batch 1 (không padding); lượt batch 16 left-padding chỉ để đo thông lượng và so argmax. |
| noul | Dữ liệu: `build_train_v0.py:146–148, 162–165`: options = `["Yes","No"]`/`["Có","Không"]` cố định, target `[p, 1−p]` với p = P(true). Bench: `config/bench_v1_questions.json` "Thứ tự Có/Không cố định, không xáo". | Train xáo cả noul (train_lora `shuffled()`), nên model học cả hai thứ tự; bench cố định [Có, Không]. Chiều không đảo (đã kiểm ở ft_decider: 0 dòng không nhận diện, decider đúng 74,1% noul val). Ngưỡng 0,5 đã kiểm trên val_v0: t* 0,57, lợi +1,4 trên dev 1.1, không đổi bức tranh (REPORT_sensitivity mục 2). |
| score | Options = mô tả mức theo thứ tự, target mềm trên mức (`build_train_v0.py:166–169`); chấm như choice | **Bị xáo khi train** (mục 2.2). Không có dòng hướng dẫn "thang có thứ tự". Dev 1.3 không còn câu score. |
| Nhiệt độ | `fit_temperature.py`: một T chung, tối thiểu CE trên val_v0 (T = 1,56 cho M1) | Chỉ ảnh hưởng Brier, không ảnh hưởng accuracy. Đối thủ dùng T theo kiểu câu. |
| Xáo lựa chọn khi train | `train_lora.py:93–95` mỗi lần nạp hoán vị `options` và `target` | Đúng với tài liệu công khai (xáo thứ tự khi train giảm thiên vị vị trí). Nhất quán khi đảo, cùng cơ sở dev 1.2, câu VI không phải noul: dự án 90,4, decider 93,9, Quyet 97,9 (REPORT_baselines_v1_2); trên toàn dev 1.3 (mọi ngôn ngữ, mọi kiểu câu) M1 → C là 88,7 → 89,6 (REPORT_train_C P2). |
| Loss | `train_lora.py:179–190`: `full_vocab` = `−Σ target_i · log_softmax(logits)[letter_i]` | = letters_k − log letter_mass (kiểm bằng số, sai lệch ≤ 1,9e-6). Target mềm: 1.117/20.000 dòng train có target không one-hot (đếm). |

### 1.2 So với decider-2b, Quyet, Jeff, laya

| | Dự án | decider-2b (v11, gói `decider/`) | Quyet-1.0-Medium (gói `quyet` 1.0.2) | Jeff-Qwen3.5-2B | laya |
|---|---|---|---|---|---|
| Khung prompt | Chat template Qwen3, khung VI/EN theo ngôn ngữ câu, `A. opt` | **Plain, không chat template, tiếng Anh cố định**: `Context:\n…\n\nQuestion: …\nOptions:\n(A) …\nAnswer: (` (notes_code_review decider B5) | Chat template, `enable_thinking=False`, **system prompt** + `State:\n…\n\nQuestion: …\n<KIND>\n\nOptions:\nA. …\n\nAnswer with one letter.` (`quyet/llm/prompt.py:16–20, 62–73`) | JSON request state-first; head `readout.safetensors` 255 mã (không có code local) | Encoder mmBERT, mỗi lựa chọn một `[MASK]` |
| Dòng hướng dẫn theo kiểu câu | Không | Không cho choice; noul là hàng riêng `(A) no (B) yes`; score: hàng riêng mỗi mức "Does the proposed answer fit?" | **Có**: choice "Choose the option that fits best."; score "…levels are ordered from lowest to highest"; noul "Choose A if the statement is true…" | Không rõ | Loại câu ghi trong chuỗi `<type> question:` |
| Đọc xác suất | softmax trên k logit chữ cái tại token cuối | `F.linear(h_slot, lm_head.weight[letters])`, mask ngoài k, /T theo kiểu, softmax (`model.py:18–27`) | `logits[:, -1][letter_ids][:k]`, softmax(/T) (`llm/runtime.py:86–91, 106`) | head riêng + T scalar 1,138 | head 2 lớp + scorer |
| noul | 2 lựa chọn Có/Không | hàng [no, yes], P(yes) = p[1], T 1,624 | A = mô tả true, B = mô tả false, T 1,4276 | "yes/no as a probability" | thứ tự nội bộ [false, true] |
| score | chữ cái trên mức, **xáo khi train** | isolated levels: mỗi mức một hàng yes/no, chuẩn hóa | chữ cái theo thứ tự + câu "ordered from lowest to highest", T 1,6039 | "a point on a scale" | `level i: mô tả` |
| Hiệu chỉnh | 1 T chung (1,56), fit trên val_v0 | T theo kiểu (1,164 / 1,624 / 1,124), fit NLL trên 108.910 / 1.677 / 535 dòng | T theo kiểu (1,3313 / 1,6039 / 1,4276) | 1 scalar | theo kiểu + bucket số lựa chọn |
| Xử lý "không có lựa chọn đúng" | Không | `ABSTAIN_PREFIXES`/`neutralize_none` (tắt ở v11); abstention probe trong tập đánh giá | Không rõ | thêm "none of these" vào 5% câu train; 3% câu cài hijack | Không rõ |
| Train | LoRA r16/α32, 7 lớp proj, lr 2e-4, batch 32, 2 epoch, 20–31k dòng, **1.250–1.938 bước** | SFT CE trên ~95 bộ + continuation epochs; RL 384 bước; LoRA **r64/α128**, 2 epoch, 42.749 dòng; "order shuffled"; nhãn lớn sub-sample ≤ 10 lựa chọn | LoRA **r16** đã merge, **120.000 bước** (`quyet_config.json`), "also tuned for Vietnamese" | Full FT, lr 1e-5, batch 256, 1 epoch, **284.747 câu** | RLCD 15.987 updates, 4 epoch |
| Điểm dev 1.3 VI có dấu / độ trễ | 79,9 / 33 ms | 86,5 / 52 ms | 89,6 / 88 ms | bị chặn | không qua kiểm |

Ba điểm rút ra:

1. **Cốt lõi giống nhau.** Dự án, decider và Quyet đều là: một forward, đọc logit chữ cái ở token kế tiếp, softmax (có T). Khác biệt ở lớp vỏ: decider dùng khung tiếng Anh cố định không chat template; Quyet có system prompt và một câu hướng dẫn theo kiểu câu; dự án có khung tiếng Việt cho câu tiếng Việt và không có hướng dẫn theo kiểu câu. Không có bằng chứng trong dự án cho thấy lớp vỏ nào tốt hơn; ft_decider nêu "khung VI so với khung EN" như giả thuyết chưa kiểm (REPORT_ft_decider mục 4.2); phép Pb (nhãn tiếng Anh trên văn bản tiếng Việt) cho +1,5 (M1) và +3,0 (C) là gợi ý nhẹ rằng chữ tiếng Anh ở phần lựa chọn không hại, có thể lợi.
2. **Điểm dự án làm đúng hơn tài liệu/đối thủ:** loss toàn từ vựng với target mềm (đối thủ chỉ nói "cross-entropy"), xáo lựa chọn mỗi lần nạp, assert ranh giới token, kiểm đẳng thức loss bằng số, cổng letter_mass trong lúc train, so merged–unmerged. Đây là phần vững nhất của dự án.
3. **Điểm dự án thiếu so với đối thủ:** (i) hướng dẫn/định dạng riêng cho noul và score; (ii) T theo kiểu câu (chỉ Brier); (iii) không có dữ liệu/hành vi "văn bản không trả lời được → chọn không đủ thông tin" trong công thức, trong khi Jeff cài 5% "none of these" và decider có abstention probe — đúng chỗ C sập ở P4; (iv) quy mô train.

### 1.3 So với tài liệu công khai

- **Multiple-choice prompting với chữ cái (MCP)**: Robinson, Rytting, Wingate (ICLR 2023, arXiv 2210.12353) cho thấy trình bày tất cả lựa chọn kèm ký hiệu và đọc ký hiệu tốt hơn cloze khi model có "symbol binding"; đây đúng là cách dự án và Quyet/decider làm. Công trình nối tiếp (arXiv 2310.12059) đo symbol binding trên đề thi trắc nghiệm tiếng Việt.
- **Thiên vị vị trí/token của chữ cái**: Zheng et al. (arXiv 2309.03882): thiên vị chọn chữ cái tồn tại ở hầu hết LLM, chủ yếu là "token bias" của chữ cái; xáo thứ tự khi train giảm được. Dự án đã xáo và nhất quán khi đảo tăng từ 70 (gốc) lên 90,4. Các nghiên cứu 2025 (arXiv 2511.21709; SCOPE arXiv 2507.18182) đề xuất LoRA debias + majority vote nhiều hoán vị, hoặc bù phân bố vị trí; cách "nhiều hoán vị lúc chấm" nhân đôi độ trễ nên không hợp ràng buộc 40 ms của dự án.
- **Loss toàn từ vựng trên "label words"**: cách thường gặp của prompt-based fine-tuning (LM-BFF, Gao–Fisch–Chen 2021, [arXiv 2012.15723](https://arxiv.org/abs/2012.15723), và các hậu bối) là softmax toàn từ vựng tại vị trí trả lời rồi lấy xác suất ở token nhãn; `letters_k` (softmax chỉ trên k chữ cái) là biến thể lệch chuẩn, và dự án đã quan sát đúng hệ quả (logit co, letter_mass sụp) rồi sửa về chuẩn.
- **LoRA**: Biderman et al. "LoRA Learns Less and Forgets Less" (ICLR 2025, arXiv 2405.09673): dùng α = 2r, lr quét 1e-5 → 5e-4 chọn mức cao nhất còn ổn định, gắn vào mọi module; với instruction fine-tuning r cao (tới 256) đóng gần hết khoảng cách với full FT; LoRA r16–64 **không đủ** khi lượng dữ liệu/kỹ năng cần học lớn. Dự án đúng α = 2r và gắn mọi lớp proj; chưa quét lr; r16 là mức Quyet-Medium cũng dùng nhưng với số bước gấp ~100.

**Kết luận câu 1:** đúng (cao), chưa "tốt" ở ba chỗ nhỏ (noul/score không có hướng dẫn riêng, T chung, không có hành vi "không đủ thông tin" trong công thức) và một chỗ lớn thuộc câu 2 (ngân sách train).

---

## 2. Câu 2: Thuật toán train đã tối ưu chưa? Lỗi tiềm ẩn?

### 2.1 Đã kiểm, không thấy lỗi (bằng chứng)

| Hạng mục | Kết quả kiểm | Bằng chứng |
|---|---|---|
| Mask / vị trí đọc khi train | Đúng. Right padding, `attention_mask`, lấy hidden tại `len−1` của từng câu (causal: padding sau không ảnh hưởng vị trí trước) | `train_lora.py:163–176` |
| Trung bình gradient | Đúng: `mean(8)/accum` × 4 micro-batch = trung bình trên 32 câu; sắp theo độ dài chỉ giảm padding | `train_lora.py:117–123, 287–291` |
| Chữ cái nhiều token / khoảng trắng | Không: assert 1 token ở cả hai script; chữ đứng sau `\n\n` nên không có biến thể ` A`; probe cho biến thể ` A`, `Ａ` ≈ 0 sau full_vocab | `train_lora.py:78`, `eval_base.py:56, 63–64`; REPORT_loss_fix bảng probe |
| BOS | Qwen không BOS; `add_special_tokens=False` cả train và chấm | `train_lora.py:89`, `eval_base.py:59` |
| Lệch prompt train↔chấm | Không: cùng `render_prompt`; dev lưu `n_tokens` và eval assert trùng | `eval_base.py:60–62` |
| Thinking mode | `enable_thinking=False` nhất quán; không có đường nào bật thinking | `common.py:93` |
| Autocast / merge bf16 | Val CE đường chấm (merge bf16, batch 1) = đường train (chưa merge, autocast, batch 16) đến chữ số thứ ba (0,776 / 0,776); letters_k bước 0 trùng bước 3 (5,495) | REPORT_loss_fix "Val nội bộ theo loại câu hỏi" và bảng val loss |
| letter_mass sau sửa loss | 0/2.000 < 0,5; trung bình 0,999; C: 0,9977 ở bước 250, ≥ 0,998 về sau | REPORT_loss_fix; `results/train_v1/train/C-Qwen3-1.7B-s1/train_log.jsonl` |
| cuDNN SDPA trên Blackwell | Không được gọi ở cả hai đường; tắt/bật trùng từng byte | PLAN_decisions 12/10 |
| Tái lập | Chấm lại trùng từng byte (adapter cũ, decider, M1 trong lượt đo lại độ trễ 3.847/3.847) | REPORT_loss_fix, REPORT_train_C mục 2 |

### 2.2 Lỗi và thiếu sót tìm thấy

**(A) Quyết định "ngừng xáo score" chưa được thực thi — mức chắc chắn: cao.**
`train_lora.py:93–95`:
```python
def shuffled(row, rng):
    perm = rng.permutation(len(row["options"]))
    return {**row, "options": [row["options"][p] for p in perm], "target": [row["target"][p] for p in perm]}
```
không kiểm `row["kind"]`. PLAN_decisions 12/10: "Câu `score` là thang có thứ tự nên không nên xáo… Áp vào lần train chính thức (bước 6)". `scripts/train_v1/run_train_C.sh` (26/10) chạy đúng script này. Tác động: 1.463/20.000 (train_v0) và 1.510/31.034 (C) dòng score học với thứ tự mức ngẫu nhiên; `val_v0` có 84 câu score. Dev 1.3 không có câu score (đếm: kind = reading_mc/topic/intent/causal/noul/choice), nên tác động lên điểm bench ≈ 0; tác động lên model là mất tính thứ bậc ở ~5% tín hiệu. Sửa: bỏ qua hoán vị khi `kind == "score"` (và cân nhắc cả noul giữ cố định nếu bench luôn cố định — nhưng xáo noul hiện có lợi cho thiên vị vị trí, nên giữ).

**(B) Mặc định `--loss letters_k` — mức chắc chắn: cao.** `train_lora.py:53` `default="letters_k"` và docstring vẫn gọi đây là "mặc định". Loss này đã được chứng minh làm sụp letter_mass (1.919/2.000 câu < 0,5). Mọi lần chạy đúng đều nhờ script shell truyền `--loss full_vocab`. Sửa: đổi mặc định hoặc bắt buộc truyền (assert).

**(C) Độ phủ độ dài state tiếng Việt — mức chắc chắn: cao về số đếm, trung bình về tác động.** `build_train_v0.py:36` `MAX_STATE_TOKENS = 1024` (đếm bằng tokenizer Qwen3-0.6B, cùng vocab 1.7B). Đếm (`len_check.py`):

| Tập | State token p50 / p90 / p99 / max | n > 512 |
|---|---|---|
| train_v0, dòng VI (4.002) | 41 / 92 / 137 / **220** | 0 |
| train_v0, dòng EN+multi | 69 / 459 / 879 / 1.021 | 1.313 |
| C.jsonl, dòng VI (15.075) | 66 / 342 / 724 / 1.020 | 787 |
| dev 1.3 ViFactCheck (316) | **847 / 1.391 / — / 2.502** | 277 (>1.024: 103) |
| dev 1.3 Belebele / ViWikiFC | 132 / 225 / 436 ; 113 / 226 / 547 | 0 ; 3 |

M1 chưa từng thấy một state tiếng Việt dài quá 220 token, rồi bị chấm trên ViFactCheck với trung vị 847; đây cũng là nguồn thua decider nặng nhất (−10,8). Nhưng C nâng p99 VI lên 724 mà ViFactCheck vẫn 72,2, và lát "văn bản dài" của C chỉ +2,7 — vậy độ dài là điều kiện cần, không giải thích được toàn bộ. Việc chọn nguồn dài thuộc mảng dữ liệu; phần phương pháp là: trần 1.024 nên nâng lên ít nhất 2.048 cho phần VI (chi phí: micro-batch có thể phải giảm 8 → 4 ở câu dài, thời gian tăng).

**(D) Ngân sách train quá nhỏ so với mọi đối thủ, ở cùng cỡ LoRA — mức chắc chắn: cao về số, trung bình về mức lợi kỳ vọng.**

| Model | Cách train | Số bước / số dòng |
|---|---|---|
| M1 | LoRA r16, batch 32, 2 epoch | 1.250 bước; 20.000 dòng |
| C | như trên | 1.938 bước; 31.034 dòng |
| Quyet-1.0-Medium | LoRA r16 (merge) | **120.000 bước** (`quyet_config.json` "step": 120000) |
| decider-2b v11 | SFT CE ~95 bộ (1 epoch + continuation) → RL 384 bước → LoRA r64/α128 2 epoch | 42.749 dòng riêng giai đoạn LoRA; tổng không nêu, "tasksource" công khai là 2,5 triệu quyết định |
| Jeff v1.2 | Full FT, lr 1e-5, batch 256, 1 epoch | 284.747 câu (≈1.100 bước ở batch 256 nhưng mỗi bước 8× lớn hơn) |

Quỹ đạo loss của dự án cũng cho thấy chưa tới vùng bão hòa: train loss M1 ở epoch 2 là 0,47–0,54 trong khi val 0,776 (tăng từ 0,756 ở epoch 1); C: train 0,37 ở bước 1.600, val 0,757 phẳng. Model đã **khớp hết** 20–31k dòng trong 2 epoch (val tăng hoặc phẳng từ epoch 1 sang 2; dev theo mốc 25/50/75/100% của C: 81,07 / 80,17 / 81,38 / 81,65, không có xu hướng), nên **thêm epoch trên cùng dữ liệu không giúp; chỉ dữ liệu mới, đa dạng hơn mới giúp**. Căn cứ cho kỳ vọng dương khi tăng bậc: decider-2b train chỉ tiếng Anh ("English only", card) mà đạt 86,3 VI có dấu; M1 nhận +6,9 VI từ 16k EN + 4k VI trên dev_v0 (dev 1.3: 70,6 → 79,9). Không dự đoán được con số; cách kiểm là đường cong 50k / 100k / 200k.

**(E) Không có tín hiệu chọn model tương quan với mục tiêu — mức chắc chắn: cao.** val_v0 là 1.000 dòng cùng nguồn với train (tasksource + PhoLaya); quy tắc cấm dùng dev để chọn; checkpoint = cuối epoch 2 cố định; không quét lr/epoch. Hệ quả nhìn thấy: val loss C tốt hơn M1 (0,757 so với 0,776) nhưng dev chỉ +1,8 (CI chứa 0) và P4 sập; val không bắt được. Khuyến nghị (giao chung mảng dữ liệu + đánh giá): tách một **tập chọn (proxy)** vài trăm câu, cùng dạng bài bench (đọc hiểu, ý định, kiểm chứng, nhân quả) nhưng **khác nguồn** bench và không nằm trong train; dùng để chọn checkpoint/lr và đo P4-kiểu (văn bản lạc đề → không đủ thông tin). Dev 1.3 giữ nguyên vai trò báo cáo.

**(F) Biến phương pháp chưa thử, rẻ — mức chắc chắn: thấp đến trung bình (giả thuyết).**
1. **Qwen3-1.7B-Base** thay bản post-train hybrid: decider đi từ Base; model gốc hiện có logit rất nhọn (logit lớn nhất 35–51, entropy gần 0) do post-training, và prompt mang khối `<think>` rỗng. Chưa có số nào trong dự án so hai bản.
2. **Khung tiếng Anh cho câu tiếng Việt** (như decider/Quyet): giả thuyết đã nêu ở ft_decider, chưa kiểm; Pb +1,5/+3,0 gợi ý nhẹ. Kiểm rẻ: chấm **model gốc** với khung EN trên dev VI (chỉ inference, ~15 phút); nếu gốc tốt hơn rõ thì train một bản.
3. **Dòng hướng dẫn theo kiểu câu** (như Quyet): "chọn Có nếu nhận định đúng theo văn bản", "các mức xếp từ thấp đến cao". Chuỗi tiếng Việt mới phải qua `config/vi_proposals.jsonl` theo quy tắc thường trực.
4. **LoRA r64/α128**: decider dùng ở giai đoạn cuối; chi phí thời gian gần như không đổi, VRAM tăng ít. Theo Biderman et al., rank chỉ bắt đầu quan trọng khi dữ liệu lớn, nên **thử sau khi tăng dữ liệu**, không thử trước.
5. **Quét lr** 1e-4 / 4e-4: lợi kỳ vọng thấp; chỉ làm kèm (D).

**(G) Điểm nhỏ, ghi để biết.** 403 dòng train đa ngữ (không en/vi) dùng khung EN — không đáng kể. Dòng VI không dấu trên bench vẫn dùng khung VI có dấu ("Văn bản:", "Câu hỏi:") — nhất quán giữa train (strip không đụng khung) và chấm, không phải lỗi. T chung thay vì theo kiểu: chỉ ảnh hưởng Brier (0,294 so với decider 0,209), không ảnh hưởng accuracy.

### 2.3 Trả lời gọn các mục câu hỏi

| Mục | Đánh giá |
|---|---|
| LoRA r16/α32 mọi proj | Hợp chuẩn (α = 2r, mọi module). Không phải nút thắt hiện tại; thử r64 sau khi có nhiều dữ liệu hơn. |
| lr 2e-4 | Trong khoảng chuẩn; chưa quét. Lợi kỳ vọng thấp. |
| batch 32, 2 epoch | Đúng cho 20k dòng; model đã khớp hết dữ liệu sau epoch 1–2 (val tăng/phẳng). Vấn đề là **số dòng**, không phải số epoch. |
| loss full_vocab | Đúng và đã kiểm. Giữ. Đổi mặc định. |
| chat template | Nhất quán, không lỗi. Chưa so với plain/Base. |
| độ dài tối đa state | 1.024 token; VI trong train_v0 ≤ 220. Nâng trần + cần nguồn VI dài (giao mảng dữ liệu). |
| loss noul/score | noul đúng; score mất thứ bậc do xáo (lỗi A) và không có hướng dẫn thứ bậc. |
| val | Cùng phân phối train, không chọn được gì; cần tập chọn proxy (E). |
| mask / token / BOS / lệch prompt / thinking | Không thấy lỗi (2.1). |

---

## 3. Câu 3: Model nền và dung lượng

**Bằng chứng trong dự án:**

| So sánh | Số | Nguồn |
|---|---|---|
| 0.6B so với 1.7B sau fine-tune cùng công thức | 1.7B hơn +5,2 VI (4 cặp seed), Belebele +11,3; cùng độ trễ ở 128 token (28 so với 29 ms) | REPORT_ft_v0 |
| Qwen3-1.7B gốc so với Qwen3.5-2B gốc (post-train), cùng đường chữ cái | VI có dấu 70,6 = 70,6 (−0,0 [−3,1; +3,2]); EN 84,6 so với 84,5; **không dấu 48,5 so với 39,0**; độ trễ 32,3 so với 57,7 ms (không có flash-linear-attention) | REPORT_baselines_v1_2 mục 1 |
| decider-2b (từ Qwen3.5-2B-**Base**) | 86,3 VI, 52 ms; train chỉ tiếng Anh | REPORT_baselines_v1_2; card |
| decider-2b + train_v0 (ft_decider) | 84,9 VI, **thấp hơn** decider gốc 86,3 | REPORT_dev_v1_1 / BRIEF_train_chinh_thuc mục 1 |
| Quyet-1.0-Medium (Qwen3.5-4B + LoRA r16, 120k bước) | 89,4–89,6 VI, 87,5 ms | REPORT_baselines_v1_2 |
| Qwen3.5-2B-Base | **không chấm** (quyết định của chủ dự án 12/10) | PLAN_decisions |

**Cách đọc:**
1. Ở điểm xuất phát, 1.7B (Qwen3) và Qwen3.5-2B không khác nhau trên tiếng Việt có dấu với đường chữ cái; Qwen3 tốt hơn rõ ở không dấu và nhanh hơn 1,8 lần. Không có bằng chứng kiến trúc/thế hệ Qwen3.5 tự thân cho lợi thế ở cỡ 2B; lợi thế của decider đến từ công thức và quy mô train (mục 2.2 D). Vì Base chưa được chấm, kết luận này là **gián tiếp**.
2. ft_decider (train_v0 làm decider tụt) cho thấy **dữ liệu kiểu train_v0 không phải thứ decider thiếu**; nó cũng cho thấy cùng 20k dòng không "nâng" một model 2B đã mạnh. Điều đó không nói gì về trần của 1.7B khi được train ở quy mô decider.
3. Khoảng cách dung lượng thật nằm ở 4B (Quyet-Medium +9,7) nhưng độ trễ 88 ms vượt xa 40 ms ở chế độ eager.
4. **Độ trễ 33 ms của 1.7B phần lớn không phải tính toán.** Ở 128 token, 0.6B và 1.7B cùng 28–29 ms (REPORT_ft_v0), tức thời gian là chi phí cố định mỗi lượt (khởi chạy kernel, Python). Sàn băng thông của RTX 5060 Ti (~448 GB/s) cho 1.7B bf16 (~3,4 GB) là khoảng 8 ms; 2B ~9 ms; 4B ~18 ms. Gói decider mặc định dùng CUDA graphs + `torch.compile` (notes_code_review A3). Nếu dự án bật CUDA graphs cho đường chấm batch 1 (hình dạng cố định theo bucket độ dài), có khả năng 1.7B xuống ~10–15 ms và Qwen3.5-2B/4B lọt dưới 40 ms — khi đó câu hỏi model nền mở lại. Đây là **giả thuyết kiểm được bằng inference trong ~1 giờ**, chưa phải kết luận; cần kiểm preds trùng với eager.

**Kết luận câu 3:** không có bằng chứng Qwen3-1.7B là giới hạn chính ở cỡ ~2B; giới hạn được chứng minh là ngân sách/công thức train. 1.7B đang được giữ vì cổng độ trễ đo ở eager; cổng này có thể nới bằng tối ưu inference mà không đổi quy tắc "batch 1". Mức chắc chắn: trung bình (vì Qwen3.5-2B-Base chưa chấm và chưa có lần train nào ở quy mô lớn trên 1.7B).

---

## 4. Câu 4: Xếp hạng thay đổi phương pháp theo lợi ích kỳ vọng / chi phí

Chi phí GPU ước theo hệ số 0,165 s/dòng/epoch (M1: 20.000 × 2 × 0,165 = 6.600 s, khớp 6.619 s). Mọi mức lợi là kỳ vọng, không phải dự báo; cột "Kiểm" nêu số đo và cổng để biết thay đổi có tác dụng.

| # | Thay đổi | Lợi kỳ vọng (căn cứ) | Chi phí | Kiểm | Chắc chắn |
|---|---|---|---|---|---|
| 1 | **Tăng ngân sách train theo bậc, bắt đầu từ phần tiếng Anh sẵn có** (tasksource, cột `license_use`; khử rò rỉ bằng `decontam.py` như hiện hành): 50k → 100k → 200k dòng, 1 epoch, giữ công thức. **Đảo quy tắc "16.000 EN cố định" của brief đã duyệt → cần chủ dự án duyệt lại.** | Lớn nhất trong các đòn bẩy phương pháp. decider "English only" đạt 86,3 VI; M1 +6,9 VI từ 20k dòng (dev_v0; dev 1.3: 70,6 → 79,9); đối thủ ở 60–100× số bước. | 100k: ~4,6 h; 200k: ~9,2 h; đường cong 3 mốc ~16 h (có thể dùng `--save_fracs` trong một lần 200k để lấy 50k/100k miễn phí) | Dev 1.3 macro VI có dấu theo mốc; lát không dấu và EN không tụt; P4 tỉ lệ NEI ≥ mức M1 (88–91%); nhất quán khi đảo | Cao về hướng, trung bình về độ lớn |
| 2 | **Tập chọn model (proxy) khác nguồn bench**, cùng 4 dạng bài; dùng cho chọn checkpoint/lr và đo P4-kiểu; val_v0 giữ cho letter_mass/T | Cho phép early-stop và bắt thói quen xấu trước khi chấm dev; hiện val không tương quan dev (0,757 < 0,776 nhưng dev +1,8, P4 sập) | 0 GPU train; công dựng dữ liệu (giao mảng dữ liệu/đánh giá) | Tương quan điểm proxy ↔ dev 1.3 trên các adapter đã có (M1, C, C@25/50/75%) phải dương rõ | Cao |
| 3 | **Sửa `shuffled()` không xáo `score`**; đổi mặc định `--loss` sang `full_vocab` (hoặc bắt buộc truyền) | Loại một lỗi đã quyết nhưng chưa làm; tránh footgun. Lợi điểm bench ≈ 0 (dev không có score); lợi cho model/val score | ~0; kiểm bằng `--dump_batches` (token hash chỉ đổi ở dòng score) | val_v0 CE trên 84 câu score giảm; `batch_dump` giữ nguyên dòng choice/noul | Cao |
| 4 | **Kiểm CUDA graphs / torch.compile cho đường chấm batch 1** (bucket độ dài, preds phải trùng eager); nếu 2B–4B lọt < 40 ms thì mở lại câu hỏi model nền | Có thể đổi cả hướng dự án (R2 đạt 89,6); nếu không thì ít nhất 1.7B có dự phòng độ trễ cho prompt dài (p90 hiện 47 ms) | ~1 h kỹ thuật, chỉ inference | Trung vị và p90 batch 1 trên dev 1.3; argmax trùng 100% với eager; đo trên GPU trống | Trung bình |
| 5 | **Nâng trần state VI lên 2.048 token** và yêu cầu mảng dữ liệu bổ sung nguồn VI dài (kiểm chứng/đọc hiểu) | ViFactCheck p50 847 token, 103/316 > 1.024; M1 chưa thấy VI > 220 token. Điều kiện cần cho −10,8 ở ViFactCheck | Thời gian train tăng theo token; có thể micro-batch 4 | Lát "văn bản dài" và ViFactCheck trên dev 1.3 | Trung bình |
| 6 | **Hành vi "không đủ thông tin" trong công thức**: dựng cặp (nhận định, văn bản lạc đề) → NEI từ chính dòng train (như Jeff cài 5% "none of these"; khuyến nghị cách C ở FABLE_doc_ket_qua 4.2) | Trực tiếp sửa K4 (P4 91 → 62/50%) | ~0 GPU thêm (nằm trong lần train kế); công dựng dữ liệu nhỏ | P4 NEI ≥ 85% cả hai nguồn; ViFactCheck không tụt; ghi vào manifest rằng P4 không còn "chưa từng thấy" | Trung bình |
| 7 | Chấm **model gốc** với khung EN trên dev VI (inference) rồi, nếu dương, một lần train với khung EN | Pb +1,5/+3,0 và giả thuyết ft_decider; decider/Quyet đều dùng khung EN | 15 phút inference; train 1,8–2,9 h nếu làm | Dev 1.3 VI có dấu và không dấu; EN không đổi | Thấp–trung bình |
| 8 | Một lần train từ **Qwen3-1.7B-Base** (ghim revision, cùng công thức) | decider đi từ Base; bản post-train có logit nhọn bất thường. Không có số trong dự án | 1,8 h (train_v0) hoặc 2,9 h (C) | Dev 1.3 so cặp với M1/C; letter_mass; Brier | Thấp–trung bình |
| 9 | Dòng hướng dẫn theo kiểu câu (noul/score) như Quyet; chuỗi VI qua `vi_proposals.jsonl` | Cải thiện noul (dự án thua decider 2,7–4,1 ở noul có dấu) và dạy thứ bậc cho score | 1 lần train; cần duyệt 2–3 chuỗi | Lát noul dev 1.3; val score CE | Thấp |
| 10 | LoRA r64/α128 | Theo Biderman et al. chỉ có lợi khi dữ liệu lớn → làm **sau** #1 | Cùng thời gian, VRAM +ít | So cặp r16 với r64 ở cùng dữ liệu | Thấp (trước #1), trung bình (sau #1) |
| 11 | Quét lr 1e-4 / 4e-4 | Thấp | 2 lần train | Dev + proxy | Thấp |
| 12 | T theo kiểu câu (choice/noul) fit trên val_v0 | Chỉ Brier/ECE (0,294 so với 0,209 của decider); không đổi accuracy | 10 phút | Brier/ECE dev 1.3 | Cao về việc không ảnh hưởng accuracy |
| — | Không khuyến nghị: trung bình nhiều hoán vị lúc chấm (nhân đôi độ trễ, vượt 40 ms); thêm epoch trên cùng dữ liệu (val tăng/phẳng, dev theo mốc phẳng) | | | | |

Thứ tự thực hiện hợp lý: #3 (sửa code, 0 chi phí) → #2 và #6 chuẩn bị song song với dữ liệu → #4 (1 giờ, có thể đổi hướng) → #1 (lần train kế, kèm `--save_fracs` để lấy đường cong) → #7/#8 như hai lần train đối chứng nếu còn ngân sách → #10/#9/#11/#12.

---

## 5. Điều chưa chắc

- Mọi mức lợi ở mục 4 là kỳ vọng; dự án chưa có lần train nào vượt 1.938 bước, nên không có điểm dữ liệu nào trên đường cong tăng theo dữ liệu của chính Qwen3-1.7B.
- Qwen3.5-2B-Base chưa được chấm; kết luận "model nền không phải giới hạn chính" dựa trên bản post-train và trên công thức của decider, là bằng chứng gián tiếp.
- Ước lượng độ trễ với CUDA graphs (~10–15 ms) là suy luận từ sàn băng thông và từ việc 0.6B = 1.7B ở 128 token; chưa đo.
- Tác động của việc không xáo score không đo được trên bench hiện tại (không có câu score).
- ft_decider cho thấy hạng mass có thể làm ECE xấu đi ở model đã hiệu chỉnh tốt; với Qwen3 từ gốc không thấy hiện tượng này, nhưng chỉ có một seed cho mỗi công thức.

## 6. File đã dùng

- Đọc: `scripts/common.py`, `scripts/train_lora.py`, `scripts/eval_base.py`, `scripts/build_train_v0.py`, `scripts/fit_temperature.py`, `scripts/ft_decider/{train_decider,dcommon}.py`, `scripts/train_v1/run_train_C.sh`, `config/{prompt_frames,train_format,bench_v1_questions,models}.json`, `results/ft_loss/REPORT_loss_fix.md`, `results/ft/REPORT_ft_v0.md`, `results/ft_decider/REPORT_ft_decider_v0.md`, `results/baselines_v1_2/{REPORT_baselines_v1_2,notes_model_api,notes_code_review}.md`, `results/baselines/{notes_code_review,notes_model_cards}.md`, `results/baselines/pkgsrc/quyet-1.0.2/quyet/llm/{prompt,runtime}.py`, `results/baselines/meta/Mapika__decider-2b/README.md` (grep), `results/baselines/meta/chinhnc__Quyet-1.0-Medium/{quyet_config.json,README.md}` (grep), `results/train_v1/REPORT_train_C.md`, `results/train_v1/phan_tich_C/FABLE_doc_ket_qua.md`, `results/dev_v1_1/REPORT_sensitivity.md` (grep), `results/probe_v1/fable_soat/REPORT_fable.md` (grep), `results/BRIEF_train_chinh_thuc.md`, `TONG_QUAN_DU_AN.md`, `PLAN_decisions.md`, `C:\Users\dungbt06\Downloads\PLAN.md`, `results/ft_loss/train/Qwen3-1.7B-s1/{summary.json,train_log.jsonl}`, `results/train_v1/train/C-Qwen3-1.7B-s1/{summary.json,train_log.jsonl}`.
- Đếm (chỉ tổng hợp, không in dòng dữ liệu; không đọc test): `data/train_v0.jsonl`, `data/val_v0.jsonl`, `results/train_v1/components/C.jsonl`, `data/bench_v1_3/dev_v1_3.jsonl`. Script ngoài repo: `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\audit_counts.py` và `len_check.py` (cùng thư mục).
- Web (truy vấn chung, không kèm dữ liệu dự án): [Mapika/decider-2b (card hiện là v12)](https://huggingface.co/Mapika/decider-2b), [chinhnc/Quyet-1.0-Medium](https://huggingface.co/chinhnc/Quyet-1.0-Medium), [mstrasser/Jeff-Qwen3.5-2B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B), [Robinson, Rytting, Wingate 2023 — MCP và symbol binding](https://arxiv.org/abs/2210.12353v3), [symbol binding trên đề thi tiếng Việt](https://arxiv.org/html/2310.12059v5), [Zheng et al. — LLMs Are Not Robust Multiple Choice Selectors](https://arxiv.org/pdf/2309.03882), [LoRA debias + majority voting](https://arxiv.org/pdf/2511.21709), [SCOPE](https://arxiv.org/html/2507.18182v1), [Biderman et al. — LoRA Learns Less and Forgets Less](https://arxiv.org/pdf/2405.09673), [laya / typed decisions (i-scoop)](https://www.i-scoop.eu/laya-system-1-decision-model/), [benchlm decision models](https://benchlm.ai/md/decision-models.md).
