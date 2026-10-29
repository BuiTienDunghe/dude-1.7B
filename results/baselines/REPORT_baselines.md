# Chấm model ngoài trên dev_v0: Qwen3.5-2B gốc và Mapika/decider-2b

Ngày 2026-10-12. Chỉ chạy inference, không train.

- Bảng đầy đủ nằm ở `results/baselines/tables_baselines.md`; số liệu thô ở `results/baselines/metrics_baselines.json`.
- **Phạm vi:** theo yêu cầu sửa lại của người dùng, lần này chỉ chấm hai model. Quyet, Laya và Jeff để sau.
- **Mốc của dự án:** Qwen3-1.7B gốc, Qwen3-1.7B adapter full_vocab (gọi tắt là "dự án"), Qwen3-0.6B ft seed 1 và seed 2. Không chấm lại các mốc này; dự đoán lấy từ `results/base_rerun`, `results/ft_loss`, `results/ft`.

## Tóm tắt số liệu (không kết luận hướng đi)

Accuracy trung bình đều theo 4 nguồn (%), [CI 95% bootstrap theo state], thứ tự lựa chọn của dev_v0, batch 1:

| | Tiếng Việt | Tiếng Anh | Không MASSIVE (vi / en) | Tụt Anh→Việt | Nhất quán khi đảo (vi) | Brier vi |
|---|---|---|---|---|---|---|
| **decider-2b** (cách ánh xạ "text") | **86,0** [83,5; 88,3] | **89,2** [87,2; 91,2] | 83,8 / 87,6 | 3,2 | 93,3 | 0,211 |
| decider-2b (cách ánh xạ "keyed") | 83,4 [80,5; 86,0] | 88,6 [86,6; 90,6] | 82,2 / 87,6 | 5,2 | 91,1 | 0,226 |
| **Qwen3-1.7B full_vocab (dự án)** | **79,8** [77,0; 82,5] | **86,4** [84,1; 88,6] | 79,3 / 86,3 | 6,7 | 88,0 | 0,304 |
| Qwen3-0.6B ft s1 / s2 | 76,7 / 73,5 | 80,2 / 80,3 | 76,1 / 71,7 (vi) | 3,5 / 6,8 | 86,6 / 85,2 | 0,368 / 0,402 |
| Qwen3-1.7B gốc | 72,9 [69,9; 75,9] | 83,4 [81,0; 85,8] | 73,4 / 81,9 | 10,5 | 73,9 | 0,474 |
| Qwen3.5-2B gốc | 70,1 [66,9; 73,1] | 82,6 [80,0; 85,1] | 73,0 / 83,3 | 12,5 | 68,0 | 0,411 |

## Bảng cùng cỡ và ba câu hỏi

| Model | Tham số | vi | en | Dự án − model này, vi [CI] (p) | Dự án − model này, en [CI] (p) | Độ trễ trung vị batch 1 |
|---|---|---|---|---|---|---|
| decider-2b (text) | 1,88 B | 86,0 | 89,2 | **−6,2** [−9,0; −3,5] (9e-7) | **−2,8** [−5,0; −0,5] (0,005) | 48,5 ms¹ |
| decider-2b (keyed) | 1,88 B | 83,4 | 88,6 | −3,7 [−6,5; −0,9] (0,004) | −2,2 [−4,3; +0,1] (0,05) | 51,3 ms¹ |
| Qwen3.5-2B gốc | 2,27 B trong repo (chỉ nạp phần văn bản) | 70,1 | 82,6 | +9,7 [+6,2; +13,3] (1e-11) | +3,8 [+1,0; +6,5] (0,01) | 54,4 ms² |
| Qwen3-1.7B gốc | 2,03 B | 72,9 | 83,4 | +6,8 [+3,8; +9,5] (8e-10) | +3,0 [+0,9; +5,1] (0,01) | 31,2 ms |
| **Qwen3-1.7B full_vocab (dự án)** | 2,03 B + LoRA đã merge | 79,8 | 86,4 | — | — | 31,1 ms |

¹ Đo trên toàn bộ lời gọi `decide()`: tách token, ghép batch, đệm tới bội số của 64 token, rồi forward. Thư viện `flash-linear-attention` không được cài, nên các lớp linear attention chạy bằng kernel PyTorch dự phòng; model card nói kernel này chậm hơn "vài lần".
² Đo bằng đường chấm của dự án (chỉ forward), cũng không có `flash-linear-attention`.
Vì hai lý do trên, độ trễ của hai model Qwen3.5 **không so trực tiếp được** với Qwen3; nhiều khả năng con số của chúng đang bị thổi lên.

**1. Trên tiếng Việt, dự án kém decider-2b 6,2 điểm** [3,5; 9,0], McNemar p=9e-7, dùng cách ánh xạ cho decider điểm cao hơn ("text").
- decider-2b hơn ở Belebele (+5,7; p=0,04) và MASSIVE (+11,7; p=5e-10); SIB-200 (+0,7) và XCOPA (+7,0; p=0,17) không có ý nghĩa thống kê.
- Model card cho thấy MASSIVE có thể đã được decider-2b nhìn thấy (bảng dữ liệu bên dưới). Nếu bỏ MASSIVE, dự án vẫn kém **4,4 điểm** [1,1; 7,8] (p=0,02).
- Dự án hơn Qwen3.5-2B gốc 9,7 điểm và hơn Qwen3-1.7B gốc 6,8 điểm.

**2. Trên tiếng Anh, dự án kém decider-2b 2,8 điểm** [0,5; 5,0] (p=0,005).
- Phần lớn khoảng cách nằm ở MASSIVE (+7,3; p=1e-5). Nếu bỏ MASSIVE, chênh lệch chỉ còn **1,2 điểm** [−1,5; +4,0] (p=0,39), tức không phân biệt được.
- decider-2b tụt từ Anh sang Việt ít hơn: 3,2 điểm, so với 6,7 của dự án.
- Như vậy, trên ba nguồn không bị nghi đã thấy, khoảng cách tiếng Anh nhỏ và không có ý nghĩa thống kê, còn khoảng cách tiếng Việt (4,4) lớn hơn. Hai mô tả đều đi kèm cảnh báo ở mục "Không chắc".

**3. Qwen3.5-2B gốc kém Qwen3-1.7B gốc 2,8 điểm trên tiếng Việt** [−6,5; +0,8] (p=0,19) **và 0,8 điểm trên tiếng Anh** [−3,8; +2,0] (p=0,69). Không chênh lệch nào có ý nghĩa thống kê.
- Theo nguồn: Qwen3.5-2B kém rõ ở MASSIVE (vi −10,0, p=0,002; en −7,7, p=1e-4). Ở Belebele (+3,3 / +3,0) và SIB-200 (+1,3 / +3,3) nó hơn một chút nhưng không có ý nghĩa; ở XCOPA kém (−6,0 / −2,0), cũng không có ý nghĩa.
- Nếu bỏ MASSIVE: vi 73,0 so với 73,4; en 83,3 so với 81,9.
- Qwen3.5-2B gốc thiên vị vị trí nặng hơn: tỉ lệ nhất quán tiếng Việt 68,0 so với 73,9 của Qwen3-1.7B gốc.
- Lưu ý: "Qwen/Qwen3.5-2B" là bản **đã post-train** (chat, có chế độ thinking và non-thinking). decider-2b được fine-tune từ **Qwen3.5-2B-Base**, một repo khác, và repo đó chưa được chấm.

**Dự án đứng ở đâu:**
- **Nhóm cùng cỡ:** dưới decider-2b ở cả hai ngôn ngữ; khoảng cách rõ ở tiếng Việt, còn ở tiếng Anh thì nhỏ và không có ý nghĩa khi bỏ MASSIVE. Trên hai model nền gốc, Qwen3-1.7B và Qwen3.5-2B, ở cả hai ngôn ngữ.
- **Nhóm nhỏ hơn và nhóm lớn hơn:** chưa có số. Quyet Tiny/Small, Laya và Quyet Medium đã được hoãn theo yêu cầu. Model nhỏ hơn duy nhất có số là Qwen3-0.6B ft của chính dự án.

## Bảng dữ liệu có thể đã thấy (theo model card và file cấu hình; không suy đoán thành kết luận)

| Model | Model card nói gì về dữ liệu train | MASSIVE | Belebele | SIB-200 | COPA/XCOPA |
|---|---|---|---|---|---|
| decider-2b | Khoảng 95 bộ dữ liệu công khai (intent, phân loại chủ đề, NLI, đọc hiểu, QA trắc nghiệm…), trajectory của agent, Mind2Web, câu hỏi do teacher viết, các trạng thái game. Danh sách đầy đủ nằm trong `decider/data/mixture.py` trên GitHub, **không có trong repo HF**. Giai đoạn LoRA v11 dùng thêm MMLU, ARC, CommonsenseQA, BoolQ, MNLI, SNLI, Banking77, RACE, OpenBookQA, LogiQA 2, MedQA, Winogrande. | **Có thể đã thấy; card mâu thuẫn.** Bảng Bespoke và `eval_results.json` ghi `massive-en-US` "trained: true", nhưng mục Evaluation liệt kê "MASSIVE scenario" là held-out. Không rõ nhãn intent của bản tiếng Việt có được dùng hay không. | Không nêu | Không nêu | Không nêu (XStoryCloze và CommitmentBank được ghi là held-out) |
| Qwen3.5-2B gốc | Không mô tả dữ liệu train (chỉ ghi "Pre-training & Post-training") | Không rõ | Không rõ | Không rõ | Không rõ |
| Dự án (Qwen3-1.7B full_vocab) | Pretrain của Qwen3: không rõ. Fine-tune trên train_v0, đã loại MASSIVE, SLURP, IntentGrasp, COPA, XCOPA, Belebele, SIB-200; số dòng trùng với dev/test = 0 | Fine-tune: không; pretrain: không rõ | như trên | như trên | như trên |

## Kiểm tra adapter

- **Qwen3.5-2B gốc:**
  - Chấm bằng đúng `scripts/eval_base.py` trong `.venv` chính, không sửa code chấm. transformers 5.19 nạp model dưới dạng `Qwen3_5ForCausalLM`, chỉ phần văn bản.
  - A..J là một token ở ranh giới trả lời với tokenizer Qwen3.5 (đây cũng là phép assert có sẵn trong eval_base).
  - Vì `n_tokens` của dev_v0 được đếm bằng tokenizer Qwen3, tôi chấm trên một bản sao dev_v0 giống hệt, chỉ khác là để trống `n_tokens` (`results/baselines/dev_v0_ntok_null.jsonl`, đã assert mọi trường khác trùng).
  - Câu hiển nhiên: 10/10 tiếng Anh và 10/10 tiếng Việt chọn đúng nội dung ở cả hai thứ tự.
- **decider-2b:**
  - Dùng gói `decider/` đi kèm repo ở revision đã ghim. Trước khi chạy, script assert từng file `.py` trong snapshot trùng sha256 với bản đã được đọc kiểm.
  - Chạy đường eager (`use_graphs=False`, không `torch.compile`, không vá kernel). Giữ temperature phát hành: assert T = 1,145 và T theo loại {choice 1,164; noul 1,624; score 1,124}; câu choice dùng 1,164.
  - Ví dụ README: billing 0,926 và yes 0,703, khớp với 0,93 và 0,70 mà card ghi.
  - Cách ánh xạ "keyed" (`(A) opt1: <nội dung>`) cho chuỗi token trùng với đường `system_one` của chính gói khi khai báo criteria {nhãn: mô tả}, 5/5 câu.
  - Câu hiển nhiên: 10/10 tiếng Anh và 10/10 tiếng Việt ở cả hai thứ tự, cả hai cách ánh xạ.
  - Mọi câu dev: xác suất cộng lại bằng 1 (sai lệch dưới 1e-4) và có đúng k phần tử (assert trong lúc chạy).
- **Cắt cụt:**
  - decider-2b mặc định cắt state ở 1.536 token mà không báo. Tôi đặt giới hạn 32.768; không câu dev nào vượt 1.536, nên 0 câu bị cắt và 0 câu bị loại.
  - Qwen3.5-2B: không có cắt cụt (eval_base đưa nguyên prompt; ngữ cảnh tối đa của model là 262.144 token).
- **Thứ tự lựa chọn:** giữ thứ tự của dev_v0, và chấm thêm một lượt thứ tự đảo cho cả hai model. Không fit lại gì.
- **trust_remote_code:** **không model nào cần.** decider-2b dùng lớp `Qwen3_5ForCausalLM` có sẵn trong transformers; gói `decider/` được import từ snapshot đã ghim.

## Bảo mật code bên thứ ba (`decider/` của decider-2b)

- Subagent đã đọc toàn bộ 19 file, tôi rà lại bằng grep. Không có lệnh gọi mạng nào ngoài HF Hub, không ghi file, không có exec/eval chuỗi, không có pickle hay `torch.load`, không có subprocess, không có code bị làm rối.
- Biến môi trường chỉ được đọc ở phần server, chế độ MPS và một tham số bộ nhớ.
- Một hành vi cần biết: gói này tự tắt backend cuDNN SDPA cho cả tiến trình, kèm chú thích rằng backend đó "sai với masked attention trên Blackwell". RTX 5060 Ti thuộc thế hệ Blackwell. Tôi chưa kiểm chứng điều này có ảnh hưởng tới các lượt chấm của dự án hay không (mục "Không chắc").
- Một bẫy đã chặn: nếu không đọc được `decider_config.json`, gói lặng lẽ chạy với T = 1. Adapter có assert để bắt trường hợp này.

## Những gì tôi không chắc

- **Có thể dự án đang bị đánh giá thấp hơn thực tế ở MASSIVE.** Lựa chọn của MASSIVE trong dev_v0 là mô tả ý định do agent viết, chưa ai duyệt (`config/translations.json`). Điều kiện này giống nhau cho mọi model, nhưng decider-2b có thể đã quen với tập nhãn MASSIVE (en-US). Vì vậy so sánh không có MASSIVE là con số thận trọng hơn.
- **decider-2b chấm tiếng Việt bằng prompt tiếng Anh cố định của tác giả** ("Context / Question / Options / Answer: ("), với nội dung tiếng Việt. Dự án dùng khung prompt tiếng Việt. Mỗi bên dùng đúng format của mình.
- **Độ trễ không so được trực tiếp** (chú thích ¹ ²). Với `flash-linear-attention`, hai model Qwen3.5 có thể nhanh hơn nhiều so với số ở đây.
- **Adapter của dự án mới có 1 seed.** Ở bước 3, nhiễu giữa hai seed của 1.7B là 0,9 điểm tiếng Việt, nhỏ hơn nhiều so với khoảng cách 6,2 điểm.
- **cuDNN SDPA trên Blackwell:** nên kiểm tra bằng cách chấm lại một model của dự án với `torch.backends.cuda.enable_cudnn_sdp(False)` rồi so từng byte. Phép kiểm này chưa làm.
- **Hai loại giá trị p:** p "gộp" là McNemar trên toàn bộ câu (nguồn đông câu nặng ký hơn), còn CI là bootstrap trên trung bình đều theo nguồn. Vì vậy đôi khi CI chứa 0 mà p gộp vẫn nhỏ, ví dụ 0.6B s1 không tính MASSIVE.

## Những gì đã có trên đĩa cho các model bị hoãn (không xóa)

- `results/baselines/meta/`: README và file cấu hình (chỉ văn bản, không có trọng số) của Quyet-1.0-Tiny/Small/Medium, laya-multilingual, laya-vi, Jeff-Qwen3.5-2B.
- `results/baselines/pkgsrc/`: wheel `quyet-1.0.2` và `laya-0.4.6` đã tải và giải nén để đọc mã nguồn, **chưa cài**. Đã kiểm sha256 khớp với PyPI. Subagent đọc một phần trước khi phạm vi đổi; đó **không phải** một lần rà soát hoàn chỉnh.
- `results/baselines/pypi/*.json`: metadata PyPI.
- Không tạo `.venv-baselines`: không model nào trong phạm vi mới cần thư viện mới. decider-2b chạy được bằng transformers 5.19 sẵn có; mã `decider/` được import từ snapshot, không cài gì vào `.venv` chính.
- HF cache trong WSL có thêm trọng số của Qwen/Qwen3.5-2B và Mapika/decider-2b.

## Ghim và file

- **Revision:**
  - Qwen/Qwen3.5-2B `15852e8c16360a2fea060d615a32b45270f8a8fc`, đã thêm vào `config/models.json`; các khóa cũ giữ nguyên.
  - Mapika/decider-2b `ff99089b6810691bd79134ec6febb5d9afa6e619`.
  - Gói đã tải: quyet 1.0.2, laya 0.4.6 (chưa dùng). Revision của các model bị hoãn ở `results/baselines/pins_models.json`.
- **Môi trường:** RTX 5060 Ti 16 GB, WSL2, torch 2.11.0+cu128, transformers 5.19.0, bf16, batch 1. Mỗi lượt chấm có giới hạn thời gian (20–120 phút); không lượt nào bị treo.

| File | Nội dung |
|---|---|
| `results/baselines/eval/{normal,reverse}/{Qwen3.5-2B,decider-2b-text,decider-2b-keyed}/` | dự đoán thô từng câu và `timing.json` |
| `results/baselines/obvious/`, `obvious_v0.jsonl` | kiểm tra bằng câu hiển nhiên |
| `results/baselines/eval/checks_decider.json` | ví dụ README và phép kiểm tương đương keyed/system_one |
| `results/baselines/notes_model_cards.md`, `notes_code_review.md` | ghi chú đọc model card và rà code (do subagent Sonnet viết; các điểm then chốt đã được tôi kiểm lại) |
| `scripts/baselines/*.py`, `setup/baselines_*.sh`, `setup/eval_qwen35.sh`, `setup/decider_checks.sh` | code |
