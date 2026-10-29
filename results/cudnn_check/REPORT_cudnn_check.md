# Kiểm cuDNN SDPA trên RTX 5060 Ti (Blackwell)

Ngày 2026-10-12. Chỉ kiểm, không train lại gì.

- Không sửa `.venv`, `eval_base.py` hay `train_lora.py`. Script mới nằm ở `scripts/cudnn_check/`, kết quả ở `results/cudnn_check/` (`backend_state.json`, `kernels.json`, `train_path.json`, `run.log`).
- Lý do kiểm: gói `decider/` của decider-2b tự tắt cuDNN SDPA, kèm chú thích "sai với masked attention trên Blackwell (torch 2.14)".

## Kết luận

**Không có bằng chứng cuDNN SDPA làm sai kết quả của dự án trên máy này.** Lý do đơn giản: ở cấu hình của dự án, cuDNN SDPA **không hề được gọi**.

- Đường chấm dùng flash attention.
- Đường train dùng memory-efficient attention.
- Tắt cuDNN không làm thay đổi một bit nào, ở cả đường chấm lẫn đường train.

## 0. Trạng thái mặc định trong `.venv` chính

- **Phần mềm và phần cứng:** torch 2.11.0+cu128, cuDNN 9.19.0, transformers 5.19.0, RTX 5060 Ti (sm_120).
- **Cờ SDPA mặc định:** `cudnn_sdp_enabled` = True, `flash_sdp_enabled` = True, `mem_efficient_sdp_enabled` = True, `math_sdp_enabled` = True. Nghĩa là cuDNN **được phép** chạy, nhưng PyTorch không chọn nó.
- **attn_implementation:** transformers chọn `sdpa` cho Qwen3, cả khi dự án chỉ định rõ lẫn khi để mặc định.
- **Kernel thực sự được gọi** (torch profiler, `kernels.json`):

| Tình huống | Kernel attention |
|---|---|
| Đường chấm: batch 1, không mask, inference | `aten::_flash_attention_forward` × 28 (flash) |
| Đường train: 8 câu right-padding có mask, bf16 autocast, gradient checkpointing, forward và backward | `aten::_efficient_attention_forward` × 56 (28 lớp × 2 vì checkpointing tính lại), `aten::_efficient_attention_backward` × 28 (memory-efficient) |

Không có kernel cuDNN nào xuất hiện.

## 1. Đường chấm: chấm lại adapter full_vocab với cuDNN SDPA tắt

- **Cách chạy:** dùng đúng `scripts/eval_base.py` (gọi qua runpy từ một wrapper đặt `torch.backends.cuda.enable_cudnn_sdp(False)`). Cùng tham số với lượt gốc: Qwen3-1.7B, `--adapter results/ft_loss/train/Qwen3-1.7B-s1/adapter`, dev_v0, batch 1. Timeout 30 phút.
- **Kết quả:** `preds.jsonl` **trùng từng byte** với `results/ft_loss/eval/normal/Qwen3-1.7B-fullvocab-s1/preds.jsonl`. Cả hai có sha256 `e26056b3…bbfa`.
- Vì vậy không có câu nào đổi quyết định, và accuracy, logit, xác suất đều chênh 0.

## 2. Đường train: một micro-batch 8 câu có padding, ba chế độ

**Cách dựng:**
- 8 câu train_v0 trải từ ngắn nhất tới dài nhất (65, 89, 90, 99, 147, 192, 377, 1.082 token). Đệm phải tới 1.082 token, nên câu ngắn nhất có 1.017 token đệm.
- Dựng đúng như `train_lora.py`: `render_prompt`, đệm phải, lấy hidden state tại vị trí len−1 rồi qua `lm_head`, bf16 autocast.
- Qwen3-1.7B gốc + adapter full_vocab, LoRA chưa merge như lúc train, chế độ train, gradient checkpointing.
- LoRA dropout được giữ giống hệt giữa các chế độ bằng cách đặt cùng seed trước mỗi forward. Backward của loss full_vocab dùng để so chuẩn gradient.

**So giữa các chế độ** (max |Δ| trên 8 câu):

| Cặp | Δlogit, k chữ cái | Δlogit, toàn từ vựng | \|Δloss\| letters_k (TB / câu lớn nhất) | \|Δloss\| full_vocab (TB / câu lớn nhất) | Δ chuẩn gradient LoRA |
|---|---|---|---|---|---|
| mặc định vs tắt cuDNN | **0** | **0** | **0 / 0** | **0 / 0** | **0** |
| mặc định vs chỉ math | 0,25 | 0,25 | 0,016 / 0,124 | 0,016 / 0,124 | 0,38 (4,3%) |
| tắt cuDNN vs chỉ math (mốc nhiễu bf16) | 0,25 | 0,25 | 0,016 / 0,124 | 0,016 / 0,124 | 0,38 (4,3%) |

**Kiểm trực tiếp lỗi mask:** so logit ở vị trí trả lời của từng câu trong batch có padding với chính câu đó chạy riêng ở batch 1, không padding, cùng chế độ.

| Chế độ | max \|Δlogit\|, k chữ cái | max \|Δlogit\|, toàn từ vựng | Đáp án (argmax trên k) trùng nhau |
|---|---|---|---|
| mặc định | 0,25 | 0,47 | 7/8 |
| tắt cuDNN | 0,25 | 0,47 | 7/8 |
| chỉ math | 0,25 | 0,38 | 8/8 |

Mức chênh 0,125–0,25 ở logit (độ lớn khoảng 20–30) là một đến hai bước làm tròn bf16. Mức này xuất hiện ở cả ba chế độ, kể cả math (không dùng kernel nhanh nào), nên đó là nhiễu bf16 do khác hình dạng batch, không phải lỗi mask. Nếu mask hỏng, câu ngắn với hơn 1.000 token đệm sẽ lệch lớn hơn nhiều; thực tế câu đó chỉ lệch 0,125.

## Những gì tôi không chắc

- **Bug mà decider nhắc tới có thể vẫn có thật, chỉ là không chạm tới dự án.** Chú thích của decider nói về torch 2.14; máy này dùng torch 2.11. Nếu sau này nâng torch hoặc transformers, hoặc thay đổi khiến PyTorch chọn cuDNN SDPA (ví dụ mask dạng khác, head_dim khác, model khác), kết luận này cần kiểm lại. Phép kiểm này chỉ bao trùm Qwen3-1.7B, đường chấm batch 1 và đường train có mask của dự án.
- **Có một câu đổi đáp án giữa batch có padding và batch 1** ở chế độ mặc định (7/8), trong khi chế độ math cho 8/8. Chênh logit lớn nhất vẫn là 0,25 như math, nên nhiều khả năng đó là một câu có hai lựa chọn sát nhau bị nhiễu bf16 đẩy qua ngưỡng. Với 8 câu thì chưa đủ để nói nhiều hơn.
- **Phần đo thông lượng ở batch 16 trong `eval_base.py`** (đệm trái, có mask) không được kiểm riêng. Số chính thức của dự án luôn lấy từ batch 1.
- **Chuẩn gradient lệch 4,3% giữa memory-efficient và math.** Đây là khác biệt số học bình thường giữa hai kernel bf16, không phải dấu hiệu lỗi. Tôi không đo mức chênh này tích lũy thế nào qua cả lần train.
