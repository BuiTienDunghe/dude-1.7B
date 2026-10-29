# Sửa công thức loss: kiểm chứng trên Qwen3-1.7B seed 1

Ngày 2026-10-12.

- Bảng đầy đủ nằm ở `results/ft_loss/tables_loss_fix.md`; số liệu thô ở `results/ft_loss/metrics_loss_fix.json`.
- So sánh với adapter 1.7B seed 1 của bước 3 (`results/ft/train/Qwen3-1.7B-s1`). Hai lần train giống hệt nhau, **chỉ khác loss**: cùng train_v0, cùng seed, LoRA r=16, 2 epoch, batch hiệu dụng 32, lr 2e-4, bf16, cùng cách xáo lựa chọn (kể cả câu score).

## Kết luận: đạt cả ba tiêu chí

| Tiêu chí | Adapter mới (full_vocab) | Adapter cũ (letters_k) | Ngưỡng | Kết quả |
|---|---|---|---|---|
| letter_mass trên dev_v0 | **0**/2.000 câu < 0,5; trung bình **0,999**; nhỏ nhất 0,629 | 1.919/2.000; trung bình 0,165 | ≤ 20 câu; ≥ 0,95 | **ĐẠT** |
| Accuracy tiếng Việt (trung bình đều) | 79,8 [77,0; 82,5] | 79,8 [77,0; 82,6] | không thấp hơn cũ quá 2 điểm | **ĐẠT** (−0,1 [−2,3; +2,2]; McNemar p=1) |
| Sinh greedy 1 token, 100 câu dev | **100/100** chữ cái hợp lệ (chặt); 0 biến thể; 100/100 trùng đường đọc logit | 100/100; 0; 100/100 | ≥ 99 | **ĐẠT** |

Theo brief, tôi đã ghi vào `PLAN_decisions.md` rằng lần train chính thức dùng loss mới.

**Lưu ý về tiêu chí sinh greedy:** adapter cũ cũng đạt 100/100. Dù phân phối gần như phẳng, token đứng đầu của nó vẫn là chữ cái đúng, nên greedy vẫn trả ra chữ cái. Vì vậy tiêu chí này không phân biệt được hai adapter. Chỗ hỏng thật của adapter cũ nằm ở **xác suất**: nếu sinh có lấy mẫu (sampling) ở nhiệt độ 1, hoặc đọc xác suất trên toàn từ vựng, thì 80–96% khối xác suất rơi ra ngoài các chữ cái. letter_mass là tiêu chí phản ánh đúng điều đó.

## Công thức và kiểm tra trước khi train

- **Loss mới (`--loss full_vocab`):**

      loss = − Σ_i target_i · log p_vocab(chữ cái i)
           = loss cũ − log letter_mass

  Loss cũ được giữ nguyên dưới tên `--loss letters_k` và vẫn là mặc định, để tái lập được bước 3. Val loss ở bước 0 tính theo letters_k là 5,495, trùng với bước 3.
- **Kiểm đẳng thức bằng số** (32 câu val; `results/ft_loss/identity_*/identity_check.json`):
  - Trên model gốc: sai lệch tuyệt đối lớn nhất 1,9e-6.
  - Trên adapter cũ, nơi −log letter_mass lên tới 6,8: sai lệch lớn nhất 4,8e-7.
  - Cả hai đều chỉ ở mức làm tròn float32.
- **Cổng chặn:** chấm lại adapter cũ cho `preds.jsonl` trùng **từng byte** với bản đã lưu (sha256 `75682ce2…f6f1`). Đường chấm không bị đổi.
- **Theo dõi trong lúc train:** đo letter_mass trên 200 câu đầu cố định của val_v0, mỗi 125 bước (10%), ở chế độ eval/no_grad (không đổi trọng số, không tiêu RNG). Kết quả: 0,786 (bước 0) → 0,996 (bước 125) → 0,999 (bước 250, qua cổng 20%) → giữ khoảng 0,999 đến bước 1.250. Không bước đo nào có câu dưới 0,5.

## So adapter mới với adapter cũ

**Accuracy (%) [CI 95%]; McNemar trên cùng câu:**

| | Gốc | Cũ | Mới | Mới − cũ (p) |
|---|---|---|---|---|
| vi — Belebele | 67,3 | 73,7 | 77,3 | +3,7 (0,07) |
| vi — SIB-200 | 74,0 | 81,3 | 81,7 | +0,3 (1) |
| vi — MASSIVE | 71,3 | 84,3 | 81,0 | −3,3 (0,05) |
| vi — XCOPA | 79,0 | 80,0 | 79,0 | −1,0 (1) |
| **vi — trung bình đều** | 72,9 | **79,8** [77,0; 82,6] | **79,8** [77,0; 82,5] | **−0,1** [−2,3; +2,2] (1) |
| en — trung bình đều | 83,4 | 87,2 | 86,4 | −0,8 [−2,1; +0,4] (0,58) |

Không nguồn nào chênh có ý nghĩa ở mức 0,05. Mức dao động ±3 điểm theo nguồn cũng cỡ chênh lệch giữa seed 1 và seed 2 của 1.7B ở bước 3.

**Brier** (trung bình đều; T fit trên val nội bộ):

| | Brier vi, T=1 | Brier vi sau T | Brier en, T=1 | Brier en sau T | T |
|---|---|---|---|---|---|
| Cũ | 0,296 | 0,292 | 0,205 | 0,208 | 1,41 |
| Mới | 0,304 | 0,290 | 0,213 | 0,208 | 1,56 |

**Đảo thứ tự lựa chọn** (tiếng Việt / tiếng Anh, trung bình đều theo nguồn):

| | acc TB hai lượt | Nhất quán | Chênh acc giữa các vị trí đáp án đúng |
|---|---|---|---|
| Cũ | 79,9 / 86,6 | 90,0 / 93,2 | 11,7 / 10,7 |
| Mới | 79,9 / 86,4 | 88,0 / 93,7 | 14,6 / 9,5 |

**letter_mass** (trung bình trên 2.000 câu của dev_v0; thứ tự thường / đảo):

| | Số câu < 0,5 | Nhỏ nhất | Trung bình (vi / en) | Val nội bộ: số < 0,5 / trung bình |
|---|---|---|---|---|
| 1.7B gốc | 18 / 14 | 0,003 | 0,988 / 0,982 | 204 / 0,793 |
| Cũ | 1.919 / 1.904 | 0,001 | 0,140 / 0,189 | 988 / 0,051 |
| **Mới** | **0 / 0** | 0,629 | 0,999 / 1,000 | **0 / 0,999** |

Ở bảng này trung bình được tính trên mọi câu. `REPORT_ft_v0.md` tính trung bình của trung bình từng nguồn, nên số của adapter cũ ở đó hơi khác (0,120 / 0,162).

**probe_full_softmax.py trên đúng 5 câu cũ:**

| | Khối trên k chữ cái | Biến thể " A" | Biến thể "Ａ" | Entropy toàn từ vựng (nat) | Logit lớn nhất | Độ lệch chuẩn logit |
|---|---|---|---|---|---|---|
| 1.7B gốc | 0,87–1,00 | | | 0,00–0,75 | 35–51 | khoảng 4,1 |
| Cũ | 0,035–0,50 | 0,0003–0,0026 | 0,0013–0,0035 | 6,1–11,0 | 7,3–10,9 | 0,92–1,11 |
| Mới | **0,999–1,000** | khoảng 0 | khoảng 0 | 0,04–0,69 | 26–28,5 | 1,65–1,89 |

Với adapter mới, top-3 ở cả 5 câu đều là các chữ cái A–J. Logit lớn nhất vẫn thấp hơn model gốc, nhưng khoảng cách với phần còn lại của từ vựng đủ lớn để khối xác suất nằm trọn trên các chữ cái.

**Val nội bộ theo loại câu hỏi** (CE trên k chữ cái / CE toàn từ vựng / accuracy theo target_argmax):

| | choice (742) | noul (174) | score (84) | Tất cả (1.000) |
|---|---|---|---|---|
| Gốc | 5,714 / 6,803 / 47,6% | 3,954 / 6,579 / 61,5% | 6,655 / 10,410 / 36,9% | 5,487 / 7,067 / 49,1% |
| Cũ | 0,798 / 5,278 / 67,3% | 0,455 / 5,256 / 78,2% | 1,065 / 5,426 / 57,1% | 0,761 / 5,287 / 68,3% |
| Mới | 0,819 / 0,820 / 68,2% | 0,446 / 0,446 / 79,3% | 1,075 / 1,075 / 54,8% | 0,776 / 0,776 / 69,0% |

Số ở bảng này lấy từ đường chấm (LoRA đã merge, bf16, batch 1). Số đo trong lúc train ở bảng sau dùng LoRA chưa merge, với autocast và batch 16, nên lệch nhau ở chữ số thứ ba.

**Val loss trong lúc train, tách hai phần** (val nội bộ 1.000 câu). (b) = full_vocab − letters_k, đúng bằng trung bình −log letter_mass vì Σ target = 1.

| Điểm đo | (a) letters_k, mới | (b) −log letter_mass, mới | full_vocab = (a)+(b), mới | letters_k, adapter cũ |
|---|---|---|---|---|
| Bước 0 | 5,495 | 1,523 | 7,018 | 5,495 |
| Cuối epoch 1 | 0,754 | 0,001 | 0,756 | 0,771 |
| Cuối epoch 2 | 0,776 | 0,001 | 0,776 | 0,761 |

- Số hạng (b) về gần 0 ngay trong 10% số bước đầu, và không làm phần (a) xấu đi một cách hệ thống.
- Phần (a) của adapter mới tốt hơn adapter cũ ở epoch 1 (0,754 so với 0,771) nhưng kém hơn ở epoch 2 (0,776 so với 0,761). Giống bước 3, val loss tăng nhẹ ở epoch 2.

**Train:** 110,3 phút so với 106,1 phút; thêm khoảng 4 phút là do 11 lần đo letter_mass. VRAM đỉnh 10,2 GB ở cả hai.

## Những gì tôi không chắc

- **Chỉ có 1 seed cho loss mới.** Các chênh lệch theo nguồn đều nằm trong mức nhiễu giữa seed đã thấy ở bước 3: MASSIVE vi −3,3 (p=0,05), Belebele vi +3,7 (p=0,07), score trên val −2,3 điểm. Chưa thể nói chúng là thật.
- **Không tách được hiệu ứng của loss khỏi tính không tất định của GPU.** Train trên GPU không tất định từng bit (gradient checkpointing, cộng dồn bf16), nên dù mọi thiết lập giống nhau, hai lần train với cùng loss cũng sẽ khác nhau một chút.
- **Tiêu chí sinh greedy không phân biệt được hai adapter** (cả hai đều 100/100). Nó chỉ xác nhận adapter mới không làm hỏng đường sinh greedy. Bằng chứng chính cho việc sửa thành công là letter_mass và entropy toàn từ vựng.
- **Brier ở T=1 của adapter mới hơi kém hơn** (vi 0,304 so với 0,296), và T lớn hơn (1,56 so với 1,41): model mới tự tin hơn một chút. Sau khi áp T thì hai adapter ngang nhau.
- **Lý do loss cũ làm 1.7B co logit mạnh hơn 0.6B** vẫn chưa giải thích được. Loss mới khắc phục hiện tượng này bất kể nguyên nhân là gì.
- **letter_mass nhỏ nhất trên dev là 0,629** (một câu). Tôi chưa soi câu đó.

## Việc chưa làm (đã ghi trong PLAN_decisions.md)

- Ngừng xáo thứ tự lựa chọn của câu `score`: để dành cho lần train chính thức, chưa áp ở lần này.

## File

| File | Nội dung |
|---|---|
| `scripts/train_lora.py` | thêm `--loss {letters_k, full_vocab}`, `--monitor_frac`, cổng `--gate_letter_mass`, `--check_identity` |
| `results/ft_loss/train/Qwen3-1.7B-s1/` | adapter, `train_log.jsonl` (có letter_mass mỗi 10% số bước và val loss theo hai định nghĩa), `summary.json` |
| `results/ft_loss/eval/{normal,reverse,val}/Qwen3-1.7B-fullvocab-s1/` | dự đoán thô từng câu |
| `results/ft_loss/eval/normal/Qwen3-1.7B-ft-s1-recheck/` | adapter cũ chấm lại (trùng từng byte) |
| `results/ft_loss/identity_{base,old}/identity_check.json` | kiểm đẳng thức |
| `results/ft_loss/probe_{old,new}.json`, `greedy_{old,new}.json` | probe toàn từ vựng và sinh greedy |
| `results/ft_loss/temp/Qwen3-1.7B-fullvocab-s1.json` | temperature |
| `scripts/greedy_gen.py`, `scripts/report_loss_fix.py`, `setup/pipeline_loss.sh` | code |
