# Train thử từ decider-2b (ft_decider_v0), một lần train

Ngày 2026-10-13. Đúng một lần train, seed 1. Chỉ chấm trên dev_v1; không mở test 1.0. **Adapter chỉ dùng nội bộ, chưa công bố**, vì train_v0 có ba bộ UIT (ViNLI, VSFC, ViSFD) đang chờ thỏa thuận. Bảng đầy đủ ở `tables_ft_decider_v0.md`, số máy đọc được ở `metrics_ft_decider_v0.json`.

## Trả lời ba câu hỏi

Model mới là decider-2b + LoRA, checkpoint cuối epoch 2. So sánh chính là với decider-2b gốc, theo cặp trên cùng câu; accuracy trung bình đều theo nguồn.

1. **Có giữ được EN và VI có dấu không? Có, nhưng VI sát mốc.**
   - EN: −0,7 [−2,2; +0,9], là nhiễu.
   - VI có dấu, khối song song: −1,8 [−3,7; +0,2]. Mức này nằm trong giới hạn "giảm không quá 2" nhưng rất sát. Riêng MASSIVE giảm −2,7 (p = 0,008).
2. **Khối bản địa có tăng không? Có, ở bản có dấu:** +4,8 [+1,9; +7,6].
   - Theo nguồn: phần lớn đến từ ViHSD (+11,7) và ViWikiFC (+4,0). ViFactCheck giảm −1,2.
   - Theo kiểu câu: noul +10,0, choice 0,0, score −1,4.
3. **Lát không dấu có hết gãy không? Không.**
   - Song song: +1,4 [−2,1; +5,0], tức 50,6 so với 49,2. Vẫn thấp hơn bản có dấu 33,6 điểm, và thấp hơn adapter bỏ dấu của Qwen3-1.7B 10,2 điểm.
   - Bản địa không dấu tăng +7,9 [+3,9; +11,7].

**Cách đọc định trước: ĐÁNG THEO.** EN và VI có dấu giảm không quá 2 điểm, và bản địa có dấu tăng ≥ 3 với CI không chứa 0. Cần đọc kết luận này cùng các lưu ý ở mục 4–5: lợi ích tập trung ở ViHSD, hiệu chỉnh xấu đi khi giữ T phát hành, và lát không dấu chưa được sửa.

---

## 1. Kiểm khả thi (bước 1)

### 1.1 Giấy phép và revision

- `Mapika/decider-2b` @ `ff99089b6810691bd79134ec6febb5d9afa6e619` (trùng `main` lúc kiểm). Thẻ YAML của card: `license: apache-2.0`, `base_model: Qwen/Qwen3.5-2B-Base`.
- Repo không có file `LICENSE` hay `NOTICE`, và card không có câu ghi công riêng. File của repo: `.gitattributes`, `README.md`, `chat_template.jinja`, `config.json`, `decider_config.json`, `eval_results.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `decider/*.py`.
- Khi phân phối model dẫn xuất, Apache-2.0 mục 4 yêu cầu:
  - kèm một bản giấy phép;
  - ghi rõ các file đã sửa;
  - giữ nguyên các thông báo bản quyền và ghi công của bản gốc;
  - nếu bản gốc có file NOTICE thì phải kèm nội dung đó. Repo này không có NOTICE.
- Adapter này chưa phân phối, chỉ dùng nội bộ.
- Mã gói `decider/*.py` được kiểm sha256 từng file với bản đã duyệt ở bước baseline, trước mỗi lần nạp.

### 1.2 Đường chấm của decider (`scripts/bench/eval_decider_v1.py`, ánh xạ "text")

- **Prompt:** dạng "plain", không chat template, không BOS:
  `Context:\n<state>\n\nQuestion: <câu hỏi>\nOptions:\n(A) …\n(B) …\nAnswer: (`
- **Đọc xác suất:** lấy logit tại token "(" cuối, giới hạn trên các token nhãn của gói (`DecisionModel.letters`: "A"=32, "B"=33, …; mỗi nhãn đúng một token). Chia cho T phát hành theo kiểu câu, rồi softmax trên k nhãn.
- **choice:** `Decider.decide`, lựa chọn giữ nguyên thứ tự, T = 1,164.
- **noul:** đường `system_one`. Một hàng với lựa chọn (A) no, (B) yes; T = 1,624. Xác suất ghi theo thứ tự [Có, Không] của dev_v1.
- **score:** isolated levels.
  - Mỗi mức một hàng `"<câu hỏi>\nProposed answer: <mức>\nDoes the proposed answer fit?"` với (A) no, (B) yes; T = 1,124.
  - P(yes) của các mức được chuẩn hóa: q_i = r_i / Σ r_j.
  - Một dòng score tốn n forward; train_v0 trung bình 4,9 mức, tối đa 10.
  - Đường này không phụ thuộc thứ tự mức: đảo thứ tự thì phân phối đảo theo, chênh 0,0. Vì vậy lỗi xáo thứ tự lựa chọn câu score đã biết không có tác dụng ở đây.

### 1.3 Ánh xạ train_v0 sang định dạng decider (`scripts/ft_decider/dcommon.py`)

- **choice:** giữ nguyên lựa chọn và target.
- **noul:** lựa chọn "Yes"/"Có" (kể cả "Co" khi đã bỏ dấu) ứng với "yes" của decider; target [1 − p, p] theo thứ tự [no, yes].
  - Đã kiểm: 0 dòng không nhận diện được trên toàn train + val, ở mọi chế độ bỏ dấu.
  - Chiều không bị đảo: decider-2b đúng 74,1% trên 174 dòng noul của val.
- **score:** giữ nguyên chữ của mức.
- Ví dụ đã ánh xạ (mỗi kiểu 5, mỗi trường cắt 200 ký tự) ở `gates/g2.json`.
- Ở định dạng decider, lựa chọn noul luôn là no/yes của gói, nên việc bỏ dấu lựa chọn noul không thay đổi gì.

### 1.4 Loss: phương án A, người dùng chọn

```
loss = CE(t, softmax(z_nhãn / T phát hành)) − log mass,   mass = Σ_nhãn softmax(z)  (toàn từ vựng, T = 1)
score: CE(t, q_isolated) + trung bình_i(−log mass_i của hàng mức i)
```

- Phần CE dùng đúng phân phối mà đường chấm tính ra: cùng hàm dựng prompt của gói (`_decide_items`, `_system_one_items`), cùng phép đọc (`TT.scaled_softmax` trên logit nhãn).
- Hạng mass chặn lỗi letters_k.
- Khi T = 1, phương án A trùng đúng loss full_vocab của Qwen3.
- **Lý do không dùng CE toàn từ vựng trên z/T (nghĩa đen của brief):**
  - Cách đó đúng bằng CE − log mass_T. T phát hành được fit cho softmax trên 2–10 nhãn; chia toàn bộ 248.320 logit cho T > 1 làm phần đuôi từ vựng phẳng ra.
  - Ở bước 0, mass_T của noul chỉ 0,026 (−log ≈ 3,79 nat, gấp khoảng 5 lần phần CE). Ở T = 1, mass là 0,61.
  - Chẩn đoán đầy đủ ở `gates/diag_loss.json` (200 dòng val, trung bình theo dòng):

| Kiểu | CE (T phát hành) | −log mass_T | −log mass (T=1) | mass (T=1) |
|---|---|---|---|---|
| choice | 0,929 | 1,201 | 0,489 | 0,629 |
| noul | 0,696 | 3,786 | 0,532 | 0,612 |
| score | 1,168 | 1,299 | 0,707 | 0,504 |

- Khối lượng thấp là đặc tính của decider, không do lệch tập token. Ở vị trí đọc, phần xác suất ngoài nhãn nằm ở các chữ cái thừa (C, D ở câu 2 lựa chọn), " A" có khoảng trắng, chữ thường và chữ số. Reader không đọc token nào trong số đó.

### 1.5 Các cổng

- **G1 (đạt):** chấm lại decider-2b trên dev_v1, preds trùng từng byte với bản đã lưu, cả "text" lẫn "keyed".
- **G2 (đạt ở batch 1):** dự đoán từ code train ở bước 0 trùng đường chấm trên 200 dòng đầu của val_v0: choice 156/156, noul 22/22, score 22/22. Chênh xác suất lớn nhất 0,0 (choice) và 4e-8 (noul/score).
  - Lần chạy đầu lệch 197/200, có hai nguồn: code train chạy dưới autocast bf16, làm hạ phép fp32 trong linear attention; và logit nhãn lấy từ phép nhân với toàn bộ lm_head thay vì F.linear trên riêng các hàng nhãn.
  - Đã sửa: bỏ autocast; logit nhãn tính đúng như `DecisionModel.slot_logits`.
  - Ở batch 8 vẫn có 2/156 câu choice đảo dự đoán (chênh xác suất lớn nhất 0,027), do kernel bf16 phụ thuộc hình dạng batch.
- **G3 (đạt, có điều kiện):** batch 32 = 8 × 4 mà chạy forward cả micro-batch một lần thì **hết bộ nhớ** ở micro-batch nặng nhất: 8 dòng có 3 dòng score, thành 23 chuỗi, 60.352 token đã pad.
  - Phiên điều phối chấp nhận cách sau: giữ batch 32 = 8 × 4, nhưng khi chạy forward/backward thì chia micro-batch thành các nhóm dòng không quá 16.384 token đã pad.
  - Mỗi nhóm gọi backward với Σ loss từng dòng / 32, nên gradient của bước đúng bằng Σ loss 32 dòng / 32.
  - Đo được: micro-batch nặng nhất đỉnh 7,95 GB; 12 bước thật đỉnh 10,6 GB; ước lượng 4,7 giờ.
  - Lần G3 đầu hỏng vì `StopIteration` ở bước ánh xạ (chưa nhận diện "Co"/"Khong"), không phải lỗi vòng lặp dữ liệu. Đã sửa trước lần đo trên.

## 2. Train (bước 2)

### 2.1 Cấu hình và thời gian

- **Cấu hình:** decider-2b @ `ff99089b`, seed 1, train_v0 (`d10c9362…`), val_v0 (`4377e710…`). 2 epoch, 1.250 bước, batch 32 = 8 × 4, lr 2e-4, warmup 37 bước rồi cosine về 0, clip 1,0.
- **Độ chính xác số:** trọng số bf16; tham số LoRA fp32 (mặc định của peft); không autocast; gradient checkpointing. Linear attention chạy bằng bản fallback PyTorch, vì không có flash-linear-attention và không cài thêm gì vào `.venv`.
- **LoRA:** r16 / α32 / dropout 0,05 trên 150 module, 15,63 triệu tham số:
  - attention đầy đủ (6 lớp): `q_proj`, `k_proj`, `v_proj`, `o_proj`;
  - linear attention (18 lớp): `in_proj_qkv`, `in_proj_z`, `out_proj`;
  - MLP (24 lớp): `gate_proj`, `up_proj`, `down_proj`.
  - Không gắn vào `in_proj_a`/`in_proj_b` (gate 16 chiều) và `lm_head`.
  - Danh sách đầy đủ ở `train/decider-2b-s1/lora_modules.json`.
- **Bỏ dấu:** y như strip_v0 (p = 0,3, nửa chỉ bỏ ở state, hash(seed, epoch, id), khóa sắp là độ dài trước khi bỏ dấu). Số dòng bị bỏ dấu trùng hệt strip_v0: epoch 0 có 605 dòng chỉ bỏ ở state và 601 dòng bỏ toàn bộ; epoch 1 có 644 và 571, trên 4.002 dòng VI.
- **Chia nhóm theo ngân sách token:** 358/5.000 micro-batch bị chia (tức 358/1.250 bước), tối đa 3 nhóm.
  - Mặt nạ dropout của LoRA dùng RNG toàn cục, nên có phụ thuộc cách chia nhóm.
  - Cách chia cố định theo dữ liệu, nên chạy lại vẫn ra cùng kết quả.
- **Thời gian:** 4 giờ 35 phút. Đỉnh VRAM 11,1 GB (đã cấp phát), 11,3 GB (đã giữ chỗ).

### 2.2 Val_v0 và khối lượng nhãn (T = 1, monitor 200 dòng)

| Thời điểm | Loss A | CE trên phân phối chấm | Khối lượng | Accuracy argmax | Val bỏ dấu VI: loss / CE / accuracy |
|---|---|---|---|---|---|
| bước 0 | 1,376 | 0,877 | 0,625 | 63,7% | 1,527 / 0,955 / 56,3% |
| cuối epoch 1 | 0,670 | 0,669 | 0,999 | 69,6% | 0,724 / 0,723 / 67,4% |
| cuối epoch 2 | 0,745 | 0,745 | 1,000 | 74,5% | 0,803 / 0,802 / 71,7% |

- **Monitor khối lượng:** 0,614 ở bước 0; 0,993 ở bước 125; **0,995 ở bước 250** (cổng: ≥ 0,5, đạt); từ đó tới bước 1.250 luôn ≥ 0,997, không dòng nào dưới 0,5.
- Loss val tăng nhẹ ở epoch 2 trong khi accuracy vẫn tăng: xác suất tự tin quá mức. Mẫu này giống các lần Qwen3 trước.

## 3. Số đo trên dev_v1 (bước 3)

Mọi preds lấy từ đường chấm của decider (batch 1, T phát hành), có thêm cờ `--adapter` để gộp LoRA. Không có cờ thì đường chấm chạy y như cũ; G1 đã xác nhận điều này. decider-2b và adapter bỏ dấu của Qwen3 dùng preds đã lưu.

### 3.1 Năm lát chính (accuracy %, trung bình đều theo nguồn)

| Lát | mới (epoch 2) | decider-2b | strip (Qwen3-1.7B) | mới epoch 1 |
|---|---|---|---|---|
| song song · EN | 88,5 | 89,2 | 86,8 | 88,6 |
| song song · VI có dấu | 84,2 | 86,0 | 80,8 | 86,2 |
| song song · VI không dấu | 50,6 | 49,2 | 60,8 | 44,4 |
| bản địa · có dấu | 70,9 | 66,1 | 66,3 | 68,7 |
| bản địa · không dấu | 50,8 | 42,9 | 56,9 | 49,7 |

### 3.2 Chênh lệch theo cặp (điểm, [CI 95% bootstrap theo state], McNemar p)

| Lát | mới − decider-2b | mới − strip (Qwen3) | mới − epoch 1 |
|---|---|---|---|
| song song · EN | −0,7 [−2,2; +0,9] (0,35) | +1,7 [−0,4; +3,8] (0,049) | −0,1 |
| song song · VI có dấu | −1,8 [−3,7; +0,2] (0,064) | +3,4 [+0,7; +6,3] (0,007) | −2,0 [−3,9; −0,2] |
| song song · VI không dấu | +1,4 [−2,1; +5,0] (0,89) | −10,2 [−14,5; −5,8] (6e-11) | +6,2 [+2,7; +9,8] |
| bản địa · có dấu | **+4,8 [+1,9; +7,6] (0,0002)** | +4,6 [+1,1; +8,0] (0,034) | +2,3 [−0,2; +4,6] |
| bản địa · không dấu | +7,9 [+3,9; +11,7] (0,0003) | −6,1 [−10,3; −1,7] (0,001) | +1,2 [−2,3; +4,7] |

### 3.3 Theo nguồn (mới − decider-2b)

- **Song song, có dấu:** MASSIVE −2,7 (p = 0,008); Belebele −1,0; SIB −1,3; XCOPA −2,0. EN: tất cả nằm trong khoảng −1,7 đến 0,0.
- **Bản địa, có dấu:** **ViHSD +11,7** (p = 0,0004; 40,7 lên 52,3), ViWikiFC +4,0 (p = 0,036), ViFactCheck −1,2.
- **Không dấu, song song:** Belebele −4,0, MASSIVE +2,3, SIB +0,3, XCOPA +7,0. Không nguồn nào có ý nghĩa thống kê.
- **Không dấu, bản địa:** ViFactCheck +10,4, ViWikiFC +8,7, ViHSD +4,7.

### 3.4 Theo kiểu câu hỏi (khối bản địa)

| Kiểu | Có dấu: mới − decider | Không dấu: mới − decider |
|---|---|---|
| choice | 0,0 [−4,7; +4,8] | +9,4 [+1,8; +17,0] |
| noul | **+10,0 [+5,7; +14,3]** | +8,3 [+2,9; +13,8] |
| score | −1,4 [−9,0; +6,2] | +3,5 [−5,6; +12,5] |

### 3.5 Hiệu chỉnh (Brier / ECE, trên xác suất đường chấm với T phát hành)

| Lát | mới (epoch 2) | decider-2b | mới epoch 1 |
|---|---|---|---|
| song song · EN | 0,185 / **0,068** | 0,164 / 0,022 | 0,175 / 0,025 |
| song song · VI có dấu | 0,221 / **0,075** | 0,211 / 0,023 | 0,226 / 0,042 |
| song song · VI không dấu | 0,655 / 0,132 | 0,626 / 0,044 | 0,656 / 0,045 |
| bản địa · có dấu | 0,404 / 0,145 | 0,436 / 0,126 | 0,427 / 0,040 |
| bản địa · không dấu | 0,572 / 0,127 | 0,611 / 0,135 | 0,552 / 0,059 |

### 3.6 Trên bản khôi phục dấu bằng vit5 (câu tiếng Việt)

| Khối | mới + vit5 | decider + vit5 | (mới + vit5) − (decider + vit5) | mới + vit5 − mới có dấu |
|---|---|---|---|---|
| song song | 78,8 | 81,9 | −3,1 [−5,7; −0,6] | −5,4 [−8,0; −2,9] |
| bản địa | 67,1 | 64,5 | +2,5 [−0,5; +5,4] | −3,9 [−6,3; −1,5] |

Ở dòng song song, CI không chứa 0 nhưng McNemar cho p = 0,078. Lý do như đã giải thích ở REPORT_restore.md và REPORT_strip_v0.md: bootstrap lấy mẫu theo state và gộp đều theo nguồn, còn McNemar tính trên từng dòng.

## 4. Diễn giải

Mọi điểm trong mục này là cách đọc số, không phải số đo.

1. **Theo quy tắc là "đáng theo", nhưng lợi ích hẹp.**
   - Bản địa có dấu tăng 4,8 điểm, chủ yếu ở ViHSD (+11,7) và kiểu noul (+10,0). Bỏ ViHSD thì trung bình hai nguồn còn lại chỉ +1,4.
   - ViHSD cũng là nguồn đã kéo mức tăng của strip_v0. decider-2b vốn yếu ở ViHSD (40,7, thấp hơn cả Qwen3 + strip là 53,0). Dữ liệu train của dự án sửa được điểm yếu đó, nhưng chưa rõ nó tổng quát tới đâu.
2. **Lát không dấu không được sửa.**
   - Trước fine-tune, hai model nền ngang nhau ở lát không dấu song song: Qwen3-1.7B gốc 48,0, decider-2b 49,2.
   - Fine-tune trên train_v0 nâng Qwen3 khoảng 11 điểm (adapter đối chứng, chưa bỏ dấu: 58,8). Bỏ dấu chỉ thêm +2,0 (60,8). Phần tăng đó đến từ chính việc fine-tune, không phải từ model nền hay từ việc bỏ dấu.
   - Cùng train_v0, có bỏ dấu, gần như không nâng decider-2b: +1,4 (49,2 lên 50,6). Ở epoch 1, lát này còn tụt xuống 44,4 rồi mới hồi lên ở epoch 2.
   - **Nguyên nhân chưa giải thích được.** Giả thuyết, đều chưa kiểm:
     - khung prompt tiếng Việt của dự án (câu hỏi và khung bằng tiếng Việt) so với khung tiếng Anh cố định của decider ("Context / Question / Options / Answer");
     - định dạng chữ cái của dự án so với định dạng gốc của decider (đọc tại "(", noul/score qua system_one);
     - hạng mass trong loss;
     - khác biệt tokenizer (Qwen3 so với Qwen3.5) khi gặp văn bản không dấu;
     - decider đã học sẵn rất nhiều dữ liệu có dấu, nên LoRA r16 trong 2 epoch không đủ đổi cách nó đọc văn bản không dấu.
3. **Epoch 1 và epoch 2 đánh đổi nhau.** Chỉ để xem, không dùng để chọn checkpoint.
   - Epoch 1 giữ nguyên VI có dấu (86,2 so với 86,0) và EN, còn hiệu chỉnh gần như decider gốc (ECE 0,025–0,042).
   - Epoch 2 đổi 2,0 điểm VI có dấu lấy +6,2 không dấu và +2,3 bản địa, đồng thời ECE tăng gấp khoảng 3 lần.
4. **Hiệu chỉnh xấu đi ở epoch 2 khi giữ T phát hành:** ECE song song từ 0,02 lên 0,07. Model tự tin hơn, khớp với loss val tăng ở epoch 2. T phát hành được fit cho decider gốc, không còn hợp với model mới. Theo brief, tôi không fit lại.
5. **So theo pipeline với bộ khôi phục vit5:** decider gốc + vit5 vẫn hơn model mới + vit5 ở khối song song (−3,1). Ở khối bản địa, model mới nhỉnh hơn (+2,5; CI chứa 0).
6. **Lưu ý về nguyên nhân** (đã nêu trước khi train):
   - Hạng mass trong loss đã đẩy khối lượng nhãn từ 0,61 lên 0,99 chỉ trong 125 bước, tức là đổi mạnh phân phối ở vị trí đọc.
   - Mức giảm nhỏ ở VI có dấu song song (−1,8; MASSIVE −2,7) và ECE xấu đi có thể một phần do hạng này, không chỉ do dữ liệu và công thức của dự án.
   - Một lần train không tách được hai nguyên nhân.

## 5. Điều chưa chắc

- **Vì sao fine-tune nâng lát không dấu của Qwen3 (+10,8 không cần bỏ dấu) mà gần như không nâng decider-2b (+1,4)** vẫn chưa rõ. Hai model nền xuất phát ngang nhau (48,0 và 49,2). Các giả thuyết ở mục 4.2 (khung prompt, định dạng đọc, hạng mass, tokenizer) chưa được kiểm, và một lần train không tách được chúng.
- **Chỉ có một seed.** Với Qwen3-1.7B, chênh giữa hai seed trên tiếng Việt là khoảng ±1 điểm. Mức −1,8 ở VI có dấu song song sát mốc 2 nên có thể đổi chiều với seed khác. Mức +4,8 bản địa có CI không chứa 0, nhưng chủ yếu dựa vào một nguồn.
- **G2 chỉ trùng hoàn toàn ở batch 1.** Train chạy micro-batch nhiều chuỗi nên có nhiễu số học bf16 theo hình dạng batch. Ở bước 0, nhiễu này làm 2/156 câu choice đảo dự đoán.
- **Linear attention chạy bằng bản fallback PyTorch**, không phải kernel mà tác giả dùng khi train (flash-linear-attention). Kết quả số có thể khác chút so với môi trường của tác giả.
- **Brier/ECE tính với T phát hành.** Nếu fit lại T thì hiệu chỉnh của model mới nhiều khả năng sẽ cải thiện; chưa làm.
- **Mặt nạ dropout phụ thuộc cách chia nhóm theo token** (358 bước có chia). Gradient về toán học không đổi.
- **ViHSD trên dev_v1** có nhãn ồn, và câu score của nó không phân biệt được các model (khoảng 28–44%).

## 6. Đề xuất (chưa chạy, chờ quyết định)

1. **Fit lại T trên val_v0 cho model mới**, chỉ inference, khoảng 10 phút, để tách phần "hiệu chỉnh xấu đi" khỏi phần "model đổi". Brief yêu cầu giữ T phát hành nên tôi chưa làm.
2. **Seed 2 với cùng công thức** (khoảng 4,7 giờ), để biết −1,8 ở VI có dấu và +4,8 bản địa có lặp lại không.
3. **Tách tác động của hạng mass:** một lần train dùng phương án N (chỉ CE trên phân phối chấm, theo dõi khối lượng). Cách này cần người dùng cho phép vì brief cấm.
4. Không chạy thêm lr, p hay seed nào.

## 7. File

- `train/decider-2b-s1/`:
  - `adapter/` (epoch 2, chính thức), `adapter_epoch1/`;
  - `train_log.jsonl`, `summary.json`, `lora_modules.json`, `stdout.txt`.
- `eval/dev_v1_ep2/`, `eval/dev_v1_ep1/`, `eval/restored_vit5/`: preds từ đường chấm của decider (`decider-2b-text` và `decider-2b-keyed`).
- `gates/`:
  - G1: `g1/`;
  - G2: `g2.json`, gồm ví dụ ánh xạ và top-10 token ở vị trí đọc;
  - G3: `g3_tb16384/g3.json`, `plan_stats.json`;
  - `diag_loss.json`, `modules.json`;
  - `gates.log`.
- `tables_ft_decider_v0.md`, `metrics_ft_decider_v0.json`, `pipeline.log`.
- Code:
  - `scripts/ft_decider/{dcommon,train_decider,g2_check,diag_loss,probe_modules,report_ft_decider}.py`;
  - `scripts/bench/eval_decider_v1.py`: thêm cờ `--adapter`; không có cờ thì chạy như cũ;
  - `setup/ftd_*.sh`.
