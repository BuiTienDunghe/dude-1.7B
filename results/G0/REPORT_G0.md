# G0: đo, không train (28/10/2026)

Theo `results/BRIEF_buoc0_G0.md`. Cách đọc kết quả đã chốt **trước** khi chấm, trong `results/G0/CACH_DOC_chot_truoc.md`:
- điểm là trung bình đều 6 nguồn (macro) trên dòng VI có dấu;
- CI là bootstrap cụm theo cặp, rút lại riêng từng nguồn.

Không train, không đụng test, không commit, không xóa file.

## Kết luận, đọc theo đúng ngưỡng đã chốt

1. **Nền 2B: giữ Qwen3-1.7B.**
   - Qwen3.5-2B-Base − Qwen3-1.7B gốc = **−2,45** [−5,41; +0,42]; ngưỡng là ≥ +3 với CI không chứa 0, nên **không đạt**.
   - Điều kiện độ trễ cũng không đạt:
     - đường graphs của mọi model chưa qua cổng đúng (mục 4), nên không có số p90 graphs hợp lệ để so;
     - số tham khảo eager không FLA của 2B-Base có p90 70,3 ms, so với 47,3 ms của C.
   - Số của 2B-Base là **cận trên**, vì không có FLA (mục 5).
2. **Điểm của Quyet: trộn lẫn, nghiêng về cỡ model.**
   - Qwen3.5-4B gốc @851bf6e806ef, khung dự án: **84,07** VI có dấu. Mức này nằm giữa 75 và 85, nên đọc là "trộn lẫn", nhưng chỉ dưới ngưỡng "chủ yếu do cỡ" 0,9 điểm.
   - Quyet-Medium đạt 89,57, tức fine-tune thêm khoảng +5,5 trên nền này.
3. **Khung prompt: giữ khung VI.** Khung EN hơn khung VI ≥ 2 điểm với CI không chứa 0 ở **0/3** model:
   - Qwen3-1.7B: +1,45 [−0,57; +3,43];
   - Qwen3.5-2B: +1,14 [−0,53; +2,79];
   - Qwen3.5-2B-Base: **−3,10** [−5,38; −0,96], khung EN còn tệ hơn.
4. **Độ trễ:** chỉ báo số, không kết luận về ngưỡng 40 ms.

   | Model (eager, batch 1) | Trung vị | p90 | p95 |
   |---|---|---|---|
   | C-s1 (1.7B), toàn dev | 31,8 ms | 47,3 ms | 85,4 ms |
   | Qwen3.5-2B-Base, không FLA, toàn dev | 53,2 ms | 70,3 ms | 117,0 ms |
   | Qwen3.5-4B, không FLA, tập con 1.000 dòng | 82,0 ms | 138,0 ms | 263,8 ms |

   Đường CUDA graphs **chưa qua cổng đúng** ở cả 4 lượt. Mọi số graphs dưới đây là THAM KHẢO.

## 1. G0b: chấm dev 1.3

Khung dự án (`render_prompt`), batch 1, toàn dev 3.847 dòng; macro 6 nguồn.

| Model | VI có dấu | VI không dấu | EN | Nguồn preds |
|---|---|---|---|---|
| Qwen3-1.7B gốc (M0) | 70,65 | 48,45 | 84,59 | `results/probe_v1/eval/dev/Qwen3-1.7B` |
| Qwen3.5-2B gốc | 70,63 | 38,99 | 84,50 | `results/bench_v1_3_trial/eval/Qwen3.5-2B` |
| **Qwen3.5-2B-Base** (b1485b2f…) | **68,20** | 37,78 | 83,91 | `results/G0/eval/dev/Qwen3.5-2B-Base` |
| **Qwen3.5-4B** (851bf6e8…) | **84,07** | 60,57 | 90,48 | `results/G0/eval/dev/Qwen3.5-4B` |
| M1 (dự án) | 79,88 | 62,37 | 87,95 | |
| C (train thử) | 81,65 | 63,97 | 88,20 | |
| decider-2b | 86,50 | 49,64 | 90,33 | |
| Quyet-Medium | 89,57 | 64,60 | 94,46 | |

**Chênh theo cặp, VI có dấu:**

| So sánh | Chênh [CI 95%] |
|---|---|
| 2B-Base − M0 | −2,45 [−5,41; +0,42] |
| 2B-Base − Qwen3.5-2B gốc | −2,43 [−4,52; −0,27] |
| 2B-Base − C | −13,45 [−16,20; −10,67] |

Theo nguồn: `results/G0/g0_tong_hop.json`.

**Ghi chú:**
- **Chat template:** tokenizer bản 2B-Base không có chat template, nên mượn template của `Qwen/Qwen3.5-2B` cùng họ, cùng vocab (`--chat_template_from`, có ghi trong `timing.json`). Đã kiểm chữ A–J là 1 token.
- **4B:** lượt đo thông lượng batch 16 phụ của `eval_base.py` bị kẹt khoảng 25 phút vì VRAM đầy 16 GB, có lẽ tràn sang RAM qua WDDM. Điều phối đồng ý dừng PID 4025 lúc 01:57, sau khi preds batch 1 đã đủ 3.847 dòng. Vì vậy 4B không có `timing.json`; độ trễ đo riêng ở mục 4.

## 2. G0c: khung prompt EN cho dòng VI có dấu (1.430 dòng)

| Model | Khung VI | Khung EN | EN − VI [CI] | Môi trường |
|---|---|---|---|---|
| Qwen3-1.7B gốc | 70,65 | 72,09 | +1,45 [−0,57; +3,43] | `.venv` |
| Qwen3.5-2B gốc | 70,63 | 71,77 | +1,14 [−0,53; +2,79] | `.venv`, chấm lại cả hai khung |
| Qwen3.5-2B-Base | 68,20 | 65,10 | −3,10 [−5,38; −0,96] | `.venv` |

Theo nguồn: khung EN tăng mạnh ở MASSIVE với hai model gốc (khoảng +8), nhưng giảm ở XCOPA và ViFactCheck (bảng theo nguồn trong JSON). Khung EN chỉ đổi chỉ dẫn và nhãn khung; nội dung, câu hỏi và lựa chọn giữ nguyên tiếng Việt. Không có chữ mới.

## 3. Môi trường và FLA

- **`.venv-g0`:** sao từ freeze của `.venv`, thêm `flash-linear-attention==0.6.0` (kéo theo `fla-core==0.6.0`, `einops==0.8.2`), các gói khác giữ nguyên phiên bản. Freeze lưu ở `results/G0/env/`.
- **FLA KHÔNG chạy được:** triton báo `RuntimeError: Failed to find C compiler` khi chạy `chunk_gated_delta_rule` (`results/G0/env/fla_import.txt`). Lúc đó WSL chưa có gcc, và sudo cần mật khẩu.
- Theo brief: ghi lỗi, **không vá, không lách**. Mọi lượt Qwen3.5 chạy trong `.venv` (đường torch, không FLA). `.venv-g0` không dùng: `fla` import được nên transformers sẽ chọn nó rồi lỗi lúc chạy.
- **Mọi số độ trễ của Qwen3.5 là CẬN TRÊN.**
- Sau đó người dùng đã tự cài build-essential (gcc 15.2). Phần đo lại có FLA là việc (A) của brief chẩn đoán, làm sau.

## 4. G0a: độ trễ (`scripts/latency_g0.py`)

**Cách đo:**
- Eager: đúng như `eval_base.py`, tức một forward với `logits_to_keep=1`, batch 1.
- CUDA graphs: như engine của decider:
  - pad phải tới 13 nhóm độ dài (64…2.048), không mask, `position_ids` mặc định;
  - logit lấy tại token thật cuối;
  - mỗi nhóm bắt một graph, dùng chung pool.
- Mỗi cấu hình đo 2 lần (4B một lần, theo brief); số trong bảng là trung bình các lần.
- GPU: `nvidia-smi` trước và sau mỗi lượt (`results/G0/nvsmi/`). Có lính gác tiến trình Windows và khung 03:00–03:05: không lượt nào bị gắn cờ.

**Số đo:**

| Cấu hình | Tập | Eager trung vị | Eager p90 | Eager p95 | Eager p99 | Graphs (THAM KHẢO): trung vị / p90 / p95 |
|---|---|---|---|---|---|---|
| C-s1 (1.7B + LoRA đã gộp) | toàn dev | 31,80 | 47,27 | 85,44 | 135,81 | 24,94 / 49,14 / 102,1 |
| C-s1 | tập con 1.000 | 30,06 | 45,80 | 84,94 | 134,38 | 24,88 / 49,15 / 102,1 |
| Qwen3.5-2B-Base, không FLA | toàn dev | 53,16 | 70,31 | 117,04 | 191,95 | 31,90 / 50,54 / 125,7 |
| Qwen3.5-4B, không FLA | tập con 1.000 | 82,03 | 138,01 | 263,75 | 436,11 | 67,86 / 126,49 / 313,5 |

p90 theo nhóm độ dài prompt (≤128, 129–256, 257–512, 513–1.024, >1.024 token của chính model) có trong `results/G0/g0_tong_hop.json` (`latency.*.p90_theo_nhom_lan1`). Ví dụ C-s1 graphs (tham khảo), toàn dev: 16,5 / 25,6 / 49,1 / 102,1 / 236,0 ms.

**Tập con** (điều phối 02:27, để giữ ngân sách GPU):
- `results/G0/rows/dev_lat_1000.jsonl`: 1.000 dòng dev, phân tầng theo nhóm độ dài `n_tokens`, seed 20261012.
- Số dòng theo nhóm: 247 / 517 / 155 / 45 / 36.
- **Lệch so với toàn dev (C-s1 eager):** trung vị −5,5%, p90 −3,1%, p95 −0,6%, p99 −1,1%.

**Cổng đúng: KHÔNG đạt ở cả 4 lượt.** Vì vậy mọi số graphs chỉ là THAM KHẢO, không dùng cho kết luận.
- Lý do vẫn ghi lại số graphs: một bản sửa đúng mà giữ được hình dạng tensor theo nhóm sẽ tốn cùng lượng tính toán, nên số này cho biết cỡ độ trễ. Đây chưa phải bằng chứng.
- **C-s1 toàn dev:** argmax trùng 99,17% (32 câu lệch), xác suất lệch tối đa 0,194, 1.897 câu lệch > 1e-3.
- **Tách nguyên nhân trên tập con C-s1**, theo đề xuất của điều phối, có thêm đường "eager có pad" (cùng pad phải, không bắt graph):

  | Phép so | Câu lệch argmax | Lệch xác suất tối đa | Câu lệch > 1e-3 |
  |---|---|---|---|
  | (i) eager không pad với eager có pad (sàn nhiễu theo hình dạng) | 3 | 0,069 | 84 |
  | (ii) eager có pad với graphs (kiểm việc bắt graph) | **5** | **0,123** | **481** |

  (ii) cũng lệch, nên **đường graphs có lỗi riêng**, ngoài nhiễu do pad.
- Câu lệch **không** tập trung ở câu bị pad nhiều (số token pad trung bình: câu lệch 30, toàn tập con 43). Câu lệch rơi vào các nhóm T = 128–256.
- Nghi vấn: kernel cuBLAS / sdpa lúc bắt graph khác lúc chạy thường. **Chưa sửa**; là việc để sau.

**Thay đổi điều kiện "GPU rảnh" lúc 02:05** (điều phối chọn):
- Sau khi dừng PID 4025, chỉ số tải GPU kẹt ở 100%, trong khi VRAM 0 MiB, công suất 27 W, không có tiến trình nào.
- Điều kiện mới: không tiến trình compute (Windows và WSL) + VRAM < 500 MiB + công suất < 40 W liên tục 10 giây.
- **Chốt kiểm:** C-s1 eager đo trước, trung vị 31,80 ms, trong 15% so với 32,6 ms đo ngày 26/10 → đạt.
- Chỉ số tải về bình thường sau lượt đo (98% khi đo, 0% sau). Xung SM và công suất ghi trong `results/G0/logs/*_xung_*.txt`.

## 4b. Có FLA (BRIEF_chan_doan_teacher.md, phần A; chạy lúc 02:46–02:51)

1. **Cài đặt:** sau khi người dùng cài build-essential (gcc 15.2), FLA 0.6.0 trong `.venv-g0` **chạy được**. Kernel `chunk_gated_delta_rule` trên sm_120 trả kết quả hữu hạn (`results/G0/env/fla_check_sau_gcc.txt`). Không vá thư viện.
2. **Kiểm đúng: KHÔNG đạt.**
   - Cách kiểm: Qwen3.5-2B-Base có FLA (`.venv-g0`) so với không FLA (`.venv`, preds G0b), trên 600 dòng dev phân tầng 100 dòng mỗi nguồn (`results/G0/rows/dev_fla_600_ntok_null.jsonl`, seed 20261028).
   - Kết quả: argmax trùng **591/600 (98,5%)**, xác suất lệch tối đa **0,062**, 359 câu lệch > 1e-3.
   - Accuracy trên 600 dòng: có FLA 60,83%, không FLA 60,50%.
   - Preds: `results/G0/eval/fla/Qwen3.5-2B-Base_FLA/`.
3. **Độ trễ có FLA: KHÔNG đo, không dùng**, đúng theo brief ("không đạt thì báo, không dùng số đo"). Hàng đợi D đã dừng trước bước đo. Phần (A) dùng 243 giây GPU.

   Ghi chú của điều phối: kernel khác cho số khác ở mức nhiễu (argmax trùng 98,5%, accuracy 60,8 so với 60,5), nhưng không đạt cổng 100% nên không dùng. Điều phối quyết định bỏ hẳn phần đo có FLA, không chạy lại: nền 2B đã bị loại vì điểm (−2,45 so với Qwen3-1.7B), nên độ trễ có FLA không còn làm đổi quyết định nào.

**Hệ quả:**
- Số độ trễ Qwen3.5 ở mục 4 (không FLA) vẫn là **cận trên**.
- Đường FLA cho kết quả khác đường torch quá ngưỡng 1e-3. Có lẽ do kernel bf16 khác thứ tự cộng dồn, vì accuracy gần như không đổi. Chưa đủ để coi hai đường là một.

## 5. Nhiệt độ, thời gian GPU, mức dùng

**Nhiệt độ:** lính gác ghi mỗi 30 giây từ 01:07 (`results/G0/logs/temp.log`).

| Lượt | Nhiệt độ cao nhất |
|---|---|
| g0b 2B-Base | 73°C |
| g0c 2B-Base khung EN | 71°C |
| g0b 4B | 78°C |
| độ trễ C-s1 | 77°C |
| độ trễ 2B-Base | 78°C |
| C-s1 tập con | 77°C |
| 4B tập con | 77°C |

Các lượt trước 01:07 chưa có log nhiệt độ. Không lượt nào vượt 85°C.

**Thời gian GPU cộng dồn** (queue log): **6.495 giây, khoảng 108 phút**, dưới trần 150 phút.
- Gồm cả bước 0: chấm lại C và đo VRAM, khoảng 10 phút.
- Gồm cả khoảng 25 phút lượt batch 16 kẹt của 4B.

**GPU dùng chung:**
- Lúc 00:32, Ollama của dự án khác chiếm 8,7 GB và tải 87%. Hàng đợi chờ tới khi điều phối báo "GPU ĐÃ RẢNH" (00:40). Mọi lượt GPU chạy từ 00:42.
- Đã tránh khung 03:00–03:05.

## File

**Kết quả:**
- `results/G0/eval/dev/` (2B-Base, 4B), `results/G0/eval/g0c/`;
- `results/G0/latency/latency_*.json`;
- `results/G0/g0_tong_hop.json`;
- `results/G0/logs/` (queue, temp, từng lượt);
- `results/G0/nvsmi/`;
- `results/G0/env/`: freeze, lỗi FLA, `models_g0.json` (revision và sha256 trọng số).

**Script:**
- `scripts/latency_g0.py`, `scripts/g0_analyze.py`;
- `scripts/eval_base.py`: thêm `--revision`, `--frame`, `--chat_template_from`, mặc định giữ nguyên;
- `setup/g0_prepare.sh`, `g0_queue.sh`, `g0_queue_b.sh`, `g0_queue_c.sh`, `g0_switch_to_c.sh`, `g0_temp_guard.sh`.

**Revision:**
- Qwen/Qwen3.5-2B-Base `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`;
- Qwen/Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
