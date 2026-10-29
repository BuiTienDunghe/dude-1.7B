# Báo cáo dữ liệu 2b′: phần CPU (bước A, B; bước C chờ "DỰNG 2b′")

- **Brief:** `results/BRIEF_2b_prime_du_lieu.md`. Người dùng duyệt 29/10, chiều.
- **Phạm vi:** chỉ CPU. Không chạy model, không GPU, không train.
  - Không sửa `config/`, `data/bench_v1_3/`, `train_lora.py`, `score_probe.py`.
  - Test chỉ được dò khử trùng, không in.
- **Script:**
  - bước A: `scripts/train_2b_prime/nguong_noul.py` (Windows python);
  - bước B: `scripts/train_2b_prime/kiem_ke_vianli.py` (WSL, `bash setup/run.sh -I …`).

## 1. Bước A — P1: ngưỡng noul (hậu xử lý, chỉ báo)

**Cách làm:**
- Quy tắc: đoán "Có" khi p(Có) / (p(Có) + p(Không)) ≥ t.
- Fit: proxy_v1, dòng noul có dấu (ViWikiFC 78 + ViFactCheck 72 = 150 dòng); t = 0,05…0,95, bước 0,01. Hòa thì lấy t gần 0,5 nhất. Fit riêng từng adapter.
- Báo: dev 1.3, không fit trên dev. Không dấu dùng cùng t* fit trên dòng có dấu.
- Cột "trước" khớp số đã báo (VI macro 81,65 / 84,84 / 83,92), tức là cùng đường argmax.

**Kết quả** (accuracy %, trước → sau):

| Mô hình | t* | proxy noul có dấu | dev noul ViFactCheck có dấu | dev noul ViWikiFC có dấu | dev noul ViFactCheck không dấu | dev noul ViWikiFC không dấu | VI macro có dấu | VI macro không dấu |
|---|---|---|---|---|---|---|---|---|
| C-s1 | 0,76 | 81,33 → 89,33 | 87,18 → 91,03 | 85,48 → 87,90 | 73,08 → 79,49 | 72,58 → 72,58 | 81,65 → 82,14 | 63,97 → 64,49 |
| 2b-s1 | 0,10 | 85,33 → 86,67 | 92,31 → 92,31 | 87,10 → 83,87 | 78,21 → 83,33 | 72,58 → 75,81 | 84,84 → 84,61 | 63,35 → 64,01 |
| 2b-s2 | 0,51 | 89,33 → 90,00 | 91,03 → 91,03 | 87,10 → 87,10 | 83,33 → 82,05 | 70,97 → 70,97 | 83,92 → 83,92 | 62,34 → 62,24 |

**Tỉ lệ đoán "Có" trên dev** (%, trước → sau; số trong ngoặc là gold):

| Mô hình | ViFactCheck có dấu | ViWikiFC có dấu | ViFactCheck không dấu | ViWikiFC không dấu |
|---|---|---|---|---|
| C-s1 | 48,7 → 42,3 (35,9) | 42,7 → 38,7 (37,9) | 57,7 → 43,6 (35,9) | 39,5 → 31,5 (37,9) |
| 2b-s1 | 30,8 → 35,9 (35,9) | 37,9 → 44,4 (37,9) | 16,7 → 24,4 (35,9) | 21,8 → 28,2 (37,9) |
| 2b-s2 | 32,1 → 32,1 (35,9) | 36,3 → 36,3 (37,9) | 19,2 → 17,9 (35,9) | 21,8 → 21,8 (37,9) |

**Ghi chú cho người đọc** (worker không kết luận):
- **C-s1:** đoán "Có" quá nhiều. t* = 0,76 kéo tỉ lệ về gần gold; VI macro có dấu +0,49.
- **2b-s1 và 2b-s2:** argmax đã gần gold khi có dấu, nên ngưỡng gần như không đổi gì.
  - Khi không dấu, 2b vẫn đoán "Có" quá ít (17–28% so với gold 36–38%).
  - Một ngưỡng fit trên dòng có dấu không sửa được chỗ này.
- **Tập fit nhỏ** (150 dòng), nên t* không ổn định:
  - 2b-s2 có 23 giá trị t cùng accuracy cao nhất, từ 0,27 đến 0,73;
  - C-s1 có 6 giá trị (0,76–0,85);
  - 2b-s1 có duy nhất t = 0,10.
- **Cách đọc theo brief mục 1.4:**
  - với C-s1 và 2b, đây chỉ là dòng chỉ báo (hậu kiểm);
  - với adapter 2b′ sau này, P1 là một dòng riêng, cùng luật, không gộp vào T4.
- **File:** `results/train_2b_prime/P1_nguong_noul.json` (kèm các t hòa và accuracy proxy theo nguồn), `P1_nguong_noul.md`.

## 2. Bước B — kiểm kê ViANLI

**Nguồn:** `results/train_v1/raw_src/LakoreAI__PhoLaya-base/sources/vianli/train.jsonl` (sha `e985e7a0…`).
- 8.012 cặp có câu hỏi `relation`: contradiction 2.473 / entailment 2.615 / neutral 2.924.
- Id cặp = `vianli:<số dòng>`.

**Loại** (theo thứ tự):

| Lý do | Số cặp |
|---|---|
| Đã có trong C / C′ (`components/C.jsonl` ∪ `C_prime.jsonl`; khớp chuỗi chuẩn hóa; 1.000 cặp, mọi cặp đều thấy trong nguồn) | 1.000 |
| Trùng cặp trong nguồn | 0 |
| Khử trùng (`kc_util.Refs`, hai chiều). Một câu chạm thì bỏ mọi cặp dùng tiền đề đó | 122 |
| **Còn lại** | **6.890** |

Khử trùng theo tham chiếu, đếm theo cặp; câu chạm có thể là tiền đề hoặc giả thuyết:

| Tham chiếu | Tiền đề chạm | Giả thuyết chạm |
|---|---|---|
| test 1.3 (chỉ đếm, không in) | 32 | 17 |
| Belebele | 20 | 6 |
| dev 1.3 | 18 | 6 |
| ViFactCheck | 12 | 9 |
| proxy_v1 | 2 | 0 |

**Số đo theo nhãn, cạnh các cột đối chiếu:**
- Tất cả là dòng có dấu.
- max_ov, cover, số lạ, tên lạ và probe dùng đúng hàm `feats` và probe của `phan_tich_bacbo.py`.
- Probe đã kiểm lại: accuracy trên KC 0,885, dev 57/74 (assert trong script).
- Với ViANLI, nhận định = giả thuyết, state = tiền đề.

| Số đo | ViANLI contradiction | ViANLI entailment | ViANLI neutral | BB dev ViWikiFC | BB proxy_v1 (Wiki + FC) | BB proxy_v1 ViWikiFC | BB KC (v3) | NEI KC (v3) |
|---|---|---|---|---|---|---|---|---|
| Số cặp / dòng | 2.148 | 2.247 | 2.495 | 74 | 45 | 22 | 637 | 918 |
| Số tiền đề khác nhau | 2.122 | 2.188 | 2.293 | — | — | — | — | — |
| Token tiền đề: trung vị / p90 | 38 / 61 | 38 / 61 | 38 / 61 | — | — | — | — | — |
| Token giả thuyết: trung vị / p90 | 21 / 33 | 19 / 31 | 23 / 35 | — | — | — | — | — |
| max_ov trung vị | 0,625 | 0,450 | 0,586 | 0,667 | 0,600 | 0,591 | 0,909 | 0,414 |
| max_ov < 0,5 / < 0,7 / < 0,9 | 0,32 / 0,59 / 0,83 | 0,55 / 0,84 / 0,97 | 0,35 / 0,67 / 0,90 | 0,16 / 0,61 / 0,96 | 0,16 / 0,64 / 0,93 | 0,23 / 0,73 / 1,00 | 0,01 / 0,30 / 0,46 | 0,58 / 0,70 / 0,88 |
| cover trung vị | 0,625 | 0,455 | 0,586 | 0,692 | 0,867 | 0,667 | 0,968 | 0,667 |
| cover < 0,5 / < 0,7 / < 0,9 | 0,32 / 0,59 / 0,83 | 0,55 / 0,84 / 0,97 | 0,34 / 0,67 / 0,90 | 0,11 / 0,54 / 0,93 | 0,02 / 0,29 / 0,60 | 0,05 / 0,55 / 0,95 | 0,00 / 0,00 / 0,06 | 0,32 / 0,54 / 0,80 |
| Phủ định (danh sách D5) | 0,164 | 0,134 | 0,121 | 0,311 | 0,289 | 0,409 | 0,093 | 0,088 |
| Lượng từ / độc quyền | 0,112 | 0,066 | 0,090 | 0,122 | 0,156 | 0,182 | 0,130 | 0,131 |
| Số lạ | 0,060 | 0,132 | 0,072 | 0,108 | 0,156 | 0,091 | 0,336 | 0,678 |
| Tên lạ | 0,030 | 0,042 | 0,021 | 0,027 | 0,089 | 0,045 | 0,275 | 0,439 |
| **Probe: phía NEI của KC** | **1.582 (0,736)** | 2.109 (0,939) | 2.056 (0,824) | **57 (0,770)** | 23 (0,511) | 18 (0,818) | 34 (0,053) | 773 (0,842) |

**Thêm:**
- **Số giả thuyết mỗi tiền đề** (pool sạch):

  | Số giả thuyết | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Số tiền đề | 609 | 1.538 | 1.036 | 10 | 9 | 2 |

  Phần lớn tiền đề có 2–3 giả thuyết, thường khác nhãn.
- **Tiền đề cũng là tiền đề của một cặp trong C′** (khác giả thuyết):
  - contradiction 481 / 2.122;
  - entailment 502 / 2.188;
  - neutral 559 / 2.293.

  Brief không loại các cặp này; ghi lại để điều phối quyết ở bước C.

**Ghi chú đọc** (chỉ mô tả, không quyết):
- **Contradiction của ViANLI giống Bác bỏ của dev về độ trùng từ:**
  - max_ov trung vị 0,625 so với 0,667; < 0,7 là 59% so với 61%;
  - probe xếp vào phía NEI 73,6% so với 77,0%.

  KC Bác bỏ thì ngược hẳn: max_ov trung vị 0,909; probe chỉ xếp 5,3% vào phía NEI.
- **ViANLI có ít phủ định hơn dev:** contradiction 16,4%, so với 31,1% ở Bác bỏ dev và 28,9% ở Bác bỏ proxy.
- **Số lạ và tên lạ ít** (6% và 3%), gần dev (11% và 3%), khác xa KC (34% và 28%).
- **Entailment của ViANLI trùng từ thấp nhất** (trung vị 0,45), thấp hơn cả neutral (0,586). Nghĩa là khi trùng từ thấp, nhãn trong ViANLI không nghiêng về NEI.
- **Độ dài:** tiền đề ViANLI là một câu báo, trung vị 38 token. Ngắn hơn nhiều so với state dev. Ở bước C, state ghép 3 tiền đề.

**Giới hạn:**
- **Regex lượng từ** khớp theo ranh giới từ, nên "chỉ" cũng khớp "chỉ huy", "chỉ đạo", "địa chỉ". Danh sách giữ nguyên theo brief, không lọc thêm.
- **Cover của ViANLI ≈ max_ov,** vì tiền đề thường chỉ có 1 câu. So cover với dev (state nhiều câu) không cùng điều kiện.
- **Cột proxy_v1 nhỏ:** 45 dòng; riêng ViWikiFC 22 dòng.

**File:**
- `results/train_2b_prime/kiem_ke_vianli.json` (kèm sha đầu vào), `kiem_ke_vianli.log`;
- `mau_vianli_bacbo.jsonl`: 40 contradiction, seed 20261029; cột id, tiền đề, giả thuyết, max_ov;
- `tam/pool_vianli.jsonl`: pool sạch 6.890 cặp kèm đặc trưng, nội bộ, cho bước C.

## 3. Bước C — dựng (sau "DỰNG 2b′"; chốt điều phối ở brief mục 2b)

- **Script:**
  - `scripts/train_2b_prime/build_2b_prime.py` (WSL), seed 20261029;
  - `scripts/train_2b_prime/d1_d4.py` (Windows python, sklearn).
- **Hai lần chạy:**
  - **Lần 1:** trượt D11(a). File lần 1 giữ nguyên ở `results/train_2b_prime/tam/lan1/`.
  - **Lần 2:** lần lấy mẫu lại duy nhất mà brief cho, cờ `--lay_mau_lai`. Đạt mọi cổng, nhưng có cảnh báo ở mục 3.4.
  - **Điều phối quyết (29/10): bỏ lần 2, lần 1 là ứng viên, ghi rõ trượt D11(a).** Lý do: trong ViANLI, tương quan bị đảo ngược hoàn toàn (probe 1,000), và phủ định lệch thêm. Đó là đạt cổng nhờ lách, không phải sửa được nguyên nhân.
  - **(Cập nhật 29/10, chiều: file chính hiện là V-e, xem mục 5. Lần 1 ở `tam/lan1/`.)** Trước V-e, file chính trong `results/train_2b_prime/` là của lần 1 (chép lại từ `tam/lan1/`, đã kiểm từng byte). Bộ lần 2 ở `tam/lan2/`. sha của cả hai lần ghi ở `manifest_2b_prime.json` (khóa `sha_hai_lan`) và ở mục 3.5.
  - Số ở mục 3.1–3.3 là của lần 1.
- **Khác nhau giữa hai lần:** chỉ thứ tự lấy trong hai ô.
  - choice Bác bỏ: lấy theo max_ov tăng dần;
  - choice NEI: lấy theo max_ov giảm dần.
  - Noul, val, cỡ khối, cơ cấu và C2 không đổi.
  - **Mức sửa:** brief ghi "hướng tới" Bác bỏ trùng từ thấp và NEI trùng từ cao. Lần 2 làm ở mức cực đại (xếp hẳn theo max_ov ở cả hai ô). Chưa thử mức nhẹ hơn, và lần sửa duy nhất đã dùng hết.
- **Val không đổi:** `val_vianli.jsonl` giống hệt từng byte giữa lần 1 và lần 2 (sha `2ff71578…`). Chọn file train nào thì val vẫn như nhau.

### 3.1 Khối ViANLI (C1)

**Cỡ:** 1.500 dòng (trần 1.580). Val 75 dòng.

| | choice | noul | tổng |
|---|---|---|---|
| Bác bỏ | 330 | 263 | 593 |
| Xác nhận | 272 | 240 | 512 |
| NEI | 223 | 172 | 395 |

- **Noul:** số "Có" chọn để noul của khối kiểm chứng VI gộp đạt 38,5% (D7). Riêng noul ViANLI có "Có" 35,6%.
- **State:** tiền đề thật + 2 tiền đề ghép. Tiền đề ghép không có tên riêng hay số chung với giả thuyết, kiểm cả hai chiều.
  - Vị trí tiền đề thật, theo nhãn (vị trí 0 / 1 / 2):

    | Nhãn | 0 | 1 | 2 |
    |---|---|---|---|
    | Xác nhận | 172 | 169 | 171 |
    | Bác bỏ | 191 | 184 | 218 |
    | NEI | 138 | 127 | 130 |

  - Một tiền đề ghép dùng tối đa 5 lần.
- **Tiền đề thật:**
  - 1.275 tiền đề khác nhau; 211 tiền đề dùng ≥ 2 lần.
  - Nhãn của các tiền đề dùng nhiều lần: Bác bỏ + Xác nhận 81; NEI + Xác nhận 54; NEI + Bác bỏ 59; cả ba nhãn 9; còn lại lẻ.
- **Trùng tiền đề với C′** (cho phép theo chốt 2):
  - 368 dòng có tiền đề thật cũng là tiền đề của một cặp trong C′;
  - 810 dòng có ít nhất một tiền đề ghép như vậy.
- **Val:**
  - 75 dòng, cùng cơ cấu: choice Xác nhận / Bác bỏ / NEI 14 / 16 / 11; noul 12 / 13 / 9; 15 dòng không dấu.
  - Dùng 150 tiền đề dành riêng, không có trong C′.
  - Không tiền đề nào xuất hiện ở cả train và val, ở bất kỳ vai nào (script assert).

### 3.2 Bỏ dấu tại chỗ (C2)

| Phần | Số dòng | Ghi chú |
|---|---|---|
| (a) ViANLI | 313 / 1.500 (20%, hash id) | Bỏ dấu: Xác nhận / Bác bỏ / NEI 102 / 131 / 80. Có dấu: 410 / 462 / 315 |
| (b) 2d | 300 | Cả 300 thuộc nhóm ưu tiên (Xác nhận / Bác bỏ, max_ov < 0,7; có 561 dòng đủ điều kiện): Xác nhận 169, Bác bỏ 131. 16 dòng bị bỏ qua vì bản không dấu chạm khử trùng (dev 15, test 1), bù theo hash |
| (c) ngoài khung | 323 (mục tiêu khoảng 410) | 2f 145; vi-tourism-admin-intent 102; multidogo 76; **pholaya/vsfc/topic 0** |
| (d) KC không dấu có sẵn | 587 | giữ nguyên |

**Cảnh báo về (c), nhóm chủ đề có 0 dòng:**
- Có 86 dòng được chọn theo hash nhưng bị bỏ qua, vì bản không dấu chạm khử trùng: 68 lần `nguoc_contain_short|test13`, phần còn lại thuộc dev, test và Belebele.
- Lựa chọn ngắn của nhóm chủ đề (như "khac", "giang vien") nằm trọn trong văn bản không dấu của test/dev. Đây là trùng chuỗi nhãn ngắn, không phải rò rỉ nội dung. Nhưng luật khử trùng coi đó là chạm, nên mình bỏ qua, không bù.

**Tổng dòng VI không dấu:** 587 (v3) → 1.523 (2b′ lần 1; lần 2 là 1.514).

### 3.3 Thành phần v3 so với 2b′

| Khối | v3 | 2b′ |
|---|---|---|
| C′ khác (VI) | 9.568 | 9.568 |
| C′ khác (EN) | 15.556 | 15.556 |
| C′ khác (multi) | 403 | 403 |
| KC | 3.000 | 3.000 |
| 2d | 1.160 | 1.160 (300 bỏ dấu) |
| 2e | 250 | 250 |
| 2f | 1.659 | 1.659 (145 bỏ dấu) |
| ViANLI khung (mới) | 0 | 1.500 |
| **Tổng** | **31.596** | **33.096** |
| Tỉ lệ VI | 49,5% | 51,8% |
| Dòng VI không dấu | 587 | 1.523 |

`train_2b_v3.jsonl` được kiểm sha `ea604a01…` trước khi dựng và không bị ghi đè.

### 3.4 Cổng D1–D12

| Mã | Lần 1 (file chính, ứng viên) | Lần 2 (bỏ, ở tam/lan2) | Ngưỡng | Lần 2 |
|---|---|---|---|---|
| D1 nhãn | kiểm chứng / Có-Không: 0; khung khác 1.077 (= v3) | như lần 1; 0 cặp có dấu / không dấu | 0; ≤ v3 | đạt |
| D2 kho chuỗi | ViANLI + C2(b): 0 chuỗi ngoài dev | như lần 1 | 0 | đạt |
| D3 khử trùng | 0 chạm (1.575 dòng ViANLI) | 0 chạm | 0 | đạt |
| D4 probe chỉ đọc nhận định | 0,372 (đa số 0,400) | **0,485** (đa số 0,400; +8,5 điểm) | ≤ đa số + 10 | đạt |
| D5′ phủ định (chênh, điểm) | 5,29 | 9,33 | ≤ proxy 20,34 + 2 | đạt |
| D5′ lượng từ (chênh, điểm) | 2,67 | 4,30 | ≤ proxy 5,96 + 2 | đạt |
| D7 | NEI choice 34,3%; noul "Có" 38,5% | như lần 1 | ≥ 30%; 37–40% | đạt |
| **D11(a)** probe Bác bỏ / NEI, khối gộp | **0,725 → trượt** | **0,603** | ≤ proxy 0,631 + 0,05 = 0,681 | đạt |
| D11(b) Bác bỏ proxy bị xếp vào phía NEI (chỉ báo) | 0,240 (v3: 0,208) | 0,260 (v3: 0,208) | — | — |
| D12 | state trung vị 121 / p90 164 / max 271 token | 119 / 163 / 271; 0 dòng vượt 2.048 | 0 | đạt |

**Cách tính:**
- **D2 với C2(c):** chỉ đếm, theo chốt 5.
  - 665 chuỗi không dấu khác nhau không có trong dev (146 câu hỏi), xuất hiện 2.318 lần. Đều là bản bỏ dấu của chữ đã có trong train.
  - Chuỗi nhãn kiểm chứng và NEI trong C2(c) trùng nodiac của dev.
- **D5′ và D11(a)** dùng proxy_v1 có dấu, choice + noul (ViWikiFC + ViFactCheck) làm mốc. Nếu chỉ lấy choice:
  - D11 proxy là 0,495, nên ngưỡng sẽ là 0,545 và cả hai lần đều trượt;
  - phủ định 20,12; lượng từ 6,14.

  Mốc choice + noul được chọn trước khi chạy và ghi trong script.

**Cảnh báo về lần 2 (worker không quyết, chỉ báo):**
- **Lần 2 đạt D11(a) nhờ một tương quan ngược trong riêng khối ViANLI, không phải nhờ khử tương quan.** Probe tách Bác bỏ / NEI:

  | Phần | Lần 1 | Lần 2 |
  |---|---|---|
  | riêng ViANLI | 0,652 (đa số 0,603) | **1,000** |
  | KC + 2d + 2e (v3) | 0,733 | 0,733 |

- **P(nhãn | dải max_ov), riêng ViANLI choice có dấu, lần 2** (Xác nhận / Bác bỏ / NEI):

  | Dải max_ov | Số dòng | Xác nhận | Bác bỏ | NEI |
  |---|---|---|---|---|
  | [0; 0,5) | 378 | 0,31 | 0,69 | 0 |
  | [0,5; 0,7) | 58 | 1,00 | 0 | 0 |
  | [0,7; 0,9) | 119 | 0,32 | 0 | 0,68 |
  | ≥ 0,9 | 104 | 0,08 | 0 | 0,92 |

  Ở lần 1, cả bốn dải đều có đủ ba nhãn.
- **Hệ quả có thể xảy ra:** dòng ViANLI có hình thức riêng (state gồm 3 câu báo ngắn), nên model có thể học luật có điều kiện: "state kiểu ViANLI, trùng từ thấp → Bác bỏ; trùng từ cao → NEI". Chỉ đo trên khối gộp thì không thấy được điều này.
- **Phủ định lệch thêm.** Chênh phủ định giữa ba nhãn (D5′) tăng từ 5,29 lên 9,33 điểm (Bác bỏ 15,7%, hai nhãn kia khoảng 6,4%). Đây là tác dụng phụ của việc chọn Bác bỏ trùng từ thấp. Vẫn dưới ngưỡng, nhưng đi ngược ý "cân phủ định" của brief.
- **D4 tăng** từ 0,372 lên 0,485: chọn theo độ trùng từ làm giả thuyết Bác bỏ / NEI khác nhau rõ hơn về chữ. Vẫn đạt, nhưng chỉ còn cách ngưỡng 1,5 điểm.
- **P(nhãn | dải max_ov), khối gộp** (choice có dấu; Xác nhận / Bác bỏ / NEI):

  | Dải max_ov | v3 | 2b′ lần 1 | 2b′ lần 2 | proxy_v1 |
  |---|---|---|---|---|
  | [0; 0,5) | 0,15 / 0,12 / 0,73 | 0,17 / 0,13 / 0,70 | 0,16 / 0,27 / 0,57 | 0,26 / 0,20 / 0,55 |
  | [0,5; 0,7) | 0,46 / 0,35 / 0,19 | 0,38 / 0,35 / 0,27 | 0,48 / 0,29 / 0,24 | 0,36 / 0,36 / 0,28 |
  | [0,7; 0,9) | 0,49 / 0,28 / 0,23 | 0,45 / 0,31 / 0,24 | 0,46 / 0,22 / 0,32 | 0,36 / 0,42 / 0,22 |
  | ≥ 0,9 | 0,62 / 0,28 / 0,10 | 0,56 / 0,32 / 0,11 | 0,54 / 0,24 / 0,21 | 0,65 / 0,27 / 0,08 |

  Khối gộp lần 2 gần proxy hơn ở dải thấp: NEI 0,57 so với 0,55 của proxy. Ở dải cao, NEI 0,21 so với 0,08 của proxy.
- **Theo brief:** đã dùng hết lần sửa duy nhất. Không lặp thêm.
- **Điều phối đã chọn lần 1** (xem đầu mục 3). Khối v3 cũng chỉ ở 0,733, tức là thêm ViANLI gần như không làm khối gộp bớt tách được bằng đặc trưng bề mặt. Bước tiếp được quyết ở phân tích trước train.

### 3.5 Mẫu đọc mù và file

- **`mau_doc_vianli.jsonl`** (lần 1; bản lần 2 ở `tam/lan2/`): 30 dòng mỗi nhãn của khối ViANLI, cộng 20 dòng không dấu của C2(c), xáo theo seed.
  - Cột `nhan` (chuỗi đáp án) và `nhan_3_lop` để riêng, điều phối che trước khi đọc.
- **`manifest_2b_prime.json`:**
  - sha đầu vào và đầu ra, seed, tham số, cờ `lay_mau_lai_D11`;
  - số dòng bị bỏ ở bước B; danh sách id đã bỏ dấu theo (a), (b), (c); toàn bộ số đo cổng.
- **Bản đầy đủ của cổng:** `tam/cong_2b_prime.json` và `tam/d1_d4.json` (D1 chi tiết ở `tam/d1_2b_prime.json`, `tam/d1_v3.json`).
- **Lần 1:** `tam/lan1/` (cùng bộ file, giống hệt thư mục chính). **Lần 2:** `tam/lan2/`.
- **sha:**

  | File | Lần 1 (thư mục chính) | Lần 2 (`tam/lan2/`) |
  |---|---|---|
  | train_2b_prime.jsonl | `c993e452…` | `2ca76281…` |
  | vianli_khung.jsonl | `3074836d…` | `f4862131…` |
  | val_vianli.jsonl | `2ff71578…` | `2ff71578…` |
  | mau_doc_vianli.jsonl | `dde42fdf…` | `41dd2afc…` |

## 4. Giới hạn

- **Nhãn ViANLI không lọc** (chốt 3). Điều phối đọc 40 cặp contradiction: khoảng 15% đáng ngờ, phần lớn là "mâu thuẫn nhờ kiến thức ngoài văn bản". 12,5% phi lý hoặc dễ nhận ra.
- **Tiền đề ghép chỉ được kiểm** không chung tên riêng hay số với giả thuyết. Một tiền đề ghép vẫn có thể tình cờ liên quan đến giả thuyết; chưa đọc kiểm.
- **max_ov của ViANLI tính trên state 3 câu** (câu khớp nhất). Thứ tự lấy ở lần 2 dùng max_ov của bước B, tính trên riêng tiền đề thật. Hai số gần nhau nhưng không trùng.
- **Mốc proxy cho D5′ và D11 nhỏ:** 190–300 dòng. Cùng một khối, mốc choice và mốc choice + noul cho kết luận khác nhau ở D11.
- **C2(c) thiếu nhóm chủ đề** (mục 3.2) và chỉ đạt 323 / khoảng 410 dòng.
- **Regex lượng từ** khớp cả "chỉ huy", "chỉ đạo", "địa chỉ".

## 5. Biến thể V-e (điều phối quyết 29/10; theo `results/huong_di_2710/FABLE_truoc_train_2b_prime.md` mục 2)

- **Thay đổi duy nhất (C2(b)):** 300 dòng 2d bỏ dấu, chọn theo hash mù trên **mọi** dòng 2d có dấu. Bỏ ưu tiên "Xác nhận / Bác bỏ có max_ov < 0,7".
  - Đây là **sai lệch có ngày so với brief C2(b)**. Lý do: C2(b) bỏ dấu đúng những dòng Bác bỏ trùng từ thấp có dấu mà P2 cần thêm (Fable, mục 1b).
  - Bản không dấu chạm khử trùng (15 dòng chạm dev) thì bù theo hash, như cũ.
  - Lệnh: `build_2b_prime.py --ve`.
- **Kiểm "giữ nguyên từng byte":**
  - Bản chính trước khi ghi đè khớp `tam/lan1/` (8 file, sha train `c993e452…`).
  - `vianli_khung.jsonl`, `val_vianli.jsonl`, `mau_doc_vianli.jsonl`: sha trước và sau khi dựng trùng nhau (`tam/sha_truoc_ve.txt`, `tam/sha_sau_ve.txt`).
  - Trong `train_2b_prime.jsonl`, chỉ 342 dòng khác lần 1, và cả 342 đều là dòng 2d. Mọi dòng khác (KC, ViANLI, C2(a), C2(c), C′) giống hệt từng byte, cùng thứ tự. Danh sách dòng không dấu ngoài 2d trùng lần 1. Cả hai bản đều có 300 dòng 2d không dấu.
  - Nhãn 2d bỏ dấu: V-e là Xác nhận 157, Bác bỏ 89, NEI 51, không xác nhận 3. Lần 1 là Xác nhận 169, Bác bỏ 131.
- **sha:** V-e train `c97bb08d…`. Lần 1 `c993e452…`. Lần 2 `2ca76281…` (bị bỏ). ViANLI, val và mẫu đọc mù của V-e trùng lần 1. Đầy đủ ở `manifest_2b_prime.json`, khóa `sha_ba_ban`.

**Cổng** (D11(a) chỉ còn là chỉ báo; cổng mới (i) và (ii) theo điều phối):

| Mã | Lần 1 | V-e | Ngưỡng | V-e |
|---|---|---|---|---|
| D1 | kiểm chứng / Có-Không 0; khung khác 1.077 (= v3) | như lần 1; 0 cặp có dấu / không dấu | 0; ≤ v3 | đạt |
| D2 | 0 chuỗi ngoài dev | 0 | 0 | đạt |
| D3 | 0 chạm | 0 chạm (C2(b): 15 dòng bỏ qua vì chạm dev, đã bù) | 0 | đạt |
| D4 | 0,372 (đa số 0,400) | 0,372 (khối ViANLI không đổi) | ≤ đa số + 10 | đạt |
| D5′ phủ định / lượng từ (chênh, điểm) | 5,29 / 2,67 | 5,49 / 2,88 | ≤ 22,34 / 7,96 | đạt |
| D7 NEI choice / noul "Có" | 34,3% / 38,5% | 34,3% / 38,5% | ≥ 30% / 37–40% | đạt |
| D12 | 0 dòng vượt | 0 | 0 | đạt |
| D11(a) khối gộp (**chỉ báo**) | 0,725 | 0,722 | — | — |
| D11(b) Bác bỏ proxy bị xếp vào phía NEI (chỉ báo) | 0,240 | 0,177 (v3 0,208) | — | — |
| **(i) Probe riêng ViANLI** | 0,631 (đa số 0,603; +2,8) | **0,631 (+2,8)**; cách chia fold của build: 0,652 (+4,9) | ≤ đa số + 10 | **đạt** |
| (ii) Bảng 1a / 1b / 1c | dưới đây | dưới đây | báo | báo |

Probe (i) cho hai số vì hai script chia fold theo hai thứ tự dòng khác nhau: `bang_ve.py` theo thứ tự file train, giống Fable (0,63); `build_2b_prime.py` theo thứ tự khối. Cả hai đều đạt.

**1a. P(nhãn | dải cover), choice, % (Xác nhận / Bác bỏ / NEI)**

| Khối | [0; 0,5) | [0,5; 0,7) | [0,7; 0,9) | ≥ 0,9 |
|---|---|---|---|---|
| v3 có dấu | n=645: 11/12/76 | n=479: 34/28/38 | n=501: 47/21/32 | n=1.053: 57/32/11 |
| lần 1 có dấu | n=781: 18/13/70 | n=513: 26/26/47 | n=615: 40/27/33 | n=1.126: 54/34/12 |
| **V-e có dấu** | **n=802: 20/15/65** | **n=593: 32/29/39** | n=562: 39/27/34 | n=1.078: 52/35/13 |
| Fable mô phỏng V-e có dấu | n=808: 19/15/66 | n=573: 31/30/38 | n=562: 39/27/34 | n=1.088: 52/36/12 |
| v3 không dấu | n=28: 0/0/100 | n=41: 0/0/100 | n=41: 0/12/88 | n=239: 46/37/16 |
| lần 1 không dấu | n=138: 34/36/30 | n=268: 39/37/24 | n=143: 36/29/35 | n=268: 46/39/16 |
| **V-e không dấu** | n=120: 24/22/53 | n=191: 28/32/39 | n=187: 37/29/34 | n=319: 52/34/13 |
| lần 1 cả hai | n=919: 20/16/64 | n=781: 31/30/39 | n=758: 40/27/33 | n=1.394: 52/35/13 |
| V-e cả hai | n=922: 20/16/63 | n=784: 31/30/39 | n=749: 39/27/34 | n=1.397: 52/35/13 |

**1b. Bác bỏ choice có cover < 0,7**

| | 2d có dấu | ViANLI có dấu | KC | **Tổng có dấu** | 2d không dấu | ViANLI không dấu | **Tổng cả hai** |
|---|---|---|---|---|---|---|---|
| v3 | 215 | — | 0 | 215 | 0 | — | 215 |
| lần 1 | 98 | 135 | 0 | 233 | 111 | 38 | 382 |
| **V-e** | **160** | 135 | 0 | **295** | 51 | 38 | **384** |
| Fable mô phỏng V-e | 161 | 135 | 0 | 296 | 54 | 38 | 388 |

**Choice có dấu, cover < 0,7** (Xác nhận / Bác bỏ / NEI):

| Bản | Xác nhận | Bác bỏ | NEI |
|---|---|---|---|
| v3 | 234 | 215 | 675 |
| lần 1 | 273 | 233 | 788 |
| V-e | 349 | 295 | 751 |
| Fable mô phỏng V-e | 336 | 296 | 749 |

**1c. Noul: P("Có" | dải cover), %**

| Khối | [0; 0,5) | [0,5; 0,7) | [0,7; 0,9) | ≥ 0,9 |
|---|---|---|---|---|
| v3 có dấu | n=128: 2 | n=101: 5 | n=125: 2 | n=791: 55 |
| lần 1 có dấu | n=289: 28 | n=268: 27 | n=250: 13 | n=868: 51 |
| V-e có dấu | n=290: 28 | n=268: 28 | n=249: 13 | n=868: 51 |
| lần 1 không dấu | n=56: 34 | n=64: 27 | n=58: 28 | n=205: 55 |
| V-e không dấu | n=55: 33 | n=64: 22 | n=59: 27 | n=205: 55 |

**Đọc** (chỉ mô tả):
- V-e khớp mô phỏng của Fable trong khoảng ±1–2 điểm và ±20 dòng mỗi dải. Phần lệch nhỏ đến từ 15 dòng 2d bị bỏ qua vì chạm khử trùng rồi được bù (mô phỏng không tính bước này).
- So với lần 1, phần có dấu được thêm 62 dòng Bác bỏ và 76 dòng Xác nhận trùng từ thấp. P(NEI | cover < 0,5) giảm từ 70 xuống 65%; P(NEI | [0,5; 0,7)) giảm từ 47 xuống 39%.
- P(NEI | cover ≥ 0,7) gần như không đổi (34 và 13%).
- Giá phải trả ở phần không dấu: 2d Bác bỏ trùng thấp không dấu giảm từ 111 xuống 51. Dải thấp không dấu chuyển từ 34/36/30 sang 24/22/53. Vẫn đủ ba nhãn.
- **File:** `tam/bang_1abc_{v3,lan1,ve}.json` (script `scripts/train_2b_prime/bang_ve.py`), `tam/cong_2b_prime.json` (V-e). Cổng lần 1 vẫn ở `tam/lan1/`.
