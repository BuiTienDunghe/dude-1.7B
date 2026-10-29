# BRIEF 2b′: phần CPU — ngưỡng noul, khối ViANLI, không dấu (29/10/2026)

**Người dùng duyệt** (chat 29/10, chiều): "được, làm phần CPU của hướng 1 đi".

Lý do và thiết kế:
- `results/huong_di_2710/FABLE_nguyen_nhan_2b_va_huong_sua.md`, mục 6: P1, P2, P4.
- `results/train_2b/PHAN_TICH_LOI_BACBO.md`: lỗi Bác bỏ → NEI, probe bề mặt.

Brief dữ liệu 2b (`results/BRIEF_2b_du_lieu.md`) vẫn áp dụng cho cách dựng, khử trùng và định dạng, trừ các điểm ghi dưới đây.

## 0. Phạm vi và ngưỡng dừng (bắt buộc)

- **Chỉ CPU. Không chạy model nào, không dùng GPU.** Không train.
- **Mức dùng tuần: ngưỡng 35%.**
  - Từ 32% trở lên thì không bắt đầu bước mới.
  - Điều phối nhắn "DỪNG USAGE" thì dừng an toàn, rồi báo lại.
- **Không đụng test:** chỉ dò khử trùng như các lần trước, không in nội dung.
- Không sửa `config/`, `data/bench_v1_3/`, `scripts/train_lora.py`, `scripts/probe_v1/score_probe.py`. Không xóa, không commit.
- **Không ghi đè** `results/train_2b/train_2b_v3.jsonl` hay file nào của 2b. Mọi đầu ra để ở thư mục mới `results/train_2b_prime/`. Script để ở `scripts/train_2b_prime/`.
- **Không thêm chữ tiếng Việt mới.**
  - Câu hỏi và chuỗi lựa chọn chép nguyên từ dòng dev cùng khung, như KC.
  - Tiền đề và giả thuyết lấy nguyên từ ViANLI.
- **Không dùng train split ViWikiFC/ViFactCheck.** Dữ liệu UIT (ViANLI) chỉ dùng nội bộ, như phần ViANLI đã có trong C′.
- Không tải gì.

## 1. Bước A — P1: ngưỡng noul (hậu xử lý trên preds có sẵn)

Viết script riêng, ví dụ `scripts/train_2b_prime/nguong_noul.py`. **Không sửa đường chấm argmax.**

1. **Đầu vào:**
   - `results/train_2b/eval/proxy/{C-s1,2b-s1,2b-s2}/preds.jsonl`;
   - `results/probe_v1/eval/dev/{C-s1,2b-s1,2b-s2}/preds.jsonl`.
2. **Fit trên proxy:**
   - Dữ liệu: các dòng noul có dấu (ViWikiFC + ViFactCheck gộp).
   - Quy tắc: đoán "Có" khi p(Có) ≥ t. Chạy t từ 0,05 đến 0,95, bước 0,01.
   - Chọn t* cho accuracy cao nhất. Hòa thì lấy t gần 0,5 nhất.
   - **Fit riêng cho từng adapter.**
3. **Báo trên dev (không fit trên dev):**
   - noul theo nguồn, có dấu và không dấu, trước và sau ngưỡng;
   - VI macro trước và sau.
   - Không dấu dùng cùng t* fit trên proxy có dấu, báo riêng.
4. **Cách đọc** (ghi trước):
   - Với C-s1, 2b-s1 và 2b-s2, P1 chỉ là **dòng chỉ báo**. Lý do: s2 đã đọc rồi, nên đây là hậu kiểm.
   - Với các adapter 2b′ sau này, P1 là **một dòng riêng, cùng luật**. Dòng này không gộp vào T4 và không dùng khi so với decider (decider không có ngưỡng).
5. **Đầu ra:**
   - `results/train_2b_prime/P1_nguong_noul.json`;
   - một mục trong báo cáo.

Bước A độc lập với các bước sau, nên làm song song được.

## 2. Bước B — kiểm kê ViANLI (CPU; xong thì **dừng, chờ điều phối**)

Nguồn: `results/train_v1/raw_src/LakoreAI__PhoLaya-base/sources/vianli/train.jsonl`.
- 16.024 dòng, tức 8.012 cặp có câu hỏi `relation`: contradiction 2.473 / entailment 2.615 / neutral 2.924.
- Mọi `probabilities` đều bằng 1,0, nên không lọc theo xác suất được.

**Loại trước khi đếm:**
- mọi cặp (tiền đề, giả thuyết) đã có trong C′ (khoảng 1.003 cặp; khớp theo chuỗi chuẩn hóa);
- mọi cặp có tiền đề chạm khử trùng (mục 4).

**Báo theo nhãn** (contradiction / entailment / neutral), cho phần còn lại:
1. Số cặp và số tiền đề khác nhau.
2. Độ dài tiền đề và giả thuyết (token Qwen3, trung vị và p90).
3. max_ov và độ phủ token, dùng đúng hàm `kc_util` mà `phan_tich_bacbo.py` dùng:
   - trung vị;
   - tỉ lệ < 0,5, < 0,7, < 0,9.
4. Tỉ lệ có phủ định ("không / chưa / chẳng / chả / đâu có"), cùng danh sách với D5.
5. Tỉ lệ có từ lượng từ hoặc độc quyền: "chỉ", "duy nhất", "độc nhất", "đầu tiên", "cuối cùng", "phần lớn", "tất cả", "mọi", "một phần".
6. Tỉ lệ có số lạ và tên lạ (số hoặc tên không có trong tiền đề).
7. **Probe bề mặt của `phan_tich_bacbo.py`** (cùng đặc trưng, cùng cách fit trên KC Bác bỏ / KC NEI):
   - áp lên contradiction ViANLI;
   - báo tỉ lệ bị xếp vào phía NEI. Dev là 57/74.
8. Đặt cạnh cùng số đo của Bác bỏ dev ViWikiFC (mục 3 của báo cáo lỗi Bác bỏ) và của Bác bỏ proxy_v1.

**Mẫu đọc:** xuất `results/train_2b_prime/mau_vianli_bacbo.jsonl`.
- 40 cặp contradiction ngẫu nhiên, seed cố định;
- các cột: tiền đề, giả thuyết, max_ov.

Điều phối sẽ đọc mẫu này để xếp vào 6 kiểu bác bỏ ở mục 4 của báo cáo lỗi Bác bỏ.

Gửi **"KIỂM KÊ ViANLI XONG"** kèm bảng, rồi **chờ**. Chỉ dựng bước C khi điều phối nhắn **"DỰNG 2b′"**. Điều phối có thể đổi cơ cấu ở bước C theo kết quả kiểm kê và sẽ ghi lại bằng văn bản.

## 2b. Điều phối đọc kiểm kê và mẫu (29/10, khoảng 15:10) → "DỰNG 2b′"

**P1:** với họ 2b, ngưỡng gần như không đổi gì: VI macro s1 −0,23, s2 0,00. t* không ổn định (s2 có 23 giá trị t hòa). Kết luận: **bỏ P1 làm đòn bẩy**. Vẫn tính cho adapter 2b′ như một dòng chỉ báo, không dùng để quyết.

**Đọc 40 cặp contradiction** (một người đọc là điều phối, không đọc mù; xếp theo 6 kiểu ở mục 4 của báo cáo lỗi Bác bỏ):

| Kiểu | Số câu | Ví dụ (số thứ tự dòng trong mẫu) |
|---|---|---|
| Thay danh từ hoặc động từ chung, kể cả trái nghĩa | 10 | 2, 6, 17, 24, 27, 31, 36, 37, 38, 39 |
| Đảo vai hoặc sự kiện ngược | 6 | 11, 15, 22, 26, 32, 35 |
| Suy luận số hoặc thời gian | 6 | 9, 14, 16, 20, 21, 25 |
| Lượng từ hoặc độc quyền | 3 | 4, 12, 13 |
| Phủ định | 3 | 7, 23, 29 |
| Kiểu KC (đổi số hoặc tên trong câu chép gần nguyên) | 1 | 1 |
| Phi lý hoặc dựa vào kiến thức ngoài văn bản, dễ nhận ra | 5 | 3, 5, 19, 28, 40 |
| Nhãn đáng ngờ (đọc giống NEI, hoặc giống Xác nhận) | 6 | 8, 10, 18, 30, 33, 34 |

- **5 kiểu bác bỏ của dev chiếm 28/40 (70%).** Kiểu KC chỉ có 1/40. Khớp với số đo: probe xếp 73,6% vào phía NEI, dev là 77%.
- **Nhãn đáng ngờ 6/40 (15%).** Phần lớn là "mâu thuẫn nhờ kiến thức ngoài văn bản". Mức này gần mức của dev (3/24 câu đáng ngờ trong phần đọc tay của worker), nên chấp nhận được, nhưng ghi vào giới hạn. Đọc mù mẫu `mau_doc_vianli.jsonl` sẽ quyết sau khi dựng.
- **Phủ định ít hơn dev** (16% so với 31%). Không cần bù: 2b mất ít câu Bác bỏ có phủ định (s2 đúng 0,739) hơn câu không phủ định (0,549).

**Chốt cho bước C** (ngoài các điểm đã ghi trong brief):
1. **Giữ cơ cấu** ở C1.
2. **Tiền đề trùng với cặp trong C′** (khác giả thuyết): **cho phép** trong train, vì cả hai đều là dữ liệu train, không chạm bench. Báo số dòng.
   - `val_vianli.jsonl` chỉ dùng tiền đề **không** có trong C′ và không có trong khối train.
3. **Không lọc** câu phi lý hay đáng ngờ, vì không có cách lọc cơ học. Chỉ báo, qua đọc mù sau.
4. **C2(c), phạm vi** (sau câu hỏi của worker): lấy 10% theo hash id của 4 nhóm, khoảng 410 dòng:
   - 2f đọc hiểu, 1.659 dòng;
   - vi-tourism-admin-intent, 993 dòng;
   - multidogo/insurance/intent, 773 dòng;
   - pholaya/vsfc/topic, 671 dòng.

   Không lấy 10% trên toàn bộ dòng VI ngoài khung (khoảng 1.100 dòng). Lý do: danh sách nhóm của Fable ở P4 chỉ có đọc hiểu, ý định và chủ đề.
5. **C2(c), cách bỏ dấu và D2:**
   - Bỏ dấu cả state, câu hỏi và lựa chọn, như cách bench dựng bản nodiac và như tiền lệ KC.
   - Bỏ dấu là phép biến đổi cơ học trên chữ đã có trong train, không phải chữ mới (Fable P4: "Chuỗi VI mới? Không").
   - D2 áp nghiêm cho khối ViANLI và C2(b): mọi chuỗi phải thuộc tập chuỗi của dev.
   - Với C2(c): đếm và báo số chuỗi không dấu mới. Chuỗi nhãn kiểm chứng và chuỗi NEI phải trùng bản nodiac của dev.
6. **D1 khi có dòng bỏ dấu:** cặp chuỗi mà một chuỗi đúng bằng `strip(chuỗi kia)` thì đếm riêng thành "cặp có dấu / không dấu", không tính là vi phạm. Các cặp gần trùng khác không được nhiều hơn v3.

## 3. Bước C — dựng (chỉ sau "DỰNG 2b′")

### C1. Khối ViANLI khung kiểm chứng (P2)

- **Cỡ:** khoảng 1.500 dòng, **≤ 1.580** (trần 5% ở mức tác vụ).
- **Ánh xạ nhãn** (như 2d): entailment → Xác nhận; contradiction → Bác bỏ; neutral → NEI.
- **Dòng choice:** khoảng 55% số dòng, cơ cấu **Bác bỏ 40% / Xác nhận 33% / NEI 27%** (tức 600/500/400 trên 1.500). **Giữ phần neutral**: Fable đã mô phỏng, bỏ phần này thì NEI ở dải trùng từ cao tụt.
- **Dòng noul:** khoảng 45% số dòng. Lấy mẫu theo nhãn sao cho **toàn khối kiểm chứng gộp** đạt "Có" 37–40%, như KC.
- **Chia choice/noul** theo seed từng dòng.
- **Câu hỏi và lựa chọn:** chép nguyên từ dòng dev cùng khung (tiền tố choice; tiền tố noul; Có/Không; 3 nhãn có đuôi "để kết luận"), giống KC.
- **State:**
  - gồm tiền đề thật và **2 tiền đề của cặp khác**;
  - vị trí tiền đề thật **ngẫu nhiên** trong 3 vị trí, seed cố định; báo phân bố vị trí theo nhãn;
  - tiền đề ghép thêm không có tên riêng hay số nào chung với giả thuyết;
  - nối các tiền đề bằng dấu cách, theo định dạng state của dev ViWikiFC.
- **Một tiền đề có nhiều giả thuyết:** cho phép, nhưng báo số tiền đề dùng ≥ 2 lần và nhãn của chúng.
- **Val:** giữ 5% số tiền đề (≤ 80 dòng) làm `val_vianli.jsonl`. Không tiền đề nào vừa ở train vừa ở val, kể cả vai tiền đề ghép thêm.
- **Manifest từng dòng:**
  - `label_source=human (ViANLI)`, `text_source=ViANLI (PhoLaya)`, `license=research-only, nội bộ`;
  - id cặp gốc, vị trí tiền đề thật, id hai tiền đề ghép, seed.

### C2. Không dấu: bỏ dấu tại chỗ (P4)

- **Bỏ dấu tại chỗ**, tức thay chính dòng đó. **Không tạo bản sao**, không tạo cặp có dấu / không dấu của cùng nhận định. Tổng số dòng không đổi vì bước này.
- **Phạm vi:**
  - (a) **20% dòng của khối ViANLI;**
  - (b) **khoảng 300 dòng 2d** (vitaminc, nli_fever) có dấu, ưu tiên dòng Xác nhận và Bác bỏ có max_ov < 0,7;
  - (c) **khoảng 10% dòng VI ngoài khung kiểm chứng** (đọc hiểu 2f, ý định, chủ đề), để văn bản không dấu không chỉ là kiểm chứng;
  - (d) giữ nguyên 587 dòng KC không dấu đã có.
- **Cách chọn:**
  - chọn theo **hash của id**, không theo nhãn;
  - trong (b), chọn trong nhóm đủ điều kiện, cũng theo hash.
  - Báo phân bố nhãn của phần đã bỏ dấu, đặt cạnh phần có dấu.
- **Hàm bỏ dấu:** đúng hàm mà `build_bench_v1.py` dùng. Chuỗi nhãn và chuỗi câu hỏi không dấu phải trùng chuỗi nodiac của dev (assert như KC).

### C3. Gộp

- `train_2b_prime.jsonl` = `train_2b_v3.jsonl` (kiểm sha `ea604a01…`) sau C2, cộng khối ViANLI. Xáo theo seed.
- Báo số dòng theo khối, và tỉ lệ VI / EN, so với v3.

## 4. Khử trùng

- **Đơn vị:** mọi tiền đề được dùng, cả tiền đề thật lẫn tiền đề ghép thêm, và mọi giả thuyết.
- **Cách làm:** `scripts/decontam.py`, cả hai chiều.
- **Tham chiếu:** như mục 4 của brief dữ liệu 2b:
  - dev 1.3; test 1.3 (chỉ dò, không in);
  - `data/proxy_v1/`; mọi file probe;
  - **mọi split** của ViWikiFC và ViFactCheck;
  - Belebele, FLORES-101, SIB-200.
- Tiền đề ViANLI là câu báo, nên phải đặc biệt so với văn bản ViFactCheck.
- **Một câu chạm thì bỏ mọi cặp dùng tiền đề đó.** Báo số bị bỏ theo tham chiếu.

## 5. Cổng kiểm dữ liệu (chốt trước)

Tính trên **khối kiểm chứng VI gộp** (KC + 2d + 2e + ViANLI), trừ khi ghi khác.

| Mã | Kiểm | Đạt khi |
|---|---|---|
| D1 | `scripts/check_label_strings.py` trên `train_2b_prime.jsonl` | 0 chuỗi gần trùng mà lệch ở nhãn kiểm chứng, Có/Không và chuỗi NEI. Các chuỗi gần trùng khác không nhiều hơn v3 |
| D2 | Kho chuỗi | Mọi câu hỏi và lựa chọn của khối ViANLI và của các dòng bỏ dấu ở C2 thuộc tập chuỗi của dev (có dấu và nodiac). 0 chuỗi khung mới |
| D3 | Khử trùng | 0 chạm |
| D4 | Probe chỉ đọc nhận định (như D4 cũ), trên dòng choice của khối ViANLI | accuracy ≤ đa số + 10 điểm |
| D5′ | Phủ định và lượng từ theo nhãn, dòng choice có dấu | Chênh lớn nhất giữa ba nhãn ≤ (chênh cùng số đo trên dòng kiểm chứng của proxy_v1) + 2 điểm, cho cả phủ định lẫn lượng từ. Báo cạnh số của v3 |
| D7 | Phân bố nhãn | Choice VI gộp: NEI ≥ 30%. Noul gộp: "Có" 37–40% |
| **D11** | **Probe bề mặt Bác bỏ / NEI** (đặc trưng của `phan_tich_bacbo.py`; LR 5-fold), trên dòng choice có dấu | (a) accuracy trên khối gộp ≤ accuracy của cùng probe trên dòng kiểm chứng proxy_v1 + 5 điểm. (b) Chỉ báo: probe fit trên khối gộp, áp lên Bác bỏ proxy_v1, cho biết tỉ lệ bị xếp vào phía NEI; báo cạnh cùng số của v3 (KC hiện 88,5%) |
| D6, D10 | Độ trùng từ và độ phủ theo nhãn | Chỉ báo, đặt cạnh proxy_v1 và v3. Thêm bảng P(nhãn \| dải trùng từ) cho các dải [0; 0,5), [0,5; 0,7), [0,7; 0,9), ≥ 0,9 |
| D12 | Độ dài | Trung vị và p90 token của state khối ViANLI. Kiểm 0 dòng vượt trần 2.048 |

**Nếu trượt** D4, D5′ hoặc D11(a):
- Được sửa **một lần, chỉ bằng lấy mẫu lại** trong pool ViANLI, trong trần 1.580:
  - hướng tới Bác bỏ trùng từ thấp và NEI trùng từ cao;
  - cân phủ định và lượng từ giữa các nhãn.
- Không sửa chữ. Không đổi KC.
- Vẫn trượt thì **dừng và báo**, không lặp thêm.

**Mẫu đọc mù:** xuất `results/train_2b_prime/mau_doc_vianli.jsonl`.
- 30 dòng mỗi nhãn của khối ViANLI, cộng 20 dòng không dấu của C2(c);
- nhãn để ở **cột riêng** để điều phối che.

Điều phối sẽ quyết sau có cho đọc mù hay không.

## 6. Đầu ra

Tất cả trong `results/train_2b_prime/`:
- `P1_nguong_noul.json`;
- `kiem_ke_vianli.json`, `mau_vianli_bacbo.jsonl`;
- `vianli_khung.jsonl`, `val_vianli.jsonl`, `train_2b_prime.jsonl`;
- `mau_doc_vianli.jsonl`;
- `manifest_2b_prime.json`: sha đầu vào, seed, số dòng theo khối và nguồn, số dòng bị bỏ theo lý do, danh sách id đã bỏ dấu;
- `REPORT_2b_prime_du_lieu.md`:
  - kết quả bước A;
  - kiểm kê bước B;
  - bảng thành phần v3 so với 2b′;
  - D1–D12;
  - giới hạn.

## 7. Kết thúc

- Hai điểm báo:
  - sau bước B: gửi **"KIỂM KÊ ViANLI XONG"**, rồi chờ;
  - sau bước C: gửi **"DỮ LIỆU 2b′ XONG"** sang session `local_ad0853e1-dbf5-4579-b3e3-3a3093b693f9`, rồi **dừng**.
- Không train, không chạy GPU, không tự quyết bước tiếp.
- Nếu một bước mất > 2 giờ CPU thì báo.
