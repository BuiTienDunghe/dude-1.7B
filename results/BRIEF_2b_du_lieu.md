# BRIEF 2b: dựng dữ liệu C′ và khối kiểm chứng đúng khung bench (28/10/2026)

**Người dùng duyệt** (chat 28/10, khoảng 08:00): "được, sang 2b đi". Điều phối hiểu câu này là duyệt cả hướng 2b lẫn việc tải Wikipedia tiếng Việt nêu ở mục 2 của `results/huong_di_2710/TOM_TAT_SANG_2810.md`.

Lý do và thiết kế: `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN.md` mục 10; `results/huong_di_2710/DU_LIEU_KHONG_NHAN_fable.md` mục 2a, 4, 5; `results/audit_2710/DU_LIEU_fable.md` mục 5.

## 0. Phạm vi

- **Chỉ dùng CPU. Không chạy model nào trên GPU**, kể cả teacher. Bên Ún đang dùng GPU.
- **Không train. Không đụng test**: chỉ dò khử trùng như các lần trước, không in nội dung.
- **Chỉ tải đúng một thứ:** Wikipedia tiếng Việt `wikimedia/wikipedia`, subset `20231101.vi`, 4 file parquet, tổng 0,73 GB. Giấy phép CC BY-SA 4.0, chỉ dùng nội bộ.
  - **Không** tải FineWeb-2, HPLT hay nguồn nào khác.
  - Tổng tải mà vượt 1 GB thì dừng và báo.
- **Không thêm chữ tiếng Việt mới.** Câu hỏi, nhãn và chuỗi lựa chọn đều chép **từ `data/bench_v1_3/dev_v1_3.jsonl`**, không lấy từ `config/`. Nhận định và văn bản chỉ được trích từ Wikipedia hoặc đổi cơ học.
  - Vì vậy **không làm** hai khối "chọn câu tiếp theo" và "chọn tiêu đề" (mỗi khối cần 1 câu hỏi mới).
  - **Không làm** khối "bỏ dấu riêng phần lựa chọn": nó làm mất ý nghĩa của hàng rào Pa.
- Không sửa `config/`, `data/bench_v1_3/`, `scripts/train_lora.py`. Không xóa file, không commit.
- **Không ghi đè** `results/train_v1/components/C.jsonl`. Mọi đầu ra để ở thư mục mới `results/train_2b/`.
- Mức dùng tuần ≥ 88% thì không bắt đầu bước mới; > 90% thì dừng.

## 1. Tải Wikipedia

- Tải vào thư mục mới và trống: `results/train_v1/raw_refs/wikimedia__wikipedia_20231101.vi/`.
  - Chỉ lấy file parquet, dùng `huggingface_hub` với `allow_patterns="20231101.vi/*"`. Không chạy code từ repo dataset.
- Ghi vào manifest LFS oid của 4 file (đầu oid lúc điều phối kiểm: `ec53cc98…`, `d1e724eb…`, `f364415c…`, `0f5c15b2…`).
- Script đọc file tải về phải để **ngoài** thư mục đó và chạy bằng `python -I`.

## 2. C′: sửa C theo audit

Đầu vào: `results/train_v1/components/C.jsonl` (sha `005ca3d5…`, kiểm lại sha trước khi dùng). **Phần EN giữ nguyên.**

| # | Việc | Chi tiết |
|---|---|---|
| a | Bỏ nguồn không có văn bản | health_fact (cả choice, noul, score), `super_glue/boolq` **không** đoạn văn (giữ `boolq_passage`), `mhardalov/exams` |
| b | Bỏ VNTC | `kornwtp/VNTC-10Topics`, 1.000 dòng. Lý do: cùng câu hỏi với SIB nhưng 10 nhãn khác, hai nhãn gần trùng chữ. Phương án đổi câu hỏi cần chữ mới, nên không làm |
| c | Bỏ dòng teacher bất đồng tự tin | Từ `results/teacher_diag/quyet_C.jsonl`: `text_lang == "vi"`, có `probs`, `agree == false`, `p_max > 0.9`, nguồn **không** thuộc vsmrc/race/quail/cosmos_qa/dream/exams. Điều phối đếm ra **146 dòng**; nhiều dòng đã bị bỏ ở (a), (b). **Ngoại lệ:** giữ dòng có nhãn gốc là NEI hoặc "trung lập" (khoảng 7 dòng) |
| d | Đưa vitaminc và nli_fever về khung bench | **State** = văn bản bằng chứng hoặc tiền đề. **Câu hỏi** = tiền tố kiểm chứng của dev (choice hoặc noul, chép nguyên chuỗi từ dòng dev) + nhận định hoặc giả thuyết. **Nhãn** = 3 chuỗi dev, có đuôi "để kết luận".<br>Ánh xạ nli_fever, ghi vào manifest như một quy ước: kéo theo → xác nhận; trung lập → NEI; mâu thuẫn → bác bỏ.<br>Noul: chỉ xác nhận → "Có"; bác bỏ và NEI → "Không" (đúng như dev).<br>Dòng score: bỏ. Dòng không tách chắc được bằng chứng và nhận định: bỏ và đếm |
| e | Ghép lạc đề → NEI | Khoảng 250 dòng choice: lấy nhận định của dòng (d), ghép với văn bản của **dòng train khác** hoặc đoạn Wikipedia **khác bài** (không lấy từ dev hay probe); nhãn = NEI. Đây là phần "từ chối" kiểu decider cho kiểm chứng |
| f | Đưa câu hỏi đọc hiểu dịch lên ô câu hỏi | Theo quy tắc từng nguồn ở audit mục 5, việc 3: race tách dòng đầu; dream và quail tách dòng cuối; cosmos_qa tách câu cuối. Dòng không tách chắc được: giữ nguyên và đếm |
| g | quail: sửa chữ nhãn NEI | Đổi lựa chọn "không đủ thông tin" thành đúng chuỗi dev "Văn bản không đủ thông tin để kết luận". Hai chuỗi cùng nghĩa, và việc đổi khớp với (h). **Ghi riêng** để đọc mẫu: nếu đọc thấy không hợp ngữ cảnh thì phương án dự phòng là bỏ quail |
| h | Tăng cường "từ chối" cho đọc hiểu | Chọn 10% dòng VI đọc hiểu **có đoạn văn** và có ≥ 3 lựa chọn (vsmrc, race, dream, cosmos_qa, quail), bỏ qua dòng đã có chuỗi NEI. Seed cố định.<br>• 75%: thêm lựa chọn "Văn bản không đủ thông tin để kết luận", đáp án giữ nguyên.<br>• 25%: thay **toàn bộ** lựa chọn bằng lựa chọn của một dòng đọc hiểu **khác đoạn văn**, đáp án = chuỗi NEI.<br>Gắn cờ `aug=abstain_add` / `aug=abstain_replace`. Làm lúc dựng dữ liệu, không làm trong trainer |
| i | Sửa VSMRC | Bỏ 21 dòng "Cả B và C" (hoặc các lựa chọn tổng hợp tương tự). Bỏ dòng có đáp án dài hơn **mọi** nhiễu quá 1,5 lần. Báo số dòng bị bỏ |

Mỗi dòng bị bỏ ghi một lý do trong manifest. Ngoài các việc trên, giữ nguyên mọi dòng khác.

## 3. Khối mới KC-Wiki: kiểm chứng dựng từ Wikipedia

**Lọc bài.**
- Bài ≥ 150 chữ, ≥ 3 đoạn.
- Loại bài sơ khai do bot (loài, đơn vị hành chính…). Dùng luật đơn giản, tự chọn, ghi trong báo cáo, ví dụ: đoạn đầu sau khi che số và tên trùng mẫu với > 500 bài khác.
- Đếm token bằng tokenizer Qwen3-1.7B của dự án. Trần state **2.048** token.

**Hai dải độ dài văn bản** (state):

| Khối | Văn bản | Mô phỏng | Pool dựng | Mặc định đưa vào mix |
|---|---|---|---|---|
| KC-ngắn | 1–3 đoạn liền nhau, 80–400 token | ViWikiFC | khoảng 2.500 | 1.800 |
| KC-dài | nhiều đoạn liền nhau, 500–1.300 token, trung vị khoảng 750 | độ dài ViFactCheck (văn bản báo; ghi rõ khác thể loại ở phần giới hạn) | khoảng 1.600 | 1.200 |

Dựng pool lớn hơn mix; cỡ mix chốt ở bước phân tích trước train.

**Kiểu dựng nhận định.** Nhận định luôn là câu trích từ bài, hoặc câu trích đã đổi cơ học:

| Mã | Nhãn | Luật |
|---|---|---|
| S-nguyen | Xác nhận | Một câu nằm trong văn bản, trích nguyên |
| S-luoc | Xác nhận | Câu trong văn bản, bỏ phần trong ngoặc hoặc bỏ mệnh đề sau dấu phẩy cuối; phần còn lại ≥ 8 chữ. **≥ 1/3 số dòng Xác nhận** thuộc kiểu này |
| R-so | Bác bỏ | Câu **nằm trong văn bản**; đổi một số hoặc ngày/năm bằng giá trị cùng loại **lấy từ cùng bài** (không có trong câu đó). Dùng regex, không dùng model |
| R-ten | Bác bỏ | Như trên, đổi một tên riêng (chuỗi ≥ 2 âm tiết viết hoa, không đứng đầu câu). Chỉ làm khi bài có ≥ 2 ứng viên cùng dạng |
| N-cungbai | NEI | Câu từ phần khác của **cùng bài**, cách văn bản ≥ 2 đoạn. Câu phải chứa ≥ 1 số, ngày hoặc tên **không xuất hiện** trong văn bản, và độ trùng từ với mọi câu trong văn bản < 0,5 |
| N-lacde | NEI | Câu từ **bài khác** |

- **Không dùng phủ định** để tạo Bác bỏ.
- Kiểu lược (bỏ ngoặc, bỏ mệnh đề) áp với cùng tỉ lệ cho cả ba nhãn, để hình dạng câu không lộ nhãn.
- Trung vị độ dài (token) của ba nhãn lệch nhau ≤ 20%.

**Tỉ lệ nhãn và khung câu hỏi.**
- **Dòng choice:**
  - Xác nhận 30%, Bác bỏ 30%, NEI 40%.
  - Trong NEI: cùng bài khoảng 85%, lạc đề khoảng 15%. Phần lạc đề đã có thêm ở mục 2e, nên NEI cùng bài phải chiếm phần lớn.
- **Dòng noul:** lấy mẫu theo nhãn sao cho "Có" chiếm khoảng 37–40%, gần dev (62,9% "Không").
- Chia choice/noul theo seed từng dòng, khoảng 55/45, như `build_proxy_v1.py`.
- Câu hỏi và lựa chọn chép nguyên từ dòng dev cùng khung: tiền tố choice, tiền tố noul "Văn bản có xác nhận phát biểu sau không? Phát biểu: …", Có/Không, 3 nhãn. **State theo đúng định dạng state của dev ViWikiFC/ViFactCheck.**

**Mục tiêu toàn khối kiểm chứng VI choice** (C′ mục 2d + 2e + KC): NEI ≥ 30%, nhắm 35%. Nếu thiếu thì tăng NEI trong KC tới tối đa 45%, hoặc giảm bớt dòng không phải NEI của vitaminc/nli_fever. Báo con số cuối cùng.

**Bỏ dấu.**
- 20% dòng KC (không phải bản sao, mà là chính dòng đó) chuyển sang không dấu toàn bộ: state, câu hỏi, lựa chọn. Gắn cờ `nodiac`.
- Dùng đúng hàm bỏ dấu mà `build_bench_v1.py` dùng. Chuỗi nhãn không dấu phải trùng chuỗi nodiac của dev.

**Val.**
- Giữ riêng 5% số bài Wikipedia làm `val_kc.jsonl` (≤ 300 dòng). Không bài nào vừa ở train vừa ở val.

**Manifest từng dòng.** Ghi: `label_source=program`, `text_source=wikipedia_vi_20231101`, `license=CC BY-SA 4.0`, mã kiểu dựng, tiêu đề bài, seed.

## 4. Khử trùng (mức bài)

Dùng cách của bước 0: `scripts/decontam.py` (trùng chính xác, MinHash ≥ 0,8, chung 13-gram, câu 4–12 chữ nằm trọn), **cả hai chiều**. Tham chiếu gồm:
- dev 1.3; test 1.3 (chỉ dò, không in);
- `data/proxy_v1/`; mọi file probe ở `results/probe_v1/*.jsonl`;
- **mọi split** của ViWikiFC (`results/train_v1/raw_refs/High-Will__ViWikiFC/`) và ViFactCheck (`results/train_v1/raw_refs/tranthaihoa__vifactcheck/`);
- Belebele, FLORES-101, SIB-200 trong `raw_refs`.

**Một câu chạm thì bỏ cả bài**, vì ViWikiFC được dựng từ chính Wikipedia tiếng Việt. Báo số bài bị bỏ theo lý do và theo tham chiếu.

Khối C′ mục 2e (ghép lạc đề) và 2h (từ chối) cũng phải qua khử trùng nếu dùng văn bản Wikipedia.

## 5. Cổng kiểm dữ liệu (chốt trước)

| Mã | Kiểm | Đạt khi |
|---|---|---|
| D1 | `scripts/check_label_strings.py` trên C′ + KC | **0 chuỗi gần trùng mà lệch** ở nhãn kiểm chứng, Có/Không và chuỗi NEI. Các chuỗi gần trùng khác (ví dụ ý định du lịch) liệt kê lại, chỉ được bớt, không được thêm so với C |
| D2 | Kho chuỗi | Mọi câu hỏi và lựa chọn của KC, mục 2d, 2e, 2h thuộc tập chuỗi của dev (VI có dấu và nodiac). Ngoại lệ: lựa chọn đọc hiểu gốc ở 2h. **0 chuỗi khung mới** |
| D3 | Khử trùng | 0 chạm |
| D4 | Probe chỉ đọc nhận định | Logistic regression trên n-gram ký tự 2–4 và từ 1–2, 5-fold, chỉ đọc nhận định, trên dòng choice KC: accuracy ≤ đa số + 10 điểm. Không đạt thì báo các đặc trưng nặng nhất, sửa luật dựng một lần rồi chạy lại. Vẫn không đạt thì dừng và báo |
| D5 | Thống kê theo nhãn | Bảng: độ dài token (trung vị, p90), tỉ lệ từ phủ định (không/chưa/chẳng/chả/đâu có), tỉ lệ có số hoặc ngày, tỉ lệ trích nguyên. Trung vị độ dài lệch ≤ 20%; tỉ lệ phủ định lệch ≤ 2 điểm |
| D6 | Độ trùng từ theo nhãn (chỉ báo, không chặn) | Độ trùng từ lớn nhất giữa nhận định và một câu của văn bản, phân bố theo nhãn, đặt cạnh cùng số đo trên dòng kiểm chứng của `data/proxy_v1` |
| D7 | Phân bố nhãn | Toàn khối kiểm chứng VI choice: NEI ≥ 30%. Noul "Có" 37–40% |
| D8 | Mẫu đọc mù | Xuất `results/train_2b/mau_doc.jsonl`: 30 dòng mỗi mã kiểu dựng ở mục 3, 30 dòng mục 2d (tách dịch hay VI gốc), 30 dòng 2f, 30 dòng quail 2g, 20 dòng mỗi loại 2h. Định dạng như `mau_soat_mu.jsonl`: nhãn để ở **cột riêng** cho điều phối che |

Worker chỉ báo số và áp đúng bảng trên, không tự quyết bước tiếp.

## 6. Đầu ra

Tất cả trong `results/train_2b/`:
- `C_prime.jsonl`: C sau mục 2, chưa có KC.
- `kc_pool_ngan.jsonl`, `kc_pool_dai.jsonl`, `val_kc.jsonl`.
- `train_2b_mac_dinh.jsonl` = C′ + KC theo mix mặc định, đã xáo theo seed. Chỉ là bản dựng sẵn; mix cuối chốt ở phân tích trước train.
- `manifest_2b.json`: sha đầu vào, LFS oid Wikipedia, seed, số dòng theo khối và theo nguồn, số dòng bị bỏ theo lý do, thời gian.
- `REPORT_2b_du_lieu.md`:
  - bảng thành phần C so với C′ so với train_2b, cho VI và EN;
  - phân bố số lựa chọn và độ dài state so với dev (chỉ số tổng hợp);
  - kết quả D1–D8;
  - giới hạn.
- Script để ở `scripts/train_2b/`.

## 7. Thước đo sau train (ghi trước, trước khi có dữ liệu)

Ngưỡng số cụ thể sẽ chốt ở bước phân tích trước train, có Fable phản biện, **trước** lần train đầu tiên. Thước đo đã cố định:

- **Chính:**
  - P4 (văn bản lạc đề → NEI);
  - recall NEI trên ViFactCheck và ViWikiFC dev (C hiện 0,21 / 0,44);
  - recall Xác nhận trên hai nguồn đó, để bắt lỗi đoán NEI quá tay;
  - VI macro.
- **Hàng rào:**
  - Pa, P5 (khối mới dùng đúng khung bench nên cần hai hàng rào này);
  - không dấu;
  - EN;
  - SIB-200 (vì bỏ VNTC);
  - Belebele (vì sửa đọc hiểu).
- **Phụ:** proxy_v1, phần kiểm chứng.

## 7b. Điều phối chốt thêm (28/10, sau câu hỏi của worker)

1. **VSMRC:**
   - Bỏ cả **138 dòng** có lựa chọn tổng hợp: 21 dòng trỏ theo chữ cái và 117 dòng trỏ theo vị trí ("Tất cả các đáp án trên"…). Lý do: trainer xáo lựa chọn nên các lựa chọn này mất nghĩa.
   - Luật độ dài đo **bằng ký tự**.
   - Báo tỉ lệ "đáp án là lựa chọn dài nhất" trước và sau khi lọc, đặt cạnh Belebele dev (22%).
2. **Tăng cường từ chối (2h): sửa tại chỗ**, không thêm bản sao, giống decider.
3. **D2:** ngoại lệ áp cho cả câu hỏi lẫn lựa chọn gốc của dòng 2h. Chuỗi NEI thêm vào phải đúng chuỗi dev.
4. **Đồng ý các cách hiểu còn lại:**
   - health_fact và boolq chỉ bỏ ở phần VI;
   - 2d giữ nguyên loại câu (choice/noul);
   - dòng packed và dòng noul không suy chắc được: bỏ và đếm;
   - state KC nối đoạn bằng dấu cách, khớp dev;
   - D4 dùng sklearn của Python Windows.

## 7c. Vòng sửa KC (28/10, sau báo cáo D6; chỉ một vòng)

**Lý do:**
- D6 cho thấy độ trùng từ tách hẳn ba nhãn KC: trung vị Xác nhận 1,00, Bác bỏ 0,94, NEI 0,23. Có 90,5% nhận định Xác nhận là đoạn liền nằm trong văn bản.
- Như vậy KC chỉ dạy NEI dễ, trái với giả thuyết của 2b (C yếu ở NEI cùng chủ đề).

**Đã kiểm quy ước** (điều phối, trên split **train** của ViWikiFC, không dùng dev/test):

| Nhãn | Trùng từ lớn nhất với 1 câu (trung vị) | Độ phủ token (trung vị) |
|---|---|---|
| Xác nhận | 0,70 | 0,79 |
| Bác bỏ | 0,71 | 0,80 |
| NEI | 0,50 | 0,67 |

- 37% NEI có độ trùng ≥ 0,6.
- Đọc 10 dòng NEI có độ trùng cao: đều là kiểu "câu bằng chứng + một chi tiết không kiểm được" (thêm "vào năm 1992", "vào cuối năm 1945", một nguyên nhân…).
- Kết luận: quy ước FEVER "đúng một phần + chi tiết không kiểm được = NEI" khớp ViWikiFC.

**Kiểu dựng mới** (đều cơ học, giữ luật "không phủ định", khử trùng như mục 4):

| Mã | Nhãn | Luật |
|---|---|---|
| N-bobang | NEI | Lấy câu s trong cửa sổ văn bản làm nhận định, rồi **xóa s khỏi state**. s phải chứa ≥ 1 số, ngày hoặc tên **không còn** ở phần state còn lại. Bỏ qua nếu câu ngay sau s mở đầu bằng từ nối chỉ ngược ("Năm đó", "Cùng năm", "Sau đó", "Khi đó", "Lúc đó", "Điều này", "Việc này", "Đây", "Ông", "Bà", "Anh", "Chị", "Họ", "Nó"), để văn bản không bị gãy |
| N-them | NEI | Câu s_p nằm trong state, chèn trước dấu chấm cuối **một cụm chép nguyên** từ câu ở xa của cùng bài: cụm thời gian ("vào năm …", "năm …", "vào ngày …", "tháng …") hoặc nơi chốn ("tại/ở" + tên riêng). Giá trị trong cụm không có trong state. Chỉ chèn khi s_p **chưa có** cụm cùng loại; nếu đã có thì phải là Bác bỏ hoặc mơ hồ, nên không chèn |
| N-ghep | NEI | Mệnh đề của s_p + " và " + mệnh đề của một câu ở xa cùng bài; câu ở xa chứa số, ngày hoặc tên không có trong state |
| S-ghep | Xác nhận | Mệnh đề của hai câu **trong state** nối bằng " và " |
| R-ghep | Bác bỏ | S-ghep có đổi một giá trị theo luật R-so / R-ten |

**Từ duy nhất được chèn** là "và" ở các kiểu ghép. Ghi vào manifest. Ngoài ra không chèn từ nào.

**Biến thể định dạng.** Giữ nguyên nghĩa, áp cùng xác suất (khoảng 25%) cho cả ba nhãn:
- "ngày 2 tháng 9 năm 1945" ↔ "2/9/1945";
- "tháng 9 năm 1945" ↔ "9/1945";
- "1.000" ↔ "1000";
- "thế kỷ XX" ↔ "thế kỷ 20".

Chỉ đổi chữ số, dấu và số La Mã, **không đổi số sang chữ**. Mục đích: nhận định Xác nhận có token không nằm nguyên trong văn bản, vì lý do đúng.

**Tỉ lệ kiểu trong KC (đích):**
- Xác nhận: nguyên 35%, lược 30%, ghép 35%.
- Bác bỏ: chia theo hình dạng nguyên/lược/ghép giống Xác nhận.
- NEI: bỏ bằng 30%, thêm 25%, ghép 15%, cùng bài (cũ) 15%, lạc đề 15%.
- Giữ tỉ lệ nhãn choice 30/30/40 và noul "Có" 37–40%.

**Cổng bổ sung, chốt trước khi dựng lại** (các ngưỡng đặt theo số tổng hợp của proxy_v1 và ViWikiFC train; đây là liên hệ nhẹ với tập chọn model, ghi ở phần giới hạn):

| Mã | Đạt khi |
|---|---|
| D6′ | Tính trên dòng KC có dấu: trung vị trùng từ lớn nhất Xác nhận ≤ 0,85; NEI ≥ 0,40; Xác nhận − NEI ≤ 0,35. NEI có trùng ≥ 0,6 chiếm ≥ 25%. Xác nhận là đoạn liền nằm trong state ≤ 50% |
| D9 | Cân hình dạng: theo từng nhãn, tỉ lệ "ghép" và tỉ lệ "có biến thể định dạng" lệch nhau ≤ 5 điểm. N-bobang và N-them tính là dạng thường, cân với S-nguyên/S-lược |
| D10 (chỉ báo) | Độ phủ token theo nhãn, đặt cạnh ViWikiFC train (0,79 / 0,80 / 0,67). Probe chỉ dùng đặc trưng trùng từ (trùng lớn nhất, độ phủ; LR 5-fold): accuracy trên KC so với trên dòng kiểm chứng của proxy_v1 |

**Sửa cổng trước khi dựng** (28/10, khoảng 09:40). Worker chỉ ra hai chỗ điều phối tính sai. Đây là lỗi tính toán của điều phối, chưa có số liệu KC v2 nào.

1. **D6′:** phép đo là "bao phủ nhận định bởi câu tốt nhất". Với phép đo này, S-nguyên và S-lược luôn bằng 1,00, nên với tỉ lệ đích 65% nguyên + lược thì trung vị Xác nhận chắc chắn là 1,00. Vế "trung vị Xác nhận ≤ 0,85" và vế "chênh ≤ 0,35" không thể đạt.
   - Nâng ghép lên ≥ 55% cũng không được, vì phần lớn nhận định sẽ thành câu nối bằng "và", một dấu vết dựng còn tệ hơn.
   - **D6′ sửa thành:**
     - NEI: trung vị ≥ 0,40, và tỉ lệ dòng ≥ 0,6 phải ≥ 25%;
     - Xác nhận: tỉ lệ dòng có bao phủ < 0,9 phải ≥ 30%;
     - Bác bỏ: tỉ lệ dòng có bao phủ < 0,9 phải ≥ 30%;
     - Xác nhận là đoạn liền trong state ≤ 50%.
   - Trung vị và mức chênh của ba nhãn chỉ báo, đặt cạnh ViWikiFC train (0,70 / 0,71 / 0,50) và proxy_v1.
   - Mục tiêu của cổng: độ trùng từ **không còn tách sạch** ba nhãn. Không đặt mục tiêu khớp phân bố thật, vì không có diễn đạt lại.
2. **D9:** NEI chỉ có 15% ghép, trong khi Xác nhận và Bác bỏ có 35%. Theo đề xuất của worker, dựng khoảng 2/3 số dòng N-cungbai và N-lacde ở dạng ghép: hai câu ở xa cùng bài, hoặc hai câu của bài khác, nối bằng " và ". Như vậy NEI ghép khoảng 35%. Tỉ lệ kiểu NEI giữ nguyên.

- Chạy lại D3, D4, D5, D7 trên KC mới.
- Xuất `results/train_2b/mau_doc_v2.jsonl`: 30 dòng mỗi kiểu mới, 30 dòng có biến thể định dạng; nhãn để ở cột riêng.

**Giới hạn còn lại:** dù đã sửa, độ phủ token của Xác nhận vẫn gần 1, vì không có diễn đạt lại. Đây là đầu vào cho quyết định tỉ lệ KC so với 2d (dòng dịch có diễn đạt lại thật) ở bước phân tích trước train.

**Đọc mù mẫu vòng 1** (310 dòng, 3 agent Opus, chia xen kẽ theo số dòng mod 3, đã che `nhan*`). Cách tính, chốt trước khi mở che:
- Một dòng tính là **lỗi rõ** khi người đọc chọn khác nhãn, độ chắc "cao", và không gắn cờ.
- Theo từng mã kiểu dựng (30 dòng): ≤ 2 lỗi rõ thì đạt; 3–4 thì điều phối xem từng dòng; ≥ 5 thì sửa luật hoặc bỏ kiểu đó.
- Với 2h (20 dòng mỗi loại): ≤ 1 thì đạt; 2–3 thì xem; ≥ 4 thì sửa.
- Tỉ lệ cờ `loi_cau` của từng kiểu được báo riêng.
- Kết quả cho R-ten và S-lược (nếu phải sửa) được gộp vào cùng lần dựng lại này.

**Đọc mù vòng 2** (`mau_doc_v2.jsonl`, 220 dòng, 3 agent Opus, cùng quy ước và cách che như vòng 1). Luật chốt trước khi mở che:
- **Nhóm theo mã kiểu.** Bác bỏ ở dạng ghép tính chung là R-ghép. Dòng có biến thể định dạng báo thêm thành một nhóm chồng lên, khoảng 46 dòng.
- Mỗi nhóm n dòng:
  - lỗi rõ ≤ 7% của n (làm tròn xuống, tối thiểu 1) thì **đạt**;
  - ≤ 15% thì điều phối xem từng dòng;
  - > 15% thì bỏ kiểu đó khỏi mix, hoặc sửa luật.
- Nhóm có n < 10 (S-nguyên, R-số, N-lạc đề) chỉ báo.

**Sai lệch có ngày (28/10, sau phản biện của Fable, trước khi train).** Cơ cấu kiểu NEI trong mix KC đổi từ 30/25/15/15/15 sang **bỏ bằng 0,31 / thêm 0,29 / ghép 0,20 / cùng bài 0,10 / lạc đề 0,10**, bằng cách tái lấy mẫu trong pool v2, có cân phủ định và biến thể định dạng. Lý do và luật: `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md` mục 9.7.

## 8. Kết thúc

- Xong thì gửi **"DỮ LIỆU 2b XONG"** sang session `local_ad0853e1-dbf5-4579-b3e3-3a3093b693f9`, rồi **dừng**.
- Không train, không chạy GPU, không sang việc khác.
- Thời gian ước lượng: vài giờ CPU. Nếu một bước mất > 2 giờ thì báo.
