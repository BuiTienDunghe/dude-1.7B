# Dịch thử 200 dòng phần B bằng subagent Sonnet

Ngày 2026-10-19. Người dùng duyệt việc này (điều phối giao). Không train, không commit, không xóa file, không đụng `data/` hay `config/`. Không đọc `results/train_v1/translated/`. Chưa dịch phần còn lại.

## Tóm tắt

- **Bản sau đọc lại** (`ban_s2_200.jsonl`) qua `kiem_ban_dich.py`: **ĐẠT, 0 lỗi, 2 cảnh báo**. Cả 2 cảnh báo đều không phải lỗi:
  - "11" là lỗi gõ "I'11" của bản gốc;
  - một lựa chọn ngắn hơn bản gốc vì tiếng Việt gọn hơn.
- **Sonnet 2 sửa 17/200 dòng (8,5%), 23 chỗ.**

  | Loại lỗi | Số chỗ |
  |---|---|
  | đổi nghĩa | 9 |
  | tự nhiên / ngữ pháp | 6 |
  | tên riêng, số | 5 |
  | thêm / bớt nội dung | 3 |
  | quan hệ lựa chọn, thứ tự lựa chọn, thuật ngữ, tiền tố | 0 |

  - Dòng ý định không phải sửa dòng nào.
  - Không chỗ sửa nào làm đổi đáp án.
- **Mức dùng cho 200 dòng:** cửa sổ 5 giờ tăng **0% → 3%**; hạn mức tuần vẫn **31%** (tăng dưới 1 điểm). Số này gồm cả phần điều phối của phiên này (Opus).
- **Thời gian:** 15:51:18 → 16:02:09, khoảng **11 phút**, chạy 2–3 subagent song song.
- **Ngoại suy cho đủ 6.944 dòng:** khoảng **120 điểm % của cửa sổ 5 giờ** (tức phải chia ít nhất 2 cửa sổ, vì dừng ở 88%), **dưới 40 điểm % tuần** (cận trên, do số tuần làm tròn), khoảng **7–8 giờ** nếu giữ mức song song như thử. Chi tiết ở mục 5.

## 1. Cách làm

1. **Mẫu:** `scripts/train_v1/pilot_sonnet_prep.py`, seed 20261019.
   - 40 dòng mỗi dạng bài (đọc hiểu, kiểm chứng, ý định, nhân quả, yes/no; dạng bài lấy từ `part_b_meta.jsonl`).
   - Xáo rồi chia 5 lô 40 dòng.
   - Danh sách id ở `ids.json`; 200 dòng nguồn ở `src_200.jsonl`; các lô ở `lo/lo_{k}_en.jsonl`.
   - Người dịch chỉ nhận id/state/question/options, không nhãn, không target.
2. **Bảng thuật ngữ** (`bang_thuat_ngu.json`, 56 chuỗi, một subagent Sonnet soạn).
   - Gồm: câu hỏi mẫu, nhãn cố định, nhãn dạng mã có trong mẫu; danh sách đầu vào ở `thuat_ngu_dau_vao.json`.
   - "None of the other options ." → "Không lựa chọn nào khác đúng."; "None of the above choices ." → "Không lựa chọn nào ở trên đúng.".
   - Nhãn trích trong ngoặc kép bên trong câu hỏi dùng đúng bản dịch của nhãn đó.
   - Ví dụ:
     - SUPPORTS / REFUTES / NOT ENOUGH INFO → "ỦNG HỘ" / "BÁC BỎ" / "KHÔNG ĐỦ THÔNG TIN";
     - entailment / neutral / contradiction → "kéo theo" / "trung lập" / "mâu thuẫn";
     - checkclaimstatus → "hỏi tình trạng yêu cầu bồi thường".
3. **Sonnet 1 dịch:** 5 subagent, mỗi lô một subagent, theo `HUONG_DAN_DICH.md` và bảng thuật ngữ. Mỗi subagent tự chạy `kiem_ban_dich.py` trên lô của mình; cả 5 lô đều ĐẠT. Bản trước khi đọc lại: `lo/lo_{k}_vi_s1.jsonl`, gộp ở `ban_s1_200.jsonl`.
4. **Sonnet 2 đọc lại độc lập:** 5 subagent khác, nhận bản EN + bản VI, không nhãn. Đối chiếu từng trường:
   - nghĩa;
   - quan hệ văn bản ↔ câu hỏi ↔ lựa chọn;
   - tên riêng, số, tiền tố, thứ tự lựa chọn;
   - bảng thuật ngữ;
   - độ tự nhiên.

   Bản sau sửa: `lo/lo_{k}_vi_s2.jsonl`, gộp ở `ban_s2_200.jsonl`. Nhật ký sửa: `lo/lo_{k}_sua.jsonl`, gộp ở `sua_200.jsonl`.
5. **Ghép và kiểm:** `scripts/train_v1/pilot_sonnet_merge.py`. Script so trực tiếp s1 với s2 để đếm dòng sửa thật: 17 dòng khác nhau = 17 dòng có nhật ký, không sót.
6. **Mức dùng** đo bằng `get_usage` trước khi bắt đầu, trước mỗi lô và sau khi xong (`muc_dung.jsonl`). Token từng subagent ở `token_subagent.jsonl`.

Chọn mốc bắt đầu sau khi cửa sổ 5 giờ reset (15:50), để số đo 5 giờ bắt đầu từ 0%.

## 2. Kết quả kiểm định dạng (`kiem_ban_dich.py`, 200 dòng)

| Bản | Kết luận | Lỗi | Cảnh báo |
|---|---|---|---|
| Sonnet 1 (trước đọc lại) | ĐẠT | 0 | 2 |
| Sonnet 2 (sau đọc lại) | ĐẠT | 0 | 2 |

Hai cảnh báo:
- `race-middle…:17446`: "state thiếu số 11". Bản gốc gõ "I'11" thay cho "I'll", không mất số thật.
- `cosmos-qa…:8895`: lựa chọn 3 ngắn hơn (tỉ lệ ký tự 0,48). "I 'm afraid of the smell of gas ." → "Tôi sợ mùi gas .", đúng nghĩa.

## 3. Sonnet 2 sửa gì

| Dạng bài | Dòng (mẫu) | Dòng sửa | Đổi nghĩa | Tự nhiên / ngữ pháp | Tên riêng, số | Thêm / bớt |
|---|---|---|---|---|---|---|
| đọc hiểu | 40 | 6 | 3 | 4 | 1 | 1 |
| kiểm chứng | 40 | 4 | 3 | 1 | 3 | 0 |
| yes/no | 40 | 4 | 2 | 0 | 1 | 1 |
| nhân quả | 40 | 3 | 1 | 1 | 0 | 1 |
| ý định | 40 | 0 | 0 | 0 | 0 | 0 |
| **Tổng** | **200** | **17** | **9** | **6** | **5** | **3** |

- Số trong các cột lỗi là số chỗ sửa (23 chỗ), không phải số dòng.
- 16 chỗ ở state, 3 chỗ ở lựa chọn, 0 chỗ ở câu hỏi.

**10 ví dụ tiêu biểu:**

1. **Đổi nghĩa, kiểm chứng** (`health_fact…:22132`): "socialized medicine" bị dịch "y tế xã hội hóa". Tiếng Việt hiểu cụm này là huy động tư nhân, gần như ngược nghĩa; sửa thành "y tế do nhà nước điều hành".
2. **Đổi nghĩa, nhân quả, ở lựa chọn** (`e-CARE…:3626`): "annual" bị dịch "cây một năm tuổi"; sửa thành "cây hằng năm".
3. **Đổi nghĩa, đọc hiểu, ở lựa chọn** (`dream…:1449`): "practically the only one" bị dịch "thực tế là"; sửa thành "gần như là người duy nhất".
4. **Đổi nghĩa, đọc hiểu, ở lựa chọn** (`race-high…:29779`): "getting him a cart" bị dịch "làm cho nó một chiếc xe đẩy"; sửa thành "sắm cho nó…". Trong bài, bác sĩ đặt xe từ một công ty.
5. **Đổi nghĩa, đọc hiểu** (`cosmos_qa…:18508`): "slimy" theo nghĩa bóng (khớp "shady character" ở câu hỏi) bị dịch "trơn tuột"; sửa thành "lươn lẹo".
6. **Đổi nghĩa, kiểm chứng** (`vitaminc…:9214`): "to win Group C" bị dịch "thắng ở bảng C"; sửa thành "đứng đầu bảng C". Cùng dòng: "Equatorial Guinea", "South Africa" để nguyên tiếng Anh; sửa thành "Guinea Xích Đạo", "Nam Phi".
7. **Đổi nghĩa, kiểm chứng (NLI)** (`nli_fever…:29974`): "for which" chỉ chung các vai/phim, bản dịch thu hẹp thành một bộ phim.
8. **Đổi nghĩa, yes/no** (`sharc…:14677`): "lifetime allowance" bị dịch "mức trợ cấp trọn đời"; sửa thành "hạn mức trọn đời", ở cả hai chỗ trong văn bản.
9. **Thêm nội dung, nhân quả** (`headline_cause…:3206`): bản dịch thêm "viên ngọc" mà bản gốc không có.
10. **Ngữ pháp, nhân quả** (`e-CARE…:6266`): "Điều gì đã xảy ra như một kết quả?" dịch từng chữ; sửa thành "Kết quả của việc này là gì?".

**Nhận xét (của tôi, khi đọc nhật ký sửa):**
- **Lỗi Sonnet 2 bỏ sót:** `e-CARE…:14657`, "maternal woman" dịch thành "người phụ nữ giàu tình mẫu tử". Câu e-CARE do người không bản ngữ viết; nghĩa gần đúng hơn là sản phụ hoặc phụ nữ mang thai. Sonnet 2 không sửa. Vậy một lượt đọc lại không bắt hết lỗi đổi nghĩa.
- **Có chỗ sửa làm câu vụng hơn:** "không quá quá đáng" → "không quá đáng phẫn nộ" (`health_fact…:7535`).
- **Nên thêm vào bảng thuật ngữ khi dịch đầy đủ:**
  - câu hỏi kèm trong state của e-CARE ("What was the cause of this?" / "What happened as a result?");
  - tiền tố "First text:" / "Second text:";
  - tên nước phổ biến.
- **Nhãn kiểm chứng "ỦNG HỘ / BÁC BỎ":** dịch sát chữ. Bench tiếng Việt dùng "Được văn bản xác nhận / Bị văn bản bác bỏ" (người dùng đã duyệt). Nên cân nhắc thống nhất trước khi dịch đầy đủ; việc này do người dùng quyết.

## 4. Token (ước tính)

Số token subagent lấy từ báo cáo sử dụng của từng lần gọi (`token_subagent.jsonl`), gồm token vào (đọc hướng dẫn, bảng thuật ngữ, lô dữ liệu, kết quả công cụ) và token ra.

| Bước | Lần gọi | Token |
|---|---|---|
| Bảng thuật ngữ | 1 | 86.619 |
| Sonnet 1 dịch | 5 | 502.991 (trung bình 100,6 nghìn mỗi lô 40 dòng) |
| Sonnet 2 đọc lại | 5 | 512.867 (trung bình 102,6 nghìn mỗi lô) |
| **Tổng** | 11 | **1.102.477** |

- Dữ liệu 200 dòng: 108.369 ký tự tiếng Anh, 118.338 ký tự tiếng Việt.
- Phần lớn token là token vào cố định mỗi lần gọi (hướng dẫn, bảng thuật ngữ, đọc file). Lô lớn hơn sẽ giảm chi phí mỗi dòng.
- Không tính phần điều phối của phiên này (Opus); phần đó nằm trong số % mức dùng ở dưới.

## 5. Mức dùng và ngoại suy

| Mốc | Thời điểm | % tuần (mọi model) | % 5 giờ | % tuần Fable |
|---|---|---|---|---|
| Điều phối đo lúc giao việc | trước 15:37 | 31 | 9 (cửa sổ cũ) | |
| Trước khi bắt đầu (sau reset 15:50) | 15:51:18 | 31 | 0 | 15 |
| Sau khi xong | 16:02:09 | 31 | 3 | 15 |

**Ngoại suy cho đủ 6.944 dòng.** Hệ số: 6.944/200 = 34,7; dòng của file đầy đủ dài hơn mẫu 1,17 lần (631 so với 542 ký tự mỗi dòng), vì file đầy đủ có tỉ lệ đọc hiểu cao hơn.

| Chỉ số | Ước | Cách tính |
|---|---|---|
| 5 giờ | khoảng **120 điểm %** | 3 × 34,7 × 1,17 |
| Tuần | **dưới 40 điểm %** | tăng dưới 1 điểm ở mẫu, nhân 40 |
| Thời gian | khoảng **7–8 giờ** | 11 phút × 34,7 × 1,17, với 2–3 subagent song song |

- **5 giờ:** quy tắc dừng ở 88%, nên phải chia ít nhất 2 cửa sổ 5 giờ.
- **Tuần:** số tuần chỉ hiện số nguyên, nên đây là cận trên thô. Cần đo lại sau lô lớn đầu tiên. Từ 31% hiện tại, cận trên khoảng 71%, vẫn dưới ngưỡng 88%.
- **Thời gian:** song song nhiều hơn thì nhanh hơn, nhưng mức dùng 5 giờ tăng nhanh hơn.
- Ngoại suy chưa tính việc chấm so sánh với bản Google (làm sau, file `SO_SANH_sonnet_google.md`).

## 6. File

Tất cả nằm trong thư mục `results/train_v1/translated_sonnet/pilot/`.

| File | Nội dung |
|---|---|
| `ids.json`, `src_200.jsonl` | mẫu |
| `bang_thuat_ngu.json`, `thuat_ngu_dau_vao.json` | bảng thuật ngữ và đầu vào của nó |
| `ban_s1_200.jsonl` | trước đọc lại |
| `ban_s2_200.jsonl` | **sau đọc lại, bản dùng** |
| `sua_200.jsonl` | nhật ký sửa |
| `lo/` | từng lô |
| `kiem_s1.json`, `kiem_s2.json` | kết quả kiểm |
| `thong_ke_pilot.json` | thống kê |
| `muc_dung.jsonl`, `token_subagent.jsonl` | mức dùng và token |
| `doc_kiem.html` | **30 dòng cho người dùng đọc** (6 mỗi dạng bài, seed 20261017; EN và VI cạnh nhau, đáp án tô xanh có dấu ✓) |

Script nằm trong `scripts/train_v1/`: `pilot_sonnet_prep.py`, `pilot_sonnet_merge.py`, `make_review_html.py`.

**Lưu ý.** Brief (mục 3.4) từng ghi rằng điều khoản của Anthropic giới hạn việc dùng đầu ra của Claude để huấn luyện model. Người dùng đã duyệt dịch thử bằng Sonnet. Có dùng bản dịch này để train hay không, và dùng cho model nào, là quyết định của người dùng.

## 7. Chuẩn bị cho bản dịch đầy đủ (19/10, chưa dịch)

Người dùng bảo "chủ động". Đã làm phần chuẩn bị; **chưa dịch dòng nào ngoài 200 dòng thử**.

- **`to_translate/HUONG_DAN_DICH_v2.md`** (bản v1 giữ nguyên). Thêm quy tắc rút ra từ dịch thử và từ đợt so sánh:
  - tên người giữ nguyên và nhất quán trong cả dòng; tên tác phẩm, sản phẩm, chương trình, chip giữ tên gốc; tên nước có tên Việt quen dùng thì dùng tên đó;
  - chuỗi không rõ nghĩa (bala, junga, proji koo…) giữ nguyên, không đoán nghĩa; văn bản cố tình vô nghĩa giữ tính vô nghĩa;
  - danh sách tiền tố phải giữ nguyên (Turn N:, text_A:, Q:, A:…), và tiền tố được dịch cố định theo bảng;
  - tiêu đề tin viết hoa theo quy ước tiếng Việt;
  - "sister"/"brother": chọn theo ngữ cảnh, không có căn cứ thì chọn một và giữ nhất quán trong cả dòng; xưng hô nhất quán;
  - thuật ngữ trong NLI phải dịch giống nhau giữa text_A và text_B; "Possible labels" giữ đủ nhãn;
  - các ví dụ dịch sai đã gặp (socialized medicine, thriller, notch-less, free-as-in-freedom, practically…);
  - không có ký tự ẩn.
- **`to_translate/bang_thuat_ngu_v2.json`** (ghép bằng `scripts/train_v1/merge_glossary_v2.py`, kiểm ĐẠT):
  - 255 chuỗi:
    - 78 câu hỏi mẫu, lựa chọn lặp, nhãn trích trong câu hỏi, 2 câu hỏi kèm của e-CARE;
    - 177 nhãn dạng mã: banking77 69, clinc_oos 91, multidogo 10, health_fact 4, nli_fever 3.
  - 75 tiền tố: 25 giữ nguyên (tên người), 50 dịch cố định (vai trò, tiêu đề mục, "First text:" → "Văn bản thứ nhất:"; "Driving Officer:" → "Giám khảo sát hạch:" sau khi đối chiếu ngữ cảnh).
  - Hai subagent Sonnet soạn. Đầu vào do `scripts/train_v1/full_glossary_prep.py` gom từ toàn bộ 6.944 dòng; đáp án một từ của bài đọc hiểu không đưa vào bảng.
  - Đã kiểm:
    - mọi câu hỏi trích nhãn đều chứa đúng bản dịch nhãn;
    - không hai nhãn cùng nguồn trùng bản dịch;
    - NFC, không ký tự ẩn;
    - phủ đủ chuỗi lặp từ 5 lần trở lên.
  - Giữ bản pilot cho mọi mục đã có, trừ 2 câu hỏi banking77 phải đổi để khớp nhãn mới.
- **`to_translate/kiem_ban_dich.py` v2:** báo lỗi khi có ký tự ẩn, cảnh báo khi tiền tố cần giữ nguyên bị đổi. Toàn bộ phần B có 1 dòng gốc chứa ký tự ẩn; bản dịch không được giữ ký tự đó.
- **Cần người dùng quyết, hoặc nên rà trước khi dịch đầy đủ:**
  - nhãn SUPPORTS/REFUTES/NOT ENOUGH INFO: giữ "ỦNG HỘ / BÁC BỎ / KHÔNG ĐỦ THÔNG TIN" hay theo bench "Được văn bản xác nhận / Bị văn bản bác bỏ / Văn bản không đủ thông tin" (đánh dấu trong `ghi_chu` của 9 mục);
  - Sonnet tự nêu 9 nhãn clinc dịch theo tên nhãn, chưa có câu mẫu để đối chiếu: gas, taxes, travel_alert, vaccines, w2, direct_deposit, user_name, international_visa, reminder.
