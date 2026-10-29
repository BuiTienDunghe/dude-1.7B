# Hướng dẫn dịch v2: dữ liệu train_v1, phần B (Anh → Việt)

Bản v2 (19/10) thay bản v1 (`HUONG_DAN_DICH.md`, giữ lại để đối chiếu). Bổ sung rút ra từ dịch thử 200 dòng và đợt so sánh với bản Google (`results/train_v1/translated_sonnet/pilot/`). Mục mới hoặc đổi so với v1 có đánh dấu **[v2]**.

File cần dịch: `part_b_to_translate.jsonl` (cùng thư mục). Mỗi dòng là một câu hỏi trắc nghiệm dùng để **huấn luyện** model ra quyết định tiếng Việt. Dữ liệu chỉ dùng nội bộ.

## Định dạng

Mỗi dòng vào:

```json
{"id": "...", "state": "văn bản", "question": "câu hỏi", "options": ["lựa chọn 1", "lựa chọn 2", ...]}
```

Trả về **một file jsonl**, mỗi dòng đúng bốn trường đó, cùng `id`, đã dịch sang tiếng Việt:

```json
{"id": "<giữ nguyên>", "state": "<bản dịch>", "question": "<bản dịch>", "options": ["<bản dịch lựa chọn 1>", "<bản dịch lựa chọn 2>", ...]}
```

- Giữ nguyên `id`, không thêm hay bỏ dòng.
- `options` rỗng (`[]`) nghĩa là câu hỏi Có/Không: lựa chọn cố định do dự án thêm sau. Trả về `[]`.
- Mã hóa UTF-8, một đối tượng JSON trên một dòng.

## Quy tắc chung

1. **Giữ nguyên số lựa chọn và thứ tự lựa chọn.** Lựa chọn thứ k của bản dịch là bản dịch của lựa chọn thứ k của bản gốc.
2. **Không đổi nghĩa.** Đáp án đúng xác định theo bản gốc. Bản dịch phải giữ quan hệ giữa văn bản, câu hỏi và từng lựa chọn: lựa chọn sai vẫn sai, lựa chọn đúng vẫn đúng, hai lựa chọn khác nghĩa không được thành cùng nghĩa.
3. **Không thêm, không bớt thông tin.** Không giải thích, không chú thích, không trả lời câu hỏi. Không tự cụ thể hóa điều bản gốc không nói.
4. **Bảng thuật ngữ bắt buộc [v2].** File `bang_thuat_ngu_v2.json` (cùng thư mục).
   - Chuỗi nào (câu hỏi, lựa chọn, tiền tố) trùng **nguyên văn** một khóa trong bảng thì dùng đúng giá trị `vi` của khóa đó.
   - Nhãn trích trong ngoặc kép bên trong câu hỏi (ví dụ `Is "pin_blocked" the correct label…`) dùng đúng bản dịch của nhãn đó trong bảng, đặt trong ngoặc kép.
5. **Tiếng Việt có dấu chuẩn, Unicode NFC.**
   - Khoảng 65 dòng gốc chưa ở dạng NFC; bản dịch vẫn phải NFC toàn bộ.
   - **Không có ký tự ẩn** (zero-width, ký tự điều khiển) **[v2]**.
6. **Trả về đúng định dạng và id.**

## Tên riêng, số, chuỗi đặc biệt [v2]

- **Tên người** giữ nguyên chữ gốc (`Li Ming` giữ là `Li Ming`, không đổi thành "Lý Minh"). Cùng một tên phải viết giống nhau ở mọi chỗ trong dòng (văn bản, câu hỏi, lựa chọn).
- **Tên tác phẩm, sản phẩm, thương hiệu, chương trình, ban nhạc, phim, sách, trò chơi, model, chip, phiên bản** giữ nguyên tên gốc, không dịch nghĩa. Ví dụ: `Before I Fall`, `ReBoot: The Guardian Code`, `Foxcatcher`, `for King and Country`, `Payroll Giving`, `Kirin 820`, `cat genie`. Có thể thêm danh từ chung tiếng Việt phía trước nếu cần cho câu tự nhiên: "phim Foxcatcher", "chương trình Payroll Giving".
- **Tên tổ chức, chức danh chính thức** ít phổ biến: giữ tên gốc. Tổ chức có tên tiếng Việt quen dùng thì dùng tên đó (Liên Hợp Quốc, Tổ chức Y tế Thế giới).
- **Tên quốc gia và địa danh lớn** có tên tiếng Việt quen dùng thì dùng tên đó (South Africa → Nam Phi, South Korea → Hàn Quốc, Equatorial Guinea → Guinea Xích Đạo). Địa danh nhỏ hoặc không có tên Việt quen dùng thì giữ nguyên.
- **Danh hiệu, biệt danh lịch sử:**
  - Danh hiệu đã có cách gọi Việt quen thuộc thì dùng cách đó (`the Great` → Đại đế).
  - Biệt danh khác giữ nguyên chữ gốc, có thể thêm nghĩa trong ngoặc ở lần đầu: "Justinian the Strong (Justinian Mạnh mẽ)".
  - Hai biệt danh khác nhau trong cùng dòng phải vẫn khác nhau.
- **Số, ngày tháng, đơn vị, số điện thoại, mã hồ sơ** giữ nguyên chữ số. Số đọc thành chữ trong hội thoại (ví dụ "fourteen eighty") dịch thành chữ tiếng Việt, giữ đúng giá trị.
- **Chuỗi không rõ nghĩa giữ nguyên, không đoán nghĩa:** tên lạ, mã, từ nhiễu, chữ dính, lỗi gõ trong hội thoại khách hàng. Ví dụ: `bala`, `junga`, `fij`, `proji koo`, `sureok`, `higood morning`, địa chỉ email, mã số. "Hi"/"hai"/"hello" ở đầu hội thoại là lời chào, dịch "chào"/"xin chào".
- **Văn bản cố tình vô nghĩa hoặc sai** (tin nhắn rác, khẩu hiệu vô nghĩa, câu chơi chữ) phải giữ tính vô nghĩa đó. Không sửa thành câu hợp lý. Có thể dịch từng chữ hoặc giữ nguyên chữ gốc trong ngoặc kép.

## Tiền tố và định dạng [v2]

**Giữ nguyên, không dịch:**
- `Turn N:` (mọi N), `text_A:`, `text_B:`, `Passage A:`, `Passage B:`, `Item A:` … `Item D:`;
- tiền tố một chữ cái `A:`, `B:`, `M:`, `W:`, `F:`, `Q:`;
- tiền tố là tên người (`Susan:`, `Mark:`, `Anna Maria:` …);
- ký hiệu `###`, `##`, `*`, `_` (chỗ trống điền khuyết), xuống dòng.

Không dịch `Turn` thành "Lượt", không dịch `Q:` thành "Hỏi:", kể cả chỉ ở một lượt.

**Dịch theo bảng thuật ngữ, cố định ở mọi dòng:**
- `First text:` → "Văn bản thứ nhất:", `Second text:` → "Văn bản thứ hai:";
- tiền tố là vai trò hoặc tiêu đề mục (`Man:`, `Woman:`, `Dad:`, `Customer:`, `Opening hours:`, `Tel:` …): dùng đúng bản dịch trong bảng.

## Cách viết [v2]

- **Tiêu đề tin (HeadlineCause, tiêu đề bài):** viết hoa theo quy ước tiếng Việt, tức chỉ hoa chữ đầu câu và tên riêng. Không giữ kiểu viết hoa từng chữ của tiếng Anh ("Donald Trump và Fed có thể sắp…", không viết "Donald Trump Và Fed Có Thể Sắp…").
- **Chữ hoa toàn bộ của bản gốc** (nhấn mạnh trong hội thoại, ví dụ "SISTER") có thể giữ hoa ở bản dịch.
- **"sister" / "brother", "aunt" / "uncle" và các từ quan hệ không rõ thứ bậc:**
  - chọn chị/em, anh/em, cô/dì/bác… theo ngữ cảnh nếu có căn cứ;
  - không có căn cứ thì chọn một cách và **giữ nhất quán trong cả dòng**;
  - không đổi giữa chừng.
- **Xưng hô trong hội thoại:** chọn một cặp xưng hô phù hợp vai (khách hàng – nhân viên, bố – con…) và giữ nhất quán trong cả dòng.
- **Thuật ngữ chuyên ngành:** dùng nghĩa theo ngữ cảnh, không theo nghĩa đen. Ví dụ:
  - "socialized medicine" → y tế do nhà nước điều hành (không phải "y tế xã hội hóa");
  - "thriller" → phim giật gân / ly kỳ (không phải "kinh dị");
  - "notch-less display" → màn hình không tai thỏ (khác "màn hình không viền");
  - "filed for unemployment" → nộp đơn xin trợ cấp thất nghiệp;
  - "free as in freedom" → tự do (không phải "miễn phí").
- **Từ đa nghĩa:** dịch theo nghĩa của câu. Ví dụ: "practically" = gần như; "I can never tell" = không bao giờ biết chắc; "slimy" (người) = lươn lẹo; "annual" (cây) = cây hằng năm.
- **Câu hỏi của bản gốc là phỏng đoán** ("What may be the reason…") thì giữ sắc thái phỏng đoán.

## Lưu ý theo loại dữ liệu

- **Câu hỏi mẫu lặp lại và nhãn cố định:** theo bảng thuật ngữ. Các câu hỏi mẫu gần nghĩa nhau đã được dịch thành các câu khác nhau; không gộp lại.
- **Nhãn "None of …":** "None of the other options ." = "Không lựa chọn nào khác đúng."; "None of the above choices ." = "Không lựa chọn nào ở trên đúng." (đã có trong bảng). **Không** dịch thành "không có lựa chọn nào khác".
- **Nhãn ý định dạng mã** (banking77, clinc_oos, multidogo) **[v2]**:
  - dịch thành cụm từ mô tả **việc người dùng muốn làm**, theo bảng thuật ngữ;
  - ví dụ `atm_support` → hỏi máy ATM nào dùng được thẻ, không viết mơ hồ "hỏi về việc hỗ trợ máy ATM";
  - các nhãn khác nhau phải ra bản dịch khác nhau.
- **Kiểm chứng và NLI:**
  - cùng một thuật ngữ xuất hiện ở text_A và text_B phải dịch **giống nhau** (ví dụ "human condition" → "thân phận con người" ở cả hai), nếu không quan hệ kéo theo sẽ khó nhận ra;
  - danh sách "Possible labels" trong câu hỏi phải giữ **đủ** mọi nhãn, đúng thứ tự.
- **Yes/no (boolq, sharc):**
  - câu hỏi phải giữ đúng điều kiện được hỏi (thời điểm, chủ thể, phủ định);
  - không bỏ chữ trong câu hỏi.
- **Nhân quả:**
  - e-CARE: câu hỏi kèm cuối state ("What was the cause of this?" / "What happened as a result?") dịch theo bảng thuật ngữ; giữ đúng chiều nguyên nhân / kết quả. Câu e-CARE do người không bản ngữ viết: dịch theo nghĩa ngữ cảnh ("maternal woman" ≈ sản phụ / phụ nữ mang thai).
  - HeadlineCause: giữ tiền tố `A:`, `B:`; nhãn "thứ nhất / thứ hai" giữ đúng chiều.
- **Câu sai chính tả hoặc lủng củng trong bản gốc:** dịch theo nghĩa rõ nhất, không sửa nội dung. Chỗ thiếu chữ (ví dụ "Do I my notes…") thì dịch sát, không tự thêm động từ.

## Kiểm trước khi gửi

```bash
python kiem_ban_dich.py part_b_to_translate.jsonl <bản_dịch>.jsonl --report ket_qua_kiem.json
```

**Script báo lỗi (kết luận phải là `ĐẠT`) khi:**
- thiếu, thừa hoặc lặp id;
- thiếu trường hoặc sai kiểu;
- số lựa chọn lệch;
- có chuỗi rỗng;
- chuỗi chưa ở dạng NFC;
- **có ký tự ẩn [v2]**;
- còn nguyên văn tiếng Anh;
- hai lựa chọn trùng nhau.

**Script cảnh báo khi:**
- chuỗi lặp dịch không nhất quán;
- độ dài lệch bất thường;
- mất số;
- **tiền tố cần giữ nguyên bị đổi hoặc mất [v2]**.

Sau khi nhận bản dịch, dự án kiểm lại bằng chính script này, khử rò rỉ lần nữa phía tiếng Việt, cho người dùng đọc kiểm 30 dòng, rồi mới dùng để train.
