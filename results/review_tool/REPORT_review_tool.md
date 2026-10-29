# Công cụ hiệu đính và cổng duyệt tiếng Việt

Ngày 2026-10-14. Không train, không chấm, không mở test. Công cụ chỉ hiển thị dev_v1.1.

## Khởi động

Chạy tại `C:\Users\dungbt06\Fine-tune`, trên một dòng, bằng Python Windows (không cần cài gì):

```
python tools/review_app/app.py --dev data/bench_v1_1/dev_v1_1.jsonl --preds_a results/dev_v1_1/eval/decider-2b-text/preds.jsonl --preds_b results/dev_v1_1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl --review_log results/human_review/review_log.jsonl --edits_dir data/human_edits --user nguoi_lam_du_an --seed 20261013 --port 8765
```

Sau đó mở http://127.0.0.1:8765.

- Thêm `--dry_run` để chỉ in số đếm rồi thoát.
- `--user` là giá trị trường "ai" trong mọi log, không suy ra từ hệ thống.

## 1. Các màn hình

Thanh điều hướng có: Trang đầu, Soát A/B/C, Mẫu và nhãn, Từng câu, Chờ duyệt, Xuất CSV, Chốt đợt.

- **Soát A/B/C:** giữ nguyên như trước.
  - Hàng A 108 câu, B 659, C 696; seed hàng C là 20261013.
  - Phím tắt: Enter / 1–9 / M / D / ←.
  - Quyết định vẫn ghi vào `results/human_review/review_log.jsonl`, đúng định dạng cũ, thêm trường "ai". Lúc dựng công cụ chưa có file này: người dùng chưa bắt đầu soát, nên không có gì bị ghi đè. Nếu file đã có, công cụ đọc tiếp và soát tiếp đúng chỗ.
- **Mẫu và nhãn:** 78 chuỗi tiếng Việt do agent viết mà bench 1.1 dùng (danh mục ở `tools/review_app/inventory.py`).
  - Mỗi chuỗi hiện cạnh bản tiếng Anh, giá trị nhãn gốc và nơi dùng.
  - Sửa tại chỗ; ba nút Duyệt / Lưu sửa / Hoàn tác.
  - Trạng thái: chưa duyệt / đã duyệt / đã sửa.

| Nhóm | Số chuỗi | Đã duyệt | Chưa duyệt |
|---|---|---|---|
| Câu hỏi mẫu (SIB, MASSIVE, XCOPA nguyên nhân/kết quả, kiểm chứng choice/noul) | 6 | 2 | 4 |
| Tên chủ đề SIB-200 | 7 | 0 | 7 |
| Mô tả ý định MASSIVE | 60 | 8 | 52 |
| Lựa chọn kiểm chứng | 3 | 3 | 0 |
| Có/Không (lựa chọn noul) | 2 | 0 | 2 |
| **Tổng** | **78** | **13** | **65** |

  - Trạng thái ban đầu: các mục config có `"_status": "người làm dự án đã duyệt ngày 2026-10-13"` là "đã duyệt"; còn lại "chưa duyệt".
  - "Từ nguồn" được ghi rõ trên trang và không tính vào số chưa duyệt: câu hỏi và lựa chọn Belebele, lựa chọn XCOPA, mọi state, nhận định trong câu kiểm chứng.
  - Các mục ViHSD trong config không dùng ở bench 1.1 nên không liệt kê.
- **Từng câu:** tìm theo từ khóa, nguồn, id; lọc theo trạng thái (chưa xem, đã sửa, mơ hồ, lệch dịch).
  - Sửa được: đáp án đúng (áp cho cả ba biến thể cùng gốc), câu hỏi tiếng Việt, từng lựa chọn tiếng Việt, state tiếng Việt có dấu.
  - **Bắt buộc chọn lý do:** sai nhãn / lỗi dịch / chính tả / mơ hồ nên loại / khác.
  - Bản gốc và bản sửa đặt cạnh nhau; có nút hoàn tác từng trường và nút "Đánh dấu mơ hồ nên loại".
  - Khối song song hiện kèm bản tiếng Anh (không sửa ở đây).
  - Nếu câu đã có quyết định ở A/B/C thì hiện quyết định đó và có nút điền sẵn đáp án. Công cụ không tự biến quyết định soát thành chỉnh sửa: người dùng phải bấm Lưu.
- **Chờ duyệt:** đọc `config/vi_proposals.jsonl` (hiện rỗng).
  - Ba lựa chọn: duyệt / sửa rồi duyệt / từ chối.
  - Nếu đề xuất là cho một chuỗi đã có trong danh mục, duyệt sẽ áp như một lần sửa ở "Mẫu và nhãn".
- **Xuất CSV:** toàn bộ log hiệu đính, UTF-8 có BOM.
- **Chốt đợt:** ghi mốc vào log, rồi `git commit` các thư mục `config/` và `data/human_edits/`. Không tự dựng bản hiệu đính.

## 2. Lưu trữ

- **Lớp ghi đè `data/human_edits/`:**
  - `edits_log.jsonl`: chỉ nối thêm; mỗi lần ghi đều flush + `os.fsync`. Mỗi dòng có id, thời điểm, ai, loại, hành động, câu/khóa, trường, giá trị cũ, giá trị mới, lý do, ghi chú.
  - `state.json`: trạng thái hiện hành, sinh lại từ log sau mỗi lần ghi (ghi tạm rồi `os.replace`). Bản ghi mới nhất là bản có hiệu lực; hoàn tác ghi thành dòng mới.
- **Không sửa** file trong `data/bench_v1_1/` hay config. Công cụ từ chối khởi động nếu `--dev` trỏ vào một file test.
- **Đề xuất của agent:** chỉ đi qua `config/vi_proposals.jsonl` (mỗi dòng: id, khoa, gia_tri_de_xuat, ban_tieng_anh, ly_do, nguoi_de_xuat, thoi_diem). Agent không tự ghi chuỗi tiếng Việt mới vào config benchmark.

## 3. Bảo mật và kết quả thử

**Cơ chế:**
- Chỉ nghe 127.0.0.1.
- Header Host phải là `127.0.0.1:<port>` hoặc `localhost:<port>`, sai thì 403. Điều này chặn DNS rebinding.
- Mọi thao tác ghi là POST `/api/*` và phải có đủ:
  - header `X-Review-Token` khớp token ngẫu nhiên 43 ký tự (`secrets.token_urlsafe`), sinh mỗi lần khởi động và nhúng vào trang;
  - `Content-Type: application/json`;
  - Origin (nếu có) là `http://127.0.0.1:<port>` hoặc `http://localhost:<port>`.
- Không endpoint ghi nào nhận GET (trả 405).
- CSP: `default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'`, cùng `X-Frame-Options: DENY` và `nosniff`.
- Mọi văn bản dữ liệu qua `html.escape`; thông báo phía trình duyệt gán bằng `textContent`.
- Ghi dữ liệu cần JavaScript; trang vẫn đọc được khi tắt JavaScript.

**Thử trên một bản chạy riêng** (cổng 8767, config, lớp ghi đè và git repo đều là bản sao trong thư mục tạm; không đụng dữ liệu thật):

| # | Thử | Kết quả |
|---|---|---|
| 1–2 | GET với Host `127.0.0.1:8767` / `localhost:8767` | 200 / 200 |
| 3 | GET với Host `evil.example` | **403** |
| 4 | GET với Host `127.0.0.1:9999` (sai cổng) | **403** |
| 5 | POST không token | **403** |
| 6 | POST token sai | **403** |
| 7 | POST token đúng, Host sai | **403** |
| 8 | POST token đúng, Origin `http://evil.example` | **403** |
| 9 | POST `Content-Type: text/plain` (kiểu form CSRF) | **415** |
| 10 | GET tới `/api/template` | **405** |
| 11 | POST hợp lệ | 200, log thêm đúng 1 dòng |
| 12 | POST sửa câu thiếu lý do | **400** ("phải chọn lý do") |

**Thử chức năng** (gửi JSON UTF-8 như trình duyệt), đều đạt:
- Sửa mẫu có chèn `<script>…</script> & "x"`: lưu nguyên văn, trang hiện bản đã escape, chỉ còn 2 thẻ `<script` của chính công cụ.
- Sửa câu hỏi có chèn `<img src=x onerror=…>`: trang sửa câu không có thẻ `<img` thô.
- Hoàn tác trường; đánh dấu loại.
- Sửa rồi duyệt đề xuất: chuỗi chuyển thành "đã sửa". Duyệt lại đề xuất đã xử lý thì bị từ chối (400).
- Quyết định soát hàng A ghi vào review log.
- Xuất CSV có BOM.
- Chốt đợt tạo commit "Chốt đợt 1" trong git repo tạm.

## 4. PHẦN 2: script dựng (`scripts/human_edits/build_revised.py`), chỉ chạy khi người dùng yêu cầu

- **Quy tắc:**
  - Sửa ở "Mẫu và nhãn" áp cho cả dev và test, chỉ lên dòng tiếng Việt có dấu của nguồn dùng chuỗi đó; bản tiếng Anh giữ nguyên.
  - Sửa ở "Từng câu" chỉ áp cho dev. Đáp án áp cho cả ba biến thể; "mơ hồ nên loại" bỏ cả ba biến thể.
  - Bản không dấu sinh lại bằng cùng hàm; n_tokens tính lại cho dòng đổi chữ.
  - Nếu test đổi thì đóng băng bản mới `test_v1.1.<k>.jsonl` (SHA + chỉ đọc), bản cũ giữ nguyên.
  - Ghi báo cáo số câu sửa theo nguồn và theo lý do, kèm danh sách thay đổi. Báo cáo không chứa văn bản test.
- **Chạy thử với lớp ghi đè rỗng** (`--check_identity`, trên `data/human_edits` thật):
  - dev trùng từng byte với `data/bench_v1_1/dev_v1_1.jsonl` (sha `18ba27b4…`);
  - test trùng hash test 1.1 (`fbad818b…`);
  - không in nội dung test.
- **Chạy thử với lớp ghi đè có chỉnh sửa** (log của môi trường thử, ra thư mục tạm; báo cáo chép ở `build_test_report_r1.json`):
  - sửa 2 tên chủ đề SIB → đổi 300 dòng dev và 120 dòng test (dòng tiếng Việt có dấu), bản không dấu sinh lại, 0 dòng tiếng Anh đổi;
  - `massive:10457` bị loại cả ba biến thể;
  - test đổi nên script đóng băng bản thử `test_v1.1.1.jsonl` (chỉ trong thư mục tạm).
- Chấm lại dòng đổi chữ và báo điểm theo hai bộ (gốc / hiệu đính) là bước sau, khi người dùng yêu cầu.

## 5. PHẦN 3: cổng duyệt (`scripts/human_edits/release_check.py`)

- **Chạy thử:** **TỪ CHỐI** (mã thoát 2), vì còn **65 chuỗi chưa duyệt**: 4 câu hỏi mẫu, 7 tên chủ đề SIB, 52 mô tả ý định MASSIVE, 2 Có/Không. Danh sách đủ ở `release_check_run.txt`.
- Đề xuất đang chờ: 0. Đề xuất chưa vào benchmark nên được liệt kê nhưng không chặn phát hành.
- Script không tự dựng bản công bố; khi đạt nó chỉ in lệnh dựng.

## 6. Git cục bộ

- Đã `git init` ở thư mục dự án. `.gitignore` là danh sách trắng: chỉ theo dõi `.gitignore`, `config/`, `data/human_edits/`.
  - Commit đầu có 12 file (config gồm `archive_v1_0/` và `vi_proposals.jsonl` rỗng; lớp ghi đè rỗng).
  - Không có `data/bench_*`, `results/`, `.venv`, `tools/` hay `scripts/`.
- Danh tính commit chỉ đặt ở mức repo (`git config --local`): **review_app <review_app@localhost>**. Không đụng cấu hình global.
  - Đổi bằng `git config --local user.name "…"` và `git config --local user.email "…"`.
- `core.autocrlf=false` ở mức repo, để git không đổi byte của config (hash config trùng manifest bench 1.1).
- Không có remote, không push.
- Commit đầu `f141099` do agent tạo, thông điệp kết thúc bằng dòng Co-Authored-By. Từ nay mỗi lần "Chốt đợt" trên công cụ là một commit.

## 7. File

- `tools/review_app/`: `app.py` (server), `inventory.py` (danh mục chuỗi), `edits.py` (log và trạng thái).
- `scripts/human_edits/`: `build_revised.py` (phần 2), `release_check.py` (phần 3).
- `data/human_edits/`: `edits_log.jsonl`, `state.json`.
- `config/vi_proposals.jsonl`.
- `results/review_tool/`: báo cáo này, `release_check_run.txt`, `build_test_report_r1.json`.

## 8. Bổ sung 2026-10-14: đề xuất theo từng câu ở màn "Chờ duyệt"

Không train, không GPU. Không có đề xuất thật nào được ghi vào `config/vi_proposals.jsonl` (file vẫn rỗng).

**Cần khởi động lại công cụ để nạp bản mới.** Dừng công cụ đang chạy ở cổng 8765 (Ctrl+C trong tab Terminal), rồi chạy lại đúng lệnh ở mục "Khởi động". Không phải thêm tham số. Tham số mới `--test` mặc định là `data/bench_v1_1/test_v1.1.jsonl`. Lần khởi động đầu tiên, `data/human_edits/state.json` được viết lại và có thêm khóa `cau_test` (đang rỗng). Đây là thay đổi dự kiến; nó sẽ vào commit ở lần "Chốt đợt" tiếp theo.

### Lược đồ đề xuất theo câu (dòng có `"loai": "cau"`)

`{id, loai:"cau", split:"dev"|"test", row_id, pair_id, nguon, truong:"state"|"question"|"option:<k>", gia_tri_cu, gia_tri_de_xuat, ly_do:"loi_dich"|"chinh_ta", bang_chung_en, giai_thich, nguoi_de_xuat, thoi_diem}`

- `row_id` là id của dòng tiếng Việt có dấu, tức `<pair_id>:vi`.
- `id` phải duy nhất trong cả file: đề xuất chuỗi mẫu và đề xuất theo câu dùng chung một không gian id.
- Nếu đề xuất có `doi_dap_an` hoặc `do_tin_cay` thì công cụ hiện ra; hai trường này không bắt buộc.
- Dòng đề xuất chuỗi mẫu (định dạng cũ) vẫn giữ nguyên và hoạt động như trước.

### Kiểm và áp đề xuất

- **Kiểm cấu trúc:** `split`, `pair_id` có thật, `row_id` khớp, `nguon` khớp, `truong` hợp lệ, `ly_do` hợp lệ. Đề xuất sai cấu trúc chỉ hiện id và lỗi, không hiện văn bản của câu, và chỉ có nút "Từ chối".
- **Kiểm nội dung:** `gia_tri_cu` phải xuất hiện **đúng 1 lần** trong giá trị hiện hành của trường, tức bản gốc cộng các sửa đã duyệt; công cụ đếm cả trường hợp chồng lấn. Nếu không đạt, thẻ ghi rõ số lần xuất hiện và chỉ cho từ chối.
- **Áp:** thay đúng chuỗi con đó. Bản ghi `cau` trong `data/human_edits/edits_log.jsonl` gồm:
  - `split`, `pair_id` (và `dev_id` nếu là dev), `truong`;
  - lý do "lỗi dịch" hoặc "chính tả";
  - `de_xuat_id`;
  - `thay_chuoi_con {cu, moi}`;
  - `gia_tri_moi` là giá trị **cả trường** sau khi thay, giữ quy tắc "bản ghi mới nhất có hiệu lực".
- **Đoạn văn dùng chung:** dev có 215 đoạn văn và test có 204 đoạn văn được nhiều câu hỏi dùng chung (Belebele, ViFactCheck, ViWikiFC), mỗi đoạn tối đa 5 câu. Không có đoạn nào dùng chung giữa dev và test.
  - Duyệt một đề xuất sửa `state` sẽ áp cho mọi câu cùng split có cùng đoạn văn hiện hành. Thẻ ghi rõ "áp cho N câu".
  - Câu nào đã được sửa đoạn văn riêng thì bỏ qua, và thẻ có ghi.
  - Sửa ở test phải có đề xuất riêng với `split:"test"`.
- **Hoàn tác:** đây là phần thêm, brief không yêu cầu. Mục "Đã xử lý" có nút hoàn tác cho từng đề xuất theo câu, kể cả đề xuất đã bị từ chối.
  - Hoàn tác đưa các trường về giá trị trước khi duyệt và đưa đề xuất về "đang chờ".
  - Chỉ làm được khi trường đó chưa bị sửa tiếp sau đó, tức là hoàn tác theo thứ tự ngược.
  - Thêm phần này để duyệt nhầm ở test vẫn sửa lại được, vì không màn nào khác hiện test.

### Hiển thị một thẻ

- **Nhãn:** thẻ test có viền đỏ và nhãn **TEST**; thẻ dev có nhãn DEV.
- **Bên tiếng Việt:** đoạn văn, câu hỏi và các lựa chọn hiện hành; chỗ đổi được bôi (đỏ gạch là cũ, xanh là mới), và khung đoạn văn tự cuộn tới chỗ đó.
- **Bên tiếng Anh:** hiện bản tương ứng, `bang_chung_en` được bôi nếu tìm thấy nguyên văn.
- **Đáp án chuẩn:** hiện ở cả VI và EN. Nếu đề xuất sửa chính lựa chọn là đáp án chuẩn thì thẻ cảnh báo riêng.
- **Thao tác:** có ô sửa đoạn đề xuất cho "Sửa rồi duyệt" và ba nút Duyệt / Sửa rồi duyệt / Từ chối.
- **Lọc và phân trang:** lọc theo loại (chuỗi mẫu / câu dev / câu test) và theo nguồn; mỗi trang 20 thẻ.

### Câu test: ngoại lệ có chủ đích

- Chỉ màn "Chờ duyệt" hiện câu test, và chỉ câu có đề xuất. Với đoạn văn dùng chung, thẻ chỉ liệt kê id của các câu cùng đoạn, không hiện câu hỏi của chúng.
- Các màn khác (Trang đầu, Soát A/B/C, Mẫu và nhãn, Từng câu, Sửa câu) không hiện test. Trang đầu chỉ có số đếm. Không có lối nào xem toàn bộ test.
- File test chỉ được đọc. Khi khởi động, công cụ kiểm hash với `manifest_test_v1.1.json`; nếu lệch thì không hiện câu test nào.
- **Đánh đổi cần biết:**
  - Sửa đã duyệt ở test lưu giá trị cả trường (có thể là cả đoạn văn) vào `edits_log.jsonl`, `state.json` và CSV xuất ra, nằm trong git cục bộ. Đây chỉ là những câu có đề xuất, và chỉ ở trên máy, không có remote.
  - `build_revised.py` cần đúng giá trị đó để dựng bản 1.2.

### Script dựng và script phát hành

- `build_revised.py` áp `cau_test` cho test, cùng cách như dev: VI có dấu, sinh lại bản không dấu, tính lại `n_tokens`; EN giữ nguyên.
  - Test đổi thì ghi `test_v1.2.jsonl` đóng băng (SHA-256, chỉ đọc). `test_v1.1` giữ nguyên.
  - Báo cáo dựng có thêm `changes_test`, chỉ gồm id, trường và lý do, không có văn bản.
  - Script chỉ chạy khi người dùng yêu cầu.
- Chạy thử với lớp ghi đè thật (rỗng): dev trùng từng byte, test trùng hash `fbad818b…`. Không thay đổi so với lần trước.
- `release_check.py` in thêm số đề xuất đang chờ theo loại (chuỗi mẫu / câu dev / câu test) và số câu đã sửa ở dev và test. Kết quả vẫn TỪ CHỐI vì còn 65 chuỗi chưa duyệt.

### Thử (thư mục tạm, cổng 8768; công cụ ở cổng 8765 không bị đụng)

Đề xuất giả do script sinh, lấy chuỗi con có thật từ dữ liệu; script chỉ in id, số đếm và kết quả đúng/sai, không in văn bản test.

- **3 đề xuất chính theo brief:**
  - 1 dev: sửa đoạn văn Belebele dùng chung 2 câu. Duyệt thì áp cho cả 2 câu.
  - 1 test: sửa đoạn văn Belebele dùng chung 2 câu, dùng "Sửa rồi duyệt". Áp cho 2 câu, nằm ở `cau_test`, không ở `cau`.
  - 1 test: sửa lựa chọn 1 của XCOPA. Từ chối, hoàn tác lại về "đang chờ", rồi duyệt.
- **Các ca biên thêm:**
  - 1 dev khác trên cùng đoạn văn: các sửa nối tiếp đúng. Hoàn tác sai thứ tự bị chặn (400); hoàn tác đúng thứ tự đưa câu về bản gốc.
  - 1 test sai `row_id`: báo lỗi, không hiện văn bản; duyệt bị 400.
  - 1 dev có `gia_tri_cu` không xuất hiện: báo 0 lần; duyệt bị 400.
  - 1 đề xuất chuỗi mẫu định dạng cũ: duyệt thì chuỗi thành "đã sửa" ở Mẫu và nhãn.
- **Kết quả thử công cụ: 49/49 đạt.** Trong đó:
  - Chỉ có 4 id câu test xuất hiện: 3 câu có đề xuất và 1 câu cùng đoạn văn. Không lộ đoạn văn nào của câu test khác. Sáu màn khác không hiện test.
  - Chuỗi XSS đặt trong `giai_thich`, `nguoi_de_xuat` và `id` đều được escape; trang chỉ có 2 thẻ script của công cụ.
  - Bảo mật:

    | Thử | Kết quả |
    |---|---|
    | POST không token | 403 |
    | POST token sai | 403 |
    | POST Host lạ | 403 |
    | POST Origin lạ | 403 |
    | POST `text/plain` | 415 |
    | GET `/api/proposal` | 405 |
    | GET Host lạ | 403 |

    Sau các lần bị chặn, log vẫn rỗng.
  - CSV có BOM và thêm các cột `split`, `pair_id`, `doan_cu`, `doan_moi`.
  - Mọi dòng log là JSON hợp lệ, ghi bằng fsync như trước.
- **Dựng thử từ lớp ghi đè tạm (WSL venv, ra thư mục tạm): 16/16 đạt.**
  - Dev đổi 604 dòng: 300 dòng SIB do tên chủ đề sửa thử, mỗi dòng kèm bản không dấu, cộng 2 câu Belebele × 2 biến thể.
  - Test đổi 246 dòng: 120 × 2 cộng 3 câu × 2.
  - Dòng EN không đổi. Chỗ sửa đúng vị trí, bản không dấu sinh lại, `n_tokens` tính lại.
  - `test_v1.2.jsonl` chỉ đọc, hash khớp file FROZEN. `test_v1.1` gốc còn nguyên.
- **Sau khi thử:** `git status` sạch; `config/vi_proposals.jsonl` và `data/human_edits/edits_log.jsonl` vẫn rỗng; hash dev và test 1.1 không đổi.
