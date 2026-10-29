# Đọc lại lượt 3: các cặp CHƯA bị sửa

Làm theo `VERIFY_INSTRUCTIONS.md`, với các bổ sung sau.

- Mọi cặp trong lô này có `loai_thay_doi` = `KHONG_DOI`: chưa bị sửa, nên `vi_truoc` = `vi_sau`. Kiểm như với `SUA`, theo mục 1, 2 và 4.
- Hai lượt Opus trước đã cho các cặp này là OK, nên chỉ ghi lỗi RÕ RÀNG:
  - dịch sai nghĩa so với EN;
  - lỗi chính tả, sót chữ, lặp chữ;
  - đáp án sai, hoặc không còn duy nhất;
  - câu mơ hồ thật sự.
- Không ghi lựa chọn từ hay văn phong. Khi phân vân thì ghi `OK`.
- `y_kien_nhan` (nếu có) là ý kiến về nhãn của các lượt trước. Đó là các câu đã được quyết GIU (giữ nhãn); chỉ ghi `DAP_AN_SAI` nếu bạn chắc.
- Với nguồn bản địa ViFactCheck và ViWikiFC: phát biểu (claim) nằm trong `question`. Không sửa lỗi văn phong của bài báo gốc; chỉ ghi lỗi làm sai nhãn.
