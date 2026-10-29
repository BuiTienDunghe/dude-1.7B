# Hướng dẫn đọc nhóm "văn bản không liên quan" (đọc mù)

Mỗi câu trong lô là một bài kiểm chứng tiếng Việt. Văn bản được **cố ý lấy từ một bài khác**, không liên quan tới nhận định. Vì vậy đáp án đúng là "Văn bản không đủ thông tin để kết luận". Model đã chọn một nhãn khác. Bạn không biết đó là model nào, và không cần đoán.

## Quy tắc

- Chỉ mở **đúng hai file**: file hướng dẫn này và file lô được giao. **Không mở file nào khác** trong thư mục, đặc biệt không mở `khoa.json` hay lô khác.
- Đọc kỹ văn bản và nhận định.
- Không chạy code, không tìm trên mạng.

## Với mỗi câu, ghi các trường sau

**`van_ban_lien_quan`**: văn bản có liên quan tới nhận định không?
- `khong`: khác hẳn chủ đề.
- `cung_chu_de_rong`: cùng lĩnh vực rộng, ví dụ cùng nói về Trung Quốc hay cùng là tin kinh tế.
- `co_chi_tiet_lien_quan`: có chi tiết liên quan trực tiếp.

**`nei_co_that_dung`**: với văn bản này, "không đủ thông tin" có thật là đáp án đúng không?
- `co`, `khong` (văn bản thật sự xác nhận hoặc bác bỏ được nhận định), hoặc `khong_chac`.

**`nhan_dinh_ngoai_doi`**: theo kiến thức phổ thông của bạn, nhận định đúng hay sai ngoài đời?
- `dung`, `sai`, hoặc `khong_biet`.

**`nguyen_nhan_kha_di`**: chọn **một** lý do có vẻ khiến model chọn nhãn khác thay vì "không đủ thông tin".

| Mã | Khi nào dùng |
|---|---|
| `nhan_dinh_sai_theo_kien_thuc` | Nhận định sai ngoài đời, nên model "bác bỏ" theo kiến thức sẵn có |
| `nhan_dinh_dung_theo_kien_thuc` | Nhận định đúng ngoài đời, nên model "xác nhận" theo kiến thức sẵn có |
| `nhan_dinh_tuyet_doi_hoa` | Nhận định có từ tuyệt đối hay cực đoan: "tất cả", "duy nhất", "không bao giờ", "nhất"… |
| `nhan_dinh_co_phu_dinh` | Nhận định chứa phủ định ("không", "chưa"…) |
| `chu_de_trung_van_ban` | Văn bản cùng chủ đề, dễ khiến model tưởng văn bản có nói tới |
| `khong_ro` | Không thấy lý do rõ ràng |

**`dau_hieu`**: danh sách ngắn các đặc điểm của nhận định. Ví dụ: `["so_lieu", "so_sanh", "ten_rieng", "thoi_gian", "phu_dinh", "tuyet_doi_hoa", "kien_thuc_pho_thong"]`; danh sách rỗng nếu không có.

**`ghi_chu`**: tiếng Việt, tối đa 40 từ.

## Kết quả

Ghi file JSONL (mỗi câu một dòng, UTF-8) vào **đúng đường dẫn được giao**. Mỗi dòng:

```json
{"ma": "P001", "van_ban_lien_quan": "...", "nei_co_that_dung": "...", "nhan_dinh_ngoai_doi": "...", "nguyen_nhan_kha_di": "...", "dau_hieu": [], "tu_tin": "cao|vua|thap", "ghi_chu": "..."}
```

Đủ số dòng bằng số câu trong lô. Xong thì trả lời ngắn: số dòng đã ghi, và đếm theo `nguyen_nhan_kha_di`.
