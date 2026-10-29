# Đọc lỗi mù bằng Opus: train thử bản C so với M1 (dev 1.3)

Ngày 26/10/2026. Theo mục 4 của `results/BRIEF_phan_tich_xu_huong.md`; điều phối giao. Không train, không chạy test.

## Cách làm

- **Mẫu** (`scripts/train_v1/doc_loi_prep.py`): seed 20261012, phân tầng theo nguồn; chi tiết ở `doc_loi/chon_mau.json`.
  - **A**: 50 câu dev VI có dấu mà M1 sai, C@100% đúng.
  - **B**: 50 câu M1 đúng, C@100% sai.
  - **P4**: 40 câu P4 VI có dấu, kiểu choice, mà C không chọn "không đủ thông tin" còn M1 chọn đúng.
- **Đọc mù:**
  - A và B trộn chung, mã trung tính X001…, chia 4 lô; P4 có 2 lô riêng.
  - Mỗi câu chỉ có văn bản, câu hỏi, lựa chọn, đáp án đúng và đáp án "một model" chọn (luôn là của C).
  - Người đọc không biết model nào, không biết nhóm; khóa giải mã (`doc_loi/khoa.json`) không đưa cho người đọc.
- **Người đọc:** 6 subagent Opus, mỗi lô một subagent. Hướng dẫn: `doc_loi/PROMPT_doc_loi.md`, `doc_loi/PROMPT_p4.md`. Kết quả thô: `doc_loi/ket_qua/*.jsonl`.
- **Lưu ý:** nhóm A có đáp án đúng nên chỉ xếp vào loại `dung_*`; nhóm B chỉ xếp vào `sai_*`. Mỗi câu chỉ có một người đọc, nên chưa đo độ đồng thuận.

**Kiểm định dạng:** 140/140 câu có kết quả; thiếu 0; lỗi 0.

## A. C mới làm đúng (M1 sai → C đúng): đúng vì lý do gì?

n = 50.

| Loại | Số câu | Tỉ lệ | belebele | massive | sib200 | vifactcheck | viwikifc | xcopa |
|---|---|---|---|---|---|---|---|---|
| `dung_ly_do_dung` | 28 | 56% | 7 | 3 | 4 | 3 | 7 | 4 |
| `dung_meo_lua_chon` | 0 | 0% | 0 | 0 | 0 | 0 | 0 | 0 |
| `dung_kien_thuc_san_co` | 2 | 4% | 0 | 0 | 0 | 0 | 0 | 2 |
| `dung_khop_be_mat` | 14 | 28% | 2 | 6 | 1 | 1 | 4 | 0 |
| `dung_nhan_dang_ngo` | 6 | 12% | 1 | 1 | 2 | 0 | 2 | 0 |
| **Tổng** | 50 | | 10 | 10 | 7 | 4 | 13 | 6 |

- Văn bản đủ căn cứ cho đáp án đúng: có 42, một phần 8, không 0.
- Độ tự tin của người đọc: cao 19, vừa 30, thấp 1.
- Câu mang cờ mơ hồ của bench: 1.

**Ví dụ ghi chú** (tối đa 2 câu mỗi loại):

- `dung_ly_do_dung`, X001 (sib200, topic): Có từ 'vùng chiến sự' dễ kéo sang chính trị; phải hiểu ý chuẩn bị cho chuyến đi mới chọn du lịch.
- `dung_ly_do_dung`, X008 (viwikifc, choice): Phải để ý chữ 'chỉ' trong nhận định và 'cơ quan đặc biệt' trong văn bản mới thấy bị bác bỏ.
- `dung_kien_thuc_san_co`, X066 (xcopa, causal): Suy luận nhân quả thường thức: bị bắt vì hành hung hợp lý hơn vào trại cai nghiện.
- `dung_kien_thuc_san_co`, X070 (xcopa, causal): Thường thức: chó sủa khi có tiếng gõ cửa; lựa chọn con mèo nằm dài không gây sủa.
- `dung_khop_be_mat`, X003 (viwikifc, choice): Nhận định gần như chép lại câu đầu của văn bản; chỉ khớp chữ cũng đủ chọn xác nhận.
- `dung_khop_be_mat`, X013 (belebele, reading_mc): Văn bản ghi nguyên văn 'bị rò rỉ với báo chí'; khớp chữ là đủ.
- `dung_nhan_dang_ngo`, X012 (belebele, reading_mc): Văn bản nói phóng photon để loại bỏ năng lượng dư thừa (D); B thêm chữ 'mạnh' không có trong văn bản. Nhãn B mơ hồ, D cũng bảo vệ được.
- `dung_nhan_dang_ngo`, X057 (viwikifc, choice): Văn bản chỉ nói trận năm 1427, không loại trừ việc nhà Minh chịu thua/rút năm 1428; nhãn bác bỏ dựa lệch năm, có thể là không đủ thông tin.

## B. C mới làm sai (M1 đúng → C sai): loại lỗi

n = 50.

| Loại | Số câu | Tỉ lệ | belebele | massive | sib200 | vifactcheck | viwikifc | xcopa |
|---|---|---|---|---|---|---|---|---|
| `sai_khong_doc_van_ban` | 8 | 16% | 1 | 4 | 3 | 0 | 0 | 0 |
| `sai_hieu_sai_phu_dinh` | 3 | 6% | 1 | 0 | 0 | 0 | 2 | 0 |
| `sai_nham_nhan_kiem_chung` | 9 | 18% | 0 | 0 | 0 | 6 | 3 | 0 |
| `sai_nham_nhan_gan_nghia` | 14 | 28% | 0 | 4 | 10 | 0 | 0 | 0 |
| `sai_suy_luan` | 11 | 22% | 7 | 0 | 0 | 0 | 1 | 3 |
| `sai_mo_ho_hoac_nhan_nghi_sai` | 4 | 8% | 1 | 1 | 1 | 0 | 1 | 0 |
| `sai_khac` | 1 | 2% | 0 | 1 | 0 | 0 | 0 | 0 |
| **Tổng** | 50 | | 10 | 10 | 14 | 6 | 7 | 3 |

- Văn bản đủ căn cứ cho đáp án đúng: có 35, một phần 14, không 1.
- Độ tự tin của người đọc: cao 16, vừa 30, thấp 4.
- Câu mang cờ mơ hồ của bench: 2.

**Ví dụ ghi chú** (tối đa 2 câu mỗi loại):

- `sai_khong_doc_van_ban`, X007 (sib200, topic): Nói về chương trình khuyến khích trẻ đọc sách, không có yếu tố chính trị; nhãn giải trí cần biết Reading Rainbow là chương trình truyền hình.
- `sai_khong_doc_van_ban`, X031 (sib200, topic): Văn bản nói chiến tranh tranh ngôi vàng (chính trị); model chọn giải trí, có lẽ liên tưởng Tam Quốc diễn nghĩa/phim/game từ kiến thức ngoài.
- `sai_hieu_sai_phu_dinh`, X029 (viwikifc, choice): Quốc hội nhấn mạnh 'cần phân biệt' hai vấn đề nên nhận định phủ định 'không đưa ra yêu cầu gì' bị bác bỏ; model xác nhận, có thể bỏ qua chữ 'không'.
- `sai_hieu_sai_phu_dinh`, X082 (viwikifc, choice): Nhận định viết 'không thể phủ nhận rằng ông không có lòng yêu nước', trái văn bản; model bỏ qua phủ định kép nên chọn xác nhận.
- `sai_nham_nhan_kiem_chung`, X005 (viwikifc, choice): Văn bản không nói gì về mức phổ biến hiện nay của thuyết Hình thức; model chọn bác bỏ thay vì không đủ thông tin.
- `sai_nham_nhan_kiem_chung`, X022 (vifactcheck, choice): Văn bản chỉ nói xử phạt hành chính, không đề cập xử lý hình sự; model chọn xác nhận, có thể theo kiến thức ngoài.
- `sai_nham_nhan_gan_nghia`, X002 (sib200, topic): Văn bản nói rõ 'nhà báo du lịch', 'nhật ký du lịch'; model bị chi tiết 'trang web' kéo sang khoa học/công nghệ.
- `sai_nham_nhan_gan_nghia`, X009 (sib200, topic): Đi bộ đường trường nằm giữa thể thao và du lịch; ranh giới hai nhãn mờ, lựa chọn du lịch khá hợp lý.
- `sai_suy_luan`, X004 (belebele, reading_mc): Văn bản nêu rõ cải thiện bài viết 'để tránh bị phê bình'; ý nuôi dưỡng sự quan tâm của khán giả gắn với hiểu biết, không phải viết lách.
- `sai_suy_luan`, X027 (xcopa, causal): Thịt chuyển màu nâu do nướng là quan hệ nhân quả hiển nhiên; model chọn đông lạnh, suy luận nhân quả sai.
- `sai_mo_ho_hoac_nhan_nghi_sai`, X006 (viwikifc, choice): Văn bản chỉ nói Kim không hề nói về dịch chuyển tức thời, không có tuyên bố mình không có khả năng; nhãn bác bỏ đáng ngờ, xác nhận hay không đủ thông tin đều bảo vệ được.
- `sai_mo_ho_hoac_nhan_nghi_sai`, X040 (belebele, reading_mc): Bản dịch chỉ nói công ty lữ hành 'liên kết' với khách sạn, không nói 'ưu đãi'; C cũng không được nêu rõ nên lựa chọn model bảo vệ được phần nào. A vẫn là đáp án tốt hơn.
- `sai_khac`, X017 (massive, intent): Lựa chọn F khớp gần nguyên văn yêu cầu; model chọn giải nghĩa thuật ngữ, có lẽ coi cụm từ ngắn là thuật ngữ cần giải thích.

## P4. Văn bản không liên quan: C không chọn "không đủ thông tin", M1 chọn đúng

n = 40. Nhãn C chọn: {'Bị văn bản bác bỏ': 40}. Nguồn: {'viwikifc': 29, 'vifactcheck': 11}.

**van_ban_lien_quan**

| Giá trị | Số câu | Tỉ lệ |
|---|---|---|
| `khong` | 37 | 92% |
| `cung_chu_de_rong` | 3 | 8% |
| `co_chi_tiet_lien_quan` | 0 | 0% |

**nei_co_that_dung**

| Giá trị | Số câu | Tỉ lệ |
|---|---|---|
| `co` | 40 | 100% |
| `khong` | 0 | 0% |
| `khong_chac` | 0 | 0% |

**nhan_dinh_ngoai_doi**

| Giá trị | Số câu | Tỉ lệ |
|---|---|---|
| `dung` | 17 | 42% |
| `sai` | 6 | 15% |
| `khong_biet` | 17 | 42% |

**nguyen_nhan_kha_di**

| Giá trị | Số câu | Tỉ lệ |
|---|---|---|
| `nhan_dinh_sai_theo_kien_thuc` | 6 | 15% |
| `nhan_dinh_dung_theo_kien_thuc` | 0 | 0% |
| `nhan_dinh_tuyet_doi_hoa` | 2 | 5% |
| `nhan_dinh_co_phu_dinh` | 2 | 5% |
| `chu_de_trung_van_ban` | 1 | 2% |
| `khong_ro` | 29 | 72% |

**Bảng chéo: nhận định ngoài đời × văn bản liên quan**

| ngoài đời \ văn bản | khong | cung_chu_de_rong | co_chi_tiet_lien_quan |
|---|---|---|---|
| dung | 14 | 3 | 0 |
| sai | 6 | 0 | 0 |
| khong_biet | 17 | 0 | 0 |

**Dấu hiệu trong nhận định** (một câu có thể nhiều dấu hiệu): {'ten_rieng': 30, 'kien_thuc_pho_thong': 16, 'so_lieu': 10, 'so_sanh': 9, 'thoi_gian': 8, 'phu_dinh': 4, 'tuyet_doi_hoa': 2, 'dai_tu_khong_ro_tham_chieu': 2, 'khoa_hoc': 1}.

**Nhãn gốc của nhận định trong bài của nó** (trước khi ghép văn bản khác): {'refuted': 15, 'nei': 12, 'supported': 13}.

**Ví dụ ghi chú** (tối đa 2 câu mỗi nguyên nhân):

- `nhan_dinh_sai_theo_kien_thuc`, P005 (viwikifc): Văn bản về tên gọi Thái Bình Dương; bóng đá thực tế là môn được theo dõi nhiều nhất ở Việt Nam nên model bác bỏ theo kiến thức.
- `nhan_dinh_sai_theo_kien_thuc`, P012 (viwikifc): Văn bản địa lý Singapore; Gaul chịu ảnh hưởng La Mã chứ không phải Mông Cổ, nhận định sai rõ nên model bác bỏ theo kiến thức.
- `nhan_dinh_tuyet_doi_hoa`, P010 (viwikifc): Văn bản Hoàng Sa 1909; nhận định khái quát mạnh 'khác với các nước khác' về ẩm thực Việt, mang tính chủ quan, dễ bị coi là sai.
- `nhan_dinh_tuyet_doi_hoa`, P039 (viwikifc): Văn bản về tiểu sử Phan Huy Chú; nhận định về giai đoạn gian băng và băng giá, nhìn chung đúng, có từ 'luôn'.
- `nhan_dinh_co_phu_dinh`, P015 (vifactcheck): Văn bản về Shakira; nhận định chính sách vé phố cổ Hội An, có 'không cần mua vé', có thể phủ định làm model nghiêng về bác bỏ.
- `nhan_dinh_co_phu_dinh`, P028 (viwikifc): Văn bản đánh giá một nhà Nho bị Nhật lợi dụng; nhận định so sánh ẩm thực Việt với Trung Quốc, Nhật Bản, có hai vế phủ định. Văn bản có nhắc Nhật, Khổng Tử.
- `chu_de_trung_van_ban`, P027 (viwikifc): Cả hai đều là lịch sử Việt Nam (Lý Nam Đế, Bà Triệu so với Nguyễn Phúc Nguyên). Nhắc tên vua khác có thể khiến model tưởng văn bản mâu thuẫn.
- `khong_ro`, P001 (viwikifc): Văn bản về Einstein, nhiệt dung riêng; nhận định lịch sử Lê Lợi 1428 đúng ngoài đời nhưng model lại bác bỏ, không thấy lý do rõ.
- `khong_ro`, P002 (viwikifc): Văn bản về dân cư Phúc Kiến; nhận định về Hoàng Sa. Hoàng Sa thường chia hai nhóm đảo, 'hai hòn đảo chính' hơi lệch nhưng không chắc.

## Số bổ sung tính từ preds (toàn tập, không phải mẫu)

- **M1**:
  - P4 choice VI có dấu (241 câu), nhãn được chọn: {'không đủ thông tin': 215, 'bác bỏ': 26}.
  - Dev kiểm chứng choice VI có dấu, câu có đáp án "không đủ thông tin" (75 câu), nhãn được chọn: {'xác nhận': 28, 'bác bỏ': 24, 'không đủ thông tin': 23}.
  - Dev kiểm chứng choice VI có dấu, mọi câu (241 câu), phân bố nhãn dự đoán: {'bác bỏ': 119, 'xác nhận': 92, 'không đủ thông tin': 30}.
- **C@100%**:
  - P4 choice VI có dấu (241 câu), nhãn được chọn: {'không đủ thông tin': 131, 'bác bỏ': 110}.
  - Dev kiểm chứng choice VI có dấu, câu có đáp án "không đủ thông tin" (75 câu), nhãn được chọn: {'bác bỏ': 28, 'không đủ thông tin': 25, 'xác nhận': 22}.
  - Dev kiểm chứng choice VI có dấu, mọi câu (241 câu), phân bố nhãn dự đoán: {'bác bỏ': 126, 'xác nhận': 86, 'không đủ thông tin': 29}.
- Phân bố đáp án thật, dev kiểm chứng choice VI có dấu: {'bác bỏ': 103, 'không đủ thông tin': 75, 'xác nhận': 63}.

## Ghi chú

- Đây là đọc của Opus, mỗi câu một người đọc; chưa đo độ đồng thuận. Số nhỏ (50/50/40), nên tỉ lệ chỉ để định hướng.
- Báo cáo chỉ trình bày số, không kết luận kịch bản K1–K9.
