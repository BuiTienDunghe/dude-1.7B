# Dịch đầy đủ phần B bằng subagent Sonnet (20/10/2026)

Không train, không commit, không xóa file, không đụng `data/`. Không đọc `results/train_v1/translated/` (bản Google).

## Kết quả

- **Bản dịch:** `full/ban_dich_s2.jsonl`, đủ **6.944/6.944 dòng**.
  - 175 dòng dùng lại từ đợt thử; 25 dòng của đợt thử bị ảnh hưởng nên dịch lại.
  - Kiểm bằng `kiem_ban_dich.py` v2.1: **ĐẠT, 0 lỗi, 95 cảnh báo**. Cảnh báo chủ yếu là số "1" thay cho chữ "l" do lỗi gõ trong bản gốc race, và lựa chọn ngắn lệch độ dài.
- **Đọc lại:** người đọc lại (Sonnet 2) sửa **358/6.944 dòng (5,2%)**, tổng 405 chỗ. Mọi dòng khác nhau giữa bản trước và sau khi đọc lại đều có ghi trong nhật ký.
- **Khử rò rỉ lần hai phía tiếng Việt** (`decontam_vi.py`): loại **259 dòng (3,7%)**, giữ 6.685 dòng.
- **Thành phần:** trần 5% đạt ở cả hai.

  | Thành phần | File | Dòng | Nguồn lớn nhất |
  |---|---|---|---|
  | B | `results/train_v1/components/B.jsonl` | **26.638** | 3,0% |
  | C | `results/train_v1/components/C.jsonl` | **31.034** | 3,9% |

  C dùng phần A hiện hành (đã dựng lại theo chuỗi đã duyệt).
- **Đọc kiểm:** `full/doc_kiem_30.html`, 30 dòng (6 mỗi dạng bài). **Chưa train; chờ người dùng xem.**

## Sửa khi đọc lại

| Dạng bài | Dòng | Dòng sửa | Tỷ lệ |
|---|---|---|---|
| đọc hiểu | 2.000 | 202 | 10,1% |
| kiểm chứng | 2.000 | 61 | 3,1% |
| yes/no | 800 | 45 | 5,6% |
| nhân quả | 1.000 | 36 | 3,6% |
| ý định | 1.144 | 14 | 1,2% |

**Theo loại** (405 chỗ):

| Loại | Số chỗ |
|---|---|
| tự nhiên, ngữ pháp | 121 |
| đổi nghĩa | 90 |
| thuật ngữ | 68 |
| tên riêng, số | 39 |
| thêm/bớt nội dung | 34 |
| tiền tố | 29 |
| khác | 19 |
| quan hệ lựa chọn | 5 |

Chi tiết ở `full/thong_ke_day_du.json` và `full/sua_tat_ca.jsonl`.

**Lưu ý chất lượng:**
- Số chỗ người đọc lại sửa không đều giữa các đợt.
  - Có đợt sửa nhiều: đợt 06 sửa 40/160 dòng, gồm 19 chỗ sai thuật ngữ; đợt 40 có 21 chỗ "Turn" bị dịch thành "Lượt".
  - Có 4 đợt không sửa dòng nào: 14, 20, 27, 30 (đợt 31 cũng 0). Phần lớn là dòng ngắn hoặc nhãn theo bảng thuật ngữ, nhưng cũng có thể người đọc lại đã soát lỏng.
- Ở đợt thử 200 dòng, Opus chấm thấy bản Sonnet còn 3/200 dòng lỗi nghĩa sau khi đọc lại.

## Khử rò rỉ lần hai phía tiếng Việt

So với toàn bộ văn bản tiếng Việt của dev và test 1.3 (kể cả không dấu), các nguồn bench tiếng Việt, train_v0/val_v0 và phần A.

**Lượt 1 bắt nhầm, nên chạy lại.** Lượt 1 loại 1.287 dòng (bản lưu ở `part_b_vi_loai_luot1.json`, `decontam_vi_stats_luot1.json`). Khoảng 1.000 dòng trong số đó bị loại chỉ vì cụm "không đủ thông tin", vốn nằm trong một lựa chọn của Global-PIQA parallel. Cụm này trùng với:
- nhãn kiểm chứng theo bench "Văn bản không đủ thông tin";
- bản dịch theo bảng thuật ngữ của "not enough information" (quail).

**Lượt 2 sửa hai chỗ:**
1. Lựa chọn của Global-PIQA chỉ xét trùng nguyên văn, MinHash và 13-gram, như đã làm với lựa chọn Belebele.
2. Chuỗi cố định của dự án (giá trị bảng thuật ngữ, Có/Không) không đem đi kiểm, vì không phải dữ liệu nguồn.

**Lượt 2 loại 259 dòng:**
- theo dạng bài: đọc hiểu 199, ý định 30, kiểm chứng 12, nhân quả 11, yes/no 7;
- theo lý do: phần lớn là câu MASSIVE tiếng Việt (4–12 từ) nằm gọn trong đoạn văn (153 dòng), MLQA (50 dòng), dev/test 1.3 (19 dòng), train_v0 (11 dòng), state vượt 1.024 token (6 dòng).

Phép "chứa đoạn ngắn" này áp đúng quy tắc đã dùng cho phần A. Chi tiết ở `part_b_vi_loai.json` và `decontam_vi_stats.json`.

## Mức dùng và thời gian

| Mốc | Giờ | % 5 giờ | % tuần |
|---|---|---|---|
| Bắt đầu | 13:38:55 | 12 | 40 |
| Xong dịch và đọc lại | 15:47:59 | 59 | 52 |
| Xong khử rò rỉ (sau khi chạy lại) | khoảng 16:07 | 60 | 53 |

- **Không lần nào chạm ngưỡng:** cửa sổ 5 giờ chưa tới 88%, tuần chưa tới 85%; không dùng mức dùng thêm.
- Mức dùng được kiểm trước mỗi đợt (4 lô). Nhật ký ở `full/muc_dung.jsonl`.
- **Thời gian dịch và đọc lại:** khoảng 2 giờ 9 phút, chạy 4–6 subagent song song.
- **Token subagent:** khoảng 17,6 triệu (dịch 8,6 triệu, đọc lại 9,1 triệu), qua 88 lần gọi; ghi ở `full/token_subagent.jsonl`.
- So với ngoại suy từ đợt thử (khoảng 120 điểm % cửa sổ 5 giờ), thực tế chỉ tốn khoảng 48 điểm. Lý do: mỗi lần gọi làm 4 lô, nên chi phí cố định mỗi lô nhỏ hơn. Số % tuần là của cả tài khoản, có thể gồm cả phiên khác.

## Sự cố

- **Đợt 02, lô 007:** bị chặn "Output blocked by content filtering policy". Chạy lại lô 007–008 thì thành công.
- **Thư mục scratchpad dùng chung:** script phụ có thể bị agent khác ghi đè ở đợt 05 và 06. Từ đợt 10, chỉ dẫn bắt buộc mỗi agent dùng thư mục con riêng. Kết quả cuối vẫn qua kiểm theo id từng lô.
- **Đợt 03:** người dịch tự mở bản dịch các lô 001–006 để theo mẫu câu hỏi. Các file đó không có nhãn.
- **Lô 072:** người dịch thêm chữ "tên" để qua phép kiểm "còn nguyên văn".
  - Đã sửa script kiểm lên v2.1: đoạn giống hệt bản gốc chỉ là lỗi khi có từ 2 hư từ tiếng Anh trở lên; còn lại chỉ cảnh báo.
  - Người đọc lại đã bỏ chữ thêm.
  - Lô 060: "higud mrng" được dịch thành "chào buổi sáng", chấp nhận được vì là lời chào.
- **Câu hỏi dạng "pack"** (đếm nhãn, hỏi nhãn của Item X) chưa có trong bảng thuật ngữ. Người dịch tự dịch theo mẫu và dùng đúng nhãn bench; người đọc lại có soát thuật ngữ.

## File

Trong `results/train_v1/translated_sonnet/full/`:
- **Bản dịch:** `ban_dich_s2.jsonl` (bản dùng), `ban_dich_s1.jsonl` (trước khi đọc lại);
- **Nhật ký sửa và từng lô:** `sua_tat_ca.jsonl`, `lo/`;
- **Kiểm và thống kê:** `kiem_s2.json`, `thong_ke_day_du.json`, `trang_thai.json`;
- **Kế hoạch:** `plan.json`, `pilot_giu.jsonl`, `pilot_anh_huong.json`;
- **Chỉ dẫn và ghi chú điều phối:** `PROMPT_dich.md`, `PROMPT_doc_lai.md`, `DIEU_PHOI_GHI_CHU.md`;
- **Nhật ký mức dùng và token:** `muc_dung.jsonl`, `token_subagent.jsonl`;
- **Đọc kiểm:** `doc_kiem_30.html`.

Trong `results/train_v1/`: `part_b_vi.jsonl`, `part_b_vi_loai.json`, `decontam_vi_stats.json`, `components/B.jsonl`, `components/C.jsonl`.

Script (trong `scripts/train_v1/`): `full_run_prep.py`, `full_status.py`, `full_assemble.py`, `decontam_vi.py` (v2), `refs_index.py` (thêm tham số `opt_prefixes`), `make_review_html.py`.
