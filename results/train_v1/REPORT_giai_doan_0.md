# Train chính thức, Giai đoạn 0: dựng và chuẩn bị dữ liệu

Ngày 2026-10-17. Không train, không dịch, không chạy model nào (cả trên test). Test 1.3 chỉ được **đọc văn bản** để khử rò rỉ, theo quyết định (a). Mọi file nằm trong `results/train_v1/`; không ghi gì vào `data/`. Không commit, không xóa file.

## Cập nhật A-duyệt (19/10): phần A dựng lại từ chuỗi tv1-* người dùng đã duyệt

Dựng bằng `scripts/train_v1/rebuild_a_approved.py`, từ bản v2 lưu ở `ban_v2/`. Số liệu ở `stage0_stats_A_duyet.json`.

- **Nguồn giá trị:** `data/human_edits/edits_log.jsonl`, gồm 28 quyết định tv1-* trên 25 khoa. Mỗi khoa lấy `gia_tri_moi` của lần duyệt mới nhất theo `thoi_diem`; không khoa nào có lần mới nhất là từ chối.
- **Dòng đổi chữ:** 2.179 dòng (ý định du lịch 1.000, VNTC 1.000, EXAMS 179). Chỉ thay chuỗi; id, nhãn và target giữ nguyên. Không còn dòng nào mang cờ `chuoi_cho_duyet`; mỗi dòng ghi `chuoi_da_duyet` (khoa → id đề xuất).
- **Lý, Hóa, Sinh** đã được duyệt nhưng không dùng, vì EXAMS đã lọc ở v2.
- **Kiểm:**
  - 0 dòng chạm chỉ mục mở rộng (phần A, và nền ở cả chỉ mục VI lẫn EN);
  - dòng đổi chữ không trùng train_v0/val_v0;
  - không state nào vượt 1.024 token;
  - trần 5% đạt (nguồn lớn nhất 5,0%, VSMRC 1.217).
- **Số dòng:**

  | | Dòng |
  |---|---|
  | Phần A | 4.396 |
  | Thành phần A (`components/A.jsonl`) | 24.349 |
  | Kế hoạch B | 26.897 |
  | Kế hoạch C | 31.293 |

  Phần B không đổi. File cần dịch giữ sha `9e279b91…`.

### Chuỗi đang dùng (chữ cuối cùng)

| Khoa | Giá trị | Đề xuất |
|---|---|---|
| ý định `ask_admin_change` | hỏi một đơn vị hành chính có bị sáp nhập, chia tách hay đổi địa giới không | `tv1-intent-ask_admin_change` |
| ý định `ask_location_mapping` | hỏi một đơn vị hành chính hiện nằm ở đâu hoặc thuộc đơn vị hành chính nào | `tv1-intent-ask_location_mapping-v2` |
| ý định `ask_new_name` | hỏi tên mới của một đơn vị hành chính sau sáp nhập | `tv1-intent-ask_new_name` |
| ý định `ask_old_name` | hỏi tên cũ hoặc các đơn vị hành chính cũ hợp thành một đơn vị hành chính hiện nay | `tv1-intent-ask_old_name-v2` |
| ý định `compare_before_after_merge` | so sánh một đơn vị hành chính trước và sau sáp nhập | `tv1-intent-compare_before_after_merge` |
| ý định `recommend_tourist_spot` | gợi ý địa điểm du lịch nên đến | `tv1-intent-recommend_tourist_spot` |
| ý định `search_tourist_address` | tìm địa chỉ hoặc đường đi đến một địa điểm du lịch | `tv1-intent-search_tourist_address-v2` |
| ý định `search_tourist_by_old_or_new` | hỏi một địa điểm du lịch thuộc đơn vị hành chính cũ hoặc mới nào | `tv1-intent-search_tourist_by_old_or_new` |
| VNTC `Chinh tri Xa hoi` | chính trị - xã hội | `tv1-vntc-chinh_tri_xa_hoi` |
| VNTC `Doi song` | đời sống | `tv1-vntc-doi_song` |
| VNTC `Khoa hoc` | khoa học | `tv1-vntc-khoa_hoc` |
| VNTC `Kinh doanh` | kinh doanh | `tv1-vntc-kinh_doanh` |
| VNTC `Phap luat` | pháp luật | `tv1-vntc-phap_luat` |
| VNTC `Suc khoe` | sức khỏe | `tv1-vntc-suc_khoe` |
| VNTC `The gioi` | thế giới | `tv1-vntc-the_gioi` |
| VNTC `The thao` | thể thao | `tv1-vntc-the_thao` |
| VNTC `Van hoa` | văn hóa | `tv1-vntc-van_hoa` |
| VNTC `Vi tinh` | vi tính | `tv1-vntc-vi_tinh` |
| EXAMS mẫu state | Môn {mon}, lớp {lop} | `tv1-exams-state` |
| EXAMS môn `History` | Lịch sử | `tv1-exams-subj-history` |
| EXAMS môn `Geography` | Địa lý | `tv1-exams-subj-geography` |
| EXAMS môn `Citizenship` | Giáo dục công dân | `tv1-exams-subj-citizenship` |

## Cập nhật v2 (quyết định người dùng 17/10, qua điều phối): số liệu hiện hành

Các mục 1–10 bên dưới mô tả bản v1 (lượt 2 của `build_stage0.py`). Bản v2 dựng bằng `scripts/train_v1/update_stage0_v2.py` từ bản lưu v1 ở `ban_v1/`, nên chạy lại ra đúng kết quả. Số liệu ở `stage0_stats_v2.json`. Mục này là số hiện hành; chỗ nào khác với mục bên dưới thì theo mục này.

**Thay đổi so với v1:**

1. **Nền:** bỏ cả 47 dòng train_v0 chạm chỉ mục mở rộng (8 VI + 39 EN), trong đó có dòng dynasent trùng nguyên văn "Too expensive?" với test 1.3. Nền còn **19.953** dòng. **Phần tasksource còn 15.961, không còn đúng 16.000.**
2. **EXAMS:** bỏ Lý, Hóa, Sinh, giữ Sử, Địa, Công dân. Còn **179 dòng**: Sử 76, Địa 77, Công dân 26. Không bù dòng.
   - EXAMS-vi train không có Toán và Văn.
   - Ba đề xuất tên môn `tv1-exams-subj-physics`, `-chemistry`, `-biology` không còn dùng; người dùng có thể từ chối trên công cụ. Tôi không sửa file đề xuất.
3. **VSMRC: 1.217, không phải 1.236.** Tổng A giảm (nền bỏ 39 dòng, EXAMS bỏ 321 dòng), nên trần 5% hạ xuống 1.217. Giữ 1.217 dòng đầu theo thứ tự chọn. Đây là chỗ lệch với quyết định "giữ 1.236": muốn giữ 1.236 thì phải nới quy tắc trần 5%.
4. **Nhân quả, thêm vào phần B: 1.000 dòng.**
   - Kiểm trước khi chọn: cả kho chưa dùng được so bằng `decontam.Index` với COPA (EN), Balanced-COPA, XCOPA mọi ngôn ngữ (21 cấu hình gồm translation-*, 42 file validation + test; parquet @ `c1471bb6`) và dev + test 1.3.
   - Sau đó chọn theo seed, mỗi `group_id` một dòng, qua `finalize` của v0 với chỉ mục tiếng Anh mở rộng + v0.

   | Nguồn | Kho | Đã dùng ở v0 | Trùng họ COPA / bench | Sạch | Lấy | Loại lúc chọn |
   |---|---|---|---|---|---|---|
   | e-CARE (`cung_khuon_copa: true`) | 770 | 26 | 5 (XCOPA 4, bench 1) | 739 | **500** (487 choice, 13 noul) | 42 cùng group_id |
   | HeadlineCause en_simple | 1.808 | 56 | 0 | 1.752 | **500** (467 choice, 29 noul, 4 score) | 54 cùng group_id; 4 trùng 13-gram v0 |

   - **Giấy phép e-CARE:** "MIT (GitHub waste-wood/e-care, bản gốc); bản HF 12ml/e-CARE @ a9c78871 KHÔNG có thẻ giấy phép; tasksource ghi BSD 2-Clause (DPI)". **HeadlineCause:** CC0-1.0 (@ 68c78423).
   - **Sửa giữa chừng:** lượt chạy đầu kiểm cả lựa chọn cố định "Yes/No" của câu noul. "No" trùng nguyên văn một lựa chọn Belebele trong bench, nên 171 dòng noul bị loại nhầm. Lượt chính thức không kiểm lựa chọn cố định đó; state, câu hỏi và lựa chọn của câu choice vẫn kiểm đủ. Số lượt đầu lưu ở `stage0_stats_v2_luot1.json`.
5. **Kiểm cuối:** phần A, phần B và nền đều **0** dòng chạm chỉ mục mở rộng. Phần nhân quả 0 dòng chạm chỉ mục họ COPA. Trần 5% đạt ở A, B, C.

**Số dòng hiện hành:**

| | Phần riêng | Thành phần (nền + phần) | Nguồn lớn nhất |
|---|---|---|---|
| A | 4.396 (VSMRC 1.217, EXAMS 179, ý định du lịch 1.000, ViANLI 1.000, VNTC 1.000); 2.179 dòng dùng chuỗi tv1-* chờ duyệt | **24.349** (`components/A.jsonl`) | 5,0% |
| B | 6.944 (đọc hiểu 2.000, ý định 1.144, kiểm chứng 2.000, yes/no 800, nhân quả 1.000) | **26.897** (chờ bản dịch) | 3,0% |
| C | A + B | **31.293** (chờ bản dịch) | 3,9% |

**File cần dịch** (giao cho agent dịch): `results/train_v1/to_translate/`

- `part_b_to_translate.jsonl`: **6.944 dòng**, sha256 `9e279b91…`. 710 dòng noul không có lựa chọn cần dịch.
- `HUONG_DAN_DICH.md`: đã thêm mục nhân quả.
- `kiem_ban_dich.py`: đã thử lại trên bản giả. Bản đúng ĐẠT; bản lỗi bắt đủ 4 lỗi (thiếu id, lệch số lựa chọn, còn nguyên văn tiếng Anh, chưa NFC).
- `chuoi_lap_lai.json`.

**Khối lượng cần dịch: 1.005.294 token tiếng Anh**

| Thành phần dòng | Token |
|---|---|
| state | 825.629 |
| câu hỏi | 80.509 |
| lựa chọn | 99.156 |

- Tính theo chuỗi khác nhau: 863.298 token.
- Theo dạng bài:

  | Dạng bài | Token |
  |---|---|
  | đọc hiểu | 591.556 |
  | kiểm chứng | 173.586 |
  | ý định | 93.730 |
  | yes/no | 90.923 |
  | nhân quả | 55.499 |

- Bản tiếng Việt ước khoảng 1,4 triệu token (tỉ lệ state đo được 1,43).

**Độ dài và thời gian** (`stage0_lengths.json`): đỉnh bộ nhớ không đổi, khoảng 10,2 GB ở micro-batch 8. Thời gian 2 epoch ước:

| Thành phần | Giờ |
|---|---|
| A | 2,34 |
| B | 2,48 |
| C | 2,98 |

**Viết sẵn cho lúc bản dịch về (chưa chạy trên bản thật):**

- **(b)** `scripts/train_v1/make_review_html.py <bản_dịch>`: 30 dòng ngẫu nhiên (seed 20261017), 6 dòng mỗi dạng bài. Mỗi dòng có bản EN và VI cạnh nhau, đáp án tô xanh có dấu ✓, nhãn mềm ghi xác suất. Ra `results/train_v1/review/doc_kiem_ban_dich.html`. Đã thử trên bản giả: ra đúng 30 dòng. **Không train khi người dùng chưa xem.**
- **(c)** `scripts/train_v1/decontam_vi.py <bản_dịch>`, gồm 4 bước:
  1. chạy `kiem_ban_dich.py` (phải ĐẠT);
  2. dựng dòng tiếng Việt, với nhãn và target lấy từ `part_b_en.jsonl`;
  3. qua `finalize` của v0 với chỉ mục **tiếng Việt** mở rộng (toàn bộ văn bản VI của dev và test 1.3, kể cả không dấu, và các nguồn bench tiếng Việt), cộng train_v0/val_v0 và phần A;
  4. ghi `part_b_vi.jsonl`, `part_b_vi_loai.json`, `components/B.jsonl`, `components/C.jsonl`, rồi kiểm trần 5%.

  Đã thử trọn trên bản giả (chuỗi tiếng Anh gắn tiền tố): kiểm định dạng ĐẠT, 6.944/6.944 dòng giữ, B 26.897, C 31.293, không vượt trần. Bản giả không phải tiếng Việt nên lượt thử này không có dòng trùng để loại.

  Chỉ mục dựng bằng `scripts/train_v1/refs_index.py` (chép nguyên văn phần chỉ mục của `build_stage0.py`, có kiểm số văn bản trùng bản v1).

**Còn chờ:**
- người dùng duyệt 25 chuỗi tv1-* trên công cụ; sau đó dựng lại A, đọc `gia_tri_moi` trong edits_log;
- bản dịch phần B.

## Tóm tắt (bản v1)

- **Phần A (tiếng Việt bản địa): 4.736 dòng**, brief đề xuất 5.500.
  - VSMRC chỉ lấy được 1.236/2.000 dòng vì vướng trần 5% mỗi nguồn (mục 5).
  - 2.500 dòng (EXAMS, ý định du lịch, VNTC) dùng 25 chuỗi tiếng Việt mới do tôi đề xuất. Các chuỗi này đang **chờ người dùng duyệt** trong `config/vi_proposals.jsonl`.
- **Phần B (tiếng Anh cần dịch): 5.944 dòng**, brief đề xuất 6.300. Mảng ý định chỉ được 1.144/1.500 dòng: banking77, clinc_oos và 5/6 miền multidogo có hơn 10 lựa chọn, nên bộ lọc của v0 loại (mục 5).
- **Khử rò rỉ:**
  - Phần A và phần B: **0 dòng trùng** với chỉ mục mở rộng (dev 1.3, test 1.3 và mọi split của các nguồn bench).
  - train_v0: chỉ mục mới bắt 47 dòng (lần kiểm v0 cũ chỉ ra 2).
    - 8 dòng tiếng Việt: đã bỏ khỏi phần nền.
    - 39 dòng tiếng Anh: **giữ lại** vì quy tắc "giữ nguyên 16.000 dòng EN", nên trái với yêu cầu "bản train mới phải bằng 0". Cần quyết (mục 9).
- **Thành phần:**

  | Thành phần | Cách dựng | Số dòng | Ghi chú |
  |---|---|---|---|
  | A | nền + phần A | 24.728 | dựng xong |
  | B | nền + phần B | 25.936 | chờ bản dịch |
  | C | nền + A + B | 30.672 | chờ bản dịch |

  Nền là train_v0 bỏ 8 dòng tiếng Việt, còn 19.992 dòng, gồm đủ 16.000 dòng tasksource.
- **File cần dịch:** `to_translate/part_b_to_translate.jsonl`, 5.944 dòng, khoảng **950 nghìn token tiếng Anh**.
  - Tính theo chuỗi khác nhau thì còn khoảng 823 nghìn token, vì câu hỏi mẫu và nhãn lặp lại nhiều.
  - Kèm `HUONG_DAN_DICH.md` và `kiem_ban_dich.py`.
- **Nguồn nhân quả ngoài họ COPA** (chỉ báo, chưa đưa vào): **e-CARE** và **HeadlineCause**, đều có split train (mục 8). Cả hai có sẵn trong tasksource và đã có mặt với số lượng nhỏ trong train_v0.

## 1. Cách làm

- **Recast dùng chung với train_v0.**
  - `ts_row`, `pl_row` và các bộ lọc của `finalize` được chép nguyên văn từ `scripts/build_train_v0.py` sang `scripts/train_v1/recast_v0.py`. Lý do: file gốc chạy ngay khi import và ghi đè `data/train_v0.jsonl`.
  - `scripts/train_v1/verify_recast_v0.py` dựng lại **21.000/21.000** dòng train_v0 + val_v0 từ nguồn, trùng từng trường. Kết quả ở `verify_recast_v0.json`.
  - Bộ lọc giữ như v0:
    - 2–10 lựa chọn;
    - target là phân phối;
    - state không rỗng;
    - lựa chọn không trích chữ cái của lựa chọn khác;
    - **state ≤ 1.024 token** (tokenizer Qwen3, như v0).
- **Nguồn mới** (VSMRC, EXAMS, ý định du lịch, VNTC) dựng theo đúng lược đồ dòng của v0 rồi qua cùng `finalize`.
  - ViANLI đi qua `pl_row`, như ViNLI ở v0. File nguồn là `sources/vianli/train.jsonl` của PhoLaya-base, revision của v0.
- **Ghim revision:**
  - mọi nguồn và bộ tham chiếu đều tải bằng commit sha;
  - các bộ chỉ có bản parquet tự chuyển thì ghim sha của `refs/convert/parquet`;
  - sha256 từng file tham chiếu ở `stage0_stats.json` (`ref_files`).
- **Chọn dòng:**
  - xáo cả kho theo seed 20261012, lấy N dòng đầu tiên qua được bộ lọc;
  - mỗi nhóm văn bản (đoạn văn, tiền đề) lấy một dòng;
  - không trùng cặp state + câu hỏi trong cùng một phần.
- **Phần B:** tasksource @ `bf1e4a87` (revision của v0).
  - Bỏ dòng đã dùng ở train_v0/val_v0, theo cả `id` lẫn `group_id`.
  - Chia đều các nguồn trong một dạng bài; nguồn nào hết dòng hoặc chạm trần thì nhường phần cho nguồn khác.
- **Trần 5% mỗi nguồn**, mức tác vụ như v0, tính trên thành phần nhỏ nhất chứa nguồn đó: A cho phần A, B cho phần B. C lớn hơn nên tự đạt. Đã kiểm bằng assert:

  | Thành phần | Nguồn chiếm nhiều nhất |
  |---|---|
  | A | 5,0% (VSMRC, đúng trần) |
  | B | 3,1% |
  | C | 4,0% |

- **Script** (thư mục `scripts/train_v1/`):
  - `fetch_refs.py`, `fetch_sources.py`: tải;
  - `build_stage0.py`: dựng;
  - `stats_stage0.py`: độ dài token;
  - `diag_short_hits.py`: chẩn đoán;
  - `add_proposals.py`: đề xuất chuỗi.
- **Log:** `log_build_stage0.txt`, `log_stats_stage0.txt`.

## 2. Phần A: tiếng Việt bản địa (4.736 dòng)

| Nguồn (revision) | Dạng bài | Kiểu | Giấy phép (`license_use`) | Nguồn nhãn | Dòng | Đề xuất brief |
|---|---|---|---|---|---|---|
| VSMRC/mrc @ `13c4449c` | đọc hiểu | choice (4) | apache-2.0 trên thẻ; văn bản Wikipedia vốn CC BY-SA | **LLM sinh** (người kiểm) | 1.236 | 2.000 |
| mhardalov/exams, crosslingual_vi @ `4ff10804` | đọc hiểu / kiến thức | choice (4) | CC BY-SA 4.0 | người gán (đề thi thật) | 500 | 500 |
| phamluan/vi-tourism-admin-intent @ `a092e219` | ý định | choice (8) | CC BY 4.0 | **mẫu câu** | 1.000 | 1.000 |
| PhoLaya-base `sources/vianli` @ `034d4273` (entailment) | yes/no | noul | research-only (PhoLaya, UIT); ViANLI gốc: thẻ mit, thân card CC BY-NC-SA 4.0, **mâu thuẫn** | người gán (vòng đối kháng người–máy) | 1.000 | 1.000 |
| kornwtp/VNTC-10Topics @ `30b4f927` | chủ đề | choice (10) | **không rõ**: thẻ mit trên bản sao HF; văn bản là tin báo 2007 | người gán (chuyên mục báo) | 1.000 | 1.000 |

- Mỗi dòng có các trường:
  - `license_verbatim`, `license_use`;
  - `nguon_nhan`: `nguoi_gan` / `llm_sinh` / `mau_cau`, kèm `nguon_nhan_ghi_chu`;
  - `dang_bai`, `dataset_revision`;
  - `chuoi_cho_duyet`: id các đề xuất chuỗi mà dòng đó dùng;
  - `phan: "A"`.
- **Chuỗi tiếng Việt mới: 25 đề xuất chờ duyệt** (`config/vi_proposals.jsonl`, id `tv1-*`):
  - 8 nhãn ý định: nguồn chỉ có mã tiếng Anh như `ask_new_name`;
  - 10 nhãn chủ đề VNTC: nguồn viết không dấu, như "Chinh tri Xa hoi";
  - mẫu state của EXAMS "Môn {mon}, lớp {lop}" và 6 tên môn: nguồn ghi tiếng Anh, như "Biology".
  - Câu hỏi **không** đề xuất mới. Ý định dùng lại "Người dùng muốn trợ lý làm gì?" (MASSIVE), chủ đề dùng lại "Văn bản này thuộc chủ đề nào?" (SIB-200). Cả hai đã có trong `config/translations.json` của bench.
  - **2.500 dòng phụ thuộc các chuỗi này**, nên thành phần A chưa train được cho tới khi người dùng duyệt (hoặc bỏ các dòng đó).
- Ghi chú từng nguồn:
  - **Ý định du lịch:** bỏ 334 câu `label_source = keyword_weak` (câu người viết nhưng nhãn do bộ phân loại từ khóa gán), chỉ giữ câu sinh từ mẫu. Một phần câu nguồn viết không dấu; giữ nguyên.
  - **VNTC:** văn bản nguồn đã bị chữ thường hóa, bỏ dấu câu và bỏ chữ số (ví dụ "gặp clb mexican ngày và houston ngày"). Bộ lọc v0 loại 125 bài dài hơn 1.024 token, nên mẫu lệch về bài ngắn. Đọc kỹ trước khi dùng.
  - **EXAMS:** không có đoạn văn, nên state chỉ ghi môn và lớp; câu hỏi và lựa chọn giữ nguyên văn đề thi.
  - **ViANLI:** 28 dòng chưa ở dạng NFC, do `pl_row` của v0 không chuẩn hóa. train_v0 cũng có 189 dòng PhoLaya như vậy. Giữ nguyên để module không đổi.

## 3. Phần B: tiếng Anh cần dịch (5.944 dòng, tasksource @ `bf1e4a87`)

| Dạng bài | Nguồn | Giấy phép (`license_use` / thẻ) | Dòng | Kiểu |
|---|---|---|---|---|
| đọc hiểu (2.000/2.000) | race/high | unspecified / other | 400 | 386 choice, 14 noul |
| | race/middle | unspecified / other | 400 | |
| | quail | **non-commercial** / cc-by-nc-sa-4.0 | 400 | |
| | cosmos_qa | commercial / cc-by-4.0 | 400 | |
| | dream | unspecified | 400 | |
| ý định (**1.144/1.500**) | multidogo/insurance/intent | commercial / cdla-permissive-1.0 | 793 | choice (10) |
| | banking77 | commercial / cc-by-4.0 | 215 | **chỉ noul** |
| | clinc_oos/plus | commercial / cc-by-3.0 | 136 | **chỉ noul** |
| kiểm chứng (2.000/2.000) | vitaminc | commercial / cc-by-sa-3.0 | 667 | |
| | nli_fever | unspecified | 667 | |
| | health_fact | commercial / mit | 666 | |
| yes/no (800/800) | super_glue/boolq | unspecified / other | 267 | |
| | super_glue/boolq_passage | unspecified / other | 267 | |
| | sharc | unspecified | 266 | |

- **Theo kiểu:** 5.233 choice, 668 noul, 43 score.
- **Nguồn nhãn:** người gán, theo bộ gốc. Câu hỏi là câu mẫu của tasksource.
- **Không dùng** IntentGrasp, SLURP, MASSIVE hay họ COPA/XCOPA; đã kiểm bằng assert trên tên nguồn.
- **Cách hiểu tên nguồn của brief:**
  - "race" = race/high + race/middle (race-c là bộ khác, không lấy);
  - "boolq" = super_glue/boolq + super_glue/boolq_passage (không lấy boolq-natural-perturbations);
  - "multidogo" = 6 miền.

## 4. File cần dịch

| File | Nội dung |
|---|---|
| `to_translate/part_b_to_translate.jsonl` | 5.944 dòng, chỉ gồm `id`, `state`, `question`, `options` (đúng thứ tự). sha256 `352fef83…` |
| `to_translate/HUONG_DAN_DICH.md` | quy tắc: giữ số và thứ tự lựa chọn; không đổi nghĩa; không thêm, bớt; tên riêng và số giữ nguyên; tiếng Việt có dấu NFC; trả về đúng định dạng và id. Có lưu ý cho câu mẫu lặp, nhãn cố định, nhãn dạng mã, hội thoại |
| `to_translate/chuoi_lap_lai.json` | câu hỏi và lựa chọn lặp từ 5 lần trở lên, để dịch nhất quán |
| `to_translate/kiem_ban_dich.py` | script kiểm bản dịch trả về; chỉ cần thư viện chuẩn Python |

- **668 dòng noul** có `options: []`. Lựa chọn Có/Không cố định lấy từ `config/train_format.json`, không cần dịch.
- **Script kiểm** báo lỗi khi:
  - thiếu, thừa hoặc lặp id;
  - thiếu trường hoặc sai kiểu;
  - số lựa chọn lệch;
  - có chuỗi rỗng;
  - chuỗi chưa ở dạng NFC;
  - còn nguyên văn tiếng Anh;
  - hai lựa chọn trùng nhau.

  Script chỉ cảnh báo khi chuỗi lặp được dịch không nhất quán, độ dài lệch bất thường, hoặc mất số.
  - Đã thử trên bản giả "đạt" và bản giả có lỗi; script bắt đúng các lỗi.
  - Phát hiện kèm: **65 dòng nguồn tiếng Anh vốn chưa ở dạng NFC**. Hướng dẫn đã ghi rằng bản dịch phải NFC.
- **Nhãn, target và metadata** để ở file riêng, ngoài thư mục `to_translate/`:
  - `part_b_meta.jsonl`: id, nguồn, kiểu, target, giấy phép, nguồn nhãn, dạng bài;
  - `part_b_en.jsonl`: dòng tiếng Anh đầy đủ.
- **Khối lượng cần dịch** (token Qwen3):

  | Phần | Token tiếng Anh |
  |---|---|
  | state | 795.221 |
  | câu hỏi | 70.578 |
  | lựa chọn | 83.996 |
  | **Tổng** | **949.795** |

  - Chuỗi khác nhau: 12.526 chuỗi, 823.354 token.
  - Câu hỏi khác nhau: 358. Lựa chọn khác nhau: 6.383.
- **Ước bản tiếng Việt.** Tỉ lệ token Việt/Anh **đo** trên 987 cặp song song của dev 1.3: trung vị 1,43 cho state, 1,28 cho cả prompt. Như vậy đầu ra vào khoảng 1,3–1,4 triệu token.
  - Khoảng 13 dòng có state tiếng Việt ước vượt 1.024 token; các dòng này sẽ bị loại khi nhận bản dịch.

## 5. Khử rò rỉ

### 5.1 Tham chiếu

Phần A so với chỉ mục tiếng Việt, phần B so với chỉ mục tiếng Anh.

- **Bench 1.3:** toàn bộ dev 1.3 và test 1.3, mọi bản (EN, VI, VI không dấu).
  - Lấy state, câu hỏi và lựa chọn.
  - Bỏ 400 chuỗi mẫu lặp từ 3 lần trở lên, như câu hỏi cố định hay nhãn chủ đề/ý định.
- **Mọi split của:**
  - Belebele (vie, eng);
  - **FLORES:** `facebook/flores` và `openlanguagedata/flores_plus` đều có cổng truy cập, nên không tải. Thay bằng `gsarti/flores_101` (parquet @ `1a45e707`, dev + devtest, vie/eng, không khóa). Thêm nữa, Belebele và SIB-200 vốn chứa văn bản FLORES-200;
  - SIB-200, MASSIVE (vi, en);
  - XCOPA (vi + translation-vi), COPA (SuperGLUE), Balanced-COPA;
  - ViFactCheck, ViWikiFC;
  - MLQA: mọi cấu hình có tiếng Việt, cộng en.en, translate-train.vi và translate-test.vi;
  - Global-PIQA: nonparallel (cả data_v0p1 và unsampled_full) và parallel, vie và eng.
- **Văn bản đã dùng:** train_v0 + val_v0.
- **Quy mô chỉ mục:** 196.204 văn bản tiếng Việt và 59.459 văn bản tiếng Anh, sau khi bỏ trùng; cộng 5.382 + 3.424 lựa chọn đáp án Belebele.

### 5.2 Phương pháp và một chỉnh sửa giữa chừng

`scripts/decontam.py`, giữ nguyên:
- trùng nguyên văn;
- MinHash Jaccard ≥ 0,8;
- chứa 13-gram;
- văn bản tham chiếu 4–12 từ nằm trọn trong văn bản train.

Phép kiểm áp lên state, câu hỏi và từng lựa chọn có từ 4 từ trở lên (v0 chỉ kiểm state).

**Lượt 1 bắt nhầm cụm từ chung.** Ở lượt 1 tôi đưa cả lựa chọn đáp án Belebele vào phép "chứa đoạn 4–12 từ". Kết quả là train_v0 có 241 dòng bị bắt, phần lớn vì cụm từ chung trong lựa chọn, không phải rò rỉ:

| Cụm từ | Số dòng bị bắt |
|---|---|
| "in the united states" | 108 |
| "a lot of money" | 13 |
| "as soon as possible" | 10 |

Chi tiết ở `diag_v0_hits.json` (của lượt 1).

**Lượt 2 (bản chính thức).** Lựa chọn đáp án Belebele (của bench và của nguồn) chỉ còn chịu trùng nguyên văn, MinHash và 13-gram. State, câu hỏi, câu XCOPA và mọi văn bản khác vẫn giữ đủ bốn phép. Brief chỉ yêu cầu ba phép đầu, nên bản chính thức vẫn chặt hơn brief.

Số liệu lượt 1 lưu ở `stage0_stats_luot1.json` và `v0_leak_luot1.json`.

### 5.3 Số dòng bị loại theo nguồn (lượt 2)

Số trong ngoặc là của lượt 1.

| Phần | Nguồn | Trùng tham chiếu | Trùng train_v0/val_v0 | Lọc khác |
|---|---|---|---|---|
| A | VSMRC | 57 (125) | 0 | 2 trích chữ cái lựa chọn; 2 cùng đoạn văn |
| A | EXAMS | 3 (7) | 0 | 2 cùng câu hỏi |
| A | ý định du lịch | 0 | 0 | 334 keyword_weak (bỏ trước); 1 trùng |
| A | ViANLI | 11 (18) | 5 | 138 cùng tiền đề |
| A | VNTC | 39 (134) | 0 | 125 state > 1.024 token |
| B | race/high | 11 (38) | 11 | |
| B | race/middle | 29 (47) | 16 | 1 |
| B | quail | 6 (29) | 93 | |
| B | cosmos_qa | 7 (7) | 5 | 1 |
| B | dream | 61 (66) | 8 | 1 state dài |
| B | banking77 | 4 | 0 | **4.472 dòng hơn 10 lựa chọn** |
| B | clinc_oos/plus | 7 | 0 | **3.788 dòng hơn 10 lựa chọn** |
| B | multidogo (5 miền) | 0 | 0 | **5.000 dòng hơn 10 lựa chọn** |
| B | multidogo/insurance | 19 | 89 | 79 trùng trong phần B |
| B | vitaminc | 17 (43) | 3 | 5 |
| B | nli_fever | 37 (60) | 26 | |
| B | health_fact | 0 (6) | 4 | 3 |
| B | boolq / boolq_passage | 0 / 3 (3 / 35) | 1 / 0 | 2 / 6 |
| B | sharc | 0 (2) | 109 | 1 |

- Trùng ở phần B chủ yếu là chứa đoạn ngắn của MASSIVE (hội thoại dream) và 13-gram của MLQA en.en (vitaminc, nli_fever: văn bản Wikipedia chung).
- Trùng train_v0 cao ở quail và sharc, vì cùng đoạn văn với dòng đã dùng ở v0.
- Chi tiết theo từng loại trùng và tag ở `stage0_stats.json` (`partA.drops`, `partB.drops`).
- **Kiểm cuối:** phần A và phần B có **0** dòng chạm chỉ mục (assert trong `build_stage0.py`).

### 5.4 train_v0 và val_v0 so với chỉ mục mở rộng

Kết quả ở `v0_leak.json`: 50 dòng chạm chỉ mục, gồm 47 ở train và 3 ở val.

- **8 dòng train tiếng Việt** (7 ViNLI, 1 ViSFD): **đã bỏ khỏi phần nền** của cả A, B, C. Gồm 2 dòng trùng ViFactCheck test đã biết từ `leakage_v1_3.json`, cùng các dòng trùng MLQA/ViFactCheck/ViWikiFC mới phát hiện. Phần VI của nền còn 3.992 dòng.
- **39 dòng train tiếng Anh:** **giữ lại**, theo quy tắc giữ nguyên 16.000 dòng EN, vì vậy trái với yêu cầu "bản train mới phải bằng 0".

  | Loại trùng | Số dòng |
  |---|---|
  | MLQA en.en / translate-test (12 dòng 13-gram, 9 dòng đoạn ngắn) | 21 |
  | câu XCOPA translation-vi trong văn bản | 6 |
  | MASSIVE | 6 |
  | Global-PIQA parallel | 3 |
  | Balanced-COPA | 1 |
  | trùng nguyên văn chuỗi 1–2 từ ("car", "Too expensive?") | 2 |

  Phần lớn không phải rò rỉ câu bench, mà là văn bản Wikipedia hay cụm câu ngắn chung. Danh sách id ở `v0_leak.json` (`giu_lai_vi_quy_tac_16000_EN`).
- val_v0 có 3 dòng chạm; không sửa val_v0.

## 6. Thành phần đã dựng (chưa train)

| Thành phần | File | Dòng | Nguồn lớn nhất | Trạng thái |
|---|---|---|---|---|
| A = nền + phần A | `components/A.jsonl` (sha `570582ad…`) | 24.728 | 5,0% | đủ dòng; 2.500 dòng chờ duyệt chuỗi |
| B = nền + phần B | `components/B_plan.json` | 25.936 | 3,1% | chờ bản dịch |
| C = nền + A + B | `components/C_plan.json` | 30.672 | 4,0% | chờ bản dịch |

- **Nền** = train_v0 (sha `d10c9362…`) bỏ 8 dòng ở mục 5.4, còn 19.992 dòng, giữ đủ 16.000 dòng tasksource.
- Dòng train giữ thứ tự lựa chọn của nguồn; vòng train tự xáo lại mỗi lần nạp, như v0.
- **Không có val mới:** val_v0 giữ nguyên.
- B và C dựng thành file đầy đủ sau khi nhận bản dịch, kiểm định dạng và khử rò rỉ phía tiếng Việt. Nhãn và target lấy từ `part_b_meta.jsonl`.

## 7. Độ dài token, micro-batch và bộ nhớ

Kết quả ở `stage0_lengths.json`. Prompt dựng đúng như lúc train (`render_prompt` + `prompt_frames.json`), tokenizer Qwen3-1.7B @ `70d244cc`. Riêng phần B tiếng Việt là ước lượng: số token tiếng Anh × 1,28 (tỉ lệ đo được).

| Thành phần | Dòng | Token TB | p50 | p90 | p99 | max | > 512 | > 1.024 |
|---|---|---|---|---|---|---|---|---|
| nền v0 | 19.992 | 215 | 135 | 492 | 954 | 2.636 | 1.831 | 108 |
| A | 24.728 | 229 | 152 | 492 | 956 | 2.636 | 2.269 | 139 |
| B (ước) | 25.936 | 227 | 150 | 531 | 934 | 2.636 | 2.765 | 119 |
| C (ước) | 30.672 | 236 | 163 | 526 | 938 | 2.636 | 3.203 | 150 |

- **Theo nguồn (p50 / max):**
  - Phần A: VNTC 473 / 1.128; VSMRC 322 / 499; EXAMS 145 / 554; ý định 207 / 226; ViANLI 135 / 262.
  - Phần B, bản tiếng Anh: quail 491 / 627; race/high 471 / 1.088; dream 146 / 951.
- Prompt dài nhất (2.636 token) nằm sẵn trong train_v0.
- **Mô phỏng chia batch** theo đúng `train_lora.py` (batch hiệu dụng 32 sắp theo độ dài, micro-batch 8, seed 1 và 2): micro-batch lớn nhất của A, B, C **bằng của v0**, khoảng 21 nghìn token.
- **Bộ nhớ (ước):** vẫn khoảng **10,2 GB ở micro-batch 8**, như lần train v0 đo được 10,25 GB. Micro-batch 4 khoảng 7 GB. **Không cần giảm micro-batch.**
- **Thời gian 2 epoch (ước):**

  | Thành phần | Giờ |
  |---|---|
  | A | 2,35 |
  | B | 2,4 |
  | C | 2,9 |

  Ước bằng cách lấy thời gian v0 (6.619 giây) nhân tỉ lệ tổng token đã đệm. Brief ước khoảng 2,8 giờ mỗi bản.
- Các con số bộ nhớ và thời gian là **ước lượng thô**, giả định phần cố định khoảng 3,7 GB, phần còn lại tỉ lệ với số token.

## 8. Nguồn nhân quả không thuộc họ COPA/XCOPA (chỉ báo, chưa đưa vào)

| Nguồn | Dạng | Split train | Giấy phép | Trong tasksource @ v0 | Đã có trong train_v0 |
|---|---|---|---|---|---|
| **e-CARE** (Du và cộng sự 2022; GitHub waste-wood/e-care; HF `12ml/e-CARE` @ `a9c78871`) | tiền đề + hỏi nguyên nhân/kết quả + 2 giả thuyết. **Cùng khuôn COPA**, nhưng dữ liệu viết độc lập: 21.324 câu, train 14.928, dev 2.132, test ẩn | có | GitHub: MIT; bản HF: không có thẻ; tasksource ghi "BSD 2-Clause (DPI)" | 770 dòng | 20 |
| **HeadlineCause** (`IlyaGusev/headline_cause` @ `68c78423`) | cặp tiêu đề tin, phân loại quan hệ nhân quả; en/ru. **Không** cùng khuôn COPA | có | CC0-1.0 | 1.808 dòng (en_simple) | 32 |

- e-CARE gần XCOPA nhất về dạng bài. Cần người dùng quyết có coi nó là "họ COPA" không: khuôn giống, dữ liệu khác. Nếu dùng, phải khử rò rỉ với COPA/XCOPA (đã có trong chỉ mục).
- Cả hai là tiếng Anh, nên sẽ phải qua agent dịch.
- Brief mục 3.3 viết rằng tasksource "chỉ có nguồn nhân quả thuộc họ XCOPA". Thực ra tasksource @ v0 còn có e-CARE, HeadlineCause, WIQA (1.448 dòng) và corr2cause (6.103 dòng), và train_v0 đã dùng một ít mỗi nguồn.

## 9. Những gì không làm được như brief, và các điểm cần quyết

1. **VSMRC 1.236/2.000** vì trần 5%. Nếu muốn 2.000, có thể coi mỗi loại câu hỏi của VSMRC (fact_check, reasoning, list, fill_blank) là một nguồn ở mức tác vụ, như PhoLaya tách theo tác vụ. Tôi không tự chọn cách này.
2. **Ý định ở phần B: 1.144/1.500.**
   - banking77 (77 nhãn), clinc_oos (151 nhãn) và 5 miền multidogo (11–17 nhãn) có hơn 10 lựa chọn, nên `finalize` của v0 loại.
   - Chỉ còn dòng noul của banking77/clinc và multidogo/insurance (10 nhãn).
   - Muốn đủ số thì phải recast mới (lấy nhãn đúng + 9 nhiễu), tức là khác module v0. Tôi chưa làm.
3. **25 chuỗi tiếng Việt chờ duyệt.** 2.500 dòng của phần A dùng chúng; A chỉ train được sau khi người dùng duyệt (hoặc bỏ 2.500 dòng đó).
   - `components/A.jsonl` hiện dùng giá trị **đề xuất** (`gia_tri_de_xuat`).
   - Khi người dùng "sửa rồi duyệt" trên công cụ, giá trị cuối được ghi vào `data/human_edits/edits_log.jsonl` (bản ghi `de_xuat`, trường `gia_tri_moi`), không ghi ngược vào `vi_proposals.jsonl`.
   - Vì vậy **A phải dựng lại sau khi duyệt**, đọc giá trị đã duyệt từ edits_log.
4. **"Giữ nguyên 16.000 dòng EN" mâu thuẫn với "bản train mới phải bằng 0 trùng"** ở 39 dòng tiếng Anh của v0 (mục 5.4). Hiện đang giữ. Bỏ 39 dòng này thì nền còn 15.961 dòng EN.
5. **FLORES gốc có cổng truy cập**, nên không xin quyền; dùng `gsarti/flores_101` thay. Belebele và SIB-200 đã phủ văn bản FLORES-200.
6. **VNTC chất lượng văn bản thấp** (mất dấu câu, chữ hoa, chữ số) và giấy phép không rõ. Đã đưa vào theo brief, nhưng nên cân nhắc.
7. **Cách khử rò rỉ đổi một chỗ so với lượt 1** (mục 5.2): lựa chọn đáp án Belebele không xét chứa đoạn ngắn. Cả hai lượt đều lưu số.
8. **Không đổi so với v0:**
   - bản không dấu vẫn do vòng train sinh (`strip_p`), không dựng sẵn;
   - dòng tiếng Việt không chuẩn hóa NFC thêm (28 dòng ViANLI, như 189 dòng v0);
   - dòng v0 không gắn cờ nguồn nhãn, để giữ nguyên từng byte.
9. **Brief mục 3.3** về nguồn nhân quả trong tasksource chưa chính xác (mục 8).

## 10. File

Tất cả nằm trong `results/train_v1/`, trừ đề xuất chuỗi:

- **Dữ liệu:** `part_a.jsonl`, `part_b_en.jsonl`, `part_b_meta.jsonl`, `components/A.jsonl`, `components/B_plan.json`, `components/C_plan.json`.
- **Cho agent dịch:** `to_translate/part_b_to_translate.jsonl`, `HUONG_DAN_DICH.md`, `chuoi_lap_lai.json`, `kiem_ban_dich.py` (cùng thư mục `to_translate/`).
- **Kiểm và số liệu:**
  - `verify_recast_v0.json`, `v0_leak.json`, `diag_v0_hits.json`;
  - `stage0_stats.json`, `stage0_lengths.json`;
  - số liệu lượt 1: `*_luot1.json`;
  - `inspect_sources.json`, `inspect_refs.json`, `refs_schema.json`, `sources_schema.json`.
- **Dữ liệu tải về:** `raw_refs/` (bộ tham chiếu, gồm cả ViFactCheck và ViWikiFC của UIT; chỉ ở máy này), `raw_src/` (nguồn phần A).
- **Đề xuất chuỗi:** `config/vi_proposals.jsonl`, id `tv1-*` (25 dòng mới, chờ duyệt).
