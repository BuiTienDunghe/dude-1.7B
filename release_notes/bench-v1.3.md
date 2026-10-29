## bench-v1.3 — Benchmark tiếng Việt 1.3 (đóng băng 17/10/2026)

> Benchmark dùng để chấm mọi phiên bản model ở trên. **Repo không chứa dữ liệu bench**, chỉ có mã băm và cách dựng.
> **Test chưa chạy model nào.** Test chỉ được chấm một lần ở cuối, theo giao thức ghi trước.

### Thành phần dev 1.3
`data/bench_v1_3/dev_v1_3.jsonl`: 3.847 dòng, sha256 `c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e`.

| Nguồn | Loại | VI có dấu | VI không dấu | EN song song |
|---|---|---|---|---|
| ViFactCheck | kiểm chứng (choice + Có/Không) | 158 | 158 | — |
| ViWikiFC | kiểm chứng (choice + Có/Không) | 285 | 285 | — |
| SIB-200 | chủ đề | 299 | 299 | 299 |
| MASSIVE | ý định | 294 | 294 | 294 |
| Belebele | đọc hiểu | 294 | 294 | 294 |
| XCOPA | nhân quả | 100 | 100 | 100 |

- **Điểm** là accuracy, trung bình đều theo nguồn (macro). Một câu XCOPA đáng 0,167 điểm macro; ViFactCheck 0,105; ViWikiFC 0,058; các nguồn còn lại khoảng 0,057.
- **Cách chấm:** model chọn chữ cái đáp án. Điểm lấy từ logit các token chữ cái, không sinh văn bản.

### Lịch sử phiên bản
| Bản | Ngày | Thay đổi | sha256 test |
|---|---|---|---|
| 1.0 | 12–13/10 | Lõi mở: Belebele, MASSIVE, XCOPA, SIB-200 (song song) và ViFactCheck, ViWikiFC, ViHSD (bản địa). Mọi dòng VI có bản không dấu | `d92d31f6…0d47` |
| 1.1 | 13/10 | Bỏ ViHSD. Sửa câu chữ kiểm chứng và 8 mô tả ý định MASSIVE, có người làm dự án duyệt | `fbad818b…` |
| 1.2 | 16/10 | Người soát toàn bộ (Opus soát dịch và đáp án, người dùng dịch lại một số đoạn Belebele); 144 quyết định nhãn; NFC; dev sha `ef88be16…` | `2bdbd210…` |
| **1.3** | **17/10** | Loại 2 cặp, đánh dấu 26 cặp mơ hồ (đọc mù 134 cặp). MLQA dời sang 1.4. **Test giữ nguyên từng byte như 1.2** | `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da` |

**Kiểm khi đóng băng:** dev↔test 0 trùng; train_v0 chạm test 2 / dev 0; không dấu 0 lệch; `release_check` đạt.

### Điểm tham chiếu trên dev 1.3 (VI có dấu)
| Model | VI có dấu | Không dấu | Ghi chú |
|---|---|---|---|
| Quyet-1.0-Medium | 89,6 | — | 4,66B tham số, khoảng 88 ms: quá chậm cho mục tiêu |
| decider-2b-text | 86,50 | 49,64 | **mốc cần vượt** |
| 2b, TB 2 seed (v0.3) | 84,38 | 62,85 | |
| C (v0.2) | 81,65 | 63,97 | |
| M1 (v0.1) | 79,88 | 62,37 | |
| Qwen3-1.7B gốc | 70,64 | 48,45 | |
