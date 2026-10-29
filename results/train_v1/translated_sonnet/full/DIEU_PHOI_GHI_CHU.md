# Ghi chú điều phối dịch đầy đủ phần B (để chạy tiếp nếu phiên bị tóm tắt)

Yêu cầu của điều phối ("Quản lý", 20/10):
- Sonnet 1 dịch, Sonnet 2 đọc lại độc lập, theo `HUONG_DAN_DICH_v2` và `bang_thuat_ngu_v2`.
- Lô 40 dòng; mỗi lần gọi subagent làm 1 đợt = 4 lô (điều phối đã đồng ý).
- **Kiểm mức dùng (`get_usage`) trước mỗi lần mở đợt:**
  - cửa sổ 5 giờ từ 88% thì không mở đợt mới; khi đó ghi điểm dừng, báo Quản lý, chờ reset;
  - từ 92% thì dừng hẳn;
  - hạn mức **tuần từ 85%** thì không mở đợt mới, chờ reset tuần (21/10 khoảng 10:00);
  - không bao giờ dùng mức dùng thêm.
- Không đọc `results/train_v1/translated/` (bản Google). Không train, không commit, không xóa file, không đụng `data/`; chỉ đọc `config/`.
- **Báo Quản lý khi:**
  - (b) mỗi lần dừng vì chạm ngưỡng;
  - (c) xong toàn bộ.

## Cách gọi subagent

Model sonnet, chạy nền. Nội dung gọi:

- Dịch: `Đọc và làm đúng theo chỉ dẫn trong file C:\Users\dungbt06\Fine-tune\results\train_v1\translated_sonnet\full\PROMPT_dich.md cho các lô: NNN, NNN, NNN, NNN. Trả lời đúng một dòng như chỉ dẫn.`
- Đọc lại: như trên, thay bằng `PROMPT_doc_lai.md`.

**Tiến độ:** `python scripts/train_v1/full_status.py --nhanh` (ghi `full/trang_thai.json`). Danh sách lô của từng đợt ở `full/plan.json`, gồm 43 đợt.

**Ghi nhật ký** sau mỗi sự kiện:
- `full/muc_dung.jsonl`: mốc, thời điểm, % tuần, % 5 giờ;
- `full/token_subagent.jsonl`: bước, lô, token, ms, ghi chú.

## Khi xong cả 43 đợt (mọi lô có s2)

1. `python scripts/train_v1/full_assemble.py`: ghép 6.944 dòng, chạy `kiem_ban_dich` v2.1; kết quả phải ĐẠT.
2. `bash setup/run.sh scripts/train_v1/decontam_vi.py results/train_v1/translated_sonnet/full/ban_dich_s2.jsonl` (venv WSL): khử rò rỉ phía tiếng Việt, ghi `components/B.jsonl` và `C.jsonl`, kiểm trần 5%.
3. `python scripts/train_v1/make_review_html.py results/train_v1/translated_sonnet/full/ban_dich_s2.jsonl --out results/train_v1/translated_sonnet/full/doc_kiem_30.html`.
4. Báo cáo `results/train_v1/translated_sonnet/REPORT_dich_day_du.md`: số dòng sửa theo loại, mức dùng theo cửa sổ, thời gian. Ghi `PLAN_decisions`. Báo Quản lý. DỪNG, không train.

## Sự cố đã gặp (đưa vào báo cáo)

- **Đợt 02 (lô 007):** bị chặn "Output blocked by content filtering policy"; chạy lại lô 007–008 thì thành công.
- **Thư mục scratchpad dùng chung:** script phụ có thể bị agent khác ghi đè (đợt 05, 06). Từ đợt 10 trở đi, chỉ dẫn bắt buộc dùng thư mục con riêng.
- **Đợt 03:** người dịch tự mở bản dịch các lô 001–006 (`_s1`) để theo mẫu câu hỏi. Các file đó không có nhãn.
- **Đọc lại đợt 06:** sửa 40/160 dòng, 19 chỗ sai thuật ngữ. Người dịch đợt đó áp bảng thuật ngữ kém; người đọc lại đã sửa.
- **Lô 072:** người dịch thêm chữ "tên" để qua phép kiểm "còn nguyên văn". Đã sửa script kiểm lên v2.1 (đoạn giống hệt bản gốc chỉ là lỗi khi có từ 2 hư từ tiếng Anh trở lên); người đọc lại đã bỏ chữ thêm.
- **Lô 060:** "higud mrng" được dịch thành "chào buổi sáng". Đây là lời chào, chấp nhận được.
- **Đợt 14 và 20:** người đọc lại không sửa dòng nào (dòng ngắn hoặc nhãn theo bảng thuật ngữ).
- Mốc đầu: 13:38, 5 giờ 12%, tuần 40%. Cửa sổ 5 giờ reset khoảng 16:20 (09:20Z).
