# Bối cảnh chung cho nghiên cứu hướng đi (27/10/2026)

Tài liệu này là bối cảnh dùng chung cho các agent nghiên cứu. Chỉ có dữ kiện, không có kết luận của điều phối.

## Dự án
- Mục tiêu: một **decision model tiếng Việt** nhỏ, chạy cục bộ, dùng **từng yêu cầu một (batch 1)**. Ưu tiên tiếng Việt, vẫn giữ tiếng Anh.
- Cách chuyển LLM thành decision model:
  - prompt qua chat template, `enable_thinking=False`;
  - các lựa chọn ghi là chữ A–J, mỗi chữ đúng 1 token;
  - đọc logit của token kế tiếp, lấy softmax trên k chữ cái.
- Ba kiểu câu: chọn một (choice), Có/Không (noul), thang điểm (score).
- Hiện tại: **Qwen3-1.7B + LoRA**.
  - LoRA r16/α32, mọi lớp proj, lr 2e-4, batch hiệu dụng 32, 2 epoch, seed 1.
  - Loss: CE trên toàn từ vựng, đích mềm đặt trên các token chữ cái ("full_vocab").
- Code: `scripts/common.py` (dựng prompt), `scripts/train_lora.py`, `scripts/eval_base.py`.

## Phần cứng và chi phí
- **Một GPU RTX 5060 Ti 16 GB** (Blackwell), WSL2, torch 2.11 cu128, transformers 5.19.
- **Bản C:** 31.034 dòng × 2 epoch = 1.938 bước; mất 2,95 giờ; VRAM đỉnh 10,25 GB. Thông lượng khoảng 21.000 dòng/giờ, độ dài trần state 1.024 token.
- **Độ trễ batch 1 của bản C:**
  - trung vị 32,6 ms, đo khi GPU rảnh;
  - p90 trên dev khoảng 47 ms;
  - prompt ViFactCheck dài tới khoảng 900 token.
- **Mục tiêu độ trễ ≤ 40 ms.** Lưu ý: số đo độ trễ của model người khác tắt CUDA graphs, nên chưa so được với nhau.
- **Claude:** chỉ dùng hạn mức gói thuê bao theo tuần, **không dùng tiền trả thêm**. Sinh hoặc dịch dữ liệu bằng Claude có làm được (Sonnet đã dịch 6.685 dòng), nhưng tốn hạn mức tuần.

## Benchmark (bench 1.3, chỉ được dùng dev; test đóng băng, không ai chạy)
- **Dev:** 1.430 cặp VI/EN từ 6 nguồn.
  - Dịch song song: Belebele, SIB-200, MASSIVE, XCOPA.
  - Tiếng Việt gốc: ViFactCheck, ViWikiFC (kiểm chứng: xác nhận / bác bỏ / không đủ thông tin).
  - Có thêm bản tiếng Việt không dấu.
- **Điểm chính:** trung bình đều theo nguồn, phần VI có dấu.

| Model | Tham số | VI có dấu | VI không dấu | EN | Độ trễ trung vị (cách đo khác nhau) |
|---|---|---|---|---|---|
| Quyet-1.0-Medium (chinhnc; Qwen3.5-4B + LoRA đã merge) | 4,66 B | 89,6 | 64,5 | 94,5 | 87,5 ms (không CUDA graphs) |
| decider-2b (Mapika; Qwen3.5-2B-Base, chỉ tiếng Anh) | 1,88 B | 86,5 | — | 90,3 | — |
| **Bản C** (dự án, train mới nhất) | 1,7 B | **81,65** | 63,97 | 88,2 | 32,6 ms |
| M1 (dự án, train_v0) | 1,7 B | 79,9 | — | 88,0 | 33 ms |
| Qwen3-1.7B gốc | 1,7 B | 70,6 | — | 84,6 | — |

- **Chỉ tiêu đã đặt:** VI ≥ 84, không dấu ≥ 61, EN ≥ 87, độ trễ ≤ 40 ms.
- **Bản C so với M1 theo nguồn:** ViFactCheck −0,6, MASSIVE +1,4, SIB −2,7, XCOPA +7, ViWikiFC +4,2.
- **Phép thử bản C:**
  - P4 (ghép nhận định với văn bản không liên quan, đáp án đúng là "không đủ thông tin"): đúng 91/88% ở M1, còn **62,5/50,3%** ở C. C hay chọn "bác bỏ".
  - Pa (lựa chọn bỏ dấu): ViWikiFC −16,8.
- **Sai số:** khoảng ±1,2 điểm; nhiễu do seed khoảng ±1.

## Dữ liệu train
- **train_v0:** 20.000 dòng.
  - 16.000 dòng EN, lấy từ tasksource;
  - 4.000 dòng VI, lấy từ PhoLaya, gồm các bộ của UIT. **Các bộ UIT chỉ dùng nội bộ**, không công bố.
- **Bản C:** 31.034 dòng = train_v0 + 4.396 dòng VI gốc + 6.685 dòng dịch bằng Sonnet.
  - VI gốc: VSMRC, EXAMS, ý định du lịch, ViANLI, VNTC.
  - Dịch: đọc hiểu, ý định, kiểm chứng, Có/Không, nhân quả.
- **Kết quả kiểm toán 27/10** (`results/audit_2710/`, đọc `TONG_HOP_dieu_phoi.md` trước):
  - Hàm chuyển đổi và cách đo đúng.
  - Quy mô train nhỏ: 1.250–1.938 bước. Audit nói Quyet-Medium train 120.000 bước; **chưa kiểm chứng**.
  - Model khớp hết dữ liệu từ mốc 25%.
  - Khoảng 1.109 dòng VI không có văn bản để đọc.
  - Khung kiểm chứng trong train khác bench, nhãn "không đủ thông tin" chỉ 12,8%.
  - Trần 1.024 token cắt văn bản dài.
  - Vài lỗi nhỏ trong code.

## Quy tắc bắt buộc
- **Chỉ đọc.** Chỉ được ghi đúng file báo cáo của mình. Không sửa code, dữ liệu hay config.
- **Không tải trọng số model hay dataset.** Đọc model card, README, bài báo, trang web, metadata HF thì được.
- **Không chạy model, không dùng GPU.** Không đụng `data/bench_v1_3/test*`.
- **Không gửi dữ liệu dự án ra ngoài** (dòng dev/train, kể cả trong câu tìm kiếm). Tìm kiếm chỉ bằng từ khóa chung.
- **Chữ tiếng Việt do agent viết** cho bench, config hay dữ liệu train đều phải qua người dùng duyệt. Một phương án cần nhiều chữ mới thì phải ghi rõ chi phí duyệt.
- **Mọi quyết định train hay tải** đều do người dùng duyệt. Báo cáo chỉ đề xuất.
