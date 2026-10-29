# BRIEF: kiểm FLA và chẩn đoán teacher (28/10/2026)

**Người dùng duyệt** (chat 28/10, khoảng 01:15):
- đã tự cài build-essential trong WSL;
- "cho chạy chẩn đoán".

Lý do làm chẩn đoán: `results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN.md`, mục 1–2.

**Không train. Không đụng test. Không dùng prompt dev hay test làm đầu vào teacher.** Các quy tắc cũ giữ nguyên:
- Nhiệt độ GPU > 85°C thì dừng; lính gác nhiệt tiếp tục chạy.
- Mức dùng tuần > 90% thì dừng; từ 88% không bắt đầu việc mới.
- Không đo độ trễ trong khung 03:00–03:05 (giờ eval đêm của Ún).
- Không viết chữ tiếng Việt mới, không sửa `config/` hay `data/`, không xóa, không commit.

Làm lần lượt: **(A) sau khi G0 xong → (B)**.

## (A) FLA: đo lại độ trễ Qwen3.5 công bằng (khoảng 15 phút GPU)

1. **Kiểm cài đặt.** Trong `.venv-g0`, kiểm `flash-linear-attention` import và chạy được, giờ đã có gcc. Không vá thư viện. Nếu vẫn lỗi thì ghi lỗi và bỏ phần (A).
2. **Kiểm đúng.** Chấm Qwen3.5-2B-Base có FLA, so với preds không FLA đã có từ G0b:
   - argmax trùng **100%**;
   - xác suất lệch ≤ 1e-3.

   Dùng toàn bộ dev, hoặc ít nhất 600 dòng phân tầng theo nguồn. Không đạt thì báo, không dùng số đo.
3. **Đo độ trễ:** 2B-Base có FLA, cả eager và CUDA graphs, cùng cách đo với G0a (trung vị, p90, p95, p99, p90 theo nhóm độ dài; 2 lần đo). Nếu còn thời gian thì đo thêm 4B có FLA.
4. Ghi kết quả vào `results/G0/REPORT_G0.md`, mục riêng **"có FLA"**.

## (B) Chẩn đoán teacher: Quyet-Medium chấm prompt train của C (khoảng 1 giờ GPU)

**Đầu vào.**
- `results/train_v1/components/C.jsonl`, sha `005ca3d5…`.
- Chỉ lấy dòng `kind ∈ {choice, noul}`, khoảng 29.500 dòng; bỏ câu score.
- Đúng văn bản như trong file, tức bản có dấu. **Không** chạy teacher trên bản bỏ dấu.

**Cách chạy.**
- Dùng adapter Quyet đã kiểm ở bước E (`scripts/baselines_v1_2/adapters.py`, `quyet-medium-text`) trong `.venv-baselines`, offline, giống lúc chấm baseline. Quyet dùng prompt của chính nó.
- Kết quả `probs` phải theo đúng thứ tự `options` của dòng C. Câu noul ánh xạ đúng như adapter.
- Batch 1, hoặc gom batch nếu adapter cho phép mà không đổi kết quả. Kiểm trên 200 dòng rằng gom batch và batch 1 ra cùng argmax.
- Ghi lại số dòng bị từ chối, bị cắt hay bị loại.

**Đầu ra.**
- `results/teacher_diag/quyet_C.jsonl`, mỗi dòng gồm: `id`, `dataset`, `text_lang`, `kind`, `k`, `gold_argmax` (lấy từ `target_argmax`), `gold_target`, `probs`, `pred`, `agree`, `p_gold`, `entropy_norm`, các cờ.
- Kèm manifest: sha đầu vào, revision Quyet, thời gian, nhiệt độ tối đa.

**Phân tích, báo cáo `results/teacher_diag/REPORT_chan_doan.md`.** Tính theo toàn bộ, theo `text_lang`, và theo từng `dataset` có ≥ 100 dòng:
- **A** = tỉ lệ argmax của teacher trùng `gold_argmax`.
- **S** = trung bình (1 − p_teacher(gold)) trên các dòng **trùng**. Với đích 0,6·teacher + 0,4·nhãn, hiệu chênh giữa đích chưng cất và nhãn là 0,6·S.
- **D90** = tỉ lệ dòng teacher **bất đồng** với nhãn mà teacher tự tin > 0,9.
- Entropy chuẩn hóa trung bình.
- Gộp nhóm VI theo dạng bài:
  - đọc hiểu: vsmrc, race, quail, cosmos_qa, dream, exams;
  - kiểm chứng/NLI: vitaminc, nli_fever, health_fact, vianli, vinli;
  - ý định: tourism, multidogo, banking77;
  - nhân quả: e-CARE, headline_cause;
  - chủ đề: vntc, vsfc/topic;
  - còn lại.

**Mẫu để soát mù sau** (điều phối làm, worker chỉ xuất file):
- Lấy 60 dòng VI teacher bất đồng tự tin (p > 0,9), phân tầng theo nhóm dạng bài, seed 20261028.
- Ghi vào `results/teacher_diag/mau_soat_mu.jsonl`. File này có `state`, `question`, `options`, kèm **cột tách riêng** cho nhãn gốc và đáp án teacher, để điều phối che hai cột này khi soát.

**Cách đọc, chốt trước khi có số, tính trên phần VI:**

| Điều kiện | Kết luận |
|---|---|
| S ≤ 0,05 **và** A ≥ 0,95 | "Chưng cất trên C gần như chỉ là làm mềm nhãn": đề xuất **bỏ lần train 2a**, đi thẳng sang 2b (teacher gán nhãn prompt mới) |
| S ≥ 0,10 **hoặc** A ≤ 0,90 | "Teacher mang thông tin khác nhãn đáng kể": 2a đáng train. Nếu A thấp thì phải soát mù trước, để biết nhãn sai hay teacher sai |
| Còn lại | "Chưa rõ": điều phối cùng Fable cân nhắc, người dùng quyết |

Worker chỉ báo số và áp đúng bảng trên, không tự quyết bước tiếp.

## Kết thúc

- Xong (A) và (B) thì gửi **"GPU XONG"** sang session `local_ad0853e1-dbf5-4579-b3e3-3a3093b693f9`, rồi **dừng**.
- Không sang bước nào khác.
- Ngân sách GPU riêng cho brief này: tối đa 1,5 giờ. Vượt thì dừng và báo.
