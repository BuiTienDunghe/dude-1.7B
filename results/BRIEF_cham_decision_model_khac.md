# Brief "CHẤM CÁC DECISION MODEL KHÁC" (người dùng, nguyên văn từ lịch sử phiên điều phối, 13/10)

## Brief gốc



<pasted_content id="8a84">
Đọc PLAN.md, PLAN_decisions.md, REPORT_ft_v0.md và REPORT_loss_fix.md trước khi
làm gì khác.

Việc lần này: chấm các decision model có sẵn của người khác trên dev_v0, để
biết model của dự án đang đứng ở đâu. Đến giờ ta mới so Qwen3-1.7B với chính nó
trước và sau fine-tune (72,9% lên 79,8% trên tiếng Việt). Ta chưa biết 79,8% là
cao hay thấp so với thị trường, và câu trả lời có thể đổi hướng cả dự án. Chỉ
chạy inference, không train gì.

MODEL CẦN CHẤM

Nhóm cùng cỡ, quanh mức 2B. Đây là nhóm so sánh trực tiếp với Qwen3-1.7B:
- Mapika/decider-2b (1,88B, nền Qwen3.5-2B-Base, chỉ tiếng Anh). Model card
  hướng dẫn nạp bằng transformers và có file decider_config.json; hãy đọc card
  để dùng đúng prompt và cách đọc xác suất của nó.
- mstrasser/Jeff-Qwen3.5-2B (2,2B, chỉ tiếng Anh). Model card ghi dữ liệu train
  có tập train của MASSIVE, nên MASSIVE là nguồn đã thấy với model này. Báo
  điểm của nó có và không có MASSIVE.
- Qwen/Qwen3.5-2B gốc, chưa fine-tune, chấm bằng đúng đường chấm chữ cái của dự
  án (scripts/eval_base.py), không qua adapter nào. Mục đích: so với Qwen3-1.7B
  gốc để biết thế hệ model nền mới hơn bao nhiêu.

Nhóm nhỏ hơn:
- chinhnc/Quyet-1.0-Tiny (183M) và chinhnc/Quyet-1.0-Small (328M)
- convaiinnovations/laya-multilingual (322M)
- vanty120/laya-vi (322M, bản tinh chỉnh cho tiếng Việt)

Nhóm lớn hơn:
- chinhnc/Quyet-1.0-Medium (4,66B)

Quyet và Laya có thư viện riêng cài bằng pip (`quyet` và `laya`). Cả hai nhận
đầu vào dạng state cùng một dict câu hỏi có type, instructions và criteria, rồi
trả về xác suất cho từng lựa chọn. Đó là những gì tôi biết từ model card; hãy
đọc tài liệu và model card của từng model trước khi viết adapter, đừng tin mô
tả của tôi về chữ ký hàm.

Cài mọi thư viện mới vào một venv riêng (ví dụ .venv-baselines). Không cài vào
.venv chính, vì môi trường chính đã được ghim phiên bản để tái lập từng byte
các kết quả cũ, và một thư viện mới có thể kéo theo phiên bản torch hay
transformers khác. Riêng Qwen3.5-2B gốc thì chạy trong .venv chính, bằng đường
chấm của dự án.

CÁCH ĐƯA CÂU HỎI VÀO MODEL

Các model ngoài không dùng prompt chữ cái của dự án. Chúng nhận lựa chọn theo
format riêng, và format có thể ảnh hưởng đến điểm. Với model nào cho phép khai
báo lựa chọn dưới dạng nhãn kèm mô tả, hãy chấm theo hai cách ánh xạ:
1. Mỗi lựa chọn là một nhãn mang nguyên văn nội dung lựa chọn, không có mô tả.
2. Mỗi lựa chọn có một khóa trung tính ngắn, và nội dung lựa chọn nằm ở phần mô tả.

Báo cả hai. Khi so với model của dự án, dùng cách nào cho model kia điểm cao
hơn. Nguyên tắc ở đây là thà đánh giá đối thủ quá cao còn hơn quá thấp: một
adapter viết sai sẽ làm model của người khác trông tệ, và ta sẽ rút ra kết luận
sai có lợi cho mình.

Với model có prompt cố định do tác giả quy định (như decider-2b), dùng đúng
prompt đó, không tự chế.

Giữ nguyên thứ tự lựa chọn của dev_v0, và chấm thêm một lượt thứ tự đảo như đã
làm với model của dự án. Dùng đúng temperature mà mỗi model phát hành, không
fit lại gì. Chạy batch 1, trên cùng GPU.

Cắt cụt là rủi ro thật: Laya mặc định chỉ đọc khoảng 1.024 token và có thể cắt
đầu vào mà không báo. Với mỗi model, hãy tìm giới hạn độ dài, đặt nó đủ lớn nếu
thư viện cho phép, và đếm số câu dev bị cắt. Với Quyet, dùng chế độ báo lỗi thay
vì cắt. Câu nào bị cắt phải được báo riêng, không tính lẫn vào điểm.

KIỂM TRA ADAPTER TRƯỚC KHI TIN KẾT QUẢ

- Chạy ví dụ mẫu trong README hoặc model card của từng model và xác nhận kết
  quả khớp với mô tả ở đó.
- Chạy 10 câu hiển nhiên tự viết (kiểu "Thủ đô của Việt Nam" với 4 lựa chọn),
  mỗi câu ở hai thứ tự lựa chọn. Model phải chọn đúng nội dung ở cả hai thứ tự.
  Việc này bắt được lỗi gán nhầm xác suất về sai lựa chọn.
- Xác suất trả về phải cộng lại bằng 1 và có đúng số phần tử bằng số lựa chọn.
- Với Qwen3.5-2B gốc: kiểm tra các chữ A đến J là một token ở ranh giới trả
  lời, như đã làm với Qwen3. Tokenizer của hai thế hệ có thể khác nhau.

Nếu một model không qua được các kiểm tra này, đừng báo điểm của nó như một kết
quả. Báo là adapter chưa đáng tin và nêu lý do.

CẦN BÁO CÁO

Cho từng model, đặt cạnh ba mốc của dự án (Qwen3-1.7B gốc, Qwen3-1.7B adapter
full_vocab, Qwen3-0.6B sau fine-tune):
- Số tham số và độ trễ trung vị batch 1 trên dev_v0.
- Accuracy và Brier tiếng Việt và tiếng Anh, theo từng nguồn và trung bình đều,
  có CI 95% bootstrap theo state.
- Mức tụt Anh sang Việt, ghép theo pair_id.
- McNemar ghép cặp với adapter full_vocab của dự án, trên cùng câu, cho tiếng
  Việt và tiếng Anh, theo từng nguồn.
- Tỉ lệ nhất quán khi đảo thứ tự lựa chọn.
- Số câu bị cắt cụt hoặc bị từ chối.

Lập riêng một bảng "cùng cỡ" gồm decider-2b, Jeff-Qwen3.5-2B, Qwen3.5-2B gốc,
Qwen3-1.7B gốc và Qwen3-1.7B adapter full_vocab. Bảng này phải trả lời được ba
câu hỏi:
1. Trên tiếng Việt, model của dự án hơn hay kém các model cùng cỡ bao nhiêu.
2. Trên tiếng Anh thì sao. Câu này cho biết công thức train của dự án có ngang
   tầm các nhóm khác không, khi yếu tố ngôn ngữ không còn che.
3. Qwen3.5-2B gốc hơn hay kém Qwen3-1.7B gốc bao nhiêu.

Thêm một bảng về khả năng model đã thấy dữ liệu: với mỗi model, ghi những gì
model card hoặc file cấu hình nói về dữ liệu train, và đánh dấu nguồn nào trong
dev_v0 có thể đã bị thấy (MASSIVE, Belebele, SIB-200, COPA). Nếu không có thông
tin, ghi là không rõ. Đừng suy đoán thành kết luận.

Đừng tự kết luận dự án nên đi hướng nào. Trình bày số, rồi nêu riêng model của
dự án đứng ở đâu so với nhóm cùng cỡ, nhóm nhỏ hơn và nhóm lớn hơn.

QUY TẮC

Các quy tắc ở mục 10 của PLAN.md vẫn áp dụng. Riêng lần này:
- Chỉ dùng dev_v0. Không đụng phần để dành cho test.
- Không sửa code chấm của dự án. Adapter cho model ngoài là code mới, đặt riêng.
- Ghim revision của từng model và phiên bản của từng thư viện.
- Mỗi lượt chấm có giới hạn thời gian.
- In dữ liệu tối đa 5 dòng, mỗi trường cắt còn 200 ký tự.
- Nếu một thư viện không cài được hoặc một model không tải được, dừng ở model
  đó, báo lại, và tiếp tục với các model còn lại. Không thay bằng model khác.

Kết quả đặt ở results/baselines/, báo cáo ở REPORT_baselines.md. Ghi vào
PLAN_decisions.md rằng bước này đã được chèn vào trước bước dựng benchmark.
</pasted_content id="8a84">


## Các tin người dùng ngay sau đó (sửa đổi phạm vi)

từ đổi 1 chút

<pasted_content id="8a84">
MODEL CẦN CHẤM

Chỉ hai model, cả hai quanh mức 2B:
- Qwen/Qwen3.5-2B gốc, chưa fine-tune, chấm bằng đúng đường chấm chữ cái của dự
  án (scripts/eval_base.py) trong .venv chính, không qua adapter nào. Mục đích:
  so với Qwen3-1.7B gốc để biết thế hệ model nền mới hơn bao nhiêu.
- Mapika/decider-2b (1,88B, nền Qwen3.5-2B-Base, chỉ tiếng Anh). Model card
  hướng dẫn nạp bằng transformers và có file decider_config.json; hãy đọc card
  để dùng đúng prompt và cách đọc xác suất của nó. Model này cài vào venv riêng
  nếu cần thư viện mới.

Các model khác (Quyet, Laya, Jeff) để sau, không tải trong lần này.
</pasted_content id="8a84">


---

dự kiến hết bao nhiêu thời gian
