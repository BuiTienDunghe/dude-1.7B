# Học từ Mapika/decider: công thức train và những gì chép được (28/10/2026)

Viết tắt nguồn:
- **GH/** = `https://github.com/Mapika/decider/blob/main/` (số dòng lấy từ bản ngày 28/10)
- **R11** = `results/baselines/meta/Mapika__decider-2b/README.md` (bản v11, lưu cục bộ)
- **HF4Bv2** = `https://huggingface.co/Mapika/decider-4b/blob/v2/README.md`
- **BC** = `results/huong_di_2710/BOI_CANH_chung.md`

## Kết luận trước

- **Lưu ý trước tiên:** Hub decider-2b nay là v12 [nguồn: GH/README.md:181], còn bản lưu cục bộ là v11. Cần xác nhận điểm 86,5 đo trên revision nào.
- **Khoảng cách chủ yếu do quy mô và độ đa dạng dữ liệu, không do cỡ model.** Họ train 1,47 triệu dòng × 1 epoch, ta train 31 nghìn dòng × 2 epoch (mục 1, 4) [suy luận].
- **Cách dạy abstain của họ nhắm thẳng vào lỗi P4.** Cỡ model không chữa được lỗi này (mục 1, 6).
- **Chưng cất KL về model cha là cơ chế giữ phân phối, không làm tăng độ chính xác** (mục 2).
- **RL không cần cho bài của ta** (mục 5).

## 1. Hỗn hợp dữ liệu SFT

**Thành phần hỗn hợp** (hạn mức trong code [nguồn: GH/decider/data/mixture.py:5-21,36-37]):

| Nhóm | Số dòng |
|---|---|
| general: mọi task gốc, ≤10 lựa chọn | khoảng 1 triệu, tối đa 20.000 dòng/task [nguồn: GH/decider/data/core.py:20] |
| Dạng đầu vào kiểu Jev: wide, padded, described, json, json_indexed, single | khoảng 155 nghìn + 8.000 cho mỗi bộ nhãn lớn |
| isolated: Score tách mức | 26.000 Score + 9.000 Choice + 1.200 routing |
| rules: sinh bằng code, nhãn chính xác, cặp đôi lật nhãn | 90.000 |
| contrastive: hai trạng thái lệch một dữ kiện, đáp án lật | lặp ×3 |
| custom, routing, commands (teacher viết) | toàn bộ |

- Số dòng thực tế của từng nhóm chỉ in ra lúc build, không công bố. Tổng: 1,47 triệu dòng, 455 triệu token [nguồn: GH/README.md:375].

**Abstention** [nguồn: GH/scripts/train.sh:12; GH/decider/data/augment.py:236-250]:
- Áp cho 10% số câu có ≥3 lựa chọn:
  - 75%: thêm một lựa chọn abstain (12 cách diễn đạt), đáp án giữ nguyên;
  - 25%: thay **toàn bộ** danh sách bằng nhãn của một task khác, đáp án đúng là abstain.
- Lựa chọn abstain luôn được giữ lại khi lấy mẫu con [nguồn: GH/decider/prompt.py:61-64].
- Bài học v5: cách cũ là xóa đáp án đúng, khiến model học "abstain khi thiếu đúng nhãn". Cách mới đưa offtopic probe từ 0,68 lên 0,83 mà các task chung không đổi [nguồn: GH/docs/HISTORY.md:146-155].

**Kiểm chứng và đọc hiểu:**
- FEVER dùng 3 nhãn, có nhãn "not enough information" [nguồn: GH/decider/data/tasks_extended.py:367-375].
- Có thêm RACE, BoolQ, MultiRC, QNLI, WikiQA, COPA [nguồn: các `@task` trong GH/decider/data/core.py và tasks_extended.py].
- Belebele và XCOPA không được train, kể cả ở 4B [nguồn: GH/MODEL_CARD_4B.md:226].
- Model còn yếu ở QuALITY (0,49), abstain_probe (0,61) và StrategyQA (0,55) [nguồn: R11:333-358].

**Lọc dữ liệu teacher:**
- Routing: teacher tự gán nhãn rồi kiểm lại bằng logit chữ cái. Nhãn bất đồng chỉ được giữ theo luật tin cậy [nguồn: GH/docs/HISTORY.md:256-258; GH/decider/data/mixture.py:57-75].
- Giai đoạn 2:
  - 8.000 dòng từ 10 họ bài sinh bằng code, đáp án do code tính; một họ là abstention [nguồn: R11:277].
  - 11.356 câu hỏi trên tài liệu do Qwen3.6-27B (bật thinking) viết. Mỗi câu được trả lời lại 2 lần trong ngữ cảnh mới, xáo lựa chọn, không thấy đáp án gốc. Chỉ giữ câu mà cả 2 lần trùng nhau; tỉ lệ giữ 89% và 91% [nguồn: GH/MODEL_CARD_4B.md:251].

## 2. Chưng cất về v10

- **Loss và nhiệt độ:** loss là KL(p_v10 ‖ p_model) trên các lựa chọn, **nhiệt độ 1**. p_v10 lấy từ v10 đã đóng băng [nguồn: R11:279,284].
- **Tỉ lệ trộn:** 20.100 dòng replay và 22.649 dòng có nhãn, khoảng 47:53. Không có hệ số riêng: loss lấy tổng trên các câu trả lời rồi chia cho số câu [nguồn: GH/MODEL_CARD_4B.md:261; cùng công thức cho 2B: GH/docs/HISTORY.md:433-435].
- **Vì sao bỏ nhãn cứng ở dòng replay** [nguồn: GH/docs/HISTORY.md:427-438]:
  - Train nhãn cứng làm logit nhọn: entropy 0,18 so với 0,28 (đo trên 4B).
  - Nhiệt độ fit vọt lên 1,935 ở 4B và 2,06 ở nhánh nhãn của 2B. Phân phối phục vụ vì thế phẳng hơn.
  - Chế độ chơi lấy mẫu sụt: bag-draw 37,9% so với 56,6% (4B).
  - Nhánh KL giữ T ở 1,145; KL trung bình 0,047; 94,9% đáp án argmax trùng v10 [nguồn: R11:45-46].
- **So sánh hai nhánh không sạch:**
  - Nhánh nhãn dùng 6.700 dòng replay, nhánh KL dùng 20.100 [nguồn: R11:292-294].
  - Hai nhánh hòa nhau trong 0,002 trên điểm chọn [nguồn: R11:117-118].
  - v11 vẫn mất 2,2 điểm trên các bộ có nhãn người [nguồn: R11:71].

## 3. Prompt, vị trí đọc chữ cái, Có/Không, thang điểm, nhiệt độ

- **Định dạng prompt:** dạng thô trên model Base, không có chat template:
  `Context:\n…\n\nQuestion: …\nOptions:\n(A) …\nAnswer: (`
  [nguồn: GH/decider/prompt.py:128-140; R11:210-216]
- **Vị trí đọc:** lấy hidden state tại dấu "(", chiếu lên các hàng chữ cái của LM head. Chữ cái không hợp lệ bị đặt −inf. **Loss là CE chỉ trên logit chữ cái** [nguồn: GH/decider/model.py:23-32; GH/decider/train.py:42-46]. Ta đang dùng CE trên toàn từ vựng.
- **Thứ tự lựa chọn:** mọi câu đều bị xáo ở mỗi dòng, kể cả no/yes. Bộ nhãn trên 10 lựa chọn được lấy mẫu con, luôn giữ đáp án đúng [nguồn: GH/decider/prompt.py:58-66,140].
- **Có/Không (noul):** là câu chọn với 2 lựa chọn `["no","yes"]` [nguồn: GH/decider/systemone.py:65-70].
- **Thang điểm (score):** mỗi mức là một dòng Có/Không riêng ("Proposed answer: … Does the proposed answer fit?"), sau đó chuẩn hóa [nguồn: GH/decider/systemone.py:84-103]. Nếu không train cho dạng này, độ chính xác sụt tới 20 điểm [nguồn: GH/docs/HISTORY.md:262-266].
- **Nhiệt độ theo loại câu:** choice 1,164; noul 1,624; score 1,124. Fit bằng NLL: dò lưới rồi golden-section [nguồn: R11:48,296-299; GH/decider/calibrate.py:145-162]. Nhiệt độ **không đổi argmax** [nguồn: R11:50-52], nên không ảnh hưởng điểm dev [suy luận].

## 4. Chọn checkpoint, tách held-out, lịch train

**Giai đoạn 1: full fine-tune** [nguồn: GH/scripts/train.sh:10-12; GH/decider/train.py:95-131]
- lr 1e-5, warm-up 150 bước, cosine về 0.
- AdamW β (0,9; 0,95), weight decay 0, clip 1,0.
- **32.768 token mỗi bước** (16.384 × accum 2), 1 epoch, khoảng 13.900 bước [suy luận].
- **Không chọn checkpoint:** chỉ eval ở cuối và lấy trọng số cuối.

**Giai đoạn LoRA** [nguồn: R11:115-124,285]
- r64/α128, lr 1e-4, warm-up 5%.
- 1.676 bước × 65.536 token, 2 epoch.
- Chọn theo luật đăng ký trước: mất tối đa 1 điểm trên các bộ có nhãn người, cộng các ngưỡng trên bộ held-out khó. Không checkpoint nào đạt, nên v11 là phương án dự phòng.

**"val Choice NLL"**
- Không tìm thấy cụm này trong tài liệu công khai. "Selection score" được nhắc tên nhưng không được định nghĩa [nguồn: R11:117; GH/MODEL_CARD_4B.md:272-273].
- Gần nhất là cổng ở giai đoạn RL: NLL trên 847 dòng val không tăng quá 0,005 [nguồn: GH/docs/RL.md:62].

**Tách held-out**
- 28 bộ không bao giờ đưa vào train [nguồn: R11:303-307].
- 6 miền teacher được giữ ra [nguồn: GH/decider/data/mixture.py:27,38].
- Template và miền tài liệu dùng cho bộ held-out chưa từng train [nguồn: GH/MODEL_CARD_4B.md:270-272].
- Khử trùng lặp với mọi file eval [nguồn: R11:289-292].

## 5. RL 384 bước

- **Reward:** chỉ tính theo kết quả, gồm bộ kiểm MiniWoB++, thắng/thua game, và log score của niềm tin so với luật xác suất chính xác. Kèm thêm KL ≤0,01 về v8 [nguồn: GH/docs/RL.md:17-46].
- **Không giúp độ chính xác chung:** 847 dòng val −0,4; OpenJev −0,8; chơi greedy +0,6. Chỉ chế độ trình duyệt lấy mẫu tăng +10,2 [nguồn: GH/docs/RL.md:73-85].
- **Chi phí:** cần 4 GPU và Chrome thật [nguồn: GH/docs/RL.md:5].
- **Kết luận: không cần** cho bài của ta [suy luận].

## 6. Cỡ model

**Số đo của họ**
- 0,8B → 2B, cùng công thức: khoảng +3 điểm in-task và +3 held-out. Model nhỏ thua ở tri thức, không thua ở định dạng [nguồn: GH/docs/RESULTS.md:175-176].
- 2B v10 → 4B v1: +2,9 in-task / +3,3 held-out [nguồn: GH/docs/RESULTS.md:113-114]. Có nhiễu: 4B dùng mixture v2 có MASSIVE và XNLI đa ngữ, và đổi optimizer [nguồn: GH/MODEL_CARD_4B.md:216-243].
- Cùng dữ liệu giai đoạn 2 (2B v11 so với 4B v2.1): regression +2,9 / +3,2; heldout_jb +12,7 [nguồn: R11:69; GH/docs/HISTORY.md:461-465].

**Theo task** [nguồn: HF4Bv2:264-291; GH/docs/RESULTS.md:313,315]:

| Task | 2B v10 | 4B |
|---|---|---|
| massive_scenario | 0,756 | 0,793 |
| QuALITY | 0,494 | 0,561 |
| VitaminC | 0,639 | 0,756 |
| massive-de-DE (4B có train MASSIVE đa ngữ) | 0,797 | 0,843 |
| **abstain_probe** | **0,606** | **0,582** |
| **offtopic_probe** | **0,841** | **0,823** |

**Hàm ý cho ta**
- Mỗi lần gấp đôi cỡ được khoảng +3 điểm trên tác vụ thường ngày [suy luận từ các số trên].
- Quyet (4,66B, công thức chưa kiểm chứng) hơn decider-2b 3,1 điểm, cùng bậc với quy luật trên [suy luận; BC:36,58].
- 1,7B so với 1,88B gần như cùng cỡ, nên khó giải thích được 4,85 điểm chênh [suy luận].
- Abstention không tăng theo cỡ, nên phải sửa bằng dữ liệu [suy luận].

## 7. Bài học cho 1 GPU 16 GB, xếp theo lợi ích/chi phí

1. **Abstain cho kiểm chứng (P4)**
   - Làm gì: ghép nhận định với văn bản lấy từ dòng khác, gán nhãn "không đủ thông tin".
   - Bằng chứng: offtopic 0,68 → 0,83, task chung không đổi [nguồn: GH/docs/HISTORY.md:153-155].
   - Chi phí: chỉ cần code, dùng lại nhãn sẵn có nên không có chữ Việt mới; train 1 lần khoảng 3 giờ.
   - Rủi ro: model học lối tắt "lệch chủ đề thì không đủ thông tin". Cách giảm là thêm cặp cùng chủ đề nhưng thiếu dữ kiện; cách này cần chữ Việt mới, người dùng phải duyệt [BC:70; suy luận].
2. **Thêm dòng, 1 epoch thay vì 2**
   - Làm gì: thêm các bộ công khai tiếng Anh (FEVER-NEI, RACE, MultiRC, COPA), có trần số dòng mỗi task.
   - Bằng chứng: decider-2b chỉ train tiếng Anh vẫn đạt 86,5 trên dev tiếng Việt [nguồn: BC:37].
   - Chi phí: 100–150 nghìn dòng mất khoảng 5–7 giờ; không tốn hạn mức Claude [suy luận].
   - Rủi ro: dữ liệu tiếng Việt bị pha loãng [suy luận].
3. **CE chỉ trên chữ cái**
   - Làm gì: tính loss chỉ trên các chữ cái lựa chọn, như họ làm [nguồn: GH/decider/train.py:42-46].
   - Chi phí: sửa ít code, train 1 lần.
   - Rủi ro: lợi ích chưa được đo, cần thử A/B [suy luận].
4. **Replay KL về M1**
   - Làm gì: lưu phân phối của M1 trên 10–20 nghìn dòng cũ, rồi train dòng replay theo KL ở T = 1.
   - Mục đích: giữ hành vi tốt của M1, như P4 đạt 91/88% [nguồn: BC:45].
   - Chi phí: vài phút suy luận.
   - Rủi ro: chưa có bằng chứng tăng độ chính xác [nguồn: R11:117-118].
5. **Xáo thứ tự lựa chọn**
   - Hiện trạng: ta đã xáo cho câu choice, còn noul và score giữ nguyên thứ tự theo quyết định ngày 12/10 [nguồn: scripts/train_lora.py:99-106].
   - Khác với họ: họ xáo cả no/yes [nguồn: GH/decider/prompt.py:140].
   - Đề xuất: có thể thử xáo noul, nhưng phải được người dùng duyệt.
6. **Giá trị thấp**
   - Nhiệt độ theo loại câu: không đổi argmax.
   - Score tách mức: tốn nhiều lượt forward, vượt mục tiêu ≤40 ms [suy luận].
   - Bỏ chat template: chỉ làm khi có A/B.
   - Full fine-tune 1,7B: khoảng 13,6 GB trước activation, khó vừa 16 GB nếu không có optimizer 8-bit [suy luận].
