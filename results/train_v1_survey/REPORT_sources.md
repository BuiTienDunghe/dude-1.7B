# Khảo sát nguồn tiếng Việt cho train_v1 (chưa dựng gì)

Ngày 2026-10-13. Đây chỉ là khảo sát: không train, không dựng train_v1, không dịch gì, không mở test 1.0. Bảng đầy đủ ở `tables_sources.md`, dữ liệu gốc của khảo sát ở `raw/` và `samples/`.

## Tóm tắt

- **Nhóm 1 (nguồn đã dùng ở train_v0):** còn rất nhiều. Cả ba bộ ViNLI, VSFC, ViSFD trong LakoreAI/PhoLaya-base đều mới dùng 700 câu hỏi mỗi tác vụ; mỗi tác vụ còn từ 7.086 đến 77.160 câu hỏi. Cùng repo còn bốn bộ chưa dùng: ViANLI, ViCTSD, UTS, VSMEC.
  - Nhưng nhóm này chỉ có NLI, cảm xúc/khía cạnh, chủ đề hẹp (4 lớp về phản hồi sinh viên) và độc hại.
  - Không có đọc hiểu, ý định hay kiểm chứng.
- **Nhóm 2 (bộ bản địa khác, mở, có split train):** đủ ứng viên cho bốn trên năm dạng bài còn thiếu, nhưng ứng viên nào cũng có điểm yếu.
  - Đọc hiểu: VSMRC, 11.426 câu. Câu hỏi do LLM sinh rồi người kiểm. Văn bản lấy từ Wikipedia, nên rủi ro trùng ViWikiFC cao.
  - Chủ đề: VNTC-10Topics (33.759 dòng, tin tức năm 2007). Ngoài ra có vài bộ tin tức mới hơn, rủi ro trùng ViFactCheck cao.
  - Ý định: chỉ có bộ sinh từ mẫu câu, miền hẹp (vi-tourism-admin-intent, 14.983 dòng). PhoATIS là bản dịch của ATIS và cấm phân phối lại.
  - Kiểm chứng: aiMy144 có claim tổng hợp và không có thẻ giấy phép. k22-hcmus không có thẻ dataset.
  - **Yes/no: không có bộ bản địa mở nào.** Chỉ dựng gián tiếp được từ NLI (entailment), phàn nàn hoặc độc hại (noul).
- **15 bộ cần email, ký thỏa thuận hoặc HF-gated.** Chỉ liệt kê, không tải. Trong đó có các bộ đọc hiểu tốt nhất: UIT-ViQuAD 2.0, ViMMRC 1.0/2.0, ViNewsQA.
- **Nhóm 3 (dịch máy):** model được dùng đầu ra để train và vừa 16 GB gồm madlad400-3b-mt, Qwen3-4B, Hy-MT2-1.8B (đều Apache-2.0) và Seed-X 7B bản lượng tử hóa (OpenMDW).
  - Hunyuan-MT-7B, HY-MT1.5, SeaLLMs-v3 và Babel cấm dùng đầu ra để cải thiện model khác.
  - NLLB chỉ dùng phi thương mại.
  - Trong tasksource còn 15–49 nghìn dòng chưa dùng ở mỗi dạng bài còn thiếu.
  - Ước lượng thô (chưa đo): dịch 5.000 dòng mất khoảng 1–3 giờ nếu chạy theo lô, khoảng 10 giờ nếu chạy batch 1.
- **Ba phương án** ở mục 5. Tôi không chọn hộ.

---

## 1. Cách làm

- Đọc card, đếm dòng và tra giấy phép do hai subagent (Sonnet) làm, theo dòng "Chi phí token" trong `PLAN_decisions.md`.
- Mọi con số của subagent đã được kiểm lại bằng script:
  - **Bộ dữ liệu:** revision, cờ gated và số dòng train của 20 ứng viên trên HF (2 ứng viên còn lại không ở trên HF) đã đối chiếu với `api/datasets/<id>` và dataset viewer (`/size`). Khớp 20/20.
  - **Model dịch:** revision, thẻ giấy phép và cờ gated đã đối chiếu với `api/models/<id>`. Khớp 44/44.
  - Đã soi trường "nguồn gốc nhãn" và "cách đổi" của từng ứng viên (mục 3).
- **Độ dài:** token Qwen3, cùng tokenizer với bộ đếm của train_v0.
  - Nhóm 2 đo trên khoảng 300 dòng split train lấy qua dataset viewer, ở 3 vị trí đầu, giữa và cuối (`scripts/survey/sample_lengths.py`).
  - Nhóm 1 đo trên file train đầy đủ (`scripts/survey/pholaya_counts.py`).
  - Tasksource đo trên 300 dòng mỗi nguồn (`scripts/survey/tasksource_mt_pool.py`).
- **Đã tải về:**
  - Split train của 6 bộ trong PhoLaya-base, đặt ở `raw/pholaya/sources/*/train.jsonl`, tổng 125 MB: vinli 34,4; visfd 57,9; vsfc 11,7; vianli 10,6; victsd 8,5; uts 2,1 MB.
  - Mẫu nhóm 2 ở `samples/`, 8,1 MB.
  - Không tải viglue (586 MB) và massive (nguồn bench_v1).
- Giấy phép được chép nguyên văn từ nguồn, ở mục 3 của `tables_sources.md`. Nguồn nào không ghi giấy phép thì ghi "không rõ".
- Đã đối chiếu `data/reserved_test.json`: file chỉ có 4 nguồn song song (Belebele, SIB-200, MASSIVE, XCOPA), đều đã loại cứng. Không ứng viên nào xung đột với file này.

## 2. Nhóm 1: phần còn lại của nguồn tiếng Việt đã dùng

Nguồn đã dùng: LakoreAI/PhoLaya-base @ `034d4273` (revision của train_v0). Giấy phép trên card (nguyên văn): `license: other (research-use-only): the mixture contains derivatives of the UIT corpora (ViSFD, VSFC, ViNLI), which are research-licensed. Do not use commercially without contacting the dataset authors.`

| Bộ | Tác vụ | Kiểu | Câu hỏi trong train | Đã dùng (train + val v0) | Còn lại | Trung vị state (token) |
|---|---|---|---|---|---|---|
| ViNLI | relation / entailment | choice / noul | 24.363 mỗi tác vụ | 700 / 700 | 23.663 / 23.663 | 66 |
| VSFC | sentiment / topic | choice / choice | 11.395 mỗi tác vụ | 700 / 700 | 10.695 / 10.695 | 13 |
| ViSFD | aspect / complaint | choice / noul | 77.860 / 7.786 | 700 / 700 | 77.160 / 7.086 | 42 |

Các bộ khác có trong cùng repo, chưa dùng ở train_v0. Card hiện tại ghi là "earlier revisions also carried ViGLUE, ViANLI, ViCTSD, VSMEC and UTS"; card không ghi giấy phép riêng cho các thư mục này.

| Bộ | Tác vụ | Câu hỏi | Trung vị state | Ghi chú |
|---|---|---|---|---|
| ViANLI | relation / entailment | 8.012 mỗi tác vụ | 60 | Bản gốc uitnlp/ViANLI: thẻ `mit`, thân card ghi CC BY-NC-SA 4.0, hai chỗ mâu thuẫn |
| ViCTSD | constructiveness / toxicity | 6.996 / 6.995 | 40 | Điều kiện truy cập mâu thuẫn (mục 6) |
| UTS | sentiment / aspect | 1.808 / 1.850 | 22 | Không rõ nguồn gốc; card PhoLaya không mô tả |
| VSMEC | emotion | 5.544 | 15 | Nguồn gốc yêu cầu ký thỏa thuận. Đã đếm trước khi biết điều đó, file đã xóa (mục 6) |

- Mỗi dòng PhoLaya là một câu hỏi trên một văn bản. Hai tác vụ của cùng một bộ dùng chung văn bản, ví dụ ViNLI có 24.363 cặp câu, mỗi cặp mang 2 câu hỏi.
- Khi dựng train_v1 phải tránh lại các văn bản đã nằm trong train_v0 và val_v0 (đã chia theo văn bản).
- Tasksource chỉ có 4 nguồn tiếng Việt. xcopa/vi và xcopa/translation-vi bị loại cứng. X-CODAH-vi (109 dòng) và X-CSQA-vi (139 dòng, đã dùng 2) quá nhỏ.

## 3. Nhóm 2: bộ tiếng Việt bản địa khác, có split train, truy cập mở

Theo dạng bài còn thiếu:

| Dạng bài | Bộ (repo) | Dòng train | Cách đổi | Trung vị state / cả dòng | Nhãn | Giấy phép (thẻ) | Rủi ro trùng bench_v1 |
|---|---|---|---|---|---|---|---|
| Đọc hiểu | VSMRC/mrc | 11.426 | choice(4): đoạn văn + câu hỏi + 4 lựa chọn | 214 / 264 | LLM sinh, người kiểm | apache-2.0 (văn bản Wikipedia vốn CC BY-SA) | **Cao** với ViWikiFC |
| Đọc hiểu / kiến thức | mhardalov/exams (vi) | 1.955 | choice(4), không có đoạn văn; bản kèm đoạn văn dài khoảng 3.600 token | 8 / 96 | đề thi thật | cc-by-sa-4.0 | Thấp (bản không kèm đoạn văn) |
| Đọc hiểu | p-storm/ViCS-MCQ | 1.085 | choice(4) trên bài giảng tin học | 67 / 145 | LLM sinh | cc-by-4.0 (đáng ngờ, vì là bài giảng của bên thứ ba) | Thấp |
| Đọc hiểu (QA) | HienNGuit/Domain-ViWikiQA | 6.074 | cần tự sinh phương án nhiễu | 299 / 323 | LLM sinh, người kiểm một phần | other | **Cao** với ViWikiFC |
| Chủ đề | kornwtp/VNTC-10Topics | 33.759 | choice(10) | 402 / 448 | chuyên mục báo (2007) | mit (bản sao trên HF) | Vừa (tin cũ) |
| Chủ đề | NamSyntax/vietnamese-news-classification | 1.348.717 | choice(11) | 57 / 103 | không rõ | apache-2.0 (tin cào, đáng ngờ) | **Cao** với ViFactCheck |
| Chủ đề | ngocleltt/vi-news-4topics | 1.909 | choice(4), tiêu đề VnExpress | 16 / 24 | không rõ | cc-by-4.0 | **Cao** với ViFactCheck |
| Chủ đề | UIT-ViON (GitHub) | khoảng 260.000 tiêu đề | choice(13) | — | bán tự động | — (không đọc được trên HF) | **Cao** với ViFactCheck |
| Ý định | phamluan/vi-tourism-admin-intent | 14.983 | choice(8) | 18 / 59 | sinh từ mẫu câu | cc-by-4.0 | Không |
| Kiểm chứng | aiMy144 verifier v3.1 | 58.017 (77% dòng mẫu có chữ tiếng Việt có dấu; dữ liệu sắp theo nguồn, cuối file phần lớn là tiếng Anh) | choice(3) hoặc noul | 50 / 99 | claim tổng hợp | **không rõ** (không có thẻ) | **Cao** với ViFactCheck |
| Kiểm chứng | k22-hcmus/fact-checking-vietnamese-news | 4.140 | choice(3) | 62 / 99 | không rõ | **không rõ** (không có card) | **Cao** (Dân Trí 2026) |
| Yes/no (gián tiếp) | ViANLI entailment, ViSFD complaint, ViCTSD toxicity, ViOCD, ViHealthQA | 4–8 nghìn mỗi bộ | noul | 24–114 | người gán (trừ ViHealthQA) | xem bảng đầy đủ | Thấp đến vừa |
| NLI (máy dịch) | facebook/xnli (vi) | 392.702 | choice(3) | 33 / 60 | **dịch máy** (train) | không có thẻ | Không |

- **Cần email, ký thỏa thuận hoặc HF-gated (15 bộ, không tải):** UIT-ViQuAD 1.0/2.0, ViNewsQA, ViMMRC 1.0, ViMMRC 2.0, ViSpamReviews, ViLegalTF/MCQ/NLI, VLSP 2025 MCQ và NLI, VlogQA, ViCoQA, ViGoEmotions, ViTHSD, UIT-ViWikiQA.
- **Không có split train:** MLQA-vi, VMLU, M3Exam-vi, VNHSGE, ViGEText, SEA-NLI.
- **Không tải được:** ISE-DSC01, ViCoV19QA.
- **Bị loại:**
  - Các nguồn của bench_v1 và bản sao của chúng.
  - ViHOS, vì dựng từ ViHSD.
  - VOZ-HSD, vì card cấm fine-tune trên nhãn AI của nó.
  - Ba bộ khác không phải bản địa hoặc không dùng được (lý do ở `tables_sources.md`).

## 4. Nhóm 3: dịch máy một phần tasksource (chỉ ước lượng, chưa dịch gì)

**Dòng tasksource chưa dùng theo dạng bài còn thiếu.** Đã bỏ nguồn bị loại cứng (IntentGrasp, SLURP) và hai tên nguồn regex bắt nhầm.

| Dạng bài | Số nguồn | Dòng chưa dùng | Token tiếng Anh TB mỗi dòng | Ví dụ nguồn |
|---|---|---|---|---|
| Đọc hiểu | 13 | 48.915 | 226 | race, race-c, quail, cosmos_qa, dream, reclor, logiqa, sciq |
| Chủ đề | 5 | 19.193 | 188 | ag_news, yahoo_answers_topics, trec, open_question_type |
| Ý định | 12 | 21.597 | 252 | banking77, clinc_oos, multidogo/*, schema-guided, topv2, snips |
| Kiểm chứng | 4 | 15.484 | 89 | vitaminc, nli_fever, health_fact, liar |
| Yes/no | 10 | 14.881 | 376 | sharc, spartqa-yn, boolq, coqa/yesno |

**Model dịch vừa 16 GB và cho phép dùng đầu ra để train** (theo card; giấy phép đã đối chiếu bằng API):

| Model | Tham số | Giấy phép | Ghi chú |
|---|---|---|---|
| google/madlad400-3b-mt | 2,9B | apache-2.0 | Model dịch chuyên dụng (T5). Card không công bố số en→vi |
| Qwen/Qwen3-4B (và bản FP8, AWQ) | 4,0B | apache-2.0 | Dịch qua prompt; không có số en→vi công bố |
| tencent/Hy-MT2-1.8B (7B chỉ chạy được bản FP8) | 2,0B | apache-2.0 | Mới (2026-05). Cần transformers ≥ 5.6 |
| ByteDance-Seed/Seed-X-PPO-7B (AWQ/GPTQ) | 7,5B | OpenMDW-1.0 | Chỉ vừa khi lượng tử hóa |
| Helsinki-NLP/opus-mt-en-vi | 72M | thẻ apache-2.0 | Rất nhanh. BLEU Tatoeba 37,2, thấp hơn rõ |

**Không dùng được để tạo dữ liệu train, hoặc có ràng buộc:**
- Tencent Hunyuan-MT-7B và HY-MT1.5, SeaLLMs-v3, Babel: điều 5(b) cấm dùng đầu ra để cải thiện model khác.
- NLLB-200: CC-BY-NC-4.0 và là "research model".
- vinai-translate: AGPL-3.0, card không nói về đầu ra.
- TranslateGemma, Vistral, Aya: gated.
- Tower: không hỗ trợ tiếng Việt.

**Ước lượng tốc độ. Chưa đo; sai số có thể gấp 2–3 lần.**
- Cách tính: card không công bố thông lượng trên phần cứng tương tự, nên tôi tính từ băng thông bộ nhớ của RTX 5060 Ti (khoảng 448 GB/s).
- Batch 1, giải mã bị giới hạn bởi việc đọc trọng số:
  - Qwen3-4B bf16 (khoảng 8 GB): tối đa khoảng 56 token/s, thực tế khoảng 35–45.
  - Hy-MT2-1.8B (khoảng 4 GB): tối đa khoảng 110 token/s.
- Chạy theo lô 16–32 câu: thông lượng có thể lên khoảng 300–700 token/s (4B) hoặc 800–1.500 token/s (1,8B).
- Một dòng tiếng Anh trung bình khoảng 220 token. Bản tiếng Việt khoảng 1,4 lần số token, tức khoảng 310 token đầu ra mỗi dòng. Hệ số 1,4 là giả định, chưa đo.
- Suy ra số dòng mỗi giờ:

| Cách chạy | Dòng mỗi giờ |
|---|---|
| 4B, batch 1 | khoảng 450 |
| 4B, theo lô | khoảng 3.500–8.000 |
| 1,8B, theo lô | khoảng 9.000–17.000 |

- Ví dụ 5.000 dòng (phương án 3): khoảng 1,1 triệu token vào và 1,5 triệu token ra. Mất khoảng 1–3 giờ nếu chạy theo lô, khoảng 11 giờ nếu chạy batch 1.
- Cần đo thử trên khoảng 50 dòng trước khi lên lịch. Bước đo thử đó sẽ là "dịch", nên chưa làm.

## 5. Ba phương án thành phần train_v1 (chưa chọn)

Chung cho cả ba:
- Giữ nguyên 16.000 dòng tiếng Anh của train_v0. Như vậy chỉ phần tiếng Việt thay đổi, so được với các lần trước.
- Số dòng không dấu mỗi epoch nếu giữ p = 0,3 là kỳ vọng 0,3 × số dòng VI. Một nửa bỏ dấu toàn bộ, một nửa chỉ bỏ ở state, như strip_v0.
- Cả ba phương án đều giữ nguyên 4.000 dòng VI của train_v0 làm tập con, rồi thêm dòng mới. Ở phương án 1, số trong bảng là tổng sau khi thêm: ví dụ ViNLI 2.400 = khoảng 1.330 dòng có sẵn ở v0 + khoảng 1.070 dòng mới, VSFC và ViSFD tương tự.
- Mọi phương án đều phải qua `scripts/decontam.py` với dev_v1 và test 1.0 trước khi dùng. Bước này phải đọc test 1.0, nên cần xin phép riêng.

| | Phương án 1: mở rộng PhoLaya | Phương án 2: bản địa mở, đủ dạng bài | Phương án 3: bản địa + dịch máy |
|---|---|---|---|
| Dòng VI | 10.000 | 12.000 | 11.000 |
| Tổng dòng | 26.000 | 28.000 | 27.000 |
| Tỉ lệ VI | 38,5% | 42,9% | 40,7% |
| Không dấu mỗi epoch (p = 0,3) | khoảng 3.000 (1.500 toàn bộ) | khoảng 3.600 (1.800 toàn bộ) | khoảng 3.300 (1.650 toàn bộ) |
| Đọc hiểu | 0 | 2.000 VSMRC + 500 EXAMS | 2.000 (dịch: race, quail, cosmos_qa, dream) |
| Chủ đề | 1.200 VSFC topic (4 lớp hẹp) | 1.500 VNTC + 666 VSFC topic có sẵn | 1.000 VSFC topic + 800 dịch (ag_news, yahoo) |
| Ý định | 0 | 1.500 vi-tourism-admin-intent (mẫu câu) | 800 dịch (banking77, clinc_oos, multidogo) |
| Kiểm chứng | 0 | 1.500 aiMy144 (chỉ dòng VI) | 800 dịch (vitaminc, nli_fever, health_fact) |
| Yes/no (noul) | 3.200 (ViNLI 1.200 + ViANLI 800 + ViSFD complaint 1.200) | 1.000 ViANLI + khoảng 1.330 noul có sẵn ở v0 | 1.000 ViNLI + 600 dịch (sharc, boolq) + khoảng 1.330 có sẵn |
| NLI, cảm xúc, khía cạnh, độc hại (choice) | 5.600 (ViNLI 1.200, ViANLI 800, VSFC sentiment 1.200, ViSFD aspect 1.200, ViCTSD 1.200) | 4.000 dòng của train_v0, giữ nguyên | 4.000 dòng của train_v0, giữ nguyên |
| Nhãn | người gán | người gán và nhãn máy/tổng hợp (VSMRC, aiMy144, intent) | người gán, cộng nhãn tiếng Anh mang qua bản dịch |
| Rủi ro chính | thiếu 3 dạng bài; giấy phép research-only; ViANLI và ViCTSD có mâu thuẫn giấy phép/truy cập; tiền đề ViANLI là câu tin tức nên cần decontam với ViFactCheck | trùng ViWikiFC (VSMRC) và ViFactCheck (aiMy144, VNTC, tin tức), bắt buộc decontam chặt; aiMy144 không rõ giấy phép | văn bản không bản địa; tốn 1–3 giờ GPU dịch và công soát; phải chọn model dịch cho phép dùng đầu ra; decontam ở mức tiếng Anh đã có từ v0 nhưng vẫn cần chạy lại |

Cách đếm hàng "Yes/no" ở phương án 2 và 3: noul của train_v0 gồm ViNLI entailment 664 và ViSFD complaint 669.

Biến thể của cả ba: giữ tổng 20.000 dòng bằng cách giảm tiếng Anh xuống 8–10 nghìn. Biến thể này thay đổi hai thứ cùng lúc, nên khó so với các lần train trước.

## 6. Điều chưa chắc, vướng mắc và sự cố

**Điều chưa chắc:**
- **Điều kiện truy cập của bộ UIT, kể cả bộ đã dùng ở train_v0.**
  - README GitHub của nhóm UIT (kietnv/VietnameseDatasets) yêu cầu email để ký thỏa thuận cho VSMEC, VSFC và các bộ khác.
  - Trang nlp.uit.edu.vn lại có link tải thẳng cho một số bộ.
  - ViCTSD và ViOCD được tác giả chính đưa lên HF, card ghi "only used for research purposes", trong khi trang của nhóm vẫn ghi phải gửi email.
  - train_v0 đang dùng ViNLI, VSFC và ViSFD qua bản trong PhoLaya-base.
  - Câu hỏi này cần người làm dự án quyết định, không chỉ cho train_v1.
- **Giấy phép "không rõ":** aiMy144 (không có thẻ), k22-hcmus (không có card), ViHealthQA (không có thẻ), UTS trong PhoLaya. ViANLI có thẻ và thân card mâu thuẫn. VSMRC và NamSyntax có thẻ permissive đặt trên văn bản vốn có giấy phép khác (Wikipedia CC BY-SA, tin cào).
- **Rủi ro rò rỉ:** mọi bộ lấy từ Wikipedia hoặc tin tức cần qua `scripts/decontam.py` với dev_v1 và test 1.0 khi dựng. Bước đó phải đọc test 1.0, nên cần người làm dự án cho phép riêng.
- **Nhãn máy:** VSMRC, ViCS-MCQ và Domain-ViWikiQA do LLM sinh; aiMy144 là claim tổng hợp; vi-tourism-admin-intent sinh từ mẫu câu. Nếu dùng, nên gắn cờ để tách được khi phân tích.
- **Tốc độ dịch** ở mục 4 là ước lượng, chưa đo. Hệ số token Việt/Anh 1,4 là giả định.
- **UIT-ViON** chỉ có trên GitHub, chưa đếm và chưa đo độ dài.

**Sự cố (đã xử lý):**
- Subagent đã lấy mẫu từ bản sao bên thứ ba của ba bộ, trái với quy tắc "không tìm bản sao": VSMEC (nguồn gốc yêu cầu ký thỏa thuận), PhoATIS (tác giả cấm phân phối lại) và Zalo e2eqa (bản tải lên không chính thức của dữ liệu cuộc thi). **Ba file mẫu đã bị xóa**; bảng ghi "mẫu đã xóa".
- Tôi tải `sources/vsmec/train.jsonl` từ PhoLaya-base để đếm, trước khi biết VSMEC yêu cầu ký thỏa thuận. **File đã xóa.** Số đếm vẫn giữ trong bảng, kèm ghi chú.
- Trước khi thấy yêu cầu ký thỏa thuận, subagent đã đọc card, schema và một dòng đầu đã cắt ngắn của các bản sao không khóa của ViQuAD 2.0 và ViMMRC. Không lưu gì từ hai bộ này.

## 7. File

- `tables_sources.md`: bảng đầy đủ, gồm giấy phép chép nguyên văn và danh sách theo từng trạng thái.
- `raw/vi_candidates.json` (63 bộ) và `raw/mt_candidates.json` (44 model): kết quả subagent, đã đối chiếu.
- `pholaya_counts.json`, `sample_lengths.json`, `tasksource_pool.json`: số đo bằng script.
- `raw/pholaya/`: split train đã tải. `samples/`: 18 file mẫu khoảng 300 dòng.
- Code: `scripts/survey/{pholaya_counts,sample_lengths,tasksource_mt_pool,build_tables}.py`, `setup/survey_*.sh`.
