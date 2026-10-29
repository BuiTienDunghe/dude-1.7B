# Thêm dữ liệu tiếng Việt tốt mà không cần người gán nhãn — Fable 5.1

Ngày 2026-10-27. Phạm vi: chỉ mảng dữ liệu (văn bản chưa nhãn, cách tạo nhãn tự động, bộ có nhãn bỏ sót, công thức dữ liệu). Công thức của đối thủ và phương pháp train do hai agent khác lo. Chỉ đọc; không tải trọng số hay dataset, không chạy model. Đã dùng ~40 lượt web. Đọc thêm trong repo: `BOI_CANH_chung.md`, `audit_2710/DU_LIEU_fable.md`, `train_v1_survey/REPORT_sources.md`, `BRIEF_train_chinh_thuc.md`, `config/bench_v1_questions.json`.

## 0. Kết luận

1. **Đòn bẩy không nằm ở số dòng mà ở dòng đúng khung, có văn bản, NEI đủ nhiều và văn bản dài.** Audit đã đo: C@25% ≈ C@100%, decider-ft-ep2 < decider-2b gốc, NEI trong train 12,8% so với 31,1% ở dev, 1.109 dòng VI không có văn bản (`DU_LIEU_fable.md` mục 2–3). Vì vậy công thức dưới đây chỉ thêm **≈ 4.700–5.200 dòng VI mới (+500 tuỳ chọn)**, dựng **bằng chương trình** từ văn bản chưa nhãn, sau khi làm bản C′ của audit (mục 5, việc 1–4, 6). Không đề xuất dùng kho web lớn để tiền huấn luyện tiếp: 21.000 dòng/giờ không chứa được, và đó là mảng của agent phương pháp. **[suy luận]**
2. **Nguồn văn bản:** Wikipedia tiếng Việt (1,3 triệu bài, 398 triệu chữ) cho kiểm chứng kiểu ViWikiFC; FineWeb-2 `vie_Latn` (61 triệu tài liệu, ODC-By) hoặc HPLT v2 (CC0) cho văn bản dài kiểu báo, **sau khi loại hẳn 9 tên miền báo nguồn của ViFactCheck và ba dự án Wikimedia nguồn của FLORES**. Kho báo binhvq đã ngừng phân phối và không có giấy phép dữ liệu — bỏ.
3. **Cách tạo nhãn:** (a) dựng bằng chương trình là cách duy nhất **không sinh chữ tiếng Việt mới** (câu nhận định là câu trích hoặc câu bị đổi số/tên; câu hỏi và nhãn dùng chuỗi bench đã duyệt) → chi phí duyệt ≈ 30 dòng đọc mẫu mỗi kiểu dựng. (b) Nhãn giả từ Quyet-Medium (Apache-2.0, card không cấm dùng đầu ra) dùng để **lọc và làm đích mềm**, không dùng để tạo văn bản; phải đo P4 của teacher trước, vì có thể chép luôn thói quen "bác bỏ". (c) Sinh bằng LLM và (d) dịch máy đều tạo chữ mới phải duyệt 100% (tiền lệ: 6.685 dòng Sonnet → sửa 5,2% khi đọc lại) và (d) vướng quyết định 17/10 (BRIEF 0(c)); chỉ giữ cho việc 8 của audit. (e) Bản không dấu dựng sẵn và xáo lựa chọn: 0 chữ mới.
4. **Bộ có nhãn bỏ sót: không tìm được bộ nào đạt cả ba điều kiện** (chưa có trong khảo sát cũ, giấy phép rõ, khớp nhiệm vụ yếu). XQuAD-vi (CC BY-SA, 1.190 câu, chỉ có validation), mMARCO-vi (Apache-2.0 nhưng dịch máy), XNLI-vi (train dịch máy, ô giấy phép trống) chỉ dùng phụ.
5. **Phát hiện kiểm được: config chuỗi nhãn NEI đã cũ.** `config/bench_v1_questions.json` ghi lựa chọn `"Văn bản không đủ thông tin"` (không đuôi); đếm trên `dev_v1_3.jsonl`: **241** chuỗi lựa chọn kết thúc bằng `để kết luận"`, **0** chuỗi `"Văn bản không đủ thông tin"` trần. Vậy dev dùng đuôi "để kết luận", 593 dòng vitaminc trong C học chữ không đuôi (audit 2.2), và giả thuyết Pa −16,8 của audit đứng vững. Người dựng dữ liệu phải chép chuỗi nhãn **từ file dev**, không từ config, và chạy probe đổi chữ nhãn. **[có nguồn: đếm cục bộ, chỉ số lượng]**

## 1. Văn bản tiếng Việt chưa nhãn

| Nguồn | Giấy phép | Quy mô vi | Chất lượng | Rủi ro trùng bench và cách khử |
|---|---|---|---|---|
| Wikipedia tiếng Việt | CC BY-SA 4.0 (văn bản Wikimedia; chưa fetch trang giấy phép) **[suy luận]** | 1.305.414 bài, 398.397.329 chữ **[có nguồn: https://vi.wikipedia.org/wiki/Đặc_biệt:Thống_kê]** | Nhiều bài sơ khai do bot (loài sinh vật, đơn vị hành chính); cần lọc bài ≥ 150 chữ, ≥ 3 đoạn **[suy luận]** | **ViWikiFC** lấy >20K nhận định từ câu bằng chứng Wikipedia **[có nguồn: https://arxiv.org/abs/2405.07615]**; không tìm được danh sách bài. Khử: `scripts/decontam.py` (trùng, MinHash ≥ 0,8, 13-gram) với dev + test 1.3 (đọc test đã duyệt 17/10) **và mọi split của ViWikiFC** (BRIEF 3.1), loại cả bài nếu bất kỳ câu nào chạm |
| FineWeb-2 `vie_Latn` | ODC-By **[có nguồn: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2]** | 61.064.248 tài liệu, 50,9 tỉ chữ, 121 GB; 96 snapshot CC 2013–4/2024; có cả phần `removed` **[có nguồn: cùng URL, README]** | Đã lọc và khử trùng toàn cục; tin tức, diễn đàn, thương mại lẫn nhau | **ViFactCheck** lấy bài 2–3/2023 từ 9 báo: Báo Chính phủ, VnExpress, Dân Trí, Người Lao Động, Tuổi Trẻ, Tin Tức, Pháp Luật TP.HCM, Thanh Niên, Tiền Phong **[có nguồn: https://arxiv.org/html/2412.15308v1]** → loại hẳn 9 tên miền theo trường URL, đơn giản hơn lọc theo ngày, cộng decontam với mọi split ViFactCheck. **Belebele** dịch tay từ FLORES tiếng Anh (Wikinews/Wikijunior/Wikivoyage, 842 bài) **[có nguồn: https://ar5iv.labs.arxiv.org/html/2106.03193]** → bản vi của các trang này không chứa đúng văn bản đó, nên bảo vệ thật là decontam theo chữ; loại thêm `vi.wikinews.org`, `vi.wikivoyage.org`, `vi.wikibooks.org` chỉ là lớp phòng xa **[suy luận]**. Thực tế: 121 GB, chỉ stream 1–2 shard parquet, mỗi lần tải là một bước cần người dùng duyệt |
| HPLT v2 cleaned | CC0 **[có nguồn: https://arxiv.org/html/2503.10267]** | 1,007×10⁸ tài liệu, 8,32×10¹⁰ token vi; Internet Archive 2012–2020 + CC 2014–2022; lọc xác suất ngôn ngữ ≥ 0,5, MinHash, điểm chất lượng ≥ 5, ≥ 500 ký tự **[có nguồn: cùng URL]** | Tương tự FineWeb-2; cũ hơn nên ít trùng ViFactCheck 2023 **[suy luận]** | Như trên |

Không cần hoặc bỏ: **CulturaX vi** (57,6 triệu tài liệu, 55,4 tỉ token, giấy phép theo mC4/OSCAR, phải chấp nhận điều kiện; cùng gốc CC nên thừa khi đã có FineWeb-2) **[có nguồn: https://huggingface.co/datasets/uonlp/CulturaX]**; **binhvq news-corpus** (MIT chỉ cho mã, chủ repo ghi rõ không áp cho dữ liệu, đã gỡ link, lưu trữ 8/2026 → bỏ, kể cả bản sao ROOTS) **[có nguồn: https://github.com/binhvq/news-corpus]**; **sailor2/community-dataset `vietnamese_news`** (snapshot ghi odc-by, snapshot khác trống → chưa rõ) **[có nguồn: https://huggingface.co/datasets/sailor2/community-dataset]**.

Lưu ý giấy phép: ODC-By/CC0 của FineWeb-2 và HPLT chỉ phủ **bản tổng hợp**, không phủ bản quyền từng bài báo; với quyết định "chỉ dùng nội bộ" (BRIEF 0(b)) thì được, nhưng phải xem lại nếu sau này công bố. **[suy luận]**

## 2. Biến văn bản chưa nhãn thành câu hỏi quyết định

### 2a. Dựng bằng chương trình (khuyến nghị chính)

Bằng chứng: QACG sinh nhận định ba nhãn từ Wikipedia không cần người: SUPPORTED từ cặp hỏi–đáp theo thực thể; REFUTED thay đáp án bằng thực thể **cùng loại** (top-5 Sense2Vec, có luật giảm trùng từ); NEI lấy thực thể từ **5 câu khác của cùng bài** không nằm trong bằng chứng. RoBERTa zero-shot tăng F1 từ 50 lên 77, bằng "2K+ ví dụ người gán"; 2/100 câu thay thực thể vô tình vẫn đúng **[có nguồn: https://ar5iv.labs.arxiv.org/html/2105.14682]**. FEVER gốc: người gán đổi câu theo 6 kiểu (diễn đạt lại, phủ định, thay thực thể giống/khác, tổng quát hoá, cụ thể hoá) và **bị can ngăn dùng "not" tầm thường**; NEI train lấy bằng chứng từ trang gần nhất (NearestP) hoặc câu ngẫu nhiên (RandomS, "không chỉ khác nghĩa mà còn không liên quan") **[có nguồn: https://ar5iv.labs.arxiv.org/html/1803.05355]**.

| Cách dựng | Nhãn | Chất lượng | Rủi ro mẹo và luật dựng |
|---|---|---|---|
| Câu trích nguyên từ đoạn (có thể lược một mệnh đề) | Xác nhận | Đúng gần 100% | Mẹo "nằm trọn trong văn bản": 8,8% giả thuyết entailment SNLI nằm trọn trong tiền đề **[có nguồn: https://ar5iv.labs.arxiv.org/html/1803.02324]**. Luật: ≥ 1/3 câu Xác nhận lấy từ câu **ghép lại** hai mệnh đề cách nhau hoặc lược bớt; không để 100% trích nguyên văn **[suy luận]** |
| Đổi số, ngày, tên riêng bằng giá trị **cùng loại lấy từ cùng bài hoặc bài cùng chủ đề** | Bác bỏ | ~98% (QACG 2/100 sai) | **Không dùng phủ định**: từ phủ định là tín hiệu mạnh nhất của nhãn mâu thuẫn, bộ phân loại chỉ đọc giả thuyết đạt 67,0% SNLI / 53,9% MNLI **[có nguồn: Gururangan, URL trên]**. Giữ độ dài câu Bác bỏ = Xác nhận (±20% token) |
| NEI cùng chủ đề: câu lấy từ **phần khác của cùng bài**, không có trong đoạn | NEI | ~90% (câu có thể được đoạn suy ra gián tiếp) **[suy luận]** | Đây là kiểu NEI của ViWikiFC/ViFactCheck. Phải chiếm phần lớn NEI, vì NEI lạc đề quá dễ |
| NEI lạc đề: câu từ **bài khác** (RandomS) | NEI | ~100% | Chữa trực tiếp P4 (C chọn Bác bỏ 110/241 khi văn bản lạc đề). Nhưng dạy mẹo "khác chủ đề → NEI" giống NSP học dự đoán chủ đề thay vì mạch văn **[có nguồn: https://arxiv.org/html/1909.11942v6]** → chỉ ~1/4 số NEI |
| Chọn câu tiếp theo, nhiễu từ **cùng bài** (kiểu SOP) | Đọc hiểu dài | Tốt nếu nhiễu cùng bài; nhiễu khác bài thì chỉ dạy chủ đề (ALBERT: NSP vô hiệu vì gộp dự đoán chủ đề và mạch văn; SOP hơn NSP trên RACE 64,0 so với 62,3) **[có nguồn: cùng URL]** | Câu hỏi là một chuỗi cố định mới → **1 chuỗi VI cần duyệt**; đoạn 500–900 token để nằm dưới trần 1.024 |
| Chọn tiêu đề hoặc chuyên mục từ metadata (tiêu đề bài, đường dẫn URL) | Đọc hiểu dài / chủ đề | Tiêu đề: tốt; chuyên mục: phụ thuộc trang | Nhiễu lấy từ bài **cùng chuyên mục** để tránh mẹo chủ đề. Chủ đề kiểu SIB: audit ghi VNTC dùng lại câu hỏi SIB với 10 nhãn khác gây −2,7, nên nếu làm thì dùng đúng 7 chữ nhãn SIB đã có trong bench, hoặc không làm (SIB đang ngang) **[suy luận]** |

Chi phí: 0 giờ GPU, 0 chữ VI do agent viết cho nhận định (chỉ sao chép hoặc đổi cơ học), câu hỏi và nhãn dùng chuỗi đã duyệt; duyệt ≈ 30 dòng đọc mẫu mỗi kiểu dựng (tiền lệ audit), tức ~150–180 dòng, cộng 1–2 chuỗi câu hỏi mới nếu thêm "chọn câu tiếp theo"/"chọn tiêu đề".

### 2b. Nhãn giả từ teacher cục bộ

Bằng chứng: Du et al. — teacher gán nhãn câu lấy từ web, giữ **mẫu tin cậy cao nhất theo từng lớp, giữ đúng tỉ lệ lớp của tập gốc**; đích mềm (KL trên softmax của teacher) hơn nhãn cứng 0,9 điểm; ở cùng số câu, câu **liên quan** cho 81,9 còn câu **ngẫu nhiên** chỉ 77,1 **[có nguồn: https://ar5iv.labs.arxiv.org/html/2010.02194]**. Ánh xạ sang dự án: loss full_vocab đã nhận đích mềm trên chữ A–J, nên lưu softmax của teacher trên k chữ cái làm target là tự nhiên **[suy luận]**.

- **Teacher:** Quyet-1.0-Medium, Apache-2.0, card không có điều khoản về đầu ra, không công bố dữ liệu train **[có nguồn: https://huggingface.co/chinhnc/Quyet-1.0-Medium]**; decider-2b Apache-2.0 nhưng card ghi English **[có nguồn: https://huggingface.co/Mapika/decider-2b]**. Việc chưng cất từ model đang được dùng làm mốc so là **quyết định của người dùng**, không phải kỹ thuật.
- **Cách dùng đề xuất:** không dùng teacher để tạo văn bản; dùng để (i) **lọc** dòng dựng bằng chương trình (bỏ dòng teacher đặt > 0,5 cho một nhiễu, hoặc mâu thuẫn với nhãn chương trình ở câu tiếp theo/tiêu đề), (ii) làm **đích mềm** pha với nhãn cứng; (iii) lấy mẫu theo tỉ lệ nhãn của dev, không theo đầu ra tự nhiên của teacher. **[suy luận]**
- **Rủi ro chặn:** P4 của Quyet chưa đo; nếu teacher cũng nghiêng "Bác bỏ" khi lạc đề thì nhãn giả chép luôn lỗi. Phải chạy probe P4 trên teacher (một lần chạy GPU → cần duyệt) trước khi gán hàng loạt.
- **Chi phí:** 1 lượt forward mỗi dòng, Quyet 4,66B bf16 ≈ 9,3 GB vừa 16 GB; 5.000 dòng ước < 1 giờ GPU **[suy luận]**, nên đo thử 50 dòng như tiền lệ dịch của BRIEF. 0 chữ VI mới.

### 2c. Sinh bằng LLM — không khuyến nghị lúc này

Claude tốn hạn mức tuần và BRIEF 3.4 đã ghi điều khoản Anthropic về đầu ra; model mở có tiếng Việt và không ràng buộc đầu ra là Sailor2 1B/8B/20B (Apache-2.0, 15 ngôn ngữ gồm vi; bản 8B thực tế 9B tham số ≈ 18 GB bf16, phải 4-bit) **[có nguồn: https://huggingface.co/sail/Sailor2-8B-Chat]** và Qwen3-4B (khảo sát cũ). Mọi dòng sinh là chữ VI mới → duyệt 100% (tiền lệ: 6.685 dòng Sonnet sửa 5,2%, Opus soát 200 dòng); chỉ cân nhắc khi (a)+(b) chạm trần.

### 2d. Dịch ngược / dịch máy

Bằng chứng ngược: lợi thế translate-train đến từ **tạo tác dịch** (giảm trùng từ giữa tiền đề và giả thuyết), model train trên dữ liệu dịch ít dựa vào trùng từ và phủ định hơn nhưng kém hơn trên bài kiểm năng lực; chênh ~2 điểm XNLI **[có nguồn: https://ar5iv.labs.arxiv.org/html/2004.04721]**. Nội bộ: 6.685 dòng dịch cho ViFactCheck −0,6. Quyết định 17/10 (BRIEF 0(c)): không dùng model dịch tại máy, dịch do agent khác người dùng cử — không đề xuất đổi. Chỉ giữ việc 8 của audit (dịch kiểm chứng có bằng chứng dài) nếu C′ và công thức dưới vẫn < 84.

### 2e. Tăng cường

- **Bỏ dấu dựng sẵn** 20% khối mới, gắn cờ (audit việc 9), và **bỏ dấu riêng phần lựa chọn** cho 10–15% dòng kiểm chứng nhắm Pa ViWikiFC −16,8: 0 chữ mới **[suy luận]**.
- **Đổi diễn đạt câu hỏi:** model nhạy với định dạng prompt tới 76 điểm, không hết khi instruction-tune **[có nguồn: https://arxiv.org/abs/2310.11324]**; mỗi diễn đạt mới là một chuỗi phải duyệt → chỉ 1–2 biến thể, dùng cho hàng rào P5.
- **Xáo lựa chọn:** đã làm; cờ không xáo noul/score thuộc agent phương pháp.

## 3. Dữ liệu có nhãn bỏ sót

Không tìm được bộ mới đạt cả ba điều kiện. Ứng viên phụ:

| Bộ | Giấy phép | Quy mô vi | Khớp nhiệm vụ yếu | Hạn chế |
|---|---|---|---|---|
| XQuAD vi **[có nguồn: https://huggingface.co/datasets/google/xquad]** | CC BY-SA 4.0 | 1.190 câu hỏi, 240 đoạn, chỉ split validation, dịch chuyên nghiệp từ SQuAD dev | Đọc hiểu đoạn + câu hỏi riêng (Belebele) | Phải tự dựng nhiễu; đoạn Wikipedia EN dịch; kiểm trùng với MLQA (vào bench 1.4) |
| mMARCO vi **[có nguồn: https://huggingface.co/datasets/unicamp-dl/mmarco]** | Apache-2.0 | bộ ba (câu hỏi, đoạn đúng, đoạn sai), 14 ngôn ngữ | Có/Không "đoạn có trả lời câu hỏi không" với **phủ định thật**, văn bản dài | Dịch máy toàn bộ |
| XNLI vi **[có nguồn: https://huggingface.co/datasets/facebook/xnli]** | Card để trống | train 392.702 (dịch máy), val 2.490, test 5.010 | NLI ba nhãn, có "trung lập" | Trùng kiểu với ViNLI đã có; giấy phép không ghi |
| Ý định, kiểm chứng tiếng Việt mới (2025–26) | — | — | — | Hai lượt tìm không ra bộ nào có giấy phép rõ ngoài ViFactCheck/ViWikiFC (bị cấm) |

## 4. Công thức dữ liệu đề xuất (sau C′ của audit)

Mọi số là gợi ý **[suy luận]**; văn bản đã qua decontam mục 1; câu hỏi và nhãn chép từ **file dev** (xem mục 5).

| Điểm yếu đo được | Khối mới | Nguồn văn bản | Dòng | Cấu tạo | Chữ VI mới cần duyệt |
|---|---|---|---|---|---|
| ViFactCheck −10,8; ViWikiFC −6,3; P4 62,5/50,3%; NEI train 12,8% | **KC-Wiki**: kiểm chứng đúng khung bench | Wikipedia vi, bài ≥ 150 chữ không phải sơ khai bot, đoạn 80–400 token | 1.500 | Xác nhận 35% / Bác bỏ 30% / NEI 35% (NEI cùng bài 25 + lạc đề 10); Bác bỏ chỉ đổi số/ngày/tên, cấm phủ định; độ dài ba lớp khớp nhau | 0 (câu hỏi, nhãn đã duyệt); đọc mẫu 30 dòng × 4 kiểu |
| ViFactCheck dài (p50 728 token), trần 1.024 | **KC-Báo-dài**: như trên, văn bản 500–900 token | FineWeb-2 vi, loại 9 tên miền ViFactCheck + Wikimedia, chọn tài liệu có cấu trúc bài báo | 1.000 | Cùng tỉ lệ nhãn; NEI cùng bài lấy từ đoạn cuối bài; có thể lọc bằng teacher (2b) | 0; đọc mẫu 30 |
| Belebele −5,4; văn bản dài; khối dịch sai khung | **ĐH-Tiếp**: chọn câu tiếp theo, 4 lựa chọn cùng bài | Wikipedia vi + FineWeb-2 (đã lọc) | 800 | Nhiễu = 3 câu khác của cùng bài, cách ≥ 2 câu; câu hỏi riêng ở ô câu hỏi | **1 chuỗi câu hỏi** |
| Belebele; chủ đề nền | **ĐH-Tiêu-đề**: chọn tiêu đề đúng, 4 lựa chọn | FineWeb-2 bài báo có tiêu đề | 500 | Nhiễu = tiêu đề bài cùng chuyên mục (theo URL) | **1 chuỗi câu hỏi** |
| Pa ViWikiFC −16,8; không dấu | **Bỏ dấu dựng sẵn** | 20% của 4 khối trên; thêm 10–15% dòng KC chỉ bỏ dấu lựa chọn | ≈ 900 | Gắn cờ `nodiac` để chấm riêng | 0 |
| MASSIVE −9,5 | Không có đòn bẩy từ văn bản chưa nhãn | — | 0 | Giữ việc 5 của audit (recast banking77/clinc, ~74 nhãn) | theo audit |
| SIB ngang | Tuỳ chọn: chủ đề từ chuyên mục URL với đúng 7 chữ nhãn SIB | FineWeb-2 | 0–500 | Chỉ nếu người dùng chấp nhận dùng lại chữ nhãn bench | 0 |

Tổng ≈ 4.700–5.200 dòng VI mới (+ 500 tuỳ chọn), 0 giờ GPU để dựng, < 1 giờ GPU nếu dùng teacher lọc, 2 chuỗi VI mới, ~180 dòng đọc mẫu. Giữ trần 5% mỗi nguồn và ghi `label_source = program|teacher`, `text_source`, `license` vào manifest.

## 5. Kiểm trước khi train

1. **Chuỗi nhãn:** 100% dòng mới dùng đúng chuỗi câu hỏi và 3 chuỗi nhãn **chép từ dev 1.3** (dev: 241/241 lựa chọn NEI có đuôi "để kết luận", 0 chuỗi trần); cập nhật `bench_v1_questions.json` cho khớp (việc sửa config cần người dùng duyệt); đổi luôn 593 dòng vitaminc trong C′ sang chuỗi có đuôi.
2. **Decontam:** 0 chạm (trùng, MinHash ≥ 0,8, 13-gram) với dev + test 1.3 **và mọi split của ViWikiFC, ViFactCheck** (BRIEF 3.1); 0 tài liệu từ 9 tên miền ViFactCheck và 3 trang Wikimedia vi.
3. **Probe chỉ-nhận-định** (kiểu Gururangan): bộ phân loại nhẹ chỉ đọc nhận định trên khối KC; nếu vượt đa số > 10 điểm thì kiểu đổi câu để lộ mẹo → sửa. **[suy luận]**
4. **Thống kê theo nhãn:** độ dài token, tỉ lệ từ phủ định, tỉ lệ số/ngày, tỉ lệ trích nguyên văn; ba lớp phải xấp xỉ nhau.
5. **Phân bố nhãn** toàn khối kiểm chứng VI (C′ + mới) ≈ 35/30/35, NEI không dưới 30%.
6. **Đọc mẫu 30 dòng mỗi kiểu dựng**, ghi tỉ lệ sai rõ; ngưỡng chấp nhận ≤ 5% như tiền lệ audit.
7. **Nếu dùng teacher:** đo P4 và P2 của teacher trước; ghi tỉ lệ dòng bị lọc theo nhãn để bảo đảm không lọc lệch về một lớp.
8. **Hàng rào sau train:** báo P4, Pa, P5 cùng điểm dev, vì khối mới dùng đúng khung bench.
