# Nguyên nhân lỗi của 2b-s1 và hướng sửa dữ liệu (2b′) — Fable 5.1, 29/10/2026

Chỉ phân tích trên CPU, một tiến trình; không chạy model, không GPU, không đọc `test_v1.3.jsonl`, không sửa file dự án ngoài file này. Mọi số tự tính lại bằng script ở `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2b2\` (`g6`–`g11`, kết quả chạy lưu cạnh trong `*_out.txt`; danh mục ở mục 7). Số của bản đọc trước (`FABLE_doc_ket_qua_2b_s1.md`, script `fable_2b/g1`–`g5`) được dẫn lại khi cần, ghi rõ.

"Bao phủ" (cover) trong toàn bài = tỉ lệ từ của nhận định có mặt trong **cả văn bản** (normalize của `decontam.py`: NFC, chữ thường, bỏ ký tự không phải chữ/số). Khác với phép đo "câu tốt nhất" của D6′. 2b-s2 đang train; các số có thể đổi khi s2 về được đánh dấu (s2?).

## Kết luận trước

1. **Khoảng cách tới 86,5 không nằm ở kiểm chứng.** Trên 2 nguồn kiểm chứng, 2b-s1 đã vượt decider-2b-text (86,34 so với 83,18 trung bình nguồn); trên 4 nguồn còn lại 2b-s1 thua 4,1 điểm nguồn (84,10 so với 88,16): MASSIVE −6,8 (= −1,13 macro), Belebele −5,1 (−0,85), XCOPA −4,0 (−0,67). Toàn bộ lỗi còn lại ở kiểm chứng chỉ đáng 4,55 điểm macro, và phần thực tế lấy lại được bằng sửa dữ liệu kiểm chứng là **+0,6 đến +1,3** → 85,4–86,1. **86,5 bằng 2b′ riêng là không chắc**; cần thêm hướng cho ý định/đọc hiểu (ngoài phạm vi 2b). (`g9` §1, §6)
2. **Cơ chế chung của cả ba lỗi là một: 2b học "nhận định không nằm nguyên trong văn bản → NEI / Không".** Trong khối kiểm chứng train (KC + 2d + 2e, có dấu), ở bao phủ < 0,5 nhãn NEI chiếm 76%, ở [0,5; 0,7) 38%; riêng KC: 100% / 100%, và KC noul có **0% "Có" khi bao phủ < 0,9**. Dev ViWikiFC cùng mức: NEI 52% / 25%, "Có" noul 24–40%. D6′ chỉ kiểm trung vị của NEI nên không bắt được. (`g6` §3)
3. **"NEI mặc định" khi không dấu là mẹo bao phủ lộ ra khi đọc kém, không phải prior nhãn.** Hồi quy trên 241 cặp orig↔nodiac: 2b p(NEI)nodiac ≈ 0,91 − 0,81·cover + 0,59·p(NEI)orig (C: 0,33 − 0,12·cover; decider: không phụ thuộc cover). ViFactCheck không dấu Xác nhận (bao phủ ≥ 0,9) p(NEI) chỉ 0,01→0,04; ViWikiFC Xác nhận bao phủ < 0,7: 0,07→0,52. KC không dấu (587 dòng, 100% dòng không dấu của 2b, C có 0) lặp đúng cấu trúc bao phủ của KC có dấu (Xác nhận 1,00 / Bác bỏ 0,97 / NEI 0,72). Prior nhãn không giải thích được: noul KC không dấu có "Có" 45% (cao hơn có dấu 39%) mà recall "Có" vẫn rơi 0,83→0,45. VNTC có dấu 100% → bỏ VNTC không làm mất tín hiệu không dấu nào. **Sửa không dấu không đóng góp vào 86,5** (VI macro là có dấu); xếp vào cột G1/độ bền. (`g6` §1, §5; `g7` §1–3)
4. **Bác bỏ ViWikiFC tụt vì KC không có một dòng Bác bỏ nào ở bao phủ < 0,7** (0/337; R-số và R-tên mọi hình dạng trung vị 0,94–0,97), trong khi 54% Bác bỏ dev ViWikiFC (và 55% proxy) ở dưới 0,7. Nguồn Bác bỏ bao phủ thấp duy nhất là 2d (67% < 0,7, n = 319, nhãn nhiễu: đọc mù trùng 18/30). 14/17 câu Bác bỏ sai bị đẩy sang NEI, tất cả ở bao phủ 0,35–0,76. (`g6` §3; `g8` §3)
5. **"Có" noul tụt là ngưỡng lệch + cùng mẹo.** 14 câu gold "Có" sai: 9 câu p(Có) < 0,1 (tự tin sai, max_ov 0,29–0,54, tức diễn đạt lại), 5 câu 0,2–0,5. Ngưỡng fit trên proxy (không trên dev) cho t* = 0,30: dev noul 88,6 → 90,6 (+4 câu, ≈ **+0,38 macro**, chi phí 0 GPU) nhưng proxy chỉ +0,7 điểm noul → yếu, phải kiểm lại trên s2. (`g8` §1–2)
6. **Ứng viên Bác bỏ-bao-phủ-thấp không cần chuỗi mới: ViANLI (relation) trong `raw_src`**, 8.012 cặp, contradiction 2.473 (bao phủ trung vị 0,62; 59% < 0,7), entailment 2.615 (0,45), neutral 2.924; 7.009 cặp chưa dùng trong C′. Tiền đề 1 câu (trung vị 27 chữ), thể loại báo → phải khử trùng D3 và ghép thêm tiền đề nhiễu. Mô phỏng 600/500/400 ghép 2 tiền đề: bao phủ Bác bỏ chỉ lên 0,67, khối kiểm chứng ở bao phủ < 0,5 về S/R/N 25/21/54 (dev ViWikiFC 23/26/52) và NEI ở [0,7; 0,9) giữ 32% — **với điều kiện giữ 400 neutral**; bỏ neutral thì NEI dải ấy tụt còn 26%. Rủi ro còn lại: dải [0,7; 0,9) là nơi ViFactCheck đòi NEI 61% mà ViWikiFC đòi Bác bỏ 57% → cần luật riêng "recall NEI ViFactCheck ≥ 0,75". (`g6` §4, `g12`)
7. **Train split ViWikiFC/ViFactCheck: không nên dùng.** Ngoài quy tắc cấm (`PLAN_decisions.md` dòng 25), **174/174 văn bản dev ViWikiFC và 112/113 văn bản dev ViFactCheck nằm nguyên trong train split gốc** (cùng đoạn, khác nhận định); ViWikiFC chỉ có 73 bài Wikipedia, 46 bài chạm dev. Loại theo bài thì ViWikiFC còn 1.927 dòng (loại theo nhóm uit_<a> + đoạn: 10.056), ViFactCheck còn 3.816 dòng — chưa trừ test. Dùng thì dev/test chỉ còn "ngoài mẫu" ở mức nhận định, không ở mức văn bản. (`g10`, `g11`)
8. **Thứ tự 2b′ theo lợi/chi phí:** (1) ngưỡng noul fit trên proxy; (2) khối ViANLI khung dev ≈ 1.500 dòng nghiêng Bác bỏ/Xác nhận bao phủ thấp; (3) không dấu: thêm bản không dấu cho lát 2d và cho dòng VI ngoài khung kiểm chứng (chỉ cho G1); (4) không dùng train split UIT; (5) phần còn lại của khoảng cách (MASSIVE/Belebele/XCOPA, +2,6 macro) để cho hướng khác. Bảng ở mục 6.

## 1. Vì sao bản không dấu rơi vào "NEI mặc định"

### 1a. Dữ liệu không dấu: C so với train_2b_v3 (`g6` §1)

| | C | train_2b_v3 |
|---|---|---|
| Dòng VI không dấu (strip_diacritics(văn bản) == văn bản) | **0 / 15.075** | **587 / 15.637** |
| Nguồn | — | kc_wiki_ngan 343, kc_wiki_dai 244; **0 dòng ngoài khung kiểm chứng** (2d, 2e, đọc hiểu, ý định, chủ đề: không có bản không dấu) |
| Choice không dấu, nhãn | — | NEI 144 / Xác nhận 111 / Bác bỏ 94 → **NEI 41,3%** (KC có dấu cũng 41,3%) |
| Noul không dấu | — | "Có" 108 / 238 = **45,4%** (KC có dấu 38,9%; dev 37,6%) |
| NEI không dấu theo kiểu | — | N-thêm 60, N-bỏ bằng 57, N-ghép 47, N-lạc đề 26, N-cùng bài 15 |
| Bao phủ KC không dấu | — | Xác nhận trung vị **1,00** (100% ≥ 0,9); Bác bỏ **0,97** (96% ≥ 0,9); NEI 0,72 (49% < 0,7) |

Vậy: với 2b, **văn bản không dấu chỉ xuất hiện trong khung kiểm chứng, và chỉ ở dạng KC** — nơi bao phủ tách nhãn gần tuyệt đối. C không có dòng không dấu nào, nhưng cũng không có dòng nào dạy "bao phủ thấp → NEI".

**Về VNTC** (`g6` §5): 1.000 dòng, 100% có dấu, 100% chữ thường, 100% không dấu câu, trung vị 362 token. Bỏ VNTC không làm mất tín hiệu đọc **không dấu** nào; chỉ mất văn bản dài chữ thường không dấu câu (SIB không dấu −3,3 có thể liên quan, nhưng n nhỏ, không tách được khỏi nhiễu).

**Prior nhãn không phải nguyên nhân:** NEI 41,3% như nhau ở KC có dấu và không dấu, còn "Có" ở noul không dấu còn cao hơn có dấu (45% so với 39%), mà recall "Có" không dấu lại rơi mạnh nhất.

### 1b. Preds: p(NEI) orig → nodiac theo nhãn gold × bao phủ (bao phủ tính trên bản orig, cùng pair_id; `g7` §1)

| Model | p(NEI) gộp orig → nodiac | Hồi quy p(NEI)nodiac ≈ a + b·cover + c·p(NEI)orig | corr(p nodiac, cover) |
|---|---|---|---|
| **2b-s1** | 0,33 → **0,53** | **0,91 − 0,81·cover** + 0,59·p_orig | **−0,53** |
| C-s1 | 0,16 → 0,30 | 0,33 − 0,12·cover + 0,33·p_orig | −0,24 |
| decider-2b-text | 0,33 → 0,33 | 0,38 − 0,06·cover − 0,04·p_orig | −0,06 |
| Qwen3-1.7B gốc | 0,02 → 0,29 | 0,41 − 0,16·cover | −0,09 |

Theo ô (2b-s1): ViWikiFC Xác nhận bao phủ < 0,7 (n = 27): 0,07 → **0,52**; [0,7; 0,9) (n = 14): 0,07 → 0,28; ≥ 0,9 (n = 5): 0,01 → 0,06. ViWikiFC Bác bỏ < 0,7 (n = 40): 0,32 → **0,71**; ≥ 0,9: 0,02 → 0,03. ViFactCheck Xác nhận ≥ 0,9 (n = 17): 0,01 → 0,04; ViFactCheck Bác bỏ [0,7; 0,9) (n = 10): 0,33 → 0,71. Tức là **khi mất dấu, 2b không đoán NEI đồng đều; nó đoán NEI đúng ở những dòng mà chữ của nhận định không nằm trong văn bản** — luật của KC. Khi dấu còn, mô hình đọc được nội dung và đè được luật này (hệ số cover trên bản có dấu: corr −0,27); khi dấu mất, đọc kém, chỉ còn luật.

Độ dài không phải yếu tố: ViWikiFC nodiac đoán NEI 56% ở cả hai dải token [0; 150) và [150; 400) (gold 28% / 18%); ViFactCheck (văn bản dài, bao phủ cao) đoán NEI 47% với gold 41% (`g7` §2).

Mọi model đều hỏng theo cách riêng trên không dấu: decider đổ về Xác nhận (96/161 ViWikiFC), base cũng vậy; C giữ phân bố nhãn gần gold nhất (65/56/40). Phần riêng của 2b là luật bao phủ.

**Noul không dấu** (`g7` §3): gold "Có", p(Không) 2b 0,20 → 0,54 (ViFactCheck) và 0,19 → 0,58 (ViWikiFC); C 0,06 → 0,13 và 0,13 → 0,33. Recall "Có" không dấu theo bao phủ orig: 2b 0,35 / 0,46 / 0,50 ở < 0,7 / [0,7; 0,9) / ≥ 0,9 (C 0,69 / 0,67 / 0,88). Ở noul, sự rơi không chỉ theo bao phủ: với 2 lựa chọn, "Không" là đáp án của cả Bác bỏ và NEI (62% noul train, 100% noul KC bao phủ < 0,9) nên "không chắc" → "Không".

**Proxy lặp lại** (`g7` §5): ViWikiFC không dấu choice 2b đoán NEI 35/72 (gold 24), recall Xác nhận 0,46 (C 0,73); noul recall "Có" 0,22 (C 0,78); ViFactCheck noul "Có" 0,42 (C 0,96).

**4 nguồn không kiểm chứng không dấu** (`g7` §4): không có dấu hiệu đổ về một nhãn riêng của 2b (acc, max-prob, tỉ lệ giữ dự đoán gần C; SIB 2b đoán "khoa hoc/cong nghe" 85 lần với gold 55, C 66). −2,0 gộp của báo cáo trước vẫn là nhiễu có chiều.

**Trả lời:** không phải KC không dấu nghiêng về NEI, không phải 2e/N-lạc đề không dấu (2e không có bản không dấu; N-lạc đề không dấu chỉ 26 dòng), không phải bỏ VNTC. Nguyên nhân là **toàn bộ 587 dòng không dấu đều là KC**, nơi "bao phủ < 0,9 ⇒ NEI (choice) / Không (noul)" đúng 97–100%; khi mất dấu model bám luật ấy.

## 2. Vì sao Bác bỏ ViWikiFC tụt ở bao phủ thấp

**P(nhãn | mức bao phủ), choice có dấu** (`g6` §3; S/R/N = Xác nhận/Bác bỏ/NEI, %):

| Tập | n | [0; 0,5) | [0,5; 0,7) | [0,7; 0,9) | ≥ 0,9 | Bác bỏ < 0,7 |
|---|---|---|---|---|---|---|
| train KC có dấu | 1.296 | n=172: **0/0/100** | n=117: **0/0/100** | n=158: 2/13/85 | n=849: 50/37/13 | **0%** (0/337) |
| train 2d | 1.132 | n=223: 33/35/32 | n=362: 44/38/18 | n=343: 67/25/8 | n=204: 87/9/4 | 67% (n=319) |
| train 2e | 250 | 0/0/100 | — | — | — | — |
| **train KC+2d+2e** | 2.678 | n=645: **11/12/76** | n=479: 34/28/**38** | n=501: 47/21/32 | n=1.053: 57/32/11 | 33% (n=656) |
| **dev ViWikiFC** | 161 | n=31: 23/26/**52** | n=69: 29/**46**/25 | n=51: 27/57/16 | n=10: 50/50/0 | **54%** (n=74) |
| proxy ViWikiFC | 72 | n=12: 25/8/67 | n=27: 37/41/22 | n=26: 38/35/27 | n=7: 43/14/43 | 55% (n=22) |
| dev ViFactCheck | 80 | — | n=5: 0/0/100 | n=28: 4/36/61 | n=47: 36/38/26 | 0% (n=28) |
| proxy ViFactCheck | 78 | — | n=3: 0/33/67 | n=22: 23/23/55 | n=53: 49/32/19 | 4% (n=23) |

- KC Bác bỏ theo kiểu × hình dạng: R-số nguyên/lược/ghép trung vị 0,97/0,97/0,97, R-tên 0,96/0,94/0,96, **0% dưới 0,7** ở cả 6 ô. Đổi một giá trị trong câu trích thì bao phủ toàn văn bản vẫn ≈ 1 (chỉ mất 1 từ); dạng "ghép" không hạ bao phủ vì cả hai mệnh đề đều ở trong văn bản. Không có cách cơ học nào trong KC tạo Bác bỏ bao phủ thấp mà không diễn đạt lại.
- Ở bao phủ < 0,7, model chỉ thấy Bác bỏ ở 2d (12% + 28% của khối), và 2d là bản dịch với nhãn nhiễu (đọc mù trùng 18/30, `DIEU_PHOI_doc_mu_vong1.md` dòng 20). Dev ViWikiFC Bác bỏ bao phủ < 0,7 có max_ov trung vị 0,55: diễn đạt lại thật.
- **Preds khớp** (`g8` §3): Bác bỏ dev ViWikiFC bao phủ [0,5; 0,7) n = 32: 2b đoán Bác bỏ 23, NEI 9 (C 27/2; decider 25/6). Mọi nhãn ViWikiFC: P(đoán NEI | bao phủ < 0,5) 2b 55%, gold 52% — **tỉ lệ đúng nhưng đặt vào sai câu** (precision NEI ViWikiFC 0,67, báo cáo trước): model dùng bao phủ thay cho nội dung. 17 câu Bác bỏ sai: 14 → NEI, 3 → Xác nhận; bao phủ 0,35–0,84; 10/17 có p(NEI) ≥ 0,75 (tự tin); C đúng 13/17, decider 9/17; chỉ 3/17 có từ phủ định → không phải mẹo phủ định (recall Bác bỏ có/không phủ định 0,87 / 0,73).
- Chi phí này **không thể sửa bằng tái lấy mẫu KC**: pool dư (700 ngắn + 400 dài) không có Bác bỏ bao phủ thấp nào (cùng luật dựng). Phải thêm nguồn có diễn đạt lại (mục 6, P2) hoặc dịch.

(s2?) Số 17 và phân bố 14/3 có thể đổi; cơ chế dữ liệu thì không.

## 3. Recall "Có" của câu Có/Không tụt 6–14 điểm

**Tỉ lệ "Có" trong train** (`g6` §2): mọi noul VI: C 40,2% (n = 3.004) → 2b 39,5% (n = 4.177): không đổi. Trong **khung kiểm chứng**: C 52,8% (n = 106, vitaminc/nli_fever khung cũ) → 2b 39,9% (n = 1.383; KC 1.355 + 2d 28), khớp dev 37,6%. Prior theo tỉ lệ không phải nguyên nhân. NEI trong noul KC: 449/1.355 = 33% (dev ViWikiFC noul 35%, ViFactCheck 22%) — không "quá nhiều".

**Nguyên nhân là P("Có" | bao phủ)** (`g6` §3, cuối):

| Bao phủ | train KC noul có dấu (n = 1.117) | dev noul 2 nguồn (n = 202) |
|---|---|---|
| [0; 0,5) | n=121: **"Có" 0%** | n=21: 24% |
| [0,5; 0,7) | n=87: **0%** | n=53: 40% |
| [0,7; 0,9) | n=119: **0%** | n=67: 36% |
| ≥ 0,9 | n=790: 55% | n=61: 43% |

Trong KC, một phát biểu chỉ là "Có" khi nó nằm nguyên trong văn bản (S-nguyên, S-lược, S-ghép đều bao phủ ≈ 1). Dev có 70% phát biểu "Có" ở bao phủ < 0,9. Model học "không nằm nguyên → Không", đúng với 76% noul không-"Có" của KC bao phủ thấp.

**Preds** (`g8` §1): 14 câu gold "Có" sai trên dev có dấu — 9 câu p(Có) 2b ≤ 0,10 (C 0,93–0,99 ở 7 câu đó; max_ov 0,29–0,54, tức diễn đạt lại); 5 câu p(Có) 0,32–0,44. Ở 126 câu gold "Không", 2b chỉ có 9 câu p(Có) ≥ 0,5 và 1 câu trong [0,3; 0,5) → model rất tự tin "Không". Recall "Có" theo bao phủ: 0,85 / 0,71 / 0,88 (< 0,7 / [0,7; 0,9) / ≥ 0,9) — ở noul, lỗi tập trung ở dải giữa, không đơn điệu như choice, vì nhiều phát biểu "Có" bao phủ thấp ở dev là của ViWikiFC mà model vẫn đọc được.

**Ngưỡng** (`g8` §2; fit trên proxy noul có dấu n = 150, không fit trên dev): 2b t* = 0,30 (proxy 0,853 → 0,860); áp lên dev: noul 88,6 → **90,6** (ViFactCheck 91,0 → 94,9, recall Có 0,79 → 0,90, recall Không giữ 0,98; ViWikiFC 87,1 → 87,9). Quy macro ≈ +0,32 + 0,06 = **+0,38**. Đường dev theo t không trơn (0,30: 90,6; 0,35: 89,6; 0,45: 88,1; 0,50: 88,6) và proxy chỉ tăng 1 câu → bằng chứng yếu; đọc lại trên s2 trước khi dùng. Với C, t* = 0,76 (ngược chiều): ngưỡng là đặc tính của từng model, đúng tiền lệ `REPORT_sensitivity`.

## 4. Còn bao nhiêu điểm có thể lấy

**Trọng số:** 1 câu nguồn s = 100/(6·n_s) điểm macro: XCOPA 0,167; ViFactCheck 0,105; ViWikiFC 0,058; Belebele/SIB/MASSIVE ≈ 0,057. +1,7 = 16 câu ViFactCheck, hoặc 29 ViWikiFC, hoặc 10 XCOPA, hoặc 30 câu một nguồn khác. SD bootstrap cụm của hiệu cặp macro ≈ 1,0–1,1 (2b − C: +3,18 [+0,91; +5,34]; decider − 2b: +1,68 [−0,30; +3,74]) → +1,7 chỉ đọc được trên trung bình 2 seed. (`g9` §1, §6, §8)

**Lỗi 2b-s1 còn lại theo nguồn** (dev VI có dấu; `g9` §1):

| Nguồn | n | 2b-s1 | C-s1 | decider-text | sai 2b | trần nếu sửa hết | decider − 2b quy macro |
|---|---|---|---|---|---|---|---|
| Belebele | 294 | 83,0 | 84,0 | 88,1 | 50 | +2,83 | **+0,85** |
| SIB-200 | 299 | 81,9 | 79,9 | 82,3 | 54 | +3,01 | +0,06 |
| MASSIVE | 294 | 89,5 | 88,1 | 96,3 | 31 | +1,76 | **+1,13** |
| XCOPA | 100 | 82,0 | 85,0 | 86,0 | 18 | +3,00 | **+0,67** |
| ViFactCheck | 158 | 89,9 | 72,2 | 83,5 | 16 | +1,69 | −1,05 |
| ViWikiFC | 285 | 82,8 | 80,7 | 82,8 | 49 | +2,87 | 0,00 |
| macro | | 84,84 | 81,65 | 86,50 | | | +1,66 |

**Kiểm chứng, theo kiểu × nhãn** (`g9` §2; tổng 4,55 điểm):

| Ô | n | sai 2b | điểm | sai C | sai decider | 2b đoán nhầm thành |
|---|---|---|---|---|---|---|
| ViFactCheck choice Xác nhận | 18 | 2 | 0,21 | 3 | 1 | Bác bỏ 2 |
| ViFactCheck choice Bác bỏ | 28 | 2 | 0,21 | 4 | 7 | NEI 2 |
| ViFactCheck choice NEI | 34 | 6 | 0,63 | 27 | 11 | Bác bỏ 4, Xác nhận 2 |
| ViFactCheck noul "Có" | 29 | **5** | 0,53 | 1 | 3 | Không 5 |
| ViFactCheck noul Bác bỏ / NEI | 32 / 17 | 0 / 1 | 0,11 | 3 / 6 | 2 / 2 | |
| ViWikiFC choice Xác nhận | 46 | 7 | 0,41 | 6 | 8 | Bác bỏ 4, NEI 2 |
| ViWikiFC choice Bác bỏ | 74 | **17** | **0,99** | 9 | 15 | NEI 14, Xác nhận 3 |
| ViWikiFC choice NEI | 41 | 9 | 0,53 | 22 | 12 | Xác nhận 5, Bác bỏ 3 |
| ViWikiFC noul "Có" | 47 | **9** | 0,53 | 6 | 4 | Không 8 |
| ViWikiFC noul Bác bỏ / NEI | 34 / 43 | 1 / 6 | 0,41 | 0 / 12 | 1 / 9 | |

Ba ô "có thể sửa bằng dữ liệu" (cùng cơ chế bao phủ): Bác bỏ ViWikiFC 0,99 + noul "Có" 1,06 + Xác nhận choice 0,62 = **2,67 điểm trần**. NEI còn sai 22 câu (1,62 điểm) là NEI kiểu khác (diễn đạt lại, suy đoán) — khó, và sửa bao phủ có thể làm mất một phần NEI đã được.

**Cận trên hợp nhất** (`g9` §3): câu decider đúng mà 2b sai ở 2 nguồn kiểm chứng = 9 + 26 câu = +2,47 macro (nếu lấy hết → 87,31); macro hợp nhất 2b ∪ decider trên 6 nguồn = 92,55. Câu 2b sai "gần đúng" (biên < 0,2): kiểm chứng 11 câu (+0,83), 4 nguồn khác 17 câu (+1,29); phần lớn lỗi là tự tin sai (p(gold) < 0,1: 27/49 ViWikiFC, 8/16 ViFactCheck). (`g9` §4)

**Có thực tế không?** Giả định hợp lý cho 2b′ (lấy lại ½ Bác bỏ ViWikiFC bị mất, ½ noul "Có", vài Xác nhận, mất 1–2 NEI): **+0,6 đến +1,3 trên dev**, tức 85,4–86,1; cộng ngưỡng noul +0,2–0,4 (chồng một phần với noul "Có"). Đầu lạc quan mới chạm 86,5; với SD 1,0 thì "VI > 86,5" không khẳng định được bằng một seed. Trên proxy/test kỳ vọng còn khoảng ½–⅔ (báo cáo trước: ViFactCheck proxy +8,7 so với dev +17,7). Phần +2,6 macro ở MASSIVE/Belebele/XCOPA là của hướng khác (ý định, đọc hiểu, nhân quả — các nguồn này bị cấm train và không có nguồn thay trong C′ ngoài vi-tourism/multidogo/banking77/clinc cho ý định; lỗi MASSIVE là nhầm ý định gần nghĩa: "đọc danh sách" ↔ "trả lời kiến thức", `g9` §5).

## 5. Ràng buộc đã kiểm (`PLAN_decisions.md`, manifest proxy)

- Dòng 25: **ViFactCheck, ViWikiFC là nguồn cấm train** (quy tắc thường trực, không chỉ chuyện proxy). Dòng 31 / 85(b): dữ liệu UIT chỉ dùng nội bộ, model chỉ dùng nội bộ — ViANLI (UIT, research-only / CC BY-NC-SA) đã có trong C′ (1.000 dòng entailment) dưới cùng điều kiện.
- Dòng 9 + `REPORT_giai_doan_0.md` mục 5: trần 5% mỗi nguồn ở mức **tác vụ**, tính trên thành phần chứa nguồn. Một khối "vianli khung kiểm chứng" là tác vụ mới (như vinli/relation khác vinli/entailment) → trần ≈ 5% × 31.600 ≈ **1.580 dòng**.
- `manifest_proxy_v1.json`: proxy dựng từ train split ViWikiFC/ViFactCheck, khử trùng với dev/test/train_v0/C, "KHÔNG BAO GIỜ TRAIN TRÊN TẬP NÀY"; dùng để chọn checkpoint và fit ngưỡng noul → P1 hợp lệ.
- D3 (brief 2b mục 4): KC khử trùng với **mọi split** ViWikiFC/ViFactCheck ở mức bài. Mọi khối mới (ViANLI) phải qua cùng D3, kể cả test (worker chạy, không in).

**Nếu vẫn muốn dùng train split UIT** (`g10`, `g11`; không đọc test):

| | ViWikiFC train | ViFactCheck train |
|---|---|---|
| Dòng / bài / văn bản | 16.738 / **73 bài Wikipedia** (856 nhóm uit_<a>) / 1.479 đoạn | 5.062 / 1.035 bài báo / 1.035 văn bản |
| Văn bản dev nằm nguyên trong train split | **174/174** (1.625 dòng train, 175 đoạn) | **112/113** (508 dòng) |
| Bài chạm dev | 46/73 title; 157/157 nhóm uit_<a> (3.725 dòng) | 112 bài |
| Văn bản proxy nằm trong train split | 150/150 (2.228 dòng); 24 nhóm uit_<a> chung với dev | 150/150 (738 dòng, 150 bài) |
| Còn lại sau loại đoạn + nhóm uit_<a> + proxy | 10.056 dòng (S/R/N 3.348/3.364/3.344; Bác bỏ trung vị 0,81, 25% < 0,7) | 3.816 dòng (773 bài; 1.322/1.250/1.244) |
| Còn lại nếu loại theo **bài** (title / Url) | **1.927 dòng, 26 title** | 3.816 dòng |
| Chưa trừ test 1.3 | cùng cỡ dev → kỳ vọng mất thêm tương tự | như trên |

Ghi chú: bao phủ Bác bỏ của train split thô (ViWikiFC 30% < 0,7; ViFactCheck 4%) lệch xa dev/proxy (54–55% ở ViWikiFC) mà chưa giải thích được (có thể do cách chọn mẫu dev); không dùng số thô này để so với ViANLI.

Đánh đổi: dev và test được dựng từ dev/test split, nhưng **cùng đoạn văn** với train split. Train trên phần "khác đoạn, cùng bài" vẫn là học thuộc 46 bài Wikipedia mà dev/test hỏi; proxy mất tư cách ngoài mẫu (proxy ⊂ train split, cùng bài). Quy tắc tối thiểu nếu người dùng đảo quyết định dòng 25: (i) loại theo **bài** (title / Url) với dev, test (decontam, không in), proxy → ViWikiFC còn ≈ 1.900 dòng từ 26 bài, không đáng; ViFactCheck còn ≈ 3.800 dòng nhưng chưa trừ test; (ii) ghi điểm ViWikiFC/ViFactCheck là "cùng bộ" và tách khỏi tuyên bố 86,5; (iii) proxy dựng lại từ bài không dùng train; (iv) ghi sai lệch có ngày. Khuyến nghị: **không làm** trong 2b′.

## 6. Bảng phương án 2b′ (xếp theo lợi ích kỳ vọng / chi phí)

Ước lượng điểm là trên **dev VI có dấu**, một seed, với SD ≈ 1,0; dải là chủ quan dựa trên số câu ở mục 4 và tỉ số proxy/dev ≈ ½ của báo cáo trước.

| # | Phương án | Nhắm lỗi | Điểm lấy lại (dev; test/proxy ≈ ½–⅔) | Rủi ro | Cổng dữ liệu cần thêm | Chuỗi VI mới? | Chi phí |
|---|---|---|---|---|---|---|---|
| **P1** | **Ngưỡng noul** là bước hậu xử lý: fit t* trên proxy noul có dấu **cho từng adapter** (2b-s1 ra 0,30, C-s1 ra 0,76 — ngưỡng là đặc tính của model, không chuyển giữa model hay seed) | noul "Có" (14 câu) | **+0,2 đến +0,4** (đo được +0,38; đọc trên dev sau khi fit trên proxy) | proxy chỉ +1 câu → có thể là nhiễu; đường chấm hiện dùng argmax nên phải sửa `score_probe` (hoặc bước hậu xử lý trên preds) và **đăng ký trước khi đọc s2**; so với decider (không ngưỡng) là không cùng điều kiện → báo thành dòng riêng, không gộp vào T4 | Luật: fit trên proxy, báo dev, không fit trên dev; làm lại cho s2 và cho mọi adapter 2b′; ghi vào bảng chỉ báo | Không | 0 GPU, 1 script |
| **P2** | **Khối ViANLI khung dev** (như 2d): relation → Xác nhận/Bác bỏ/NEI, câu hỏi và chuỗi nhãn chép từ dev; ≈ 1.500 dòng (≤ trần 5%), cơ cấu **Bác bỏ 600 / Xác nhận 500 / NEI 400 — phải giữ phần neutral**; 55/45 choice/noul; state = tiền đề thật + **2** tiền đề của dòng khác (đã khử trùng, không chung tên riêng/số với câu nói), **vị trí tiền đề thật ngẫu nhiên** (tránh mẹo vị trí như MLQA); **không** lấy 1.003 cặp đã có trong C′. **Mô phỏng** (`g12`, mẫu 600/500/400, seed cố định): ghép 2 chỉ nâng bao phủ Bác bỏ ViANLI 0,62 → 0,67 (53% vẫn < 0,7), state trung vị 84 chữ (ViWikiFC 134); khối KC+2d+2e+ViANLI có S/R/N theo dải [0; 0,5) 25/21/**54** (hiện 11/12/76; dev ViWikiFC 23/26/52), [0,5; 0,7) 36/31/33 (hiện 34/28/38), [0,7; 0,9) **37/31/31** (hiện 47/21/32; NEI giữ nguyên nhờ neutral), ≥ 0,9 50/37/14. Biến thể bỏ neutral (300 R + 300 S) kéo NEI ở [0,5; 0,7) xuống 28% và ở [0,7; 0,9) xuống 26% → **không làm** | Bác bỏ ViWikiFC bao phủ thấp (17 câu), noul "Có" diễn đạt lại (14), Xác nhận choice (9); hạ P(NEI \| bao phủ < 0,5) của khối từ 76% về ≈ 54% (dev 52%) mà không hạ NEI ở [0,7; 0,9) | **+0,4 đến +0,9** (lấy 5–9 Bác bỏ, 3–6 "Có", 1–3 Xác nhận; có thể mất 1–3 NEI) | (a) **xung đột dải [0,7; 0,9)**: ViFactCheck dev đòi NEI 61% ở dải này (17/28 câu, chính là nơi +21 NEI của 2b-s1), ViWikiFC đòi Bác bỏ 57%; P2 đưa Bác bỏ ở dải này từ 21% lên 31% → nguy cơ mất NEI ViFactCheck; đo bằng luật riêng (cột cổng); (b) thể loại báo VnExpress, trùng kho ViFactCheck → D3 bắt buộc với dev/test/proxy/ViFactCheck mọi split; (c) nhiễu ghép có thể vô tình xác nhận/bác bỏ (xác suất thấp; lọc không chung tên/số như 2e); (d) D7: NEI khối 36,3% → ≈ 34% (vẫn ≥ 30); (e) 1.000 cặp đã ở C′ dạng noul: không dùng lại cùng cặp | D1, D2, D3 (mọi split UIT + dev + test + proxy), D4 chỉ đọc nhận định, D5 phủ định (contradiction 17% / entailment 14% → cân khi lấy mẫu), **D6″ theo từng dải**: khối kiểm chứng có Bác bỏ ≥ 25% ở [0; 0,7), và NEI ở [0,7; 0,9) không giảm quá 5 điểm so với 32% hiện tại; sau ghép đo lại: Bác bỏ ViANLI trung vị ≤ 0,70 và ≥ 45% dưới 0,7 (không đạt thì giảm số câu nhiễu); D7; đọc mù 30 dòng/nhãn. **Luật đọc bậc 1 thêm: recall NEI ViFactCheck choice ≥ 0,75** (s1: 0,82), tách khỏi T2 gộp | **Không** (câu hỏi, nhãn chép dev; tiền đề và câu nói là của ViANLI) | CPU vài giờ dựng + 1 lần train ≈ 4 h GPU |
| P2b | Biến thể của P2 không thêm nguồn: bớt 200 dòng N-bỏ bằng / N-lạc đề / N-cùng bài (bao phủ thấp) trong KC, bù bằng N-thêm/N-ghép — **không khả thi**: pool dư chỉ còn 9 N-thêm + 2 N-ghép | như P2 | ≈ 0 nếu không dựng lại pool | T2 (recall NEI) có thể tụt | — | Không | Phải dựng lại KC |
| P3 | Nhân đôi 319 dòng Bác bỏ 2d (bao phủ trung vị 0,60) | Bác bỏ bao phủ thấp | +0,1 đến +0,3 | nhãn 2d nhiễu (đọc mù trùng 18/30) → nhân đôi nhiễu; bản dịch | D5, D7 | Không | 0 CPU đáng kể |
| **P4** | **Không dấu (chỉ cho G1, 0 điểm vào 86,5):** (a) **bỏ dấu tại chỗ** (thay chính dòng đó, không nhân bản — đúng tiền lệ brief 2b mục 3) ≈ 300 dòng 2d (có Xác nhận/Bác bỏ bao phủ thấp) và ≈ 10% dòng VI ngoài khung kiểm chứng (đọc hiểu 2f, ý định, chủ đề) để văn bản không dấu không chỉ là KC; tổng dòng không đổi, không tạo cặp có dấu/không dấu của cùng nhận định; (b) giữ 587 dòng KC không dấu; nếu có P2 thì 20% khối ViANLI cũng bỏ dấu tại chỗ | NEI mặc định không dấu (ViWikiFC 91/161), "Có" noul không dấu 0,45 | không dấu macro +1 đến +2 (G1); **0** cho VI có dấu | tiền lệ strip_v0: +2/+3 không ý nghĩa; tăng độ dài token 15–20% ở dòng không dấu; phần bỏ dấu phải cân nhãn như phần có dấu (chọn theo hash id, không theo nhãn) | Kiểm chuỗi nhãn không dấu trùng dev (assert như KC); D7 trên phần không dấu | Không (strip_diacritics) | CPU phút, cùng lần train với P2 |
| P5 | Train split ViWikiFC/ViFactCheck | mọi lỗi kiểm chứng | lớn trên dev nhưng **không sạch** (mục 5) | cấm (dòng 25); 174/174 và 112/113 văn bản dev nằm trong train split; proxy mất ngoài mẫu; chỉ còn 1.927 (ViWikiFC) / 3.816 (ViFactCheck) dòng nếu loại theo bài, chưa trừ test | đảo quyết định có ngày; D3 theo bài với dev/test/proxy; proxy mới; báo điểm "cùng bộ" riêng | Không | Không khuyến nghị |
| P6 | Dịch thêm vitaminc/nli_fever (EN còn trong tasksource) để tăng Bác bỏ bao phủ thấp | như P2 | +0,2 đến +0,5 | nhãn FEVER nhiễu như 2d; cần agent dịch (người dùng cử), chi phí Opus; chuỗi dịch máy phải báo riêng | D1–D7 như 2d | **Có** (bản dịch; đi qua quy trình dịch của dự án, không qua tôi) | Cao |
| — | Ngoài 2b′: MASSIVE (+1,13), Belebele (+0,85), XCOPA (+0,67) so decider | ý định gần nghĩa, đọc hiểu, nhân quả | +1 đến +2,5 nếu có nguồn | nguồn bench bị cấm; cần nguồn ý định/đọc hiểu mới hoặc 2c | — | có thể | hướng riêng |

**Gói đề nghị cho 2b′:** P1 (ngay, đọc lại khi s2 về) + P2 + P4, một lần train 2 seed nếu bậc 1 đạt. Kỳ vọng dev VI có dấu **85,4–86,1**, không dấu +1–2, Bác bỏ ViWikiFC về ≥ 0,82, "Có" noul về ≥ 0,88. Luật đọc nên đặt trước: T2 ≥ 0,70 (giữ ≥ 7/8 NEI đã được), **recall NEI ViFactCheck choice ≥ 0,75** (tách riêng, vì P2 đẩy Bác bỏ vào dải [0,7; 0,9)), T3a precision NEI ≥ 0,70, **T3c mới: recall Bác bỏ ViWikiFC ≥ 0,82 và recall "Có" noul gộp ≥ 0,88** (đo ở argmax, trước ngưỡng P1), G1 ≥ 63,35 (2b-s1), và chỉ báo "tỉ lệ đoán NEI ViWikiFC không dấu ≤ 40%". Ngưỡng P1 báo thành dòng riêng. Nếu muốn 86,5 có cổng, phải nói rõ với người dùng rằng 2b′ một mình chỉ tới được ≈ 86 ở đầu lạc quan; phần còn lại là hướng ý định/đọc hiểu.

**Số sẽ đổi khi 2b-s2 về:** 17 Bác bỏ ViWikiFC sai, 91/161 đoán NEI không dấu, 14 noul "Có" sai, t* của ngưỡng noul, các ước lượng cột "điểm lấy lại". Cơ chế (P(nhãn | bao phủ) trong train) không đổi.

## 7. Script và file đã dùng

Thư mục `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2b2\`; đường dẫn gốc truyền qua tham số (mặc định `C:/Users/dungbt06/Fine-tune/`); `PYTHONIOENCODING=utf-8`.

| Script | Việc | Đọc |
|---|---|---|
| `g6_train_comp.py` | dòng không dấu C/2b theo nguồn, nhãn, kiểu, bao phủ; prior khung kiểm chứng; **P(nhãn \| bao phủ)** train/dev/proxy; KC Bác bỏ theo kiểu × hình dạng; noul P("Có" \| bao phủ); ViANLI thô theo nhãn; VNTC; pool dư | `C.jsonl`, `train_2b_v3.jsonl`, `dev_v1_3.jsonl`, `proxy_v1.jsonl`, `raw_src/.../vianli/train.jsonl`, `kc_pool_*_v2.jsonl`, `scripts/common.py` |
| `g7_nodiac_preds.py` | cặp orig↔nodiac: p(NEI) theo gold × bao phủ × model, hồi quy; phân bố nhãn đoán; theo độ dài; noul p(Không); 4 nguồn khác; proxy nodiac | dev, `eval/dev/{2b-s1,C-s1,decider-2b-text,Qwen3-1.7B}/preds.jsonl`, `train_2b/eval/proxy/*/preds.jsonl` |
| `g8_noul_refuted_preds.py` | 14 câu noul "Có" sai; p(Có) theo nhãn; ngưỡng fit trên proxy, đọc trên dev; Bác bỏ ViWikiFC theo bao phủ (2b/C/decider); 17 câu sai; phủ định | như trên |
| `g9_headroom.py` | điểm theo nguồn, trọng số câu, trần; lỗi kiểm chứng theo kiểu × nhãn; ô cặp 2b × decider, hợp nhất; biên; 4 nguồn khác theo nhãn; kịch bản +1,7; nodiac/EN các model; bootstrap cụm SD | dev, preds 5 model |
| `g10_uit_split_check.py` | ViWikiFC `train.csv`, ViFactCheck `train.parquet` (WSL `.venv`, pyarrow): cột, nhãn, bao phủ, trùng văn bản/bài với dev và proxy | `raw_refs/High-Will__ViWikiFC/data/train.csv`, `raw_refs/tranthaihoa__vifactcheck/data/train-*.parquet`, dev, proxy |
| `g11_uit_clean_remainder.py` | phần còn lại của train split sau khi loại đoạn/bài chạm dev và proxy (WSL) | như trên |
| `g12_sim_p2.py` | mô phỏng P2: nối mẫu ViANLI 600/500/400 (và 300R/300S) vào khối KC+2d+2e, state thô và ghép 1–3 tiền đề nhiễu; P(nhãn \| bao phủ), bao phủ theo nhãn, độ dài state. Không ghi file dữ liệu | `train_2b_v3.jsonl`, ViANLI thô |
| `kcf.py` | hàm đo (chép từ `fable_2b/kcf.py`: normalize, cover, max_ov, claim_of) | — |

Không đọc `test_v1.3.jsonl`; phần chạm test của ViANLI và train split UIT chưa kiểm, ghi là việc của worker qua `decontam.py`. Không chạy model, không GPU, không tải gì.
