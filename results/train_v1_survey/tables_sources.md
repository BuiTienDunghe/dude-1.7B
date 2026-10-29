# Bảng khảo sát nguồn tiếng Việt cho train_v1

Số dòng train và revision của mọi ứng viên trên HF đã đối chiếu lại bằng API (`api/datasets/<id>` và dataset viewer `/size`): khớp 100%. Độ dài: token Qwen3 (tokenizer bộ đếm của train_v0) trên ~300 dòng split train lấy qua dataset viewer.

## 1. Nhóm 1 — phần còn lại của nguồn tiếng Việt đã dùng (LakoreAI/PhoLaya-base @ `034d4273`)

Mỗi dòng của PhoLaya là một câu hỏi trên một văn bản; một văn bản có thể mang nhiều câu hỏi (vd. ViNLI: relation + entailment trên cùng cặp câu).

| Bộ | Tác vụ (câu hỏi) | Kiểu | Câu hỏi trong train | Đã dùng (train_v0 + val_v0) | Còn lại | Trung vị state (token) | Dung lượng file train |
|---|---|---|---|---|---|---|---|
| vinli | relation | choice | 24,363 | 700 | 23,663 | 66 | 34.4 MB |
| vinli | entailment | noul | 24,363 | 700 | 23,663 | 66 | 34.4 MB |
| vsfc | sentiment | choice | 11,395 | 700 | 10,695 | 13 | 11.7 MB |
| vsfc | topic | choice | 11,395 | 700 | 10,695 | 13 | 11.7 MB |
| visfd | aspect | choice | 77,860 | 700 | 77,160 | 42 | 57.9 MB |
| visfd | complaint | noul | 7,786 | 700 | 7,086 | 42 | 57.9 MB |
| uts | sentiment | choice | 1,808 | 0 | 1,808 | 22 | 2.1 MB |
| uts | aspect | choice | 1,850 | 0 | 1,850 | 22 | 2.1 MB |
| vianli | relation | choice | 8,012 | 0 | 8,012 | 60 | 10.6 MB |
| vianli | entailment | noul | 8,012 | 0 | 8,012 | 60 | 10.6 MB |
| victsd | constructiveness | choice | 6,996 | 0 | 6,996 | 40 | 8.5 MB |
| victsd | toxicity | choice | 6,995 | 0 | 6,995 | 40 | 8.5 MB |
| vsmec | emotion | choice | 5,544 | 0 | 5,544 | 15 | 3.9 MB |

vsmec: đã đếm từ bản trong PhoLaya trước khi biết nguồn gốc yêu cầu ký thỏa thuận; file tải về đã xóa. viglue (586 MB) và massive (bench_v1) không đếm.

Nguồn tiếng Việt sẵn có trong tasksource-jev-typed-decisions (`multilingual/*vi`):

| Nguồn | Dòng | Đã dùng | Bị loại (bench) | Kiểu | Trung vị token cả dòng |
|---|---|---|---|---|---|
| multilingual/xcopa/translation-vi | 30 | 0 | có | choice | 36 |
| multilingual/xcopa/vi | 31 | 0 | có | choice, noul | 42 |
| multilingual/xcsr/X-CODAH-vi | 109 | 0 | không | choice, noul | 100 |
| multilingual/xcsr/X-CSQA-vi | 139 | 2 | không | choice, noul | 51 |

## 2. Nhóm 2 — bộ tiếng Việt khác, có split train, truy cập mở

| Bộ | Repo @ revision | Dòng train | Dạng bài | Cách đổi | Trung vị state / cả dòng (token) | Nhãn | Truy cập | Rủi ro trùng bench_v1 |
|---|---|---|---|---|---|---|---|---|
| VSMRC/mrc (Vietnamese Text Segmentation and Multiple-Choice  | VSMRC/mrc @ `13c4449c` | 11426 | multiple-choice reading comprehension (passage + question + 2-4 option | choice: state = `context` (passage, <=~1000 chars), question = `question`, options = `choices` (297/300 sampled rows have 4 options, a few 2-3), gold = `correct | 214 / 264 | mixed | open | HIGH vs ViWikiFC (both Wikipedia-derived; check passage/title overlap before mixing). Also Apache-2.0 tag on Wikipedia-d |
| EXAMS (Vietnamese subset) - mhardalov/exams, config crosslin | mhardalov/exams @ `4ff10804` | 1955 | high-school exam multiple-choice QA (subjects incl. Biology, grade 12) | choice: question = `question.stem`, options = `question.choices.text`, gold = `answerKey` (A-D). State: the plain config has none; `crosslingual_with_para_vi` a | 8 / 96 | human-annotated | open | LOW-MEDIUM: Wikipedia paragraphs are only in the *_with_para configs (ViWikiFC overlap possible there); exam stems thems |
| ViCS-MCQ (Vietnamese Computer Science MCQ) | p-storm/ViCS-MCQ @ `72076a98` | 1085 | context-grounded MCQ (OS and Database lecture notes), 4 options | choice: state = `context` (lecture paragraph), question = `question`, options = [A,B,C,D], gold = `answer` letter. | 67 / 144 | machine-generated | open | none with Wikipedia/news; but CC BY claim over third-party lecture notes (PTIT authors) is questionable. |
| HienNGuit/Domain-ViWikiQA (Vietnamese WikiQA, domain-labelle | HienNGuit/Domain-ViWikiQA @ `bf7019a4` | 6074 | generative/extractive QA over Wikipedia section chunks (no options); r | noul: state = `context`, question = `question`, ask 'is this proposed answer correct?' using `answer` vs a distractor answer from another row (distractors must  | 299 / 323 | mixed | open | HIGH vs ViWikiFC (Wikipedia). |
| ViBidLQA_v1 (Vietnamese bidding-law QA) | ntphuc149/ViBidLQA_v1 @ `dc439e13` | 1928 | extractive/generative legal QA over a law article | noul/choice requires synthetic distractors (no options in data). State = `context` (law provision), question = `question`. | 152 / 188 | mixed | open | none with Wikipedia/news; narrow legal domain. |
| ura-hcmut/zalo_e2eqa (Zalo AI Challenge E2E QA, open-domain  | ura-hcmut/zalo_e2eqa @ `63494521` | 600 | open-domain factoid QA, question + short answer, no passage | none useful (no passage, no options). Could be used only as question-only items (e.g. noul 'is X the answer?') with synthesised negatives. | mẫu đã xóa (bản sao bên thứ ba) | unknown | open | unknown (answers are Wikipedia-style facts). |
| GreenNode/zalo-ai-legal-text-retrieval-vn (Zalo AI Challenge | GreenNode/zalo-ai-legal-text-retrieval-vn @ `12d76d4d` | 2505 | legal passage retrieval (query, relevant article) with graded score | noul: state = legal article text (from `corpus`), question = query, label = relevant yes/no using qrels (negatives from non-relevant articles). Requires a join  | — | human-annotated | open | none with Wikipedia/news. |
| aiMy144 Vietnamese fact-checking verifier data v3.1 (claims  | aiMy144/vietnamese-fact-checking-verifier-data-v @ `6bba4242` | 58017 | claim verification: (evidence_text, claim) -> SUPPORTED / REFUTED / NO | choice(3): state = `evidence_text`, question = `claim`, options = [SUPPORTED, REFUTED, NOT_ENOUGH_INFO]; or noul (claim supported?). Filter to Vietnamese rows f | 50 / 99 | machine-generated | open | HIGH vs ViFactCheck (news and fact-check-portal sources incl. VnExpress, VAFC). Not Wikipedia. |
| k22-hcmus/fact-checking-vietnamese-news | k22-hcmus/fact-checking-vietnamese-news @ `6d690dd7` | 4140 | claim verification with news evidence passages, labels support / refut | choice(3): state = joined `evidence_text`, question = `claim`, options = [support, refute, nei]. | 62 / 99 | unknown | open | HIGH vs ViFactCheck (news-derived). |
| VNTC 10-topics news classification (HF copy kornwtp/VNTC-10T | kornwtp/VNTC-10Topics @ `30b4f927` | 33759 | news topic classification, 10 classes (Chinh tri Xa hoi, Doi song, Kho | choice(10): state = article `text`, question = null, options = the 10 topic names. Texts are long (p50 1.5k chars, p90 3.9k, max 27k) - truncate to first ~1k ch | 402 / 448 | unknown | open | MEDIUM vs ViFactCheck (news domain, but old 2007 articles so low chance of the same documents). |
| NamSyntax/vietnamese-news-classification (1.3M news, 11 topi | NamSyntax/vietnamese-news-classification @ `b6ee76ee` | 1348717 | news topic classification, 11 classes (Thoi su, The gioi, Kinh doanh,  | choice(11): state = `text` (title + description/sapo, p50 188 chars), question = null, options = 11 topic names (sample uses the ClassLabel names from the card  | 57 / 103 | unknown | open | HIGH vs ViFactCheck (news). Scraped-news redistribution under Apache-2.0 is legally doubtful. |
| ngocleltt/vi-news-4topics-classification | ngocleltt/vi-news-4topics-classification @ `c7d8c884` | 1909 | news headline topic classification, 4 classes (world, sport, tech, bus | choice(4): state = headline `text` (p50 52 chars), question = null, options = 4 topics. | 16 / 24 | unknown | open | HIGH vs ViFactCheck (VnExpress news). |
| vanhai123/vietnamese-news-dataset (tiny topic set) | vanhai123/vietnamese-news-dataset @ `b3169868` | 858 | news headline/short text topic classification (labels like giao duc, t | choice: state = `content`, question = null, options = label set (sample file leaves options null; the label list was not enumerated). | 14 / 15 | unknown | open | HIGH vs ViFactCheck (news). |
| phamluan/vi-tourism-admin-intent (Vietnamese tourism & admin | phamluan/vi-tourism-admin-intent @ `a092e219` | 14983 | intent classification, 8 intents (ask_admin_change, ask_location_mappi | choice(8): state = user utterance `text` (p50 55 chars), question = null (or a fixed 'what does the user want?'), options = 8 intent names (should be rewritten  | 18 / 59 | machine-generated | open | none with Wikipedia/news (place names from administrative tables). |
| PhoATIS (VinAI, Vietnamese ATIS) - HF copy ura-hcmut/PhoATIS | https://github.com/VinAIResearch/JointIDSF/tree/main/PhoATIS @ `HF mirro` | 4478 | airline-travel intent detection (utterance -> intent; some labels are  | choice(~24 observed labels, heavily skewed: 'flight' 3,284 of 4,478 train rows): state = utterance, question = null, options = intent names (write Vietnamese de | mẫu đã xóa (bản sao bên thứ ba) | human-annotated | open | none with Wikipedia/news/social media. |
| ViANLI (Vietnamese Adversarial NLI) - uitnlp/ViANLI | uitnlp/ViANLI @ `0fec8d6e` | 8012 | natural language inference, 3-way (entailment / neutral / contradictio | choice(3): state = premise, question = hypothesis, options = [entailment, neutral, contradiction]; or noul 'does the premise entail the hypothesis'. NLI is seco | 38 / 68 | mixed | open | MEDIUM-HIGH vs ViFactCheck (news-derived; premises are news sentences). |
| XNLI Vietnamese (facebook/xnli, config vi) | facebook/xnli @ `b8dd5d7a` | 392702 | natural language inference, 3-way | choice(3): state = premise, question = hypothesis, options = [entailment, neutral, contradiction]. Use only validation/test if you want human translations; trai | 32 / 60 | machine-translated | open | none with Wikipedia/news/social (but MNLI government genre). |
| UIT-ViCTSD (Constructive and Toxic Speech Detection) - tarud | tarudesu/ViCTSD @ `65a073f2` | 7000 | two binary tasks on news comments: Constructiveness (0/1) and Toxicity | noul: state = `Comment` (optionally with Title as context), question = 'Is this comment toxic?' / 'Is it constructive?', label from Toxicity / Constructiveness. | 27 / 28 | human-annotated | open | LOW vs ViHSD (ViCTSD comes from VnExpress news comments, ViHSD from Facebook/YouTube); MEDIUM vs news-derived sets (VnEx |
| UIT-ViOCD (Complaint detection) - tarudesu/ViOCD | tarudesu/ViOCD @ `25e18516` | 4387 | binary complaint detection on e-commerce product reviews (label 0.0/1. | noul: state = `review`, question = 'Is the customer complaining?'; label from `label`. | 24 / 24 | human-annotated | open | MEDIUM vs UIT-ViSFD (already used; the mobile category is also smartphone reviews). |
| UIT-VSMEC (emotion) - HF copy ura-hcmut/UIT-VSMEC | ura-hcmut/UIT-VSMEC @ `ab642b18` | 5548 | emotion classification, 7 labels (Enjoyment, Sadness, Anger, Fear, Dis | choice(7): state = `Sentence`, question = null, options = 7 emotions. | mẫu đã xóa (bản sao bên thứ ba) | human-annotated | open | MEDIUM vs ViHSD-like social-media text (different collection, but same platform type). |
| ViHealthQA (Vietnamese healthcare QA) - tarudesu/ViHealthQA | tarudesu/ViHealthQA @ `d90a62de` | 7009 | question -> answer passage matching (SPBERTQA); question + long answer | noul (as implemented in the sample file): state = `answer` text, question = `question`, label = whether this answer addresses the question (positives = the give | 114 / 146 | unknown | open | MEDIUM vs news-derived sets (VnExpress health pages). |
| UIT-ViON (online news titles, 13 topics) - GitHub kh4nh12/UI | https://github.com/kh4nh12/UIT-ViON-Dataset @ `a0877b93` | — | news title topic classification (TECHNOLOGY, TRAVEL, EDUCATION, ENTERT | choice(13): state = title, question = null, options = topics. | — | mixed | open | HIGH vs ViFactCheck (news outlets incl. VnExpress, Dan Tri). |

### Cần email / ký thỏa thuận / HF-gated — chỉ liệt kê, không tải

| Bộ | URL | Ghi chú |
|---|---|---|
| UIT-ViQuAD 1.0 / 2.0 (Vietnamese MRC; 2.0 adds ~12K unanswerable quest | https://github.com/kietnv/VietnameseDatasets | email-or-agreement; DISCLOSURE: ungated HF mirrors of this agreement-only corpus exist (taidng/UIT-ViQuAD2.0 [sha 406f09a4...], tuanquocbd/, nhphuc210/, noname002/ ...). I came across taidng/UIT-ViQuAD2.0 in a name searc |
| UIT-ViNewsQA (health-news MRC) | https://github.com/kietnv/VietnameseDatasets | email-or-agreement; No HF copy was searched for (rule). Not downloaded. |
| ViMMRC 1.0 (Vietnamese multiple-choice MRC, grades 1-5 textbook passag | https://github.com/kietnv/VietnameseDatasets | email-or-agreement; Ungated third-party mirrors exist (ura-hcmut/ViMMRC, stair-lab/ViMMRC, Angelectronic/ViMMRC, bachelo/VimMRC). Per the authoritative source the data needs a signed agreement, so no sample was taken and |
| ViMMRC 2.0 - uitnlp/vimmrc2.0 | https://huggingface.co/datasets/uitnlp/vimmrc2.0 | HF-gated + signed agreement by email; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. gated='auto' on HF plus an emailed agreement per GitHub. Nothing downloa |
| ViSpamReviews (spam review detection) - sonlam1102/vispamdetection (v1 | https://huggingface.co/datasets/sonlam1102/vispamdetection | HF-gated; Official repo is gated (auto) and the authors ask readers to contact them. The ungated visolex mirror was NOT sampled. Low relevance (spam detection). |
| ViLegalTF (legal true/false QA) | https://huggingface.co/datasets/ntphuc149/ViLegalTF | HF-gated; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. Part of the ViLegalLM suite. Narrow legal domain. |
| ViLegalMCQ (legal multiple-choice QA) | https://huggingface.co/datasets/ntphuc149/ViLegalMCQ | HF-gated; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. Part of the ViLegalLM suite. Narrow legal domain. |
| ViLegalNLI (legal citation sufficiency NLI) | https://huggingface.co/datasets/ntphuc149/ViLegalNLI | HF-gated; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. Part of the ViLegalLM suite. Narrow legal domain. |
| VLSP 2025 legal MCQ (ura-hcmut/vlsp2025-mcq-non-reasoning and -reasoni | https://huggingface.co/datasets/ura-hcmut/vlsp2025-mcq-non-reasoning | HF-gated; Form-gated: nothing downloaded. Non-commercial. Also in the collection: vlsp2025-qa-*, legal-document-detection/revision, legal-corpus. |
| VLSP 2025 legal citation-usefulness NLI (ura-hcmut/vlsp2025-nli-non-re | https://huggingface.co/datasets/ura-hcmut/vlsp2025-nli-non-reasoning | HF-gated; Form-gated: nothing downloaded. |
| UIT-VlogQA (QA over Vietnamese vlog transcripts) - uitnlp/vlogqa | https://huggingface.co/datasets/uitnlp/vlogqa | HF-gated + signed agreement by email; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. |
| UIT-ViCoQA (conversational QA on health news) - uitnlp/vicoqa | https://huggingface.co/datasets/uitnlp/vicoqa | HF-gated + signed agreement by email; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. |
| UIT-ViGoEmotions (27-way emotion) - uitnlp/vigoemotions | https://huggingface.co/datasets/uitnlp/vigoemotions | HF-gated; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. |
| ViTHSD (targeted hate speech) - sonlam1102/vithsd | https://huggingface.co/datasets/sonlam1102/vithsd | HF-gated; gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. |
| UIT-ViWikiQA (sentence-extraction MRC) - SEACrowd/uit_viwikiqa loader | https://huggingface.co/datasets/SEACrowd/uit_viwikiqa | email-or-agreement; The HF repo ships only a loader: 'This is a local dataset. You have to obtain this dataset separately from [homepage]'. Term 3 forbids distributing edited versions, which conflicts with releasing a fi |

### Không có split train

| Bộ | URL | Ghi chú |
|---|---|---|
| MLQA Vietnamese (facebook/mlqa, configs mlqa.vi.* / mlqa.*.vi) | https://huggingface.co/datasets/facebook/mlqa | open; Also bundled in google/xtreme. Dev/test only (511/5,495 for vi). |
| VMLU (Vietnamese multitask language understanding benchmark; mirror tr | https://vmlu.ai | open; Benchmark: do not train on it. Not sampled. |
| M3Exam Vietnamese (chiayewken/m3exam config vietnamese; also SeaLLMs/S | https://huggingface.co/datasets/chiayewken/m3exam | open; Evaluation benchmark (dev 28 / test 1,789). |
| VNHSGE (Vietnamese National High School Graduation Exam) - Elfsong/vnh | https://huggingface.co/datasets/Elfsong/vnhsge-v | open; Third-party upload; test only. Original VNHSGE project not checked. |
| ViGEText_17to23 (uitnlp; Vietnamese general-education MCQ, symbol bind | https://huggingface.co/datasets/uitnlp/ViGEText_17to23 | open; Evaluation set; 35 train rows are few-shot exemplars. |
| SEA-NLI (aisingapore; culturally grounded NLI incl. Vietnamese) | https://huggingface.co/datasets/aisingapore/SEA-NLI | HF-gated (manual); gated: card README returns HTTP 401 without login; I did not log in. Access condition below is from the public HF API metadata. Test only anyway. Other aisingapore sets (NLU-Sentiment-Analysis, Safety |

### Không tải được / không tồn tại

| Bộ | URL | Ghi chú |
|---|---|---|
| ISE-DSC01 (UIT Data Science Challenge 2023 fact-checking) | https://codalab.lisn.upsaclay.fr/competitions/ (UIT DSC 2023; exact page not located) | unavailable; Only competition-platform distribution (CodaLab) and participant GitHub repos (e.g. DanQuang/Verdict-Classification-for-Fact-Checking-at-DSC-2023, code only) were found; no official public release or  |
| UIT-ViCoV19QA (COVID-19 community QA) | https://huggingface.co/datasets/SEACrowd/uit_vicov19qa | unavailable; Loader only; no download location found for the data itself. |
| Native Vietnamese yes/no (boolean) QA set - GAP | n/a | unavailable; No native, open Vietnamese BoolQ-style dataset was found (HF language:vi filters for QA/NLI/fact-checking/multiple-choice, keyword searches 'boolq', 'yes_no', 'vietnamese boolean', and two web searche |

### Bị loại (nguồn bench_v1 hoặc dẫn xuất, hoặc điều khoản cấm)

| Bộ | URL | Ghi chú |
|---|---|---|
| ViFactCheck (copies: tranthaihoa/vifactcheck, neyugntuan/vifactcheck) | https://huggingface.co/datasets/tranthaihoa/vifactcheck | open; HARD EXCLUDE: ViFactCheck is on the hard-exclude list. Second copy neyugntuan/vifactcheck sha c1d0fbf006ab4b1d13ceb2f430b2957365877bf4. Nothing was downloaded or sampled. Used as the reference for 'ne |
| ViWikiFC (copies: High-Will/ViWikiFC, NghiemAbe/ViWikiFC) | https://huggingface.co/datasets/High-Will/ViWikiFC | open; HARD EXCLUDE: ViWikiFC is on the hard-exclude list. Nothing downloaded or sampled. Reference for 'Wikipedia-derived' overlap risk. |
| ViHSD (uitnlp/vihsd, mirror ura-hcmut/UIT-ViHSD) | https://huggingface.co/datasets/uitnlp/vihsd | HF-gated; HARD EXCLUDE: ViHSD is on the hard-exclude list. Nothing downloaded. Reference for 'social-media-comment' overlap risk. |
| ViHOS (phusroyal/ViHOS, hate and offensive spans) | https://huggingface.co/datasets/phusroyal/ViHOS | open; HARD EXCLUDE: DERIVATIVE of ViHSD. ACL paper (https://aclanthology.org/2023.eacl-main.47.pdf, Sec. 3.1): 'ViHOS consists of 11,056 comments derived from the ViHSD dataset (Luu et al., 2021).' Not samp |
| VOZ-HSD (tarudesu/VOZ-HSD, 10.7M AI-labelled forum comments) | https://huggingface.co/datasets/tarudesu/VOZ-HSD | open; NOT on the hard-exclude list; excluded because the card forbids using the AI labels for fine-tuning. The training data of the labelling classifier (ViSoBERT-HSD) was not checked and may involve ViHSD. |
| MASSIVE (qanastek/MASSIVE; also AmazonScience/massive vi-VN as bundled | https://huggingface.co/datasets/qanastek/MASSIVE | open; HARD EXCLUDE: MASSIVE is on the hard-exclude list. IMPORTANT: LakoreAI/PhoLaya-base data/train.jsonl contains 22,069 MASSIVE-derived train questions (12.3% of its 179,231 train rows, per manifest.json |
| Belebele (facebook/belebele, vie_Latn) | https://huggingface.co/datasets/facebook/belebele | open; HARD EXCLUDE: Belebele. Seen only in the HF language:vi listing; not examined. |
| SIB-200 (Davlan/sib200, vie_Latn) | https://huggingface.co/datasets/Davlan/sib200 | open; HARD EXCLUDE: SIB-200. Not examined. |
| XCOPA (cambridgeltl/xcopa, vi) | https://huggingface.co/datasets/cambridgeltl/xcopa | open; HARD EXCLUDE: XCOPA/COPA family. Not examined. |
| FLORES-200, SLURP, IntentGrasp, COPA / Balanced-COPA | n/a | n/a; HARD EXCLUDE: Hard-excluded by instruction. None of these surfaced as separate Vietnamese-relevant repos in my searches (SLURP is English-only; FLORES/COPA/IntentGrasp not searched for beyond the list |
| ShynBui/Vietnamese_Reading_Comprehension_Dataset | https://huggingface.co/datasets/ShynBui/Vietnamese_Reading_Comprehension_Dataset | open; NOT on the hard-exclude list; excluded because it is machine-translated SQuAD (not native), word-segmented, with an MIT tag over SQuAD (CC BY-SA 4.0) content. Not sampled. |
| huyleit/vietnamese-social-emotion | https://huggingface.co/datasets/huyleit/vietnamese-social-emotion | open; NOT on the hard-exclude list; excluded: derivative of UIT-VSMEC (agreement/licence-dependent) mixed with machine-translated GoEmotions. Not sampled. |
| FinnPham52/vietnamese-legal-claim-verification | https://huggingface.co/datasets/FinnPham52/vietnamese-legal-claim-verification | open; NOT on the hard-exclude list; excluded: English claims, no card, no licence, small. Not sampled. |

### Đã dùng

| Bộ | URL | Ghi chú |
|---|---|---|
| UIT-ViNLI (as used via LakoreAI/PhoLaya-base; source presencesw/vinli_ | https://huggingface.co/datasets/LakoreAI/PhoLaya-base | open; Not surveyed deeply (per instruction). |
| UIT-VSFC (as used via LakoreAI/PhoLaya-base; source uitnlp/vietnamese_ | https://huggingface.co/datasets/LakoreAI/PhoLaya-base | open; Not surveyed deeply (per instruction). |
| UIT-ViSFD (as used via LakoreAI/PhoLaya-base; source visolex/ViSFD) | https://huggingface.co/datasets/LakoreAI/PhoLaya-base | open; Not surveyed deeply (per instruction). |
| LakoreAI/PhoLaya-base (the collection already used) - contents of sour | https://huggingface.co/datasets/LakoreAI/PhoLaya-base | open; sources/ directory holds: massive, uts, vianli, victsd, viglue, vinli, visfd, vsfc, vsmec (each dev/test/train.jsonl). The current card (v1.1) says only four sources are in the mixture - ViSFD 85,646, |

## 3. Giấy phép / điều kiện truy cập — chép nguyên văn từ nguồn

- **VSMRC/mrc (Vietnamese Text Segmentation and Multiple-Choice Reading Comprehension; MC part** — nguồn: https://huggingface.co/datasets/VSMRC/mrc/raw/main/README.md
  > HF card YAML `license: apache-2.0`; the card body contains no other text (front matter only). A derivative card (p-storm/VSMRC-mrc-ABCD) says 'Dataset gốc dùng license **Apache-2.0** — sử dụng tự do, kể cả mục đích thương mại, chỉ cần giữ thông báo bản quyền/attribution.' but that is a third party statement.
- **EXAMS (Vietnamese subset) - mhardalov/exams, config crosslingual_with_para_vi** — nguồn: https://huggingface.co/datasets/mhardalov/exams/raw/main/README.md
  > YAML `license: cc-by-sa-4.0`; body: 'The dataset, which contains paragraphs from Wikipedia, is licensed under CC-BY-SA 4.0. The code in this repository is licensed according the [LICENSE file](https://raw.githubusercontent.com/mhardalov/exams-qa/main/LICENSE).'
- **ViCS-MCQ (Vietnamese Computer Science MCQ)** — nguồn: https://huggingface.co/datasets/p-storm/ViCS-MCQ/raw/main/README.md
  > YAML `license: cc-by-4.0`; body: 'Bộ dữ liệu được phân phối dưới giấy phép **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.'
- **HienNGuit/Domain-ViWikiQA (Vietnamese WikiQA, domain-labelled)** — nguồn: https://huggingface.co/datasets/HienNGuit/Domain-ViWikiQA/raw/main/README.md
  > YAML `license: other`; the card body contains no licence sentence (no terms stated).
- **ViBidLQA_v1 (Vietnamese bidding-law QA)** — nguồn: https://huggingface.co/datasets/ntphuc149/ViBidLQA_v1/raw/main/README.md
  > YAML `license: apache-2.0`; body: 'This dataset is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.'
- **ura-hcmut/zalo_e2eqa (Zalo AI Challenge E2E QA, open-domain short answers)** — nguồn: https://huggingface.co/datasets/ura-hcmut/zalo_e2eqa/raw/main/README.md
  > YAML `license: mit`; no body text.
- **GreenNode/zalo-ai-legal-text-retrieval-vn (Zalo AI Challenge 2021 legal text retrieval)** — nguồn: https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn/raw/main/README.md
  > YAML `license: mit`; no licence sentence in body.
- **aiMy144 Vietnamese fact-checking verifier data v3.1 (claims + evidence, 3-way)** — nguồn: https://huggingface.co/datasets/aiMy144/vietnamese-fact-checking-verifier-data-v3-1/raw/main/README.md ; https://huggingface.co/datasets/aiMy144/vietnamese-fact-checking-claims-v3-1/raw/main/README.md ; https://huggingface.co/datasets/aiMy144/viet-fact-checking/raw/main/README.md
  > Verifier repo: no `license:` tag and no licence sentence in the card. Claims repo (rev e20c1edd...): YAML `license: other` with no terms text. Evidence corpus aiMy144/viet-fact-checking (rev 61ec51be...): YAML `license: cc-by-4.0`.
- **k22-hcmus/fact-checking-vietnamese-news** — nguồn: https://huggingface.co/api/datasets/k22-hcmus/fact-checking-vietnamese-news
  > None. No licence tag, no README, no LICENSE file in the repo listing.
- **VNTC 10-topics news classification (HF copy kornwtp/VNTC-10Topics; original GitHub duyvule** — nguồn: https://huggingface.co/datasets/kornwtp/VNTC-10Topics/raw/main/README.md ; https://raw.githubusercontent.com/duyvuleo/VNTC/master/LICENSE
  > HF YAML `license: mit`; HF body: 'ref: https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics/Ver1.1'. Original GitHub LICENSE (commit 533a3d6e1a78d73cde5dcfaf867cf8fe62c1fca8): 'MIT License\n\nCopyright (c) 2019 Vu Cong Duy Hoang\n\nPermission is hereby granted, free of charge, to any person obtaining a copy ...'
- **NamSyntax/vietnamese-news-classification (1.3M news, 11 topics)** — nguồn: https://huggingface.co/datasets/NamSyntax/vietnamese-news-classification/raw/main/README.md
  > YAML `license: apache-2.0`; card body contains no licence sentence.
- **ngocleltt/vi-news-4topics-classification** — nguồn: https://huggingface.co/datasets/ngocleltt/vi-news-4topics-classification/raw/main/README.md
  > YAML `license: cc-by-4.0`; body: 'The data was collected from Vietnamese news websites (e.g., VnExpress). Only headlines were used to ensure lightweight data and avoid copyright issues.' and 'This dataset was manually curated and cleaned for research purposes.'
- **vanhai123/vietnamese-news-dataset (tiny topic set)** — nguồn: https://huggingface.co/datasets/vanhai123/vietnamese-news-dataset/raw/main/README.md
  > YAML `license: other`; card body has no licence sentence.
- **phamluan/vi-tourism-admin-intent (Vietnamese tourism & administrative-merger intent)** — nguồn: https://huggingface.co/datasets/phamluan/vi-tourism-admin-intent/raw/main/README.md
  > YAML `license: cc-by-4.0`; body has no separate licence sentence.
- **PhoATIS (VinAI, Vietnamese ATIS) - HF copy ura-hcmut/PhoATIS; original github.com/VinAIRes** — nguồn: https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/README.md ; https://huggingface.co/datasets/ura-hcmut/PhoATIS/raw/main/README.md
  > GitHub README (JointIDSF): 'By downloading [our dataset](https://github.com/VinAIResearch/JointIDSF/tree/main/PhoATIS), USER agrees: * to use the dataset for research or educational purposes only. * to **not** distribute the dataset or part of the dataset in any original or modified form. * and to cite our paper above whenever the dataset is employed to help produce published results or incorporated into other software.' GitHub repo licence (code): AGPL-3.0. The ura-hcmut HF mirror declares YAML `license: cc-by-nc-sa-4.0` (conflicts with the no-redistribution clause; not from the authors).
- **ViANLI (Vietnamese Adversarial NLI) - uitnlp/ViANLI** — nguồn: https://huggingface.co/datasets/uitnlp/ViANLI/raw/main/README.md
  > YAML `license: mit`; body '### Dataset License - **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) *(Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License)*' (TAG AND BODY DISAGREE).
- **XNLI Vietnamese (facebook/xnli, config vi)** — nguồn: https://huggingface.co/datasets/facebook/xnli/raw/main/README.md ; https://raw.githubusercontent.com/facebookresearch/XNLI/main/LICENSE
  > HF card: no `license:` tag; body 'Licensing Information: [More Information Needed]'. GitHub (facebookresearch/XNLI, commit 9624695001060244cf8d5d893b5236093a21a42c): README 'XNLI is licensed under the license found in the LICENSE file.' and LICENSE file titled 'Attribution-NonCommercial 4.0 International'.
- **UIT-ViCTSD (Constructive and Toxic Speech Detection) - tarudesu/ViCTSD** — nguồn: https://huggingface.co/datasets/tarudesu/ViCTSD/raw/main/README.md
  > No licence tag. Card body: 'The provided dataset is only used for research purposes!'
- **UIT-ViOCD (Complaint detection) - tarudesu/ViOCD** — nguồn: https://huggingface.co/datasets/tarudesu/ViOCD/raw/main/README.md
  > No licence tag. Card body: 'The provided dataset is only used for research purposes!'
- **UIT-VSMEC (emotion) - HF copy ura-hcmut/UIT-VSMEC** — nguồn: https://huggingface.co/datasets/ura-hcmut/UIT-VSMEC/raw/main/README.md ; https://raw.githubusercontent.com/kietnv/VietnameseDatasets/main/README.md ; https://nlp.uit.edu.vn/datasets
  > ura-hcmut HF mirror: YAML `license: cc-by-nc-sa-4.0` (declared by the mirror owner, not the authors), body only '# UIT-VSMEC Dataset'. Authors: github.com/kietnv/VietnameseDatasets README (commit 82c1b664a4f555d357757326cf0f7b755f57c2c5): 'Please contact us via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to sign the corpus user agreement and then receive the corpus.' BUT nlp.uit.edu.vn/datasets lists UIT-VSMEC with a Google Drive download link ('Please download this dataset/corpus') and no agreement sentence.
- **ViHealthQA (Vietnamese healthcare QA) - tarudesu/ViHealthQA** — nguồn: https://huggingface.co/datasets/tarudesu/ViHealthQA/raw/main/README.md
  > No licence tag. Card body: 'The provided dataset is only used for research purposes!' and 'The dataset may contain personal information crawled along with the contents of various sources. Please make a filter in pre-processing data before starting your research training.'
- **UIT-ViON (online news titles, 13 topics) - GitHub kh4nh12/UIT-ViON-Dataset** — nguồn: https://raw.githubusercontent.com/kh4nh12/UIT-ViON-Dataset/main/README.md
  > None. GitHub repo has no LICENSE file (GitHub API license: null) and the README (861 bytes) has no terms sentence.
- **UIT-ViQuAD 1.0 / 2.0 (Vietnamese MRC; 2.0 adds ~12K unanswerable questions)** — nguồn: https://raw.githubusercontent.com/kietnv/VietnameseDatasets/main/README.md ; https://nlp.uit.edu.vn/datasets ; https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0/raw/main/README.md
  > Authors' GitHub README (commit 82c1b664a4f555d357757326cf0f7b755f57c2c5), UIT-ViQuAD section: 'Please contact us via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to sign the corpus user agreement and then receive the corpus.' nlp.uit.edu.vn/datasets (WebFetch summary, not raw HTML): 'Request by email ... to sign the corpus user agreement and then receive the corpus.' and also 'Our dataset is freely available on our website.' The third-party mirror taidng/UIT-ViQuAD2.0 (sha 406f09a45cc106a8f7b7fd0c25078883fe58cb1f) has NO licence tag and no licence sentence in its card ('Specific questions about the test set or the dataset should be directed to the authors').
- **UIT-ViNewsQA (health-news MRC)** — nguồn: https://raw.githubusercontent.com/kietnv/VietnameseDatasets/main/README.md ; https://nlp.uit.edu.vn/datasets
  > Authors' GitHub README (commit 82c1b664a4f555d357757326cf0f7b755f57c2c5): 'Please contact us via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to sign the corpus user agreement and then receive the corpus.' and abstract 'Our corpus is freely available on our website in order to encourage the research community'. nlp.uit.edu.vn/datasets (WebFetch summary): 'Our corpus is freely available on our website' with no agreement step stated (conflicting).
- **ViMMRC 1.0 (Vietnamese multiple-choice MRC, grades 1-5 textbook passages)** — nguồn: https://raw.githubusercontent.com/kietnv/VietnameseDatasets/main/README.md ; https://huggingface.co/datasets/ura-hcmut/ViMMRC/raw/main/README.md ; https://huggingface.co/datasets/SEACrowd/vimmrc/raw/main/README.md
  > Authors' GitHub README (commit 82c1b664a4f555d357757326cf0f7b755f57c2c5): 'Please contact us via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to sign the corpus user agreement and then receive the corpus.' plus abstract: 'The corpus is freely available at our website for research purposes.' SEACrowd loader card: 'Unknown (unknown) / The corpus is freely available at our website for research purposes.' Third-party mirror ura-hcmut/ViMMRC (sha fe68800e37aaa84d80b1d93466b36c3fa60d8bcb): YAML `license: cc-by-nc-nd-4.0` only (not from the authors).
- **ViMMRC 2.0 - uitnlp/vimmrc2.0** — nguồn: https://huggingface.co/api/datasets/uitnlp/vimmrc2.0 ; https://raw.githubusercontent.com/sonlam1102/vimmrc2/main/README.md
  > HF YAML `license: cc-by-sa-4.0`. API description: 'The dataset is freely available for research purposes only. Users need to sign the data agreement before receiving the dataset.' GitHub sonlam1102/vimmrc2 README (commit 9f662eedf1288dc5559c39617f6017e254e58543): 'Please contact Mr Son Luu via this email: sonlt@uit.edu.vn for the dataset ... Users need to sign the data agreement before receiving the dataset.' and 'Note that the dataset is gated, and users need to sign the data agreement form and send it to Mr Son Luu (as mentioned in Data inquiry) to access it.' (tag cc-by-sa-4.0 conflicts with 'research purposes only').
- **ViSpamReviews (spam review detection) - sonlam1102/vispamdetection (v1) and vispamdetectio** — nguồn: https://raw.githubusercontent.com/sonlam1102/vispamdetection/main/README.md ; https://huggingface.co/datasets/visolex/ViSpamReviews/raw/main/README.md
  > HF YAML `license: mit` (gated, auto). Authors' GitHub README (sonlam1102/vispamdetection, commit 01a965c7dc67a1059153d2cfc26b681fc8b9cfb4): 'The dataset is used for research purposes. ACCESS the dataset on HuggingFace here (gated)' and 'Please contact Mr. Son Luu (Son T. Luu) if you want to use or discuss this dataset (both version 1 and version 2).' GitHub licence: none. The mirror visolex/ViSpamReviews (sha 3527ed2c779bca5162145bd03575647c5993d937) says 'Refer to the GitHub repo’s LICENSE. If unspecified, assume **CC BY 4.0**.' - a guess by the mirror owner.
- **ViLegalTF (legal true/false QA)** — nguồn: https://huggingface.co/api/datasets/ntphuc149/ViLegalTF
  > HF YAML `license: apache-2.0`. Access condition: HF gated, mode 'auto' (click-through); the form text is not exposed by the public API and the card README is not readable without login (HTTP 401) - not quoted.
- **ViLegalMCQ (legal multiple-choice QA)** — nguồn: https://huggingface.co/api/datasets/ntphuc149/ViLegalMCQ
  > HF YAML `license: apache-2.0`. Access condition: HF gated, mode 'auto' (click-through); the form text is not exposed by the public API and the card README is not readable without login (HTTP 401) - not quoted.
- **ViLegalNLI (legal citation sufficiency NLI)** — nguồn: https://huggingface.co/api/datasets/ntphuc149/ViLegalNLI
  > HF YAML `license: apache-2.0`. Access condition: HF gated, mode 'auto' (click-through); the form text is not exposed by the public API and the card README is not readable without login (HTTP 401) - not quoted.
- **VLSP 2025 legal MCQ (ura-hcmut/vlsp2025-mcq-non-reasoning and -reasoning)** — nguồn: https://huggingface.co/api/datasets/ura-hcmut/vlsp2025-mcq-non-reasoning
  > YAML `license: cc-by-nc-sa-4.0`. extra_gated_description: 'Part of the VLSP 2025: Vietnamese Legal Reasoning & QA Datasets collection (9 datasets). Released for non-commercial research under CC BY-NC-SA 4.0. Access is granted automatically once you submit this form.' extra_gated_prompt: "By requesting access, you agree to use this dataset for non-commercial research only and to cite the paper 'Can Small Language Models Handle Vietnamese Legal Reasoning? Insights from Multi-Task Evaluation' (VLSP 2025) in any work that uses it. Citing the paper is a condition of access." Form fields: full name, email, affiliation, country, position, intended use, two checkboxes.
- **VLSP 2025 legal citation-usefulness NLI (ura-hcmut/vlsp2025-nli-non-reasoning and -reasoni** — nguồn: https://huggingface.co/api/datasets/ura-hcmut/vlsp2025-nli-non-reasoning
  > Same gate as the MCQ sets: YAML `license: cc-by-nc-sa-4.0`; extra_gated_description 'Released for non-commercial research under CC BY-NC-SA 4.0. Access is granted automatically once you submit this form.'; extra_gated_prompt 'By requesting access, you agree to use this dataset for non-commercial research only and to cite the paper ... Citing the paper is a condition of access.'
- **UIT-VlogQA (QA over Vietnamese vlog transcripts) - uitnlp/vlogqa** — nguồn: https://huggingface.co/api/datasets/uitnlp/vlogqa
  > HF YAML `license: cc-by-sa-4.0`. API description: 'You need to sign the agreement and send to the author via the email in "Dataset Contact" to access to the dataset.'
- **UIT-ViCoQA (conversational QA on health news) - uitnlp/vicoqa** — nguồn: https://huggingface.co/api/datasets/uitnlp/vicoqa ; https://raw.githubusercontent.com/kietnv/VietnameseDatasets/main/README.md
  > HF YAML `license: mit`; gated 'auto'. Authors' GitHub list (kietnv README): 'To access this dataset, please complete and sign the dataset user agreement and then send it via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to receive the dataset.' (UIT-ViCoQA section).
- **UIT-ViGoEmotions (27-way emotion) - uitnlp/vigoemotions** — nguồn: https://huggingface.co/api/datasets/uitnlp/vigoemotions
  > HF YAML `license: mit`; gated 'auto'; form text not exposed.
- **ViTHSD (targeted hate speech) - sonlam1102/vithsd** — nguồn: https://huggingface.co/api/datasets/sonlam1102/vithsd
  > HF YAML `license: mit`; gated 'auto'.
- **UIT-ViWikiQA (sentence-extraction MRC) - SEACrowd/uit_viwikiqa loader** — nguồn: https://huggingface.co/datasets/SEACrowd/uit_viwikiqa/raw/main/README.md
  > YAML `license: other`; body: 'The user of UIT-ViWikiQA developed by the NLP@UIT research group must respect the following terms and conditions: 1. The dataset is only used for non-profit research for image captioning. 2. The dataset is not allowed to be used in commercial systems. 3. Do not redistribute the dataset. This dataset may be modified or improved to serve a research purpose better, but the edited dataset may not be distributed. 4. Summaries, analyses, and interpretations of the properties of the dataset may be derived and published, provided it is not possible to reconstruct the information from these summaries. 5. Published research works that use the dataset must cite the following paper: ...' (sic, 'image captioning' is in the card).
- **UIT-ViNLI (as used via LakoreAI/PhoLaya-base; source presencesw/vinli_4_label)** — nguồn: https://huggingface.co/datasets/LakoreAI/PhoLaya-base/raw/main/README.md
  > YAML `license: other`, `license_name: research-use-only`. Body: '`license: other` (research-use-only): the mixture contains derivatives of the UIT corpora (ViSFD, VSFC, ViNLI), which are research-licensed. Do not use commercially without contacting the dataset authors. MASSIVE vi-VN is CC BY 4.0. Attribute each source per its own license.'
- **UIT-VSFC (as used via LakoreAI/PhoLaya-base; source uitnlp/vietnamese_students_feedback)** — nguồn: https://huggingface.co/datasets/LakoreAI/PhoLaya-base/raw/main/README.md
  > YAML `license: other`, `license_name: research-use-only`. Body: '`license: other` (research-use-only): the mixture contains derivatives of the UIT corpora (ViSFD, VSFC, ViNLI), which are research-licensed. Do not use commercially without contacting the dataset authors. MASSIVE vi-VN is CC BY 4.0. Attribute each source per its own license.' (uitnlp/vietnamese_students_feedback HF tag: `license: unknown`)
- **UIT-ViSFD (as used via LakoreAI/PhoLaya-base; source visolex/ViSFD)** — nguồn: https://huggingface.co/datasets/LakoreAI/PhoLaya-base/raw/main/README.md
  > YAML `license: other`, `license_name: research-use-only`. Body: '`license: other` (research-use-only): the mixture contains derivatives of the UIT corpora (ViSFD, VSFC, ViNLI), which are research-licensed. Do not use commercially without contacting the dataset authors. MASSIVE vi-VN is CC BY 4.0. Attribute each source per its own license.'
- **LakoreAI/PhoLaya-base (the collection already used) - contents of sources/** — nguồn: https://huggingface.co/datasets/LakoreAI/PhoLaya-base/raw/main/README.md
  > YAML `license: other`, `license_name: research-use-only`. Body: '`license: other` (research-use-only): the mixture contains derivatives of the UIT corpora (ViSFD, VSFC, ViNLI), which are research-licensed. Do not use commercially without contacting the dataset authors. MASSIVE vi-VN is CC BY 4.0. Attribute each source per its own license.'

## 4. Nhóm 3 — model dịch Anh → Việt chạy được trên 16 GB (chưa dịch gì; chỉ đọc card)

Revision, thẻ giấy phép và cờ gated đã đối chiếu lại bằng API HF: khớp 44/44.

| Repo @ revision | Tham số | Giấy phép (thẻ) | Hạn chế | Vừa 16 GB | Chất lượng en→vi đã công bố |
|---|---|---|---|---|---|
| vinai/vinai-translate-en2vi @ `13af6b73` | 448M (stated | agpl-3.0 | AGPL-3.0 strong/network copyleft on the weights/code if modified or served; commercial use allowed by AGPL but copyleft obligations apply. Neither card nor READ | yes | PhoMT test set, English-to-Vietnamese SacreBLEU 44.2 |
| vinai/vinai-translate-en2vi-v2 @ `82f8c91b` | ~0.42B (DERI | agpl-3.0 | AGPL-3.0 copyleft on weights/code if modified or served; treatment of outputs not stated. Only pytorch_model.bin (pickle) shipped. | yes | PhoMT test set, English-to-Vietnamese SacreBLEU 44.3 |
| VietAI/envit5-translation @ `840bc881` | ~275M (EnViT | openrail | OpenRAIL-family licence = responsible-AI use-based restrictions; the card does not reproduce the terms nor say which OpenRAIL variant. Check the exact text befo | yes | PhoMT test set En->Vi, EnViT5-base finetuned on MTet+PhoMT (6.2M pairs), MTet paper Table 1 SacreBLEU (case-sensitive) 45.47; PhoMT test set En->Vi, EnViT5-base |
| Helsinki-NLP/opus-mt-en-vi @ `989c9fb9` | ~72M (DERIVE | apache-2.0 | HF tag says apache-2.0, but upstream Tatoeba-Challenge README states: 'Training data is released under the CC-BY-NC-SA 4.0 license' and test/dev data CC-BY 2.0  | yes | Tatoeba-test.eng.vie (short, simple sentences) BLEU 37.2; Tatoeba-test.eng.vie chrF2 0.542 |
| facebook/nllb-200-distilled-600M @ `f8d333a0` | 600M nominal | cc-by-nc-4.0 | Non-commercial (CC-BY-NC-4.0). Card declares it a research model not released for production deployment; 512-token training length; not intended for domain-spec | yes | FLORES-200, direction eng_Latn-vie_Latn (split not stated in the CSV; card says Flores-200) chrF++ 56.4 |
| facebook/nllb-200-distilled-1.3B @ `7be3e246` | 1.3B nominal | cc-by-nc-4.0 | Non-commercial (CC-BY-NC-4.0). Card declares it a research model not released for production deployment; 512-token training length; not intended for domain-spec | yes | FLORES-200, direction eng_Latn-vie_Latn (split not stated in the CSV; card says Flores-200) chrF++ 58.4 |
| facebook/nllb-200-3.3B @ `1a07f7d1` | 3.3B nominal | cc-by-nc-4.0 | Non-commercial (CC-BY-NC-4.0). Card declares it a research model not released for production deployment; 512-token training length; not intended for domain-spec | yes | FLORES-200, direction eng_Latn-vie_Latn (split not stated in the CSV; card says Flores-200) chrF++ 59.3 |
| google/madlad400-3b-mt @ `fa184c67` | 2.94B (F32 s | apache-2.0 | None beyond Apache-2.0. Card disclaimer: card text was written by Juarez Bochi (not an author) based on the paper; model is a research release with unassessed d | yes | — |
| google/madlad400-7b-mt @ `1ff63ce7` | 8.30B (F32 s | apache-2.0 | None beyond Apache-2.0. Card disclaimer: card text was written by Juarez Bochi (not an author) based on the paper; model is a research release with unassessed d | quantized only | — |
| google/madlad400-10b-mt @ `9f279762` | 10.71B (F32  | apache-2.0 | None beyond Apache-2.0. Card disclaimer: card text was written by Juarez Bochi (not an author) based on the paper; model is a research release with unassessed d | quantized only | — |
| Qwen/Qwen3-4B @ `1cfa9a72` | 4.02B (BF16  | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen3-4B-FP8 @ `96b30dc1` | 4.02B nomina | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen3-4B-AWQ @ `74d4bd2b` | 4.02B nomina | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f` | 4.02B (BF16  | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen3-8B @ `b968826d` | 8.19B (BF16  | apache-2.0 | None beyond Apache-2.0. | quantized only | — |
| Qwen/Qwen3-8B-FP8 @ `220b46e3` | 8.19B (7.0B  | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen3-8B-AWQ @ `4da05a8e` | 8.19B (AWQ I | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen2.5-7B-Instruct @ `a09a3545` | 7.62B (BF16  | apache-2.0 | None beyond Apache-2.0. | quantized only | — |
| Qwen/Qwen3.5-4B @ `851bf6e8` | 4.66B (BF16  | apache-2.0 | None beyond Apache-2.0. | yes | — |
| Qwen/Qwen3.5-9B @ `c2022362` | 9.65B (BF16  | apache-2.0 | None beyond Apache-2.0. | quantized only | — |
| tencent/Hunyuan-MT-7B @ `9305c783` | 8.03B (BF16  | — | BLOCKER for creating training data: licence 5(b) forbids using the Works 'or any Output or results' to improve any other AI model (translated data used to fine- | quantized only | — |
| tencent/HY-MT1.5-7B @ `397085d4` | 8.03B (BF16  | — | BLOCKER for creating training data: licence 5(b) forbids using the Works 'or any Output or results' to improve any other AI model (translated data used to fine- | quantized only | — |
| tencent/HY-MT1.5-1.8B @ `dbad0378` | 2.04B (BF16  | — | BLOCKER for creating training data: licence 5(b) forbids using the Works 'or any Output or results' to improve any other AI model (translated data used to fine- | yes | — |
| tencent/Hy-MT2-7B @ `9b0eb4e8` | 8.03B (BF16  | apache-2.0 | None beyond Apache-2.0 (no output-use or territory restriction, unlike Hunyuan-MT-7B and HY-MT1.5). Newest (2026-05) dedicated MT family. | quantized only | — |
| tencent/Hy-MT2-7B-FP8 @ `883d09eb` | 8.03B nomina | apache-2.0 | None beyond Apache-2.0. | yes | — |
| tencent/Hy-MT2-1.8B @ `9a341cd1` | 2.04B (BF16  | apache-2.0 | None beyond Apache-2.0. | yes | — |
| tencent/Hy-MT2-30B-A3B @ `d3ead4db` | 30.06B total | apache-2.0 | None beyond Apache-2.0. | no | — |
| ByteDance-Seed/Seed-X-PPO-7B @ `6ef78fc0` | 7.51B (BF16  | other | OpenMDW-1.0 is permissive: no restriction on use or on outputs; only retain licence copy/notices on redistribution; patent-suit termination clause. Verdict: usa | quantized only | — |
| ByteDance-Seed/Seed-X-Instruct-7B @ `a41af5ab` | 7.51B (BF16  | other | OpenMDW-1.0 is permissive: no restriction on use or on outputs; only retain licence copy/notices on redistribution; patent-suit termination clause. Verdict: usa | quantized only | — |
| ByteDance-Seed/Seed-X-PPO-7B-AWQ-Int4 @ `3c253e97` | 7.5B nominal | other | OpenMDW-1.0 (permissive; no output restriction). | yes | — |
| ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8 @ `d4f23d35` | 7.5B nominal | other | OpenMDW-1.0 (permissive; no output restriction). | yes | — |
| google/translategemma-4b-it @ `10042cb0` | ~4.3B text p | gemma | GATED (HF gated='manual'). Logged-out model page shows: 'This repository is publicly accessible, but you have to accept the conditions to access its files and c | yes | WMT24++ en->vi_VN, TranslateGemma 4B (Appendix A Table 4) MetricX-24 (lower is better) 2.87; WMT24++ en->vi_VN, baseline Gemma 3 same size (Appendix A Table 4)  |
| google/translategemma-12b-it @ `d1b225e1` | ~12.2B text  | gemma | GATED (HF gated='manual'). Logged-out model page shows: 'This repository is publicly accessible, but you have to accept the conditions to access its files and c | no | WMT24++ en->vi_VN, TranslateGemma 12B (Appendix A Table 4) MetricX-24 (lower is better) 2.2; WMT24++ en->vi_VN, baseline Gemma 3 same size (Appendix A Table 4)  |
| google/translategemma-27b-it @ `7d10f0b7` | ~27.4B text  | gemma | GATED (HF gated='manual'). Logged-out model page shows: 'This repository is publicly accessible, but you have to accept the conditions to access its files and c | no | WMT24++ en->vi_VN, TranslateGemma 27B (Appendix A Table 4) MetricX-24 (lower is better) 1.97; WMT24++ en->vi_VN, baseline Gemma 3 same size (Appendix A Table 4) |
| google/gemma-4-E4B-it @ `ee0ef602` | 7.99B stored | apache-2.0 | Apache-2.0 per card (verify the linked gemma_4_license page before relying on it). | quantized only | — |
| google/gemma-4-12B-it @ `707f0a3b` | 11.96B (BF16 | apache-2.0 | Apache-2.0 per card (verify the linked gemma_4_license page before relying on it). | quantized only | — |
| SeaLLMs/SeaLLMs-v3-7B-Chat @ `a9900348` | 7.62B (BF16  | other | BLOCKER for creating training data: licence 5(b) forbids using the Materials or any output therefrom to improve any other LLM (excluding SeaLLMs derivatives). A | quantized only | FLORES-200 zero-shot, average over all source languages translating INTO vi (NOT en->vi specific) chrF 43.78 |
| sail/Sailor2-8B-Chat @ `577d5b27` | 8.55B (BF16  | apache-2.0 | None beyond Apache-2.0. | quantized only | — |
| Tower-Babel/Babel-9B-Chat @ `e30379ef` | 9.01B (BF16  | other | BLOCKER for creating training data: licence 5(b) forbids using the Materials or any output therefrom to improve any other LLM (excluding SeaLLMs derivatives). A | quantized only | — |
| CohereLabs/aya-expanse-8b @ `5062468b` | 8.03B (F16 s | cc-by-nc-4.0 | GATED (HF gated='auto'): logged-out page shows 'This repository is publicly accessible, but you have to accept the conditions to access its files and content.'; | quantized only | — |
| Viet-Mistral/Vistral-7B-Chat @ `d331b64e` | 7.29B (BF16  | afl-3.0 | GATED (HF gated='manual'): logged-out page shows 'This repository is publicly accessible, but you have to accept the conditions to access its files and content. | quantized only | — |
| Unbabel/Tower-Plus-9B @ `c59d17c5` | 9.24B (BF16  | cc-by-nc-sa-4.0 | Non-commercial (CC-BY-NC-*); tag/body disagree on ShareAlike. | quantized only | — |
| Unbabel/Tower-Plus-2B @ `4d779ca9` | 2.61B (BF16  | cc-by-nc-sa-4.0 | Non-commercial (CC-BY-NC-*); tag/body disagree on ShareAlike. | yes | — |
| Unbabel/TowerInstruct-7B-v0.2 @ `a240dfd3` | 6.74B (F32 s | cc-by-nc-4.0 | Non-commercial plus Llama 2 Community License. | quantized only | — |

### Tác vụ tasksource (tiếng Anh) thuộc dạng bài còn thiếu — ứng viên để dịch

| Dạng bài | Nguồn | Dòng | Đã dùng v0 | Bị loại | Kiểu | Trung vị / p90 token cả dòng |
|---|---|---|---|---|---|---|
| đọc hiểu | cloth | 2,767 | 54 | không | choice, noul | 42 / 71 |
| đọc hiểu | cosmos_qa | 4,649 | 89 | không | choice, noul | 125 / 169 |
| đọc hiểu | dream | 2,533 | 50 | không | choice, noul | 155 / 389 |
| đọc hiểu | logiqa | 5,635 | 97 | không | choice, noul | 164 / 237 |
| đọc hiểu | openbookqa | 2,504 | 48 | không | choice, noul | 35 / 50 |
| đọc hiểu | qasc | 3,114 | 64 | không | choice, noul | 43 / 54 |
| đọc hiểu | quail | 4,739 | 81 | không | choice, noul | 444 / 491 |
| đọc hiểu | quartz | 1,406 | 23 | không | choice, noul | 37 / 53 |
| đọc hiểu | race-c | 4,277 | 72 | không | choice, noul | 529 / 907 |
| đọc hiểu | race/high | 4,813 | 95 | không | choice, noul | 428 / 536 |
| đọc hiểu | race/middle | 3,538 | 63 | không | choice, noul | 292 / 406 |
| đọc hiểu | reclor | 4,451 | 75 | không | choice, noul | 189 / 236 |
| đọc hiểu | sciq | 5,389 | 89 | không | choice, noul | 36 / 45 |
| chủ đề | ag_news | 3,018 | 61 | không | choice, noul, score | 70 / 119 |
| chủ đề | open_question_type | 4,733 | 78 | không | choice, noul, score | 48 / 61 |
| chủ đề | stackoverflow-questions | 3,491 | 77 | không | choice, score, noul | 320 / 951 |
| chủ đề | trec | 3,451 | 7 | không | choice, noul, score | 253 / 323 |
| chủ đề | yahoo_answers_topics | 4,787 | 64 | không | choice, noul, score | 88 / 219 |
| ý định | IntentGrasp/all | 15,978 | 0 | có | choice, noul | 110 / 440 |
| ý định | banking77 | 4,697 | 3 | không | choice, noul | 404 / 417 |
| ý định | clinc_oos/plus | 3,937 | 3 | không | choice, noul | 510 / 515 |
| ý định | dnd_style_intents | 4,609 | 12 | không | choice, noul, score | 64 / 96 |
| ý định | multidogo/airline/intent | 1,000 | 0 | không | choice | 105 / 156 |
| ý định | multidogo/fastfood/intent | 1,000 | 0 | không | choice | 129 / 194 |
| ý định | multidogo/finance/intent | 1,000 | 0 | không | choice | 136 / 194 |
| ý định | multidogo/insurance/intent | 1,000 | 20 | không | choice | 109 / 164 |
| ý định | multidogo/media/intent | 1,000 | 0 | không | choice | 118 / 151 |
| ý định | multidogo/software/intent | 1,000 | 0 | không | choice | 121 / 165 |
| ý định | schema-guided/intent | 1,000 | 0 | không | choice | 348 / 531 |
| ý định | slurp/intent | 1,000 | 0 | có | choice | 252 / 257 |
| ý định | snips_built_in_intents | 396 | 4 | không | choice, noul, score | 54 / 65 |
| ý định | topv2/intent | 1,000 | 0 | không | choice | 227 / 233 |
| kiểm chứng | health_fact | 3,621 | 59 | không | choice, noul, score | 35 / 71 |
| kiểm chứng | liar | 2,996 | 57 | không | choice, noul, score | 51 / 70 |
| kiểm chứng | nli_fever | 4,942 | 91 | không | choice, noul, score | 106 / 242 |
| kiểm chứng | vitaminc | 4,215 | 83 | không | choice, noul, score | 80 / 174 |
| yes/no | babi_nli/yes-no-questions | 328 | 9 | không | choice, noul, score | 74 / 152 |
| yes/no | boolq-natural-perturbations | 1,446 | 22 | không | choice, noul, score | 24 / 29 |
| yes/no | coqa/yesno | 1,000 | 18 | không | choice | 528 / 663 |
| yes/no | hallmarks-of-cancer/yesno | 1,000 | 16 | không | choice | 61 / 89 |
| yes/no | qasper/yesno | 279 | 0 | không | choice | 5103 / 8342 |
| yes/no | sharc | 3,828 | 65 | không | choice, noul, score | 140 / 258 |
| yes/no | spartqa-yn | 4,850 | 88 | không | choice, noul, score | 164 / 292 |
| yes/no | super_glue/boolq | 813 | 17 | không | choice, noul, score | 23 / 54 |
| yes/no | super_glue/boolq_passage | 586 | 13 | không | choice, noul, score | 147 / 299 |
| yes/no | websrc/yesno | 1,000 | 1 | không | choice | 1572 / 4429 |

Bỏ khỏi bảng vì regex tên nguồn bắt nhầm: `citation_intent` (ý định trích dẫn, không phải ý định người dùng); `natural-language-satisfiability` (regex bắt nhầm ('atis' trong 'satisfiability')). `IntentGrasp/all` và `slurp/intent` là nguồn bị loại cứng (MASSIVE/SLURP).
