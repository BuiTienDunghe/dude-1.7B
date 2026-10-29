# Soát 65 chuỗi mẫu tiếng Việt "chưa duyệt" (màn "Mẫu và nhãn")

Ngày 2026-10-14. Soát bằng Opus 5.5, hai lượt:

- quét theo `results/bench_candidates/pilot2/audit/TEMPLATE_REVIEW.md`;
- soi lại độc lập theo `TEMPLATE_VERIFY.md`.

Không đổi trạng thái duyệt nào. Chỉ chuỗi cần sửa được ghi thành đề xuất trong `config/vi_proposals.jsonl` (định dạng đề xuất chuỗi mẫu, khóa trùng khóa của danh mục). Khi duyệt một đề xuất trong công cụ, công cụ áp nó như một lần sửa ở "Mẫu và nhãn".

## Tóm tắt

| Nhóm | Số chuỗi | OK | Đề xuất sửa |
|---|---|---|---|
| Câu hỏi mẫu | 4 | 4 | 0 |
| Tên chủ đề SIB-200 | 7 | 7 | 0 |
| Mô tả ý định MASSIVE | 52 | 50 | 2 |
| Có/Không | 2 | 2 | 0 |

Lượt 1: OK 63, SUA 2. Lượt 2: {'XAC_NHAN': 65}.

## Đề xuất sửa (đã ghi vào `config/vi_proposals.jsonl`, chờ duyệt ở màn "Chờ duyệt")

| id đề xuất | Khóa | EN | Hiện hành | Đề xuất | Lý do | Tin cậy (lượt 1 / lượt 2) |
|---|---|---|---|---|---|---|
| `tpl-massive_intents-audio_volume_other` | `translations.massive_intents.audio_volume_other` | other volume adjustments | điều chỉnh âm lượng khác | **điều chỉnh âm lượng theo cách khác** | EN "other volume adjustments" là nhóm còn lại ngoài tăng/giảm/tắt tiếng, nhưng "điều chỉnh âm lượng khác" dễ đọc thành "chỉnh một âm lượng khác" vì "khác" bám vào "âm lượng"; sửa để "khác" bổ nghĩa cho cách điều chỉnh (mức độ chắc chắn vừa phải). | 0.6 / 0.65 |
| `tpl-massive_intents-takeaway_query` | `translations.massive_intents.takeaway_query` | ask about a takeaway order or delivery | hỏi về đơn hoặc dịch vụ đồ ăn mang về | **hỏi về đơn đồ ăn mang về hoặc việc giao đồ ăn** | EN "ask about a takeaway order or delivery": bản hiện hành đổi "delivery" thành "dịch vụ" (rộng hơn, mất ý giao hàng) và để "đơn" treo trước "hoặc"; bản sửa giữ đủ hai vế "order" và "delivery", vẫn phân biệt với takeaway_order "đặt đồ ăn mang về". | 0.7 / 0.7 |

## Chuỗi OK: duyệt nhanh ở màn "Mẫu và nhãn"

Cả hai lượt đều kết luận OK cho các chuỗi dưới đây. Khi người làm dự án đồng ý, bấm "Duyệt" từng dòng.

### Câu hỏi mẫu (4)

| Khóa | EN | Nhãn gốc | Tiếng Việt |
|---|---|---|---|
| `translations.questions.sib200` | Which topic does this text belong to? | — | Văn bản này thuộc chủ đề nào? |
| `translations.questions.massive` | What does the user want the assistant to do? | — | Người dùng muốn trợ lý làm gì? |
| `translations.questions.xcopa_cause` | What is the more plausible cause? | — | Nguyên nhân nào hợp lý hơn? |
| `translations.questions.xcopa_effect` | What is the more plausible effect? | — | Kết quả nào hợp lý hơn? |

### Tên chủ đề SIB-200 (7)

| Khóa | EN | Nhãn gốc | Tiếng Việt |
|---|---|---|---|
| `translations.sib200_topics.entertainment` | entertainment | category = entertainment | giải trí |
| `translations.sib200_topics.geography` | geography | category = geography | địa lý |
| `translations.sib200_topics.health` | health | category = health | sức khỏe |
| `translations.sib200_topics.politics` | politics | category = politics | chính trị |
| `translations.sib200_topics.science/technology` | science/technology | category = science/technology | khoa học/công nghệ |
| `translations.sib200_topics.sports` | sports | category = sports | thể thao |
| `translations.sib200_topics.travel` | travel | category = travel | du lịch |

### Mô tả ý định MASSIVE (50)

| Khóa | EN | Nhãn gốc | Tiếng Việt |
|---|---|---|---|
| `translations.massive_intents.alarm_query` | check existing alarms | intent = alarm_query | xem các báo thức đã đặt |
| `translations.massive_intents.alarm_remove` | remove an alarm | intent = alarm_remove | xóa báo thức |
| `translations.massive_intents.alarm_set` | set an alarm | intent = alarm_set | đặt báo thức |
| `translations.massive_intents.audio_volume_down` | turn the volume down | intent = audio_volume_down | giảm âm lượng |
| `translations.massive_intents.audio_volume_mute` | mute the audio | intent = audio_volume_mute | tắt tiếng |
| `translations.massive_intents.audio_volume_up` | turn the volume up | intent = audio_volume_up | tăng âm lượng |
| `translations.massive_intents.calendar_query` | check the calendar | intent = calendar_query | xem lịch |
| `translations.massive_intents.calendar_remove` | remove a calendar event | intent = calendar_remove | xóa sự kiện khỏi lịch |
| `translations.massive_intents.calendar_set` | add a calendar event or reminder | intent = calendar_set | thêm sự kiện hoặc lời nhắc vào lịch |
| `translations.massive_intents.cooking_query` | answer a cooking question | intent = cooking_query | trả lời câu hỏi về nấu ăn |
| `translations.massive_intents.cooking_recipe` | find a recipe | intent = cooking_recipe | tìm công thức nấu ăn |
| `translations.massive_intents.datetime_convert` | convert time between time zones | intent = datetime_convert | đổi giờ giữa các múi giờ |
| `translations.massive_intents.datetime_query` | tell the date or time | intent = datetime_query | cho biết ngày hoặc giờ |
| `translations.massive_intents.email_addcontact` | add an email contact | intent = email_addcontact | thêm liên hệ email |
| `translations.massive_intents.email_query` | check emails | intent = email_query | kiểm tra email |
| `translations.massive_intents.email_querycontact` | look up contact information | intent = email_querycontact | tra cứu thông tin liên hệ |
| `translations.massive_intents.email_sendemail` | send an email | intent = email_sendemail | gửi email |
| `translations.massive_intents.general_joke` | tell a joke | intent = general_joke | kể chuyện cười |
| `translations.massive_intents.iot_cleaning` | start the robot vacuum cleaner | intent = iot_cleaning | bật robot hút bụi |
| `translations.massive_intents.iot_coffee` | make coffee | intent = iot_coffee | pha cà phê |
| `translations.massive_intents.iot_hue_lightchange` | change the light color | intent = iot_hue_lightchange | đổi màu đèn |
| `translations.massive_intents.iot_hue_lightdim` | dim the lights | intent = iot_hue_lightdim | giảm độ sáng đèn |
| `translations.massive_intents.iot_hue_lightoff` | turn the lights off | intent = iot_hue_lightoff | tắt đèn |
| `translations.massive_intents.iot_hue_lighton` | turn the lights on | intent = iot_hue_lighton | bật đèn |
| `translations.massive_intents.iot_hue_lightup` | brighten the lights | intent = iot_hue_lightup | tăng độ sáng đèn |
| `translations.massive_intents.iot_wemo_off` | turn off a smart plug | intent = iot_wemo_off | tắt ổ cắm thông minh |
| `translations.massive_intents.iot_wemo_on` | turn on a smart plug | intent = iot_wemo_on | bật ổ cắm thông minh |
| `translations.massive_intents.lists_createoradd` | create a list or add to a list | intent = lists_createoradd | tạo danh sách hoặc thêm vào danh sách |
| `translations.massive_intents.lists_query` | read out a list | intent = lists_query | xem nội dung danh sách |
| `translations.massive_intents.lists_remove` | remove from a list or delete a list | intent = lists_remove | xóa mục khỏi danh sách hoặc xóa danh sách |
| `translations.massive_intents.music_query` | tell which music is playing | intent = music_query | cho biết bài nhạc đang phát |
| `translations.massive_intents.news_query` | give the news | intent = news_query | đọc tin tức |
| `translations.massive_intents.play_audiobook` | play an audiobook | intent = play_audiobook | phát sách nói |
| `translations.massive_intents.play_game` | play a game | intent = play_game | chơi trò chơi |
| `translations.massive_intents.play_music` | play music | intent = play_music | phát nhạc |
| `translations.massive_intents.play_podcasts` | play a podcast | intent = play_podcasts | phát podcast |
| `translations.massive_intents.play_radio` | play the radio | intent = play_radio | mở radio |
| `translations.massive_intents.qa_currency` | answer a currency exchange question | intent = qa_currency | trả lời câu hỏi về tỷ giá tiền tệ |
| `translations.massive_intents.qa_definition` | define a word or term | intent = qa_definition | giải nghĩa một từ hoặc thuật ngữ |
| `translations.massive_intents.qa_factoid` | answer a factual question | intent = qa_factoid | trả lời một câu hỏi kiến thức |
| `translations.massive_intents.qa_maths` | do a math calculation | intent = qa_maths | tính toán |
| `translations.massive_intents.qa_stock` | give stock market information | intent = qa_stock | cung cấp thông tin chứng khoán |
| `translations.massive_intents.recommendation_events` | recommend events | intent = recommendation_events | gợi ý sự kiện |
| `translations.massive_intents.recommendation_locations` | recommend places | intent = recommendation_locations | gợi ý địa điểm |
| `translations.massive_intents.recommendation_movies` | recommend movies | intent = recommendation_movies | gợi ý phim |
| `translations.massive_intents.social_post` | post on social media or send a complaint | intent = social_post | đăng lên mạng xã hội hoặc gửi phản ánh |
| `translations.massive_intents.social_query` | check social media updates | intent = social_query | xem cập nhật trên mạng xã hội |
| `translations.massive_intents.takeaway_order` | order takeaway food | intent = takeaway_order | đặt đồ ăn mang về |
| `translations.massive_intents.transport_taxi` | book a taxi | intent = transport_taxi | đặt taxi |
| `translations.massive_intents.weather_query` | give the weather | intent = weather_query | cho biết thời tiết |

### Có/Không (2)

| Khóa | EN | Nhãn gốc | Tiếng Việt |
|---|---|---|---|
| `train_format.noul_options.vi.0` | Yes | đúng khi nhãn gốc = supported | Có |
| `train_format.noul_options.vi.1` | No | đúng khi nhãn gốc = refuted hoặc nei | Không |

## File

- Đầu vào: `results/bench_candidates/pilot2/audit/templates_in.jsonl`, `templates_context.json`.
- Kết quả lượt 1: `templates_out.jsonl`.
- Kết quả lượt 2: `templates_verify_in.jsonl`, `templates_verify_out.jsonl`.
- Code: `scripts/pilot/build_pilot2_inputs.py`, `make_template_verify.py`, `finish_template_review.py`.
