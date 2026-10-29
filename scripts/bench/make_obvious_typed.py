"""Câu hiển nhiên cho kiểm adapter noul/score (Việt và Anh). Score dùng thang trung tính có thứ tự (không ngôn từ thù ghét),
mỗi câu ở hai chiều thang (xuôi và ngược) để bắt lỗi gán nhầm chỉ số mức. Ra: results/dev_v1/obvious_typed.jsonl"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, write_jsonl  # noqa: E402

NOUL = [  # (state_vi, claim_vi, state_en, claim_en, true?)
    ("Hà Nội là thủ đô của Việt Nam.", "Hà Nội là thủ đô của Việt Nam.", "Hanoi is the capital of Vietnam.", "Hanoi is the capital of Vietnam.", True),
    ("Con mèo màu trắng đang ngủ trên ghế.", "Con mèo màu đen.", "The white cat is sleeping on the chair.", "The cat is black.", False),
    ("Lan sinh năm 1990 ở Huế.", "Lan sinh ở Huế.", "Lan was born in 1990 in Hue.", "Lan was born in Hue.", True),
    ("Cửa hàng đóng cửa vào Chủ nhật.", "Cửa hàng mở cửa vào Chủ nhật.", "The shop is closed on Sundays.", "The shop is open on Sundays.", False),
    ("Nam đã ăn hai quả táo.", "Nam đã ăn táo.", "Nam ate two apples.", "Nam ate apples.", True),
    ("Trời mưa suốt cả ngày hôm qua.", "Hôm qua trời nắng đẹp cả ngày.", "It rained all day yesterday.", "Yesterday was sunny all day.", False),
    ("Chuyến bay cất cánh lúc 8 giờ sáng.", "Chuyến bay cất cánh vào buổi sáng.", "The flight took off at 8 am.", "The flight took off in the morning.", True),
    ("Hùng cao 1m80, còn Minh cao 1m60.", "Minh cao hơn Hùng.", "Hung is 1.80 m tall and Minh is 1.60 m tall.", "Minh is taller than Hung.", False),
    ("Công ty có 50 nhân viên.", "Công ty có nhân viên.", "The company has 50 employees.", "The company has employees.", True),
    ("Bài kiểm tra bị hủy vì bão.", "Bài kiểm tra diễn ra bình thường.", "The exam was cancelled because of the storm.", "The exam took place as usual.", False),
]
SCORE = [  # (state_vi, q_vi, levels_vi, state_en, q_en, levels_en, gold index in levels order)
    ("Nhiệt độ hôm nay là 40 độ C, nắng gắt.", "Thời tiết hôm nay nóng ở mức nào?", ["mát", "ấm", "rất nóng"],
     "Today it is 40 degrees Celsius with blazing sun.", "How hot is the weather today?", ["cool", "warm", "very hot"], 2),
    ("Nhiệt độ hôm nay là 12 độ C, gió lạnh.", "Thời tiết hôm nay nóng ở mức nào?", ["mát", "ấm", "rất nóng"],
     "Today it is 12 degrees Celsius with a cold wind.", "How hot is the weather today?", ["cool", "warm", "very hot"], 0),
    ("Bộ phim hay tuyệt vời, tôi sẽ xem lại nhiều lần!", "Người viết thích bộ phim ở mức nào?", ["không thích", "bình thường", "rất thích"],
     "The movie was wonderful, I will watch it again many times!", "How much did the writer like the movie?", ["disliked it", "neutral", "loved it"], 2),
    ("Bộ phim dở tệ, phí tiền và thời gian.", "Người viết thích bộ phim ở mức nào?", ["không thích", "bình thường", "rất thích"],
     "The movie was terrible, a waste of money and time.", "How much did the writer like the movie?", ["disliked it", "neutral", "loved it"], 0),
    ("Món ăn tạm được, không ngon cũng không dở.", "Người viết đánh giá món ăn ở mức nào?", ["tệ", "bình thường", "rất ngon"],
     "The food was okay, neither good nor bad.", "How did the writer rate the food?", ["bad", "average", "excellent"], 1),
]
rows = []
for i, (sv, cv, se, ce, t) in enumerate(NOUL):
    for lang, s, c in (("vi", sv, cv), ("en", se, ce)):
        q = f"Văn bản có hỗ trợ phát biểu sau không? Phát biểu: {c}" if lang == "vi" else f"Does the text support this statement? Statement: {c}"
        opts = ["Có", "Không"] if lang == "vi" else ["Yes", "No"]
        rows.append({"id": f"obv_noul:{i}:{lang}", "pair_id": f"obv_noul:{i}", "source": "obvious", "lang": lang, "kind": "noul",
                     "variant": "orig", "state": s, "question": q, "options": opts, "target": 0 if t else 1, "cluster": f"obv_noul:{i}"})
for i, (sv, qv, lv, se, qe, le, g) in enumerate(SCORE):
    for lang, s, q, lv_ in (("vi", sv, qv, lv), ("en", se, qe, le)):
        for direction in ("up", "down"):
            levels = lv_ if direction == "up" else lv_[::-1]
            tgt = g if direction == "up" else len(lv_) - 1 - g
            rows.append({"id": f"obv_score:{i}:{lang}:{direction}", "pair_id": f"obv_score:{i}", "source": "obvious", "lang": lang,
                         "kind": "score", "variant": "orig", "state": s, "question": q, "options": levels, "target": tgt,
                         "cluster": f"obv_score:{i}"})
write_jsonl(RESULTS / "dev_v1" / "obvious_typed.jsonl", rows)
print(len(rows), "rows")
