"""10 câu hiển nhiên tự viết (song song Việt–Anh), 4 lựa chọn, đáp án đúng ở các vị trí khác nhau.
Dùng để kiểm adapter: model phải chọn đúng NỘI DUNG ở cả thứ tự thường và thứ tự đảo.
Ra: results/baselines/obvious_v0.jsonl (cùng định dạng dev_v0, n_tokens = null)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, write_jsonl

ITEMS = [  # (state_vi, q_vi, opts_vi, state_en, q_en, opts_en, target)
    ("Hà Nội là thủ đô của Việt Nam.", "Thủ đô của Việt Nam là thành phố nào?", ["Hà Nội", "Bangkok", "Paris", "Tokyo"],
     "Hanoi is the capital of Vietnam.", "Which city is the capital of Vietnam?", ["Hanoi", "Bangkok", "Paris", "Tokyo"], 0),
    ("Nước sôi ở 100 độ C ở mực nước biển.", "Nước sôi ở bao nhiêu độ C ở mực nước biển?", ["0 độ", "100 độ", "37 độ", "500 độ"],
     "Water boils at 100 degrees Celsius at sea level.", "At what temperature does water boil at sea level?", ["0 degrees", "100 degrees", "37 degrees", "500 degrees"], 1),
    ("Con mèo của Lan có bộ lông màu trắng như tuyết.", "Con mèo của Lan có màu gì?", ["đen", "đỏ", "trắng", "xanh lá"],
     "Lan's cat has fur as white as snow.", "What color is Lan's cat?", ["black", "red", "white", "green"], 2),
    ("Một tuần có bảy ngày.", "Một tuần có bao nhiêu ngày?", ["năm", "mười", "ba", "bảy"],
     "A week has seven days.", "How many days are in a week?", ["five", "ten", "three", "seven"], 3),
    ("Anh Minh vừa trúng xổ số và cười rất tươi.", "Tâm trạng của anh Minh thế nào?", ["vui", "buồn", "giận dữ", "sợ hãi"],
     "Minh just won the lottery and is smiling widely.", "How does Minh feel?", ["happy", "sad", "angry", "scared"], 0),
    ("Trời đang mưa to, mọi người mở ô khi ra đường.", "Thời tiết hiện tại thế nào?", ["nắng gắt", "mưa", "tuyết rơi", "có bão cát"],
     "It is raining heavily and people open umbrellas outside.", "What is the weather like?", ["sunny", "rainy", "snowy", "a sandstorm"], 1),
    ("Hai cộng hai bằng bốn.", "Hai cộng hai bằng mấy?", ["một", "ba", "bốn", "chín"],
     "Two plus two equals four.", "What is two plus two?", ["one", "three", "four", "nine"], 2),
    ("Mặt trời mọc ở hướng đông mỗi buổi sáng.", "Mặt trời mọc ở hướng nào?", ["tây", "bắc", "nam", "đông"],
     "The sun rises in the east every morning.", "In which direction does the sun rise?", ["west", "north", "south", "east"], 3),
    ("Bác sĩ khám bệnh cho bệnh nhân trong bệnh viện.", "Ai khám bệnh cho bệnh nhân?", ["bác sĩ", "phi công", "đầu bếp", "thợ điện"],
     "The doctor examines patients in the hospital.", "Who examines the patients?", ["the doctor", "the pilot", "the chef", "the electrician"], 0),
    ("Cá sống dưới nước và thở bằng mang.", "Cá sống ở đâu?", ["trên cây", "dưới nước", "trong sa mạc", "trên mây"],
     "Fish live in water and breathe with gills.", "Where do fish live?", ["in trees", "in water", "in the desert", "in the clouds"], 1),
]
rows = []
for i, (sv, qv, ov, se, qe, oe, t) in enumerate(ITEMS):
    for lang, s, q, o in (("en", se, qe, oe), ("vi", sv, qv, ov)):
        rows.append({"id": f"obvious:{i}:{lang}", "pair_id": f"obvious:{i}", "source": "obvious", "lang": lang, "kind": "check",
                     "state": s, "question": q, "options": o, "target": t, "split_origin": "self-written", "n_tokens": None,
                     "cluster": f"obvious:{i}", "gold_label": None, "flags": []})
out = RESULTS / "baselines" / "obvious_v0.jsonl"
write_jsonl(out, rows)
print(out, len(rows))
