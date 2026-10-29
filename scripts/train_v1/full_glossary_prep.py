"""Chuẩn bị bảng thuật ngữ v2 cho bản dịch ĐẦY ĐỦ phần B (chưa dịch dòng dữ liệu nào; chỉ thư viện chuẩn).
Gom từ to_translate/part_b_to_translate.jsonl:
  - câu hỏi và lựa chọn lặp >= 5 lần (như chuoi_lap_lai.json);
  - lựa chọn dạng mã (chữ thường/số/gạch dưới) và nhãn trích trong ngoặc kép bên trong câu hỏi (phải dịch giống nhãn ở lựa chọn);
  - câu hỏi kèm trong state của e-CARE;
  - tiền tố người nói/định dạng ở đầu dòng của state (để hướng dẫn chia: giữ nguyên hay dịch cố định).
Mỗi mục kèm số lần, nguồn, 1 ví dụ ngữ cảnh ngắn, và bản dịch pilot (nếu có trong pilot/bang_thuat_ngu.json).
Ra: results/train_v1/translated_sonnet/full_prep/thuat_ngu_v2_dau_vao_{1,2}.json (1 = câu hỏi/lựa chọn/nhãn thường; 2 = nhãn dạng mã),
tien_to.json."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
O = T / "translated_sonnet/full_prep"
O.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(T / "to_translate/part_b_to_translate.jsonl", encoding="utf-8") if l.strip()]
meta = {json.loads(l)["id"]: json.loads(l) for l in open(T / "part_b_meta.jsonl", encoding="utf-8") if l.strip()}
pilot = json.load(open(T / "translated_sonnet/pilot/bang_thuat_ngu.json", encoding="utf-8"))
code = re.compile(r"^[a-z0-9_]+$")
qn, on, quoted = Counter(r["question"] for r in rows), Counter(o for r in rows for o in r["options"]), Counter()
src, ctx = defaultdict(set), {}
for r in rows:
    s = meta[r["id"]]["source"]
    for x in re.findall(r'"([^"]{1,80})"', r["question"]):
        quoted[x] += 1; src[x].add(s); ctx.setdefault(x, ("cau_hoi", r["question"][:200]))
    src[r["question"]].add(s); ctx.setdefault(r["question"], ("state", r["state"][:200]))
    for o in r["options"]:
        src[o].add(s); ctx.setdefault(o, ("state", r["state"][:200] + " || lựa chọn: " + " | ".join(r["options"])[:200]))
terms = {}


def put(t, loai, n):
    if t in terms:
        terms[t]["loai"] = sorted(set(terms[t]["loai"]) | {loai}); terms[t]["so_lan"] = max(terms[t]["so_lan"], n); return
    terms[t] = {"loai": [loai], "so_lan": n, "nguon": sorted(src[t])[:4], "vi_du": ctx.get(t, ("", ""))[1], "vi_pilot": pilot.get(t, {}).get("vi")}


for q, n in qn.items():
    if n >= 5:
        put(q, "cau_hoi_lap", n)
LABEL_SRC = {"banking77", "clinc_oos/plus", "multidogo/insurance/intent", "health_fact", "nli_fever", "vitaminc", "headline_cause/en_simple"}
for o, n in on.items():
    # dạng mã chỉ khi là nhãn thật: có gạch dưới, lặp >= 5, hoặc thuộc nguồn nhãn cố định (đáp án một từ của bài đọc hiểu KHÔNG vào bảng)
    if n >= 5 or (code.match(o) and ("_" in o or src[o] & LABEL_SRC)):
        put(o, "lua_chon_dang_ma" if code.match(o) else "lua_chon_lap", n)
for x, n in quoted.items():
    if not (src[x] & LABEL_SRC or n >= 5 or "_" in x or x in on and on[x] >= 5):
        continue                                     # trích dẫn thường (câu trả lời, tên) trong câu hỏi noul: dịch theo ngữ cảnh, không vào bảng
    put(x, "nhan_trong_ngoac_dang_ma" if code.match(x) else "nhan_trong_ngoac", n)
for x in ("What was the cause of this?", "What happened as a result?"):
    terms[x] = {"loai": ["cau_hoi_kem_ecare"], "so_lan": sum(1 for r in rows if r["state"].endswith(x)), "nguon": ["e-CARE"], "vi_du": "", "vi_pilot": None}
is_code = lambda t, v: any(l.endswith("dang_ma") for l in v["loai"]) and not re.match(r"^\d+$", t)
p1 = {t: v for t, v in terms.items() if not is_code(t, v)}
p2 = {t: v for t, v in terms.items() if is_code(t, v)}
(O / "thuat_ngu_v2_dau_vao_1.json").write_text(json.dumps(p1, ensure_ascii=False, indent=1), encoding="utf-8")
(O / "thuat_ngu_v2_dau_vao_2.json").write_text(json.dumps(p2, ensure_ascii=False, indent=1), encoding="utf-8")
pre = Counter()
for r in rows:
    for line in r["state"].split("\n"):
        m = re.match(r"^([A-Za-z_ ]{1,24}?\d*\s*:)", line)
        if m:
            pre[re.sub(r"\d+", "N", m.group(1))] += 1
(O / "tien_to.json").write_text(json.dumps(dict(pre.most_common()), ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"phan_1": len(p1), "phan_2_dang_ma": len(p2), "co_ban_pilot": sum(1 for v in terms.values() if v["vi_pilot"]),
                  "tien_to": len(pre), "theo_loai": dict(Counter(l for v in terms.values() for l in v["loai"]))}, ensure_ascii=False))
