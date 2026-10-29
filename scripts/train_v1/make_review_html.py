"""Việc (b), viết sẵn, CHẠY KHI CÓ BẢN DỊCH: xuất 30 dòng ngẫu nhiên của phần B (seed cố định, chia đều theo dạng bài: đọc hiểu, ý định,
kiểm chứng, yes/no, nhân quả) thành một file HTML để người dùng đọc kiểm: mỗi dòng có đoạn văn, câu hỏi và lựa chọn, bản EN và bản VI cạnh
nhau, đáp án đánh dấu. KHÔNG train khi người dùng chưa xem.

Cách chạy (Python Windows hoặc venv WSL đều được, chỉ dùng thư viện chuẩn):
    python scripts/train_v1/make_review_html.py <bản_dịch>.jsonl [--n 30] [--seed 20261017] [--out results/train_v1/review/doc_kiem_ban_dich.html]
Đọc results/train_v1/part_b_en.jsonl (bản EN + nhãn) và config/train_format.json (Có/Không của câu noul)."""
import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("dich")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=20261017)
ap.add_argument("--out", default=str(ROOT / "results/train_v1/review/doc_kiem_ban_dich.html"))
a = ap.parse_args()

en = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "results/train_v1/part_b_en.jsonl", encoding="utf-8") if l.strip()}
vi = {json.loads(l)["id"]: json.loads(l) for l in open(a.dich, encoding="utf-8") if l.strip()}
fmt = json.load(open(ROOT / "config/train_format.json", encoding="utf-8"))
NAME = {"doc_hieu": "Đọc hiểu", "y_dinh": "Ý định", "kiem_chung": "Kiểm chứng", "yes_no": "Yes/no", "nhan_qua": "Nhân quả"}
by = defaultdict(list)
for i, r in sorted(en.items()):
    if i in vi:
        by[r["dang_bai"]].append(i)
groups = [g for g in NAME if by[g]]
rng = random.Random(a.seed)
per, extra = divmod(a.n, len(groups))
pick = []
for k, g in enumerate(groups):          # chia đều; phần dư cho các dạng bài đầu danh sách
    pick += [(g, i) for i in rng.sample(by[g], min(len(by[g]), per + (1 if k < extra else 0)))]
missing = sorted(set(en) - set(vi))

e = lambda s: html.escape(s or "").replace("\n", "<br>")


def opts(r_en, r_vi):
    if r_en["kind"] == "noul":
        o_en, o_vi = fmt["noul_options"]["en"], fmt["noul_options"]["vi"]
    else:
        o_en, o_vi = r_en["options"], r_vi["options"]
    t = r_en["target"]
    best = max(range(len(t)), key=lambda k: t[k])
    rows = []
    for k, (x, y) in enumerate(zip(o_en, o_vi)):
        mark = ' class="dung"' if k == best else ""
        p = f' <span class="p">({t[k]:.2f})</span>' if 0 < t[k] < 1 else ""
        tick = "✓ " if k == best else ""
        rows.append(f"<tr{mark}><td class='l'>{chr(65 + k)}</td><td>{tick}{e(x)}{p}</td><td>{tick}{e(y)}{p}</td></tr>")
    return "".join(rows)


cards = []
for n, (g, i) in enumerate(pick, 1):
    r, v = en[i], vi[i]
    flag = " · cùng khuôn COPA" if r.get("cung_khuon_copa") else ""
    cards.append(f"""<section class="card"><h2>{n}. {NAME[g]} <span class="meta">{e(r['source'])} · {e(r['kind'])}{flag} · <code>{e(i)}</code></span></h2>
<table><tr><th></th><th>Tiếng Anh (gốc)</th><th>Tiếng Việt (bản dịch)</th></tr>
<tr><td class='l'>Văn bản</td><td>{e(r['state'])}</td><td>{e(v['state'])}</td></tr>
<tr><td class='l'>Câu hỏi</td><td>{e(r['question'])}</td><td>{e(v['question'])}</td></tr>
{opts(r, v)}</table></section>""")
page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đọc kiểm bản dịch</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--ok:#e6f4ea;--okb:#1e7e34;--card:#fafafa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--mut:#aaa;--line:#333;--ok:#173d24;--okb:#6fd38a;--card:#1d1d1d}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}}
h1{{font-size:22px}} h2{{font-size:16px;margin:0 0 8px}} .meta{{color:var(--mut);font-weight:normal;font-size:13px}}
table{{border-collapse:collapse;width:100%;table-layout:fixed}} td,th{{border-top:1px solid var(--line);padding:6px;vertical-align:top;text-align:left;overflow-wrap:anywhere}}
td.l{{width:80px;color:var(--mut)}} tr.dung td{{background:var(--ok)}} tr.dung td:not(.l){{font-weight:600;color:var(--okb)}} .p{{color:var(--mut);font-weight:normal}}
</style></head><body><h1>Đọc kiểm bản dịch phần B: {len(pick)} dòng</h1>
<p>Dòng chọn ngẫu nhiên (seed {a.seed}), chia đều theo dạng bài: {", ".join(f"{NAME[g]} {sum(1 for x, _ in pick if x == g)}" for g in groups)}.
Đáp án đúng (theo nhãn bản gốc) tô xanh, có dấu ✓; nhãn mềm ghi xác suất. Câu Có/Không: lựa chọn cố định của dự án.
Bản dịch có {len(vi)} / {len(en)} dòng của phần B; mẫu chỉ lấy trong các dòng đã dịch.</p>
<p>Cần xem: bản dịch có giữ đúng nghĩa không; đáp án đánh dấu có còn đúng với bản tiếng Việt không; lựa chọn có đúng thứ tự không; tên riêng, số có giữ nguyên không.</p>
{''.join(cards)}</body></html>"""
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(page, encoding="utf-8")
print(json.dumps({"out": a.out, "dong": len(pick), "theo_dang_bai": {g: sum(1 for x, _ in pick if x == g) for g in groups}, "thieu_trong_ban_dich": len(missing)},
                 ensure_ascii=False))
