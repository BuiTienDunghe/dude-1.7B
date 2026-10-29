"""So sánh Sonnet ↔ Google trên 200 dòng dịch thử: GIẢI MÃ kết quả chấm mù (Opus) và tổng hợp (chỉ thư viện chuẩn).
Đọc pilot/so_sanh/cham/lo_{k}_out.jsonl + cham_map.json + kiem_khach_quan.json. Ra: pilot/so_sanh/tong_hop.json, pilot/SO_SANH_sonnet_google.md
(bảng số; phần nhận định viết tay nối sau) và pilot/so_sanh.html (30 dòng, 6 mỗi dạng bài, EN | Sonnet | Google, đáp án đánh dấu, nhận xét người chấm;
ghi nguồn vì người dùng đọc sau khi đã chấm mù)."""
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
P = T / "translated_sonnet/pilot"
S = P / "so_sanh"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
mp = json.load(open(S / "cham_map.json", encoding="utf-8"))
kq = json.load(open(S / "kiem_khach_quan.json", encoding="utf-8"))
out = {}
for f in sorted((S / "cham").glob("lo_*_out.jsonl")):
    for r in rd(f):
        out[r["id_cham"]] = r
assert set(out) == set(mp), (len(out), len(mp))
NAME = {"doc_hieu": "Đọc hiểu", "y_dinh": "Ý định", "kiem_chung": "Kiểm chứng", "yes_no": "Yes/no", "nhan_qua": "Nhân quả"}
rows = []
for cid, m in mp.items():
    r = out[cid]
    by = {m["X"]: r["X"], m["Y"]: r["Y"]}
    win = {"X": m["X"], "Y": m["Y"], "ngang": "ngang"}[r["ket_luan"]]
    # nhận xét giữ nguyên chữ của người chấm (nói theo X/Y); cách giải mã ghi kèm, không thay chữ tự động (tránh đụng các từ như "Y tế")
    rows.append({"id_cham": cid, "id": m["id"], "dang_bai": m["dang_bai"], "sonnet": by["sonnet"], "google": by["google"], "thang": win,
                 "nhan_xet_goc": r["nhan_xet"], "X_la": m["X"]})


def agg(rs):
    n = len(rs)
    c = Counter(r["thang"] for r in rs)
    f = lambda who, k: sum(bool(r[who][k]) for r in rs)
    return {"n": n, "sonnet_thang": c["sonnet"], "google_thang": c["google"], "ngang": c["ngang"],
            **{f"{who}_{k}": f(who, k) for who in ("sonnet", "google") for k in ("doi_nghia", "doi_dap_an")},
            "sonnet_tu_nhien_tb": round(sum(r["sonnet"]["tu_nhien"] for r in rs) / n, 2), "google_tu_nhien_tb": round(sum(r["google"]["tu_nhien"] for r in rs) / n, 2)}


groups = [g for g in NAME if any(r["dang_bai"] == g for r in rows)]
tab = {g: agg([r for r in rows if r["dang_bai"] == g]) for g in groups}
tab["tong"] = agg(rows)
# vị trí X/Y có thiên lệch không
pos = Counter((r["X_la"], out[r["id_cham"]]["ket_luan"]) for r in rows)


def gap(r):
    s, g = r["sonnet"], r["google"]
    return (abs(s["doi_dap_an"] - g["doi_dap_an"]) * 4 + abs(s["doi_nghia"] - g["doi_nghia"]) * 2 + abs(s["tu_nhien"] - g["tu_nhien"]))


top = sorted(rows, key=lambda r: (-gap(r), r["id_cham"]))[:10]
(S / "tong_hop.json").write_text(json.dumps({"bang": tab, "vi_tri_X_Y": {f"{a}|{b}": n for (a, b), n in pos.items()}, "top10": top, "dong": rows},
                                            ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- bảng số cho báo cáo
pct = lambda a, n: f"{a} ({100 * a / n:.0f}%)"
L = ["| Dạng bài | n | Sonnet thắng | Google thắng | Ngang | Đổi nghĩa: Sonnet | Đổi nghĩa: Google | Đổi đáp án: Sonnet | Đổi đáp án: Google | Tự nhiên TB: Sonnet | Tự nhiên TB: Google |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for g in groups + ["tong"]:
    t = tab[g]; n = t["n"]
    L.append(f"| {NAME.get(g, '**Tổng**')} | {n} | {pct(t['sonnet_thang'], n)} | {pct(t['google_thang'], n)} | {pct(t['ngang'], n)} | {pct(t['sonnet_doi_nghia'], n)} | "
             f"{pct(t['google_doi_nghia'], n)} | {pct(t['sonnet_doi_dap_an'], n)} | {pct(t['google_doi_dap_an'], n)} | {t['sonnet_tu_nhien_tb']:.2f} | {t['google_tu_nhien_tb']:.2f} |")
(S / "bang_so.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- HTML 30 dòng
src = {r["id"]: r for r in rd(P / "src_200.jsonl")}
son = {r["id"]: r for r in rd(P / "ban_s2_200.jsonl")}
goo = {r["id"]: r for r in rd(S / "google_200.jsonl")}
meta = {r["id"]: r for r in rd(T / "part_b_meta.jsonl") if r["id"] in src}
fmt = json.load(open(ROOT / "config/train_format.json", encoding="utf-8"))
rng = random.Random(20261017)
pick = []
for g in groups:
    pick += rng.sample(sorted([r for r in rows if r["dang_bai"] == g], key=lambda r: r["id_cham"]), 6)
e = lambda s: html.escape(s or "").replace("\n", "<br>")


def card(n, r):
    i = r["id"]; m = meta[i]; t = m["target"]
    if m["kind"] == "noul":
        oe, os_, og = fmt["noul_options"]["en"], fmt["noul_options"]["vi"], fmt["noul_options"]["vi"]
    else:
        oe, os_, og = src[i]["options"], son[i]["options"], goo[i]["options"]
    best = max(range(len(t)), key=lambda k: t[k])
    trs = "".join(f"<tr{' class=dung' if k == best else ''}><td class=l>{chr(65 + k)}</td><td>{'✓ ' if k == best else ''}{e(a)}</td>"
                  f"<td>{'✓ ' if k == best else ''}{e(b)}</td><td>{'✓ ' if k == best else ''}{e(c)}</td></tr>" for k, (a, b, c) in enumerate(zip(oe, os_, og)))
    s, g = r["sonnet"], r["google"]
    badge = lambda x: (f"đổi nghĩa{' – ĐỔI ĐÁP ÁN' if x['doi_dap_an'] else ''}: {e(x['cho'])}" if x["doi_nghia"] else "không lỗi đổi nghĩa") + f" · tự nhiên {x['tu_nhien']}/5"
    win = {"sonnet": "Sonnet tốt hơn", "google": "Google tốt hơn", "ngang": "Ngang"}[r["thang"]]
    return f"""<section class=card><h2>{n}. {NAME[r['dang_bai']]} <span class=meta>{e(m['source'])} · {e(m['kind'])} · <code>{e(i)}</code></span></h2>
<table><tr><th></th><th>Tiếng Anh (gốc)</th><th>Sonnet (dịch + đọc lại)</th><th>Google Translate (agent ngoài)</th></tr>
<tr><td class=l>Văn bản</td><td>{e(src[i]['state'])}</td><td>{e(son[i]['state'])}</td><td>{e(goo[i]['state'])}</td></tr>
<tr><td class=l>Câu hỏi</td><td>{e(src[i]['question'])}</td><td>{e(son[i]['question'])}</td><td>{e(goo[i]['question'])}</td></tr>{trs}
<tr class=ch><td class=l>Người chấm</td><td><b>{win}</b><br>{e(r['nhan_xet_goc'])}<br><span class=meta>(X = {r['X_la'].capitalize()}, Y = {'Google' if r['X_la'] == 'sonnet' else 'Sonnet'})</span></td>
<td>{badge(s)}</td><td>{badge(g)}</td></tr></table></section>"""


t = tab["tong"]
page = f"""<!doctype html><html lang=vi><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>So sánh bản dịch</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--ok:#e6f4ea;--okb:#1e7e34;--card:#fafafa;--ch:#fff8e1}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--mut:#aaa;--line:#333;--ok:#173d24;--okb:#6fd38a;--card:#1d1d1d;--ch:#2a2615}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0;overflow-x:auto}}
h1{{font-size:22px}} h2{{font-size:16px;margin:0 0 8px}} .meta{{color:var(--mut);font-weight:normal;font-size:13px}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;min-width:720px}} td,th{{border-top:1px solid var(--line);padding:6px;vertical-align:top;text-align:left;overflow-wrap:anywhere}}
td.l{{width:80px;color:var(--mut)}} tr.dung td{{background:var(--ok)}} tr.dung td:not(.l){{font-weight:600;color:var(--okb)}} tr.ch td{{background:var(--ch);font-size:14px}}
</style></head><body><h1>So sánh bản dịch: Sonnet và Google Translate, 30 dòng</h1>
<p>Trên cả 200 dòng chấm mù (Opus, X/Y xáo ngẫu nhiên): Sonnet thắng {t['sonnet_thang']}, Google thắng {t['google_thang']}, ngang {t['ngang']}.
Lỗi đổi nghĩa: Sonnet {t['sonnet_doi_nghia']}, Google {t['google_doi_nghia']} dòng. Lỗi đổi đáp án: Sonnet {t['sonnet_doi_dap_an']}, Google {t['google_doi_dap_an']} dòng.
Độ tự nhiên trung bình: Sonnet {t['sonnet_tu_nhien_tb']:.2f}, Google {t['google_tu_nhien_tb']:.2f} (thang 1–5).</p>
<p>30 dòng dưới đây chọn ngẫu nhiên (seed 20261017), 6 dòng mỗi dạng bài. Đáp án đúng tô xanh, có dấu ✓. Câu Có/Không dùng lựa chọn cố định. Hàng vàng là nhận xét của người chấm; nguồn chỉ ghi ở đây, người chấm không biết.</p>
{''.join(card(n, r) for n, r in enumerate(pick, 1))}</body></html>"""
(P / "so_sanh.html").write_text(page, encoding="utf-8")
print(json.dumps({"tong": tab["tong"], "vi_tri": {f"{a}|{b}": n for (a, b), n in pos.items()}}, ensure_ascii=False))
