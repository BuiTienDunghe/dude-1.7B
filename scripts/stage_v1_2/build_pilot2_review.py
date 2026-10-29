"""File đọc kiểm cho người: 20 câu MLQA recast 2 (từ 50 câu validation) + 20 câu Global-PIQA lô 1 (đã chấm thử, KHÔNG phải lô sạch).
Tự chứa (CSS nội tuyến, không script, không tài nguyên ngoài), mọi văn bản qua html.escape. Python Windows, thư viện chuẩn.
Ra: results/review/pilot2_review.html + results/review/pilot2_review_sheet.csv (UTF-8 có BOM, cột verdict để trống cho người điền).
"""
import csv
import hashlib
import html
import io
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "results/bench_candidates"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
e = lambda s: html.escape("" if s is None else str(s), quote=True)
SEED = 20261014
LET = "ABCD"

# ---------------------------------------------------------------- dữ liệu
m_rows = {r["id"]: r for r in read(B / "pilot2/mlqa2_rows.jsonl")}
m_pairs = sorted({r["pair_id"] for r in m_rows.values()})
m_pick = sorted(random.Random(f"{SEED}|review|mlqa").sample(m_pairs, 20))
m_ver = {x["id"]: x for x in read(B / "pilot2/audit/verify_out_mlqa2.jsonl")}
g_rows = {r["id"]: r for r in read(B / "pilot/pilot_rows.jsonl") if r["source"] == "gpiqa"}
g_pairs = sorted({r["pair_id"] for r in g_rows.values()})
g_pick = sorted(random.Random(f"{SEED}|review|gpiqa").sample(g_pairs, 20))
g_iss = {x["audit_id"]: x for x in read(B / "pilot2/gpiqa_issues_100.jsonl")}
import csv as _csv  # noqa: E402
tsv = {f"gpiqa:{r['example_id']}": r for r in _csv.DictReader(open(B / "pilot/raw/gpiqa/nonparallel_vie_latn.tsv", encoding="utf-8"), delimiter="\t")}


def highlight(text, opts, gold):
    """Bôi mọi lần xuất hiện của từng lựa chọn trong đoạn văn: đáp án xanh, nhiễu đỏ nhạt. Escape từng đoạn."""
    spans = []
    for j, o in enumerate(opts):
        k = text.find(o)
        while o and k >= 0:
            spans.append((k, k + len(o), j))
            k = text.find(o, k + 1)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, pos = [], 0
    for s, t, j in spans:
        if s < pos:
            continue
        out.append(e(text[pos:s]))
        cls = "hg" if j == gold else "hd"
        out.append(f'<span class="{cls}" title="lựa chọn {LET[j]}">{e(text[s:t])}</span>')
        pos = t
    out.append(e(text[pos:]))
    return "".join(out)


def opts_html(opts, gold):
    return '<ol class="opts" type="A">' + "".join(f'<li class="{"gold" if j == gold else ""}">{e(o)}{" ✔ đáp án chuẩn" if j == gold else ""}</li>'
                                                  for j, o in enumerate(opts)) + "</ol>"


CSS = """body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;margin:0;padding:16px;line-height:1.45;color:#1b1b1b;background:#fff}
main{max-width:1200px;margin:0 auto}h1{font-size:1.5rem}h2{font-size:1.25rem;margin-top:2.2rem;border-bottom:2px solid #ccc;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.9rem}th,td{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}th{background:#f2f2f2}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:.85rem}.item{border:1px solid #bbb;border-radius:6px;padding:8px 10px;margin:12px 0;background:#fcfcfc}
.iid{font-family:ui-monospace,Consolas,monospace;background:#e8eefc;padding:1px 6px;border-radius:4px;font-weight:600}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:800px){.grid2{grid-template-columns:1fr}}
.text{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;padding:6px;border-radius:4px}
.gold{font-weight:700;background:#e6f6e6}.hg{background:#c9f0c9;font-weight:600}.hd{background:#fbdcdc}
.warn{border:2px solid #c0392b;background:#fff3f2;padding:8px 10px;border-radius:6px;margin:10px 0}.note{border-left:4px solid #888;background:#f6f6f6;padding:6px 10px;margin:8px 0}
ol.opts{margin:.3rem 0 .3rem 1.4rem;padding:0}ol.opts li{margin:2px 0}.lab{color:#555;font-size:.85rem}nav.toc a{margin-right:.6rem}"""

H, sheet = [], []
H.append(f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đọc kiểm chạy thử đợt 2</title><style>{CSS}</style></head><body><main>
<h1>Đọc kiểm chạy thử đợt 2: MLQA (4 cụm từ trả lời) và Global-PIQA</h1>
<p class="note">Dựng bằng <span class="mono">scripts/stage_v1_2/build_pilot2_review.py</span>. Seed lấy mẫu {SEED}. Mỗi mục có mã riêng (item_id), trùng cột item_id của
<span class="mono">results/review/pilot2_review_sheet.csv</span>; điền cột verdict ở file đó. Mã verdict: OK / SAI_NHAN / CAU_HOI_MO_HO / DICH_SAI / NHIEU_CUNG_DUNG / NHIEU_QUA_DE / CHINH_TA / KHAC.</p>
<nav class="toc"><b>Mục lục.</b> <a href="#mlqa">Phần A: MLQA (20)</a> <a href="#gpiqa">Phần B: Global-PIQA lô 1 (20)</a></nav>
<h2 id="mlqa">Phần A: MLQA, recast "4 cụm từ trả lời" (20 câu ngẫu nhiên từ 50 câu validation)</h2>
<p class="warn">Đáp án chuẩn là span do người gán (MLQA; bản Việt do người dịch chuyên nghiệp đánh dấu). <b>3 nhiễu do LLM (Opus) chọn</b>, chép nguyên văn từ đoạn văn,
và đã qua một lượt Opus kiểm độc lập. Câu hỏi là câu hỏi MLQA nguyên văn. Đoạn Việt và đoạn Anh là hai đoạn Wikipedia viết tự nhiên, <b>không phải bản dịch của nhau</b>.
Trong đoạn văn: <span class="hg">đáp án</span>, <span class="hd">nhiễu</span>. Cần đọc: nhiễu nào cũng đúng? Đáp án có đúng không? Nhiễu có quá dễ không?</p>""")
for n, pid in enumerate(m_pick, 1):
    iid = f"M-{n:02d}"
    v, en = m_rows[pid + ":vi"], m_rows[pid + ":en"]
    ver = m_ver.get(pid, {})
    weak = {l: ver.get(l, {}).get("nhieu_yeu", []) for l in ("vi", "en")}
    H.append(f'<div class="item"><div><span class="iid">{iid}</span> <span class="lab">{e(pid)} · loại đáp án: {e(next((f for f in v["flags"] if f.startswith("loai=")), "")[5:])}'
             f' · lượt kiểm độc lập ghi nhiễu yếu: VI {len(weak["vi"])}, EN {len(weak["en"])}</span></div><div class="grid2">'
             f'<div><b>Tiếng Việt</b><div class="text">{highlight(v["state"], v["options"], v["target"])}</div><p><b>Câu hỏi:</b> {e(v["question"])}</p>{opts_html(v["options"], v["target"])}</div>'
             f'<div><b>Tiếng Anh</b><div class="text">{highlight(en["state"], en["options"], en["target"])}</div><p><b>Question:</b> {e(en["question"])}</p>{opts_html(en["options"], en["target"])}</div>'
             f'</div></div>')
    sheet.append({"item_id": iid, "nguon": "mlqa_recast2", "pair_id": pid, "dap_an_vi": v["options"][v["target"]], "ghi_chu_may": "nhiễu do LLM chọn",
                  "verdict": "", "ghi_chu": ""})
H.append("""<h2 id="gpiqa">Phần B: Global-PIQA vi, lô 1 (20 câu ngẫu nhiên từ 50 câu lô 1)</h2>
<p class="warn">Lô 1 <b>đã được chấm thử</b> (decider-2b và adapter đối chứng), nên KHÔNG phải lô sạch cho test. Bản Việt do người Việt viết (bản gốc); bản Anh là bản dịch của bộ gốc.
Kèm kết luận soát của Opus (quét → soi lại) để đối chiếu: cần đọc xem nhãn có đúng, có mơ hồ không, và chính tả.</p>""")
for n, pid in enumerate(g_pick, 1):
    iid = f"G-{n:02d}"
    v = g_rows[pid + ":vi"]; t = tsv[pid]; it = g_iss[pid]
    en_opts = [t["eng_translated0"].strip(), t["eng_translated1"].strip()]
    fix = f'<p class="lab">Đề xuất sửa (chưa áp): {e(it["gia_tri_cu"])} → {e(it["gia_tri_de_xuat"])}</p>' if it["gia_tri_cu"] else ""
    cls = "note" if it["verdict_cuoi"] == "OK" else "warn"
    H.append(f'<div class="item"><div><span class="iid">{iid}</span> <span class="lab">{e(pid)}</span></div><div class="grid2">'
             f'<div><b>Tiếng Việt (bản gốc)</b><div class="text">{e(v["state"])}</div><p><b>Câu hỏi (mẫu đề xuất, chưa duyệt):</b> {e(v["question"])}</p>'
             f'{opts_html(v["options"], v["target"])}</div>'
             f'<div><b>Tiếng Anh (bản dịch nguyên câu của bộ gốc)</b>{opts_html(en_opts, v["target"])}</div></div>'
             f'<p class="{cls}"><b>Soát Opus:</b> {e(it["verdict_cuoi"])}'
             + (f' · đáp án theo Opus: {LET[it["dap_an_dung_theo_ban"]]}' if isinstance(it["dap_an_dung_theo_ban"], int) else "")
             + f' — {e(it["giai_thich"])}</p>{fix}</div>')
    sheet.append({"item_id": iid, "nguon": "gpiqa_lo1", "pair_id": pid, "dap_an_vi": v["options"][v["target"]], "ghi_chu_may": f"Opus: {it['verdict_cuoi']}",
                  "verdict": "", "ghi_chu": ""})
H.append("</main></body></html>")
page = "\n".join(H)
assert page.count("<script") == 0 and "http://" not in page.replace('http-equiv', '') and "src=" not in page
(ROOT / "results/review").mkdir(exist_ok=True)
(ROOT / "results/review/pilot2_review.html").write_text(page, encoding="utf-8")
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=list(sheet[0].keys()))
w.writeheader(); w.writerows(sheet)
(ROOT / "results/review/pilot2_review_sheet.csv").write_text("﻿" + buf.getvalue(), encoding="utf-8")
print(json.dumps({"mlqa": len(m_pick), "gpiqa": len(g_pick), "html_bytes": len(page.encode()), "sha256": hashlib.sha256(page.encode()).hexdigest()[:16]}))
