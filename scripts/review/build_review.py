"""Bản đọc kiểm benchmark cho người: results/review/bench_v1_review.html (tự chứa) + results/review/review_sheet.csv (UTF-8 BOM).

Chỉ đọc dev_v1 (không mở test 1.0), config/ và preds đã lưu; không ghi gì vào data/bench_v1 hay config/. Không cần GPU.
Mọi văn bản lấy từ dữ liệu/config được html.escape trước khi chèn. Không JS, không tài nguyên ngoài (thu gọn bằng <details>).
Phần 3: 15 state mỗi nguồn, rng = random.Random(f"review|<nguồn>|{SEED}"), SEED = 20261012 (common.SEED).
Phần 4: dòng tiếng Việt có dấu (variant orig) mà decider-2b (ánh xạ text, results/dev_v1/eval/decider-2b-text) và adapter đối chứng
  (Qwen3-1.7B full_vocab, results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1) cùng chọn MỘT đáp án sai; xếp theo trung bình xác suất hai model
  gán cho đáp án sai đó, giảm dần; tối đa 10 mỗi nguồn.
"""
import csv
import html
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, LETTERS, RESULTS, SEED, load_json, read_jsonl, sha256_file  # noqa: E402

OUT = RESULTS / "review"
OUT.mkdir(parents=True, exist_ok=True)
DEV = DATA / "bench_v1" / "dev_v1.jsonl"
man = load_json(DATA / "bench_v1" / "manifest_dev_v1.json")
assert sha256_file(DEV) == man["sha256"]["dev_v1.jsonl"]
rows = read_jsonl(DEV)
byid = {r["id"]: r for r in rows}
Q = load_json(CONFIG / "bench_v1_questions.json")
TR = load_json(CONFIG / "translations.json")
FMT = load_json(CONFIG / "train_format.json")
P_DEC = {x["id"]: x for x in read_jsonl(RESULTS / "dev_v1/eval/decider-2b-text/preds.jsonl")}
P_CTL = {x["id"]: x for x in read_jsonl(RESULTS / "dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl")}
SOURCES = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc", "vihsd"]
NAME = {"belebele": "Belebele", "sib200": "SIB-200", "massive": "MASSIVE", "xcopa": "XCOPA", "vifactcheck": "ViFactCheck",
        "viwikifc": "ViWikiFC", "vihsd": "ViHSD"}
PARALLEL = {"belebele", "sib200", "massive", "xcopa"}
e = lambda s: html.escape("" if s is None else str(s), quote=True)

# ---------------------------------------------------------------- nhãn gốc theo lựa chọn
TOPIC_BY_TEXT = {lang: {v[lang]: k for k, v in TR["sib200_topics"].items()} for lang in ("en", "vi")}
INTENT_BY_TEXT = {lang: {v[lang]: k for k, v in TR["massive_intents"].items()} for lang in ("en", "vi")}
FC_OPT = {v: k for k, v in Q["factcheck_choice"]["options"].items()}          # chữ lựa chọn -> nhãn chuẩn hóa
FC_ORIG = {"vifactcheck": {"supported": "labels = 0", "refuted": "labels = 1", "nei": "labels = 2"},
           "viwikifc": {"supported": "gold_label = Supports", "refuted": "gold_label = Refutes", "nei": "gold_label = Not_Enough_Information"}}
HSD_ORIG = {"clean": "label_id = 0 (CLEAN)", "offensive": "label_id = 1 (OFFENSIVE)", "hate": "label_id = 2 (HATE)"}
HSD_LEVEL = dict(zip(Q["vihsd_score"]["levels"], ["clean", "offensive", "hate"]))


def option_label(r, j):
    """Giá trị nhãn gốc ứng với lựa chọn j của dòng r (None nếu nguồn không có tập nhãn cố định / không lưu)."""
    o, s, k = r["options"][j], r["source"], r["kind"]
    if s == "sib200":
        return "category = " + TOPIC_BY_TEXT[r["lang"]].get(o, "?")
    if s == "massive":
        return "intent = " + INTENT_BY_TEXT[r["lang"]].get(o, "?")
    if s in ("vifactcheck", "viwikifc"):
        if k == "choice":
            lab = FC_OPT.get(o)
            return f"{lab} ({FC_ORIG[s][lab]})" if lab else None
        return "đúng khi nhãn gốc = supported" if j == 0 else "đúng khi nhãn gốc = refuted hoặc nei"
    if s == "vihsd":
        if k == "score":
            lab = HSD_LEVEL.get(o)
            return HSD_ORIG.get(lab) if lab else None
        return "đúng khi nhãn gốc = offensive hoặc hate" if j == 0 else "đúng khi nhãn gốc = clean"
    return None


def gold_value(r):
    s = r["source"]
    if s in ("belebele", "xcopa"):
        return "không lưu trong dev_v1 (lựa chọn đã xáo; chỉ lưu vị trí đáp án đúng sau xáo)"
    if s in ("vifactcheck", "viwikifc"):
        return f"{r['gold_label']} ({FC_ORIG[s][r['gold_label']]})"
    if s == "vihsd":
        return HSD_ORIG[r["gold_label"]]
    return r["gold_label"]


# ---------------------------------------------------------------- PHẦN 1: tổng quan
def overview(s):
    allr = [r for r in rows if r["source"] == s]
    vi = [r for r in allr if r["lang"] == "vi" and r["variant"] == "orig"]
    en = [r for r in allr if r["lang"] == "en"]
    kinds = Counter(r["kind"] for r in vi)
    nopt = Counter(len(r["options"]) for r in vi)
    if s in ("sib200", "vifactcheck", "viwikifc", "vihsd"):
        dist = Counter(r["gold_label"] for r in vi)
        dist_note = "theo nhãn gốc"
    else:
        dist = Counter(LETTERS[r["target"]] for r in vi)
        dist_note = "theo vị trí đáp án đúng sau xáo (nguồn không có tập nhãn cố định trên mọi câu)"
    # mức đoán theo lớp đa số: trong từng kiểu câu, luôn chọn giá trị phổ biến nhất (nhãn gốc nếu có tập cố định, nếu không thì vị trí)
    hit = 0
    for k in kinds:
        g = [r for r in vi if r["kind"] == k]
        key = (lambda r: r["gold_label"]) if s in ("sib200", "vifactcheck", "viwikifc", "vihsd") else (lambda r: r["target"])
        hit += Counter(key(r) for r in g).most_common(1)[0][1]
    st_tok = statistics.median([r["n_state_tokens"] for r in vi if r.get("n_state_tokens") is not None])
    st_chr = statistics.median([len(r["state"]) for r in vi])
    src_q = {"belebele": "Câu hỏi và lựa chọn: từ nguồn (bản tiếng Việt của Belebele).",
             "sib200": "Câu hỏi và tên chủ đề: do agent viết (config/translations.json, chưa duyệt).",
             "massive": "Câu hỏi và mô tả ý định: do agent viết (config/translations.json, chưa duyệt). Mỗi câu có đáp án đúng + 9 ý định gây nhiễu chọn ngẫu nhiên.",
             "xcopa": "Câu hỏi (nguyên nhân/kết quả): do agent viết (config/translations.json, chưa duyệt). Hai lựa chọn: từ nguồn.",
             "vifactcheck": "Câu hỏi và lựa chọn: do agent viết (config/bench_v1_questions.json, chưa duyệt); nhận định chèn vào câu hỏi: từ nguồn.",
             "viwikifc": "Câu hỏi và lựa chọn: do agent viết (config/bench_v1_questions.json, chưa duyệt); nhận định chèn vào câu hỏi: từ nguồn.",
             "vihsd": "Câu hỏi và lựa chọn: do agent viết (config/bench_v1_questions.json, chưa duyệt)."}[s]
    return {"rows": len(allr), "vi_orig": len(vi), "en": len(en), "states": len({r["state"] for r in vi}),
            "kinds": dict(kinds), "nopt": dict(sorted(nopt.items())), "dist": dist, "dist_note": dist_note,
            "majority": hit / len(vi), "state_tok": st_tok, "state_chr": st_chr, "license": man["licenses"][s],
            "revision": man["revisions"].get({"belebele": "facebook/belebele", "sib200": "Davlan/sib200", "massive": "mteb/amazon_massive_intent",
                                              "xcopa": "cambridgeltl/xcopa", "vifactcheck": "tranthaihoa/vifactcheck",
                                              "viwikifc": "High-Will/ViWikiFC", "vihsd": "uitnlp/vihsd"}[s]),
            "who": "người (PLAN.md mục 8: " + ("khối A1 'Song song, người dịch', nhãn người" if s in PARALLEL else "khối B 'Bản địa, nhãn người'") +
                   "); chi tiết quy trình gán: không rõ trong tài liệu dự án",
            "q_src": src_q}


# ---------------------------------------------------------------- HTML helpers
CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;margin:0;padding:16px;line-height:1.45;color:#1b1b1b;background:#fff}
main{max-width:1200px;margin:0 auto}
h1{font-size:1.5rem}h2{font-size:1.25rem;margin-top:2.2rem;border-bottom:2px solid #ccc;padding-bottom:.2rem}h3{font-size:1.1rem;margin-top:1.6rem}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.9rem}th,td{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}
th{background:#f2f2f2}.mono{font-family:ui-monospace,Consolas,monospace;font-size:.85rem}
.item{border:1px solid #bbb;border-radius:6px;padding:8px 10px;margin:12px 0;background:#fcfcfc}
.iid{font-family:ui-monospace,Consolas,monospace;background:#e8eefc;padding:1px 6px;border-radius:4px;font-weight:600}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:800px){.grid2{grid-template-columns:1fr}}
.text{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;padding:6px;border-radius:4px}
.nod{color:#555;font-size:.88rem}.gold{font-weight:700;background:#e6f6e6}.wrong{background:#fdeaea}
.warn{border:2px solid #c0392b;background:#fff3f2;padding:8px 10px;border-radius:6px;margin:10px 0}
.note{border-left:4px solid #888;background:#f6f6f6;padding:6px 10px;margin:8px 0}
ol.opts{margin:.3rem 0 .3rem 1.4rem;padding:0}ol.opts li{margin:2px 0}
nav.toc a{margin-right:.6rem}.lab{color:#555;font-size:.85rem}
"""


def opts_html(r, highlight=None):
    li = []
    for j, o in enumerate(r["options"]):
        cls = "gold" if j == r["target"] else ("wrong" if highlight is not None and j == highlight else "")
        mark = " ✓ đúng" if j == r["target"] else (" ✗ hai model cùng chọn" if highlight is not None and j == highlight else "")
        lab = option_label(r, j)
        li.append(f'<li class="{cls}">({LETTERS[j]}) {e(o)}{e(mark)}' + (f' <span class="lab">[{e(lab)}]</span>' if lab else "") + "</li>")
    return '<ol class="opts" style="list-style:none">' + "".join(li) + "</ol>"


def question_block(r, highlight=None):
    return (f'<div><b>Câu hỏi ({e(r["lang"])}):</b> <span class="text" style="display:inline">{e(r["question"])}</span></div>'
            + opts_html(r, highlight))


def item_html(iid, vi, extra=""):
    """vi: dòng tiếng Việt có dấu của một câu (pair). Hiện cặp EN (khối song song), bản không dấu 300 ký tự đầu, câu hỏi, lựa chọn."""
    pid = vi["pair_id"]
    en = byid.get(pid + ":en")
    nd = byid.get(vi["id"] + ":nodiac")
    hl = None
    if extra:
        hl = P_DEC[vi["id"]]["pred"]
    head = (f'<div id="{e(iid)}"><span class="iid">{e(iid)}</span> &nbsp;<span class="mono">dev_id: {e(pid)}</span>'
            f' &nbsp;· kiểu câu: {e(vi["kind"])} &nbsp;· đáp án đúng: ({LETTERS[vi["target"]]}) &nbsp;· nhãn gốc: {e(gold_value(vi))}</div>')
    if en is not None:
        body = ('<div class="grid2"><div>' + question_block(en) + '</div><div>' + question_block(vi, hl) + '</div></div>')
    else:
        body = question_block(vi, hl)
    nod = (f'<details class="nod"><summary>Bản không dấu (300 ký tự đầu của state; câu hỏi và lựa chọn cũng bỏ dấu)</summary>'
           f'<div class="text">{e(nd["state"][:300])}{"…" if len(nd["state"]) > 300 else ""}</div>'
           f'<div>{e(nd["question"])}</div><div>{" | ".join(e(o) for o in nd["options"])}</div></details>') if nd else ""
    return f'<div class="item">{head}{extra}{body}{nod}</div>'


def state_html(vi):
    en = byid.get(vi["pair_id"] + ":en")
    if en is not None:
        return (f'<div class="grid2"><div><b>State (EN)</b><div class="text">{e(en["state"])}</div></div>'
                f'<div><b>State (VI, có dấu, đầy đủ)</b><div class="text">{e(vi["state"])}</div></div></div>')
    return f'<div><b>State (VI, có dấu, đầy đủ)</b><div class="text">{e(vi["state"])}</div></div>'


WARN = ('<div class="warn"><b>Cảnh báo nội dung.</b> Phần ViHSD dưới đây hiện nguyên văn bình luận mạng xã hội, có thể chứa lời lẽ '
        'xúc phạm hoặc thù ghét (đó chính là đối tượng gán nhãn của bộ dữ liệu).</div>')
csv_rows = []
H = [f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
     f'<title>Đọc kiểm bench_v1 (dev_v1)</title><style>{CSS}</style></head><body><main>',
     '<h1>Bản đọc kiểm benchmark bench_v1 — tập dev_v1</h1>',
     f'<p class="note">Dựng bằng <span class="mono">scripts/review/build_review.py</span> từ <span class="mono">data/bench_v1/dev_v1.jsonl</span> '
     f'(SHA-256 <span class="mono">{e(man["sha256"]["dev_v1.jsonl"])}</span>, khớp manifest). Không dùng test 1.0. Seed lấy mẫu phần 3: {SEED}. '
     'Mỗi mục cần đọc kiểm có mã riêng (item_id) trùng với cột item_id của <span class="mono">review_sheet.csv</span>. '
     'Mã verdict: OK / SAI_NHAN / CAU_HOI_MO_HO / DICH_SAI / LUA_CHON_LOI / KHAC.</p>']
toc = ['<nav class="toc"><b>Mục lục.</b> <a href="#p1">Phần 1</a><a href="#p2">Phần 2</a>',
       '<br>Phần 3: ' + "".join(f'<a href="#p3-{s}">{NAME[s]}</a>' for s in SOURCES),
       '<br>Phần 4: ' + "".join(f'<a href="#p4-{s}">{NAME[s]}</a>' for s in SOURCES) + '</nav>']
H += toc

# ------------------------------------------------ PHẦN 1
H.append('<h2 id="p1">Phần 1. Tổng quan từng nguồn</h2>')
H.append('<table><tr><th>Nguồn</th><th>Số dòng dev_v1 (mọi biến thể) / dòng VI có dấu / số state VI khác nhau</th><th>Kiểu câu</th>'
         '<th>Số lựa chọn</th><th>Phân bố nhãn (dòng VI có dấu)</th><th>Đoán theo lớp đa số</th><th>Trung vị độ dài state VI</th>'
         '<th>Giấy phép (manifest) · revision</th><th>Nhãn gốc do ai gán</th><th>Câu hỏi và lựa chọn tiếng Việt</th></tr>')
ov = {}
for s in SOURCES:
    o = ov[s] = overview(s)
    dist = ", ".join(f"{e(k)}: {v}" for k, v in sorted(o["dist"].items(), key=lambda x: -x[1])[:12]) + (f" … ({len(o['dist'])} giá trị)" if len(o["dist"]) > 12 else "")
    H.append(f'<tr><td><a href="#p3-{s}">{NAME[s]}</a></td><td>{o["rows"]} / {o["vi_orig"]} / {o["states"]}'
             + (f' (EN: {o["en"]} dòng)' if o["en"] else "") + f'</td><td>{e(o["kinds"])}</td><td>{e(o["nopt"])}</td>'
             f'<td>{dist}<br><span class="lab">{e(o["dist_note"])}</span></td><td>{100 * o["majority"]:.1f}%</td>'
             f'<td>{o["state_tok"]:.0f} token Qwen · {o["state_chr"]:.0f} ký tự</td><td>{e(o["license"])}<br><span class="mono">{e(o["revision"])}</span></td>'
             f'<td>{e(o["who"])}</td><td>{e(o["q_src"])}</td></tr>')
H.append('</table><p class="note">"Đoán theo lớp đa số": trong từng kiểu câu, luôn chọn giá trị phổ biến nhất — nhãn gốc với nguồn có tập nhãn cố định '
         '(SIB-200, ViFactCheck, ViWikiFC, ViHSD), vị trí đáp án sau xáo với nguồn còn lại (Belebele, MASSIVE, XCOPA). '
         'Ba nguồn bản địa được lấy mẫu cân bằng theo nhãn khi dựng bench (build_bench_v1.balanced_sample). '
         'config/bench_v1_questions.json và config/translations.json đều ghi "reviewed": false.</p>')

# ------------------------------------------------ PHẦN 2
H.append('<h2 id="p2">Phần 2. Toàn bộ mẫu câu hỏi và nhãn</h2>')
H.append(f'<p class="note">Nguồn: config/translations.json (trạng thái ghi trong file: "{e(TR["_status"])}") và config/bench_v1_questions.json '
         f'("{e(Q["_status"])}"). Lựa chọn noul tiếng Việt lấy từ config/train_format.json: {e(FMT["noul_options"]["vi"])}.</p>')
H.append('<table><tr><th>item_id</th><th>Nguồn</th><th>Kiểu câu</th><th>Câu hỏi tiếng Việt</th><th>Câu hỏi tiếng Anh</th><th>Ghi chú</th></tr>')


def p2(iid, src, kind, vi, en, note):
    H.append(f'<tr id="{e(iid)}"><td><span class="iid">{e(iid)}</span></td><td>{e(src)}</td><td>{e(kind)}</td><td>{e(vi)}</td><td>{e(en)}</td><td>{e(note)}</td></tr>')
    csv_rows.append([iid, src, 2, "", "", ""])


p2("P2-belebele-q", "Belebele", "reading_mc", "(lấy từ nguồn, mỗi câu một câu hỏi riêng)", "(lấy từ nguồn)", "Không có mẫu; câu hỏi và 4 lựa chọn là bản người dịch của Belebele.")
p2("P2-sib200-q", "SIB-200", "topic", TR["questions"]["sib200"]["vi"], TR["questions"]["sib200"]["en"], "")
p2("P2-massive-q", "MASSIVE", "intent", TR["questions"]["massive"]["vi"], TR["questions"]["massive"]["en"], "")
p2("P2-xcopa-q-cause", "XCOPA", "causal", TR["questions"]["xcopa_cause"]["vi"], TR["questions"]["xcopa_cause"]["en"], "Dùng khi question = cause; hai lựa chọn lấy từ nguồn.")
p2("P2-xcopa-q-effect", "XCOPA", "causal", TR["questions"]["xcopa_effect"]["vi"], TR["questions"]["xcopa_effect"]["en"], "Dùng khi question = effect; hai lựa chọn lấy từ nguồn.")
p2("P2-factcheck-choice-q", "ViFactCheck, ViWikiFC", "choice", Q["factcheck_choice"]["question"], "— (khối bản địa chỉ có tiếng Việt)", "{claim} = nhận định từ nguồn.")
p2("P2-factcheck-noul-q", "ViFactCheck, ViWikiFC", "noul", Q["factcheck_noul"]["question"], "—", f'true_when: {Q["factcheck_noul"]["true_when"]}. {Q["factcheck_noul"]["options_note"]}')
p2("P2-vihsd-score-q", "ViHSD", "score", Q["vihsd_score"]["question"], "—", Q["vihsd_score"]["_note"])
p2("P2-vihsd-noul-q", "ViHSD", "noul", Q["vihsd_noul"]["question"], "—", f'true_when: {Q["vihsd_noul"]["true_when"]}. {Q["vihsd_noul"]["options_note"]}')
H.append('</table>')

H.append('<h3>Lựa chọn ↔ nhãn gốc</h3><table><tr><th>item_id</th><th>Nguồn</th><th>Lựa chọn tiếng Việt</th><th>Lựa chọn tiếng Anh</th><th>Giá trị nhãn gốc trong bộ dữ liệu</th></tr>')


def p2o(iid, src, vi, en, orig):
    H.append(f'<tr id="{e(iid)}"><td><span class="iid">{e(iid)}</span></td><td>{e(src)}</td><td>{e(vi)}</td><td>{e(en)}</td><td>{e(orig)}</td></tr>')
    csv_rows.append([iid, src, 2, "", "", ""])


for k, v in sorted(TR["sib200_topics"].items()):
    p2o(f"P2-sib200-topic-{k}", "SIB-200", v["vi"], v["en"], f"category = {k}")
for k, v in sorted(TR["massive_intents"].items()):
    p2o(f"P2-massive-intent-{k}", "MASSIVE", v["vi"], v["en"], f"intent = {k}")
for lab, txt in Q["factcheck_choice"]["options"].items():
    p2o(f"P2-factcheck-choice-{lab}", "ViFactCheck, ViWikiFC", txt, "—", f"ViFactCheck: {FC_ORIG['vifactcheck'][lab]}; ViWikiFC: {FC_ORIG['viwikifc'][lab]}")
p2o("P2-factcheck-noul-co", "ViFactCheck, ViWikiFC", FMT["noul_options"]["vi"][0], "—", "đúng khi nhãn gốc = supported")
p2o("P2-factcheck-noul-khong", "ViFactCheck, ViWikiFC", FMT["noul_options"]["vi"][1], "—", "đúng khi nhãn gốc = refuted hoặc nei")
for lv, lab in HSD_LEVEL.items():
    p2o(f"P2-vihsd-score-{lab}", "ViHSD", lv, "—", HSD_ORIG[lab])
p2o("P2-vihsd-noul-co", "ViHSD", FMT["noul_options"]["vi"][0], "—", "đúng khi nhãn gốc = OFFENSIVE (1) hoặc HATE (2)")
p2o("P2-vihsd-noul-khong", "ViHSD", FMT["noul_options"]["vi"][1], "—", "đúng khi nhãn gốc = CLEAN (0)")
H.append('</table>')
H.append('<h3>ViHSD: ánh xạ nhãn 0/1/2 và định nghĩa nhãn</h3>'
         '<ul><li>Thẻ dữ liệu <span class="mono">uitnlp/vihsd</span> @ <span class="mono">88e81b36</span> (README): chỉ ghi tên ba nhãn — '
         '"Each has one of three labels: HATE, OFFENSIVE, CLEAN". <b>Không có định nghĩa nhãn và không ghi mã số</b> trong thẻ.</li>'
         '<li>Bài báo gốc: Luu, Nguyen &amp; Nguyen (2021), "A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts", '
         'arXiv:2103.11528 (<span class="mono">https://arxiv.org/abs/2103.11528</span>; bản HTML: <span class="mono">https://ar5iv.labs.arxiv.org/html/2103.11528</span>).'
         '<ul><li>Mã số (mục 3.5 "Dataset Overview"): "labelled as CLEAN (0), OFFENSIVE (1), and HATE (2)".</li>'
         '<li>Định nghĩa (mục 3.2 "Annotation Guidelines", Bảng 1; trích ngắn, xem đầy đủ ở bài báo):'
         '<ul><li>CLEAN: "The comments have no harassment at all."</li>'
         '<li>OFFENSIVE: "The comments contain harassment contents, even profanity words, but do not attack any specific object."</li>'
         '<li>HATE: "The comments have harassment and abusive contents and directly aim at an individual or a group of people …" (câu tiếp tục trong Bảng 1).</li></ul></li></ul></li>'
         '<li>Dự án dùng: <span class="mono">label_id</span> 0 → sạch, 1 → xúc phạm, 2 → thù ghét (scripts/bench/build_bench_v1.py, HSD_LAB), khớp mã số của bài báo.</li></ul>')

# ------------------------------------------------ PHẦN 3
H.append(f'<h2 id="p3">Phần 3. Mẫu ngẫu nhiên — 15 state mỗi nguồn (seed {SEED})</h2>')
H.append(f'<p class="note">Với mỗi nguồn: lấy các state tiếng Việt có dấu khác nhau (theo trường cluster), xếp theo cluster, rồi chọn 15 bằng '
         f'<span class="mono">random.Random("review|&lt;nguồn&gt;|{SEED}").sample</span>. Với mỗi state, hiện mọi câu hỏi gắn với nó. '
         'Khối song song: state và câu hỏi tiếng Anh bên trái, tiếng Việt bên phải. Bản không dấu: 300 ký tự đầu (bấm để mở). '
         'Mỗi câu hỏi là một mục đọc kiểm (một dòng CSV). Phần ViHSD có cảnh báo nội dung.</p>')
n3 = Counter()
for s in SOURCES:
    H.append(f'<h3 id="p3-{s}">Phần 3 · {NAME[s]}</h3>')
    if s == "vihsd":
        H.append(WARN)
    vi = [r for r in rows if r["source"] == s and r["lang"] == "vi" and r["variant"] == "orig"]
    by_cl = defaultdict(list)
    for r in vi:
        by_cl[r["cluster"]].append(r)
    pick = random.Random(f"review|{s}|{SEED}").sample(sorted(by_cl), 15)
    for ci, cl in enumerate(pick, 1):
        qs = sorted(by_cl[cl], key=lambda r: r["id"])
        H.append(f'<div class="item" style="background:#f7f9ff"><b>State {ci}/15</b> <span class="mono">cluster: {e(cl)}</span> · {len(qs)} câu hỏi'
                 + state_html(qs[0]) + "</div>")
        for r in qs:
            n3[s] += 1
            iid = f"P3-{s}-{n3[s]:03d}"
            H.append(item_html(iid, r))
            csv_rows.append([iid, NAME[s], 3, "", "", r["pair_id"]])

# ------------------------------------------------ PHẦN 4
H.append('<h2 id="p4">Phần 4. Mẫu nghi vấn</h2>')
H.append('<div class="warn"><b>Mẫu này được chọn vì model sai, không đại diện cho cả bộ.</b> Chỉ dòng tiếng Việt có dấu (variant orig) mà '
         'decider-2b (ánh xạ "text", preds đã lưu ở results/dev_v1/eval/decider-2b-text) và adapter đối chứng (Qwen3-1.7B full_vocab, '
         'results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1) cùng chọn MỘT đáp án và đáp án đó khác đáp án đúng. Xếp theo trung bình xác suất '
         'hai model gán cho đáp án sai đó, giảm dần; tối đa 10 câu mỗi nguồn. Phần ViHSD có cảnh báo nội dung.</div>')
n4 = Counter(); n4_pool = {}
for s in SOURCES:
    H.append(f'<h3 id="p4-{s}">Phần 4 · {NAME[s]}</h3>')
    if s == "vihsd":
        H.append(WARN)
    cand = []
    for r in rows:
        if r["source"] != s or r["lang"] != "vi" or r["variant"] != "orig":
            continue
        a, b = P_DEC[r["id"]], P_CTL[r["id"]]
        if a["pred"] == b["pred"] != r["target"]:
            cand.append(((a["probs"][a["pred"]] + b["probs"][b["pred"]]) / 2, r))
    n4_pool[s] = len(cand)
    cand.sort(key=lambda x: (-x[0], x[1]["id"]))
    H.append(f'<p>{len(cand)} dòng thỏa điều kiện; hiện {min(10, len(cand))}.</p>')
    for score, r in cand[:10]:
        n4[s] += 1
        iid = f"P4-{s}-{n4[s]:03d}"
        a, b = P_DEC[r["id"]], P_CTL[r["id"]]
        extra = (f'<div class="note">Hai model cùng chọn ({LETTERS[a["pred"]]}). decider-2b: P = {a["probs"][a["pred"]]:.3f} '
                 f'(đáp án đúng {a["probs"][r["target"]]:.3f}); adapter đối chứng: P = {b["probs"][b["pred"]]:.3f} '
                 f'(đáp án đúng {b["probs"][r["target"]]:.3f}). Trung bình dùng để xếp: {score:.3f}.</div>')
        H.append('<div class="item" style="background:#f7f9ff">' + state_html(r) + "</div>")
        H.append(item_html(iid, r, extra))
        csv_rows.append([iid, NAME[s], 4, "", "", r["pair_id"]])

H.append('</main></body></html>')
(OUT / "bench_v1_review.html").write_text("\n".join(H), encoding="utf-8")
with open(OUT / "review_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "nguồn", "phần", "verdict", "ghi_chú", "dev_id"])
    for x in csv_rows:
        assert len(x) == 6, x
        w.writerow(x)
stats = {"part2_items": sum(1 for x in csv_rows if x[2] == 2), "part3_items": dict(n3), "part4_items": dict(n4), "part4_pool": n4_pool,
         "seed": SEED, "dev_sha256": man["sha256"]["dev_v1.jsonl"],
         "html_bytes": (OUT / "bench_v1_review.html").stat().st_size, "csv_rows": len(csv_rows)}
(OUT / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False))
