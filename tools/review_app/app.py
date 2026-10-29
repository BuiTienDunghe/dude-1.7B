"""Công cụ soát nhãn và hiệu đính benchmark (chỉ dev). Server Python thuần thư viện chuẩn, chỉ nghe 127.0.0.1; mở bằng trình duyệt.

Khởi động (tại thư mục dự án, dev_v1.1), trên một dòng:
  python tools/review_app/app.py --dev data/bench_v1_1/dev_v1_1.jsonl
      --preds_a results/dev_v1_1/eval/decider-2b-text/preds.jsonl
      --preds_b results/dev_v1_1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl
      --review_log results/human_review/review_log.jsonl --edits_dir data/human_edits --user nguoi_lam_du_an --seed 20261013 --port 8765
  rồi mở http://127.0.0.1:8765

Màn hình:
  Soát (hàng A/B/C, như trước): Enter = đúng; 1..9 = chọn lại đáp án; M = mơ hồ; D = lệch dịch (khối song song); ← = câu trước.
     Quyết định ghi vào --review_log (định dạng cũ, thêm trường "ai"); mở lại thì tiếp tục câu chưa soát.
  Mẫu và nhãn: mọi chuỗi tiếng Việt do agent viết mà benchmark dùng (tools/review_app/inventory.py); duyệt / sửa tại chỗ / hoàn tác.
  Từng câu: tìm theo từ khóa, nguồn, id; lọc theo trạng thái; sửa đáp án, câu hỏi, từng lựa chọn, state (tiếng Việt có dấu), lý do
     bắt buộc; bản gốc và bản sửa cạnh nhau; hoàn tác từng trường; đánh dấu "mơ hồ nên loại".
  Chờ duyệt: đề xuất của agent đọc từ config/vi_proposals.jsonl: duyệt / sửa rồi duyệt / từ chối. Hai loại:
     - đề xuất chuỗi mẫu (định dạng cũ): duyệt thì áp như một lần sửa ở "Mẫu và nhãn";
     - đề xuất theo câu (loai "cau", split dev|test): thay đúng một chuỗi con của state / câu hỏi / một lựa chọn (gia_tri_cu phải xuất hiện
       đúng 1 lần trong giá trị hiện hành); duyệt thì ghi như một chỉnh sửa "Từng câu" (lý do "lỗi dịch" / "chính tả", có cờ split).
       Sửa đoạn văn áp cho mọi câu cùng split có chung đoạn văn đó. Đề xuất đã xử lý hoàn tác được (theo thứ tự ngược).
     NGOẠI LỆ CÓ CHỦ ĐÍCH với quy tắc "chỉ hiển thị dev": màn này hiện các câu test CÓ đề xuất (nhãn TEST), và chỉ những câu đó;
     các màn khác không hiện test, không có lối xem toàn bộ test. File test (--test) chỉ đọc, phải trùng hash manifest.
  Xuất CSV (UTF-8 có BOM) toàn bộ chỉnh sửa; Chốt đợt: ghi mốc vào log và commit git cục bộ (config/ + data/human_edits/).
Mọi chỉnh sửa nằm ở --edits_dir (edits_log.jsonl chỉ nối thêm + state.json); không sửa file dữ liệu benchmark.

Bảo mật (công cụ GHI được dữ liệu, trình duyệt có thể đang mở trang khác):
  - Chỉ nghe 127.0.0.1. Header Host phải là 127.0.0.1:<port> hoặc localhost:<port>, sai thì 403 (chặn DNS rebinding).
  - Mọi ghi là POST tới /api/*, phải có header X-Review-Token bằng token ngẫu nhiên sinh mỗi lần khởi động (nhúng vào trang), và
    Content-Type application/json; có Origin thì Origin phải là http://127.0.0.1:<port> hoặc http://localhost:<port>. Sai thì 403.
    Không có endpoint ghi nào nhận GET.
  - CSP chặn mọi nguồn ngoài; mọi văn bản dữ liệu được html.escape (phía server) hoặc gán qua textContent (phía trình duyệt).
Không tải tài nguyên ngoài, không cài gì. Ghi chỉ cần JavaScript; trang vẫn đọc được khi tắt JavaScript.
"""
import argparse
import csv
import difflib
import hashlib
import html
import io
import json
import os
import random
import re
import secrets
import subprocess
import sys
import threading
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as E  # noqa: E402
from inventory import FROM_SOURCE_NOTE, GROUPS, inventory  # noqa: E402

for _s in (sys.stdout, sys.stderr):          # console Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
LETTERS = "ABCDEFGHIJ"
NAME = {"belebele": "Belebele", "sib200": "SIB-200", "massive": "MASSIVE", "xcopa": "XCOPA", "vifactcheck": "ViFactCheck",
        "viwikifc": "ViWikiFC", "vihsd": "ViHSD"}
ROW_B, ROW_C = ("xcopa", "massive", "sib200"), ("belebele", "viwikifc", "vifactcheck")
DECISIONS = {"dung": "đáp án chuẩn đúng", "sai": "đáp án chuẩn sai — chọn lại", "mo_ho": "mơ hồ, không có một đáp án duy nhất",
             "dich_lech": "bản dịch lệch nghĩa so với tiếng Anh"}
e = lambda s: html.escape("" if s is None else str(s), quote=True)

ap = argparse.ArgumentParser()
ap.add_argument("--dev", required=True)
ap.add_argument("--preds_a", required=True)
ap.add_argument("--preds_b", required=True)
ap.add_argument("--name_a", default="decider-2b")
ap.add_argument("--name_b", default="adapter đối chứng")
ap.add_argument("--review_log", "--log", dest="review_log", default="results/human_review/review_log.jsonl")
ap.add_argument("--edits_dir", default="data/human_edits")
ap.add_argument("--proposals", default="config/vi_proposals.jsonl")
ap.add_argument("--test", default="data/bench_v1_1/test_v1.1.jsonl", help="chỉ dùng để hiện các câu test có đề xuất ở màn Chờ duyệt")
ap.add_argument("--second_opinion", default="results/translation_audit/second_opinion/second_opinion.jsonl",
                help="ý kiến thứ hai (Opus) cho đề xuất; chỉ hiển thị, không đổi logic duyệt")
ap.add_argument("--dung_sai", default="results/bench_v1_2_trial/dung_sai_items.json", help="danh sách câu dev đúng→sai chỉ vì sửa chữ (màn /dung_sai)")
ap.add_argument("--config_dir", default="config")
ap.add_argument("--repo", default=".", help="thư mục git cục bộ để commit khi Chốt đợt")
ap.add_argument("--no_git", action="store_true")
ap.add_argument("--user", default="nguoi_lam_du_an", help="giá trị trường 'ai' trong log (không suy ra từ hệ thống)")
ap.add_argument("--seed", type=int, default=20261013)
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--dry_run", action="store_true")
args = ap.parse_args()
if "test" in Path(args.dev).name.lower():
    sys.exit("Công cụ chỉ hiển thị dev; --dev trỏ vào một file test.")


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


rows = read_jsonl(args.dev)
byid = {r["id"]: r for r in rows}
PA = {x["id"]: x for x in read_jsonl(args.preds_a)}
PB = {x["id"]: x for x in read_jsonl(args.preds_b)}
vi = [r for r in rows if r["lang"] == "vi" and r["variant"] == "orig"]
VI_BY_PID = {r["pair_id"]: r for r in vi}
if any(r["id"] not in PA or r["id"] not in PB for r in vi):
    sys.exit("preds thiếu dòng tiếng Việt có dấu của --dev; kiểm tra đường dẫn preds.")
DEV_GROUPS = {}                     # đoạn văn gốc -> các câu dùng chung (Belebele, ViFactCheck, ViWikiFC)
for r in vi:
    DEV_GROUPS.setdefault(r["state"], []).append(r["pair_id"])

# ---------------------------------------------------------------- ba hàng A/B/C (giữ nguyên cách dựng)
A = []
for r in vi:
    a, b = PA[r["id"]], PB[r["id"]]
    if a["pred"] == b["pred"] != r["target"]:
        A.append(((a["probs"][a["pred"]] + b["probs"][b["pred"]]) / 2, r))
A.sort(key=lambda x: (-x[0], x[1]["id"]))
in_a = {r["id"] for _, r in A}
Bq = [r for r in vi if r["source"] in ROW_B and r["id"] not in in_a]
per = {s: [r for r in vi if r["source"] == s and r["id"] not in in_a] for s in ROW_C}
for s in ROW_C:
    random.Random(f"C|{s}|{args.seed}").shuffle(per[s])
Cq = []
while any(per.values()):
    for s in ROW_C:
        if per[s]:
            Cq.append(per[s].pop(0))
QUEUE = [("A", r) for _, r in A] + [("B", r) for r in Bq] + [("C", r) for r in Cq]
CNT = Counter(h for h, _ in QUEUE)
ITEM_ID = {}
for h in "ABC":
    n = 0
    for hh, r in QUEUE:
        if hh == h:
            n += 1
            ITEM_ID[r["pair_id"]] = f"{h}-{n:04d}"
DEV_SHA = sha256(args.dev)
has_sensitive = any(r["source"] == "vihsd" for r in vi)
INV = inventory(args.config_dir)
INV_BY_KEY = {x["key"]: x for x in INV}
EDITS_LOG = Path(args.edits_dir) / "edits_log.jsonl"

# ---------------------------------------------------------------- test: chỉ để hiện câu test CÓ đề xuất ở màn Chờ duyệt
TEST_VI, TEST_EN, TEST_GROUPS, TEST_SHA, TEST_ERR = {}, {}, {}, "", ""
if not Path(args.test).exists():
    TEST_ERR = f"không có file test {args.test}"
elif Path(args.test).resolve() == Path(args.dev).resolve():
    sys.exit("--test trùng --dev.")
else:
    TEST_SHA = sha256(args.test)
    for mp in sorted(Path(args.test).parent.glob("manifest_test*.json")):
        want = json.loads(mp.read_text(encoding="utf-8")).get("sha256", {}).get(Path(args.test).name)
        if want and want != TEST_SHA:
            TEST_ERR = f"file test khác hash trong {mp.name}; không hiện câu test"
    if not TEST_ERR:
        for r in read_jsonl(args.test):
            if r["lang"] == "en":
                TEST_EN[r["pair_id"]] = r
            elif r["variant"] == "orig":
                TEST_VI[r["pair_id"]] = r
                TEST_GROUPS.setdefault(r["state"], []).append(r["pair_id"])

STATE = E.derive_state(E.read_log(EDITS_LOG), INV)
if not args.dry_run:
    Path(args.review_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.edits_dir).mkdir(parents=True, exist_ok=True)
TOKEN = secrets.token_urlsafe(32)
HOSTS = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
ORIGINS = {f"http://{h}" for h in HOSTS}
LOCK = threading.Lock()


def load_review():
    latest = {}
    if os.path.exists(args.review_log):
        for x in read_jsonl(args.review_log):
            if x.get("dev_sha256") in (None, DEV_SHA):
                latest[x["dev_id"]] = x
    return latest


DECIDED = load_review()
if not args.dry_run:
    E.write_state(args.edits_dir, STATE)


def refresh_state():
    global STATE
    STATE = E.derive_state(E.read_log(EDITS_LOG), INV)
    E.write_state(args.edits_dir, STATE)


def append_review(rec):
    with open(args.review_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
    DECIDED[rec["dev_id"]] = rec


def first_undecided(start=0):
    for i in list(range(start, len(QUEUE))) + list(range(0, start)):
        if QUEUE[i][1]["pair_id"] not in DECIDED:
            return i
    return None


# ---------------------------------------------------------------- hiển thị chung
CSS = """body{font-family:system-ui,"Segoe UI",Roboto,Arial,sans-serif;margin:0;padding:10px 16px;color:#1b1b1b;background:#fff;line-height:1.45}
nav{display:flex;flex-wrap:wrap;gap:10px;align-items:center;border-bottom:1px solid #ccc;padding-bottom:6px;margin-bottom:8px}nav a{font-weight:600}
.bar{background:#eee;border-radius:4px;height:8px;width:140px;display:inline-block;vertical-align:middle}.bar i{display:block;height:8px;background:#4a7;border-radius:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.text{white-space:pre-wrap;word-break:break-word;border:1px solid #ccc;border-radius:4px;padding:8px;background:#fcfcfc;max-height:46vh;overflow:auto}
mark{background:#fff3b0}.opt{padding:3px 6px;border-radius:4px;margin:2px 0}.gold{background:#d9f2d9;font-weight:700;border:1px solid #7b7}
.models{background:#fdeaea;border:1px solid #e99;padding:4px 8px;border-radius:4px;display:inline-block}.iid{font-family:Consolas,monospace;background:#e8eefc;padding:1px 6px;border-radius:4px}
.dec{background:#f3f3f3;border-radius:6px;padding:8px;margin-top:10px}button{margin:2px;padding:5px 9px;font-size:.92rem}.warn{border:2px solid #c0392b;background:#fff3f2;padding:6px 10px;border-radius:6px}
.done{background:#eef7ff;border:1px solid #9bd;padding:4px 8px;border-radius:4px}.small{font-size:.85rem;color:#555}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}th{background:#f2f2f2}
.st-chua{background:#fff6d6}.st-da{background:#e6f6e6}textarea,input[type=text]{width:100%;box-sizing:border-box;font:inherit}
.chg{background:#fdf0d5}#msg{position:fixed;bottom:8px;right:8px;background:#333;color:#fff;padding:6px 10px;border-radius:6px;display:none}
.card{border:1px solid #bbb;border-radius:6px;padding:8px 10px;margin:12px 0}.card.t{border:2px solid #c0392b;background:#fffafa}.card .text{position:relative}
.so{border-left:4px solid #8a6d3b;background:#fbf6ec;padding:4px 8px;margin:6px 0;font-size:.9rem}.badge{font-weight:700;padding:1px 7px;border-radius:4px;color:#fff}.badge.test{background:#c0392b}.badge.dev{background:#2c6fbb}
del{background:#f9d0d0;text-decoration:line-through}ins{background:#c9f0c9;text-decoration:none;font-weight:600}"""

JS = """<script>
var TOKEN=%s;
function post(url,obj){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':TOKEN},body:JSON.stringify(obj),credentials:'same-origin'})
 .then(function(r){return r.json().then(function(j){if(!r.ok||!j.ok){throw new Error(j.error||('HTTP '+r.status));}return j;});})
 .catch(function(err){var m=document.getElementById('msg');m.textContent='Lỗi: '+err.message;m.style.display='block';throw err;});}
function val(id){var x=document.getElementById(id);return x?x.value:'';}
</script>"""


def page(title, body, extra_js=""):
    nav = ('<nav><a href="/">Trang đầu</a><a href="/review">Soát A/B/C</a><a href="/templates">Mẫu và nhãn</a><a href="/items">Từng câu</a>'
           '<a href="/proposals">Chờ duyệt</a><a href="/dung_sai">Đúng→sai</a><a href="/export.csv">Xuất CSV</a><a href="/milestone">Chốt đợt</a>'
           f'<span class="small">người dùng: {e(args.user)} · dev {e(DEV_SHA[:12])}…</span></nav>')
    warn = ('<p class="warn">Cảnh báo nội dung: dữ liệu có văn bản mạng xã hội có thể xúc phạm.</p>' if has_sensitive else "")
    nojs = '<noscript><p class="warn">Trang đọc được khi tắt JavaScript, nhưng mọi thao tác ghi cần JavaScript.</p></noscript>'
    return (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>{e(title)}</title><style>{CSS}</style></head><body>{nav}{nojs}{warn}{body}'
            f'<div id="msg"></div>{JS % json.dumps(TOKEN)}{extra_js}</body></html>')


def progress_html():
    done_by_row, done_by_src, tot_src = Counter(), Counter(), Counter()
    for h, r in QUEUE:
        tot_src[r["source"]] += 1
        if r["pair_id"] in DECIDED:
            done_by_row[h] += 1; done_by_src[r["source"]] += 1
    bar = lambda d, t: f'<span class="bar"><i style="width:{0 if not t else 100 * d / t:.1f}%"></i></span> {d}/{t}'
    return ('<div>' + " · ".join(f"Hàng {h}: {bar(done_by_row[h], CNT[h])}" for h in "ABC" if CNT[h]) + '</div><div class="small">' +
            " · ".join(f"{e(NAME.get(s, s))}: {done_by_src[s]}/{t}" for s, t in sorted(tot_src.items())) + "</div>")


WORD = re.compile(r"\w+", re.UNICODE)


def claim_of(r):
    for key in ("Nhận định: ", "Phát biểu: "):
        if key in r["question"]:
            return r["question"].split(key, 1)[1]
    return ""


def state_html(r, text=None):
    st = r["state"] if text is None else text
    if r["source"] not in ("vifactcheck", "viwikifc"):
        return e(st)
    cw = {w.lower() for w in WORD.findall(claim_of(r))}
    parts = re.split(r"(?<=[.!?…])(\s+)", st)
    sents = [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    scores = [len({w.lower() for w in WORD.findall(s)} & cw) for s, _ in sents]
    top = [t for t in sorted(set(scores), reverse=True) if t > 0][:2]
    return "".join(f'<mark title="trùng {sc} từ với nhận định">{e(s)}</mark>{e(sep)}' if sc in top else e(s) + e(sep) for (s, sep), sc in zip(sents, scores))


def options_html(r, gold=None):
    gold = r["target"] if gold is None else gold
    return "".join(f'<div class="opt{" gold" if j == gold else ""}"><b>{j + 1} · ({LETTERS[j]})</b> {e(o)}{" ✓ đáp án chuẩn" if j == gold else ""}</div>'
                   for j, o in enumerate(r["options"]))


# ---------------------------------------------------------------- trang đầu
def page_home():
    st = STATE["dem_mau"]
    pend = E.pending_proposals(load_props(), STATE)
    kinds = Counter(p.get("split") if p.get("loai") == "cau" else "mẫu" for p in pend)
    body = (f'<h1>Soát nhãn và hiệu đính — dev</h1>{progress_html()}'
            f'<ul><li><a href="/review">Soát hàng A/B/C</a> — câu chưa soát tiếp theo. Seed hàng C: {args.seed}.</li>'
            f'<li><a href="/templates">Mẫu và nhãn</a>: {len(INV)} chuỗi — chưa duyệt {st["chưa duyệt"]}, đã duyệt {st["đã duyệt"]}, đã sửa {st["đã sửa"]}.</li>'
            f'<li><a href="/items">Từng câu</a>: {len(vi)} câu tiếng Việt có dấu; đang có chỉnh sửa ở {len(STATE["cau"])} câu.</li>'
            f'<li><a href="/proposals">Chờ duyệt</a>: {len(pend)} đề xuất (chuỗi mẫu {kinds["mẫu"]}, câu dev {kinds["dev"]}, câu test {kinds["test"]}). '
            f'Câu test đã có chỉnh sửa (qua đề xuất): {len(STATE["cau_test"])}.</li>'
            f'<li>Số đợt đã chốt: {len(STATE["dot"])}. Bản ghi trong log hiệu đính: {STATE["so_ban_ghi"]}.</li></ul>')
    return page("Trang đầu", body)


# ---------------------------------------------------------------- soát A/B/C
def page_item(k):
    h, r = QUEUE[k]
    pid = r["pair_id"]; en = byid.get(pid + ":en"); prev = DECIDED.get(pid)
    head = (f'<div><span class="iid">{e(ITEM_ID[pid])}</span> hàng {h} · câu {k + 1}/{len(QUEUE)} · {e(NAME.get(r["source"], r["source"]))} · '
            f'kiểu câu {e(r["kind"])} · <span class="small">dev_id: {e(pid)}</span> · <a href="/edit?dev_id={urllib.parse.quote(pid)}">sửa câu này</a></div>')
    models = ""
    if h == "A":
        a = PA[r["id"]]
        models = (f'<p class="models">Hai model chọn: <b>({LETTERS[a["pred"]]}) {e(r["options"][a["pred"]])}</b> — {e(args.name_a)} P={a["probs"][a["pred"]]:.2f}, '
                  f'{e(args.name_b)} P={PB[r["id"]]["probs"][PB[r["id"]]["pred"]]:.2f}</p>')
    vi_block = f'<h3>Tiếng Việt (có dấu)</h3><div class="text">{state_html(r)}</div><p><b>Câu hỏi:</b> {e(r["question"])}</p>{options_html(r)}'
    body = (f'<div class="grid"><div><h3>Tiếng Anh</h3><div class="text">{e(en["state"])}</div><p><b>Question:</b> {e(en["question"])}</p>{options_html(en)}</div><div>{vi_block}</div></div>'
            if en is not None else vi_block)
    done = ""
    if prev:
        na = f' → ({LETTERS[prev["dap_an_moi"]]})' if prev.get("dap_an_moi") is not None else ""
        done = f'<p class="done">Đã quyết định: <b>{e(DECISIONS[prev["quyet_dinh"]])}{e(na)}</b> lúc {e(prev["thoi_diem"])}. Quyết định mới ghi thêm một dòng và thay thế.</p>'
    n_opt = len(r["options"])
    nums = "".join(f'<button onclick="send(\'sai:{j}\')" title="phím {j + 1}">{j + 1} · chọn ({LETTERS[j]})</button>' for j in range(n_opt) if j != r["target"])
    btn_d = '<button onclick="send(\'dich_lech\')" title="phím D">D · bản dịch lệch nghĩa</button>' if en is not None else ""
    form = (f'<div class="dec"><button onclick="send(\'dung\')" title="phím Enter" autofocus>Enter · đáp án chuẩn đúng</button> {nums} '
            f'<button onclick="send(\'mo_ho\')" title="phím M">M · mơ hồ</button> {btn_d}<br><label class="small">Ghi chú (tùy chọn; Ctrl+Enter = "đúng"): '
            f'<input type="text" id="note" value="{e(prev.get("ghi_chu", "") if prev else "")}"></label></div>'
            f'<p class="small"><a href="/item?k={max(0, k - 1)}" id="back">← câu trước</a> · <a href="/item?k={min(len(QUEUE) - 1, k + 1)}">câu sau →</a> · '
            f'<a href="/review">câu chưa soát tiếp theo</a></p>')
    js = """<script>
function send(v){post('/api/review',{k:%d,d:v,note:val('note')}).then(function(j){location.href=j.next;});}
document.addEventListener('keydown',function(ev){var inNote=document.activeElement&&document.activeElement.id==='note';
if(inNote){if(ev.key==='Enter'&&ev.ctrlKey){ev.preventDefault();send('dung');}return;}
if(ev.key==='Enter'){ev.preventDefault();send('dung');}
else if(/^[1-9]$/.test(ev.key)){var j=parseInt(ev.key,10)-1;if(j<%d&&j!==%d){ev.preventDefault();send('sai:'+j);}}
else if(ev.key==='m'||ev.key==='M'){ev.preventDefault();send('mo_ho');}
else if((ev.key==='d'||ev.key==='D')&&%s){ev.preventDefault();send('dich_lech');}
else if(ev.key==='ArrowLeft'){ev.preventDefault();location.href=document.getElementById('back').href;}});
</script>""" % (k, n_opt, int(r["target"]), "true" if en is not None else "false")
    return page(f"Soát {ITEM_ID[pid]}", progress_html() + head + models + body + done + form, js)


def api_review(d):
    k = int(d["k"]); dec = str(d.get("d", "")); note = str(d.get("note", "")).strip()
    if not 0 <= k < len(QUEUE):
        raise ValueError("k ngoài phạm vi")
    h, r = QUEUE[k]
    new = None
    if dec.startswith("sai:"):
        new = int(dec.split(":", 1)[1])
        dec, new = ("dung", None) if new == r["target"] else ("sai", new)
        if new is not None and not 0 <= new < len(r["options"]):
            raise ValueError("đáp án ngoài phạm vi")
    if dec not in DECISIONS or (dec == "dich_lech" and byid.get(r["pair_id"] + ":en") is None):
        raise ValueError("quyết định không hợp lệ")
    append_review({"item_id": ITEM_ID[r["pair_id"]], "dev_id": r["pair_id"], "nguon": r["source"], "hang": h, "quyet_dinh": dec, "dap_an_moi": new,
                   "dap_an_chuan": r["target"], "ghi_chu": note, "thoi_diem": E.now(), "ai": args.user, "dev_sha256": DEV_SHA})
    nxt = first_undecided(k + 1)
    return {"next": "/done" if nxt is None else f"/item?k={nxt}"}


# ---------------------------------------------------------------- mẫu và nhãn
def page_templates():
    rows_ = []
    for g in GROUPS:
        rows_.append(f'<tr><th colspan="7">{e(g)}</th></tr>')
        for x in INV:
            if x["group"] != g:
                continue
            m = STATE["mau"][x["key"]]; kid = hashlib.sha1(x["key"].encode()).hexdigest()[:10]
            cls = "st-chua" if m["trang_thai"] == "chưa duyệt" else "st-da"
            k_js = e(json.dumps(x["key"]))
            rows_.append(f'<tr class="{cls}"><td class="small">{e(x["key"])}<br>{e(x["used_in"])}</td><td>{e(x["en"])}</td><td class="small">{e(x["label"])}</td>'
                         f'<td><textarea id="v{kid}" rows="2">{e(m["gia_tri"])}</textarea>'
                         + (f'<div class="small">gốc: {e(m["goc"])}</div>' if m["gia_tri"] != m["goc"] else "") +
                         f'</td><td><b>{e(m["trang_thai"])}</b>' + (f'<div class="small">{e(m.get("ai"))} · {e(m.get("cap_nhat"))}</div>' if m.get("cap_nhat") else "") +
                         f'</td><td><input type="text" id="n{kid}" placeholder="lý do / ghi chú (tùy chọn)"></td>'
                         f'<td><button onclick="tp({k_js},\'duyet\',\'{kid}\')">Duyệt</button>'
                         f'<button onclick="tp({k_js},\'sua\',\'{kid}\')">Lưu sửa</button>'
                         f'<button onclick="tp({k_js},\'hoan_tac\',\'{kid}\')">Hoàn tác</button></td></tr>')
    st = STATE["dem_mau"]
    body = (f'<h1>Mẫu và nhãn</h1><p>{len(INV)} chuỗi tiếng Việt do agent viết — chưa duyệt <b>{st["chưa duyệt"]}</b>, đã duyệt {st["đã duyệt"]}, đã sửa {st["đã sửa"]}. '
            f'Trạng thái gốc lấy từ config ("_status": đã duyệt ngày 2026-10-13); mọi thay đổi ghi vào lớp ghi đè, không sửa config.</p>'
            f'<p class="small">{e(FROM_SOURCE_NOTE)}</p><table><tr><th>Khóa · dùng ở</th><th>Tiếng Anh</th><th>Nhãn gốc</th><th>Tiếng Việt (hiện hành)</th>'
            f'<th>Trạng thái</th><th>Ghi chú</th><th></th></tr>{"".join(rows_)}</table>')
    js = """<script>function tp(key,act,kid){post('/api/template',{key:key,action:act,value:val('v'+kid),note:val('n'+kid)}).then(function(){location.reload();});}</script>"""
    return page("Mẫu và nhãn", body, js)


def api_template(d):
    key, act = str(d["key"]), str(d["action"])
    if key not in INV_BY_KEY or act not in ("duyet", "sua", "hoan_tac"):
        raise ValueError("khóa hoặc thao tác không hợp lệ")
    cur = STATE["mau"][key]
    rec = {"ai": args.user, "loai": "mau", "khoa": key, "hanh_dong": act, "gia_tri_cu": cur["gia_tri"], "gia_tri_moi": cur["gia_tri"], "ly_do": "",
           "ghi_chu": str(d.get("note", ""))}
    if act == "sua":
        v = str(d.get("value", ""))
        if not v.strip():
            raise ValueError("giá trị rỗng")
        rec["gia_tri_moi"] = v
    if act == "hoan_tac":
        rec["gia_tri_moi"] = cur["goc"]
    E.append(EDITS_LOG, rec); refresh_state()
    return {}


# ---------------------------------------------------------------- từng câu
def item_status(pid):
    s = set()
    c = STATE["cau"].get(pid, {})
    rv = DECIDED.get(pid)
    if c:
        s.add("đã sửa")
    if rv and rv["quyet_dinh"] == "mo_ho" or "loai_bo" in c:
        s.add("mơ hồ")
    if rv and rv["quyet_dinh"] == "dich_lech":
        s.add("lệch dịch")
    if not rv and not c:
        s.add("chưa xem")
    return s


def page_items(qs):
    q = qs.get("q", [""])[0].strip().lower(); src = qs.get("source", [""])[0]; stf = qs.get("status", [""])[0]; pg = int(qs.get("page", ["0"])[0] or 0)
    res = []
    for r in vi:
        pid = r["pair_id"]
        if src and r["source"] != src:
            continue
        if stf and stf not in item_status(pid):
            continue
        if q and q not in pid.lower() and q not in r["question"].lower() and q not in r["state"].lower() and not any(q in o.lower() for o in r["options"]):
            continue
        res.append(r)
    show = res[pg * 50:(pg + 1) * 50]
    opts_src = "".join(f'<option value="{e(s)}"{" selected" if s == src else ""}>{e(NAME[s])}</option>' for s in sorted({r["source"] for r in vi}))
    opts_st = "".join(f'<option{" selected" if s == stf else ""}>{e(s)}</option>' for s in ("chưa xem", "đã sửa", "mơ hồ", "lệch dịch"))
    form = (f'<form method="get" action="/items"><input type="text" name="q" value="{e(q)}" placeholder="từ khóa hoặc id" style="width:40%">'
            f'<select name="source"><option value="">mọi nguồn</option>{opts_src}</select><select name="status"><option value="">mọi trạng thái</option>{opts_st}</select>'
            f'<button>Tìm</button></form>')
    lst = "".join(f'<tr><td><a href="/edit?dev_id={urllib.parse.quote(r["pair_id"])}">{e(r["pair_id"])}</a><br><span class="small">{e(ITEM_ID.get(r["pair_id"], ""))}</span></td>'
                  f'<td>{e(NAME[r["source"]])} · {e(r["kind"])}</td><td>{e(", ".join(sorted(item_status(r["pair_id"]))))}</td><td>{e(r["question"][:160])}</td></tr>' for r in show)
    pages = "".join(f'<a href="/items?{e(urllib.parse.urlencode({"q": q, "source": src, "status": stf, "page": p}))}">{p + 1}</a> ' for p in range((len(res) + 49) // 50))
    body = f'<h1>Từng câu</h1>{form}<p>{len(res)} câu. Trang: {pages}</p><table><tr><th>dev_id</th><th>Nguồn</th><th>Trạng thái</th><th>Câu hỏi</th></tr>{lst}</table>'
    return page("Từng câu", body)


def base_val(r, field):
    if field == "dap_an":
        return r["target"]
    if field == "cau_hoi":
        return r["question"]
    if field == "state":
        return r["state"]
    if field.startswith("lua_chon:"):
        return r["options"][int(field.split(":")[1])]
    return False   # loai_bo


def field_cur(r, pid, field, split="dev"):
    c = (STATE["cau"] if split == "dev" else STATE["cau_test"]).get(pid, {})
    return c[field]["gia_tri_moi"] if field in c else base_val(r, field)


def page_edit(pid):
    r = VI_BY_PID.get(pid)
    if r is None:
        return page("Không có", "<p>Không có câu này trong dev (chỉ câu tiếng Việt có dấu).</p>")
    en = byid.get(pid + ":en"); c = STATE["cau"].get(pid, {})
    reasons = "".join(f"<option>{e(x)}</option>" for x in E.REASONS)
    sel = f'<select id="reason"><option value="">— chọn lý do (bắt buộc) —</option>{reasons}</select>'
    rv = DECIDED.get(pid)
    rv_html = ""
    if rv:
        na = f' → ({LETTERS[rv["dap_an_moi"]]})' if rv.get("dap_an_moi") is not None else ""
        rv_html = f'<p class="done">Quyết định soát (hàng {e(rv["hang"])}): <b>{e(DECISIONS[rv["quyet_dinh"]])}{e(na)}</b>{(" — " + e(rv["ghi_chu"])) if rv.get("ghi_chu") else ""}</p>'

    def fld(field, label, orig, cur, multiline):
        fid = re.sub(r"[^a-z0-9]", "_", field)
        changed = field in c
        inp = (f'<textarea id="f_{fid}" rows="{8 if field == "state" else 2}">{e(cur)}</textarea>' if multiline else f'<input type="text" id="f_{fid}" value="{e(cur)}">')
        meta = f'<div class="small">sửa: {e(c[field]["ly_do"])} · {e(c[field]["ai"])} · {e(c[field]["thoi_diem"])}</div>' if changed else ""
        f_js = e(json.dumps(field))
        return (f'<tr class="{"chg" if changed else ""}"><th>{e(label)}</th><td><div class="text">{e(orig)}</div></td><td>{inp}{meta}</td>'
                f'<td><button onclick="sv({f_js},\'f_{fid}\')">Lưu</button>' + (f'<button onclick="ud({f_js})">Hoàn tác</button>' if changed else "") + "</td></tr>")

    gold_cur = field_cur(r, pid, "dap_an")
    gold_sel = "".join(f'<option value="{j}"{" selected" if j == gold_cur else ""}>({LETTERS[j]}) {e(o[:80])}</option>' for j, o in enumerate(r["options"]))
    trs = [f'<tr class="{"chg" if "dap_an" in c else ""}"><th>Đáp án đúng (áp cho cả ba biến thể)</th><td>({LETTERS[r["target"]]}) {e(r["options"][r["target"]])}</td>'
           f'<td><select id="f_dap_an">{gold_sel}</select></td><td><button onclick="sv(\'dap_an\',\'f_dap_an\',true)">Lưu</button>'
           + ('<button onclick="ud(\'dap_an\')">Hoàn tác</button>' if "dap_an" in c else "") + "</td></tr>",
           fld("cau_hoi", "Câu hỏi (VI)", r["question"], field_cur(r, pid, "cau_hoi"), True)]
    trs += [fld(f"lua_chon:{j}", f"Lựa chọn ({LETTERS[j]}) (VI)", o, field_cur(r, pid, f"lua_chon:{j}"), False) for j, o in enumerate(r["options"])]
    trs.append(fld("state", "State (VI, có dấu)", r["state"], field_cur(r, pid, "state"), True))
    excl = "loai_bo" in c
    en_html = (f'<details open><summary>Bản tiếng Anh (không sửa ở đây; sửa chữ tiếng Việt thì bản tiếng Anh giữ nguyên)</summary><div class="text">{e(en["state"])}</div>'
               f'<p><b>Question:</b> {e(en["question"])}</p>{options_html(en)}</details>') if en is not None else ""
    apply_btn = (f'<button onclick="document.getElementById(\'f_dap_an\').value=\'{int(rv["dap_an_moi"])}\';document.getElementById(\'reason\').value=\'sai nhãn\';">'
                 f'Điền sẵn đáp án theo quyết định soát</button>') if rv and rv.get("dap_an_moi") is not None else ""
    ex_html = ('<b>Câu này đang bị đánh dấu loại khỏi bản hiệu đính.</b> <button onclick="ud(\'loai_bo\')">Bỏ đánh dấu loại</button>' if excl
               else '<button onclick="ex()">Đánh dấu mơ hồ nên loại</button>')
    body = (f'<h1><span class="iid">{e(ITEM_ID.get(pid, ""))}</span> {e(pid)}</h1><p>{e(NAME[r["source"]])} · kiểu câu {e(r["kind"])}</p>{rv_html}{en_html}'
            f'<p>Lý do: {sel} Ghi chú: <input type="text" id="note" style="width:40%"> {apply_btn}</p>'
            f'<table><tr><th>Trường</th><th>Bản gốc (dev_v1.1)</th><th>Bản sửa (hiện hành)</th><th></th></tr>{"".join(trs)}</table><p>{ex_html}</p>'
            f'<h3>Xem trước state (bản sửa)</h3><div class="text">{state_html(r, field_cur(r, pid, "state"))}</div>')
    js = """<script>var PID=%s;
function sv(field,id,isint){var r=val('reason');if(!r){var m=document.getElementById('msg');m.textContent='Chọn lý do trước khi lưu';m.style.display='block';return;}
 var v=val(id);post('/api/item',{dev_id:PID,field:field,action:'sua',value:isint?parseInt(v,10):v,reason:r,note:val('note')}).then(function(){location.reload();});}
function ud(field){post('/api/item',{dev_id:PID,field:field,action:'hoan_tac',value:null,reason:'',note:val('note')}).then(function(){location.reload();});}
function ex(){post('/api/item',{dev_id:PID,field:'loai_bo',action:'sua',value:true,reason:'mơ hồ nên loại',note:val('note')}).then(function(){location.reload();});}
</script>""" % json.dumps(pid).replace("</", "<\\/")
    return page(f"Sửa {pid}", body, js)


def api_item(d):
    pid, field, act = str(d["dev_id"]), str(d["field"]), str(d["action"])
    r = VI_BY_PID.get(pid)
    if r is None or act not in ("sua", "hoan_tac"):
        raise ValueError("câu hoặc thao tác không hợp lệ")
    valid = {"dap_an", "cau_hoi", "state", "loai_bo"} | {f"lua_chon:{j}" for j in range(len(r["options"]))}
    if field not in valid:
        raise ValueError("trường không hợp lệ")
    rec = {"ai": args.user, "loai": "cau", "split": "dev", "dev_id": pid, "nguon": r["source"], "truong": field, "hanh_dong": act, "gia_tri_cu": field_cur(r, pid, field),
           "gia_tri_moi": None, "ly_do": str(d.get("reason", "")), "ghi_chu": str(d.get("note", ""))}
    if act == "sua":
        if rec["ly_do"] not in E.REASONS:
            raise ValueError("phải chọn lý do")
        v = d.get("value")
        if field == "dap_an":
            v = int(v)
            if not 0 <= v < len(r["options"]):
                raise ValueError("đáp án ngoài phạm vi")
        elif field == "loai_bo":
            v = True
        else:
            v = str(v)
            if not v.strip():
                raise ValueError("giá trị rỗng")
        rec["gia_tri_moi"] = v
    elif field not in STATE["cau"].get(pid, {}):
        raise ValueError("trường này chưa có chỉnh sửa")
    E.append(EDITS_LOG, rec); refresh_state()
    return {}


# ---------------------------------------------------------------- chờ duyệt
PAGE_N = 20


def load_props():
    return E.read_proposals(args.proposals) if Path(args.proposals).exists() else []


SO_LABEL = {"DONG_Y": "đồng ý", "DONG_Y_NHUNG_CO_CHUOI_TOT_HON": "đồng ý, nhưng có chuỗi tốt hơn", "BAC_GIU_NGUYEN": "bác, giữ nguyên bản gốc",
            "CAN_NGUOI_XEM": "cần người xem"}


def load_so():
    p = Path(args.second_opinion)
    return {x["id"]: x for x in E.read_log(p)} if p.exists() else {}


def so_html(so, pid):
    """Ý kiến thứ hai: CHỈ ĐỌC, không ảnh hưởng thao tác duyệt."""
    x = so.get(pid)
    if not x:
        return ""
    alt = f' · chuỗi thay thế gợi ý: <ins>{e(x.get("chuoi_thay_the"))}</ins>' if x.get("chuoi_thay_the") else ""
    gold = f' · đáp án sau sửa: {e(x.get("gold_sau_sua"))}' if x.get("gold_sau_sua") not in (None, "", "DUNG_DUY_NHAT", "KHONG_AP_DUNG") else ""
    moi = f' · lỗi mới: {e(x.get("loi_moi"))}' if x.get("loi_moi") else ""
    return (f'<p class="so"><b>Ý kiến thứ hai (Opus, chỉ để tham khảo): {e(SO_LABEL.get(x.get("ket_luan"), x.get("ket_luan")))}</b>'
            f' — {e(x.get("ly_do"))}{alt}{gold}{moi}</p>')


def rows_of(split):
    return (VI_BY_PID, DEV_GROUPS) if split == "dev" else (TEST_VI, TEST_GROUPS)


def en_of(split, pid):
    return byid.get(pid + ":en") if split == "dev" else TEST_EN.get(pid)


def check_cau(p):
    """Kiểm một đề xuất theo câu -> (lỗi cấu trúc, lỗi nội dung, ngữ cảnh). Lỗi cấu trúc: thẻ không hiện văn bản nào của câu."""
    split = p.get("split")
    if split not in ("dev", "test"):
        return "split phải là dev hoặc test", "", None
    if split == "test" and TEST_ERR:
        return TEST_ERR, "", None
    vi_by, groups = rows_of(split)
    pid = str(p.get("pair_id", ""))
    r = vi_by.get(pid)
    if r is None:
        return f"không có pair_id này trong {split}", "", None
    if p.get("row_id") != r["id"]:
        return "row_id không khớp dòng tiếng Việt có dấu của pair_id", "", None
    if p.get("nguon") != r["source"]:
        return "nguon không khớp", "", None
    t = str(p.get("truong", ""))
    m = re.fullmatch(r"option:(\d+)", t)
    if t not in ("state", "question") and not (m and int(m.group(1)) < len(r["options"])):
        return "truong phải là state, question hoặc option:<k> có thật", "", None
    if p.get("ly_do") not in E.PROPOSAL_REASON:
        return "ly_do phải là loi_dich hoặc chinh_ta", "", None
    if not isinstance(p.get("gia_tri_cu"), str) or not isinstance(p.get("gia_tri_de_xuat"), str):
        return "gia_tri_cu và gia_tri_de_xuat phải là chuỗi", "", None
    field = E.PROPOSAL_FIELD(t)
    cur, old = field_cur(r, pid, field, split), p["gia_tri_cu"]
    ctx = {"split": split, "r": r, "pid": pid, "field": field, "cur": cur, "old": old, "i": cur.find(old) if old else -1, "targets": [pid], "skipped": []}
    if field == "state":                 # đoạn văn dùng chung: áp cho mọi câu cùng split có cùng đoạn văn hiện hành
        same = groups[r["state"]]
        ctx["targets"] = [q for q in same if field_cur(vi_by[q], q, "state", split) == cur]
        ctx["skipped"] = [q for q in same if q not in ctx["targets"]]
    n = E.occurrences(cur, old)
    return "", ("" if n == 1 else f"gia_tri_cu xuất hiện {n} lần trong giá trị hiện hành của trường (cần đúng 1 lần)"), ctx


def changed_html(text, i, old, new):
    return f'{e(text[:i])}<del>{e(old)}</del><ins>{e(new)}</ins>{e(text[i + len(old):])}'


def hl(text, sub):
    i = text.find(sub) if sub else -1
    return e(text) if i < 0 else f'{e(text[:i])}<mark>{e(sub)}</mark>{e(text[i + len(sub):])}'


def badge(split):
    return {"test": '<span class="badge test">TEST</span>', "dev": '<span class="badge dev">DEV</span>'}.get(split, "")


def card_cau(p, so=None):
    cid = "k" + hashlib.sha1(str(p["id"]).encode()).hexdigest()[:12]
    split = p.get("split")
    err, err2, ctx = check_cau(p)
    id_js = e(json.dumps(p["id"]))
    link = f' · <a href="/edit?dev_id={urllib.parse.quote(ctx["pid"])}">mở ở Từng câu</a>' if ctx and split == "dev" else ""
    head = (f'<div>{badge(split)} <b>{e(p["id"])}</b> · {e(NAME.get(p.get("nguon"), p.get("nguon")))} · <span class="small">{e(p.get("pair_id"))}</span> · '
            f'trường <b>{e(p.get("truong"))}</b> · {e(E.PROPOSAL_REASON.get(p.get("ly_do"), p.get("ly_do")))} · '
            f'<span class="small">{e(p.get("nguoi_de_xuat"))} · {e(p.get("thoi_diem"))}</span>{link}</div>')
    note = f'<input type="text" id="pn{cid}" placeholder="ghi chú (tùy chọn)" style="width:40%">'
    btn_tc = f'<button onclick="pc({id_js},\'tu_choi\',\'{cid}\')">Từ chối</button>'
    cls = "card t" if split == "test" else "card"
    if err:
        return f'<div class="{cls}">{head}{so_html(so or {}, p["id"])}<p class="warn">Đề xuất không hợp lệ: {e(err)}. Không hiện văn bản.</p><p>{note} {btn_tc}</p></div>'
    r, pid, new, old, i = ctx["r"], ctx["pid"], p["gia_tri_de_xuat"], ctx["old"], ctx["i"]
    en, gold, ev = en_of(split, pid), r["target"], p.get("bang_chung_en") or ""

    def vi_val(field):
        v = field_cur(r, pid, field, split)
        return changed_html(v, i, old, new) if field == ctx["field"] and not err2 else e(v)

    vi_opts = "".join(f'<div class="opt{" gold" if j == gold else ""}"><b>({LETTERS[j]})</b> {vi_val(f"lua_chon:{j}")}{" ✓ đáp án chuẩn" if j == gold else ""}</div>'
                      for j in range(len(r["options"])))
    vi_html = (f'<div><h4>Tiếng Việt hiện hành <span class="small">(<del>cũ</del> <ins>đề xuất</ins>)</span></h4><div class="text">{vi_val("state")}</div>'
               f'<p><b>Câu hỏi:</b> {vi_val("cau_hoi")}</p>{vi_opts}</div>')
    en_html = ""
    if en is not None:
        en_opts = "".join(f'<div class="opt{" gold" if j == en["target"] else ""}"><b>({LETTERS[j]})</b> {hl(o, ev)}{" ✓" if j == en["target"] else ""}</div>'
                          for j, o in enumerate(en["options"]))
        en_html = f'<div><h4>Tiếng Anh <span class="small">(<mark>bằng chứng</mark>)</span></h4><div class="text">{hl(en["state"], ev)}</div><p><b>Question:</b> {hl(en["question"], ev)}</p>{en_opts}</div>'
    gold_line = (f'<p><b>Đáp án chuẩn:</b> ({LETTERS[gold]}) {e(field_cur(r, pid, f"lua_chon:{gold}", split))}'
                 + (f' — EN: {e(en["options"][gold])}' if en is not None else "") + "</p>")
    if ctx["field"].startswith("lua_chon:"):
        k = int(ctx["field"].split(":")[1])
        gold_line += ('<p class="warn">Đề xuất sửa chính lựa chọn là đáp án chuẩn.</p>' if k == gold
                      else f'<p class="small">Đề xuất sửa lựa chọn ({LETTERS[k]}), không phải đáp án chuẩn.</p>')
    info = (f'<p><b>Đổi:</b> <del>{e(old)}</del> → <ins>{e(new)}</ins></p><p><b>Bằng chứng EN:</b> {e(ev) if ev else "—"} · <b>Giải thích:</b> {e(p.get("giai_thich"))}')
    for k_, lab in (("doi_dap_an", "agent cho là đổi đáp án"), ("do_tin_cay", "độ tin cậy")):
        if k_ in p:
            info += f' · {lab}: {e(p[k_])}'
    info += "</p>"
    if ctx["field"] == "state" and len(ctx["targets"]) > 1:
        info += f'<p class="small">Đoạn văn dùng chung: áp cho {len(ctx["targets"])} câu ({e(", ".join(ctx["targets"]))}).</p>'
    if ctx["skipped"]:
        info += f'<p class="small">Không áp cho {len(ctx["skipped"])} câu cùng đoạn văn gốc vì đoạn văn của chúng đã được sửa riêng: {e(", ".join(ctx["skipped"]))}.</p>'
    if err2:
        acts = f'<p class="warn">{e(err2)}. Chỉ từ chối được.</p><p>{note} {btn_tc}</p>'
    else:
        acts = (f'<p><label>Đoạn đề xuất (sửa ở đây rồi bấm "Sửa rồi duyệt"): <textarea id="p{cid}" rows="2">{e(new)}</textarea></label></p>'
                f'<p>{note} <button onclick="pc({id_js},\'duyet\',\'{cid}\')">Duyệt</button>'
                f'<button onclick="pc({id_js},\'sua_roi_duyet\',\'{cid}\')">Sửa rồi duyệt</button>{btn_tc}</p>')
    body = f'<div class="grid">{en_html}{vi_html}</div>' if en_html else vi_html
    return f'<div class="{cls}">{head}{so_html(so or {}, p["id"])}{info}{body}{gold_line}{acts}</div>'


def processed_html(props):
    trs = []
    for p in props:
        dx = STATE["de_xuat"].get(p["id"])
        if dx is None:
            continue
        id_js = e(json.dumps(p["id"]))
        if p.get("loai") == "cau":
            n_ap = len(dx.get("ap", []))
            trs.append(f'<tr><td>{badge(p.get("split"))} {e(p["id"])}</td><td class="small">{e(p.get("pair_id"))} · {e(p.get("truong"))}</td>'
                       f'<td>{e(dx["hanh_dong"])}{f" · áp {n_ap} câu" if n_ap else ""}</td>'
                       f'<td class="small"><del>{e(str(p.get("gia_tri_cu"))[:120])}</del> → <ins>{e(str(dx.get("gia_tri_moi") or "—")[:120])}</ins></td>'
                       f'<td class="small">{e(dx.get("ai"))} · {e(dx["thoi_diem"])}</td><td><button onclick="pu({id_js})">Hoàn tác</button></td></tr>')
        else:
            trs.append(f'<tr><td>{e(p["id"])}</td><td class="small">{e(p.get("khoa"))}</td><td>{e(dx["hanh_dong"])}</td><td class="small">{e(dx.get("gia_tri_moi") or "—")}</td>'
                       f'<td class="small">{e(dx.get("ai"))} · {e(dx["thoi_diem"])}</td><td class="small">hoàn tác ở "Mẫu và nhãn"</td></tr>')
    return (f'<details><summary>Đã xử lý ({len(trs)})</summary><p class="small">Hoàn tác một đề xuất theo câu đưa các trường nó đã sửa về giá trị trước khi duyệt '
            f'và đưa đề xuất về "đang chờ"; chỉ làm được khi các trường đó chưa bị sửa tiếp sau đó.</p>'
            f'<table><tr><th>id</th><th>câu · trường</th><th>xử lý</th><th>đổi</th><th>ai · lúc</th><th></th></tr>{"".join(trs)}</table></details>')


def page_proposals(qs):
    xem, src = qs.get("xem", [""])[0], qs.get("nguon", [""])[0]
    pg = max(0, int(qs.get("page", ["0"])[0] or 0))
    props = load_props()
    so = load_so()
    pend = E.pending_proposals(props, STATE)
    pend_mau = [p for p in pend if p.get("loai") != "cau"]
    cau_all = [p for p in pend if p.get("loai") == "cau"]
    cnt = Counter(p.get("split") for p in cau_all)
    pend_cau = [p for p in cau_all if (not xem or p.get("split") == xem) and (not src or p.get("nguon") == src)] if xem != "mau" else []
    sel_x = "".join(f'<option value="{v}"{" selected" if v == xem else ""}>{lab}</option>'
                    for v, lab in (("", "mọi loại"), ("mau", "chuỗi mẫu"), ("dev", "câu dev"), ("test", "câu test")))
    sel_s = "".join(f'<option value="{e(s)}"{" selected" if s == src else ""}>{e(NAME.get(s, s))}</option>' for s in sorted({str(p.get("nguon")) for p in cau_all}))
    form = (f'<form method="get" action="/proposals"><select name="xem">{sel_x}</select><select name="nguon"><option value="">mọi nguồn</option>{sel_s}</select>'
            f'<button>Lọc</button></form>')
    pages = "".join(f'<a href="/proposals?{e(urllib.parse.urlencode({"xem": xem, "nguon": src, "page": k}))}">{k + 1}</a> '
                    for k in range((len(pend_cau) + PAGE_N - 1) // PAGE_N))
    rows_ = []
    for p in pend_mau if xem in ("", "mau") else []:
        pid = re.sub(r"[^a-zA-Z0-9]", "_", str(p["id"]))
        cur = STATE["mau"].get(p.get("khoa"), {}).get("gia_tri")
        id_js = e(json.dumps(p["id"]))
        rows_.append(f'<tr><td class="small">{e(p["id"])}<br>{e(p.get("khoa"))}</td><td>{e(cur) if cur is not None else "<i>(chuỗi mới, chưa có trong config)</i>"}</td>'
                     f'<td><textarea id="p{pid}" rows="2">{e(p.get("gia_tri_de_xuat"))}</textarea><div class="small">EN: {e(p.get("ban_tieng_anh"))} · lý do: {e(p.get("ly_do"))}'
                     f' · {e(p.get("nguoi_de_xuat"))} · {e(p.get("thoi_diem"))}</div>{so_html(so, p["id"])}</td><td><input type="text" id="pn{pid}" placeholder="ghi chú"></td>'
                     f'<td><button onclick="pp({id_js},\'duyet\',\'{pid}\')">Duyệt</button><button onclick="pp({id_js},\'sua_roi_duyet\',\'{pid}\')">Sửa rồi duyệt</button>'
                     f'<button onclick="pp({id_js},\'tu_choi\',\'{pid}\')">Từ chối</button></td></tr>')
    test_note = (f'Câu test chỉ hiện ở màn này, chỉ những câu có đề xuất (ngoại lệ có chủ đích; thẻ viền đỏ, nhãn TEST). File test chỉ đọc '
                 f'({e(TEST_SHA[:12])}…); duyệt chỉ ghi vào lớp ghi đè, test chỉ đổi khi dựng bản 1.2 theo yêu cầu.' if not TEST_ERR else f'Câu test: {e(TEST_ERR)}.')
    mau_tbl = (f'<h2>Chuỗi mẫu ({len(pend_mau)})</h2><table><tr><th>id · khóa</th><th>Giá trị hiện hành</th><th>Đề xuất</th><th>Ghi chú</th><th></th></tr>{"".join(rows_)}</table>'
               if rows_ else "")
    cards = "".join(card_cau(p, so) for p in pend_cau[pg * PAGE_N:(pg + 1) * PAGE_N])
    body = (f'<h1>Chờ duyệt</h1><p>Đọc từ <span class="iid">{e(args.proposals)}</span>: {len(props)} đề xuất, {len(pend)} đang chờ '
            f'(chuỗi mẫu {len(pend_mau)}, câu dev {cnt["dev"]}, câu test {cnt["test"]}), {len(props) - len(pend)} đã xử lý.</p>'
            f'<p class="small">{test_note}</p>{form}{mau_tbl}'
            + (f'<h2>Theo câu ({len(pend_cau)} theo bộ lọc)</h2><p>Trang: {pages}</p>{cards}' if xem != "mau" else "")
            + processed_html(props))
    js = """<script>function pp(id,act,pid){post('/api/proposal',{id:id,action:act,value:val('p'+pid),note:val('pn'+pid)}).then(function(){location.reload();});}
function pc(id,act,cid){post('/api/proposal',{id:id,action:act,value:val('p'+cid),note:val('pn'+cid)}).then(function(){location.reload();});}
function pu(id){if(!confirm('Hoàn tác đề xuất '+id+'?')){return;}post('/api/proposal',{id:id,action:'hoan_tac',value:'',note:''}).then(function(){location.reload();});}
document.querySelectorAll('.card .text').forEach(function(t){var x=t.querySelector('ins,del,mark');if(x){t.scrollTop=Math.max(0,x.offsetTop-40);}});
</script>"""
    return page("Chờ duyệt", body, js)


def api_proposal_cau(p, act, d):
    note = str(d.get("note", ""))
    if act == "hoan_tac":
        return undo_cau(p, note)
    if act not in ("duyet", "sua_roi_duyet", "tu_choi") or p["id"] in STATE["de_xuat"]:
        raise ValueError("đề xuất đã xử lý hoặc thao tác không hợp lệ")
    err, err2, ctx = check_cau(p)
    hdr = {"ai": args.user, "loai": "de_xuat", "de_xuat_id": p["id"], "split": p.get("split"), "pair_id": p.get("pair_id"), "truong": p.get("truong"),
           "hanh_dong": act, "ghi_chu": note}
    if act == "tu_choi":
        E.append(EDITS_LOG, {**hdr, "gia_tri_moi": None})
        refresh_state()
        return {}
    if err or err2:
        raise ValueError(err or err2)
    v = str(d.get("value", "")) if act == "sua_roi_duyet" else p["gia_tri_de_xuat"]
    if v == ctx["old"]:
        raise ValueError("đoạn đề xuất giống đoạn cũ")
    split, field, i, old = ctx["split"], ctx["field"], ctx["i"], ctx["old"]
    vi_by, _ = rows_of(split)
    E.append(EDITS_LOG, {**hdr, "gia_tri_moi": v})
    for q in ctx["targets"]:
        r = vi_by[q]
        cur = field_cur(r, q, field, split)
        rec = {"ai": args.user, "loai": "cau", "split": split, "pair_id": q, "nguon": r["source"], "truong": field, "hanh_dong": "sua", "gia_tri_cu": cur,
               "gia_tri_moi": cur[:i] + v + cur[i + len(old):], "ly_do": E.PROPOSAL_REASON[p["ly_do"]], "ghi_chu": note, "de_xuat_id": p["id"],
               "thay_chuoi_con": {"cu": old, "moi": v}}
        if split == "dev":
            rec["dev_id"] = q
        E.append(EDITS_LOG, rec)
    refresh_state()
    return {"ap": len(ctx["targets"])}


def undo_cau(p, note):
    dx = STATE["de_xuat"].get(p["id"])
    if dx is None:
        raise ValueError("đề xuất chưa được xử lý")
    split, aps = dx.get("split"), dx.get("ap", [])
    vi_by = rows_of(split)[0] if split in ("dev", "test") else {}
    for a in aps:
        r = vi_by.get(a["pair_id"])
        if r is None or field_cur(r, a["pair_id"], a["truong"], split) != a["sau"]:
            raise ValueError(f"{a['pair_id']}: trường {a['truong']} đã đổi sau đề xuất này; hoàn tác các thay đổi sau trước")
    for a in aps:
        r = vi_by[a["pair_id"]]
        back_to_base = a["truoc"] == base_val(r, a["truong"])
        rec = {"ai": args.user, "loai": "cau", "split": split, "pair_id": a["pair_id"], "nguon": r["source"], "truong": a["truong"],
               "hanh_dong": "hoan_tac" if back_to_base else "sua", "gia_tri_cu": a["sau"], "gia_tri_moi": None if back_to_base else a["truoc"],
               "ly_do": E.PROPOSAL_REASON.get(p.get("ly_do"), "khác"), "ghi_chu": f"hoàn tác đề xuất {p['id']}. {note}".strip(), "hoan_tac_de_xuat": p["id"]}
        if split == "dev":
            rec["dev_id"] = a["pair_id"]
        E.append(EDITS_LOG, rec)
    E.append(EDITS_LOG, {"ai": args.user, "loai": "de_xuat", "de_xuat_id": p["id"], "split": split, "pair_id": p.get("pair_id"), "truong": p.get("truong"),
                         "hanh_dong": "hoan_tac", "ghi_chu": note})
    refresh_state()
    return {}


def api_proposal(d):
    props = {p["id"]: p for p in load_props()}
    pid, act = str(d["id"]), str(d["action"])
    if pid not in props:
        raise ValueError("không có đề xuất này")
    if props[pid].get("loai") == "cau":
        return api_proposal_cau(props[pid], act, d)
    if act not in ("duyet", "sua_roi_duyet", "tu_choi") or pid in STATE["de_xuat"]:
        raise ValueError("đề xuất hoặc thao tác không hợp lệ")
    p = props[pid]
    v = str(d.get("value", "")) if act == "sua_roi_duyet" else p.get("gia_tri_de_xuat")
    if act != "tu_choi" and not str(v).strip():
        raise ValueError("giá trị rỗng")
    E.append(EDITS_LOG, {"ai": args.user, "loai": "de_xuat", "de_xuat_id": pid, "khoa": p.get("khoa"), "hanh_dong": act,
                         "gia_tri_moi": None if act == "tu_choi" else v, "ghi_chu": str(d.get("note", ""))})
    if act != "tu_choi" and p.get("khoa") in INV_BY_KEY:     # đề xuất cho chuỗi đã có: áp như một lần sửa ở "Mẫu và nhãn"
        E.append(EDITS_LOG, {"ai": args.user, "loai": "mau", "khoa": p["khoa"], "hanh_dong": "sua", "gia_tri_cu": STATE["mau"][p["khoa"]]["gia_tri"],
                             "gia_tri_moi": v, "ly_do": f"duyệt đề xuất {pid}", "ghi_chu": str(d.get("note", ""))})
    refresh_state()
    return {}


# ---------------------------------------------------------------- đúng → sai (mục B, chỉ dev)
def load_dung_sai():
    p = Path(args.dung_sai)
    return {x["pair_id"]: x for x in json.load(open(p, encoding="utf-8"))["items"]} if p.exists() else {}


def word_diff(a, b):
    """-> (html cũ, html mới): so theo từ, chỗ khác bôi <del>/<ins>; mọi đoạn đều escape."""
    ta, tb = re.findall(r"\S+|\s+", a), re.findall(r"\S+|\s+", b)
    oa, ob = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        sa, sb = e("".join(ta[i1:i2])), e("".join(tb[j1:j2]))
        if op == "equal":
            oa.append(sa); ob.append(sb)
        else:
            oa.append(f"<del>{sa}</del>" if sa else ""); ob.append(f"<ins>{sb}</ins>" if sb else "")
    return "".join(oa), "".join(ob)


def dung_sai_plan(pid, order=None):
    """Kế hoạch "hoàn tác về bản cũ" cho cặp dev pid: gỡ mọi đề xuất đã áp lên cặp, mới nhất trước (thứ tự trong log). Mô phỏng trước cả chồng
    hoàn tác: một trường (của pid hay câu cùng đoạn) có sửa khác chồng lên thì báo lỗi, không ghi gì. -> (ids đề xuất, mọi cặp bị đổi)."""
    if order is None:
        order = {r["de_xuat_id"]: k for k, r in enumerate(E.read_log(EDITS_LOG)) if r.get("loai") == "de_xuat"}
    manual = [f for f, v in STATE["cau"].get(pid, {}).items() if f not in ("loai_bo", "dap_an") and not v.get("de_xuat_id")]
    if manual:
        raise ValueError(f"{pid}: có sửa tay không qua đề xuất ({', '.join(manual)}); hoàn tác ở màn Từng câu")
    dx = sorted(((i, v) for i, v in STATE["de_xuat"].items() if v.get("split") == "dev" and any(a["pair_id"] == pid for a in v.get("ap", []))),
                key=lambda t: order.get(t[0], -1), reverse=True)
    if not dx:
        raise ValueError(f"{pid}: không còn đề xuất đã áp để hoàn tác")
    cur, pairs = {}, set()
    for i, v in dx:
        for a in v["ap"]:
            k = (a["pair_id"], a["truong"])
            if k not in cur:
                r = VI_BY_PID.get(a["pair_id"])
                if r is None:
                    raise ValueError(f"{a['pair_id']}: không có trong dev")
                cur[k] = field_cur(r, a["pair_id"], a["truong"])
            if cur[k] != a["sau"]:
                raise ValueError(f"{a['pair_id']} · {a['truong']}: có sửa khác chồng lên sau đề xuất {i}; không tự ghi đè, hoàn tác sửa đó trước")
            cur[k] = a["truoc"]; pairs.add(a["pair_id"])
    return [i for i, _ in dx], sorted(pairs)


def has_applied(pid):
    return any(v.get("split") == "dev" and any(a["pair_id"] == pid for a in v.get("ap", [])) for v in STATE["de_xuat"].values())


def page_dung_sai():
    items = load_dung_sai()
    order = {r["de_xuat_id"]: k for k, r in enumerate(E.read_log(EDITS_LOG)) if r.get("loai") == "de_xuat"}
    pulled = {q: p for p, d in STATE["dung_sai"].items() if d["hanh_dong"] == "hoan_tac" for q in d.get("keo_theo", [])}
    cards = []
    for pid, it in items.items():
        cid = "d" + hashlib.sha1(pid.encode("utf-8")).hexdigest()[:10]
        dec = STATE["dung_sai"].get(pid)
        try:
            ids, pairs = dung_sai_plan(pid, order)
            others = [q for q in pairs if q != pid]
            plan, can_undo = (f"Hoàn tác sẽ gỡ đề xuất {', '.join(ids)}" + (f"; KÉO THEO {len(others)} câu cùng đoạn văn: {', '.join(others)}" if others else "")), True
        except ValueError as ex:
            plan, can_undo = f"Không hoàn tác được: {ex}", False
        trs = []
        for f, lab in (("state", "Văn bản"), ("question", "Câu hỏi")):
            o, n = word_diff(it["vi_cu"][f], it["vi_moi"][f])
            trs.append(f'<tr class="{"chg" if o != n else ""}"><th>{lab}</th><td><div class="text">{o}</div></td><td><div class="text">{n}</div></td></tr>')
        for j, (oo, nn) in enumerate(zip(it["vi_cu"]["options"], it["vi_moi"]["options"])):
            o, n = word_diff(oo, nn)
            trs.append(f'<tr class="{"chg" if o != n else ""}"><th>({LETTERS[j]}){" ✓ đáp án" if j == it["dap_an"] else ""}</th><td>{o}</td><td>{n}</td></tr>')
        en = it.get("en")
        en_html = (f'<details open><summary>Bản tiếng Anh</summary><div class="text">{e(en["state"])}</div><p><b>Question:</b> {e(en["question"])}</p>'
                   f'{options_html({"options": en["options"], "target": en["dap_an"]})}</details>' if en else '<p class="small">Nguồn tiếng Việt gốc, không có bản tiếng Anh.</p>')
        pr = "".join(f'<tr><td>{e(m)}{"" if v["tinh_vao_dem"] else " <span class=small>(tham khảo, không tính)</span>"}</td>'
                     f'<td>({LETTERS[v["cu"]]}) {"đúng" if v["dung_cu"] else "sai"}</td>'
                     f'<td class="{"chg" if v["dung_cu"] and not v["dung_moi"] else ""}">({LETTERS[v["moi"]]}) {"đúng" if v["dung_moi"] else "sai"}</td></tr>'
                     for m, v in it["du_doan"].items())
        st_ = ""
        if dec:
            st_ = (f'<p class="done">Đã quyết: <b>{"giữ bản mới" if dec["hanh_dong"] == "giu_moi" else "hoàn tác về bản cũ"}</b> · {e(dec["thoi_diem"])}'
                   + (f' · đã gỡ {e(", ".join(dec["de_xuat"]))}' if dec.get("de_xuat") else "")
                   + (f' · kéo theo {e(", ".join(dec["keo_theo"]))}' if dec.get("keo_theo") else "") + (f' · {e(dec["ghi_chu"])}' if dec.get("ghi_chu") else "") + "</p>")
        if pid in pulled and not (dec and dec["hanh_dong"] == "hoan_tac"):
            st_ += f'<p class="done">Đã về bản cũ do hoàn tác câu cùng đoạn {e(pulled[pid])}.</p>'
        pid_js = e(json.dumps(pid))
        btn = ("" if (dec and dec["hanh_dong"] == "hoan_tac") or not has_applied(pid) else f'<button onclick="ds({pid_js},\'giu_moi\',\'{cid}\')">Giữ bản mới</button>') + \
              (f'<button onclick="ds({pid_js},\'hoan_tac\',\'{cid}\')">Hoàn tác về bản cũ</button>' if can_undo else "")
        cards.append(f'<div class="card" id="{cid}"><div><span class="badge dev">DEV</span> <b>{e(pid)}</b> · {e(NAME.get(it["nguon"], it["nguon"]))} · '
                     f'nhóm <b>{"chính" if it["nhom"] == "chinh" else "phụ"}</b> · {it["so_model_dung_sang_sai"]} model đúng → sai ({e(", ".join(it["model_dung_sang_sai"]))}) · '
                     f'trường đã sửa: {e(", ".join(it["truong_da_sua"]))}</div>{st_}'
                     f'<table><tr><th></th><th>Bản cũ (1.1)</th><th>Bản mới (1.2 thử)</th></tr>{"".join(trs)}</table>{en_html}'
                     f'<table><tr><th>Model</th><th>Dự đoán ở 1.1</th><th>Dự đoán ở 1.2 thử</th></tr>{pr}</table>'
                     f'<p class="small">Đề xuất đã áp: {e(", ".join(it["de_xuat_da_ap"]) or "—")}. {e(plan)}</p>'
                     f'<p><input type="text" id="n{cid}" placeholder="ghi chú (không bắt buộc)" style="width:45%"> {btn}</p></div>')
    n_dec = sum(p in STATE["dung_sai"] for p in items)
    body = (f'<h1>Đúng → sai chỉ vì sửa chữ (dev)</h1><p>{len(items)} câu dev ({sum(x["nhom"] == "chinh" for x in items.values())} nhóm chính: từ 2 model trở lên '
            f'chuyển đúng → sai; còn lại nhóm phụ: 1 model). Đã quyết {n_dec}/{len(items)}. Đọc từ <span class="iid">{e(args.dung_sai)}</span>.</p>'
            '<p class="small">"Giữ bản mới" chỉ ghi quyết định. "Hoàn tác về bản cũ" gỡ đúng các đề xuất đã áp lên câu (mới nhất trước), đánh dấu các đề xuất đó là '
            'từ chối; đoạn văn dùng chung thì mọi câu cùng đoạn cũng về bản cũ. Có sửa khác chồng lên thì báo lỗi, không ghi gì. Chuỗi mẫu không đổi ở màn này.</p>'
            + "".join(cards))
    js = """<script>function ds(pid,act,cid){if(act==='hoan_tac'&&!confirm('Hoàn tác về bản cũ cho '+pid+'? Câu cùng đoạn văn (nếu có) cũng về bản cũ.')){return;}
post('/api/dung_sai',{pair_id:pid,action:act,note:val('n'+cid)}).then(function(j){if(j.keo_theo&&j.keo_theo.length){alert('Đã hoàn tác. Kéo theo '+j.keo_theo.length+' câu cùng đoạn: '+j.keo_theo.join(', '));}location.reload();});}
document.querySelectorAll('.card .text').forEach(function(t){var x=t.querySelector('ins,del');if(x){t.scrollTop=Math.max(0,x.offsetTop-40);}});
</script>"""
    return page("Đúng → sai", body, js)


def api_dung_sai(d):
    pid, act, note = str(d["pair_id"]), str(d["action"]), str(d.get("note", ""))
    if pid not in load_dung_sai() or pid not in VI_BY_PID or act not in ("giu_moi", "hoan_tac"):
        raise ValueError("câu hoặc thao tác không hợp lệ")
    if act == "giu_moi":
        if STATE["dung_sai"].get(pid, {}).get("hanh_dong") == "hoan_tac":
            raise ValueError("câu đã hoàn tác về bản cũ; muốn dùng lại bản mới thì hoàn tác quyết định 'từ chối' của các đề xuất ở Chờ duyệt rồi duyệt lại")
        if not has_applied(pid):
            raise ValueError("câu này không còn đề xuất đã áp (đã về bản cũ, có thể do hoàn tác câu cùng đoạn); không ghi 'giữ bản mới'")
        E.append(EDITS_LOG, {"ai": args.user, "loai": "dung_sai", "pair_id": pid, "hanh_dong": "giu_moi", "ghi_chu": note})
        refresh_state()
        return {"keo_theo": []}
    ids, pairs = dung_sai_plan(pid)            # kiểm cả chồng trước khi ghi
    props = {p["id"]: p for p in load_props()}
    miss = [i for i in ids if i not in props]
    if miss:
        raise ValueError(f"không thấy đề xuất {', '.join(miss)} trong {args.proposals}")
    for i in ids:
        undo_cau(props[i], f"hoàn tác ở màn đúng→sai ({pid}). {note}".strip())
        E.append(EDITS_LOG, {"ai": args.user, "loai": "de_xuat", "de_xuat_id": i, "split": "dev", "pair_id": props[i].get("pair_id"), "truong": props[i].get("truong"),
                             "hanh_dong": "tu_choi", "gia_tri_moi": None, "ghi_chu": f"từ chối sau khi hoàn tác ở màn đúng→sai ({pid}). {note}".strip()})
        refresh_state()
    others = [q for q in pairs if q != pid]
    E.append(EDITS_LOG, {"ai": args.user, "loai": "dung_sai", "pair_id": pid, "hanh_dong": "hoan_tac", "ghi_chu": note, "de_xuat": ids, "keo_theo": others})
    refresh_state()
    return {"de_xuat": ids, "keo_theo": others}


# ---------------------------------------------------------------- xuất CSV, chốt đợt
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["thoi_diem", "ai", "loai", "hanh_dong", "khoa", "dev_id", "nguon", "truong", "gia_tri_cu", "gia_tri_moi", "ly_do", "ghi_chu", "de_xuat_id", "dot", "id",
            "split", "pair_id", "doan_cu", "doan_moi"]
    w.writerow(cols)
    for r in E.read_log(EDITS_LOG):
        r = {**r, "doan_cu": (r.get("thay_chuoi_con") or {}).get("cu"), "doan_moi": (r.get("thay_chuoi_con") or {}).get("moi")}
        w.writerow([json.dumps(r.get(c), ensure_ascii=False) if isinstance(r.get(c), (dict, list, bool)) else ("" if r.get(c) is None else r.get(c)) for c in cols])
    return "﻿" + buf.getvalue()


def page_milestone():
    body = (f'<h1>Chốt đợt</h1><p>Đã chốt {len(STATE["dot"])} đợt. Chốt đợt ghi một mốc vào log hiệu đính rồi commit git cục bộ '
            f'(config/ và data/human_edits/; không đẩy lên đâu). Việc dựng bản hiệu đính chỉ chạy khi người làm dự án yêu cầu.</p>'
            f'<p>Ghi chú: <input type="text" id="note" style="width:50%"> <button onclick="ms()">Chốt đợt</button></p><p id="res" class="small"></p>')
    js = """<script>function ms(){post('/api/milestone',{note:val('note')}).then(function(j){document.getElementById('res').textContent='Đã chốt đợt '+j.dot+'. git: '+j.git;});}</script>"""
    return page("Chốt đợt", body, js)


def api_milestone(d):
    n = len(STATE["dot"]) + 1
    summ = {"mau": STATE["dem_mau"], "so_cau_co_sua": len(STATE["cau"]), "so_cau_test_co_sua": len(STATE["cau_test"]), "so_de_xuat_da_xu_ly": len(STATE["de_xuat"]),
            "so_quyet_dinh_soat": len(DECIDED)}
    E.append(EDITS_LOG, {"ai": args.user, "loai": "chot_dot", "hanh_dong": "chot", "dot": n, "tom_tat": summ, "ghi_chu": str(d.get("note", ""))})
    refresh_state()
    if args.no_git:
        return {"dot": n, "git": "bỏ qua (--no_git)"}
    try:
        subprocess.run(["git", "-C", args.repo, "add", "--", "config", os.path.relpath(args.edits_dir, args.repo)], check=True, capture_output=True, timeout=60)
        msg = f"Chốt đợt {n} (người dùng: {args.user})\n\n{json.dumps(summ, ensure_ascii=False)}\n{str(d.get('note', ''))}".strip()
        out = subprocess.run(["git", "-C", args.repo, "commit", "-m", msg], capture_output=True, timeout=60, text=True)
        if out.returncode != 0 and "nothing to commit" not in (out.stdout + out.stderr):
            return {"dot": n, "git": "lỗi commit: " + (out.stderr or out.stdout)[:200]}
        h = subprocess.run(["git", "-C", args.repo, "rev-parse", "--short", "HEAD"], capture_output=True, timeout=30, text=True).stdout.strip()
        return {"dot": n, "git": f"commit {h}"}
    except Exception as ex:   # mốc đã ghi; báo lỗi git cho người dùng
        return {"dot": n, "git": f"lỗi git: {type(ex).__name__}"}


# ---------------------------------------------------------------- HTTP
CSP = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
API = {"/api/dung_sai": api_dung_sai, "/api/review": api_review, "/api/template": api_template, "/api/item": api_item, "/api/proposal": api_proposal, "/api/milestone": api_milestone}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8", headers=None):
        b = body.encode("utf-8")
        self.send_response(code)
        for k, v in (("Content-Type", ctype), ("Content-Security-Policy", CSP), ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                     ("Referrer-Policy", "no-referrer"), ("X-Frame-Options", "DENY"), *((headers or {}).items()), ("Content-Length", str(len(b)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def log_message(self, *a):
        pass

    def _host_ok(self):
        return self.headers.get("Host", "") in HOSTS

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, "Host không hợp lệ")
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        if u.path.startswith("/api/"):
            return self._send(405, "chỉ nhận POST", headers={"Allow": "POST"})
        try:
            if u.path == "/":
                return self._send(200, page_home())
            if u.path == "/review":
                k = first_undecided()
                return self._send(303, "", headers={"Location": "/done" if k is None else f"/item?k={k}"})
            if u.path == "/item":
                k = int(qs.get("k", ["0"])[0] or 0)
                return self._send(200, page_item(max(0, min(len(QUEUE) - 1, k))))
            if u.path == "/done":
                return self._send(200, page("Xong", f"{progress_html()}<p>Đã có quyết định cho mọi câu. <a href='/item?k=0'>Câu đầu</a>.</p>"))
            if u.path == "/templates":
                return self._send(200, page_templates())
            if u.path == "/items":
                return self._send(200, page_items(qs))
            if u.path == "/edit":
                return self._send(200, page_edit(qs.get("dev_id", [""])[0]))
            if u.path == "/proposals":
                return self._send(200, page_proposals(qs))
            if u.path == "/dung_sai":
                return self._send(200, page_dung_sai())
            if u.path == "/milestone":
                return self._send(200, page_milestone())
            if u.path == "/export.csv":
                return self._send(200, export_csv(), "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="human_edits_export.csv"'})
        except ValueError:
            return self._send(400, "tham số không hợp lệ")
        return self._send(404, "không có")

    def do_POST(self):
        if not self._host_ok():
            return self._json(403, {"ok": False, "error": "Host không hợp lệ"})
        origin = self.headers.get("Origin")
        if origin is not None and origin not in ORIGINS:
            return self._json(403, {"ok": False, "error": "Origin không hợp lệ"})
        if not secrets.compare_digest(self.headers.get("X-Review-Token", ""), TOKEN):
            return self._json(403, {"ok": False, "error": "thiếu hoặc sai token"})
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._json(415, {"ok": False, "error": "cần application/json"})
        fn = API.get(urllib.parse.urlparse(self.path).path)
        if fn is None:
            return self._json(404, {"ok": False, "error": "không có"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 1_000_000:
                return self._json(413, {"ok": False, "error": "quá lớn"})
            data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            with LOCK:
                out = fn(data)
            return self._json(200, {"ok": True, **out})
        except (ValueError, KeyError, TypeError, IndexError) as ex:
            return self._json(400, {"ok": False, "error": str(ex)[:200]})


if __name__ == "__main__" and args.dry_run:
    pend = E.pending_proposals(load_props(), STATE)
    chk = [check_cau(p) for p in pend if p.get("loai") == "cau"]
    print(json.dumps({"rows": dict(CNT), "by_row_source": {f"{h}|{s}": n for (h, s), n in sorted(Counter((h, r["source"]) for h, r in QUEUE).items())},
                      "seed_row_C": args.seed, "dev_sha256": DEV_SHA, "has_sensitive": has_sensitive, "test_sha256": TEST_SHA, "test_err": TEST_ERR,
                      "mau_va_nhan": {"tong": len(INV), "theo_nhom": {g: sum(x["group"] == g for x in INV) for g in GROUPS}, "trang_thai": STATE["dem_mau"],
                                      "chua_duyet_theo_nhom": {g: sum(x["group"] == g and STATE["mau"][x["key"]]["trang_thai"] == "chưa duyệt" for x in INV) for g in GROUPS}},
                      "de_xuat_cho": dict(Counter(p.get("split") if p.get("loai") == "cau" else "mau" for p in pend)),
                      "de_xuat_cau_loi": {"cau_truc": sum(bool(a) for a, _, _ in chk), "noi_dung": sum(bool(b) for a, b, _ in chk if not a)}}, ensure_ascii=False))
    sys.exit(0)

if __name__ == "__main__":
    print(json.dumps({"rows": dict(CNT), "decided": len(DECIDED), "seed_row_C": args.seed, "dev_sha256": DEV_SHA[:16], "mau_va_nhan": STATE["dem_mau"],
                      "url": f"http://127.0.0.1:{args.port}"}, ensure_ascii=False), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
