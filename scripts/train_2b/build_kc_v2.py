"""Vòng sửa KC (results/BRIEF_2b_du_lieu.md mục 7c): KC v2 trên cùng các bài Wikipedia đã lọc và khử trùng (results/train_2b/tam/bai_sach.jsonl).
C′ (mục 2) giữ nguyên; bài "cho" của 2e (tam/bai_cho.jsonl) không dùng lại. Không chạy model (tokenizer Qwen3-1.7B chỉ để đếm token). 0 chữ
tiếng Việt mới: nhận định là câu trích, câu trích đổi cơ học, hoặc ghép mệnh đề trích bằng " và " (từ duy nhất được chèn); cụm chèn ở
N-them chép nguyên từ câu khác của cùng bài; biến thể định dạng chỉ đổi chữ số / dấu / số La Mã (dạng ngày đổi qua lại có thêm / bớt các từ
khung "ngày", "tháng", "năm" của chính ngày đó).

Hình dạng nhận định (cân giữa các nhãn, cổng D9):
  nguyen  một câu (hoặc câu đã đổi một giá trị / đã chèn cụm);  luoc  câu đã lược (bỏ ngoặc / bỏ mệnh đề sau dấu phẩy cuối);
  ghep    mệnh đề của hai câu nối bằng " và ".
Kiểu dựng:
  Xác nhận  S-nguyen / S-luoc / S-ghep (hai câu trong văn bản)
  Bác bỏ    R-so / R-ten trên một hình dạng như Xác nhận (nguyen / luoc / ghep), đổi một giá trị theo luật cũ
  NEI       N-bobang  câu s trong cửa sổ làm nhận định rồi XÓA s khỏi state; s có >= 1 số / tên không còn ở phần state còn lại;
                      bỏ qua nếu câu ngay sau s mở đầu bằng từ nối chỉ ngược (danh sách NOI_NGUOC)
            N-them    câu trong state + một cụm thời gian ("vào năm …", "năm …", "vào ngày …", "tháng …") hoặc nơi chốn ("tại / ở" + tên riêng)
                      chép nguyên từ câu ở xa (>= 2 đoạn) của cùng bài, chèn trước dấu câu cuối; giá trị trong cụm không có trong state;
                      chỉ chèn khi câu chưa có cụm cùng loại
            N-ghep    mệnh đề của câu trong state + " và " + mệnh đề của câu ở xa cùng bài có số / tên không có trong state
            N-cungbai, N-lacde  như v1 (dạng nguyen / luoc), và một phần ở dạng ghép (hai câu ở xa / hai câu của bài khác) để cân D9
Biến thể định dạng (xác suất P_VARIANT cho mọi nhãn): "ngày 2 tháng 9 năm 1945" ↔ "2/9/1945"; "tháng 9 năm 1945" ↔ "9/1945";
"1.000" ↔ "1000" (không đụng năm); "thế kỷ XX" ↔ "thế kỷ 20". Không đổi số sang chữ.
Luật chất lượng sau đọc mù vòng 1 (results/train_2b/DIEU_PHOI_doc_mu_vong1.md), áp cho mọi kiểu: lược bằng kc_util.luoc_v2 (ngoặc cân có
dấu cách đứng trước, không kết thúc bằng hư từ; bỏ mệnh đề chỉ ở ", " + từ mở đầu mệnh đề phụ); bỏ câu kiểu trích dẫn và câu có tỉ lệ âm
tiết mang dấu < 0,3 (kc_util.cau_dung_duoc); R-so / R-ten: giá trị thay không có trong câu gốc trước khi lược và câu liền trước / sau,
không là tên năm can chi, ưu tiên ứng viên cùng từ đứng trước.
Còn lại như build_kc.py (v1): dải độ dài, choice / noul 55/45, 20% không dấu, val 5% số bài, khử trùng lại từng dòng.
Ra results/train_2b/{kc_pool_ngan_v2.jsonl, kc_pool_dai_v2.jsonl, val_kc_v2.jsonl} + tam/kc_log_v2.json (v1 giữ nguyên).
Dùng: bash setup/run.sh -I scripts/train_2b/build_kc_v2.py"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import (CAN_CHI, DEV, LO, OUT, ROOT, SEED, TMP, UP, Refs, cau_dung_duoc, h64, luoc_v2, name_spans, normalize, num_spans, overlap, read_jsonl,  # noqa: E402
                     sentences, u01, write_jsonl)

sys.path.insert(0, str(ROOT / "scripts"))
from common import strip_diacritics  # noqa: E402

N_SHORT, N_LONG, VAL_FRAC, VAL_MAX = 2500, 1600, 0.05, 300
LAB_P = {"choice": [("supported", 0.29), ("refuted", 0.29), ("nei", 0.42)], "noul": [("supported", 0.37), ("refuted", 0.28), ("nei", 0.35)]}
SHAPE_SR = {"nguyen": 0.35, "luoc": 0.30, "ghep": 0.35}               # Xác nhận, Bác bỏ
N_TYPES = {"N-bobang": 0.30, "N-them": 0.25, "N-ghep": 0.15, "N-cungbai": 0.15, "N-lacde": 0.15}
P_N_EXTRA_GHEP = 2 / 3          # phần N-cungbai / N-lacde dựng ở dạng ghép, để NEI ghép ~ 15 + 20 = 35%
P_LUOC_N = 0.30 / 0.65          # NEI dạng thường: lược cùng tỉ lệ như Xác nhận trong phần không ghép
P_VARIANT, P_RSO = 0.25, 0.6
LABS = ["supported", "refuted", "nei"]
NOI_NGUOC = ("Năm đó", "Cùng năm", "Sau đó", "Khi đó", "Lúc đó", "Điều này", "Việc này", "Đây", "Ông", "Bà", "Anh", "Chị", "Họ", "Nó")

os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"])
REFS = Refs()


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def touched(state, claim):
    for t in [state, claim] + sentences(state):
        h = REFS.hit(t)
        if h:
            return h
    return None


# ------------------------------------------------------------------ chữ dev
dev = read_jsonl(DEV)
FC_OPT, PRE = {}, {}
for r in dev:
    if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] in ("vifactcheck", "viwikifc"):
        if r["kind"] == "choice" and r["gold_provenance"] == "source":
            FC_OPT[r["gold_label"]] = r["options"][r["target"]]
        mk = "Nhận định: " if r["kind"] == "choice" else "Phát biểu: "
        PRE.setdefault(r["kind"], set()).add(r["question"].split(mk, 1)[0] + mk)
PRE = {k: next(iter(v)) for k, v in PRE.items()}
NOUL = next(r["options"] for r in dev if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] == "vifactcheck" and r["kind"] == "noul")
DEV_STR = {s for r in dev if r["source"] in ("vifactcheck", "viwikifc") for s in r["options"]}
DEV_PRE = {r["question"].split(mk, 1)[0] + mk for r in dev if r["source"] in ("vifactcheck", "viwikifc")
           for mk in ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ") if mk in r["question"]}

# ------------------------------------------------------------------ bài
DONOR = {d["id"] for d in read_jsonl(TMP / "bai_cho.jsonl")}
arts = read_jsonl(TMP / "bai_sach.jsonl")
for a in arts:
    a["sents"] = [sentences(p) for p in a["pars"]]
    a["split"] = "val" if u01(SEED, "val", a["id"]) < VAL_FRAC else "train"
arts.sort(key=lambda a: h64(SEED, "kc", a["id"]))
by_split = defaultdict(list)
for a in arts:
    by_split[a["split"]].append(a)
log = defaultdict(Counter)
END = re.compile(r"[.!?…][\"”’')\]]*$")


def ok_claim(s):
    n = len(s.split())
    return 8 <= n <= 45 and END.search(s) and "\n" not in s and (num_spans(s) or name_spans(s)) and cau_dung_duoc(s)


def keys(s):
    return [v for _, _, v, _ in num_spans(s)] + [n for _, _, n, _ in name_spans(s)]


def new_keys(s, state_norm):
    return [k for k in keys(s) if normalize(k) and f" {normalize(k)} " not in f" {state_norm} "]


def core(s):
    m = END.search(s)
    return s[:m.start()].rstrip() if m else s.rstrip()


LOWER_OK = set("Ông Bà Anh Chị Họ Nó Năm Sau Trong Khi Đây Các Những Một Theo Vào Từ Tại Ở Cũng Ngoài Hiện Đến Với Do Đó Này Ngày Tháng "
               "Hai Ba Bốn Nhiều Mỗi Tuy Dù Nếu Vì Bởi Cùng Lúc Thời Hầu Phần Đa Số Toàn Tất Gần Hơn".split())


def join(a, b):
    """Mệnh đề a + " và " + mệnh đề b (chữ đầu của b hạ thường khi là từ thường, giữ khi là tên riêng)."""
    b0 = core(b)
    w = b0.split(" ", 1)
    if w and w[0] in LOWER_OK and not (len(w) > 1 and w[1][:1].isupper()):
        b0 = w[0].lower() + (" " + w[1] if len(w) > 1 else "")
    return core(a) + " và " + b0 + "."


def shape_claim(s, shape, s2=None):
    """Câu theo hình dạng: (nhận định, cách lược) hoặc None. Câu nguồn phải dùng được (không phải mảnh trích dẫn, đủ dấu tiếng Việt)."""
    if not cau_dung_duoc(s) or (s2 is not None and not cau_dung_duoc(s2)):
        return None
    if shape == "nguyen":
        return s, None
    if shape == "luoc":
        x = luoc_v2(s)
        return (x[0], x[1]) if x else None
    if s2 is None:
        return None
    a = luoc_v2(s); b = luoc_v2(s2)
    a = a[0] if a and len(s.split()) > 22 else s                      # câu dài thì lược trước khi ghép cho gọn
    b = b[0] if b and len(s2.split()) > 22 else s2
    return join(a, b), "ghep"


# ------------------------------------------------------------------ biến thể định dạng
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman2int(s):
    v = 0
    for i, c in enumerate(s):
        x = ROMAN[c]; v += -x if i + 1 < len(s) and ROMAN[s[i + 1]] > x else x
    return v


def int2roman(n):
    out = ""
    for v, c in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += c; n -= v
    return out


VARIANTS = [
    ("ngay_gach", re.compile(r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{3,4})"), lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}"),
    ("gach_ngay", re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{3,4})(?![\d/])"), lambda m: f"ngày {m.group(1)} tháng {m.group(2)} năm {m.group(3)}"),
    ("thang_gach", re.compile(r"(?<!ngày \d )(?<!ngày \d\d )(?<![\d/])tháng (\d{1,2}) năm (\d{3,4})"), lambda m: f"{m.group(1)}/{m.group(2)}"),
    ("gach_thang", re.compile(r"(?<![\d/])(\d{1,2})/(\d{4})(?![\d/])"), lambda m: f"tháng {m.group(1)} năm {m.group(2)}"),
    ("bo_cham_nghin", re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+)(?![\d.,]\d)"), lambda m: m.group(1).replace(".", "")),
    ("them_cham_nghin", re.compile(r"(?<![\d.,])(\d{5,}|[1-9]\d{3})(?![\d.,]\d)"),
     lambda m: (f"{int(m.group(1)):,}".replace(",", ".") if not (1000 <= int(m.group(1)) <= 2100) else None)),
    ("the_ky_so", re.compile(r"(thế k[ỷỉ]) ([IVXLC]+)\b"), lambda m: f"{m.group(1)} {roman2int(m.group(2))}"),
    ("the_ky_la_ma", re.compile(r"(thế k[ỷỉ]) (\d{1,2})\b"), lambda m: f"{m.group(1)} {int2roman(int(m.group(2)))}"),
]


def variant(claim, rid):
    """(nhận định đổi định dạng, loại) hoặc None; giữ nguyên nghĩa."""
    cands = []
    for name, rx, f in VARIANTS:
        for m in rx.finditer(claim):
            new = f(m)
            if new and new != m.group(0):
                cands.append((name, m.start(), m.end(), new))
    if not cands:
        return None
    name, st, en, new = cands[h64(SEED, "var", rid) % len(cands)]
    return claim[:st] + new + claim[en:], name


# ------------------------------------------------------------------ dựng nhận định
TIME = re.compile(r"(?:vào )?(?:ngày \d{1,2} tháng \d{1,2}(?: năm \d{3,4})?|tháng \d{1,2} năm \d{3,4}|năm \d{3,4})")
HAS_TIME = re.compile(r"\b(năm|ngày|tháng) \d|\b(1\d{3}|20\d{2})\b")
PLACE = re.compile(r"(?:tại|ở) ((?:[" + UP + r"][" + LO + r"]*)(?: [" + UP + r"][" + LO + r"]*)+)")
HAS_PLACE = re.compile(r"\b(tại|ở) [" + UP + r"]")


def windows(a, band):
    nt = a["ntok"]; P = len(nt); out = []
    if band == "ngan":
        for p0 in range(P):
            for L in (1, 2, 3):
                if p0 + L <= P and 80 <= sum(nt[p0:p0 + L]) + L <= 400:
                    out.append((p0, p0 + L - 1))
    else:
        tgt = min(1300, max(500, math.exp(math.log(750) + 0.25 * _gauss(a["id"]))))
        for p0 in range(P):
            s, p1 = 0, p0
            while p1 < P and s < tgt:
                s += nt[p1] + 1; p1 += 1
            if 500 <= s <= 1300 and s >= tgt:
                out.append((p0, p1 - 1))
    out.sort(key=lambda w: h64(SEED, "win", a["id"], band, w))
    return out


def _gauss(aid):
    u1, u2 = max(1e-12, u01(SEED, "g1", aid)), u01(SEED, "g2", aid)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def pick(dist, u):
    acc = 0
    for k, p in dist.items() if isinstance(dist, dict) else dist:
        acc += p
        if u < acc:
            return k
    return list(dist)[-1] if isinstance(dist, dict) else dist[-1][0]


_AV = {}


def _prev_word(s, st):
    w = s[:st].split()
    return w[-1].lower().strip(",.;:()\"“”") if w else ""


def article_values(a):
    """Giá trị cùng loại trong bài + các từ đứng trước mỗi giá trị (để ưu tiên ứng viên cùng từ đứng trước). Cache theo bài."""
    if a["id"] in _AV:
        return _AV[a["id"]]
    nums, names, prev = defaultdict(set), defaultdict(set), defaultdict(set)
    flat = [s for ss in a["sents"] for s in ss]
    for s in flat:
        for st, _, v, t in num_spans(s):
            nums[t].add(v); prev[v].add(_prev_word(s, st))
        for st, _, n, k in name_spans(s):
            if not CAN_CHI.fullmatch(n):                 # bỏ tên năm can chi
                names[k].add(n); prev[n].add(_prev_word(s, st))
    idx = {}
    for i, s in enumerate(flat):
        idx.setdefault(s, i)
    _AV[a["id"]] = (nums, names, prev, flat, idx)
    return _AV[a["id"]]


def context_text(a, goc):
    """Câu gốc trước khi lược + câu liền trước và liền sau của mỗi câu gốc."""
    _, _, _, flat, idx = article_values(a)
    out = []
    for s in ([goc] if isinstance(goc, str) else goc):
        i = idx.get(s)
        out += [s] if i is None else flat[max(0, i - 1):i + 2]
    return " ".join(out)


def swap(c, a, rkind, rid, goc):
    """Đổi một giá trị (v2): giá trị thay không có trong câu gốc trước khi lược, không có ở câu liền trước / liền sau; không là tên năm
    can chi; ưu tiên ứng viên có cùng từ đứng trước với giá trị gốc."""
    nums, names, prev, _, _ = article_values(a)
    ctx = context_text(a, goc)
    order = lambda xs, tag: sorted(xs, key=lambda x: h64(SEED, tag, rid, x))
    rank = lambda cands, pw, tag: sorted(cands, key=lambda x: (0 if pw and pw in prev[x] else 1, h64(SEED, tag, rid, x)))
    if rkind == "R-so":
        spans = num_spans(c); have = {v for _, _, v, _ in spans}
        for st, en, v, t in order(spans, "rs_span"):
            cands = [x for x in nums[t] if x not in have and x != v and x not in c
                     and not re.search(r"(?<![\d.,])" + re.escape(x) + r"(?!\d)", ctx)]
            if cands:
                nv = rank(cands, _prev_word(c, st), "rs_val")[0]
                return c[:st] + nv + c[en:], {"doi": [v, nv], "loai_so": t}
    else:
        for st, en, n, k in order(name_spans(c), "rt_span"):
            if CAN_CHI.fullmatch(n):
                continue
            pool = names[k]
            if len(pool) < 2:
                continue
            cands = [x for x in pool if x != n and x not in c and n not in x and x not in n and x not in ctx]
            if cands:
                nv = rank(cands, _prev_word(c, st), "rt_val")[0]
                return c[:st] + nv + c[en:], {"doi": [n, nv]}
    return None


def candidates(a, p0, p1, plan, rid, pool_split):
    """Sinh (nhận định, kiểu, hình dạng, cách lược, câu gốc, state hoặc None, thêm) theo kế hoạch của dòng."""
    label, typ, shape = plan["label"], plan["typ"], plan["shape"]
    win = [s for ss in a["sents"][p0:p1 + 1] for s in ss]
    win_text = " ".join(a["pars"][p0:p1 + 1]); win_norm = normalize(win_text)
    order = lambda xs, tag: sorted(xs, key=lambda x: h64(SEED, tag, rid, str(x)))
    P = len(a["pars"])
    far = [s for i in range(P) if i <= p0 - 2 or i >= p1 + 2 for s in a["sents"][i]]
    pairs = lambda xs, tag, k=40: order([(x, y) for i, x in enumerate(xs) for y in xs[i + 1:]], tag)[:k]
    if label in ("supported", "refuted"):
        src = pairs(win, "pair") if shape == "ghep" else [(s, None) for s in order(win, "S")]
        for s, s2 in src:
            m = shape_claim(s, shape, s2)
            if not m or not ok_claim(m[0]):
                continue
            goc = s if s2 is None else [s, s2]
            if label == "supported":
                yield m[0], f"S-{shape}", shape, m[1], goc, None, {}
            else:
                for rk in plan["rkinds"]:
                    sw = swap(m[0], a, rk, rid, goc)
                    if sw:
                        yield sw[0], rk, shape, m[1], goc, None, sw[1]
                        break
        return
    if typ == "N-bobang":
        for pi in range(p0, p1 + 1):
            ss = a["sents"][pi]
            for j in order(range(len(ss)), f"bb{pi}"):
                s = ss[j]
                nxt = ss[j + 1] if j + 1 < len(ss) else (a["sents"][pi + 1][0] if pi + 1 <= p1 and a["sents"][pi + 1] else "")
                if nxt.startswith(NOI_NGUOC):
                    continue
                pars = [" ".join(x for k, x in enumerate(a["sents"][q]) if not (q == pi and k == j)) if q == pi else a["pars"][q]
                        for q in range(p0, p1 + 1)]
                st = " ".join(p for p in pars if p.strip())
                m = shape_claim(s, shape)
                if not m or not ok_claim(m[0]) or not new_keys(m[0], normalize(st)):
                    continue
                yield m[0], "N-bobang", shape, m[1], s, st, {"moi": new_keys(m[0], normalize(st))[:3]}
        return
    if typ == "N-them":
        phrases = []
        for s in far:
            for m in TIME.finditer(s):
                vals = re.findall(r"\d+", m.group(0))
                if all(f" {v} " not in f" {win_norm} " for v in vals):
                    phrases.append(("thoi_gian", m.group(0)))
            for m in PLACE.finditer(s):
                if normalize(m.group(1)) not in win_norm:
                    phrases.append(("noi_chon", m.group(0)))
        if not phrases:
            return
        for s in order(win, "Nt"):
            m = shape_claim(s, shape)
            if not m:
                continue
            for kind_ph, ph in order(sorted(set(phrases)), "Nt_ph")[:6]:
                if (kind_ph == "thoi_gian" and HAS_TIME.search(m[0])) or (kind_ph == "noi_chon" and HAS_PLACE.search(m[0])):
                    continue
                c = core(m[0]) + " " + ph + "."
                if ok_claim(c):
                    yield c, "N-them", shape, m[1], s, None, {"cum_chen": ph, "loai_cum": kind_ph}
                    break
        return
    if typ == "N-ghep":
        fk = [s for s in far if new_keys(s, win_norm)]
        for s, s2 in order([(x, y) for x in win for y in fk], "Ng")[:60]:
            m = shape_claim(s, "ghep", s2)
            if m and ok_claim(m[0]):
                yield m[0], "N-ghep", "ghep", "ghep", [s, s2], None, {"moi": new_keys(s2, win_norm)[:3]}
        return
    # N-cungbai, N-lacde (nguyen / luoc / ghep)
    if typ == "N-cungbai":
        pool = far
    else:
        others = by_split[pool_split]; start = h64(SEED, "Nl", rid) % len(others); pool = []
        for j in range(30):
            b = others[(start + j * 13) % len(others)]
            if b["id"] != a["id"]:
                bs = [s for x in b["sents"] for s in x]
                pool += [(s, b) for s in order(bs, "Nl_s")[:8]]
    items = [x if isinstance(x, tuple) else (x, None) for x in pool]
    src = (order([(x, y) for i, x in enumerate(items) for y in items[i + 1:i + 4] if (x[1] or {}).get("id") == (y[1] or {}).get("id")], "Nc2")[:60]
           if shape == "ghep" else [(x, None) for x in order(items, "Nc1")])
    for x, y in src:
        m = shape_claim(x[0], shape, y[0] if y else None)
        if not m or not ok_claim(m[0]):
            continue
        if typ == "N-cungbai" and not new_keys(m[0], win_norm):
            continue
        if not all(overlap(m[0], w) < 0.5 for w in win):
            continue
        extra = {"moi": new_keys(m[0], win_norm)[:3]} if typ == "N-cungbai" else {"bai_khac": x[1]["id"], "tieu_de_khac": x[1]["title"]}
        yield m[0], typ, shape, m[1], x[0] if y is None else [x[0], y[0]], None, extra


def plan_of(rid, kind):
    label = pick(LAB_P[kind], u01(SEED, "lab", rid))
    p = {"label": label}
    if label in ("supported", "refuted"):
        p["typ"] = None; p["shape"] = pick(SHAPE_SR, u01(SEED, "shape", rid))
        p["rkinds"] = ["R-so", "R-ten"] if u01(SEED, "rk", rid) < P_RSO else ["R-ten", "R-so"]
    else:
        p["typ"] = pick(N_TYPES, u01(SEED, "ntyp", rid))
        if p["typ"] == "N-ghep":
            p["shape"] = "ghep"
        elif p["typ"] in ("N-cungbai", "N-lacde") and u01(SEED, "nextra", rid) < P_N_EXTRA_GHEP:
            p["shape"] = "ghep"
        else:
            p["shape"] = "luoc" if u01(SEED, "nluoc", rid) < P_LUOC_N else "nguyen"
    p["variant"] = u01(SEED, "variant", rid) < P_VARIANT
    return p


def make_row(a, band, pool_split):
    rid = f"kc2:{band}:{a['id']}"
    kind = "choice" if u01(SEED, "kind", rid) < 0.55 else "noul"
    plan = plan_of(rid, kind)
    label = plan["label"]
    for w in windows(a, band)[:8]:
        n_try = 0
        for claim, kieu, shape, cach, goc, st_over, extra in candidates(a, w[0], w[1], plan, rid, pool_split):
            n_try += 1
            if n_try > 40:
                break
            var = None
            if plan["variant"]:
                v = variant(claim, rid)
                if not v:
                    continue
                claim, var = v
            if not ok_claim(claim):
                continue
            state = st_over or " ".join(a["pars"][w[0]:w[1] + 1])
            h = touched(state, claim)
            if h:
                log["bo_cham"][f"{h[0]}|{h[1]}"] += 1
                continue
            nst = ntok(state)
            if not ((80 <= nst <= 400) if band == "ngan" else (500 <= nst <= 1300)):
                log["bo_token_ngoai_dai"][kieu] += 1
                continue
            if kind == "choice":
                perm = sorted(range(3), key=lambda j: h64(SEED, "perm", rid, j))
                options = [FC_OPT[LABS[p]] for p in perm]; t = perm.index(LABS.index(label)); q = PRE["choice"] + claim
            else:
                options = list(NOUL); t = 0 if label == "supported" else 1; q = PRE["noul"] + claim
            r = {"id": rid, "dataset": "wikimedia/wikipedia:20231101.vi", "source": f"kc_wiki_{band}", "kind": kind, "lang": "vi", "text_lang": "vi",
                 "license_use": "CC BY-SA 4.0 (chỉ dùng nội bộ)", "split": pool_split, "state": state, "question": q, "options": options,
                 "target": [1.0 if j == t else 0.0 for j in range(len(options))], "target_argmax": t, "n_state_tokens": nst, "variant": "program",
                 "flags": [], "gold_label": label, "label_source": "program", "text_source": "wikipedia_vi_20231101", "license": "CC BY-SA 4.0",
                 "kieu": kieu, "hinh_dang": shape, "luoc": cach if cach != "ghep" else None, "bien_the": var, "tu_chen": "và" if shape == "ghep" else None,
                 "tieu_de": a["title"], "wiki_id": a["id"], "doan": [w[0], w[1]], "cau_goc": goc, "xoa_cau_khoi_state": st_over is not None,
                 "seed": SEED, "kc_ban": 2, "claim_ntok": ntok(claim), **extra}
            if u01(SEED, "nodiac", rid) < 0.20:
                r["state"], r["question"] = strip_diacritics(r["state"]), strip_diacritics(r["question"])
                r["options"] = [strip_diacritics(o) for o in r["options"]]
                r["flags"] = ["nodiac"]; r["n_state_tokens"] = ntok(r["state"])
            assert all(o in DEV_STR for o in r["options"]) and any(r["question"].startswith(p) for p in DEV_PRE), r["id"]
            log["kieu"][f"{band}|{pool_split}|{kind}|{kieu}|{shape}|{'bt' if var else '-'}"] += 1
            return r
    log["bai_khong_dung_duoc"][f"{band}|{plan['typ'] or label}|{plan['shape']}|bt={plan['variant']}"] += 1
    return None


pools = {("train", "ngan"): [], ("train", "dai"): [], ("val", "ngan"): [], ("val", "dai"): []}
cap = {("train", "ngan"): N_SHORT, ("train", "dai"): N_LONG, ("val", "ngan"): VAL_MAX * 3 // 5, ("val", "dai"): VAL_MAX * 2 // 5}
n_seen = 0
for a in arts:
    if a["id"] in DONOR:
        continue
    n_seen += 1
    sp = a["split"]
    long_ok = sum(a["ntok"]) >= 900 and len(a["pars"]) >= 6
    for band in (["dai", "ngan"] if long_ok else ["ngan"]):
        if len(pools[(sp, band)]) >= cap[(sp, band)]:
            continue
        r = make_row(a, band, sp)
        if r:
            pools[(sp, band)].append(r)
            break
    if all(len(pools[k]) >= cap[k] for k in pools):
        break

for (sp, band), rows in pools.items():
    log["so_dong"][f"{sp}|{band}"] = len(rows)
write_jsonl(OUT / "kc_pool_ngan_v2.jsonl", pools[("train", "ngan")])
write_jsonl(OUT / "kc_pool_dai_v2.jsonl", pools[("train", "dai")])
val = pools[("val", "ngan")] + pools[("val", "dai")]
write_jsonl(OUT / "val_kc_v2.jsonl", val)
tr_ids = {r["wiki_id"] for k, v in pools.items() if k[0] == "train" for r in v} | DONOR
va_ids = {r["wiki_id"] for r in val}
assert not tr_ids & va_ids
res = {"so_bai_da_xet": n_seen, "so_bai_sach": len(arts), "bo_bai_cho_2e": len(DONOR), "log": {k: dict(v) for k, v in log.items()},
       "ti_le_nhan": LAB_P, "hinh_dang_S_R": SHAPE_SR, "kieu_NEI": N_TYPES, "p_N_cungbai_lacde_ghep": P_N_EXTRA_GHEP, "p_luoc_N_thuong": P_LUOC_N,
       "p_bien_the": P_VARIANT, "p_rso": P_RSO, "noi_nguoc": NOI_NGUOC, "tu_chen_duy_nhat": "và"}
(TMP / "kc_log_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res["log"].items() if k != "kieu"}, ensure_ascii=False, indent=1))
