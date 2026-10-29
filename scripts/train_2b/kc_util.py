"""Hàm dùng chung cho dữ liệu 2b (results/BRIEF_2b_du_lieu.md): làm sạch bài Wikipedia, tách câu, tìm số / tên riêng, lược câu, chỉ mục khử trùng.
Thư viện chuẩn + decontam (datasketch) của dự án. Không chạy model."""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from decontam import Index, normalize  # noqa: E402,F401

SEED = 20261028
WIKI = ROOT / "results/train_v1/raw_refs/wikimedia__wikipedia_20231101.vi/20231101.vi"
OUT = ROOT / "results/train_2b"
TMP = OUT / "tam"

# Tiêu đề mục mà từ đó trở đi chỉ còn chú thích / tham khảo / liên kết: cắt bài tại đây.
REF_HEADS = {"chú thích", "tham khảo", "liên kết ngoài", "xem thêm", "đọc thêm", "ghi chú", "nguồn", "thư mục", "tài liệu tham khảo",
             "nguồn tham khảo", "chú giải", "liên kết", "thư mục chọn lọc", "sách tham khảo"}
ABBR = {"tp", "tt", "st", "mr", "mrs", "ms", "dr", "ths", "ts", "gs", "pgs", "th", "jr", "sr", "co", "inc", "ltd", "no", "vol", "pp", "ed",
        "tr", "vs", "v.v", "ft", "mt", "ct", "cty", "sđd"}
# Từ chức năng hay đứng ngay sau một tên riêng; âm tiết thường khác đứng sau thì coi là tên còn tiếp bằng chữ thường ("Họ Cá trích",
# "Đại học Cornell"…) và không nhận làm tên để đổi.
AFTER_OK = set("""là và của ở tại trong với cho từ đến được đã có năm ngày tháng thuộc nằm khi vào sau trước theo bởi như cũng không này đó
các những một hai ba bốn người nhưng mà thì nên để về ra lên đi làm gồm giữa bị do qua tới rằng sinh mất cùng hay hoặc vì sẽ đang vẫn còn
chỉ rất lại đều nơi gần thành trở bên phía nhằm thay dưới trên chống đối kể tuy dù nếu""".split())


def h64(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def u01(*parts):
    return h64(*parts) / 2 ** 64


def words(s):
    return s.split()


def is_prose(line):
    s = line.strip()
    if len(s.split()) < 12 or s[0] in "-*•#|=!{[" or s.count("|") > 1:
        return False
    if not re.search(r"[.!?…][\"”’')\]]*$", s):
        return False
    digits = sum(c.isdigit() for c in s)
    return digits / max(1, len(s)) < 0.15


def prose_paragraphs(text):
    """Đoạn văn xuôi theo thứ tự, cắt ở mục chú thích / tham khảo đầu tiên. Bỏ tiêu đề mục, danh sách, bảng."""
    out = []
    for line in text.split("\n"):
        s = " ".join(line.split())
        if not s:
            continue
        if s.lower().rstrip(":").strip() in REF_HEADS:
            break
        if is_prose(s):
            out.append(s)
    return out


_LATIN = [chr(c) for c in list(range(0x41, 0x7B)) + list(range(0xC0, 0x250)) + list(range(0x1E00, 0x1F00)) if chr(c).isalpha()]
UP = "".join(c for c in _LATIN if c.isupper())
LO = "".join(c for c in _LATIN if c.islower())
_SPLIT = re.compile(r"(?<=[.!?…])([\"”’')\]]*)\s+(?=[\"“'(\[]?[" + UP + r"0-9])")


def _upper_start(t):
    return bool(t) and t[0].isupper()


def sentences(par):
    """Tách câu: sau . ! ? … (có thể kèm ngoặc / nháy đóng) + khoảng trắng + chữ hoa hoặc số. Không tách sau chữ viết tắt
    (danh sách ABBR) hay chữ cái đơn viết hoa (tên viết tắt kiểu "J. R. R.")."""
    out, start = [], 0
    for m in _SPLIT.finditer(par):
        end = m.start() + len(m.group(1))
        prev = par[start:end].rstrip(".!?…\"”’')]").split()
        last = prev[-1] if prev else ""
        lw = last.lower().strip("(\"“")
        if lw in ABBR or (len(last) == 1 and last.isupper()) or re.fullmatch(r"[A-ZĐ]\.?[A-ZĐ]?", last):
            continue
        out.append(par[start:end].strip()); start = m.end()
    out.append(par[start:].strip())
    return [s for s in out if s]


NUM = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")


def num_spans(s):
    """[(start, end, giá trị, loại)]; loại: nam (1000–2100), ngay (sau 'ngày'), thang (sau 'tháng', 1–12), so:<mẫu chữ số>."""
    out = []
    for m in NUM.finditer(s):
        v = m.group(0)
        before = s[:m.start()].rstrip().lower()
        if re.fullmatch(r"\d{4}", v) and 1000 <= int(v) <= 2100:
            t = "nam"
        elif before.endswith("ngày") and v.isdigit() and 1 <= int(v) <= 31:
            t = "ngay"
        elif before.endswith("tháng") and v.isdigit() and 1 <= int(v) <= 12:
            t = "thang"
        else:
            t = "so:" + re.sub(r"\d", "0", v)
        out.append((m.start(), m.end(), v, t))
    return out


_TOK = re.compile(r"\S+")
_NAMEWORD = re.compile("^[" + UP + "][" + LO + "]*$")


def name_spans(s):
    """Tên riêng: chuỗi >= 2 âm tiết liền nhau, mỗi âm tiết viết hoa chữ đầu (chỉ chữ cái), không đứng đầu câu.
    [(start, end, chuỗi, số âm tiết)]."""
    toks = []
    for m in _TOK.finditer(s):
        w = m.group(0)
        core = w.rstrip(",.;:)!?\"”’")
        toks.append((m.start(), m.start() + len(core), core, core != w))   # (đầu, cuối phần chữ, chữ, có dấu câu dính sau)
    out, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and _NAMEWORD.match(toks[j][2]):
            j += 1
            if toks[j - 1][3]:
                break                                                       # dấu câu sau âm tiết: hết tên
        nxt = toks[j][2] if j < len(toks) else ""
        ends_ok = toks[j - 1][3] or j == len(toks) or not nxt[:1].isalpha() or nxt[:1].isupper() or nxt.lower() in AFTER_OK
        if j - i >= 2 and i > 0 and ends_ok:
            out.append((toks[i][0], toks[j - 1][1], s[toks[i][0]:toks[j - 1][1]], j - i))
        i = max(j, i + 1)
    return out


def luoc(s):
    """Kiểu lược: bỏ phần trong ngoặc tròn; nếu không có ngoặc thì bỏ mệnh đề sau dấu phẩy cuối. Trả về (câu mới, cách) hoặc None
    nếu không lược được hoặc phần còn lại < 8 chữ."""
    if "(" in s and ")" in s:
        t = re.sub(r"\s*\([^()]*\)", "", s)
        t = re.sub(r"\s+([,.;:])", r"\1", t).strip()
        if t != s and len(t.split()) >= 8:
            return t, "bo_ngoac"
    if "," in s:
        body = s.rstrip()
        end = re.search(r"[.!?…][\"”’')\]]*$", body)
        tail = end.group(0) if end else "."
        core = body[:end.start()] if end else body
        k = core.rfind(", ")
        if k < 0:
            return None
        t = core[:k].rstrip() + (tail if tail.startswith((".", "!", "?", "…")) else ".")
        if len(t.split()) >= 8 and t.count("(") == t.count(")") and t.count("\"") % 2 == 0:
            return t, "bo_menh_de"
    return None


def wset(s):
    return set(normalize(s).split())


def overlap(claim, sent):
    a = wset(claim)
    return len(a & wset(sent)) / max(1, len(a))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def nfc(s):
    return unicodedata.normalize("NFC", s)


# ------------------------------------------------------------------ tham chiếu khử trùng (mục 4 của brief)
DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
TEST = ROOT / "data/bench_v1_3/test_v1.3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
TEST_SHA = "2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da"
RAW = ROOT / "results/train_v1/raw_refs"
MARKS = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")


def ref_texts():
    """[(văn bản, tag)] của mọi tham chiếu: dev, test 1.3 (chỉ dò, không in), proxy_v1, probe_v1/*.jsonl, mọi split ViWikiFC và ViFactCheck,
    Belebele, FLORES-101, SIB-200 tiếng Việt. Câu hỏi / lựa chọn lặp >= 3 lần trong một file coi là khung (như build_proxy_v1)."""
    import hashlib as _h
    from collections import Counter

    import pandas as pd
    assert _h.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA and _h.sha256(TEST.read_bytes()).hexdigest() == TEST_SHA
    out = []
    files = [("dev13", DEV), ("test13", TEST), ("proxy_v1", ROOT / "data/proxy_v1/proxy_v1.jsonl")]
    files += [("probe_v1", p) for p in sorted((ROOT / "results/probe_v1").glob("*.jsonl"))]
    for tag, f in files:
        rows = [r for r in read_jsonl(f) if r.get("lang", "vi") == "vi"]
        rep = Counter(normalize(t) for r in rows for t in [r.get("question") or ""] + list(r.get("options") or []))
        for r in rows:
            out.append((r.get("state") or "", tag))
            q = r.get("question") or ""
            for mk in MARKS:
                if mk in q:
                    out.append((q.split(mk, 1)[1], tag)); q = ""
            if q and rep[normalize(q)] < 3:
                out.append((q, tag))
            out += [(o, tag) for o in r.get("options") or [] if rep[normalize(o)] < 3]
    for sp in ("train", "dev", "test"):
        d = pd.read_csv(RAW / f"High-Will__ViWikiFC/data/{sp}.csv", dtype=str, keep_default_na=False)
        for c in ("context", "claim", "evidence"):
            out += [(t, "viwikifc") for t in d[c]]
        d = pd.read_parquet(RAW / f"tranthaihoa__vifactcheck/data/{sp}-00000-of-00001.parquet")
        for c in ("Context", "Statement", "Evidence"):
            out += [(t, "vifactcheck") for t in d[c] if isinstance(t, str)]
    for line in open(RAW / "facebook__belebele/data/vie_Latn.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out += [(r[k], "belebele") for k in ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4")]
    for sp in ("dev", "devtest"):
        out += [(t, "flores101") for t in pd.read_parquet(RAW / f"gsarti__flores_101/vie/{sp}/0000.parquet")["sentence"]]
    for sp in ("train", "dev", "test"):
        d = pd.read_csv(RAW / f"Davlan__sib200/data/vie_Latn/{sp}.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
        out += [(t, "sib200") for t in d["text"]]
    return [(t, tag) for t, tag in out if isinstance(t, str) and t.strip()]


class Refs:
    """Chỉ mục tham chiếu: decontam.Index (trùng, MinHash >= 0,8, chung 13-gram, văn bản tham chiếu 4–12 từ nằm trọn trong ứng viên)
    + chiều ngược (câu ứng viên 4–12 từ nằm trọn trong văn bản tham chiếu) + trùng chính xác văn bản < 4 từ."""

    def __init__(self):
        self.idx, self.short, blob, seen, self.n_by_tag = Index(), {}, [], set(), {}
        for t, tag in ref_texts():
            n = normalize(t)
            if not n or n in seen:
                continue
            seen.add(n); self.n_by_tag[tag] = self.n_by_tag.get(tag, 0) + 1
            if len(n.split()) >= 4:
                self.idx.add(t, tag); blob.append(f"{n}\t{tag}")
            else:
                self.short.setdefault(n, tag)
        self.blob = "\n" + "\n".join(blob) + "\n"

    def _tag_at(self, pos):
        e = self.blob.find("\n", pos)
        return self.blob[self.blob.rfind("\t", 0, e) + 1:e]

    def hit(self, text):
        """(loại, tag) của lần chạm đầu tiên, hoặc None."""
        n = normalize(text)
        if not n:
            return None
        k = len(n.split())
        if k < 4:
            return ("exact_short", self.short[n]) if n in self.short else None
        h = self.idx.check(text)
        if h:
            return h
        if k <= 12:
            for pat in (f"\n{n}\t", f"\n{n} ", f" {n} ", f" {n}\t"):
                p = self.blob.find(pat)
                if p >= 0:
                    return ("nguoc_contain_short", self._tag_at(p + 1))
        return None


# ------------------------------------------------------------------ luật chất lượng KC v2 (đọc mù vòng 1, results/train_2b/DIEU_PHOI_doc_mu_vong1.md)
MO_MENH_DE = ("và", "nhưng", "trong khi", "khi", "sau khi", "trước khi", "với", "để", "do", "vì", "nhằm", "cụ thể là", "bao gồm", "trong đó", "rồi")
HU_TU_CUOI = {"thuộc", "của", "là", "và", "ở", "tại", "với", "cho", "từ", "đến", "về"}
TRICH_DAN = re.compile(r"ISBN|\beds?\.|\bPress\b|\bNXB\b|nhà xuất bản|\btr\.|\bpp\.|truy cập|bản gốc|lưu trữ", re.I)
_VI_DAU = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
CAN = "Giáp Ất Bính Đinh Mậu Kỷ Canh Tân Nhâm Quý".split()
CHI = "Tý Sửu Dần Mão Thìn Tỵ Ngọ Mùi Thân Dậu Tuất Hợi".split()
CAN_CHI = re.compile(r"\b(" + "|".join(CAN) + r") (" + "|".join(CHI) + r")\b")


def ti_le_am_tiet_dau(s):
    toks = [t for t in re.findall(r"\w+", s) if t.isalpha()]
    return sum(any(c.lower() in _VI_DAU for c in t) for t in toks) / max(1, len(toks))


def cau_dung_duoc(s):
    """Câu dùng làm nhận định (v2): không phải mảnh trích dẫn thư mục, tỉ lệ âm tiết mang dấu tiếng Việt >= 0,3."""
    return not TRICH_DAN.search(s) and ti_le_am_tiet_dau(s) >= 0.3


def _end_ok(t):
    w = re.sub(r"[.!?…\"”’')\]]+$", "", t).split()
    return bool(w) and w[-1].lower() not in HU_TU_CUOI


def luoc_v2(s):
    """Lược (v2): bỏ ngoặc tròn cân, không lồng, có dấu cách đứng trước (không bỏ ngoặc dính trong một từ như "Bis(…)"), câu sau khi bỏ
    không kết thúc bằng hư từ; nếu không thì bỏ mệnh đề sau dấu phẩy cuối mà ngay sau ", " là từ mở đầu mệnh đề phụ (MO_MENH_DE).
    Phần còn lại >= 8 chữ. Trả về (câu mới, cách) hoặc None."""
    if re.search(r" \([^()]*\)", s):
        t = re.sub(r" \([^()]*\)", "", s)
        t = re.sub(r"\s+([,.;:])", r"\1", t).strip()
        if t != s and len(t.split()) >= 8 and "(" not in t and ")" not in t and _end_ok(t):
            return t, "bo_ngoac"
    end = re.search(r"[.!?…][\"”’')\]]*$", s.rstrip())
    tail = end.group(0) if end else "."
    body = s.rstrip()[:end.start()] if end else s.rstrip()
    cut = None
    for m in re.finditer(r", ", body):
        rest = body[m.end():].lower()
        if any(rest == w or rest.startswith(w + " ") for w in MO_MENH_DE):
            cut = m.start()
    if cut is None:
        return None
    t = body[:cut].rstrip() + (tail if tail.startswith((".", "!", "?", "…")) else ".")
    if len(t.split()) >= 8 and t.count("(") == t.count(")") and t.count("\"") % 2 == 0 and _end_ok(t):
        return t, "bo_menh_de"
    return None
