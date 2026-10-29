"""Kiểm chữ nhãn cố định giữa một file train bất kỳ và DEV 1.3 (audit 27/10; bước 0 ngày 28/10). Không chạy model, không đọc test. Thư viện chuẩn.

Bộ chữ nhãn chuẩn lấy TỪ DEV 1.3 (không lấy từ config), theo từng khung:
  kiem_chung  : lựa chọn của câu choice ViFactCheck / ViWikiFC ("Được văn bản xác nhận", "Bị văn bản bác bỏ", "Văn bản không đủ thông tin để kết luận");
  co_khong    : lựa chọn của câu noul (Có / Không);
  sib200      : 7 tên chủ đề SIB-200;
  massive     : 60 mô tả ý định MASSIVE;
mỗi khung tách theo biến thể dev (vi|orig, vi|nodiac, en|orig).
Với mỗi lựa chọn trong file train (chuẩn hóa: NFC, chữ thường, gộp khoảng trắng, bỏ dấu câu ở hai đầu):
  trung      : trùng đúng một nhãn dev;
  gan_trung  : KHÔNG trùng nhưng gần một nhãn dev: chứa nhau (chuỗi ngắn >= 2 từ nằm trong chuỗi dài <= 10 từ; bỏ câu dài tình cờ chứa một
               từ như "health"), hoặc giống nhau >= 0,9 (difflib) khi độ dài lệch <= 20%. Đây là dấu hiệu chữ nhãn lệch dev
               (ví dụ "Văn bản không đủ thông tin" so với "Văn bản không đủ thông tin để kết luận").
Báo theo nguồn của file train (trường source) và khung. --strict: thoát mã 1 nếu có gan_trung.
Dùng: python scripts/check_label_strings.py <file_train.jsonl> [--out <báo cáo.json>] [--strict]"""
import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"

ap = argparse.ArgumentParser()
ap.add_argument("train")
ap.add_argument("--out", type=Path)
ap.add_argument("--strict", action="store_true")
a = ap.parse_args()

sys.path.insert(0, str(ROOT / "scripts"))
from common import read_jsonl, sha256_file  # noqa: E402

assert sha256_file(DEV) == DEV_SHA, "dev 1.3 sai sha"


def norm(s):
    s = " ".join(unicodedata.normalize("NFC", str(s)).lower().split())
    return re.sub(r"^[\s.,;:!?\"'()\-]+|[\s.,;:!?\"'()\-]+$", "", s)


def frame(r):
    if r["source"] in ("vifactcheck", "viwikifc"):
        return "kiem_chung" if r["kind"] == "choice" else "co_khong"
    return r["source"] if r["source"] in ("sib200", "massive") else None


LAB = defaultdict(set)                      # (khung, biến thể) -> {nhãn chuẩn hóa}
RAW = {}                                    # nhãn chuẩn hóa -> chữ gốc trong dev
for r in read_jsonl(DEV):
    f = frame(r)
    if f:
        for o in r["options"]:
            LAB[(f, f"{r['lang']}|{r['variant']}")].add(norm(o)); RAW[norm(o)] = o
ALL = {lb: k for k, s in LAB.items() for lb in s}


def near(o):
    """Nhãn dev gần o (không trùng); None nếu không có."""
    for lb in ALL:
        short, long_ = (o, lb) if len(o) <= len(lb) else (lb, o)
        if len(short.split()) >= 2 and len(long_.split()) <= 10 and short in long_:
            return lb
        if abs(len(o) - len(lb)) <= 0.2 * max(len(o), len(lb)) and o[:4] == lb[:4] and difflib.SequenceMatcher(None, o, lb).ratio() >= 0.9:
            return lb
    return None


trung = defaultdict(Counter)                # nguồn train -> Counter((khung, biến thể, chữ dev))
gan = defaultdict(Counter)                  # nguồn train -> Counter((chữ train, chữ dev gần, khung))
n_rows = Counter()
cache = {}
for line in open(a.train, encoding="utf-8"):
    r = json.loads(line)
    src = r.get("source", "?")
    n_rows[src] += 1
    for o in r["options"]:
        k = norm(o)
        if k in ALL:
            f, v = ALL[k]
            trung[src][(f, v, RAW[k])] += 1
        elif len(k) <= 80:
            if k not in cache:
                cache[k] = near(k)
            lb = cache[k]
            if lb:
                gan[src][(o, RAW[lb], ALL[lb][0])] += 1

dev_cover = {f"{f}|{v}": {"so_nhan_dev": len(s), "so_nhan_co_trong_train": len({RAW[x] for x in s} & {t[2] for c in trung.values() for t in c})}
             for (f, v), s in sorted(LAB.items())}
res = {"file_train": str(a.train), "file_train_sha256": sha256_file(a.train), "dev_sha256": DEV_SHA,
       "so_dong_theo_nguon": dict(n_rows.most_common()),
       "trung_dung_chu_dev": {s: [{"khung": f, "bien_the": v, "chu": t, "so_lan": n} for (f, v, t), n in c.most_common()] for s, c in sorted(trung.items())},
       "gan_trung_LECH_dev": {s: [{"chu_train": t, "chu_dev_gan": d, "khung": f, "so_lan": n} for (t, d, f), n in c.most_common()] for s, c in sorted(gan.items())},
       "do_phu_nhan_dev": dev_cover}
tong_gan = sum(sum(c.values()) for c in gan.values())
res["tong"] = {"lan_trung": sum(sum(c.values()) for c in trung.values()), "lan_gan_trung_lech": tong_gan,
               "chuoi_gan_trung_khac_nhau": len({(t, d) for c in gan.values() for (t, d, f) in c})}
out = json.dumps(res, ensure_ascii=False, indent=1)
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out, encoding="utf-8")
print(json.dumps(res["tong"], ensure_ascii=False))
for s, c in sorted(gan.items()):
    for (t, d, f), n in c.most_common(5):
        print(f"GẦN TRÙNG [{s}] khung {f}: train {t!r} ≠ dev {d!r} ({n} lần)")
sys.exit(1 if (a.strict and tong_gan) else 0)
