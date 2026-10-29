"""Ba cổng trước khi push `_github/dude-1.7B/`. Trượt cổng nào thì không push.

1. Rò rỉ test: chuỗi riêng của mọi file test đã đóng băng (state, câu hỏi, lựa chọn; ≥ 25 ký tự; không có trong dev cùng bản
   hay trong config) không được xuất hiện trong file xuất. Chỉ in số lần chạm theo file, KHÔNG in nội dung (như decontam).
2. Bí mật: không có chuỗi giống token/khóa.
3. Kích thước: không file nào > 10 MB, tổng < 50 MB.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "_github" / "dude-1.7B"
BENCH = [("bench_v1", "test_v1.0.jsonl", "dev_v1.jsonl"), ("bench_v1_1", "test_v1.1.jsonl", "dev_v1_1.jsonl"),
         ("bench_v1_2", "test_v1.2.jsonl", "dev_v1_2.jsonl"), ("bench_v1_3", "test_v1.3.jsonl", "dev_v1_3.jsonl")]
SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                    r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,})", re.I)


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def chuoi(r):
    out = [r.get("state"), r.get("question")] + list(r.get("options") or [])
    return [norm(x) for x in out if isinstance(x, str)]


def main():
    dev_set, test_meta = set(), {}
    for d, t, v in BENCH:
        for line in open(ROOT / "data" / d / v, encoding="utf-8"):
            dev_set.update(chuoi(json.loads(line)))
        for line in open(ROOT / "data" / d / t, encoding="utf-8"):
            r = json.loads(line)
            for kieu, x in [("state", r.get("state")), ("question", r.get("question"))] + [("option", o) for o in r.get("options") or []]:
                if isinstance(x, str):
                    test_meta.setdefault(norm(x), (d, r.get("source"), kieu))
    for p in (ROOT / "config").glob("*.json"):
        dev_set.add(norm(p.read_text(encoding="utf-8")))
    dev_text = "\n".join(dev_set)  # chuỗi mẫu nằm trong dev/config (kể cả là chuỗi con) thì không phải nội dung riêng của test
    rieng = {s for s in test_meta if len(s) >= 25 and s not in dev_set}
    mau = {}
    for s in rieng:  # chuỗi dài: dò 3 đoạn 60 ký tự (đầu, giữa, cuối)
        if len(s) <= 80:
            parts = [s]
        else:
            m = len(s) // 2
            parts = [s[:60], s[m - 30:m + 30], s[-60:]]
        for x in parts:
            if x not in dev_text:
                mau.setdefault(x, test_meta[s])
    print(f"chuỗi riêng của test: {len(rieng)}; mẫu dò (không nằm trong dev/config): {len(mau)}")

    files = [p for p in DEST.rglob("*") if p.is_file() and ".git" not in p.parts]
    cham, bimat, to = {}, {}, []
    tong = 0
    for p in files:
        sz = p.stat().st_size
        tong += sz
        if sz > 10e6:
            to.append(str(p.relative_to(DEST)))
        try:
            text = norm(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        hit = [meta for s, meta in mau.items() if s in text]
        if hit:
            cham[str(p.relative_to(DEST))] = hit
        k = len(SECRET.findall(text))
        if k:
            bimat[str(p.relative_to(DEST))] = k
    print(f"file: {len(files)}, tổng {tong / 1e6:.2f} MB")
    print(f"[1] rò rỉ test: {sum(len(v) for v in cham.values())} lần chạm ở {len(cham)} file (chỉ in bản / nguồn / loại trường)")
    for f, hit in sorted(cham.items()):
        c = {}
        for meta in hit:
            c["/".join(str(x) for x in meta)] = c.get("/".join(str(x) for x in meta), 0) + 1
        print(f"    {f}: {len(hit)}  {c}")
    print(f"[2] bí mật: {sum(bimat.values())} ở {len(bimat)} file")
    for f, n in sorted(bimat.items()):
        print(f"    {f}: {n}")
    print(f"[3] file > 10 MB: {to}; tổng < 50 MB: {tong < 50e6}")
    ok = not cham and not bimat and not to and tong < 50e6
    print("KẾT QUẢ:", "ĐẠT" if ok else "TRƯỢT — KHÔNG PUSH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
