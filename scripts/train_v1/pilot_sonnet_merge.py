"""Dịch thử 200 dòng bằng Sonnet: GHÉP + KIỂM (chỉ thư viện chuẩn).
Đọc results/train_v1/translated_sonnet/pilot/lo/lo_{k}_vi_s1.jsonl (Sonnet 1 dịch), lo_{k}_vi_s2.jsonl (Sonnet 2 đọc lại, bản sau sửa),
lo_{k}_sua.jsonl (nhật ký sửa của Sonnet 2). Ghi: ban_s1_200.jsonl (trước sửa), ban_s2_200.jsonl (sau sửa = bản dùng), sua_200.jsonl,
kiem_s1.json / kiem_s2.json (kiem_ban_dich.py), thong_ke_pilot.json (sửa theo dạng bài / loại lỗi; đối chiếu nhật ký với khác biệt thật)."""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
P = T / "translated_sonnet/pilot"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
meta = {r["id"]: r for r in rd(T / "part_b_meta.jsonl")}
src = {r["id"]: r for r in rd(P / "src_200.jsonl")}
lots = sorted(P.glob("lo/lo_*_en.jsonl"), key=lambda p: int(p.name.split("_")[1]))
s1, s2, sua = [], [], []
for p in lots:
    k = p.name.split("_")[1]
    s1 += rd(P / f"lo/lo_{k}_vi_s1.jsonl"); s2 += rd(P / f"lo/lo_{k}_vi_s2.jsonl")
    f = P / f"lo/lo_{k}_sua.jsonl"
    sua += [{**x, "lo": int(k)} for x in rd(f)] if f.exists() else []
for name, rows in (("ban_s1_200.jsonl", s1), ("ban_s2_200.jsonl", s2), ("sua_200.jsonl", sua)):
    with open(P / name, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
kiem = {}
for tag, name in (("s1", "ban_s1_200.jsonl"), ("s2", "ban_s2_200.jsonl")):
    rc = subprocess.run([sys.executable, str(T / "to_translate/kiem_ban_dich.py"), str(P / "src_200.jsonl"), str(P / name), "--report", str(P / f"kiem_{tag}.json")],
                        capture_output=True, text=True, encoding="utf-8")
    kiem[tag] = json.load(open(P / f"kiem_{tag}.json", encoding="utf-8"))
a, b = {r["id"]: r for r in s1}, {r["id"]: r for r in s2}
diff = defaultdict(list)
for i in b:
    for fld in ("state", "question"):
        if a[i][fld] != b[i][fld]:
            diff[i].append(fld)
    for k, (x, y) in enumerate(zip(a[i]["options"], b[i]["options"])):
        if x != y:
            diff[i].append(f"option:{k}")
logged = {x["id"] for x in sua}
st = {
    "dong": len(b), "dong_s2_sua_that": len(diff), "dong_co_nhat_ky": len(logged), "muc_nhat_ky": len(sua),
    "sua_khong_ghi_nhat_ky": sorted(set(diff) - logged), "nhat_ky_khong_co_khac_biet": sorted(logged - set(diff)),
    "dong_sua_theo_dang_bai": dict(Counter(meta[i]["dang_bai"] for i in diff)),
    "dong_theo_dang_bai": dict(Counter(meta[i]["dang_bai"] for i in b)),
    "muc_sua_theo_loai_loi": dict(Counter(x.get("loai_loi", "?") for x in sua)),
    "dong_sua_theo_dang_bai_loai_loi": {g: dict(Counter(x.get("loai_loi", "?") for x in sua if meta[x["id"]]["dang_bai"] == g)) for g in sorted({meta[i]["dang_bai"] for i in b})},
    "truong_sua": dict(Counter(f.split(":")[0] for v in diff.values() for f in v)),
    "ky_tu_en": sum(len(r["state"]) + len(r["question"]) + sum(map(len, r["options"])) for r in src.values()),
    "ky_tu_vi": sum(len(r["state"]) + len(r["question"]) + sum(map(len, r["options"])) for r in s2),
    "kiem": {t: {k: v for k, v in x.items() if not k.startswith("chi_tiet")} for t, x in kiem.items()},
}
(P / "thong_ke_pilot.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in st.items() if k not in ("sua_khong_ghi_nhat_ky", "nhat_ky_khong_co_khac_biet")}, ensure_ascii=False))
