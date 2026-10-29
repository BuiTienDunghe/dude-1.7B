"""Ghép bản dịch đầy đủ phần B (chỉ thư viện chuẩn): pilot giữ lại (full/pilot_giu.jsonl) + s2 của mọi lô, theo đúng thứ tự file dịch.
Ra (results/train_v1/translated_sonnet/full/): ban_dich_s1.jsonl (trước đọc lại; dòng pilot giữ lại lấy bản pilot s1), ban_dich_s2.jsonl (bản dùng),
sua_tat_ca.jsonl, kiem_s2.json (kiem_ban_dich v2 trên 6.944 dòng), thong_ke_day_du.json (dòng sửa theo dạng bài / loại; đối chiếu nhật ký với khác biệt)."""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
O = T / "translated_sonnet/full"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
src = rd(T / "to_translate/part_b_to_translate.jsonl")
meta = {r["id"]: r for r in rd(T / "part_b_meta.jsonl")}
plan = json.load(open(O / "plan.json", encoding="utf-8"))
lots = [l for d in plan["dot"].values() for l in d]
s1, s2, sua = {}, {}, []
for r in rd(O / "pilot_giu.jsonl"):
    s2[r["id"]] = r
for r in rd(T / "translated_sonnet/pilot/ban_s1_200.jsonl"):
    if r["id"] in s2:
        s1[r["id"]] = r
for x in rd(T / "translated_sonnet/pilot/sua_200.jsonl"):
    if x["id"] in s2:
        sua.append({**x, "lo": "pilot"})
for lo in lots:
    for r in rd(O / f"lo/lo_{lo}_vi_s1.jsonl"):
        s1[r["id"]] = r
    for r in rd(O / f"lo/lo_{lo}_vi_s2.jsonl"):
        s2[r["id"]] = r
    f = O / f"lo/lo_{lo}_sua.jsonl"
    sua += [{**x, "lo": lo} for x in rd(f)] if f.exists() else []
miss = [r["id"] for r in src if r["id"] not in s2]
assert not miss, (len(miss), miss[:3])
for name, d in (("ban_dich_s1.jsonl", s1), ("ban_dich_s2.jsonl", s2), ("sua_tat_ca.jsonl", None)):
    with open(O / name, "w", encoding="utf-8", newline="\n") as f:
        for r in (sua if d is None else [d[x["id"]] for x in src]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
rc = subprocess.run([sys.executable, str(T / "to_translate/kiem_ban_dich.py"), str(T / "to_translate/part_b_to_translate.jsonl"), str(O / "ban_dich_s2.jsonl"),
                     "--report", str(O / "kiem_s2.json")], capture_output=True, text=True, encoding="utf-8")
k = json.load(open(O / "kiem_s2.json", encoding="utf-8"))
diff = defaultdict(list)
for r in src:
    a, b = s1[r["id"]], s2[r["id"]]
    diff[r["id"]] += [f for f in ("state", "question") if a[f] != b[f]] + [f"option:{j}" for j, (x, y) in enumerate(zip(a["options"], b["options"])) if x != y]
changed = {i for i, v in diff.items() if v}
st = {"dong": len(src), "dong_sua": len(changed), "muc_sua": len(sua), "dong_sua_khong_ghi_nhat_ky": len(changed - {x["id"] for x in sua}),
      "dong_sua_theo_dang_bai": dict(Counter(meta[i]["dang_bai"] for i in changed)), "dong_theo_dang_bai": dict(Counter(meta[r["id"]]["dang_bai"] for r in src)),
      "muc_sua_theo_loai": dict(Counter(x.get("loai_loi", "?") for x in sua)),
      "muc_sua_theo_dang_bai_loai": {g: dict(Counter(x.get("loai_loi", "?") for x in sua if meta[x["id"]]["dang_bai"] == g)) for g in sorted({m["dang_bai"] for m in meta.values()})},
      "kiem": {kk: vv for kk, vv in k.items() if not kk.startswith("chi_tiet")}, "kiem_ma_thoat": rc.returncode}
(O / "thong_ke_day_du.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(st, ensure_ascii=False))
