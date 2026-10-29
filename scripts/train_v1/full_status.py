"""Tiến độ dịch đầy đủ phần B (chỉ thư viện chuẩn): với mỗi lô, đã có s1 / s2 / sua chưa và s1, s2 có ĐẠT kiem_ban_dich không (kiểm khi file có).
Dùng để chạy tiếp sau khi dừng. In: số lô theo trạng thái, đợt tiếp theo cần dịch / cần đọc lại. Ghi full/trang_thai.json.
Tham số --nhanh: chỉ xét có file và đủ dòng, không chạy kiem."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
O = T / "translated_sonnet/full"
plan = json.load(open(O / "plan.json", encoding="utf-8"))
fast = "--nhanh" in sys.argv
n = lambda p: sum(1 for l in open(p, encoding="utf-8") if l.strip()) if p.exists() else -1


def ok(en, vi):
    if fast:
        return n(vi) == n(en)
    r = subprocess.run([sys.executable, str(T / "to_translate/kiem_ban_dich.py"), str(en), str(vi)], capture_output=True, text=True, encoding="utf-8")
    return r.returncode == 0


st = {}
for d, lots in plan["dot"].items():
    for lo in lots:
        en = O / f"lo/lo_{lo}_en.jsonl"
        s1, s2, sua = (O / f"lo/lo_{lo}_{x}.jsonl" for x in ("vi_s1", "vi_s2", "sua"))
        st[lo] = {"dot": d, "s1": s1.exists() and ok(en, s1), "s2": s2.exists() and sua.exists() and ok(en, s2)}
need_s1 = [d for d, lots in plan["dot"].items() if not all(st[l]["s1"] for l in lots)]
need_s2 = [d for d, lots in plan["dot"].items() if all(st[l]["s1"] for l in lots) and not all(st[l]["s2"] for l in lots)]
out = {"lo_s1_xong": sum(v["s1"] for v in st.values()), "lo_s2_xong": sum(v["s2"] for v in st.values()), "so_lo": len(st),
       "dot_can_dich": need_s1, "dot_can_doc_lai": need_s2, "chi_tiet": st}
(O / "trang_thai.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "chi_tiet"}, ensure_ascii=False))
