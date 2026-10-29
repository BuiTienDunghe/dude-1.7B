"""So sánh 200 dòng dịch thử Sonnet với bản agent ngoài (Google Translate), BƯỚC CHUẨN BỊ (chỉ thư viện chuẩn).
Ngoại lệ có phạm vi (người dùng, 19/10): chỉ ĐỌC results/train_v1/translated/ban_dich.jsonl để lấy đúng 200 id của mẫu; không dùng để train / làm nháp;
không sửa gì trong thư mục đó.
- Trích 200 dòng của bản Google -> pilot/so_sanh/google_200.jsonl (chỉ để so sánh).
- Kiểm khách quan cả hai bản bằng kiem_ban_dich.py; đếm lỗi/cảnh báo theo loại; độ nhất quán chuỗi lặp (khóa bảng thuật ngữ có trong mẫu:
  số cách dịch khác nhau ở mỗi bản); cách dịch "None of the other options ." / "None of the above choices .".
- Chấm mù: mỗi dòng ghép EN + bản X + bản Y (X/Y xáo theo seed 20261019, không ghi nguồn) + đáp án đúng (từ part_b_meta.jsonl); 5 lô 40 dòng
  -> pilot/so_sanh/cham/lo_{k}_in.jsonl; bảng giải mã -> pilot/so_sanh/cham_map.json (người chấm không được đọc).
Ra: pilot/so_sanh/kiem_khach_quan.json."""
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
P = T / "translated_sonnet/pilot"
S = P / "so_sanh"
(S / "cham").mkdir(parents=True, exist_ok=True)
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
src = {r["id"]: r for r in rd(P / "src_200.jsonl")}
order = json.load(open(P / "ids.json", encoding="utf-8"))["thu_tu"]
son = {r["id"]: r for r in rd(P / "ban_s2_200.jsonl")}
goo = {}
with open(T / "translated/ban_dich.jsonl", encoding="utf-8") as f:     # chỉ lấy 200 id của mẫu
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r.get("id") in src:
                goo[r["id"]] = r
with open(S / "google_200.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for i in order:
        if i in goo:
            f.write(json.dumps(goo[i], ensure_ascii=False) + "\n")
meta = {r["id"]: r for r in rd(T / "part_b_meta.jsonl") if r["id"] in src}
fmt = json.load(open(ROOT / "config/train_format.json", encoding="utf-8"))

# ---------------------------------------------------------------- kiểm khách quan
def loai(msg):
    m = re.search(r": (.*)$", msg)
    s = m.group(1) if m else msg
    s = re.sub(r"option:\d+", "option", s)
    s = re.sub(r"\(.*?\)|\[.*?\]|'.*?'|\".*?\"|\d+", "", s)
    return re.sub(r"\s+", " ", s).strip()[:60]


kq = {"thieu_id_google": sorted(set(src) - set(goo))}
for tag, name in (("sonnet", P / "ban_s2_200.jsonl"), ("google", S / "google_200.jsonl")):
    subprocess.run([sys.executable, str(T / "to_translate/kiem_ban_dich.py"), str(P / "src_200.jsonl"), str(name), "--report", str(S / f"kiem_{tag}.json")],
                   capture_output=True, text=True, encoding="utf-8")
    k = json.load(open(S / f"kiem_{tag}.json", encoding="utf-8"))
    kq[tag] = {"ket_luan": k["ket_luan"], "loi": k["loi"], "canh_bao": k["canh_bao"],
               "loi_theo_loai": dict(Counter(loai(x) for x in k["chi_tiet_loi"])), "canh_bao_theo_loai": dict(Counter(loai(x) for x in k["chi_tiet_canh_bao"]))}
gloss = json.load(open(P / "bang_thuat_ngu.json", encoding="utf-8"))
nhat_quan = {}
for tag, ver in (("sonnet", son), ("google", goo)):
    m = defaultdict(set)
    for i, r in src.items():
        if i not in ver:
            continue
        v = ver[i]
        if r["question"] in gloss:
            m[r["question"]].add(v.get("question", "").strip())
        for x, y in zip(r["options"], v.get("options", [])):
            if x in gloss:
                m[x].add(y.strip())
    nhat_quan[tag] = {"chuoi_lap_co_trong_mau": len(m), "chuoi_dich_hon_1_kieu": sum(1 for s in m.values() if len(s) > 1),
                      "vi_du": {k: sorted(s)[:4] for k, s in m.items() if len(s) > 1}}
kq["nhat_quan_chuoi_lap"] = nhat_quan
none = {}
for key in ("None of the other options .", "None of the above choices ."):
    for tag, ver in (("sonnet", son), ("google", goo)):
        none.setdefault(key, {})[tag] = dict(Counter(ver[i]["options"][k].strip() for i, r in src.items() if i in ver
                                                      for k, x in enumerate(r["options"]) if x == key))
kq["none_of_the_other_options"] = none
(S / "kiem_khach_quan.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- lô chấm mù
rng = random.Random(20261019)
mp = {}
items = []
for n, i in enumerate(order, 1):
    if i not in goo:
        continue
    r, m = src[i], meta[i]
    x_is = "sonnet" if rng.random() < 0.5 else "google"
    X, Y = (son[i], goo[i]) if x_is == "sonnet" else (goo[i], son[i])
    if m["kind"] == "noul":
        p = m["target"][0]
        dap_an = f"Câu hỏi Có/Không (lựa chọn cố định Có/Không, không cần dịch). Đáp án đúng: {'Có (Yes)' if p >= 0.5 else 'Không (No)'}" + \
                 (f", xác suất Có = {p:.2f}" if 0 < p < 1 else "")
    else:
        t = m["target"]; b = max(range(len(t)), key=lambda k: t[k])
        dap_an = f"Lựa chọn đúng: số {b + 1} (đếm từ 1): {r['options'][b]!r}" + (f"; phân phối {t}" if max(t) < 1 else "")
    cid = f"c{n:03d}"
    mp[cid] = {"id": i, "X": x_is, "Y": "google" if x_is == "sonnet" else "sonnet", "dang_bai": m["dang_bai"]}
    pick = lambda v: {"state": v.get("state", ""), "question": v.get("question", ""), "options": v.get("options", [])}
    items.append({"id_cham": cid, "en": pick(r), "X": pick(X), "Y": pick(Y), "dap_an": dap_an})
for k in range(0, len(items), 40):
    with open(S / f"cham/lo_{k // 40 + 1}_in.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items[k:k + 40]:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
(S / "cham_map.json").write_text(json.dumps(mp, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"google_co": len(goo), "lo_cham": (len(items) + 39) // 40, "X_la_sonnet": sum(v["X"] == "sonnet" for v in mp.values()),
                  "kiem": {t: {k: v for k, v in kq[t].items()} for t in ("sonnet", "google")},
                  "nhat_quan": {t: {k: v for k, v in nhat_quan[t].items() if k != "vi_du"} for t in nhat_quan}, "none": none}, ensure_ascii=False))
