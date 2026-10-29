"""Phân tích lỗi Bác bỏ ViWikiFC theo dòng (sau REPORT_2b_bac2.md; người dùng yêu cầu 29/10). Chỉ CPU, chỉ đọc preds đã có; không chấm
model, không đọc test. Dữ liệu UIT chỉ dùng nội bộ.

Dòng phân tích: dev 1.3, ViWikiFC, gold "refuted" (choice có dấu là chính; kèm không dấu và Có/Không). Mô hình: C-s1, 2b-s1, 2b-s2.
Đặc trưng mỗi dòng (từ đã chuẩn hoá decontam.normalize, như kiem_kc_v2.py):
  max_ov  = max theo câu của state |từ nhận định ∩ từ câu| / |từ nhận định|; cover = |từ nhận định ∩ từ state| / |từ nhận định|;
  phu_dinh_nd / phu_dinh_cau = nhận định / câu trùng nhiều nhất có từ phủ định (không, chưa, chẳng, chả, đâu có, không hề…);
  lech_phu_dinh = đúng một bên có phủ định; so_la = có số trong nhận định không xuất hiện trong state; ten_la = có tên riêng (>= 2 âm tiết
  viết hoa) trong nhận định không xuất hiện trong state; do_dai = số từ nhận định.
So sánh với dòng Bác bỏ trong train_2b_v3.jsonl (KC theo kiểu dựng, vitaminc, nli_fever) trên cùng đặc trưng.
Ra results/train_2b/tam/bacbo_dong.jsonl (từng dòng, nội bộ) và results/train_2b/tam/phan_tich_bacbo.json; in tóm tắt.
Dùng: bash setup/run.sh -I scripts/train_2b/phan_tich_bacbo.py"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import DEV, OUT, ROOT, TMP, name_spans, normalize, num_spans, overlap, read_jsonl, sentences, wset, write_jsonl  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
EV = ROOT / "results/probe_v1/eval/dev"
MODELS = ("C-s1", "2b-s1", "2b-s2")
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
NEG = re.compile(r"\b(không|chưa|chẳng|chả|khong|chua|chang|cha)\b", re.I)
LAB_OPT = {"Được văn bản xác nhận": "supported", "Bị văn bản bác bỏ": "refuted", "Văn bản không đủ thông tin để kết luận": "nei",
           "Duoc van ban xac nhan": "supported", "Bi van ban bac bo": "refuted", "Van ban khong du thong tin de ket luan": "nei",
           "Có": "co", "Không": "khong", "Co": "co", "Khong": "khong"}


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def feats(r):
    c, st = claim_of(r), r["state"] or ""
    ss = sentences(st) or [st]
    best = max(ss, key=lambda s: overlap(c, s))
    nst = normalize(st)
    nums = [v for _, _, v, _ in num_spans(c)]
    names = [x for _, _, x, _ in name_spans(c)]
    return {"max_ov": round(overlap(c, best), 3), "cover": round(len(wset(c) & wset(st)) / max(1, len(wset(c))), 3),
            "phu_dinh_nd": bool(NEG.search(c)), "phu_dinh_cau": bool(NEG.search(best)), "lech_phu_dinh": bool(NEG.search(c)) != bool(NEG.search(best)),
            "so_la": any(normalize(v) not in nst for v in nums), "co_so": bool(nums), "ten_la": any(normalize(x) not in nst for x in names),
            "do_dai": len(c.split()), "n_cau_state": len(ss), "nhan_dinh": c, "cau_gan_nhat": best}


def bin_ov(x):
    return "<0,5" if x < 0.5 else "0,5–0,7" if x < 0.7 else "0,7–0,9" if x < 0.9 else "≥0,9"


dev = read_jsonl(DEV)
P = {m: {json.loads(l)["id"]: json.loads(l) for l in open(EV / m / "preds.jsonl", encoding="utf-8")} for m in MODELS}


def label(m, r):
    return LAB_OPT[r["options"][P[m][r["id"]]["pred"]]]


def prob(m, r, lab):
    i = next(i for i, o in enumerate(r["options"]) if LAB_OPT[o] == lab)
    return P[m][r["id"]]["probs"][i]


rows_out, res = [], {}
for var in ("orig", "nodiac"):
    for kind in ("choice", "noul"):
        for gold in ("refuted", "supported", "nei"):
            rs = [r for r in dev if r["source"] == "viwikifc" and r["variant"] == var and r["kind"] == kind and r["gold_label"] == gold]
            if not rs:
                continue
            key = f"{var}|{kind}|{gold}"
            o = {"n": len(rs), "acc": {m: round(sum(P[m][r["id"]]["correct"] for r in rs) / len(rs), 3) for m in MODELS},
                 "du_doan": {m: dict(Counter(label(m, r) for r in rs)) for m in MODELS}}
            if kind == "choice":
                o["p_tb"] = {m: {lab: round(sum(prob(m, r, lab) for r in rs) / len(rs), 3) for lab in ("supported", "refuted", "nei")} for m in MODELS}
            res[key] = o
            if var == "orig" and kind == "choice":
                for r in rs:
                    f = feats(r)
                    rows_out.append({"id": r["id"], "gold": gold, **{f"dung_{m}": bool(P[m][r["id"]]["correct"]) for m in MODELS},
                                     **{f"doan_{m}": label(m, r) for m in MODELS},
                                     **{f"p_{m}": {lab: round(prob(m, r, lab), 3) for lab in ("supported", "refuted", "nei")} for m in MODELS}, **f})

ref = [x for x in rows_out if x["gold"] == "refuted"]
# 1. chuyển trạng thái đúng/sai C → s1 → s2
res["mau_dung_sai_C_s1_s2"] = dict(Counter("".join("Đ" if x[f"dung_{m}"] else "S" for m in MODELS) for x in ref))
mat = [x for x in ref if x["dung_C-s1"] and not x["dung_2b-s1"] and not x["dung_2b-s2"]]
res["mat_ca_hai_seed"] = {"n": len(mat), "doan_s1": dict(Counter(x["doan_2b-s1"] for x in mat)), "doan_s2": dict(Counter(x["doan_2b-s2"] for x in mat))}


# 2. tỉ lệ sai theo đặc trưng, từng mô hình, cho Bác bỏ và Xác nhận (choice có dấu)
def by(rows, keyf):
    g = defaultdict(list)
    for x in rows:
        g[keyf(x)].append(x)
    return {k: {"n": len(v), **{m: round(sum(x[f"dung_{m}"] for x in v) / len(v), 3) for m in MODELS},
                "doan_NEI_s2": sum(x["doan_2b-s2"] == "nei" for x in v)} for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


res["theo_dac_trung"] = {}
for gold in ("refuted", "supported", "nei"):
    rs = [x for x in rows_out if x["gold"] == gold]
    res["theo_dac_trung"][gold] = {"max_ov": by(rs, lambda x: bin_ov(x["max_ov"])), "phu_dinh_nd": by(rs, lambda x: x["phu_dinh_nd"]),
                                   "lech_phu_dinh": by(rs, lambda x: x["lech_phu_dinh"]), "so_la": by(rs, lambda x: x["so_la"]),
                                   "ten_la": by(rs, lambda x: x["ten_la"])}

# 3. đặc trưng của dòng Bác bỏ trong train_2b_v3 (có dấu, choice + Có/Không) theo nguồn / kiểu dựng
kieu = {r["id"]: r.get("kieu") for f in ("kc_pool_ngan_v2.jsonl", "kc_pool_dai_v2.jsonl") for r in read_jsonl(OUT / f)}
tr = [r for r in read_jsonl(OUT / "train_2b_v3.jsonl") if r.get("gold_label") == "refuted" and r.get("variant", "orig") != "nodiac"
      and not any(mk in r["question"] for mk in ("Nhan dinh: ", "Phat bieu: "))]
grp = defaultdict(list)
for r in tr:
    try:
        f = feats(r)
    except StopIteration:
        continue
    g = r["source"] if not r["source"].startswith("kc_") else f"{r['source']}|{kieu.get(r['id'])}"
    grp[g].append(f)
grp["DEV viwikifc (choice có dấu)"] = [x for x in ref]


def summ(fs):
    n = len(fs)
    ov = sorted(f["max_ov"] for f in fs)
    return {"n": n, "max_ov_trung_vi": ov[n // 2] if n else None, "ti_le_max_ov<0,7": round(sum(v < 0.7 for v in ov) / n, 3),
            "phu_dinh_nd": round(sum(f["phu_dinh_nd"] for f in fs) / n, 3), "lech_phu_dinh": round(sum(f["lech_phu_dinh"] for f in fs) / n, 3),
            "so_la": round(sum(f["so_la"] for f in fs) / n, 3), "ten_la": round(sum(f["ten_la"] for f in fs) / n, 3),
            "do_dai_trung_vi": sorted(f["do_dai"] for f in fs)[n // 2]}


res["train_vs_dev_bac_bo"] = {g: summ(v) for g, v in sorted(grp.items()) if v}

# 4. Dòng NEI của KC theo kiểu dựng (cùng đặc trưng), để so với dòng Bác bỏ của dev
gn = defaultdict(list)
trn = [r for r in read_jsonl(OUT / "train_2b_v3.jsonl") if r["source"].startswith("kc_") and r.get("gold_label") == "nei"
       and not any(mk in r["question"] for mk in ("Nhan dinh: ", "Phat bieu: "))]
for r in trn:
    gn[f"{r['source']}|{kieu.get(r['id'])}"].append(feats(r))
res["kc_nei_theo_kieu"] = {g: summ(v) for g, v in sorted(gn.items())}

# 5. Probe bề mặt: logistic regression (numpy, chuẩn hoá) phân biệt KC Bác bỏ với KC NEI chỉ bằng đặc trưng trùng từ / phủ định / số / tên;
# áp lên dev ViWikiFC choice có dấu. Câu hỏi: dòng Bác bỏ dev có bị probe xếp vào phía "NEI của KC" không, và dòng đó 2b có sai nhiều hơn không.
import numpy as np  # noqa: E402

FK = ("max_ov", "cover", "phu_dinh_nd", "lech_phu_dinh", "so_la", "ten_la", "do_dai")
vec = lambda f: [float(f[k]) for k in FK]
Xr = [vec(f) for g, v in grp.items() if g.startswith("kc_") for f in v]
Xn = [vec(f) for v in gn.values() for f in v]
X = np.array(Xr + Xn); y = np.array([1] * len(Xr) + [0] * len(Xn))
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1])
for _ in range(3000):
    p = 1 / (1 + np.exp(-Z @ w)); w -= 0.1 * (Z.T @ (p - y) / len(y) + 1e-3 * np.r_[0, w[1:]])
acc_train = float(((1 / (1 + np.exp(-Z @ w)) > 0.5) == y).mean())
score = lambda f: float(1 / (1 + np.exp(-(np.r_[1, (np.array(vec(f)) - mu) / sd] @ w))))
pr = {}
for gold in ("refuted", "supported", "nei"):
    rs = [x for x in rows_out if x["gold"] == gold]
    nei_side = [x for x in rs if score(x) < 0.5]; ref_side = [x for x in rs if score(x) >= 0.5]
    pr[gold] = {"n": len(rs), "phia_NEI_cua_KC": len(nei_side),
                **{f"acc_{m}": {"phia_NEI": round(sum(x[f"dung_{m}"] for x in nei_side) / max(1, len(nei_side)), 3),
                                "phia_BacBo": round(sum(x[f"dung_{m}"] for x in ref_side) / max(1, len(ref_side)), 3)} for m in MODELS},
                "doan_NEI_s2": {"phia_NEI": sum(x["doan_2b-s2"] == "nei" for x in nei_side), "phia_BacBo": sum(x["doan_2b-s2"] == "nei" for x in ref_side)}}
for x in rows_out:
    x["probe_p_bacbo_kc"] = round(score(x), 3)
res["probe_KC_BacBo_vs_NEI"] = {"dac_trung": FK, "trong_so_chuan_hoa": dict(zip(("chan",) + FK, [round(float(v), 3) for v in w])),
                                "acc_tren_KC": round(acc_train, 3), "n_KC": {"bac_bo": len(Xr), "nei": len(Xn)}, "ap_len_dev": pr}
write_jsonl(TMP / "bacbo_dong.jsonl", rows_out)
(TMP / "phan_tich_bacbo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1))
