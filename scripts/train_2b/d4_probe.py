"""Cổng D4 của results/BRIEF_2b_du_lieu.md: probe chỉ đọc nhận định (kiểu Gururangan) trên dòng choice của KC (kc_pool_ngan + kc_pool_dai).
Logistic regression trên n-gram ký tự 2–4 (char_wb) + từ 1–2 (TF-IDF), 5-fold phân tầng, seed 20261028. Đạt khi accuracy <= tỉ lệ nhãn
đa số + 10 điểm. Báo thêm: chỉ dòng có dấu, và 15 đặc trưng nặng nhất mỗi nhãn (fit trên toàn bộ) để sửa luật dựng nếu không đạt.
Chạy bằng Python Windows (có scikit-learn; .venv WSL không có, không cài thêm). Chỉ đọc jsonl.
--v3: chạy trên dòng KC của train_2b_v3.jsonl (mix sau tái lấy mẫu). --v2: chạy trên KC v2 (brief mục 7c) và thêm probe chỉ dùng đặc trưng trùng từ của D10 (đặc trưng do kiem_kc_v2.py tính).
Dùng: python -I scripts/train_2b/d4_probe.py [--v2 | --v3]"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/train_2b"
SEED = 20261028
MK = ("Nhận định: ", "Nhan dinh: ")
V3 = "--v3" in sys.argv          # tái lấy mẫu (PHAN_TICH_TRUOC_TRAIN_2b mục 9.7): dòng KC trong train_2b_v3.jsonl, thêm probe D10
V2 = "--v2" in sys.argv or V3    # vòng sửa KC (brief mục 7c): đọc pool v2, thêm probe D10
if V3:
    rows = [r for r in (json.loads(l) for l in open(OUT / "train_2b_v3.jsonl", encoding="utf-8")) if r["source"].startswith("kc_")]
else:
    rows = [json.loads(l) for f in (("kc_pool_ngan_v2.jsonl", "kc_pool_dai_v2.jsonl") if V2 else ("kc_pool_ngan.jsonl", "kc_pool_dai.jsonl"))
            for l in open(OUT / f, encoding="utf-8")]
rows = [r for r in rows if r["kind"] == "choice"]


def claim(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def run(rs, top=False):
    X_txt = [claim(r) for r in rs]
    y = np.array([r["gold_label"] for r in rs])
    maj = max(Counter(y).values()) / len(y)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_txt, y):
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        Xtr = hstack([vc.fit_transform([X_txt[i] for i in tr]), vw.fit_transform([X_txt[i] for i in tr])]).tocsr()
        Xte = hstack([vc.transform([X_txt[i] for i in te]), vw.transform([X_txt[i] for i in te])]).tocsr()
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        accs.append(float((m.predict(Xte) == y[te]).mean()))
    res = {"n": len(rs), "nhan": dict(Counter(y)), "da_so": round(maj, 4), "acc_5fold": round(float(np.mean(accs)), 4),
           "acc_tung_fold": [round(a, 4) for a in accs], "vuot_da_so_diem": round(100 * (np.mean(accs) - maj), 2)}
    res["dat"] = bool(np.mean(accs) <= maj + 0.10)
    if top:
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        X = hstack([vc.fit_transform(X_txt), vw.fit_transform(X_txt)]).tocsr()
        names = [f"c:{x}" for x in vc.get_feature_names_out()] + [f"w:{x}" for x in vw.get_feature_names_out()]
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
        res["dac_trung_nang_nhat"] = {c: [names[j] for j in np.argsort(-m.coef_[i])[:15]] for i, c in enumerate(m.classes_)}
    return res


out = {"tat_ca_choice_kc": run(rows, top=True), "chi_dong_co_dau": run([r for r in rows if "nodiac" not in r.get("flags", [])])}
out["dat"] = out["tat_ca_choice_kc"]["dat"]
if V2:
    # D10 (chỉ báo): probe chỉ dùng đặc trưng trùng từ (trùng lớn nhất với một câu, độ phủ token; tính ở kiem_kc_v2.py), LR 5-fold;
    # accuracy trên dòng choice KC v2 (có dấu) so với trên dòng kiểm chứng của proxy_v1 (choice + noul, nhãn 3 lớp gốc).
    F = [json.loads(l) for l in open(OUT / "tam" / ("d10_dac_trung_v3.jsonl" if V3 else "d10_dac_trung.jsonl"), encoding="utf-8")]

    def probe_ov(fs):
        X = np.array([[f["max_ov"], f["cover"]] for f in fs]); y = np.array([f["nhan"] for f in fs])
        accs = [float((LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)]
        return {"n": len(fs), "da_so": round(max(Counter(y).values()) / len(y), 4), "acc_5fold": round(float(np.mean(accs)), 4)}

    out["D10_probe_trung_tu"] = {"kc_v2_choice_co_dau": probe_ov([f for f in F if f["bo"] == "kc" and f["kind"] == "choice"]),
                                 "proxy_v1_kiem_chung": probe_ov([f for f in F if f["bo"] == "proxy_v1"])}
(OUT / "tam" / ("d4_d10_probe_v3.json" if V3 else "d4_d10_probe_v2.json" if V2 else "d4_probe.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
