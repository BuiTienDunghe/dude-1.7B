"""Cổng D1 và D4 của 2b′ (results/BRIEF_2b_prime_du_lieu.md mục 5). Chạy bằng Python Windows (có scikit-learn). Chỉ đọc jsonl, không chạy model.
D1: scripts/check_label_strings.py trên train_2b_prime.jsonl và train_2b_v3.jsonl. Cặp mà một bên đúng bằng strip(bên kia) đếm riêng ("cặp có dấu / không dấu"), không tính vi phạm. Đạt khi 0 lần gần trùng lệch dev ở khung kiem_chung /
    co_khong (gồm chuỗi NEI) và số lần gần trùng lệch ở các khung khác không nhiều hơn v3.
D4: probe chỉ đọc nhận định (như scripts/train_2b/d4_probe.py: TF-IDF ký tự 2–4 + từ 1–2, LR, 5-fold phân tầng, seed 20261028) trên dòng
    choice của khối ViANLI (vianli_khung.jsonl). Đạt khi accuracy <= tỉ lệ nhãn đa số + 10 điểm. Báo thêm: chỉ dòng có dấu.
Ra results/train_2b_prime/tam/{d1_2b_prime.json, d1_v3.json, d1_d4.json}.
Dùng: python -X utf8 -I scripts/train_2b_prime/d1_d4.py"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
DST = ROOT / "results/train_2b_prime"
SEED = 20261028
MK = ("Nhận định: ", "Nhan dinh: ")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "scripts"))
from common import strip_diacritics  # noqa: E402

# ------------------------------------------------------------------ D1
d1 = {}
for ten, f in (("2b_prime", DST / "train_2b_prime.jsonl"), ("v3", ROOT / "results/train_2b/train_2b_v3.jsonl")):
    out = DST / f"tam/d1_{ten}.json"
    subprocess.run([sys.executable, "-X", "utf8", "-I", str(ROOT / "scripts/check_label_strings.py"), str(f), "--out", str(out)], check=True,
                   capture_output=True)
    r = json.loads(out.read_text(encoding="utf-8"))
    by, cap_dau = Counter(), Counter()
    for items in r["gan_trung_LECH_dev"].values():
        for x in items:
            if strip_diacritics(x["chu_train"]) == x["chu_dev_gan"] or x["chu_train"] == strip_diacritics(x["chu_dev_gan"]):
                cap_dau[x["khung"]] += x["so_lan"]      # cặp có dấu / không dấu: đếm riêng, không tính vi phạm (điều phối, brief mục 2b điểm 6)
            else:
                by[x["khung"]] += x["so_lan"]
    d1[ten] = {"tong": r["tong"], "gan_trung_lech_theo_khung": dict(by), "cap_co_dau_khong_dau_theo_khung": dict(cap_dau)}
fc = ("kiem_chung", "co_khong")
d1["dat"] = (sum(d1["2b_prime"]["gan_trung_lech_theo_khung"].get(k, 0) for k in fc) == 0
             and sum(v for k, v in d1["2b_prime"]["gan_trung_lech_theo_khung"].items() if k not in fc)
             <= sum(v for k, v in d1["v3"]["gan_trung_lech_theo_khung"].items() if k not in fc))


# ------------------------------------------------------------------ D4
def claim(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def run(rs):
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
    return {"n": len(rs), "nhan": dict(Counter(y)), "da_so": round(maj, 4), "acc_5fold": round(float(np.mean(accs)), 4),
            "acc_tung_fold": [round(a, 4) for a in accs], "vuot_da_so_diem": round(100 * (np.mean(accs) - maj), 2), "dat": bool(np.mean(accs) <= maj + 0.10)}


rows = [json.loads(l) for l in open(DST / "vianli_khung.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["kind"] == "choice"]
d4 = {"tat_ca_choice_vianli": run(rows), "chi_co_dau": run([r for r in rows if "nodiac" not in (r.get("flags") or [])])}
d4["dat"] = d4["tat_ca_choice_vianli"]["dat"]
res = {"D1": d1, "D4": d4}
(DST / "tam/d1_d4.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
