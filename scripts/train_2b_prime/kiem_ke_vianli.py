"""Bước B của 2b′ (results/BRIEF_2b_prime_du_lieu.md mục 2): kiểm kê ViANLI trước khi dựng. Chỉ CPU (tokenizer + numpy), không chạy model,
không đọc test ngoài dò khử trùng (không in). Dữ liệu UIT (ViANLI qua PhoLaya) chỉ dùng nội bộ.

Nguồn: results/train_v1/raw_src/LakoreAI__PhoLaya-base/sources/vianli/train.jsonl (dòng có câu hỏi `relation`; id cặp = vianli:<số dòng>).
Loại trước khi đếm:
  (1) cặp (tiền đề, giả thuyết) đã có trong C / C′ (results/train_v1/components/C.jsonl ∪ results/train_2b/C_prime.jsonl, khớp decontam.normalize);
  (2) cặp trùng chuỗi chuẩn hoá với cặp đã giữ (giữ lần đầu; nhãn mâu thuẫn thì bỏ cả nhóm);
  (3) khử trùng (kc_util.Refs, hai chiều, tham chiếu như brief dữ liệu 2b mục 4): tiền đề hoặc giả thuyết chạm → bỏ mọi cặp dùng tiền đề đó.
Báo theo nhãn (contradiction / entailment / neutral): số cặp, số tiền đề; token Qwen3 (trung vị, p90) của tiền đề / giả thuyết; max_ov, cover
(hàm feats của phan_tich_bacbo.py: nhận định = giả thuyết, state = tiền đề); phủ định (danh sách D5); lượng từ / độc quyền; số lạ, tên lạ;
probe bề mặt KC Bác bỏ / KC NEI (fit đúng như phan_tich_bacbo.py, kiểm lại acc 0,885 và dev 57/74) → tỉ lệ bị xếp vào phía NEI.
Cột đối chiếu: Bác bỏ dev ViWikiFC (choice có dấu), Bác bỏ proxy_v1 (choice có dấu, ViWikiFC + ViFactCheck), Bác bỏ KC trong v3.
Ra results/train_2b_prime/kiem_ke_vianli.json, mau_vianli_bacbo.jsonl (40 contradiction, seed cố định),
results/train_2b_prime/tam/pool_vianli.jsonl (pool sạch + đặc trưng, nội bộ, cho bước C).
Dùng: bash setup/run.sh -I scripts/train_2b_prime/kiem_ke_vianli.py"""
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train_2b"))
from kc_util import DEV, OUT, Refs, name_spans, normalize, num_spans, overlap, read_jsonl, sentences, wset, write_jsonl  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SRC = ROOT / "results/train_v1/raw_src/LakoreAI__PhoLaya-base/sources/vianli/train.jsonl"
C_FILES = (ROOT / "results/train_v1/components/C.jsonl", OUT / "C_prime.jsonl")
PROXY = ROOT / "data/proxy_v1/proxy_v1.jsonl"
DST = ROOT / "results/train_2b_prime"
SEED = 20261029
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
NEG_P = re.compile(r"\b(không|chưa|chẳng|chả|khong|chua|chang|cha)\b", re.I)   # đặc trưng probe, như phan_tich_bacbo.py
NEG_D5 = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b")                       # danh sách D5 (kiem_2b.py)
LUONG = re.compile(r"\b(chỉ|duy nhất|độc nhất|đầu tiên|cuối cùng|phần lớn|tất cả|mọi|một phần)\b")
LAB = {"contradiction": "refuted", "entailment": "supported", "neutral": "nei"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def feats(claim, st):   # = phan_tich_bacbo.feats, nhận định và state truyền thẳng
    ss = sentences(st) or [st]
    best = max(ss, key=lambda s: overlap(claim, s))
    nst = normalize(st)
    nums = [v for _, _, v, _ in num_spans(claim)]
    names = [x for _, _, x, _ in name_spans(claim)]
    return {"max_ov": round(overlap(claim, best), 3), "cover": round(len(wset(claim) & wset(st)) / max(1, len(wset(claim))), 3),
            "phu_dinh_nd": bool(NEG_P.search(claim)), "phu_dinh_cau": bool(NEG_P.search(best)),
            "lech_phu_dinh": bool(NEG_P.search(claim)) != bool(NEG_P.search(best)),
            "so_la": any(normalize(v) not in nst for v in nums), "co_so": bool(nums), "ten_la": any(normalize(x) not in nst for x in names),
            "do_dai": len(claim.split()), "n_cau_state": len(ss),
            "phu_dinh_D5": bool(NEG_D5.search(claim.lower())), "luong_tu": bool(LUONG.search(claim.lower()))}


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


# ------------------------------------------------------------------ 1. đọc ViANLI, gom cặp
src_sha = sha(SRC)
pairs = []
for n, line in enumerate(open(SRC, encoding="utf-8")):
    x = json.loads(line)
    q = json.loads(x["questions"])
    if "relation" not in q:
        continue
    st, g = json.loads(x["state"]), json.loads(x["gold"])["relation"]
    assert set(st) == {"doan_van", "cau_noi"} and all(v == 1.0 or v == 0.0 for v in g["probabilities"].values())
    pairs.append({"id": f"vianli:{n}", "tien_de": st["doan_van"], "gia_thuyet": st["cau_noi"], "nhan_goc": g["label"]})
n_raw = Counter(p["nhan_goc"] for p in pairs)

key = lambda a, b: (normalize(a), normalize(b))
c_keys = set()
for f in C_FILES:
    for r in read_jsonl(f):
        if str(r["id"]).startswith("pl:vianli:"):
            a, b = r["state"].split("\nCâu nói: ")
            c_keys.add(key(a.removeprefix("Đoạn văn: "), b))
c_prem = {k[0] for k in c_keys}

bo = defaultdict(list)
groups = defaultdict(list)
for p in pairs:
    groups[key(p["tien_de"], p["gia_thuyet"])].append(p)
uniq = []
for k, g in groups.items():
    if k in c_keys:
        bo["da_co_trong_C_Cprime"] += [p["id"] for p in g]
    elif len({p["nhan_goc"] for p in g}) > 1:
        bo["trung_cap_nhan_mau_thuan"] += [p["id"] for p in g]
    else:
        uniq.append(g[0]); bo["trung_cap_giu_lan_dau"] += [p["id"] for p in g[1:]]

# ------------------------------------------------------------------ 2. khử trùng (hai chiều), theo tiền đề
refs = Refs()
by_prem = defaultdict(list)
for p in uniq:
    by_prem[normalize(p["tien_de"])].append(p)
cham = Counter()
pool = []
for k, ps in by_prem.items():
    h = refs.hit(ps[0]["tien_de"])
    why = f"tien_de:{h[1]}" if h else None
    if not why:
        for p in ps:
            h = refs.hit(p["gia_thuyet"])
            if h:
                why = f"gia_thuyet:{h[1]}"; break
    if why:
        cham[why] += len(ps); bo["khu_trung"] += [p["id"] for p in ps]
    else:
        pool += ps

# ------------------------------------------------------------------ 3. đặc trưng + token
from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"])
for p in pool:
    p.update(feats(p["gia_thuyet"], p["tien_de"]))
    p["tok_tien_de"] = len(TOK(p["tien_de"])["input_ids"]); p["tok_gia_thuyet"] = len(TOK(p["gia_thuyet"])["input_ids"])
    p["tien_de_co_trong_C"] = normalize(p["tien_de"]) in c_prem

# ------------------------------------------------------------------ 4. probe bề mặt: fit đúng như phan_tich_bacbo.py
kieu = {r["id"]: r.get("kieu") for f in ("kc_pool_ngan_v2.jsonl", "kc_pool_dai_v2.jsonl") for r in read_jsonl(OUT / f)}
v3 = read_jsonl(OUT / "train_2b_v3.jsonl")
is_orig = lambda r: r.get("variant", "orig") != "nodiac" and not any(mk in r["question"] for mk in ("Nhan dinh: ", "Phat bieu: "))
kc_ref, kc_nei = [], []
for r in v3:
    if not r["source"].startswith("kc_") or not is_orig(r):
        continue
    if r.get("gold_label") == "refuted":
        try:
            kc_ref.append(feats(claim_of(r), r["state"] or ""))
        except StopIteration:
            continue
    elif r.get("gold_label") == "nei":
        kc_nei.append(feats(claim_of(r), r["state"] or ""))
FK = ("max_ov", "cover", "phu_dinh_nd", "lech_phu_dinh", "so_la", "ten_la", "do_dai")
vec = lambda f: [float(f[k]) for k in FK]
X = np.array([vec(f) for f in kc_ref + kc_nei]); y = np.array([1] * len(kc_ref) + [0] * len(kc_nei))
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1])
for _ in range(3000):
    pp = 1 / (1 + np.exp(-Z @ w)); w -= 0.1 * (Z.T @ (pp - y) / len(y) + 1e-3 * np.r_[0, w[1:]])
acc_kc = float(((1 / (1 + np.exp(-Z @ w)) > 0.5) == y).mean())
score = lambda f: float(1 / (1 + np.exp(-(np.r_[1, (np.array(vec(f)) - mu) / sd] @ w))))

dev = read_jsonl(DEV)
dev_ref = [feats(claim_of(r), r["state"] or "") for r in dev if r["source"] == "viwikifc" and r["variant"] == "orig" and r["kind"] == "choice"
           and r["gold_label"] == "refuted"]
n_dev_nei_side = sum(score(f) < 0.5 for f in dev_ref)
assert abs(acc_kc - 0.885) < 0.0015 and len(dev_ref) == 74 and n_dev_nei_side == 57, (acc_kc, len(dev_ref), n_dev_nei_side)   # khớp báo cáo lỗi Bác bỏ
px = read_jsonl(PROXY)
px_ref = [feats(claim_of(r), r["state"] or "") for r in px if r["source"] in ("viwikifc", "vifactcheck") and r["variant"] == "orig"
          and r["kind"] == "choice" and r["gold_label"] == "refuted"]
px_ref_wiki = [feats(claim_of(r), r["state"] or "") for r in px if r["source"] == "viwikifc" and r["variant"] == "orig"
               and r["kind"] == "choice" and r["gold_label"] == "refuted"]
for p in pool:
    p["probe_p_bacbo_kc"] = round(score(p), 3)


# ------------------------------------------------------------------ 5. tổng hợp
def q(v, x):
    v = sorted(v)
    return v[min(len(v) - 1, int(x * len(v)))] if v else None


def summ(fs, tok=False):
    n = len(fs)
    ov = [f["max_ov"] for f in fs]; cv = [f["cover"] for f in fs]
    o = {"n": n, "max_ov_trung_vi": q(ov, 0.5), **{f"max_ov<{t}": round(sum(v < t for v in ov) / n, 3) for t in (0.5, 0.7, 0.9)},
         "cover_trung_vi": q(cv, 0.5), **{f"cover<{t}": round(sum(v < t for v in cv) / n, 3) for t in (0.5, 0.7, 0.9)},
         "phu_dinh_D5": round(sum(f["phu_dinh_D5"] for f in fs) / n, 3), "luong_tu": round(sum(f["luong_tu"] for f in fs) / n, 3),
         "so_la": round(sum(f["so_la"] for f in fs) / n, 3), "ten_la": round(sum(f["ten_la"] for f in fs) / n, 3),
         "lech_phu_dinh": round(sum(f["lech_phu_dinh"] for f in fs) / n, 3),
         "probe_phia_NEI": sum(score(f) < 0.5 for f in fs), "probe_phia_NEI_ti_le": round(sum(score(f) < 0.5 for f in fs) / n, 3)}
    if tok:
        o.update({"so_tien_de": len({normalize(f["tien_de"]) for f in fs}),
                  "tien_de_cung_co_trong_C": len({normalize(f["tien_de"]) for f in fs if f["tien_de_co_trong_C"]}),
                  "tok_tien_de_trung_vi": q([f["tok_tien_de"] for f in fs], 0.5), "tok_tien_de_p90": q([f["tok_tien_de"] for f in fs], 0.9),
                  "tok_gia_thuyet_trung_vi": q([f["tok_gia_thuyet"] for f in fs], 0.5),
                  "tok_gia_thuyet_p90": q([f["tok_gia_thuyet"] for f in fs], 0.9)})
    return o


res = {"sha": {"vianli_train": src_sha, "C": sha(C_FILES[0]), "C_prime": sha(C_FILES[1]), "dev": sha(DEV), "proxy_v1": sha(PROXY),
               "train_2b_v3": sha(OUT / "train_2b_v3.jsonl")},
       "cap_goc_theo_nhan": dict(n_raw), "cap_C_Cprime_khop_chuoi": len(c_keys),
       "bo": {k: len(v) for k, v in bo.items()}, "khu_trung_theo_tham_chieu": dict(cham.most_common()),
       "con_lai_theo_nhan": {lab: summ([p for p in pool if p["nhan_goc"] == lab], tok=True) for lab in LAB},
       "doi_chieu": {"BB_dev_viwikifc_choice_co_dau": summ([{**f} for f in dev_ref]), "BB_proxy_v1_choice_co_dau": summ(px_ref),
                     "BB_proxy_v1_viwikifc": summ(px_ref_wiki), "BB_KC_v3_co_dau": summ(kc_ref), "NEI_KC_v3_co_dau": summ(kc_nei)},
       "probe": {"dac_trung": FK, "acc_tren_KC": round(acc_kc, 4), "n_KC": {"bac_bo": len(kc_ref), "nei": len(kc_nei)},
                 "kiem_lai_dev_57_74": [n_dev_nei_side, len(dev_ref)]},
       "ghi_chu": ["nhận định = giả thuyết ViANLI, state = tiền đề (thường 1 câu, nên max_ov ≈ cover)",
                   "lượng từ khớp \\b theo từ: 'chỉ' cũng khớp 'chỉ huy', 'chỉ đạo', 'địa chỉ' (danh sách theo brief, không lọc thêm)",
                   "khử trùng: một câu (tiền đề hoặc giả thuyết) chạm → bỏ mọi cặp dùng tiền đề đó"]}
# phân bố số giả thuyết / tiền đề
cnt = Counter(normalize(p["tien_de"]) for p in pool)
res["gia_thuyet_moi_tien_de"] = dict(sorted(Counter(cnt.values()).items()))

DST.mkdir(parents=True, exist_ok=True); (DST / "tam").mkdir(exist_ok=True)
(DST / "kiem_ke_vianli.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
write_jsonl(DST / "tam/pool_vianli.jsonl", pool)
con = sorted([p for p in pool if p["nhan_goc"] == "contradiction"], key=lambda p: p["id"])
mau = random.Random(SEED).sample(con, 40)
write_jsonl(DST / "mau_vianli_bacbo.jsonl", [{"id": p["id"], "tien_de": p["tien_de"], "gia_thuyet": p["gia_thuyet"], "max_ov": p["max_ov"]} for p in mau])
print(json.dumps(res, ensure_ascii=False, indent=1))
