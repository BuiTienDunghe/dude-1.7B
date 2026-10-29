"""Bước lọc sau đọc mù vòng 2 (điều phối, 28/10): train_2b_v2_loc.jsonl từ train_2b_mac_dinh_v2.jsonl. Không dựng lại KC, không chạy model.
Bỏ các dòng KC:
  (a) N-bobang mà nhận định vẫn được một câu bất kỳ của state còn lại bao phủ >= 0,9 (bài có câu trùng ở chỗ khác);
  (b) N-bobang mà câu ngay sau câu bị xóa trong bài gốc (câu bị xóa đứng đầu cửa sổ thì đó cũng là câu mở đầu state) có 5 chữ đầu chứa
      "này", "đó", "ấy", "khác", hoặc mở đầu bằng từ nối chỉ ngược mở rộng (NOI_MO_RONG) → state dễ gãy;
  (c) mọi dòng KC có state chứa "[[" hoặc "]]" (sót mã wiki).
Bù mỗi dòng bị bỏ bằng dòng chưa dùng trong pool v2 cùng mã kiểu, cùng kind, cùng dải ngắn / dài, đã qua cùng bộ lọc, theo thứ tự xáo sẵn
của pool. Rồi chạy lại D5, D6′, D7, D9 trên mix sau lọc (chỉ báo số, không chỉnh để ép qua cổng).
Ra results/train_2b/train_2b_v2_loc.jsonl + tam/loc_kc_v2.json. Dùng: bash setup/run.sh -I scripts/train_2b/loc_kc_v2.py"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import OUT, SEED, TMP, h64, normalize, overlap, read_jsonl, sentences, wset, write_jsonl  # noqa: E402

NOI_MO_RONG = ("Ngoài ra", "Tuy nhiên", "Do đó", "Vì vậy", "Vì thế", "Mặt khác", "Bên cạnh đó", "Cũng", "Còn")
CHI_NGUOC = {"này", "đó", "ấy", "khác"}
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

SRC = OUT / "train_2b_mac_dinh_v2.jsonl"
TRAIN = read_jsonl(SRC)
PN, PD = read_jsonl(OUT / "kc_pool_ngan_v2.jsonl"), read_jsonl(OUT / "kc_pool_dai_v2.jsonl")
ART = {a["id"]: a for a in read_jsonl(TMP / "bai_sach.jsonl")}


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def next_sentence(r):
    """Câu ngay sau câu bị xóa trong bài gốc (bản có dấu)."""
    a = ART[r["wiki_id"]]
    flat = [s for p in a["pars"] for s in sentences(p)]
    try:
        i = flat.index(r["cau_goc"])
    except ValueError:
        return None
    return flat[i + 1] if i + 1 < len(flat) else None


def gay(s):
    if not s:
        return False
    w5 = [x.lower().strip(",.;:()\"“”") for x in s.split()[:5]]
    return bool(CHI_NGUOC & set(w5)) or any(s == p or s.startswith(p + " ") or s.startswith(p + ",") for p in NOI_MO_RONG)


def ly_do(r):
    """Lý do bỏ (a / b / c) hoặc None."""
    if "[[" in r["state"] or "]]" in r["state"]:
        return "c_ma_wiki"
    if r["kieu"] == "N-bobang":
        c = claim_of(r)
        if max((overlap(c, s) for s in sentences(r["state"])), default=0) >= 0.9:
            return "a_bobang_van_bao_phu"
        if gay(next_sentence(r)):
            return "b_bobang_cau_sau_chi_nguoc"
    return None


mix_ids = {r["id"] for r in TRAIN if r["source"].startswith("kc_")}
dropped, by = [], Counter()
spare = defaultdict(list)                      # (kiểu, kind, nguồn) -> dòng pool chưa dùng, đã qua lọc, theo thứ tự pool
for r in PN + PD:
    if r["id"] not in mix_ids and ly_do(r) is None:
        spare[(r["kieu"], r["kind"], r["source"])].append(r)
out, thieu = [], Counter()
for r in TRAIN:
    if not r["source"].startswith("kc_"):
        out.append(r); continue
    why = ly_do(r)
    if why is None:
        out.append(r); continue
    by[why] += 1
    k = (r["kieu"], r["kind"], r["source"])
    rep = spare[k].pop(0) if spare[k] else None
    dropped.append({"id": r["id"], "ly_do": why, "thay_bang": rep["id"] if rep else None})
    if rep:
        out.append(rep)
    else:
        thieu[str(k)] += 1
out.sort(key=lambda r: h64(SEED, "mix", r["id"]))
assert len({r["id"] for r in out}) == len(out)
write_jsonl(OUT / "train_2b_v2_loc.jsonl", out)
pool_by = Counter(ly_do(r) for r in PN + PD)

# ------------------------------------------------------------------ D5, D6′, D7, D9 trên mix sau lọc (cùng định nghĩa như kiem_kc_v2.py)
KC = [r for r in out if r["source"].startswith("kc_")]
KCD = [r for r in KC if "nodiac" not in r.get("flags", [])]
LAB = ("supported", "refuted", "nei")
NEG = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def d5(rows):
    o = {}
    for l in LAB:
        rs = [r for r in rows if r["gold_label"] == l]; cl = [claim_of(r) for r in rs]
        o[l] = {"n": len(rs), "token_trung_vi": q([r["claim_ntok"] for r in rs], 0.5),
                "ti_le_phu_dinh": round(sum(bool(NEG.search(c.lower())) for c in cl) / max(1, len(cl)), 4)}
    meds = [o[l]["token_trung_vi"] for l in LAB]; negs = [o[l]["ti_le_phu_dinh"] for l in LAB]
    o["lech_trung_vi"] = round((max(meds) - min(meds)) / min(meds), 4); o["lech_phu_dinh_diem"] = round(100 * (max(negs) - min(negs)), 2)
    o["dat"] = o["lech_trung_vi"] <= 0.20 and o["lech_phu_dinh_diem"] <= 2
    return o


f = [{"nhan": r["gold_label"], "ov": max((overlap(claim_of(r), s) for s in sentences(r["state"])), default=0.0),
      "lien": normalize(claim_of(r)) in normalize(r["state"])} for r in KCD]
med = {l: q([x["ov"] for x in f if x["nhan"] == l], 0.5) for l in LAB}
share = lambda l, cond: round(sum(cond(x) for x in f if x["nhan"] == l) / max(1, sum(x["nhan"] == l for x in f)), 4)
d6 = {"trung_vi": med, "NEI_ge_0_6": share("nei", lambda x: x["ov"] >= 0.6), "lt_0_9": {l: share(l, lambda x: x["ov"] < 0.9) for l in LAB},
      "S_doan_lien": share("supported", lambda x: x["lien"])}
d6["nguong"] = {"NEI_trung_vi_ge_0_40": med["nei"] >= 0.40, "NEI_ge_0_6_chiem_ge_25": d6["NEI_ge_0_6"] >= 0.25,
                "S_lt_0_9_ge_30": d6["lt_0_9"]["supported"] >= 0.30, "R_lt_0_9_ge_30": d6["lt_0_9"]["refuted"] >= 0.30,
                "S_doan_lien_le_50": d6["S_doan_lien"] <= 0.50}
d6["dat"] = all(d6["nguong"].values())
fc = [r for r in out if r["text_lang"] == "vi" and r["kind"] == "choice" and (r["source"].startswith("kc_") or {"2d", "2e"} & set(r.get("flags", [])))]
c7 = Counter(r["gold_label"] for r in fc); kn = [r for r in KC if r["kind"] == "noul"]
d7 = {"n": len(fc), "nhan": dict(c7), "ti_le_nei": round(c7["nei"] / len(fc), 4), "noul_co_kc": round(sum(r["target_argmax"] == 0 for r in kn) / len(kn), 4)}
d7["dat"] = d7["ti_le_nei"] >= 0.30 and 0.37 <= d7["noul_co_kc"] <= 0.40
d9 = {l: {"ghep": round(sum(r["hinh_dang"] == "ghep" for r in KC if r["gold_label"] == l) / sum(r["gold_label"] == l for r in KC), 4),
          "bien_the": round(sum(bool(r["bien_the"]) for r in KC if r["gold_label"] == l) / sum(r["gold_label"] == l for r in KC), 4)} for l in LAB}
for k in ("ghep", "bien_the"):
    v = [d9[l][k] for l in LAB]; d9[f"lech_{k}_diem"] = round(100 * (max(v) - min(v)), 2)
d9["dat"] = d9["lech_ghep_diem"] <= 5 and d9["lech_bien_the_diem"] <= 5
res = {"dau_vao": {SRC.name: sha(SRC)}, "file": {"train_2b_v2_loc.jsonl": sha(OUT / "train_2b_v2_loc.jsonl")}, "so_dong": len(out),
       "bo_trong_mix_theo_ly_do": dict(by), "khong_bu_duoc": dict(thieu), "dong_bo": dropped,
       "ca_pool_v2_theo_ly_do": {str(k): v for k, v in pool_by.items()}, "D5_kc_choice_co_dau": d5([r for r in KCD if r["kind"] == "choice"]),
       "D6_phay_kc_co_dau": d6, "D7": d7, "D9_kc_mix": d9, "noi_nguoc_mo_rong": NOI_MO_RONG, "tu_chi_nguoc_5_chu_dau": sorted(CHI_NGUOC)}
(TMP / "loc_kc_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "dong_bo"}, ensure_ascii=False, indent=1))
