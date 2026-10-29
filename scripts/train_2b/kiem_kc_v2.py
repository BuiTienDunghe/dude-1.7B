"""Cổng của vòng sửa KC (results/BRIEF_2b_du_lieu.md mục 7c) trên KC v2: ráp train_2b_mac_dinh_v2.jsonl (C′ giữ nguyên + 1.800 KC-ngắn v2 +
1.200 KC-dài v2, xáo theo seed) và chạy D3, D5, D6′, D7, D9, D10 (phần thống kê). D4 và probe của D10 chạy bằng d4_probe.py --v2 (cần sklearn).
Phép đo trùng từ như D6 vòng 1 và như điều phối dùng trên ViWikiFC train: trùng lớn nhất = max theo câu của state |từ nhận định ∩ từ câu| /
|từ nhận định|; độ phủ token = |từ nhận định ∩ từ state| / |từ nhận định| (từ đã chuẩn hoá decontam.normalize).
Chỉ báo số và áp ngưỡng đã chốt trước; không tự quyết bước tiếp.
Ra results/train_2b/{train_2b_mac_dinh_v2.jsonl, mau_doc_v2.jsonl, manifest_kc_v2.json, tam/kiem_kc_v2.json, tam/d10_dac_trung.jsonl}.
Dùng: bash setup/run.sh -I scripts/train_2b/kiem_kc_v2.py"""
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import DEV, OUT, ROOT, SEED, TMP, Refs, h64, normalize, overlap, read_jsonl, sentences, wset, write_jsonl  # noqa: E402

T0 = time.time()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CP = read_jsonl(OUT / "C_prime.jsonl")
PN, PD, VAL = read_jsonl(OUT / "kc_pool_ngan_v2.jsonl"), read_jsonl(OUT / "kc_pool_dai_v2.jsonl"), read_jsonl(OUT / "val_kc_v2.jsonl")
dev = read_jsonl(DEV)
KC_MIX = PN[:1800] + PD[:1200]
TRAIN = sorted(CP + KC_MIX, key=lambda r: h64(SEED, "mix", r["id"]))
assert len({r["id"] for r in TRAIN}) == len(TRAIN)
write_jsonl(OUT / "train_2b_mac_dinh_v2.jsonl", TRAIN)
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
res = {"ghi_chu": "lần dựng chốt KC v2, sau đọc mù vòng 1 (đã gộp 4 sửa chất lượng)"}


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def flags(r):
    return set(r.get("flags", []))


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def max_ov(c, st):
    return max((overlap(c, s) for s in sentences(st)), default=0.0)


def cover(c, st):
    a = wset(c)
    return len(a & wset(st)) / max(1, len(a))


ALL = PN + PD
KCD = [r for r in ALL if "nodiac" not in flags(r)]
LAB = ("supported", "refuted", "nei")

# ------------------------------------------------------------------ D3
REFS = Refs()
hits = Counter(); hit_rows = []
for r in ALL + VAL:
    for t in [r["state"], claim_of(r)] + sentences(r["state"]):
        h = REFS.hit(t)
        if h:
            hits[f"{h[0]}|{h[1]}"] += 1; hit_rows.append(r["id"]); break
res["D3"] = {"so_dong_kiem": len(ALL + VAL), "cham": dict(hits), "dong_cham": hit_rows[:50], "dat": not hits}

# ------------------------------------------------------------------ D5
NEG = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b")


def d5(rows):
    out = {}
    for lab in LAB:
        rs = [r for r in rows if r["gold_label"] == lab]; cl = [claim_of(r) for r in rs]
        out[lab] = {"n": len(rs), "token_trung_vi": q([r["claim_ntok"] for r in rs], 0.5), "token_p90": q([r["claim_ntok"] for r in rs], 0.9),
                    "ti_le_phu_dinh": round(sum(bool(NEG.search(c.lower())) for c in cl) / max(1, len(cl)), 4),
                    "ti_le_co_so_hoac_ngay": round(sum(bool(re.search(r"\d", c)) for c in cl) / max(1, len(cl)), 4)}
    meds = [out[l]["token_trung_vi"] for l in LAB]; negs = [out[l]["ti_le_phu_dinh"] for l in LAB]
    out["lech_trung_vi"] = round((max(meds) - min(meds)) / min(meds), 4)
    out["lech_phu_dinh_diem"] = round(100 * (max(negs) - min(negs)), 2)
    out["dat"] = out["lech_trung_vi"] <= 0.20 and out["lech_phu_dinh_diem"] <= 2
    return out


res["D5"] = {"kc_choice_co_dau": d5([r for r in KCD if r["kind"] == "choice"]), "kc_tat_ca_co_dau": d5(KCD)}
res["D5"]["dat"] = res["D5"]["kc_choice_co_dau"]["dat"]

# ------------------------------------------------------------------ D6′ + D10 (thống kê) + đặc trưng cho probe D10
feat = []
for r in KCD:
    c = claim_of(r)
    feat.append({"bo": "kc", "id": r["id"], "kind": r["kind"], "nhan": r["gold_label"], "kieu": r["kieu"], "max_ov": round(max_ov(c, r["state"]), 4),
                 "cover": round(cover(c, r["state"]), 4), "lien_trong_state": normalize(c) in normalize(r["state"])})
PX = [r for r in read_jsonl(ROOT / "data/proxy_v1/proxy_v1.jsonl") if r["source"] in ("vifactcheck", "viwikifc") and r["variant"] == "orig"]
for r in PX:
    c = claim_of(r)
    feat.append({"bo": "proxy_v1", "id": r["id"], "kind": r["kind"], "nhan": r["gold_label"], "kieu": r["source"], "max_ov": round(max_ov(c, r["state"]), 4),
                 "cover": round(cover(c, r["state"]), 4), "lien_trong_state": normalize(c) in normalize(r["state"])})
write_jsonl(TMP / "d10_dac_trung.jsonl", feat)


def dist(xs):
    return {"n": len(xs), "trung_vi": q(xs, 0.5), "p10": q(xs, 0.1), "p90": q(xs, 0.9), "ge_0_6": round(sum(x >= 0.6 for x in xs) / max(1, len(xs)), 3)}


fk = [f for f in feat if f["bo"] == "kc"]; fp = [f for f in feat if f["bo"] == "proxy_v1"]
d6 = {"kc_co_dau": {l: dist([f["max_ov"] for f in fk if f["nhan"] == l]) for l in LAB},
      "proxy_v1": {l: dist([f["max_ov"] for f in fp if f["nhan"] == l]) for l in LAB}}
med = {l: d6["kc_co_dau"][l]["trung_vi"] for l in LAB}
lien_S = sum(f["lien_trong_state"] for f in fk if f["nhan"] == "supported") / max(1, sum(f["nhan"] == "supported" for f in fk))
lt09 = {l: round(sum(f["max_ov"] < 0.9 for f in fk if f["nhan"] == l) / max(1, sum(f["nhan"] == l for f in fk)), 4) for l in LAB}
d6["ti_le_bao_phu_lt_0_9"] = lt09
# ngưỡng sửa trước khi dựng (brief 7c "Sửa cổng trước khi dựng"); trung vị và mức chênh ba nhãn chỉ báo
d6["nguong"] = {"NEI_trung_vi_ge_0_40": med["nei"] >= 0.40, "NEI_ge_0_6_chiem_ge_25": d6["kc_co_dau"]["nei"]["ge_0_6"] >= 0.25,
                "S_lt_0_9_ge_30": lt09["supported"] >= 0.30, "R_lt_0_9_ge_30": lt09["refuted"] >= 0.30, "S_doan_lien_le_50": lien_S <= 0.50}
d6["chi_bao_trung_vi"] = {"kc": med, "viwikifc_train_dieu_phoi": {"supported": 0.70, "refuted": 0.71, "nei": 0.50},
                          "proxy_v1": {l: d6["proxy_v1"][l]["trung_vi"] for l in LAB}}
d6["S_doan_lien_trong_state"] = round(lien_S, 4); d6["S_tru_NEI"] = round(med["supported"] - med["nei"], 4)
d6["dat"] = all(d6["nguong"].values())
d6["theo_kieu_trung_vi"] = {k: q([f["max_ov"] for f in fk if f["kieu"] == k], 0.5) for k in sorted({f["kieu"] for f in fk})}
res["D6_phay"] = d6
res["D10_thong_ke"] = {"do_phu_token_trung_vi": {"kc_co_dau": {l: q([f["cover"] for f in fk if f["nhan"] == l], 0.5) for l in LAB},
                                                "proxy_v1": {l: q([f["cover"] for f in fp if f["nhan"] == l], 0.5) for l in LAB},
                                                "viwikifc_train_dieu_phoi": {"supported": 0.79, "refuted": 0.80, "nei": 0.67}},
                       "probe": "xem tam/d4_d10_probe_v2.json"}

# ------------------------------------------------------------------ D7
fc_choice = [r for r in TRAIN if r["text_lang"] == "vi" and r["kind"] == "choice" and (r["source"].startswith("kc_") or {"2d", "2e"} & flags(r))]
c7 = Counter(r["gold_label"] for r in fc_choice)
kc_noul = [r for r in KC_MIX if r["kind"] == "noul"]
res["D7"] = {"kiem_chung_vi_choice": {"n": len(fc_choice), "nhan": dict(c7), "ti_le_nei": round(c7["nei"] / len(fc_choice), 4)},
             "kc_mix_choice_nhan": dict(Counter(r["gold_label"] for r in KC_MIX if r["kind"] == "choice")),
             "noul_co_kc_mix": round(sum(r["target_argmax"] == 0 for r in kc_noul) / max(1, len(kc_noul)), 4)}
res["D7"]["dat"] = res["D7"]["kiem_chung_vi_choice"]["ti_le_nei"] >= 0.30 and 0.37 <= res["D7"]["noul_co_kc_mix"] <= 0.40

# ------------------------------------------------------------------ D9
d9 = {}
for l in LAB:
    rs = [r for r in ALL if r["gold_label"] == l]
    d9[l] = {"n": len(rs), "ti_le_ghep": round(sum(r["hinh_dang"] == "ghep" for r in rs) / len(rs), 4),
             "ti_le_bien_the": round(sum(bool(r["bien_the"]) for r in rs) / len(rs), 4),
             "ti_le_luoc": round(sum(r["hinh_dang"] == "luoc" for r in rs) / len(rs), 4)}
for k in ("ti_le_ghep", "ti_le_bien_the"):
    v = [d9[l][k] for l in LAB]
    d9[f"lech_{k}_diem"] = round(100 * (max(v) - min(v)), 2)
d9["dat"] = d9["lech_ti_le_ghep_diem"] <= 5 and d9["lech_ti_le_bien_the_diem"] <= 5
d9["kieu"] = dict(Counter(r["kieu"] for r in ALL)); d9["kieu_x_hinh_dang"] = dict(Counter(f"{r['kieu']}|{r['hinh_dang']}" for r in ALL))
d9["bien_the_loai"] = dict(Counter(r["bien_the"] for r in ALL if r["bien_the"]))
res["D9"] = d9

# ------------------------------------------------------------------ mẫu đọc v2
L = "ABCDEFGHIJ"


def sample(rows, n, tag):
    return sorted(rows, key=lambda r: h64(SEED, "mau2", tag, r["id"]))[:n]


def rec(r, nhom):
    t = r["target_argmax"]
    d = {"ma": None, "id": r["id"], "nhom": nhom, "dataset": r["dataset"], "kind": r["kind"], "state": r["state"], "question": r["question"],
         "options": [f"{L[i]}. {o}" for i, o in enumerate(r["options"])], "nhan": f"{L[t]}. {r['options'][t]}"}
    for k in ("kieu", "hinh_dang", "luoc", "bien_the", "cau_goc", "doi", "cum_chen", "moi", "tieu_de", "tieu_de_khac", "xoa_cau_khoi_state"):
        d[f"nhan_{k}"] = r.get(k)
    return d


cd = [r for r in ALL if "nodiac" not in flags(r)]
mau = []
for kieu in ("N-bobang", "N-them", "N-ghep", "S-ghep"):
    mau += [rec(r, "KC2") for r in sample([x for x in cd if x["kieu"] == kieu], 30, kieu)]
mau += [rec(r, "KC2") for r in sample([x for x in cd if x["hinh_dang"] == "ghep" and x["kieu"] in ("R-so", "R-ten")], 30, "R-ghep")]
mau += [rec(r, "KC2") for r in sample([x for x in cd if x["bien_the"] and x["id"] not in {m["id"] for m in mau}], 30, "bien_the")]
# kiểm các sửa sau đọc mù vòng 1: 20 S-lược, 20 R-tên dựng lại
mau += [rec(r, "KC2") for r in sample([x for x in cd if x["kieu"] == "S-luoc" and x["id"] not in {m["id"] for m in mau}], 20, "S-luoc")]
mau += [rec(r, "KC2") for r in sample([x for x in cd if x["kieu"] == "R-ten" and x["id"] not in {m["id"] for m in mau}], 20, "R-ten")]
mau.sort(key=lambda m: h64(SEED, "mau2_xao", m["id"]))
for i, m in enumerate(mau):
    m["ma"] = f"V{i + 1:03d}"
write_jsonl(OUT / "mau_doc_v2.jsonl", mau)
res["mau_doc_v2"] = {"so_dong": len(mau), "nhom_dung": "N-bobang, N-them, N-ghep, S-ghep, R-ghep (R-so / R-ten dạng ghép), biến thể định dạng: 30 mỗi nhóm; S-lược 20, R-tên 20 (kiểm sửa sau đọc mù vòng 1)"}

kcl = json.loads((TMP / "kc_log_v2.json").read_text(encoding="utf-8"))
man = {"brief": "results/BRIEF_2b_du_lieu.md mục 7c", "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED,
       "ghi_chu": res["ghi_chu"], "C_prime_giu_nguyen": sha(OUT / "C_prime.jsonl"),
       "file": {p.name: sha(p) for p in [OUT / "kc_pool_ngan_v2.jsonl", OUT / "kc_pool_dai_v2.jsonl", OUT / "val_kc_v2.jsonl",
                                          OUT / "train_2b_mac_dinh_v2.jsonl", OUT / "mau_doc_v2.jsonl"]},
       "so_dong": {"kc_pool_ngan_v2": len(PN), "kc_pool_dai_v2": len(PD), "val_kc_v2": len(VAL), "train_2b_mac_dinh_v2": len(TRAIN)},
       "tu_chen_duy_nhat": "và (kiểu ghép)", "kc_log": kcl, "giay_kiem": round(time.time() - T0)}
(OUT / "manifest_kc_v2.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
(TMP / "kiem_kc_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: (v.get("dat") if isinstance(v, dict) else v) for k, v in res.items()}, ensure_ascii=False))
print(json.dumps({"D6_phay": {k: d6[k] for k in ("nguong", "ti_le_bao_phu_lt_0_9", "chi_bao_trung_vi", "S_doan_lien_trong_state", "S_tru_NEI", "theo_kieu_trung_vi")}, "D9": {k: d9[k] for k in d9 if k.startswith(("lech", "supported", "refuted", "nei"))},
                  "D7": res["D7"], "D10": res["D10_thong_ke"]}, ensure_ascii=False, indent=1))
