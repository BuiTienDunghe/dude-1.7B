"""Tái lấy mẫu KC trong pool v2 (sai lệch có ngày so với brief 7c; luật: results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md mục 9.7, theo
mô phỏng f_joint_redraw.py của Fable). Không dựng lại KC, không chạy model.
  1. Từ train_2b_v2_loc.jsonl: giữ nguyên C′ và mọi dòng KC Xác nhận / Bác bỏ; rút lại các dòng KC NEI theo cơ cấu bỏ bằng 0,31 / thêm 0,29 /
     ghép 0,20 / cùng bài 0,10 / lạc đề 0,10 trong cùng ô (dải ngắn / dài, kind); số dòng NEI mỗi ô giữ như cũ. Lấy theo thứ tự xáo sẵn
     của pool; ô thiếu kiểu thì bù bằng N-bobang (như mô phỏng).
  2. Mọi dòng rút mới (và mọi dòng dùng để hoán đổi) phải qua bộ lọc (a), (b), (c) của loc_kc_v2.py.
  3. Cân phủ định (dòng choice có dấu) và biến thể định dạng giữa ba nhãn bằng hoán đổi trong cùng ô (dải, kind, nhãn, kiểu, có / không dấu),
     ít hoán đổi nhất: mỗi vòng hoán đổi đủ số dòng để kéo nhãn lệch nhất về giữa; dừng khi phủ định lệch <= 1,5 điểm và biến thể <= 3
     điểm (biên an toàn dưới ngưỡng 2 / 5), tối đa 6 vòng.
Rồi chạy cổng một lần trên mix mới: D3, D5, D6′, D7, D9, D10 (thống kê + đặc trưng cho probe); D4 và probe D10 do d4_probe.py --v3.
Ra results/train_2b/{train_2b_v3.jsonl, manifest_v3.json} + tam/{v3_cong.json, d10_dac_trung_v3.jsonl}. Không ghi đè file cũ.
Dùng: bash setup/run.sh -I scripts/train_2b/build_v3.py"""
import hashlib
import json
import re
import statistics
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import OUT, SEED, TMP, ROOT, Refs, h64, normalize, overlap, read_jsonl, sentences, wset, write_jsonl  # noqa: E402

T0 = time.time()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NEI_W = {"N-bobang": 0.31, "N-them": 0.29, "N-ghep": 0.20, "N-cungbai": 0.10, "N-lacde": 0.10}
LAB = ("supported", "refuted", "nei")
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
NEG = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b")
SRC = OUT / "train_2b_v2_loc.jsonl"
TRAIN = read_jsonl(SRC)
POOL = read_jsonl(OUT / "kc_pool_ngan_v2.jsonl") + read_jsonl(OUT / "kc_pool_dai_v2.jsonl")
ART = {a["id"]: a for a in read_jsonl(TMP / "bai_sach.jsonl")}


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def dia(r):
    return "nodiac" not in set(r.get("flags", []))


def isneg(r):
    return bool(NEG.search(claim_of(r).lower()))


# ------------------------------------------------------------------ bộ lọc (a), (b), (c), chép đúng loc_kc_v2.py
NOI_MO_RONG = ("Ngoài ra", "Tuy nhiên", "Do đó", "Vì vậy", "Vì thế", "Mặt khác", "Bên cạnh đó", "Cũng", "Còn")
CHI_NGUOC = {"này", "đó", "ấy", "khác"}


def next_sentence(r):
    flat = [s for p in ART[r["wiki_id"]]["pars"] for s in sentences(p)]
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
    if "[[" in r["state"] or "]]" in r["state"]:
        return "c_ma_wiki"
    if r["kieu"] == "N-bobang":
        if max((overlap(claim_of(r), s) for s in sentences(r["state"])), default=0) >= 0.9:
            return "a_bobang_van_bao_phu"
        if gay(next_sentence(r)):
            return "b_bobang_cau_sau_chi_nguoc"
    return None


pool_ok = [r for r in POOL if ly_do(r) is None]
loc_bo = Counter(ly_do(r) for r in POOL if ly_do(r))

# ------------------------------------------------------------------ 1. rút lại NEI
C_PRIME = [r for r in TRAIN if not r["source"].startswith("kc_")]
MIX = [r for r in TRAIN if r["source"].startswith("kc_")]
need = Counter((r["source"], r["kind"]) for r in MIX if r["gold_label"] == "nei")
by_cell = defaultdict(list)
for r in pool_ok:
    if r["gold_label"] == "nei":
        by_cell[(r["source"], r["kind"], r["kieu"])].append(r)
new = [r for r in MIX if r["gold_label"] != "nei"]
rut = {}
for (src, kind), n in sorted(need.items()):
    got = []
    for kieu, w in NEI_W.items():
        got += by_cell[(src, kind, kieu)][:int(round(n * w))]
    if len(got) < n:
        ids = {r["id"] for r in got}
        got += [r for r in by_cell[(src, kind, "N-bobang")] if r["id"] not in ids][:n - len(got)]
    got = got[:n]
    assert len(got) == n, (src, kind, n, len(got))
    rut[f"{src}|{kind}"] = dict(Counter(r["kieu"] for r in got))
    new += got
truoc_hoan_doi = list(new)

# ------------------------------------------------------------------ 3. cân phủ định và biến thể bằng hoán đổi trong cùng ô
used = {r["id"] for r in new}
rest = defaultdict(list)
for r in pool_ok:
    if r["id"] not in used:
        rest[(r["source"], r["kind"], r["gold_label"], r["kieu"], dia(r))].append(r)


def stats(rows):
    s = {}
    for l in LAB:
        rs = [r for r in rows if r["gold_label"] == l]; rd = [r for r in rs if dia(r) and r["kind"] == "choice"]
        s[l] = {"neg": 100 * sum(isneg(r) for r in rd) / len(rd), "bt": 100 * sum(bool(r["bien_the"]) for r in rs) / len(rs),
                "ghep": 100 * sum(r["hinh_dang"] == "ghep" for r in rs) / len(rs), "n": len(rs), "nd": len(rd)}
    return s


n_swap = Counter()


def swap_towards(rows, lab, pred_out, pred_in, max_n, tag, only_dia_choice=False):
    n = 0
    for i, r in enumerate(rows):
        if n >= max_n:
            break
        if r["gold_label"] != lab or not pred_out(r) or (only_dia_choice and not (dia(r) and r["kind"] == "choice")):
            continue
        key = (r["source"], r["kind"], r["gold_label"], r["kieu"], dia(r))
        cands = [c for c in rest[key] if pred_in(c)]
        if cands:
            c = cands[0]; rest[key].remove(c); rest[key].append(r); rows[i] = c; n += 1; n_swap[tag] += 1
    return n


for it in range(6):
    s = stats(new); negs = {l: s[l]["neg"] for l in LAB}
    hi, lo = max(negs, key=negs.get), min(negs, key=negs.get)
    if negs[hi] - negs[lo] > 1.5:
        k = int(round((negs[hi] - negs[lo]) / 2 * s[hi]["nd"] / 100)) + 1
        swap_towards(new, hi, isneg, lambda c: not isneg(c) and not c["bien_the"], k, "phu_dinh", only_dia_choice=True)
    s = stats(new); bts = {l: s[l]["bt"] for l in LAB}
    hi, lo = max(bts, key=bts.get), min(bts, key=bts.get)
    if bts[hi] - bts[lo] > 3:
        k = int(round((bts[hi] - bts[lo]) / 2 * s[lo]["n"] / 100)) + 1
        a = swap_towards(new, lo, lambda r: not r["bien_the"] and not isneg(r), lambda c: bool(c["bien_the"]) and not isneg(c), k, "bien_the")
        if a < k:
            swap_towards(new, hi, lambda r: bool(r["bien_the"]) and not isneg(r), lambda c: not c["bien_the"] and not isneg(c), k - a, "bien_the")
    s = stats(new)
    if max(s[l]["neg"] for l in LAB) - min(s[l]["neg"] for l in LAB) <= 1.5 and max(s[l]["bt"] for l in LAB) - min(s[l]["bt"] for l in LAB) <= 3:
        break
assert all(ly_do(r) is None for r in new) and len({r["id"] for r in new}) == len(new) == len(MIX)
assert Counter((r["source"], r["kind"], r["gold_label"]) for r in new) == Counter((r["source"], r["kind"], r["gold_label"]) for r in MIX)
old_ids = {r["id"] for r in MIX}
OUTROWS = sorted(C_PRIME + new, key=lambda r: h64(SEED, "mix", r["id"]))
write_jsonl(OUT / "train_2b_v3.jsonl", OUTROWS)

# ------------------------------------------------------------------ cổng
res = {"loc_pool_v2": dict(loc_bo), "rut_lai_NEI_theo_o": rut, "hoan_doi": dict(n_swap), "so_dong_KC_khac_v2_loc": sum(r["id"] not in old_ids for r in new),
       "so_dong_KC_NEI_khac_v2_loc": sum(r["id"] not in old_ids for r in new if r["gold_label"] == "nei")}
REFS = Refs()
hits = Counter()
for r in new:
    for t in [r["state"], claim_of(r)] + sentences(r["state"]):
        h = REFS.hit(t)
        if h:
            hits[f"{h[0]}|{h[1]}"] += 1; break
res["D3"] = {"so_dong": len(new), "cham": dict(hits), "dat": not hits}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def d5(rows):
    o = {}
    for l in LAB:
        rs = [r for r in rows if r["gold_label"] == l]
        o[l] = {"n": len(rs), "token_trung_vi": q([r["claim_ntok"] for r in rs], 0.5), "ti_le_phu_dinh": round(sum(isneg(r) for r in rs) / max(1, len(rs)), 4)}
    meds = [o[l]["token_trung_vi"] for l in LAB]; negs = [o[l]["ti_le_phu_dinh"] for l in LAB]
    o["lech_trung_vi"] = round((max(meds) - min(meds)) / min(meds), 4); o["lech_phu_dinh_diem"] = round(100 * (max(negs) - min(negs)), 2)
    o["dat"] = o["lech_trung_vi"] <= 0.20 and o["lech_phu_dinh_diem"] <= 2
    return o


md = [r for r in new if dia(r)]
res["D5"] = {"kc_choice_co_dau": d5([r for r in md if r["kind"] == "choice"]), "kc_tat_ca_co_dau": d5(md)}
res["D5"]["dat"] = res["D5"]["kc_choice_co_dau"]["dat"]
cov = lambda c, st: len(wset(c) & wset(st)) / max(1, len(wset(c)))
F = {r["id"]: (max((overlap(claim_of(r), s) for s in sentences(r["state"])), default=0.0), cov(claim_of(r), r["state"]),
               normalize(claim_of(r)) in normalize(r["state"])) for r in md}
med = {l: q([F[r["id"]][0] for r in md if r["gold_label"] == l], 0.5) for l in LAB}
share = lambda l, f: round(sum(f(F[r["id"]]) for r in md if r["gold_label"] == l) / max(1, sum(r["gold_label"] == l for r in md)), 4)
d6 = {"trung_vi_chi_bao": med, "NEI_ge_0_6": share("nei", lambda x: x[0] >= 0.6), "lt_0_9": {l: share(l, lambda x: x[0] < 0.9) for l in LAB},
      "S_doan_lien": share("supported", lambda x: x[2])}
d6["nguong"] = {"NEI_trung_vi_ge_0_40": med["nei"] >= 0.40, "NEI_ge_0_6_chiem_ge_25": d6["NEI_ge_0_6"] >= 0.25,
                "S_lt_0_9_ge_30": d6["lt_0_9"]["supported"] >= 0.30, "R_lt_0_9_ge_30": d6["lt_0_9"]["refuted"] >= 0.30, "S_doan_lien_le_50": d6["S_doan_lien"] <= 0.50}
d6["dat"] = all(d6["nguong"].values())
res["D6_phay"] = d6
fc = [r for r in OUTROWS if r["text_lang"] == "vi" and r["kind"] == "choice" and (r["source"].startswith("kc_") or {"2d", "2e"} & set(r.get("flags", [])))]
c7 = Counter(r["gold_label"] for r in fc); kn = [r for r in new if r["kind"] == "noul"]
res["D7"] = {"n": len(fc), "nhan": dict(c7), "ti_le_nei": round(c7["nei"] / len(fc), 4), "noul_co_kc": round(sum(r["target_argmax"] == 0 for r in kn) / len(kn), 4)}
res["D7"]["dat"] = res["D7"]["ti_le_nei"] >= 0.30 and 0.37 <= res["D7"]["noul_co_kc"] <= 0.40
s = stats(new)
d9 = {l: {"ghep": round(s[l]["ghep"], 2), "bien_the": round(s[l]["bt"], 2)} for l in LAB}
for k in ("ghep", "bien_the"):
    v = [d9[l][k] for l in LAB]; d9[f"lech_{k}_diem"] = round(max(v) - min(v), 2)
d9["dat"] = d9["lech_ghep_diem"] <= 5 and d9["lech_bien_the_diem"] <= 5
res["D9"] = d9
res["kieu_KC"] = dict(Counter(r["kieu"] for r in new))
res["co_cau_NEI"] = {k: round(v / sum(r["gold_label"] == "nei" for r in new), 4) for k, v in Counter(r["kieu"] for r in new if r["gold_label"] == "nei").items()}
# D10: thống kê + đặc trưng cho probe
PX = [r for r in read_jsonl(ROOT / "data/proxy_v1/proxy_v1.jsonl") if r["source"] in ("vifactcheck", "viwikifc") and r["variant"] == "orig"]
feat = [{"bo": "kc", "id": r["id"], "kind": r["kind"], "nhan": r["gold_label"], "max_ov": round(F[r["id"]][0], 4), "cover": round(F[r["id"]][1], 4)} for r in md]
feat += [{"bo": "proxy_v1", "id": r["id"], "kind": r["kind"], "nhan": r["gold_label"],
          "max_ov": round(max((overlap(claim_of(r), x) for x in sentences(r["state"])), default=0.0), 4), "cover": round(cov(claim_of(r), r["state"]), 4)} for r in PX]
write_jsonl(TMP / "d10_dac_trung_v3.jsonl", feat)
d2e = [r for r in OUTROWS if r["text_lang"] == "vi" and r["kind"] == "choice" and {"2d", "2e"} & set(r.get("flags", []))]
blk = [r for r in new if r["kind"] == "choice" and dia(r)] + d2e
cv = [(cov(claim_of(r), r["state"]), r["gold_label"]) for r in blk]
ph = statistics.mean(l == "nei" for c, l in cv if c >= 0.9); pl = statistics.mean(l == "nei" for c, l in cv if c < 0.9)
res["D10_thong_ke"] = {"do_phu_token_trung_vi_kc": {l: q([F[r["id"]][1] for r in md if r["gold_label"] == l], 0.5) for l in LAB},
                       "do_phu_token_trung_vi_proxy": {l: q([f["cover"] for f in feat if f["bo"] == "proxy_v1" and f["nhan"] == l], 0.5) for l in LAB},
                       "muc_khoi_P_NEI_cover_ge_0_9": round(ph, 3), "muc_khoi_P_NEI_cover_lt_0_9": round(pl, 3), "ti_so": round(pl / ph, 2),
                       "probe": "tam/d4_d10_probe_v3.json"}
man = {"luat": "results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md mục 9.7; brief 7c 'Sai lệch có ngày'", "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "dau_vao": {SRC.name: sha(SRC), "kc_pool_ngan_v2.jsonl": sha(OUT / "kc_pool_ngan_v2.jsonl"), "kc_pool_dai_v2.jsonl": sha(OUT / "kc_pool_dai_v2.jsonl")},
       "file": {"train_2b_v3.jsonl": sha(OUT / "train_2b_v3.jsonl")}, "so_dong": len(OUTROWS), "so_dong_KC": len(new), "seed": SEED,
       "co_cau_NEI_dich": NEI_W, "ghi_chu": "C′ và dòng KC Xác nhận / Bác bỏ giữ như train_2b_v2_loc, trừ dòng bị hoán đổi ở bước cân", **res,
       "giay": round(time.time() - T0)}
(OUT / "manifest_v3.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
(TMP / "v3_cong.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
