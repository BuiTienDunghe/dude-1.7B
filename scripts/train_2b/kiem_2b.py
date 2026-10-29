"""Mục 5 + 6 của results/BRIEF_2b_du_lieu.md: ráp train_2b_mac_dinh.jsonl (C′ + 1.800 KC-ngắn + 1.200 KC-dài, xáo theo seed) và chạy cổng
D1, D2, D3, D5, D6, D7, D8 (D4 chạy riêng bằng scripts/train_2b/d4_probe.py vì cần sklearn). Không chạy model; tokenizer đã đếm ở bước dựng.
Chỉ báo số và áp ngưỡng đã chốt trước; không tự quyết bước tiếp.
Ra results/train_2b/{train_2b_mac_dinh.jsonl, mau_doc.jsonl, manifest_2b.json, tam/kiem_2b.json, tam/d1_*.json}.
Dùng: bash setup/run.sh -I scripts/train_2b/kiem_2b.py"""
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import DEV, OUT, ROOT, SEED, TMP, Refs, h64, normalize, read_jsonl, sentences, u01, write_jsonl  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from common import strip_diacritics  # noqa: E402

T0 = time.time()
N_MIX = {"kc_wiki_ngan": 1800, "kc_wiki_dai": 1200}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
C = read_jsonl(ROOT / "results/train_v1/components/C.jsonl")
CP = read_jsonl(OUT / "C_prime.jsonl")
PN, PD, VAL = read_jsonl(OUT / "kc_pool_ngan.jsonl"), read_jsonl(OUT / "kc_pool_dai.jsonl"), read_jsonl(OUT / "val_kc.jsonl")
dev = read_jsonl(DEV)
KC_MIX = PN[:N_MIX["kc_wiki_ngan"]] + PD[:N_MIX["kc_wiki_dai"]]
TRAIN = sorted(CP + KC_MIX, key=lambda r: h64(SEED, "mix", r["id"]))
assert len({r["id"] for r in TRAIN}) == len(TRAIN)
write_jsonl(OUT / "train_2b_mac_dinh.jsonl", TRAIN)
res = {}
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")


def claim_of(r):
    for mk in MK:
        if mk in r["question"]:
            return r["question"].split(mk, 1)[1]
    return None


def flags(r):
    return set(r.get("flags", []))


# ------------------------------------------------------------------ thành phần
def comp(rows):
    c = Counter()
    for r in rows:
        c[f"{r['text_lang']}|tong"] += 1
        c[f"{r['text_lang']}|{r['source']}"] += 1
    return c


cc, cp, ct = comp(C), comp(CP), comp(TRAIN)
res["thanh_phan"] = {k: {"C": cc.get(k, 0), "C_prime": cp.get(k, 0), "train_2b": ct.get(k, 0)}
                     for k in sorted(set(cc) | set(cp) | set(ct), key=lambda k: (k.split("|")[0], -ct.get(k, 0)))}
res["kind_vi"] = {n: dict(Counter(r["kind"] for r in rows if r["text_lang"] == "vi")) for n, rows in (("C", C), ("C_prime", CP), ("train_2b", TRAIN))}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def lens(rows):
    t = [r["n_state_tokens"] for r in rows if r.get("n_state_tokens") is not None]
    return {"n": len(t), "trung_vi": q(t, 0.5), "p90": q(t, 0.9), "max": max(t) if t else None}


DEVV = [r for r in dev if r["lang"] == "vi" and r["variant"] == "orig"]
res["so_lua_chon_va_do_dai"] = {
    "dev_vi_theo_nguon": {s: {"so_lua_chon": dict(Counter(len(r["options"]) for r in DEVV if r["source"] == s)),
                              "token_state": lens([r for r in DEVV if r["source"] == s])} for s in sorted({r["source"] for r in DEVV})},
    "train_2b_vi_theo_khoi": {}}
blocks = defaultdict(list)
for r in TRAIN:
    if r["text_lang"] != "vi":
        continue
    f = flags(r)
    b = (r["source"] if r["source"].startswith("kc_") else "2e" if "2e" in f else "2d" if "2d" in f else
         "doc_hieu" if r["source"] in ("vsmrc/mrc", "race/high", "race/middle", "dream", "cosmos_qa", "quail") else "vi_khac")
    blocks[b].append(r)
for b, rows in sorted(blocks.items()):
    res["so_lua_chon_va_do_dai"]["train_2b_vi_theo_khoi"][b] = {"n": len(rows), "so_lua_chon": dict(Counter(len(r["options"]) for r in rows)),
                                                               "token_state": lens(rows)}

# ------------------------------------------------------------------ D1
def run_d1(path, name):
    out = TMP / f"d1_{name}.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/check_label_strings.py"), str(path), "--out", str(out)],   # không -I: giữ PYTHONHASHSEED=0
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    return json.loads(out.read_text(encoding="utf-8"))


d1c = run_d1(ROOT / "results/train_v1/components/C.jsonl", "C")
d1t = run_d1(OUT / "train_2b_mac_dinh.jsonl", "train_2b")
allkc = TMP / "kc_tat_ca.jsonl"
write_jsonl(allkc, PN + PD + VAL)
d1k = run_d1(allkc, "kc_pool_va_val")


def pairs(d1):
    return {(x["chu_train"], x["chu_dev_gan"], x["khung"]): x["so_lan"] for v in d1["gan_trung_LECH_dev"].values() for x in v}


pc, pt, pk = pairs(d1c), pairs(d1t), pairs(d1k)
NEIS = {r["options"][r["target"]] for r in dev if r["source"] in ("vifactcheck", "viwikifc") and r["gold_label"] == "nei" and r["kind"] == "choice"}
cam = lambda k: k[2] in ("kiem_chung", "co_khong") or k[1] in NEIS
res["D1"] = {"C": {"chuoi_gan_trung_lech": len(pc), "trong_khung_cam": sum(v for k, v in pc.items() if cam(k))},
             "train_2b": {"chuoi_gan_trung_lech": len(pt), "trong_khung_cam": sum(v for k, v in pt.items() if cam(k)),
                          "danh_sach": [{"train": k[0], "dev": k[1], "khung": k[2], "so_lan": v} for k, v in sorted(pt.items(), key=lambda x: -x[1])],
                          "them_so_voi_C": sorted({(k[0], k[2]) for k in pt} - {(k[0], k[2]) for k in pc})},   # so theo chữ train + khung
             "kc_pool_va_val": {"chuoi_gan_trung_lech": len(pk), "trong_khung_cam": sum(v for k, v in pk.items() if cam(k))},
             "C_danh_sach": [{"train": k[0], "dev": k[1], "khung": k[2], "so_lan": v} for k, v in sorted(pc.items(), key=lambda x: -x[1])]}
res["D1"]["dat"] = (res["D1"]["train_2b"]["trong_khung_cam"] == 0 and res["D1"]["kc_pool_va_val"]["trong_khung_cam"] == 0
                    and not res["D1"]["train_2b"]["them_so_voi_C"] and res["D1"]["kc_pool_va_val"]["chuoi_gan_trung_lech"] == 0)

# ------------------------------------------------------------------ D2
DEV_OPT = {o for r in dev if r["source"] in ("vifactcheck", "viwikifc") for o in r["options"]}
DEV_PRE = {r["question"].split(mk, 1)[0] + mk for r in dev if r["source"] in ("vifactcheck", "viwikifc") for mk in MK if mk in r["question"]}
bad = Counter(); checked = Counter()
for r in PN + PD + VAL + [x for x in CP if {"2d", "2e"} & flags(x)]:
    tag = r["source"] if r["source"].startswith("kc_") else ("2e" if "2e" in flags(r) else "2d")
    checked[tag] += 1
    pre = next((r["question"].split(mk, 1)[0] + mk for mk in MK if mk in r["question"]), None)
    if pre not in DEV_PRE:
        bad[f"{tag}|cau_hoi"] += 1
    if any(o not in DEV_OPT for o in r["options"]):
        bad[f"{tag}|lua_chon"] += 1
for r in CP:
    if r.get("aug") == "abstain_add" and r["options"][-1] not in DEV_OPT:
        bad["2h_add|chuoi_nei"] += 1
    if r.get("aug") == "abstain_replace" and r["options"][r["target_argmax"]] not in DEV_OPT:
        bad["2h_replace|chuoi_nei"] += 1
    if r.get("aug"):
        checked[r["aug"]] += 1
new_pre = {next((r["question"].split(mk, 1)[0] + mk for mk in MK if mk in r["question"]), None) for r in PN + PD + VAL} - DEV_PRE
res["D2"] = {"so_dong_kiem": dict(checked), "vi_pham": dict(bad), "chuoi_khung_moi": sorted(x for x in new_pre if x),
             "ngoai_le": "2h: câu hỏi và lựa chọn đọc hiểu gốc (chỉ kiểm chuỗi NEI thêm vào)", "dat": not bad and not new_pre}

# ------------------------------------------------------------------ D3
REFS = Refs()
hits = Counter(); hit_rows = []
d3_rows = PN + PD + VAL + [x for x in CP if "2e" in flags(x) and str(x.get("van_ban_tu", "")).startswith("wiki:")]
for r in d3_rows:
    texts = [r["state"], claim_of(r) or ""] + sentences(r["state"])
    for t in texts:
        h = REFS.hit(t)
        if h:
            hits[f"{h[0]}|{h[1]}"] += 1; hit_rows.append(r["id"]); break
res["D3"] = {"so_dong_kiem": len(d3_rows), "cham": dict(hits), "dong_cham": hit_rows[:50], "dat": not hits,
             "tham_chieu": REFS.n_by_tag}
# thêm, chỉ báo: dòng 2d / 2f / 2h của C′ (văn bản có sẵn từ C) có chạm dev / test 1.3 không
info = Counter()
for r in CP:
    if {"2d", "2f"} & flags(r) or r.get("aug"):
        for t in [r["state"], claim_of(r) or r["question"]]:
            h = REFS.hit(t)
            if h:
                info[f"{h[0]}|{h[1]}"] += 1; break
res["D3"]["chi_bao_C_prime_2d_2f_2h"] = dict(info)

# ------------------------------------------------------------------ D5, D6
NEG = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b")
NUMD = re.compile(r"\d")
KCD = [r for r in PN + PD if "nodiac" not in flags(r)]


def stats(rows):
    out = {}
    for lab in ("supported", "refuted", "nei"):
        rs = [r for r in rows if r["gold_label"] == lab]
        cl = [claim_of(r) for r in rs]
        out[lab] = {"n": len(rs), "token_trung_vi": q([r["claim_ntok"] for r in rs], 0.5), "token_p90": q([r["claim_ntok"] for r in rs], 0.9),
                    "ti_le_phu_dinh": round(sum(bool(NEG.search(c.lower())) for c in cl) / max(1, len(cl)), 4),
                    "ti_le_co_so_hoac_ngay": round(sum(bool(NUMD.search(c)) for c in cl) / max(1, len(cl)), 4),
                    "ti_le_nam_nguyen_trong_van_ban": round(sum(normalize(c) in normalize(r["state"]) for c, r in zip(cl, rs)) / max(1, len(cl)), 4),
                    "ti_le_cau_nguyen_khong_luoc_khong_doi": round(sum(r["kieu"] in ("S-nguyen", "N-cungbai", "N-lacde") and not r["luoc"]
                                                                       for r in rs) / max(1, len(rs)), 4),
                    "ti_le_luoc": round(sum(bool(r["luoc"]) for r in rs) / max(1, len(rs)), 4)}
    meds = [v["token_trung_vi"] for v in out.values()]
    negs = [v["ti_le_phu_dinh"] for v in out.values()]
    out["lech_trung_vi"] = round((max(meds) - min(meds)) / min(meds), 4)
    out["lech_phu_dinh_diem"] = round(100 * (max(negs) - min(negs)), 2)
    out["dat"] = out["lech_trung_vi"] <= 0.20 and out["lech_phu_dinh_diem"] <= 2
    return out


res["D5"] = {"kc_choice_co_dau": stats([r for r in KCD if r["kind"] == "choice"]), "kc_tat_ca_co_dau": stats(KCD),
             "kieu": dict(Counter(r["kieu"] for r in PN + PD)),
             "S_luoc_tren_S": round(sum(r["kieu"] == "S-luoc" for r in PN + PD) / max(1, sum(r["gold_label"] == "supported" for r in PN + PD)), 4)}
res["D5"]["dat"] = res["D5"]["kc_choice_co_dau"]["dat"] and res["D5"]["S_luoc_tren_S"] >= 1 / 3


def max_ov(claim, state):
    from kc_util import overlap
    return max((overlap(claim, s) for s in sentences(state)), default=0.0)


def dist(xs):
    return {"n": len(xs), "tb": round(statistics.mean(xs), 3) if xs else None, "p10": q(xs, 0.1), "trung_vi": q(xs, 0.5), "p90": q(xs, 0.9),
            "ti_le_ge_0_9": round(sum(x >= 0.9 for x in xs) / max(1, len(xs)), 3)}


PX = [r for r in read_jsonl(ROOT / "data/proxy_v1/proxy_v1.jsonl") if r["source"] in ("vifactcheck", "viwikifc") and r["variant"] == "orig"]
res["D6"] = {"kc_co_dau": {lab: dist([round(max_ov(claim_of(r), r["state"]), 3) for r in KCD if r["gold_label"] == lab]) for lab in ("supported", "refuted", "nei")},
             "proxy_v1_kiem_chung": {lab: dist([round(max_ov(claim_of(r), r["state"]), 3) for r in PX if r["gold_label"] == lab])
                                     for lab in ("supported", "refuted", "nei")},
             "ghi_chu": "chỉ báo, không chặn"}

# ------------------------------------------------------------------ D7
VI_FC_CHOICE = [r for r in TRAIN if r["text_lang"] == "vi" and r["kind"] == "choice" and (r["source"].startswith("kc_") or {"2d", "2e"} & flags(r))]
lab_of = lambda r: r["gold_label"]
c7 = Counter(lab_of(r) for r in VI_FC_CHOICE)
kc_noul = [r for r in KC_MIX if r["kind"] == "noul"]
fc_noul = [r for r in TRAIN if r["text_lang"] == "vi" and r["kind"] == "noul" and (r["source"].startswith("kc_") or "2d" in flags(r))]
res["D7"] = {"kiem_chung_vi_choice": {"n": len(VI_FC_CHOICE), "nhan": dict(c7), "ti_le_nei": round(c7["nei"] / len(VI_FC_CHOICE), 4),
                                      "theo_khoi": {b: dict(Counter(lab_of(r) for r in VI_FC_CHOICE if (r["source"] if r["source"].startswith("kc_")
                                                                                                         else "2e" if "2e" in flags(r) else "2d") == b))
                                                    for b in ("2d", "2e", "kc_wiki_ngan", "kc_wiki_dai")}},
             "noul_co_kc_mix": round(sum(r["target_argmax"] == 0 for r in kc_noul) / max(1, len(kc_noul)), 4),
             "noul_co_kiem_chung_vi_ca_2d": round(sum(r["target_argmax"] == 0 for r in fc_noul) / max(1, len(fc_noul)), 4),
             "dev_noul_co": round(sum(r["target"] == 0 for r in DEVV if r["kind"] == "noul") / max(1, sum(r["kind"] == "noul" for r in DEVV)), 4)}
res["D7"]["dat"] = res["D7"]["kiem_chung_vi_choice"]["ti_le_nei"] >= 0.30 and 0.37 <= res["D7"]["noul_co_kc_mix"] <= 0.40

# ------------------------------------------------------------------ D8: mẫu đọc mù
L = "ABCDEFGHIJ"


def sample(rows, n, tag):
    return sorted(rows, key=lambda r: h64(SEED, "mau", tag, r["id"]))[:n]


def rec(r, nhom, extra=None):
    t = r["target_argmax"]
    d = {"ma": None, "id": r["id"], "nhom": nhom, "dataset": r["dataset"], "kind": r["kind"], "state": r["state"], "question": r["question"],
         "options": [f"{L[i]}. {o}" for i, o in enumerate(r["options"])], "nhan": f"{L[t]}. {r['options'][t]}"}
    d.update({f"nhan_{k}": v for k, v in (extra or {}).items()})
    return d


mau = []
for kieu in ("S-nguyen", "S-luoc", "R-so", "R-ten", "N-cungbai", "N-lacde"):
    for r in sample([x for x in PN + PD if x["kieu"] == kieu], 30, kieu):
        mau.append(rec(r, "KC", {"kieu": kieu, "luoc": r["luoc"], "cau_goc": r["cau_goc"], "doi": r.get("doi"), "tieu_de": r["tieu_de"],
                                 "bai_khac": r.get("tieu_de_khac")}))
for r in sample([x for x in CP if "2d" in flags(x)], 30, "2d"):
    mau.append(rec(r, "2d", {"nguon": r["source"], "ban": "dịch (vitaminc / nli_fever là bản dịch từ tiếng Anh)", "gold_label": r["gold_label"]}))
for r in sample([x for x in CP if "2f" in flags(x)], 30, "2f"):
    mau.append(rec(r, "2f", {"nguon": r["source"], "cau_hoi_goc": r.get("question_goc")}))
for r in sample([x for x in CP if "2g" in flags(x)], 30, "2g"):
    mau.append(rec(r, "2g", {"nguon": r["source"]}))
for aug in ("abstain_add", "abstain_replace"):
    for r in sample([x for x in CP if x.get("aug") == aug], 20, aug):
        mau.append(rec(r, "2h", {"aug": aug, "nguon": r["source"], "lua_chon_goc": r.get("options_goc"), "dong_cho": r.get("aug_cho")}))
mau.sort(key=lambda m: h64(SEED, "mau_xao", m["id"]))
for i, m in enumerate(mau):
    m["ma"] = f"M{i + 1:03d}"
write_jsonl(OUT / "mau_doc.jsonl", mau)
res["D8"] = {"so_dong": len(mau), "theo_nhom": dict(Counter(m["nhom"] for m in mau)),
             "cot_can_che": "nhan và mọi cột bắt đầu bằng nhan_ (kiểu dựng, câu gốc, chữ bị đổi, nguồn)"}

# ------------------------------------------------------------------ manifest
cpl = json.loads((TMP / "c_prime_log.json").read_text(encoding="utf-8"))
kcl = json.loads((TMP / "kc_log.json").read_text(encoding="utf-8"))
loc = json.loads((TMP / "loc_bai.json").read_text(encoding="utf-8"))
tai = json.loads((OUT / "manifest_tai_wiki.json").read_text(encoding="utf-8"))
man = {"brief": "results/BRIEF_2b_du_lieu.md", "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "seed_common": 20261012,
       "dau_vao": {"C.jsonl": cpl["sha256"], "dev_v1_3": sha(DEV), "quyet_C.jsonl": sha(ROOT / "results/teacher_diag/quyet_C.jsonl")},
       "wikipedia": {"repo": tai["repo"], "revision": tai["revision"], "lfs_oid": {k: v["lfs_oid_sha256"] for k, v in tai["file"].items()}},
       "file": {p.name: sha(p) for p in [OUT / "C_prime.jsonl", OUT / "kc_pool_ngan.jsonl", OUT / "kc_pool_dai.jsonl", OUT / "val_kc.jsonl",
                                          OUT / "train_2b_mac_dinh.jsonl", OUT / "mau_doc.jsonl"]},
       "so_dong": {"C": len(C), "C_prime": len(CP), "kc_pool_ngan": len(PN), "kc_pool_dai": len(PD), "val_kc": len(VAL), "train_2b_mac_dinh": len(TRAIN),
                   "mix_mac_dinh": N_MIX},
       "so_dong_theo_nguon": res["thanh_phan"], "c_prime_viec": cpl["log"], "chu_tu_dev": cpl["chu_tu_dev"], "anh_xa_nli_fever": cpl["anh_xa_nli_fever"],
       "loc_bai_wiki": loc, "kc": kcl, "dong_bi_bo": cpl["dong_bi_bo"],
       "dong_bi_bo_theo_ly_do": dict(Counter(d["ly_do"] for d in cpl["dong_bi_bo"])), "giay_kiem": round(time.time() - T0)}
(OUT / "manifest_2b.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
(TMP / "kiem_2b.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: (v.get("dat") if isinstance(v, dict) else None) for k, v in res.items() if k.startswith("D")}, ensure_ascii=False))
print(json.dumps({"D7": res["D7"], "D5": res["D5"]["kc_choice_co_dau"], "D3": {k: res["D3"][k] for k in ("so_dong_kiem", "cham", "dat")}},
                 ensure_ascii=False, indent=1))
