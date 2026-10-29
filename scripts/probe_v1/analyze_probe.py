"""Phân tích xu hướng theo BRIEF_phan_tich_xu_huong.md (mục 2, 3, 4 và "Cập nhật 26/10") trên preds ĐÃ CÓ: không chấm model, không train,
chỉ dev 1.3 và probe_v1. Thư viện chuẩn + numpy (python Windows chạy được). Mọi ngưỡng, lát, seed chốt ở đây TRƯỚC khi có model M2.

Dùng: python scripts/probe_v1/analyze_probe.py --m1 "<tên>" [--m2 "<tên>"] [--m0 "<tên>"] [--r "<tên>" ...]
        [--dev_eval results/bench_v1_3_trial/eval] [--probe_eval results/probe_v1/eval] [--out results/probe_v1/phan_tich/<tên>.json]
  <tên> = thư mục con chứa preds.jsonl. Preds dev tìm ở --dev_eval/<tên>/ rồi --probe_eval/dev/[normal/]<tên>/; preds phép thử ở
  --probe_eval/<phép>/[normal/]<tên>/. Thiếu preds phép thử nào thì bỏ phần đó, ghi "chua_co". Không có --m2 thì bỏ mọi phần M2 (chấm mốc).
Ra (JSON + .md cùng tên):
  lat_cat    : accuracy từng model theo mọi lát của mục 2, khoảng tin cậy bootstrap 95% theo cụm (trường cluster của dev: cùng đoạn văn thì
               cùng cụm), 1.000 lần, seed cố định; nếu có M2: chênh M2 − M1 theo cặp câu, CI bootstrap cùng cụm, McNemar chính xác.
  chuyen_doi : ma trận M1 → M2 theo nguồn (khi có M2).
  trung_loi  : từng model M với từng R: Jaccard tập câu sai, P(R sai | M sai), tỉ lệ sai riêng của R.
  hieu_chinh : Brier (đa lớp), ECE 10 ngăn đều theo xác suất cao nhất, độ tự tin trung bình khi sai / khi đúng.
  phep_thu   : P1′ (p1b), P5, Pa (lựa chọn bỏ dấu), Pb (nhãn tiếng Anh): so theo cặp với cùng câu ở dev (CI, McNemar); Pa thêm mốc dòng
               không dấu toàn phần, Pb thêm mốc dòng EN cùng cặp; P1′ − P5 theo cặp (phần do đổi chữ nhãn). P2: tỉ lệ chọn cùng đáp án khi
               đảo. P3: so đoán mò và "luôn chọn đáp án hay gặp". P4: tỉ lệ chọn "không đủ thông tin"/"Không", kèm mốc thiên vị sẵn có
               (trên câu dev gốc của cùng nhận định, model chọn nhãn đó khi đáp án KHÔNG phải nó).
  so_cau_bi_loai: số câu model ngoài loại (excluded / rejected / truncated) theo phép thử.
  nham_nhan  : (c) SIB-200 / MASSIVE: phân bố nhãn dự đoán so với phân bố đáp án, tỉ lệ dồn vào nhãn theo dõi ("health", "sports" ở SIB;
               "recommendation_locations" ở MASSIVE), khoảng cách biến phân toàn phần, ma trận nhầm (SIB đủ; MASSIVE 10 cặp nhầm nhiều nhất).
  p3_phan_bo_nhan: (d) P3 kiểm chứng: phân bố nhãn dự đoán khi bỏ văn bản, so với phân bố đáp án và phân bố dự đoán ở dev cùng câu."""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import CONFIG, SEED, read_jsonl, sha256_file, strip_diacritics  # noqa: E402

DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
PR = ROOT / "results/probe_v1"
PROBE_FILES = {"p1b": "p1b_doi_chu_nhan.jsonl", "p3": "p3_bo_van_ban.jsonl", "p4": "p4_van_ban_khong_lien_quan.jsonl",
               "p5": "p5_doi_cau_hoi.jsonl", "pa": "pa_lua_chon_khong_dau.jsonl", "pb": "pb_nhan_tieng_anh.jsonl"}
PAIRED = ("p1b", "p5", "pa", "pb")
N_BOOT = 1000
FC_LAB = {"Được văn bản xác nhận": "supported", "Bị văn bản bác bỏ": "refuted", "Văn bản không đủ thông tin để kết luận": "nei",
          "Có": "co", "Không": "khong"}
FC_LAB.update({strip_diacritics(k): v for k, v in list(FC_LAB.items())})
NEI_LIKE = {k for k, v in FC_LAB.items() if v in ("nei", "khong")}
THEO_DOI = {"sib200": ("health", "sports"), "massive": ("recommendation_locations",)}

ap = argparse.ArgumentParser()
ap.add_argument("--m1", required=True)
ap.add_argument("--m2")
ap.add_argument("--m0")
ap.add_argument("--r", action="append", default=[])
ap.add_argument("--dev_eval", type=Path, default=ROOT / "results/bench_v1_3_trial/eval")
ap.add_argument("--probe_eval", type=Path, default=PR / "eval")
ap.add_argument("--out", type=Path)
ap.add_argument("--probe_dir", type=Path, default=PR)   # thư mục chứa file phép thử (chỉ đổi khi tự kiểm script)
a = ap.parse_args()
a.dev_eval, a.probe_eval, a.probe_dir = a.dev_eval.resolve(), a.probe_eval.resolve(), a.probe_dir.resolve()
PR = a.probe_dir

assert sha256_file(DEV) == DEV_SHA, "dev 1.3 sai sha"
dev = read_jsonl(DEV)
D = {r["id"]: r for r in dev}
BY = {(r["pair_id"], r["lang"], r["variant"]): r for r in dev}
MODELS = {"M1": a.m1, **({"M2": a.m2} if a.m2 else {}), **({"M0": a.m0} if a.m0 else {}), **{f"R{i + 1}": n for i, n in enumerate(a.r)}}
HAS_M2 = "M2" in MODELS
M_CODES = [c for c in MODELS if c.startswith("M")]


def load(path):
    out, n_ex, n = {}, 0, 0
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        n += 1
        if p.get("excluded") or p.get("probs") is None:
            n_ex += 1
            continue
        out[p["id"]] = p
    return out, {"so_dong": n, "bi_loai": n_ex}


def find(phep, name):
    roots = ([a.dev_eval / name] if phep == "dev" else []) + [a.probe_eval / phep / name, a.probe_eval / phep / "normal" / name]
    for d in roots:
        if (d / "preds.jsonl").exists():
            return d / "preds.jsonl"
    return None


PRED, EXCL, SRC = {}, defaultdict(dict), {}
for code, name in MODELS.items():
    p = find("dev", name)
    assert p, f"không thấy preds dev của {name}"
    PRED[(code, "dev")], EXCL[code]["dev"] = load(p)
    SRC[code] = str(p.relative_to(ROOT))
    for ph in list(PROBE_FILES) + ["p2"]:
        q = find(ph, name)
        if q:
            PRED[(code, ph)], EXCL[code][ph] = load(q)


# ---------------------------------------------------------------- lát cắt (mục 2)
def k_bucket(k):
    return "2" if k == 2 else ("3-4" if k <= 4 else ">=5")


len_vi = {r["pair_id"]: r["n_state_tokens"] for r in dev if r["lang"] == "vi" and r["variant"] == "orig"}
CUT = {}
for src in sorted({r["source"] for r in dev}):
    v = sorted(len_vi[r["pair_id"]] for r in dev if r["source"] == src and r["lang"] == "vi" and r["variant"] == "orig")
    CUT[src] = (v[len(v) // 3], v[2 * len(v) // 3])   # ngưỡng tam phân theo độ dài state VI có dấu của cặp, riêng từng nguồn
revised_pair = {r["pair_id"] for r in dev if r.get("vi_provenance") == "revised" or r.get("vi_revised_fields")}


def slices(r):
    """Các (lát, nhóm) mà một dòng dev thuộc về."""
    s = [("tat_ca", "tat_ca"), ("nguon", r["source"]), ("ngon_ngu", f"{r['lang']}|{r['variant']}"),
         ("kieu_cau", "noul" if r["kind"] == "noul" else "chon_mot"), ("so_lua_chon", k_bucket(len(r["options"])))]
    if r["lang"] == "vi" and r["variant"] == "orig":
        s.append(("goc_vs_dich (VI có dấu)", "nguon_goc_VI" if r["block"] == "native" else "nguon_dich"))
        s.append(("nguon (VI có dấu)", r["source"]))
    lo, hi = CUT[r["source"]]
    L = len_vi.get(r["pair_id"], r["n_state_tokens"])
    s.append(("do_dai_van_ban", "ngan" if L <= lo else ("vua" if L <= hi else "dai")))
    if r["lang"] == "vi":
        s.append(("da_sua_o_1_2 (dòng VI)", "da_sua" if r["pair_id"] in revised_pair else "giu_nguyen"))
    s.append(("mo_ho", "co_co" if r.get("mo_ho") else "khong_co"))
    return s


rng = np.random.default_rng(SEED)
ORIG = {}


def cluster_of(i):
    return D[i]["cluster"] if i in D else D[ORIG[i]]["cluster"]


def boot_ci(ids, val):
    """CI 95% bootstrap theo cụm (cluster của dev) cho trung bình của val(id) trên các id: rút lại cụm có hoàn lại, N_BOOT lần."""
    if not ids:
        return [None, None]
    rng = np.random.default_rng(SEED)   # seed riêng mỗi lần: CI của một nhóm không đổi theo tập model hay thứ tự tính
    s, n = defaultdict(float), defaultdict(int)
    for i in ids:
        c = cluster_of(i)
        s[c] += float(val(i)); n[c] += 1
    keys = sorted(s)
    S, N = np.array([s[k] for k in keys]), np.array([n[k] for k in keys])
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    v = np.sort(S[idx].sum(1) / N[idx].sum(1))
    return [round(float(v[int(0.025 * N_BOOT)]), 4), round(float(v[int(0.975 * N_BOOT) - 1]), 4)]


def mcnemar(b, c):
    """p hai phía chính xác (nhị thức) trên số cặp lệch b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(Pa_, Pb_, ids, ka=lambda i: i, kb=lambda i: i):
    """So theo cặp: acc A (id ka(i)), acc B (id kb(i)), chênh B − A, CI bootstrap theo cụm, McNemar."""
    ids = [i for i in ids if ka(i) in Pa_ and kb(i) in Pb_]
    if not ids:
        return None
    ca = lambda i: int(Pa_[ka(i)]["correct"]); cb = lambda i: int(Pb_[kb(i)]["correct"])
    b = sum(ca(i) and not cb(i) for i in ids); c = sum(cb(i) and not ca(i) for i in ids)
    acc_a = sum(map(ca, ids)) / len(ids); acc_b = sum(map(cb, ids)) / len(ids)
    return {"n": len(ids), "acc_A": round(acc_a, 4), "acc_B": round(acc_b, 4), "chenh_B_tru_A": round(acc_b - acc_a, 4),
            "ci95": boot_ci(ids, lambda i: cb(i) - ca(i)), "mcnemar_p": round(mcnemar(b, c), 4)}


groups = defaultdict(list)
for r in dev:
    for s in slices(r):
        groups[s].append(r["id"])
lat = {}
for (sl, g), ids in sorted(groups.items()):
    ent = {"so_cau": len(ids)}
    for code in MODELS:
        P = PRED[(code, "dev")]
        ii = [i for i in ids if i in P]
        ent[code] = {"acc": round(sum(P[i]["correct"] for i in ii) / max(1, len(ii)), 4), "ci95": boot_ci(ii, lambda i, P=P: P[i]["correct"]), "n": len(ii)}
    if HAS_M2:
        x = paired(PRED[("M1", "dev")], PRED[("M2", "dev")], ids)
        ent["M2_tru_M1"] = {"chenh": x["chenh_B_tru_A"], "ci95": x["ci95"], "mcnemar_p": x["mcnemar_p"]}
    lat.setdefault(sl, {})[g] = ent

# ---------------------------------------------------------------- ma trận chuyển đổi M1 -> M2 (mục 4)
chuyen = {}
if HAS_M2:
    for src in sorted({r["source"] for r in dev}) + ["tat_ca"]:
        m = Counter()
        for r in dev:
            i = r["id"]
            if src in ("tat_ca", r["source"]) and i in PRED[("M1", "dev")] and i in PRED[("M2", "dev")]:
                x, y = PRED[("M1", "dev")][i]["correct"], PRED[("M2", "dev")][i]["correct"]
                m[("dung" if x else "sai") + "->" + ("dung" if y else "sai")] += 1
        chuyen[src] = dict(sorted(m.items()))

# ---------------------------------------------------------------- trùng lỗi với R (mục 4)
trung = {}
for rc in [c for c in MODELS if c.startswith("R")]:
    R = PRED[(rc, "dev")]
    for mc in M_CODES:
        M = PRED[(mc, "dev")]
        common = [i for i in M if i in R]
        em, er = {i for i in common if not M[i]["correct"]}, {i for i in common if not R[i]["correct"]}
        per = {}
        for src in sorted({D[i]["source"] for i in common}):
            a_ = {i for i in em if D[i]["source"] == src}; b_ = {i for i in er if D[i]["source"] == src}
            per[src] = round(len(a_ & b_) / max(1, len(a_ | b_)), 4)
        trung[f"{mc}_voi_{rc} ({MODELS[rc]})"] = {
            "so_cau_chung": len(common), "jaccard_tap_sai": round(len(em & er) / max(1, len(em | er)), 4),
            "P(R sai | M sai)": round(len(em & er) / max(1, len(em)), 4), "ti_le_sai_cua_R": round(len(er) / max(1, len(common)), 4),
            "ti_le_sai_cua_M": round(len(em) / max(1, len(common)), 4), "jaccard_theo_nguon": per}


# ---------------------------------------------------------------- hiệu chỉnh (P7)
def calib(P, ids):
    ids = [i for i in ids if i in P]
    if not ids:
        return None
    br, bins, cw, cr = 0.0, defaultdict(list), [], []
    for i in ids:
        p = P[i]["probs"]; t = P[i]["target"]
        br += sum((q - (1.0 if j == t else 0.0)) ** 2 for j, q in enumerate(p))
        conf = max(p)
        bins[min(9, int(conf * 10))].append((conf, P[i]["correct"]))
        (cr if P[i]["correct"] else cw).append(conf)
    ece = sum(len(v) / len(ids) * abs(sum(c for c, _ in v) / len(v) - sum(k for _, k in v) / len(v)) for v in bins.values())
    return {"n": len(ids), "brier": round(br / len(ids), 4), "ece_10": round(ece, 4),
            "tu_tin_tb_khi_sai": round(sum(cw) / max(1, len(cw)), 4), "tu_tin_tb_khi_dung": round(sum(cr) / max(1, len(cr)), 4)}


hieu = {}
for code in MODELS:
    P = PRED[(code, "dev")]
    hieu[code] = {"tat_ca": calib(P, list(D))}
    for g in sorted({f"{r['lang']}|{r['variant']}" for r in dev}):
        hieu[code][g] = calib(P, [r["id"] for r in dev if f"{r['lang']}|{r['variant']}" == g])

# ---------------------------------------------------------------- phép thử (mục 3 + cập nhật 26/10)
PROW = {}
for ph, fn in PROBE_FILES.items():
    if (PR / fn).exists():
        PROW[ph] = {r["id"]: r for r in read_jsonl(PR / fn)}
        for i, r in PROW[ph].items():
            ORIG[i] = r["orig_id"]


def grp(r):
    return f"{r['source']}|{r['lang']}|{r['variant']}|{'noul' if r['kind'] == 'noul' else 'chon_mot'}"


def khung(r):
    """Nhóm gộp theo khung câu hỏi P5 / P1′ (cả hai nguồn kiểm chứng gộp lại) và biến thể."""
    k = r.get("p5_khung") or ("kiem_chung_choice" if r.get("probe") == "P1b" else None)
    return f"khung:{k}|{r['variant']}" if k else None


def nodiac_id(i):
    r = D[ORIG[i]]
    x = BY.get((r["pair_id"], "vi", "nodiac"))
    return x["id"] if x else None


def en_id(i):
    r = D[ORIG[i]]
    x = BY.get((r["pair_id"], "en", "orig"))
    return x["id"] if x else None


phep = {}
for code in MODELS:
    Pd = PRED[(code, "dev")]
    res = {}
    for ph in PAIRED:
        if (code, ph) not in PRED or ph not in PROW:
            res[ph] = "chua_co"; continue
        Pp, rows = PRED[(code, ph)], PROW[ph]
        G = defaultdict(list)
        for i, r in rows.items():
            G[grp(r)].append(i); G["tat_ca"].append(i)
            if khung(r):
                G[khung(r)].append(i)
        out = {}
        for g, ids in sorted(G.items()):
            e = paired(Pd, Pp, ids, ka=lambda i: ORIG[i])      # A = dev cùng câu, B = phép thử
            if e is None:
                continue
            e = {"n": e["n"], "acc_dev": e["acc_A"], "acc_phep_thu": e["acc_B"], "chenh": e["chenh_B_tru_A"], "ci95": e["ci95"], "mcnemar_p": e["mcnemar_p"]}
            if ph == "pa":    # mốc: dòng không dấu toàn phần của cùng cặp
                x = paired(Pd, Pp, ids, ka=nodiac_id)
                e["moc_dong_khong_dau_toan_phan"] = {"acc": x["acc_A"], "chenh_pa_tru_moc": x["chenh_B_tru_A"], "ci95": x["ci95"]} if x else None
            if ph == "pb":    # mốc: dòng EN cùng cặp
                x = paired(Pd, Pp, ids, ka=en_id)
                e["moc_dong_EN"] = {"acc": x["acc_A"], "chenh_pb_tru_moc": x["chenh_B_tru_A"], "ci95": x["ci95"]} if x else None
            out[g] = e
        res[ph] = out
    if (code, "p1b") in PRED and (code, "p5") in PRED and "p5" in PROW:     # P1′ − P5: cùng câu hỏi mới, chỉ khác chữ nhãn
        p5_of = {r["orig_id"]: i for i, r in PROW["p5"].items()}
        ids = [i for i in PROW["p1b"] if PROW["p1b"][i]["orig_id"] in p5_of]
        G = defaultdict(list)
        for i in ids:
            G[grp(PROW["p1b"][i])].append(i); G["tat_ca"].append(i); G[khung(PROW["p1b"][i])].append(i)
        res["p1b_tru_p5"] = {g: paired(PRED[(code, "p5")], PRED[(code, "p1b")], v, ka=lambda i: p5_of[PROW["p1b"][i]["orig_id"]]) for g, v in sorted(G.items())}
    else:
        res["p1b_tru_p5"] = "chua_co"
    if (code, "p2") in PRED:
        Pr = PRED[(code, "p2")]
        G = defaultdict(list)
        for i in Pr:
            if i in Pd:
                G[grp(D[i])].append(i); G["tat_ca"].append(i)
        res["p2"] = {g: {"n": len(ids), "cung_lua_chon_khi_dao": round(sum(Pd[i]["pred"] == Pr[i]["k"] - 1 - Pr[i]["pred"] for i in ids) / len(ids), 4),
                         "acc_dev": round(sum(Pd[i]["correct"] for i in ids) / len(ids), 4), "acc_dao": round(sum(Pr[i]["correct"] for i in ids) / len(ids), 4)}
                     for g, ids in sorted(G.items())}
    else:
        res["p2"] = "chua_co"
    for ph in ("p3", "p4"):
        if (code, ph) not in PRED or ph not in PROW:
            res[ph] = "chua_co"; continue
        Pp, rows = PRED[(code, ph)], PROW[ph]
        G = defaultdict(list)
        for i, r in rows.items():
            if i in Pp:
                G[grp(r)].append(i)
        out = {}
        for g, ids in sorted(G.items()):
            rs = [rows[i] for i in ids]
            e = {"n": len(ids), "acc_phep_thu": round(sum(Pp[i]["correct"] for i in ids) / len(ids), 4),
                 "doan_mo_1_tren_k": round(sum(1 / len(r["options"]) for r in rs) / len(rs), 4), "ci95": boot_ci(ids, lambda i: Pp[i]["correct"])}
            if ph == "p3":
                e["acc_dev_cung_cau"] = round(sum(Pd[ORIG[i]]["correct"] for i in ids if ORIG[i] in Pd) / len(ids), 4)
                top = Counter(r["options"][r["target"]] if r["source"] != "belebele" else r["target"] for r in rs).most_common(1)[0][1]
                e["luon_chon_dap_an_hay_gap"] = round(top / len(rs), 4)
            else:
                # mốc thiên vị: trên câu dev gốc của cùng nhận định, model chọn "không đủ thông tin"/"Không" khi đáp án gốc KHÔNG phải nhãn đó
                base_ids = [ORIG[i] for i in ids if ORIG[i] in Pd]
                wrong_nei = [j for j in base_ids if D[j]["options"][D[j]["target"]] not in NEI_LIKE]
                e["dev_chon_nhan_nay_khi_dap_an_khac"] = round(sum(D[j]["options"][Pd[j]["pred"]] in NEI_LIKE for j in wrong_nei) / max(1, len(wrong_nei)), 4)
                e["dev_so_cau_dap_an_khac"] = len(wrong_nei)
            out[g] = e
        res[ph] = out
    phep[code] = res

# ---------------------------------------------------------------- (c) nhầm nhãn SIB-200 / MASSIVE
_tr = json.loads((CONFIG / "translations.json").read_text(encoding="utf-8"))["massive_intents"]
EN2K = {v["en"]: k for k, v in _tr.items()}


def opt_labels(r):
    """Nhãn (khóa) của từng vị trí lựa chọn: lấy theo dòng EN cùng cặp (mọi biến thể và phép thử đều giữ vị trí)."""
    src = D[r["orig_id"]] if "orig_id" in r else r
    e = BY[(src["pair_id"], "en", "orig")]
    return [EN2K.get(o, o) if r["source"] == "massive" else o for o in e["options"]]


def label_dist(P, rows):
    gold, pred, conf = Counter(), Counter(), Counter()
    for r in rows:
        if r["id"] not in P:
            continue
        lab = opt_labels(r)
        g, p = lab[r["target"]], lab[P[r["id"]]["pred"]]
        gold[g] += 1; pred[p] += 1; conf[(g, p)] += 1
    n = sum(gold.values())
    if not n:
        return None
    labs = sorted(set(gold) | set(pred))
    tvd = 0.5 * sum(abs(pred[x] - gold[x]) for x in labs) / n
    return gold, pred, conf, n, tvd


nham = {}
for code in MODELS:
    nham[code] = {}
    sets = [(f"dev|{lv}", PRED[(code, "dev")], [r for r in dev if f"{r['lang']}|{r['variant']}" == lv]) for lv in ("vi|orig", "vi|nodiac", "en|orig")]
    for ph in ("pa", "pb"):
        if (code, ph) in PRED and ph in PROW:
            sets.append((ph, PRED[(code, ph)], list(PROW[ph].values())))
    for src in ("sib200", "massive"):
        for name, P, rows in sets:
            x = label_dist(P, [r for r in rows if r["source"] == src])
            if not x:
                continue
            gold, pred, conf, n, tvd = x
            e = {"n": n, "tvd_phan_bo_du_doan_vs_dap_an": round(tvd, 4),
                 "nhan_theo_doi": {lb: {"ti_le_dap_an": round(gold[lb] / n, 4), "ti_le_du_doan": round(pred[lb] / n, 4),
                                        "du_doan_tren_dap_an": round(pred[lb] / max(1, gold[lb]), 3)} for lb in THEO_DOI[src]},
                 "du_doan_nhieu_hon_dap_an_nhat": sorted(({"nhan": lb, "du_doan": pred[lb], "dap_an": gold[lb]} for lb in set(pred) | set(gold)),
                                                          key=lambda d: d["dap_an"] - d["du_doan"])[:5]}
            if src == "sib200":
                labs = sorted(set(gold) | set(pred))
                e["ma_tran (hàng = đáp án, cột = dự đoán)"] = {g: {p: conf[(g, p)] for p in labs} for g in labs}
            else:
                e["cap_nham_nhieu_nhat"] = [{"dap_an": g, "du_doan": p, "so_cau": c} for (g, p), c in conf.most_common() if g != p][:10]
            nham[code][f"{src}|{name}"] = e

# ---------------------------------------------------------------- (d) P3 kiểm chứng: phân bố nhãn dự đoán khi bỏ văn bản
p3pb = {}
if "p3" in PROW:
    for code in MODELS:
        if (code, "p3") not in PRED:
            p3pb[code] = "chua_co"; continue
        Pp, Pd = PRED[(code, "p3")], PRED[(code, "dev")]
        G = defaultdict(list)
        for i, r in PROW["p3"].items():
            if r["source"] in ("vifactcheck", "viwikifc") and i in Pp:
                G[grp(r)].append(r)
        out = {}
        for g, rs in sorted(G.items()):
            n = len(rs)
            dist = lambda f: {k: round(v / n, 4) for k, v in sorted(Counter(f(r) for r in rs).items())}
            out[g] = {"n": n, "dap_an": dist(lambda r: FC_LAB[r["options"][r["target"]]]),
                      "du_doan_P3": dist(lambda r: FC_LAB[r["options"][Pp[r["id"]]["pred"]]]),
                      "du_doan_dev_cung_cau": dist(lambda r: FC_LAB[r["options"][Pd[r["orig_id"]]["pred"]]] if r["orig_id"] in Pd else "thieu")}
        p3pb[code] = out

res = {"CANH_BAO": "Phân tích trên dev 1.3 và bộ phân tích probe_v1; không phải điểm benchmark; không dùng để chọn checkpoint.",
       "models": MODELS, "nguon_preds_dev": SRC, "phep_thu_co_preds": sorted(f"{c}:{p}" for c, p in PRED if p != "dev"),
       "so_cau_bi_loai": dict(EXCL),
       "cai_dat": {"seed": SEED, "n_boot": N_BOOT, "cum_bootstrap": "trường cluster của dev 1.3 (dòng phép thử theo câu dev gốc)",
                   "nguong_do_dai (token state VI có dấu của cặp, tam phân riêng từng nguồn)": CUT,
                   "da_sua_o_1_2": "cặp có dòng vi_provenance = revised hoặc vi_revised_fields khác rỗng; chỉ tính dòng VI",
                   "ece": "10 ngăn đều theo xác suất cao nhất", "nhan_theo_doi_nham_nhan": THEO_DOI},
       "lat_cat": lat, "chuyen_doi_M1_sang_M2": chuyen or "khong_co_M2", "trung_loi": trung, "hieu_chinh": hieu, "phep_thu": phep,
       "nham_nhan": nham, "p3_phan_bo_nhan": p3pb}
out = a.out or PR / "phan_tich" / f"{(a.m2 or a.m1).replace(' ', '_').replace('(', '').replace(')', '')}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- bản tóm tắt markdown
pct = lambda x: "—" if x is None else f"{100 * x:.1f}"
ci = lambda c: "" if not c or c[0] is None else f" [{100 * c[0]:.1f}; {100 * c[1]:.1f}]"
L = [f"# Phân tích xu hướng: {', '.join(f'{c} = {n}' for c, n in MODELS.items())}", "", res["CANH_BAO"], ""]
L += ["## Lát cắt (accuracy %, CI 95% bootstrap theo cụm" + ("; chênh M2 − M1 kèm McNemar)" if HAS_M2 else ")"), "",
      "| Lát | Nhóm | n | " + " | ".join(MODELS) + (" | M2 − M1 | p McNemar |" if HAS_M2 else " |"),
      "|---|---|---|" + "---|" * len(MODELS) + ("---|---|" if HAS_M2 else "")]
for sl, gs in lat.items():
    for g, e in gs.items():
        cells = [pct(e[c]["acc"]) + ci(e[c]["ci95"]) for c in MODELS]
        tail = ""
        if HAS_M2:
            d = e["M2_tru_M1"]
            tail = f" {100 * d['chenh']:+.1f}{ci(d['ci95'])} | {d['mcnemar_p']:.3g} |"
        L.append(f"| {sl} | {g} | {e['so_cau']} | " + " | ".join(cells) + " |" + tail)
if HAS_M2:
    L += ["", "## Chuyển đổi M1 → M2", "", "| Nguồn | đúng→đúng | đúng→sai | sai→đúng | sai→sai |", "|---|---|---|---|---|"]
    for s, m in chuyen.items():
        L.append(f"| {s} | {m.get('dung->dung', 0)} | {m.get('dung->sai', 0)} | {m.get('sai->dung', 0)} | {m.get('sai->sai', 0)} |")
L += ["", "## Trùng lỗi với model tham chiếu", "", "| Cặp | Jaccard tập sai | P(R sai \\| M sai) | tỉ lệ sai của R | tỉ lệ sai của M |", "|---|---|---|---|---|"]
for k, v in trung.items():
    L.append(f"| {k} | {v['jaccard_tap_sai']:.3f} | {v['P(R sai | M sai)']:.3f} | {v['ti_le_sai_cua_R']:.3f} | {v['ti_le_sai_cua_M']:.3f} |")
L += ["", "## Hiệu chỉnh (toàn dev)", "", "| Model | Brier | ECE | tự tin khi sai | tự tin khi đúng |", "|---|---|---|---|---|"]
for c, v in hieu.items():
    t = v["tat_ca"]
    L.append(f"| {c} | {t['brier']:.4f} | {t['ece_10']:.4f} | {t['tu_tin_tb_khi_sai']:.3f} | {t['tu_tin_tb_khi_dung']:.3f} |")
L += ["", "## Phép thử so theo cặp với dev (acc dev → acc phép thử, chênh, CI)", ""]
for ph in PAIRED + ("p2", "p3", "p4"):
    groups_ = sorted({g for c in MODELS if isinstance(phep[c].get(ph), dict) for g in phep[c][ph]})
    if not groups_:
        L.append(f"- {ph}: chưa có preds"); continue
    L += [f"### {ph}", "", "| Nhóm | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for g in groups_:
        cells = []
        for c in MODELS:
            e = phep[c].get(ph, {}).get(g) if isinstance(phep[c].get(ph), dict) else None
            if not e:
                cells.append("—")
            elif ph in PAIRED:
                cells.append(f"{pct(e['acc_dev'])} → {pct(e['acc_phep_thu'])} ({100 * e['chenh']:+.1f}{ci(e['ci95'])})")
            elif ph == "p2":
                cells.append(f"cùng lựa chọn {pct(e['cung_lua_chon_khi_dao'])}; acc {pct(e['acc_dev'])} → {pct(e['acc_dao'])}")
            elif ph == "p3":
                cells.append(f"{pct(e['acc_phep_thu'])}{ci(e['ci95'])} (dev {pct(e['acc_dev_cung_cau'])}; đoán mò {pct(e['doan_mo_1_tren_k'])}; hay gặp {pct(e['luon_chon_dap_an_hay_gap'])})")
            else:
                cells.append(f"{pct(e['acc_phep_thu'])}{ci(e['ci95'])} (mốc thiên vị {pct(e['dev_chon_nhan_nay_khi_dap_an_khac'])})")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("")
L += ["## dev → P5 → P1′ theo nguồn (acc %; chênh P5 − dev và P1′ − P5 theo cặp câu, CI 95% bootstrap theo cụm, p McNemar)", "",
      "P5 chỉ đổi câu hỏi; P1′ dùng chính câu hỏi P5 và đổi thêm chữ nhãn (chỉ kiểm chứng choice), nên P1′ − P5 là phần do đổi chữ nhãn. "
      "Dòng khung:* gộp hai nguồn kiểm chứng; dòng tat_ca của P5 gồm cả 4 khung, còn P1′ − P5 ở dòng đó chỉ tính trên câu kiểm chứng choice.", "",
      "| Model | Nhóm | n | dev | P5 | P5 − dev | P1′ | P1′ − P5 |", "|---|---|---|---|---|---|---|---|"]
for c in MODELS:
    p5_, d15 = phep[c].get("p5"), phep[c].get("p1b_tru_p5")
    if not isinstance(p5_, dict):
        L.append(f"| {c} | chưa có preds P5 | | | | | | |"); continue
    for g, e in p5_.items():
        x = d15.get(g) if isinstance(d15, dict) else None
        tail = f"{pct(x['acc_B'])} | {100 * x['chenh_B_tru_A']:+.1f}{ci(x['ci95'])}, p {x['mcnemar_p']:.3g} |" if x else "— | — |"
        L.append(f"| {c} | {g} | {e['n']} | {pct(e['acc_dev'])} | {pct(e['acc_phep_thu'])} | {100 * e['chenh']:+.1f}{ci(e['ci95'])}, p {e['mcnemar_p']:.3g} | " + tail)
L.append("")
L += ["## Nhầm nhãn (c): tỉ lệ dự đoán / tỉ lệ đáp án của nhãn theo dõi; TVD", "", "| Model | Nguồn \\| tập | TVD | nhãn theo dõi |", "|---|---|---|---|"]
for c, v in nham.items():
    for k, e in v.items():
        L.append(f"| {c} | {k} | {e['tvd_phan_bo_du_doan_vs_dap_an']:.3f} | " + "; ".join(
            f"{lb}: {pct(x['ti_le_du_doan'])}/{pct(x['ti_le_dap_an'])}" for lb, x in e["nhan_theo_doi"].items()) + " |")
L += ["", "## P3 kiểm chứng (d): phân bố nhãn dự đoán khi bỏ văn bản", "", "| Model | Nhóm | đáp án | dự đoán P3 | dự đoán dev cùng câu |", "|---|---|---|---|---|"]
for c, v in p3pb.items():
    if not isinstance(v, dict):
        continue
    for g, e in v.items():
        f = lambda d: ", ".join(f"{k} {pct(x)}" for k, x in d.items())
        L.append(f"| {c} | {g} | {f(e['dap_an'])} | {f(e['du_doan_P3'])} | {f(e['du_doan_dev_cung_cau'])} |")
L += ["", "## Số câu bị loại (model ngoài)", "", "```", json.dumps(dict(EXCL), ensure_ascii=False), "```"]
out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(out, out.with_suffix(".md"))
