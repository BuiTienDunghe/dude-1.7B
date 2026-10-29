"""Bước C của 2b′ (results/BRIEF_2b_prime_du_lieu.md mục 3–6, chốt điều phối mục 2b). Chỉ CPU (tokenizer + numpy), không chạy model,
không đọc test ngoài dò khử trùng (không in). Dữ liệu UIT (ViANLI qua PhoLaya) chỉ dùng nội bộ. Không chuỗi tiếng Việt mới: câu hỏi / lựa
chọn chép từ dev (như KC), tiền đề / giả thuyết nguyên văn ViANLI; bỏ dấu dùng common.strip_diacritics (hàm của build_bench_v1.py).

C1. Khối ViANLI khung kiểm chứng, từ pool sạch của bước B (results/train_2b_prime/tam/pool_vianli.jsonl):
  - val trước: tiền đề không có trong C′, xếp theo hash; M_VAL tiền đề đầu dành riêng cho val (vai thật và vai ghép); train không dùng
    tiền đề nào trong đó ở bất kỳ vai nào. Val ~ 5% khối (≤ 80 dòng), cùng cơ cấu.
  - kind theo hash từng cặp (choice khi u < 0,55); choice: Bác bỏ / Xác nhận / NEI = 330 / 272 / 223; noul: N_NOUL dòng, số "Có" chọn để
    noul của khối kiểm chứng VI gộp (KC + 2d + 2e + ViANLI) đạt 38,5%, phần còn lại chia Bác bỏ : NEI = 270 : 177 (để toàn khối ~600/500/400).
    Trong mỗi ô (kind, nhãn) lấy theo hash.
  - state = tiền đề thật + 2 tiền đề của cặp khác (không có tên riêng / số chung với giả thuyết, hai chiều), vị trí thật ngẫu nhiên theo hash,
    nối bằng dấu cách. Lựa chọn choice hoán vị theo hash như KC.
C2. Bỏ dấu tại chỗ (không bản sao): (a) 20% dòng ViANLI theo hash id; (b) 300 dòng 2d có dấu theo hash trong nhóm Xác nhận / Bác bỏ có
    max_ov < 0,7 (thiếu thì bù theo hash từ phần 2d còn lại); (c) 10% dòng của 2f + ý định (vi-tourism-admin-intent, multidogo/insurance/
    intent) + chủ đề (pholaya/vsfc/topic) theo hash id; (d) giữ 587 dòng KC không dấu.
C3. train_2b_prime.jsonl = train_2b_v3.jsonl (sha ea604a01…) sau C2 + khối ViANLI, xáo theo seed.
Cổng: D2, D3, D5′, D6/D10, D7, D11, D12 ở đây; D1 (check_label_strings.py) và D4 (cần sklearn) chạy bằng d1_d4.py (Python Windows).
Ra results/train_2b_prime/{vianli_khung.jsonl, val_vianli.jsonl, train_2b_prime.jsonl, mau_doc_vianli.jsonl, manifest_2b_prime.json,
  tam/cong_2b_prime.json}. Không ghi đè file nào của 2b.
Dùng: bash setup/run.sh -I scripts/train_2b_prime/build_2b_prime.py [--lay_mau_lai | --ve]  (--ve: biến thể V-e, chỉ đổi C2(b) sang hash mù trên mọi dòng 2d;
  --lay_mau_lai: lần sửa duy nhất cho D4 / D5′ / D11(a),
  chỉ đổi thứ tự lấy trong ô choice Bác bỏ (max_ov tăng dần) và choice NEI (max_ov giảm dần); lần 1 giữ ở tam/lan1/)"""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train_2b"))
from kc_util import DEV, OUT, Refs, h64, name_spans, normalize, num_spans, overlap, read_jsonl, sentences, u01, wset, write_jsonl  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from common import strip_diacritics  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
SEED = 20261029
DST = ROOT / "results/train_2b_prime"
V3 = OUT / "train_2b_v3.jsonl"
V3_SHA = "ea604a0138c20f42b4beae67cdeb3fe02908c0fa96cd63b104c469cc54a11635"
POOL = DST / "tam/pool_vianli.jsonl"
PROXY = ROOT / "data/proxy_v1/proxy_v1.jsonl"
C_FILES = (ROOT / "results/train_v1/components/C.jsonl", OUT / "C_prime.jsonl")
LABS = ["supported", "refuted", "nei"]
NHAN = {"contradiction": "refuted", "entailment": "supported", "neutral": "nei"}
Q_CHOICE = {"refuted": 330, "supported": 272, "nei": 223}
N_NOUL, CO_DICH, R_N_NOUL = 675, 0.385, (270, 177)
VAL_ROWS, M_VAL = 75, 150
P_NODIAC_A, N_NODIAC_B, P_NODIAC_C = 0.20, 300, 0.10
NHOM_C = {"vi-tourism-admin-intent", "multidogo/insurance/intent", "pholaya/vsfc/topic"}
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
NEG_P = re.compile(r"\b(không|chưa|chẳng|chả|khong|chua|chang|cha)\b", re.I)
NEG_D5 = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b")
LUONG = re.compile(r"\b(chỉ|duy nhất|độc nhất|đầu tiên|cuối cùng|phần lớn|tất cả|mọi|một phần)\b")
YES = {"Có", "Co"}

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(V3) == V3_SHA
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"])


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def dia(r):
    return "nodiac" not in set(r.get("flags") or [])


def fl(r):
    return set(r.get("flags") or [])


# ------------------------------------------------------------------ chữ dev (như build_kc_v2.py)
dev = read_jsonl(DEV)
FC_OPT, PRE = {}, {}
for r in dev:
    if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] in ("vifactcheck", "viwikifc"):
        if r["kind"] == "choice" and r["gold_provenance"] == "source":
            FC_OPT[r["gold_label"]] = r["options"][r["target"]]
        mk = "Nhận định: " if r["kind"] == "choice" else "Phát biểu: "
        PRE.setdefault(r["kind"], set()).add(r["question"].split(mk, 1)[0] + mk)
assert all(len(v) == 1 for v in PRE.values())
PRE = {k: next(iter(v)) for k, v in PRE.items()}
NOUL = next(r["options"] for r in dev if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] == "vifactcheck" and r["kind"] == "noul")
DEV_STR = {s for r in dev if r["source"] in ("vifactcheck", "viwikifc") for s in r["options"]}
DEV_PRE = {r["question"].split(mk, 1)[0] + mk for r in dev if r["source"] in ("vifactcheck", "viwikifc") for mk in MK if mk in r["question"]}
DEV_ALL_STR = {s for r in dev if r["lang"] == "vi" for s in [r["question"]] + list(r["options"])}

# ------------------------------------------------------------------ pool, C′
pool = read_jsonl(POOL)
for p in pool:
    p["nhan"] = NHAN[p["nhan_goc"]]; p["kp"] = normalize(p["tien_de"])
c_prem = set()
for f in C_FILES:
    for r in read_jsonl(f):
        if str(r["id"]).startswith("pl:vianli:"):
            c_prem.add(normalize(r["state"].split("\nCâu nói: ")[0].removeprefix("Đoạn văn: ")))
prem_text = {}
for p in pool:
    prem_text.setdefault(p["kp"], p["tien_de"])
prem_ids = defaultdict(list)
for p in pool:
    prem_ids[p["kp"]].append(p["id"])


def ents(s):
    return {normalize(x) for _, _, x, _ in name_spans(s)} | {normalize(v) for _, _, v, _ in num_spans(s)}


ENT = {k: ents(t) for k, t in prem_text.items()}
NORM = {k: " " + k + " " for k in prem_text}

# ------------------------------------------------------------------ val trước: tiền đề không có trong C′, M_VAL tiền đề đầu theo hash
elig_val = sorted((k for k in prem_text if k not in c_prem), key=lambda k: h64(SEED, "val", k))
VALP = set(elig_val[:M_VAL])
kind_of = lambda pid: "choice" if u01(SEED, "kind", pid) < 0.55 else "noul"


def ghep(pid, hyp, true_k, allowed):
    """2 tiền đề ghép: khác tiền đề thật, thuộc allowed, không tên riêng / số chung với giả thuyết (hai chiều)."""
    rng = random.Random(h64(SEED, "ghep", pid))
    he, hn = ents(hyp), " " + normalize(hyp) + " "
    out = []
    for _ in range(5000):
        k = allowed[rng.randrange(len(allowed))]
        if k == true_k or k in out:
            continue
        if any(f" {e} " in NORM[k] for e in he) or any(f" {e} " in hn for e in ENT[k]):
            continue
        out.append(k)
        if len(out) == 2:
            return out
    raise RuntimeError(pid)


def make_row(p, kind, split, allowed):
    rid = f"vianli2b:{p['id'].split(':')[1]}"
    extra = ghep(p["id"], p["gia_thuyet"], p["kp"], allowed)
    pos = int(u01(SEED, "vitri", rid) * 3)
    parts = [prem_text[k] for k in extra]; parts.insert(pos, p["tien_de"])
    state = " ".join(parts)
    label = p["nhan"]
    if kind == "choice":
        perm = sorted(range(3), key=lambda j: h64(SEED, "perm", rid, j))
        options = [FC_OPT[LABS[j]] for j in perm]; t = perm.index(LABS.index(label)); q = PRE["choice"] + p["gia_thuyet"]
    else:
        options = list(NOUL); t = 0 if label == "supported" else 1; q = PRE["noul"] + p["gia_thuyet"]
    r = {"id": rid, "dataset": "LakoreAI/PhoLaya-base:vianli", "source": "vianli_khung", "kind": kind, "lang": "vi", "text_lang": "vi",
         "license_use": "research-only, nội bộ", "split": split, "state": state, "question": q, "options": options,
         "target": [1.0 if j == t else 0.0 for j in range(len(options))], "target_argmax": t, "n_state_tokens": ntok(state), "variant": "khung_dev",
         "flags": [], "gold_label": label, "label_source": "human (ViANLI)", "text_source": "ViANLI (PhoLaya)", "license": "research-only, nội bộ",
         "id_cap_goc": p["id"], "nhan_goc": p["nhan_goc"], "vi_tri_tien_de": pos,
         "id_tien_de_ghep": [prem_ids[k][0] for k in extra], "tien_de_ghep_chuan": extra, "seed": SEED, "claim_ntok": ntok(p["gia_thuyet"])}
    if u01(SEED, "nodiac", rid) < P_NODIAC_A:
        r["state"], r["question"] = strip_diacritics(r["state"]), strip_diacritics(r["question"])
        r["options"] = [strip_diacritics(o) for o in r["options"]]
        r["flags"] = ["nodiac"]; r["n_state_tokens"] = ntok(r["state"])
    assert all(o in DEV_STR for o in r["options"]) and any(r["question"].startswith(x) for x in DEV_PRE), r["id"]
    return r


# ------------------------------------------------------------------ noul "Có" đích: khối kiểm chứng VI gộp
v3 = read_jsonl(V3)
is_fc = lambda r: r["text_lang"] == "vi" and (r["source"].startswith("kc_") or r["source"] == "vianli_khung" or bool({"2d", "2e"} & fl(r)))
noul_old = [r for r in v3 if is_fc(r) and r["kind"] == "noul"]
co_old = sum(r["options"][r["target_argmax"]] in YES for r in noul_old)
n_co = int(round(CO_DICH * (len(noul_old) + N_NOUL) - co_old))
n_r = int(round((N_NOUL - n_co) * R_N_NOUL[0] / sum(R_N_NOUL))); n_n = N_NOUL - n_co - n_r
Q_NOUL = {"supported": n_co, "refuted": n_r, "nei": n_n}
QUOTA = {("choice", l): Q_CHOICE[l] for l in LABS} | {("noul", l): Q_NOUL[l] for l in LABS}
scale = VAL_ROWS / sum(QUOTA.values())
QUOTA_VAL = {k: max(1, int(round(v * scale))) for k, v in QUOTA.items()}

cells_val, cells_tr = defaultdict(list), defaultdict(list)
for p in sorted(pool, key=lambda p: h64(SEED, "chon", p["id"])):
    (cells_val if p["kp"] in VALP else cells_tr)[(kind_of(p["id"]), p["nhan"])].append(p)
LAY_MAU_LAI = "--lay_mau_lai" in sys.argv
if LAY_MAU_LAI:   # lần sửa duy nhất cho D11(a) (brief mục 5): dòng choice Bác bỏ lấy từ trùng từ thấp, NEI từ trùng từ cao (max_ov của bước B);
    # noul, val, cỡ và cơ cấu giữ nguyên
    cells_tr[("choice", "refuted")].sort(key=lambda p: (p["max_ov"], h64(SEED, "chon", p["id"])))
    cells_tr[("choice", "nei")].sort(key=lambda p: (-p["max_ov"], h64(SEED, "chon", p["id"])))
allowed_val = sorted(VALP)
allowed_tr = sorted(k for k in prem_text if k not in VALP)
val_rows = [make_row(p, k[0], "val", allowed_val) for k, n in QUOTA_VAL.items() for p in cells_val[k][:n]]
assert all(len(cells_val[k]) >= n for k, n in QUOTA_VAL.items()), {k: len(cells_val[k]) for k in QUOTA_VAL}
blk = [make_row(p, k[0], "train", allowed_tr) for k, n in QUOTA.items() for p in cells_tr[k][:n]]
assert all(len(cells_tr[k]) >= n for k, n in QUOTA.items())
assert len(blk) <= 1580 and len(val_rows) <= 80
tr_true = {p["kp"] for k, n in QUOTA.items() for p in cells_tr[k][:n]}
tr_all = tr_true | {k for r in blk for k in r["tien_de_ghep_chuan"]}
va_true = {p["kp"] for k, n in QUOTA_VAL.items() for p in cells_val[k][:n]}
va_all = va_true | {k for r in val_rows for k in r["tien_de_ghep_chuan"]}
assert not tr_all & va_all and not va_all & c_prem

# ------------------------------------------------------------------ C2: bỏ dấu tại chỗ trên v3
REFS = Refs()


def strip_row(r):
    n = {**r, "state": strip_diacritics(r["state"]), "question": strip_diacritics(r["question"]), "options": [strip_diacritics(o) for o in r["options"]],
         "flags": sorted(fl(r) | {"nodiac"})}
    n["n_state_tokens"] = ntok(n["state"])
    return n


def texts(r):
    q = r["question"]
    c = next((q.split(mk, 1)[1] for mk in MK if mk in q), None)
    return [r["state"], c if c else q] + sentences(r["state"]) + ([] if c else list(r["options"]))


def hit(r):
    for t in texts(r):
        h = REFS.hit(t)
        if h:
            return h
    return None


def ov(r):
    c = claim_of(r)
    return max((overlap(c, s) for s in sentences(r["state"]) or [r["state"]]), default=0.0)


d2 = [r for r in v3 if "2d" in fl(r) and dia(r)]
VE = "--ve" in sys.argv   # biến thể V-e (điều phối 29/10, theo FABLE_truoc_train_2b_prime.md mục 2): sai lệch có ngày với brief C2(b);
# 300 dòng 2d chọn theo hash mù trên mọi dòng 2d có dấu, bỏ ưu tiên Xác nhận / Bác bỏ max_ov < 0,7; chạm khử trùng thì bù theo hash như cũ
elig_b = sorted([r for r in d2 if VE or (r["gold_label"] in ("supported", "refuted") and ov(r) < 0.7)], key=lambda r: h64(SEED, "nodiac_b", r["id"]))
eb_ids = {r["id"] for r in elig_b}
rest_b = sorted([r for r in d2 if r["id"] not in eb_ids], key=lambda r: h64(SEED, "nodiac_b", r["id"]))
pick_b, bo_c2 = [], Counter()
for r in elig_b + rest_b:
    if len(pick_b) >= N_NODIAC_B:
        break
    n = strip_row(r)
    h = hit(n)
    if h:
        bo_c2[f"b|{h[0]}|{h[1]}"] += 1; continue
    pick_b.append(r["id"])
grp_c = [r for r in v3 if r["text_lang"] == "vi" and dia(r) and ("2f" in fl(r) or r["source"] in NHOM_C)]
pick_c = []
for r in grp_c:
    if u01(SEED, "nodiac_c", r["id"]) < P_NODIAC_C:
        h = hit(strip_row(r))
        if h:
            bo_c2[f"c|{h[0]}|{h[1]}"] += 1; continue
        pick_c.append(r["id"])
SB, SC = set(pick_b), set(pick_c)
v3_new = [strip_row(r) if r["id"] in SB | SC else r for r in v3]
for r in v3_new:
    if r["id"] in SB:
        assert all(o in DEV_STR for o in r["options"]) and any(r["question"].startswith(x) for x in DEV_PRE), r["id"]

# D3 trên khối ViANLI (train + val): state, giả thuyết, từng câu
d3 = Counter(); bo_d3 = set()
for r in blk + val_rows:
    h = hit(r)
    if h:
        d3[f"{h[0]}|{h[1]}"] += 1; bo_d3.add(r["id"])
assert not bo_d3, dict(d3)   # pool đã sạch ở bước B; chạm ở đây thì dừng và báo, không tự sửa

OUTROWS = sorted(v3_new + blk, key=lambda r: h64(SEED, "mix", r["id"]))
DST.mkdir(parents=True, exist_ok=True)
write_jsonl(DST / "vianli_khung.jsonl", blk)
write_jsonl(DST / "val_vianli.jsonl", val_rows)
write_jsonl(DST / "train_2b_prime.jsonl", OUTROWS)

# ------------------------------------------------------------------ cổng
res = {"lay_mau_lai_D11": LAY_MAU_LAI, "bien_the": "V-e" if VE else ("lan2" if LAY_MAU_LAI else "lan1")}
# D2: kho chuỗi
d2c = {"vianli_va_C2b": {"so_dong": len(blk) + len(val_rows) + len(SB), "chuoi_ngoai_dev": 0}}
new_c = Counter()
for r in v3_new:
    if r["id"] in SC:
        for s in [r["question"]] + list(r["options"]):
            if s not in DEV_ALL_STR:
                new_c[s] += 1
nhan_kc = {strip_diacritics(x) for x in FC_OPT.values()} | set(FC_OPT.values())
c_label_ok = all(o in DEV_STR for r in v3_new if r["id"] in SC for o in r["options"] if normalize(strip_diacritics(o)) in {normalize(x) for x in nhan_kc})
d2c["C2c"] = {"so_dong": len(SC), "chuoi_khong_dau_ngoai_dev_khac_nhau": len(new_c), "luot": sum(new_c.values()),
              "cau_hoi_khac_nhau_ngoai_dev": len({r["question"] for r in v3_new if r["id"] in SC and r["question"] not in DEV_ALL_STR}),
              "vi_du": [s for s, _ in new_c.most_common(5)],
              "nhan_kiem_chung_trung_dev": c_label_ok}
d2c["dat_nghiem_vianli_C2b"] = True
res["D2"] = d2c
res["D3"] = {"so_dong_vianli_kiem": len(blk) + len(val_rows), "cham": dict(d3), "C2_bo_vi_cham_sau_bo_dau": dict(bo_c2), "dat": not d3}


def feats(claim, st):   # = phan_tich_bacbo.feats
    ss = sentences(st) or [st]
    best = max(ss, key=lambda s: overlap(claim, s))
    nst = normalize(st)
    nums = [v for _, _, v, _ in num_spans(claim)]
    names = [x for _, _, x, _ in name_spans(claim)]
    return {"max_ov": overlap(claim, best), "cover": len(wset(claim) & wset(st)) / max(1, len(wset(claim))),
            "phu_dinh_nd": bool(NEG_P.search(claim)), "lech_phu_dinh": bool(NEG_P.search(claim)) != bool(NEG_P.search(best)),
            "so_la": any(normalize(v) not in nst for v in nums), "ten_la": any(normalize(x) not in nst for x in names), "do_dai": len(claim.split()),
            "neg": bool(NEG_D5.search(claim.lower())), "luong": bool(LUONG.search(claim.lower()))}


def F(rows):
    return [(feats(claim_of(r), r["state"] or ""), r["gold_label"]) for r in rows]


blk_new = [r for r in OUTROWS if is_fc(r) and r["kind"] == "choice" and dia(r)]
blk_v3 = [r for r in v3 if is_fc(r) and r["kind"] == "choice" and dia(r)]
px = [r for r in read_jsonl(PROXY) if r["source"] in ("vifactcheck", "viwikifc") and r["variant"] == "orig"]
FB, FV, FP = F(blk_new), F(blk_v3), F(px)
FPc = F([r for r in px if r["kind"] == "choice"])
FA = F([r for r in blk if r["kind"] == "choice" and dia(r)])


def rate_gap(fs, key):
    o = {l: round(100 * sum(f[key] for f, y in fs if y == l) / max(1, sum(y == l for _, y in fs)), 2) for l in LABS}
    o["chenh_diem"] = round(max(o[l] for l in LABS) - min(o[l] for l in LABS), 2)
    return o


d5 = {}
for key, ten in (("neg", "phu_dinh"), ("luong", "luong_tu")):
    d5[ten] = {"khoi_2b_prime": rate_gap(FB, key), "proxy_v1_kiem_chung": rate_gap(FP, key), "proxy_v1_choice": rate_gap(FPc, key),
               "khoi_v3": rate_gap(FV, key), "rieng_vianli": rate_gap(FA, key)}
    d5[ten]["dat"] = d5[ten]["khoi_2b_prime"]["chenh_diem"] <= d5[ten]["proxy_v1_kiem_chung"]["chenh_diem"] + 2
d5["dat"] = d5["phu_dinh"]["dat"] and d5["luong_tu"]["dat"]
res["D5_phay"] = d5


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


BANDS = ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


def d6(fs):
    o = {"n": dict(Counter(y for _, y in fs)), "max_ov_trung_vi": {l: q([f["max_ov"] for f, y in fs if y == l], 0.5) for l in LABS},
         "cover_trung_vi": {l: q([f["cover"] for f, y in fs if y == l], 0.5) for l in LABS}, "P_nhan_theo_dai_max_ov": {}}
    for a, b in BANDS:
        ys = [y for f, y in fs if a <= f["max_ov"] < b]
        o["P_nhan_theo_dai_max_ov"][f"[{a};{min(b, 1)})"] = {"n": len(ys), **{l: round(sum(y == l for y in ys) / max(1, len(ys)), 3) for l in LABS}}
    return o


res["D6_D10"] = {"khoi_2b_prime": d6(FB), "khoi_v3": d6(FV), "proxy_v1_kiem_chung": d6(FP), "rieng_vianli": d6(FA)}

fc_choice = [r for r in OUTROWS if is_fc(r) and r["kind"] == "choice"]
fc_noul = [r for r in OUTROWS if is_fc(r) and r["kind"] == "noul"]
c7 = Counter(r["gold_label"] for r in fc_choice)
res["D7"] = {"choice_n": len(fc_choice), "choice_nhan": dict(c7), "ti_le_nei": round(c7["nei"] / len(fc_choice), 4),
             "noul_n": len(fc_noul), "noul_co": round(sum(r["options"][r["target_argmax"]] in YES for r in fc_noul) / len(fc_noul), 4),
             "noul_co_rieng_vianli": round(sum(r["target_argmax"] == 0 for r in blk if r["kind"] == "noul") / max(1, sum(r["kind"] == "noul" for r in blk)), 4)}
res["D7"]["dat"] = res["D7"]["ti_le_nei"] >= 0.30 and 0.37 <= res["D7"]["noul_co"] <= 0.40

# D11: probe bề mặt Bác bỏ / NEI (đặc trưng phan_tich_bacbo, LR numpy như phan_tich_bacbo, 5-fold phân tầng theo seed)
FK = ("max_ov", "cover", "phu_dinh_nd", "lech_phu_dinh", "so_la", "ten_la", "do_dai")


def fit(fs):
    X = np.array([[float(f[k]) for k in FK] for f, _ in fs]); y = np.array([1 if l == "refuted" else 0 for _, l in fs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-Z @ w)); w -= 0.1 * (Z.T @ (p - y) / len(y) + 1e-3 * np.r_[0, w[1:]])
    return lambda f: float(1 / (1 + np.exp(-(np.r_[1, (np.array([float(f[k]) for k in FK]) - mu) / sd] @ w))))


def cv5(fs):
    fs = [x for x in fs if x[1] in ("refuted", "nei")]
    by = defaultdict(list)
    for i, (_, l) in enumerate(fs):
        by[l].append(i)
    fold = {}
    for l, idx in by.items():
        random.Random(SEED).shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % 5
    accs = []
    for k in range(5):
        tr = [fs[i] for i in range(len(fs)) if fold[i] != k]; te = [fs[i] for i in range(len(fs)) if fold[i] == k]
        s = fit(tr)
        accs.append(sum((s(f) >= 0.5) == (l == "refuted") for f, l in te) / len(te))
    c = Counter(l for _, l in fs)
    return {"n": dict(c), "da_so": round(max(c.values()) / len(fs), 4), "acc_5fold": round(float(np.mean(accs)), 4)}


px_ref = [x for x in FP if x[1] == "refuted"]
d11 = {"a": {"khoi_2b_prime": cv5(FB), "proxy_v1_kiem_chung": cv5(FP), "proxy_v1_choice": cv5(FPc), "khoi_v3": cv5(FV), "rieng_vianli": cv5(FA)}}
d11["a"]["dat"] = d11["a"]["khoi_2b_prime"]["acc_5fold"] <= d11["a"]["proxy_v1_kiem_chung"]["acc_5fold"] + 0.05
sb, sv = fit([x for x in FB if x[1] in ("refuted", "nei")]), fit([x for x in FV if x[1] in ("refuted", "nei")])
d11["b"] = {"BB_proxy_bi_xep_phia_NEI": {"probe_fit_khoi_2b_prime": round(sum(sb(f) < 0.5 for f, _ in px_ref) / len(px_ref), 3),
                                         "probe_fit_khoi_v3": round(sum(sv(f) < 0.5 for f, _ in px_ref) / len(px_ref), 3), "n": len(px_ref)}}
res["D11"] = d11

st_tok = [r["n_state_tokens"] for r in blk]
full = [r["n_state_tokens"] + ntok(r["question"]) + 60 for r in blk]
res["D12"] = {"state_tok_trung_vi": q(st_tok, 0.5), "state_tok_p90": q(st_tok, 0.9), "state_tok_max": max(st_tok),
              "uoc_tinh_prompt_max": max(full), "vuot_2048": sum(x > 2048 for x in full)}
res["D12"]["dat"] = res["D12"]["vuot_2048"] == 0

# ------------------------------------------------------------------ báo thêm
tp = Counter(p["kp"] for k, n in QUOTA.items() for p in cells_tr[k][:n])
multi = {k for k, c in tp.items() if c >= 2}
res["khoi_vianli"] = {
    "so_dong": len(blk), "kind_nhan": {f"{k}|{l}": n for (k, l), n in QUOTA.items()}, "nhan": dict(Counter(r["gold_label"] for r in blk)),
    "nodiac": sum(not dia(r) for r in blk), "nodiac_nhan": dict(Counter(r["gold_label"] for r in blk if not dia(r))),
    "vi_tri_tien_de_theo_nhan": {l: dict(Counter(r["vi_tri_tien_de"] for r in blk if r["gold_label"] == l)) for l in LABS},
    "tien_de_that_khac_nhau": len(tp), "tien_de_dung_ge_2_lan": len(multi),
    "dong_tien_de_co_trong_C_prime": sum(1 for k, n in QUOTA.items() for p in cells_tr[k][:n] if p["kp"] in c_prem),
    "dong_tien_de_ghep_co_trong_C_prime": sum(any(k in c_prem for k in r["tien_de_ghep_chuan"]) for r in blk),
    "val": {"so_dong": len(val_rows), "kind_nhan": dict(Counter(f"{r['kind']}|{r['gold_label']}" for r in val_rows)),
            "tien_de_danh_rieng": len(VALP), "nodiac": sum(not dia(r) for r in val_rows)}}
# nhãn của tiền đề dùng ≥ 2 lần (vai thật)
lab_by_k = defaultdict(list)
for k, n in QUOTA.items():
    for p in cells_tr[k][:n]:
        lab_by_k[p["kp"]].append(p["nhan"])
res["khoi_vianli"]["nhan_cua_tien_de_ge_2_lan"] = dict(Counter("+".join(sorted(v)) for v in lab_by_k.values() if len(v) >= 2))
ghep_use = Counter(k for r in blk for k in r["tien_de_ghep_chuan"])
res["khoi_vianli"]["tien_de_ghep_dung_toi_da"] = max(ghep_use.values())
res["C2"] = {"a_vianli": sum(not dia(r) for r in blk), "b_2d": len(SB), "b_2d_trong_nhom_uu_tien": len(SB & eb_ids),
             "b_nhom_uu_tien_co_san": len(elig_b), "c_ngoai_khung": len(SC), "c_nhom_co_san": len(grp_c), "d_kc_giu": sum(r["source"].startswith("kc_") and not dia(r) for r in v3),
             "nhan_b_bo_dau_vs_co_dau": {"bo_dau": dict(Counter(r["gold_label"] for r in v3 if r["id"] in SB)),
                                         "con_co_dau_2d": dict(Counter(r["gold_label"] for r in d2 if r["id"] not in SB))},
             "nguon_c": dict(Counter(("2f" if "2f" in fl(r) else r["source"]) for r in v3 if r["id"] in SC)),
             "nhan_a_bo_dau_vs_co_dau": {"bo_dau": dict(Counter(r["gold_label"] for r in blk if not dia(r))), "co_dau": dict(Counter(r["gold_label"] for r in blk if dia(r)))}}


def comp(rows):
    c = Counter()
    for r in rows:
        if r["source"] == "vianli_khung":
            c["ViANLI khung (mới)"] += 1
        elif r["source"].startswith("kc_"):
            c["KC"] += 1
        elif "2d" in fl(r):
            c["2d"] += 1
        elif "2e" in fl(r):
            c["2e"] += 1
        elif "2f" in fl(r):
            c["2f"] += 1
        else:
            c[f"C′ khác ({r['text_lang']})"] += 1
    nv = sum(r["text_lang"] == "vi" for r in rows)
    return {"so_dong": len(rows), "khoi": dict(c), "VI": nv, "ti_le_VI": round(nv / len(rows), 4),
            "VI_khong_dau": sum(r["text_lang"] == "vi" and not dia(r) for r in rows)}


res["thanh_phan"] = {"v3": comp(v3), "2b_prime": comp(OUTROWS)}

# mẫu đọc mù: 30 dòng mỗi nhãn của khối ViANLI + 20 dòng không dấu của C2(c); nhãn ở cột riêng
rng = random.Random(SEED)
mau = []
for l in LABS:
    for r in rng.sample(sorted([r for r in blk if r["gold_label"] == l], key=lambda r: r["id"]), 30):
        mau.append({"id": r["id"], "phan": "vianli", "state": r["state"], "question": r["question"], "options": r["options"],
                    "nhan": r["options"][r["target_argmax"]], "nhan_3_lop": l})
for r in rng.sample(sorted([r for r in v3_new if r["id"] in SC], key=lambda r: r["id"]), 20):
    mau.append({"id": r["id"], "phan": "C2c_khong_dau", "state": r["state"], "question": r["question"], "options": r["options"],
                "nhan": r["options"][r["target_argmax"]], "nhan_3_lop": r.get("gold_label")})
rng.shuffle(mau)
write_jsonl(DST / "mau_doc_vianli.jsonl", mau)

man = {"brief": "results/BRIEF_2b_prime_du_lieu.md mục 3–6, chốt mục 2b", "lay_mau_lai_D11": LAY_MAU_LAI, "bien_the": res["bien_the"], "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED,
       "dau_vao": {"train_2b_v3.jsonl": V3_SHA, "pool_vianli.jsonl": sha(POOL), "dev": sha(DEV), "proxy_v1": sha(PROXY),
                   "C.jsonl": sha(C_FILES[0]), "C_prime.jsonl": sha(C_FILES[1]),
                   "vianli_train_goc": json.load(open(DST / "kiem_ke_vianli.json", encoding="utf-8"))["sha"]["vianli_train"]},
       "file": {f: sha(DST / f) for f in ("vianli_khung.jsonl", "val_vianli.jsonl", "train_2b_prime.jsonl", "mau_doc_vianli.jsonl")},
       "tham_so": {"Q_CHOICE": Q_CHOICE, "Q_NOUL": Q_NOUL, "CO_DICH": CO_DICH, "VAL_ROWS": VAL_ROWS, "M_VAL": M_VAL, "QUOTA_VAL": {f"{k[0]}|{k[1]}": v for k, v in QUOTA_VAL.items()},
                   "P_NODIAC_A": P_NODIAC_A, "N_NODIAC_B": N_NODIAC_B, "P_NODIAC_C": P_NODIAC_C, "NHOM_C": sorted(NHOM_C)},
       "bo_o_buoc_B": json.load(open(DST / "kiem_ke_vianli.json", encoding="utf-8"))["bo"],
       "id_bo_dau": {"a_vianli": sorted(r["id"] for r in blk if not dia(r)), "a_val": sorted(r["id"] for r in val_rows if not dia(r)),
                     "b_2d": sorted(SB), "c_ngoai_khung": sorted(SC)},
       **res, "giay": round(time.time() - T0)}
(DST / "manifest_2b_prime.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
(DST / "tam").mkdir(exist_ok=True)
(DST / "tam/cong_2b_prime.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
print("GIAY", round(time.time() - T0))
