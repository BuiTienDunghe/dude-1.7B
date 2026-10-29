"""Soát nhãn dev 1.2 (việc 2): gộp bước A (mù) + B, thống kê, độ nhạy, kiểm chất lượng; ghi đề xuất sửa chữ (nhóm 3) vào config/vi_proposals.jsonl ở
trạng thái CHỜ DUYỆT (chỉ khi gia_tri_cu xuất hiện đúng 1 lần ở cả bản dev 1.2 lẫn giá trị hiện hành của công cụ). Không sửa dev/test/bench/config khác.
Ra: results/label_review_v1_2/review.jsonl, summary.json. Python Windows.
  analyze.py            -> thống kê + review.jsonl (không ghi đề xuất)
  analyze.py --write_proposals -> thêm: ghi đề xuất nhóm 3 (idempotent theo id)"""
import datetime
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/review_app"))
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402

O = ROOT / "results/label_review_v1_2"
DEV = ROOT / "data/bench_v1_2/dev_v1_2.jsonl"
DEV_SHA = "ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b"
LET = "ABCDEFGHIJ"
assert hashlib.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = {r["id"]: r for r in read(DEV)}
pairs = json.load(open(O / "pairs.json", encoding="utf-8"))
A = {x["item_id"]: x for k in pairs["lo"] for x in read(O / f"A/batch_{k}.jsonl")}
Bm = {x["item_id"]: x for k in pairs["lo"] for x in read(O / f"B/batch_{k}.jsonl")}
G = {x["item_id"]: x for k in pairs["lo"] for x in read(O / f"gold/batch_{k}.jsonl")}
ALL = [p for k in pairs["lo"] for p in pairs["lo"][k]]
assert set(A) == set(Bm) == set(G) == set(ALL), (len(A), len(Bm), len(G), len(ALL))
src = {p: rows[p + ":vi"]["source"] for p in ALL}

# ---------------------------------------------------------------- review.jsonl
review = []
for p in ALL:
    a, b, g = A[p], Bm[p], G[p]
    review.append({"id": p, "nguon": src[p], "tra_loi_mu": {"vi": a["tra_loi_vi"], "en": a.get("tra_loi_en"), "ly_do": a["ly_do"],
                                                          "nhieu_dap_an_bao_ve_duoc": a.get("nhieu_dap_an_bao_ve_duoc"), "cac_dap_an_bao_ve_duoc": a.get("cac_dap_an_bao_ve_duoc"),
                                                          "do_tin_cay": a.get("do_tin_cay"), "ghi_chu_ban_dich": a.get("ghi_chu_ban_dich")},
                   "gold": {"vi": g["dap_an_chuan_vi"], "en": g["dap_an_chuan_en"]}, "nhom": int(b["nhom"]), "bang_chung": b.get("bang_chung", ""),
                   "de_xuat": b.get("de_xuat", {}), "ghi_chu": b.get("ghi_chu", "")})
(O / "review.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in review), encoding="utf-8")
R = {x["id"]: x for x in review}

# ---------------------------------------------------------------- thống kê
S = {"so_cap": len(ALL), "nhom_theo_nguon": {}, "nhom": dict(Counter(x["nhom"] for x in review))}
for s in sorted(set(src.values())):
    S["nhom_theo_nguon"][s] = dict(sorted(Counter(R[p]["nhom"] for p in ALL if src[p] == s).items()))
vi_ok = [R[p]["tra_loi_mu"]["vi"] == R[p]["gold"]["vi"] for p in ALL]
en_ps = [p for p in ALL if R[p]["gold"]["en"]]
en_ok = [R[p]["tra_loi_mu"]["en"] == R[p]["gold"]["en"] for p in en_ps]
S["mu_trung_gold"] = {"vi": {"trung": sum(vi_ok), "n": len(vi_ok)}, "en": {"trung": sum(en_ok), "n": len(en_ok)},
                      "vi_theo_nguon": {s: f"{sum(R[p]['tra_loi_mu']['vi'] == R[p]['gold']['vi'] for p in ALL if src[p] == s)}/{sum(1 for p in ALL if src[p] == s)}"
                                        for s in sorted(set(src.values()))}}
# Quyet (và model khác) ở nhóm 1 so với nhóm 2: chọn gold bao nhiêu lần (dòng VI có dấu; dòng EN)
PRED = {"Quyet-1.0-Medium": ROOT / "results/baselines_v1_2/eval/normal/quyet-medium-text/preds.jsonl",
        "decider-2b (text)": ROOT / "results/dev_v1_2/eval/decider-2b-text/preds.jsonl",
        "dự án (full_vocab)": ROOT / "results/dev_v1_2/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl",
        "Qwen3-1.7B gốc": ROOT / "results/dev_v1_2/eval/Qwen3-1.7B/preds.jsonl",
        "Qwen3.5-2B gốc": ROOT / "results/baselines_v1_2/eval/normal/Qwen3.5-2B/preds.jsonl",
        "decider-2b (keyed)": ROOT / "results/dev_v1_2/eval/decider-2b-keyed/preds.jsonl"}
PR = {m: {x["id"]: x for x in read(f)} for m, f in PRED.items()}
S["chon_gold_theo_nhom"] = {}
for m, pr in PR.items():
    d = {}
    for nh in (1, 2, 3, 4, 5):
        ps = [p for p in ALL if R[p]["nhom"] == nh]
        vi = [pr[p + ":vi"]["pred"] == rows[p + ":vi"]["target"] for p in ps]
        en = [pr[p + ":en"]["pred"] == rows[p + ":en"]["target"] for p in ps if p + ":en" in rows]
        d[nh] = {"vi": f"{sum(vi)}/{len(vi)}", "en": f"{sum(en)}/{len(en)}"}
    S["chon_gold_theo_nhom"][m] = d

# ---------------------------------------------------------------- độ nhạy: áp nhóm 2 (đổi nhãn) và bỏ nhóm 4 (loại câu); KHÔNG sửa bench
SRC_VI = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
SRCS = {("vi", "orig"): SRC_VI, ("vi", "nodiac"): SRC_VI, ("en", "orig"): SRC_VI[:4]}
new_t, drop = {}, set()
for p in ALL:
    x = R[p]
    if x["nhom"] == 2 and x["de_xuat"].get("dap_an_moi"):
        t = LET.index(x["de_xuat"]["dap_an_moi"])
        ap = x["de_xuat"].get("ap_dung", "vi+en")
        for suf in ([":vi", ":vi:nodiac"] + ([":en"] if "en" in ap else [])):
            if p + suf in rows:
                new_t[p + suf] = t
    if x["nhom"] == 4:
        drop.add(p)


def macro(pr, lang, var, alt):
    out = []
    for s in SRCS[(lang, var)]:
        ids = [i for i, r in rows.items() if r["source"] == s and r["lang"] == lang and r["variant"] == var and (not alt or r["pair_id"] not in drop)]
        out.append(sum(pr[i]["pred"] == (new_t.get(i, rows[i]["target"]) if alt else rows[i]["target"]) for i in ids) / len(ids))
    return 100 * sum(out) / len(out)


S["do_nhay"] = {"so_dong_doi_nhan": len(new_t), "so_cap_bo": len(drop), "diem": {}}
for m, pr in PR.items():
    S["do_nhay"]["diem"][m] = {f"{l}|{v}": [round(macro(pr, l, v, False), 1), round(macro(pr, l, v, True), 1)] for (l, v) in SRCS}

# ---------------------------------------------------------------- kiểm chất lượng (người soát thứ hai)
if (O / "qc/B.jsonl").exists():
    QA = {x["item_id"]: x for x in read(O / "qc/A.jsonl")}; QB = {x["item_id"]: x for x in read(O / "qc/B.jsonl")}
    qc = pairs["qc"]
    S["kiem_chat_luong"] = {"so_cap": len(qc),
                            "mu_vi_dong_y": sum(QA[p]["tra_loi_vi"] == R[p]["tra_loi_mu"]["vi"] for p in qc),
                            "mu_en_dong_y": f"{sum(QA[p].get('tra_loi_en') == R[p]['tra_loi_mu']['en'] for p in qc if R[p]['gold']['en'])}/{sum(1 for p in qc if R[p]['gold']['en'])}",
                            "nhom_dong_y": sum(int(QB[p]["nhom"]) == R[p]["nhom"] for p in qc),
                            "nhom_dong_y_gop_giu_vs_doi": sum((int(QB[p]["nhom"]) in (1, 5)) == (R[p]["nhom"] in (1, 5)) for p in qc),
                            "bat_dong": [{"id": p, "nhom_1": R[p]["nhom"], "nhom_2": int(QB[p]["nhom"]), "mu_vi_1": R[p]["tra_loi_mu"]["vi"], "mu_vi_2": QA[p]["tra_loi_vi"]}
                                         for p in qc if int(QB[p]["nhom"]) != R[p]["nhom"] or QA[p]["tra_loi_vi"] != R[p]["tra_loi_mu"]["vi"]]}

# ---------------------------------------------------------------- đề xuất nhóm 3 (sửa chữ, chờ duyệt)
log = E.read_log(ROOT / "data/human_edits/edits_log.jsonl")
st = E.derive_state(log, inventory(ROOT / "config"))
v11 = {r["id"]: r for r in read(ROOT / "data/bench_v1_1/dev_v1_1.jsonl")}
FIELD = {"state": ("state", "state"), "question": ("question", "cau_hoi")}


def cur_tool(pid, field):
    base = v11[pid + ":vi"]
    tf = FIELD[field][1] if field in FIELD else "lua_chon:" + field.split(":")[1]
    c = st["cau"].get(pid, {})
    if tf in c:
        return c[tf]["gia_tri_moi"]
    return base[FIELD[field][0]] if field in FIELD else base["options"][int(field.split(":")[1])]


def cur_dev(pid, field):
    r = rows[pid + ":vi"]
    return r[FIELD[field][0]] if field in FIELD else r["options"][int(field.split(":")[1])]


props, bad = [], []
existing = {p["id"] for p in read(ROOT / "config/vi_proposals.jsonl")}
for p in ALL:
    x = R[p]
    for d in ([x["de_xuat"]] if isinstance(x["de_xuat"], dict) else x["de_xuat"]):
        if not (x["nhom"] == 3 and d.get("loai") == "sua_chu"):
            continue
        f, old, new = d["truong"], d["gia_tri_cu"], d["gia_tri_de_xuat"]
        old_n, new_n = unicodedata.normalize("NFC", old), unicodedata.normalize("NFC", new)
        ok_dev = cur_dev(p, f).count(old_n) == 1
        ok_tool = cur_tool(p, f).count(old) == 1 or unicodedata.normalize("NFC", cur_tool(p, f)).count(old_n) == 1
        pid_ = "lr-" + hashlib.sha1(f"{p}|{f}|{old_n}|{new_n}".encode("utf-8")).hexdigest()[:12]
        rec = {"id": pid_, "loai": "cau", "split": "dev", "row_id": p + ":vi", "pair_id": p, "nguon": src[p], "truong": f, "gia_tri_cu": old_n,
               "gia_tri_de_xuat": new_n, "ly_do": "loi_dich", "bang_chung_en": d.get("bang_chung_en", "") or x["bang_chung"][:300],
               "giai_thich": "Soát nhãn mù dev 1.2 (nhóm 3: bản dịch làm đổi nghĩa so với EN). " + (d.get("giai_thich") or ""),
               "doi_dap_an": False, "nguoi_de_xuat": "agent_label_review_opus", "thoi_diem": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        (props if ok_dev and ok_tool and old_n != new_n else bad).append({**rec, "_kiem": {"dev_1_lan": ok_dev, "cong_cu_1_lan": ok_tool}})
S["de_xuat_nhom3"] = {"hop_le": len(props), "khong_hop_le": [{"id": b["pair_id"], "truong": b["truong"], **b["_kiem"]} for b in bad]}
if "--write_proposals" in sys.argv:
    new = [p for p in props if p["id"] not in existing]
    with open(ROOT / "config/vi_proposals.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        for p in new:
            fh.write(json.dumps({k: v for k, v in p.items() if k != "_kiem"}, ensure_ascii=False) + "\n")
    S["de_xuat_nhom3"]["da_ghi"] = [p["id"] for p in new]
(O / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in S.items() if k not in ("chon_gold_theo_nhom",)}, ensure_ascii=False)[:3000])
