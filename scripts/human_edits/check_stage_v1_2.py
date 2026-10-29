"""Kiểm bản dựng 1.2 ở khu tạm trước khi đóng băng (bước D). Dùng: check_stage_v1_2.py <thư mục dựng> (có dev_v1_2_trial.jsonl, test_v1_2_trial.jsonl).
(1) mọi dòng có gold_provenance, dòng VI có vi_provenance; hash khi bỏ provenance; (2) mọi dòng không dấu = strip_diacritics(bản VI có dấu hiện hành);
(3) rò rỉ theo đúng cách build_bench_v1_1: dev↔test (phải 0) và train_v0 + val_v0 với từng tập (định dạng leakage_v1_1.json);
(4) provenance theo tập × nguồn, đáp án ngược nguồn. Ra <thư mục>/check_v1_2.json và <thư mục>/leakage_v1_2.json. Chạy bằng venv WSL."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DATA, read_jsonl, strip_diacritics  # noqa: E402
from decontam import Index  # noqa: E402

S = Path(sys.argv[1]).resolve()
PROV = ("vi_provenance", "vi_revised_fields", "gold_provenance")
out = {}
rows12 = {"dev": read_jsonl(S / "dev_v1_2_trial.jsonl"), "test": read_jsonl(S / "test_v1_2_trial.jsonl")}
rows11 = {"dev": read_jsonl(DATA / "bench_v1_1/dev_v1_1.jsonl"), "test": read_jsonl(DATA / "bench_v1_1/test_v1.1.jsonl")}
for s, rs in rows12.items():
    b = "".join(json.dumps({k: v for k, v in r.items() if k not in PROV}, ensure_ascii=False) + "\n" for r in rs).encode("utf-8")
    out[f"{s}_sha_bo_provenance"] = hashlib.sha256(b).hexdigest()
    out[f"{s}_moi_dong_co_gold_provenance"] = all("gold_provenance" in r for r in rs)
    out[f"{s}_moi_dong_vi_co_vi_provenance"] = all("vi_provenance" in r for r in rs if r["lang"] == "vi")
    out[f"{s}_id_duy_nhat"] = len({r["id"] for r in rs}) == len(rs)
bad = []
for s, rs in rows12.items():
    by = {r["id"]: r for r in rs}
    for r in rs:
        if r["lang"] == "vi" and r["variant"] == "orig":
            nd = by[r["id"] + ":nodiac"]
            if not (nd["state"] == strip_diacritics(r["state"]) and nd["question"] == strip_diacritics(r["question"]) and
                    nd["options"] == [strip_diacritics(o) for o in r["options"]] and nd["target"] == r["target"]):
                bad.append(r["id"])
out["nodiac_khong_khop"] = bad
out["nodiac_kiem"] = {s: sum(1 for r in rs if r["lang"] == "vi" and r["variant"] == "orig") for s, rs in rows12.items()}
out["nodiac_kiem_rieng_revised"] = {s: sum(1 for r in rs if r["lang"] == "vi" and r["variant"] == "orig" and r["vi_provenance"] == "revised")
                                    for s, rs in rows12.items()}
# rò rỉ: đúng cách build_bench_v1_1 (chỉ mục state "orig" của từng tập; train_v0 + val_v0)
dev, test = rows12["dev"], rows12["test"]
ti, di = Index(), Index()
for r in test:
    if r["variant"] == "orig":
        ti.add(r["state"], f"test:{r['source']}")
for r in dev:
    if r["variant"] == "orig":
        di.add(r["state"], f"dev:{r['source']}")
leak = {"dev_states_hitting_test": sum(bool(ti.check(r["state"])) for r in dev if r["variant"] == "orig"),
        "test_states_hitting_dev": sum(bool(di.check(r["state"])) for r in test if r["variant"] == "orig")}
train = read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl")
for name, idx in (("test", ti), ("dev_v1_2", di)):
    hits = Counter()
    for t in train:
        h = idx.check(t["state"])
        if h:
            hits[f"{h[0]}|{h[1]}|train_source={t['source']}"] += 1
    leak[f"train_v0_vs_{name}"] = {"total": sum(hits.values()), "by_kind_target_trainsource": dict(hits.most_common())}
(S / "leakage_v1_2.json").write_text(json.dumps(leak, ensure_ascii=False, indent=1), encoding="utf-8")
out["leakage"] = leak
prov = defaultdict(Counter)
gold_over = []
r11 = {s: {r["id"]: r for r in rs} for s, rs in rows11.items()}
for s, rs in rows12.items():
    for r in rs:
        if r["lang"] == "vi" and r["variant"] == "orig":
            prov[f"{s}|{r['source']}"][r["vi_provenance"]] += 1
            for f in r["vi_revised_fields"]:
                prov[f"{s}|{r['source']}"]["sua:" + f.split(":")[0]] += 1
            if r["gold_provenance"] == "overridden":
                gold_over.append({"split": s, "pair_id": r["pair_id"], "nguon": r["source"], "dap_an_nguon": r11[s][r["id"]]["target"], "dap_an_1_2": r["target"]})
out["provenance_dong_vi_co_dau"] = {k: dict(v) for k, v in sorted(prov.items())}
out["dap_an_nguoc_nguon"] = gold_over
(S / "check_v1_2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("provenance_dong_vi_co_dau", "dap_an_nguoc_nguon", "leakage")}, ensure_ascii=False))
print("leak:", json.dumps({k: (v if isinstance(v, int) else v["total"]) for k, v in leak.items()}), "| gold overridden:", dict(Counter(g["split"] for g in gold_over)))
