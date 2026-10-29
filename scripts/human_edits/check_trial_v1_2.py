"""Kiểm bản dựng thử 1.2 (mục A, C của brief): (1) bỏ trường provenance thì trùng hash bản thử trước; (2) mọi dòng không dấu sinh từ bản VI
đã sửa (common.strip_diacritics); (3) rò rỉ train_v0 + val_v0 với dev/test 1.1 và 1.2 bằng đúng 3 phép của scripts/decontam.py (exact / near /
contain), theo nguồn × tập; (4) thống kê provenance. Ra results/bench_v1_2_trial/check_v1_2_trial.json. Chạy bằng venv WSL."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DATA, read_jsonl, strip_diacritics  # noqa: E402
from decontam import Index  # noqa: E402

T = ROOT / "results/bench_v1_2_trial"
PROV = ("vi_provenance", "vi_revised_fields", "gold_provenance")
EXPECT = {"dev": sys.argv[1] if len(sys.argv) > 1 else None, "test": sys.argv[2] if len(sys.argv) > 2 else None}
out = {}
rows12 = {"dev": read_jsonl(T / "dev_v1_2_trial.jsonl"), "test": read_jsonl(T / "test_v1_2_trial.jsonl")}
rows11 = {"dev": read_jsonl(DATA / "bench_v1_1/dev_v1_1.jsonl"), "test": read_jsonl(DATA / "bench_v1_1/test_v1.1.jsonl")}
# (1) trùng bản thử trước khi bỏ provenance
for s, rs in rows12.items():
    b = "".join(json.dumps({k: v for k, v in r.items() if k not in PROV}, ensure_ascii=False) + "\n" for r in rs).encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    out[f"{s}_sha_bo_provenance"] = h
    out[f"{s}_trung_ban_thu_truoc"] = (EXPECT[s] is None) or h.startswith(EXPECT[s])
    out[f"{s}_moi_dong_co_gold_provenance"] = all("gold_provenance" in r for r in rs)
    out[f"{s}_moi_dong_vi_co_vi_provenance"] = all("vi_provenance" in r for r in rs if r["lang"] == "vi")
# (2) bản không dấu sinh từ bản VI hiện hành
bad = []
for s, rs in rows12.items():
    by = {r["id"]: r for r in rs}
    for r in rs:
        if r["lang"] == "vi" and r["variant"] == "orig":
            nd = by[r["id"] + ":nodiac"]
            ok = nd["state"] == strip_diacritics(r["state"]) and nd["question"] == strip_diacritics(r["question"]) and \
                 nd["options"] == [strip_diacritics(o) for o in r["options"]] and nd["target"] == r["target"]
            if not ok:
                bad.append(r["id"])
out["nodiac_khong_khop"] = bad
out["nodiac_kiem"] = {s: sum(1 for r in rs if r["lang"] == "vi" and r["variant"] == "orig") for s, rs in rows12.items()}
out["nodiac_kiem_rieng_revised"] = {s: sum(1 for r in rs if r["lang"] == "vi" and r["variant"] == "orig" and r["vi_provenance"] == "revised")
                                    for s, rs in rows12.items()}
# (3) rò rỉ train_v0 + val_v0 (chỉ mục = state của bench, quét state của train như build_bench_v1)
train = read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl")
leak = {}
for ver, R in (("1.1", rows11), ("1.2", rows12)):
    for s, rs in R.items():
        ix = Index()
        for r in rs:
            if r["variant"] == "orig":
                ix.add(r["state"], f"{r['source']}|{r['lang']}")
        hits = Counter()
        for t in train:
            h = ix.check(t["state"])
            if h:
                hits[f"{h[1]}|{h[0]}"] += 1
        leak[f"{ver}|{s}"] = dict(sorted(hits.items()))
out["ro_ri_train_v0_val_v0"] = leak
# (4) provenance
prov = defaultdict(Counter)
gold_over = []
for s, rs in rows12.items():
    for r in rs:
        if r["lang"] == "vi" and r["variant"] == "orig":
            prov[f"{s}|{r['source']}"][r["vi_provenance"]] += 1
            for f in r["vi_revised_fields"]:
                prov[f"{s}|{r['source']}"]["sua:" + f.split(":")[0]] += 1
        if r["variant"] == "orig" and r["gold_provenance"] == "overridden" and (r["lang"] == "vi"):
            old = next(x for x in rows11[s] if x["id"] == r["id"])
            gold_over.append({"split": s, "pair_id": r["pair_id"], "nguon": r["source"], "dap_an_nguon": old["target"], "dap_an_1_2": r["target"]})
out["provenance_dong_vi_co_dau"] = {k: dict(v) for k, v in sorted(prov.items())}
out["dap_an_nguoc_nguon"] = gold_over
(T / "check_v1_2_trial.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("ro_ri_train_v0_val_v0", "provenance_dong_vi_co_dau", "dap_an_nguoc_nguon")}, ensure_ascii=False))
print("leak:", json.dumps({k: sum(v.values()) for k, v in leak.items()}), "| gold overridden:", Counter(g["split"] for g in gold_over))
