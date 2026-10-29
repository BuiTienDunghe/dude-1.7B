"""Gộp preds chấm lại cho dev 1.2 khi cập nhật dev: dòng của tập chấm lại (rescore_dev.jsonl) lấy preds mới nếu dòng có trong lượt chấm cập nhật,
ngược lại lấy preds cũ (chỉ khi nội dung dòng không đổi so với tập chấm lại cũ). Kiểm thêm: dòng EN trong lượt cập nhật (không đổi) phải cho đúng
preds như bản thử (tái lập). Dùng: merge_rescore_v1_2.py <eval_rescore cũ> <rescore_dev.jsonl cũ> <lượt cập nhật> <eval_rescore ra>. Python Windows."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD, OLD_RS, UPD, OUT = (Path(a) for a in sys.argv[1:5])
NEW_RS = ROOT / "results/bench_v1_2_stage/rescore/rescore_dev.jsonl"
SUBS = {"decider-2b-text": "decider/decider-2b-text", "decider-2b-keyed": "decider/decider-2b-keyed", "decider-ft-ep2-text": "ftd/decider-2b-text",
        "decider-ft-ep2-keyed": "ftd/decider-2b-keyed", "Qwen3-1.7B-fullvocab-s1": "control", "Qwen3-1.7B": "base", "strip-s1": "strip"}
read = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}
new_rs, old_rs = read(NEW_RS), read(OLD_RS)
assert OUT.resolve() != OLD.resolve()
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(OLD, OUT)
rep = {}
for m, sub in SUBS.items():
    po, pu = read(OLD / sub / "preds.jsonl"), read(UPD / sub / "preds.jsonl")
    ptr = read(ROOT / f"results/bench_v1_2_trial/eval/{m}/preds.jsonl")
    out, n_upd = [], 0
    for i, r in new_rs.items():
        if i in pu:
            out.append(pu[i]); n_upd += 1
        else:
            assert old_rs.get(i) == r, f"{m}: dòng {i} đổi nhưng không có trong lượt cập nhật"
            out.append(po[i])
    en_ok = all(pu[i]["pred"] == ptr[i]["pred"] and pu[i]["probs"] == ptr[i]["probs"] for i in pu if i not in new_rs and i in ptr)
    with open(OUT / sub / "preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    rep[m] = {"dong": len(out), "lay_tu_luot_cap_nhat": n_upd, "dong_khong_doi_tai_lap_dung_ban_thu": en_ok}
print(json.dumps(rep, ensure_ascii=False))
assert all(v["dong_khong_doi_tai_lap_dung_ban_thu"] for v in rep.values())
