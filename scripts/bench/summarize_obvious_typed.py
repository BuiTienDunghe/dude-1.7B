"""Tóm tắt kiểm câu hiển nhiên noul/score cho từng model: noul đúng/sai; score đúng ở thang xuôi, thang ngược, cả hai,
và dấu hiệu gán nhầm (đúng ở thang xuôi nhưng sai ở thang ngược một cách hệ thống)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, read_jsonl  # noqa: E402

R = RESULTS / "dev_v1"
paths = {"decider-2b": R / "obvious_typed_eval/decider-2b-text/preds.jsonl",
         "Qwen3-1.7B gốc": R / "obvious_typed_eval_project/Qwen3-1.7B/preds.jsonl",
         "dự án (full_vocab)": R / "obvious_typed_eval_project/Qwen3-1.7B-fullvocab-s1/preds.jsonl"}
for name, p in paths.items():
    P = {x["id"]: x for x in read_jsonl(p)}
    out = []
    for lang in ("vi", "en"):
        noul = [x for i, x in P.items() if i.startswith("obv_noul") and i.endswith(lang)]
        up = [x for i, x in P.items() if i.startswith("obv_score") and f":{lang}:up" in i]
        down = {i.replace(":down", ":up"): x for i, x in P.items() if i.startswith("obv_score") and f":{lang}:down" in i}
        both = sum(x["correct"] and down[x["id"]]["correct"] for x in up)
        up_only = sum(x["correct"] and not down[x["id"]]["correct"] for x in up)
        out.append(f"{lang}: noul {sum(x['correct'] for x in noul)}/{len(noul)}; score xuôi {sum(x['correct'] for x in up)}/{len(up)}, "
                   f"ngược {sum(x['correct'] for x in down.values())}/{len(down)}, cả hai {both}/{len(up)}, chỉ đúng thang xuôi {up_only}")
        sums = all(abs(sum(x["probs"]) - 1) < 1e-4 and len(x["probs"]) == len(x.get("probs")) for x in P.values())
    print(f"{name}: " + " | ".join(out) + f" | tổng xác suất = 1: {sums}")
