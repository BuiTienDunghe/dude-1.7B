"""Kiểm: trên các dòng dev_v0 nằm trong dev_v1, đường chấm của dự án cho đúng các trường probs/pred/correct/letter_mass/top_token
như kết quả cũ (base: results/base_rerun; full_vocab: results/ft_loss). preds dev_v1 có thêm trường logits nên so theo trường."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, read_jsonl  # noqa: E402

KEYS = ["probs", "pred", "correct", "letter_mass", "top_token", "target", "k"]
for name, old in (("Qwen3-1.7B", RESULTS / "base_rerun/Qwen3-1.7B/preds.jsonl"),
                  ("Qwen3-1.7B-fullvocab-s1", RESULTS / "ft_loss/eval/normal/Qwen3-1.7B-fullvocab-s1/preds.jsonl")):
    o = {p["id"]: p for p in read_jsonl(old)}
    n = {p["id"]: p for p in read_jsonl(RESULTS / "dev_v1/eval" / name / "preds.jsonl")}
    diff = [i for i in o if any(o[i][k] != n[i][k] for k in KEYS)]
    print(name, "dev_v0 rows:", len(o), "identical fields:", len(o) - len(diff), "different:", len(diff), diff[:3])
