"""Chấm một file dòng DEV bằng adapter model ngoài của bước E (thứ tự gốc, batch 1). Chạy bằng .venv-baselines, OFFLINE.
Dùng: score_rows_ext.py <khóa: quyet-medium> <file dòng> <thư mục ra>. Không đọc test."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/baselines_v1_2"))
import adapters as A  # noqa: E402
import harness as H  # noqa: E402

assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "phải chạy offline"
key, rows_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
MODELS = {"quyet-medium": ("chinhnc/Quyet-1.0-Medium", "Quyet", "text")}
repo, kind, mapping = MODELS[key]
snap = Path(json.load(open(ROOT / "results/baselines_v1_2/downloads.json", encoding="utf-8"))[repo]["snapshot"])
ad = getattr(A, kind)(snap, mapping)
H.run(H.read_rows(rows_path), ad.score, out, f"{key}-{mapping}", orders=("normal",), meta={"model": repo, "mapping": mapping, "rows_file": str(rows_path)})
