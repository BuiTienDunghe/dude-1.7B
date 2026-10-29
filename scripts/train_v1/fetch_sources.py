"""Giai đoạn 0 train_v1: tải (ghim revision) split train của các nguồn phần A vào results/train_v1/raw_src/; in schema + phân bố nhãn.
PhoLaya ViANLI: dùng đúng file sources/vianli/train.jsonl @ revision train_v0 (034d4273)."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/train_v1/raw_src"
SRC = {
    "VSMRC/mrc": ("13c4449cdee58b62dd9bbca555fcbc0cc2630b38", ["train.jsonl", "README.md"]),
    "mhardalov/exams": ("4ff10804abb3341f8815cacd778181177bba7edd", ["crosslingual_vi/train-00000-of-00001.parquet", "README.md"]),
    "phamluan/vi-tourism-admin-intent": ("a092e219b0bf9a08e736f4425b8ebe0717029f20", ["data/train.jsonl", "README.md"]),
    "kornwtp/VNTC-10Topics": ("30b4f927d8c62b130d49621f511d61a84d588d20", ["train.csv", "README.md"]),
    "LakoreAI/PhoLaya-base": ("034d4273fb2600582cb789c45defe86b1c07e12d", ["sources/vianli/train.jsonl", "README.md"]),
    "uitnlp/ViANLI": ("0fec8d6ecb043a61c609f9b51f80401fdf1e84d3", ["README.md"]),
}
out = {}
for repo, (rev, files) in SRC.items():
    out[repo] = {"revision": rev, "files": {}}
    for f in files:
        p = Path(hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=RAW / repo.replace("/", "__")))
        if f.endswith(".md"):
            continue
        df = pd.read_parquet(p) if f.endswith(".parquet") else pd.read_csv(p) if f.endswith(".csv") else pd.read_json(p, lines=True)
        info = {"rows": len(df), "cols": {c: (str(df[c].dtype), str(df[c].iloc[0])[:200]) for c in df.columns}}
        for c in df.columns:
            if df[c].dtype == object and df[c].map(lambda v: isinstance(v, str)).all() and df[c].nunique() <= 40:
                info.setdefault("dist", {})[c] = dict(Counter(df[c]).most_common(40))
        out[repo]["files"][f] = info
    print(repo, flush=True)
(ROOT / "results/train_v1/sources_schema.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
