"""Giai đoạn 0 train_v1: tải (ghim revision) các bộ tham chiếu khử rò rỉ vào results/train_v1/raw_refs/, in cột + kiểu + 1 giá trị cắt ngắn mỗi file.
Chỉ đọc dữ liệu; không chạy model. Bộ UIT (ViFactCheck, ViWikiFC) chỉ ở máy này."""
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/train_v1/raw_refs"
api = HfApi()
# repo -> (revision, [file patterns]); revision = commit sha (repo) hoặc sha của refs/convert/parquet
REFS = {
    "tranthaihoa/vifactcheck": ("2df623632299a3459ad95feeff9ac67d7205a42a", lambda f: f.startswith("data/")),
    "High-Will/ViWikiFC": ("0089cbaac1a58458893b11c5cc067c4a37843f8c", lambda f: f.startswith("data/")),
    "facebook/mlqa": ("0d43d0f635e3d5768fcc252c97bf59cff847859e", lambda f: f.endswith(".parquet") and (".vi" in f.split("/")[0] or f.startswith("mlqa.vi.") or f.startswith("mlqa.en.en/"))),
    "mrlbenchmarks/global-piqa-nonparallel": ("6777742fa3634c0583cda3b7f8a482ea7b1b0937", lambda f: f.endswith(("vie_latn.tsv", "eng_latn.tsv"))),
    "mrlbenchmarks/global-piqa-parallel": ("b0b18516a8bc2cb1106bce3dd4db32848ca715ea", lambda f: f.endswith(("vie_latn.tsv", "eng_latn.tsv"))),
    "gsarti/flores_101": ("1a45e707ea4b5ea3d6c71341f18bfca6a6e356a3", lambda f: f.split("/")[0] in ("vie", "eng") and f.endswith(".parquet")),
    "facebook/belebele": ("7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", lambda f: f in ("data/vie_Latn.jsonl", "data/eng_Latn.jsonl")),
    "Davlan/sib200": ("38977a667f6fc264d5c26ec57a01e16db040b358", lambda f: f.split("/")[1:2] in (["vie_Latn"], ["eng_Latn"]) and f.endswith(".tsv")),
    "mteb/amazon_massive_intent": ("940fd47a81eaa7f2cc7b129674d945d618ac38c2", lambda f: f.endswith(("/en.json.gz", "/vi.json.gz"))),
    "cambridgeltl/xcopa": ("c1471bb6b030caadddd02b7785224bfa285737b4", lambda f: f.split("/")[0] in ("vi", "translation-vi") and f.endswith(".parquet")),
    "aps/super_glue": ("3de24cf8022e94f4ee4b9d55a6f539891524d646", lambda f: f.startswith("copa/") and f.endswith(".parquet")),
    "pkavumba/balanced-copa": ("e800e8e6db40b8e6e5bd00a7a08343a2932f7516", lambda f: f.endswith(".parquet")),
}
PARQUET_REF = {"facebook/mlqa", "gsarti/flores_101", "cambridgeltl/xcopa", "pkavumba/balanced-copa"}


def read(p):
    s = str(p)
    if s.endswith(".parquet"):
        return pd.read_parquet(p)
    if s.endswith(".csv"):
        return pd.read_csv(p)
    if s.endswith(".tsv"):
        return pd.read_csv(p, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    if s.endswith((".jsonl", ".json.gz")):
        return pd.read_json(p, lines=True)
    raise ValueError(s)


out = {}
for repo, (rev, keep) in REFS.items():
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev) if keep(f)]
    out[repo] = {"revision": rev, "parquet_convert": repo in PARQUET_REF, "files": {}}
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=RAW / repo.replace("/", "__"))
        df = read(p)
        out[repo]["files"][f] = {"rows": len(df), "cols": {c: (str(df[c].dtype), str(df[c].iloc[0])[:80]) for c in df.columns}}
    print(repo, len(files), flush=True)
(ROOT / "results/train_v1/refs_schema.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
