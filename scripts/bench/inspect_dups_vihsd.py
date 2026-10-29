"""Tìm khóa trùng trong ViFactCheck/ViWikiFC (dev/test) và in schema ViHSD (KHÔNG in văn bản)."""
from collections import Counter

from datasets import load_dataset
from huggingface_hub import HfApi

R = {"tranthaihoa/vifactcheck": "2df623632299a3459ad95feeff9ac67d7205a42a", "High-Will/ViWikiFC": "0089cbaac1a58458893b11c5cc067c4a37843f8c"}
for split in ("dev", "test"):
    d = load_dataset("tranthaihoa/vifactcheck", revision=R["tranthaihoa/vifactcheck"], split=split).to_pandas()
    k = d["index"].astype(str) + "-" + d["annotation_id"].astype(str)
    dup = k[k.duplicated(keep=False)]
    print(f"vifactcheck {split}: rows {len(d)} dup keys {dup.nunique()} dup rows {len(dup)}",
          "| identical dup rows:", int(d[k.duplicated(keep=False)].duplicated(subset=["Statement", "Context", "labels"], keep=False).sum()))
    w = load_dataset("High-Will/ViWikiFC", revision=R["High-Will/ViWikiFC"], split=split).to_pandas()
    dupw = w[w.pairID.duplicated(keep=False)]
    print(f"viwikifc {split}: rows {len(w)} dup pairID {dupw.pairID.nunique()} dup rows {len(dupw)}",
          "| same claim+label among dups:", int(dupw.duplicated(subset=["pairID", "claim", "gold_label"], keep=False).sum()))

info = HfApi().dataset_info("uitnlp/vihsd")
print("vihsd sha", info.sha, "files", [s.rfilename for s in info.siblings])
for split in ("dev", "test"):
    v = load_dataset("uitnlp/vihsd", revision=info.sha, split=split if split != "dev" else "validation").to_pandas() \
        if False else load_dataset("uitnlp/vihsd", revision=info.sha, data_files={split: f"{split}.csv"}, split=split).to_pandas()
    print(f"vihsd {split}: rows {len(v)} columns {list(v.columns)} dtypes {dict(v.dtypes.astype(str))}")
    for c in v.columns:
        if v[c].dtype != object or v[c].nunique() < 10:
            print("   ", c, Counter(v[c]).most_common(5))
    txt = next(c for c in v.columns if v[c].dtype == object and v[c].nunique() > 100)
    print("    text col:", txt, "null:", int(v[txt].isna().sum()), "len chars median:", int(v[txt].fillna("").str.len().median()))
