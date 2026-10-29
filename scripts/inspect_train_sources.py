"""In revision, config, split, schema và tối đa 5 dòng mẫu (mỗi trường cắt 200 ký tự) của hai nguồn train."""
from collections import Counter

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

api = HfApi()
cut = lambda v: str(v)[:200]

for repo in ["tasksource/tasksource-jev-typed-decisions", "LakoreAI/PhoLaya-base"]:
    print("=" * 100)
    info = api.dataset_info(repo)
    lic = info.card_data.get("license") if info.card_data else None
    print(repo, "sha", info.sha, "| license", lic, "| gated", getattr(info, "gated", None))
    cfgs = get_dataset_config_names(repo, revision=info.sha)
    print("configs:", cfgs[:30], "..." if len(cfgs) > 30 else "")
    files = [s.rfilename for s in info.siblings]
    print("files:", len(files), files[:20])
    cfg = "default" if "default" in cfgs else cfgs[0]
    ds = load_dataset(repo, cfg, revision=info.sha)
    print("config", cfg, {k: len(v) for k, v in ds.items()})
    tr = ds["train"]
    print("features:", tr.features)
    for c in tr.column_names:
        if c in ("source", "subset", "task", "dataset", "kind", "lang", "language", "config", "origin", "type"):
            cnt = Counter(tr[c])
            print(f"  {c}: {len(cnt)} distinct; top:", cnt.most_common(12))
    for i in range(min(5, len(tr))):
        print("  ex:", {k: cut(v) for k, v in tr[i].items()})
