"""In revision (commit sha), splits, schema và vài dòng mẫu của từng dataset nguồn."""
from huggingface_hub import HfApi
from datasets import load_dataset, get_dataset_config_names

api = HfApi()
REPOS = {
    "facebook/belebele": ["vie_Latn", "eng_Latn"],
    "Davlan/sib200": ["vie_Latn", "eng_Latn"],
    "AmazonScience/massive": ["vi-VN", "en-US"],
    "mteb/amazon_massive_intent": ["vi", "en"],
    "cambridgeltl/xcopa": ["vi"],
}

for repo, cfgs in REPOS.items():
    print("=" * 100)
    try:
        info = api.dataset_info(repo)
        print(repo, "sha", info.sha, "| license", (info.card_data or {}).get("license") if info.card_data else None)
    except Exception as e:
        print(repo, "INFO FAIL", type(e).__name__, e)
        continue
    for cfg in cfgs:
        try:
            ds = load_dataset(repo, cfg, revision=info.sha)
        except Exception as e:
            print(f"  [{cfg}] LOAD FAIL {type(e).__name__}: {str(e)[:300]}")
            continue
        print(f"  [{cfg}]", {k: len(v) for k, v in ds.items()})
        first = next(iter(ds.values()))
        print("   features:", first.features)
        for i in range(2):
            print("   ex:", {k: (str(v)[:120]) for k, v in first[i].items()})
