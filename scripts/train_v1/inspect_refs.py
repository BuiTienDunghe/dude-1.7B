"""Giai đoạn 0: soi các bộ tham chiếu khử rò rỉ (FLORES, MLQA, Global-PIQA, ViFactCheck, ViWikiFC): revision, gated, file. Không tải dữ liệu."""
import json
from huggingface_hub import HfApi
api = HfApi()
out = {}
for repo in ["facebook/flores", "openlanguagedata/flores_plus", "Muennighoff/flores200", "gsarti/flores_101", "facebook/mlqa", "mrlbenchmarks/global-piqa-nonparallel",
             "mrlbenchmarks/global-piqa-parallel", "tranthaihoa/vifactcheck", "High-Will/ViWikiFC", "facebook/belebele", "Davlan/sib200", "mteb/amazon_massive_intent",
             "AmazonScience/massive", "cambridgeltl/xcopa", "aps/super_glue", "pkavumba/balanced-copa"]:
    try:
        info = api.dataset_info(repo)
        files = [s.rfilename for s in info.siblings]
        sel = [f for f in files if any(k in f.lower() for k in ("vie", "vi.", "/vi", "eng", "en.", "readme", "devtest", "dev", "copa", ".py"))][:30]
        out[repo] = {"sha": info.sha, "gated": info.gated, "n_files": len(files), "files_sel": sel, "tags": [t for t in info.tags if t.startswith("license")]}
    except Exception as e:
        out[repo] = {"error": repr(e)[:200]}
    try:
        refs = api.list_repo_refs(repo, repo_type="dataset")
        out.setdefault(repo, {})["converts"] = [(c.name, c.target_commit) for c in refs.converts]
        if refs.converts:
            pf = api.list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet")
            out[repo]["parquet_sel"] = [f for f in pf if any(k in f.lower() for k in ("vie", "vi", "eng", "en.en", "en/"))][:40]
    except Exception as e:
        out.setdefault(repo, {})["refs_error"] = repr(e)[:200]
open("/mnt/c/Users/dungbt06/Fine-tune/results/train_v1/inspect_refs.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")
