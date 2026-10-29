"""Thu metadata của các model baseline ở revision ghim: sha, danh sách file, README, các file .json/.py nhỏ.
Chỉ tải file văn bản (không tải trọng số, không chạy code). Ra: results/baselines/meta/<org>__<name>/ và pins_models.json
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPOS = ["Mapika/decider-2b", "mstrasser/Jeff-Qwen3.5-2B", "Qwen/Qwen3.5-2B", "chinhnc/Quyet-1.0-Tiny", "chinhnc/Quyet-1.0-Small",
         "chinhnc/Quyet-1.0-Medium", "convaiinnovations/laya-multilingual", "vanty120/laya-vi"]
OUT = Path(__file__).resolve().parents[2] / "results" / "baselines" / "meta"
api = HfApi()
pins = {}
for repo in REPOS:
    d = OUT / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:
        pins[repo] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(repo, "INFO FAIL", pins[repo]["error"]); continue
    files = [(s.rfilename, s.size) for s in info.siblings]
    card = info.card_data.to_dict() if info.card_data else {}
    pins[repo] = {"sha": info.sha, "license": card.get("license"), "base_model": card.get("base_model"),
                  "datasets": card.get("datasets"), "library_name": getattr(info, "library_name", None),
                  "pipeline_tag": info.pipeline_tag, "tags": info.tags, "gated": getattr(info, "gated", None),
                  "safetensors_params": (info.safetensors.total if getattr(info, "safetensors", None) else None),
                  "files": files}
    for name, size in files:
        if (name.endswith((".md", ".json", ".py", ".txt", ".yaml", ".yml", ".jinja")) and (size or 0) < 2_000_000
                and not name.startswith(("tokenizer", "vocab", "merges"))) or name == "README.md":
            try:
                p = hf_hub_download(repo, name, revision=info.sha)
                dst = d / name.replace("/", "__")
                dst.write_bytes(Path(p).read_bytes())
            except Exception as e:
                print(repo, name, "DL FAIL", e)
    print(repo, info.sha, "files:", len(files), "py:", [f for f, _ in files if f.endswith(".py")],
          "params:", pins[repo]["safetensors_params"])
(OUT.parent / "pins_models.json").write_text(json.dumps(pins, indent=1, default=str), encoding="utf-8")
