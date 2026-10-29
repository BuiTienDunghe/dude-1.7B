"""Bước E: tải trọng số các model ngoài về HF cache (WSL), ghim đúng revision trong results/baselines/pins_models.json. Chỉ tải, không import
gói của bên thứ ba, không chạy mã nào của repo. Từ chối repo có file .py hoặc .bin/.pt/.pkl (chỉ nhận safetensors + cấu hình văn bản)."""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / "results/baselines/pins_models.json").read_text(encoding="utf-8"))
REPOS = ["mstrasser/Jeff-Qwen3.5-2B", "chinhnc/Quyet-1.0-Tiny", "chinhnc/Quyet-1.0-Small", "convaiinnovations/laya-multilingual", "vanty120/laya-vi",
         "chinhnc/Quyet-1.0-Medium"]
out = {}
for repo in REPOS:
    rev = PINS[repo]["sha"]
    files = [s.rfilename for s in HfApi().model_info(repo, revision=rev, files_metadata=False).siblings]
    bad = [f for f in files if f.endswith((".py", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"))]
    if bad:
        out[repo] = {"loi": f"repo có file không nhận: {bad}"}; print(repo, "BỎ QUA:", bad, flush=True); continue
    try:
        p = snapshot_download(repo, revision=rev)
        sizes = {f.name: f.stat().st_size for f in Path(p).rglob("*") if f.is_file()}
        out[repo] = {"revision": rev, "snapshot": p, "files": len(sizes), "bytes": sum(sizes.values())}
        print(repo, "OK", rev[:12], f"{sum(sizes.values()) / 1e9:.2f} GB", flush=True)
    except Exception as ex:
        out[repo] = {"loi": f"{type(ex).__name__}: {str(ex)[:300]}"}; print(repo, "LỖI", type(ex).__name__, str(ex)[:200], flush=True)
(ROOT / "results/baselines_v1_2/downloads.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
sys.exit(0 if all("loi" not in v for v in out.values()) else 1)
