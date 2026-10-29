#!/usr/bin/env bash
# G0 (BRIEF_buoc0_G0.md, người dùng duyệt 28/10): (1) tạo .venv-g0 sao từ freeze của .venv + flash-linear-attention (giữ nguyên mọi phiên bản cũ
# bằng ràng buộc -c; không đụng .venv / .venv-baselines); (2) tải Qwen/Qwen3.5-2B-Base (ghim revision hiện tại) và Qwen/Qwen3.5-4B @851bf6e806ef,
# chỉ file cấu hình, tokenizer và trọng số; ghi revision, danh sách file, sha256 trọng số. Không đăng nhập HF (model công khai).
# Chạy (WSL): bash setup/g0_prepare.sh
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy HF_HUB_DISABLE_PROGRESS_BARS=1
G=results/G0; mkdir -p $G/env
uv pip freeze --python .venv/bin/python > $G/env/freeze_venv.txt
if [ ! -x .venv-g0/bin/python ]; then
  uv venv --python 3.12 .venv-g0 > $G/env/log_venv.txt 2>&1
  uv pip install --python .venv-g0/bin/python -r $G/env/freeze_venv.txt --index-url https://download.pytorch.org/whl/cu128 \
     --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match >> $G/env/log_venv.txt 2>&1
  echo "sao .venv exit=$?"
fi
uv pip install --python .venv-g0/bin/python flash-linear-attention -c $G/env/freeze_venv.txt \
   --index-url https://pypi.org/simple >> $G/env/log_venv.txt 2>&1
echo "fla exit=$?"
uv pip freeze --python .venv-g0/bin/python > $G/env/freeze_venv_g0.txt
diff $G/env/freeze_venv.txt $G/env/freeze_venv_g0.txt > $G/env/diff_freeze.txt; echo "khác biệt freeze: $(wc -l < $G/env/diff_freeze.txt) dòng"
.venv-g0/bin/python - <<'EOF' > $G/env/fla_import.txt 2>&1
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability(0))
try:
    import fla, triton
    print("fla", fla.__version__, "triton", triton.__version__)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, T, H, K = 1, 64, 2, 64
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16); k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q); g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32); beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
    o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    torch.cuda.synchronize(); print("fla chunk_gated_delta_rule chạy được trên sm_%d%d:" % torch.cuda.get_device_capability(0), tuple(o.shape), bool(torch.isfinite(o.float()).all()))
except Exception as e:
    import traceback; traceback.print_exc(); print("FLA_LOI", repr(e))
EOF
echo "fla_import: $(tail -1 $G/env/fla_import.txt)"
.venv/bin/python - <<'EOF'
import hashlib, json, os
from huggingface_hub import HfApi, snapshot_download
api = HfApi(); out = {}
for repo, rev in (("Qwen/Qwen3.5-2B-Base", None), ("Qwen/Qwen3.5-4B", "851bf6e806ef")):
    info = api.model_info(repo, revision=rev)
    sha = info.sha
    path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "*.model"])
    files = {}
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        files[f] = {"bytes": os.path.getsize(p)}
        if f.endswith(".safetensors"):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for ch in iter(lambda: fh.read(1 << 24), b""):
                    h.update(ch)
            files[f]["sha256"] = h.hexdigest()
    out[repo] = {"revision": sha, "revision_yeu_cau": rev, "snapshot": path, "files": files,
                 "license": (info.card_data or {}).get("license") if info.card_data else None}
    print(repo, sha, sum(v["bytes"] for v in files.values()) / 1e9, "GB")
json.dump(out, open("results/G0/env/models_g0.json", "w"), indent=1)
EOF
echo G0_PREP_DONE
