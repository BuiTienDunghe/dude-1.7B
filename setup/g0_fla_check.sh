#!/usr/bin/env bash
# (A) bước 1 của BRIEF_chan_doan_teacher.md: kiểm FLA chạy được trong .venv-g0 sau khi người dùng cài build-essential (gcc). Không vá thư viện.
cd /mnt/c/Users/dungbt06/Fine-tune
which gcc && gcc --version | head -1
.venv-g0/bin/python - <<'EOF' 2>&1 | tee results/G0/env/fla_check_sau_gcc.txt
import torch, fla, triton
print("torch", torch.__version__, "fla", fla.__version__, "triton", triton.__version__, "sm", torch.cuda.get_device_capability(0))
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
B, T, H, K = 1, 64, 2, 64
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16); k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
v = torch.randn_like(q); g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32); beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
torch.cuda.synchronize()
print("FLA_OK", tuple(o.shape), bool(torch.isfinite(o.float()).all()))
EOF
