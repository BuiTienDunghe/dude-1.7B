#!/usr/bin/env bash
# Tạo bộ câu hiển nhiên và chạy kiểm tra adapter decider-2b (ví dụ README, tương đương keyed/system_one).
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
bash setup/run.sh scripts/baselines/make_obvious.py
.venv/bin/python - <<'EOF'
import importlib.util as u
print("fla installed:", u.find_spec("fla") is not None, "| causal_conv1d installed:", u.find_spec("causal_conv1d") is not None)
EOF
timeout 2400 bash setup/run.sh -X faulthandler scripts/baselines/eval_decider.py results/baselines/obvious_v0.jsonl results/baselines/eval --checks \
  > results/baselines/log_decider_checks.txt 2>&1
echo "checks exit=$?"
grep -v unauthenticated results/baselines/log_decider_checks.txt | tail -30
