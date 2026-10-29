#!/usr/bin/env bash
# Thu metadata model baseline và thông tin gói PyPI (không cài, không chạy code bên thứ ba).
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 900 bash setup/run.sh scripts/baselines/fetch_meta.py 2>&1 | grep -v unauthenticated | cut -c1-300
mkdir -p results/baselines/pypi
for p in quyet laya; do
  curl -sf "https://pypi.org/pypi/$p/json" -o "results/baselines/pypi/$p.json" || echo "pypi $p: FAIL"
done
.venv/bin/python scripts/baselines/pypi_info.py results/baselines/pypi/*.json
