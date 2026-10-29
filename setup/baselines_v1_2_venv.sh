#!/usr/bin/env bash
# Bước E: venv riêng .venv-baselines cho gói ngoài (quyet, laya). KHÔNG đụng .venv chính.
# torch / transformers / safetensors / huggingface_hub / numpy dùng lại bản ghim của .venv chính qua một file .pth (chỉ đọc), nên phiên bản thư viện
# nền giống hệt lần chấm của dự án. Gói ngoài cài --no-deps từ wheel đã tải và kiểm sha256 với PyPI (results/baselines/pkgsrc, results/baselines/pypi).
# Dùng: baselines_v1_2_venv.sh create | install
set -euo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
V=.venv-baselines
case "${1:-}" in
  create)
    [ -d $V ] && { echo "$V đã có"; exit 0; }
    ~/.local/bin/uv venv $V --python "$(readlink -f .venv/bin/python)"
    SP=$(ls -d $V/lib/python3.*/site-packages)
    echo "/mnt/c/Users/dungbt06/Fine-tune/.venv/lib/python3.12/site-packages" > $SP/zz_main_venv_readonly.pth
    $V/bin/python -c "import torch, transformers, safetensors, huggingface_hub, numpy; print('torch', torch.__version__, 'transformers', transformers.__version__, 'safetensors', safetensors.__version__, 'hub', huggingface_hub.__version__, 'numpy', numpy.__version__)"
    ;;
  install)
    for w in results/baselines/pkgsrc/quyet-1.0.2-py3-none-any.whl results/baselines/pkgsrc/laya-0.4.6-py3-none-any.whl; do
      n=$(basename $w); pkg=${n%%-*}
      want=$(python3 -c "import json,sys; d=json.load(open('results/baselines/pypi/$pkg.json')); print([u['digests']['sha256'] for u in d['urls'] if u['filename']=='$n'][0])")
      got=$(sha256sum $w | cut -d' ' -f1)
      [ "$want" = "$got" ] || { echo "sha256 $n KHÁC PyPI"; exit 1; }
      ~/.local/bin/uv pip install --no-deps --offline --python $V/bin/python $w
    done
    $V/bin/python -m pip --version >/dev/null 2>&1 || true
    ~/.local/bin/uv pip freeze --python $V/bin/python
    ;;
  *) echo "dùng: create | install"; exit 2;;
esac
