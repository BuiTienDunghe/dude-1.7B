#!/usr/bin/env bash
# Tạo venv riêng trong thư mục dự án bằng uv (Python 3.12) và cài PyTorch build CUDA 12.8+
# (cần cho GPU Blackwell sm_120 như RTX 5060 Ti).
set -euo pipefail
PROJ=/mnt/c/Users/dungbt06/Fine-tune
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version
cd "$PROJ"
export UV_LINK_MODE=copy
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=4.51" datasets accelerate numpy scipy pandas tiktoken sentencepiece
uv pip freeze --python .venv/bin/python > requirements.lock.txt
