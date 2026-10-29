"""Bước 1: chạy scripts/eval_base.py nguyên vẹn (qua runpy) với cuDNN SDPA tắt; mọi tham số dòng lệnh chuyển thẳng cho eval_base.

Dùng: eval_nocudnn.py [--keep-cudnn] <tham số của eval_base.py ...>
  --keep-cudnn: không tắt (lượt đối chứng của chính wrapper).
"""
import runpy
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
args = sys.argv[1:]
keep = "--keep-cudnn" in args
args = [a for a in args if a != "--keep-cudnn"]
if not keep:
    torch.backends.cuda.enable_cudnn_sdp(False)
print("cudnn_sdp_enabled:", torch.backends.cuda.cudnn_sdp_enabled(), flush=True)
sys.argv = [str(SCRIPTS / "eval_base.py")] + args
runpy.run_path(str(SCRIPTS / "eval_base.py"), run_name="__main__")
