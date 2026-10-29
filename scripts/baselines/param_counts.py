"""Số tham số (safetensors) của các model dùng trong bảng baseline, ở revision ghim."""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import model_revisions

api = HfApi()
out = {}
for m, rev in model_revisions().items():
    info = api.model_info(m, revision=rev, files_metadata=False)
    st = getattr(info, "safetensors", None)
    out[m] = {"total": st.total if st else None, "by_dtype": dict(st.parameters) if st else None}
print(json.dumps(out, indent=1))
