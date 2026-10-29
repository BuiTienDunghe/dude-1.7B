"""Bước 1 của results/BRIEF_2b_du_lieu.md: tải Wikipedia tiếng Việt wikimedia/wikipedia subset 20231101.vi (4 file parquet, ~0,73 GB,
CC BY-SA 4.0, chỉ dùng nội bộ). Chỉ lấy parquet bằng huggingface_hub (allow_patterns="20231101.vi/*"), không chạy code của repo dataset.
Kiểm trước khi tải: đúng 4 file, tổng < 1 GB, LFS oid khớp đầu oid điều phối đã kiểm; sau khi tải: sha256 từng file = LFS oid.
Ra results/train_v1/raw_refs/wikimedia__wikipedia_20231101.vi/ (thư mục mới, trống) + manifest_tai.json trong results/train_2b/.
Dùng: bash setup/run.sh -I scripts/train_2b/tai_wiki.py"""
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path("/mnt/c/Users/dungbt06/Fine-tune")
DST = ROOT / "results/train_v1/raw_refs/wikimedia__wikipedia_20231101.vi"
OUT = ROOT / "results/train_2b"
REPO, PAT = "wikimedia/wikipedia", "20231101.vi/*"
OID_DAU = ["ec53cc98", "d1e724eb", "f364415c", "0f5c15b2"]

assert not DST.exists() or not any(DST.iterdir()), "thư mục đích phải mới và trống"
info = HfApi().dataset_info(REPO, files_metadata=True)
files = sorted((s for s in info.siblings if s.rfilename.startswith("20231101.vi/")), key=lambda s: s.rfilename)
oid = {s.rfilename: s.lfs.sha256 for s in files}
tot = sum(s.size for s in files)
assert len(files) == 4 and tot < 1e9, (len(files), tot)
assert [oid[f.rfilename][:8] for f in files] == OID_DAU, oid
t0 = time.time()
snapshot_download(REPO, repo_type="dataset", revision=info.sha, allow_patterns=PAT, local_dir=DST)
got = {}
for f in files:
    p = DST / f.rfilename
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == oid[f.rfilename], (f.rfilename, h)
    got[f.rfilename] = {"bytes": p.stat().st_size, "lfs_oid_sha256": h}
OUT.mkdir(parents=True, exist_ok=True)
man = {"repo": REPO, "revision": info.sha, "allow_patterns": PAT, "thu_muc": str(DST.relative_to(ROOT)), "file": got,
       "tong_bytes": tot, "giay": round(time.time() - t0), "giay_phep": "CC BY-SA 4.0 (chỉ dùng nội bộ)",
       "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
(OUT / "manifest_tai_wiki.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(man, ensure_ascii=False))
