#!/usr/bin/env bash
# Tải wheel của quyet/laya (không cài, không phụ thuộc) và giải nén để đọc mã nguồn trước khi cài.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
D=results/baselines/pkgsrc
mkdir -p $D
for spec in quyet==1.0.2 laya==0.4.6; do
  .venv/bin/python -m pip --version >/dev/null 2>&1 && PIP=".venv/bin/python -m pip" || PIP="$HOME/.local/bin/uv pip"
  curl -sf -o /dev/null https://pypi.org/simple/ || echo "pypi unreachable"
done
for spec in quyet/1.0.2/quyet-1.0.2-py3-none-any.whl laya/0.4.6/laya-0.4.6-py3-none-any.whl; do :; done
.venv/bin/python - <<'EOF'
import json, urllib.request, zipfile, io, hashlib, pathlib
D = pathlib.Path("results/baselines/pkgsrc")
for name, ver in (("quyet", "1.0.2"), ("laya", "0.4.6")):
    meta = json.load(open(f"results/baselines/pypi/{name}.json"))
    whl = next(u for u in meta["releases"][ver] if u["filename"].endswith(".whl"))
    data = urllib.request.urlopen(whl["url"]).read()
    assert hashlib.sha256(data).hexdigest() == whl["digests"]["sha256"], name
    (D / whl["filename"]).write_bytes(data)
    out = D / f"{name}-{ver}"
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(name, ver, "sha256 ok,", len(files), "files:", files[:60])
EOF
