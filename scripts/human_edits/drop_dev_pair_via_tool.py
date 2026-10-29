"""Loại MỘT cặp dev qua API của công cụ đang chạy (POST /api/item, field loai_bo; có token + Host), không nối thẳng vào edits_log.
Công cụ không chạy hoặc không dùng đúng lớp ghi đè thì dừng, không ghi gì, không tự khởi động lại. Python Windows.
  python scripts/human_edits/drop_dev_pair_via_tool.py --pair_id massive:4129 --note "..." [--port 8765] [--dry_run]"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/review_app"))
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
ap = argparse.ArgumentParser()
ap.add_argument("--pair_id", required=True)
ap.add_argument("--note", required=True)
ap.add_argument("--reason", default="khác")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--edits_dir", default=str(ROOT / "data/human_edits"))
ap.add_argument("--dry_run", action="store_true")
args = ap.parse_args()
HOST = f"127.0.0.1:{args.port}"
LOG = Path(args.edits_dir) / "edits_log.jsonl"
assert args.reason in E.REASONS
dev = {json.loads(l)["pair_id"] for l in open(ROOT / "data/bench_v1_1/dev_v1_1.jsonl", encoding="utf-8") if l.strip()}
assert args.pair_id in dev, f"{args.pair_id} không có trong dev 1.1"
st = E.derive_state(E.read_log(LOG), inventory(ROOT / "config"))
if "loai_bo" in st["cau"].get(args.pair_id, {}):
    print(f"{args.pair_id} đã bị loại sẵn; không ghi gì."); sys.exit(0)
if args.dry_run:
    print(f"sẽ loại {args.pair_id}, lý do '{args.reason}', ghi chú: {args.note}"); sys.exit(0)
try:
    html = urllib.request.urlopen(urllib.request.Request(f"http://{HOST}/", headers={"Host": HOST}), timeout=10).read().decode("utf-8")
    token = json.loads(re.search(r'var TOKEN=("[^"]+")', html).group(1))
except Exception as ex:
    print(f"Công cụ không chạy ở {HOST} ({type(ex).__name__}). KHÔNG ghi gì; báo phiên điều phối."); sys.exit(3)
m_ = re.search(r"Bản ghi trong log hiệu đính: (\d+)", html)
n_log = len(E.read_log(LOG))
if not m_ or int(m_.group(1)) != n_log:
    print(f"Công cụ báo {m_.group(1) if m_ else '?'} bản ghi, {LOG} có {n_log}: không dùng đúng lớp ghi đè. KHÔNG ghi gì."); sys.exit(4)
body = json.dumps({"dev_id": args.pair_id, "field": "loai_bo", "action": "sua", "value": True, "reason": args.reason, "note": args.note}).encode("utf-8")
req = urllib.request.Request(f"http://{HOST}/api/item", data=body, method="POST",
                             headers={"Host": HOST, "Content-Type": "application/json", "X-Review-Token": token})
r = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
assert r.get("ok"), r
log2 = E.read_log(LOG)
assert len(log2) == n_log + 1 and log2[-1].get("truong") == "loai_bo" and (log2[-1].get("dev_id") or log2[-1].get("pair_id")) == args.pair_id, "log không tăng đúng 1 bản ghi"
st2 = E.derive_state(log2, inventory(ROOT / "config"))
assert "loai_bo" in st2["cau"].get(args.pair_id, {})
print(json.dumps({"da_loai": args.pair_id, "ban_ghi": log2[-1]["id"], "thoi_diem": log2[-1]["thoi_diem"], "so_ban_ghi_log": len(log2)}, ensure_ascii=False))
