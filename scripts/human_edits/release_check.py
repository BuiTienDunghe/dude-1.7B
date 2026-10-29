"""PHẦN 3: cổng duyệt trước khi dựng bản công bố.

Từ chối (mã thoát 2) nếu còn chuỗi tiếng Việt do agent viết mà benchmark dùng (tools/review_app/inventory.py) ở trạng thái
"chưa duyệt" sau khi áp lớp ghi đè của người (data/human_edits/edits_log.jsonl), và in danh sách các chuỗi đó (mỗi dòng cắt 200 ký tự).
Đề xuất đang chờ trong config/vi_proposals.jsonl được liệt kê để tham khảo; chúng chưa vào benchmark nên không chặn.
Chuỗi lấy nguyên từ nguồn không thuộc cổng này. Không đọc, không in dữ liệu test.
Nếu không còn chuỗi chưa duyệt: in "ĐẠT" và lệnh dựng (scripts/human_edits/build_revised.py); script này không tự dựng.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "review_app"))
import edits as E  # noqa: E402
from inventory import GROUPS, inventory  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
ap = argparse.ArgumentParser()
ap.add_argument("--config_dir", default=str(ROOT / "config"))
ap.add_argument("--edits_dir", default=str(ROOT / "data" / "human_edits"))
ap.add_argument("--proposals", default=str(ROOT / "config" / "vi_proposals.jsonl"))
args = ap.parse_args()
INV = inventory(args.config_dir)
st = E.derive_state(E.read_log(Path(args.edits_dir) / "edits_log.jsonl"), INV)
unapproved = [x for x in INV if st["mau"][x["key"]]["trang_thai"] == "chưa duyệt"]
props = E.read_proposals(args.proposals) if Path(args.proposals).exists() else []
pending = E.pending_proposals(props, st)
print(f"Chuỗi tiếng Việt do agent viết: {len(INV)} — chưa duyệt {st['dem_mau']['chưa duyệt']}, đã duyệt {st['dem_mau']['đã duyệt']}, đã sửa {st['dem_mau']['đã sửa']}.")
kinds = {k: sum(1 for p in pending if (p.get("split") if p.get("loai") == "cau" else "mẫu") == k) for k in ("mẫu", "dev", "test")}
print(f"Đề xuất đang chờ trong {args.proposals}: {len(pending)} (chuỗi mẫu {kinds['mẫu']}, câu dev {kinds['dev']}, câu test {kinds['test']}; "
      f"không chặn phát hành; chưa vào benchmark).")
print(f"Câu đã có chỉnh sửa của người: dev {len(st['cau'])}, test {len(st['cau_test'])} (test chỉ đổi khi dựng bản 1.2 theo yêu cầu).")
if unapproved:
    print(f"TỪ CHỐI dựng bản công bố: còn {len(unapproved)} chuỗi chưa duyệt.")
    for g in GROUPS:
        xs = [x for x in unapproved if x["group"] == g]
        if xs:
            print(f"[{g}] {len(xs)}")
            for x in xs:
                print(f"  {x['key']} | {st['mau'][x['key']]['gia_tri']}"[:200])
    sys.exit(2)
print("ĐẠT: mọi chuỗi tiếng Việt do agent viết đã được người làm dự án duyệt hoặc sửa.")
print("Dựng bản hiệu đính: python scripts/human_edits/build_revised.py --out_dir data/bench_v1_1_rev (chỉ khi người làm dự án yêu cầu).")
