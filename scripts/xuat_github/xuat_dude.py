"""Xuất bản sao danh sách trắng của dự án sang repo riêng `_github/dude-1.7B/` để push lên GitHub (private).

Chỉ chép code, cấu hình nhỏ và báo cáo. Không chép dữ liệu, adapter, preds, test hay tài liệu của bên thứ ba.
Repo git cục bộ của cổng duyệt (danh sách trắng ở `.gitignore` gốc) không bị đụng tới.

Dùng: python scripts/xuat_github/xuat_dude.py   (chạy lại được; xóa sạch thư mục đích trừ `.git`, các file README.md / RELEASES.md /
MANIFEST_SHA.md / release_notes/ do điều phối viết tay, rồi chép lại)
"""
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "_github" / "dude-1.7B"
GIU_TAY = {".git", "README.md", "RELEASES.md", "MANIFEST_SHA.md", "release_notes", ".gitattributes", ".gitignore"}

FILE_GOC = ["PLAN_decisions.md", "TONG_QUAN_DU_AN.md", "BAO_CAO_SANG_14_10.md", "requirements.lock.txt"]
CODE_EXT = {".py", ".sh", ".ps1"}
LOAI_TRU_PHAN = ("raw_refs", "raw_src", "raw", ".cache", "__pycache__", "node_modules",
                 # báo cáo soát dịch toàn bench (chạm test, kiem_xuat.py bắt được) và ứng viên bench 1.4 (soát MLQA test, README bên thứ ba)
                 "translation_audit", "translation_audit_v1_2", "bench_candidates")


def chon():
    out = []
    for f in FILE_GOC:
        if (ROOT / f).is_file():
            out.append(Path(f))
    for goc in ("scripts", "setup", "tools/review_app"):
        for p in (ROOT / goc).rglob("*"):
            if p.is_file() and not any(x in p.parts for x in LOAI_TRU_PHAN) and (p.suffix in CODE_EXT or p.suffix == ".md"):
                out.append(p.relative_to(ROOT))
    for p in (ROOT / "config").glob("*.json"):  # chỉ file .json cấp đầu; bỏ vi_proposals.jsonl và archive_v1_0/
        out.append(p.relative_to(ROOT))
    for p in (ROOT / "results").rglob("*.md"):
        if not any(x in p.parts for x in LOAI_TRU_PHAN):
            out.append(p.relative_to(ROOT))
    out.append(Path("results/TONG_KET_DU_AN_29_10.html"))
    return sorted(set(out))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for p in DEST.iterdir():
        if p.name in GIU_TAY:
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    files = chon()
    tong = 0
    for rel in files:
        src, dst = ROOT / rel, DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        tong += src.stat().st_size
    (DEST / "_danh_sach_file.txt").write_text("\n".join(str(f).replace("\\", "/") for f in files) + "\n", encoding="utf-8")
    print(f"chép {len(files)} file, {tong / 1e6:.2f} MB → {DEST}")


if __name__ == "__main__":
    main()
