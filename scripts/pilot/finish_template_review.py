"""Việc 3: ghi đề xuất chuỗi mẫu (chỉ chuỗi cần sửa) vào config/vi_proposals.jsonl theo định dạng đề xuất chuỗi mẫu đang có, và viết
results/review_tool/REPORT_template_review.md (liệt kê chuỗi OK để người dùng duyệt nhanh). Không đổi trạng thái duyệt."""
import datetime
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results/bench_candidates/pilot2/audit"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
inp = read(A / "templates_in.jsonl")
l1 = {x["khoa"]: x for x in read(A / "templates_out.jsonl")}
l2 = {x["khoa"]: x for x in read(A / "templates_verify_out.jsonl")}
assert len(inp) == len(l1) == len(l2) == 65
now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
props, ok_rows, sua_rows = [], [], []
for x in inp:
    k, a, b = x["khoa"], l1[x["khoa"]], l2[x["khoa"]]
    if b["ket_qua_cuoi"] == "SUA":
        val = b.get("gia_tri_cuoi") or a.get("gia_tri_de_xuat")
        pid = "tpl-" + k.replace("translations.", "").replace(".", "-")
        props.append({"id": pid, "khoa": k, "gia_tri_de_xuat": val, "ban_tieng_anh": x["en"],
                      "ly_do": f"[{a.get('loai_van_de') or '-'}; soát Opus 2 lượt, lượt 2 {b['ket_luan']}] {a['ly_do']}",
                      "nguoi_de_xuat": "agent_opus (soát chuỗi mẫu, phiên Decision model tiếng Việt)", "thoi_diem": now})
        sua_rows.append((x, val, a, b))
    else:
        ok_rows.append(x)
p = ROOT / "config/vi_proposals.jsonl"
ids = {json.loads(l)["id"] for l in open(p, encoding="utf-8") if l.strip()}
assert not ids & {q["id"] for q in props}, "id trùng"
with open(p, "rb") as f:
    f.seek(-1, os.SEEK_END); last = f.read(1)
with open(p, "a", encoding="utf-8") as f:
    if last != b"\n":
        f.write("\n")
    for q in props:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
    f.flush(); os.fsync(f.fileno())

e = lambda s: str(s).replace("|", "\\|")
L = ["# Soát 65 chuỗi mẫu tiếng Việt \"chưa duyệt\" (màn \"Mẫu và nhãn\")", "",
     f"Ngày {now[:10]}. Soát bằng Opus 5.5, hai lượt:",
     "",
     "- quét theo `results/bench_candidates/pilot2/audit/TEMPLATE_REVIEW.md`;",
     "- soi lại độc lập theo `TEMPLATE_VERIFY.md`.",
     "",
     "Không đổi trạng thái duyệt nào. Chỉ chuỗi cần sửa được ghi thành đề xuất trong `config/vi_proposals.jsonl` (định dạng đề xuất chuỗi mẫu, khóa trùng khóa của danh mục). Khi duyệt một đề xuất trong công cụ, công cụ áp nó như một lần sửa ở \"Mẫu và nhãn\".", "",
     "## Tóm tắt", "", "| Nhóm | Số chuỗi | OK | Đề xuất sửa |", "|---|---|---|---|"]
for g in ("Câu hỏi mẫu", "Tên chủ đề SIB-200", "Mô tả ý định MASSIVE", "Có/Không"):
    n = sum(x["nhom"] == g for x in inp); s = sum(x["nhom"] == g for x, *_ in sua_rows)
    L.append(f"| {g} | {n} | {n - s} | {s} |")
L += ["", f"Lượt 1: OK {sum(v['ket_luan'] == 'OK' for v in l1.values())}, SUA {sum(v['ket_luan'] == 'SUA' for v in l1.values())}. "
      f"Lượt 2: {dict(Counter(v['ket_luan'] for v in l2.values()))}.", "",
      "## Đề xuất sửa (đã ghi vào `config/vi_proposals.jsonl`, chờ duyệt ở màn \"Chờ duyệt\")", "",
      "| id đề xuất | Khóa | EN | Hiện hành | Đề xuất | Lý do | Tin cậy (lượt 1 / lượt 2) |", "|---|---|---|---|---|---|---|"]
for (x, val, a, b), q in zip(sua_rows, props):
    L.append(f"| `{q['id']}` | `{x['khoa']}` | {e(x['en'])} | {e(x['vi_hien_hanh'])} | **{e(val)}** | {e(a['ly_do'])} | {a.get('do_tin_cay')} / {b.get('do_tin_cay')} |")
L += ["", "## Chuỗi OK: duyệt nhanh ở màn \"Mẫu và nhãn\"", "",
      "Cả hai lượt đều kết luận OK cho các chuỗi dưới đây. Khi người làm dự án đồng ý, bấm \"Duyệt\" từng dòng.", ""]
for g in ("Câu hỏi mẫu", "Tên chủ đề SIB-200", "Mô tả ý định MASSIVE", "Có/Không"):
    xs = [x for x in ok_rows if x["nhom"] == g]
    if not xs:
        continue
    L += [f"### {g} ({len(xs)})", "", "| Khóa | EN | Nhãn gốc | Tiếng Việt |", "|---|---|---|---|"]
    L += [f"| `{x['khoa']}` | {e(x['en'])} | {e(x['nhan_goc'])} | {e(x['vi_hien_hanh'])} |" for x in xs]
    L.append("")
L += ["## File", "",
      "- Đầu vào: `results/bench_candidates/pilot2/audit/templates_in.jsonl`, `templates_context.json`.",
      "- Kết quả lượt 1: `templates_out.jsonl`.",
      "- Kết quả lượt 2: `templates_verify_in.jsonl`, `templates_verify_out.jsonl`.",
      "- Code: `scripts/pilot/build_pilot2_inputs.py`, `make_template_verify.py`, `finish_template_review.py`."]
(ROOT / "results/review_tool/REPORT_template_review.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(json.dumps({"de_xuat": [q["id"] for q in props], "ok": len(ok_rows), "vi_proposals_lines": sum(1 for l in open(p, encoding="utf-8") if l.strip())}, ensure_ascii=False))
