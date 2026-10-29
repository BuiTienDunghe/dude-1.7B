"""Gộp ý kiến thứ hai (Opus) cho đề xuất đang chờ: out_*.jsonl + out_template.jsonl -> second_opinion.jsonl (dùng cho công cụ, chỉ đọc),
second_opinion.csv (UTF-8 có BOM) và REPORT_second_opinion.md. Không ghi vào config. Python Windows, thư viện chuẩn."""
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/translation_audit/second_opinion"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
props = {p["id"]: p for p in read(ROOT / "config/vi_proposals.jsonl")}
units = [u for p in sorted(D.glob("units_*.jsonl")) for u in read(p)]
kiem = {pp["id"]: pp["kiem_may"] for u in units for pp in u["props"]}
expected = set(kiem) | {json.loads(l)["de_xuat"]["id"] for l in open(D / "template_units.jsonl", encoding="utf-8") if l.strip()}
so = {}
for p in sorted(D.glob("out_*.jsonl")):
    for x in read(p):
        so[x["id"]] = x
missing = sorted(expected - set(so))
extra = sorted(set(so) - expected)
KL = ("DONG_Y", "DONG_Y_NHUNG_CO_CHUOI_TOT_HON", "BAC_GIU_NGUYEN", "CAN_NGUOI_XEM")
bad_kl = [i for i, x in so.items() if x.get("ket_luan") not in KL]
rows = []
for i in sorted(expected & set(so), key=lambda i: (props[i].get("split", "mẫu"), props[i].get("nguon", ""), i)):
    p, x = props[i], so[i]
    rows.append({"id": i, "split": p.get("split", "mẫu"), "nguon": p.get("nguon", p.get("khoa", "")), "pair_id": p.get("pair_id", ""),
                 "truong": p.get("truong", p.get("khoa", "")), "gia_tri_cu": p.get("gia_tri_cu", ""), "gia_tri_de_xuat": p.get("gia_tri_de_xuat", ""),
                 "ket_luan": x.get("ket_luan"), "chuoi_thay_the": x.get("chuoi_thay_the") or "", "ly_do": x.get("ly_do", ""),
                 "gold_sau_sua": x.get("gold_sau_sua", ""), "loi_moi": x.get("loi_moi") or "", "do_tin_cay": x.get("do_tin_cay"),
                 "doi_dap_an_lan_1": p.get("doi_dap_an", ""), "nfc_de_xuat": (kiem.get(i) or {}).get("nfc_de_xuat", "")})
with open(D / "second_opinion.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({k: r[k] for k in ("id", "ket_luan", "chuoi_thay_the", "ly_do", "gold_sau_sua", "loi_moi", "do_tin_cay")}, ensure_ascii=False) + "\n")
buf = io.StringIO()
cols = ["id", "split", "nguon", "pair_id", "truong", "gia_tri_cu", "gia_tri_de_xuat", "ket_luan", "chuoi_thay_the", "ly_do"]
w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
(D / "second_opinion.csv").write_text("\ufeff" + buf.getvalue(), encoding="utf-8")

e = lambda s: str("" if s is None else s).replace("|", "\\|").replace("\n", " ")
cnt = Counter(r["ket_luan"] for r in rows)
by = defaultdict(Counter)
for r in rows:
    by[(r["split"], r["nguon"] if r["split"] != "mẫu" else "chuỗi mẫu")][r["ket_luan"]] += 1
L = ["# Ý kiến thứ hai cho đề xuất đang chờ duyệt", "",
     f"Ngày 2026-10-14. Một lượt Opus 5.5 **độc lập** (agent mới, không phải agent đã đề xuất) cho ý kiến về {len(rows)}/{len(expected)} đề xuất đang chờ trong "
     "`config/vi_proposals.jsonl` (chỉ đọc; không ghi gì vào config, không tạo đề xuất mới). Hướng dẫn: `SECOND_OPINION.md`. "
     "Đây là ý kiến tham khảo để duyệt nhanh; người làm dự án vẫn quyết từng đề xuất trong công cụ.", "",
     "## Tổng", "", "| Kết luận | Số đề xuất |", "|---|---|"]
L += [f"| {k} | {cnt[k]} |" for k in KL]
L += ["", f"Thiếu kết luận: {len(missing)}; kết luận không hợp lệ: {len(bad_kl)}.",
      f"Gold sau khi sửa không còn đúng hoặc mơ hồ (theo ý kiến thứ hai): {sum(1 for r in rows if r['gold_sau_sua'] in ('KHONG_CON_DUNG', 'MO_HO'))}. "
      f"Đề xuất gây lỗi mới: {sum(1 for r in rows if r['loi_moi'])}. Chuỗi đề xuất không ở dạng NFC (kiểm máy): {sum(1 for r in rows if r['nfc_de_xuat'] is False)}.", "",
      "## Theo split và nguồn", "", "| Split | Nguồn | " + " | ".join(KL) + " | Tổng |", "|---|---|" + "---|" * (len(KL) + 1)]
for (s, n), c in sorted(by.items()):
    L.append(f"| {s} | {n} | " + " | ".join(str(c[k]) for k in KL) + f" | {sum(c.values())} |")
for title, keys in (("BAC_GIU_NGUYEN (nên từ chối)", ("BAC_GIU_NGUYEN",)), ("CAN_NGUOI_XEM", ("CAN_NGUOI_XEM",)),
                    ("DONG_Y_NHUNG_CO_CHUOI_TOT_HON (duyệt bằng \"Sửa rồi duyệt\" với chuỗi thay thế)", ("DONG_Y_NHUNG_CO_CHUOI_TOT_HON",))):
    xs = [r for r in rows if r["ket_luan"] in keys]
    L += ["", f"## {title}: {len(xs)}", "", "| id | split | nguồn | trường | cũ → đề xuất | chuỗi thay thế | gold sau sửa | lý do |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| `{r['id']}` | {r['split']} | {e(r['nguon'])} | {e(r['truong'])} | {e(r['gia_tri_cu'][:120])} → {e(r['gia_tri_de_xuat'][:120])} | {e(r['chuoi_thay_the'][:160])} | "
          f"{e(r['gold_sau_sua'])} | {e(r['ly_do'])}{(' Lỗi mới: ' + e(r['loi_moi'])) if r['loi_moi'] else ''} |" for r in xs]
multi = [u for u in units if len(u["props"]) > 1]
L += ["", f"## Dòng có từ 2 đề xuất trở lên: {len(multi)} dòng (nên duyệt CÙNG LÚC)", "",
      "Các đề xuất trên cùng một dòng thường phụ thuộc nhau, ví dụ sửa đoạn văn đi kèm sửa một lựa chọn hay câu hỏi. Duyệt lẻ một cái có thể làm đoạn văn và lựa chọn lệch nhau, "
      "hoặc làm đáp án không còn duy nhất; lượt ý kiến thứ hai đã nêu vài cặp như vậy. Đoạn văn dùng chung giữa nhiều câu (Belebele #1/#2…) thì không cần đề xuất riêng cho từng câu: "
      "công cụ áp một đề xuất sửa đoạn văn cho mọi câu cùng split có cùng đoạn văn hiện hành. Đề xuất trùng ở câu kia sau đó sẽ báo \"xuất hiện 0 lần\"; khi đó chỉ việc từ chối.", "",
      "| split | row_id | đề xuất (trường) |", "|---|---|---|"]
L += [f"| {u['split']} | `{e(u['row_id'])}` | " + ", ".join(f"`{pp['id']}` ({e(pp['truong'])})" for pp in u["props"]) + " |" for u in multi]
L += ["", "## File", "", "- `second_opinion.csv` (UTF-8 có BOM): id, split, nguon, pair_id, truong, gia_tri_cu, gia_tri_de_xuat, ket_luan, chuoi_thay_the, ly_do.",
      "- `second_opinion.jsonl`: bản gọn cho công cụ (màn \"Chờ duyệt\" hiện ý kiến thứ hai ở dạng chỉ đọc).",
      "- `units_*.jsonl` (đầu vào theo dòng benchmark, kèm kiểm máy), `out_*.jsonl` (đầu ra từng lô), `template_units.jsonl`, `out_template.jsonl`.",
      "- Code: `scripts/second_opinion/build_units.py`, `report_second_opinion.py`."]
(D / "REPORT_second_opinion.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(json.dumps({"n": len(rows), "expected": len(expected), "missing": len(missing), "extra": extra[:5], "bad": bad_kl[:5], "ket_luan": dict(cnt)}, ensure_ascii=False))
