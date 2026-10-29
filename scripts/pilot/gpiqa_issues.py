"""Việc 1: gộp kết quả soát Opus (quét → soi lại) của 100 câu Global-PIQA vi (lô 1 = pilot/, lô 2 = pilot2/) thành danh sách
chính tả / nhãn sai / mơ hồ / lỗi phía EN. Chỉ báo cáo, chưa ghi đề xuất theo câu. Ra: pilot2/gpiqa_issues_100.jsonl, pilot2/gpiqa_issues_100.md."""
import json
from collections import Counter
from pathlib import Path

B = Path(__file__).resolve().parents[2] / "results/bench_candidates"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
items = []
for lo, scan, ver in ((1, B / "pilot/audit/out_gpiqa.jsonl", B / "pilot/audit/verify_out_gpiqa.jsonl"),
                      (2, B / "pilot2/audit/out_gpiqa_b.jsonl", B / "pilot2/audit/verify_out_gpiqa_b.jsonl")):
    s = read(scan); v = {x["audit_id"]: x for x in read(ver)}
    assert len(s) == 50
    for x in s:
        y = v.get(x["audit_id"])
        final = y["verdict_cuoi"] if y else ("OK" if x["verdict"] == "OK" else None)
        assert final is not None, x["audit_id"]
        src = y or x
        items.append({"audit_id": x["audit_id"], "lo": lo, "da_cham_model": lo == 1, "verdict_cuoi": final, "ben_sai": src.get("ben_sai", "-"),
                      "truong": src.get("truong", "-"), "doi_dap_an": bool(src.get("doi_dap_an")) if final != "OK" else False,
                      "gia_tri_cu": src.get("gia_tri_cu") or "", "gia_tri_de_xuat": src.get("gia_tri_de_xuat") or "",
                      "dap_an_dung_theo_ban": src.get("dap_an_dung_theo_ban"), "giai_thich": src.get("giai_thich", ""), "do_tin_cay": src.get("do_tin_cay"),
                      "ket_luan_soi_lai": y["ket_luan"] if y else "(không bị gắn cờ ở lượt quét)"})
assert len(items) == 100 and len({i["audit_id"] for i in items}) == 100
with open(B / "pilot2/gpiqa_issues_100.jsonl", "w", encoding="utf-8") as f:
    for i in items:
        f.write(json.dumps(i, ensure_ascii=False) + "\n")
e = lambda s: str(s if s is not None else "").replace("|", "\\|").replace("\n", " ")
cnt = {lo: Counter(i["verdict_cuoi"] for i in items if i["lo"] == lo) for lo in (1, 2)}
tot = Counter(i["verdict_cuoi"] for i in items)
L = ["| Kết luận cuối | Lô 1 (đã chấm model) | Lô 2 (chưa chấm, giữ sạch) | Tổng 100 |", "|---|---|---|---|"]
for k in ("OK", "NHAN_NGUON_SAI_CHAC", "NHAN_NGUON_CO_THE_SAI", "MO_HO", "LOI_CHINH_TA", "DICH_SAI", "DICH_CHUA_CHUAN", "TACH_TIEN_TO_LOI"):
    L.append(f"| {k} | {cnt[1][k]} | {cnt[2][k]} | {tot[k]} |")
L += ["", f"Có `doi_dap_an: true`: lô 1 {sum(i['doi_dap_an'] for i in items if i['lo'] == 1)}, lô 2 {sum(i['doi_dap_an'] for i in items if i['lo'] == 2)}.", ""]
for title, keys in (("Nhãn sai chắc chắn", ("NHAN_NGUON_SAI_CHAC",)), ("Nhãn có thể sai / mơ hồ", ("NHAN_NGUON_CO_THE_SAI", "MO_HO")),
                    ("Lỗi chính tả trong bản Việt (bản gốc do người viết)", ("LOI_CHINH_TA",)),
                    ("Lỗi phía tiếng Anh (bản dịch của bộ gốc, hoặc tách phần đầu chung)", ("DICH_SAI", "DICH_CHUA_CHUAN", "TACH_TIEN_TO_LOI"))):
    xs = [i for i in items if i["verdict_cuoi"] in keys]
    L += [f"#### {title} ({len(xs)})", "", "| Câu | Lô | Kết luận | Bên/trường | Sửa (cũ → mới) | Đáp án theo Opus | Giải thích |", "|---|---|---|---|---|---|---|"]
    for i in xs:
        fix = f"{e(i['gia_tri_cu'])} → {e(i['gia_tri_de_xuat'])}" if i["gia_tri_cu"] else ""
        L.append(f"| `{i['audit_id'].replace('gpiqa:', '')}` | {i['lo']} | {i['verdict_cuoi']} | {i['ben_sai']}/{i['truong']} | {fix} | "
                 f"{'' if i['dap_an_dung_theo_ban'] is None else i['dap_an_dung_theo_ban']} | {e(i['giai_thich'])} |")
    L.append("")
(B / "pilot2/gpiqa_issues_100.md").write_text("\n".join(L), encoding="utf-8")
print(json.dumps({"lo1": dict(cnt[1]), "lo2": dict(cnt[2]), "tong": dict(tot)}, ensure_ascii=False))
