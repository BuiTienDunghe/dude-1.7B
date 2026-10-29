"""Việc 2 (stage_v1_2): gộp soát Opus (quét → soi lại) 197 câu MLQA test-candidate -> stage_v1_2/mlqa_test_fixes.jsonl (CHƯA áp),
tóm tắt nối vào REPORT_stage_v1_2.md, và file đọc kiểm 30 câu (seed 20261014): results/review/mlqa_test_review.html + mlqa_test_review_sheet.csv.
Tự chứa, escape mọi văn bản, không script, không tài nguyên ngoài. Python Windows, thư viện chuẩn."""
import csv
import glob
import html
import io
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "results/bench_candidates/stage_v1_2"
A = S / "audit_mlqa"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
e = lambda s: html.escape("" if s is None else str(s), quote=True)
SEED, LET = 20261014, "ABCD"
rows = {r["id"]: r for r in read(S / "mlqa_test_candidates.jsonl")}
scan = {x["audit_id"]: x for f in sorted(glob.glob(str(A / "out_*.jsonl"))) for x in read(f)}
ver = {x["audit_id"]: x for x in read(A / "verify_out.jsonl")}
assert len(scan) == 197 and set(ver) == {i for i, x in scan.items() if x["verdict"] != "OK"}
final = {}
for i, x in scan.items():
    v = ver.get(i)
    src = v or x
    final[i] = {"pair_id": i, "verdict_cuoi": v["verdict_cuoi"] if v else x["verdict"], "ket_luan_soi_lai": v["ket_luan"] if v else "(không gắn cờ)",
                "ben_sai": src.get("ben_sai", "-"), "truong": src.get("truong", "-"), "doi_dap_an": bool(src.get("doi_dap_an")),
                "gia_tri_cu": src.get("gia_tri_cu") or "", "gia_tri_de_xuat": src.get("gia_tri_de_xuat") or "", "dap_an_goi_y": src.get("dap_an_goi_y") or "",
                "giai_thich": src.get("giai_thich", ""), "do_tin_cay": src.get("do_tin_cay")}
fixes = [f for f in final.values() if f["verdict_cuoi"] != "OK"]
bad_occ = [f["pair_id"] for f in fixes if f["truong"] == "question" and f["gia_tri_cu"] and rows[f["pair_id"] + ":vi"]["question"].count(f["gia_tri_cu"]) != 1]
with open(S / "mlqa_test_fixes.jsonl", "w", encoding="utf-8") as fo:
    for f in sorted(fixes, key=lambda f: f["pair_id"]):
        fo.write(json.dumps(f, ensure_ascii=False) + "\n")
cnt = Counter(f["verdict_cuoi"] for f in final.values())
kl = Counter(v["ket_luan"] for v in ver.values())
sec = ["", "## 4. Soát câu hỏi và đáp án tiếng Việt của 197 câu MLQA test (bổ sung)", "",
       "- **Cách soát:** Opus 5.5 hai lượt. Lượt quét đi qua cả 197 câu theo `audit_mlqa/AUDIT_MLQA_TEST.md`; lượt soi lại độc lập chỉ xét các câu bị gắn cờ, theo `VERIFY_MLQA_TEST.md`.",
       "- **Phạm vi:** câu hỏi tiếng Việt (chính tả; dịch so với câu hỏi tiếng Anh) và đáp án chuẩn tiếng Việt (có trả lời đúng câu hỏi không). Đoạn văn là Wikipedia gốc, không soát.",
       f"- **Lượt quét:** gắn cờ {len(ver)}/197 câu. Lượt soi lại: {dict(kl)}.",
       f"- **Kết luận cuối:** {dict(cnt)}.",
       f"- **Đổi đáp án:** {sum(f['doi_dap_an'] for f in fixes)} câu có `doi_dap_an`.",
       f"- **Kiểm máy:** {len(bad_occ)} câu có `gia_tri_cu` không xuất hiện đúng 1 lần.",
       "- Phát hiện thêm: một số câu hỏi gốc không ở dạng NFC (Unicode tổ hợp). Những câu này giữ nguyên chuỗi gốc trong `gia_tri_cu`.",
       f"- **File sửa:** `mlqa_test_fixes.jsonl` gồm {len(fixes)} dòng, CHƯA áp, cùng kiểu `gpiqa_lo2_fixes.jsonl`. Câu nhãn sai hoặc mơ hồ có thêm `dap_an_goi_y`.",
       "- **Gợi ý:**",
       "  - câu `NHAN_NGUON_NGHI_SAI` hoặc `MO_HO`: nên loại khỏi bộ, hoặc để người quyết;",
       "  - lỗi câu hỏi (`DICH_SAI`, `LOI_CHINH_TA`, `DICH_CHUA_CHUAN`): sửa qua đề xuất theo câu khi bộ được đưa vào bench.",
       "- **File đọc kiểm:** `results/review/mlqa_test_review.html` và `mlqa_test_review_sheet.csv`, gồm 30 câu chọn ngẫu nhiên (seed 20261014).",
       "", "| Câu | Kết luận | Trường | Sửa (cũ → mới) / đáp án gợi ý | Giải thích |", "|---|---|---|---|---|"]
q = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
for f in sorted(fixes, key=lambda f: (f["verdict_cuoi"], f["pair_id"])):
    fx = f"{q(f['gia_tri_cu'])} → {q(f['gia_tri_de_xuat'])}" if f["gia_tri_cu"] else (f"đáp án gợi ý: {q(f['dap_an_goi_y'])}" if f["dap_an_goi_y"] else "")
    sec.append(f"| `{f['pair_id']}` | {f['verdict_cuoi']} | {f['truong']} | {fx} | {q(f['giai_thich'])} |")
rp = S / "REPORT_stage_v1_2.md"
txt = rp.read_text(encoding="utf-8")
marker = "\n## 4. Soát câu hỏi và đáp án tiếng Việt của 197 câu MLQA test (bổ sung)"
if marker in txt:
    txt = txt[:txt.index(marker)]
rp.write_text(txt.rstrip("\n") + "\n" + "\n".join(sec) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- HTML 30 câu
pairs = sorted({r["pair_id"] for r in rows.values()})
pick = sorted(random.Random(SEED).sample(pairs, 30))


def highlight(text, opts, gold):
    spans = []
    for j, o in enumerate(opts):
        k = text.find(o)
        while o and k >= 0:
            spans.append((k, k + len(o), j)); k = text.find(o, k + 1)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, pos = [], 0
    for s, t, j in spans:
        if s < pos:
            continue
        out.append(e(text[pos:s])); out.append(f'<span class="{"hg" if j == gold else "hd"}" title="lựa chọn {LET[j]}">{e(text[s:t])}</span>'); pos = t
    out.append(e(text[pos:]))
    return "".join(out)


opts_html = lambda opts, gold: '<ol class="opts" type="A">' + "".join(
    f'<li class="{"gold" if j == gold else ""}">{e(o)}{" ✔ đáp án chuẩn" if j == gold else ""}</li>' for j, o in enumerate(opts)) + "</ol>"
CSS = (ROOT / "scripts/stage_v1_2/build_pilot2_review.py").read_text(encoding="utf-8").split('CSS = """', 1)[1].split('"""', 1)[0]
H = [f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đọc kiểm MLQA test</title><style>{CSS}</style></head><body><main>
<h1>Đọc kiểm 30 câu ứng viên test MLQA (recast "4 cụm từ trả lời")</h1>
<p class="note">Dựng bằng <span class="mono">scripts/stage_v1_2/finish_mlqa_test.py</span> từ <span class="mono">stage_v1_2/mlqa_test_candidates.jsonl</span>
(khu tạm; chưa quyết đưa vào bản 1.2; chưa chấm model nào). 30 câu chọn ngẫu nhiên, seed {SEED}. Điền cột verdict ở <span class="mono">results/review/mlqa_test_review_sheet.csv</span>.
Mã verdict: OK / SAI_NHAN / CAU_HOI_MO_HO / DICH_SAI / NHIEU_CUNG_DUNG / NHIEU_QUA_DE / CHINH_TA / KHAC.</p>
<p class="warn">Đáp án chuẩn là span do người gán (MLQA). <b>3 nhiễu do LLM (Opus) chọn</b>, chép nguyên văn từ đoạn văn, có ít nhất 1 nhiễu nằm cùng câu với đáp án ở mỗi phía, và đã qua một lượt Opus kiểm độc lập.
Đoạn Việt và đoạn Anh là hai đoạn Wikipedia viết tự nhiên, không phải bản dịch của nhau. Trong đoạn văn: <span class="hg">đáp án</span>, <span class="hd">nhiễu</span>.
Mỗi câu kèm kết luận soát câu hỏi/đáp án của Opus (2 lượt) và đề xuất sửa (chưa áp).</p>"""]
sheet = []
for n, pid in enumerate(pick, 1):
    iid = f"T-{n:02d}"
    v, en, f = rows[pid + ":vi"], rows[pid + ":en"], final[pid]
    fix = (f'<p class="lab">Đề xuất sửa (chưa áp): {e(f["gia_tri_cu"])} → {e(f["gia_tri_de_xuat"])}</p>' if f["gia_tri_cu"] else "") + \
          (f'<p class="lab">Đáp án gợi ý: {e(f["dap_an_goi_y"])}</p>' if f["dap_an_goi_y"] else "")
    H.append(f'<div class="item"><div><span class="iid">{iid}</span> <span class="lab">{e(pid)} · {e(next((x for x in v["flags"] if x.startswith("loai=")), "")[5:])}</span></div><div class="grid2">'
             f'<div><b>Tiếng Việt</b><div class="text">{highlight(v["state"], v["options"], v["target"])}</div><p><b>Câu hỏi:</b> {e(v["question"])}</p>{opts_html(v["options"], v["target"])}</div>'
             f'<div><b>Tiếng Anh</b><div class="text">{highlight(en["state"], en["options"], en["target"])}</div><p><b>Question:</b> {e(en["question"])}</p>{opts_html(en["options"], en["target"])}</div></div>'
             f'<p class="{"note" if f["verdict_cuoi"] == "OK" else "warn"}"><b>Soát Opus:</b> {e(f["verdict_cuoi"])}' + (f' — {e(f["giai_thich"])}' if f["giai_thich"] else "") + f'</p>{fix}</div>')
    sheet.append({"item_id": iid, "pair_id": pid, "dap_an_vi": v["options"][v["target"]], "ghi_chu_may": f"nhiễu do LLM chọn; Opus: {f['verdict_cuoi']}", "verdict": "", "ghi_chu": ""})
H.append("</main></body></html>")
page = "\n".join(H)
assert "<script" not in page and "src=" not in page
out_html = ROOT / "results/review/mlqa_test_review.html"
others = {p.name.lower() for p in (ROOT / "results/review").iterdir() if p.name != out_html.name and p.name != "mlqa_test_review_sheet.csv"}
assert out_html.name.lower() not in others and "mlqa_test_review_sheet.csv" not in others
out_html.write_text(page, encoding="utf-8")
buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(sheet[0])); w.writeheader(); w.writerows(sheet)
(ROOT / "results/review/mlqa_test_review_sheet.csv").write_text("\ufeff" + buf.getvalue(), encoding="utf-8")
print(json.dumps({"fixes": len(fixes), "final": dict(cnt), "soi_lai": dict(kl), "bad_occ": bad_occ, "html_items": len(pick),
                  "html_flagged": sum(final[p]["verdict_cuoi"] != "OK" for p in pick)}, ensure_ascii=False))
