"""Dịch ĐẦY ĐỦ phần B (người dùng chốt 20/10): CHUẨN BỊ (chỉ thư viện chuẩn, không dịch).
- 200 dòng dịch thử (pilot/ban_s2_200.jsonl): giữ nếu mọi chuỗi trùng khóa bảng thuật ngữ v2 (câu hỏi, lựa chọn, nhãn trích trong ngoặc kép)
  đã dùng đúng bản dịch v2; ngược lại là "bị ảnh hưởng" (do đổi nhãn kiểm chứng theo bench, 5 nhãn clinc, hoặc nhãn banking77 đổi khi soạn v2) -> dịch lại.
- Dòng cần dịch = 6.944 - dòng pilot giữ lại; chia lô 40 dòng theo thứ tự file -> full/lo/lo_{NNN}_en.jsonl; đợt = 4 lô.
Ra: results/train_v1/translated_sonnet/full/plan.json, lo/*.jsonl, pilot_giu.jsonl (bản s2 dùng lại), pilot_anh_huong.json."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
P = T / "translated_sonnet/pilot"
O = T / "translated_sonnet/full"
(O / "lo").mkdir(parents=True, exist_ok=True)
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
LOT, PER_DOT = 40, 4
src = rd(T / "to_translate/part_b_to_translate.jsonl")
g = json.load(open(T / "to_translate/bang_thuat_ngu_v2.json", encoding="utf-8"))["thuat_ngu"]
pen = {r["id"]: r for r in rd(P / "src_200.jsonl")}
ps2 = {r["id"]: r for r in rd(P / "ban_s2_200.jsonl")}
keep, hit = [], {}
for i, e in pen.items():
    v = ps2[i]
    why = []
    if e["question"] in g and v["question"].strip() != g[e["question"]]["vi"].strip():
        why.append(f"question: {e['question'][:60]}")
    for lab in re.findall(r'"([^"]{1,80})"', e["question"]):
        if lab in g and f'"{g[lab]["vi"]}"' not in v["question"]:
            why.append(f"nhãn trích: {lab}")
    for x, y in zip(e["options"], v["options"]):
        if x in g and y.strip() != g[x]["vi"].strip():
            why.append(f"option: {x}")
    if why:
        hit[i] = why
    else:
        keep.append(v)
todo = [r for r in src if r["id"] not in {k["id"] for k in keep}]
lots = []
for k in range(0, len(todo), LOT):
    name = f"{k // LOT + 1:03d}"
    with open(O / f"lo/lo_{name}_en.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in todo[k:k + LOT]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lots.append(name)
with open(O / "pilot_giu.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(O / "pilot_anh_huong.json").write_text(json.dumps(hit, ensure_ascii=False, indent=1), encoding="utf-8")
dots = [lots[k:k + PER_DOT] for k in range(0, len(lots), PER_DOT)]
plan = {"tong_dong": len(src), "pilot_giu": len(keep), "pilot_dich_lai": len(hit), "can_dich": len(todo), "so_lo": len(lots), "dong_moi_lo": LOT,
        "lo_moi_dot": PER_DOT, "so_dot": len(dots), "dot": {f"{d + 1:02d}": v for d, v in enumerate(dots)},
        "ly_do_dich_lai": dict(Counter(w.split(":")[0] for v in hit.values() for w in v))}
(O / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in plan.items() if k != "dot"}, ensure_ascii=False))
