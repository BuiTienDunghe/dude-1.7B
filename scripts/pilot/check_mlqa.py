"""Kiểm máy móc cho MLQA chạy thử: (1) ranh giới câu dính liền (dấu chấm + chữ hoa không có khoảng trắng); (2) chuỗi đáp án xuất hiện ở
nhiều câu (answer_start có thể trỏ nhầm câu). Chỉ in số đếm và id."""
import ast, json, re, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, "/mnt/c/Users/dungbt06/Fine-tune/scripts/pilot")
P = Path("/mnt/c/Users/dungbt06/Fine-tune/results/bench_candidates/pilot")
src = (P / "../../../scripts/pilot/build_pilot.py").resolve()
ns = {}
code = src.read_text(encoding="utf-8")
ns = {"re": re}; exec(code[code.index("ABBR = "):code.index("def mlqa_item")], ns)   # chỉ lấy hàm tách câu
split_sents = ns["split_sents"]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(P / "pilot_rows.jsonl", encoding="utf-8")}
glued = re.compile(r"[a-zà-ỹđ0-9\)][.!?][A-ZÀ-ỸĐ]")
res = {}
for lang in ("vi", "en"):
    t = {x["id"]: x for x in pq.read_table(P / f"raw/mlqa/{lang}.{lang}.validation.parquet").to_pylist()}
    pids = sorted({r["pair_id"] for r in rows.values() if r["source"] == "mlqa"})
    g, multi, multi_ids = 0, 0, []
    for pid in pids:
        qid = next(k for k in t if k.startswith(pid.split(":")[1]))
        x = t[qid]; ctx = x["context"]; ans = ast.literal_eval(x["answers"]) if isinstance(x["answers"], str) else x["answers"]
        a = ans["text"][0]
        if glued.search(ctx):
            g += 1
        sp = split_sents(ctx)
        occ = [m.start() for m in re.finditer(re.escape(a), ctx)]
        sents = {i for o in occ for i, (s, e) in enumerate(sp) if s <= o and o + len(a) <= e}
        if len(sents) > 1:
            multi += 1; multi_ids.append(pid)
    res[lang] = {"n": len(pids), "doan_co_ranh_gioi_dinh_lien": g, "dap_an_xuat_hien_o_nhieu_cau": multi, "ids": multi_ids}
print(json.dumps(res, ensure_ascii=False))
