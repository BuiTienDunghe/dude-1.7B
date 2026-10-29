"""Đầu vào cho đợt chạy thử 2 (results/bench_candidates/pilot2/). Không train, không đụng bench/test 1.1.
  1. Global-PIQA: 50 câu CÒN LẠI (ngoài 50 câu lô 1), đơn vị soát giống lô 1. KHÔNG chấm model trên các câu này.
  2. MLQA (dev = validation): ứng viên cho recast "4 cụm từ trả lời"; khác đoạn văn của lô 1, bỏ đoạn trùng ViWikiFC của lô 1.
     Một câu hỏi mỗi đoạn, seed 20261014, lấy 60 ứng viên để còn ~50 sau khi loại.
  3. 65 chuỗi mẫu tiếng Việt "chưa duyệt" (tools/review_app/inventory.py + lớp ghi đè hiện hành) và ngữ cảnh cho người soát.
"""
import ast
import csv
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tools" / "review_app"))
from common import read_jsonl, sha256_file, write_jsonl  # noqa: E402
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402

SEED = 20261014
P1, P2 = ROOT / "results/bench_candidates/pilot", ROOT / "results/bench_candidates/pilot2"
(P2 / "audit").mkdir(parents=True, exist_ok=True)
man1 = json.load(open(P1 / "manifest_pilot.json", encoding="utf-8"))

# ---------------------------------------------------------------- 1. Global-PIQA 50 câu còn lại
assert sha256_file(P1 / "raw/gpiqa/nonparallel_vie_latn.tsv") == man1["sources"]["gpiqa"]["sha256"]
gp = list(csv.DictReader(open(P1 / "raw/gpiqa/nonparallel_vie_latn.tsv", encoding="utf-8"), delimiter="\t"))
used = set(man1["sources"]["gpiqa"]["picked_index"])
rest = [i for i in range(len(gp)) if i not in used]
assert len(rest) == 50


def en_prefix(t0, t1):
    w0, w1 = t0.split(" "), t1.split(" ")
    k = 0
    while k < min(len(w0), len(w1)) - 1 and w0[k] == w1[k]:
        k += 1
    return " ".join(w0[:k]), " ".join(w0[k:]), " ".join(w1[k:])


units = []
for i in rest:
    r = gp[i]
    pre, o0, o1 = en_prefix(r["eng_translated0"].strip(), r["eng_translated1"].strip())
    q_en, q_vi = man1["question_templates"]["gpiqa"]["en"], man1["question_templates"]["gpiqa"]["vi"]
    lab = int(r["label"])
    vi_opts = [r["solution0"].strip(), r["solution1"].strip()]
    units.append({"split": "pilot2", "source": "gpiqa", "vi_state": r["prompt"].strip(), "en_state": pre,
                  "questions": [{"audit_id": f"gpiqa:{r['example_id']}", "kind": "plausible",
                                 "vi": {"question": q_vi, "options": vi_opts, "gold_index": lab, "gold_text": vi_opts[lab]},
                                 "en": {"question": q_en, "options": [o0, o1], "gold_index": lab, "gold_text": [o0, o1][lab]}}]})
if "--mlqa_b" not in sys.argv:
    write_jsonl(P2 / "audit/units_gpiqa_b.jsonl", units)

# ---------------------------------------------------------------- 2. MLQA ứng viên
vi_t = pq.read_table(P1 / "raw/mlqa/vi.vi.validation.parquet").to_pylist()
en_t = {x["id"]: x for x in pq.read_table(P1 / "raw/mlqa/en.en.validation.parquet").to_pylist()}
p1_rows = read_jsonl(P1 / "pilot_rows.jsonl")
p1_ctx = {r["state"] for r in p1_rows if r["source"] == "mlqa" and r["lang"] == "vi" and r["variant"] == "orig"}
p1_ctx_excluded = p1_ctx  # gồm cả đoạn trùng ViWikiFC (mlqa:b2e48158c172) của lô 1
ans = lambda x: ast.literal_eval(x["answers"]) if isinstance(x["answers"], str) else x["answers"]
by_ctx, drop = {}, {"lo_1": 0, "dai_qua_400_tu": 0, "span_khong_khop": 0}
for r in vi_t:
    if r["context"].strip() in p1_ctx_excluded:
        drop["lo_1"] += 1; continue
    e = en_t[r["id"]]
    if len(r["context"].split()) > 400 or len(e["context"].split()) > 400:
        drop["dai_qua_400_tu"] += 1; continue
    av, ae = ans(r), ans(e)
    if r["context"][av["answer_start"][0]:][:len(av["text"][0])] != av["text"][0] or \
            e["context"][ae["answer_start"][0]:][:len(ae["text"][0])] != ae["text"][0]:
        drop["span_khong_khop"] += 1; continue
    by_ctx.setdefault(r["context"], []).append(r["id"])
rng = random.Random(SEED)
cands = [rng.choice(sorted(v)) for _, v in sorted(by_ctx.items())]
rng.shuffle(cands)
if "--mlqa_b" in sys.argv:          # lô ứng viên bổ sung (cùng thứ tự seed), chia hai phần cho hai agent
    for name, sl in (("gen_in_mlqa_b1.jsonl", cands[60:130]), ("gen_in_mlqa_b2.jsonl", cands[130:200])):
        out = []
        for qid in sl:
            v, e = next(x for x in vi_t if x["id"] == qid), en_t[qid]
            out.append({"id": f"mlqa2:{qid[:12]}", "mlqa_id": qid, "vi": {"context": v["context"].strip(), "question": v["question"], "answer": ans(v)["text"][0]},
                        "en": {"context": e["context"].strip(), "question": e["question"], "answer": ans(e)["text"][0]}})
        write_jsonl(P2 / "audit" / name, out)
    print(json.dumps({"b1": len(cands[60:130]), "b2": len(cands[130:200]), "pool": len(cands)}))
    sys.exit(0)
cands = cands[:60]
gen = []
for qid in cands:
    v, e = next(x for x in vi_t if x["id"] == qid), en_t[qid]
    av, ae = ans(v), ans(e)
    gen.append({"id": f"mlqa2:{qid[:12]}", "mlqa_id": qid,
                "vi": {"context": v["context"].strip(), "question": v["question"], "answer": av["text"][0]},
                "en": {"context": e["context"].strip(), "question": e["question"], "answer": ae["text"][0]}})
write_jsonl(P2 / "audit/gen_in_mlqa.jsonl", gen)

# ---------------------------------------------------------------- 3. chuỗi mẫu chưa duyệt
inv = inventory(ROOT / "config")
st = E.derive_state(E.read_log(ROOT / "data/human_edits/edits_log.jsonl"), inv)
un = [x for x in inv if st["mau"][x["key"]]["trang_thai"] == "chưa duyệt"]
write_jsonl(P2 / "audit/templates_in.jsonl", [{"khoa": x["key"], "nhom": x["group"], "en": x["en"], "nhan_goc": x["label"], "vi_hien_hanh": st["mau"][x["key"]]["gia_tri"],
                                                 "dung_o": x["used_in"], "ghi_chu": x["note"]} for x in un])
dev = read_jsonl(ROOT / "data/bench_v1_1/dev_v1_1.jsonl")
ex = [next(r for r in dev if r["source"] == s and r["lang"] == l and r["variant"] == "orig") for s in ("massive", "sib200") for l in ("en", "vi")]
ctx = {"tat_ca_chuoi_cung_nhom": [{"khoa": x["key"], "nhom": x["group"], "en": x["en"], "vi": st["mau"][x["key"]]["gia_tri"], "trang_thai": st["mau"][x["key"]]["trang_thai"]}
                                  for x in inv if x["group"] in ("Mô tả ý định MASSIVE", "Tên chủ đề SIB-200", "Câu hỏi mẫu", "Có/Không")],
       "vi_du_dong_dev": [{k: r[k] for k in ("id", "lang", "state", "question", "options", "target")} for r in ex]}
(P2 / "audit/templates_context.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"gpiqa_b": len(units), "mlqa_candidates": len(gen), "mlqa_ctx_pool": len(by_ctx), "mlqa_drop": drop, "templates": len(un)}, ensure_ascii=False))
