"""Cổng G2: so đường dữ liệu 200 bước đầu.
- strip_p 0 (code mới) với tham chiếu (code cũ): hash id dòng và hash token phải trùng ở mọi bước;
- strip_p 0,3 với tham chiếu: hash id dòng trùng ở mọi bước; hash token chỉ được khác ở những bước có dòng bị bỏ dấu.
Ra: results/ft_strip/gates/g2_compare.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, read_jsonl  # noqa: E402

G = RESULTS / "ft_strip" / "gates"
ref = read_jsonl(G / "ref_batches.jsonl")
p0 = read_jsonl(G / "dump_p0" / "batch_dump.jsonl")
p3 = read_jsonl(G / "dump_p0.3" / "batch_dump.jsonl")
assert len(ref) == len(p0) == len(p3) == 200
res = {
    "p0_vs_ref": {"ids_equal_steps": sum(a["ids_sha"] == b["ids_sha"] for a, b in zip(ref, p0)),
                  "tokens_equal_steps": sum(a["tokens_sha"] == b["tokens_sha"] for a, b in zip(ref, p0)), "steps": 200},
    "p03_vs_ref": {"ids_equal_steps": sum(a["ids_sha"] == b["ids_sha"] for a, b in zip(ref, p3)),
                   "steps_with_stripped_rows": sum(b["n_stripped"] > 0 for b in p3),
                   "tokens_differ_only_where_stripped": all((a["tokens_sha"] == b["tokens_sha"]) == (b["n_stripped"] == 0) for a, b in zip(ref, p3)),
                   "stripped_rows_200_steps": sum(b["n_stripped"] for b in p3)},
}
res["PASS"] = (res["p0_vs_ref"]["ids_equal_steps"] == 200 and res["p0_vs_ref"]["tokens_equal_steps"] == 200
               and res["p03_vs_ref"]["ids_equal_steps"] == 200 and res["p03_vs_ref"]["tokens_differ_only_where_stripped"])
(G / "g2_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("G2", json.dumps(res))
