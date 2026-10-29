"""Trung vị độ dài state (token Qwen3, tokenizer bộ đếm của train_v0) trên các file mẫu của khảo sát train_v1
(results/train_v1_survey/samples/*.jsonl, ~300 dòng split train mỗi bộ, lấy qua dataset viewer). Kèm tỉ lệ dòng có chữ tiếng Việt
có dấu (để bắt mẫu lẫn tiếng Anh). Không in nội dung. Ra: results/train_v1_survey/sample_lengths.json"""
import json, re, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, model_revisions, read_jsonl, strip_diacritics  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
res = {}
for p in sorted((RESULTS / "train_v1_survey" / "samples").glob("*.jsonl")):
    rows = read_jsonl(p)
    st = [len(tok.encode(r.get("state") or "", add_special_tokens=False)) for r in rows]
    full = [len(tok.encode("\n".join([r.get("state") or "", r.get("question") or ""] + list(r.get("options") or [])), add_special_tokens=False)) for r in rows]
    vi = np.mean([strip_diacritics(" ".join([r.get("state") or "", r.get("question") or ""])) != " ".join([r.get("state") or "", r.get("question") or ""]) for r in rows])
    res[p.stem] = {"n": len(rows), "state_tokens_median": float(np.median(st)), "state_tokens_p90": float(np.percentile(st, 90)),
                   "row_tokens_median": float(np.median(full)), "share_rows_with_vi_diacritics": float(vi),
                   "has_question": float(np.mean([bool(r.get("question")) for r in rows])), "has_options": float(np.mean([bool(r.get("options")) for r in rows]))}
    print(p.stem, res[p.stem]["n"], res[p.stem]["state_tokens_median"], res[p.stem]["row_tokens_median"], round(res[p.stem]["share_rows_with_vi_diacritics"], 2))
(RESULTS / "train_v1_survey" / "sample_lengths.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
