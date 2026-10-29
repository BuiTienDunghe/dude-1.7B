"""Kiểm hàm bỏ dấu sau khi chuyển sang common.py: dựng lại các dòng không dấu của test 1.0 và dev_v1 từ các dòng có dấu
(đúng quy trình add_nodiac của build_bench_v1.py), ghi vào thư mục tạm, so SHA-256 với file đã đóng băng / manifest.
Chỉ đọc file test để so hash; không model nào đọc nó; không ghi gì vào data/bench_v1.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_diacritics, write_jsonl  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
frames = load_json(CONFIG / "prompt_frames.json")
TMP = RESULTS / "ft_strip" / "tmp_regen"; TMP.mkdir(parents=True, exist_ok=True)
res = {}
for name, path, manifest, key in (("test_v1.0", DATA / "bench_v1/test_v1.0.jsonl", DATA / "bench_v1/manifest_test_v1.0.json", "test_v1.0.jsonl"),
                                  ("dev_v1", DATA / "bench_v1/dev_v1.jsonl", DATA / "bench_v1/manifest_dev_v1.json", "dev_v1.jsonl")):
    rows = read_jsonl(path)
    out = []
    for r in rows:
        if r["variant"] == "nodiac":
            continue
        out.append(r)
        if r["lang"] == "vi":
            n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]),
                 "question": strip_diacritics(r["question"]), "options": [strip_diacritics(o) for o in r["options"]]}
            n["n_tokens"] = len(tok.encode(render_prompt(n, tok, frames), add_special_tokens=False))
            n["n_state_tokens"] = len(tok.encode(n["state"], add_special_tokens=False))
            out.append(n)
    tmp = TMP / f"{name}.jsonl"
    write_jsonl(tmp, out)
    res[name] = {"rows": len(out), "regenerated_sha256": sha256_file(tmp), "frozen_file_sha256": sha256_file(path),
                 "manifest_sha256": load_json(manifest)["sha256"][key]}
    res[name]["identical"] = res[name]["regenerated_sha256"] == res[name]["frozen_file_sha256"] == res[name]["manifest_sha256"]
(RESULTS / "ft_strip" / "regen_nodiac_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
