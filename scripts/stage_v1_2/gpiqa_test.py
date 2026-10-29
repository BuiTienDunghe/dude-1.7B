"""Ứng viên test bản 1.2 (KHU TẠM): Global-PIQA vi, LÔ 2 (50 câu chưa chấm model nào). KHÔNG chấm. KHÔNG ghi vào data/, KHÔNG đóng băng.
- VI: state = prompt, lựa chọn = solution0/1 (nguyên văn người Việt viết, CHƯA sửa chính tả), câu hỏi = mẫu đề xuất pilot-gpiqa-question-vi (chưa duyệt).
- EN: dùng NGUYÊN câu dịch eng_translated0/1 làm lựa chọn, state EN rỗng (cờ en_state_rong; khung prompt sẽ hiện "Text:" trống),
  câu hỏi EN "Which of the two statements is more plausible?". Không tách phần đầu chung như đợt chạy thử (hỏng 5/100).
- Nhãn = nhãn gốc. 6 câu nhãn sai/mơ hồ theo soát Opus (2 lượt) gắn cờ cho_quyet_nhan + gợi ý của Opus; người làm dự án quyết.
- Sửa chính tả / lỗi phía EN của Opus: KHÔNG áp; ghi riêng vào gpiqa_lo2_fixes.jsonl để chuyển thành đề xuất theo câu khi bộ được chọn.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import CONFIG, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_diacritics, write_jsonl  # noqa: E402

P1 = ROOT / "results/bench_candidates/pilot"
OUT = ROOT / "results/bench_candidates/stage_v1_2"
OUT.mkdir(parents=True, exist_ok=True)
man1 = json.load(open(P1 / "manifest_pilot.json", encoding="utf-8"))
TSV = P1 / "raw/gpiqa/nonparallel_vie_latn.tsv"
assert sha256_file(TSV) == man1["sources"]["gpiqa"]["sha256"]
gp = list(csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"))
lo1 = set(man1["sources"]["gpiqa"]["picked_index"])
lo2 = [i for i in range(len(gp)) if i not in lo1]
issues = {x["audit_id"]: x for x in read_jsonl(ROOT / "results/bench_candidates/pilot2/gpiqa_issues_100.jsonl") if x["lo"] == 2}
assert len(lo2) == 50 and len(issues) == 50
Q = {"vi": man1["question_templates"]["gpiqa"]["vi"], "en": "Which of the two statements is more plausible?"}
rows, fixes = [], []
for i in lo2:
    r = gp[i]
    pid = f"gpiqa:{r['example_id']}"
    it = issues[pid]
    flags = [f"cultural={r['approx_cultural_score']}", f"llm_used={r['llm_used']}", "lo2_chua_cham"]
    extra = {}
    if it["verdict_cuoi"] in ("NHAN_NGUON_SAI_CHAC", "NHAN_NGUON_CO_THE_SAI", "MO_HO"):
        flags.append("cho_quyet_nhan")
        extra = {"soat_opus": {"ket_luan": it["verdict_cuoi"], "dap_an_theo_opus": it["dap_an_dung_theo_ban"], "giai_thich": it["giai_thich"]}}
    elif it["verdict_cuoi"] != "OK":
        flags.append("co_de_xuat_sua:" + it["verdict_cuoi"])
        fixes.append({"pair_id": pid, **{k: it[k] for k in ("verdict_cuoi", "ben_sai", "truong", "gia_tri_cu", "gia_tri_de_xuat", "giai_thich", "do_tin_cay")}})
    lab = int(r["label"])
    base = {"pair_id": pid, "source": "gpiqa", "kind": "plausible", "target": lab, "split_origin": "global-piqa-nonparallel:test",
            "n_tokens": None, "cluster": pid, "gold_label": None, "block": "stage_v1_2", "variant": "orig", **extra}
    rows.append({"id": f"{pid}:en", "lang": "en", "state": "", "question": Q["en"],
                 "options": [r["eng_translated0"].strip(), r["eng_translated1"].strip()], "flags": flags + ["en_state_rong"], **base})
    rows.append({"id": f"{pid}:vi", "lang": "vi", "state": r["prompt"].strip(), "question": Q["vi"],
                 "options": [r["solution0"].strip(), r["solution1"].strip()], "flags": flags, **base})
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
frames = load_json(CONFIG / "prompt_frames.json")
full = []
for r in rows:
    full.append(r)
    if r["lang"] == "vi":
        n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]), "question": strip_diacritics(r["question"]),
             "options": [strip_diacritics(x) for x in r["options"]]}
        for a, b in [(r["state"], n["state"]), (r["question"], n["question"])] + list(zip(r["options"], n["options"])):
            assert re.findall(r"\d+", a) == re.findall(r"\d+", b), r["id"]
        full.append(n)
for r in full:
    r["n_tokens"] = len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))
    r["n_state_tokens"] = len(tok.encode(r["state"], add_special_tokens=False))
write_jsonl(OUT / "gpiqa_test_candidates.jsonl", full)
write_jsonl(OUT / "gpiqa_lo2_fixes.jsonl", fixes)
cq = [r["pair_id"] for r in rows if r["lang"] == "vi" and "cho_quyet_nhan" in r["flags"]]
from collections import Counter  # noqa: E402
man = {"name": "stage_v1_2 Global-PIQA vi test candidates — lô 2 (KHU TẠM; chưa quyết; chưa chấm; chưa đóng băng)",
       "source": man1["sources"]["gpiqa"] | {"picked_index_lo2": lo2}, "license": "CC BY-SA 4.0; cấm dùng để train (README)",
       "rows": len(full), "pairs": len(rows) // 2, "cho_quyet_nhan": cq, "so_cau_co_de_xuat_sua": len(fixes),
       "fixes_by_kind": dict(Counter(f["verdict_cuoi"] for f in fixes)), "label_dist": dict(Counter(r["target"] for r in rows if r["lang"] == "vi")),
       "question_templates": Q, "question_vi_status": "đề xuất pilot-gpiqa-question-vi, CHƯA DUYỆT",
       "en": "nguyên câu dịch eng_translated0/1; state EN rỗng", "sha256": sha256_file(OUT / "gpiqa_test_candidates.jsonl"),
       "ghi_chu": "Lô 1 (50 câu) đã chấm thử ở đợt chạy thử: KHÔNG thuộc lát này. Không chấm model nào trên lô 2."}
(OUT / "manifest_gpiqa_test_candidates.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("rows", "pairs", "cho_quyet_nhan", "so_cau_co_de_xuat_sua", "fixes_by_kind", "label_dist")}, ensure_ascii=False))
