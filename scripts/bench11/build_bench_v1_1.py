"""Dựng bench bản 1.1 từ bản 1.0 (không lấy mẫu lại): bỏ ViHSD (mọi dòng), áp câu chữ mới đã duyệt ngày 2026-10-13.

- Giữ nguyên câu đã lấy mẫu, id, thứ tự lựa chọn (seed xáo cũ) và nhãn gây nhiễu của MASSIVE. Lựa chọn được đổi theo KHÓA nhãn:
  chữ cũ (config/archive_v1_0/*.json) -> khóa (ý định MASSIVE / nhãn kiểm chứng) -> chữ mới (config/*.json). Câu hỏi kiểm chứng:
  tách nhận định khỏi mẫu cũ rồi ghép vào mẫu mới.
- Bản không dấu sinh lại bằng đúng phép của build_bench_v1.add_nodiac (common.strip_diacritics trên state, câu hỏi, lựa chọn).
  Dòng không đổi chữ: bản không dấu sinh lại phải trùng từng trường với bản 1.0.
- n_tokens / n_state_tokens: giữ nguyên cho dòng không đổi; tính lại (render_prompt, tokenizer Qwen3-0.6B như bản 1.0) cho dòng đổi chữ.
- Đọc test_v1.0.jsonl bằng script (chỉ đọc, không in nội dung, không chạy model). Ghi test_v1.1.jsonl, đóng băng (SHA-256 + chỉ đọc).
- Kiểm rò rỉ lại với train_v0 + val_v0 (scripts/decontam.Index) và giữa dev/test.
Ra: data/bench_v1_1/{dev_v1_1.jsonl, test_v1.1.jsonl, manifest_dev_v1_1.json, manifest_test_v1.1.json, leakage_v1_1.json, FROZEN_test_v1.1.sha256}
"""
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, SEED, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_diacritics, write_jsonl  # noqa: E402
from decontam import Index  # noqa: E402

OLD_Q, OLD_T = load_json(CONFIG / "archive_v1_0/bench_v1_questions.json"), load_json(CONFIG / "archive_v1_0/translations.json")
NEW_Q, NEW_T = load_json(CONFIG / "bench_v1_questions.json"), load_json(CONFIG / "translations.json")
IN, OUT = DATA / "bench_v1", DATA / "bench_v1_1"
man_dev0, man_test0 = load_json(IN / "manifest_dev_v1.json"), load_json(IN / "manifest_test_v1.0.json")
assert sha256_file(IN / "dev_v1.jsonl") == man_dev0["sha256"]["dev_v1.jsonl"]
assert sha256_file(IN / "test_v1.0.jsonl") == man_test0["sha256"]["test_v1.0.jsonl"]
OUT.mkdir(parents=True, exist_ok=True)

INTENT = {lang: {v[lang]: k for k, v in OLD_T["massive_intents"].items()} for lang in ("en", "vi")}
FC_KEY = {v: k for k, v in OLD_Q["factcheck_choice"]["options"].items()}
PFX = {"choice": OLD_Q["factcheck_choice"]["question"].split("{claim}")[0], "noul": OLD_Q["factcheck_noul"]["question"].split("{claim}")[0]}
for k in PFX:
    assert OLD_Q[f"factcheck_{k}"]["question"].endswith("{claim}")


def transform(r):
    """-> (dòng mới, đổi chữ hay không). Chỉ dòng variant orig."""
    s = r["source"]
    if s == "massive":
        opts = [NEW_T["massive_intents"][INTENT[r["lang"]][o]][r["lang"]] for o in r["options"]]
        assert NEW_T["massive_intents"][r["gold_label"]][r["lang"]] == opts[r["target"]]
        return {**r, "options": opts}, opts != r["options"]
    if s in ("vifactcheck", "viwikifc"):
        k = r["kind"]
        assert r["question"].startswith(PFX[k]), r["id"]
        claim = r["question"][len(PFX[k]):]
        q = NEW_Q[f"factcheck_{k}"]["question"].format(claim=claim)
        opts = [NEW_Q["factcheck_choice"]["options"][FC_KEY[o]] for o in r["options"]] if k == "choice" else list(r["options"])
        if k == "choice":
            assert FC_KEY[r["options"][r["target"]]] == r["gold_label"]
        return {**r, "question": q, "options": opts}, (q != r["question"] or opts != r["options"])
    return dict(r), False


def nodiac(r):   # đúng phép build_bench_v1.add_nodiac
    n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]),
         "question": strip_diacritics(r["question"]), "options": [strip_diacritics(o) for o in r["options"]]}
    for a, b in ((r["state"], n["state"]), (r["question"], n["question"])) + tuple(zip(r["options"], n["options"])):
        assert re.findall(r"\d+", a) == re.findall(r"\d+", b), r["id"]
    return n


from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
frames = load_json(CONFIG / "prompt_frames.json")
ntok = lambda r: len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))


def build(rows):
    old = {r["id"]: r for r in rows}
    out, stats = [], Counter()
    for r in rows:
        if r["variant"] != "orig":
            continue
        if r["source"] == "vihsd":
            stats["dropped_vihsd_orig"] += 1
            continue
        n, changed = transform(r)
        if changed:
            n["n_tokens"] = ntok(n); stats[f"changed_orig|{r['source']}|{r['lang']}"] += 1
        else:
            assert n == r
            if stats["ntok_rechecked"] < 200:           # kiểm: tính lại n_tokens cho dòng không đổi ra đúng số cũ
                assert ntok(n) == r["n_tokens"], r["id"]; stats["ntok_rechecked"] += 1
        out.append(n)
        if n["lang"] == "vi":
            d = nodiac(n)
            d["n_tokens"] = ntok(d) if changed else old[r["id"] + ":nodiac"]["n_tokens"]
            d["n_state_tokens"] = old[r["id"] + ":nodiac"]["n_state_tokens"]       # state không đổi
            if not changed:
                assert d == old[r["id"] + ":nodiac"], r["id"]          # bản không dấu sinh lại trùng từng trường với bản 1.0
                stats["nodiac_identical_to_v1_0"] += 1
            else:
                stats[f"changed_nodiac|{r['source']}"] += 1
            out.append(d)
    stats["dropped_vihsd_total"] = sum(1 for r in rows if r["source"] == "vihsd")
    assert len({r["id"] for r in out}) == len(out)
    return out, dict(stats)


dev, st_dev = build(read_jsonl(IN / "dev_v1.jsonl"))
test, st_test = build(read_jsonl(IN / "test_v1.0.jsonl"))

# chống rò rỉ (như build_bench_v1)
leak = {}
ti, di = Index(), Index()
for r in test:
    if r["variant"] == "orig":
        ti.add(r["state"], f"test:{r['source']}")
for r in dev:
    if r["variant"] == "orig":
        di.add(r["state"], f"dev:{r['source']}")
leak["dev_states_hitting_test"] = sum(bool(ti.check(r["state"])) for r in dev if r["variant"] == "orig")
leak["test_states_hitting_dev"] = sum(bool(di.check(r["state"])) for r in test if r["variant"] == "orig")
for name, idx in (("test", ti), ("dev_v1_1", di)):
    hits = Counter()
    for t in read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl"):
        h = idx.check(t["state"])
        if h:
            hits[f"{h[0]}|{h[1]}|train_source={t['source']}"] += 1
    leak[f"train_v0_vs_{name}"] = {"total": sum(hits.values()), "by_kind_target_trainsource": dict(hits.most_common())}
assert leak["dev_states_hitting_test"] == 0 and leak["test_states_hitting_dev"] == 0, leak

dp, tp = OUT / "dev_v1_1.jsonl", OUT / "test_v1.1.jsonl"
for p in (tp,):
    if p.exists():
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
write_jsonl(dp, dev); write_jsonl(tp, test)
(OUT / "leakage_v1_1.json").write_text(json.dumps(leak, ensure_ascii=False, indent=1), encoding="utf-8")


def counts(rows_):
    return {"rows": len(rows_), "by_source": dict(Counter(r["source"] for r in rows_)),
            "by_source_lang_variant": dict(Counter(f"{r['source']}|{r['lang']}|{r['variant']}" for r in rows_)),
            "by_kind": dict(Counter(f"{r['source']}|{r['kind']}" for r in rows_ if r["variant"] == "orig")),
            "target_hist_orig": dict(Counter(f"{r['source']}|{r['kind']}|{r['target']}" for r in rows_ if r["variant"] == "orig" and r["lang"] == "vi")),
            "gold_label_hist_native": dict(Counter(f"{r['source']}|{r['gold_label']}" for r in rows_ if r["block"] == "native" and r["variant"] == "orig"))}


ga = load_json(RESULTS / "dev_v1_1/gate_a.json")
common_meta = {
    "version": "1.1", "built_from": {"dev": f"data/bench_v1/dev_v1.jsonl ({man_dev0['sha256']['dev_v1.jsonl']})",
                                     "test": f"data/bench_v1/test_v1.0.jsonl ({man_test0['sha256']['test_v1.0.jsonl']})"},
    "seed": SEED, "revisions": {k: v for k, v in man_dev0["revisions"].items() if k != "uitnlp/vihsd"},
    "licenses": {k: v for k, v in man_dev0["licenses"].items() if k != "vihsd"},
    "changes_vs_1_0": ["Bỏ ViHSD khỏi dev và test (mọi dòng, mọi biến thể).",
                       "Kiểm chứng (ViFactCheck, ViWikiFC): 'hỗ trợ' -> 'xác nhận' trong câu hỏi choice, lựa chọn 'Được văn bản xác nhận', câu hỏi noul.",
                       "MASSIVE: đổi mô tả 8 ý định (music_settings, music_dislikeness, music_likeness, transport_query, transport_traffic, "
                       "transport_ticket [chỉ tiếng Việt], general_greet, general_quirky), cả lựa chọn đúng lẫn lựa chọn gây nhiễu.",
                       "Không lấy mẫu lại; giữ id, thứ tự lựa chọn, nhãn gây nhiễu. Bản không dấu sinh lại bằng cùng hàm."],
    "config": {"bench_v1_questions.json": sha256_file(CONFIG / "bench_v1_questions.json"), "translations.json": sha256_file(CONFIG / "translations.json"),
               "previous_configs": "config/archive_v1_0/ (bản dùng cho bench 1.0)",
               "reviewed_items": "factcheck_choice, factcheck_noul, 8 ý định MASSIVE đã liệt kê: người làm dự án đã duyệt ngày 2026-10-13; mục khác giữ trạng thái chưa duyệt"},
    "notes": ["Không có kiểu câu score (score chỉ có ở ViHSD, đã bỏ).",
              "Khối bản địa chỉ gồm kiểm chứng (ViFactCheck, ViWikiFC): choice 3 lựa chọn và noul Có/Không.",
              "SIB-200 có trần: nhãn theo chủ đề của bài gốc (FLORES), một câu lẻ có thể không đủ để suy ra chủ đề.",
              "Cổng A (results/dev_v1_1/gate_a.json): đáp án dev_v1 khớp trường nhãn nguồn 100% (Belebele 600/600, MASSIVE 600/600; "
              "SIB-200 600/600, XCOPA 100/100). Hai câu nghi lỗi nguồn giữ nguyên trong dữ liệu: "
              "belebele:…/High_School_Earth_Science/Telescopes#2 và massive:10457 (đáp án khớp nhãn nguồn)."],
    "gate_a_pass": ga["GATE_PASS"], "n_tokens": "rendered prompt, tokenizer Qwen/Qwen3-0.6B (tính lại cho dòng đổi chữ)",
    "nodiac": "strip combining marks (NFD) and đ/Đ→d/D on state, question, options; prompt frame words unchanged",
    "leakage": {k: (v if not isinstance(v, dict) or "total" not in v else v["total"]) for k, v in leak.items()}}
(OUT / "manifest_dev_v1_1.json").write_text(json.dumps({"name": "dev_v1_1", "sha256": {dp.name: sha256_file(dp)}, "counts": counts(dev),
                                                        "build_stats": st_dev, **common_meta}, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "manifest_test_v1.1.json").write_text(json.dumps({"name": "bench_v1 test", "frozen": True, "sha256": {tp.name: sha256_file(tp)},
                                                         "counts": counts(test), "build_stats": st_test, **common_meta,
                                                         "note": "Không sửa file này sau khi đóng băng. Không chạy model nào trên tập này. Bản 1.0 giữ ở data/bench_v1/ làm lưu trữ."},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "FROZEN_test_v1.1.sha256").write_text(f"{sha256_file(tp)}  test_v1.1.jsonl\n", encoding="utf-8")
os.chmod(tp, stat.S_IREAD)
print(json.dumps({"dev": counts(dev)["by_source"], "test": counts(test)["by_source"], "dev_sha": sha256_file(dp), "test_sha": sha256_file(tp),
                  "build_stats_dev": st_dev, "build_stats_test": {k: v for k, v in st_test.items()}, "leak": common_meta["leakage"]}, ensure_ascii=False))
