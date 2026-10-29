"""Cổng A (bench 1.1): đáp án đúng của dev_v1 có khớp trường nhãn của dữ liệu gốc không — toàn bộ Belebele và MASSIVE
(và thêm SIB-200, XCOPA để tham khảo). Chỉ đọc dev_v1 và dữ liệu nguồn (revision đã ghim, cache); không đụng test.
Hai câu nghi lỗi: P4-belebele-003 (…/High_School_Earth_Science/Telescopes#2) và P4-massive-001 (massive:10457).
In tối đa vài dòng, cắt 200 ký tự. Ra: results/dev_v1_1/gate_a.json
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, load_json, read_jsonl  # noqa: E402

REV = {"facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", "mteb/amazon_massive_intent": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
       "Davlan/sib200": "38977a667f6fc264d5c26ec57a01e16db040b358", "cambridgeltl/xcopa": "042f78955ba48e6404616762fa6e05e839c3907a"}
load = lambda r, c, s: load_dataset(r, c, revision=REV[r], split=s).to_pandas()
tr = load_json(CONFIG / "translations.json")
dev = read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")
OUT = RESULTS / "dev_v1_1"; OUT.mkdir(parents=True, exist_ok=True)
res = {}

# Belebele: trường nhãn nguồn = correct_answer_num (1..4) trỏ vào mc_answer1..4
bv, be = load("facebook/belebele", "vie_Latn", "test"), load("facebook/belebele", "eng_Latn", "test")
B = {}
for lang, df in (("vi", bv), ("en", be)):
    for x in df.itertuples(index=False):
        B[(lang, f"belebele:{x.link}#{x.question_number}")] = x
ok = bad = 0; bad_ids = []
for r in dev:
    if r["source"] != "belebele" or r["variant"] != "orig":
        continue
    x = B[(r["lang"], r["pair_id"])]
    src_opts = [getattr(x, f"mc_answer{i}") for i in range(1, 5)]
    gold_src = src_opts[int(x.correct_answer_num) - 1]
    good = sorted(r["options"]) == sorted(src_opts) and r["options"][r["target"]] == gold_src and r["state"] == x.flores_passage and r["question"] == x.question
    ok += good; bad += not good
    if not good:
        bad_ids.append(r["id"])
res["belebele"] = {"rows_checked": ok + bad, "match": ok, "mismatch": bad, "mismatch_ids": bad_ids[:20]}

# MASSIVE: trường nhãn nguồn = label (tên ý định); lựa chọn đúng = mô tả của ý định đó trong translations.json (bản 1.0)
mv, me = load("mteb/amazon_massive_intent", "vi", "validation"), load("mteb/amazon_massive_intent", "en", "validation")
Mm = {}
for lang, df in (("vi", mv), ("en", me)):
    for x in df.itertuples(index=False):
        Mm[(lang, f"massive:{int(x.id)}")] = x
ok = bad = 0; bad_ids = []
for r in dev:
    if r["source"] != "massive" or r["variant"] != "orig":
        continue
    x = Mm[(r["lang"], r["pair_id"])]
    want = tr["massive_intents"][x.label][r["lang"]]
    dis = next(f for f in r["flags"] if f.startswith("distractors="))[len("distractors="):].split(",")
    good = (r["gold_label"] == x.label and r["options"][r["target"]] == want and r["state"] == x.text
            and sorted(r["options"]) == sorted(tr["massive_intents"][l][r["lang"]] for l in [x.label] + dis))
    ok += good; bad += not good
    if not good:
        bad_ids.append(r["id"])
res["massive"] = {"rows_checked": ok + bad, "match": ok, "mismatch": bad, "mismatch_ids": bad_ids[:20]}

# SIB-200 (tham khảo): category; XCOPA (tham khảo): label chỉ vào choice1/choice2
topics = tr["sib200_topics"]
S = {}
for sp in ("validation", "train"):
    for lang, cfg in (("vi", "vie_Latn"), ("en", "eng_Latn")):
        for x in load("Davlan/sib200", cfg, sp).itertuples(index=False):
            S[(lang, f"sib200:{x.index_id}")] = x
ok = bad = 0
for r in dev:
    if r["source"] == "sib200" and r["variant"] == "orig":
        x = S[(r["lang"], r["pair_id"])]
        good = r["gold_label"] == x.category and r["options"][r["target"]] == topics[x.category][r["lang"]] and r["state"] == x.text
        ok += good; bad += not good
res["sib200_reference"] = {"rows_checked": ok + bad, "match": ok, "mismatch": bad}
xv = load("cambridgeltl/xcopa", "vi", "validation")
X = {f"xcopa:{int(x.idx)}": x for x in xv.itertuples(index=False)}
ok = bad = 0
for r in dev:
    if r["source"] == "xcopa" and r["variant"] == "orig" and r["lang"] == "vi":
        x = X.get(r["pair_id"])
        if x is None:
            bad += 1; continue
        good = r["options"][r["target"]] == (x.choice1 if int(x.label) == 0 else x.choice2) and r["state"] == x.premise
        ok += good; bad += not good
res["xcopa_vi_reference"] = {"rows_checked": ok + bad, "match": ok, "mismatch": bad}


# hai câu nghi lỗi
def detail(pid, lang):
    r = next(r for r in dev if r["pair_id"] == pid and r["lang"] == lang and r["variant"] == "orig")
    out = {"dev_target_text": r["options"][r["target"]], "dev_target_letter": "ABCDEFGHIJ"[r["target"]]}
    if pid.startswith("belebele"):
        x = B[(lang, pid)]
        out.update(source_correct_answer_num=int(x.correct_answer_num), source_answer_text=getattr(x, f"mc_answer{int(x.correct_answer_num)}"))
    else:
        x = Mm[(lang, pid)]
        out.update(source_label=x.label, source_label_description=tr["massive_intents"][x.label][lang])
    out["match"] = out["dev_target_text"] == out.get("source_answer_text", out.get("source_label_description"))
    return out


res["suspects"] = {"P4-belebele-003": {lang: detail("belebele:https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#2", lang) for lang in ("vi", "en")},
                   "P4-massive-001": {lang: detail("massive:10457", lang) for lang in ("vi", "en")}}
res["GATE_PASS"] = res["belebele"]["mismatch"] == 0 and res["massive"]["mismatch"] == 0
(OUT / "gate_a.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("belebele", "massive", "sib200_reference", "xcopa_vi_reference", "GATE_PASS")}, ensure_ascii=False)[:900])
for k, v in res["suspects"].items():
    print(k, json.dumps(v["vi"], ensure_ascii=False)[:200])
