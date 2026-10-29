"""Phần 5a: dựng tập test bench_v1.0 (đóng băng) và dev_v1.

Test 1.0:
  - song song Anh–Việt (người dịch): Belebele (244 đoạn để dành), SIB-200 test (cả 204), MASSIVE test, XCOPA test (vi) +
    COPA test (en, ghép theo idx; nhãn lấy từ XCOPA) — recast đúng như dev_v0 (cùng translations.json, cùng cách xáo);
  - tiếng Việt bản địa: ViFactCheck test, ViWikiFC test (ViHSD: bộ có cổng truy cập, chưa được cấp quyền → không có).
dev_v1:
  - phần song song = nguyên dev_v0 (không dựng lại; chỉ thêm trường block/variant);
  - phần bản địa = 300 câu mỗi nguồn từ split dev, đã loại mọi câu có văn bản trùng/gần trùng/chứa đoạn với split test.
Khối bản địa: mỗi dòng nguồn sinh đúng một câu hỏi, loại (choice / noul) chia theo seed từng dòng; choice 3 lựa chọn xáo theo
dòng; noul Có/Không, đúng khi nhãn gốc là "được hỗ trợ". Mọi dòng tiếng Việt có thêm bản không dấu (variant = "nodiac").
Ra: data/bench_v1/{test_v1.0.jsonl, dev_v1.jsonl, manifest_test_v1.0.json, manifest_dev_v1.json, leakage_v1.json}
"""
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, SEED, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, write_jsonl  # noqa: E402
from decontam import Index, normalize  # noqa: E402

OUT = DATA / "bench_v1"; OUT.mkdir(parents=True, exist_ok=True)
REV = {
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    "Davlan/sib200": "38977a667f6fc264d5c26ec57a01e16db040b358",
    "mteb/amazon_massive_intent": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
    "cambridgeltl/xcopa": "042f78955ba48e6404616762fa6e05e839c3907a",
    "aps/super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    "tranthaihoa/vifactcheck": "2df623632299a3459ad95feeff9ac67d7205a42a",
    "High-Will/ViWikiFC": "0089cbaac1a58458893b11c5cc067c4a37843f8c",
    "uitnlp/vihsd": "88e81b36ca376867640dad9df295e0f7d499ab80",
}
LICENSE = {"belebele": "cc-by-sa-4.0", "sib200": "cc-by-sa-4.0", "massive": "cc-by-4.0 (mirror card apache-2.0)",
           "xcopa": "cc-by-4.0 (vi); COPA en: BSD 2-Clause", "vifactcheck": "mit", "viwikifc": "cc-by-4.0",
           "vihsd": "mit (gated on HF; access granted to the project owner's account)"}
N = 300
tr = load_json(CONFIG / "translations.json")
qcfg = load_json(CONFIG / "bench_v1_questions.json")
fmt = load_json(CONFIG / "train_format.json")
rng0 = np.random.default_rng(SEED + 501)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
frames = load_json(CONFIG / "prompt_frames.json")
log = defaultdict(Counter)


def load(repo, cfg, split):
    return load_dataset(repo, cfg, revision=REV[repo], split=split).to_pandas()


def pair_rng(source, pair_id):
    h = hashlib.sha256(f"{SEED}:{source}:{pair_id}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def balanced_sample(keys, labels, n, rng):
    groups = defaultdict(list)
    for k, l in zip(keys, labels):
        groups[l].append(k)
    order = {l: list(rng.permutation(len(g))) for l, g in sorted(groups.items())}
    out, depth = [], 0
    while len(out) < n:
        progressed = False
        for l in sorted(groups):
            if depth < len(groups[l]) and len(out) < n:
                out.append(groups[l][order[l][depth]]); progressed = True
        if not progressed:
            break
        depth += 1
    return out


from common import strip_diacritics  # noqa: E402  (hàm đã chuyển sang common.py, thân hàm giữ nguyên)


def row(pid, source, lang, kind, block, state, question, options, target, origin, cluster, gold_label=None, flags=None):
    return {"id": f"{pid}:{lang}", "pair_id": pid, "source": source, "lang": lang, "kind": kind, "block": block, "variant": "orig",
            "state": state, "question": question, "options": options, "target": int(target), "split_origin": origin,
            "n_tokens": None, "cluster": cluster, "gold_label": gold_label, "flags": flags or []}


def state_cluster(source, state):
    return f"{source}:" + hashlib.sha1(normalize(state).encode()).hexdigest()[:12]


# ===================================================================== khối bản địa: nạp, ánh xạ nhãn
FC_LAB = {0: "supported", 1: "refuted", 2: "nei"}
WK_LAB = {"Supports": "supported", "Refutes": "refuted", "Not_Enough_Information": "nei"}
HSD_LAB = {0: "clean", 1: "offensive", 2: "hate"}
HSD_ORDER = ["clean", "offensive", "hate"]


def native_frame(source, split):
    items = _native_frame(source, split)
    bad = [it for it in items if not isinstance(it["state"], str) or not it["state"].strip() or not isinstance(it["claim"], str)]
    log["native_dropped_empty_state_or_claim"][f"{source}:{split}"] += len(bad)
    return [it for it in items if it not in bad]


def _native_frame(source, split):
    if source == "vifactcheck":
        d = load("tranthaihoa/vifactcheck", None, split)
        return [{"key": f"{int(r['index'])}-{int(r['annotation_id'])}", "state": r["Context"], "claim": r["Statement"],
                 "label": FC_LAB[int(r["labels"])], "origin": f"vifactcheck:{split}"} for _, r in d.iterrows()]
    if source == "vihsd":
        d = load_dataset("uitnlp/vihsd", revision=REV["uitnlp/vihsd"], data_files={split: f"{split}.csv"}, split=split).to_pandas()
        return [{"key": f"{split}-{i}", "state": r["free_text"], "claim": "", "label": HSD_LAB[int(r["label_id"])],
                 "origin": f"vihsd:{split}"} for i, r in d.iterrows()
                if isinstance(r["free_text"], str) and r["free_text"].strip() not in ("", "#ERROR!")]
    d = load("High-Will/ViWikiFC", None, split)
    # pairID có lặp lại với nhận định khác nhau: khóa = pairID + băm nội dung nhận định
    return [{"key": r["pairID"] + "~" + hashlib.sha1(r["claim"].encode()).hexdigest()[:8], "state": r["context"], "claim": r["claim"],
             "label": WK_LAB[r["gold_label"]], "origin": f"viwikifc:{split}"} for _, r in d.iterrows()]


def native_rows(source, items, split_tag):
    out = []
    for it in items:
        pid = f"{source}:{split_tag}:{it['key']}"
        rng = pair_rng(source, pid)
        is_choice = rng.random() < 0.5
        if source == "vihsd":
            if is_choice:   # nửa này là score ba mức theo thứ tự, không xáo
                kind, opts, target = "score", list(qcfg["vihsd_score"]["levels"]), HSD_ORDER.index(it["label"])
                q = qcfg["vihsd_score"]["question"]
            else:
                kind, opts, target = "noul", list(fmt["noul_options"]["vi"]), 0 if it["label"] != "clean" else 1
                q = qcfg["vihsd_noul"]["question"]
            out.append(row(pid, source, "vi", kind, "native", it["state"], q, opts, target, it["origin"],
                           state_cluster(source, it["state"]), gold_label=it["label"]))
            continue
        if is_choice:
            labs = ["supported", "refuted", "nei"]
            perm = list(rng.permutation(3))
            opts = [qcfg["factcheck_choice"]["options"][labs[p]] for p in perm]
            target = perm.index(labs.index(it["label"]))
            q = qcfg["factcheck_choice"]["question"].format(claim=it["claim"])
            kind = "choice"
        else:
            opts = list(fmt["noul_options"]["vi"])
            target = 0 if it["label"] == "supported" else 1
            q = qcfg["factcheck_noul"]["question"].format(claim=it["claim"])
            kind = "noul"
        out.append(row(pid, source, "vi", kind, "native", it["state"], q, opts, target, it["origin"],
                       state_cluster(source, it["state"]), gold_label=it["label"]))
    return out


# ===================================================================== dev_v1
dev_v0 = read_jsonl(DATA / "dev_v0.jsonl")
dev = [{**r, "block": "parallel", "variant": "orig"} for r in dev_v0]
dev_native = []
for source in ("vifactcheck", "viwikifc", "vihsd"):
    test_items = native_frame(source, "test")
    tix = Index()
    for it in test_items:
        tix.add(it["state"], f"{source}:test")
    cand = []
    for it in native_frame(source, "dev"):
        if tix.check(it["state"]):
            log["dev_native_dropped_overlap_with_test_split"][source] += 1
        else:
            cand.append(it)
    cand.sort(key=lambda x: x["key"])
    keys = balanced_sample([c["key"] for c in cand], [c["label"] for c in cand], N, rng0)
    byk = {c["key"]: c for c in cand}
    dev_native += native_rows(source, [byk[k] for k in sorted(keys)], "dev")
dev += dev_native

dev_index = Index()
for r in dev:
    dev_index.add(r["state"], f"dev:{r['source']}")
    if r["source"] == "xcopa":
        for o in r["options"]:
            dev_index.add(o, "dev:xcopa_option")


# chiều ngược lại: một câu test ngắn (4–12 từ) nằm nguyên vẹn trong văn bản dev
dev_ngrams = set()
for r in dev:
    for t in [r["state"]] + (r["options"] if r["source"] == "xcopa" else []):
        w = normalize(t).split()
        for n in range(4, 13):
            for i in range(len(w) - n + 1):
                dev_ngrams.add(hash(tuple(w[i:i + n])))


def hits_dev(*texts):
    for t in texts:
        h = dev_index.check(t)
        if h:
            return h
        w = normalize(t).split()
        if 4 <= len(w) <= 12 and hash(tuple(w)) in dev_ngrams:
            return ("contained_in_dev", "dev")
    return None


# ===================================================================== test 1.0, khối song song
test = []
reserved = load_json(DATA / "reserved_test.json")
# Belebele: 244 đoạn để dành
bv, be = load("facebook/belebele", "vie_Latn", "test"), load("facebook/belebele", "eng_Latn", "test")
bm = bv.merge(be, on=["link", "question_number"], suffixes=("_vi", "_en"))
bm = bm[bm.link.isin(set(reserved["belebele"]["links"]))].copy()
bm["key"] = bm.link + "#" + bm.question_number.astype(str); bm["label"] = bm.correct_answer_num_en.astype(int)
ok = [not hits_dev(x.flores_passage_vi, x.flores_passage_en) for x in bm.itertuples()]
log["test_dropped_overlap_with_dev"]["belebele"] = int(len(ok) - sum(ok))
bm = bm[ok].sort_values("key")
for k in sorted(balanced_sample(list(bm.key), list(bm.label), N, rng0)):
    x = bm[bm.key == k].iloc[0]
    pid = f"belebele:test:{k}"
    rng = pair_rng("belebele", pid); perm = list(rng.permutation(4))
    for lang in ("en", "vi"):
        opts = [x[f"mc_answer{i}_{lang}"] for i in range(1, 5)]
        test.append(row(pid, "belebele", lang, "choice", "parallel", x[f"flores_passage_{lang}"], x[f"question_{lang}"],
                        [opts[p] for p in perm], perm.index(x.label - 1), "belebele:test (reserved half)", f"belebele:{x.link}"))
# SIB-200: cả split test
topics = sorted(tr["sib200_topics"])
sv, se = load("Davlan/sib200", "vie_Latn", "test"), load("Davlan/sib200", "eng_Latn", "test")
sm = sv.merge(se, on="index_id", suffixes=("_vi", "_en")).sort_values("index_id")
for x in sm.itertuples():
    if hits_dev(x.text_vi, x.text_en):
        log["test_dropped_overlap_with_dev"]["sib200"] += 1; continue
    pid = f"sib200:test:{x.index_id}"
    rng = pair_rng("sib200", pid); perm = list(rng.permutation(len(topics)))
    for lang in ("en", "vi"):
        opts = [tr["sib200_topics"][t][lang] for t in topics]
        test.append(row(pid, "sib200", lang, "choice", "parallel", getattr(x, f"text_{lang}"), tr["questions"]["sib200"][lang],
                        [opts[p] for p in perm], perm.index(topics.index(x.category_en)), "sib200:test", pid, gold_label=x.category_en))
# MASSIVE test: 1 đúng + 9 nhiễu
intents = sorted(tr["massive_intents"])
ma, mb = load("mteb/amazon_massive_intent", "vi", "test"), load("mteb/amazon_massive_intent", "en", "test")
mm = ma.merge(mb, on="id", suffixes=("_vi", "_en")); mm["iid"] = mm.id.astype(int); mm = mm.sort_values("iid")
okm = [not hits_dev(x.text_vi, x.text_en) for x in mm.itertuples()]
log["test_dropped_overlap_with_dev"]["massive"] = int(len(okm) - sum(okm))
mm = mm[okm]
for iid in sorted(balanced_sample(list(mm.iid), list(mm.label_en), N, rng0)):
    x = mm[mm.iid == iid].iloc[0]
    pid = f"massive:test:{iid}"
    rng = pair_rng("massive", pid)
    pool = [i for i in intents if i != x.label_en]
    labels = [x.label_en] + [pool[i] for i in sorted(rng.choice(len(pool), 9, replace=False))]
    perm = list(rng.permutation(10))
    for lang in ("en", "vi"):
        opts = [tr["massive_intents"][l][lang] for l in labels]
        test.append(row(pid, "massive", lang, "choice", "parallel", x[f"text_{lang}"], tr["questions"]["massive"][lang],
                        [opts[p] for p in perm], perm.index(0), "massive:test", pid, gold_label=x.label_en,
                        flags=[f"distractors={','.join(labels[1:])}"]))
# XCOPA test (vi) + COPA test (en) theo idx; nhãn từ XCOPA
xv = load("cambridgeltl/xcopa", "vi", "test"); ce = load("aps/super_glue", "copa", "test")
xm = xv.merge(ce, on="idx", suffixes=("_vi", "_en")).sort_values("idx")
assert len(xm) == 500 and (xm.question_vi == xm.question_en).all()
okx = [not hits_dev(x.premise_vi, x.premise_en, x.choice1_vi, x.choice2_vi, x.choice1_en, x.choice2_en) for x in xm.itertuples()]
log["test_dropped_overlap_with_dev"]["xcopa"] = int(len(okx) - sum(okx))
xm = xm[okx]
for idx in sorted(balanced_sample(list(xm.idx), list(xm.label_vi), N, rng0)):
    x = xm[xm.idx == idx].iloc[0]
    pid = f"xcopa:test:{idx}"
    rng = pair_rng("xcopa", pid); perm = list(rng.permutation(2))
    q = tr["questions"][f"xcopa_{x.question_vi}"]
    for lang in ("en", "vi"):
        opts = [x[f"choice1_{lang}"], x[f"choice2_{lang}"]]
        test.append(row(pid, "xcopa", lang, "choice", "parallel", x[f"premise_{lang}"], q[lang], [opts[p] for p in perm],
                        perm.index(int(x.label_vi)), "xcopa:test" if lang == "vi" else "super_glue/copa:test (label from xcopa)", pid,
                        flags=[f"copa_question={x.question_vi}"] + (["xcopa_changed=True"] if x.changed else [])))
# ===================================================================== test 1.0, khối bản địa
for source in ("vifactcheck", "viwikifc", "vihsd"):
    cand = []
    for it in native_frame(source, "test"):
        if hits_dev(it["state"]):
            log["test_dropped_overlap_with_dev"][source] += 1
        else:
            cand.append(it)
    cand.sort(key=lambda x: x["key"])
    keys = balanced_sample([c["key"] for c in cand], [c["label"] for c in cand], N, rng0)
    byk = {c["key"]: c for c in cand}
    test += native_rows(source, [byk[k] for k in sorted(keys)], "test")


# ===================================================================== bản không dấu + n_tokens
def add_nodiac(rows):
    out = []
    for r in rows:
        out.append(r)
        if r["lang"] == "vi":
            n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]),
                 "question": strip_diacritics(r["question"]), "options": [strip_diacritics(o) for o in r["options"]]}
            for a, b in ((r["state"], n["state"]), (r["question"], n["question"])) + tuple(zip(r["options"], n["options"])):
                assert re.findall(r"\d+", a) == re.findall(r"\d+", b), (r["id"], a[:80])
            assert n["pair_id"] == r["pair_id"] and n["target"] == r["target"] and len(n["options"]) == len(r["options"])
            out.append(n)
    return out


test, dev = add_nodiac(test), add_nodiac(dev)
for r in test + dev:
    if r["n_tokens"] is None or r["variant"] == "nodiac":
        r["n_tokens"] = len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))
    r["n_state_tokens"] = len(tok.encode(r["state"], add_special_tokens=False))
assert len({r["id"] for r in test}) == len(test) and len({r["id"] for r in dev}) == len(dev)
# dev_v0 giữ nguyên từng trường
v0 = {r["id"]: r for r in dev_v0}
for r in dev:
    if r["id"] in v0:
        assert all(r[k] == v0[r["id"]][k] for k in v0[r["id"]]), r["id"]

# ===================================================================== chống rò rỉ
leak = {}
test_index = Index()
for r in test:
    test_index.add(r["state"], f"test:{r['source']}")
leak["dev_states_hitting_test"] = sum(bool(test_index.check(r["state"])) for r in dev if r["variant"] == "orig")
leak["test_states_hitting_dev"] = sum(bool(dev_index.check(r["state"])) for r in test if r["variant"] == "orig")
assert leak["dev_states_hitting_test"] == 0 and leak["test_states_hitting_dev"] == 0, leak
leak["dropped_while_building"] = {k: dict(v) for k, v in log.items()}
for name, rows_ in (("test", test), ("dev_v1", dev)):
    idx = Index()
    for r in rows_:
        if r["variant"] == "orig":
            idx.add(r["state"], f"{name}:{r['source']}")
    hits = Counter()
    for t in read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl"):
        h = idx.check(t["state"])
        if h:
            hits[f"{h[0]}|{h[1]}|train_source={t['source']}"] += 1
    leak[f"train_v0_vs_{name}"] = {"total": sum(hits.values()), "by_kind_target_trainsource": dict(hits.most_common())}

# ===================================================================== ghi file
tp, dp = OUT / "test_v1.0.jsonl", OUT / "dev_v1.jsonl"
write_jsonl(tp, test); write_jsonl(dp, dev)
(OUT / "leakage_v1.json").write_text(json.dumps(leak, ensure_ascii=False, indent=1), encoding="utf-8")


def counts(rows_):
    return {"rows": len(rows_), "by_source": dict(Counter(r["source"] for r in rows_)),
            "by_source_lang_variant": dict(Counter(f"{r['source']}|{r['lang']}|{r['variant']}" for r in rows_)),
            "by_kind": dict(Counter(f"{r['source']}|{r['kind']}" for r in rows_ if r["variant"] == "orig")),
            "target_hist_orig": dict(Counter(f"{r['source']}|{r['kind']}|{r['target']}" for r in rows_ if r["variant"] == "orig" and r["lang"] == "vi")),
            "gold_label_hist_native": dict(Counter(f"{r['source']}|{r['gold_label']}" for r in rows_ if r["block"] == "native" and r["variant"] == "orig"))}


common_meta = {"seed": SEED, "revisions": REV, "licenses": LICENSE, "questions_config": "config/bench_v1_questions.json (reviewed: false)",
               "translations": "config/translations.json (reviewed: false)", "n_tokens": "rendered prompt, tokenizer Qwen/Qwen3-0.6B",
               "nodiac": "strip combining marks (NFD) and đ/Đ→d/D on state, question, options; prompt frame words unchanged",
               "leakage": leak}
(OUT / "manifest_test_v1.0.json").write_text(json.dumps({"name": "bench_v1 test", "version": "1.0", "frozen": True,
    "sha256": {tp.name: sha256_file(tp), "bench_v1_questions.json": sha256_file(CONFIG / "bench_v1_questions.json"),
               "translations.json": sha256_file(CONFIG / "translations.json")},
    "counts": counts(test), **common_meta,
    "note": "Không sửa file này sau khi đóng băng. Lỗi phát hiện sau đó ghi lại để sửa ở bản 1.1. Không chạy model nào trên tập này."},
    ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "manifest_dev_v1.json").write_text(json.dumps({"name": "dev_v1", "sha256": {dp.name: sha256_file(dp)},
    "parallel_part": "dev_v0 unchanged (ids and fields identical) + nodiac variants of its vi rows",
    "counts": counts(dev), **common_meta}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"test": counts(test)["by_source_lang_variant"], "dev": counts(dev)["by_source_lang_variant"],
                  "leak": {k: (v if not isinstance(v, dict) or "total" not in v else v["total"]) for k, v in leak.items()}},
                 ensure_ascii=False, indent=1))
