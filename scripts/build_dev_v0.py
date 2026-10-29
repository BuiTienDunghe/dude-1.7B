"""Dựng tập dev_v0 song song Anh–Việt từ Belebele, SIB-200, MASSIVE (mirror mteb) và XCOPA (+ COPA tiếng Anh).

Ra: data/dev_v0.jsonl, data/reserved_test.json, data/manifest_dev_v0.json
"""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common import CONFIG, DATA, LETTERS, SEED, load_json, model_revisions, render_prompt, sha256_file, write_jsonl

# Revision ghim của các dataset (commit sha trên Hugging Face Hub, tra ngày 2026-10-12).
DATASETS = {
    "facebook/belebele": {"rev": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", "license": "cc-by-sa-4.0"},
    "Davlan/sib200": {"rev": "38977a667f6fc264d5c26ec57a01e16db040b358", "license": "cc-by-sa-4.0"},
    "mteb/amazon_massive_intent": {"rev": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
                                   "license": "apache-2.0 (mirror card); original AmazonScience/massive is cc-by-4.0"},
    "cambridgeltl/xcopa": {"rev": "042f78955ba48e6404616762fa6e05e839c3907a", "license": "cc-by-4.0"},
    "aps/super_glue": {"rev": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
                       "license": "HF card: 'other'; original COPA homepage (asgordon.github.io/copa.html): BSD 2-Clause"},
}
N_BELEBELE, N_SIB_TRAIN, N_MASSIVE = 300, 201, 300
N_DISTRACTORS = 9

tr = load_json(CONFIG / "translations.json")
frames = load_json(CONFIG / "prompt_frames.json")


def load(repo, cfg, split):
    return load_dataset(repo, cfg, revision=DATASETS[repo]["rev"], split=split).to_pandas()


def pair_rng(source, pair_id):
    """RNG riêng cho từng pair_id: hai câu Anh/Việt cùng pair_id nhận cùng thứ tự lựa chọn và cùng nhãn nhiễu."""
    h = hashlib.sha256(f"{SEED}:{source}:{pair_id}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def balanced_sample(keys, labels, n, rng):
    """Lấy n phần tử cân theo nhãn nhất có thể: xoay vòng qua các nhãn (đã sắp xếp), mỗi nhãn đã xáo theo seed.
    Nhãn hết phần tử thì các nhãn còn lại bù vào. keys phải đã được sắp xếp ổn định."""
    groups = defaultdict(list)
    for k, l in zip(keys, labels):
        groups[l].append(k)
    order = {l: list(rng.permutation(len(g))) for l, g in sorted(groups.items())}
    out, depth = [], 0
    while len(out) < n:
        progressed = False
        for l in sorted(groups):
            if depth < len(groups[l]) and len(out) < n:
                out.append(groups[l][order[l][depth]])
                progressed = True
        if not progressed:
            raise ValueError(f"not enough items: wanted {n}, have {len(out)}")
        depth += 1
    return out


def shuffled(options_by_lang, gold, rng):
    """Xáo thứ tự lựa chọn một lần cho cả cặp; trả về options theo từng ngôn ngữ và chỉ số đáp án đúng."""
    perm = list(rng.permutation(len(next(iter(options_by_lang.values())))))
    return {lang: [opts[i] for i in perm] for lang, opts in options_by_lang.items()}, perm.index(gold)


rows, reserved, unused, notes = [], {}, {}, []
seed_rng = np.random.default_rng(SEED)


def emit(source, kind, pair_id, cluster, per_lang, options, target, gold_label=None, flags=None):
    for lang in ("en", "vi"):
        st, q, origin = per_lang[lang]
        rows.append({
            "id": f"{pair_id}:{lang}", "pair_id": pair_id, "source": source, "lang": lang, "kind": kind,
            "state": st, "question": q, "options": options[lang], "target": int(target),
            "split_origin": origin, "n_tokens": None,
            "cluster": cluster, "gold_label": gold_label, "flags": flags or [],
        })


# ---------------- Belebele: chia theo đoạn văn (khóa = link), một nửa dev, một nửa để dành test
r = "facebook/belebele"
bv, be = load(r, "vie_Latn", "test"), load(r, "eng_Latn", "test")
bm = bv.merge(be, on=["link", "question_number"], suffixes=("_vi", "_en"))
assert len(bm) == 900 and (bm.correct_answer_num_vi == bm.correct_answer_num_en).all()
links = sorted(bm.link.unique())
perm = seed_rng.permutation(len(links))
half = len(links) // 2
dev_links = sorted(links[i] for i in perm[:half])
test_links = sorted(links[i] for i in perm[half:])
reserved["belebele"] = {"rule": "passages (keyed by link) in the test half; all their questions",
                        "links": test_links,
                        "question_keys": sorted(f"{l}#{q}" for l, q in bm[bm.link.isin(test_links)][["link", "question_number"]].itertuples(index=False))}
dev_pool = bm[bm.link.isin(dev_links)].sort_values(["link", "question_number"]).reset_index(drop=True)
dev_pool["key"] = dev_pool.link + "#" + dev_pool.question_number.astype(str)
dev_pool["label"] = dev_pool.correct_answer_num_en.astype(int)
picked = set(balanced_sample(list(dev_pool.key), list(dev_pool.label), N_BELEBELE, seed_rng))
unused["belebele"] = sorted(set(dev_pool.key) - picked)
for x in dev_pool[dev_pool.key.isin(picked)].itertuples(index=False):
    pid = f"belebele:{x.key}"
    opts = {lang: [getattr(x, f"mc_answer{i}_{lang}") for i in range(1, 5)] for lang in ("en", "vi")}
    options, target = shuffled(opts, x.label - 1, pair_rng("belebele", pid))
    emit("belebele", "reading_mc", pid, f"belebele:{x.link}",
         {lang: (getattr(x, f"flores_passage_{lang}"), getattr(x, f"question_{lang}"), "test (dev half, passage split)")
          for lang in ("en", "vi")}, options, target)

# ---------------- SIB-200: toàn bộ validation + 201 câu train cân theo nhãn; không đụng test
r = "Davlan/sib200"
topics = sorted(tr["sib200_topics"])
def sib(split):
    a, b = load(r, "vie_Latn", split), load(r, "eng_Latn", split)
    m = a.merge(b, on="index_id", suffixes=("_vi", "_en")).sort_values("index_id").reset_index(drop=True)
    assert len(m) == len(a) == len(b) and (m.category_vi == m.category_en).all()
    assert set(m.category_en) <= set(topics), set(m.category_en) - set(topics)
    return m
sv, st_ = sib("validation"), sib("train")
reserved["sib200"] = {"rule": "whole 'test' split", "index_ids": sorted(int(i) for i in sib("test").index_id)}
picked = set(balanced_sample(list(st_.index_id), list(st_.category_en), N_SIB_TRAIN, seed_rng))
unused["sib200"] = sorted(int(i) for i in set(st_.index_id) - picked)
sib_rows = [(x, "validation") for x in sv.itertuples(index=False)] + \
           [(x, "train") for x in st_[st_.index_id.isin(picked)].itertuples(index=False)]
for x, origin in sib_rows:
    pid = f"sib200:{x.index_id}"
    opts = {lang: [tr["sib200_topics"][t][lang] for t in topics] for lang in ("en", "vi")}
    options, target = shuffled(opts, topics.index(x.category_en), pair_rng("sib200", pid))
    emit("sib200", "topic", pid, pid,
         {lang: (getattr(x, f"text_{lang}"), tr["questions"]["sib200"][lang], origin) for lang in ("en", "vi")},
         options, target, gold_label=x.category_en)

# ---------------- MASSIVE (mirror mteb/amazon_massive_intent): 300 câu validation cân theo nhãn, 1 đúng + 9 nhiễu
r = "mteb/amazon_massive_intent"
intents = sorted(tr["massive_intents"])
assert len(intents) == 60
ma, mb = load(r, "vi", "validation"), load(r, "en", "validation")
mm = ma.merge(mb, on="id", suffixes=("_vi", "_en"))
mm["iid"] = mm.id.astype(int)
mm = mm.sort_values("iid").reset_index(drop=True)
assert len(mm) == len(ma) == len(mb) and (mm.label_vi == mm.label_en).all()
all_labels = set(load(r, "en", "train").label) | set(mm.label_en)
assert all_labels == set(intents), all_labels ^ set(intents)
reserved["massive"] = {"rule": "whole 'test' split", "ids": sorted(int(i) for i in load(r, "en", "test").id)}
picked = set(balanced_sample(list(mm.iid), list(mm.label_en), N_MASSIVE, seed_rng))
unused["massive"] = {"validation_not_sampled": len(set(mm.iid) - picked), "train": "whole split unused"}
for x in mm[mm.iid.isin(picked)].itertuples(index=False):
    pid = f"massive:{x.iid}"
    rng = pair_rng("massive", pid)
    pool = [i for i in intents if i != x.label_en]
    labels = [x.label_en] + [pool[i] for i in sorted(rng.choice(len(pool), N_DISTRACTORS, replace=False))]
    opts = {lang: [tr["massive_intents"][l][lang] for l in labels] for lang in ("en", "vi")}
    options, target = shuffled(opts, 0, rng)
    emit("massive", "intent", pid, pid,
         {lang: (getattr(x, f"text_{lang}"), tr["questions"]["massive"][lang], "validation") for lang in ("en", "vi")},
         options, target, gold_label=x.label_en, flags=[f"distractors={','.join(labels[1:])}"])

# ---------------- XCOPA vi validation + COPA tiếng Anh (SuperGLUE copa validation, khớp theo idx)
xv = load("cambridgeltl/xcopa", "vi", "validation")
ce = load("aps/super_glue", "copa", "validation")
xm = xv.merge(ce, on="idx", suffixes=("_vi", "_en")).sort_values("idx").reset_index(drop=True)
assert len(xm) == 100 and (xm.question_vi == xm.question_en).all() and (xm.label_vi == xm.label_en).all()
reserved["xcopa"] = {"rule": "whole 'test' split (XCOPA vi test = COPA test, 500 items)",
                     "idx": sorted(int(i) for i in load("cambridgeltl/xcopa", "vi", "test").idx)}
for x in xm.itertuples(index=False):
    pid = f"xcopa:{x.idx}"
    opts = {lang: [getattr(x, f"choice1_{lang}"), getattr(x, f"choice2_{lang}")] for lang in ("en", "vi")}
    options, target = shuffled(opts, int(x.label_vi), pair_rng("xcopa", pid))
    q = tr["questions"][f"xcopa_{x.question_vi}"]
    emit("xcopa", "causal", pid, pid,
         {"vi": (x.premise_vi, q["vi"], "xcopa:validation"), "en": (x.premise_en, q["en"], "super_glue/copa:validation")},
         options, target, gold_label=None,
         flags=[f"copa_question={x.question_vi}"] + (["xcopa_changed=True"] if x.changed else []))

# ---------------- n_tokens = số token của prompt đã render đầy đủ (tokenizer Qwen3, chung cho hai model)
tok_name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(tok_name, revision=model_revisions()[tok_name])
for row in rows:
    assert 2 <= len(row["options"]) <= len(LETTERS)
    row["n_tokens"] = len(tok.encode(render_prompt(row, tok, frames), add_special_tokens=False))
rows.sort(key=lambda r: (["belebele", "sib200", "massive", "xcopa"].index(r["source"]), r["pair_id"], r["lang"]))

out = DATA / "dev_v0.jsonl"
write_jsonl(out, rows)
res_path = DATA / "reserved_test.json"
res_path.write_text(json.dumps(reserved, ensure_ascii=False, indent=1), encoding="utf-8")

by_src = defaultdict(list)
for row in rows:
    by_src[row["source"]].append(row)
stats = {s: {"rows": len(v), "pairs": len(v) // 2,
             "split_origin": dict(Counter(r["split_origin"] for r in v)),
             "n_options": dict(Counter(len(r["options"]) for r in v)),
             "target_letter_hist": dict(sorted(Counter(LETTERS[r["target"]] for r in v if r["lang"] == "en").items())),
             "gold_label_hist": dict(Counter(r["gold_label"] for r in v if r["lang"] == "en" and r["gold_label"])),
             "n_tokens_max": max(r["n_tokens"] for r in v),
             "n_tokens_mean": {lang: round(float(np.mean([r["n_tokens"] for r in v if r["lang"] == lang])), 1) for lang in ("en", "vi")}}
         for s, v in by_src.items()}
manifest = {
    "name": "dev_v0", "seed": SEED, "created": "2026-10-12",
    "datasets": DATASETS,
    "files": {p.name: sha256_file(p) for p in [out, res_path, CONFIG / "translations.json", CONFIG / "prompt_frames.json"]},
    "n_tokens_definition": f"tokens of the fully rendered chat prompt (enable_thinking=False), tokenizer {tok_name}@{model_revisions()[tok_name]}",
    "pairing_keys": {"belebele": "(link, question_number)", "sib200": "index_id", "massive": "id",
                     "xcopa": "idx (XCOPA vi validation <-> SuperGLUE COPA validation)"},
    "stats": stats,
    "reserved_for_test": {k: {"rule": v["rule"], "count": len(next(x for kk, x in v.items() if kk != "rule"))} for k, v in reserved.items()},
    "unused_not_reserved": {k: (len(v) if isinstance(v, list) else v) for k, v in unused.items()},
    "notes": [
        "MASSIVE: AmazonScience/massive refused by datasets>=4 (loading script massive.py); used mirror mteb/amazon_massive_intent (configs vi, en).",
        "MASSIVE validation has 59 of the 60 intents (audio_volume_other only in train/test); it can appear as a distractor but never as gold.",
        "SIB-200 validation (99) is taken whole and is not label-balanced; only the 201 train rows are balanced.",
        "XCOPA English side: aps/super_glue copa validation; idx/question/label agree on 100/100. XCOPA rows with changed=True: idx 14, 21, 22 (flagged).",
        "translations.json (questions, topic names, intent descriptions in both vi and en) is NOT human-reviewed.",
    ],
}
(DATA / "manifest_dev_v0.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: manifest[k] for k in ["files", "stats", "reserved_for_test", "unused_not_reserved"]}, ensure_ascii=False, indent=1))
