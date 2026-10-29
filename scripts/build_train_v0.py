"""Dựng train_v0 (~20.000 quyết định) + val nội bộ 1.000 dòng từ tasksource-jev-typed-decisions và PhoLaya-base.

- Loại mọi nguồn liên quan tới dev_v0 (MASSIVE/SLURP/IntentGrasp, COPA/XCOPA/balanced-COPA, Belebele, SIB-200, FLORES),
  xác định qua sources.yaml của tasksource; PhoLaya chỉ lấy ViNLI, VSFC, ViSFD.
- Lọc trùng/gần trùng/chứa đoạn với văn bản của dev_v0, phần để dành test và toàn bộ các bộ nguồn của dev_v0.
- Giới hạn 5% tính theo nguồn ở mức tác vụ (tasksource `source`; PhoLaya `pholaya/<bộ>/<câu hỏi>`), theo quyết định của người dùng.
- Val nội bộ chia theo văn bản gốc: các câu dùng chung group_id hoặc chung văn bản nằm cùng một phía.

Ra: data/train_v0.jsonl, data/val_v0.jsonl, data/manifest_train_v0.json

QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common import CONFIG, DATA, LETTERS, SEED, load_json, model_revisions, read_jsonl, sha256_file, write_jsonl
from decontam import Index, normalize

TS = ("tasksource/tasksource-jev-typed-decisions", "bf1e4a874575dfb16852af90f62eb739a737fd17")
PL = ("LakoreAI/PhoLaya-base", "034d4273fb2600582cb789c45defe86b1c07e12d")
DEV_DS = {  # các bộ nguồn của dev_v0, ghim đúng revision của bước trước
    "belebele": ("facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"),
    "sib200": ("Davlan/sib200", "38977a667f6fc264d5c26ec57a01e16db040b358"),
    "massive": ("mteb/amazon_massive_intent", "940fd47a81eaa7f2cc7b129674d945d618ac38c2"),
    "xcopa": ("cambridgeltl/xcopa", "042f78955ba48e6404616762fa6e05e839c3907a"),
    "copa": ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646"),
}
N_TS, N_PL = 16000, 4000          # train
N_VAL_TS, N_VAL_PL = 800, 200     # val nội bộ, cùng tỉ lệ
CAP = 0.05                        # tối đa 5% train cho một nguồn (mức tác vụ)
MAX_OPTIONS = len(LETTERS)
# Trần độ dài state (token Qwen3): tham số --max_state_tokens, mặc định 2.048 từ 28/10 (audit 27/10). train_v0/val_v0 đã dựng với 1.024:
# muốn dựng lại đúng v0 thì chạy với --max_state_tokens 1024.
_ap = argparse.ArgumentParser()
_ap.add_argument("--max_state_tokens", type=int, default=2048)
MAX_STATE_TOKENS = _ap.parse_args().max_state_tokens
PL_TASKS = ["vinli/relation", "vinli/entailment", "vsfc/sentiment", "vsfc/topic", "visfd/aspect", "visfd/complaint"]

fmt = load_json(CONFIG / "train_format.json")
rng = np.random.default_rng(SEED)
tok_name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(tok_name, revision=model_revisions()[tok_name])
drops = Counter()

# ---------------------------------------------------------------- 1) danh sách nguồn tasksource bị loại (theo sources.yaml)
EXCL_DATASETS = {
    "AmazonScience/massive": "MASSIVE", "mteb/MassiveIntentClassification": "MASSIVE",
    "tasksource/slurp": "SLURP (văn bản gốc của MASSIVE en-US)", "yuweiyin/IntentGrasp": "IntentGrasp (chứa câu lệnh MASSIVE/SLURP)",
    "cambridgeltl/xcopa": "XCOPA", "pkavumba/balanced-copa": "Balanced-COPA (dựng từ COPA)",
}
EXCL_PATTERN = re.compile(r"(?i)flores|belebele|sib-?200|massive|slurp|xcopa|copa|hwu64")
sy = yaml.safe_load(open(hf_hub_download(TS[0], "sources.yaml", repo_type="dataset", revision=TS[1]), encoding="utf-8"))


def strings(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield str(k); yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif x is not None:
        yield str(x)


excluded = {}
for name, meta in sy["sources"].items():
    names = [meta.get("dataset", "")] + list(meta.get("originals") or []) + list((meta.get("revisions") or {}).keys())
    hit = next((EXCL_DATASETS[n] for n in names if n in EXCL_DATASETS), None)
    if hit is None and meta.get("dataset") == "aps/super_glue" and meta.get("config") == "copa":
        hit = "SuperGLUE COPA (bản tiếng Anh của XCOPA)"
    if hit is None and (EXCL_PATTERN.search(name) or any(EXCL_PATTERN.search(s) for s in names)):
        hit = "tên/bộ gốc khớp mẫu " + EXCL_PATTERN.pattern
    if hit:
        excluded[name] = hit

# ---------------------------------------------------------------- 2) chỉ mục văn bản của dev_v0, phần để dành test và các bộ nguồn
index = Index()
dev_rows = read_jsonl(DATA / "dev_v0.jsonl")
for r in dev_rows:
    index.add(r["state"], f"dev_v0:{r['source']}")
    if r["source"] == "xcopa":
        for o in r["options"]:
            index.add(o, "dev_v0:xcopa_option")
reserved = load_json(DATA / "reserved_test.json")
res_links = set(reserved["belebele"]["links"])
for cfg in ("vie_Latn", "eng_Latn"):
    for x in load_dataset(DEV_DS["belebele"][0], cfg, revision=DEV_DS["belebele"][1], split="test"):
        index.add(x["flores_passage"], "reserved:belebele" if x["link"] in res_links else "source:belebele")
    for sp in ("train", "validation", "test"):
        for x in load_dataset(DEV_DS["sib200"][0], cfg, revision=DEV_DS["sib200"][1], split=sp):
            index.add(x["text"], "reserved:sib200" if sp == "test" else "source:sib200")
for cfg in ("vi", "en"):
    for sp in ("train", "validation", "test"):
        for x in load_dataset(DEV_DS["massive"][0], cfg, revision=DEV_DS["massive"][1], split=sp):
            index.add(x["text"], "reserved:massive" if sp == "test" else "source:massive")
for sp in ("validation", "test"):
    for x in load_dataset(DEV_DS["xcopa"][0], "vi", revision=DEV_DS["xcopa"][1], split=sp):
        for t in (x["premise"], x["choice1"], x["choice2"]):
            index.add(t, "reserved:xcopa" if sp == "test" else "source:xcopa")
for sp in ("train", "validation", "test"):
    for x in load_dataset(DEV_DS["copa"][0], "copa", revision=DEV_DS["copa"][1], split=sp):
        for t in (x["premise"], x["choice1"], x["choice2"]):
            index.add(t, "reserved:copa" if sp == "test" else "source:copa")
print("index docs:", index.n)

# ---------------------------------------------------------------- 3) chuyển về một dạng chung
LETTER_REF = [re.compile(p) for p in (
    r"\b[A-J]\b\s*(?:,|and|&|or)\s*\b[A-J]\b",
    r"\([a-jA-J1-9]\)\s*(?:,|and|&|or)\s*\(?[a-jA-J1-9]\)",
    r"(?i)\b(?:options?|answers?|choices?|statements?)\s*\(?[A-J1-9]\)?(?:\s|$|,|\.)",
)]


def cites_other_options(options):
    return any(p.search(o) for o in options for p in LETTER_REF)


def finalize(row):
    """Áp các bộ lọc chung; trả về row hoặc None (và đếm lý do loại)."""
    if not (2 <= len(row["options"]) <= MAX_OPTIONS):
        drops[f"options_not_2_to_{MAX_OPTIONS}"] += 1; return None
    if abs(sum(row["target"]) - 1) > 1e-4 or min(row["target"]) < 0:
        drops["target_not_distribution"] += 1; return None
    if not row["state"].strip():
        drops["empty_state"] += 1; return None
    if cites_other_options(row["options"]):
        drops["options_cite_other_options"] += 1; return None
    n = len(tok.encode(row["state"], add_special_tokens=False))
    if n > MAX_STATE_TOKENS:
        drops[f"state_over_{MAX_STATE_TOKENS}_tokens"] += 1; return None
    row["n_state_tokens"] = n
    hit = index.check(row["state"])
    if hit:
        drops[f"decontam:{hit[0]}:{hit[1]}"] += 1; return None
    return row


def ts_row(x):
    src = x["source"]
    if src.startswith("multilingual/"):
        text_lang = "vi" if re.search(r"(?:[-/_])vi$", src) else "multi"
    else:
        text_lang = "en"
    lang = "vi" if text_lang == "vi" else "en"   # khung prompt chỉ có en/vi
    if x["kind"] == "noul":
        p = float(x["target"][0])
        options, target = list(fmt["noul_options"][lang]), [p, 1 - p]
    else:
        options, target = list(x["options"]), [float(t) for t in x["target"]]
    return {"id": f"ts:{x['id']}", "source": src, "dataset": "tasksource", "lang": lang, "text_lang": text_lang,
            "kind": x["kind"], "state": x["state"], "question": x["question"], "options": options, "target": target,
            "group_keys": [f"g:{x['group_id']}"], "license_use": x["license_use"], "variant": x["variant"]}


def pl_row(x, ds_name, n):
    state = json.loads(x["state"]); qs = json.loads(x["questions"]); gold = json.loads(x["gold"])
    (qid, q), = qs.items()
    labels = fmt["pholaya_state_labels"]
    text = next(iter(state.values())) if len(state) == 1 else "\n".join(f"{labels[k]}: {v}" for k, v in state.items())
    probs = gold[qid]["probabilities"]
    if q["type"] == "noul":
        p = float(probs["true"])
        assert abs(p + float(probs["false"]) - 1) < 1e-6
        options, target = list(fmt["noul_options"]["vi"]), [p, 1 - p]
    else:
        keys = list(q["criteria"])
        assert set(keys) == set(probs), (keys, probs)
        options, target = [q["criteria"][k] for k in keys], [float(probs[k]) for k in keys]
    task = "aspect" if qid.startswith("aspect_") else qid
    return {"id": f"pl:{ds_name}:{n}", "source": f"pholaya/{ds_name}/{task}", "dataset": f"pholaya/{ds_name}",
            "lang": "vi", "text_lang": "vi", "kind": q["type"], "state": text, "question": q["instructions"],
            "options": options, "target": target, "group_keys": [f"f:{normalize(v)}" for v in state.values()],
            "license_use": "research-only (PhoLaya-base, UIT corpora)", "variant": qid}


# ---------------------------------------------------------------- 4) lấy mẫu tasksource
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
src_col = ts["source"]
pool_by_src = Counter(src_col)
keep_idx = [i for i, s in enumerate(src_col) if s not in excluded]
excluded_rows = {s: pool_by_src.get(s, 0) for s in excluded}
order = rng.permutation(len(keep_idx))
ts_rows, per_src = [], Counter()
need_ts = N_TS + N_VAL_TS
for j in order:
    if len(ts_rows) >= need_ts:
        break
    x = ts[int(keep_idx[j])]
    if per_src[x["source"]] >= int(CAP * (N_TS + N_PL)):
        drops["source_cap_reached"] += 1; continue
    if x["kind"] != "noul" and len(x["options"]) > MAX_OPTIONS:   # lọc sớm trước khi chuyển dạng
        drops[f"options_not_2_to_{MAX_OPTIONS}"] += 1; continue
    r = finalize(ts_row(x))
    if r:
        ts_rows.append(r); per_src[r["source"]] += 1
print("tasksource sampled", len(ts_rows))

# ---------------------------------------------------------------- 5) lấy mẫu PhoLaya: chia đều 6 tác vụ
pl_pool = defaultdict(list)
for ds_name in ("vinli", "vsfc", "visfd"):
    path = hf_hub_download(PL[0], f"sources/{ds_name}/train.jsonl", repo_type="dataset", revision=PL[1])
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            r = pl_row(json.loads(line), ds_name, n)
            pl_pool[r["source"].split("/", 1)[1]].append(r)
assert sorted(pl_pool) == sorted(PL_TASKS), sorted(pl_pool)
per_task = (N_PL + N_VAL_PL) // len(PL_TASKS)
pl_rows = []
for task in PL_TASKS:
    got = 0
    for j in rng.permutation(len(pl_pool[task])):
        if got >= per_task:
            break
        r = finalize(pl_pool[task][int(j)])
        if r:
            pl_rows.append(r); got += 1
    assert got == per_task, (task, got)
print("pholaya sampled", len(pl_rows))

# ---------------------------------------------------------------- 6) chia val theo văn bản gốc (union-find trên group_id / văn bản)
rows = ts_rows + pl_rows
parent = list(range(len(rows)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a


owner = {}
for i, r in enumerate(rows):
    for k in r["group_keys"] + [f"s:{normalize(r['state'])}"]:
        if k in owner:
            parent[find(i)] = find(owner[k])
        else:
            owner[k] = i
comps = defaultdict(list)
for i in range(len(rows)):
    comps[find(i)].append(i)
val_idx = set()
for stream, quota in (("tasksource", N_VAL_TS), ("pholaya", N_VAL_PL)):
    cands = sorted(c for c, m in comps.items() if rows[m[0]]["dataset"].startswith(stream))
    taken = 0
    for j in rng.permutation(len(cands)):
        m = comps[cands[int(j)]]
        if taken + len(m) > quota:
            continue
        val_idx.update(m); taken += len(m)
        if taken == quota:
            break
    assert taken == quota, (stream, taken)
train = [r for i, r in enumerate(rows) if i not in val_idx]
val = [r for i, r in enumerate(rows) if i in val_idx]
assert not ({normalize(r["state"]) for r in train} & {normalize(r["state"]) for r in val})
assert not ({k for r in train for k in r["group_keys"]} & {k for r in val for k in r["group_keys"]})

# val: xáo thứ tự lựa chọn một lần lúc dựng và lưu lại (val loss / temperature xác định)
for r in val:
    perm = rng.permutation(len(r["options"]))
    r["options"] = [r["options"][p] for p in perm]
    r["target"] = [r["target"][p] for p in perm]
for split, rs in (("train", train), ("val", val)):
    for r in rs:
        r["split"] = split
        r["target_argmax"] = int(np.argmax(r["target"]))
        del r["group_keys"]

# ---------------------------------------------------------------- 7) kiểm tra cuối: trùng với dev/test phải bằng 0; giới hạn nguồn
final_hits = Counter()
for r in train + val:
    hit = index.check(r["state"])
    if hit:
        final_hits[f"{hit[0]}:{hit[1]}"] += 1
assert sum(final_hits.values()) == 0, final_hits
src_share = Counter(r["source"] for r in train)
assert max(src_share.values()) <= CAP * len(train), src_share.most_common(3)

write_jsonl(DATA / "train_v0.jsonl", train)
write_jsonl(DATA / "val_v0.jsonl", val)


def counts(rs, key):
    return dict(Counter(r[key] for r in rs).most_common())


ds_share = Counter(r["dataset"] if r["dataset"] != "tasksource" else r["source"] for r in train)
manifest = {
    "name": "train_v0", "seed": SEED, "created": "2026-10-12",
    "sources": {
        "tasksource": {"repo": TS[0], "revision": TS[1], "config": "default", "split": "train",
                       "license": "per-row `license`/`license_use`; card: other",
                       "language": "inferred from source name (multilingual/* = non-English; */vi = Vietnamese)"},
        "pholaya": {"repo": PL[0], "revision": PL[1], "files": [f"sources/{d}/train.jsonl" for d in ("vinli", "vsfc", "visfd")],
                    "license": "research-use-only (UIT ViNLI, VSFC, ViSFD). Dùng nội bộ cho thí nghiệm chọn model; không dùng thương mại.",
                    "excluded_parts": "massive (cùng nguồn với dev_v0)"},
    },
    "excluded_tasksource_sources": {s: {"reason": excluded[s], "rows_in_default_train": excluded_rows[s]} for s in sorted(excluded)},
    "flores_note": "sources.yaml không có nguồn nào khớp flores/belebele/sib200; kiểm tra lọc trùng văn bản bên dưới là bằng chứng bổ sung.",
    "filters": {"max_options": MAX_OPTIONS, "max_state_tokens": MAX_STATE_TOKENS, "state_token_counter": f"{tok_name}",
                "source_cap": f"{CAP:.0%} of train, source = task level (user decision)",
                "dropped_while_sampling": dict(drops.most_common())},
    "decontamination": {"method": "exact / MinHash Jaccard>=0.8 on char-5 shingles / word-13-gram or whole-text (4-12 words) containment, on normalized text",
                        "reference": "dev_v0 states (+XCOPA options), reserved test portions, and all splits of Belebele, SIB-200, MASSIVE (vi,en), XCOPA-vi, COPA",
                        "reference_docs": index.n,
                        "removed_during_sampling": sum(v for k, v in drops.items() if k.startswith("decontam:")),
                        "final_overlap_train_val_vs_reference": sum(final_hits.values())},
    "val_split": "by source text: union of tasksource group_id, normalized state, and each PhoLaya state field; no component straddles",
    "noul_format": "options Yes/No (Có/Không), target [p, 1-p]; config/train_format.json (unreviewed)",
    "option_order": "train rows stored in source order and reshuffled at every load; val rows shuffled once at build and stored",
    "counts": {split: {"rows": len(rs), "by_kind": counts(rs, "kind"), "by_lang_frame": counts(rs, "lang"),
                       "by_text_lang": counts(rs, "text_lang"), "by_license_use": counts(rs, "license_use"),
                       "by_dataset": counts(rs, "dataset"), "n_sources": len({r['source'] for r in rs}),
                       "n_options": dict(sorted(Counter(len(r["options"]) for r in rs).items())),
                       "soft_target_rows": sum(max(r["target"]) < 1 for r in rs)}
               for split, rs in (("train", train), ("val", val))},
    "train_source_share_top": {s: round(n / len(train), 4) for s, n in src_share.most_common(15)},
    "train_dataset_share_top": {s: round(n / len(train), 4) for s, n in ds_share.most_common(10)},
    "train_by_source": dict(src_share.most_common()),
    "files": {p.name: sha256_file(p) for p in [DATA / "train_v0.jsonl", DATA / "val_v0.jsonl", CONFIG / "train_format.json"]},
}
(DATA / "manifest_train_v0.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: manifest[k] for k in ["filters", "decontamination", "counts", "train_source_share_top", "train_dataset_share_top"]},
                 ensure_ascii=False, indent=1))
print("excluded sources:", len(excluded), sorted(excluded))
