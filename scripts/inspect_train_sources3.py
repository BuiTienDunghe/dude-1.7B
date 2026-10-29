"""Soi các nguồn tasksource có thể liên quan tới dev_v0 và định dạng từng nguồn con của PhoLaya.
In tối đa 5 dòng mỗi nhóm, mỗi trường cắt 200 ký tự."""
import json
import re
from collections import Counter

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
TS, PL = "tasksource/tasksource-jev-typed-decisions", "LakoreAI/PhoLaya-base"
ts_rev, pl_rev = api.dataset_info(TS).sha, api.dataset_info(PL).sha
cut = lambda v: str(v)[:200]

ts = load_dataset(TS, "default", revision=ts_rev, split="train")
srcs = Counter(ts["source"])
ymlp = hf_hub_download(TS, "sources.yaml", repo_type="dataset", revision=ts_rev)
yml = open(ymlp, encoding="utf-8").read()
print("#### sources.yaml: first 1500 chars")
print(yml[:1500])
print("#### sources.yaml lines mentioning dev_v0-related datasets")
for i, line in enumerate(yml.splitlines()):
    if re.search(r"(?i)flores|massive|copa|belebele|sib-?200|sib200|slurp|hwu|xcopa", line):
        print(f"  {i}: {line[:200]}")
import sys; sys.exit(0) if "--yaml-only" in sys.argv else None
pat = re.compile(r"(?i)intent|slurp|hwu|snips|clinc|banking|alexa|flores|nllb|wiki_?news|wikivoyage|wikibooks|topic|commonsense|causal|e-care|choice|plausib|cause")
print("#### sources matching intent/flores/causal patterns")
for s, n in sorted(srcs.items()):
    if pat.search(s):
        print(f"  {n:6d}  {s}")

idx = [i for i, s in enumerate(ts["source"]) if s.startswith("IntentGrasp")]
sub = ts.select(idx)
print("\n#### IntentGrasp: ids/question samples")
print(" question top:", Counter(sub["question"]).most_common(5))
print(" id prefixes:", Counter(i.split(":")[0] for i in sub["id"]).most_common(10))
for r in sub.select(range(5)):
    print("  ", {k: cut(r[k]) for k in ["id", "state", "question", "options"]})

print("\n#### AGENTS.md (first 4000 chars)")
print(open(hf_hub_download(TS, "AGENTS.md", repo_type="dataset", revision=ts_rev), encoding="utf-8").read()[:4000])

print("\n#### PhoLaya per-source train files")
for s in ["vinli", "vsfc", "visfd", "massive"]:
    p = hf_hub_download(PL, f"sources/{s}/train.jsonl", repo_type="dataset", revision=pl_rev)
    lines = open(p, encoding="utf-8").read().splitlines()
    rows = [json.loads(l) for l in lines]
    print(f"== {s}: {len(rows)} rows; keys {list(rows[0].keys())}")
    qtypes, nq = Counter(), Counter()
    for r in rows:
        q = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        nq[len(q)] += 1
        for qid, v in q.items():
            qtypes[(re.sub(r"\d+", "#", qid), v["type"], len(v.get("criteria") or {}))] += 1
    print("   questions per row:", dict(nq))
    print("   (qid,type,k):", qtypes.most_common(12))
    for r in rows[:2]:
        print("   ex:", {k: cut(v) for k, v in r.items()})
    noul = [r for r in rows if '"noul"' in (r["questions"] if isinstance(r["questions"], str) else json.dumps(r["questions"]))]
    for r in noul[:2]:
        print("   noul ex:", {k: cut(v) for k, v in r.items()})

mix = load_dataset(PL, "default", revision=pl_rev, split="train")
print("\nmix train rows", len(mix))
