"""Đọc README/manifest của hai nguồn train và liệt kê nguồn con, để xác định ngữ nghĩa noul/score và cách lọc."""
import json
import re
from collections import Counter

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
TS, PL = "tasksource/tasksource-jev-typed-decisions", "LakoreAI/PhoLaya-base"
ts_rev, pl_rev = api.dataset_info(TS).sha, api.dataset_info(PL).sha

readme = open(hf_hub_download(TS, "README.md", repo_type="dataset", revision=ts_rev), encoding="utf-8").read()
print("#### TASKSOURCE README (without yaml header), first 6000 chars")
body = readme.split("---", 2)[-1]
print(body[:6000])
for kw in ["noul", "score", "target", "options"]:
    for m in list(re.finditer(kw, body))[:3]:
        print(f"[{kw}]...", body[max(0, m.start() - 150): m.start() + 300].replace("\n", " "))

ts = load_dataset(TS, "default", revision=ts_rev, split="train")
srcs = Counter(ts["source"])
pat = re.compile(r"(?i)massive|copa|belebele|sib|flores|xcopa|amazon|super_?glue|xnli|xstory|paws-x|m_?arc|global|tydi|mlqa|xquad|multiling|vietnam|vi_|_vi\b")
print("\n#### tasksource sources matching suspicious patterns:")
for s, n in sorted(srcs.items()):
    if pat.search(s):
        print(f"  {n:6d}  {s}")
print("n sources", len(srcs), "rows", len(ts))
print("variants:", Counter(ts["variant"]).most_common(15))
print("license_use:", Counter(ts["license_use"]).most_common())
# ví dụ score và noul
for kind in ["score", "noul"]:
    sub = ts.filter(lambda r: r["kind"] == kind).select(range(3))
    for r in sub:
        print(kind, {k: str(r[k])[:200] for k in ["source", "variant", "question", "options", "target"]})

print("\n#### PhoLaya files")
info = api.dataset_info(PL)
print([s.rfilename for s in info.siblings])
man = json.load(open(hf_hub_download(PL, "manifest.json", repo_type="dataset", revision=pl_rev), encoding="utf-8"))
print("manifest:", json.dumps(man, ensure_ascii=False)[:3000])
plr = open(hf_hub_download(PL, "README.md", repo_type="dataset", revision=pl_rev), encoding="utf-8").read()
print("#### PhoLaya README first 5000 chars")
print(plr[:5000])
