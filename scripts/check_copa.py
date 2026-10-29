"""So khớp XCOPA-vi validation với aps/super_glue copa validation theo idx; in license trong dataset card."""
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset

api = HfApi()
xs = api.dataset_info("cambridgeltl/xcopa").sha
cs = api.dataset_info("aps/super_glue").sha
x = load_dataset("cambridgeltl/xcopa", "vi", revision=xs, split="validation").to_pandas()
c = load_dataset("aps/super_glue", "copa", revision=cs, split="validation").to_pandas()
m = x.merge(c, on="idx", suffixes=("_vi", "_en"))
print("merged", len(m), "question agree", (m.question_vi == m.question_en).mean(), "label agree", (m.label_vi == m.label_en).mean())
print(m[(m.question_vi != m.question_en) | (m.label_vi != m.label_en)][["idx", "question_vi", "question_en", "label_vi", "label_en"]])
pd_opts = ["idx", "changed", "question_vi", "label_vi", "premise_vi", "premise_en", "choice1_vi", "choice1_en", "choice2_vi", "choice2_en"]
for _, r in m[m.changed | m.idx.isin([0, 17, 42, 63, 99])].iterrows():
    print("-" * 80)
    for k in pd_opts:
        print(f"  {k:12s} {r[k]}")

readme = open(hf_hub_download("aps/super_glue", "README.md", repo_type="dataset", revision=cs), encoding="utf-8").read()
import re
for mt in re.finditer(r"(?i)licen[sc]", readme):
    print("README>>", readme[max(0, mt.start() - 200): mt.start() + 400].replace("\n", " | "))
    print()
