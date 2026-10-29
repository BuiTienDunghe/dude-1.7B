"""Phần 5a, bước soi: schema thật của ViFactCheck, ViWikiFC, ViHSD (không đăng nhập), phân bố độ dài bài ViFactCheck,
và kiểm XCOPA test vi ghép với COPA test en theo idx. In tối đa 2 dòng mẫu mỗi bộ, mỗi trường cắt 200 ký tự;
ViHSD không in văn bản (chỉ schema và phân bố nhãn).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import model_revisions  # noqa: E402

api = HfApi()
cut = lambda v: str(v)[:200]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=model_revisions()["Qwen/Qwen3-1.7B"])

for repo, show_text in (("tranthaihoa/vifactcheck", True), ("High-Will/ViWikiFC", True), ("uitnlp/vihsd", False)):
    print("=" * 90)
    try:
        info = api.dataset_info(repo)
        print(repo, "sha", info.sha, "| gated", getattr(info, "gated", None), "| license",
              (info.card_data or {}).get("license") if info.card_data else None)
        print("  files:", [s.rfilename for s in info.siblings][:20])
        cfgs = get_dataset_config_names(repo, revision=info.sha)
        print("  configs:", cfgs)
        ds = load_dataset(repo, cfgs[0] if cfgs else None, revision=info.sha)
    except Exception as e:
        print(repo, "FAIL", type(e).__name__, str(e)[:300]); continue
    print("  splits:", {k: len(v) for k, v in ds.items()})
    first = next(iter(ds.values()))
    print("  features:", first.features)
    for sp, d in ds.items():
        for c in d.column_names:
            if any(s in c.lower() for s in ("label", "verdict", "class", "nhan")):
                print(f"  [{sp}] {c}:", Counter(d[c]).most_common(8))
    if show_text:
        for i in range(2):
            print("  ex:", {k: cut(v) for k, v in first[i].items()})
    if repo == "tranthaihoa/vifactcheck":
        for sp, d in ds.items():
            col = next(c for c in d.column_names if c.lower() in ("context", "content", "article", "full_text", "evidence_full"))
            lens = np.array([len(tok.encode(t or "", add_special_tokens=False)) for t in d[col]])
            print(f"  [{sp}] '{col}' tokens: median {np.median(lens):.0f} p75 {np.percentile(lens,75):.0f} p90 {np.percentile(lens,90):.0f} "
                  f"max {lens.max()} | share>1024 {np.mean(lens>1024):.2f} >2048 {np.mean(lens>2048):.2f} >4000 {np.mean(lens>4000):.2f}")

print("=" * 90)
xs = api.dataset_info("cambridgeltl/xcopa").sha; cs = api.dataset_info("aps/super_glue").sha
x = load_dataset("cambridgeltl/xcopa", "vi", revision="042f78955ba48e6404616762fa6e05e839c3907a", split="test").to_pandas()
c = load_dataset("aps/super_glue", "copa", revision="3de24cf8022e94f4ee4b9d55a6f539891524d646", split="test").to_pandas()
m = x.merge(c, on="idx", suffixes=("_vi", "_en"))
print("xcopa test vi", len(x), "copa test en", len(c), "merged", len(m), "question agree", (m.question_vi == m.question_en).mean(),
      "en labels", Counter(c.label).most_common(3), "changed", int(x.changed.sum()))
for _, r in m.sample(3, random_state=0).iterrows():
    print("  ", r.idx, r.question_vi, "|", cut(r.premise_vi), "||", cut(r.premise_en), "|", cut(r.choice1_vi), "||", cut(r.choice1_en))
