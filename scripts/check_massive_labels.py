"""Liệt kê nhãn MASSIVE theo split để biết đủ 60 ý định ở đâu."""
from huggingface_hub import HfApi
from datasets import load_dataset

s = HfApi().dataset_info("mteb/amazon_massive_intent").sha
sets = {sp: set(load_dataset("mteb/amazon_massive_intent", "en", revision=s, split=sp)["label"]) for sp in ["train", "validation", "test"]}
allv = set().union(*sets.values())
print({k: len(v) for k, v in sets.items()}, "union", len(allv))
print("missing in validation:", sorted(allv - sets["validation"]))
print(sorted(allv))
