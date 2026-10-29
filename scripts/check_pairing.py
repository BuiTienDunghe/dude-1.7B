"""Kiểm tra khóa ghép Anh–Việt của từng bộ và tìm COPA tiếng Anh khớp với XCOPA-vi theo idx."""
from collections import Counter
from huggingface_hub import HfApi
from datasets import load_dataset

api = HfApi()
sha = lambda r: api.dataset_info(r).sha

# --- Belebele: khóa câu = (link, question_number); khóa đoạn = link
r = "facebook/belebele"; s = sha(r)
vi = load_dataset(r, "vie_Latn", revision=s, split="test").to_pandas()
en = load_dataset(r, "eng_Latn", revision=s, split="test").to_pandas()
for name, df in [("vi", vi), ("en", en)]:
    print(f"belebele {name}: rows={len(df)} links={df.link.nunique()} passages={df.flores_passage.nunique()}"
          f" dup(link,qn)={df.duplicated(['link','question_number']).sum()}")
# link có nhiều đoạn văn khác nhau không?
multi = en.groupby("link").flores_passage.nunique()
print("  links with >1 distinct EN passage:", (multi > 1).sum())
m = vi.merge(en, on=["link", "question_number"], suffixes=("_vi", "_en"))
print("  merged rows:", len(m), "answer agree:", (m.correct_answer_num_vi == m.correct_answer_num_en).mean())
# đoạn văn trong cùng một link phải ánh xạ 1-1 giữa hai ngôn ngữ
pp = m.groupby("flores_passage_en").flores_passage_vi.nunique()
print("  EN passage -> #VI passages max:", pp.max())

# --- SIB-200: khóa = index_id
r = "Davlan/sib200"; s = sha(r)
for split in ["train", "validation", "test"]:
    a = load_dataset(r, "vie_Latn", revision=s, split=split).to_pandas()
    b = load_dataset(r, "eng_Latn", revision=s, split=split).to_pandas()
    mm = a.merge(b, on="index_id", suffixes=("_vi", "_en"))
    print(f"sib200 {split}: vi={len(a)} en={len(b)} merged={len(mm)} cat agree={(mm.category_vi==mm.category_en).mean():.3f}"
          f" uniq ids={a.index_id.is_unique}")
    if split == "validation":
        print("  labels:", Counter(a.category))
# chồng lấn index_id giữa các split
ids = {sp: set(load_dataset(r, "eng_Latn", revision=s, split=sp)["index_id"]) for sp in ["train", "validation", "test"]}
print("  overlap train&test", len(ids["train"] & ids["test"]), "val&test", len(ids["validation"] & ids["test"]))

# --- MASSIVE (mirror mteb): khóa = id
r = "mteb/amazon_massive_intent"; s = sha(r)
a = load_dataset(r, "vi", revision=s, split="validation").to_pandas()
b = load_dataset(r, "en", revision=s, split="validation").to_pandas()
mm = a.merge(b, on="id", suffixes=("_vi", "_en"))
print(f"massive val: vi={len(a)} en={len(b)} merged={len(mm)} label agree={(mm.label_vi==mm.label_en).mean():.4f}"
      f" n_labels={a.label.nunique()} uniq ids={a.id.is_unique}")
print("  label counts min/max:", a.label.value_counts().min(), a.label.value_counts().max())
print(mm.sample(5, random_state=0)[["id", "label_vi", "text_vi", "text_en"]].to_string())
tr = set(load_dataset(r, "en", revision=s, split="train")["id"]); te = set(load_dataset(r, "en", revision=s, split="test")["id"])
print("  id overlap val&train", len(set(b.id) & tr), "val&test", len(set(b.id) & te))

# --- XCOPA vi + ứng viên COPA tiếng Anh
r = "cambridgeltl/xcopa"; s = sha(r)
x = load_dataset(r, "vi", revision=s, split="validation").to_pandas()
print("xcopa vi val:", len(x), "idx range", x.idx.min(), x.idx.max(), "changed:", x.changed.sum(), Counter(x.question))
for repo, cfg in [("aps/super_glue", "copa"), ("super_glue", "copa"), ("pkavumba/balanced-copa", None)]:
    try:
        info = api.dataset_info(repo)
        print("CANDIDATE", repo, info.sha, "license", info.card_data.get("license") if info.card_data else None)
        d = load_dataset(repo, cfg, revision=info.sha) if cfg else load_dataset(repo, revision=info.sha)
        print("  splits", {k: len(v) for k, v in d.items()}, d[list(d)[0]].features)
    except Exception as e:
        print("CANDIDATE", repo, "FAIL", type(e).__name__, str(e)[:200])
