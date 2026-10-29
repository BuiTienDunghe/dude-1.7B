"""Chỉ mục tham chiếu khử rò rỉ của train_v1, chép nguyên văn mục 1 + 1b của build_stage0.py (lượt 2) thành hàm để dùng lại
(update_stage0_v2.py, decontam_vi.py). build(langs) dựng chỉ mục cho các ngôn ngữ cần; với langs=("vi","en") số văn bản phải trùng
stage0_stats.json["ref_docs"] (check_same_as_stage0 kiểm điều đó)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import DATA, RESULTS, read_jsonl, sha256_file
from decontam import Index, normalize

RAWR = RESULTS / "train_v1" / "raw_refs"


class Both:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def check(self, text):
        return self.a.check(text) or self.b.check(text)


def build(langs=("vi", "en"), with_v0=True, opt_prefixes=("mc_answer",)):
    """opt_prefixes: cột nào là "lựa chọn đáp án" (chỉ trùng nguyên văn / MinHash / 13-gram). Mặc định như build_stage0 (chỉ mc_answer của Belebele)."""
    IX = {"vi": Index(), "en": Index()}
    # Lựa chọn đáp án Belebele (bench + nguồn): chỉ trùng nguyên văn / MinHash / 13-gram, KHÔNG xét "chứa đoạn 4-12 từ".
    IXO = {"vi": Index(), "en": Index()}
    ref_count = Counter()
    seen_ref = defaultdict(set)
    stats = {}

    def add(lang, text, tag, opt=False):
        if lang not in langs:
            return
        if isinstance(text, str) and text.strip():
            ref_count[f"{lang}|{tag}{'|lua_chon' if opt else ''}"] += 1
            k = normalize(text)
            if k not in seen_ref[(lang, opt)]:
                seen_ref[(lang, opt)].add(k); (IXO if opt else IX)[lang].add(text, tag)

    bench = {sp: read_jsonl(DATA / "bench_v1_3" / f) for sp, f in (("dev", "dev_v1_3.jsonl"), ("test", "test_v1.3.jsonl"))}
    rep = Counter(normalize(t) for rs in bench.values() for r in rs for t in [r["question"]] + list(r["options"]))
    skipped_tpl = set()
    for sp, rs in bench.items():
        for r in rs:
            lang = "vi" if r["lang"] == "vi" else "en"
            add(lang, r["state"], f"bench13:{sp}:{r['source']}")
            for j, t in enumerate([r["question"]] + list(r["options"])):
                if rep[normalize(t)] >= 3:
                    skipped_tpl.add(normalize(t)); continue
                add(lang, t, f"bench13:{sp}:{r['source']}", opt=j > 0 and r["source"] != "xcopa")
    stats["bench_template_strings_skipped"] = len(skipped_tpl)

    def rd(repo, f):
        p = RAWR / repo.replace("/", "__") / f
        s = str(p)
        if s.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif s.endswith(".csv"):
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        elif s.endswith(".tsv"):
            df = pd.read_csv(p, sep="\t", quoting=3, dtype=str, keep_default_na=False)
        else:
            df = pd.read_json(p, lines=True)
        return df, sha256_file(p)

    ref_files = {}

    def ref(repo, files, cols_by_lang, tag):
        for f in files:
            df, h = rd(repo, f)
            ref_files[f"{repo}/{f}"] = {"rows": len(df), "sha256": h}
            for lang, cols in cols_by_lang(f).items():
                for c in cols:
                    if c in df.columns:
                        for t in df[c]:
                            add(lang, t, f"{tag}:{f.split('/')[0] if '/' in f else f}", opt=c.startswith(tuple(opt_prefixes)))

    def ls(repo):
        return sorted(str(p.relative_to(RAWR / repo.replace("/", "__"))).replace("\\", "/") for p in (RAWR / repo.replace("/", "__")).rglob("*")
                      if p.is_file() and p.suffix in (".parquet", ".csv", ".tsv", ".jsonl", ".gz") and ".cache" not in p.parts)

    ref("tranthaihoa/vifactcheck", ls("tranthaihoa/vifactcheck"), lambda f: {"vi": ["Statement", "Context", "Evidence"]}, "ref:vifactcheck")
    ref("High-Will/ViWikiFC", ls("High-Will/ViWikiFC"), lambda f: {"vi": ["claim", "evidence", "context"]}, "ref:viwikifc")

    def mlqa_cols(f):
        cfg = f.split("/")[0]
        if cfg.startswith("mlqa-translate-test"):
            return {"en": ["context", "question"]}
        if cfg.startswith("mlqa-translate-train"):
            return {"vi": ["context", "question"]}
        c, q = cfg.split(".")[1:3]
        out = defaultdict(list)
        if c in ("vi", "en"):
            out[c].append("context")
        if q in ("vi", "en"):
            out[q].append("question")
        return dict(out)

    ref("facebook/mlqa", ls("facebook/mlqa"), mlqa_cols, "ref:mlqa")
    lang_of = lambda f: "vi" if "vie" in f.lower() else "en"
    GP = ["prompt", "solution0", "solution1", "solution2", "solution3"]
    ref("mrlbenchmarks/global-piqa-nonparallel", ls("mrlbenchmarks/global-piqa-nonparallel"),
        lambda f: {lang_of(f): GP, "en": (GP if lang_of(f) == "en" else []) + ["eng_translated0", "eng_translated1"]}, "ref:gpiqa_np")
    ref("mrlbenchmarks/global-piqa-parallel", ls("mrlbenchmarks/global-piqa-parallel"),
        lambda f: {lang_of(f): GP, "en": (GP if lang_of(f) == "en" else []) + ["eng_prompt"] + [f"eng_solution{i}" for i in range(4)]}, "ref:gpiqa_p")
    ref("gsarti/flores_101", ls("gsarti/flores_101"), lambda f: {lang_of(f): ["sentence"]}, "ref:flores101")
    BB = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
    ref("facebook/belebele", ls("facebook/belebele"), lambda f: {lang_of(f): BB}, "ref:belebele")
    ref("Davlan/sib200", ls("Davlan/sib200"), lambda f: {lang_of(f): ["text"]}, "ref:sib200")
    ref("mteb/amazon_massive_intent", ls("mteb/amazon_massive_intent"), lambda f: {"vi" if f.endswith("vi.json.gz") else "en": ["text"]}, "ref:massive")
    CP = ["premise", "choice1", "choice2"]
    # chỉ vi + translation-vi như build_stage0 (file XCOPA các ngôn ngữ khác tải thêm ở raw_refs/xcopa_all, dùng riêng cho kiểm nguồn nhân quả)
    ref("cambridgeltl/xcopa", [f for f in ls("cambridgeltl/xcopa") if f.split("/")[0] in ("vi", "translation-vi")],
        lambda f: {"en" if f.startswith("translation") else "vi": CP}, "ref:xcopa")
    ref("aps/super_glue", ls("aps/super_glue"), lambda f: {"en": CP}, "ref:copa")
    ref("pkavumba/balanced-copa", ls("pkavumba/balanced-copa"), lambda f: {"en": CP}, "ref:balanced_copa")
    for ix in IXO.values():
        ix.short.clear()
    stats["ref_docs"] = {"vi": IX["vi"].n, "en": IX["en"].n, "lua_chon_vi": IXO["vi"].n, "lua_chon_en": IXO["en"].n}
    stats["ref_units_by_tag"] = dict(sorted(ref_count.items()))
    stats["ref_files"] = ref_files
    CHK = {lang: Both(IX[lang], IXO[lang]) for lang in IX}
    V0 = None
    if with_v0:
        V0 = Index()
        for r in read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl"):
            V0.add(r["state"], f"v0:{r['split']}")
    return CHK, V0, stats


def check_same_as_stage0(stats, langs=("vi", "en")):
    """Số văn bản và tag phải trùng lượt 2 của build_stage0 (cùng code, cùng file tham chiếu)."""
    s0 = json.load(open(RESULTS / "train_v1" / "ban_v1" / "stage0_stats.json", encoding="utf-8"))   # bản lưu của lượt 2 build_stage0
    for lang in langs:
        for k in (lang, f"lua_chon_{lang}"):
            assert stats["ref_docs"][k] == s0["ref_docs"][k], (k, stats["ref_docs"][k], s0["ref_docs"][k])
        a = {k: v for k, v in stats["ref_units_by_tag"].items() if k.startswith(lang + "|")}
        b = {k: v for k, v in s0["ref_units_by_tag"].items() if k.startswith(lang + "|")}
        assert a == b, lang
    return True
