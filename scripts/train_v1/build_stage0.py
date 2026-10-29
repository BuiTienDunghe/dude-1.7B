"""Giai đoạn 0 train_v1 (KHÔNG train, KHÔNG dịch): chọn dòng phần A (tiếng Việt bản địa) và phần B (tiếng Anh cần dịch), khử rò rỉ, dựng sẵn thành phần A/B/C.

Recast: recast_v0 (ts_row, pl_row, bộ lọc finalize lấy nguyên văn từ build_train_v0.py; verify_recast_v0.py dựng lại đúng 21.000/21.000 dòng v0).
Nguồn mới (VSMRC, EXAMS-vi, vi-tourism-admin-intent, VNTC) dựng theo đúng lược đồ dòng của v0 rồi qua cùng finalize.
Khử rò rỉ: scripts/decontam.py (exact / MinHash >= 0,8 / chứa 13-gram, đoạn ngắn 4-12 từ) với
  - toàn bộ văn bản dev 1.3 + test 1.3 (state; câu hỏi và lựa chọn trừ chuỗi mẫu lặp >= 3 lần), mọi bản (EN, VI, VI không dấu);
  - mọi split của Belebele, FLORES (bản flores_101 không khóa), SIB-200, MASSIVE, XCOPA (vi + translation-vi), COPA, Balanced-COPA, ViFactCheck,
    ViWikiFC, MLQA (mọi cấu hình có tiếng Việt + en.en), Global-PIQA (nonparallel + parallel, vie/eng);
  - train_v0 + val_v0 (tránh văn bản đã dùng).
  Phần A so với chỉ mục tiếng Việt, phần B với chỉ mục tiếng Anh (phần B khử lại phía tiếng Việt khi nhận bản dịch).
Giới hạn 5%/nguồn (mức tác vụ như v0) tính trên tổng của thành phần nhỏ nhất chứa nguồn đó (A cho phần A, B cho phần B); C lớn hơn nên tự đạt.
Ra results/train_v1/: part_a.jsonl, part_b_en.jsonl, part_b_meta.jsonl, to_translate/part_b_to_translate.jsonl, to_translate/chuoi_lap_lai.json,
components/A.jsonl, components/{B,C}_plan.json, v0_leak.json, stage0_stats.json. Không ghi data/. Không chạy model.
QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, DATA, RESULTS, SEED, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, write_jsonl  # noqa: E402
from decontam import Index, normalize  # noqa: E402
from recast_v0 import make_finalize, pl_row, ts_row  # noqa: E402

OUT = RESULTS / "train_v1"
RAWR, RAWS = OUT / "raw_refs", OUT / "raw_src"
CAP = 0.05
man0 = load_json(DATA / "manifest_train_v0.json")
TS = (man0["sources"]["tasksource"]["repo"], man0["sources"]["tasksource"]["revision"])
tr = load_json(CONFIG / "translations.json")
props = {p["id"]: p for p in map(json.loads, (CONFIG / "vi_proposals.jsonl").read_text(encoding="utf-8").splitlines()) if p}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])   # bộ đếm state của v0
rng = np.random.default_rng(SEED)
stats = {"seed": SEED}


def nfc(s):
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s


# ================================================================ 1) chỉ mục tham chiếu
IX = {"vi": Index(), "en": Index()}
# Lựa chọn đáp án Belebele (bench + nguồn): chỉ trùng nguyên văn / MinHash / 13-gram, KHÔNG xét "chứa đoạn 4-12 từ" (lượt đầu cho thấy phép đó bắt
# cụm từ chung như "in the united states" trong 108 dòng v0). State, câu hỏi và văn bản khác giữ đủ 4 phép như decontam.py.
IXO = {"vi": Index(), "en": Index()}
ref_count = Counter()
seen_ref = defaultdict(set)


class Both:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def check(self, text):
        return self.a.check(text) or self.b.check(text)


CHK = {lang: Both(IX[lang], IXO[lang]) for lang in IX}


def add(lang, text, tag, opt=False):
    """Thêm một văn bản tham chiếu; văn bản đã có (sau chuẩn hóa) chỉ đếm, không thêm lần hai (cùng kết quả kiểm, đỡ tính MinHash)."""
    if isinstance(text, str) and text.strip():
        ref_count[f"{lang}|{tag}{'|lua_chon' if opt else ''}"] += 1
        k = normalize(text)
        if k not in seen_ref[(lang, opt)]:
            seen_ref[(lang, opt)].add(k); (IXO if opt else IX)[lang].add(text, tag)


# 1a) bench 1.3: mọi dòng dev + test; câu hỏi / lựa chọn lặp >= 3 lần là chuỗi mẫu (không phải dữ liệu) nên bỏ khỏi chỉ mục
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
                        add(lang, t, f"{tag}:{f.split('/')[0] if '/' in f else f}", opt=c.startswith("mc_answer"))


def ls(repo):
    return sorted(str(p.relative_to(RAWR / repo.replace("/", "__"))).replace("\\", "/") for p in (RAWR / repo.replace("/", "__")).rglob("*")
                  if p.is_file() and p.suffix in (".parquet", ".csv", ".tsv", ".jsonl", ".gz") and ".cache" not in p.parts)


ref("tranthaihoa/vifactcheck", ls("tranthaihoa/vifactcheck"), lambda f: {"vi": ["Statement", "Context", "Evidence"]}, "ref:vifactcheck")
ref("High-Will/ViWikiFC", ls("High-Will/ViWikiFC"), lambda f: {"vi": ["claim", "evidence", "context"]}, "ref:viwikifc")


def mlqa_cols(f):
    cfg = f.split("/")[0]
    if cfg.startswith("mlqa-translate-test"):          # MLQA vi test dịch sang tiếng Anh
        return {"en": ["context", "question"]}
    if cfg.startswith("mlqa-translate-train"):         # SQuAD dịch sang tiếng Việt
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
ref("cambridgeltl/xcopa", ls("cambridgeltl/xcopa"), lambda f: {"en" if f.startswith("translation") else "vi": CP}, "ref:xcopa")
ref("aps/super_glue", ls("aps/super_glue"), lambda f: {"en": CP}, "ref:copa")
ref("pkavumba/balanced-copa", ls("pkavumba/balanced-copa"), lambda f: {"en": CP}, "ref:balanced_copa")
for ix in IXO.values():
    ix.short.clear()
stats["ref_docs"] = {"vi": IX["vi"].n, "en": IX["en"].n, "lua_chon_vi": IXO["vi"].n, "lua_chon_en": IXO["en"].n}
stats["ref_units_by_tag"] = dict(sorted(ref_count.items()))
stats["ref_files"] = ref_files
print("ref index", stats["ref_docs"], flush=True)

# 1b) văn bản đã dùng: train_v0 + val_v0
V0 = Index()
train0, val0 = read_jsonl(DATA / "train_v0.jsonl"), read_jsonl(DATA / "val_v0.jsonl")
for r in train0 + val0:
    V0.add(r["state"], f"v0:{r['split']}")

# ================================================================ 2) train_v0 / val_v0 so với chỉ mục mở rộng (v0 chỉ từng so với dev_v0 + phần để dành)
v0_hits = []
for r in train0 + val0:
    units = [("state", r["state"])] + [(f, t) for f, t in [("question", r["question"] or "")] + [("option", o) for o in r["options"]] if len(normalize(t).split()) >= 4]
    for lang in ("vi", "en"):
        for field, t in units:
            h = CHK[lang].check(t)
            if h:
                v0_hits.append({"id": r["id"], "split": r["split"], "source": r["source"], "lang": r["lang"], "index": lang, "field": field, "kind": h[0], "tag": h[1]})
                break
        else:
            continue
        break
drop_v0 = {h["id"] for h in v0_hits if h["split"] == "train" and h["lang"] == "vi"}   # phần VI của v0 được bỏ; 16.000 dòng tasksource giữ nguyên
(OUT / "v0_leak.json").write_text(json.dumps({"hits": v0_hits, "bo_khoi_thanh_phan": sorted(drop_v0),
                                              "giu_lai_vi_quy_tac_16000_EN": sorted({h["id"] for h in v0_hits if h["split"] == "train" and h["lang"] != "vi"})},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
base0 = [r for r in train0 if r["id"] not in drop_v0]
stats["v0_vs_ref"] = {"hits": len(v0_hits), "by": dict(Counter(f"{h['split']}|{h['lang']}|{h['source']}|{h['kind']}|{h['tag']}" for h in v0_hits)),
                      "bo_khoi_thanh_phan": len(drop_v0), "base_rows": len(base0)}
print("v0 hits", len(v0_hits), flush=True)

# ================================================================ 3) phần A: tiếng Việt bản địa
dropsA = defaultdict(Counter)
finA = {}


def finalizer(src, lang):
    if src not in finA:
        finA[src] = make_finalize(tok, [(f"ref_{lang}", CHK[lang]), ("v0", V0)], dropsA[src], max_state_tokens=1024)  # dựng với 1.024 như v0
    return finA[src]


def onehot(k, n):
    return [1.0 if i == k else 0.0 for i in range(n)]


def P(pid):
    return props[pid]["gia_tri_de_xuat"]


REV = {"VSMRC/mrc": "13c4449cdee58b62dd9bbca555fcbc0cc2630b38", "mhardalov/exams": "4ff10804abb3341f8815cacd778181177bba7edd",
       "phamluan/vi-tourism-admin-intent": "a092e219b0bf9a08e736f4425b8ebe0717029f20", "kornwtp/VNTC-10Topics": "30b4f927d8c62b130d49621f511d61a84d588d20",
       "LakoreAI/PhoLaya-base": "034d4273fb2600582cb789c45defe86b1c07e12d", "uitnlp/ViANLI (thẻ giấy phép)": "0fec8d6ecb043a61c609f9b51f80401fdf1e84d3"}
LIC = {
    "vsmrc/mrc": ("apache-2.0 (thẻ HF); văn bản lấy từ Wikipedia tiếng Việt vốn CC BY-SA", "commercial (thẻ) / văn bản CC BY-SA"),
    "exams/vi": ("cc-by-sa-4.0 (thẻ HF)", "commercial (CC BY-SA 4.0)"),
    "vi-tourism-admin-intent": ("cc-by-4.0 (thẻ HF)", "commercial (CC BY 4.0)"),
    "vntc": ("mit (thẻ bản sao HF kornwtp/VNTC-10Topics; bản gốc GitHub duyvuleo/VNTC); văn bản tin báo 2007", "không rõ (thẻ mit trên bản sao; văn bản báo)"),
    "pholaya/vianli/entailment": ("PhoLaya-base: other (research-use-only); ViANLI gốc: thẻ mit, thân card CC BY-NC-SA 4.0 (mâu thuẫn)",
                                  "research-only (PhoLaya-base, UIT corpora); gốc non-commercial (CC BY-NC-SA 4.0, chặt hơn thẻ mit)"),
}
NHAN = {"vsmrc/mrc": "llm_sinh", "exams/vi": "nguoi_gan", "vi-tourism-admin-intent": "mau_cau", "vntc": "nguoi_gan", "pholaya/vianli/entailment": "nguoi_gan"}
NHAN_GHICHU = {"vsmrc/mrc": "câu hỏi + phương án do LLM sinh, người kiểm (card/bài báo arXiv 2506.15978)",
               "exams/vi": "đề thi thật (EXAMS, Hardalov và cộng sự 2020)",
               "vi-tourism-admin-intent": "sinh từ mẫu câu, nhãn đúng theo mẫu (label_source=synthetic_template); bỏ 334 câu keyword_weak",
               "vntc": "chuyên mục báo do tòa soạn gán (VNTC 2007)",
               "pholaya/vianli/entailment": "giả thuyết do người viết, vòng đối kháng người-máy, kiểm kép (card ViANLI)"}
DANG = {"vsmrc/mrc": "doc_hieu", "exams/vi": "doc_hieu_kien_thuc", "vi-tourism-admin-intent": "y_dinh", "vntc": "chu_de", "pholaya/vianli/entailment": "yes_no"}
DSREV = {"vsmrc/mrc": "VSMRC/mrc", "exams/vi": "mhardalov/exams", "vi-tourism-admin-intent": "phamluan/vi-tourism-admin-intent", "vntc": "kornwtp/VNTC-10Topics",
         "pholaya/vianli/entailment": "LakoreAI/PhoLaya-base"}


def extra(src, pending=()):
    return {"license_verbatim": LIC[src][0], "license_use": LIC[src][1], "nguon_nhan": NHAN[src], "nguon_nhan_ghi_chu": NHAN_GHICHU[src], "dang_bai": DANG[src],
            "dataset_revision": f"{DSREV[src]}@{REV[DSREV[src]]}", "chuoi_cho_duyet": list(pending), "phan": "A"}


def cand_vsmrc():
    df = pd.read_json(RAWS / "VSMRC__mrc/train.jsonl", lines=True)
    for i, x in df.iterrows():
        opts = [nfc(o) for o in x["choices"]]
        yield {"id": f"vsmrc:train:{i}", "source": "vsmrc/mrc", "dataset": "VSMRC/mrc", "lang": "vi", "text_lang": "vi", "kind": "choice",
               "state": nfc(x["context"]), "question": nfc(x["question"]), "options": opts, "target": onehot(int(x["correctchoice"]), len(opts)),
               "group_keys": [f"f:{normalize(x['context'])}"], "variant": x["questiontype"], **extra("vsmrc/mrc")}


def cand_exams():
    df = pd.read_parquet(RAWS / "mhardalov__exams/crosslingual_vi/train-00000-of-00001.parquet")
    for _, x in df.iterrows():
        q, info = x["question"], x["info"]
        labels, texts = list(q["choices"]["label"]), [nfc(t) for t in q["choices"]["text"]]
        subj = f"tv1-exams-subj-{info['subject'].lower()}"
        yield {"id": f"exams:vi:{x['id']}", "source": "exams/vi", "dataset": "mhardalov/exams", "lang": "vi", "text_lang": "vi", "kind": "choice",
               "state": P("tv1-exams-state").format(mon=P(subj), lop=int(info["grade"])), "question": nfc(q["stem"]), "options": texts,
               "target": onehot(labels.index(x["answerKey"]), len(texts)), "group_keys": [f"q:{normalize(q['stem'])}"], "variant": f"{info['subject']}/{info['grade']}",
               **extra("exams/vi", ["tv1-exams-state", subj])}


INTENTS = ["ask_admin_change", "ask_location_mapping", "ask_new_name", "ask_old_name", "compare_before_after_merge", "recommend_tourist_spot",
           "search_tourist_address", "search_tourist_by_old_or_new"]


def cand_intent():
    df = pd.read_json(RAWS / "phamluan__vi-tourism-admin-intent/data/train.jsonl", lines=True)
    opts = [P(f"tv1-intent-{k}") for k in INTENTS]
    for _, x in df.iterrows():
        if x["label_source"] != "synthetic_template":
            dropsA["vi-tourism-admin-intent"]["label_source_keyword_weak"] += 1; continue
        yield {"id": f"tourism_intent:train:{x['id']}", "source": "vi-tourism-admin-intent", "dataset": "phamluan/vi-tourism-admin-intent", "lang": "vi",
               "text_lang": "vi", "kind": "choice", "state": nfc(x["text"]), "question": tr["questions"]["massive"]["vi"], "options": opts,
               "target": onehot(INTENTS.index(x["label"]), len(opts)), "group_keys": [], "variant": x["label"],
               **extra("vi-tourism-admin-intent", [f"tv1-intent-{k}" for k in INTENTS])}


VNTC = ["Chinh tri Xa hoi", "Doi song", "Khoa hoc", "Kinh doanh", "Phap luat", "Suc khoe", "The gioi", "The thao", "Van hoa", "Vi tinh"]


def cand_vntc():
    df = pd.read_csv(RAWS / "kornwtp__VNTC-10Topics/train.csv", dtype=str, keep_default_na=False)
    pids = [f"tv1-vntc-{k.lower().replace(' ', '_')}" for k in VNTC]
    opts = [P(p) for p in pids]
    for i, x in df.iterrows():
        yield {"id": f"vntc:train:{i}", "source": "vntc", "dataset": "kornwtp/VNTC-10Topics", "lang": "vi", "text_lang": "vi", "kind": "choice",
               "state": nfc(x["text"]), "question": tr["questions"]["sib200"]["vi"], "options": opts, "target": onehot(VNTC.index(x["label"]), len(opts)),
               "group_keys": [], "variant": x["label"], **extra("vntc", pids)}


def cand_vianli():
    with open(RAWS / "LakoreAI__PhoLaya-base/sources/vianli/train.jsonl", encoding="utf-8") as f:
        for n, line in enumerate(f):
            r = pl_row(json.loads(line), "vianli", n)
            if r["source"] == "pholaya/vianli/entailment":
                r.pop("license_use")
                yield {**r, **extra("pholaya/vianli/entailment")}


def take(gen, src, lang, quota, seen_keys, seen_text, excl_groups=frozenset()):
    """Xáo cả kho theo SEED, lấy quota dòng đầu tiên qua finalize; một dòng mỗi nhóm văn bản; không trùng state+câu hỏi trong cùng phần."""
    pool = list(gen)
    order = rng.permutation(len(pool))
    fin, got, tried = finalizer(src, lang), [], 0
    for j in order:
        if len(got) >= quota:
            break
        r = pool[int(j)]; tried += 1
        if any(k in seen_keys or k in excl_groups for k in r["group_keys"]):
            dropsA[src]["cung_nhom_van_ban"] += 1; continue
        key = (normalize(r["state"]), normalize(r["question"] or ""))
        if key in seen_text:
            dropsA[src]["trung_state_cau_hoi"] += 1; continue
        r = fin(r)
        if r:
            got.append(r); seen_keys.update(r["group_keys"]); seen_text.add(key)
    return got, {"kho": len(pool), "da_xet": tried, "lay": len(got), "chi_tieu": quota}


QA = {"exams/vi": (cand_exams, 500), "vi-tourism-admin-intent": (cand_intent, 1000), "pholaya/vianli/entailment": (cand_vianli, 1000), "vntc": (cand_vntc, 1000)}
partA, infoA, sk, st = [], {}, set(), set()
for src, (gen, q) in QA.items():
    rows, infoA[src] = take(gen(), src, "vi", q, sk, st); partA += rows
    print("A", src, infoA[src], flush=True)
# VSMRC (đề xuất 2.000) bị giới hạn 5%: x <= 0,05 * (base + phần A khác + x)
others = len(partA)
cap_vsmrc = int(CAP * (len(base0) + others) / (1 - CAP))
rows, infoA["vsmrc/mrc"] = take(cand_vsmrc(), "vsmrc/mrc", "vi", min(2000, cap_vsmrc), sk, st); partA += rows
infoA["vsmrc/mrc"]["de_xuat_brief"] = 2000
infoA["vsmrc/mrc"]["tran_5pct"] = cap_vsmrc
print("A vsmrc", infoA["vsmrc/mrc"], flush=True)
for r in partA:
    r["split"] = "train"; r["target_argmax"] = int(np.argmax(r["target"])); r.pop("group_keys", None)
nA = len(base0) + len(partA)
shareA = Counter(r["source"] for r in base0 + partA)
assert max(shareA.values()) <= CAP * nA, shareA.most_common(3)

# ================================================================ 4) phần B: tasksource @ revision v0, dòng chưa dùng
GROUPS = {
    "doc_hieu": (2000, ["race/high", "race/middle", "quail", "cosmos_qa", "dream"]),
    "y_dinh": (1500, ["banking77", "clinc_oos/plus", "multidogo/airline/intent", "multidogo/fastfood/intent", "multidogo/finance/intent",
                      "multidogo/insurance/intent", "multidogo/media/intent", "multidogo/software/intent"]),
    "kiem_chung": (2000, ["vitaminc", "nli_fever", "health_fact"]),
    "yes_no": (800, ["super_glue/boolq", "super_glue/boolq_passage", "sharc"]),
}
FORBID = ("intentgrasp", "slurp", "massive", "copa", "xcopa")
assert not any(f in s.lower() for _, (_, ss) in GROUPS.items() for s in ss for f in FORBID)
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
src_col, id_col, gid_col = ts["source"], ts["id"], ts["group_id"]
pos = {i: k for k, i in enumerate(id_col)}
used_ids = {r["id"][3:] for r in train0 + val0 if r["dataset"] == "tasksource"}
used_groups = {gid_col[pos[i]] for i in used_ids}
v0_src = Counter(r["source"] for r in base0)
nB_plan = len(base0) + sum(q for q, _ in GROUPS.values())
capB = int(CAP * nB_plan)
dropsB = defaultdict(Counter)
partB, infoB = [], {}
seenB = set()
for g, (quota, srcs) in GROUPS.items():
    gens, info = {}, {}
    for s in srcs:
        idx = [k for k, x in enumerate(src_col) if x == s]
        info[s] = {"kho": len(idx), "da_dung_v0": sum(1 for k in idx if id_col[k] in used_ids)}
        idx = [k for k in idx if id_col[k] not in used_ids and gid_col[k] not in used_groups]
        info[s]["chua_dung"] = len(idx)
        order = rng.permutation(len(idx))
        fin = make_finalize(tok, [("ref_en", CHK["en"]), ("v0", V0)], dropsB[s], max_state_tokens=1024)  # dựng với 1.024 như v0

        def gen(idx=idx, order=order, fin=fin, s=s):
            for j in order:
                r = ts_row(ts[int(idx[int(j)])])
                key = (normalize(r["state"]), normalize(r["question"] or ""), tuple(r["options"]))
                if key in seenB:
                    dropsB[s]["trung_trong_phan_B"] += 1; continue
                r = fin(r)
                if r:
                    seenB.add(key)
                    yield r
        gens[s] = gen()
    got = defaultdict(list)
    active = list(srcs)
    while sum(map(len, got.values())) < quota and active:      # chia đều vòng tròn; nguồn hết dòng hoặc chạm trần thì nhường cho nguồn khác
        for s in list(active):
            if sum(map(len, got.values())) >= quota:
                break
            if v0_src[s] + len(got[s]) >= capB:
                active.remove(s); continue
            r = next(gens[s], None)
            if r is None:
                active.remove(s); continue
            got[s].append(r)
    for s in srcs:
        info[s]["lay"] = len(got[s]); info[s]["v0_trong_train"] = v0_src[s]
        info[s]["kinds"] = dict(Counter(r["kind"] for r in got[s]))
        for r in got[s]:
            r.update(split="train", target_argmax=int(np.argmax(r["target"])), group_id=r.pop("group_keys")[0][2:], phan="B", dang_bai=g,
                     nguon_nhan="nguoi_gan", license_verbatim=ts[pos[r["id"][3:]]]["license"], dataset_revision=f"{TS[0]}@{TS[1]}")
            partB.append(r)
    infoB[g] = {"chi_tieu": quota, "lay": sum(map(len, got.values())), "nguon": info}
    print("B", g, infoB[g]["lay"], flush=True)
nB = len(base0) + len(partB)
shareB = Counter(r["source"] for r in base0 + partB)
assert max(shareB.values()) <= CAP * nB, shareB.most_common(3)
nC = len(base0) + len(partA) + len(partB)
shareC = Counter(r["source"] for r in base0 + partA + partB)
assert max(shareC.values()) <= CAP * nC

# ================================================================ 5) kiểm cuối: 0 trùng với tham chiếu (state, câu hỏi, lựa chọn >= 4 từ)
final = Counter()
for rows, lang in ((partA, "vi"), (partB, "en"), (base0, "vi"), (base0, "en")):
    for r in rows:
        for t in [r["state"]] + [t for t in [r["question"] or ""] + r["options"] if len(normalize(t).split()) >= 4]:
            h = CHK[lang].check(t)
            if h:
                final[f"{r.get('phan', 'v0')}|{lang}|{h[0]}|{h[1]}"] += 1; break
stats["final_overlap"] = dict(final)
assert not any(k.startswith(("A|", "B|")) for k in final), final

# ================================================================ 6) ghi phần A, phần B, file cần dịch
write_jsonl(OUT / "part_a.jsonl", partA)
write_jsonl(OUT / "part_b_en.jsonl", partB)
META = ["id", "source", "dataset", "dataset_revision", "kind", "variant", "target", "target_argmax", "license_use", "license_verbatim", "nguon_nhan", "dang_bai",
        "group_id", "n_state_tokens", "text_lang"]
write_jsonl(OUT / "part_b_meta.jsonl", [{k: r[k] for k in META} for r in partB])
TT = OUT / "to_translate"
exp = [{"id": r["id"], "state": r["state"], "question": r["question"] or "", "options": [] if r["kind"] == "noul" else list(r["options"])} for r in partB]
write_jsonl(TT / "part_b_to_translate.jsonl", exp)
qc = Counter(x["question"] for x in exp)
oc = Counter(o for x in exp for o in x["options"])
rep_strings = {"cau_hoi_lap_lai": {q: n for q, n in qc.most_common() if n >= 5}, "lua_chon_lap_lai": {o: n for o, n in oc.most_common() if n >= 5}}
(TT / "chuoi_lap_lai.json").write_text(json.dumps(rep_strings, ensure_ascii=False, indent=1), encoding="utf-8")
ntok = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0
tt_tok = {"state": sum(ntok(x["state"]) for x in exp), "question": sum(ntok(x["question"]) for x in exp), "options": sum(ntok(o) for x in exp for o in x["options"])}
uniq = {x["state"] for x in exp} | {x["question"] for x in exp} | {o for x in exp for o in x["options"]}
stats["to_translate"] = {"rows": len(exp), "noul_rows_khong_co_lua_chon": sum(1 for x in exp if not x["options"]), "tokens_en": {**tt_tok, "tong": sum(tt_tok.values())},
                         "chuoi_khac_nhau": len(uniq), "tokens_en_chuoi_khac_nhau": sum(ntok(s) for s in uniq),
                         "cau_hoi_khac_nhau": len(qc), "lua_chon_khac_nhau": len(oc),
                         "sha256": sha256_file(TT / "part_b_to_translate.jsonl")}

# ================================================================ 7) thành phần
write_jsonl(OUT / "components/A.jsonl", base0 + partA)
plan = lambda name, parts: {"name": name, "base_train_v0": {"file": "data/train_v0.jsonl", "sha256": sha256_file(DATA / "train_v0.jsonl"), "rows": len(base0),
                                                            "bo_ra": sorted(drop_v0)},
                            "parts": parts, "rows_total": len(base0) + sum(p["rows"] for p in parts.values())}
pA = {"A": {"file": "results/train_v1/part_a.jsonl", "sha256": sha256_file(OUT / "part_a.jsonl"), "rows": len(partA)}}
pB = {"B": {"file_en": "results/train_v1/part_b_en.jsonl", "sha256_en": sha256_file(OUT / "part_b_en.jsonl"), "rows": len(partB),
            "trang_thai": "CHỜ BẢN DỊCH: state/câu hỏi/lựa chọn tiếng Việt lấy từ bản dịch đã qua kiểm định dạng và khử rò rỉ phía tiếng Việt; nhãn/target giữ theo part_b_meta.jsonl"}}
(OUT / "components/B_plan.json").write_text(json.dumps(plan("B", pB), ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "components/C_plan.json").write_text(json.dumps(plan("C", {**pA, **pB}), ensure_ascii=False, indent=1), encoding="utf-8")
stats["components"] = {"A": {"rows": nA, "max_source_share": round(max(shareA.values()) / nA, 4), "sha256": sha256_file(OUT / "components/A.jsonl")},
                       "B": {"rows": nB, "max_source_share": round(max(shareB.values()) / nB, 4)},
                       "C": {"rows": nC, "max_source_share": round(max(shareC.values()) / nC, 4)}, "cap_B_plan": capB}
stats["partA"] = {"info": infoA, "drops": {s: dict(c.most_common()) for s, c in dropsA.items()}}
stats["partB"] = {"info": infoB, "drops": {s: dict(c.most_common()) for s, c in dropsB.items()}}
stats["files"] = {p.name: sha256_file(p) for p in [OUT / "part_a.jsonl", OUT / "part_b_en.jsonl", OUT / "part_b_meta.jsonl", OUT / "components/A.jsonl"]}
(OUT / "stage0_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: stats[k] for k in ("ref_docs", "v0_vs_ref", "components", "to_translate", "final_overlap")}, ensure_ascii=False)[:3000])
