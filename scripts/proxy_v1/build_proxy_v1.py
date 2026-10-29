"""Dựng TẬP CHỌN MODEL riêng ("proxy") data/proxy_v1/ (bước 0, BRIEF_buoc0_G0.md mục 4; người dùng duyệt "duyệt 0 + G0" ngày 28/10 —
brief này cho phép ghi data/proxy_v1/, ngoại lệ của quy tắc "không ghi data/"). Dùng để chọn checkpoint và fit ngưỡng noul;
KHÔNG BAO GIỜ TRAIN TRÊN TẬP NÀY. Không chạy model. Không thêm chữ tiếng Việt mới.

Nguồn: phần KHÔNG thuộc dev hay test 1.3 của chính các nguồn bench, đọc từ file nguồn gốc đã tải (results/train_v1/raw_refs/):
  ViFactCheck : split train (dev 1.3 dùng split dev, test 1.3 dùng split test);
  ViWikiFC    : split train (như trên);
  SIB-200     : split train trừ 200 câu dev 1.3 đã dùng (split validation đã dùng hết cho dev);
  MASSIVE     : split train (dev 1.3 dùng validation, test 1.3 dùng test);
  XCOPA       : XCOPA tiếng Việt không có split train; chỉ còn các câu split test gốc KHÔNG vào test 1.3 (bản EN lấy từ COPA test theo idx,
                nhãn từ XCOPA, như bench);
  Belebele    : chỉ có một split test; lấy các câu có ĐOẠN VĂN không xuất hiện trong dev hay test 1.3 (kiểm theo link; còn 250 câu tiếng Việt).
Khung và chữ: đúng khung bench (common.render_prompt khi chấm). MỌI chữ nhãn, câu hỏi mẫu lấy TỪ DEV 1.3 (không từ config):
  kiểm chứng choice/noul: phần câu hỏi trước "Nhận định: " / "Phát biểu: " của dev + nhận định của nguồn; lựa chọn theo nhãn dev;
  SIB-200: 7 chủ đề EN (sắp xếp) và chữ VI cùng vị trí ở dòng VI cùng cặp của dev; câu hỏi VI/EN của dev;
  MASSIVE: khóa ý định ↔ chữ EN suy ra từ nhãn đúng + danh sách nhiễu (flags) của dev; chữ VI cùng vị trí; câu hỏi của dev;
  XCOPA: câu hỏi nguyên nhân / kết quả VI/EN của dev.
Cách dựng như build_bench_v1: kiểm chứng mỗi dòng nguồn sinh 1 câu, choice (3 lựa chọn xáo theo dòng) hay noul theo seed từng dòng; MASSIVE
1 đúng + 9 nhiễu; SIB 7 chủ đề xáo theo dòng; XCOPA 2 lựa chọn xáo theo dòng; mọi dòng VI có bản không dấu. Lấy mẫu cân bằng nhãn, seed cố định,
mã cặp "proxy:<nguồn>:<khóa>" (khác bench).
Khử trùng (scripts/decontam.py: trùng chính xác, MinHash >= 0,8, chung 13-gram, câu 4–12 từ nằm trọn trong văn bản kia; xét HAI chiều):
  với dev và test 1.3 (chỉ dùng để khử trùng, không in nội dung test), với data/train_v0.jsonl, val_v0 và results/train_v1/components/C.jsonl.
  Câu ứng viên chạm bất kỳ văn bản nào thì bỏ; kiểm lại toàn bộ tập sau cùng: phải 0 chạm.
ViFactCheck và ViWikiFC là dữ liệu UIT: CHỈ DÙNG NỘI BỘ.
Ra data/proxy_v1/proxy_v1.jsonl, manifest_proxy_v1.json. Chạy (venv WSL): bash setup/run.sh scripts/proxy_v1/build_proxy_v1.py"""
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, read_jsonl, sha256_file, strip_diacritics, write_jsonl  # noqa: E402
from decontam import Index, normalize  # noqa: E402

N = 150
BUF = 60                                   # lấy dư trước kiểm chiều ngược
OUT = DATA / "proxy_v1"
RAW = RESULTS / "train_v1" / "raw_refs"
DEV = DATA / "bench_v1_3" / "dev_v1_3.jsonl"
TEST = DATA / "bench_v1_3" / "test_v1.3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
TEST_SHA = "2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da"
TRAIN_FILES = [DATA / "train_v0.jsonl", DATA / "val_v0.jsonl", RESULTS / "train_v1/components/C.jsonl"]
assert sha256_file(DEV) == DEV_SHA and sha256_file(TEST) == TEST_SHA
dev, test = read_jsonl(DEV), read_jsonl(TEST)
BY = {(r["pair_id"], r["lang"], r["variant"]): r for r in dev}
log = defaultdict(Counter)


def pair_rng(source, pid):
    h = hashlib.sha256(f"{SEED}:{source}:{pid}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def balanced_order(keys, labels, rng):
    """Toàn bộ khóa theo thứ tự lấy mẫu cân bằng nhãn (vòng lần lượt qua các nhãn), như balanced_sample của build_bench_v1."""
    groups = defaultdict(list)
    for k, l in zip(keys, labels):
        groups[l].append(k)
    order = {l: list(rng.permutation(len(g))) for l, g in sorted(groups.items())}
    out, depth = [], 0
    while True:
        prog = False
        for l in sorted(groups):
            if depth < len(groups[l]):
                out.append(groups[l][order[l][depth]]); prog = True
        if not prog:
            return out
        depth += 1


# ------------------------------------------------------------------ chữ của dev 1.3
vio = [r for r in dev if r["lang"] == "vi" and r["variant"] == "orig"]
FC_OPT, PRE = {}, {}
for r in vio:
    if r["source"] in ("vifactcheck", "viwikifc"):
        if r["kind"] == "choice" and r["gold_provenance"] == "source":
            FC_OPT[r["gold_label"]] = r["options"][r["target"]]
        mk = "Nhận định: " if r["kind"] == "choice" else "Phát biểu: "
        PRE.setdefault(r["kind"], set()).add(r["question"].split(mk, 1)[0] + mk)
assert set(FC_OPT) == {"supported", "refuted", "nei"} and all(len(v) == 1 for v in PRE.values()), (FC_OPT, PRE)
PRE = {k: next(iter(v)) for k, v in PRE.items()}
NOUL = next(r["options"] for r in vio if r["kind"] == "noul")
Q = {}
for r in dev:
    if r["variant"] == "orig" and r["source"] in ("sib200", "massive"):
        Q.setdefault((r["source"], r["lang"]), set()).add(r["question"])
    if r["variant"] == "orig" and r["source"] == "xcopa":
        cq = next(f.split("=")[1] for f in r["flags"] if f.startswith("copa_question="))
        Q.setdefault((f"xcopa_{cq}", r["lang"]), set()).add(r["question"])
assert all(len(v) == 1 for v in Q.values()), Q
Q = {k: next(iter(v)) for k, v in Q.items()}
# SIB: chủ đề EN (sắp xếp như bench) -> chữ VI cùng vị trí
SIB = {}
for r in dev:
    if r["source"] == "sib200" and r["lang"] == "en":
        v = BY[(r["pair_id"], "vi", "orig")]
        for e, x in zip(r["options"], v["options"]):
            assert SIB.setdefault(e, x) == x
TOPICS = sorted(SIB)
assert len(TOPICS) == 7
# MASSIVE: khóa ý định -> chữ EN (giao các tập lựa chọn của mọi câu dev có khóa đó), rồi chữ VI cùng vị trí
cand = {}
for r in dev:
    if r["source"] == "massive" and r["lang"] == "en":
        keys = [r["gold_label"]] + next(f.split("=", 1)[1] for f in r["flags"] if f.startswith("distractors=")).split(",")
        for k in keys:
            cand[k] = cand.get(k, set(r["options"])) & set(r["options"])
done = {}
while len(done) < len(cand):
    prog = False
    for k, s in cand.items():
        s = s - set(done.values())
        if k not in done and len(s) == 1:
            done[k] = next(iter(s)); prog = True
    assert prog, "không suy được ánh xạ khóa ý định MASSIVE từ dev"
EN2VI = {}
for r in dev:
    if r["source"] == "massive" and r["lang"] == "en":
        for e, x in zip(r["options"], BY[(r["pair_id"], "vi", "orig")]["options"]):
            assert EN2VI.setdefault(e, x) == x
MAS = {k: {"en": e, "vi": EN2VI[e]} for k, e in done.items()}
INTENTS = sorted(MAS)
assert len(INTENTS) == 60


def row(pid, source, lang, kind, block, state, question, options, target, origin, cluster, gold, flags=None):
    return {"id": f"{pid}:{lang}", "pair_id": pid, "source": source, "lang": lang, "kind": kind, "block": block, "variant": "orig",
            "state": state, "question": question, "options": options, "target": int(target), "split_origin": origin, "n_tokens": None,
            "cluster": cluster, "gold_label": gold, "flags": flags or [], "proxy_v1": True}


def cluster_of(source, state):
    return f"{source}:" + hashlib.sha1(normalize(state).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ ứng viên theo nguồn
def items_factcheck(source):
    if source == "vifactcheck":
        d = pd.read_parquet(RAW / "tranthaihoa__vifactcheck/data/train-00000-of-00001.parquet")
        lab = {0: "supported", 1: "refuted", 2: "nei"}
        its = [{"key": f"{int(x['index'])}-{int(x['annotation_id'])}", "state": x["Context"], "claim": x["Statement"],
                "label": lab[int(x["labels"])]} for _, x in d.iterrows()]
    else:
        d = pd.read_csv(RAW / "High-Will__ViWikiFC/data/train.csv", dtype=str, keep_default_na=False)
        lab = {"Supports": "supported", "Refutes": "refuted", "Not_Enough_Information": "nei"}
        its = [{"key": x["pairID"] + "~" + hashlib.sha1(x["claim"].encode()).hexdigest()[:8], "state": x["context"], "claim": x["claim"],
                "label": lab[x["gold_label"]]} for _, x in d.iterrows() if x["gold_label"] in lab]
    return [it for it in its if isinstance(it["state"], str) and it["state"].strip() and isinstance(it["claim"], str) and it["claim"].strip()]


def build_factcheck(source, it):
    pid = f"proxy:{source}:{it['key']}"
    rng = pair_rng(source, pid)
    if rng.random() < 0.5:
        labs = ["supported", "refuted", "nei"]; perm = list(rng.permutation(3))
        opts, tgt, kind = [FC_OPT[labs[p]] for p in perm], perm.index(labs.index(it["label"])), "choice"
        q = PRE["choice"] + it["claim"]
    else:
        opts, tgt, kind, q = list(NOUL), 0 if it["label"] == "supported" else 1, "noul", PRE["noul"] + it["claim"]
    return [row(pid, source, "vi", kind, "native", it["state"], q, opts, tgt, f"{source}:train", cluster_of(source, it["state"]), it["label"])]


def items_sib():
    used = {r["pair_id"].split(":")[-1] for r in dev + test if r["source"] == "sib200"}
    rd = lambda l: pd.read_csv(RAW / f"Davlan__sib200/data/{l}/train.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
    m = rd("vie_Latn").merge(rd("eng_Latn"), on="index_id", suffixes=("_vi", "_en"))
    log["sib200"]["train_split"] = len(m)
    m = m[~m.index_id.isin(used)]
    log["sib200"]["train_tru_dev_test"] = len(m)
    return [{"key": x.index_id, "vi": x.text_vi, "en": x.text_en, "label": x.category_en} for x in m.itertuples()]


def build_sib(it):
    pid = f"proxy:sib200:{it['key']}"
    rng = pair_rng("sib200", pid); perm = list(rng.permutation(7))
    out = []
    for lang in ("en", "vi"):
        opts = [t if lang == "en" else SIB[t] for t in TOPICS]
        out.append(row(pid, "sib200", lang, "choice", "parallel", it[lang], Q[("sib200", lang)], [opts[p] for p in perm],
                       perm.index(TOPICS.index(it["label"])), "sib200:train (ngoài dev/test 1.3)", pid, it["label"]))
    return out


def items_massive():
    rd = lambda l: pd.DataFrame([json.loads(x) for x in gzip.open(RAW / f"mteb__amazon_massive_intent/train/{l}.json.gz", "rt", encoding="utf-8")])
    a, b = rd("vi"), rd("en")
    lab = "label" if "label" in a.columns else "label_text"
    m = a.merge(b, on="id", suffixes=("_vi", "_en"))
    log["massive"]["train_split"] = len(m)
    return [{"key": str(x["id"]), "vi": x["text_vi"], "en": x["text_en"], "label": x[f"{lab}_en"]} for _, x in m.iterrows() if x[f"{lab}_en"] in MAS]


def build_massive(it):
    pid = f"proxy:massive:{it['key']}"
    rng = pair_rng("massive", pid)
    pool = [i for i in INTENTS if i != it["label"]]
    labels = [it["label"]] + [pool[i] for i in sorted(rng.choice(len(pool), 9, replace=False))]
    perm = list(rng.permutation(10))
    return [row(pid, "massive", lang, "choice", "parallel", it[lang], Q[("massive", lang)], [[MAS[l][lang] for l in labels][p] for p in perm],
                perm.index(0), "massive:train", pid, it["label"], [f"distractors={','.join(labels[1:])}"]) for lang in ("en", "vi")]


def items_xcopa():
    used = {r["pair_id"].split(":")[-1] for r in test if r["source"] == "xcopa"}
    xv = pd.read_parquet(RAW / "cambridgeltl__xcopa/vi/test/0000.parquet")
    ce = pd.read_parquet(RAW / "aps__super_glue/copa/test-00000-of-00001.parquet")
    m = xv.merge(ce, on="idx", suffixes=("_vi", "_en"))
    assert (m.question_vi == m.question_en).all()
    log["xcopa"]["test_split"] = len(m)
    m = m[~m.idx.astype(str).isin(used)]
    log["xcopa"]["test_tru_test_1_3"] = len(m)
    return [{"key": str(int(x.idx)), "x": x, "label": int(x.label_vi)} for x in m.itertuples()]


def build_xcopa(it):
    x = it["x"]; pid = f"proxy:xcopa:{it['key']}"
    rng = pair_rng("xcopa", pid); perm = list(rng.permutation(2))
    out = []
    for lang in ("en", "vi"):
        opts = [getattr(x, f"choice1_{lang}"), getattr(x, f"choice2_{lang}")]
        out.append(row(pid, "xcopa", lang, "choice", "parallel", getattr(x, f"premise_{lang}"), Q[(f"xcopa_{x.question_vi}", lang)],
                       [opts[p] for p in perm], perm.index(it["label"]),
                       "xcopa:test (không vào test 1.3)" if lang == "vi" else "super_glue/copa:test (nhãn từ xcopa)", pid, None,
                       [f"copa_question={x.question_vi}"] + (["xcopa_changed=True"] if bool(x.changed) else [])))
    return out


def items_belebele():
    """Câu Belebele có đoạn văn (link) không thuộc dev hay test 1.3; cặp VI–EN theo (link, question_number)."""
    used = {r["cluster"].split(":", 1)[1] for r in dev + test if r["source"] == "belebele"}
    rd = lambda l: pd.DataFrame([json.loads(x) for x in open(RAW / f"facebook__belebele/data/{l}.jsonl", encoding="utf-8")])
    m = rd("vie_Latn").merge(rd("eng_Latn"), on=["link", "question_number"], suffixes=("_vi", "_en"))
    log["belebele"]["cau_tieng_viet"] = len(m)
    m = m[~m.link.isin(used)]
    log["belebele"]["cau_co_doan_van_ngoai_dev_test"] = len(m)
    return [{"key": f"{x.link}#{x.question_number}", "x": x, "label": int(x.correct_answer_num_en)} for x in m.itertuples()]


def build_belebele(it):
    x = it["x"]; pid = f"proxy:belebele:{it['key']}"
    rng = pair_rng("belebele", pid); perm = list(rng.permutation(4))
    out = []
    for lang in ("en", "vi"):
        opts = [getattr(x, f"mc_answer{i}_{lang}") for i in range(1, 5)]
        out.append(row(pid, "belebele", lang, "choice", "parallel", getattr(x, f"flores_passage_{lang}"), getattr(x, f"question_{lang}"),
                       [opts[p] for p in perm], perm.index(it["label"] - 1), "belebele:test (đoạn văn ngoài dev/test 1.3)", f"belebele:{x.link}", None))
    return out


# ------------------------------------------------------------------ chỉ mục tham chiếu (dev, test 1.3, train_v0, val_v0, C)
REF, SHORT_EXACT, REF_TEXTS = Index(), set(), []


def add_ref(t, tag):
    if not isinstance(t, str) or not t.strip():
        return
    n = normalize(t)
    if len(n.split()) >= 4:
        REF.add(t, tag); REF_TEXTS.append(n)
    else:
        SHORT_EXACT.add(n)


for name, rows in (("dev13", dev), ("test13", test)) + tuple((f.stem, read_jsonl(f)) for f in TRAIN_FILES):
    rep = Counter(normalize(t) for r in rows for t in [r["question"] or ""] + list(r["options"]))
    for r in rows:
        add_ref(r["state"], name)
        q = r["question"] or ""
        for mk in ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: "):
            if mk in q:
                add_ref(q.split(mk, 1)[1], name); q = ""
        if q and rep[normalize(q)] < 3:
            add_ref(q, name)
        for o in r["options"]:
            if rep[normalize(o)] < 3:
                add_ref(o, name)
REF_BLOB = "\n" + "\n".join(REF_TEXTS) + "\n"


def texts_of(rows):
    out = []
    for r in rows:
        out.append(r["state"])
        for mk in ("Nhận định: ", "Phát biểu: "):
            if mk in r["question"]:
                out.append(r["question"].split(mk, 1)[1])
        if r["source"] == "xcopa":
            out += r["options"]
        if r["source"] == "belebele":
            out += [r["question"]] + list(r["options"])
    return out


def hit(rows):
    """Lý do chạm (chiều thuận: ứng viên chứa / trùng / gần văn bản tham chiếu; chiều ngược: câu ứng viên 4–12 từ nằm trọn trong văn bản
    tham chiếu; câu dưới 4 từ: trùng chính xác) hoặc None."""
    for t in texts_of(rows):
        n = normalize(t)
        if not n:
            continue
        if len(n.split()) < 4:
            if n in SHORT_EXACT or f"\n{n}\n" in REF_BLOB:
                return "exact_short"
            continue
        h = REF.check(t)
        if h:
            return f"{h[0]}:{h[1]}"
        if len(n.split()) <= 12 and (f" {n} " in REF_BLOB or f"\n{n} " in REF_BLOB or f" {n}\n" in REF_BLOB or f"\n{n}\n" in REF_BLOB):
            return "nguoc_contain_short"
    return None


# ------------------------------------------------------------------ chọn mẫu
rng0 = np.random.default_rng(SEED + 2810)
plans = [("belebele", items_belebele(), build_belebele), ("vifactcheck", items_factcheck("vifactcheck"), lambda it: build_factcheck("vifactcheck", it)),
         ("viwikifc", items_factcheck("viwikifc"), lambda it: build_factcheck("viwikifc", it)),
         ("sib200", items_sib(), build_sib), ("massive", items_massive(), build_massive), ("xcopa", items_xcopa(), build_xcopa)]
proxy, seen_cluster = [], set()
for source, its, build in plans:
    byk = {it["key"]: it for it in its}
    order = balanced_order(sorted(byk), [byk[k]["label"] for k in sorted(byk)], rng0)
    log[source]["ung_vien"] = len(order)
    got = 0
    for k in order:
        if got >= N:
            break
        rows = build(byk[k])
        if source in ("vifactcheck", "viwikifc") and rows[0]["cluster"] in seen_cluster:
            log[source]["bo_trung_doan_van_trong_proxy"] += 1          # mỗi đoạn văn chỉ một nhận định, để mẫu đa dạng và cụm bootstrap độc lập
            continue
        h = hit(rows)
        if h:
            log[source]["bo_" + h.split(":")[0]] += 1; continue
        seen_cluster.add(rows[0]["cluster"])
        proxy += rows; got += 1
    log[source]["lay"] = got

# bản không dấu
full = []
for r in proxy:
    full.append(r)
    if r["lang"] == "vi":
        n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]),
             "question": strip_diacritics(r["question"]), "options": [strip_diacritics(o) for o in r["options"]]}
        assert re.findall(r"\d+", r["state"]) == re.findall(r"\d+", n["state"])
        full.append(n)

# kiểm lại toàn bộ tập: 0 chạm
recheck = Counter()
for pid in sorted({r["pair_id"] for r in proxy}):
    rs = [r for r in proxy if r["pair_id"] == pid]
    h = hit(rs)
    if h:
        recheck[h] += 1
assert not recheck, recheck
OUT.mkdir(parents=True, exist_ok=True)
write_jsonl(OUT / "proxy_v1.jsonl", full)
man = {"ten": "proxy_v1: tập chọn model riêng", "CANH_BAO": "KHÔNG BAO GIỜ TRAIN TRÊN TẬP NÀY. Chỉ để chọn checkpoint và fit ngưỡng noul. "
       "ViFactCheck, ViWikiFC (UIT): CHỈ DÙNG NỘI BỘ.", "cho_phep_ghi_data": "BRIEF_buoc0_G0.md mục 4, người dùng duyệt 28/10 ('duyệt 0 + G0')",
       "file": {"proxy_v1.jsonl": sha256_file(OUT / "proxy_v1.jsonl")}, "seed": SEED, "seed_lay_mau": SEED + 2810,
       "dev_sha256": DEV_SHA, "test_sha256": TEST_SHA, "khu_trung_voi": [str(p.relative_to(DATA.parent)) for p in [DEV, TEST] + TRAIN_FILES],
       "khu_trung_cach": "decontam.Index (exact, MinHash >= 0,8, 13-gram, câu 4–12 từ nằm trọn) chiều thuận + chiều ngược (câu ứng viên 4–12 từ nằm "
                         "trọn trong văn bản tham chiếu) + trùng chính xác câu < 4 từ; câu hỏi / lựa chọn lặp >= 3 lần trong một file coi là khung, không tính",
       "kiem_lai_sau_cung": "0 chạm", "so_cau": dict(Counter(f"{r['source']}|{r['lang']}|{r['variant']}|{r['kind']}" for r in full)),
       "so_cap": dict(Counter(r["source"] for r in proxy if r["lang"] == "vi" and r["variant"] == "orig")),
       "nhan_dung": dict(Counter(f"{r['source']}|{r['gold_label']}" for r in proxy if r["lang"] == "vi" and r["source"] in ("vifactcheck", "viwikifc", "sib200"))),
       "nhat_ky_lay_mau": {k: dict(v) for k, v in log.items()},
       "belebele": "từ các đoạn văn (link) không thuộc dev hay test 1.3 của split test gốc (xem nhat_ky_lay_mau.belebele)",
       "chu_lay_tu_dev": {"kiem_chung_choice_tien_to": PRE["choice"], "kiem_chung_noul_tien_to": PRE["noul"], "nhan_kiem_chung": FC_OPT,
                          "noul": NOUL, "cau_hoi": {f"{k[0]}|{k[1]}": v for k, v in Q.items()}, "sib200": SIB, "so_y_dinh_massive": len(MAS)}}
(OUT / "manifest_proxy_v1.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("file", "so_cau", "so_cap", "nhat_ky_lay_mau")}, ensure_ascii=False))
