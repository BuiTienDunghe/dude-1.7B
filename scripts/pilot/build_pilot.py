"""Chạy thử 2 benchmark bổ sung (lõi mở): MLQA vi và Global-PIQA nonparallel vi, 50 câu mỗi bộ, ở khu tạm.
KHÔNG đụng data/bench_v1_1 hay test. Seed 20261014. Chạy bằng venv WSL: bash setup/run.sh scripts/pilot/build_pilot.py

Nguồn (đã tải, kiểm hash, ở results/bench_candidates/pilot/raw/ — chỉ đọc, không thực thi):
  - facebook/mlqa @ 397ed406 (repo chỉ có mlqa.py); dữ liệu: bản parquet HF của đúng repo, refs/convert/parquet @ 0d43d0f6,
    chỉ split validation của mlqa.vi.vi và mlqa.en.en (cùng id).
  - mrlbenchmarks/global-piqa-nonparallel @ 6777742f, data/nonparallel_vie_latn.tsv (test, 100 câu); chỉ dùng 50 câu theo seed.

Recast (chuỗi tiếng Việt của câu hỏi do agent viết = ĐỀ XUẤT, chưa duyệt; đã ghi vào config/vi_proposals.jsonl):
  - MLQA, choice(k câu): state = đoạn Wikipedia; option = các câu của đoạn (tách câu bằng quy tắc dưới); gold = câu chứa span đáp án.
    Giữ đoạn có 3..10 câu, span nằm gọn trong một câu, ở CẢ bản vi và en (cùng id). Một câu hỏi mỗi đoạn (theo seed).
    Đoạn vi và en KHÔNG phải bản dịch của nhau (MLQA khai thác câu song song ở câu đáp án); chỉ câu hỏi là bản dịch chuyên nghiệp.
  - Global-PIQA, choice(2): state = prompt; option = solution0/1; gold = label. Bản EN: chỉ có eng_translated0/1 (cả câu), nên
    state EN = tiền tố từ chung của hai bản dịch, option EN = phần còn lại (cờ en_prefix_split).
Ra (results/bench_candidates/pilot/): pilot_rows.jsonl (EN, VI có dấu, VI không dấu), manifest_pilot.json, decontam_pilot.json,
  floors_pilot.json, audit/units_{mlqa,gpiqa}.jsonl (cho bước soát bằng Opus).
"""
import ast
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_diacritics, write_jsonl  # noqa: E402
from decontam import Index  # noqa: E402

SEED, N = 20261014, 50
OUT = Path(__file__).resolve().parents[2] / "results" / "bench_candidates" / "pilot"
RAW = OUT / "raw"
Q_TPL = {  # câu hỏi recast; VI là ĐỀ XUẤT của agent (pilot-mlqa-question-vi, pilot-gpiqa-question-vi trong config/vi_proposals.jsonl)
    "mlqa": {"en": "Which sentence in the text contains the answer to the following question: {q}",
             "vi": "Câu nào trong văn bản chứa câu trả lời cho câu hỏi sau: {q}"},
    "gpiqa": {"en": "Which continuation of the text is more plausible?",
              "vi": "Phần nối tiếp nào hợp lý hơn cho văn bản trên?"},
}
HASH = {"mlqa/vi.vi.validation.parquet": "0352bf538d0b4b5cf96cd3f9d86bef3fec1db1569900a1e4efabb8032c7a9f3b",
        "mlqa/en.en.validation.parquet": "0f6d63f190339874d099fc22b29fff12141ee87a09e77d89988d064b3055d748",
        "gpiqa/nonparallel_vie_latn.tsv": "3efde85b8cddece3a44ac3dbb66869a89a5cd30dff6a7b6b3c05f4eeb7a10099"}
for f, h in HASH.items():
    assert sha256_file(RAW / f) == h, f

# ----------------------------------------------------------------------------- tách câu
ABBR = {"tp", "st", "dr", "mr", "mrs", "ms", "jr", "sr", "vs", "etc", "co", "inc", "ltd", "no", "vol", "gs", "pgs", "ts", "ths", "tt", "mt",
        "ft", "lb", "oz", "approx", "fig", "e.g", "i.e", "u.s", "u.k", "a.d", "b.c", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "sept", "oct", "nov", "dec", "c", "ca", "cf", "al", "pp", "p", "q", "tr", "op", "ed", "eds", "rev", "gen", "gov", "sen", "rep", "prof", "mt"}
CAND = re.compile(r"[.!?…]+[\"”’)\]]*\s+")


def split_sents(text):
    """-> [(start, end)] trên chuỗi gốc (đã bỏ khoảng trắng hai đầu từng câu)."""
    cuts, pos = [], 0
    for m in CAND.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"“‘(["):
            continue
        tok = re.search(r"(\S+)$", text[:m.start()])
        tok = tok.group(1) if tok else ""
        core = tok.lower().strip("(\"“‘[")
        if text[m.start()] == "." and (core in ABBR or (len(tok) == 1 and tok.isupper()) or re.fullmatch(r"(?:[a-z]\.)+[a-z]", core)):
            continue
        end = m.start() + len(m.group(0).rstrip())
        cuts.append((pos, end)); pos = m.end()
    cuts.append((pos, len(text)))
    out = []
    for s, e in cuts:
        seg = text[s:e]
        ls = len(seg) - len(seg.lstrip()); rs = len(seg.rstrip())
        if seg.strip():
            out.append((s + ls, s + rs))
    return out


def mlqa_item(r):
    """-> (dict, lý do loại)."""
    ctx, ans = r["context"], ast.literal_eval(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
    a0, t0 = ans["answer_start"][0], ans["text"][0]
    if ctx[a0:a0 + len(t0)] != t0:
        return None, "span_khong_khop"
    sp = split_sents(ctx)
    gold = [i for i, (s, e) in enumerate(sp) if s <= a0 and a0 + len(t0) <= e]
    if len(gold) != 1:
        return None, "span_vat_qua_cau"
    if not 3 <= len(sp) <= 10:
        return None, f"so_cau_{'<3' if len(sp) < 3 else '>10'}"
    return {"state": ctx.strip(), "options": [ctx[s:e] for s, e in sp], "target": gold[0], "q": r["question"], "answer": t0}, None


vi_t = pq.read_table(RAW / "mlqa/vi.vi.validation.parquet").to_pylist()
en_t = {x["id"]: x for x in pq.read_table(RAW / "mlqa/en.en.validation.parquet").to_pylist()}
drop = Counter()
by_ctx = defaultdict(list)
eligible_all = []          # mọi câu hỏi hợp lệ (để đo phân phối vị trí trên toàn bộ, không chỉ 50)
for r in vi_t:
    vi_it, why = mlqa_item(r)
    if why:
        drop["vi:" + why] += 1; continue
    en_it, why = mlqa_item(en_t[r["id"]])
    if why:
        drop["en:" + why] += 1; continue
    eligible_all.append((r["id"], vi_it, en_it))
    by_ctx[vi_it["state"]].append((r["id"], vi_it, en_it))
rng = random.Random(SEED)
one_per_ctx = [rng.choice(sorted(v, key=lambda x: x[0])) for _, v in sorted(by_ctx.items())]
rng.shuffle(one_per_ctx)
mlqa_pick = sorted(one_per_ctx[:N], key=lambda x: x[0])

# ----------------------------------------------------------------------------- Global-PIQA
gp = list(csv.DictReader(open(RAW / "gpiqa/nonparallel_vie_latn.tsv", encoding="utf-8"), delimiter="\t"))
assert len(gp) == 100
gp_idx = sorted(random.Random(SEED).sample(range(len(gp)), N))
gp_pick = [gp[i] for i in gp_idx]


def en_prefix(t0, t1):
    w0, w1 = t0.split(" "), t1.split(" ")
    k = 0
    while k < min(len(w0), len(w1)) - 1 and w0[k] == w1[k]:
        k += 1
    return " ".join(w0[:k]), " ".join(w0[k:]), " ".join(w1[k:]), k


# ----------------------------------------------------------------------------- dòng
def row(source, pid, lang, state, question, options, target, flags, cluster, origin, kind):
    return {"id": f"{pid}:{lang}", "pair_id": pid, "source": source, "lang": lang, "kind": kind, "state": state, "question": question,
            "options": options, "target": target, "split_origin": origin, "n_tokens": None, "cluster": cluster, "gold_label": None,
            "flags": flags, "block": "pilot", "variant": "orig"}


rows, gp_meta = [], Counter()
for qid, vi_it, en_it in mlqa_pick:
    pid = f"mlqa:{qid[:12]}"
    for lang, it in (("en", en_it), ("vi", vi_it)):
        rows.append(row("mlqa", pid, lang, it["state"], Q_TPL["mlqa"][lang].format(q=it["q"]), it["options"], it["target"],
                        [f"n_sent={len(it['options'])}", f"gold_pos={it['target']}"], pid, "mlqa:validation", "sentence_select"))
for r in gp_pick:
    pid = f"gpiqa:{r['example_id']}"
    pre, o0, o1, k = en_prefix(r["eng_translated0"].strip(), r["eng_translated1"].strip())
    fl = [f"cultural={r['approx_cultural_score']}", f"llm_used={r['llm_used']}"]
    gp_meta["en_prefix_words>0" if k else "en_prefix_empty"] += 1
    rows.append(row("gpiqa", pid, "en", pre if k else "", Q_TPL["gpiqa"]["en"], [o0, o1], int(r["label"]), fl + ["en_prefix_split"], pid,
                    "global-piqa-nonparallel:test", "plausible"))
    rows.append(row("gpiqa", pid, "vi", r["prompt"].strip(), Q_TPL["gpiqa"]["vi"], [r["solution0"].strip(), r["solution1"].strip()],
                    int(r["label"]), fl, pid, "global-piqa-nonparallel:test", "plausible"))

# bản không dấu (đúng phép của bench) + n_tokens (tokenizer Qwen3-0.6B như bench 1.0/1.1)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
frames = load_json(CONFIG / "prompt_frames.json")
full = []
for r in rows:
    full.append(r)
    if r["lang"] == "vi":
        n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]), "question": strip_diacritics(r["question"]),
             "options": [strip_diacritics(o) for o in r["options"]]}
        for a, b in [(r["state"], n["state"]), (r["question"], n["question"])] + list(zip(r["options"], n["options"])):
            assert re.findall(r"\d+", a) == re.findall(r"\d+", b), r["id"]
        full.append(n)
for r in full:
    r["n_tokens"] = len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))
    r["n_state_tokens"] = len(tok.encode(r["state"], add_special_tokens=False))
write_jsonl(OUT / "pilot_rows.jsonl", full)

# ----------------------------------------------------------------------------- decontam (ba phép như bench: exact / near MinHash 0,8 / contain 13-gram|câu ngắn)
def build_index(texts):
    ix = Index()
    for t, tag in texts:
        ix.add(t, tag)
    return ix


pilot_texts = []
for r in full:
    if r["variant"] != "orig":
        continue
    pilot_texts.append((r["state"], f"{r['pair_id']}|{r['lang']}|state"))
    for j, o in enumerate(r["options"]):
        if r["source"] == "gpiqa":
            pilot_texts.append((o, f"{r['pair_id']}|{r['lang']}|option{j}"))
for qid, vi_it, en_it in mlqa_pick:
    pid = f"mlqa:{qid[:12]}"
    pilot_texts += [(vi_it["q"], f"{pid}|vi|question"), (en_it["q"], f"{pid}|en|question")]
pix = build_index(pilot_texts)


def scan(corpus_name, texts):
    hits = defaultdict(set)
    for t, ctag in texts:
        h = pix.check(t)
        if h:
            hits[h[1]].add(f"{h[0]}<-{ctag}")
    return {k: sorted(v)[:5] for k, v in sorted(hits.items())}


def corpus_rows(rows_, tag):
    out = []
    for r in rows_:
        out.append((r["state"], f"{tag}:{r.get('source')}:state"))
        if r.get("question"):
            out.append((r["question"], f"{tag}:{r.get('source')}:question"))
        for o in r.get("options") or []:
            out.append((o, f"{tag}:{r.get('source')}:option"))
    return out


dec = {"phep": "exact / near (MinHash Jaccard>=0,8, shingle 5 ký tự) / contain (13-gram từ; văn bản 4–12 từ phải nằm trọn) — scripts/decontam.py",
       "huong": "chỉ mục = văn bản chạy thử (state, câu hỏi gốc MLQA, lựa chọn Global-PIQA); quét văn bản của từng kho",
       "pilot_texts": len(pilot_texts)}
tv = read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl")
dec["train_v0+val_v0"] = scan("train", corpus_rows(tv, "train_v0"))
dec["dev_v1_1"] = scan("dev", corpus_rows([r for r in read_jsonl(DATA / "bench_v1_1/dev_v1_1.jsonl") if r["variant"] == "orig"], "dev_v1_1"))
dec["test_v1_1"] = scan("test", corpus_rows([r for r in read_jsonl(DATA / "bench_v1_1/test_v1.1.jsonl") if r["variant"] == "orig"], "test_v1_1"))
from datasets import load_dataset  # noqa: E402
wk = []
for split in ("train", "dev", "test"):
    try:
        d = load_dataset("High-Will/ViWikiFC", revision="0089cbaac1a58458893b11c5cc067c4a37843f8c", split=split)
    except Exception as ex:   # tên split khác: ghi lại
        dec.setdefault("viwikifc_split_loi", []).append(f"{split}: {type(ex).__name__}")
        continue
    wk += [(x["context"], f"viwikifc:{split}:context") for x in d] + [(x["claim"], f"viwikifc:{split}:claim") for x in d]
dec["viwikifc_all_splits_n_texts"] = len(wk)
dec["viwikifc_all_splits"] = scan("viwikifc", wk)
# chiều ngược: văn bản ViWikiFC nằm trong đoạn MLQA (claim ngắn, câu trong đoạn)
wix = build_index(list({t: tag for t, tag in wk}.items()))
rev = defaultdict(set)
for t, tag in pilot_texts:
    if tag.startswith("mlqa"):
        h = wix.check(t)
        if h:
            rev[tag].add(f"{h[0]}<-{h[1]}")
dec["viwikifc_reverse"] = {k: sorted(v) for k, v in rev.items()}
dec["tong_cau_trung"] = {k: len({x.split("|")[0] for x in v}) for k, v in dec.items() if isinstance(v, dict) and k not in ("phep",)}
(OUT / "decontam_pilot.json").write_text(json.dumps(dec, ensure_ascii=False, indent=1), encoding="utf-8")

# ----------------------------------------------------------------------------- mức sàn
WORD = re.compile(r"\w+", re.UNICODE)


def overlap_pick(q, opts):
    qw = {w.lower() for w in WORD.findall(q)}
    sc = [len(qw & {w.lower() for w in WORD.findall(o)}) for o in opts]
    return sc.index(max(sc))


floors = {}
for src in ("mlqa", "gpiqa"):
    for lang, var in (("en", "orig"), ("vi", "orig"), ("vi", "nodiac")):
        rs = [r for r in full if r["source"] == src and r["lang"] == lang and r["variant"] == var]
        f = {"n": len(rs), "doan_deu": sum(1 / len(r["options"]) for r in rs) / len(rs),
             "luon_dau": sum(r["target"] == 0 for r in rs) / len(rs), "luon_cuoi": sum(r["target"] == len(r["options"]) - 1 for r in rs) / len(rs),
             "dai_nhat": sum(max(range(len(r["options"])), key=lambda j: len(r["options"][j])) == r["target"] for r in rs) / len(rs)}
        pos = Counter(r["target"] for r in rs)
        f["vi_tri_pho_bien_nhat_trong_mau"] = {"vi_tri": pos.most_common(1)[0][0], "acc": pos.most_common(1)[0][1] / len(rs)}
        f["phan_phoi_vi_tri"] = dict(sorted(pos.items()))
        if src == "mlqa":
            qs = {f"mlqa:{qid[:12]}": (vi_it["q"] if lang == "vi" else en_it["q"]) for qid, vi_it, en_it in mlqa_pick}
            f["trung_tu_cau_hoi"] = sum(overlap_pick(strip_diacritics(qs[r["pair_id"]]) if var == "nodiac" else qs[r["pair_id"]], r["options"]) == r["target"]
                                        for r in rs) / len(rs)
        floors[f"{src}|{lang}|{var}"] = f
# vị trí câu gold trên TOÀN BỘ câu hợp lệ (không chỉ 50), theo vị trí tương đối
for lang in ("vi", "en"):
    its = [(v if lang == "vi" else e) for _, v, e in eligible_all]
    pos = Counter("dau" if it["target"] == 0 else ("cuoi" if it["target"] == len(it["options"]) - 1 else "giua") for it in its)
    by_k = Counter(it["target"] for it in its)
    floors[f"mlqa|{lang}|toan_bo_hop_le"] = {"n": len(its), "vi_tri_tuong_doi": dict(pos), "luon_dau": by_k[0] / len(its),
                                             "doan_deu": sum(1 / len(it["options"]) for it in its) / len(its),
                                             "trung_tu_cau_hoi": sum(overlap_pick(it["q"], it["options"]) == it["target"] for it in its) / len(its)}
(OUT / "floors_pilot.json").write_text(json.dumps(floors, ensure_ascii=False, indent=1), encoding="utf-8")

# ----------------------------------------------------------------------------- đơn vị soát (định dạng results/translation_audit)
(OUT / "audit").mkdir(exist_ok=True)
for src in ("mlqa", "gpiqa"):
    units = []
    for pid in sorted({r["pair_id"] for r in full if r["source"] == src}):
        v = next(r for r in full if r["id"] == pid + ":vi"); en = next(r for r in full if r["id"] == pid + ":en")
        side = lambda r: {"question": r["question"], "options": r["options"], "gold_index": r["target"], "gold_text": r["options"][r["target"]]}
        units.append({"split": "pilot", "source": src, "vi_state": v["state"], "en_state": en["state"],
                      "questions": [{"audit_id": pid, "kind": v["kind"], "vi": side(v), "en": side(en)}]})
    write_jsonl(OUT / "audit" / f"units_{src}.jsonl", units)

man = {"name": "pilot_bench_candidates", "seed": SEED, "n_per_source": N, "rows": len(full),
       "counts": dict(Counter(f"{r['source']}|{r['lang']}|{r['variant']}" for r in full)),
       "sources": {"mlqa": {"repo": "facebook/mlqa", "revision": "397ed406c1a7902140303e7faf60fff35b58d285",
                            "data": "refs/convert/parquet @ 0d43d0f635e3d5768fcc252c97bf59cff847859e: mlqa.vi.vi/validation/0000.parquet, mlqa.en.en/validation/0000.parquet",
                            "sha256": {k: v for k, v in HASH.items() if k.startswith("mlqa")}, "license_tag": "cc-by-sa-3.0",
                            "vi_validation_questions": len(vi_t), "vi_contexts": len({r["context"] for r in vi_t}),
                            "eligible_questions": len(eligible_all), "eligible_contexts": len(by_ctx), "dropped": dict(drop)},
                   "gpiqa": {"repo": "mrlbenchmarks/global-piqa-nonparallel", "revision": "6777742fa3634c0583cda3b7f8a482ea7b1b0937",
                             "file": "data/nonparallel_vie_latn.tsv", "git_blob": "789ffc04c30174049fce3835079728490a5a4798",
                             "sha256": HASH["gpiqa/nonparallel_vie_latn.tsv"], "license_tag": "cc-by-sa-4.0", "picked_index": gp_idx,
                             "en_prefix": dict(gp_meta)}},
       "question_templates": Q_TPL, "question_templates_status": "VI: đề xuất của agent, CHƯA DUYỆT (config/vi_proposals.jsonl)",
       "nodiac": "scripts/common.strip_diacritics", "n_tokens": "rendered prompt, tokenizer Qwen/Qwen3-0.6B",
       "sha256": {"pilot_rows.jsonl": sha256_file(OUT / "pilot_rows.jsonl")}}
(OUT / "manifest_pilot.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"rows": len(full), "mlqa": man["sources"]["mlqa"], "gpiqa_en_prefix": dict(gp_meta),
                  "decontam": dec["tong_cau_trung"], "viwikifc_n": dec["viwikifc_all_splits_n_texts"], "viwikifc_err": dec.get("viwikifc_split_loi")},
                 ensure_ascii=False))
