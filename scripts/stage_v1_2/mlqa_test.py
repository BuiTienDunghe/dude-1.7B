"""Ứng viên test bản 1.2 (KHU TẠM, chưa quyết đưa vào): MLQA vi recast "4 cụm từ trả lời", từ split TEST.
KHÔNG chấm model nào trên các câu này. KHÔNG ghi vào data/, KHÔNG đóng băng.

Quy tắc (REPORT_pilot2 §3): (a) mỗi phía có ≥1 nhiễu nằm cùng câu với đáp án; (b) hai lượt Opus: chọn nhiễu, rồi kiểm độc lập;
(c) cờ "nhieu_do_LLM_chon"; (d) loại đoạn trùng ViWikiFC; cộng decontam đủ kho như đợt chạy thử.
Nguồn: facebook/mlqa @ 397ed406, parquet refs/convert/parquet @ 0d43d0f6: mlqa.vi.vi/test, mlqa.en.en/test (cùng id). Seed 20261014.

Giai đoạn (venv WSL):
  prep        : lọc trước bằng script (span khớp; đoạn ≤ 400 từ; ≥ 4 câu mỗi phía; câu chứa đáp án ≥ 12 từ mỗi phía; khác đoạn của chạy thử;
                decontam đoạn + câu hỏi với train_v0+val, dev_v1.1, test 1.1, ViWikiFC (3 split), MLQA chạy thử) -> work/cands.jsonl (thứ tự seed)
  batch K     : ghi work/gen_in_K.jsonl (60 ứng viên, vị trí 60K..60K+59)
  check       : kiểm mọi work/gen_out_*.jsonl bằng script (nguyên văn, không chồng đáp án, khác nhau, quy tắc (a)) -> work/verify_in_K.jsonl
                cho các câu đạt chưa được kiểm; in phễu
  assemble    : giữ câu đạt cả hai phía ở work/verify_out_*.jsonl theo thứ tự seed, tối đa 300; xáo lựa chọn (cùng hoán vị EN/VI); thêm
                bản không dấu; n_tokens; mức sàn heuristic (không chấm model) -> stage_v1_2/mlqa_test_candidates.jsonl + manifest
"""
import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import CONFIG, DATA, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_diacritics, write_jsonl  # noqa: E402

SEED, TARGET, BATCH = 20261014, 300, 60
RAW = ROOT / "results/bench_candidates/pilot/raw/mlqa"
OUT = ROOT / "results/bench_candidates/stage_v1_2"
W = OUT / "work_mlqa"
W.mkdir(parents=True, exist_ok=True)
HASH = {"vi.vi.test.parquet": "e8538e64ace62d038ba73572e17aef852538d1d7e060dce5da2d7488dbf225ec",
        "en.en.test.parquet": "78e612ddce0503ebd583b02a807d0d218992daf7f53c027a2edf11bf4564f1cf"}
GLUED = re.compile(r"(?<=[a-zà-ỹđ0-9\)\]])([.!?])(?=[A-ZÀ-ỸĐ])")
SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ỸĐ0-9\"“(])")
stage = sys.argv[1]


def sent_spans(text):
    """Câu (start, end) trên chuỗi gốc: tách sau dấu câu + khoảng trắng + chữ hoa/số, và cả ranh giới dính liền 'castle.In'."""
    cuts = [m.end() for m in SPLIT.finditer(text)] + [m.end() for m in GLUED.finditer(text)]
    pos, out = 0, []
    for c in sorted(set(cuts)) + [len(text)]:
        seg = text[pos:c]
        if seg.strip():
            ls = len(seg) - len(seg.lstrip()); out.append((pos + ls, pos + len(seg.rstrip())))
        pos = c
    return out


def gold_sentence(ctx, a0, a):
    for s, e in sent_spans(ctx):
        if s <= a0 and a0 + len(a) <= e:
            return ctx[s:e]
    return None


if stage == "prep":
    import pyarrow.parquet as pq
    from common import sha256_file as h
    for f, v in HASH.items():
        assert h(RAW / f) == v, f
    vi_t = pq.read_table(RAW / "vi.vi.test.parquet").to_pylist()
    en_t = {x["id"]: x for x in pq.read_table(RAW / "en.en.test.parquet").to_pylist()}
    ans = lambda x: ast.literal_eval(x["answers"]) if isinstance(x["answers"], str) else x["answers"]
    pilot_ctx = {r["state"] for p in (ROOT / "results/bench_candidates/pilot/pilot_rows.jsonl", ROOT / "results/bench_candidates/pilot2/mlqa2_rows.jsonl")
                 for r in read_jsonl(p) if r["source"].startswith("mlqa")}
    drop, by_ctx = Counter(), {}
    for r in vi_t:
        e = en_t.get(r["id"])
        if e is None:
            drop["khong_co_en"] += 1; continue
        if r["context"].strip() in pilot_ctx:
            drop["doan_chay_thu"] += 1; continue
        av, ae = ans(r), ans(e)
        a0v, tv, a0e, te = av["answer_start"][0], av["text"][0], ae["answer_start"][0], ae["text"][0]
        if r["context"][a0v:a0v + len(tv)] != tv or e["context"][a0e:a0e + len(te)] != te:
            drop["span_khong_khop"] += 1; continue
        if len(r["context"].split()) > 400 or len(e["context"].split()) > 400:
            drop["dai_qua_400_tu"] += 1; continue
        if len(sent_spans(r["context"])) < 4 or len(sent_spans(e["context"])) < 4:
            drop["it_hon_4_cau"] += 1; continue
        gv, ge = gold_sentence(r["context"], a0v, tv), gold_sentence(e["context"], a0e, te)
        if gv is None or ge is None:
            drop["dap_an_vat_qua_cau"] += 1; continue
        if len(gv.split()) < 12 or len(ge.split()) < 12:
            drop["cau_dap_an_ngan"] += 1; continue
        by_ctx.setdefault(r["context"], []).append({"mlqa_id": r["id"], "vi": {"context": r["context"].strip(), "question": r["question"], "answer": tv,
                                                                              "cau_chua_dap_an": gv},
                                                    "en": {"context": e["context"].strip(), "question": e["question"], "answer": te,
                                                           "cau_chua_dap_an": ge}})
    rng = random.Random(SEED)
    cands = [rng.choice(sorted(v, key=lambda x: x["mlqa_id"])) for _, v in sorted(by_ctx.items())]
    rng.shuffle(cands)
    # decontam trước khi tốn Opus (3 phép của scripts/decontam.py): chỉ mục = đoạn + câu hỏi ứng viên; quét các kho
    from decontam import Index
    from datasets import load_dataset
    ix = Index()
    for i, c in enumerate(cands):
        for lang in ("vi", "en"):
            ix.add(c[lang]["context"], f"{i}|{lang}|state"); ix.add(c[lang]["question"], f"{i}|{lang}|question")
    hits = {}
    corp = []
    for r in read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl"):
        corp += [(r["state"], "train_v0")] + ([(r["question"], "train_v0")] if r.get("question") else [])
    for name, p in (("dev_v1_1", DATA / "bench_v1_1/dev_v1_1.jsonl"), ("test_v1_1", DATA / "bench_v1_1/test_v1.1.jsonl")):
        corp += [(r["state"], name) for r in read_jsonl(p) if r["variant"] == "orig"] + [(r["question"], name) for r in read_jsonl(p) if r["variant"] == "orig"]
    for split in ("train", "dev", "test"):
        d = load_dataset("High-Will/ViWikiFC", revision="0089cbaac1a58458893b11c5cc067c4a37843f8c", split=split)
        corp += [(x["context"], "viwikifc") for x in d] + [(x["claim"], "viwikifc") for x in d]
    for t, tag in corp:
        hit = ix.check(t)
        if hit:
            hits.setdefault(int(hit[1].split("|")[0]), set()).add(f"{hit[0]}<-{tag}")
    # chiều ngược với ViWikiFC (quy tắc d): đoạn ứng viên chứa 13-gram của văn bản ViWikiFC
    wix = Index()
    for t, tag in corp:
        if tag == "viwikifc":
            wix.add(t, tag)
    for i, c in enumerate(cands):
        hit = wix.check(c["vi"]["context"])
        if hit:
            hits.setdefault(i, set()).add(f"{hit[0]}<-viwikifc(nguoc)")
    keep = [c for i, c in enumerate(cands) if i not in hits]
    for k, c in enumerate(keep):
        c["id"] = f"mlqaT:{c['mlqa_id'][:12]}"; c["thu_tu"] = k
    assert len({c["id"] for c in keep}) == len(keep)
    write_jsonl(W / "cands.jsonl", keep)
    funnel = {"vi_test_questions": len(vi_t), "vi_test_contexts": len({r["context"] for r in vi_t}), "loai_loc_truoc": dict(drop),
              "doan_hop_le": len(cands), "loai_decontam": len(hits),
              "decontam_theo_kho": dict(Counter(x.split("<-")[1] for v in hits.values() for x in v)), "ung_vien": len(keep)}
    (W / "prep_funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(funnel, ensure_ascii=False))
    sys.exit(0)

cands = read_jsonl(W / "cands.jsonl")
by_id = {c["id"]: c for c in cands}

if stage == "batch":
    k = int(sys.argv[2])
    sl = cands[k * BATCH:(k + 1) * BATCH]
    write_jsonl(W / f"gen_in_{k:03d}.jsonl", [{"id": c["id"], "vi": c["vi"], "en": c["en"]} for c in sl])
    print(json.dumps({"batch": k, "n": len(sl), "tu": k * BATCH}))
    sys.exit(0)

if stage == "batch2":   # từ vị trí 300 trở đi: chỉ gửi ứng viên qua lọc heuristic (heur.py), vẫn theo thứ tự seed; tệp gen_in_1KK
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from heur import ok as heur_ok
    k = int(sys.argv[2])
    pool = [c for c in cands[300:] if heur_ok(c)]
    sl = pool[k * BATCH:(k + 1) * BATCH]
    write_jsonl(W / f"gen_in_{100 + k:03d}.jsonl", [{"id": c["id"], "vi": c["vi"], "en": c["en"]} for c in sl])
    print(json.dumps({"batch2": k, "n": len(sl), "pool": len(pool)}))
    sys.exit(0)


def script_check(side, nhieu):
    ctx, gold, gs = side["context"], side["answer"], side["cau_chua_dap_an"]
    errs = []
    if len(nhieu) != 3:
        errs.append("khong_du_3")
    for d in nhieu:
        if d not in ctx:
            errs.append("khong_nguyen_van")
        if not d.strip() or d.strip() == gold.strip() or d in gold or gold in d:
            errs.append("chong_dap_an")
    if len({d.strip().lower() for d in nhieu}) != len(nhieu):
        errs.append("trung_nhau")
    if not any(d in gs for d in nhieu):
        errs.append("quy_tac_a")
    return errs


def gen_outs():
    out = {}
    for p in sorted(W.glob("gen_out_*.jsonl")):
        for x in read_jsonl(p):
            out[x["id"]] = x
    return out


def ver_outs():
    out = {}
    for p in sorted(W.glob("verify_out_*.jsonl")):
        for x in read_jsonl(p):
            out[x["id"]] = x
    return out


if stage == "check":
    go, vo = gen_outs(), ver_outs()
    queued = {x["id"] for p in W.glob("verify_in_*.jsonl") for x in read_jsonl(p)}
    log, ok = Counter(), []
    for c in cands:
        o = go.get(c["id"])
        if o is None:
            continue
        log["da_chon_nhieu"] += 1
        if o.get("khong_du") or o.get("dap_an_nghi_sai"):
            log["loai:" + ("khong_du" if o.get("khong_du") else "dap_an_nghi_sai")] += 1; continue
        ev, ee = script_check(c["vi"], o.get("vi_nhieu", [])), script_check(c["en"], o.get("en_nhieu", []))
        if ev or ee:
            log["loai_script:" + ",".join(sorted(set(ev + ee)))] += 1; continue
        log["qua_script"] += 1
        ok.append(c["id"])
    new = [i for i in ok if i not in queued]
    k = len(list(W.glob("verify_in_*.jsonl")))
    for j in range(0, len(new), BATCH):
        chunk = new[j:j + BATCH]
        write_jsonl(W / f"verify_in_{k:03d}.jsonl", [{"id": i, **{l: {"context": by_id[i][l]["context"], "question": by_id[i][l]["question"],
                                                                     "dap_an": by_id[i][l]["answer"], "nhieu": go[i][f"{l}_nhieu"]} for l in ("vi", "en")}}
                                                    for i in chunk])
        k += 1
    passed = sum(1 for i in ok if i in vo and vo[i]["vi"]["dat"] and vo[i]["en"]["dat"])
    print(json.dumps({**log, "cho_kiem_moi": len(new), "da_kiem": sum(1 for i in ok if i in vo), "dat_ca_hai_phia": passed}, ensure_ascii=False))
    sys.exit(0)

# ============================================================================ assemble
go, vo = gen_outs(), ver_outs()
order = [c["id"] for c in cands if c["id"] in go]
# đối chiếu: nhiễu đã được kiểm độc lập phải trùng nhiễu trong file gen_out cuối cùng (agent có thể sửa file sau khi đã xếp hàng kiểm),
# và gen_out cuối vẫn qua script (không khong_du / dap_an_nghi_sai, nguyên văn, quy tắc (a))
vin = {x["id"]: x for p in W.glob("verify_in_*.jsonl") for x in read_jsonl(p)}
lech = []
for i in list(vo):
    o = go.get(i)
    bad = (o is None or o.get("khong_du") or o.get("dap_an_nghi_sai") or i not in vin
           or any(vin[i][l]["nhieu"] != o.get(f"{l}_nhieu") for l in ("vi", "en"))
           or script_check(by_id[i]["vi"], o.get("vi_nhieu", [])) or script_check(by_id[i]["en"], o.get("en_nhieu", [])))
    if bad:
        lech.append(i); vo.pop(i)
passed = [i for i in order if i in vo and vo[i]["vi"]["dat"] and vo[i]["en"]["dat"]]
keep = passed[:TARGET]
rows = []
for i in keep:
    c, o = by_id[i], go[i]
    perm = list(range(4)); random.Random(f"{SEED}|{i}").shuffle(perm)
    for lang in ("en", "vi"):
        opts0 = [c[lang]["answer"]] + o[f"{lang}_nhieu"]
        rows.append({"id": f"{i}:{lang}", "pair_id": i, "source": "mlqa_ans", "lang": lang, "kind": "answer_mc", "state": c[lang]["context"],
                     "question": c[lang]["question"], "options": [opts0[p] for p in perm], "target": perm.index(0), "split_origin": "mlqa:test",
                     "n_tokens": None, "cluster": i, "gold_label": None, "flags": ["nhieu_do_LLM_chon", f"loai={o.get('loai_dap_an')}"],
                     "block": "stage_v1_2", "variant": "orig", "mlqa_id": c["mlqa_id"]})
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
frames = load_json(CONFIG / "prompt_frames.json")
full = []
for r in rows:
    full.append(r)
    if r["lang"] == "vi":
        n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]), "question": strip_diacritics(r["question"]),
             "options": [strip_diacritics(x) for x in r["options"]]}
        for a, b in [(r["state"], n["state"]), (r["question"], n["question"])] + list(zip(r["options"], n["options"])):
            assert re.findall(r"\d+", a) == re.findall(r"\d+", b), r["id"]
        full.append(n)
for r in full:
    r["n_tokens"] = len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))
    r["n_state_tokens"] = len(tok.encode(r["state"], add_special_tokens=False))
write_jsonl(OUT / "mlqa_test_candidates.jsonl", full)
WORD = re.compile(r"\w+", re.UNICODE)
words = lambda s: {w.lower() for w in WORD.findall(s)}


def locate_pick(r):
    q = words(r["question"])
    for s, e in sorted(sent_spans(r["state"]), key=lambda se: -len(words(r["state"][se[0]:se[1]]) & q)):
        inside = [j for j, o in enumerate(r["options"]) if o in r["state"][s:e]]
        if inside:
            return inside[0]
    return 0


floors = {}
for lang, var in (("en", "orig"), ("vi", "orig"), ("vi", "nodiac")):
    rs = [r for r in full if r["lang"] == lang and r["variant"] == var]
    n = len(rs)
    if not n:
        continue
    pos = Counter(r["target"] for r in rs)
    floors[f"{lang}|{var}"] = {"n": n, "doan_deu": 0.25, "vi_tri_pho_bien_nhat": pos.most_common(1)[0][1] / n,
                               "dai_nhat": sum(max(range(4), key=lambda j: len(r["options"][j])) == r["target"] for r in rs) / n,
                               "lua_chon_trung_tu_cau_hoi": sum(max(range(4), key=lambda j: (len(words(r["options"][j]) & words(r["question"])), -j)) == r["target"]
                                                               for r in rs) / n,
                               "meo_dinh_vi_cau": sum(locate_pick(r) == r["target"] for r in rs) / n}
vfail = Counter()
for i in order:
    if i in vo:
        for l in ("vi", "en"):
            if not vo[i][l]["dap_an_dung"]:
                vfail[f"{l}:dap_an_sai"] += 1
            if vo[i][l]["nhieu_cung_dung"]:
                vfail[f"{l}:nhieu_cung_dung"] += 1
gstat = Counter()
for i in order:
    o = go[i]
    gstat["khong_du" if o.get("khong_du") else ("dap_an_nghi_sai" if o.get("dap_an_nghi_sai") else "co_nhieu")] += 1
man = {"name": "stage_v1_2 MLQA test candidates (KHU TẠM; chưa quyết đưa vào 1.2; chưa chấm; chưa đóng băng)",
       "source": {"repo": "facebook/mlqa", "revision": "397ed406c1a7902140303e7faf60fff35b58d285",
                  "parquet": "refs/convert/parquet @ 0d43d0f635e3d5768fcc252c97bf59cff847859e (mlqa.vi.vi/test, mlqa.en.en/test)", "sha256": HASH,
                  "license": "CC BY-SA 3.0 (dữ liệu; README GitHub facebookresearch/MLQA @ 24bf470b)"},
       "seed": SEED, "rules": ["(a) ≥1 nhiễu cùng câu với đáp án ở mỗi phía (script kiểm)", "(b) Opus chọn nhiễu + Opus kiểm độc lập",
                               "(c) cờ nhieu_do_LLM_chon", "(d) loại đoạn trùng ViWikiFC (cả hai chiều) + decontam train_v0/val, dev_v1.1, test 1.1, chạy thử"],
       "phieu": {"prep": json.load(open(W / "prep_funnel.json", encoding="utf-8")), "da_gui_chon_nhieu": len(order), "ket_qua_chon_nhieu": dict(gstat),
                 "da_kiem_doc_lap": sum(1 for i in order if i in vo), "khong_dat_kiem": dict(vfail), "dat_ca_hai_phia": len(passed), "giu": len(keep),
                 "loai_vi_gen_out_doi_sau_khi_kiem": len(lech)},
       "loc_truoc_heuristic": {"ap_tu_vi_tri": 300, "mo_ta": "scripts/stage_v1_2/heur.py: câu chứa đáp án phải có số khác (đáp án có số) / chữ hoa khác "
                                                         "(đáp án là tên riêng) / ≥16 từ (còn lại), ở cả hai phía; không đổi quy tắc (a)",
                               "do_tren_dot_1_300_ung_vien": "giữ 32/35 câu đủ nhiễu (91%), cắt 113/300 ứng viên (38%)",
                               "pool_sau_vi_tri_300": "1139/1854 qua lọc", "quyet_dinh": "phiên điều phối 2026-10-14: chọn (ii) giữ (a) ở cả hai phía, chấp nhận ~200 câu"},
       "rows": len(full), "sha256": sha256_file(OUT / "mlqa_test_candidates.jsonl"), "floors_heuristic": floors,
       "ghi_chu": "gold = span người gán (MLQA, người dịch chuyên nghiệp đánh dấu ở bản vi); 3 nhiễu do Opus chọn; câu hỏi = câu hỏi MLQA nguyên văn. "
                  "KHÔNG chấm model nào trên các câu này."}
(OUT / "manifest_mlqa_test_candidates.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"giu": len(keep), "rows": len(full), "phieu": man["phieu"], "floors": floors}, ensure_ascii=False))
