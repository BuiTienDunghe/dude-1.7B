"""MLQA recast 2: choice(4) trên CỤM TỪ TRẢ LỜI. Gold = span đáp án do người gán (MLQA); 3 nhiễu = cụm chép nguyên văn từ chính đoạn
văn, do Opus chọn (audit/gen_out_mlqa.jsonl), rồi MỘT lượt Opus độc lập kiểm (audit/verify_out_mlqa2.jsonl); câu không đạt thì loại.
=> nhiễu do LLM chọn (nhãn gold vẫn của người): phải báo riêng nếu dùng.

Giai đoạn:
  build_mlqa2.py verify_in : kiểm nhiễu bằng script (nguyên văn, không chồng đáp án, khác nhau) -> audit/verify_in_mlqa2.jsonl
  build_mlqa2.py build     : giữ câu đạt cả hai phía, lấy 50 theo thứ tự ứng viên (seed); xáo lựa chọn (cùng hoán vị cho EN và VI);
                             thêm bản không dấu; n_tokens; decontam (3 phép); mức sàn -> mlqa2_rows.jsonl, floors_mlqa2.json, decontam_mlqa2.json
Câu hỏi = câu hỏi MLQA nguyên văn (người dịch), không có chuỗi tiếng Việt nào do agent viết.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import CONFIG, DATA, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_diacritics, write_jsonl  # noqa: E402

SEED, N = 20261014, 50
P2 = ROOT / "results/bench_candidates/pilot2"
A = P2 / "audit"
stage = sys.argv[1]
BATCHES = [("gen_in_mlqa.jsonl", "gen_out_mlqa.jsonl"), ("gen_in_mlqa_b1.jsonl", "gen_out_mlqa_b1.jsonl"), ("gen_in_mlqa_b2.jsonl", "gen_out_mlqa_b2.jsonl")]
gen_in, gen_out, order = {}, {}, []      # lô đầu 60 ứng viên + 2 lô bổ sung 70 + 70 (cùng thứ tự seed) vì chỉ ~1/3 ứng viên đủ 3 nhiễu cùng loại
for fi, fo in BATCHES:
    if not (A / fi).exists():
        continue
    xs = read_jsonl(A / fi); ys = {x["id"]: x for x in read_jsonl(A / fo)}
    assert set(ys) == {x["id"] for x in xs}, (fo, len(ys), len(xs))
    gen_in.update({x["id"]: x for x in xs}); gen_out.update(ys); order += [x["id"] for x in xs]
assert len(order) == len(set(order))


def script_check(side, nhieu):
    ctx, gold = side["context"], side["answer"]
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
    return errs


if stage == "verify_in":
    out, log = [], Counter()
    for i in order:
        g, o = gen_in[i], gen_out[i]
        if o.get("khong_du") or o.get("dap_an_nghi_sai"):
            log["loai_o_buoc_chon:" + ("khong_du" if o.get("khong_du") else "dap_an_nghi_sai")] += 1
            continue
        ev, ee = script_check(g["vi"], o["vi_nhieu"]), script_check(g["en"], o["en_nhieu"])
        if ev or ee:
            log["loai_o_script:" + ",".join(sorted(set(ev + ee)))] += 1
            continue
        out.append({"id": i, "vi": {"context": g["vi"]["context"], "question": g["vi"]["question"], "dap_an": g["vi"]["answer"], "nhieu": o["vi_nhieu"]},
                    "en": {"context": g["en"]["context"], "question": g["en"]["question"], "dap_an": g["en"]["answer"], "nhieu": o["en_nhieu"]}})
    write_jsonl(A / "verify_in_mlqa2.jsonl", out)
    avail = {"ung_vien": len(order), "vi_du_3_nhieu": sum(len(gen_out[i]["vi_nhieu"]) == 3 for i in order),
             "en_du_3_nhieu": sum(len(gen_out[i]["en_nhieu"]) == 3 for i in order),
             "loai_dap_an": dict(Counter(gen_out[i].get("loai_dap_an") for i in order))}
    (A / "gen_stats_mlqa2.json").write_text(json.dumps({**avail, "vao_buoc_kiem": len(out), "loai": dict(log)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({**avail, "vao_buoc_kiem": len(out), "loai": dict(log)}, ensure_ascii=False))
    sys.exit(0)

# ============================================================================ build
ver = {x["id"]: x for x in read_jsonl(A / "verify_out_mlqa2.jsonl")}
vin = [x["id"] for x in read_jsonl(A / "verify_in_mlqa2.jsonl")]
assert set(ver) == set(vin)
passed = [i for i in vin if ver[i]["vi"]["dat"] and ver[i]["en"]["dat"]]
fail = Counter()
for i in vin:
    for lang in ("vi", "en"):
        v = ver[i][lang]
        if not v["dap_an_dung"]:
            fail[f"{lang}:dap_an_sai"] += 1
        if v["nhieu_cung_dung"]:
            fail[f"{lang}:nhieu_cung_dung"] += 1
keep = passed[:N]
weak = sum(len(ver[i][l]["nhieu_yeu"]) for i in keep for l in ("vi", "en"))

rows = []
for i in keep:
    g, o = gen_in[i], gen_out[i]
    perm = list(range(4)); random.Random(f"{SEED}|{i}").shuffle(perm)          # cùng hoán vị cho EN và VI
    for lang in ("en", "vi"):
        opts0 = [g[lang]["answer"]] + o[f"{lang}_nhieu"]
        opts = [opts0[p] for p in perm]
        rows.append({"id": f"{i}:{lang}", "pair_id": i, "source": "mlqa_ans", "lang": lang, "kind": "answer_mc", "state": g[lang]["context"],
                     "question": g[lang]["question"], "options": opts, "target": perm.index(0), "split_origin": "mlqa:validation",
                     "n_tokens": None, "cluster": i, "gold_label": None, "flags": [f"loai={o.get('loai_dap_an')}", "nhieu_do_LLM_chon"],
                     "block": "pilot2", "variant": "orig"})
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
write_jsonl(P2 / "mlqa2_rows.jsonl", full)

# ---------------------------------------------------------------------------- mức sàn
WORD = re.compile(r"\w+", re.UNICODE)
words = lambda s: {w.lower() for w in WORD.findall(s)}
GLUED = re.compile(r"(?<=[a-zà-ỹđ0-9\)\]])([.!?])(?=[A-ZÀ-ỸĐ])")


def sents(text):
    t = GLUED.sub(r"\1 ", text)
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÀ-ỸĐ0-9\"“(])", t)
    return [p for p in parts if p.strip()]


def locate_pick(r):
    """Mẹo định vị: câu trong đoạn trùng nhiều từ nhất với câu hỏi -> chọn lựa chọn nằm trong câu đó (nhiều thì lấy lựa chọn trùng
    câu hỏi ít nhất... đơn giản: lựa chọn đầu tiên theo thứ tự); không có lựa chọn nào trong câu đó thì xét câu kế tiếp theo điểm."""
    q = words(r["question"])
    ss = sorted(sents(r["state"]), key=lambda s: -len(words(s) & q))
    for s in ss:
        inside = [j for j, o in enumerate(r["options"]) if o in s]
        if inside:
            return inside[0]
    return 0


floors = {}
for lang, var in (("en", "orig"), ("vi", "orig"), ("vi", "nodiac")):
    rs = [r for r in full if r["lang"] == lang and r["variant"] == var]
    n = len(rs)
    ov = lambda r: max(range(4), key=lambda j: (len(words(r["options"][j]) & words(r["question"])), -j))
    first_in_text = lambda r: min(range(4), key=lambda j: r["state"].find(r["options"][j]) if r["options"][j] in r["state"] else 10 ** 9)
    pos = Counter(r["target"] for r in rs)
    floors[f"{lang}|{var}"] = {
        "n": n, "doan_deu": 0.25, "phan_phoi_vi_tri": dict(sorted(pos.items())),
        "vi_tri_pho_bien_nhat": pos.most_common(1)[0][1] / n,
        "dai_nhat": sum(max(range(4), key=lambda j: len(r["options"][j])) == r["target"] for r in rs) / n,
        "ngan_nhat": sum(min(range(4), key=lambda j: len(r["options"][j])) == r["target"] for r in rs) / n,
        "lua_chon_trung_tu_cau_hoi_nhieu_nhat": sum(ov(r) == r["target"] for r in rs) / n,
        "meo_dinh_vi_cau": sum(locate_pick(r) == r["target"] for r in rs) / n,
        "xuat_hien_som_nhat_trong_doan": sum(first_in_text(r) == r["target"] for r in rs) / n,
        "gold_cung_cau_voi_it_nhat_1_nhieu": sum(any(r["options"][r["target"]] in s and any(r["options"][j] in s for j in range(4) if j != r["target"])
                                                     for s in sents(r["state"])) for r in rs) / n,
    }
(P2 / "floors_mlqa2.json").write_text(json.dumps(floors, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------------------- decontam (3 phép như bench)
from decontam import Index  # noqa: E402
pix = Index()
for r in full:
    if r["variant"] == "orig":
        pix.add(r["state"], f"{r['pair_id']}|{r['lang']}|state"); pix.add(r["question"], f"{r['pair_id']}|{r['lang']}|question")


def scan(texts):
    hits = defaultdict(set)
    for t, tag in texts:
        h = pix.check(t)
        if h:
            hits[h[1]].add(f"{h[0]}<-{tag}")
    return {k: sorted(v)[:5] for k, v in sorted(hits.items())}


def corpus(rows_, tag):
    out = []
    for r in rows_:
        out += [(r["state"], f"{tag}:{r.get('source')}:state")] + ([(r["question"], f"{tag}:{r.get('source')}:question")] if r.get("question") else [])
    return out


dec = {"train_v0+val_v0": scan(corpus(read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl"), "train_v0")),
       "dev_v1_1": scan(corpus([r for r in read_jsonl(DATA / "bench_v1_1/dev_v1_1.jsonl") if r["variant"] == "orig"], "dev_v1_1")),
       "test_v1_1": scan(corpus([r for r in read_jsonl(DATA / "bench_v1_1/test_v1.1.jsonl") if r["variant"] == "orig"], "test_v1_1"))}
from datasets import load_dataset  # noqa: E402
wk = []
for split in ("train", "dev", "test"):
    d = load_dataset("High-Will/ViWikiFC", revision="0089cbaac1a58458893b11c5cc067c4a37843f8c", split=split)
    wk += [(x["context"], f"viwikifc:{split}:context") for x in d] + [(x["claim"], f"viwikifc:{split}:claim") for x in d]
dec["viwikifc_all_splits"] = scan(wk)
dec["pilot1_mlqa"] = scan(corpus([r for r in read_jsonl(ROOT / "results/bench_candidates/pilot/pilot_rows.jsonl") if r["source"] == "mlqa" and r["variant"] == "orig"], "pilot1"))
dec["tong_cau_trung"] = {k: len({x.split("|")[0] for x in v}) for k, v in dec.items()}
(P2 / "decontam_mlqa2.json").write_text(json.dumps(dec, ensure_ascii=False, indent=1), encoding="utf-8")
man = {"seed": SEED, "ung_vien": len(order), "vao_buoc_kiem": len(vin), "dat_ca_hai_phia": len(passed), "giu": len(keep), "khong_dat": dict(fail),
       "nhieu_yeu_trong_50": weak, "rows": len(full), "sha256": sha256_file(P2 / "mlqa2_rows.jsonl"),
       "ghi_chu": "gold = span người gán (MLQA); 3 nhiễu do Opus chọn, chép nguyên văn từ đoạn; một lượt Opus độc lập kiểm; câu hỏi = câu hỏi MLQA nguyên văn"}
(P2 / "manifest_mlqa2.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({**man, "floors": floors, "decontam": dec["tong_cau_trung"]}, ensure_ascii=False))
