"""Giai đoạn 0 train_v1, cập nhật v2 theo quyết định người dùng (điều phối báo 17/10). KHÔNG train, KHÔNG dịch.
Đầu vào: bản lưu lượt 2 của build_stage0.py ở results/train_v1/ban_v1/ (tự sao lần đầu chạy; các lần sau chỉ đọc từ đó, nên chạy lại ra cùng kết quả).
 1) Nền = train_v0 bỏ MỌI dòng train chạm chỉ mục mở rộng (8 VI + 39 EN, v0_leak.json). Phần EN không còn đúng 16.000.
 2) Phần A: EXAMS bỏ môn Lý, Hóa, Sinh (Toán, Văn không có trong EXAMS-vi train); giữ Sử, Địa, Công dân; không bù dòng. VSMRC giữ các dòng đã chọn
    nhưng cắt theo trần 5% của tổng A mới (trần là quy tắc cứng; tổng A giảm nên trần giảm).
 3) Phần B: thêm e-CARE và HeadlineCause (tasksource @ revision v0, recast bằng recast_v0), khoảng 500 dòng mỗi nguồn:
    TRƯỚC khi chọn, cả kho chưa dùng được kiểm bằng decontam.Index với COPA (EN), Balanced-COPA, XCOPA mọi ngôn ngữ (gồm translation-*),
    dev + test 1.3; dòng trùng bị bỏ. Sau đó chọn (seed cố định, một dòng mỗi group_id), qua finalize với chỉ mục tiếng Anh mở rộng + v0. Trần 5%.
Ra: ghi đè part_a.jsonl, part_b_en.jsonl, part_b_meta.jsonl, to_translate/part_b_to_translate.jsonl, to_translate/chuoi_lap_lai.json,
components/A.jsonl, components/{B,C}_plan.json; thêm stage0_stats_v2.json. Bản trước giữ ở ban_v1/.
QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, SEED, load_json, model_revisions, read_jsonl, sha256_file, write_jsonl  # noqa: E402
from decontam import Index, normalize  # noqa: E402
from recast_v0 import make_finalize, ts_row  # noqa: E402
from refs_index import build, check_same_as_stage0  # noqa: E402

OUT = RESULTS / "train_v1"
B1 = OUT / "ban_v1"
CAP = 0.05
KEEP = ["part_a.jsonl", "part_b_en.jsonl", "part_b_meta.jsonl", "to_translate/part_b_to_translate.jsonl", "to_translate/chuoi_lap_lai.json",
        "components/A.jsonl", "components/B_plan.json", "components/C_plan.json", "stage0_stats.json", "stage0_lengths.json", "v0_leak.json"]
if not B1.exists():
    for f in KEEP:
        (B1 / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(OUT / f, B1 / f)
man0 = load_json(DATA / "manifest_train_v0.json")
TS = (man0["sources"]["tasksource"]["repo"], man0["sources"]["tasksource"]["revision"])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
st = {"seed": SEED, "dau_vao": {f: sha256_file(B1 / f) for f in KEEP}}

# ---------------------------------------------------------------- chỉ mục mở rộng (cùng code, kiểm trùng số văn bản với lượt 2)
CHK, V0, rstats = build()
check_same_as_stage0(rstats)
print("ref index ok", rstats["ref_docs"], flush=True)

# ---------------------------------------------------------------- 1) nền
leak = json.load(open(B1 / "v0_leak.json", encoding="utf-8"))
drop0 = {h["id"] for h in leak["hits"] if h["split"] == "train"}
assert "ts:dynabench-dynasent-8074025339:train:ff029677187969ed:dynasent-r1_votes" in drop0
train0, val0 = read_jsonl(DATA / "train_v0.jsonl"), read_jsonl(DATA / "val_v0.jsonl")
base = [r for r in train0 if r["id"] not in drop0]
st["nen"] = {"bo": len(drop0), "bo_vi": sum(1 for r in train0 if r["id"] in drop0 and r["lang"] == "vi"),
             "bo_en": sum(1 for r in train0 if r["id"] in drop0 and r["lang"] != "vi"), "con": len(base),
             "tasksource_con": sum(1 for r in base if r["dataset"] == "tasksource"), "vi_con": sum(1 for r in base if r["lang"] == "vi")}

# ---------------------------------------------------------------- 2) phần A
A1 = read_jsonl(B1 / "part_a.jsonl")
DROP_SUBJ = {"Physics", "Chemistry", "Biology", "Mathematics", "Math"}
ex_before = Counter(r["variant"].split("/")[0] for r in A1 if r["source"] == "exams/vi")
A = [r for r in A1 if not (r["source"] == "exams/vi" and r["variant"].split("/")[0] in DROP_SUBJ)]
ex_after = Counter(r["variant"].split("/")[0] for r in A if r["source"] == "exams/vi")
non_v = len(base) + sum(1 for r in A if r["source"] != "vsmrc/mrc")
cap_v = int(CAP * non_v / (1 - CAP))
vs = [r for r in A if r["source"] == "vsmrc/mrc"]
A = [r for r in A if r["source"] != "vsmrc/mrc"] + vs[:cap_v]          # giữ thứ tự chọn ban đầu, cắt phần cuối
order1 = {x["id"]: k for k, x in enumerate(A1)}
A.sort(key=lambda r: order1[r["id"]])
nA = len(base) + len(A)
shareA = Counter(r["source"] for r in base + A)
assert max(shareA.values()) <= CAP * nA, shareA.most_common(2)
st["phan_A"] = {"exams_truoc": dict(ex_before), "exams_sau": dict(ex_after), "exams_khong_co_trong_nguon": ["Mathematics (Toán)", "Literature (Văn)"],
                "vsmrc_truoc": len(vs), "vsmrc_tran": cap_v, "vsmrc_sau": min(len(vs), cap_v), "dong": len(A), "theo_nguon": dict(Counter(r["source"] for r in A)),
                "de_xuat_khong_con_dung": ["tv1-exams-subj-physics", "tv1-exams-subj-chemistry", "tv1-exams-subj-biology"]}
print("A", st["phan_A"], flush=True)

# ---------------------------------------------------------------- 3) phần B: nhân quả
# 3a) chỉ mục họ COPA + dev/test 1.3 (kiểm trước khi chọn)
api = HfApi()
XREV = "c1471bb6b030caadddd02b7785224bfa285737b4"            # refs/convert/parquet của cambridgeltl/xcopa (như build_stage0)
xfiles = [f for f in api.list_repo_files("cambridgeltl/xcopa", repo_type="dataset", revision=XREV) if f.endswith(".parquet")]
COP = Index()
cop_count = Counter()


def cadd(t, tag):
    if isinstance(t, str) and t.strip():
        COP.add(t, tag); cop_count[tag.split(":")[0] + ":" + tag.split(":")[1]] += 1


import pandas as pd  # noqa: E402
xsha = {}
for f in xfiles:
    p = hf_hub_download("cambridgeltl/xcopa", f, repo_type="dataset", revision=XREV, local_dir=OUT / "raw_refs" / "xcopa_all")
    xsha[f] = sha256_file(p)
    for _, x in pd.read_parquet(p).iterrows():
        for c in ("premise", "choice1", "choice2"):
            cadd(x[c], f"xcopa:{f.split('/')[0]}")
for sub, repo in (("copa", "aps__super_glue"), ("balanced_copa", "pkavumba__balanced-copa")):
    for p in sorted((OUT / "raw_refs" / repo).rglob("*.parquet")):
        if ".cache" in p.parts or (sub == "copa" and "copa" not in p.parts):
            continue
        for _, x in pd.read_parquet(p).iterrows():
            for c in ("premise", "choice1", "choice2"):
                cadd(x[c], f"{sub}:{p.parent.name}")
bench = {sp: read_jsonl(DATA / "bench_v1_3" / f) for sp, f in (("dev", "dev_v1_3.jsonl"), ("test", "test_v1.3.jsonl"))}
rep = Counter(normalize(t) for rs in bench.values() for r in rs for t in [r["question"]] + list(r["options"]))
for sp, rs in bench.items():
    for r in rs:
        cadd(r["state"], f"bench13:{sp}")
        for t in [r["question"]] + list(r["options"]):
            if rep[normalize(t)] < 3:
                cadd(t, f"bench13:{sp}")
st["chi_muc_ho_copa"] = {"van_ban": COP.n, "theo_nguon": dict(cop_count), "xcopa_files": len(xfiles), "xcopa_revision": XREV}

# 3b) kho chưa dùng, kiểm trước khi chọn, rồi chọn
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
src_col, id_col, gid_col = ts["source"], ts["id"], ts["group_id"]
pos = {i: k for k, i in enumerate(id_col)}
used_ids = {r["id"][3:] for r in train0 + val0 if r["dataset"] == "tasksource"}
used_groups = {gid_col[pos[i]] for i in used_ids}
B1rows = read_jsonl(B1 / "part_b_en.jsonl")
NEW = {
    "e-CARE": {"license_verbatim": "MIT (GitHub waste-wood/e-care, bản gốc); bản HF 12ml/e-CARE @ a9c788710144e6edd31f925345fa107c0634e223 KHÔNG có thẻ "
                                  "giấy phép; tasksource ghi 'BSD 2-Clause License (DPI)'",
               "license_use": "commercial (MIT, theo bản gốc GitHub)", "cung_khuon_copa": True,
               "goc": "waste-wood/e-care (GitHub); HF 12ml/e-CARE @ a9c788710144e6edd31f925345fa107c0634e223"},
    "headline_cause/en_simple": {"license_verbatim": "cc0-1.0 (IlyaGusev/headline_cause @ 68c78423fe4653638a87ec91140488bb3cb97061)", "license_use": "commercial (CC0-1.0)",
                                 "cung_khuon_copa": False, "goc": "IlyaGusev/headline_cause @ 68c78423fe4653638a87ec91140488bb3cb97061 (en/simple)"},
}
QUOTA = 500
base_src = Counter(r["source"] for r in base)
nB_plan = len(base) + len(B1rows) + QUOTA * len(NEW)
capB = int(CAP * nB_plan)
rng = np.random.default_rng([SEED, 2])
causal, info = [], {}
seen_keys = {(normalize(r["state"]), normalize(r["question"] or ""), tuple(r["options"])) for r in B1rows}
for s, meta in NEW.items():
    idx = [k for k, x in enumerate(src_col) if x == s]
    d = Counter()
    clean = []
    for k in idx:
        if id_col[k] in used_ids or gid_col[k] in used_groups:
            d["da_dung_v0"] += 1; continue
        r = ts_row(ts[k])
        # lựa chọn Yes/No của câu noul là chuỗi cố định của dự án (train_format.json), không phải dữ liệu: không kiểm (lượt đầu chúng trùng
        # nguyên văn lựa chọn "No" của một câu Belebele trong bench và kéo theo 171 dòng noul bị loại nhầm)
        units = [r["state"], r["question"] or ""] + ([] if r["kind"] == "noul" else r["options"])
        hit = next((h for t in units for h in [COP.check(t)] if h), None)
        if hit:
            d[f"ho_copa_bench:{hit[0]}:{hit[1].split(':')[0]}"] += 1; continue
        clean.append(r)
    nclean = len(clean)
    fin = make_finalize(tok, [("ref_en", CHK["en"]), ("v0", V0)], d, max_state_tokens=1024)  # dựng với 1.024 như v0
    got, groups = [], set()
    for j in rng.permutation(len(clean)):
        if len(got) >= min(QUOTA, capB - base_src[s]):
            break
        r = clean[int(j)]
        g = r["group_keys"][0]
        if g in groups:
            d["cung_group_id"] += 1; continue
        key = (normalize(r["state"]), normalize(r["question"] or ""), tuple(r["options"]))
        if key in seen_keys:
            d["trung_trong_phan_B"] += 1; continue
        r = fin(r)
        if r:
            groups.add(g); seen_keys.add(key)
            r.update(split="train", target_argmax=int(np.argmax(r["target"])), group_id=r.pop("group_keys")[0][2:], phan="B", dang_bai="nhan_qua",
                     nguon_nhan="nguoi_gan", license_verbatim=meta["license_verbatim"], license_use=meta["license_use"],
                     cung_khuon_copa=meta["cung_khuon_copa"], dataset_revision=f"{TS[0]}@{TS[1]}; gốc: {meta['goc']}")
            got.append(r)
    causal += got
    info[s] = {"kho": len(idx), "sach_sau_kiem_ho_copa": nclean, "lay": len(got), "kinds": dict(Counter(r["kind"] for r in got)),
               "v0_trong_nen": base_src[s], "loai": dict(d.most_common())}
    print("B", s, info[s], flush=True)
B = B1rows + causal
nB = len(base) + len(B)
shareB = Counter(r["source"] for r in base + B)
assert max(shareB.values()) <= CAP * nB, shareB.most_common(2)
nC = len(base) + len(A) + len(B)
shareC = Counter(r["source"] for r in base + A + B)
assert max(shareC.values()) <= CAP * nC
st["phan_B"] = {"dong": len(B), "nhan_qua": info, "theo_dang_bai": dict(Counter(r["dang_bai"] for r in B))}

# ---------------------------------------------------------------- 4) kiểm cuối
final = Counter()
for rows, lang in ((A, "vi"), (B, "en"), (base, "vi"), (base, "en")):
    for r in rows:
        for t in [r["state"]] + [t for t in [r["question"] or ""] + r["options"] if len(normalize(t).split()) >= 4]:
            h = CHK[lang].check(t)
            if h:
                final[f"{r.get('phan', 'v0')}|{lang}|{h[0]}|{h[1]}"] += 1; break
for r in causal:
    if any(COP.check(t) for t in [r["state"], r["question"] or ""] + ([] if r["kind"] == "noul" else r["options"])):
        final["B_nhan_qua|ho_copa"] += 1
st["kiem_cuoi"] = dict(final)
assert not final, final

# ---------------------------------------------------------------- 5) ghi
write_jsonl(OUT / "part_a.jsonl", A)
write_jsonl(OUT / "part_b_en.jsonl", B)
META = ["id", "source", "dataset", "dataset_revision", "kind", "variant", "target", "target_argmax", "license_use", "license_verbatim", "nguon_nhan", "dang_bai",
        "group_id", "n_state_tokens", "text_lang"]
write_jsonl(OUT / "part_b_meta.jsonl", [{**{k: r[k] for k in META}, **({"cung_khuon_copa": r["cung_khuon_copa"]} if "cung_khuon_copa" in r else {})} for r in B])
TT = OUT / "to_translate"
exp = [{"id": r["id"], "state": r["state"], "question": r["question"] or "", "options": [] if r["kind"] == "noul" else list(r["options"])} for r in B]
write_jsonl(TT / "part_b_to_translate.jsonl", exp)
qc, oc = Counter(x["question"] for x in exp), Counter(o for x in exp for o in x["options"])
(TT / "chuoi_lap_lai.json").write_text(json.dumps({"cau_hoi_lap_lai": {q: n for q, n in qc.most_common() if n >= 5},
                                                   "lua_chon_lap_lai": {o: n for o, n in oc.most_common() if n >= 5}}, ensure_ascii=False, indent=1), encoding="utf-8")
ntok = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0
tt = {"state": sum(ntok(x["state"]) for x in exp), "question": sum(ntok(x["question"]) for x in exp), "options": sum(ntok(o) for x in exp for o in x["options"])}
uniq = {x["state"] for x in exp} | {x["question"] for x in exp} | {o for x in exp for o in x["options"]}
st["to_translate"] = {"rows": len(exp), "noul_rows_khong_co_lua_chon": sum(1 for x in exp if not x["options"]), "tokens_en": {**tt, "tong": sum(tt.values())},
                      "chuoi_khac_nhau": len(uniq), "tokens_en_chuoi_khac_nhau": sum(ntok(s) for s in uniq),
                      "theo_dang_bai": {g: {"dong": sum(1 for r in B if r["dang_bai"] == g),
                                            "tokens_en": sum(ntok(x["state"]) + ntok(x["question"]) + sum(ntok(o) for o in x["options"])
                                                             for x, r in zip(exp, B) if r["dang_bai"] == g)} for g in sorted({r["dang_bai"] for r in B})},
                      "sha256": sha256_file(TT / "part_b_to_translate.jsonl")}
write_jsonl(OUT / "components/A.jsonl", base + A)
plan = lambda name, parts: {"name": name, "ban": "v2 (quyết định người dùng 17/10)",
                            "base_train_v0": {"file": "data/train_v0.jsonl", "sha256": sha256_file(DATA / "train_v0.jsonl"), "rows": len(base), "bo_ra": sorted(drop0),
                                              "ghi_chu": f"bỏ {st['nen']['bo_vi']} dòng VI + {st['nen']['bo_en']} dòng EN chạm chỉ mục mở rộng; phần tasksource còn "
                                                         f"{st['nen']['tasksource_con']} (không còn đúng 16.000)"},
                            "parts": parts, "rows_total": len(base) + sum(p["rows"] for p in parts.values())}
pA = {"A": {"file": "results/train_v1/part_a.jsonl", "sha256": sha256_file(OUT / "part_a.jsonl"), "rows": len(A),
            "trang_thai": "2.xxx dòng dùng chuỗi tv1-* chờ người dùng duyệt; dựng lại sau khi duyệt, đọc gia_tri_moi trong edits_log".replace(
                "2.xxx", f"{sum(1 for r in A if r['chuoi_cho_duyet']):,}".replace(",", "."))}}
pB = {"B": {"file_en": "results/train_v1/part_b_en.jsonl", "sha256_en": sha256_file(OUT / "part_b_en.jsonl"), "rows": len(B),
            "trang_thai": "CHỜ BẢN DỊCH: kiểm bằng to_translate/kiem_ban_dich.py, khử rò rỉ phía tiếng Việt bằng scripts/train_v1/decontam_vi.py, "
                          "người dùng đọc kiểm 30 dòng (scripts/train_v1/make_review_html.py) rồi mới dùng"}}
(OUT / "components/B_plan.json").write_text(json.dumps(plan("B", pB), ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "components/C_plan.json").write_text(json.dumps(plan("C", {**pA, **pB}), ensure_ascii=False, indent=1), encoding="utf-8")
st["thanh_phan"] = {"A": {"rows": nA, "max_share": round(max(shareA.values()) / nA, 4), "sha256": sha256_file(OUT / "components/A.jsonl")},
                    "B": {"rows": nB, "max_share": round(max(shareB.values()) / nB, 4)}, "C": {"rows": nC, "max_share": round(max(shareC.values()) / nC, 4)},
                    "cap_B": capB}
st["files"] = {f: sha256_file(OUT / f) for f in ["part_a.jsonl", "part_b_en.jsonl", "part_b_meta.jsonl", "components/A.jsonl"]}
(OUT / "stage0_stats_v2.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: st[k] for k in ("nen", "thanh_phan", "to_translate", "kiem_cuoi")}, ensure_ascii=False))
