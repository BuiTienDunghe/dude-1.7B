"""PHẦN 2: dựng benchmark hiệu đính một cách xác định = bản gốc (data/bench_v1_1) + lớp ghi đè của người (data/human_edits).
Chỉ chạy khi người làm dự án yêu cầu (sau một mốc "Chốt đợt"); KHÔNG tự chạy theo mốc.

- Sửa ở "Mẫu và nhãn" (khóa trong tools/review_app/inventory.py) áp cho cả dev và test: thay đúng chuỗi cũ bằng chuỗi mới trên dòng
  tiếng Việt có dấu của nguồn dùng chuỗi đó. Bản tiếng Anh giữ nguyên.
- Sửa ở "Từng câu" (state["cau"]) áp cho dev: đáp án (cả ba biến thể cùng gốc: EN, có dấu, không dấu), câu hỏi / lựa chọn / state tiếng Việt;
  câu có "mơ hồ nên loại" bị loại khỏi bản hiệu đính (cả ba biến thể).
- Sửa ở test chỉ đến từ đề xuất theo câu đã được người duyệt ở màn "Chờ duyệt" (state["cau_test"], split "test"): áp cùng cách cho test.
- Bản không dấu sinh lại từ văn bản đã sửa bằng common.strip_diacritics (đúng phép của bench). n_tokens / n_state_tokens tính lại cho
  dòng đổi chữ (render_prompt, tokenizer Qwen3-0.6B như bench 1.0/1.1).
- Test chỉ đọc qua script, không in nội dung. Nếu test đổi: ghi test_v1.<k>.jsonl mới (lần đầu là test_v1.2.jsonl), đóng băng (SHA-256 + chỉ đọc);
  test_v1.1 giữ nguyên.
- Báo cáo: số câu sửa theo nguồn và theo lý do, danh sách thay đổi (ra --out_dir/build_report_r<k>.json; chỉ id, trường, lý do; không chứa văn bản).

Cách dùng:
  build_revised.py --out_dir data/bench_v1_1_rev            (dựng thật)
  build_revised.py --check_identity                         (chạy thử: phải ra dev trùng từng byte và test trùng hash khi lớp ghi đè rỗng)
"""
import argparse
import hashlib
import json
import os
import re
import stat
import unicodedata
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "review_app"))
from common import CONFIG, DATA, load_json, model_revisions, read_jsonl, render_prompt, strip_diacritics  # noqa: E402
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base_dir", default=str(DATA / "bench_v1_1"))
ap.add_argument("--edits_dir", default=str(DATA / "human_edits"))
ap.add_argument("--config_dir", default=str(CONFIG))
ap.add_argument("--out_dir", default=str(DATA / "bench_v1_1_rev"))
ap.add_argument("--check_identity", action="store_true")
ap.add_argument("--test_labels", default=None, help="quyết định nhãn test (đổi nhãn / loại), chỉ dùng dòng split='test'; ví dụ results/human_review/label_decisions_remote.jsonl")
ap.add_argument("--nfc", action="store_true", help="chuẩn hóa NFC mọi trường tiếng Việt (state, question, options); bản không dấu sinh lại từ bản NFC")
ap.add_argument("--provenance", action="store_true", help="thêm vi_provenance (source|revised), vi_revised_fields, gold_provenance (source|overridden) vào mọi dòng")
ap.add_argument("--trial", action="store_true", help="dựng thử: ghi ra --out_dir ngoài data/, KHÔNG chmod, KHÔNG ghi FROZEN")
args = ap.parse_args()
base = Path(args.base_dir)
man_dev, man_test = load_json(base / "manifest_dev_v1_1.json"), load_json(base / "manifest_test_v1.1.json")
DEV_P, TEST_P = base / "dev_v1_1.jsonl", base / "test_v1.1.jsonl"
sha = lambda b: hashlib.sha256(b).hexdigest()
dev_bytes0, test_bytes0 = DEV_P.read_bytes(), TEST_P.read_bytes()
assert sha(dev_bytes0) == man_dev["sha256"]["dev_v1_1.jsonl"] and sha(test_bytes0) == man_test["sha256"]["test_v1.1.jsonl"], "bản gốc khác manifest"
dev, test = read_jsonl(DEV_P), read_jsonl(TEST_P)
INV = inventory(args.config_dir)
state = E.derive_state(E.read_log(Path(args.edits_dir) / "edits_log.jsonl"), INV)
tmpl = {k: v for k, v in state["mau"].items() if v["gia_tri"] != v["goc"]}
ITEMS = {"dev": state["cau"], "test": state["cau_test"]}
TEST_LAB = {}      # nhãn test: công cụ không có đường sửa test, nên quyết định nằm ở file riêng (chỉ dòng split=test; dòng dev đã nằm trong lớp ghi đè)
if args.test_labels:      # dòng MỚI NHẤT theo (split, pair_id) thắng: một dòng "giu" ghi sau "loai"/"doi_nhan" là hoàn tác
    latest = {}
    for x in read_jsonl(args.test_labels):
        if x.get("split") == "test":
            latest[x["pair_id"]] = x
    TEST_LAB = {p: x for p, x in latest.items() if x.get("hanh_dong") in ("loai", "doi_nhan")}
NFC = (lambda t: unicodedata.normalize("NFC", t)) if args.nfc else (lambda t: t)
nfc_cmp = lambda t: unicodedata.normalize("NFC", t)
CHANGES = []       # mỗi cặp đổi: split, pair_id, nguon, loai_thay_doi (một cặp có thể thuộc nhiều loại), do_chuoi_mau
EN_NOT_NFC = Counter()
_tok = {}


def ntok(r, field="prompt"):
    if "tok" not in _tok:
        from transformers import AutoTokenizer
        _tok["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
        _tok["frames"] = load_json(CONFIG / "prompt_frames.json")
    tok = _tok["tok"]
    if field == "state":
        return len(tok.encode(r["state"], add_special_tokens=False))
    return len(tok.encode(render_prompt(r, tok, _tok["frames"]), add_special_tokens=False))


def nodiac_of(r, old_nd):
    n = {**r, "id": r["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(r["state"]),
         "question": strip_diacritics(r["question"]), "options": [strip_diacritics(o) for o in r["options"]]}
    for a, b in ((r["state"], n["state"]), (r["question"], n["question"])) + tuple(zip(r["options"], n["options"])):
        assert re.findall(r"\d+", a) == re.findall(r"\d+", b), r["id"]
    n["n_tokens"] = ntok(n)
    n["n_state_tokens"] = old_nd["n_state_tokens"] if n["state"] == old_nd["state"] else ntok(n, "state")
    return n


def template_apply(r, changes):
    """Áp sửa ở "Mẫu và nhãn" lên một dòng tiếng Việt có dấu. -> (dòng mới, danh sách khóa đã áp)."""
    s, kind, used = r["source"], r["kind"], []
    q, opts = r["question"], list(r["options"])
    for key, m in tmpl.items():
        old, new = m["goc"], m["gia_tri"]
        if key == "translations.questions.sib200" and s == "sib200" and q == old:
            q = new; used.append(key)
        elif key == "translations.questions.massive" and s == "massive" and q == old:
            q = new; used.append(key)
        elif key in ("translations.questions.xcopa_cause", "translations.questions.xcopa_effect") and s == "xcopa" and q == old and \
                f"copa_question={key.rsplit('_', 1)[1]}" in r["flags"]:
            q = new; used.append(key)
        elif key in ("bench_v1_questions.factcheck_choice.question", "bench_v1_questions.factcheck_noul.question") and s in ("vifactcheck", "viwikifc") \
                and kind == key.split(".")[1].split("_")[1]:
            pre_old, pre_new = old.split("{claim}")[0], new.split("{claim}")
            assert old.endswith("{claim}") and len(pre_new) == 2 and pre_new[1] == "", f"mẫu câu hỏi phải kết thúc bằng {{claim}}: {key}"
            assert q.startswith(pre_old), r["id"]
            q = pre_new[0] + q[len(pre_old):]; used.append(key)
        elif (key.startswith("translations.sib200_topics.") and s == "sib200") or (key.startswith("translations.massive_intents.") and s == "massive") or \
                (key.startswith("bench_v1_questions.factcheck_choice.options.") and s in ("vifactcheck", "viwikifc") and kind == "choice"):
            if old in opts:
                opts = [new if o == old else o for o in opts]; used.append(key)
        elif key.startswith("train_format.noul_options.vi.") and s in ("vifactcheck", "viwikifc") and kind == "noul":
            j = int(key.rsplit(".", 1)[1])
            assert opts[j] == old, r["id"]
            opts[j] = new; used.append(key)
    return {**r, "question": q, "options": opts}, used


def rebuild(rows, is_dev):
    by_id = {r["id"]: r for r in rows}
    out, log, used_keys = [], [], Counter()
    split = "dev" if is_dev else "test"
    items = ITEMS[split]
    drop = {pid for pid, c in items.items() if "loai_bo" in c}
    drop_lab = {p for p, x in TEST_LAB.items() if x["hanh_dong"] == "loai"} if not is_dev else set()
    relab = {p: int(x["dap_an_moi"]) for p, x in TEST_LAB.items() if x["hanh_dong"] == "doi_nhan"} if not is_dev else {}
    for r in rows:
        if r["variant"] == "nodiac":
            continue                                   # sinh lại ngay sau dòng có dấu tương ứng
        if r["lang"] == "en" and any(t != nfc_cmp(t) for t in [r["state"], r["question"]] + list(r["options"])):
            EN_NOT_NFC[split] += 1
        if r["pair_id"] in drop or r["pair_id"] in drop_lab:
            if r["lang"] == "vi":
                why = items[r["pair_id"]]["loai_bo"]["ly_do"] if r["pair_id"] in drop else "quyết định nhãn test: loại"
                log.append({"split": split, "pair_id": r["pair_id"], "nguon": r["source"], "thay_doi": "loại", "ly_do": why})
                CHANGES.append({"split": split, "pair_id": r["pair_id"], "nguon": r["source"], "loai_thay_doi": ["loai_cau"], "do_chuoi_mau": False})
            continue
        n = dict(r); changed = False
        if r["lang"] == "vi":
            n, used = template_apply(n, None)
            changed |= bool(used); used_keys.update(used)
            tmpl_view = (nfc_cmp(n["state"]), nfc_cmp(n["question"]), [nfc_cmp(o) for o in n["options"]])   # sau chuỗi mẫu, trước sửa của người
        c = items.get(r["pair_id"], {})
        if "dap_an" in c and c["dap_an"]["gia_tri_moi"] != r["target"]:
            n["target"] = int(c["dap_an"]["gia_tri_moi"]); changed = True
        if r["pair_id"] in relab:
            assert 0 <= relab[r["pair_id"]] < len(r["options"]), r["id"]
            if relab[r["pair_id"]] != r["target"]:
                n["target"] = relab[r["pair_id"]]; changed = True
        if r["lang"] == "vi":
            if "cau_hoi" in c:
                n["question"] = c["cau_hoi"]["gia_tri_moi"]
            if "state" in c:
                n["state"] = c["state"]["gia_tri_moi"]
            for f, v in c.items():
                if f.startswith("lua_chon:"):
                    n["options"] = list(n["options"]); n["options"][int(f.split(":")[1])] = v["gia_tri_moi"]
            n["state"], n["question"], n["options"] = NFC(n["state"]), NFC(n["question"]), [NFC(o) for o in n["options"]]
            text_changed = (n["question"], n["options"], n["state"]) != (r["question"], r["options"], r["state"])
            cats = (["doi_dap_an"] if n["target"] != r["target"] else []) +                    (["sua_state"] if nfc_cmp(n["state"]) != nfc_cmp(r["state"]) else []) +                    (["sua_cau_hoi"] if nfc_cmp(n["question"]) != nfc_cmp(r["question"]) else []) +                    (["sua_lua_chon"] if [nfc_cmp(o) for o in n["options"]] != [nfc_cmp(o) for o in r["options"]] else [])
            if not cats and text_changed:
                cats = ["chi_doi_nfc"]
            if cats:
                CHANGES.append({"split": split, "pair_id": r["pair_id"], "nguon": r["source"], "loai_thay_doi": cats, "do_chuoi_mau": bool(used)})
            if args.provenance:     # phần đã sửa = khác bản sau-chuỗi-mẫu (NFC và chuỗi mẫu không tính)
                rev = (["state"] if nfc_cmp(n["state"]) != tmpl_view[0] else []) + (["question"] if nfc_cmp(n["question"]) != tmpl_view[1] else []) +                       [f"option:{j}" for j, o in enumerate(n["options"]) if nfc_cmp(o) != tmpl_view[2][j]]
                n["vi_provenance"], n["vi_revised_fields"] = ("revised" if rev else "source"), rev
            if text_changed:
                n["n_tokens"] = ntok(n)
                if n["state"] != r["state"]:
                    n["n_state_tokens"] = ntok(n, "state")
            changed |= text_changed
        if c and r["lang"] == "vi":
            for f, v in c.items():
                log.append({"split": split, "pair_id": r["pair_id"], "nguon": r["source"], "truong": f, "ly_do": v["ly_do"], "de_xuat_id": v.get("de_xuat_id"),
                            "ai": v["ai"], "thoi_diem": v["thoi_diem"]})
        if args.provenance:
            n["gold_provenance"] = "overridden" if n["target"] != r["target"] else "source"
        out.append(n)
        if r["lang"] == "vi":
            old_nd = by_id[r["id"] + ":nodiac"]
            nd = nodiac_of(n, old_nd) if changed else dict(old_nd)
            if args.provenance:
                for k_ in ("vi_provenance", "vi_revised_fields", "gold_provenance"):
                    nd[k_] = n[k_]
            out.append(nd)
    return out, log, used_keys


def dumps(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


dev_new, dev_log, dev_used = rebuild(dev, True)
test_new, test_log, test_used = rebuild(test, False)
dev_b, test_b = dumps(dev_new), dumps(test_new)
res = {"template_overrides": sorted(tmpl), "item_overrides": len(ITEMS["dev"]), "item_overrides_test": len(ITEMS["test"]), "dev_rows": len(dev_new), "test_rows": len(test_new),
       "dev_identical_to_base": dev_b == dev_bytes0, "test_identical_to_base": test_b == test_bytes0,
       "dev_sha256": sha(dev_b), "test_sha256": sha(test_b), "test_sha256_base": man_test["sha256"]["test_v1.1.jsonl"],
       "template_keys_applied_dev": dict(dev_used), "template_keys_applied_test": dict(test_used)}
if args.check_identity:
    print(json.dumps({k: res[k] for k in ("template_overrides", "item_overrides", "item_overrides_test", "dev_rows", "test_rows", "dev_identical_to_base",
                                          "test_identical_to_base", "dev_sha256", "test_sha256", "test_sha256_base")}, ensure_ascii=False))
    sys.exit(0 if res["dev_identical_to_base"] and res["test_identical_to_base"] else 1)


def cross_leak(dev_rows, test_rows):
    """Rò rỉ dev↔test đúng cách build_bench_v1_1 (chỉ mục state 'orig'; khớp chính xác / MinHash / 13-gram + câu ngắn nằm trọn). Phải bằng 0."""
    from decontam import Index
    ti, di = Index(), Index()
    for r in test_rows:
        if r["variant"] == "orig":
            ti.add(r["state"], r["id"])
    for r in dev_rows:
        if r["variant"] == "orig":
            di.add(r["state"], r["id"])
    hd = [{"dev_id": r["id"], "kieu": h[0], "test_id": h[1]} for r in dev_rows if r["variant"] == "orig" for h in [ti.check(r["state"])] if h]
    ht = [{"test_id": r["id"], "kieu": h[0], "dev_id": h[1]} for r in test_rows if r["variant"] == "orig" for h in [di.check(r["state"])] if h]
    return {"dev_states_hitting_test": len(hd), "test_states_hitting_dev": len(ht), "chi_tiet": hd + ht}


LEAK = cross_leak(dev_new, test_new)    # chốt kiểm: dựng thử thì ghi + cảnh báo; dựng thật thì chặn
if LEAK["dev_states_hitting_test"] or LEAK["test_states_hitting_dev"]:
    print("CẢNH BÁO rò rỉ dev↔test:", json.dumps(LEAK, ensure_ascii=False), file=sys.stderr)
    assert args.trial, "rò rỉ dev↔test khác 0: không dựng bản thật"

if args.trial:      # dựng thử: không đóng băng, không chmod, không ghi vào data/
    out = Path(args.out_dir).resolve()
    assert DATA.resolve() not in out.parents and out != DATA.resolve(), "dựng thử không được ghi vào data/"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_v1_2_trial.jsonl").write_bytes(dev_b); (out / "test_v1_2_trial.jsonl").write_bytes(test_b)
    with open(out / "changes_v1_2_trial.jsonl", "w", encoding="utf-8") as f:
        for c_ in CHANGES:
            f.write(json.dumps(c_, ensure_ascii=False) + chr(10))
    rep = {**res, "trial": True, "nfc": args.nfc, "test_labels": args.test_labels, "test_label_decisions": dict(Counter(x["hanh_dong"] for x in TEST_LAB.values())),
           "en_rows_not_nfc": dict(EN_NOT_NFC), "n_changed_pairs": dict(Counter(c_["split"] for c_ in CHANGES)),
           "base": {"dev_v1_1.jsonl": man_dev["sha256"]["dev_v1_1.jsonl"], "test_v1.1.jsonl": man_test["sha256"]["test_v1.1.jsonl"]},
           "changes_dev": dev_log, "changes_test": test_log, "milestones": state["dot"], "edits_log_records": state["so_ban_ghi"], "leak_dev_test": LEAK}
    (out / "build_report_v1_2_trial.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({**{k_: rep[k_] for k_ in ("dev_rows", "test_rows", "dev_sha256", "test_sha256", "n_changed_pairs", "en_rows_not_nfc", "test_label_decisions",
                                                "template_overrides")}, "leak_dev_test": {k_: LEAK[k_] for k_ in ("dev_states_hitting_test", "test_states_hitting_dev")}},
                     ensure_ascii=False))
    sys.exit(0)

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
k = 1 + sum(1 for p in out.glob("dev_v1_1_r*.jsonl"))
(out / f"dev_v1_1_r{k}.jsonl").write_bytes(dev_b)
if not res["test_identical_to_base"]:
    j = 2 + sum(1 for p in out.glob("test_v1.*.jsonl"))
    tp = out / f"test_v1.{j}.jsonl"
    tp.write_bytes(test_b); os.chmod(tp, stat.S_IREAD)
    (out / f"FROZEN_test_v1.{j}.sha256").write_text(f"{sha(test_b)}  {tp.name}\n", encoding="utf-8")
    res["test_file"] = tp.name
else:
    res["test_file"] = "không đổi: dùng data/bench_v1_1/test_v1.1.jsonl"
by_src = Counter(x["nguon"] for x in dev_log); by_reason = Counter(x.get("ly_do") for x in dev_log)
rep = {**res, "dev_file": f"dev_v1_1_r{k}.jsonl", "changes_dev": dev_log, "changes_by_source": dict(by_src), "changes_by_reason": dict(by_reason),
       "excluded_pairs": sorted({x["pair_id"] for x in dev_log if x.get("thay_doi", "").startswith("loại")}),
       "changes_test": test_log, "changes_by_source_test": dict(Counter(x["nguon"] for x in test_log)),
       "changes_by_reason_test": dict(Counter(x.get("ly_do") for x in test_log)), "milestones": state["dot"]}
(out / f"build_report_r{k}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k_: rep[k_] for k_ in ("dev_file", "test_file", "dev_rows", "test_rows", "changes_by_source", "changes_by_reason", "changes_by_source_test",
                                         "changes_by_reason_test")}, ensure_ascii=False))
