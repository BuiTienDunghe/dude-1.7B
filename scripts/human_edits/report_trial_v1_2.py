"""Báo cáo dựng thử bench 1.2 (khu tạm results/bench_v1_2_trial/; chưa đóng băng). Python Windows, thư viện chuẩn.
- Thống kê thay đổi theo split × nguồn × loại (đổi đáp án, sửa câu hỏi, sửa lựa chọn, sửa state, loại câu) + cột "chỉ đổi do NFC".
  Một cặp có thể thuộc nhiều loại: đếm cặp ở MỌI loại nó thuộc; "chỉ đổi do NFC" chỉ tính cặp đổi byte mà không thuộc loại nào khác.
- Manifest: hash, tùy chọn dựng, test_ids_nguoi_da_xem, test_ids_agent_da_doc, trạng thái release_check.
- Điểm DEV: preds 1.2 = preds 1.1 cho dòng trùng byte + preds chấm lại cho dòng đổi byte; dòng bị loại bỏ khỏi điểm. Bảng model × nguồn × biến thể:
  1.1 (đủ), 1.1 chỉ trên các cặp còn lại, 1.2. Ô nhảy > 2 điểm (1.2 so với 1.1): liệt kê cặp đổi đúng/sai và cặp bị loại.
Ra: manifest_v1_2_trial.json, REPORT_bench_v1_2_trial.md, scores_v1_2_trial.json, eval/<model>/preds.jsonl (preds dev 1.2 đã gộp).
"""
import hashlib
import json
_PROV = ("vi_provenance", "vi_revised_fields", "gold_provenance")
def _core(l):
    """Dòng JSONL bỏ các trường provenance (để so với 1.1 / bản thử trước)."""
    d = json.loads(l)
    return json.dumps({k: v for k, v in d.items() if k not in _PROV}, ensure_ascii=False)
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/bench_v1_2_trial"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
lines = lambda p: [l for l in open(p, encoding="utf-8") if l.strip()]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
D11, T11 = ROOT / "data/bench_v1_1/dev_v1_1.jsonl", ROOT / "data/bench_v1_1/test_v1.1.jsonl"
D12, T12 = T / "dev_v1_2_trial.jsonl", T / "test_v1_2_trial.jsonl"
br = json.load(open(T / "build_report_v1_2_trial.json", encoding="utf-8"))
changes = read(T / "changes_v1_2_trial.jsonl")
CH = {(c["split"], c["pair_id"]): c for c in changes}
CATS = ["doi_dap_an", "sua_cau_hoi", "sua_lua_chon", "sua_state", "loai_cau", "chi_doi_nfc"]
CAT_VI = {"doi_dap_an": "đổi đáp án", "sua_cau_hoi": "sửa câu hỏi", "sua_lua_chon": "sửa lựa chọn", "sua_state": "sửa state", "loai_cau": "loại câu",
          "chi_doi_nfc": "chỉ đổi do NFC"}
NAME = {"belebele": "Belebele", "sib200": "SIB-200", "massive": "MASSIVE", "xcopa": "XCOPA", "vifactcheck": "ViFactCheck", "viwikifc": "ViWikiFC"}

# ---------------------------------------------------------------- thống kê thay đổi
rows11 = {"dev": read(D11), "test": read(T11)}
pairs11 = {s: Counter(r["source"] for r in rs if r["lang"] == "vi" and r["variant"] == "orig") for s, rs in rows11.items()}
stat = defaultdict(Counter)
for c in changes:
    for k in c["loai_thay_doi"]:
        stat[(c["split"], c["nguon"])][k] += 1
    stat[(c["split"], c["nguon"])]["tong_cap_doi"] += 1
    stat[(c["split"], c["nguon"])]["do_chuoi_mau"] += bool(c["do_chuoi_mau"])

# ---------------------------------------------------------------- id test đã xem
seen = defaultdict(set)
for p in read(ROOT / "config/vi_proposals.jsonl"):
    if p.get("loai") == "cau" and p.get("split") == "test":
        seen["vi_proposals"].add(p["pair_id"])
for x in read(ROOT / "results/human_review/label_decisions_remote.jsonl"):
    if x.get("split") == "test":
        seen["label_decisions_remote"].add(x["pair_id"])
for f in sorted((ROOT / "results/translation_audit/label_second").glob("in_*.jsonl")):
    for x in read(f):
        if str(x.get("audit_id", "")).startswith("test|"):
            seen["label_issues"].add(x["pair_id"])
test_pairs_all = sorted({r["pair_id"] for r in rows11["test"]})
nguoi = sorted(set().union(*seen.values()))
assert set(nguoi) <= set(test_pairs_all)

# ---------------------------------------------------------------- release_check
rc = subprocess.run([sys.executable, str(ROOT / "scripts/human_edits/release_check.py")], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
rc_head = [l for l in rc.stdout.splitlines() if l and not l.startswith("  ")][:6]

# ---------------------------------------------------------------- điểm dev
MODELS = {"decider-2b-text": "decider/decider-2b-text", "decider-2b-keyed": "decider/decider-2b-keyed", "decider-ft-ep2-text": "ftd/decider-2b-text",
          "decider-ft-ep2-keyed": "ftd/decider-2b-keyed", "Qwen3-1.7B-fullvocab-s1": "control", "Qwen3-1.7B": "base", "strip-s1": "strip"}
old_line = {json.loads(l)["id"]: l for l in lines(D11)}
dev12_lines = lines(D12)
dev12 = {json.loads(l)["id"]: json.loads(l) for l in dev12_lines}
changed_ids = {json.loads(l)["id"] for l in dev12_lines if _core(old_line[json.loads(l)["id"]]) != _core(l)}
dev11 = {r["id"]: r for r in rows11["dev"]}
VARS = [("en", "orig"), ("vi", "orig"), ("vi", "nodiac")]
cell = lambda r: (r["source"], r["lang"], r["variant"])
# cặp "source": bản VI giữ nguyên như nguồn (NFC, chuỗi mẫu không tính) và đáp án theo nguồn
PAIR_SRC = {r["pair_id"]: (r.get("vi_provenance") == "source" and r.get("gold_provenance") == "source") for r in dev12.values() if r["lang"] == "vi" and r["variant"] == "orig"}
scores, flags, missing_models = {}, [], []
for m, sub in MODELS.items():
    p11 = {x["id"]: x for x in read(ROOT / f"results/dev_v1_1/eval/{m}/preds.jsonl")}
    rp = T / f"eval_rescore/{sub}/preds.jsonl"
    if not rp.exists():
        missing_models.append(m); continue
    pr = {x["id"]: x for x in read(rp)}
    assert changed_ids <= set(pr), (m, len(changed_ids - set(pr)))
    assert set(dev11) <= set(p11), m
    p12 = {i: (pr[i] if i in changed_ids else p11[i]) for i in dev12}
    (T / f"eval/{m}").mkdir(parents=True, exist_ok=True)
    with open(T / f"eval/{m}/preds.jsonl", "w", encoding="utf-8") as f:
        for i in dev12:
            x = dict(p12[i]); x["target"] = dev12[i]["target"]; x["correct"] = x["pred"] == dev12[i]["target"]; x["rescored"] = i in changed_ids
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    c11 = {i: int(p11[i]["pred"] == r["target"]) for i, r in dev11.items()}
    c12 = {i: int(p12[i]["pred"] == r["target"]) for i, r in dev12.items()}
    sc = {}
    for src in sorted({r["source"] for r in dev11.values()}) + ["*"]:
        for l, v in VARS:
            ids11 = [i for i, r in dev11.items() if (src == "*" or r["source"] == src) and (r["lang"], r["variant"]) == (l, v)]
            if not ids11:
                continue
            ids_rem = [i for i in ids11 if i in dev12]
            a11 = sum(c11[i] for i in ids11) / len(ids11); a11r = sum(c11[i] for i in ids_rem) / len(ids_rem); a12 = sum(c12[i] for i in ids_rem) / len(ids_rem)
            ids_src = [i for i in ids_rem if PAIR_SRC.get(dev12[i]["pair_id"], False)]
            a12s = sum(c12[i] for i in ids_src) / len(ids_src) if ids_src else float("nan")
            sc[f"{src}|{l}|{v}"] = {"n11": len(ids11), "n12": len(ids_rem), "acc11": a11, "acc11_con_lai": a11r, "acc12": a12, "n12_source": len(ids_src), "acc12_source": a12s}
            if src != "*" and abs(a12 - a11) > 0.02:
                flip = [{"pair_id": dev11[i]["pair_id"], "id": i, "dung_11": c11[i], "dung_12": c12[i],
                         "loai_thay_doi": CH.get(("dev", dev11[i]["pair_id"]), {}).get("loai_thay_doi", ["không đổi"])} for i in ids_rem if c11[i] != c12[i]]
                dropped = [i for i in ids11 if i not in dev12]
                flags.append({"model": m, "nguon": src, "bien_the": f"{l}|{v}", "acc11": a11, "acc11_con_lai": a11r, "acc12": a12,
                              "doi_dung_sai": flip, "bi_loai": {"n": len(dropped), "dung_o_11": sum(c11[i] for i in dropped)}})
    scores[m] = sc
(T / "scores_v1_2_trial.json").write_text(json.dumps({"scores": scores, "flags": flags, "changed_rows": len(changed_ids), "missing_models": missing_models},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- manifest
man = {"name": "bench 1.2 — DỰNG THỬ (khu tạm; CHƯA đóng băng; không ghi data/)", "build_cmd": "bash scripts/human_edits/run_trial_v1_2.sh",
       "options": {"nfc": br["nfc"], "test_labels": br["test_labels"], "template_overrides": br["template_overrides"], "edits_log_records": br["edits_log_records"]},
       "base": br["base"], "base_unchanged": {"dev_v1_1.jsonl": sha(D11) == br["base"]["dev_v1_1.jsonl"], "test_v1.1.jsonl": sha(T11) == br["base"]["test_v1.1.jsonl"]},
       "files": {"dev_v1_2_trial.jsonl": sha(D12), "test_v1_2_trial.jsonl": sha(T12)}, "rows": {"dev": br["dev_rows"], "test": br["test_rows"]},
       "pairs_11": {s: sum(c.values()) for s, c in pairs11.items()}, "changed_pairs": br["n_changed_pairs"], "en_rows_not_nfc": br["en_rows_not_nfc"],
       "test_label_decisions_applied": br["test_label_decisions"],
       "change_counting": "một cặp được đếm ở MỌI loại nó thuộc; 'chi_doi_nfc' chỉ cho cặp đổi byte mà không thuộc loại nào khác; 'loai_cau' loại trừ các loại khác",
       "changes_by_split_source": {f"{s}|{n}": dict(c) for (s, n), c in sorted(stat.items())},
       "test_ids_nguoi_da_xem": nguoi, "test_ids_nguoi_da_xem_nguon": {k: len(v) for k, v in seen.items()},
       "test_ids_agent_da_doc": test_pairs_all, "test_ids_agent_da_doc_ghi_chu": "lượt kiểm tra cuối (Opus) đã đọc toàn bộ test 1.1 (verify + reread 40 lô)",
       "release_check": {"exit_code": rc.returncode, "dau_ra": rc_head},
       "dev_rescore": {"changed_rows": len(changed_ids), "models": list(scores), "missing_models": missing_models}}
CHK = json.load(open(T / "check_v1_2_trial.json", encoding="utf-8")) if (T / "check_v1_2_trial.json").exists() else {}
if CHK:
    man["provenance"] = {
        "dinh_nghia": {"vi_provenance": "source = văn bản tiếng Việt giữ nguyên như nguồn (chuẩn hóa NFC và đổi chuỗi mẫu không tính); revised = đã sửa state/câu hỏi/lựa chọn (vi_revised_fields liệt kê)",
                       "gold_provenance": "source = đáp án như nhãn gốc (target của 1.1 trước mọi chỉnh sửa); overridden = khác nhãn gốc; áp cho cả 3 biến thể của cặp"},
        "dong_vi_co_dau_theo_tap_nguon": CHK["provenance_dong_vi_co_dau"], "dap_an_nguoc_nguon": CHK["dap_an_nguoc_nguon"],
        "quy_trinh": "Đề xuất do LLM (Opus, Fable) viết, người làm dự án duyệt.",
        "cach_duyet_nguoi_dung_mo_ta": "đọc từng câu với đề xuất đổi đáp án và nhãn; đề xuất dịch mà Opus đồng ý thì duyệt theo lô sau khi đọc mẫu, nếu cần thiết thì gọi fable ra",
        "so_lieu_tu_log": {"edits_log_de_xuat_nguoi_dung_tu_xa": 545, "edits_log_de_xuat_ghi_chu_uy_quyen": 132, "edits_log_chuoi_mau_excel": 66,
                           "edits_log_chuoi_mau_uy_quyen": 2, "label_decisions_remote": 143,
                           "ghi_chu": "đếm theo trường ai/ghi_chu của data/human_edits/edits_log.jsonl và results/human_review/label_decisions_remote.jsonl; "
                                      "143 quyết định nhãn đều mang ghi chú 'ủy quyền' (141 ghi ai = người làm dự án từ xa, 2 ghi 'ủy quyền agent'). "
                                      "Log không tách được các nhóm 5 trên trang / 1 lô 350 / 6 trong chat / 182 đề xuất kiểm tra cuối."},
        "so_lieu_dieu_phoi_cung_cap": {"nguoi_dung_tu_bam": "5 trên trang, 1 lô 350, 6 trong chat, Excel chuỗi mẫu",
                                       "theo_uy_quyen_theo_agent": "132 đề xuất, 141 nhãn, 182 đề xuất kiểm tra cuối, và các điều chỉnh an toàn"}}
    man["ro_ri_train_v0_val_v0"] = CHK["ro_ri_train_v0_val_v0"]
    UC = ROOT / "results/human_review/user_confirmations.jsonl"
    can = {(c_["split"], c_["pair_id"]) for c_ in changes if "loai_cau" in c_["loai_thay_doi"] or "doi_dap_an" in c_["loai_thay_doi"]}
    conf = {(x.get("split"), x.get("pair_id")) for x in read(UC)} if UC.exists() else set()
    man["xac_nhan_nguoi_dung"] = {"file": "results/human_review/user_confirmations.jsonl", "co_file": UC.exists(),
                                  "so_cau_loai_hoac_doi_dap_an": len(can), "da_xac_nhan": len(can & conf),
                                  "chua_xac_nhan": sorted(f"{a_}|{b_}" for a_, b_ in can - conf)}
    man["nodiac_sinh_tu_ban_da_sua"] = {"khong_khop": CHK["nodiac_khong_khop"], "so_dong_kiem": CHK["nodiac_kiem"], "trong_do_revised": CHK["nodiac_kiem_rieng_revised"]}
(T / "manifest_v1_2_trial.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- báo cáo
pct = lambda x: f"{100 * x:.1f}"
L = ["# Dựng thử bench 1.2 (khu tạm, CHƯA đóng băng)", "",
     "Bản 1.2 = bench 1.1 sau khi người soát: áp lớp ghi đè `data/human_edits` (đề xuất đã duyệt, sửa câu, loại câu, đổi đáp án dev), quyết định nhãn test "
     "(`results/human_review/label_decisions_remote.jsonl`, chỉ dòng split = test), và chuẩn hóa NFC mọi trường tiếng Việt (bản không dấu sinh lại từ bản NFC). "
     "MLQA để bản 1.3, Global-PIQA không làm. Không train; không chạy model nào trên test.", "",
     f"- Dựng: `bash scripts/human_edits/run_trial_v1_2.sh` (gọi `build_revised.py --trial --nfc --test_labels …`). Kiểm đồng nhất với lớp ghi đè rỗng: dev trùng byte, test trùng hash 1.1.",
     f"- Ra: `dev_v1_2_trial.jsonl` ({br['dev_rows']} dòng, sha `{man['files']['dev_v1_2_trial.jsonl'][:12]}…`), `test_v1_2_trial.jsonl` ({br['test_rows']} dòng, "
     f"sha `{man['files']['test_v1_2_trial.jsonl'][:12]}…`). Không chmod, không FROZEN, không ghi `data/`. dev/test 1.1 giữ nguyên hash: {man['base_unchanged']}.",
     f"- Nhãn test đã áp: {br['test_label_decisions']} (dòng `giu` không đổi gì). Chuỗi mẫu đã sửa: {', '.join(br['template_overrides'])}. "
     f"Dòng tiếng Anh không NFC: {br['en_rows_not_nfc'] or 0} (bản Anh giữ nguyên).",
     f"- `release_check.py`: mã thoát {rc.returncode}. {' / '.join(rc_head[:3])}", "",
     "## Tóm tắt số liệu", "", "| Model | EN 1.1 → 1.2 (còn lại) | VI có dấu 1.1 → 1.2 (còn lại) | VI không dấu 1.1 → 1.2 (còn lại) |", "|---|---|---|---|"]
for m_, sc_ in scores.items():
    L.append(f"| {m_} | " + " | ".join(f"{pct(sc_[k_]['acc11'])} → {pct(sc_[k_]['acc12'])} ({pct(sc_[k_]['acc11_con_lai'])})" for k_ in ("*|en|orig", "*|vi|orig", "*|vi|nodiac")) + " |")
L += ["", f"Số ô (model × nguồn × biến thể) nhảy quá 2 điểm so với 1.1: {len(flags)} (mục 4).", "",
     "## 1. Thay đổi theo nguồn và loại", "",
     "Cách đếm: một cặp (EN + VI + VI không dấu cùng `pair_id`) được đếm ở **mọi** loại nó thuộc (ví dụ sửa state và đổi đáp án). "
     "\"Chỉ đổi do NFC\" là cặp đổi byte nhưng chữ sau NFC không đổi. \"Loại câu\" không tính thêm loại nào khác. Cột \"do chuỗi mẫu\": cặp có lựa chọn đổi vì 2 chuỗi mẫu MASSIVE đã sửa.", ""]
for s in ("dev", "test"):
    L += [f"### {s}", "", "| Nguồn | Số cặp 1.1 | " + " | ".join(CAT_VI[k] for k in CATS) + " | Cặp có đổi | trong đó do chuỗi mẫu |", "|---" * (len(CATS) + 4) + "|"]
    tot = Counter()
    for src in sorted(pairs11[s]):
        c = stat.get((s, src), Counter()); tot.update(c); tot["n"] += pairs11[s][src]
        L.append(f"| {NAME.get(src, src)} | {pairs11[s][src]} | " + " | ".join(str(c[k]) for k in CATS) + f" | {c['tong_cap_doi']} | {c['do_chuoi_mau']} |")
    L.append(f"| **Tổng** | {tot['n']} | " + " | ".join(str(tot[k]) for k in CATS) + f" | {tot['tong_cap_doi']} | {tot['do_chuoi_mau']} |")
    L.append("")
L += ["### Theo chuỗi mẫu đã sửa (số cặp có lựa chọn hoặc câu hỏi đổi vì chuỗi đó)", "", "| Khóa chuỗi mẫu | dev | test |", "|---|---|---|"]
for k_ in sorted(set(br.get("template_keys_applied_dev", {})) | set(br.get("template_keys_applied_test", {}))):
    L.append(f"| `{k_}` | {br.get('template_keys_applied_dev', {}).get(k_, 0)} | {br.get('template_keys_applied_test', {}).get(k_, 0)} |")
L.append("")
if CHK:
    L += ["## A. Nguồn gốc văn bản và đáp án (provenance)", "",
          "Mỗi dòng có thêm `gold_provenance`; dòng tiếng Việt có thêm `vi_provenance` và `vi_revised_fields`. "
          "NFC và đổi chuỗi mẫu không tính là sửa. Số dòng VI có dấu (bản không dấu giống hệt):", "",
          "| Tập · nguồn | source | revised | sửa state | sửa câu hỏi | sửa lựa chọn |", "|---|---|---|---|---|---|"]
    for k_, v_ in CHK["provenance_dong_vi_co_dau"].items():
        L.append(f"| {k_.replace('|', ' · ')} | {v_.get('source', 0)} | {v_.get('revised', 0)} | {v_.get('sua:state', 0)} | {v_.get('sua:question', 0)} | {v_.get('sua:option', 0)} |")
    go = CHK["dap_an_nguoc_nguon"]
    L += ["", f"Đáp án đi ngược nguồn (`gold_provenance = overridden`): {len(go)} cặp (dev {sum(g['split'] == 'dev' for g in go)}, test {sum(g['split'] == 'test' for g in go)}):", ""]
    L += [f"- {g['split']} `{g['pair_id']}`: {g['dap_an_nguon']} → {g['dap_an_1_2']}" for g in go]
    L += ["", "Quy trình (ghi trong manifest): đề xuất do LLM (Opus, Fable) viết, người làm dự án duyệt. Cách duyệt theo lời người làm dự án: "
          "\"đọc từng câu với đề xuất đổi đáp án và nhãn; đề xuất dịch mà Opus đồng ý thì duyệt theo lô sau khi đọc mẫu, nếu cần thiết thì gọi fable ra\". "
          "Số liệu từ log (manifest `provenance.so_lieu_tu_log`): 545 bản ghi duyệt đề xuất do người làm dự án bấm từ xa; 132 bản ghi duyệt đề xuất mang ghi chú ủy quyền; "
          "66 bản ghi chuỗi mẫu từ Excel; 143 quyết định nhãn từ xa (đều mang ghi chú ủy quyền). Log không tách được các nhóm 5 trên trang / lô 350 / 6 trong chat / 182 đề xuất kiểm tra cuối; "
          "các số đó do điều phối cung cấp và được ghi riêng.", ""]
    DS = T / "dung_sai_items.json"
    if DS.exists():
        ds = json.load(open(DS, encoding="utf-8"))["items"]
        chinh = [x for x in ds if x["nhom"] == "chinh"]
        L += ["## B. Câu dev đúng → sai chỉ vì sửa chữ", "",
              "Cặp dev có `vi_provenance = revised`, `gold_provenance = source`, không bị loại, mà model chuyển đúng (1.1) → sai (1.2 thử) ở VI có dấu. "
              "Đếm trên 5 model riêng (decider-2b text, decider-ft-ep2 text, đối chứng, Qwen3-1.7B gốc, strip; bản keyed chỉ để tham khảo). "
              f"Nhóm chính (từ 2 model trở lên, đúng tiêu chí brief): **{len(chinh)}** cặp; nhóm phụ (1 model, đưa thêm để đọc): {len(ds) - len(chinh)} cặp; tổng {len(ds)}. "
              f"Theo nguồn: {', '.join(f'{s} {n}' for s, n in sorted(Counter(x['nguon'] for x in ds).items()))}. "
              f"{sum(bool(x['cau_cung_doan_van']) for x in ds)} cặp có đoạn văn dùng chung (hoàn tác sẽ kéo theo câu cùng đoạn); sửa tay không qua đề xuất: "
              f"{sum(bool(x['sua_tay_khong_qua_de_xuat']) for x in ds)}.", "",
              "Nhóm chính: " + ", ".join(f"`{x['pair_id']}` ({x['so_model_dung_sang_sai']})" for x in chinh) + ".", "",
              "Dữ liệu: `results/bench_v1_2_trial/dung_sai_items.json`. Quyết định ghi qua công cụ (màn \"Đúng→sai\", `POST /api/dung_sai` "
              "{pair_id, action: giu_moi | hoan_tac, note}); \"hoàn tác\" gỡ đúng các đề xuất đã áp (mới nhất trước), đánh dấu từ chối, "
              "câu cùng đoạn về bản cũ theo; có sửa chồng lên thì báo lỗi, không ghi.", ""]
    L += ["## C. Kỹ thuật", "",
          f"- Bỏ 3 trường provenance thì bản dựng này trùng hash bản thử trước: dev `{CHK['dev_sha_bo_provenance'][:12]}…`, test `{CHK['test_sha_bo_provenance'][:12]}…`.",
          f"- Bản không dấu sinh từ bản VI hiện hành (strip_diacritics trên state, câu hỏi, lựa chọn; cùng đáp án): khớp {CHK['nodiac_kiem']} dòng, trong đó revised {CHK['nodiac_kiem_rieng_revised']}; không khớp: {len(CHK['nodiac_khong_khop'])}.",
          "- Rò rỉ train_v0 + val_v0 (3 phép như lần dựng 1.1: khớp chính xác, MinHash ≥ 0,8, 13-gram / câu ngắn nằm trọn; chỉ mục là state của bench, quét state của train):", "",
          "| Bản · tập | Số lần trùng theo nguồn và phép |", "|---|---|"]
    for k_, v_ in CHK["ro_ri_train_v0_val_v0"].items():
        L.append(f"| {k_.replace('|', ' · ')} | {', '.join(f'{a_}: {b_}' for a_, b_ in v_.items()) or '0'} |")
    L.append("")
L += ["## 2. Câu test đã xem (ghi trong manifest)", "",
      f"- `test_ids_nguoi_da_xem`: **{len(nguoi)}** cặp test đã hiện cho người: có đề xuất theo câu ({len(seen['vi_proposals'])}), có quyết định nhãn từ xa "
      f"({len(seen['label_decisions_remote'])}), hoặc nằm trong danh sách vấn đề nhãn (label_issues, {len(seen['label_issues'])}).",
      f"- `test_ids_agent_da_doc`: **{len(test_pairs_all)}** cặp, tức toàn bộ test 1.1, vì lượt kiểm tra cuối Opus đã đọc hết.", "",
      "## 3. Điểm DEV 1.2 so với 1.1", "",
      f"Dòng dev trùng byte với 1.1 dùng lại preds; {len(changed_ids)} dòng đổi byte (kể cả chỉ do NFC) được chấm lại, cùng điều kiện như lần chấm 1.1 "
      "(`scripts/human_edits/rescore_dev_v1_2.sh`); dòng bị loại bỏ khỏi điểm 1.2. Độ đúng tính theo đáp án của từng bản (1.2 dùng đáp án đã sửa). "
      "\"1.1 trên cặp còn lại\" tách tác động của việc loại câu khỏi tác động của việc sửa.", ""]
if missing_models:
    L.append(f"Chưa có preds chấm lại cho: {', '.join(missing_models)}.")
for m, sc in scores.items():
    L += [f"### {m}", "", "| Nguồn | biến thể | n 1.1 → 1.2 | 1.1 | 1.1 trên cặp còn lại | 1.2 toàn bộ | 1.2 chỉ dòng source (n) | 1.2 − 1.1 |", "|---|---|---|---|---|---|---|---|"]
    for key, v in sc.items():
        src, l, var = key.split("|")
        d = 100 * (v["acc12"] - v["acc11"])
        L.append(f"| {NAME.get(src, 'Tất cả' if src == '*' else src)} | {l}{' không dấu' if var == 'nodiac' else ''} | {v['n11']} → {v['n12']} | {pct(v['acc11'])} | "
                 f"{pct(v['acc11_con_lai'])} | {pct(v['acc12'])} | {pct(v['acc12_source'])} ({v['n12_source']}) | {'**' if abs(d) > 2 and src != '*' else ''}{d:+.1f}{'**' if abs(d) > 2 and src != '*' else ''} |")
    L.append("")
L += ["## 4. Các ô nhảy hơn 2 điểm và câu gây ra", ""]
if not flags:
    L.append("Không có ô nào (model × nguồn × biến thể) thay đổi quá 2 điểm.")
for fl in flags:
    L.append(f"- **{fl['model']} · {NAME.get(fl['nguon'], fl['nguon'])} · {fl['bien_the']}**: {pct(fl['acc11'])} → {pct(fl['acc12'])} "
             f"(1.1 trên cặp còn lại {pct(fl['acc11_con_lai'])}). Bị loại {fl['bi_loai']['n']} dòng (đúng ở 1.1: {fl['bi_loai']['dung_o_11']}). "
             f"Đổi đúng/sai {len(fl['doi_dung_sai'])} cặp:")
    for x in fl["doi_dung_sai"][:25]:
        L.append(f"  - `{x['pair_id']}`: {'đúng' if x['dung_11'] else 'sai'} → {'đúng' if x['dung_12'] else 'sai'} ({', '.join(CAT_VI.get(k, k) for k in x['loai_thay_doi'])})")
    if len(fl["doi_dung_sai"]) > 25:
        L.append(f"  - … còn {len(fl['doi_dung_sai']) - 25} cặp (đủ danh sách ở `scores_v1_2_trial.json`).")
PREV = T / "prev_r1"
if (PREV / "scores_v1_2_trial.json").exists():
    ps = json.load(open(PREV / "scores_v1_2_trial.json", encoding="utf-8"))["scores"]
    pm = json.load(open(PREV / "manifest_v1_2_trial.json", encoding="utf-8"))
    L += ["", "## 5. So với lần dựng thử trước (chỉ khác ở chuỗi mẫu)", "",
          f"Lần trước: dev `{pm['files']['dev_v1_2_trial.jsonl'][:12]}…`, test `{pm['files']['test_v1_2_trial.jsonl'][:12]}…`, "
          f"{pm['options']['edits_log_records']} bản ghi lớp ghi đè, chuỗi mẫu đã sửa: {len(pm['options']['template_overrides'])}. "
          f"Lần này: dev `{man['files']['dev_v1_2_trial.jsonl'][:12]}…`, test `{man['files']['test_v1_2_trial.jsonl'][:12]}…`, {br['edits_log_records']} bản ghi, "
          f"chuỗi mẫu đã sửa: {len(br['template_overrides'])}. Giữa hai lần chỉ có thêm bản ghi chuỗi mẫu (duyệt + sửa), nên chênh điểm dưới đây chỉ do chuỗi mẫu.", "",
          "| Model | Nguồn | biến thể | 1.2 lần trước | 1.2 lần này | chênh |", "|---|---|---|---|---|---|"]
    for m_, sc_ in scores.items():
        for key_, v_ in sc_.items():
            src_ = key_.split("|")[0]
            if src_ in ("massive", "vifactcheck", "viwikifc") and key_ in ps.get(m_, {}):
                a0 = ps[m_][key_]["acc12"]; a1 = v_["acc12"]
                L.append(f"| {m_} | {NAME[src_]} | {key_.split('|', 1)[1].replace('|', ' ')} | {pct(a0)} | {pct(a1)} | {100 * (a1 - a0):+.1f} |")
L += ["", "## 6. File", "",
      "- `dev_v1_2_trial.jsonl`, `test_v1_2_trial.jsonl`, `changes_v1_2_trial.jsonl` (loại thay đổi từng cặp), `build_report_v1_2_trial.json`, `manifest_v1_2_trial.json`.",
      "- `rescore/rescore_dev.jsonl` (dòng dev đổi byte), `eval_rescore/` (preds chấm lại + log), `eval/<model>/preds.jsonl` (preds dev 1.2 đã gộp), `scores_v1_2_trial.json`.",
      "- Code: `scripts/human_edits/build_revised.py` (thêm `--trial`, `--nfc`, `--test_labels`), `run_trial_v1_2.sh`, `rescore_dev_v1_2.sh`, `report_trial_v1_2.py`.", "",
      "Chưa đóng băng gì. Khi người làm dự án bảo đóng băng: chạy `build_revised.py` không có `--trial` (ghi test mới chỉ đọc + FROZEN) và giữ test 1.1 để đối chiếu."]
(T / "REPORT_bench_v1_2_trial.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(json.dumps({"changed_rows": len(changed_ids), "models": list(scores), "missing": missing_models, "flags": len(flags), "nguoi_da_xem": len(nguoi),
                  "release_check_exit": rc.returncode}, ensure_ascii=False))
