"""Bench 1.3, đóng băng (2/2): từ bản dựng lại results/bench_v1_3_trial/rebuild/ (lớp ghi đè hiện tại) gắn cờ mo_ho y như bản thử, rồi BẮT BUỘC:
dev trùng sha bản thử (dev_v1_3_trial.jsonl), test trùng sha bản thử và test 1.2 đã đóng băng; rò rỉ dev↔test = 0; không dấu 0 lệch; provenance đủ;
release_check mã 0; data/bench_v1_3/test_v1.3.jsonl chưa tồn tại. Lệch là dừng, không ghi gì.
Ghi data/bench_v1_3/: dev_v1_3.jsonl, test_v1.3.jsonl (chỉ đọc), FROZEN_test_v1.3.sha256, leakage_v1_3.json, manifest_dev_v1_3.json,
manifest_test_v1.3.json (chỉ đọc). Không đụng data/bench_v1_2/. Python Windows."""
import datetime
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/bench_v1_3_trial"
S = T / "rebuild"
OUT = ROOT / "data/bench_v1_3"
B12 = ROOT / "data/bench_v1_2"
LOG = ROOT / "data/human_edits/edits_log.jsonl"
FLAG = "mo_ho"
sha_b = lambda b: hashlib.sha256(b).hexdigest()
sha = lambda p: sha_b(Path(p).read_bytes())
qd = json.load(open(ROOT / "results/label_review_v1_2/quyet_dinh_28_mo_ho.json", encoding="utf-8"))
MOHO = {x["pair_id"] for x in qd["giu_nhan_mo_ho"]}
LOAI = {x["pair_id"] for x in qd["loai"]}
TRIAL_DEV, TRIAL_TEST = (T / "dev_v1_3_trial.jsonl").read_bytes(), (T / "test_v1_3_trial.jsonl").read_bytes()
B12_SHA = {f: sha(B12 / f) for f in ("dev_v1_2.jsonl", "test_v1.2.jsonl", "manifest_test_v1.2.json", "FROZEN_test_v1.2.sha256")}

rows = [json.loads(l) for l in (S / "dev_v1_2_trial.jsonl").read_bytes().decode("utf-8").splitlines() if l.strip()]
dev_b = "".join(json.dumps({**r, FLAG: True} if r["pair_id"] in MOHO else r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
test_b = (S / "test_v1_2_trial.jsonl").read_bytes()
br = json.load(open(S / "build_report_v1_2_trial.json", encoding="utf-8"))
chk = json.load(open(S / "check_v1_2.json", encoding="utf-8"))
leak = json.load(open(S / "leakage_v1_2.json", encoding="utf-8"))
rc = subprocess.run([sys.executable, str(ROOT / "scripts/human_edits/release_check.py")], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
errs = []
if dev_b != TRIAL_DEV:
    errs.append(f"dev dựng lại khác bản thử ({sha_b(dev_b)[:12]} vs {sha_b(TRIAL_DEV)[:12]})")
if test_b != TRIAL_TEST or test_b != (B12 / "test_v1.2.jsonl").read_bytes():
    errs.append("test dựng lại khác bản thử / test 1.2")
if leak["dev_states_hitting_test"] or leak["test_states_hitting_dev"] or br.get("leak_dev_test", {}).get("dev_states_hitting_test"):
    errs.append("rò rỉ dev↔test khác 0")
if chk["nodiac_khong_khop"] or not all(chk[f"{s}_{k}"] for s in ("dev", "test") for k in ("moi_dong_co_gold_provenance", "moi_dong_vi_co_vi_provenance", "id_duy_nhat")):
    errs.append("không dấu / provenance / id không đạt")
if len(json.loads("[" + ",".join(l for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()) + "]")) != br["edits_log_records"]:
    errs.append("lớp ghi đè đổi sau khi dựng lại")
if rc.returncode != 0:
    errs.append(f"release_check mã {rc.returncode}")
if (OUT / "test_v1.3.jsonl").exists() or (OUT / "FROZEN_test_v1.3.sha256").exists():
    errs.append("data/bench_v1_3/ đã có test 1.3 đóng băng")
if errs:
    print("KHÔNG ĐÓNG BĂNG:", json.dumps(errs, ensure_ascii=False)); sys.exit(2)

dev = [json.loads(l) for l in dev_b.decode("utf-8").splitlines() if l.strip()]
test = [json.loads(l) for l in test_b.decode("utf-8").splitlines() if l.strip()]


def counts(rs):
    return {"rows": len(rs), "by_source": dict(Counter(r["source"] for r in rs)), "by_source_lang_variant": dict(Counter(f"{r['source']}|{r['lang']}|{r['variant']}" for r in rs)),
            "by_kind": dict(Counter(f"{r['source']}|{r['kind']}" for r in rs if r["variant"] == "orig")),
            "target_hist_orig": dict(Counter(f"{r['source']}|{r['kind']}|{r['target']}" for r in rs if r["lang"] == "vi" and r["variant"] == "orig"))}


m12d = json.load(open(B12 / "manifest_dev_v1_2.json", encoding="utf-8"))
chg = json.load(open(T / "changes_vs_1_2.json", encoding="utf-8"))
log = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
common = {
    "version": "1.3", "seed": m12d["seed"], "revisions": m12d["revisions"], "licenses": m12d["licenses"],
    "dinh_nghia": "1.3 = 1.2 + quyết định người dùng 17/10 trên soát nhãn mù dev 1.2 (loại 2 cặp ViWikiFC; gắn cờ mơ hồ 26 cặp) + đề xuất lr-07c4a55b9f82 "
                  "(người dùng sửa rồi duyệt). Không thêm gì khác.",
    "mlqa": "MLQA DỜI SANG 1.4 (lát riêng, khi người dùng có bản xem xét 30 câu; quyết định 17/10 'chọn 1, đóng băng 1.3 trước').",
    "built_from": {"bench_v1_2": {"dev_v1_2.jsonl": B12_SHA["dev_v1_2.jsonl"], "test_v1.2.jsonl": B12_SHA["test_v1.2.jsonl"]},
                   "edits_log": f"data/human_edits/edits_log.jsonl ({len(log)} bản ghi, {sha(LOG)})",
                   "label_decisions": f"results/human_review/label_decisions_remote.jsonl ({sha(ROOT / 'results/human_review/label_decisions_remote.jsonl')})",
                   "quyet_dinh_mo_ho": f"results/label_review_v1_2/quyet_dinh_28_mo_ho.json ({sha(ROOT / 'results/label_review_v1_2/quyet_dinh_28_mo_ho.json')})",
                   "config": {f: sha(ROOT / "config" / f) for f in ("bench_v1_questions.json", "translations.json")}},
    "build_cmd": "bash scripts/bench_v1_3/run_rebuild_v1_3.sh (build_revised.py --trial --nfc --provenance --test_labels …) rồi python scripts/bench_v1_3/freeze_v1_3.py",
    "quyet_dinh_nguoi_dung": {
        "nguon": qd["nguon_quyet_dinh"], "thoi_diem": qd["thoi_diem"],
        "loai": sorted(LOAI), "loai_ghi_qua": "POST /api/item (lý do 'mơ hồ nên loại'; agent ghi theo quyết định người dùng)",
        "mo_ho": {"truong": FLAG, "y_nghia": "true trên mọi dòng (EN, VI, VI không dấu) của cặp người dùng giữ và gắn nhãn mơ hồ; vắng = không mơ hồ. Câu mơ hồ VẪN tính "
                                            "vào điểm chính; báo cáo có lát bỏ câu mơ hồ.", "so_cap": len(MOHO), "cap": sorted(MOHO)},
        "lr-07c4a55b9f82": "belebele …/Airline_alliances#1, lựa chọn C: 'Hãng hàng không có thể cung cấp thêm nhiều quyền lợi ' -> 'Hãng hàng không có thể đang bổ sung "
                           "thêm quyền lợi mới ' (người dùng sửa rồi duyệt trên công cụ 2026-10-17 03:31; dấu cách cuối có sẵn từ nguồn)"},
    "thay_doi_so_voi_1_2": {"dev": {k: v for k, v in chg["dev"].items()}, "test": "không đổi: trùng từng byte test 1.2"},
    "provenance": {"dinh_nghia": m12d["provenance"]["dinh_nghia"], "dong_vi_co_dau_theo_tap_nguon": chk["provenance_dong_vi_co_dau"],
                   "dap_an_nguoc_nguon": chk["dap_an_nguoc_nguon"], "quy_trinh": m12d["provenance"]["quy_trinh"],
                   "cach_duyet_nguoi_dung_mo_ta": m12d["provenance"]["cach_duyet_nguoi_dung_mo_ta"],
                   "soat_nhan_mu_1_2": "results/label_review_v1_2/REPORT_label_review.md (134 cặp; nhóm 1: 105, nhóm 3: 1, nhóm 4: 28; không nhãn sai có căn cứ)"},
    "leakage": {"dev_states_hitting_test": leak["dev_states_hitting_test"], "test_states_hitting_dev": leak["test_states_hitting_dev"],
                "train_v0_vs_test": leak["train_v0_vs_test"]["total"], "train_v0_vs_dev_v1_3": leak["train_v0_vs_dev_v1_2"]["total"]},
    "release_check": {"exit_code": rc.returncode, "dau_ra": [l for l in rc.stdout.splitlines() if l and not l.startswith("  ")][:4]},
    "n_tokens": m12d["n_tokens"], "nodiac": m12d["nodiac"], "nfc": m12d["nfc"],
    "notes": ["Không chạy model nào trên test 1.3; test 1.2 và 1.1 giữ nguyên ở data/bench_v1_2/, data/bench_v1_1/.",
              "Test 1.3 trùng từng byte test 1.2 (MLQA dời sang 1.4)."],
}
md = {"name": "dev_v1_3", "sha256": {"dev_v1_3.jsonl": sha_b(dev_b)}, "counts": counts(dev), **common}
mt = {"name": "bench_v1 test", "frozen": True, "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
      "sha256": {"test_v1.3.jsonl": sha_b(test_b)}, "trung_byte_test_v1_2": True, "counts": counts(test), **common,
      "note": "Không sửa file này sau khi đóng băng. Không chạy model nào trên tập này."}
md_s, mt_s, leak_s = (json.dumps(x, ensure_ascii=False, indent=1) for x in (md, mt, leak))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "dev_v1_3.jsonl").write_bytes(dev_b)
tp = OUT / "test_v1.3.jsonl"
tp.write_bytes(test_b); os.chmod(tp, stat.S_IREAD)
(OUT / "FROZEN_test_v1.3.sha256").write_text(f"{sha_b(test_b)}  test_v1.3.jsonl\n", encoding="utf-8")
(OUT / "leakage_v1_3.json").write_text(leak_s, encoding="utf-8")
(OUT / "manifest_dev_v1_3.json").write_text(md_s, encoding="utf-8")
mp = OUT / "manifest_test_v1.3.json"
mp.write_text(mt_s, encoding="utf-8"); os.chmod(mp, stat.S_IREAD)
assert all(sha(B12 / f) == h for f, h in B12_SHA.items()), "data/bench_v1_2 bị đổi"
print(json.dumps({"dev_v1_3.jsonl": sha_b(dev_b), "test_v1.3.jsonl": sha_b(test_b), "dev_rows": len(dev), "test_rows": len(test),
                  "test_read_only": not os.access(tp, os.W_OK), "bench_v1_2_khong_doi": True, "leak": md["leakage"], "release_check": rc.returncode}, ensure_ascii=False))
