"""Bench 1.3, DỰNG THỬ, bước 2: từ bản dựng ở results/bench_v1_3_trial/base/ (lớp ghi đè hiện tại, xem run_build_v1_3_trial.sh)
- gắn cờ mơ hồ: trường "mo_ho": true trên MỌI dòng (EN, VI, VI không dấu) của 26 cặp người dùng giữ và gắn nhãn mơ hồ
  (results/label_review_v1_2/quyet_dinh_28_mo_ho.json, mục giu_nhan_mo_ho). Dòng khác KHÔNG thêm trường (vắng = không mơ hồ), để dòng không đổi
  giữ nguyên byte so với dev 1.2. Câu mơ hồ VẪN tính vào điểm chính; báo cáo thêm lát "bỏ câu mơ hồ".
- kiểm: 2 cặp người dùng quyết loại đã vắng; test 1.3 trùng TỪNG BYTE test 1.2 đã đóng băng (lát MLQA tạm dừng); so dev với dev 1.2 theo nguồn × loại.
Ra (khu tạm, không ghi data/, không đóng băng): dev_v1_3_trial.jsonl, test_v1_3_trial.jsonl, changes_vs_1_2.json, manifest_v1_3_trial.json, REPORT_bench_v1_3_trial.md,
rescore/rescore_dev.jsonl (dòng dev cần chấm lại: mới hoặc đổi nội dung so với dev 1.2, không tính riêng trường mo_ho). Python Windows."""
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/bench_v1_3_trial"
BASE = T / "base"
B12 = ROOT / "data/bench_v1_2"
read_b = lambda p: [l for l in Path(p).read_bytes().decode("utf-8").splitlines() if l.strip()]
sha = lambda b: hashlib.sha256(b).hexdigest()
FLAG = "mo_ho"
NAME = {"belebele": "Belebele", "sib200": "SIB-200", "massive": "MASSIVE", "xcopa": "XCOPA", "vifactcheck": "ViFactCheck", "viwikifc": "ViWikiFC"}

qd = json.load(open(ROOT / "results/label_review_v1_2/quyet_dinh_28_mo_ho.json", encoding="utf-8"))
MOHO = {x["pair_id"] for x in qd["giu_nhan_mo_ho"]}
LOAI = {x["pair_id"] for x in qd["loai"]}
assert len(MOHO) == 26 and len(LOAI) == 2 and not (MOHO & LOAI)
m12d = json.load(open(B12 / "manifest_dev_v1_2.json", encoding="utf-8"))
m12t = json.load(open(B12 / "manifest_test_v1.2.json", encoding="utf-8"))
dev12_b, test12_b = (B12 / "dev_v1_2.jsonl").read_bytes(), (B12 / "test_v1.2.jsonl").read_bytes()
assert sha(dev12_b) == m12d["sha256"]["dev_v1_2.jsonl"] and sha(test12_b) == m12t["sha256"]["test_v1.2.jsonl"]
base_dev, base_test = (BASE / "dev_v1_2_trial.jsonl").read_bytes(), (BASE / "test_v1_2_trial.jsonl").read_bytes()
br = json.load(open(BASE / "build_report_v1_2_trial.json", encoding="utf-8"))
chk = json.load(open(BASE / "check_v1_2.json", encoding="utf-8"))
leak = json.load(open(BASE / "leakage_v1_2.json", encoding="utf-8"))

# ---------------------------------------------------------------- test: phải trùng từng byte test 1.2
test_same = base_test == test12_b
assert test_same, "test dựng ra KHÁC test 1.2 đã đóng băng"

# ---------------------------------------------------------------- dev: gắn cờ mơ hồ
dev_rows = [json.loads(l) for l in base_dev.decode("utf-8").splitlines() if l.strip()]
pairs_in = {r["pair_id"] for r in dev_rows}
assert MOHO <= pairs_in, sorted(MOHO - pairs_in)
assert not (LOAI & pairs_in), "cặp đã quyết loại vẫn còn"
out = []
for r in dev_rows:
    if r["pair_id"] in MOHO:
        r = {**r, FLAG: True}
    out.append(r)
dev13_b = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out).encode("utf-8")
(T / "dev_v1_3_trial.jsonl").write_bytes(dev13_b)
(T / "test_v1_3_trial.jsonl").write_bytes(base_test)

# ---------------------------------------------------------------- so với dev 1.2
d12 = {json.loads(l)["id"]: json.loads(l) for l in read_b(B12 / "dev_v1_2.jsonl")}
d13 = {r["id"]: r for r in out}
core = lambda r: {k: v for k, v in r.items() if k != FLAG}
per_pair = defaultdict(lambda: {"them": 0, "bot": 0, "doi_noi_dung": 0, "chi_co_mo_ho": 0, "truong": set()})
rescore = []
for i in set(d12) | set(d13):
    a, b = d12.get(i), d13.get(i)
    if a == b:
        continue
    p = (a or b)["pair_id"]
    if a is None:
        per_pair[p]["them"] += 1; rescore.append(b)
    elif b is None:
        per_pair[p]["bot"] += 1
    elif core(a) != core(b):
        per_pair[p]["doi_noi_dung"] += 1; per_pair[p]["truong"] |= {k for k in core(b) if a.get(k) != b.get(k)}; rescore.append(b)
    else:
        per_pair[p]["chi_co_mo_ho"] += 1
loai_thay_doi = {}
for p, d in per_pair.items():
    if d["bot"] and not (d["them"] or d["doi_noi_dung"]):
        loai_thay_doi[p] = "loai_cau"
    elif d["doi_noi_dung"] or d["them"]:
        loai_thay_doi[p] = "sua_noi_dung:" + ",".join(sorted(d["truong"]))
    else:
        loai_thay_doi[p] = "gan_co_mo_ho"
src = {p: (d13.get(p + ":vi") or d12.get(p + ":vi"))["source"] for p in per_pair}
stat = defaultdict(Counter)
for p, k in loai_thay_doi.items():
    stat[src[p]][k.split(":")[0]] += 1
(T / "rescore").mkdir(exist_ok=True)
(T / "rescore/rescore_dev.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rescore, key=lambda r: r["id"])), encoding="utf-8")
chg = {"dev": {"cap_doi": len(per_pair), "theo_loai": dict(Counter(k.split(":")[0] for k in loai_thay_doi.values())),
               "theo_nguon_loai": {s: dict(c) for s, c in sorted(stat.items())}, "chi_tiet": {p: loai_thay_doi[p] for p in sorted(loai_thay_doi)},
               "dong_can_cham_lai": len(rescore)},
       "test": {"trung_tung_byte_test_1_2": test_same, "sha256": sha(base_test)}}
(T / "changes_vs_1_2.json").write_text(json.dumps(chg, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- lát
vi_pairs = lambda rows, f=lambda r: True: Counter(r["source"] for r in rows if r["lang"] == "vi" and r["variant"] == "orig" and f(r))
slices = {"dev": {"chinh": dict(vi_pairs(out)), "bo_cau_mo_ho": dict(vi_pairs(out, lambda r: not r.get(FLAG))), "mo_ho": dict(vi_pairs(out, lambda r: bool(r.get(FLAG)))),
                  "mlqa": "TẠM DỪNG: chưa có bản xem xét 30 câu của người dùng"},
          "test": {"chinh": dict(vi_pairs([json.loads(l) for l in base_test.decode("utf-8").splitlines() if l.strip()])),
                   "mlqa": "TẠM DỪNG: chưa có bản xem xét 30 câu của người dùng"}}
rc = subprocess.run([sys.executable, str(ROOT / "scripts/human_edits/release_check.py")], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
rc_head = [l for l in rc.stdout.splitlines() if l and not l.startswith("  ")][:4]
man = {"name": "bench 1.3 — DỰNG THỬ (khu tạm; CHƯA đóng băng; không ghi data/)", "version": "1.3-trial",
       "dinh_nghia_1_3": "1.2 + quyết định người dùng 17/10 trên soát nhãn mù (loại 2 cặp ViWikiFC; gắn cờ mơ hồ 26 cặp) + đề xuất lr-07c4a55b9f82 (đã duyệt, "
                         "sửa rồi duyệt) + lát MLQA riêng (TẠM DỪNG)",
       "files": {"dev_v1_3_trial.jsonl": sha(dev13_b), "test_v1_3_trial.jsonl": sha(base_test)}, "rows": {"dev": len(out), "test": br["test_rows"]},
       "built_from": {"dev_v1_2.jsonl": m12d["sha256"]["dev_v1_2.jsonl"], "test_v1.2.jsonl": m12t["sha256"]["test_v1.2.jsonl"],
                      "edits_log_records": br["edits_log_records"], "quyet_dinh_mo_ho": "results/label_review_v1_2/quyet_dinh_28_mo_ho.json"},
       "co_mo_ho": {"truong": FLAG, "y_nghia": "true trên mọi dòng (EN, VI, VI không dấu) của cặp người dùng giữ và gắn nhãn mơ hồ; vắng = không mơ hồ. "
                                               "Câu mơ hồ vẫn tính vào điểm chính; báo cáo có lát bỏ câu mơ hồ.", "so_cap": len(MOHO), "cap": sorted(MOHO)},
       "loai_theo_nguoi_dung": sorted(LOAI), "thay_doi_so_voi_1_2": chg, "lat": slices,
       "kiem": {"test_trung_byte_1_2": test_same, "leak_dev_test": {k: leak[k] for k in ("dev_states_hitting_test", "test_states_hitting_dev")},
                "train_v0_val_v0": {"test": leak["train_v0_vs_test"]["total"], "dev": leak["train_v0_vs_dev_v1_2"]["total"]},
                "nodiac_khong_khop": len(chk["nodiac_khong_khop"]), "provenance_du": all(chk[f"{s}_moi_dong_co_gold_provenance"] for s in ("dev", "test")),
                "release_check": {"exit_code": rc.returncode, "dau_ra": rc_head}}}
(T / "manifest_v1_3_trial.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"dev_rows": len(out), "dev_sha": sha(dev13_b)[:16], "test_same": test_same, "changes": {k: v for k, v in chg["dev"].items() if k != "chi_tiet"},
                  "slices_dev": slices["dev"], "rescore": len(rescore), "release_check": rc.returncode}, ensure_ascii=False))
