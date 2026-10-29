"""Bước D (3/3): ĐÓNG BĂNG bench 1.2 từ bản dựng ở khu tạm results/bench_v1_2_stage/ vào data/bench_v1_2/ (giữ nguyên data/bench_v1_1/).
Chỉ chạy khi mọi điều kiện đạt (nếu không: dừng, không ghi gì):
  - check_v1_2.json: mọi dòng có provenance, id duy nhất, bản không dấu khớp, rò rỉ dev↔test = 0;
  - diff_vs_trial_v1_2.json: không có cặp "khong_giai_thich"; release_check mã 0;
  - lớp ghi đè và quyết định nhãn không đổi sau khi dựng; data/bench_v1_2/test_v1.2.jsonl chưa tồn tại (không ghi đè bản đã đóng băng).
Ghi: dev_v1_2.jsonl, test_v1.2.jsonl (chỉ đọc), FROZEN_test_v1.2.sha256, leakage_v1_2.json, manifest_dev_v1_2.json, manifest_test_v1.2.json (chỉ đọc).
--update_dev "lý do": CẬP NHẬT DEV sau khi đã đóng băng. Bắt buộc test dựng ra trùng TỪNG BYTE với data/bench_v1_2/test_v1.2.jsonl và FROZEN;
chỉ ghi dev_v1_2.jsonl, manifest_dev_v1_2.json (thêm mục "cap_nhat": ngày, lý do, sha cũ/mới, cặp đổi) và leakage_v1_2.json; không đụng test, FROZEN,
manifest_test. Python Windows, thư viện chuẩn."""
import argparse
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
S = ROOT / "results/bench_v1_2_stage"
OUT = ROOT / "data/bench_v1_2"
B11 = ROOT / "data/bench_v1_1"
LOG = ROOT / "data/human_edits/edits_log.jsonl"
LAB = ROOT / "results/human_review/label_decisions_remote.jsonl"
UC = ROOT / "results/human_review/user_confirmations.jsonl"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
sha_b = lambda b: hashlib.sha256(b).hexdigest()
sha = lambda p: sha_b(Path(p).read_bytes())
ap = argparse.ArgumentParser()
ap.add_argument("--update_dev", default=None, help="lý do cập nhật dev (test đã đóng băng phải trùng từng byte)")
args = ap.parse_args()

# ---------------------------------------------------------------- điều kiện
chk = json.load(open(S / "check_v1_2.json", encoding="utf-8"))
dif = json.load(open(S / "diff_vs_trial_v1_2.json", encoding="utf-8"))
br = json.load(open(S / "build_report_v1_2_trial.json", encoding="utf-8"))
leak = json.load(open(S / "leakage_v1_2.json", encoding="utf-8"))
dev_b, test_b = (S / "dev_v1_2_trial.jsonl").read_bytes(), (S / "test_v1_2_trial.jsonl").read_bytes()
log = read(LOG)
errs = []
for s in ("dev", "test"):
    for k in (f"{s}_moi_dong_co_gold_provenance", f"{s}_moi_dong_vi_co_vi_provenance", f"{s}_id_duy_nhat"):
        if not chk[k]:
            errs.append(k)
    if dif[s]["cap_theo_loai"].get("khong_giai_thich"):
        errs.append(f"{s}: khác biệt không giải thích được {dif[s]['cap_theo_loai']['khong_giai_thich'][:5]}")
if chk["nodiac_khong_khop"]:
    errs.append(f"không dấu không khớp: {len(chk['nodiac_khong_khop'])}")
if leak["dev_states_hitting_test"] or leak["test_states_hitting_dev"]:
    errs.append(f"rò rỉ dev↔test: {leak['dev_states_hitting_test']} / {leak['test_states_hitting_dev']}")
if br.get("leak_dev_test", {}).get("dev_states_hitting_test") or br.get("leak_dev_test", {}).get("test_states_hitting_dev"):
    errs.append("build_revised báo rò rỉ dev↔test")
if sha_b(dev_b) != br["dev_sha256"] or sha_b(test_b) != br["test_sha256"] or sha_b(dev_b) != dif["ban_dung"]["dev"]:
    errs.append("file khu tạm khác hash lúc dựng")
if len(log) != br["edits_log_records"]:
    errs.append(f"lớp ghi đè đổi sau khi dựng ({br['edits_log_records']} -> {len(log)} bản ghi): dựng lại")
if os.path.getmtime(LAB) > os.path.getmtime(S / "dev_v1_2_trial.jsonl"):
    errs.append("label_decisions_remote.jsonl đổi sau khi dựng: dựng lại")
if args.update_dev is None and ((OUT / "test_v1.2.jsonl").exists() or (OUT / "FROZEN_test_v1.2.sha256").exists()):
    errs.append("data/bench_v1_2/ đã có test 1.2 đóng băng: không ghi đè")
if args.update_dev is not None:
    frozen = (OUT / "FROZEN_test_v1.2.sha256").read_text(encoding="utf-8").split()[0]
    mt_old = json.load(open(OUT / "manifest_test_v1.2.json", encoding="utf-8"))
    if not ((OUT / "test_v1.2.jsonl").read_bytes() == test_b and sha_b(test_b) == frozen == mt_old["sha256"]["test_v1.2.jsonl"]):
        errs.append("test dựng ra KHÁC test 1.2 đã đóng băng: không cập nhật")
    if os.access(OUT / "test_v1.2.jsonl", os.W_OK):
        errs.append("test_v1.2.jsonl không còn chỉ đọc")
    if not args.update_dev.strip():
        errs.append("thiếu lý do cập nhật")
for f, want in (("dev_v1_1.jsonl", "18ba27b4eb1fd5dade68a48a9ef220cd58871370da58f95bb1ff6283b4ced119"),
                ("test_v1.1.jsonl", "fbad818b11b7ca8e23767a7287e9db7dae9a6a767042c04eb79c053a8b811b0c")):
    if sha(B11 / f) != want:
        errs.append(f"{f} khác hash 1.1")
rc = subprocess.run([sys.executable, str(ROOT / "scripts/human_edits/release_check.py")], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
if rc.returncode != 0:
    errs.append(f"release_check mã {rc.returncode}")
if errs:
    print("KHÔNG ĐÓNG BĂNG:", json.dumps(errs, ensure_ascii=False))
    sys.exit(2)

# ---------------------------------------------------------------- số liệu cho manifest
dev = [json.loads(l) for l in dev_b.decode("utf-8").splitlines() if l.strip()]
test = [json.loads(l) for l in test_b.decode("utf-8").splitlines() if l.strip()]


def counts(rows):
    c = {"rows": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
         "by_source_lang_variant": dict(Counter(f"{r['source']}|{r['lang']}|{r['variant']}" for r in rows)),
         "by_kind": dict(Counter(f"{r['source']}|{r['kind']}" for r in rows if r["variant"] == "orig")),
         "target_hist_orig": dict(Counter(f"{r['source']}|{r['kind']}|{r['target']}" for r in rows if r["lang"] == "vi" and r["variant"] == "orig"))}
    return c


props = {p["id"]: p for p in read(ROOT / "config/vi_proposals.jsonl")}
note = lambda r: r.get("ghi_chu") or ""
dx = [r for r in log if r.get("loai") == "de_xuat"]
mau = [r for r in log if r.get("loai") == "mau"]
ds = [r for r in log if r.get("loai") == "dung_sai"]
lab = read(LAB)
latest_lab = {}
for x in lab:
    latest_lab[(x["split"], x["pair_id"])] = x
from importlib import util as _u  # noqa: E402
sys.path.insert(0, str(ROOT / "tools/review_app"))
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402
st = E.derive_state(log, inventory(ROOT / "config"))
applied = {i: v for i, v in st["de_xuat"].items() if v["hanh_dong"] in ("duyet", "sua_roi_duyet")}
so_lieu = {
    "edits_log_ban_ghi": len(log), "edits_log_sha256": sha(LOG),
    "de_xuat_ban_ghi": len(dx),
    "de_xuat_tu_xa_nguoi_dung_bam": sum("[từ xa" in note(r) and "ủy quyền" not in note(r) for r in dx),
    "de_xuat_ghi_chu_uy_quyen": sum("ủy quyền" in note(r) for r in dx),
    "de_xuat_theo_chat_nguoi_dung": sum("[từ xa" not in note(r) and "ủy quyền" not in note(r) and "chat" in note(r) for r in dx),
    "de_xuat_khac": sum("[từ xa" not in note(r) and "ủy quyền" not in note(r) and "chat" not in note(r) for r in dx),
    "de_xuat_dang_ap": len(applied),
    "de_xuat_dang_ap_theo_nguoi_viet": dict(Counter(props.get(i, {}).get("nguoi_de_xuat", "?") for i in applied)),
    "chuoi_mau_excel": sum("[excel]" in note(r) for r in mau), "chuoi_mau_tu_xa": sum("[từ xa" in note(r) for r in mau),
    "chuoi_mau_theo_chat": sum("chat" in note(r) and "[excel]" not in note(r) and "[từ xa" not in note(r) for r in mau),
    "dung_sai": {"giu_moi": sum(r["hanh_dong"] == "giu_moi" for r in ds), "hoan_tac": sum(r["hanh_dong"] == "hoan_tac" for r in ds)},
    "label_decisions_remote_ban_ghi": len(lab), "label_decisions_remote_sha256": sha(LAB),
    "label_decisions_moi_nhat_theo_cap": {f"{s_}|{h_}": n_ for (s_, h_), n_ in sorted(Counter((k[0], v["hanh_dong"]) for k, v in latest_lab.items()).items())},
    "label_decisions_ghi_chu_uy_quyen": sum("ủy quyền" in (x.get("ghi_chu") or "") for x in lab),
    "cach_dem": "de_xuat_tu_xa_nguoi_dung_bam: bản ghi đề xuất có '[từ xa' và không có 'ủy quyền' trong ghi_chu (gồm cả hoàn tác / từ chối ở màn Đúng→sai từ xa); "
                "de_xuat_ghi_chu_uy_quyen: ghi_chu có 'ủy quyền'; de_xuat_theo_chat_nguoi_dung: điều phối ghi qua công cụ theo lời người dùng trong chat; "
                "de_xuat_khac: còn lại (điều phối ghi theo quyết định người dùng chọn trên trang). Với 1.650 bản ghi đầu, cách đếm này cho đúng 545 / 132 / 66 như lần dựng thử."}
changes = read(S / "changes_v1_2_trial.jsonl")
can = {(c["split"], c["pair_id"]) for c in changes if "loai_cau" in c["loai_thay_doi"] or "doi_dap_an" in c["loai_thay_doi"]}
conf = {(x.get("split"), x.get("pair_id")) for x in read(UC)} if UC.exists() else set()
chg_stat = Counter()
for c in changes:
    for k in c["loai_thay_doi"]:
        chg_stat[f"{c['split']}|{k}"] += 1
    chg_stat[f"{c['split']}|tong_cap_doi"] += 1
    chg_stat[f"{c['split']}|do_chuoi_mau"] += bool(c["do_chuoi_mau"])
seen = {}
seen["vi_proposals"] = {p["pair_id"] for p in props.values() if p.get("loai") == "cau" and p.get("split") == "test"}
seen["label_decisions_remote"] = {x["pair_id"] for x in lab if x.get("split") == "test"}
seen["label_issues"] = {x["pair_id"] for f in sorted((ROOT / "results/translation_audit/label_second").glob("in_*.jsonl")) for x in read(f)
                        if str(x.get("audit_id", "")).startswith("test|")}
nguoi = sorted(set().union(*seen.values()))
m11 = json.load(open(B11 / "manifest_test_v1.1.json", encoding="utf-8"))
cfg = {f: sha(ROOT / "config" / f) for f in ("bench_v1_questions.json", "translations.json")}
common = {
    "version": "1.2", "seed": m11["seed"], "revisions": m11["revisions"], "licenses": m11["licenses"],
    "built_from": {"dev_v1_1.jsonl": f"data/bench_v1_1/dev_v1_1.jsonl ({sha(B11 / 'dev_v1_1.jsonl')})",
                   "test_v1.1.jsonl": f"data/bench_v1_1/test_v1.1.jsonl ({sha(B11 / 'test_v1.1.jsonl')})",
                   "edits_log": f"data/human_edits/edits_log.jsonl ({len(log)} bản ghi, {sha(LOG)})",
                   "label_decisions": f"results/human_review/label_decisions_remote.jsonl ({len(lab)} bản ghi, {sha(LAB)}; dòng MỚI NHẤT theo (split, pair_id) có hiệu lực)",
                   "config": cfg},
    "build_cmd": "bash scripts/human_edits/run_build_v1_2.sh (build_revised.py --nfc --provenance --test_labels …) rồi python scripts/human_edits/freeze_v1_2.py",
    "build_options": {"nfc": br["nfc"], "provenance": True, "template_overrides": br["template_overrides"], "test_label_decisions_applied": br["test_label_decisions"],
                      "en_rows_not_nfc": br["en_rows_not_nfc"]},
    "changes_vs_1_1": {"so_cap_doi": br["n_changed_pairs"], "theo_loai": dict(sorted(chg_stat.items())),
                       "cach_dem": "một cặp đếm ở mọi loại nó thuộc; 'chi_doi_nfc' chỉ cho cặp đổi byte mà không thuộc loại nào khác; 'loai_cau' loại trừ các loại khác",
                       "chi_tiet": "results/bench_v1_2_stage/changes_v1_2_trial.jsonl"},
    "provenance": {
        "dinh_nghia": {"vi_provenance": "source = văn bản tiếng Việt giữ nguyên như nguồn (chuẩn hóa NFC và đổi chuỗi mẫu không tính); revised = đã sửa state/câu hỏi/lựa chọn (vi_revised_fields liệt kê)",
                       "gold_provenance": "source = đáp án như nhãn gốc (target của 1.1 trước mọi chỉnh sửa); overridden = khác nhãn gốc; áp cho cả 3 biến thể của cặp"},
        "dong_vi_co_dau_theo_tap_nguon": chk["provenance_dong_vi_co_dau"], "dap_an_nguoc_nguon": chk["dap_an_nguoc_nguon"],
        "quy_trinh": "Đề xuất do LLM (Opus, Fable) viết, người làm dự án duyệt.",
        "cach_duyet_nguoi_dung_mo_ta": "đọc từng câu với đề xuất đổi đáp án và nhãn; đề xuất dịch mà Opus đồng ý thì duyệt theo lô sau khi đọc mẫu, nếu cần thiết thì gọi fable ra",
        "ban_dich_nguoi_dung_tu_viet": {"mo_ta": "người dùng tự dịch hoặc bảo chỉnh trong chat; điều phối áp qua đề xuất (nguoi_de_xuat = nguoi_lam_du_an hoặc dieu_phoi) và công cụ",
                                        "cap_dev": dif["dev"]["cap_theo_loai"].get("nguoi_dung_trong_chat", [])},
        "so_lieu_tu_log": so_lieu,
        "so_lieu_dieu_phoi_cung_cap": {"nguoi_dung_tu_bam": "5 trên trang, 1 lô 350, 6 trong chat, Excel chuỗi mẫu",
                                       "theo_uy_quyen_theo_agent": "132 đề xuất, 141 nhãn, 182 đề xuất kiểm tra cuối, và các điều chỉnh an toàn",
                                       "ghi_chu": "log không tách được các nhóm 5 trên trang / 1 lô 350 / 6 trong chat / 182 đề xuất kiểm tra cuối; số do điều phối cung cấp"}},
    "dung_sai_muc_B": {"so_cau_dua_doc": 27, "giu_moi": so_lieu["dung_sai"]["giu_moi"], "hoan_tac": dif["dung_sai"]["hoan_tac"],
                       "file": "results/bench_v1_2_trial/dung_sai_items.json"},
    "xac_nhan_nguoi_dung": {"file": "results/human_review/user_confirmations.jsonl", "co_file": UC.exists(), "so_cau_loai_hoac_doi_dap_an": len(can),
                            "da_xac_nhan": len(can & conf), "ghi_chu": "người dùng chưa bấm 'Đồng ý (đã đọc)' cho câu nào; file chưa tồn tại" if not UC.exists() else ""},
    "so_voi_ban_thu_cuoi": {"ban_thu": dif["ban_thu"], "theo_loai": {s: dif[s]["theo_loai"] for s in ("dev", "test")},
                            "chi_tiet": "results/bench_v1_2_stage/diff_vs_trial_v1_2.json"},
    "release_check": {"exit_code": rc.returncode, "dau_ra": [l for l in rc.stdout.splitlines() if l and not l.startswith("  ")][:4]},
    "n_tokens": "rendered prompt, tokenizer Qwen/Qwen3-0.6B (tính lại cho dòng đổi chữ)",
    "nodiac": "strip combining marks (NFD) and đ/Đ→d/D on state, question, options; sinh lại từ bản VI đã sửa (đã kiểm: 0 lệch)",
    "nfc": "mọi trường tiếng Việt chuẩn hóa NFC; dòng tiếng Anh giữ nguyên",
    "leakage": {"dev_states_hitting_test": leak["dev_states_hitting_test"], "test_states_hitting_dev": leak["test_states_hitting_dev"],
                "train_v0_vs_test": leak["train_v0_vs_test"]["total"], "train_v0_vs_dev_v1_2": leak["train_v0_vs_dev_v1_2"]["total"]},
    "notes": ["Bản 1.2 = bench 1.1 sau khi người soát: lớp ghi đè data/human_edits (đề xuất đã duyệt, sửa câu, loại câu, đổi đáp án dev), quyết định nhãn test, chuẩn hóa NFC.",
              "MLQA để bản 1.3; Global-PIQA không làm.",
              "1 cặp dev bị loại vì trùng gần với test do sửa dịch: dev massive:4129 ('tôi đã đặt báo thức nào ngay bây giờ') chứa trọn test massive:test:1483 "
              "('tôi đã đặt báo thức nào', sau khi sửa 'lời nhắc' -> 'báo thức' cho khớp EN 'alarms'); test giữ nguyên để giữ quy tắc dev↔test = 0; "
              f"dev MASSIVE còn {sum(1 for r in dev if r['source'] == 'massive' and r['lang'] == 'vi' and r['variant'] == 'orig')} cặp.",
              "Không chạy model nào trên test 1.2; test 1.1 giữ ở data/bench_v1_1/ làm đối chiếu."],
}
md = {"name": "dev_v1_2", "sha256": {"dev_v1_2.jsonl": sha_b(dev_b)}, "counts": counts(dev), **common}
if args.update_dev is not None:
    md_old = json.load(open(OUT / "manifest_dev_v1_2.json", encoding="utf-8"))
    old_rows = {json.loads(l)["id"]: json.loads(l) for l in open(OUT / "dev_v1_2.jsonl", encoding="utf-8") if l.strip()}
    new_rows = {r["id"]: r for r in dev}
    doi = sorted({(old_rows.get(i) or new_rows.get(i))["pair_id"] for i in set(old_rows) | set(new_rows) if old_rows.get(i) != new_rows.get(i)})
    md["cap_nhat"] = md_old.get("cap_nhat", []) + [{"thoi_diem": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "ly_do": args.update_dev,
                                                     "sha_cu": md_old["sha256"]["dev_v1_2.jsonl"], "sha_moi": sha_b(dev_b), "cap_doi": doi,
                                                     "test_khong_doi": mt_old["sha256"]["test_v1.2.jsonl"]}]
    md["dong_bang_test_luc"] = mt_old["frozen_at"]
mt = {"name": "bench_v1 test", "frozen": True, "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
      "sha256": {"test_v1.2.jsonl": sha_b(test_b)}, "counts": counts(test), **common,
      "test_ids_nguoi_da_xem": nguoi, "test_ids_nguoi_da_xem_nguon": {k: len(v) for k, v in seen.items()},
      "test_ids_agent_da_doc": "toàn bộ cặp test 1.1 (lượt kiểm tra cuối Opus đã đọc hết: verify + reread 40 lô)",
      "note": "Không sửa file này sau khi đóng băng. Không chạy model nào trên tập này. Bản 1.1 giữ ở data/bench_v1_1/ làm đối chiếu."}
md_s, mt_s, leak_s = (json.dumps(x, ensure_ascii=False, indent=1) for x in (md, mt, leak))   # dựng xong mọi thứ rồi mới ghi một lượt
if args.update_dev is not None:     # chỉ dev: không đụng test, FROZEN, manifest_test
    (OUT / "dev_v1_2.jsonl").write_bytes(dev_b)
    (OUT / "leakage_v1_2.json").write_text(leak_s, encoding="utf-8")
    (OUT / "manifest_dev_v1_2.json").write_text(md_s, encoding="utf-8")
    print(json.dumps({"cap_nhat_dev": md["cap_nhat"][-1], "dev_rows": len(dev), "test_sha_khong_doi": sha(OUT / "test_v1.2.jsonl")}, ensure_ascii=False))
    sys.exit(0)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "dev_v1_2.jsonl").write_bytes(dev_b)
tp = OUT / "test_v1.2.jsonl"
tp.write_bytes(test_b)
os.chmod(tp, stat.S_IREAD)
(OUT / "FROZEN_test_v1.2.sha256").write_text(f"{sha_b(test_b)}  test_v1.2.jsonl\n", encoding="utf-8")
(OUT / "leakage_v1_2.json").write_text(leak_s, encoding="utf-8")
(OUT / "manifest_dev_v1_2.json").write_text(md_s, encoding="utf-8")
mp = OUT / "manifest_test_v1.2.json"
mp.write_text(mt_s, encoding="utf-8")
os.chmod(mp, stat.S_IREAD)
print(json.dumps({"dev_v1_2.jsonl": sha_b(dev_b), "test_v1.2.jsonl": sha_b(test_b), "dev_rows": len(dev), "test_rows": len(test),
                  "test_read_only": not os.access(tp, os.W_OK), "so_lieu_tu_log": {k: v for k, v in so_lieu.items() if k not in ("cach_dem",)}},
                 ensure_ascii=False))
