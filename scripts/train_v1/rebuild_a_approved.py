"""Giai đoạn 0 train_v1: dựng lại phần A + thành phần A từ các chuỗi tv1-* NGƯỜI DÙNG ĐÃ DUYỆT (data/human_edits/edits_log.jsonl).
Quy tắc: với mỗi "khoa", lấy gia_tri_moi của bản ghi de_xuat (duyet / sua_roi_duyet) MỚI NHẤT theo thoi_diem (cùng thời điểm thì bản ghi sau trong log);
khoa nào có lần mới nhất là tu_choi thì dừng. Không đổi dòng nào khác: cùng tập id, cùng nhãn/target, chỉ thay chuỗi (lựa chọn ý định, lựa chọn
VNTC, state EXAMS). Lý/Hóa/Sinh đã duyệt nhưng không dùng (EXAMS đã lọc ở v2). Dòng đổi chữ được kiểm lại state <= 1.024 token và khử rò rỉ
(chỉ mục tiếng Việt mở rộng + train_v0/val_v0, như build_stage0), rồi kiểm toàn bộ A + nền lần cuối.
Đầu vào: bản v2 lưu ở results/train_v1/ban_v2/ (tự sao lần đầu). Ra: ghi đè part_a.jsonl, components/A.jsonl, components/C_plan.json;
thêm stage0_stats_A_duyet.json.
QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, model_revisions, read_jsonl, sha256_file, write_jsonl  # noqa: E402
from decontam import normalize  # noqa: E402
from refs_index import build, check_same_as_stage0  # noqa: E402

OUT = RESULTS / "train_v1"
B2 = OUT / "ban_v2"
CAP = 0.05
KEEP = ["part_a.jsonl", "components/A.jsonl", "components/B_plan.json", "components/C_plan.json", "stage0_stats_v2.json", "stage0_lengths.json"]
if not B2.exists():
    for f in KEEP:
        (B2 / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(OUT / f, B2 / f)
TT_SHA = "9e279b91606ecdac9436a2b673dc26b6ac49b6ec3b34fa2c9fe04bfd99917c14"
assert sha256_file(OUT / "to_translate/part_b_to_translate.jsonl") == TT_SHA

# ---------------------------------------------------------------- giá trị đã duyệt
log = [json.loads(l) for l in (DATA / "human_edits/edits_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
dec = [(r["thoi_diem"], k, r) for k, r in enumerate(log) if r.get("loai") == "de_xuat" and str(r.get("de_xuat_id", "")).startswith("tv1-")]
latest = {}
for t, k, r in sorted(dec, key=lambda x: (x[0], x[1])):
    latest[r["khoa"]] = r
bad = {kh: r["hanh_dong"] for kh, r in latest.items() if r["hanh_dong"] not in ("duyet", "sua_roi_duyet")}
assert not bad, bad
VAL = {kh: r["gia_tri_moi"] for kh, r in latest.items()}
SRC = {kh: r["de_xuat_id"] for kh, r in latest.items()}
INTENTS = ["ask_admin_change", "ask_location_mapping", "ask_new_name", "ask_old_name", "compare_before_after_merge", "recommend_tourist_spot",
           "search_tourist_address", "search_tourist_by_old_or_new"]
VNTC = ["Chinh tri Xa hoi", "Doi song", "Khoa hoc", "Kinh doanh", "Phap luat", "Suc khoe", "The gioi", "The thao", "Van hoa", "Vi tinh"]
K_INT = [f"train_v1.intent_tourism.label.{k}" for k in INTENTS]
K_VNTC = [f"train_v1.vntc.label.{k}" for k in VNTC]
K_TPL = "train_v1.exams.state_template"
SUBJ_USED = ["History", "Geography", "Citizenship"]
K_SUBJ = {s: f"train_v1.exams.subject.{s}" for s in SUBJ_USED}
need = K_INT + K_VNTC + [K_TPL] + list(K_SUBJ.values())
assert all(k in VAL for k in need), [k for k in need if k not in VAL]
st = {"so_quyet_dinh_tv1": len(dec), "khoa": len(latest), "dau_vao": {f: sha256_file(B2 / f) for f in KEEP}}

# ---------------------------------------------------------------- thay chuỗi
A2 = read_jsonl(B2 / "part_a.jsonl")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
changed, A = [], []
for r in A2:
    r = dict(r)
    used = []
    if r["source"] == "vi-tourism-admin-intent":
        assert len(r["options"]) == 8
        r["options"] = [VAL[k] for k in K_INT]; used = K_INT
    elif r["source"] == "vntc":
        assert len(r["options"]) == 10
        r["options"] = [VAL[k] for k in K_VNTC]; used = K_VNTC
    elif r["source"] == "exams/vi":
        subj, grade = r["variant"].split("/")
        assert subj in SUBJ_USED, subj
        r["state"] = VAL[K_TPL].format(mon=VAL[K_SUBJ[subj]], lop=int(grade))
        r["n_state_tokens"] = len(tok.encode(r["state"], add_special_tokens=False)); used = [K_TPL, K_SUBJ[subj]]
    else:
        assert not r["chuoi_cho_duyet"], r["id"]
    if used:
        assert r["chuoi_cho_duyet"], r["id"]
        r["chuoi_cho_duyet"] = []
        r["chuoi_da_duyet"] = {k: SRC[k] for k in used}
        changed.append(r["id"])
    A.append(r)
assert [r["id"] for r in A] == [r["id"] for r in A2]
assert all(a["target"] == b["target"] for a, b in zip(A, A2))
assert not any(r["chuoi_cho_duyet"] for r in A)

# ---------------------------------------------------------------- kiểm: token, khử rò rỉ (dòng đổi chữ + toàn bộ lần cuối), trần
CHK, V0, rstats = build(langs=("vi", "en"))
check_same_as_stage0(rstats)
leak = json.load(open(OUT / "ban_v1/v0_leak.json", encoding="utf-8"))
drop0 = {h["id"] for h in leak["hits"] if h["split"] == "train"}
base = [r for r in read_jsonl(DATA / "train_v0.jsonl") if r["id"] not in drop0]
assert len(base) == json.load(open(B2 / "components/B_plan.json", encoding="utf-8"))["base_train_v0"]["rows"]
hits = Counter()
units = lambda r: [r["state"]] + [t for t in [r["question"] or ""] + r["options"] if len(normalize(t).split()) >= 4]
for r in A:
    if r["n_state_tokens"] > 1024:
        hits["state_over_1024"] += 1
    if r["id"] in set(changed):
        for t in units(r):
            h = V0.check(t)
            if h:
                hits[f"A_doi_chu|v0|{h[0]}|{h[1]}"] += 1; break
for rows, lang, name in ((A, "vi", "A"), (base, "vi", "nen"), (base, "en", "nen")):
    for r in rows:
        for t in units(r):
            h = CHK[lang].check(t)
            if h:
                hits[f"{name}|{lang}|{h[0]}|{h[1]}"] += 1; break
assert not hits, hits
nA = len(base) + len(A)
share = Counter(r["source"] for r in base + A)
assert max(share.values()) <= CAP * nA

# ---------------------------------------------------------------- ghi
write_jsonl(OUT / "part_a.jsonl", A)
write_jsonl(OUT / "components/A.jsonl", base + A)
cplan = json.load(open(B2 / "components/C_plan.json", encoding="utf-8"))
cplan["ban"] = "v2 + phần A dựng lại từ chuỗi tv1-* đã duyệt"
cplan["parts"]["A"].update(sha256=sha256_file(OUT / "part_a.jsonl"), rows=len(A),
                           trang_thai="chuỗi tv1-* đã duyệt (edits_log, lần duyệt mới nhất mỗi khoa); không còn dòng chờ duyệt")
cplan["rows_total"] = len(base) + sum(p["rows"] for p in cplan["parts"].values())
(OUT / "components/C_plan.json").write_text(json.dumps(cplan, ensure_ascii=False, indent=1), encoding="utf-8")
bplan = json.load(open(OUT / "components/B_plan.json", encoding="utf-8"))
assert bplan["parts"]["B"]["sha256_en"] == sha256_file(OUT / "part_b_en.jsonl") == cplan["parts"]["B"]["sha256_en"]
st.update({
    "gia_tri_dung": {"y_dinh": {k: VAL[f"train_v1.intent_tourism.label.{k}"] for k in INTENTS}, "vntc": {k: VAL[f"train_v1.vntc.label.{k}"] for k in VNTC},
                     "exams_state": VAL[K_TPL], "exams_mon": {s: VAL[K_SUBJ[s]] for s in SUBJ_USED}},
    "nguon_gia_tri": {k: SRC[k] for k in need},
    "da_duyet_khong_dung": {k: VAL[k] for k in VAL if k not in need},
    "dong_doi_chu": len(changed), "theo_nguon_doi_chu": dict(Counter(i.split(":")[0] for i in changed)),
    "kiem": {"cham_chi_muc": 0, "state_over_1024": 0, "max_share_A": round(max(share.values()) / nA, 4), "rows_part_A": len(A), "rows_A": nA,
             "B_rows": bplan["rows_total"], "C_rows": cplan["rows_total"], "to_translate_sha": TT_SHA},
    "files": {f: sha256_file(OUT / f) for f in ["part_a.jsonl", "components/A.jsonl", "components/C_plan.json"]}})
(OUT / "stage0_stats_A_duyet.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(st, ensure_ascii=False))
