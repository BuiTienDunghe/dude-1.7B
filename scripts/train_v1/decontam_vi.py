"""Việc (c), viết sẵn, CHẠY KHI CÓ BẢN DỊCH: khử rò rỉ lần hai trên BẢN TIẾNG VIỆT của phần B, rồi dựng thành phần B và C.
  1) kiểm định dạng bằng results/train_v1/to_translate/kiem_ban_dich.py (phải ĐẠT, không thì dừng);
  2) dựng dòng tiếng Việt: nhãn/target/metadata lấy từ part_b_en.jsonl (không lấy từ bản dịch); câu noul dùng Có/Không của config/train_format.json;
     cờ dich_may = true; id = id gốc + ":vi";
  3) qua finalize của recast_v0 (bộ lọc v0, state <= 1.024 token) với chỉ mục TIẾNG VIỆT mở rộng của refs_index (toàn bộ văn bản VI của dev
     và test 1.3 kể cả không dấu; Belebele/FLORES/SIB-200/MASSIVE/XCOPA vi/ViFactCheck/ViWikiFC/MLQA/Global-PIQA tiếng Việt) và train_v0/val_v0;
     thêm: so với chính phần A (tránh trùng giữa hai phần);
  4) ghi part_b_vi.jsonl, part_b_vi_loai.json, components/B.jsonl (nền + B), components/C.jsonl (nền + A + B), kiểm trần 5%.
KHÔNG train. Người dùng phải đọc kiểm 30 dòng (make_review_html.py) trước khi train.
Cách chạy (venv WSL): bash setup/run.sh scripts/train_v1/decontam_vi.py <bản_dịch>.jsonl [--out_dir results/train_v1]
QUY TẮC CHỮ NHÃN (từ 28/10, audit 27/10): mọi bộ dựng dữ liệu lấy chữ nhãn / lựa chọn cố định / câu hỏi mẫu TỪ DEV 1.3
(data/bench_v1_3/dev_v1_3.jsonl) hoặc từ lớp ghi đè người dùng đã duyệt (data/human_edits/edits_log.jsonl), KHÔNG lấy từ
config/bench_v1_questions.json hay config/translations.json (hai file này lệch dev, ví dụ "Văn bản không đủ thông tin" so với
"...để kết luận"). Kiểm bằng scripts/check_label_strings.py trước khi dùng file train."""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, DATA, RESULTS, load_json, model_revisions, read_jsonl, sha256_file, write_jsonl  # noqa: E402
from decontam import Index, normalize  # noqa: E402
from recast_v0 import make_finalize  # noqa: E402
from refs_index import build, check_same_as_stage0  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dich")
ap.add_argument("--out_dir", default=str(RESULTS / "train_v1"))
a = ap.parse_args()
T1 = RESULTS / "train_v1"
OUT = Path(a.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
CAP = 0.05

# 1) định dạng
rc = subprocess.run([sys.executable, str(T1 / "to_translate/kiem_ban_dich.py"), str(T1 / "to_translate/part_b_to_translate.jsonl"), a.dich,
                     "--report", str(OUT / "ket_qua_kiem_ban_dich.json")], capture_output=True, text=True, encoding="utf-8")
print(rc.stdout.splitlines()[0] if rc.stdout else rc.stderr[-500:])
if rc.returncode != 0:
    sys.exit("Bản dịch KHÔNG ĐẠT kiểm định dạng; dừng (xem ket_qua_kiem_ban_dich.json).")

# 2) dòng tiếng Việt
fmt = load_json(CONFIG / "train_format.json")
en = read_jsonl(T1 / "part_b_en.jsonl")
vi = {r["id"]: r for r in read_jsonl(a.dich)}
rows = []
for r in en:
    v = vi[r["id"]]
    rows.append({**r, "id": r["id"] + ":vi", "id_en": r["id"], "lang": "vi", "text_lang": "vi", "dich_may": True, "state": v["state"], "question": v["question"],
                 "options": list(fmt["noul_options"]["vi"]) if r["kind"] == "noul" else list(v["options"]), "n_state_tokens_en": r.get("n_state_tokens")})

# 3) khử rò rỉ phía tiếng Việt
# v2 (20/10): lựa chọn của Global-PIQA (solution*, eng_solution*) cũng chỉ xét trùng nguyên văn / MinHash / 13-gram như lựa chọn Belebele
# (lượt 1: cụm "không đủ thông tin" trong một lựa chọn Global-PIQA bắt nhầm khoảng 1.000 dòng mang nhãn cố định của dự án).
CHK, V0, rstats = build(langs=("vi",), opt_prefixes=("mc_answer", "solution", "eng_solution"))
# chuỗi cố định của dự án (giá trị bảng thuật ngữ v2, Có/Không) không phải dữ liệu nguồn: không đem đi kiểm rò rỉ
_g = load_json(RESULTS / "train_v1/to_translate/bang_thuat_ngu_v2.json")
FIXED = {normalize(v["vi"]) for part in ("thuat_ngu", "tien_to") for v in _g[part].values()} | {normalize(x) for x in load_json(CONFIG / "train_format.json")["noul_options"]["vi"]}


class SkipFixed:
    def __init__(self, ix):
        self.ix = ix

    def check(self, text):
        return None if normalize(text) in FIXED else self.ix.check(text)


CHK = {"vi": SkipFixed(CHK["vi"])}
V0 = SkipFixed(V0)
PA_ = Index()
partA = read_jsonl(T1 / "part_a.jsonl")
for r in partA:
    PA_.add(r["state"], "phan_A")
PA = SkipFixed(PA_)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
drops = Counter()
keep, lost = [], []
for r in rows:
    d = Counter()                                   # bộ đếm riêng từng dòng để ghi đúng lý do loại
    x = make_finalize(tok, [("ref_vi", CHK["vi"]), ("v0", V0), ("phan_A", PA)], d, max_state_tokens=1024)(dict(r))  # B/C dựng với 1.024
    drops.update(d)
    if x:
        keep.append(x)
    else:
        lost.append({"id": r["id_en"], "source": r["source"], "dang_bai": r["dang_bai"], "ly_do": next(iter(d))})

# 4) thành phần
plan = json.load(open(T1 / "components/B_plan.json", encoding="utf-8"))
bo = set(plan["base_train_v0"]["bo_ra"])
base = [r for r in read_jsonl(DATA / "train_v0.jsonl") if r["id"] not in bo]
assert len(base) == plan["base_train_v0"]["rows"]
res = {"ban_dich": {"file": a.dich, "sha256": sha256_file(a.dich)}, "dong_vao": len(rows), "giu": len(keep), "loai": len(lost),
       "loai_theo_ly_do": dict(drops.most_common()), "loai_theo_nguon": dict(Counter(x["source"] for x in lost)),
       "chi_muc_vi": rstats["ref_docs"]}
for name, parts in (("B", keep), ("C", partA + keep)):
    comp = base + parts
    share = Counter(r["source"] for r in comp)
    res[f"thanh_phan_{name}"] = {"rows": len(comp), "max_share": round(max(share.values()) / len(comp), 4),
                                 "vuot_tran": {s: n for s, n in share.items() if n > CAP * len(comp)}}
write_jsonl(OUT / "part_b_vi.jsonl", keep)
(OUT / "part_b_vi_loai.json").write_text(json.dumps(lost, ensure_ascii=False, indent=1), encoding="utf-8")
write_jsonl(OUT / "components/B.jsonl", base + keep)
write_jsonl(OUT / "components/C.jsonl", base + partA + keep)
res["files"] = {p: sha256_file(OUT / p) for p in ("part_b_vi.jsonl", "components/B.jsonl", "components/C.jsonl")}
res["ghi_chu"] = "C dùng part_a.jsonl hiện tại; nếu A dựng lại sau khi duyệt chuỗi tv1-* thì chạy lại script này. KHÔNG train trước khi người dùng đọc kiểm."
(OUT / "decontam_vi_stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "files"}, ensure_ascii=False))
