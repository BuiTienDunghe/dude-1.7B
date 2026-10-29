"""Chuẩn bị đọc lỗi MÙ bằng Opus cho train thử bản C (BRIEF_phan_tich_xu_huong.md mục 4; điều phối giao 26/10). Chỉ dev 1.3 và probe_v1 P4.
Không chấm model, không train, không đọc test. Thư viện chuẩn.

Nhóm (khóa giải mã để riêng ở khoa.json, KHÔNG đưa cho người đọc):
  A "C mới đúng": dòng dev VI có dấu, M1 sai → C@100% đúng; 50 câu.
  B "C mới sai" : dòng dev VI có dấu, M1 đúng → C@100% sai; 50 câu.
  A và B trộn chung, mã trung tính X001…, xáo theo seed, chia 4 lô 25 câu. Mỗi câu chỉ có: văn bản, câu hỏi, lựa chọn, đáp án đúng,
  đáp án "một model" chọn (luôn là đáp án của C). Người đọc không biết model nào, không biết câu thuộc nhóm nào.
  P4: dòng P4 VI có dấu (choice) mà C KHÔNG chọn "không đủ thông tin" còn M1 chọn đúng; 40 câu, mã P001…, 2 lô 20 câu.
Lấy mẫu phân tầng theo nguồn: chia theo tỉ lệ số câu có trong nhóm của từng nguồn (phần dư lớn nhất), mỗi nguồn có câu thì ít nhất 1; seed SEED.
Ra results/train_v1/phan_tich_C/doc_loi/: lo_01..04.md, lo_p4_01..02.md, khoa.json, chon_mau.json.
Dùng: python scripts/train_v1/doc_loi_prep.py"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import LETTERS, SEED, read_jsonl, sha256_file  # noqa: E402

DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
EV = ROOT / "results/probe_v1/eval"
OUT = ROOT / "results/train_v1/phan_tich_C/doc_loi"
M1, C = "Qwen3-1.7B-fullvocab-s1 (dự án)", "C-s1"
NEI = "Văn bản không đủ thông tin để kết luận"

assert sha256_file(DEV) == DEV_SHA
dev = {r["id"]: r for r in read_jsonl(DEV)}
p4 = {r["id"]: r for r in read_jsonl(ROOT / "results/probe_v1/p4_van_ban_khong_lien_quan.jsonl")}
load = lambda ph, n: {json.loads(l)["id"]: json.loads(l) for l in open(EV / ph / n / "preds.jsonl", encoding="utf-8")}
d1, dc, p1, pc = load("dev", M1), load("dev", C), load("p4", M1), load("p4", C)


def stratified(rows, n, rng):
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by[r["source"]].append(r)
    tot = sum(len(v) for v in by.values())
    if tot <= n:
        return sorted(rows, key=lambda r: r["id"])
    quota = {s: max(1, int(n * len(v) / tot)) for s, v in by.items()}
    rem = sorted(by, key=lambda s: -(n * len(by[s]) / tot - int(n * len(by[s]) / tot)))
    i = 0
    while sum(quota.values()) < n:
        s = rem[i % len(rem)]
        if quota[s] < len(by[s]):
            quota[s] += 1
        i += 1
    while sum(quota.values()) > n:
        s = max(quota, key=lambda s: quota[s])
        quota[s] -= 1
    out = []
    for s in sorted(by):
        out += rng.sample(by[s], min(quota[s], len(by[s])))
    return out


rng = random.Random(SEED)
vio = [r for r in dev.values() if r["lang"] == "vi" and r["variant"] == "orig"]
A_pool = [r for r in vio if not d1[r["id"]]["correct"] and dc[r["id"]]["correct"]]
B_pool = [r for r in vio if d1[r["id"]]["correct"] and not dc[r["id"]]["correct"]]
P_pool = [r for r in p4.values() if r["variant"] == "orig" and r["kind"] == "choice"
          and p1[r["id"]]["correct"] and r["options"][pc[r["id"]]["pred"]] != NEI]
A, B, P = stratified(A_pool, 50, rng), stratified(B_pool, 50, rng), stratified(P_pool, 40, rng)


def item(code, r, pred):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
    return (f"### {code}\n\n**Văn bản:**\n\n{r['state']}\n\n**Câu hỏi:** {r['question']}\n\n**Các lựa chọn:**\n\n{opts}\n\n"
            f"**Đáp án đúng:** {LETTERS[r['target']]}. {r['options'][r['target']]}\n\n"
            f"**Model chọn:** {LETTERS[pred]}. {r['options'][pred]}\n")


mix = [("A", r) for r in A] + [("B", r) for r in B]
rng.shuffle(mix)
khoa = {}
OUT.mkdir(parents=True, exist_ok=True)
for k in range(4):
    part = mix[25 * k:25 * (k + 1)]
    txt = [f"# Lô {k + 1:02d}: {len(part)} câu\n"]
    for j, (g, r) in enumerate(part):
        code = f"X{25 * k + j + 1:03d}"
        khoa[code] = {"nhom": g, "id": r["id"], "source": r["source"], "kind": r["kind"], "mo_ho": bool(r.get("mo_ho")),
                      "m1_pred": d1[r["id"]]["pred"], "c_pred": dc[r["id"]]["pred"], "target": r["target"]}
        txt.append(item(code, r, dc[r["id"]]["pred"]))
    (OUT / f"lo_{k + 1:02d}.md").write_text("\n".join(txt), encoding="utf-8")
rng.shuffle(P)
for k in range(2):
    part = P[20 * k:20 * (k + 1)]
    txt = [f"# Lô P4-{k + 1:02d}: {len(part)} câu\n"]
    for j, r in enumerate(part):
        code = f"P{20 * k + j + 1:03d}"
        khoa[code] = {"nhom": "P4", "id": r["id"], "source": r["source"], "partner_id": r["partner_id"], "c_pred": pc[r["id"]]["pred"],
                      "c_pred_nhan": r["options"][pc[r["id"]]["pred"]], "orig_gold_label": r["orig_gold_label"]}
        txt.append(item(code, r, pc[r["id"]]["pred"]))
    (OUT / f"lo_p4_{k + 1:02d}.md").write_text("\n".join(txt), encoding="utf-8")
(OUT / "khoa.json").write_text(json.dumps(khoa, ensure_ascii=False, indent=1), encoding="utf-8")
info = {"seed": SEED, "dev_sha256": DEV_SHA,
        "kho_A_M1_sai_C_dung": dict(Counter(r["source"] for r in A_pool)), "mau_A": dict(Counter(r["source"] for r in A)),
        "kho_B_M1_dung_C_sai": dict(Counter(r["source"] for r in B_pool)), "mau_B": dict(Counter(r["source"] for r in B)),
        "kho_P4": dict(Counter(r["source"] for r in P_pool)), "mau_P4": dict(Counter(r["source"] for r in P)),
        "P4_nhan_C_chon_trong_mau": dict(Counter(khoa[c]["c_pred_nhan"] for c in khoa if c.startswith("P")))}
(OUT / "chon_mau.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False, indent=1))
