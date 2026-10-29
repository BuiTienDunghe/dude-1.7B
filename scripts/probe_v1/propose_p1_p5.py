"""Đề xuất chữ cho P1 "đổi chữ nhãn" và P5 "đổi câu hỏi" của probe_v1 (BRIEF_phan_tich_xu_huong.md mục 3). Chữ do worker (Opus) soạn;
NGƯỜI DÙNG DUYỆT qua công cụ duyệt (config/vi_proposals.jsonl -> data/human_edits/edits_log.jsonl). Script này không tự duyệt, không dùng chữ.

Dùng: python scripts/probe_v1/propose_p1_p5.py            # chỉ kiểm, in bảng
      python scripts/probe_v1/propose_p1_p5.py --ghi      # kiểm đạt thì append vào config/vi_proposals.jsonl (từ chối nếu đã có khóa probe_v1.*)
Kiểm: (1) chuỗi gốc lấy từ dev 1.3 phải khớp; (2) đề xuất không trùng (chuẩn hóa, có dấu và bỏ dấu) mọi câu hỏi / lựa chọn của
train_v1 A/B/C, dev 1.3, config/translations.json, config/bench_v1_questions.json, config/train_format.json, bảng thuật ngữ v2;
(3) trong cùng một nguồn, chuỗi gốc + v1 + v2 của mọi nhãn đôi một khác nhau; (4) độ giống chữ với chuỗi gốc (Jaccard theo từ) in ra để soát.
Bản không dấu của P1/P5 do script dựng file tạo bằng common.strip_diacritics, không phải đề xuất riêng."""
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, DATA, RESULTS, read_jsonl, strip_diacritics  # noqa: E402

from probe_strings import P1, P1_SOURCES, P5, P5_NOTE  # noqa: E402,F401


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(re.sub(r"[\s.,;:!?()\"'/-]+", " ", s).split())


def jacc(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / max(1, len(A | B))


dev = read_jsonl(DATA / "bench_v1_3" / "dev_v1_3.jsonl")
# (1) chuỗi gốc phải có trong dev
dev_opts = defaultdict(set)
for r in dev:
    if r["lang"] == "vi" and r["variant"] == "orig":
        for o in r["options"]:
            dev_opts[r["source"]].add(o)
# bản tiếng Anh để người duyệt đối chiếu: lấy từ dòng EN cùng cặp (cùng vị trí lựa chọn) trong dev
_by = {(r["pair_id"], r["lang"], r["variant"]): r for r in dev}
VI2EN = {"Được văn bản xác nhận": "supported: the text confirms the claim", "Bị văn bản bác bỏ": "refuted: the text contradicts the claim",
         "Văn bản không đủ thông tin để kết luận": "not enough information in the text", "Có": "yes (the text confirms the statement)",
         "Không": "no (the text does not confirm the statement)"}
for r in dev:
    e = _by.get((r["pair_id"], "en", "orig"))
    if r["lang"] == "vi" and r["variant"] == "orig" and e and r["source"] in ("sib200", "massive", "xcopa"):
        VI2EN[r["question"]] = e["question"]
        if r["source"] != "xcopa":
            VI2EN.update(zip(r["options"], e["options"]))
err = []
for src, lab, cur, v1, v2 in P1:
    for s in P1_SOURCES[src]:
        if cur not in dev_opts[s]:
            err.append(f"chuỗi gốc không có trong dev: {s} {lab} {cur!r}")
for src in ("sib200", "massive"):
    have = {cur for s_, _, cur, _, _ in P1 if s_ == src}
    if have != dev_opts[src]:
        err.append(f"{src}: thiếu/thừa nhãn {sorted(dev_opts[src] ^ have)}")
dev_q = {r["question"] for r in dev if r["lang"] == "vi" and r["variant"] == "orig"}
for k, cur, new in P5:
    if "{claim}" in cur:
        pre = cur.split("{claim}")[0]
        if not any(q.startswith(pre) for q in dev_q):
            err.append(f"câu hỏi gốc không có trong dev: {k}")
    elif cur not in dev_q:
        err.append(f"câu hỏi gốc không có trong dev: {k}")

# (2) tập chuỗi đã dùng
used = defaultdict(set)
def add(s, tag):
    if isinstance(s, str) and s.strip():
        for x in (norm(s), norm(strip_diacritics(s))):
            used[x].add(tag)
for comp in ("A", "B", "C"):
    with open(RESULTS / "train_v1" / "components" / f"{comp}.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            add(t["question"], f"train_v1:{comp}:cau_hoi")
            for o in t["options"]:
                add(o, f"train_v1:{comp}:lua_chon")
for r in dev:
    add(r["question"], "dev13:cau_hoi")
    for o in r["options"]:
        add(o, "dev13:lua_chon")


def walk(x, tag):
    if isinstance(x, dict):
        for v in x.values():
            walk(v, tag)
    elif isinstance(x, list):
        for v in x:
            walk(v, tag)
    elif isinstance(x, str) and len(x) < 300:
        add(x, tag)


for p in ("translations.json", "bench_v1_questions.json", "train_format.json"):
    walk(json.loads((CONFIG / p).read_text(encoding="utf-8")), f"config:{p}")
walk(json.loads((RESULTS / "train_v1/to_translate/bang_thuat_ngu_v2.json").read_text(encoding="utf-8")), "bang_thuat_ngu_v2")

for _, lab, cur, _, _ in P1:                  # tự kiểm chỉ mục: chuỗi gốc (có trong dev) phải bị coi là đã dùng
    if norm(cur) not in used:
        err.append(f"chỉ mục hỏng: chuỗi gốc {cur!r} không có trong tập đã dùng")
rows = []
for src, lab, cur, v1, v2 in P1:
    for ver, s in (("v1", v1), ("v2", v2)):
        rows.append({"khoa": f"probe_v1.P1.{src}.{lab}.{ver}", "phep": "P1", "nguon": src, "nhan": lab, "goc": cur, "de_xuat": s})
for k, cur, new in P5:
    rows.append({"khoa": f"probe_v1.P5.{k}", "phep": "P5", "nguon": k, "nhan": "", "goc": cur, "de_xuat": new})
for x in rows:
    probe = x["de_xuat"].replace("{claim}", "").strip()
    hit = used.get(norm(probe)) or used.get(norm(strip_diacritics(probe)))
    x["trung"] = sorted(hit) if hit else []
    x["jaccard_voi_goc"] = round(jacc(x["de_xuat"].replace("{claim}", ""), x["goc"].replace("{claim}", "")), 2)
    if hit:
        err.append(f"trùng chuỗi đã dùng: {x['khoa']} {x['de_xuat']!r} {sorted(hit)[:3]}")
# (3) đôi một khác nhau trong cùng nguồn (kể cả bỏ dấu)
for src in P1_SOURCES:
    seen = {}
    for s_, lab, cur, v1, v2 in P1:
        if s_ != src:
            continue
        for s in (cur, v1, v2):
            k = norm(strip_diacritics(s))
            if k in seen and seen[k] != (lab, s):
                err.append(f"trùng trong nguồn {src}: {s!r} ({lab}) với {seen[k]}")
            seen[k] = (lab, s)

for x in rows:
    print(f"{x['khoa']}\t{x['goc']}\t{x['de_xuat']}\tJ={x['jaccard_voi_goc']}")
print(f"\nP1: {sum(x['phep'] == 'P1' for x in rows)}  P5: {sum(x['phep'] == 'P5' for x in rows)}  lỗi: {len(err)}")
for e in err:
    print("LỖI", e)
if err:
    sys.exit(1)

if "--ghi" in sys.argv:
    prop = CONFIG / "vi_proposals.jsonl"
    old = [json.loads(l) for l in open(prop, encoding="utf-8") if l.strip()]
    assert not any(str(o.get("khoa", "")).startswith("probe_v1.") for o in old), "đã có đề xuất probe_v1.* trong vi_proposals.jsonl: không ghi lặp"
    now = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    out = []
    for x in rows:
        if x["phep"] == "P1":
            ly_do = (f"probe_v1 P1 'đổi chữ nhãn' (BRIEF_phan_tich_xu_huong.md mục 3), CHỈ dùng cho bộ phân tích, KHÔNG train, KHÔNG benchmark. "
                     f"Chuỗi hiện tại trong dev 1.3: '{x['goc']}' (nguồn {', '.join(P1_SOURCES[x['nguon']])}). Cách diễn đạt {x['khoa'][-2:]} "
                     f"cùng nghĩa, khác chữ; đã kiểm không trùng câu hỏi/lựa chọn của train_v1 A/B/C, dev 1.3, config và bảng thuật ngữ v2 "
                     f"(kể cả bỏ dấu). Bản không dấu tạo bằng strip_diacritics. Duyệt / sửa / bỏ đều được.")
        else:
            ly_do = (f"probe_v1 P5 'đổi câu hỏi' (BRIEF_phan_tich_xu_huong.md mục 3), CHỈ dùng cho bộ phân tích, KHÔNG train, KHÔNG benchmark. "
                     f"Câu hỏi hiện tại trong dev 1.3: '{x['goc']}'. {{claim}} được thay bằng nhận định của từng câu. Lựa chọn giữ nguyên chữ gốc "
                     f"(P1 và P5 tách riêng). Đã kiểm không trùng câu hỏi của train_v1 A/B/C, dev 1.3 và config. Duyệt / sửa / bỏ đều được.")
        out.append({"id": x["khoa"].replace("probe_v1.", "pv1-").replace(".", "-"), "khoa": x["khoa"], "gia_tri_de_xuat": x["de_xuat"],
                    "ban_tieng_anh": VI2EN.get(x["goc"], ""), "ly_do": ly_do, "nguoi_de_xuat": "worker probe_v1 (Opus), theo brief điều phối 26/10", "thoi_diem": now})
    with open(prop, "a", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"đã append {len(out)} đề xuất vào {prop}")
