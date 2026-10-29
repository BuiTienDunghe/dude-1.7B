"""Hai phép thử probe_v1 KHÔNG cần duyệt chữ (quyết định 26/10, mục "Cập nhật 26/10" của BRIEF_phan_tich_xu_huong.md), dựng từ DEV 1.3.
BỘ PHÂN TÍCH, KHÔNG PHẢI BENCHMARK, KHÔNG BAO GIỜ DÙNG ĐỂ TRAIN. Không đọc test. Chỉ thư viện chuẩn.

(a) pa_lua_chon_khong_dau.jsonl: dòng VI có dấu của ViFactCheck, ViWikiFC (choice và noul), SIB-200, MASSIVE; lựa chọn = strip_diacritics(lựa chọn),
    văn bản và câu hỏi GIỮ DẤU; thứ tự và đáp án giữ nguyên. So với dòng không dấu toàn phần đã có trong dev để tách phần "đọc không dấu".
(b) pb_nhan_tieng_anh.jsonl: dòng VI có dấu của SIB-200, MASSIVE; lựa chọn = lựa chọn của dòng EN cùng cặp (cùng vị trí; script kiểm ánh xạ
    VI↔EN nhất quán trên toàn dev và đáp án EN = đáp án VI); văn bản và câu hỏi giữ tiếng Việt.
Kiểm trùng train_v1 (components/A, B, C): state / câu hỏi (>= 4 từ, trừ câu hỏi mẫu lặp >= 3 lần trong dev) trùng nguyên văn đã chuẩn hóa; và đếm số dòng train dùng đúng chuỗi lựa chọn
    mới làm lựa chọn (để biết chữ nhãn đó model đã gặp lúc train hay chưa; không phải rò rỉ).
Ra results/probe_v1/: hai file trên, mục "Pa_Pb" của manifest_probe_v1.json.
Dùng: python scripts/probe_v1/build_pa_pb.py"""
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, read_jsonl, sha256_file, strip_diacritics, write_jsonl  # noqa: E402

DEV = DATA / "bench_v1_3" / "dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
OUT = RESULTS / "probe_v1"
COMP = RESULTS / "train_v1" / "components"

assert sha256_file(DEV) == DEV_SHA, "dev 1.3 sai sha"
dev = read_jsonl(DEV)
assert not any(":test:" in r["id"] for r in dev)
by = {(r["pair_id"], r["lang"], r["variant"]): r for r in dev}
VIO = [r for r in dev if r["lang"] == "vi" and r["variant"] == "orig"]


def base(r, probe):
    x = {k: v for k, v in r.items() if k != "n_tokens"}       # n_tokens của dev không còn đúng (eval_base.py kiểm khi có trường này)
    x.update({"id": f"{r['id']}:{probe}", "probe": probe, "orig_id": r["id"], "orig_options": r["options"]})
    return x


pa = []
for r in VIO:
    if r["source"] in ("vifactcheck", "viwikifc", "sib200", "massive"):
        x = base(r, "Pa")
        x["options"] = [strip_diacritics(o) for o in r["options"]]
        assert len(set(x["options"])) == len(x["options"]), r["id"]
        pa.append(x)

vi2en = defaultdict(set)
pb = []
for r in VIO:
    if r["source"] in ("sib200", "massive"):
        e = by[(r["pair_id"], "en", "orig")]
        assert e["target"] == r["target"] and len(e["options"]) == len(r["options"]), r["id"]
        for v, en in zip(r["options"], e["options"]):
            vi2en[(r["source"], v)].add(en)
        x = base(r, "Pb")
        x.update({"options": list(e["options"]), "en_id": e["id"]})
        pb.append(x)
bad = {k: sorted(v) for k, v in vi2en.items() if len(v) > 1}
assert not bad, f"ánh xạ VI↔EN không nhất quán: {bad}"
assert len({en for k, v in vi2en.items() for en in v}) == len(vi2en), "hai nhãn VI trùng một nhãn EN"


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


FIXED = {k for k, v in Counter(norm(r["question"]) for r in dev).items() if v >= 3}   # câu hỏi mẫu (như build_p3_p4.py): phần A dùng lại có chủ ý
texts = defaultdict(set)
for x in pa + pb:
    for t in (x["state"], x["question"]):
        if len(re.findall(r"\w+", t)) >= 4 and norm(t) not in FIXED:
            texts[norm(t)].add(x["id"])
new_opts = {f: {norm(o) for x in rows for o in x["options"]} for f, rows in (("Pa", pa), ("Pb", pb))}
overlap = {}
for comp in ("A", "B", "C"):
    hit, used, n = set(), {"Pa": Counter(), "Pb": Counter()}, 0
    with open(COMP / f"{comp}.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            n += 1
            for s in (t["state"], t["question"]):
                if norm(s) in texts:
                    hit |= texts[norm(s)]
            for o in t["options"]:
                for p in ("Pa", "Pb"):
                    if norm(o) in new_opts[p]:
                        used[p][norm(o)] += 1
    overlap[comp] = {"file_sha256": sha256_file(COMP / f"{comp}.jsonl"), "rows": n, "state_cau_hoi_trung_nguyen_van": len(hit),
                     "vi_du": sorted(hit)[:10],
                     "lua_chon_moi_co_trong_train (số lần làm lựa chọn; không phải rò rỉ)": {p: {"so_chuoi": len(c), "so_lan": sum(c.values()),
                                                                                                   "nhieu_nhat": dict(c.most_common(8))} for p, c in used.items()}}

OUT.mkdir(parents=True, exist_ok=True)
write_jsonl(OUT / "pa_lua_chon_khong_dau.jsonl", pa)
write_jsonl(OUT / "pb_nhan_tieng_anh.jsonl", pb)
sec = {"files": {f: sha256_file(OUT / f) for f in ("pa_lua_chon_khong_dau.jsonl", "pb_nhan_tieng_anh.jsonl")},
       "Pa": {"cach_dung": "dòng VI có dấu; lựa chọn bỏ dấu (strip_diacritics), văn bản và câu hỏi giữ dấu; so với dev cùng câu (orig_id) và với dòng không dấu toàn phần",
              "so_cau": dict(Counter(f"{x['source']}|{x['kind']}" for x in pa))},
       "Pb": {"cach_dung": "dòng VI có dấu của SIB-200, MASSIVE; lựa chọn = lựa chọn của dòng EN cùng cặp (cùng vị trí, cùng đáp án); văn bản và câu hỏi tiếng Việt",
              "so_cau": dict(Counter(x["source"] for x in pb)), "so_nhan": {s: sum(1 for k in vi2en if k[0] == s) for s in ("sib200", "massive")}},
       "trung_train_v1": overlap}
mpath = OUT / "manifest_probe_v1.json"
man = json.loads(mpath.read_text(encoding="utf-8"))
man["Pa_Pb"] = sec
mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(sec, ensure_ascii=False, indent=1))
