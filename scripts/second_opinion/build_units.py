"""Ý kiến thứ hai cho đề xuất đang chờ trong config/vi_proposals.jsonl (chỉ đọc config). Gộp theo dòng: mỗi đơn vị = dòng VI đầy đủ
(dev_v1.1 hoặc test 1.1, áp lớp ghi đè hiện hành) + dòng EN tương ứng (nếu có) + mọi đề xuất trên dòng đó + kiểm máy (số lần gia_tri_cu
xuất hiện, NFC, khoảng trắng). Đề xuất mẫu (loai cũ) đi kèm danh mục chuỗi. Ra results/translation_audit/second_opinion/units_KK.jsonl
và template_units.jsonl. Python Windows, thư viện chuẩn."""
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/review_app"))
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402

OUT = ROOT / "results/translation_audit/second_opinion"
OUT.mkdir(parents=True, exist_ok=True)
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
props = read(ROOT / "config/vi_proposals.jsonl")
inv = inventory(ROOT / "config")
st = E.derive_state(E.read_log(ROOT / "data/human_edits/edits_log.jsonl"), inv)
pend = E.pending_proposals(props, st)
rows = {"dev": {r["id"]: r for r in read(ROOT / "data/bench_v1_1/dev_v1_1.jsonl") if r["variant"] == "orig"},
        "test": {r["id"]: r for r in read(ROOT / "data/bench_v1_1/test_v1.1.jsonl") if r["variant"] == "orig"}}
FIELD = {"state": "state", "question": "cau_hoi"}


def cur_value(split, r, truong):
    edits = (st["cau"] if split == "dev" else st["cau_test"]).get(r["pair_id"], {})
    f = E.PROPOSAL_FIELD(truong)
    if f in edits:
        return edits[f]["gia_tri_moi"]
    if truong == "state":
        return r["state"]
    if truong == "question":
        return r["question"]
    return r["options"][int(truong.split(":")[1])]


def view(split, r):
    edits = (st["cau"] if split == "dev" else st["cau_test"]).get(r["pair_id"], {})
    g = lambda f, base: edits[f]["gia_tri_moi"] if f in edits else base
    return {"state": g("state", r["state"]), "question": g("cau_hoi", r["question"]),
            "options": [g(f"lua_chon:{j}", o) for j, o in enumerate(r["options"])], "gold_index": r["target"]}


units = defaultdict(lambda: {"props": []})
tmpl = []
for p in pend:
    if p.get("loai") != "cau":
        tmpl.append(p)
        continue
    split = p["split"]; r = rows[split].get(p["row_id"])
    assert r is not None, p["id"]
    cur = cur_value(split, r, p["truong"])
    new = p["gia_tri_de_xuat"]
    checks = {"so_lan_gia_tri_cu": E.occurrences(cur, p["gia_tri_cu"]), "nfc_de_xuat": unicodedata.is_normalized("NFC", new),
              "nfc_gia_tri_cu": unicodedata.is_normalized("NFC", p["gia_tri_cu"]), "hai_dau_cach_de_xuat": "  " in new,
              "khoang_trang_dau_cuoi_doi": (new[:1].isspace() != p["gia_tri_cu"][:1].isspace()) or (new[-1:].isspace() != p["gia_tri_cu"][-1:].isspace()),
              "flores": "[FLORES" in p.get("giai_thich", "")}
    u = units[(split, p["row_id"])]
    u.update({"split": split, "row_id": p["row_id"], "pair_id": p["pair_id"], "nguon": p["nguon"], "vi": view(split, r)})
    en = rows[split].get(p["pair_id"] + ":en")
    u["en"] = {"state": en["state"], "question": en["question"], "options": en["options"], "gold_index": en["target"]} if en else None
    u["props"].append({k: p[k] for k in ("id", "truong", "gia_tri_cu", "gia_tri_de_xuat", "ly_do", "giai_thich", "doi_dap_an", "bang_chung_en")} | {"kiem_may": checks})
ulist = sorted(units.values(), key=lambda u: (u["split"], u["nguon"], u["row_id"]))
# chia lô cân theo độ dài văn bản (Belebele dài)
size = lambda u: len(json.dumps(u, ensure_ascii=False))
total = sum(size(u) for u in ulist)
K = 9
target = total / K
batches, cur_b, acc = [], [], 0
for u in ulist:
    cur_b.append(u); acc += size(u)
    if acc >= target and len(batches) < K - 1:
        batches.append(cur_b); cur_b, acc = [], 0
batches.append(cur_b)
for k, b in enumerate(batches):
    with open(OUT / f"units_{k:02d}.jsonl", "w", encoding="utf-8") as f:
        for u in b:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
INV = {x["key"]: x for x in inv}
with open(OUT / "template_units.jsonl", "w", encoding="utf-8") as f:
    for p in tmpl:
        x = INV.get(p.get("khoa"))
        f.write(json.dumps({"de_xuat": p, "danh_muc": ({k: x[k] for k in ("key", "group", "en", "label", "vi", "used_in")} if x else None),
                            "cung_nhom": [{"key": y["key"], "en": y["en"], "vi": st["mau"][y["key"]]["gia_tri"]} for y in inv if x and y["group"] == x["group"]]},
                           ensure_ascii=False) + "\n")
flags = defaultdict(int)
for u in ulist:
    for p in u["props"]:
        c = p["kiem_may"]
        flags["so_lan!=1"] += c["so_lan_gia_tri_cu"] != 1; flags["khong_nfc"] += not c["nfc_de_xuat"]; flags["hai_dau_cach"] += c["hai_dau_cach_de_xuat"]
        flags["flores"] += c["flores"]
print(json.dumps({"pending": len(pend), "cau": sum(len(u["props"]) for u in ulist), "units": len(ulist), "batches": [len(b) for b in batches],
                  "template": len(tmpl), "kiem_may": dict(flags)}, ensure_ascii=False))
