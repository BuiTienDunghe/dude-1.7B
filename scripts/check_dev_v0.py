"""Kiểm tra dev_v0: cấu trúc cặp, không chạm phần để dành test, và in 5 cặp Anh–Việt mỗi nguồn để duyệt bằng mắt."""
import json
from collections import defaultdict

import numpy as np

from common import DATA, LETTERS, read_jsonl

rows = read_jsonl(DATA / "dev_v0.jsonl")
reserved = json.loads((DATA / "reserved_test.json").read_text(encoding="utf-8"))
FIELDS = ["id", "pair_id", "source", "lang", "kind", "state", "question", "options", "target", "split_origin", "n_tokens"]

# 1) đủ trường, id duy nhất, mỗi pair_id có đúng một en và một vi
assert all(all(f in r for f in FIELDS) for r in rows)
assert len({r["id"] for r in rows}) == len(rows)
pairs = defaultdict(dict)
for r in rows:
    pairs[r["pair_id"]][r["lang"]] = r
assert all(set(p) == {"en", "vi"} for p in pairs.values())

# 2) cùng pair_id: cùng target, cùng số lựa chọn, cùng bộ nhãn nhiễu (MASSIVE), cùng nhãn gốc
for pid, p in pairs.items():
    en, vi = p["en"], p["vi"]
    assert en["target"] == vi["target"] and len(en["options"]) == len(vi["options"]), pid
    assert en["gold_label"] == vi["gold_label"] and en["flags"] == vi["flags"], pid
    assert 0 <= en["target"] < len(en["options"]) <= len(LETTERS)
    assert en["state"] != vi["state"], pid  # bản Việt không phải bản Anh chép lại
print(f"OK structure: {len(rows)} rows, {len(pairs)} pairs")

# 3) thứ tự lựa chọn giống nhau giữa en/vi: với SIB/MASSIVE kiểm tra qua bảng dịch
tr = json.loads(open(DATA.parent / "config/translations.json", encoding="utf-8").read())
en2vi = {v["en"]: v["vi"] for d in ("sib200_topics", "massive_intents") for v in tr[d].values()}
n = 0
for p in pairs.values():
    if p["en"]["source"] in ("sib200", "massive"):
        assert [en2vi[o] for o in p["en"]["options"]] == p["vi"]["options"]
        n += 1
print(f"OK option order identical across en/vi for {n} label-based pairs")

# 4) không câu nào thuộc phần để dành test
bad = []
res_bele = set(reserved["belebele"]["question_keys"]) | set(reserved["belebele"]["links"])
for r in rows:
    s, key = r["source"], r["pair_id"].split(":", 1)[1]
    if s == "belebele" and (key in res_bele or key.rsplit("#", 1)[0] in res_bele):
        bad.append(r["id"])
    if s == "sib200" and int(key) in set(reserved["sib200"]["index_ids"]):
        bad.append(r["id"])
    if s == "massive" and int(key) in set(reserved["massive"]["ids"]):
        bad.append(r["id"])
    if s == "xcopa" and "validation" not in r["split_origin"]:
        bad.append(r["id"])
    if s in ("sib200", "massive", "xcopa") and "test" in r["split_origin"]:
        bad.append(r["id"])
assert not bad, bad[:10]
# chéo: không đoạn văn Belebele nào nằm ở cả hai nửa
assert not ({r["cluster"].split(":", 1)[1] for r in rows if r["source"] == "belebele"} & set(reserved["belebele"]["links"]))
print("OK no row belongs to the reserved test portions")

# 5) in 5 cặp mỗi nguồn
rng = np.random.default_rng(0)
for src in ["belebele", "sib200", "massive", "xcopa"]:
    pids = sorted(pid for pid, p in pairs.items() if p["en"]["source"] == src)
    print("\n" + "=" * 30, src, "=" * 30)
    for pid in sorted(rng.choice(pids, 5, replace=False)):
        print(f"--- {pid}  target={LETTERS[pairs[pid]['en']['target']]}  {pairs[pid]['en']['flags'][:1] if src == 'xcopa' else ''}")
        for lang in ("en", "vi"):
            r = pairs[pid][lang]
            st = r["state"] if len(r["state"]) < 260 else r["state"][:260] + "…"
            print(f"  [{lang}] STATE: {st}")
            print(f"  [{lang}] Q: {r['question']}")
            print(f"  [{lang}] OPTS: " + " | ".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"])))
