"""Kiểm bản đã khôi phục dấu của dev_v1: bất biến, chất lượng so với bản có dấu gốc, độ trễ.

1. Bất biến: bỏ dấu (common.strip_diacritics) của bản đã khôi phục phải ra đúng bản không dấu ban đầu, chỉ cho lệch khoảng trắng
   (so sau khi xóa mọi khoảng trắng). Đếm theo trường (state / câu hỏi / lựa chọn) và theo dòng; in tối đa 5 ví dụ, cắt 200 ký tự.
2. Chất lượng (so với bản có dấu gốc, NFC, phân biệt hoa/thường, bỏ qua khoảng trắng):
   - tỉ lệ âm tiết đúng: âm tiết = cụm giữa hai khoảng trắng của bản gốc có ít nhất một chữ cái; đúng khi mọi ký tự khớp.
     Dòng thỏa bất biến: so từng ký tự theo vị trí; dòng vi phạm: căn bằng difflib.
   - tỉ lệ âm tiết có dấu đúng (âm tiết mà bỏ dấu làm thay đổi);
   - tỉ lệ state / câu hỏi / mọi lựa chọn khớp hoàn toàn.
3. Độ trễ của bộ khôi phục, batch 1, theo độ dài state (token Qwen, trường n_state_tokens của dev_v1).

Cách dùng: check_restore.py <rows_dir> <name>  -> <rows_dir>/quality_<name>.json
"""
import difflib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, read_jsonl, strip_diacritics  # noqa: E402

rows_dir, name = Path(sys.argv[1]), sys.argv[2]
rest = read_jsonl(rows_dir / f"dev_v1_vi_restored_{name}.jsonl")
base = name[:-len("_fixed")] if name.endswith("_fixed") else name  # bản đã sửa dùng log/meta của bản thô
log = {r["id"]: r for r in read_jsonl(rows_dir / f"log_{base}.jsonl")}
dev = {r["id"]: r for r in read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")}

nfc = lambda s: unicodedata.normalize("NFC", s)
nows = lambda s: re.sub(r"\s+", "", s)


def syl_eval(orig, restored):
    o, r = nfc(orig), nfc(restored)
    toks = o.split()
    on, rn = "".join(toks), nows(r)
    if len(on) == len(rn):
        mask = [a == b for a, b in zip(on, rn)]
    else:
        mask = [False] * len(on)
        for a, b, n in difflib.SequenceMatcher(None, on, rn, autojunk=False).get_matching_blocks():
            for k in range(n):
                mask[a + k] = True
    n = c = nd = cd = 0
    pos = 0
    for t in toks:
        ok = all(mask[pos:pos + len(t)])
        pos += len(t)
        if not any(ch.isalpha() for ch in t):
            continue
        n += 1; c += ok
        if strip_diacritics(t) != t:
            nd += 1; cd += ok
    return n, c, nd, cd, on == rn


from collections import Counter
viol = {"state": 0, "question": 0, "options": 0, "row": 0}
dropped, inserted = Counter(), Counter()
viol_kind = Counter()  # truncated / chỉ mất ký tự không phải chữ-số / có đổi chữ-số
viol_by_src = defaultdict(int)
examples = []
agg = defaultdict(lambda: defaultdict(float))
lat = defaultdict(list)
for r in rest:
    oid = r["id"].split(":restore:")[0]
    o, nd = dev[oid], dev[oid + ":nodiac"]
    bad = []
    for f in ("state", "question"):
        if nows(strip_diacritics(r[f])) != nows(nd[f]):
            bad.append((f, nd[f], r[f]))
    if len(r["options"]) != len(nd["options"]) or any(nows(strip_diacritics(a)) != nows(b) for a, b in zip(r["options"], nd["options"])):
        bad.append(("options", " | ".join(nd["options"]), " | ".join(r["options"])))
    for f, _, _ in bad:
        viol[f] += 1
    if bad:
        viol["row"] += 1; viol_by_src[r["source"]] += 1
        alnum_change = False
        for f, a, b in bad:
            sa, sb = nows(a), nows(strip_diacritics(b))
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
                if op in ("delete", "replace"):
                    dropped.update(sa[i1:i2]); alnum_change |= any(ch.isalnum() for ch in sa[i1:i2])
                if op in ("insert", "replace"):
                    inserted.update(sb[j1:j2]); alnum_change |= any(ch.isalnum() for ch in sb[j1:j2])
        viol_kind["truncated_output" if log[oid]["truncated"] else "letters_or_digits_changed" if alnum_change
                  else "only_non_alnum_chars_lost_or_added"] += 1
        if len(examples) < 50:
            f, a, b = bad[0]
            sa, sb = nows(a), nows(strip_diacritics(b))
            k = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
            examples.append({"id": oid[:120], "source": r["source"], "field": f, "at_char": k,
                             "nodiac_near": a[max(0, k - 60):k + 140][:200], "restored_near": b[max(0, k - 60):k + 140][:200]})
    for key in (r["source"], "block:" + r["block"], "all"):
        g = agg[key]
        n, c, ndd, cd, ex = syl_eval(o["state"], r["state"])
        g["rows"] += 1; g["syl"] += n; g["syl_ok"] += c; g["dsyl"] += ndd; g["dsyl_ok"] += cd; g["state_exact"] += ex
        nq, cq, _, _, exq = syl_eval(o["question"], r["question"])
        g["q_syl"] += nq; g["q_syl_ok"] += cq; g["question_exact"] += exq
        g["options_exact"] += all(nfc(nows(a)) == nfc(nows(b)) for a, b in zip(o["options"], r["options"]))
        g["row_violation"] += bool(bad)
    L = log[oid]
    ns = int(o.get("n_state_tokens") or 0)
    b = "≤128" if ns <= 128 else "129–512" if ns <= 512 else "513–1024" if ns <= 1024 else ">1024"
    for key in (b, "all"):
        lat[key].append((L["total_ms"], L["state_ms"], L["calls"]))

# in và lưu tối đa 5 ví dụ; ưu tiên nguồn khác ViHSD (không in nội dung ViHSD trừ khi cần)
examples = sorted(examples, key=lambda e: e["source"] == "vihsd")[:5]
quality = {}
for k, g in agg.items():
    quality[k] = {"rows": int(g["rows"]), "syllable_acc": g["syl_ok"] / max(g["syl"], 1), "n_syllables": int(g["syl"]),
                  "diacritic_syllable_acc": g["dsyl_ok"] / max(g["dsyl"], 1), "n_diacritic_syllables": int(g["dsyl"]),
                  "state_exact": g["state_exact"] / g["rows"], "question_syllable_acc": g["q_syl_ok"] / max(g["q_syl"], 1),
                  "question_exact": g["question_exact"] / g["rows"], "options_all_exact": g["options_exact"] / g["rows"],
                  "rows_with_invariant_violation": int(g["row_violation"])}
latency = {}
for k, v in lat.items():
    t = np.array([x[0] for x in v]); s = np.array([x[1] for x in v]); c = np.array([x[2] for x in v])
    latency[k] = {"rows": len(v), "total_p50_ms": float(np.percentile(t, 50)), "total_p90_ms": float(np.percentile(t, 90)),
                  "total_mean_ms": float(t.mean()), "state_p50_ms": float(np.percentile(s, 50)), "calls_mean": float(c.mean())}
meta = json.loads((rows_dir / f"meta_{base}.json").read_text(encoding="utf-8"))
res = {"name": name, "rows": len(rest), "invariant_violations": viol, "violation_kinds": dict(viol_kind),
       "dropped_chars_top": dropped.most_common(25), "inserted_chars_top": inserted.most_common(25), "violations_by_source": dict(viol_by_src), "examples": examples,
       "quality": quality, "latency": latency, "truncated_outputs": meta["total_truncated"], "forced_splits": meta["total_forced_splits"]}
(rows_dir / f"quality_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("kinds", json.dumps(dict(viol_kind)), "dropped", json.dumps(dropped.most_common(10), ensure_ascii=False)[:200])
print(json.dumps({"name": name, "rows": len(rest), "violations": viol, "truncated": meta["total_truncated"],
                  "forced_splits": meta["total_forced_splits"]}, ensure_ascii=False))
a = quality["all"]
print(f"syllable_acc {a['syllable_acc']:.4f} diacritic_syl_acc {a['diacritic_syllable_acc']:.4f} state_exact {a['state_exact']:.3f} "
      f"question_exact {a['question_exact']:.3f} options_exact {a['options_all_exact']:.3f}")
for e in examples:
    print(f"{e['id'][:150]} {e['field']} @{e['at_char']}")
    print("  không dấu: " + e["nodiac_near"].replace("\n", " ")[:200])
    print("  khôi phục: " + e["restored_near"].replace("\n", " ")[:200])
