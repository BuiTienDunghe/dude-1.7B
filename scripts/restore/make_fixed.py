"""Bản "đã sửa" (kiểm độ nhạy) của đầu ra khôi phục dấu: chiếu đầu ra thô về đầu vào không dấu để bất biến luôn đúng.
Dựng được trong pipeline thật vì lúc suy luận vẫn có đầu vào không dấu.

Quy tắc, áp riêng cho state, câu hỏi và từng lựa chọn:
1. Tách đầu vào không dấu N thành các từ theo khoảng trắng; giữ nguyên mọi khoảng trắng của N.
2. Tách đầu ra thô R thành các từ theo khoảng trắng.
3. Khóa của một từ = các ký tự chữ-số của nó (bỏ dấu câu và ký hiệu); với từ của R thì lấy khóa sau khi bỏ dấu.
   Căn hai dãy khóa bằng difflib (autojunk tắt).
4. Từ của N nằm trong khối khớp: lấy dấu từ từ tương ứng của R theo từng ký tự (căn ký tự trong từ bằng difflib; ký tự khớp
   lấy chữ có dấu của R, ký tự không khớp — thường là dấu câu R đã làm rơi, như ngoặc kép cong — giữ nguyên ký tự của N).
5. Từ của N không nằm trong khối khớp (R bỏ sót, lặp, bịa, hoặc bị cắt ở max_length): giữ nguyên từ không dấu.
Kết quả F luôn thỏa strip_diacritics(F) == N (kiểm bằng assert, khớp cả khoảng trắng).

Cách dùng: make_fixed.py <rows_dir> <name>
Ra: <rows_dir>/dev_v1_vi_restored_<name>_fixed.jsonl (id = id có dấu + ":restore:<name>_fixed"), <rows_dir>/fixed_stats_<name>.json
"""
import difflib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, read_jsonl, sha256_file, strip_diacritics, write_jsonl  # noqa: E402

rows_dir, name = Path(sys.argv[1]), sys.argv[2]
rest = read_jsonl(rows_dir / f"dev_v1_vi_restored_{name}.jsonl")
dev = {r["id"]: r for r in read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")}
key = lambda w: "".join(c for c in w if c.isalnum())


def transfer(nw, rw):
    """Từ nw (không dấu) và rw (có dấu, khóa trùng): lấy dấu của rw theo ký tự."""
    rw = unicodedata.normalize("NFC", rw)
    pairs = [(c, strip_diacritics(c)) for c in rw]
    pairs = [(c, x) for c, x in pairs if len(x) == 1]  # bỏ ký tự mà bỏ dấu không ra đúng một ký tự (vd. dấu kết hợp đứng riêng)
    out = list(nw)
    for a, b, n in difflib.SequenceMatcher(None, nw, "".join(x for _, x in pairs), autojunk=False).get_matching_blocks():
        for k in range(n):
            out[a + k] = pairs[b + k][0]
    return "".join(out)


def fix(N, R):
    """-> (F, số từ có chữ-số của N, số từ phải giữ nguyên không dấu)."""
    parts = re.split(r"(\s+)", N)
    nwords = parts[0::2]
    rwords = R.split()
    nk = [key(w) for w in nwords]
    rk = [key(strip_diacritics(w)) for w in rwords]
    out = list(nwords)
    matched = [False] * len(nwords)
    for a, b, n in difflib.SequenceMatcher(None, nk, rk, autojunk=False).get_matching_blocks():
        for k in range(n):
            out[a + k] = transfer(nwords[a + k], rwords[b + k])
            matched[a + k] = True
    parts[0::2] = out
    F = "".join(parts)
    assert strip_diacritics(F) == N, (N[:80], F[:80])
    n_words = sum(bool(k) for k in nk)
    n_kept = sum(bool(k) and not m for k, m in zip(nk, matched))
    return F, n_words, n_kept


out_rows = []
st = defaultdict(lambda: defaultdict(int))
for r in rest:
    oid = r["id"].split(":restore:")[0]
    nd = dev[oid + ":nodiac"]
    new = dict(r)
    tot = kept = 0
    for f in ("state", "question"):
        new[f], w, k = fix(nd[f], r[f]); tot += w; kept += k
        st[r["source"]][f + "_words"] += w; st[r["source"]][f + "_kept_nodiac"] += k
    opts = []
    for a, b in zip(nd["options"], r["options"]):
        F, w, k = fix(a, b); opts.append(F); tot += w; kept += k
        st[r["source"]]["options_words"] += w; st[r["source"]]["options_kept_nodiac"] += k
    new["options"] = opts
    new["id"] = f"{oid}:restore:{name}_fixed"
    new["variant"] = f"restored_{name}_fixed"
    st[r["source"]]["rows"] += 1
    st[r["source"]]["rows_changed_by_fix"] += any(new[f] != r[f] for f in ("state", "question", "options"))
    out_rows.append(new)
p = rows_dir / f"dev_v1_vi_restored_{name}_fixed.jsonl"
write_jsonl(p, out_rows)
res = {"name": name + "_fixed", "rows": len(out_rows), "sha256": sha256_file(p), "by_source": {k: dict(v) for k, v in st.items()}}
for s, v in res["by_source"].items():
    w = v["state_words"] + v["question_words"] + v["options_words"]
    k = v["state_kept_nodiac"] + v["question_kept_nodiac"] + v["options_kept_nodiac"]
    v["all_words"], v["all_kept_nodiac"], v["kept_share"] = w, k, k / max(w, 1)
(rows_dir / f"fixed_stats_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({s: (v["rows_changed_by_fix"], v["all_kept_nodiac"], round(v["kept_share"], 4)) for s, v in res["by_source"].items()}))
