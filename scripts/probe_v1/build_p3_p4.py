"""Bộ phép thử chẩn đoán probe_v1 (BRIEF_phan_tich_xu_huong.md mục 3): dựng P3 "bỏ văn bản" và P4 "văn bản không liên quan" từ DEV 1.3.
BỘ PHÂN TÍCH, KHÔNG PHẢI BENCHMARK, KHÔNG BAO GIỜ DÙNG ĐỂ TRAIN. Không đọc test. Chỉ dùng thư viện chuẩn (python Windows chạy được).

P3: Belebele (VI có dấu, VI không dấu, EN), ViFactCheck và ViWikiFC (VI có dấu, không dấu; cả choice lẫn noul):
    state = "" (khung prompt vẫn in nhãn "Văn bản:" rồi để trống: không thêm chữ tiếng Việt mới); câu hỏi, lựa chọn, đáp án giữ nguyên.
P4: ViFactCheck và ViWikiFC: nhận định của mỗi câu (bản có dấu) ghép với state của một câu KHÁC cùng nguồn, KHÁC BÀI
    (vifactcheck: khác cluster = khác bài báo; viwikifc: khác mã bài Wikipedia, trường thứ 3 của id uit_*), seed cố định.
    Bản không dấu dùng đúng cặp đó (state không dấu của câu được ghép, lấy từ dòng không dấu của dev).
    Đáp án: choice -> lựa chọn "Văn bản không đủ thông tin để kết luận" (vị trí lấy theo từng dòng); noul -> "Không"
    (noul không có nhãn "không đủ thông tin"; văn bản không liên quan thì không xác nhận phát biểu).
    Bỏ cặp khi state mới có thể chứa thông tin của nhận định (xem LOC). Hết ứng viên thì bỏ câu.
Kiểm trùng train_v1: mọi state / câu hỏi / lựa chọn (trừ chuỗi cố định lặp >= 3 lần trong dev) so khớp nguyên văn đã chuẩn hóa
    với state / câu hỏi / lựa chọn của components/A, B, C; nhận định và câu hỏi Belebele (>= 10 từ) thêm kiểm chứa 10-gram trong state train.
Ra results/probe_v1/: p3_bo_van_ban.jsonl, p4_van_ban_khong_lien_quan.jsonl, manifest_probe_v1.json.
Dùng: python scripts/probe_v1/build_p3_p4.py"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, read_jsonl, sha256_file, write_jsonl  # noqa: E402

DEV = DATA / "bench_v1_3" / "dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
OUT = RESULTS / "probe_v1"
COMP = RESULTS / "train_v1" / "components"
NEI = {"orig": "Văn bản không đủ thông tin để kết luận", "nodiac": "Van ban khong du thong tin de ket luan"}
KHONG = {"orig": "Không", "nodiac": "Khong"}
CLAIM_MARK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
# Ngưỡng lọc P4 (chốt trước khi xem kết quả; đo trên dev: nhận định với state của chính nó trung vị 0,50 bigram trùng,
# với state ngẫu nhiên khác bài 90% <= 0,057; Jaccard bigram state-state ngẫu nhiên 95% <= 0,022)
LOC = {"bigram_nhan_dinh_toi_da": 0.10, "jaccard_state_toi_da": 0.02, "ten_rieng": "mọi cụm chữ hoa (trừ chữ đầu câu) của nhận định không được có trong state mới",
       "so": "mọi số >= 2 chữ số của nhận định không được có trong state mới"}
NGRAM = 10

assert sha256_file(DEV) == DEV_SHA, "dev 1.3 sai sha"
dev = read_jsonl(DEV)
assert not any(":test:" in r["id"] for r in dev)
by = {(r["pair_id"], r["lang"], r["variant"]): r for r in dev}


def claim_of(q):
    for m in CLAIM_MARK:
        if m in q:
            return q.split(m, 1)[1]
    raise ValueError(q)


def words(s):
    return re.findall(r"\w+", s.lower())


def bigrams(s):
    t = words(s)
    return set(zip(t, t[1:]))


def names(c):
    """Cụm chữ hoa liên tiếp trong nhận định, bỏ chữ đầu câu; cụm có từ 2 chữ trở lên giữ nguyên cụm, cụm 1 chữ giữ chữ đó."""
    toks = re.findall(r"\w+", c)
    runs, cur = [], []
    for i, t in enumerate(toks):
        if i > 0 and (t[0].isupper() or any(ch.isupper() for ch in t[1:])):
            cur.append(t)
        else:
            if cur:
                runs.append(" ".join(cur))
            cur = []
    if cur:
        runs.append(" ".join(cur))
    return runs


def article(r):
    if r["source"] == "viwikifc":
        return "viwikifc:" + r["pair_id"].split(":")[2].split("~")[0].split("_")[2]
    return r["cluster"]


def leak(c, state):
    """Lý do bỏ cặp (claim c, state mới) hoặc None."""
    ws = set(re.findall(r"\w+", state))
    for n in names(c):
        if (n in state) if " " in n else (n in ws):
            return "ten_rieng"
    for x in re.findall(r"\d{2,}", c):
        if x in re.findall(r"\d+", state):
            return "so"
    b = bigrams(c)
    if b and len(b & bigrams(state)) / len(b) > LOC["bigram_nhan_dinh_toi_da"]:
        return "bigram_nhan_dinh"
    return None


def base(r, probe):
    x = {k: v for k, v in r.items() if k not in ("n_tokens",)}   # n_tokens của dev không còn đúng (eval_base.py kiểm khi có trường này)
    x.update({"id": f"{r['id']}:{probe}", "probe": probe, "orig_id": r["id"], "orig_target": r["target"], "orig_gold_label": r["gold_label"]})
    return x


# ---------------- P3
P3_SRC = ("belebele", "vifactcheck", "viwikifc")
p3 = []
for r in dev:
    if r["source"] in P3_SRC:
        x = base(r, "P3")
        x.update({"state": "", "n_state_tokens": 0})
        p3.append(x)

# ---------------- P4
rng = random.Random(SEED)
p4, drops, picked = [], Counter(), {}
fc = sorted([r for r in dev if r["source"] in ("vifactcheck", "viwikifc") and r["variant"] == "orig"], key=lambda r: r["pair_id"])
for src in ("vifactcheck", "viwikifc"):
    pool = {}
    for r in fc:
        if r["source"] == src:
            pool.setdefault(r["state"], r)            # một ứng viên cho mỗi state (nhiều câu chung một đoạn)
    pool = sorted(pool.values(), key=lambda r: r["pair_id"])
    for r in [x for x in fc if x["source"] == src]:
        c = claim_of(r["question"])
        cand = [p for p in pool if article(p) != article(r) and p["cluster"] != r["cluster"]]
        rng.shuffle(cand)
        why = Counter()
        for p in cand:
            a, b = bigrams(r["state"]), bigrams(p["state"])
            if len(a & b) / max(1, len(a | b)) > LOC["jaccard_state_toi_da"]:
                why["jaccard_state"] += 1
                continue
            w = leak(c, p["state"])
            if w:
                why[w] += 1
                continue
            picked[r["pair_id"]] = (p, dict(why))
            break
        else:
            drops[(src, r["kind"])] += 1
for r in fc:
    if r["pair_id"] not in picked:
        continue
    p, why = picked[r["pair_id"]]
    for variant in ("orig", "nodiac"):
        rv, pv = by[(r["pair_id"], "vi", variant)], by[(p["pair_id"], "vi", variant)]
        x = base(rv, "P4")
        x.pop("mo_ho", None)                          # cờ mơ hồ nói về nhãn gốc, không áp cho cặp mới
        tgt = rv["options"].index(NEI[variant] if rv["kind"] == "choice" else KHONG[variant])
        x.update({"state": pv["state"], "n_state_tokens": pv["n_state_tokens"], "target": tgt, "gold_label": "nei",
                  "gold_provenance": "probe_v1:P4", "partner_id": pv["id"], "partner_pair_id": p["pair_id"], "partner_cluster": p["cluster"],
                  "partner_article": article(p), "p4_ung_vien_bi_loai": why})
        p4.append(x)


# ---------------- kiểm trùng train_v1 (A/B/C)
def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


rep = Counter(norm(t) for r in dev for t in [r["question"]] + list(r["options"]))
FIXED = {k for k, v in rep.items() if v >= 3}
probe_texts = defaultdict(set)                       # chuỗi chuẩn hóa -> id phép thử
grams = defaultdict(set)
for x in p3 + p4:
    for t in [x["state"], x["question"]] + list(x["options"]):
        if t and norm(t) not in FIXED:
            probe_texts[norm(t)].add(x["id"])
    q = claim_of(x["question"]) if x["source"] != "belebele" else x["question"]
    w = words(q)
    for i in range(len(w) - NGRAM + 1):
        grams[" ".join(w[i:i + NGRAM])].add(x["id"])
overlap = {}
for comp in ("A", "B", "C"):
    hit_exact, short, hit_gram, n = set(), Counter(), {}, 0
    with open(COMP / f"{comp}.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            n += 1
            for s in [t["state"], t["question"]] + list(t["options"]):
                if isinstance(s, str) and norm(s) in probe_texts:
                    if len(words(s)) >= 4:              # chuỗi ngắn hơn 4 từ (lựa chọn kiểu "fear", "Hoa Kỳ", "ba") không tính là rò rỉ
                        hit_exact |= probe_texts[norm(s)]
                    else:
                        short[norm(s)] += 1
            w = words(t["state"])
            for i in range(len(w) - NGRAM + 1):
                g = " ".join(w[i:i + NGRAM])
                if g in grams:
                    for pid in grams[g]:
                        hit_gram.setdefault(pid, {"dong_train": t["id"], "cum": g})
    overlap[comp] = {"file_sha256": sha256_file(COMP / f"{comp}.jsonl"), "rows": n, "trung_nguyen_van_tu_4_tu": len(hit_exact),
                     "vi_du_trung_nguyen_van": sorted(hit_exact)[:10],
                     "chuoi_ngan_duoi_4_tu_trung (không tính)": dict(short.most_common(15)), "so_chuoi_ngan_khac_nhau": len(short),
                     f"chua_{NGRAM}gram (đọc tay)": hit_gram}

# ---------------- ghi
OUT.mkdir(parents=True, exist_ok=True)
write_jsonl(OUT / "p3_bo_van_ban.jsonl", p3)
write_jsonl(OUT / "p4_van_ban_khong_lien_quan.jsonl", p4)


def chance(rows):
    """Mức đoán mò (trung bình 1/k) và tỉ lệ nếu luôn chọn đáp án hay gặp nhất (theo chuỗi đáp án) của từng nhóm."""
    g = defaultdict(list)
    for x in rows:
        g[f"{x['source']}|{x['lang']}|{x['variant']}|{x['kind']}"].append(x)
    out = {}
    for k, rs in sorted(g.items()):
        top = Counter(r["options"][r["target"]] for r in rs).most_common(1)[0]
        pos = Counter(r["target"] for r in rs).most_common(1)[0]
        out[k] = {"so_cau": len(rs), "doan_mo_1_tren_k": round(sum(1 / len(r["options"]) for r in rs) / len(rs), 4),
                  "luon_chon_vi_tri_hay_gap": {"vi_tri": pos[0], "ti_le": round(pos[1] / len(rs), 4)}}
        if rs[0]["source"] != "belebele":               # Belebele không có nhãn cố định: chỉ có ý nghĩa theo vị trí
            out[k]["luon_chon_dap_an_hay_gap"] = {"dap_an": top[0], "ti_le": round(top[1] / len(rs), 4)}
    return out


man = {
    "ten": "probe_v1: bộ phép thử chẩn đoán P3, P4 (P1, P5 chờ duyệt chữ)",
    "CANH_BAO": "BỘ PHÂN TÍCH, KHÔNG PHẢI BENCHMARK. KHÔNG BAO GIỜ DÙNG ĐỂ TRAIN (kể cả chọn checkpoint hay chỉnh siêu tham số). Chỉ dựng từ dev 1.3; không đụng test.",
    "brief": "results/BRIEF_phan_tich_xu_huong.md",
    "nguon": {"dev_v1_3.jsonl": DEV_SHA},
    "seed": SEED,
    "files": {f: sha256_file(OUT / f) for f in ("p3_bo_van_ban.jsonl", "p4_van_ban_khong_lien_quan.jsonl")},
    "P3": {"cach_dung": "state = '' (khung prompt vẫn in nhãn văn bản rồi để trống); câu hỏi, lựa chọn, đáp án giữ nguyên; id = id dev + ':P3'",
           "nguon": list(P3_SRC), "so_cau": dict(Counter(f"{x['source']}|{x['lang']}|{x['variant']}" for x in p3)), "muc_doan_mo": chance(p3)},
    "P4": {"cach_dung": "nhận định của mỗi câu ghép với state của câu khác cùng nguồn, khác bài; đáp án choice = 'Văn bản không đủ thông tin để kết luận', "
                        "noul = 'Không' (diễn giải của worker: noul không có nhãn không đủ thông tin; điều phối có thể bác)",
           "khac_bai": "vifactcheck: khác cluster (bài báo); viwikifc: khác mã bài Wikipedia (trường thứ 3 của id uit_*) và khác cluster",
           "loc": LOC, "cau_bi_bo_vi_het_ung_vien": {f"{s}|{k}": n for (s, k), n in sorted(drops.items())},
           "so_cau_goc_vao": dict(Counter(f"{r['source']}|{r['kind']}" for r in fc)),
           "so_cau": dict(Counter(f"{x['source']}|{x['variant']}|{x['kind']}" for x in p4)),
           "so_doan_duoc_ghep_khac_nhau": len({x["partner_pair_id"] for x in p4}),
           "dap_an_goc_cua_cau_trong_P4": dict(Counter(f"{x['source']}|{x['orig_gold_label']}" for x in p4 if x["variant"] == "orig")),
           "muc_doan_mo": chance(p4)},
    "trung_train_v1": {"cach_kiem": f"nguyên văn đã chuẩn hóa (trừ chuỗi cố định lặp >= 3 lần trong dev) và chứa {NGRAM}-gram của nhận định / câu hỏi Belebele trong state train",
                       **overlap},
}
man["P2_P6"] = "P2 (đảo thứ tự lựa chọn) = dev 1.3 chấm với eval_base.py --reverse; P6 (không dấu) = các dòng variant nodiac có sẵn trong dev 1.3. Không cần file riêng."
_old = OUT / "manifest_probe_v1.json"
if _old.exists():
    for _k, _v in json.loads(_old.read_text(encoding="utf-8")).items():
        man.setdefault(_k, _v)                        # mục do script khác ghi (P1_P5, Pa_Pb): giữ nguyên khi dựng lại P3/P4
(OUT / "manifest_probe_v1.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("files", "trung_train_v1")}, ensure_ascii=False, indent=1))
print(json.dumps(man["P4"]["cau_bi_bo_vi_het_ung_vien"], ensure_ascii=False), json.dumps(man["P4"]["so_cau"], ensure_ascii=False))
