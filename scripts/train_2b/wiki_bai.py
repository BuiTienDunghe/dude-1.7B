"""Bước 3 (lọc bài) + 4 (khử trùng mức bài) của results/BRIEF_2b_du_lieu.md trên Wikipedia tiếng Việt 20231101.vi. Chỉ CPU, không chạy model
(tokenizer Qwen3-1.7B của dự án chỉ để đếm token).

Lượt 1, mọi bài: đoạn văn xuôi (kc_util.prose_paragraphs: cắt ở mục chú thích / tham khảo, bỏ tiêu đề mục, danh sách, bảng); số chữ;
mẫu đoạn đầu = 2 câu đầu của đoạn văn xuôi đầu tiên, chữ thường, che số (0) và che chữ viết hoa đầu (X, gộp X liền nhau).
Bài đạt: >= 150 chữ văn xuôi, >= 3 đoạn, mẫu đoạn đầu không trùng với > 500 bài khác (luật loại bài sơ khai do bot: loài, xã, huyện…),
tiêu đề không trùng tiêu đề bài của ViWikiFC (mọi split; ViWikiFC dựng từ chính Wikipedia tiếng Việt).
Lượt 2, bài đạt theo thứ tự xáo cố định (seed 20261028): khử trùng từng đoạn và từng câu với mọi tham chiếu (kc_util.Refs), một câu chạm
thì bỏ cả bài; bài sạch thì đếm token từng đoạn bằng tokenizer Qwen3-1.7B (revision trong config/models.json). Dừng khi đủ bài sạch
cho hai dải độ dài (mặc định: 7.000 bài sạch, trong đó >= 2.600 bài có >= 900 token văn xuôi và >= 6 đoạn).
Ra results/train_2b/tam/bai_sach.jsonl (bài sạch: id, tiêu đề, đoạn, token từng đoạn) + tam/loc_bai.json (số bài bị loại theo lý do,
theo tham chiếu). Dùng: bash setup/run.sh -I scripts/train_2b/wiki_bai.py"""
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import RAW, ROOT, SEED, TMP, WIKI, Refs, h64, prose_paragraphs, sentences, write_jsonl  # noqa: E402

N_CLEAN, N_LONG, LONG_TOK, LONG_PAR = 7000, 2600, 900, 6
FILES = sorted(WIKI.glob("train-*.parquet"))
assert len(FILES) == 4


def template(pars):
    s = " ".join(sentences(pars[0])[:2]) if pars else ""
    s = re.sub(r"\d+(?:[.,]\d+)*", "0", s)
    s = " ".join("X" if w[:1].isupper() else w.lower() for w in s.split())
    return re.sub(r"(X )+X", "X", s)


def pass1(fi):
    import pyarrow.parquet as pq
    t = pq.read_table(FILES[fi], columns=["id", "title", "text"])
    out = []
    for i, (aid, title, text) in enumerate(zip(t["id"].to_pylist(), t["title"].to_pylist(), t["text"].to_pylist())):
        pars = prose_paragraphs(text)
        nw = sum(len(p.split()) for p in pars)
        out.append((fi, i, aid, title, len(pars), nw, h64(template(pars))))
    return out


REFS = None


def check_article(item):
    """(id, None) nếu sạch, hoặc (id, (loại, tag, phần))."""
    aid, pars = item
    for p in pars:
        h = REFS.hit(p)
        if h:
            return aid, (h[0], h[1], "doan")
        for s in sentences(p):
            h = REFS.hit(s)
            if h:
                return aid, (h[0], h[1], "cau")
    return aid, None


def main():
    global REFS
    t0 = time.time()
    TMP.mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool:
        meta = [x for part in pool.map(pass1, range(4)) for x in part]
    tpl = Counter(m[6] for m in meta)
    import pandas as pd
    wfc_titles = set()
    for sp in ("train", "dev", "test"):
        d = pd.read_csv(RAW / f"High-Will__ViWikiFC/data/{sp}.csv", dtype=str, keep_default_na=False)
        wfc_titles |= {" ".join(t.split()).lower() for t in d["title"]}
        wfc_titles |= {" ".join(l.rsplit("/", 1)[-1].replace("_", " ").split()).lower() for l in d["link"]}
    loai = Counter()
    ok = []
    for m in meta:
        fi, i, aid, title, npar, nw, th = m
        if nw < 150 or npar < 3:
            loai["ngan_duoi_150_chu_hoac_3_doan"] += 1
        elif tpl[th] > 501:
            loai["so_khai_bot_mau_doan_dau_trung_gt_500_bai"] += 1
        elif " ".join(title.split()).lower() in wfc_titles:
            loai["trung_tieu_de_viwikifc"] += 1
        else:
            ok.append(m)
    print(f"lượt 1: {len(meta)} bài, đạt {len(ok)}; {dict(loai)}; {time.time() - t0:.0f}s", flush=True)
    ok.sort(key=lambda m: h64(SEED, "bai", m[2]))
    REFS = Refs()
    print(f"tham chiếu: {REFS.n_by_tag}; {time.time() - t0:.0f}s", flush=True)

    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    rev = json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=rev)
    import pyarrow.parquet as pq
    tables = {}
    clean, cham, n_long, pos = [], Counter(), 0, 0
    by_ref = Counter()
    with Pool(14) as pool:
        while (len(clean) < N_CLEAN or n_long < N_LONG) and pos < len(ok):
            batch = ok[pos:pos + 3000]; pos += len(batch)
            items, info = [], {}
            for fi, i, aid, title, *_ in batch:
                if fi not in tables:
                    tables[fi] = pq.read_table(FILES[fi], columns=["text"])["text"]
                pars = prose_paragraphs(tables[fi][i].as_py())
                items.append((aid, pars)); info[aid] = (title, pars, fi, i)
            for aid, h in pool.imap(check_article, items, chunksize=20):
                if h:
                    cham[f"{h[0]}|{h[2]}"] += 1; by_ref[h[1]] += 1
                    continue
                title, pars, fi, i = info[aid]
                ntok = [len(x) for x in tok(pars, add_special_tokens=False)["input_ids"]]
                rec = {"id": aid, "title": title, "pars": pars, "ntok": ntok, "file": FILES[fi].name, "row": i}
                clean.append(rec)
                n_long += sum(ntok) >= LONG_TOK and len(pars) >= LONG_PAR
            print(f"đã xét {pos}: sạch {len(clean)} (dài {n_long}), chạm {sum(cham.values())}; {time.time() - t0:.0f}s", flush=True)
    write_jsonl(TMP / "bai_sach.jsonl", clean)
    res = {"tong_bai": len(meta), "dat_luot_1": len(ok), "loai_luot_1": dict(loai), "da_xet_khu_trung": pos,
           "bo_vi_cham": dict(cham), "bo_vi_cham_theo_tham_chieu": dict(by_ref), "bai_sach": len(clean), "bai_sach_du_dai": n_long,
           "so_van_ban_tham_chieu_theo_nguon": REFS.n_by_tag, "luat_so_khai": "mẫu 2 câu đầu đoạn đầu (che số, che chữ viết hoa) trùng > 500 bài khác",
           "giay": round(time.time() - t0)}
    (TMP / "loc_bai.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
