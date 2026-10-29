"""Mục 3 của results/BRIEF_2b_du_lieu.md: khối KC-Wiki, kiểm chứng đúng khung bench dựng bằng chương trình từ bài Wikipedia tiếng Việt đã lọc và
khử trùng (results/train_2b/tam/bai_sach.jsonl, do wiki_bai.py). Không chạy model (tokenizer Qwen3-1.7B chỉ để đếm token). 0 chữ tiếng Việt
mới: nhận định là câu trích từ bài hoặc câu trích đã đổi cơ học; câu hỏi, lựa chọn chép từ dev 1.3.

Mỗi bài cho tối đa một dòng (một dải độ dài). Văn bản (state) = các đoạn văn xuôi liền nhau nối bằng dấu cách (đúng định dạng state của
dev ViWikiFC / ViFactCheck: không xuống dòng):
  KC-ngắn: 1–3 đoạn, 80–400 token;  KC-dài: nhiều đoạn, 500–1.300 token, độ dài đích lấy log-chuẩn quanh 750.
Kiểu dựng nhận định (mọi nhận định 8–45 chữ, có >= 1 số hoặc tên riêng, để ba nhãn cùng hình dạng):
  S-nguyen  Xác nhận  câu trong văn bản, trích nguyên
  S-luoc    Xác nhận  câu trong văn bản, bỏ phần trong ngoặc, hoặc bỏ mệnh đề sau dấu phẩy cuối (còn >= 8 chữ)
  R-so      Bác bỏ    câu trong văn bản, đổi một số / năm / ngày / tháng bằng giá trị cùng loại lấy từ cùng bài (không có trong câu)
  R-ten     Bác bỏ    câu trong văn bản, đổi một tên riêng (>= 2 âm tiết viết hoa, không đứng đầu câu) bằng tên cùng số âm tiết của cùng bài;
                      chỉ khi bài có >= 2 ứng viên cùng dạng
  N-cungbai NEI       câu ở phần khác của cùng bài, cách văn bản >= 2 đoạn, chứa >= 1 số / tên không có trong văn bản, trùng từ với mọi
                      câu của văn bản < 0,5
  N-lacde   NEI       câu của bài khác (cùng phần train / val), trùng từ với mọi câu của văn bản < 0,5
Kiểu lược (bỏ ngoặc / bỏ mệnh đề) áp cùng xác suất 0,4 cho cả ba nhãn. Không dùng phủ định.
Tỉ lệ rút: choice Xác nhận 29 / Bác bỏ 29 / NEI 42 (NEI: cùng bài 85, lạc đề 15); noul Xác nhận 37 / Bác bỏ 28 / NEI 35. Rút cao hơn đích
(30/30/40; "Có" 37–40%) một chút vì bài không dựng được NEI cùng bài / Bác bỏ thì bị bỏ, làm tỉ lệ thực lệch về Xác nhận.
choice / noul theo seed từng dòng 55/45. 20% dòng (chính dòng đó) chuyển không dấu toàn bộ bằng common.strip_diacritics (hàm của
build_bench_v1), gắn cờ nodiac. Mỗi dòng (state đã nối, nhận định) và mỗi đoạn "cho" được khử trùng lại với mọi tham chiếu
(kc_util.Refs): chỗ nối hai đoạn có thể tạo chuỗi chạm mà bước lọc từng đoạn không thấy; chạm thì bỏ cửa sổ đó. 5% số bài (theo hash id) làm val_kc.jsonl (<= 300 dòng); không bài nào ở cả hai phần.
Ra results/train_2b/{kc_pool_ngan.jsonl, kc_pool_dai.jsonl, val_kc.jsonl} (thứ tự trong pool đã xáo theo seed: lấy N dòng đầu là mẫu
ngẫu nhiên) + tam/bai_cho.jsonl (đoạn văn của bài "cho" cho C′ mục 2e) + tam/kc_log.json.
Dùng: bash setup/run.sh -I scripts/train_2b/build_kc.py"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import (DEV, OUT, ROOT, SEED, TMP, Refs, h64, luoc, name_spans, normalize, num_spans, overlap, read_jsonl, sentences, u01,  # noqa: E402
                     write_jsonl)

sys.path.insert(0, str(ROOT / "scripts"))
from common import strip_diacritics  # noqa: E402

N_SHORT, N_LONG, N_DONOR, VAL_FRAC, VAL_MAX = 2500, 1600, 400, 0.05, 300
P_LUOC = 0.4
LAB_P = {"choice": [("supported", 0.29), ("refuted", 0.29), ("nei", 0.42)], "noul": [("supported", 0.37), ("refuted", 0.28), ("nei", 0.35)]}
P_LACDE, P_RSO = 0.15, 0.6
LABS = ["supported", "refuted", "nei"]

os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"])


REFS = Refs()


def touched(state, claim):
    """Khử trùng lại dòng đã dựng (state nối nhiều đoạn có thể tạo chuỗi chạm ở chỗ nối; nhận định đã lược / đổi)."""
    for t in [state, claim] + sentences(state):
        h = REFS.hit(t)
        if h:
            return h
    return None


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


# ------------------------------------------------------------------ chữ dev
dev = read_jsonl(DEV)
FC_OPT, PRE = {}, {}
for r in dev:
    if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] in ("vifactcheck", "viwikifc"):
        if r["kind"] == "choice" and r["gold_provenance"] == "source":
            FC_OPT[r["gold_label"]] = r["options"][r["target"]]
        mk = "Nhận định: " if r["kind"] == "choice" else "Phát biểu: "
        PRE.setdefault(r["kind"], set()).add(r["question"].split(mk, 1)[0] + mk)
PRE = {k: next(iter(v)) for k, v in PRE.items()}
NOUL = next(r["options"] for r in dev if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] == "vifactcheck" and r["kind"] == "noul")
DEV_STR = {s for r in dev if r["source"] in ("vifactcheck", "viwikifc") for s in r["options"]}
DEV_PRE = {r["question"].split(mk, 1)[0] + mk for r in dev if r["source"] in ("vifactcheck", "viwikifc")
           for mk in ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ") if mk in r["question"]}

# ------------------------------------------------------------------ bài
arts = read_jsonl(TMP / "bai_sach.jsonl")
for a in arts:
    a["sents"] = [sentences(p) for p in a["pars"]]
    a["split"] = "val" if u01(SEED, "val", a["id"]) < VAL_FRAC else "train"
arts.sort(key=lambda a: h64(SEED, "kc", a["id"]))
by_split = defaultdict(list)
for a in arts:
    by_split[a["split"]].append(a)
log = defaultdict(Counter)


def ok_claim(s):
    n = len(s.split())
    return 8 <= n <= 45 and re.search(r"[.!?…][\"”’')\]]*$", s) and "\n" not in s and (num_spans(s) or name_spans(s))


def keys(s):
    return [v for _, _, v, _ in num_spans(s)] + [n for _, _, n, _ in name_spans(s)]


def maybe_luoc(s, want):
    """(câu, cách lược) theo yêu cầu; None nếu yêu cầu lược mà không lược được."""
    if not want:
        return s, None
    x = luoc(s)
    return (x[0], x[1]) if x else None


def windows(a, band):
    """Các cửa sổ (p0, p1) hợp dải, thứ tự theo seed của bài."""
    nt = a["ntok"]; P = len(nt); out = []
    if band == "ngan":
        for p0 in range(P):
            for L in (1, 2, 3):
                if p0 + L <= P and 80 <= sum(nt[p0:p0 + L]) + L <= 400:
                    out.append((p0, p0 + L - 1))
    else:
        tgt = min(1300, max(500, math.exp(math.log(750) + 0.25 * _gauss(a["id"]))))
        for p0 in range(P):
            s, p1 = 0, p0
            while p1 < P and s < tgt:
                s += nt[p1] + 1; p1 += 1
            if 500 <= s <= 1300 and s >= tgt:
                out.append((p0, p1 - 1))
    out.sort(key=lambda w: h64(SEED, "win", a["id"], band, w))
    return out


def _gauss(aid):
    u1, u2 = max(1e-12, u01(SEED, "g1", aid)), u01(SEED, "g2", aid)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def pick_label(kind, rid):
    u = u01(SEED, "lab", rid); acc = 0
    for lab, p in LAB_P[kind]:
        acc += p
        if u < acc:
            return lab
    return LAB_P[kind][-1][0]


def article_values(a):
    nums, names = defaultdict(set), defaultdict(set)
    for ss in a["sents"]:
        for s in ss:
            for _, _, v, t in num_spans(s):
                nums[t].add(v)
            for _, _, n, k in name_spans(s):
                names[k].add(n)
    return nums, names


def build_claim(a, p0, p1, kind_lab, want_luoc, rid, pool_split):
    """(nhận định, kiểu, cách lược, câu gốc, thông tin thêm) hoặc None."""
    win_sents = [s for ss in a["sents"][p0:p1 + 1] for s in ss]
    win_text = normalize(" ".join(a["pars"][p0:p1 + 1]))
    rng_order = lambda xs, tag: sorted(xs, key=lambda x: h64(SEED, tag, rid, x))
    if kind_lab == "S":
        for s in rng_order(win_sents, "S"):
            m = maybe_luoc(s, want_luoc)
            if m and ok_claim(m[0]):
                return m[0], "S-luoc" if m[1] else "S-nguyen", m[1], s, {}
        return None
    if kind_lab in ("R-so", "R-ten"):
        nums, names = article_values(a)
        for s in rng_order(win_sents, kind_lab):
            m = maybe_luoc(s, want_luoc)
            if not m or not ok_claim(m[0]):
                continue
            c = m[0]
            if kind_lab == "R-so":
                spans = num_spans(c); have = {v for _, _, v, _ in spans}
                for st, en, v, t in rng_order(spans, "rs_span"):
                    cands = sorted(x for x in nums[t] if x not in have and x != v and x not in c)
                    if not cands:
                        continue
                    nv = cands[h64(SEED, "rs_val", rid) % len(cands)]
                    return c[:st] + nv + c[en:], "R-so", m[1], s, {"doi": [v, nv], "loai_so": t}
            else:
                spans = name_spans(c)
                for st, en, n, k in rng_order(spans, "rt_span"):
                    pool = names[k]
                    if len(pool) < 2:
                        continue
                    cands = sorted(x for x in pool if x != n and x not in c and n not in x and x not in n)
                    if not cands:
                        continue
                    nv = cands[h64(SEED, "rt_val", rid) % len(cands)]
                    return c[:st] + nv + c[en:], "R-ten", m[1], s, {"doi": [n, nv]}
        return None
    if kind_lab == "N-cungbai":
        P = len(a["pars"])
        far = [s for i in range(P) if i <= p0 - 2 or i >= p1 + 2 for s in a["sents"][i]]
        for s in rng_order(far, "Nc"):
            m = maybe_luoc(s, want_luoc)
            if not m or not ok_claim(m[0]):
                continue
            new = [k for k in keys(m[0]) if normalize(k) and normalize(k) not in win_text]
            if new and all(overlap(m[0], w) < 0.5 for w in win_sents):
                return m[0], "N-cungbai", m[1], s, {"moi": new[:3]}
        return None
    # N-lacde
    others = by_split[pool_split]
    start = h64(SEED, "Nl", rid) % len(others)
    for j in range(60):
        b = others[(start + j * 13) % len(others)]
        if b["id"] == a["id"]:
            continue
        ss = [s for x in b["sents"] for s in x]
        for s in rng_order(ss, "Nl_s")[:20]:
            m = maybe_luoc(s, want_luoc)
            if m and ok_claim(m[0]) and all(overlap(m[0], w) < 0.5 for w in win_sents):
                return m[0], "N-lacde", m[1], s, {"bai_khac": b["id"], "tieu_de_khac": b["title"]}
    return None


def make_row(a, band, pool_split):
    rid = f"kc:{band}:{a['id']}"
    kind = "choice" if u01(SEED, "kind", rid) < 0.55 else "noul"
    label = pick_label(kind, rid)
    if label == "supported":
        kinds = ["S"]
    elif label == "refuted":
        kinds = ["R-so", "R-ten"] if u01(SEED, "rk", rid) < P_RSO else ["R-ten", "R-so"]
    else:
        kinds = ["N-lacde"] if u01(SEED, "nk", rid) < P_LACDE else ["N-cungbai"]
    want_luoc = u01(SEED, "luoc", rid) < P_LUOC
    for w in windows(a, band)[:8]:
        for kl in kinds:
            got = build_claim(a, w[0], w[1], kl, want_luoc, rid, pool_split)
            if got:
                break
        else:
            continue
        claim, kieu, cach, goc, extra = got
        state = " ".join(a["pars"][w[0]:w[1] + 1])
        h = touched(state, claim)
        if h:
            log["bo_cua_so_cham"][f"{h[0]}|{h[1]}"] += 1
            continue
        nst = ntok(state)
        if not ((80 <= nst <= 400) if band == "ngan" else (500 <= nst <= 1300)):
            log["bo_cua_so"]["token_ngoai_dai"] += 1
            continue
        if kind == "choice":
            perm = sorted(range(3), key=lambda j: h64(SEED, "perm", rid, j))
            options = [FC_OPT[LABS[p]] for p in perm]; t = perm.index(LABS.index(label)); q = PRE["choice"] + claim
        else:
            options = list(NOUL); t = 0 if label == "supported" else 1; q = PRE["noul"] + claim
        r = {"id": rid, "dataset": "wikimedia/wikipedia:20231101.vi", "source": f"kc_wiki_{band}", "kind": kind, "lang": "vi", "text_lang": "vi",
             "license_use": "CC BY-SA 4.0 (chỉ dùng nội bộ)", "split": pool_split, "state": state, "question": q, "options": options,
             "target": [1.0 if j == t else 0.0 for j in range(len(options))], "target_argmax": t, "n_state_tokens": nst, "variant": "program",
             "flags": [], "gold_label": label, "label_source": "program", "text_source": "wikipedia_vi_20231101", "license": "CC BY-SA 4.0",
             "kieu": kieu, "luoc": cach, "tieu_de": a["title"], "wiki_id": a["id"], "doan": [w[0], w[1]], "cau_goc": goc, "seed": SEED,
             "claim_ntok": ntok(claim), **extra}
        if u01(SEED, "nodiac", rid) < 0.20:
            r["state"], r["question"] = strip_diacritics(r["state"]), strip_diacritics(r["question"])
            r["options"] = [strip_diacritics(o) for o in r["options"]]
            r["flags"] = ["nodiac"]; r["n_state_tokens"] = ntok(r["state"])
        assert all(o in DEV_STR for o in r["options"]) and any(r["question"].startswith(p) for p in DEV_PRE), r["id"]
        log["kieu"][f"{band}|{pool_split}|{kind}|{kieu}"] += 1
        return r
    log["bai_khong_dung_duoc"][f"{band}|{kinds[0]}|luoc={want_luoc}"] += 1
    return None


pools = {("train", "ngan"): [], ("train", "dai"): [], ("val", "ngan"): [], ("val", "dai"): []}
cap = {("train", "ngan"): N_SHORT, ("train", "dai"): N_LONG, ("val", "ngan"): VAL_MAX * 3 // 5, ("val", "dai"): VAL_MAX * 2 // 5}
donor, used = [], set()
for a in arts:
    sp = a["split"]
    long_ok = sum(a["ntok"]) >= 900 and len(a["pars"]) >= 6
    order = (["dai", "ngan"] if long_ok else ["ngan"])
    done = False
    for band in order:
        if len(pools[(sp, band)]) >= cap[(sp, band)]:
            continue
        r = make_row(a, band, sp)
        if r:
            pools[(sp, band)].append(r); used.add(a["id"]); done = True
            break
    if not done and sp == "train" and len(donor) < N_DONOR and a["id"] not in used:
        ws = windows(a, "ngan")
        if ws:
            p0, p1 = ws[0]
            text = " ".join(a["pars"][p0:p1 + 1])
            if not touched(text, ""):
                donor.append({"id": a["id"], "title": a["title"], "text": text, "doan": [p0, p1]})
                used.add(a["id"])
    if all(len(pools[k]) >= cap[k] for k in pools) and len(donor) >= N_DONOR:
        break

for (sp, band), rows in pools.items():
    log["so_dong"][f"{sp}|{band}"] = len(rows)
write_jsonl(OUT / "kc_pool_ngan.jsonl", pools[("train", "ngan")])
write_jsonl(OUT / "kc_pool_dai.jsonl", pools[("train", "dai")])
val = pools[("val", "ngan")] + pools[("val", "dai")]
write_jsonl(OUT / "val_kc.jsonl", val)
write_jsonl(TMP / "bai_cho.jsonl", donor)
tr_ids = {r["wiki_id"] for k, v in pools.items() if k[0] == "train" for r in v} | {d["id"] for d in donor}
va_ids = {r["wiki_id"] for r in val}
assert not tr_ids & va_ids
for k, v in pools.items():                       # N-lacde: bài khác cùng phần train / val
    for r in v:
        if r["kieu"] == "N-lacde":
            assert r["bai_khac"] not in (va_ids if k[0] == "train" else tr_ids)
res = {"so_bai_sach": len(arts), "so_bai_val_theo_hash": sum(a["split"] == "val" for a in arts), "bai_cho_2e": len(donor),
       "log": {k: dict(v) for k, v in log.items()}, "p_luoc": P_LUOC, "ti_le_nhan": LAB_P, "p_lacde_trong_nei": P_LACDE, "p_rso_trong_bac_bo": P_RSO}
(TMP / "kc_log.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
