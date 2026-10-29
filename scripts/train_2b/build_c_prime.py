"""Mục 2 của results/BRIEF_2b_du_lieu.md: C′ = C (results/train_v1/components/C.jsonl, sha 005ca3d5…) sửa theo audit. Phần EN và multi giữ
nguyên; mọi việc chỉ đụng dòng text_lang == "vi". Không chạy model; 0 chữ tiếng Việt mới: câu hỏi khung và chữ nhãn chép TỪ DEV 1.3.

  a  bỏ nguồn không có văn bản: health_fact (choice, noul, score), super_glue/boolq (không đoạn văn; giữ boolq_passage), exams/vi (mhardalov/exams)
  b  bỏ VNTC (kornwtp/VNTC-10Topics)
  c  bỏ dòng teacher bất đồng tự tin (results/teacher_diag/quyet_C.jsonl: vi, có probs, agree false, p_max > 0,9, nguồn không thuộc
     vsmrc/race/quail/cosmos_qa/dream/exams); giữ dòng có nhãn gốc NEI / "trung lập"
  i  VSMRC: bỏ dòng có lựa chọn tổng hợp (trỏ chữ cái "Cả A và B"… hoặc trỏ vị trí "Tất cả các đáp án trên", "Không có đáp án nào đúng"…);
     bỏ dòng có đáp án dài hơn MỌI nhiễu quá 1,5 lần (đếm ký tự)
  d  vitaminc, nli_fever về khung bench: state = bằng chứng / tiền đề; câu hỏi = tiền tố kiểm chứng của dev + nhận định / giả thuyết;
     nhãn = 3 chuỗi dev. nli_fever: kéo theo → xác nhận, trung lập → NEI, mâu thuẫn → bác bỏ. Noul: chỉ xác nhận → "Có". Bỏ score,
     packed (Item A/B…), noul hỏi nhãn mà không suy ra chắc Có/Không. Đổi cơ học: bỏ khoảng trắng trước dấu câu.
  f  câu hỏi đọc hiểu dịch lên ô câu hỏi (choice): race tách dòng đầu; dream tách dòng cuối; quail tách dòng cuối nếu dòng đó chỉ là
     câu hỏi, không thì tách câu cuối (câu hỏi quail thường dính sau câu cuối của đoạn cuối); cosmos_qa tách câu cuối. Chỉ tách khi phần
     tách ra kết thúc bằng "?" (quail: cả ":", câu hỏi điền tiếp) hoặc có "_" (chỗ trống); còn lại giữ nguyên và đếm
  g  quail: lựa chọn "không đủ thông tin" → chuỗi NEI của dev
  h  tăng cường "từ chối" cho đọc hiểu: 10% dòng VI choice có đoạn văn, >= 3 lựa chọn, không có lựa chọn kiểu NEI / "không lựa chọn nào";
     75% thêm chuỗi NEI (đáp án giữ nguyên), 25% thay toàn bộ lựa chọn bằng lựa chọn của dòng đọc hiểu khác đoạn văn + chuỗi NEI (đáp án NEI);
     sửa tại chỗ, gắn aug=abstain_add / abstain_replace
  e  ghép lạc đề → NEI: 250 dòng choice, nhận định của dòng (d) + văn bản khác: nửa là bằng chứng của dòng (d) khác (không chung số / tên,
     trùng từ < 0,2), nửa là 1–3 đoạn Wikipedia của bài "cho" (results/train_2b/tam/bai_cho.jsonl, đã khử trùng, không dùng ở KC)
Ra results/train_2b/C_prime.jsonl + tam/c_prime_log.json (số dòng theo việc, dòng bị bỏ kèm lý do).
Dùng: bash setup/run.sh -I scripts/train_2b/build_c_prime.py"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_util import DEV, DEV_SHA, OUT, ROOT, SEED, TMP, h64, name_spans, num_spans, overlap, read_jsonl, u01, write_jsonl  # noqa: E402

C_PATH = ROOT / "results/train_v1/components/C.jsonl"
C_SHA = "005ca3d547aa2f9e2f356a8f75eb7f3f17addba1cc4d74e3a7b44a7b39c86e89"
assert hashlib.sha256(C_PATH.read_bytes()).hexdigest() == C_SHA
assert hashlib.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA
C = read_jsonl(C_PATH)
dev = read_jsonl(DEV)

# ------------------------------------------------------------------ chữ của dev 1.3 (như build_proxy_v1)
FC_OPT, PRE = {}, {}
for r in dev:
    if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] in ("vifactcheck", "viwikifc"):
        if r["kind"] == "choice" and r["gold_provenance"] == "source":
            FC_OPT[r["gold_label"]] = r["options"][r["target"]]
        mk = "Nhận định: " if r["kind"] == "choice" else "Phát biểu: "
        PRE.setdefault(r["kind"], set()).add(r["question"].split(mk, 1)[0] + mk)
assert set(FC_OPT) == {"supported", "refuted", "nei"} and all(len(v) == 1 for v in PRE.values())
PRE = {k: next(iter(v)) for k, v in PRE.items()}
NOUL = next(r["options"] for r in dev if r["lang"] == "vi" and r["variant"] == "orig" and r["source"] == "vifactcheck" and r["kind"] == "noul")
NEI = FC_OPT["nei"]
LABS = ["supported", "refuted", "nei"]

log = defaultdict(Counter)
dropped = []                      # (id, lý do)


def drop(r, why):
    dropped.append({"id": r["id"], "ly_do": why}); log["bo"][why] += 1


def onehot(k, i):
    return [1.0 if j == i else 0.0 for j in range(k)]


def flag(r, f):
    r["flags"] = list(r.get("flags", [])) + [f]


def fc_row(base, claim, evidence, label, kind, rid, extra_flag):
    """Dòng kiểm chứng đúng khung dev; choice: 3 nhãn dev xáo theo dòng; noul: Có iff xác nhận."""
    r = {k: base[k] for k in ("dataset", "source", "lang", "license_use", "split", "text_lang")}
    r.update({"id": rid, "kind": kind, "state": evidence, "variant": "khung_dev", "n_state_tokens": None, "gold_label": label})
    if kind == "choice":
        perm = sorted(range(3), key=lambda j: h64(SEED, "perm", rid, j))
        r["options"] = [FC_OPT[LABS[p]] for p in perm]
        r["question"] = PRE["choice"] + claim
        t = perm.index(LABS.index(label))
    else:
        r["options"] = list(NOUL); r["question"] = PRE["noul"] + claim
        t = 0 if label == "supported" else 1
    r["target"], r["target_argmax"] = onehot(len(r["options"]), t), t
    r["flags"] = [extra_flag]
    return r


def tidy(s):
    """Đổi cơ học: bỏ khoảng trắng trước dấu câu / sau ngoặc mở, gộp khoảng trắng (văn bản dịch giữ dấu cách kiểu token hoá)."""
    s = " ".join(s.split())
    s = re.sub(r"\s+([.,;:!?%)\]”])", r"\1", s)
    s = re.sub(r"([(\[“])\s+", r"\1", s)
    return s.strip()


# ------------------------------------------------------------------ a, b, c
quyet = {q["id"]: q for q in read_jsonl(ROOT / "results/teacher_diag/quyet_C.jsonl")}
RC_KEYS = ("vsmrc", "race", "quail", "cosmos_qa", "dream", "exams")
keep = []
for r in C:
    if r["text_lang"] != "vi":
        keep.append(r); continue
    if r["source"] in ("health_fact", "super_glue/boolq", "exams/vi"):
        drop(r, f"a_khong_van_ban:{r['source']}"); continue
    if r["source"] == "vntc":
        drop(r, "b_vntc"); continue
    q = quyet.get(r["id"])
    if (q and q.get("probs") is not None and not q["agree"] and q["p_max"] > 0.9
            and not any(k in (q["source"] or "").lower() for k in RC_KEYS)):
        gold = r["options"][r["target_argmax"]].strip().lower()
        if gold in ("trung lập", "văn bản không đủ thông tin", NEI.lower()):
            log["c"]["giu_ngoai_le_nei_trung_lap"] += 1
        else:
            drop(r, "c_teacher_bat_dong_tu_tin"); continue
    keep.append(r)
q_all = [q for q in quyet.values() if q["text_lang"] == "vi" and q.get("probs") is not None and not q["agree"] and q["p_max"] > 0.9
         and not any(k in (q["source"] or "").lower() for k in RC_KEYS)]
log["c"]["tong_theo_quy_tac"] = len(q_all)

# ------------------------------------------------------------------ i: VSMRC
LETTER = re.compile(r"^(vì\s+)?(cả\s+(hai\s+)?(đáp án\s+|phương án\s+)?|đáp án\s+)?[A-D]\s+(và|,|hoặc)\s+[A-D]\b", re.I)
ABOVE = re.compile(r"^(tất cả|cả hai|cả ba|cả bốn)\s+(các\s+)?(đáp án|lựa chọn|phương án|ý|câu|yếu tố|điều)\b.*\btrên\b|"
                   r"^(tất cả|cả hai|cả ba)\s+(các\s+)?(đáp án|lựa chọn|phương án|ý)(\s+(đều\s+)?(đúng|sai))?\.?$|"
                   r"^không có\s+(đáp án|phương án|lựa chọn)\s+(nào\s+)?(đúng|ở trên)", re.I)
k2 = []
for r in keep:
    if r["text_lang"] == "vi" and r["source"] == "vsmrc/mrc":
        opts = [o.strip() for o in r["options"]]
        if any(LETTER.search(o) for o in opts):
            drop(r, "i_vsmrc_tong_hop_chu_cai"); continue
        if any(ABOVE.search(o) for o in opts):
            drop(r, "i_vsmrc_tong_hop_vi_tri"); continue
        g = opts[r["target_argmax"]]
        if len(opts) > 1 and all(len(g) > 1.5 * len(o) for i, o in enumerate(opts) if i != r["target_argmax"]):
            drop(r, "i_vsmrc_dap_an_dai_gt_1_5_lan"); continue
    k2.append(r)
keep = k2

# ------------------------------------------------------------------ d: vitaminc, nli_fever
VC_LAB = {"được văn bản xác nhận": "supported", "bị văn bản bác bỏ": "refuted", "văn bản không đủ thông tin": "nei"}
NF_LAB = {"kéo theo": "supported", "trung lập": "nei", "mâu thuẫn": "refuted"}
FORMATS = [re.compile(r"^Văn bản thứ nhất:\n(.*?)\n\nVăn bản thứ hai:\n(.*)$", re.S), re.compile(r"^Passage A:\n(.*?)\n\nPassage B:\n(.*)$", re.S),
           re.compile(r"^text_A:\s*(.*?)\ntext_B:\s*(.*)$", re.S), re.compile(r"^A:\s*(.*?)\nB:\s*(.*)$", re.S)]


def split_pair(state):
    if state.startswith("Item A:"):
        return None
    for f in FORMATS:
        m = f.match(state)
        if m and "\n" not in m.group(1).strip() and "\n" not in m.group(2).strip():
            return m.group(1).strip(), m.group(2).strip()
    return None


d_rows = []
k2 = []
for r in keep:
    if r["text_lang"] != "vi" or r["source"] not in ("vitaminc", "nli_fever"):
        k2.append(r); continue
    src = r["source"]; lab = VC_LAB if src == "vitaminc" else NF_LAB
    if r["kind"] == "score":
        drop(r, "d_score"); continue
    pr = split_pair(r["state"])
    if pr is None:
        drop(r, "d_packed_hoac_khong_tach_duoc"); continue
    claim, ev = (pr[0], pr[1]) if src == "vitaminc" else (pr[1], pr[0])
    if r["kind"] == "choice":
        g = lab.get(r["options"][r["target_argmax"]].strip().lower())
        if g is None or not all(o.strip().lower() in lab for o in r["options"]):
            drop(r, "d_nhan_la"); continue
    else:
        m = re.search(r"Có phải \"([^\"]+)\" là (?:nhãn|câu trả lời) đúng", r["question"])
        x = lab.get(m.group(1).strip().lower()) if m else None
        if x is None:
            drop(r, "d_noul_khong_doc_duoc_nhan"); continue
        yes = r["target_argmax"] == 0
        if yes:
            g = x
        elif x == "supported":
            g = "khong_xac_nhan"
        else:
            drop(r, "d_noul_khong_suy_ra_chac"); continue
    claim, ev = tidy(claim), tidy(ev)
    if len(claim.split()) < 3 or len(ev.split()) < 5:
        drop(r, "d_qua_ngan"); continue
    label = g if g != "khong_xac_nhan" else "refuted_or_nei"
    nr = fc_row(r, claim, ev, "refuted" if label == "refuted_or_nei" else label, r["kind"], r["id"] + ":2d", "2d")
    if label == "refuted_or_nei":
        nr["gold_label"] = "khong_xac_nhan"          # noul "Không": bác bỏ hoặc NEI, không phân biệt
    nr["claim_len_gt_evidence"] = len(claim) > len(ev)
    log["d"][f"{src}|{r['kind']}|{nr['gold_label']}"] += 1
    d_rows.append(nr); k2.append(nr)
keep = k2
log["d"]["nhan_dinh_dai_hon_bang_chung"] = sum(r.pop("claim_len_gt_evidence") for r in d_rows)

# ------------------------------------------------------------------ f, g
RACE = ("race/high", "race/middle")


def split_q(r):
    s = r["state"].strip()
    if r["source"] in RACE:
        if "\n" not in s:
            return None
        q, rest = s.split("\n", 1)
    elif r["source"] == "dream" and "\n" in s:
        rest, q = s.rsplit("\n", 1)
    elif r["source"] == "quail":
        # câu hỏi quail là dòng cuối nếu dòng đó chỉ có câu hỏi; thường câu hỏi dính sau câu cuối của đoạn cuối: tách câu cuối.
        # Câu hỏi quail có thể kết thúc bằng "?" hoặc ":" (câu hỏi dạng điền tiếp: "…, tác giả có lẽ:").
        last = s.rsplit("\n", 1)[-1].strip()
        if "\n" in s and len(last.split()) <= 40 and re.search(r"[?:][\"”]?$", last) and not re.search(r"[.!…][\"”]?\s", last):
            rest, q = s.rsplit("\n", 1)
        else:
            m = re.search(r"^(.*[.!?…][\"”]?)\s+([^\n]*?[?:][\"”]?)\s*$", s, re.S)
            if not m:
                return None
            rest, q = m.group(1), m.group(2)
    else:                                            # cosmos_qa: câu cuối
        m = re.search(r"^(.*[.!?…\"”])\s+([^.!?…]*\?)\s*$", s, re.S)
        if not m:
            return None
        rest, q = m.group(1), m.group(2)
    q, rest = q.strip(), rest.strip()
    endq = ("?", "?\"", "?”", ":") if r["source"] == "quail" else ("?",)
    if not rest or not (q.endswith(endq) or "_" in q) or len(q.split()) > 60:
        return None
    return q, rest


for r in keep:
    if r["text_lang"] != "vi" or r["kind"] != "choice":
        continue
    if r["source"] in RACE + ("dream", "quail", "cosmos_qa"):
        sp = split_q(r)
        if sp is None:
            log["f"][f"{r['source']}|khong_tach_chac"] += 1
        else:
            r["question_goc"] = r["question"]; r["question"], r["state"] = sp; flag(r, "2f")
            log["f"][f"{r['source']}|tach"] += 1
    if r["source"] == "quail":
        n = [i for i, o in enumerate(r["options"]) if o.strip().lower() == "không đủ thông tin"]
        if n:
            r["options"] = [NEI if i in n else o for i, o in enumerate(r["options"])]; flag(r, "2g"); log["g"]["doi_chu_nei"] += 1

# ------------------------------------------------------------------ h
RC_SRC = ("vsmrc/mrc",) + RACE + ("dream", "cosmos_qa", "quail")
NOTA = re.compile(r"không (lựa chọn|đáp án|phương án|câu trả lời) nào|không có thông tin|không đủ thông tin|không thể xác định|"
                  r"không được (nêu|đề cập)|tất cả các", re.I)


def has_nei_like(r):
    return any(NOTA.search(o) for o in r["options"])


elig = [r for r in keep if r["text_lang"] == "vi" and r["kind"] == "choice" and r["source"] in RC_SRC and r["state"].strip()
        and len(r["options"]) >= 3]
log["h"]["du_dieu_kien_truoc_loc_nei"] = len(elig)
elig = [r for r in elig if not has_nei_like(r)]
log["h"]["du_dieu_kien"] = len(elig)
donors = sorted(elig, key=lambda r: h64(SEED, "2h_cho", r["id"]))
for r in elig:
    if u01(SEED, "2h", r["id"]) >= 0.10:
        continue
    if u01(SEED, "2h_kieu", r["id"]) < 0.75:
        r["options"] = r["options"] + [NEI]
        r["target"] = list(r["target"]) + [0.0]
        r["aug"] = "abstain_add"; flag(r, "aug=abstain_add"); log["h"]["abstain_add"] += 1
    else:
        st = r["state"].strip()
        start = h64(SEED, "2h_vt", r["id"]) % len(donors)
        dn = next(d for d in donors[start:] + donors[:start] if d["id"] != r["id"] and d["state"].strip() != st
                  and "aug" not in d and len(d["options"]) + 1 <= 10)
        opts = [o for o in dn["options"]] + [NEI]
        r["options_goc"] = r["options"]; r["options"] = opts
        r["target"] = [0.0] * (len(opts) - 1) + [1.0]
        r["aug"] = "abstain_replace"; r["aug_cho"] = dn["id"]; flag(r, "aug=abstain_replace"); log["h"]["abstain_replace"] += 1
    r["target_argmax"] = max(range(len(r["target"])), key=lambda j: r["target"][j])

# ------------------------------------------------------------------ e
donor_wiki = read_jsonl(TMP / "bai_cho.jsonl")


def keys_of(s):
    return {v for _, _, v, _ in num_spans(s)} | {n for _, _, n, _ in name_spans(s)}


cand = sorted(d_rows, key=lambda r: h64(SEED, "2e", r["id"]))
e_rows, used_claims = [], set()
N_E = 250
wi = 0
for i, r in enumerate(cand):
    if len(e_rows) >= N_E:
        break
    claim = r["question"].split("Nhận định: " if r["kind"] == "choice" else "Phát biểu: ", 1)[1]
    if claim in used_claims:
        continue
    use_wiki = len(e_rows) % 2 == 1
    text, src_tag = None, None
    if use_wiki:
        for _ in range(len(donor_wiki)):
            b = donor_wiki[wi % len(donor_wiki)]; wi += 1
            if overlap(claim, b["text"]) < 0.2 and not (keys_of(claim) & keys_of(b["text"])):
                text, src_tag = b["text"], f"wiki:{b['id']}"
                break
    else:
        for j in range(1, len(cand)):
            o = cand[(i + j * 7) % len(cand)]
            if o["id"] == r["id"] or o["state"] == r["state"]:
                continue
            if overlap(claim, o["state"]) < 0.2 and not (keys_of(claim) & keys_of(o["state"])):
                text, src_tag = o["state"], f"c2d:{o['id']}"
                break
    if text is None:
        log["e"]["khong_tim_duoc_van_ban"] += 1; continue
    used_claims.add(claim)
    nr = fc_row(r, claim, text, "nei", "choice", r["id"].replace(":2d", "") + ":2e", "2e")
    nr["source"] = r["source"]; nr["van_ban_tu"] = src_tag
    if src_tag.startswith("wiki:"):
        nr["text_source"] = "wikipedia_vi_20231101"; nr["license"] = "CC BY-SA 4.0"
    e_rows.append(nr); log["e"]["wiki" if use_wiki else "c2d"] += 1
keep += e_rows

import os  # noqa: E402
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"])
for r in keep:                                       # state đổi ở 2d, 2e, 2f: đếm lại token
    if {"2d", "2e", "2f"} & set(r.get("flags", [])):
        r["n_state_tokens"] = len(TOK(r["state"], add_special_tokens=False)["input_ids"])
    if r["text_lang"] == "vi":
        log["c_prime_vi"][f"{r['source']}|{r['kind']}"] += 1
assert len({r["id"] for r in keep}) == len(keep)
write_jsonl(OUT / "C_prime.jsonl", keep)
res = {"dau_vao": str(C_PATH.relative_to(ROOT)), "sha256": C_SHA, "so_dong_C": len(C), "so_dong_C_prime": len(keep),
       "chu_tu_dev": {"tien_to_choice": PRE["choice"], "tien_to_noul": PRE["noul"], "nhan": FC_OPT, "noul": NOUL},
       "anh_xa_nli_fever": {"kéo theo": "xác nhận", "trung lập": "NEI", "mâu thuẫn": "bác bỏ"},
       "log": {k: dict(v) for k, v in log.items()}, "dong_bi_bo": dropped}
(TMP / "c_prime_log.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: dict(v) for k, v in log.items() if k != "c_prime_vi"}, ensure_ascii=False, indent=1))
