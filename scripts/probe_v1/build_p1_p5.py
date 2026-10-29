"""Dựng P1′ "đổi chữ nhãn" và P5 "đổi câu hỏi" của probe_v1 từ chữ NGƯỜI DÙNG ĐÃ DUYỆT, theo bộ phép thử chốt lại ngày 26/10
(mục "Cập nhật 26/10" của results/BRIEF_phan_tich_xu_huong.md). BỘ PHÂN TÍCH, KHÔNG PHẢI BENCHMARK, KHÔNG BAO GIỜ DÙNG ĐỂ TRAIN.
Chỉ dùng dev 1.3; không đọc test. Chỉ thư viện chuẩn.

Giá trị đã duyệt, theo khoa (vd. probe_v1.P5.sib200), lấy LẦN DUYỆT MỚI NHẤT, bỏ qua lần từ chối:
  - data/human_edits/edits_log.jsonl: với mỗi đề xuất (de_xuat_id), bản ghi de_xuat mới nhất theo thoi_diem (cùng thời điểm thì bản ghi sau);
    là duyet / sua_roi_duyet thì tính là một lần duyệt (gia_tri_moi, thời điểm của bản ghi); tu_choi / hoan_tac thì không.
  - results/probe_v1/fable_soat/quyet_dinh_nguoi_dung.json và results/probe_v1/quyet_dinh_chat_pv1b.json (nếu có): quyết định người dùng
    trong chat 26/10, điều phối ghi lại; mục được duyệt = duyệt nguyên chữ gia_tri_de_xuat của đề xuất đó, thời điểm = thoi_diem của
    đề xuất (nên mọi lần duyệt sau đó trên công cụ đều mới hơn).
  Khoa probe_v1.P5.sib200 / .massive có hai đề xuất (pv1-* bị bỏ, pv1b-* chờ duyệt): chỉ lần duyệt được tính.
P1′ (p1b_doi_chu_nhan.jsonl): mọi dòng VI (có dấu, không dấu) của ViFactCheck, ViWikiFC kiểu choice: câu hỏi = câu P5 kiểm chứng choice đã duyệt
  ({claim} = nhận định của chính câu), lựa chọn = chữ đã duyệt của probe_v1.P1b.kiem_chung.{supported,refuted,nei}; thứ tự và chỉ số đáp án giữ nguyên.
  Chênh P1′ − P5 trên cùng câu là phần do đổi chữ nhãn.
P5 (p5_doi_cau_hoi.jsonl): 4 khung kiem_chung_choice, kiem_chung_noul (ViFactCheck, ViWikiFC), sib200, massive; mọi dòng VI có dấu và không dấu;
  thay câu hỏi, giữ lựa chọn. Không có P5 XCOPA, không có P1 cho noul / SIB / MASSIVE (bỏ theo quyết định 26/10).
Bản không dấu = strip_diacritics(chữ đã duyệt), nhận định không dấu lấy từ chính dòng không dấu.
Phép nào / khung nào chưa đủ giá trị duyệt thì BỎ QUA và ghi vào manifest, không dừng cả script.
Ra results/probe_v1/: p1b_doi_chu_nhan.jsonl, p5_doi_cau_hoi.jsonl (nếu có khung nào), mục P1_P5 của manifest_probe_v1.json.
Dùng: python scripts/probe_v1/build_p1_p5.py
      python scripts/probe_v1/build_p1_p5.py --thu_de_xuat --out <thư mục nháp>   # THỬ CƠ CHẾ bằng chữ đang chờ duyệt; cấm ghi vào results/probe_v1"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, DATA, RESULTS, read_jsonl, sha256_file, strip_diacritics, write_jsonl  # noqa: E402
from probe_strings import P1, P5  # noqa: E402  (chuỗi gốc của dev; chữ thay lấy từ chữ đã duyệt)

DEV = DATA / "bench_v1_3" / "dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
OUT = RESULTS / "probe_v1"
CHAT = RESULTS / "probe_v1" / "fable_soat" / "quyet_dinh_nguoi_dung.json"
CHAT2 = RESULTS / "probe_v1" / "quyet_dinh_chat_pv1b.json"   # người dùng đồng ý trong chat 26/10 ("Hay test thử"), điều phối ghi
THU = "--thu_de_xuat" in sys.argv
if THU:
    OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    assert OUT != (RESULTS / "probe_v1").resolve(), "chế độ thử không được ghi vào results/probe_v1"
P5_KHUNG = ("kiem_chung_choice", "kiem_chung_noul", "sib200", "massive")
NHAN_P1B = {cur: f"probe_v1.P1b.kiem_chung.{lab}" for g, lab, cur, _, _ in P1 if g == "kiem_chung"}   # chuỗi dev -> khoa
Q_GOC = {k: cur for k, cur, _ in P5}

# ---------------- giá trị
props = [json.loads(l) for l in open(CONFIG / "vi_proposals.jsonl", encoding="utf-8") if l.strip()]
PROP = {p["id"]: p for p in props if str(p.get("khoa", "")).startswith("probe_v1.")}
duyet = []                                            # (thời điểm, thứ tự, khoa, giá trị, nguồn)


def chat_approved(q):
    """id đề xuất được duyệt trong một file quyết định chat: {"quyet_dinh": {id: "duyet" | ...}} hoặc danh sách id ở "duyet" / "ids"."""
    if isinstance(q.get("quyet_dinh"), dict):
        ids = [i for i, v in q["quyet_dinh"].items() if v == "duyet"]
    else:
        ids = list(q.get("duyet") or q.get("ids") or [])
    miss = [i for i in ids if i not in PROP]
    assert not miss, f"id không có trong vi_proposals.jsonl: {miss}"
    return ids
if THU:
    for n, p in enumerate(PROP.values()):
        if p["id"].startswith("pv1b-") or p["khoa"] in ("probe_v1.P5.kiem_chung_choice", "probe_v1.P5.kiem_chung_noul"):
            duyet.append((p["thoi_diem"], n, p["khoa"], p["gia_tri_de_xuat"], f"THU:{p['id']}"))
    NGUON_GIA_TRI = "THỬ: gia_tri_de_xuat đang chờ duyệt trong config/vi_proposals.jsonl (KHÔNG dùng để chấm)"
else:
    log = [json.loads(l) for l in (DATA / "human_edits/edits_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    last = {}
    for k, r in sorted(enumerate(log), key=lambda x: (x[1].get("thoi_diem", ""), x[0])):
        if r.get("loai") == "de_xuat" and str(r.get("de_xuat_id", "")) in PROP:
            last[r["de_xuat_id"]] = (k, r)
    for did, (k, r) in last.items():
        if r["hanh_dong"] in ("duyet", "sua_roi_duyet"):
            duyet.append((r["thoi_diem"], k, PROP[did]["khoa"], r["gia_tri_moi"], f"edits_log:{did}"))
    NGUON_GIA_TRI = {"edits_log.jsonl": sha256_file(DATA / "human_edits/edits_log.jsonl")}
    for fchat, tag in ((CHAT, "chat_26_10"), (CHAT2, "chat_26_10_pv1b")):
        if not fchat.exists():
            NGUON_GIA_TRI[fchat.name] = "KHÔNG CÓ FILE (bỏ qua)"
            continue
        NGUON_GIA_TRI[fchat.name] = sha256_file(fchat)
        for did in chat_approved(json.loads(fchat.read_text(encoding="utf-8"))):
            # chữ = đúng gia_tri_de_xuat của đề xuất (đọc từ config/vi_proposals.jsonl, chỉ đọc)
            duyet.append((PROP[did]["thoi_diem"], -1, PROP[did]["khoa"], PROP[did]["gia_tri_de_xuat"], f"{tag}:{did}"))
VAL, VAL_SRC = {}, {}
for t, k, kh, v, src in sorted(duyet, key=lambda x: (x[0], x[1])):
    VAL[kh], VAL_SRC[kh] = v, src                     # lần duyệt mới nhất thắng

assert sha256_file(DEV) == DEV_SHA, "dev 1.3 sai sha"
dev = read_jsonl(DEV)
assert not any(":test:" in r["id"] for r in dev)
VI = [r for r in dev if r["lang"] == "vi"]
MARK = {"kiem_chung_choice": ("Nhận định: ", "Nhan dinh: "), "kiem_chung_noul": ("Phát biểu: ", "Phat bieu: ")}


def base(r, probe):
    x = {k: v for k, v in r.items() if k != "n_tokens"}       # n_tokens của dev không còn đúng (eval_base.py kiểm khi có trường này)
    x.update({"id": f"{r['id']}:{probe}", "probe": probe, "orig_id": r["id"]})
    return x


def khung(r):
    if r["source"] in ("vifactcheck", "viwikifc"):
        return "kiem_chung_choice" if r["kind"] == "choice" else "kiem_chung_noul"
    return r["source"] if r["source"] in ("sib200", "massive") else None


def new_question(r, k):
    cur, new = Q_GOC[k], VAL[f"probe_v1.P5.{k}"]
    nd = r["variant"] == "nodiac"
    if k in MARK:
        mk = MARK[k][nd]
        pre, claim = r["question"].split(mk, 1)
        assert (strip_diacritics(cur.split("{claim}")[0]) if nd else cur.split("{claim}")[0]) == pre + mk, r["id"]
        return (strip_diacritics(new.replace("{claim}", "\x00")) if nd else new.replace("{claim}", "\x00")).replace("\x00", claim)
    assert r["question"] == (strip_diacritics(cur) if nd else cur), r["id"]
    return strip_diacritics(new) if nd else new


thieu = {}
co_p5 = [k for k in P5_KHUNG if f"probe_v1.P5.{k}" in VAL]
for k in P5_KHUNG:
    if k not in co_p5:
        thieu[f"P5.{k}"] = [f"probe_v1.P5.{k}"]
p5 = []
for r in VI:
    k = khung(r)
    if k in co_p5:
        x = base(r, "P5")
        x.update({"question": new_question(r, k), "orig_question": r["question"], "p5_khung": k})
        p5.append(x)

p1b = []
miss = [kh for kh in NHAN_P1B.values() if kh not in VAL] + ([] if "kiem_chung_choice" in co_p5 else ["probe_v1.P5.kiem_chung_choice"])
if miss:
    thieu["P1b.kiem_chung"] = miss
else:
    m = {cur: VAL[kh] for cur, kh in NHAN_P1B.items()}
    m.update({strip_diacritics(cur): strip_diacritics(VAL[kh]) for cur, kh in NHAN_P1B.items()})
    for r in VI:
        if khung(r) == "kiem_chung_choice":
            x = base(r, "P1b")
            x.update({"question": new_question(r, "kiem_chung_choice"), "orig_question": r["question"],
                      "options": [m[o] for o in r["options"]], "orig_options": r["options"]})   # KeyError = lựa chọn lạ: dừng
            assert len(set(x["options"])) == 3, r["id"]
            p1b.append(x)

# ---------------- kiểm trùng train_v1 (A/B/C): state là state dev (đã kiểm 0 trùng ở P3/P4/Pa); ở đây đếm số dòng train dùng đúng
# câu hỏi mới (phần trước {claim} với kiểm chứng) hay chữ lựa chọn mới (để biết model đã gặp chữ đó lúc train chưa; không phải rò rỉ)
def _n(s):
    return " ".join(s.lower().split())


moi_q = {k: _n(VAL[f"probe_v1.P5.{k}"].split("{claim}")[0]) for k in co_p5}
moi_q.update({k + "|khong_dau": _n(strip_diacritics(v)) for k, v in list(moi_q.items())})
moi_o = {_n(x) for x in p1b for x in x["options"]}
trung = {}
if not THU:
    for comp in ("A", "B", "C"):
        cq, co, n = Counter(), Counter(), 0
        with open(RESULTS / "train_v1" / "components" / f"{comp}.jsonl", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                n += 1
                q = _n(t["question"])
                for k, v in moi_q.items():
                    if v and q.startswith(v):
                        cq[k] += 1
                for o in t["options"]:
                    if _n(o) in moi_o:
                        co[_n(o)] += 1
        trung[comp] = {"rows": n, "file_sha256": sha256_file(RESULTS / "train_v1" / "components" / f"{comp}.jsonl"),
                       "dong_dung_cau_hoi_moi": dict(cq), "lan_dung_chu_lua_chon_moi": dict(co)}

OUT.mkdir(parents=True, exist_ok=True)
files = {}
for fn, rows in (("p1b_doi_chu_nhan.jsonl", p1b), ("p5_doi_cau_hoi.jsonl", p5)):
    if rows:
        write_jsonl(OUT / fn, rows)
        files[fn] = sha256_file(OUT / fn)
sec = {"nguon_gia_tri": NGUON_GIA_TRI, "gia_tri_dung": {kh: {"chu": VAL[kh], "nguon": VAL_SRC[kh]} for kh in sorted(VAL)},
       "files": files, "chua_du_duyet (không dựng)": thieu, "trung_train_v1": trung or "không kiểm (chế độ thử)",
       "P1b": dict(Counter(f"{x['source']}|{x['variant']}|{x['kind']}" for x in p1b)),
       "P5": dict(Counter(f"{x['p5_khung']}|{x['source']}|{x['variant']}" for x in p5)),
       "ghi_chu": "P1′ (mã P1b) = câu hỏi P5 kiểm chứng choice + 3 lựa chọn mới; P5 đổi câu hỏi, giữ lựa chọn. Đáp án và thứ tự lựa chọn như dev; "
                  "so theo cặp với dev (và P1′ với P5) qua orig_id. Bỏ theo quyết định 26/10: P1 noul, P1 SIB, P1 MASSIVE, P5 XCOPA."}
mpath = OUT / "manifest_probe_v1.json"
man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
    "CANH_BAO": "BỘ PHÂN TÍCH, KHÔNG PHẢI BENCHMARK. KHÔNG BAO GIỜ DÙNG ĐỂ TRAIN."}
man["P1_P5"] = sec
mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(sec, ensure_ascii=False, indent=1))
