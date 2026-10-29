"""Bước D (2/3): so bản dựng 1.2 ở khu tạm (results/bench_v1_2_stage/) với bản thử cuối (results/bench_v1_2_trial/), phân loại MỌI khác biệt theo
bản ghi lớp ghi đè / quyết định nhãn có SAU lần dựng thử:
  hoan_tac_B               : hoàn tác ở màn Đúng→sai (câu được chọn + câu cùng đoạn văn)
  nguoi_dung_trong_chat    : bản dịch người dùng gửi trong chat, hoặc người dùng bảo chỉnh trong chat (áp qua đề xuất)
  quyet_dinh_tren_trang    : quyết định mới / đổi ý trên trang (artifact) áp qua đề xuất hoặc hoàn tác đề xuất
  chuoi_mau_transport_query: chuỗi mẫu MASSIVE transport_query đổi
  nhan_test                : quyết định nhãn test (loại / đổi nhãn / giữ) ghi sau lần dựng thử
  loai_do_trung_test       : cặp dev bị loại vì trùng gần với test (giữ quy tắc dev↔test = 0)
  chi_provenance           : nội dung không đổi, chỉ khác trường provenance
Cặp đổi nội dung mà không thuộc loại nào -> "khong_giai_thich" (chặn đóng băng). Ghi thêm rescore/rescore_dev.jsonl: các dòng dev 1.2 có nội dung
khác bản thử hoặc không có trong bản thử (cần chấm lại). Python Windows, thư viện chuẩn."""
import datetime
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S, T = ROOT / "results/bench_v1_2_stage", ROOT / "results/bench_v1_2_trial"
PROV = ("vi_provenance", "vi_revised_fields", "gold_provenance")
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
core = lambda r: {k: v for k, v in r.items() if k not in PROV}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

br_t = json.load(open(T / "build_report_v1_2_trial.json", encoding="utf-8"))
br_s = json.load(open(S / "build_report_v1_2_trial.json", encoding="utf-8"))
N0 = br_t["edits_log_records"]
T_TIME = datetime.datetime.fromtimestamp(os.path.getmtime(T / "dev_v1_2_trial.jsonl")).astimezone()
log = read(ROOT / "data/human_edits/edits_log.jsonl")
assert len(log) == br_s["edits_log_records"], "lớp ghi đè đã đổi sau khi dựng khu tạm: dựng lại"
new = log[N0:]
assert all(datetime.datetime.fromisoformat(r["thoi_diem"]) > T_TIME for r in new) and datetime.datetime.fromisoformat(log[N0 - 1]["thoi_diem"]) < T_TIME
dx_note = {}
for r in new:
    if r.get("loai") == "de_xuat":
        dx_note[r["de_xuat_id"]] = r.get("ghi_chu") or ""
pid_of = lambda r: r.get("dev_id") or r.get("pair_id")


def cat_of(r):
    g = r.get("ghi_chu") or ""
    if r.get("truong") == "loai_bo" and "trùng gần với test" in g:
        return "loai_do_trung_test"
    if "đúng→sai" in g:
        return "hoan_tac_B"
    i = r.get("de_xuat_id") or r.get("hoan_tac_de_xuat")
    if i and "chat" in dx_note.get(i, "") and "đổi ý" not in g:
        return "nguoi_dung_trong_chat"
    return "quyet_dinh_tren_trang"


why = defaultdict(set)          # (split, pair) -> loại
for r in new:
    if r.get("loai") == "cau":
        why[(r.get("split") or "dev", pid_of(r))].add(cat_of(r))
st_ds = [r for r in new if r.get("loai") == "dung_sai"]
for r in st_ds:
    if r["hanh_dong"] == "hoan_tac":
        for q in [r["pair_id"], *r.get("keo_theo", [])]:
            why[("dev", q)].add("hoan_tac_B")
tq = [r for r in new if r.get("loai") == "mau" and r.get("khoa") == "translations.massive_intents.transport_query"]
assert len(tq) == 1 and all(r.get("khoa") == "translations.massive_intents.transport_query" for r in new if r.get("loai") == "mau")
TQ_OLD, TQ_NEW = tq[0]["gia_tri_cu"], tq[0]["gia_tri_moi"]
lab_new = [x for x in read(ROOT / "results/human_review/label_decisions_remote.jsonl") if datetime.datetime.fromisoformat(x["thoi_diem"]) > T_TIME]
for x in lab_new:
    why[(x["split"], x["pair_id"])].add("nhan_test" if x["split"] == "test" else "nhan_dev_bo_qua")

res, rescore = {}, []
for split, fn in (("dev", "dev_v1_2_trial.jsonl"), ("test", "test_v1_2_trial.jsonl")):
    A = {r["id"]: r for r in read(T / fn)}
    B = {r["id"]: r for r in read(S / fn)}
    pairs = defaultdict(lambda: {"them": [], "bot": [], "noi_dung": [], "chi_provenance": [], "truong": set()})
    for i in set(A) | set(B):
        a, b = A.get(i), B.get(i)
        if a == b:
            continue
        p = (a or b)["pair_id"]
        if a is None:
            pairs[p]["them"].append(i)
        elif b is None:
            pairs[p]["bot"].append(i)
        elif core(a) != core(b):
            pairs[p]["noi_dung"].append(i)
            pairs[p]["truong"] |= {k for k in core(a) if a.get(k) != b.get(k)}
        else:
            pairs[p]["chi_provenance"].append(i)
        if split == "dev" and b is not None and (a is None or core(a) != core(b)):
            rescore.append(b)
    out = defaultdict(list)
    for p, d in sorted(pairs.items()):
        cats = set(why.get((split, p), set()))
        content = bool(d["them"] or d["bot"] or d["noi_dung"])
        src = (B.get(p + ":vi") or A.get(p + ":vi") or {}).get("source") or p.split(":")[0]
        if content and src == "massive" and d["truong"] <= {"options", "n_tokens"} and p + ":vi" in A and p + ":vi" in B:
            a, b = A[p + ":vi"], B[p + ":vi"]
            dif = [(x, y) for x, y in zip(a["options"], b["options"]) if x != y]
            if dif and all(x == TQ_OLD and y == TQ_NEW for x, y in dif):
                cats.add("chuoi_mau_transport_query")
        if not content:
            cats = {"chi_provenance"}
        elif not cats:
            cats = {"khong_giai_thich"}
        for c in sorted(cats):
            out[c].append(p)
        out["_chi_tiet"].append({"pair_id": p, "loai": sorted(cats), "them": len(d["them"]), "bot": len(d["bot"]), "noi_dung": len(d["noi_dung"]),
                                 "chi_provenance": len(d["chi_provenance"]), "truong": sorted(d["truong"])})
    res[split] = {"so_cap_khac": len(pairs), "theo_loai": {k: len(v) for k, v in sorted(out.items()) if k != "_chi_tiet"},
                  "cap_theo_loai": {k: v for k, v in sorted(out.items()) if k != "_chi_tiet"}, "chi_tiet": out["_chi_tiet"],
                  "dong_them": sum(d_["them"] for d_ in out["_chi_tiet"]), "dong_bot": sum(d_["bot"] for d_ in out["_chi_tiet"])}
(S / "rescore").mkdir(exist_ok=True)
with open(S / "rescore/rescore_dev.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in sorted(rescore, key=lambda r: r["id"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summ = {"ban_thu": {"dev": sha(T / "dev_v1_2_trial.jsonl"), "test": sha(T / "test_v1_2_trial.jsonl"), "ban_ghi_lop_ghi_de": N0, "thoi_diem": T_TIME.isoformat()},
        "ban_dung": {"dev": sha(S / "dev_v1_2_trial.jsonl"), "test": sha(S / "test_v1_2_trial.jsonl"), "ban_ghi_lop_ghi_de": len(log)},
        "ban_ghi_moi": dict(Counter(f"{r.get('loai')}|{r.get('hanh_dong')}" for r in new)), "quyet_dinh_nhan_moi": [
            {k: x.get(k) for k in ("split", "pair_id", "hanh_dong", "dap_an_moi", "thoi_diem")} for x in lab_new],
        "dung_sai": {"giu_moi": sum(r["hanh_dong"] == "giu_moi" for r in st_ds), "hoan_tac": [{"pair_id": r["pair_id"], "keo_theo": r.get("keo_theo", []),
                                                                                                 "de_xuat": r.get("de_xuat", [])} for r in st_ds if r["hanh_dong"] == "hoan_tac"]},
        "transport_query": {"cu": TQ_OLD, "moi": TQ_NEW}, "dong_dev_can_cham_lai": len(rescore), **res}
(S / "diff_vs_trial_v1_2.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({s: {k: res[s][k] for k in ("so_cap_khac", "theo_loai", "dong_them", "dong_bot")} for s in res}, ensure_ascii=False))
print("khong_giai_thich:", {s: res[s]["cap_theo_loai"].get("khong_giai_thich", []) for s in res}, "| dev rows to rescore:", len(rescore))
print("nhan moi:", summ["quyet_dinh_nhan_moi"])
