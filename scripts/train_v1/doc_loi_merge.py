"""Ghép kết quả đọc lỗi MÙ bằng Opus (doc_loi_prep.py), giải mã bằng khoa.json, đếm theo nhóm / nguồn / loại, ghi results/train_v1/phan_tich_C/doc_loi_opus.md.
Không chấm model, không train, không đọc test. Thư viện chuẩn.
Dùng: python scripts/train_v1/doc_loi_merge.py"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/train_v1/phan_tich_C/doc_loi"
OUT = ROOT / "results/train_v1/phan_tich_C/doc_loi_opus.md"
LOAI_DUNG = ["dung_ly_do_dung", "dung_meo_lua_chon", "dung_kien_thuc_san_co", "dung_khop_be_mat", "dung_nhan_dang_ngo"]
LOAI_SAI = ["sai_khong_doc_van_ban", "sai_hieu_sai_phu_dinh", "sai_nham_nhan_kiem_chung", "sai_nham_nhan_gan_nghia", "sai_suy_luan",
            "sai_mo_ho_hoac_nhan_nghi_sai", "sai_khac"]
P4_FIELDS = {"van_ban_lien_quan": ["khong", "cung_chu_de_rong", "co_chi_tiet_lien_quan"], "nei_co_that_dung": ["co", "khong", "khong_chac"],
             "nhan_dinh_ngoai_doi": ["dung", "sai", "khong_biet"],
             "nguyen_nhan_kha_di": ["nhan_dinh_sai_theo_kien_thuc", "nhan_dinh_dung_theo_kien_thuc", "nhan_dinh_tuyet_doi_hoa",
                                    "nhan_dinh_co_phu_dinh", "chu_de_trung_van_ban", "khong_ro"]}

khoa = json.loads((D / "khoa.json").read_text(encoding="utf-8"))
res, loi = {}, []
for f in sorted((D / "ket_qua").glob("*.jsonl")):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            if x["ma"] in res:
                loi.append(f"trùng mã {x['ma']}")
            res[x["ma"]] = x
thieu = sorted(set(khoa) - set(res))
for m, x in res.items():
    k = khoa.get(m)
    if k is None:
        loi.append(f"mã lạ {m}"); continue
    if k["nhom"] == "P4":
        for fld, ok in P4_FIELDS.items():
            if x.get(fld) not in ok:
                loi.append(f"{m}: {fld}={x.get(fld)!r} ngoài danh sách")
    else:
        ok = LOAI_DUNG if k["nhom"] == "A" else LOAI_SAI    # nhóm A: model (C) chọn đúng; nhóm B: chọn sai
        if x.get("loai") not in ok:
            loi.append(f"{m} (nhóm {k['nhom']}): loai={x.get('loai')!r} không khớp nhánh đúng/sai")

pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "—"
L = ["# Đọc lỗi mù bằng Opus: train thử bản C so với M1 (dev 1.3)", "",
     "Ngày 26/10/2026. Theo mục 4 của `results/BRIEF_phan_tich_xu_huong.md`; điều phối giao. Không train, không chạy test.", "",
     "## Cách làm", "",
     "- **Mẫu** (`scripts/train_v1/doc_loi_prep.py`): seed 20261012, phân tầng theo nguồn; chi tiết ở `doc_loi/chon_mau.json`.",
     "  - **A**: 50 câu dev VI có dấu mà M1 sai, C@100% đúng.",
     "  - **B**: 50 câu M1 đúng, C@100% sai.",
     "  - **P4**: 40 câu P4 VI có dấu, kiểu choice, mà C không chọn \"không đủ thông tin\" còn M1 chọn đúng.",
     "- **Đọc mù:**",
     "  - A và B trộn chung, mã trung tính X001…, chia 4 lô; P4 có 2 lô riêng.",
     "  - Mỗi câu chỉ có văn bản, câu hỏi, lựa chọn, đáp án đúng và đáp án \"một model\" chọn (luôn là của C).",
     "  - Người đọc không biết model nào, không biết nhóm; khóa giải mã (`doc_loi/khoa.json`) không đưa cho người đọc.",
     "- **Người đọc:** 6 subagent Opus, mỗi lô một subagent. Hướng dẫn: `doc_loi/PROMPT_doc_loi.md`, `doc_loi/PROMPT_p4.md`. Kết quả thô: `doc_loi/ket_qua/*.jsonl`.",
     "- **Lưu ý:** nhóm A có đáp án đúng nên chỉ xếp vào loại `dung_*`; nhóm B chỉ xếp vào `sai_*`. Mỗi câu chỉ có một người đọc, nên chưa đo độ đồng thuận.", "",
     f"**Kiểm định dạng:** {len(res)}/{len(khoa)} câu có kết quả; thiếu {len(thieu)}; lỗi {len(loi)}.", ""]
if thieu or loi:
    L += ["```", *([f"thiếu: {', '.join(thieu)}"] if thieu else []), *loi, "```", ""]

for g, ten, ds in (("A", "A. C mới làm đúng (M1 sai → C đúng): đúng vì lý do gì?", LOAI_DUNG),
                   ("B", "B. C mới làm sai (M1 đúng → C sai): loại lỗi", LOAI_SAI)):
    ms = [m for m in khoa if khoa[m]["nhom"] == g and m in res]
    srcs = sorted({khoa[m]["source"] for m in ms})
    c = Counter(res[m]["loai"] for m in ms)
    L += [f"## {ten}", "", f"n = {len(ms)}.", "", "| Loại | Số câu | Tỉ lệ | " + " | ".join(srcs) + " |", "|---|---|---|" + "---|" * len(srcs)]
    for lo in ds:
        L.append(f"| `{lo}` | {c[lo]} | {pct(c[lo], len(ms))} | " + " | ".join(str(sum(1 for m in ms if res[m]["loai"] == lo and khoa[m]["source"] == s)) for s in srcs) + " |")
    L.append(f"| **Tổng** | {len(ms)} | | " + " | ".join(str(sum(1 for m in ms if khoa[m]["source"] == s)) for s in srcs) + " |")
    vb = Counter(res[m].get("van_ban_du_can_cu") for m in ms); tt = Counter(res[m].get("tu_tin") for m in ms)
    mh = sum(khoa[m]["mo_ho"] for m in ms)
    L += ["", f"- Văn bản đủ căn cứ cho đáp án đúng: có {vb['co']}, một phần {vb['mot_phan']}, không {vb['khong']}.",
          f"- Độ tự tin của người đọc: cao {tt['cao']}, vừa {tt['vua']}, thấp {tt['thap']}.",
          f"- Câu mang cờ mơ hồ của bench: {mh}.", "", "**Ví dụ ghi chú** (tối đa 2 câu mỗi loại):", ""]
    for lo in ds:
        ex = [m for m in sorted(ms) if res[m]["loai"] == lo][:2]
        for m in ex:
            L.append(f"- `{lo}`, {m} ({khoa[m]['source']}, {khoa[m]['kind']}): {res[m].get('ghi_chu', '')}")
    L.append("")

ms = [m for m in khoa if khoa[m]["nhom"] == "P4" and m in res]
L += ["## P4. Văn bản không liên quan: C không chọn \"không đủ thông tin\", M1 chọn đúng", "",
      f"n = {len(ms)}. Nhãn C chọn: {dict(Counter(khoa[m]['c_pred_nhan'] for m in ms))}. "
      f"Nguồn: {dict(Counter(khoa[m]['source'] for m in ms))}.", ""]
for fld, ok in P4_FIELDS.items():
    c = Counter(res[m].get(fld) for m in ms)
    L += [f"**{fld}**", "", "| Giá trị | Số câu | Tỉ lệ |", "|---|---|---|"] + [f"| `{v}` | {c[v]} | {pct(c[v], len(ms))} |" for v in ok] + [""]
ct = Counter((res[m].get("nhan_dinh_ngoai_doi"), res[m].get("van_ban_lien_quan")) for m in ms)
L += ["**Bảng chéo: nhận định ngoài đời × văn bản liên quan**", "",
      "| ngoài đời \\ văn bản | " + " | ".join(P4_FIELDS["van_ban_lien_quan"]) + " |", "|---|---|---|---|"]
for a_ in P4_FIELDS["nhan_dinh_ngoai_doi"]:
    L.append(f"| {a_} | " + " | ".join(str(ct[(a_, b_)]) for b_ in P4_FIELDS["van_ban_lien_quan"]) + " |")
dh = Counter(d for m in ms for d in res[m].get("dau_hieu", []))
og = Counter(khoa[m]["orig_gold_label"] for m in ms)
L += ["", f"**Dấu hiệu trong nhận định** (một câu có thể nhiều dấu hiệu): {dict(dh.most_common())}.", "",
      f"**Nhãn gốc của nhận định trong bài của nó** (trước khi ghép văn bản khác): {dict(og)}.", "",
      "**Ví dụ ghi chú** (tối đa 2 câu mỗi nguyên nhân):", ""]
for lo in P4_FIELDS["nguyen_nhan_kha_di"]:
    for m in [m for m in sorted(ms) if res[m].get("nguyen_nhan_kha_di") == lo][:2]:
        L.append(f"- `{lo}`, {m} ({khoa[m]['source']}): {res[m].get('ghi_chu', '')}")
L += ["", "## Ghi chú", "",
      "- Đây là đọc của Opus, mỗi câu một người đọc; chưa đo độ đồng thuận. Số nhỏ (50/50/40), nên tỉ lệ chỉ để định hướng.",
      "- Báo cáo chỉ trình bày số, không kết luận kịch bản K1–K9."]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"{len(res)}/{len(khoa)} câu; thiếu {len(thieu)}; lỗi {len(loi)}")
for x in loi[:10]:
    print("LỖI", x)
