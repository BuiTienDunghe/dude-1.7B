"""Ghép bảng thuật ngữ v2 cho bản dịch đầy đủ phần B (chỉ thư viện chuẩn).
Vào: translated_sonnet/full_prep/bang_thuat_ngu_v2_phan1.json, _phan2.json, bang_tien_to_v2.json (Sonnet soạn).
Kiểm: không trùng khóa giữa hai phần; vi khác rỗng, NFC, không ký tự ẩn; trong mỗi nguồn nhãn (phần 2) không hai nhãn cùng bản dịch;
câu hỏi trích nhãn trong ngoặc kép chứa đúng bản dịch nhãn; tiền tố 'dich' giữ dấu hai chấm; phủ đủ chuỗi lặp >= 5 lần của file dịch.
Ra: to_translate/bang_thuat_ngu_v2.json {"_ghi_chu", "thuat_ngu": {en: {vi, ghi_chu}}, "tien_to": {en: {cach, vi}}} và kiem_bang_thuat_ngu_v2.json."""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
F = T / "translated_sonnet/full_prep"
p1 = json.load(open(F / "bang_thuat_ngu_v2_phan1.json", encoding="utf-8"))
p2 = json.load(open(F / "bang_thuat_ngu_v2_phan2.json", encoding="utf-8"))
tt = json.load(open(F / "bang_tien_to_v2.json", encoding="utf-8"))
HIDDEN = re.compile(r"[​-‏ -‮⁠-⁤﻿­]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
errs, warns = [], []
both = set(p1) & set(p2)
if both:
    errs.append(f"khóa trùng giữa hai phần: {sorted(both)[:5]}")
terms = {**{k: {"vi": v["vi"], "ghi_chu": v.get("ghi_chu", "")} for k, v in p2.items()}, **{k: {"vi": v["vi"], "ghi_chu": v.get("ghi_chu", "")} for k, v in p1.items()}}
for k, v in list(terms.items()) + [(k, v) for k, v in tt.items()]:
    s = v["vi"]
    if not s.strip():
        errs.append(f"rỗng: {k}")
    if unicodedata.normalize("NFC", s) != s:
        errs.append(f"chưa NFC: {k}")
    if HIDDEN.search(s):
        errs.append(f"ký tự ẩn: {k}")
by = defaultdict(dict)
for k, v in p2.items():
    for other, vi in by[v.get("nguon", "?")].items():
        if vi == v["vi"]:
            errs.append(f"hai nhãn cùng bản dịch trong {v.get('nguon')}: {k} / {other}")
    by[v.get("nguon", "?")][k] = v["vi"]
for k, v in terms.items():
    for lab in re.findall(r'"([^"]{1,80})"', k):
        if lab in terms and f'"{terms[lab]["vi"]}"' not in v["vi"]:
            errs.append(f"câu hỏi không chứa đúng bản dịch nhãn {lab!r}: {k[:60]}")
for k, v in tt.items():
    if v["cach"] == "dich" and not v["vi"].rstrip().endswith(":"):
        errs.append(f"tiền tố mất dấu hai chấm: {k}")
rep = json.load(open(T / "to_translate/chuoi_lap_lai.json", encoding="utf-8"))
miss = [s for d in rep.values() for s in d if s not in terms]
if miss:
    warns.append(f"{len(miss)} chuỗi lặp >= 5 lần chưa có trong bảng: {miss[:8]}")
out = {"_ghi_chu": "Bảng thuật ngữ v2 (Sonnet soạn 19/10, ghép bằng scripts/train_v1/merge_glossary_v2.py). Dùng cùng HUONG_DAN_DICH_v2.md. "
                   "Chuỗi trùng NGUYÊN VĂN một khóa của 'thuat_ngu' thì dùng đúng 'vi'. 'tien_to': cach=giu_nguyen thì giữ nguyên, cach=dich thì dùng 'vi'. "
                   "Tiền tố Turn N:, text_A:, text_B:, Passage X:, Item X:, một chữ cái (A:, B:, M:, W:, F:, Q:) luôn giữ nguyên (xem hướng dẫn). "
                   "Nhãn SUPPORTS/REFUTES/NOT ENOUGH INFO theo bench (người dùng quyết 20/10; chuỗi lấy từ config/bench_v1_questions.json).",
       "thuat_ngu": dict(sorted(terms.items())), "tien_to": dict(sorted(tt.items()))}
(T / "to_translate/bang_thuat_ngu_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
res = {"thuat_ngu": len(terms), "phan1": len(p1), "phan2": len(p2), "tien_to": len(tt),
       "tien_to_giu_nguyen": sum(v["cach"] == "giu_nguyen" for v in tt.values()), "loi": errs, "canh_bao": warns, "ket_luan": "ĐẠT" if not errs else "KHÔNG ĐẠT"}
(F / "kiem_bang_thuat_ngu_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
