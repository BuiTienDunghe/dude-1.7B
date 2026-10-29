"""Dựng results/train_v1_survey/tables_sources.md từ các file khảo sát đã kiểm:
raw/vi_candidates.json (subagent; revision/gated/số dòng train đã đối chiếu lại bằng API HF), pholaya_counts.json, sample_lengths.json,
tasksource_pool.json, raw/mt_candidates.json (revision/giấy phép/gated đã đối chiếu bằng API HF). Không in nội dung dữ liệu."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, load_json  # noqa: E402

S = RESULTS / "train_v1_survey"
vi = load_json(S / "raw" / "vi_candidates.json")
pl = load_json(S / "pholaya_counts.json")
ln = load_json(S / "sample_lengths.json")
mt = load_json(S / "raw" / "mt_candidates.json")
tsp = load_json(S / "tasksource_pool.json") if (S / "tasksource_pool.json").exists() else None
DELETED = {"vsmec_ura", "phoatis_ura", "zalo_e2eqa_ura"}  # mẫu lấy từ bản sao bên thứ ba, đã xóa
cut = lambda s, n: (str(s) if s is not None else "—").replace("\n", " ").replace("|", "/")[:n]


def stem(r):
    f = r.get("sample_file")
    return Path(str(f).replace("\\", "/")).stem if f else None


L = ["# Bảng khảo sát nguồn tiếng Việt cho train_v1", "",
     "Số dòng train và revision của mọi ứng viên trên HF đã đối chiếu lại bằng API (`api/datasets/<id>` và dataset viewer `/size`): "
     "khớp 100%. Độ dài: token Qwen3 (tokenizer bộ đếm của train_v0) trên ~300 dòng split train lấy qua dataset viewer.", ""]
L += ["## 1. Nhóm 1 — phần còn lại của nguồn tiếng Việt đã dùng (LakoreAI/PhoLaya-base @ `034d4273`)", "",
      "Mỗi dòng của PhoLaya là một câu hỏi trên một văn bản; một văn bản có thể mang nhiều câu hỏi (vd. ViNLI: relation + entailment trên cùng cặp câu).", "",
      "| Bộ | Tác vụ (câu hỏi) | Kiểu | Câu hỏi trong train | Đã dùng (train_v0 + val_v0) | Còn lại | Trung vị state (token) | Dung lượng file train |",
      "|---|---|---|---|---|---|---|---|"]
for ds, rec in pl["sources"].items():
    for t, e in rec["tasks"].items():
        L.append(f"| {ds} | {t} | {', '.join(e['kinds'])} | {e['questions']:,} | {e['used_in_train_v0_val_v0']:,} | "
                 f"{e['questions'] - e['used_in_train_v0_val_v0']:,} | {rec['state_tokens_median']:.0f} | {rec['file_bytes'] / 1e6:.1f} MB |")
L += ["", "vsmec: đã đếm từ bản trong PhoLaya trước khi biết nguồn gốc yêu cầu ký thỏa thuận; file tải về đã xóa. "
      "viglue (586 MB) và massive (bench_v1) không đếm.", ""]
if tsp:
    L += ["Nguồn tiếng Việt sẵn có trong tasksource-jev-typed-decisions (`multilingual/*vi`):", "",
          "| Nguồn | Dòng | Đã dùng | Bị loại (bench) | Kiểu | Trung vị token cả dòng |", "|---|---|---|---|---|---|"]
    for s, e in tsp["vi_sources"].items():
        L.append(f"| {s} | {e['rows']:,} | {e['used_v0']} | {'có' if e['excluded'] else 'không'} | {', '.join(e['kinds'])} | {e['tokens_median']:.0f} |")
    L.append("")
L += ["## 2. Nhóm 2 — bộ tiếng Việt khác, có split train, truy cập mở", "",
      "| Bộ | Repo @ revision | Dòng train | Dạng bài | Cách đổi | Trung vị state / cả dòng (token) | Nhãn | Truy cập | Rủi ro trùng bench_v1 |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in vi:
    if r["status"] != "candidate":
        continue
    st = stem(r)
    if st in DELETED:
        lens = "mẫu đã xóa (bản sao bên thứ ba)"
    elif st and st in ln:
        lens = f"{ln[st]['state_tokens_median']:.0f} / {ln[st]['row_tokens_median']:.0f}"
    else:
        lens = "—"
    repo = cut(r["url"], 80).replace("https://huggingface.co/datasets/", "")
    L.append(f"| {cut(r['name'], 60)} | {repo} @ `{cut(r['revision'], 8)}` | {cut(r['train_rows'], 10)} | {cut(r['task'], 70)} | "
             f"{cut(r['conversion'], 160)} | {lens} | {cut(r['label_origin'], 30)} | {cut(r['access'], 40)} | {cut(r['overlap_risk'], 120)} |")
for status, title in (("agreement-only", "Cần email / ký thỏa thuận / HF-gated — chỉ liệt kê, không tải"),
                      ("no train split", "Không có split train"), ("unavailable", "Không tải được / không tồn tại"),
                      ("excluded", "Bị loại (nguồn bench_v1 hoặc dẫn xuất, hoặc điều khoản cấm)"), ("already used", "Đã dùng")):
    L += ["", f"### {title}", "", "| Bộ | URL | Ghi chú |", "|---|---|---|"]
    for r in vi:
        if r["status"] == status:
            L.append(f"| {cut(r['name'], 70)} | {cut(r['url'], 90)} | {cut(r.get('access'), 50)}; {cut(r.get('notes'), 200)} |")
L += ["", "## 3. Giấy phép / điều kiện truy cập — chép nguyên văn từ nguồn", ""]
for r in vi:
    if r["status"] in ("candidate", "agreement-only", "already used"):
        L += [f"- **{cut(r['name'], 90)}** — nguồn: {cut(r.get('license_source_url'), 300)}",
              f"  > {cut(r.get('license_verbatim'), 1500) if r.get('license_verbatim') else 'không rõ'}"]
L += ["", "## 4. Nhóm 3 — model dịch Anh → Việt chạy được trên 16 GB (chưa dịch gì; chỉ đọc card)", "",
      "Revision, thẻ giấy phép và cờ gated đã đối chiếu lại bằng API HF: khớp 44/44.", "",
      "| Repo @ revision | Tham số | Giấy phép (thẻ) | Hạn chế | Vừa 16 GB | Chất lượng en→vi đã công bố |", "|---|---|---|---|---|---|"]
for r in mt:
    q = "; ".join(f"{x.get('benchmark')} {x.get('metric')} {x.get('value')}" for x in (r.get("quality_published") or []))[:160] or "—"
    L.append(f"| {cut(r['repo'], 60)} @ `{cut(r['revision'], 8)}` | {cut(r['params'], 12)} | {cut(r['license_tag'], 30)} | "
             f"{cut(r.get('restrictions'), 160)} | {cut(r['fits_16gb'], 16)} | {q} |")
if tsp:
    L += ["", "### Tác vụ tasksource (tiếng Anh) thuộc dạng bài còn thiếu — ứng viên để dịch", "",
          "| Dạng bài | Nguồn | Dòng | Đã dùng v0 | Bị loại | Kiểu | Trung vị / p90 token cả dòng |", "|---|---|---|---|---|---|---|"]
    FALSE_POS = {"citation_intent": "ý định trích dẫn, không phải ý định người dùng",
                 "natural-language-satisfiability": "regex bắt nhầm ('atis' trong 'satisfiability')"}
    for form, d in tsp["forms"].items():
        for s, e in d.items():
            if s in FALSE_POS:
                continue
            L.append(f"| {form} | {s} | {e['rows']:,} | {e['used_v0']} | {'có' if e['excluded'] else 'không'} | {', '.join(e['kinds'])} | "
                     f"{e['tokens_median']:.0f} / {e['tokens_p90']:.0f} |")
if tsp:
    L += ["", "Bỏ khỏi bảng vì regex tên nguồn bắt nhầm: " + "; ".join(f"`{k}` ({v})" for k, v in FALSE_POS.items()) + ". "
          "`IntentGrasp/all` và `slurp/intent` là nguồn bị loại cứng (MASSIVE/SLURP)."]
(S / "tables_sources.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("written", len(L))
