"""Mục G (bench 1.1): file đọc kiểm 15 câu ngẫu nhiên cho mỗi nguồn thay thế ĐỀ XUẤT (chưa thêm vào benchmark).
Ra: results/review/candidates_review.html (tự chứa, cùng định dạng bench_v1_review.html), results/review/candidates_rows.json.

Ứng viên (từ results/train_v1_survey/REPORT_sources.md): bản địa, nhãn người, không trùng train_v0, không phải dạng kiểm chứng.
Chỉ lấy dữ liệu qua dataset viewer của Hugging Face (repo công khai, không gated), 15 dòng split train ở vị trí ngẫu nhiên
random.Random(f"cand|<tên>|{SEED}").sample(range(số dòng train), 15). Mọi văn bản được html.escape. Không JS, không tài nguyên ngoài.
Câu hỏi và lựa chọn tiếng Việt cho từng ứng viên là ĐỀ XUẤT của agent, chưa duyệt; nhãn hiện đúng giá trị gốc của bộ dữ liệu.
"""
import html
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa: E402

SEED = 20261013
e = lambda s: html.escape("" if s is None else str(s), quote=True)
SURVEY = {r["name"]: r for r in json.load(open(RESULTS / "train_v1_survey/raw/vi_candidates.json", encoding="utf-8"))}


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as f:
                return json.load(f)
        except Exception:
            time.sleep(2 + 3 * k)
    raise RuntimeError(url)


def survey_entry(prefix):
    return next(v for k, v in SURVEY.items() if k.startswith(prefix))


CANDS = [
    {"key": "victsd", "name": "UIT-ViCTSD", "repo": "tarudesu/ViCTSD", "config": "default", "survey": "UIT-ViCTSD",
     "paper": "Nguyen, Nguyen & Nguyen (2021), \"Constructive and Toxic Speech Detection for Open-domain Social Media Comments in Vietnamese\", arXiv:2103.10069",
     "paper_url": "https://arxiv.org/abs/2103.10069",
     "iaa": "Mục 3.3 (Dataset Evaluation): \"the inter-annotator agreement we gained for constructive and toxic speech detection is 59.48% and 58.74% IAA, respectively\" "
            "(bài báo gọi thước đo là Fleiss' Kappa nhưng báo dạng phần trăm; vòng gán nhãn đầu chỉ 21.7% và 30.4%).",
     "warn": True,
     "kind": "noul ×2 (tính xây dựng; độc hại) — nhãn nhị phân, KHÔNG có thứ bậc",
     "questions": [("Constructiveness", "Bình luận này có mang tính xây dựng không?"), ("Toxicity", "Bình luận này có độc hại không?")],
     "state": lambda r: (f"Tiêu đề bài báo: {r.get('Title')}\nChủ đề: {r.get('Topic')}\n\nBình luận: {r.get('Comment')}"),
     "access": "Repo HF của tác giả chính (card: \"official repository\"), không gated; trang nlp.uit.edu.vn ghi phải gửi email ký thỏa thuận — hai nguồn mâu thuẫn, cần người làm dự án quyết."},
    {"key": "viocd", "name": "UIT-ViOCD", "repo": "tarudesu/ViOCD", "config": "default", "survey": "UIT-ViOCD",
     "paper": "Nguyen et al. (2021), \"Vietnamese Complaint Detection on E-Commerce Websites\" (bản v1: \"Vietnamese Open-domain Complaint Detection in E-Commerce Websites\"), arXiv:2104.11969",
     "paper_url": "https://arxiv.org/abs/2104.11969",
     "iaa": "Tóm tắt (arXiv v3): \"achieve the inter-annotator agreement Am of 87%\". Bài báo không nêu rõ trong tóm tắt Am là thước đo nào (một tóm tắt bản khác ghi Fleiss' Kappa).",
     "warn": False,
     "kind": "noul (phàn nàn hay không) — nhãn nhị phân, KHÔNG có thứ bậc",
     "questions": [("label", "Đánh giá này có phải là lời phàn nàn không?")],
     "state": lambda r: f"Lĩnh vực: {r.get('domain')}\n\nĐánh giá: {r.get('review')}",
     "access": "Repo HF của tác giả (\"official repository\"), không gated; trang nlp.uit.edu.vn ghi phải gửi email — mâu thuẫn như ViCTSD. "
               "Gần với tác vụ complaint của ViSFD đã dùng ở train_v0 (khác dữ liệu, cùng dạng)."},
    {"key": "exams_vi", "name": "EXAMS (tiếng Việt)", "repo": "mhardalov/exams", "config": "crosslingual_vi", "survey": "EXAMS",
     "paper": "Hardalov et al. (2020), \"EXAMS: A Multi-Subject High School Examinations Dataset for Cross-Lingual and Multilingual Question Answering\", arXiv:2011.03080",
     "paper_url": "https://arxiv.org/abs/2011.03080",
     "iaa": "Không công bố. Đáp án lấy từ đề thi chính thức (bài báo: tách câu hỏi, lựa chọn và đáp án đúng bằng biểu thức chính quy rồi soát tay lỗi hiển thị); không có bước gán nhãn của người nên không có độ đồng thuận.",
     "warn": False,
     "kind": "choice (4 lựa chọn, câu hỏi đề thi THPT) — KHÔNG có thứ bậc; không có đoạn văn (bản kèm đoạn văn là đoạn Wikipedia truy hồi, rất dài)",
     "questions": None,
     "state": lambda r: f"Môn / khối: {r.get('info', {}).get('subject')} / {r.get('info', {}).get('grade')}",
     "access": "Công khai, không gated; thẻ cc-by-sa-4.0."},
]

CSS = (Path(RESULTS / "review/bench_v1_review.html").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0])
H = ['<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
     f'<title>Đọc kiểm nguồn đề xuất</title><style>{CSS}</style></head><body><main>',
     '<h1>Đọc kiểm nguồn thay thế đề xuất (CHƯA thêm vào benchmark)</h1>',
     f'<p class="note">Dựng bằng <span class="mono">scripts/bench11/candidates_review.py</span>. 15 dòng split train mỗi nguồn, vị trí ngẫu nhiên với seed {SEED}, '
     'lấy qua dataset viewer của Hugging Face. Câu hỏi tiếng Việt là đề xuất của agent, chưa duyệt; nhãn hiện đúng giá trị gốc. '
     '<b>Không ứng viên nào có nhãn thứ bậc</b> (ưu tiên của brief): trong khảo sát, nguồn có nhãn thứ bậc gần nhất là UIT-VSFC (cảm xúc 3 mức), đã nằm trong train_v0.</p>',
     '<nav class="toc"><b>Mục lục.</b> ' + "".join(f'<a href="#{c["key"]}">{e(c["name"])}</a>' for c in CANDS) + '</nav>']
dump = {}
for c in CANDS:
    sv = survey_entry(c["survey"])
    info = get(f"https://huggingface.co/api/datasets/{c['repo']}")
    size = get(f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(c['repo'])}")
    n = next(s["num_rows"] for s in size["size"]["splits"] if s["config"] == c["config"] and s["split"] == "train")
    offs = sorted(random.Random(f"cand|{c['key']}|{SEED}").sample(range(n), 15))
    got = []
    for o in offs:
        d = get(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(c['repo'])}&config={c['config']}&split=train&offset={o}&length=1")
        got.append((o, d["rows"][0]["row"]))
    dump[c["key"]] = {"repo": c["repo"], "sha_at_fetch": info.get("sha"), "survey_revision": sv.get("revision"), "train_rows": n, "offsets": offs,
                      "rows": [r for _, r in got]}
    H.append(f'<h2 id="{c["key"]}">{e(c["name"])}</h2>')
    if c["warn"]:
        H.append('<div class="warn"><b>Cảnh báo nội dung.</b> Bình luận mạng xã hội, có thể chứa lời lẽ độc hại (đó là một nhãn của bộ dữ liệu).</div>')
    H.append('<table>'
             f'<tr><th>Repo @ revision</th><td><span class="mono">{e(c["repo"])} @ {e(info.get("sha"))}</span>'
             f'{"" if info.get("sha") == sv.get("revision") else " (khác revision lúc khảo sát: " + e(sv.get("revision")) + ")"}</td></tr>'
             f'<tr><th>Số dòng split train</th><td>{n}</td></tr><tr><th>Dạng bài đề xuất</th><td>{e(c["kind"])}</td></tr>'
             f'<tr><th>Nhãn</th><td>{e(sv.get("label_origin"))}: {e(sv.get("label_origin_evidence"))}</td></tr>'
             f'<tr><th>Giấy phép (nguyên văn)</th><td>{e(sv.get("license_verbatim"))}<br><span class="mono">{e(sv.get("license_source_url"))}</span></td></tr>'
             f'<tr><th>Điều kiện truy cập</th><td>{e(c["access"])}</td></tr>'
             f'<tr><th>Độ đồng thuận người gán nhãn</th><td>{e(c["iaa"])}<br>{e(c["paper"])} — <span class="mono">{e(c["paper_url"])}</span></td></tr>'
             f'<tr><th>Rủi ro trùng</th><td>{e(sv.get("overlap_risk"))}</td></tr></table>')
    for k, (o, r) in enumerate(got, 1):
        iid = f"G-{c['key']}-{k:02d}"
        H.append(f'<div class="item"><div id="{iid}"><span class="iid">{iid}</span> <span class="mono">train offset {o}</span></div>'
                 f'<div class="text">{e(c["state"](r))}</div>')
        if c["questions"]:
            for field, q in c["questions"]:
                v = r.get(field)
                yes = str(v) in ("1", "1.0", "True")
                H.append(f'<div><b>Câu hỏi đề xuất (noul):</b> {e(q)}</div><ol class="opts" style="list-style:none">'
                         f'<li class="{"gold" if yes else ""}">(A) Có{" ✓ đúng" if yes else ""}</li><li class="{"" if yes else "gold"}">(B) Không{"" if yes else " ✓ đúng"}</li></ol>'
                         f'<div class="lab">nhãn gốc: {e(field)} = {e(v)}</div>')
        else:
            q = r["question"]; texts, labels = q["choices"]["text"], q["choices"]["label"]
            H.append(f'<div><b>Câu hỏi (từ nguồn):</b> {e(q["stem"])}</div><ol class="opts" style="list-style:none">' +
                     "".join(f'<li class="{"gold" if lab == r["answerKey"] else ""}">({e(lab)}) {e(t)}{" ✓ đúng" if lab == r["answerKey"] else ""}</li>' for t, lab in zip(texts, labels)) +
                     f'</ol><div class="lab">nhãn gốc: answerKey = {e(r["answerKey"])}</div>')
        H.append('</div>')
H.append('</main></body></html>')
(RESULTS / "review/candidates_review.html").write_text("\n".join(H), encoding="utf-8")
(RESULTS / "review/candidates_rows.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: {"sha": v["sha_at_fetch"][:8], "survey_rev": str(v["survey_revision"])[:8], "train_rows": v["train_rows"]} for k, v in dump.items()}))
