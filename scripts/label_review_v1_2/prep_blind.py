"""Soát nhãn dev 1.2 (việc 2), bước chuẩn bị: các cặp trong results/baselines_v1_2/ung_vien_soat_nhan.jsonl có ít nhất một dòng VI có dấu hoặc EN là
ứng viên. Ghi gói MÙ cho người soát (không có đáp án chuẩn, không phiếu model, không đáp án đa số, không tên model):
results/label_review_v1_2/blind/batch_<k>.jsonl. Không sửa dev, test, bench hay config. Python Windows.
  prep_blind.py blind          -> gói mù (bước A)
  prep_blind.py gold <k>       -> đáp án chuẩn lô k cho bước B (chỉ chạy SAU khi lô k xong bước A): results/label_review_v1_2/gold/batch_<k>.jsonl
  prep_blind.py qc             -> 15 cặp ngẫu nhiên (seed cố định) cho người soát thứ hai: results/label_review_v1_2/qc/blind.jsonl (+ gold khi gọi 'qc_gold')
  prep_blind.py qc_gold"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/bench_v1_2/dev_v1_2.jsonl"
DEV11 = ROOT / "data/bench_v1_1/dev_v1_1.jsonl"
DEV_SHA = "ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b"
OUT = ROOT / "results/label_review_v1_2"
LET = "ABCDEFGHIJ"
N_BATCH, QC_N, QC_SEED = 5, 15, 20261016
assert hashlib.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA, "dev 1.2 khác sha"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip()}
cand = [json.loads(l) for l in open(ROOT / "results/baselines_v1_2/ung_vien_soat_nhan.jsonl", encoding="utf-8") if l.strip()]
V11 = {json.loads(l)["id"]: json.loads(l) for l in open(DEV11, encoding="utf-8") if l.strip()}
PAIRS = sorted({c["pair_id"] for c in cand if c["variant"] == "orig"}, key=lambda p: (rows[p + ":vi"]["source"], p))
DATASET = {
    "belebele": "Belebele: đọc hiểu trắc nghiệm 4 lựa chọn. Đoạn văn lấy từ FLORES-200 (Wikinews, Wikivoyage, Wikibooks); câu hỏi và lựa chọn do người viết "
                "bằng tiếng Anh. Bản tiếng Việt là bản trong benchmark của dự án.",
    "sib200": "SIB-200: phân loại chủ đề cho một câu của FLORES-200, 7 nhãn. Nhãn gán theo chủ đề của BÀI gốc chứa câu đó, nên một câu lẻ có thể không đủ "
              "để suy ra chủ đề.",
    "massive": "MASSIVE: câu nói với trợ lý ảo, nhãn là ý định (intent) do người gán. Các lựa chọn là mô tả ý định; bản tiếng Việt của câu nói là của bộ "
               "MASSIVE vi-VN, mô tả ý định tiếng Việt do dự án dịch và người làm dự án duyệt.",
    "xcopa": "XCOPA (bản tiếng Việt của COPA): cho một tiền đề, chọn nguyên nhân hoặc kết quả hợp lý hơn trong hai lựa chọn.",
    "vifactcheck": "ViFactCheck: kiểm chứng nhận định tiếng Việt dựa trên đoạn bằng chứng lấy từ bài báo; nhãn gốc là được xác nhận / bị bác bỏ / "
                   "không đủ thông tin. Chỉ có tiếng Việt.",
    "viwikifc": "ViWikiFC: kiểm chứng nhận định tiếng Việt dựa trên đoạn Wikipedia tiếng Việt; nhãn gốc là được xác nhận / bị bác bỏ / không đủ thông "
                "tin. Chỉ có tiếng Việt.",
}


def view(r):
    return {"state": r["state"], "question": r["question"], "options": {LET[j]: o for j, o in enumerate(r["options"])}}


def blind_item(p):
    vi, en = rows[p + ":vi"], rows.get(p + ":en")
    return {"item_id": p, "nguon": vi["source"], "kieu": vi["kind"], "mo_ta_dataset": DATASET[vi["source"]], "vi": view(vi), "en": view(en) if en else None}


def gold_item(p):
    vi, en = rows[p + ":vi"], rows.get(p + ":en")
    o = V11.get(p + ":vi")
    return {"item_id": p, "dap_an_chuan_vi": LET[vi["target"]], "dap_an_chuan_en": LET[en["target"]] if en else None,
            "nhan_goc_dataset": LET[o["target"]] if o else None, "dap_an_da_bi_doi_o_1_2": vi["gold_provenance"] == "overridden",
            "ban_vi_da_sua_o_1_2": vi["vi_revised_fields"]}


def write(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")


mode = sys.argv[1]
chunks = [PAIRS[i * len(PAIRS) // N_BATCH:(i + 1) * len(PAIRS) // N_BATCH] for i in range(N_BATCH)]
qc = sorted(random.Random(QC_SEED).sample(PAIRS, QC_N))
if mode == "blind":
    for k, ch in enumerate(chunks, 1):
        write(OUT / f"blind/batch_{k}.jsonl", [blind_item(p) for p in ch])
    (OUT / "pairs.json").write_text(json.dumps({"dev_sha": DEV_SHA, "so_cap": len(PAIRS), "lo": {k: ch for k, ch in enumerate(chunks, 1)}, "qc": qc,
                                                "qc_seed": QC_SEED}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"so_cap": len(PAIRS), "lo": [len(c) for c in chunks], "nguon_lo": [sorted({rows[p + ':vi']['source'] for p in c}) for c in chunks]}, ensure_ascii=False))
elif mode == "gold":                 # gold <k>: chỉ tạo đáp án chuẩn cho lô k, sau khi lô đó xong bước A
    k = int(sys.argv[2]); ch = chunks[k - 1]
    a_done = {json.loads(l)["item_id"] for l in open(OUT / f"A/batch_{k}.jsonl", encoding="utf-8") if l.strip()}
    assert a_done == set(ch), f"lô {k} chưa xong bước A ({len(a_done)}/{len(ch)})"
    write(OUT / f"gold/batch_{k}.jsonl", [gold_item(p) for p in ch])
    print(f"gold lô {k}: {len(ch)} mục")
elif mode == "qc":
    write(OUT / "qc/blind.jsonl", [blind_item(p) for p in qc])
    print(json.dumps({"qc": qc}, ensure_ascii=False))
elif mode == "qc_gold":
    assert {json.loads(l)["item_id"] for l in open(OUT / "qc/A.jsonl", encoding="utf-8") if l.strip()} == set(qc), "người soát thứ hai chưa xong bước A"
    write(OUT / "qc/gold.jsonl", [gold_item(p) for p in qc])
    print("qc gold written")
