"""Adapter chấm Mapika/decider-2b (revision ghim) bằng chính gói `decider/` đi kèm repo, không tự chế prompt.

Dùng: eval_decider.py <rows.jsonl> <out_dir> [--checks]
- Nạp: snapshot_download ở revision ghim; mã `decider/*.py` trong snapshot phải trùng sha256 với bản đã đọc kiểm
  (results/baselines/meta/Mapika__decider-2b). Decider(folder, device="cuda", dtype=bf16, use_graphs=False): đường eager,
  không torch.compile / CUDA graph / vá kernel. Temperature phát hành giữ nguyên (không truyền temperature=), assert giá trị.
- Mỗi câu là một yêu cầu `decide(state, [{"question", "options"}])` (một câu hỏi một hàng, batch 1). Câu "choice" dùng
  T = 1.164 phát hành. Xác suất lấy từ probs_list (không làm tròn, đúng thứ tự lựa chọn đưa vào).
- Hai cách ánh xạ lựa chọn:
    text  : mỗi lựa chọn là nguyên văn nội dung, không mô tả ("(A) <nội dung>");
    keyed : khóa trung tính ngắn + nội dung ở phần mô tả ("(A) opt1: <nội dung>") — đúng cách system_one hiển thị
            criteria {nhãn: mô tả}; tương đương được kiểm bằng so token với _system_one_items.
- Hai thứ tự: normal (thứ tự dev_v0) và reverse (đảo ngược, target = k-1-target).
- Cắt cụt: decide() mặc định cắt state ở 1.536 token (không báo). Ta đặt max_ctx_tokens = 32.768 và đếm số câu
  có "Context:\n"+state vượt 1.536 token (lẽ ra bị cắt ở mặc định) và vượt 32.768 (bị loại, không tính điểm).
- --checks: chạy ví dụ README, kiểm tương đương keyed/system_one, rồi thôi.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, read_jsonl, write_jsonl

REPO, REV = "Mapika/decider-2b", "ff99089b6810691bd79134ec6febb5d9afa6e619"
CAP_DEFAULT, CAP = 1536, 32768
rows_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
checks_only = "--checks" in sys.argv

folder = snapshot_download(REPO, revision=REV)
reviewed = RESULTS / "baselines" / "meta" / "Mapika__decider-2b"
for p in (Path(folder) / "decider").glob("*.py"):
    a = hashlib.sha256(p.read_bytes()).hexdigest()
    b = hashlib.sha256((reviewed / f"decider__{p.name}").read_bytes()).hexdigest()
    assert a == b, f"decider/{p.name} khác bản đã đọc kiểm"
sys.path.insert(0, folder)
from decider.infer import Decider  # noqa: E402

torch.manual_seed(0)
d = Decider(folder, device="cuda", dtype=torch.bfloat16, use_graphs=False)
assert abs(d.T - 1.145) < 1e-9 and d.T_by_type == {"choice": 1.164, "noul": 1.624, "score": 1.124}, (d.T, d.T_by_type)
assert d.layout in ("plain", None, "state_first") or d.chat is None, d.layout
assert d.neutralize_none is False and d.chat is None
tok = d.m.tok
meta = {"repo": REPO, "revision": REV, "T": d.T, "T_by_type": d.T_by_type, "layout": str(d.layout), "use_graphs": False,
        "dtype": "bfloat16", "max_ctx_tokens": CAP, "name": d.name}


def opts_for(options, mapping):
    return list(options) if mapping == "text" else [f"opt{i + 1}: {o}" for i, o in enumerate(options)]


if checks_only:
    out = {}
    ex = d.decide("My card was charged twice for the same purchase.",
                  [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
                   {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
    out["readme_example"] = [{"choice": r["choice"], "confidence": round(r["confidence"], 4), "probs_list": [round(x, 4) for x in r["probs_list"]]} for r in ex]
    out["readme_expected"] = "[{'choice': 'billing', 'confidence': 0.93}, {'choice': 'yes', 'confidence': 0.70}]  (v11, decider-ai 1.4.0, eager)"
    # keyed (decide với "optN: nội dung") phải cho đúng token như system_one với criteria {optN: nội dung}
    eq = []
    for r in read_jsonl(rows_path)[:5]:
        _, it_dec = d._decide_items([(r["state"], [{"question": r["question"], "options": opts_for(r["options"], "keyed")}])], CAP)
        _, _, it_sys = d._system_one_items(r["state"], {"q": {"type": "choice", "instructions": r["question"],
                                                              "criteria": {f"opt{i + 1}": o for i, o in enumerate(r["options"])}}},
                                           True, layout="state_first")
        eq.append(it_dec[0]["ids"] == it_sys[0]["ids"])
    out["keyed_equals_system_one_tokens"] = eq
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "checks_decider.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    sys.exit(0)

rows = read_jsonl(rows_path)
for mapping in ("text", "keyed"):
    for order in ("normal", "reverse"):
        preds, lat, n_over_default, n_over_cap = [], [], 0, 0
        # warmup
        for r in rows[:10]:
            d.decide(r["state"], [{"question": r["question"], "options": opts_for(r["options"], mapping)}], max_ctx_tokens=CAP)
        for r in rows:
            options = r["options"][::-1] if order == "reverse" else r["options"]
            target = len(options) - 1 - r["target"] if order == "reverse" else r["target"]
            n_ctx = len(tok.encode("Context:\n" + r["state"], add_special_tokens=False))
            over_default = n_ctx > CAP_DEFAULT; n_over_default += over_default
            if n_ctx > CAP:
                n_over_cap += 1
                preds.append({"id": r["id"], "excluded": "state over 32768 tokens"}); continue
            q = [{"question": r["question"], "options": opts_for(options, mapping)}]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            res = d.decide(r["state"], q, max_ctx_tokens=CAP)[0]
            torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
            p = res["probs_list"]
            assert len(p) == len(options) and abs(sum(p) - 1) < 1e-4, (r["id"], len(p), sum(p))
            pred = max(range(len(p)), key=p.__getitem__)
            preds.append({"id": r["id"], "pair_id": r.get("pair_id"), "source": r["source"], "lang": r["lang"], "k": len(p),
                          "target": target, "probs": p, "pred": pred, "correct": pred == target, "letter_mass": None,
                          "n_ctx_tokens": n_ctx, "over_default_cap_1536": over_default, "mapping": mapping, "reversed": order == "reverse"})
        name = f"decider-2b-{mapping}"
        write_jsonl(out_dir / order / name / "preds.jsonl", preds)
        timing = {**meta, "mapping": mapping, "order": order, "rows": len(rows), "batch1_median_latency_ms": 1000 * statistics.median(lat),
                  "batch1_p90_latency_ms": 1000 * sorted(lat)[int(0.9 * len(lat))], "latency_includes": "tokenize + collate + forward (decide call)",
                  "n_over_default_cap_1536": n_over_default, "n_excluded_over_32768": n_over_cap,
                  "hardware": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "flash_linear_attention": False}
        (out_dir / order / name / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
        acc = sum(x.get("correct", False) for x in preds) / len(preds)
        print(f"{mapping} {order}: acc {acc:.4f} median {timing['batch1_median_latency_ms']:.1f} ms over1536 {n_over_default}")
