"""Bước E: chấm một model ngoài (Quyet / Laya) — chạy bằng .venv-baselines với HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1.
Dùng: run_external.py <khóa> checks|wiring|dev [--force]
  wiring: so xác suất của adapter với hàm predict() CÔNG KHAI của chính gói (cùng câu hỏi, cùng state) trên mọi câu hiển nhiên, hai thứ tự, hai
          cách ánh xạ: phải khớp đến 4 chữ số (predict làm tròn 4). Khớp hết nghĩa là adapter đi đúng đường của tác giả, kể cả cách gán noul theo chữ;
          câu hiển nhiên còn sai là do model. Laya-multilingual: chạy lại ví dụ README ở fp32 (amp tắt) để xem bf16 có phải nguyên nhân.
  checks: ví dụ README (so với số / nhãn card nêu), 10 câu hiển nhiên mỗi ngôn ngữ ở 2 thứ tự (câu chọn đáp án: results/baselines/obvious_v0.jsonl;
          câu Có/Không: results/dev_v1/obvious_typed.jsonl), cho từng cách ánh xạ; xác suất đủ k phần tử, tổng = 1 (harness assert).
          Ghi results/baselines_v1_2/checks/<khóa>.json. Model qua khi mọi câu hiển nhiên đúng ở cả hai thứ tự và ví dụ README khớp.
  dev   : chỉ chạy khi checks đạt (hoặc --force: vẫn chấm nhưng ghi "tin_cay": false, báo cáo không đưa vào bảng chính); DEV 1.2 (kiểm sha), hai thứ tự, từng cách ánh xạ, batch 1, có giới hạn thời gian mỗi lượt.
Không bao giờ đọc test."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "results/baselines_v1_2"
DEV = ROOT / "data/bench_v1_2/dev_v1_2.jsonl"
DEV_SHA = "ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b"
MODELS = {"quyet-tiny": ("chinhnc/Quyet-1.0-Tiny", "Quyet", 3600), "quyet-small": ("chinhnc/Quyet-1.0-Small", "Quyet", 3600),
          "quyet-medium": ("chinhnc/Quyet-1.0-Medium", "Quyet", 10800), "laya-multilingual": ("convaiinnovations/laya-multilingual", "Laya", 3600),
          "laya-vi": ("vanty120/laya-vi", "Laya", 3600)}
key, stage = sys.argv[1], sys.argv[2]
assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "phải chạy offline"
repo, kind, limit_s = MODELS[key]
snap = Path(json.load(open(B / "downloads.json", encoding="utf-8"))[repo]["snapshot"])
if kind == "Laya":      # laya có thể ghi lại tokenizer_config.json trong thư mục nạp: dùng bản sao có quyền ghi, không đụng HF cache
    dst = Path.home() / "baselines_v1_2_snap" / repo.replace("/", "__")
    if not dst.exists():
        shutil.copytree(snap, dst, symlinks=False)
    snap = dst
import adapters as A  # noqa: E402

MAPPINGS = ["text", "keyed"]
(B / "checks").mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def load(mapping):
    return getattr(A, kind)(snap, mapping)


def obvious_rows():
    ch = H.read_rows(ROOT / "results/baselines/obvious_v0.jsonl")
    no = [r for r in H.read_rows(ROOT / "results/dev_v1/obvious_typed.jsonl") if r["kind"] == "noul"]
    return ch + no


if stage == "checks":
    out = {"model": repo, "revision": snap.name if kind == "Quyet" else json.load(open(B / "downloads.json"))[repo]["revision"], "mappings": {}}
    ad = load("text")
    # ví dụ README
    if kind == "Quyet":
        ex = ad.m.predict({"message": "Please close my card, I lost it yesterday."},
                          {"intent": {"type": "choice", "instructions": "What does the customer want?",
                                      "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
                           "urgent": {"type": "noul", "instructions": "The request is urgent."},
                           "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}})
        a = ex["answers"]
        ok = (list(a["intent"]["probabilities"]) == ["cancel", "limit", "other"] and abs(sum(a["intent"]["probabilities"].values()) - 1) < 1e-3
              and "noul" in a["urgent"] and "legend" in a["mood"])
        out["readme"] = {"pass": ok, "kiem": "hình dạng kết quả (card không nêu số)", "intent": a["intent"]["probabilities"], "urgent": a["urgent"]["noul"],
                         "mood": a["mood"].get("score"), "intent_choice": a["intent"]["choice"]}
    elif repo.endswith("laya-multilingual"):
        ex = ad.a.predict({"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।"},
                          {"department": {"type": "choice", "instructions": "Which team should handle `body`?",
                                          "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages", "sales": "pricing"}},
                           "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"}})
        a = ex["answers"]
        out["readme"] = {"pass": a["department"]["choice"] == "billing", "kiem": "card nêu nhãn billing", "department": a["department"]["probabilities"],
                         "refund_requested": a["refund_requested"]["noul"]}
    else:
        st = ("Chào shop, tôi đã chuyển khoản 2 lần cho đơn hàng #4521 tháng 3. "
              "Nhờ shop hoàn lại khoản bị trừ trùng trong hôm nay, nếu không tôi sẽ huỷ tài khoản.")
        qs = {"bo_phan": {"type": "choice", "instructions": "Bộ phận nào nên xử lý yêu cầu này?",
                          "criteria": {"thanh_toan": "hoá đơn, thanh toán, hoàn tiền", "ky_thuat": "lỗi, sự cố, hệ thống", "khac": "các vấn đề khác"}},
              "muc_do_khan": {"type": "score", "instructions": "Yêu cầu này khẩn cấp đến mức nào?", "criteria": ["không gấp", "sớm", "rất gấp"]},
              "nguy_co_roi_bo": {"type": "noul", "instructions": "Khách hàng có đe doạ huỷ hoặc rời bỏ dịch vụ không?"}}
        a = ad.a.predict(st, qs)["answers"]
        got = {"thanh_toan": a["bo_phan"]["probabilities"]["thanh_toan"], "score": a["muc_do_khan"]["score"],
               "p_rat_gap": a["muc_do_khan"]["probabilities"]["2"], "noul": a["nguy_co_roi_bo"]["noul"]}
        want = {"thanh_toan": 0.96, "score": 1.71, "p_rat_gap": 0.75, "noul": 0.94}
        out["readme"] = {"pass": a["bo_phan"]["choice"] == "thanh_toan" and all(abs(got[k] - want[k]) <= 0.03 for k in want),
                         "kiem": "card nêu số làm tròn, dung sai ±0,03", "got": got, "want": want}
    rows = obvious_rows()
    probs_by_map = {}
    for mapping in MAPPINGS:
        ad_m = ad if mapping == "text" else load(mapping)
        res = H.run(rows, ad_m.score, B / "checks/obvious", f"{key}-{mapping}", meta={"model": repo, "mapping": mapping}, log=log)
        P = {o: {json.loads(l)["id"]: json.loads(l) for l in open(B / f"checks/obvious/{o}/{key}-{mapping}/preds.jsonl", encoding="utf-8")}
             for o in ("normal", "reverse")}
        fail = [i for i in P["normal"] if not (P["normal"][i]["correct"] and P["reverse"][i]["correct"])]
        by = {f"{r['kind']}|{r['lang']}": sum(1 for x in rows if x["kind"] == r["kind"] and x["lang"] == r["lang"]) for r in rows}
        out["mappings"][mapping] = {"pass": not fail, "sai": fail, "so_cau": by}
        probs_by_map[mapping] = {i: P["normal"][i]["probs"] for i in P["normal"]}
        if mapping != "text":
            del ad_m
    if key == "quyet-medium":   # Medium ẩn khóa khi có mô tả: hai cách ánh xạ phải cho cùng prompt -> cùng xác suất
        same = all(probs_by_map["text"][i] == probs_by_map["keyed"][i] for i in probs_by_map["text"])
        out["medium_keyed_bang_text"] = same
    out["pass"] = out["readme"]["pass"] and any(v["pass"] for v in out["mappings"].values())
    (B / f"checks/{key}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: (v if k != "mappings" else {m: {"pass": x["pass"], "sai": len(x["sai"])} for m, x in v.items()}) for k, v in out.items()},
                   ensure_ascii=False))
elif stage == "wiring":
    import torch
    ck = json.load(open(B / f"checks/{key}.json", encoding="utf-8"))
    rows = obvious_rows()
    res = {}
    for mapping in MAPPINGS:
        ad = load(mapping)
        worst, n = 0.0, 0
        for r0 in rows:
            for rr in (r0, H.reverse_row(r0)):
                s_ = ad.score(rr)
                if rr["kind"] == "noul":
                    qdef = {"type": "noul", "instructions": rr["question"]}
                else:
                    qdef = {"type": "choice", "instructions": rr["question"], "criteria": A.criteria_of(rr["options"], mapping)}
                if kind == "Quyet":
                    ans = ad.m.predict(rr["state"], {"q": qdef})["answers"]["q"]
                else:
                    ans = ad.a.predict(rr["state"], {"q": qdef}, max_len=ad.MAX_LEN, head_max_len=ad.HEAD_MAX)["answers"]["q"]
                if rr["kind"] == "noul":
                    it, _ = A.noul_map(rr["options"]); pub = [None, None]; pub[it] = ans["noul"]; pub[1 - it] = round(1 - ans["noul"], 4)
                    diff = abs(s_["probs"][it] - ans["noul"])
                else:
                    pub = list(ans["probabilities"].values())
                    diff = max(abs(a_ - b_) for a_, b_ in zip(s_["probs"], pub))
                worst = max(worst, diff); n += 1
        res[mapping] = {"so_lan_goi": n, "lech_lon_nhat": worst, "khop_4_chu_so": worst <= 6e-5}
        del ad; torch.cuda.empty_cache()
    ck["wiring"] = res
    if key == "laya-multilingual":
        ad = load("text"); ad.a.amp_enabled = False
        ex = ad.a.predict({"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।"},
                          {"department": {"type": "choice", "instructions": "Which team should handle `body`?",
                                          "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages", "sales": "pricing"}}})
        ck["readme_fp32"] = {"department": ex["answers"]["department"]["probabilities"], "choice": ex["answers"]["department"]["choice"]}
    (B / f"checks/{key}.json").write_text(json.dumps(ck, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({"wiring": res, "readme_fp32": ck.get("readme_fp32")}, ensure_ascii=False))
elif stage == "dev":
    ck = json.load(open(B / f"checks/{key}.json", encoding="utf-8"))
    FORCE = "--force" in sys.argv
    assert hashlib.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA
    rows = H.read_rows(DEV)
    maps = [m for m in MAPPINGS if ck["mappings"][m]["pass"] or FORCE]
    if key == "quyet-medium" and ck.get("medium_keyed_bang_text") and "text" in maps:
        maps = ["text"]          # keyed cho đúng cùng prompt (đã kiểm), không chạy lặp
    assert (ck["readme"]["pass"] and maps) or FORCE, "adapter chưa đáng tin: không chấm dev (dùng --force để chấm mà không đưa vào bảng chính)"
    for mapping in maps:
        ad = load(mapping)
        tin_cay = bool(ck["readme"]["pass"] and ck["mappings"][mapping]["pass"])
        H.run(rows, ad.score, B / "eval", f"{key}-{mapping}", meta={"model": repo, "mapping": mapping, "dev_sha": DEV_SHA, "tin_cay": tin_cay},
              limit_s=limit_s, log=log)
        del ad
        import torch
        torch.cuda.empty_cache()
