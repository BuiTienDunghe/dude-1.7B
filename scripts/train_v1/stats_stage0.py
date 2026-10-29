"""Giai đoạn 0 train_v1: thống kê độ dài token (prompt dựng đúng như lúc train: render_prompt + prompt_frames.json, tokenizer Qwen3-1.7B ghim)
và ước micro-batch / bộ nhớ GPU / thời gian cho A, B, C. Không nạp model.
- Phần B chưa dịch: đo bản tiếng Anh; ước bản tiếng Việt bằng tỉ lệ token VI/EN ĐO trên các cặp song song của dev 1.3 (Belebele, SIB-200, MASSIVE, XCOPA).
- Mô phỏng chia batch đúng train_lora.py: rng(seed*1000+epoch).permutation, batch hiệu dụng 32 sắp theo độ dài, micro-batch 8, right padding.
- Bộ nhớ: lần train v0 (Qwen3-1.7B, micro-batch 8) đo đỉnh 10,25 GB; ước phần cố định ~3,7 GB (trọng số bf16 3,44 GB + LoRA/Adam) và cho phần còn
  lại tỉ lệ với số token của micro-batch lớn nhất. Đây là ước lượng thô.
Ra: results/train_v1/stage0_lengths.json."""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, load_json, model_revisions, read_jsonl, render_prompt  # noqa: E402

OUT = RESULTS / "train_v1"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=model_revisions()["Qwen/Qwen3-1.7B"])
frames = load_json(CONFIG / "prompt_frames.json")
plen = lambda r: len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))
slen = lambda s: len(tok.encode(s, add_special_tokens=False))
partA = read_jsonl(OUT / "part_a.jsonl")
partB = read_jsonl(OUT / "part_b_en.jsonl")
idsA = {r["id"] for r in partA}
base = [r for r in read_jsonl(OUT / "components/A.jsonl") if r["id"] not in idsA]

# tỉ lệ token VI/EN trên cặp song song dev 1.3
dev = read_jsonl(DATA / "bench_v1_3/dev_v1_3.jsonl")
by = {}
for r in dev:
    if r["variant"] == "orig":
        by.setdefault(r["pair_id"], {})[r["lang"]] = r
ratio_p, ratio_s, per_src = [], [], {}
for p, d in by.items():
    if "en" in d and "vi" in d:
        rp = plen(d["vi"]) / plen(d["en"]); rs = slen(d["vi"]["state"]) / max(1, slen(d["en"]["state"]))
        ratio_p.append(rp); ratio_s.append(rs); per_src.setdefault(d["vi"]["source"], []).append(rp)
R_P, R_S = float(np.median(ratio_p)), float(np.median(ratio_s))


def desc(x):
    x = np.asarray(x)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": int(x.max()), "gt_512": int((x > 512).sum()), "gt_1024": int((x > 1024).sum())}


L = {"base_v0": [plen(r) for r in base], "A": [plen(r) for r in partA], "B_en": [plen(r) for r in partB]}
L["B_vi_uoc"] = [int(round(x * R_P)) for x in L["B_en"]]
stateB_vi = [slen(r["state"]) * R_S for r in partB]
comp = {"v0": L["base_v0"], "A": L["base_v0"] + L["A"], "B_uoc": L["base_v0"] + L["B_vi_uoc"], "C_uoc": L["base_v0"] + L["A"] + L["B_vi_uoc"]}


def simulate(lens, seed, epochs=2, eff=32, micro=8):
    lens = np.asarray(lens); steps = len(lens) // eff; mb = []
    for ep in range(epochs):
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(lens))
        for b in range(steps):
            s = np.sort(lens[order[b * eff:(b + 1) * eff]])
            for m in range(eff // micro):
                mb.append(micro * int(s[m * micro:(m + 1) * micro].max()))
    mb = np.asarray(mb)
    return {"micro_batch_token_max": int(mb.max()), "p99": float(np.percentile(mb, 99)), "mean": round(float(mb.mean()), 1), "padded_tokens_total": int(mb.sum())}


sim = {k: {f"seed{s}": simulate(v, s) for s in (1, 2)} for k, v in comp.items()}
FIXED, PEAK0, SEC0 = 3.7, 10.25, 6618.6
t0 = sim["v0"]["seed1"]["micro_batch_token_max"]; pad0 = sim["v0"]["seed1"]["padded_tokens_total"]
est = {}
for k, s in sim.items():
    tm = max(s[x]["micro_batch_token_max"] for x in s)
    est[k] = {"rows": len(comp[k]), "vram_uoc_gb_micro8": round(FIXED + (PEAK0 - FIXED) * tm / t0, 1),
              "micro4_token_max": tm // 2, "vram_uoc_gb_micro4": round(FIXED + (PEAK0 - FIXED) * tm / 2 / t0, 1),
              "gio_uoc_2_epoch": round(SEC0 * s["seed1"]["padded_tokens_total"] / pad0 / 3600, 2)}
res = {"tokenizer": "Qwen/Qwen3-1.7B@" + model_revisions()["Qwen/Qwen3-1.7B"], "ty_le_vi_en": {"prompt_median": R_P, "state_median": R_S,
       "prompt_p90": float(np.percentile(ratio_p, 90)), "so_cap": len(ratio_p), "theo_nguon": {s: float(np.median(v)) for s, v in per_src.items()}},
       "prompt_tokens": {k: desc(v) for k, v in L.items()}, "prompt_tokens_thanh_phan": {k: desc(v) for k, v in comp.items()},
       "B_state_vi_uoc_gt_1024": int(sum(x > 1024 for x in stateB_vi)), "mo_phong_batch": sim, "uoc_luong": est,
       "theo_nguon_A": {s: desc([plen(r) for r in partA if r["source"] == s]) for s in sorted({r["source"] for r in partA})},
       "theo_nguon_B_en": {s: desc([plen(r) for r in partB if r["source"] == s]) for s in sorted({r["source"] for r in partB})},
       "gia_dinh": {"phan_co_dinh_gb": FIXED, "dinh_v0_gb": PEAK0, "thoi_gian_v0_s": SEC0,
                    "ghi_chu": "bộ nhớ ~ cố định + hoạt hóa tỉ lệ token micro-batch lớn nhất; thời gian tỉ lệ tổng token đã đệm; đều là ước lượng thô"}}
(OUT / "stage0_lengths.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"ty_le": res["ty_le_vi_en"], "uoc": est, "tp": res["prompt_tokens_thanh_phan"]}, ensure_ascii=False))
