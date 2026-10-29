"""G0a (BRIEF_buoc0_G0.md): đo độ trễ batch 1 trên dev 1.3 cùng một cách cho mọi model, đường eager và đường CUDA graphs. Không train, không test.

Eager    : đúng đường của eval_base.py: ids đã nằm trên GPU, một forward với logits_to_keep=1, đồng bộ, đo thời gian.
CUDA graph: như engine của decider-2b: right padding tới nhóm độ dài cố định T_BUCKETS, KHÔNG cần attention mask (các lớp đều nhân quả nên
           token đệm ở sau không ảnh hưởng vị trí trước), một forward của backbone rồi lm_head trên đúng vị trí token thật cuối (tương đương
           logits_to_keep=1). Mỗi nhóm độ dài bắt một graph (dùng chung graph pool), bắt TRƯỚC khi đo. Thời gian đo gồm: chép ids vào bộ đệm tĩnh,
           replay, đồng bộ. Câu dài hơn 2.048 token chạy eager (ghi riêng).
Cổng đúng: trên toàn dev, argmax của đường graph trùng đường eager 100% và |xác suất (softmax trên k chữ cái) graph − eager| <= 1e-3.
           Không đạt: ghi độ lệch lớn nhất và đánh dấu "khong_dung_so_do_graph".
Lặp: mỗi đường đo --repeats lần (mặc định 2) trong cùng tiến trình; trung vị hoặc p90 của hai lần lệch > 10% thì đo thêm lần 3.
Báo: trung vị, p90, p95, p99 toàn dev; p90 theo nhóm độ dài prompt (token của chính model): <=128, 129–256, 257–512, 513–1024, >1024.
Dùng: latency_g0.py <model_id> --revision R [--adapter DIR] [--chat_template_from repo@rev] --name TÊN --out_dir DIR [--rows JSONL]"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, DATA, LETTERS, load_json, read_jsonl, render_prompt, sha256_file

T_BUCKETS = [64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048]
GROUPS = [(0, 128), (129, 256), (257, 512), (513, 1024), (1025, 10 ** 9)]
ap = argparse.ArgumentParser()
ap.add_argument("model_id")
ap.add_argument("--revision", required=True)
ap.add_argument("--adapter", type=Path)
ap.add_argument("--chat_template_from")
ap.add_argument("--name", required=True)
ap.add_argument("--out_dir", type=Path, required=True)
ap.add_argument("--rows", type=Path, default=DATA / "bench_v1_3" / "dev_v1_3.jsonl")
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--modes", default="eager,graphs")
a = ap.parse_args()
assert "test" not in a.rows.name.lower()
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False

frames = load_json(CONFIG / "prompt_frames.json")
rows = read_jsonl(a.rows)
tok = AutoTokenizer.from_pretrained(a.model_id, revision=a.revision)
tsrc = None
if a.chat_template_from and not tok.chat_template:
    sid, srev = a.chat_template_from.split("@")
    tok.chat_template = AutoTokenizer.from_pretrained(sid, revision=srev).chat_template; tsrc = a.chat_template_from
model = AutoModelForCausalLM.from_pretrained(a.model_id, revision=a.revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload().eval().to(torch.bfloat16)
letter_ids = [tok.convert_tokens_to_ids(L) for L in LETTERS]
assert all(len(tok.encode(L, add_special_tokens=False)) == 1 for L in LETTERS)
backbone, lm_head = model.model, model.lm_head
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc = [tok.encode(render_prompt(r, tok, frames), add_special_tokens=False) for r in rows]
lens = [len(e) for e in enc]


def bucket(n):
    return next((b for b in T_BUCKETS if b >= n), None)


def probs_of(lg, k):
    return torch.softmax(lg[letter_ids[:k]].float(), -1).cpu().numpy()


@torch.inference_mode()
def eager_pass():
    lat, P = [], []
    for e, r in zip(enc, rows):
        x = torch.tensor([e], device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg = model(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1]
        torch.cuda.synchronize(); lat.append(1000 * (time.perf_counter() - t0))
        P.append(probs_of(lg, len(r["options"])))
    return lat, P


@torch.inference_mode()
def eager_pad_pass():
    """Chẩn đoán (điều phối 28/10): đúng hàm forward và đúng pad phải tới nhóm T như đường graphs, nhưng KHÔNG bắt graph.
    So với eager không pad = sàn nhiễu do hình dạng tensor; so với graphs = kiểm riêng việc bắt graph."""
    lat, P = [], []
    for e, r in zip(enc, rows):
        n = len(e); T = bucket(n) or n
        ids = torch.full((1, T), pad, dtype=torch.long, device="cuda"); ids[0, :n] = torch.tensor(e, device="cuda")
        idx = torch.tensor([n - 1], device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg = lm_head(backbone(input_ids=ids, use_cache=False).last_hidden_state[0].index_select(0, idx))[0]
        torch.cuda.synchronize(); lat.append(1000 * (time.perf_counter() - t0))
        P.append(probs_of(lg, len(r["options"])))
    return lat, P


def compare(Pa, Pb):
    d = [float(np.abs(x - y).max()) for x, y in zip(Pa, Pb)]
    same = [int(np.argmax(x) == np.argmax(y)) for x, y in zip(Pa, Pb)]
    return {"argmax_trung": sum(same) / len(same), "so_cau_lech_argmax": len(same) - sum(same), "lech_xac_suat_max": max(d),
            "so_cau_lech_qua_1e-3": int(sum(x > 1e-3 for x in d))}


graphs, pool = {}, None
capture_err = None


@torch.inference_mode()
def capture(T):
    global pool
    pool = pool or torch.cuda.graph_pool_handle()
    s_ids = torch.full((1, T), pad, dtype=torch.long, device="cuda")
    s_idx = torch.zeros(1, dtype=torch.long, device="cuda")
    fwd = lambda: lm_head(backbone(input_ids=s_ids, use_cache=False).last_hidden_state[0].index_select(0, s_idx))[0]
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fwd()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        s_out = fwd()
    return s_ids, s_idx, s_out, g


@torch.inference_mode()
def graph_pass():
    lat, P, eager_long = [], [], 0
    for e, r in zip(enc, rows):
        n = len(e); T = bucket(n)
        if T is None:                                       # > 2.048 token: eager
            x = torch.tensor([e], device="cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lg = model(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1]
            torch.cuda.synchronize(); lat.append(1000 * (time.perf_counter() - t0)); eager_long += 1
            P.append(probs_of(lg, len(r["options"]))); continue
        s_ids, s_idx, s_out, g = graphs[T]
        x = torch.tensor(e, device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        s_ids.fill_(pad); s_ids[0, :n].copy_(x); s_idx.fill_(n - 1)
        g.replay()
        torch.cuda.synchronize(); lat.append(1000 * (time.perf_counter() - t0))
        P.append(probs_of(s_out, len(r["options"])))
    return lat, P, eager_long


def stats(lat):
    v = np.array(lat)
    out = {"n": len(v), "trung_vi": float(np.median(v)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
           "p99": float(np.percentile(v, 99)), "trung_binh": float(v.mean())}
    out["p90_theo_nhom_do_dai"] = {}
    for lo, hi in GROUPS:
        sel = [x for x, n in zip(lat, lens) if lo <= n <= hi]
        key = f">{lo - 1}" if hi > 10 ** 8 else f"{lo}-{hi}"
        out["p90_theo_nhom_do_dai"][key] = {"n": len(sel), "p90": float(np.percentile(sel, 90)) if sel else None,
                                            "trung_vi": float(np.median(sel)) if sel else None}
    return out


def run_mode(fn):
    passes, P = [], None
    for i in range(max(1, min(2, a.repeats))):
        res = fn()
        passes.append(stats(res[0])); P = res[1]
        if len(res) > 2:
            passes[-1]["cau_chay_eager_vi_qua_2048"] = res[2]
        if i == 1:
            m0, m1 = passes[0], passes[1]
            if max(abs(m0[k] - m1[k]) / min(m0[k], m1[k]) for k in ("trung_vi", "p90")) > 0.10:
                res = fn(); passes.append(stats(res[0])); P = res[1]
                if len(res) > 2:
                    passes[-1]["cau_chay_eager_vi_qua_2048"] = res[2]
            break
    agg = {k: float(np.mean([p[k] for p in passes])) for k in ("trung_vi", "p90", "p95", "p99")}
    return {"cac_lan": passes, "trung_binh_cac_lan": agg, "so_lan": len(passes),
            "do_tre_tung_cau_lan_cuoi": {r["id"]: round(x, 3) for r, x in zip(rows, res[0])}}, P   # thêm 28/10: để so tập con với toàn dev


modes = a.modes.split(",")
res = {"model": a.model_id, "revision": a.revision, "adapter": str(a.adapter) if a.adapter else None, "chat_template_from": tsrc,
       "rows_file": str(a.rows), "rows_sha256": sha256_file(a.rows), "so_cau": len(rows),
       "do_dai_prompt": {"trung_vi": float(np.median(lens)), "p90": float(np.percentile(lens, 90)), "p95": float(np.percentile(lens, 95)),
                         "max": int(max(lens)), "so_cau_qua_2048": int(sum(n > 2048 for n in lens))},
       "fla": None, "env": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
                            "gpu": torch.cuda.get_device_name(0)}}
try:
    import fla
    res["fla"] = fla.__version__
except Exception as e:  # noqa: BLE001
    res["fla"] = f"không có ({type(e).__name__})"
with torch.inference_mode():                                # warmup chung
    for e in enc[:10]:
        model(input_ids=torch.tensor([e], device="cuda"), logits_to_keep=1, use_cache=False)
torch.cuda.synchronize()
P_eager = None
if "eager" in modes:
    res["eager"], P_eager = run_mode(eager_pass)
P_epad = None
if "eager_pad" in modes:
    res["eager_pad"], P_epad = run_mode(eager_pad_pass)
    if P_eager is not None:
        res["so_sanh_eager_khong_pad_vs_eager_pad"] = compare(P_eager, P_epad)
if "graphs" in modes:
    t0 = time.time()
    try:
        for T in sorted({bucket(n) for n in lens if bucket(n)}):
            graphs[T] = capture(T)
        res["graphs_bat"] = {"so_graph": len(graphs), "giay": time.time() - t0, "nhom": sorted(graphs)}
        res["graphs"], P_graph = run_mode(graph_pass)
        if P_epad is not None:
            res["so_sanh_eager_pad_vs_graphs"] = compare(P_epad, P_graph)
        if P_eager is not None:
            d = [float(np.abs(x - y).max()) for x, y in zip(P_eager, P_graph)]
            same = [int(np.argmax(x) == np.argmax(y)) for x, y in zip(P_eager, P_graph)]
            # chẩn đoán (28/10): độ lệch từng câu kèm độ dài thật và nhóm pad, để xem câu lệch có tập trung ở câu bị pad nhiều không
            res["cong_dung_tung_cau"] = [{"id": r["id"], "n": n, "T": bucket(n), "pad": (bucket(n) or n) - n, "lech": round(x, 5),
                                          "argmax_trung": bool(s_)} for r, n, x, s_ in zip(rows, lens, d, same)]
            res["cong_dung"] = {"argmax_trung": sum(same) / len(same), "so_cau_lech_argmax": len(same) - sum(same),
                                "lech_xac_suat_max": max(d), "so_cau_lech_qua_1e-3": int(sum(x > 1e-3 for x in d)),
                                "dat": sum(same) == len(same) and max(d) <= 1e-3}
            if not res["cong_dung"]["dat"]:
                res["graphs"]["CANH_BAO"] = "khong_dung_so_do_graph: không qua cổng đúng"
    except Exception as e:  # noqa: BLE001
        import traceback
        res["graphs_loi"] = traceback.format_exc()[-3000:]
res["vram_dinh_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
a.out_dir.mkdir(parents=True, exist_ok=True)
(a.out_dir / f"latency_{a.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
short = {m: res[m]["trung_binh_cac_lan"] for m in ("eager", "graphs") if m in res}
print(json.dumps({"name": a.name, **short, "cong_dung": res.get("cong_dung"), "graphs_loi": bool(res.get("graphs_loi"))}, ensure_ascii=False))
