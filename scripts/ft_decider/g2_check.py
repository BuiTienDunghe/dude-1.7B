"""Cổng G2 (+ soi ánh xạ): ở bước 0 (LoRA khởi tạo, B = 0), dự đoán từ code train (dcommon.build_lora_model + forward_full +
readout + row_dist) phải trùng dự đoán của đường chấm decider (Decider.decide cho choice; _system_one_items/_system_one_probs như
eval_decider_v1 cho noul/score) trên 200 dòng đầu của val_v0. Ghi thêm: mass bước 0 theo kiểu câu, loss bước 0, 5 ví dụ ánh xạ mỗi
kiểu (mỗi trường cắt 200 ký tự), kiểm chiều noul, kiểm bất biến của score khi đảo thứ tự mức. Ra: results/ft_decider/gates/g2.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, read_jsonl  # noqa: E402
import dcommon as D  # noqa: E402

folder = D.load_folder()
from decider import temperature as TT  # noqa: E402
from decider.infer import Decider  # noqa: E402
from decider.systemone import combine_isolated  # noqa: E402

torch.manual_seed(0)
d = Decider(folder, device="cuda", dtype=torch.bfloat16, use_graphs=False)
assert d.T_by_type == {"choice": 1.164, "noul": 1.624, "score": 1.124} and d.neutralize_none is False and d.chat is None and d.isolated_levels
M = D.build_lora_model(folder, 1)
M["model"].eval()
pad = d.m.tok.pad_token_id
OUT = RESULTS / "ft_decider" / "gates"; OUT.mkdir(parents=True, exist_ok=True)


def eval_path(m):
    """Đúng phép tính của scripts/bench/eval_decider_v1.py, trả về thứ tự lựa chọn của decider (noul: [no, yes])."""
    if m["kind"] == "choice":
        return d.decide(m["state"], [{"question": m["question"], "options": m["options"]}], max_ctx_tokens=D.CAP)[0]["probs_list"]
    spec = {"type": "noul", "instructions": m["question"]} if m["kind"] == "noul" else \
        {"type": "score", "instructions": m["question"], "criteria": m["levels"]}
    _, _, items = d._system_one_items(m["state"], {"q": spec}, True, D.CAP, "state_first", None)
    flat = d._system_one_probs(items, "state_first", 65536, TT.for_items(d.T, d.T_by_type, items))
    if m["kind"] == "noul":
        p_no, p_yes = flat[0][:2]; s = p_no + p_yes
        return [p_no / s, p_yes / s]
    return list(combine_isolated([float(x[1]) for x in flat])[0])


@torch.no_grad()
def train_path(ms, batch=8):
    out = []
    for s in range(0, len(ms), batch):
        chunk = ms[s:s + batch]
        preps = [(m, D.items_for(d, m)) for m in chunk]
        items = [it for _, its in preps for it in its]
        temps = D.slot_temps(d, items)
        ps, lfs = D.readout(D.forward_full(M, items, pad), temps)
        c = 0
        for m, its in preps:
            n = sum(len(it["slots"]) for it in its)
            p, lf = ps[c:c + n], lfs[c:c + n]; c += n
            out.append({"dist": D.row_dist(m, p).tolist(), "loss": D.row_loss(m, p, lf, "C").item(), "loss_A": D.row_loss(m, p, lf, "A").item(),
                        "mass": D.row_log_mass(m, lf, "T").exp().item()})
        # khối lượng ở T = 1 (không chia nhiệt độ), từng dòng: trung bình các slot của dòng
        o = D.forward_full(M, items, pad); pr1 = torch.log_softmax(o["full"], -1)[:, o["letters"]]; c = 0
        for j, (m, its) in enumerate(preps):
            n = sum(len(it["slots"]) for it in its)
            out[len(out) - len(preps) + j]["mass_T1"] = float(torch.stack([torch.logsumexp(pr1[c + q, :int(o["nopts"][c + q])], -1) for q in range(n)]).mean().exp())
            c += n
    return out


rows = read_jsonl(DATA / "val_v0.jsonl")[:200]
ms = [D.to_decider(r) for r in rows]
ev = [eval_path(m) for m in ms]
tr1 = train_path(ms, batch=1)
tr8 = train_path(ms, batch=8)
argmax = lambda v: max(range(len(v)), key=v.__getitem__)
res = {"n": len(ms), "by_kind": {}}
for kind in ("choice", "noul", "score"):
    ix = [i for i, m in enumerate(ms) if m["kind"] == kind]
    if not ix:
        continue
    res["by_kind"][kind] = {
        "n": len(ix),
        "pred_equal_batch1": sum(argmax(ev[i]) == argmax(tr1[i]["dist"]) for i in ix),
        "pred_equal_batch8": sum(argmax(ev[i]) == argmax(tr8[i]["dist"]) for i in ix),
        "max_abs_prob_diff_batch1": max(max(abs(a - b) for a, b in zip(ev[i], tr1[i]["dist"])) for i in ix),
        "max_abs_prob_diff_batch8": max(max(abs(a - b) for a, b in zip(ev[i], tr8[i]["dist"])) for i in ix),
        "step0_mass_mean": sum(tr1[i]["mass"] for i in ix) / len(ix),
        "step0_mass_T1_mean": sum(tr1[i]["mass_T1"] for i in ix) / len(ix),
        "step0_mass_T1_below_0.5": sum(tr1[i]["mass_T1"] < 0.5 for i in ix),
        "step0_mass_min": min(tr1[i]["mass"] for i in ix),
        "step0_mass_below_0.5": sum(tr1[i]["mass"] < 0.5 for i in ix),
        "step0_loss_mean": sum(tr1[i]["loss"] for i in ix) / len(ix),
        "step0_acc_vs_target_argmax": sum(argmax(ev[i]) == argmax(ms[i]["target"]) for i in ix) / len(ix),
    }
# G2 = dự đoán trùng ở batch 1 (cùng phép tính, cùng hình dạng tensor). Batch 8 chỉ để đo nhiễu số học bf16 theo hình dạng batch.
res["PASS"] = all(v["pred_equal_batch1"] == v["n"] for v in res["by_kind"].values())
res["batch8_all_equal"] = all(v["pred_equal_batch8"] == v["n"] for v in res["by_kind"].values())

# chiều noul: lựa chọn "Yes"/"Có" của dữ liệu phải ứng với "yes" của decider (ô 1)
nl = [(r, m) for r, m in zip(rows, ms) if m["kind"] == "noul"]
res["noul_direction"] = {"rows": len(nl),
                         "p_yes_equals_target_of_yes_option": sum(abs(m["target"][1] - r["target"][[o in D.YES for o in r["options"]].index(True)]) < 1e-9 for r, m in nl)}
# score: đảo thứ tự mức thì phân phối đường chấm đảo theo (đường isolated không phụ thuộc thứ tự mức)
sc = [m for m in ms if m["kind"] == "score"][:5]
res["score_order_invariance_max_abs_diff"] = max(
    max(abs(a - b) for a, b in zip(eval_path(m)[::-1], eval_path({**m, "levels": m["levels"][::-1], "target": m["target"][::-1]}))) for m in sc) if sc else None


# 5 ví dụ ánh xạ mỗi kiểu (bỏ nguồn ViHSD nếu có), mỗi trường cắt 200 ký tự, kèm 200 ký tự cuối của prompt decider
def cut(x):
    return str(x).replace("\n", "\\n")[:200]


ex = {}
for kind in ("choice", "noul", "score"):
    ex[kind] = []
    for r, m in zip(rows, ms):
        if m["kind"] != kind or "vihsd" in r["id"].lower() or len(ex[kind]) >= 5:
            continue
        its = D.items_for(d, m)
        ex[kind].append({"id": cut(r["id"]), "orig_options": [cut(o) for o in r["options"]], "orig_target": r["target"],
                         "question": cut(m["question"]), "decider_options_or_levels": [cut(o) for o in m.get("options", m.get("levels", ["no", "yes"]))],
                         "decider_target": [round(t, 4) for t in m["target"]], "n_prompt_rows": len(its),
                         "prompt_tail": cut(d.m.tok.decode(its[0]["ids"])[-200:])})
res["examples"] = ex

# chẩn đoán khối lượng hợp lệ: tập token reader dùng; top-10 ở vị trí đọc; mass ở T=1 và ở T phát hành; token đầu chuỗi
from decider.prompt import label_table, letter_ids  # noqa: E402
tk = d.m.tok
names, lab_ids, _ = label_table(tk)
reader_ids = d.m.letters.tolist()
assert reader_ids == lab_ids == list(letter_ids(tk)) and M["letters"].tolist() == reader_ids
res["reader_token_set"] = {"source": "decider.prompt.label_table / DecisionModel.letters (buffer của chính model chấm)",
                           "first_10": [{"label": n, "id": i, "decoded": tk.decode([i])} for n, i in zip(names[:10], lab_ids[:10])],
                           "train_letters_identical_to_reader": True,
                           "choice_uses": "10 nhãn đầu, mask theo số lựa chọn (A..)", "noul_score_rows_use": "A = no, B = yes"}
variants = {}
for L in "ABCDEFGHIJ":
    variants[L] = {v: tk.encode(v, add_special_tokens=False) for v in (L, " " + L, L.lower(), " " + L.lower())}
res["label_variants_token_ids"] = variants
top = {}
with torch.no_grad():
    for kind in ("choice", "noul", "score"):
        top[kind] = []
        for r, m in zip(rows, ms):
            if m["kind"] != kind or len(top[kind]) >= 3:
                continue
            its = D.items_for(d, m)[:1]
            o = D.forward_full(M, its, pad)
            pr = torch.softmax(o["full"][0], -1)
            v, ix = pr.topk(10)
            k = int(o["nopts"][0])
            T = D.slot_temps(d, its)[0]
            top[kind].append({"id": cut(r["id"])[:80], "k": k, "mass_T1": float(pr[M["letters"][:k]].sum()),
                              "mass_T": float(torch.softmax(o["full"][0] / T, -1)[M["letters"][:k]].sum()),
                              "top10": [(tk.decode([int(i)]), round(float(p), 4)) for p, i in zip(v, ix)],
                              "first_tokens": tk.decode(its[0]["ids"][:4]), "starts_with_bos": its[0]["ids"][0] == tk.bos_token_id if tk.bos_token_id is not None else False})
res["top10_at_read_position"] = top
for kind, lst in top.items():
    for x in lst:
        print(kind, round(x["mass_T1"], 3), round(x["mass_T"], 3), json.dumps(x["top10"][:10], ensure_ascii=False)[:180])
res["lora_modules"] = len(M["lora_modules"]); res["n_trainable"] = M["n_trainable"]
(OUT / "g2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("by_kind", "PASS", "noul_direction", "score_order_invariance_max_abs_diff", "lora_modules", "n_trainable")}))
