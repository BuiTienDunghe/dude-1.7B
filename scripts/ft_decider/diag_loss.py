"""Chẩn đoán trước khi chọn loss (bước 0, không train). Trên val_v0:
- Bảng theo kiểu câu (200 dòng đầu): trung bình CE_eval(T) (CE với phân phối đường chấm, T phát hành), −log mass_T (khối lượng nhãn
  trên toàn từ vựng sau chia T), −log mass_T1 (không chia T). Phương án C (nghĩa đen) = CE_eval − log mass_T;
  phương án A = CE_eval − log mass_T1; phương án N = CE_eval.
- Chiều noul: accuracy argmax của decider-2b (đường chấm) so với target đã ánh xạ trên MỌI dòng noul của val_v0.
Ra: results/ft_decider/gates/diag_loss.json
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
from decider.infer import Decider  # noqa: E402

d = Decider(folder, device="cpu", dtype=torch.bfloat16, use_graphs=False)
d.m.lm = None
M = D.build_lora_model(folder, 1)
M["model"].eval()
pad = d.m.tok.pad_token_id
val = read_jsonl(DATA / "val_v0.jsonl")


@torch.no_grad()
def per_row(rows):
    out = []
    for r in rows:
        m = D.to_decider(r)
        its = D.items_for(d, m)
        temps = D.slot_temps(d, its)
        o = D.forward_full(M, its, pad)
        ps, lfs = D.readout(o, temps)
        q = D.row_dist(m, ps); t = torch.tensor(m["target"], device=q.device)
        ce = float(-(t * torch.log(q.clamp_min(1e-30))).sum())
        lmT = float(D.row_log_mass(m, lfs, "T"))
        lm1 = float(D.row_log_mass(m, lfs, "1"))
        out.append({"kind": m["kind"], "ce_eval_T": ce, "neg_log_mass_T": -lmT, "neg_log_mass_T1": -lm1, "mass_T1": float(torch.tensor(lm1).exp()),
                    "correct": int(torch.argmax(q) == torch.argmax(t))})
    return out


rows200 = per_row(val[:200])
tab = {}
for k in ("choice", "noul", "score"):
    xs = [x for x in rows200 if x["kind"] == k]
    tab[k] = {"n": len(xs), **{f: sum(x[f] for x in xs) / len(xs) for f in ("ce_eval_T", "neg_log_mass_T", "neg_log_mass_T1", "mass_T1")},
              "mass_T1_below_0.5": sum(x["mass_T1"] < 0.5 for x in xs)}
    tab[k]["loss_C_literal"] = tab[k]["ce_eval_T"] + tab[k]["neg_log_mass_T"]
    tab[k]["loss_A"] = tab[k]["ce_eval_T"] + tab[k]["neg_log_mass_T1"]
monitor_mass_T1_mean = sum(x["mass_T1"] for x in rows200) / len(rows200)
noul_rows = [r for r in val if r["kind"] == "noul"]
nr = per_row(noul_rows)
res = {"table_first200": tab, "monitor200_mass_T1_mean": monitor_mass_T1_mean,
       "noul_direction_all_val": {"rows": len(nr), "acc_argmax_vs_mapped_target": sum(x["correct"] for x in nr) / len(nr)}}
(RESULTS / "ft_decider" / "gates" / "diag_loss.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res))
