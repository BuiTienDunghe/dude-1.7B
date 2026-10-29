"""Phần dùng chung cho train và kiểm khả thi trên decider-2b, ở ĐỊNH DẠNG GỐC của decider (không dùng định dạng chữ cái của dự án).

Dựng prompt: chính các hàm của gói decider mà đường chấm (scripts/bench/eval_decider_v1.py) gọi:
  choice -> Decider._decide_items([(state, [{"question", "options"}])])        (= d.decide, ánh xạ "text")
  noul   -> Decider._system_one_items(state, {"q": {"type": "noul", "instructions"}}, independent=True, state_first)
  score  -> Decider._system_one_items(state, {"q": {"type": "score", "instructions", "criteria"}}, ...) với isolated levels:
            mỗi mức một hàng "…\\nProposed answer: <mức>\\nDoes the proposed answer fit?" với lựa chọn [no, yes].
Đọc xác suất: tại mỗi slot ("Answer: (") lấy logit toàn từ vựng z = lm_head(h). Đường chấm dùng softmax giới hạn trên các token nhãn
  hợp lệ, chia cho T phát hành theo kiểu câu (choice 1,164; noul 1,624; score 1,124): p = softmax(z[nhãn]/T). Ở đây tính đúng phép đó
  từ z (TT.scaled_softmax), và thêm log-xác suất trên TOÀN từ vựng log softmax(z/T)[nhãn] cho loss.
Phân phối của đường chấm cho một dòng:
  choice: p trên các lựa chọn;  noul: [P(no), P(yes)] (thứ tự của decider);  score: combine_isolated(P(yes) từng mức) = r_i / Σ r_j.
Loss (row_loss, chọn phương án bằng LOSS_VARIANTS): CE giữa phân phối đích và phân phối đường chấm (T phát hành), cộng hạng
  −log mass trên toàn từ vựng: A = mass ở T=1; C (nghĩa đen, CE toàn từ vựng trên z/T) = mass ở T; N = không có hạng mass.
  score: CE với q = r/Σr (r_i = P(yes) giới hạn của hàng mức i) + trung bình −log mass_i của các hàng mức (một dòng score tốn n forward).
"""
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa: E402

REPO, REV = "Mapika/decider-2b", "ff99089b6810691bd79134ec6febb5d9afa6e619"
CAP = 32768                       # giới hạn token state như đường chấm
# Lựa chọn noul của train_v0/val_v0 (config/train_format.json), kể cả dạng đã bỏ dấu (bỏ dấu "all" chạm cả lựa chọn). Ở định dạng
# decider, lựa chọn noul luôn là ["no", "yes"] của gói, nên bỏ dấu lựa chọn noul không đổi gì ngoài việc nhận ra lựa chọn "yes".
YES, NO = ("Yes", "Có", "Co"), ("No", "Không", "Khong")


def load_folder():
    """Tải (đã cache) repo đã ghim, kiểm sha256 từng file decider/*.py với bản đã duyệt ở bước baseline, thêm vào sys.path."""
    from huggingface_hub import snapshot_download
    folder = snapshot_download(REPO, revision=REV)
    reviewed = RESULTS / "baselines" / "meta" / "Mapika__decider-2b"
    for p in (Path(folder) / "decider").glob("*.py"):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256((reviewed / f"decider__{p.name}").read_bytes()).hexdigest(), p.name
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return folder


def to_decider(row):
    """Dòng định dạng dự án -> câu hỏi định dạng decider + target trên các lựa chọn của decider."""
    k = row["kind"]
    if k == "noul":
        opts = list(row["options"])
        iy = next(i for i, o in enumerate(opts) if o in YES)
        ino = next(i for i, o in enumerate(opts) if o in NO)
        assert {iy, ino} == {0, 1}, opts
        p_yes = float(row["target"][iy])
        return {"kind": "noul", "state": row["state"], "question": row["question"], "target": [1.0 - p_yes, p_yes]}
    if k == "score":
        return {"kind": "score", "state": row["state"], "question": row["question"], "levels": list(row["options"]),
                "target": [float(t) for t in row["target"]]}
    return {"kind": "choice", "state": row["state"], "question": row["question"], "options": list(row["options"]),
            "target": [float(t) for t in row["target"]]}


def items_for(d, m, cap=CAP):
    """Các hàng prompt (item của gói decider) cho một câu đã ánh xạ — đúng hàm đường chấm dùng."""
    if m["kind"] == "choice":
        _, items = d._decide_items([(m["state"], [{"question": m["question"], "options": m["options"]}])], cap)
        return items
    spec = {"type": "noul", "instructions": m["question"]} if m["kind"] == "noul" else \
        {"type": "score", "instructions": m["question"], "criteria": m["levels"]}
    _, _, items = d._system_one_items(m["state"], {"q": spec}, True, cap, "state_first", None)
    return items


def slot_temps(d, items):
    """T của từng slot, đúng như đường chấm (TT.for_items + TT.slot_temperatures)."""
    from decider import temperature as TT
    T = TT.slot_temperatures(TT.for_items(d.T, d.T_by_type, items), items)
    return T if isinstance(T, (list, tuple)) else [T] * sum(len(it["slots"]) for it in items)


def readout(out, temps):
    """out: kết quả forward_full. -> (p giới hạn từng slot [k], log xác suất toàn từ vựng tại nhãn từng slot [k]:
    dict "T" = sau chia T phát hành, "1" = không chia).
    p = TT.scaled_softmax(logit nhãn đã mask, T) — đúng phép của đường chấm (Decider.decide_batch / _system_one_probs)."""
    from decider import temperature as TT
    zr, full, nopts = out["zr"], out["full"], out["nopts"]
    P = TT.scaled_softmax(zr, list(temps))
    t = torch.tensor(list(temps), dtype=full.dtype).to(full.device)[:, None]
    LT = torch.log_softmax(full / t, -1)[:, out["letters"]]
    L1 = torch.log_softmax(full, -1)[:, out["letters"]]
    n = zr.shape[0]
    ps = [P[i, :int(nopts[i])] for i in range(n)]
    lfs = [{"T": LT[i, :int(nopts[i])], "1": L1[i, :int(nopts[i])]} for i in range(n)]
    return ps, lfs


def row_dist(m, ps):
    """Phân phối đường chấm của một dòng từ xác suất giới hạn của các slot của nó."""
    if m["kind"] == "score":
        r = torch.stack([p[1] for p in ps])
        return r / r.sum()
    assert len(ps) == 1
    return ps[0]


LOSS_VARIANTS = {
    "A": "CE(t, phân phối đường chấm ở T phát hành) − log mass toàn từ vựng ở T=1",
    "C": "CE toàn từ vựng trên z/T (nghĩa đen) = CE(t, phân phối đường chấm) − log mass toàn từ vựng ở T",
    "N": "CE(t, phân phối đường chấm) — không có hạng mass (chỉ theo dõi)",
}


def mass_key(variant):
    """Khối lượng mà loss phạt (và monitor/cổng đo): ở T phát hành với C, ở T=1 với A và N."""
    return "T" if variant == "C" else "1"


def row_loss(m, ps, lfs, variant):
    """Loss một dòng. score: CE với phân phối isolated q = r/Σr (r_i = P(yes) giới hạn của hàng mức i) + trung bình −log mass các hàng."""
    assert variant in LOSS_VARIANTS
    t = torch.tensor(m["target"], device=ps[0].device, dtype=torch.float32)
    if m["kind"] == "score":
        r = torch.stack([p[1] for p in ps])
        ce = -(t * (torch.log(r) - torch.log(r.sum()))).sum()
    else:
        ce = -(t * torch.log(ps[0])).sum()
    if variant == "N":
        return ce
    return ce - row_log_mass(m, lfs, mass_key(variant))


def row_log_mass(m, lfs, key="1"):
    """log khối lượng nhãn hợp lệ trung bình trên các slot của một dòng (choice/noul: một slot)."""
    return torch.stack([torch.logsumexp(lf[key], -1) for lf in lfs]).mean()


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]


def build_lora_model(folder, seed):
    """Model train: lớp DecisionModel của gói (nạp tokenizer + Qwen3_5ForCausalLM bf16, tắt cuDNN SDPA như gói) + LoRA r16/α32/
    dropout 0,05 trên TARGETS (attention đầy đủ q/k/v/o, linear attention in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down).
    -> dict(model, backbone, lm_head, letters, lora_modules, n_trainable)."""
    from peft import LoraConfig, get_peft_model
    from decider.model import DecisionModel
    dm = DecisionModel(folder, dtype=torch.bfloat16, grad_ckpt=False)
    lm = dm.lm.to("cuda")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.enable_input_require_grads()
    torch.manual_seed(seed)
    model = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                          target_modules=TARGETS))
    inner = model.base_model.model
    return {"model": model, "backbone": inner.model, "lm_head": inner.lm_head, "letters": dm.letters.cuda(),
            "lora_modules": sorted({n.split(".lora_A")[0] for n, _ in model.named_parameters() if ".lora_A" in n}),
            "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def forward_full(M, items, pad_id):
    """Forward như đường chấm (DecisionModel.slot_logits, không autocast: trọng số bf16, linear attention giữ phép fp32 của nó; LoRA
    fp32 do peft tự đổi kiểu). collate của gói (pad phải, bội 64).
    -> dict(zr = logit nhãn [N, 255] đã mask như slot_logits, full = logit toàn từ vựng [N, V] với cột nhãn thay bằng zr, nopts, letters)."""
    import torch.nn.functional as F
    from decider.model import collate
    from decider.prompt import MAX_OPTIONS
    b = collate(items, pad_id)
    nopts = b["nopts"].cuda()
    h = M["backbone"](input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).last_hidden_state
    hs = h[b["slot_batch"].cuda(), b["slot_idx"].cuda()]
    W = M["lm_head"].weight
    zr = F.linear(hs, W[M["letters"]]).float()                                  # = DecisionModel.slot_logits trước khi mask
    full = F.linear(hs, W).float().index_copy(1, M["letters"], zr)              # cột nhãn trùng đúng zr
    ar = torch.arange(MAX_OPTIONS, device=zr.device)[None, :]
    zr = zr.masked_fill(ar >= nopts[:, None], float("-inf"))
    return {"zr": zr, "full": full, "nopts": b["nopts"], "letters": M["letters"]}
