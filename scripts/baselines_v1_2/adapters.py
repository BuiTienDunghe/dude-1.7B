"""Adapter model ngoài cho bước E (chạy trong .venv-baselines; gói quyet 1.0.2 / laya 0.4.6 đã rà mã, xem results/baselines_v1_2/notes_*.md).
Mỗi adapter có score(row) -> {"probs": [...theo đúng thứ tự row["options"]], "truncated", "rejected", "n_ctx_tokens", "note"}.
Hai cách ánh xạ câu chọn đáp án (brief): "text" = mỗi lựa chọn là một nhãn mang nguyên văn nội dung; "keyed" = khóa trung tính A, B, ... và nội dung
nằm ở mô tả. Câu Có/Không (kind noul) dùng kiểu câu hỏi gốc "noul" của model (không phụ thuộc thứ tự lựa chọn), xác suất gán theo CHỮ của lựa chọn.
Nhiệt độ: đúng bản phát hành (assert khớp file cấu hình trong snapshot)."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

KEYS = "ABCDEFGHIJ"


def _norm(s):
    s = unicodedata.normalize("NFD", str(s).strip().lower()).replace("đ", "d")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def noul_map(options):
    """Chỉ số của lựa chọn 'đúng' và 'sai' trong row["options"] (Có/Co/Yes vs Không/Khong/No)."""
    t = {"co": True, "yes": True, "khong": False, "no": False}
    v = [t.get(_norm(o)) for o in options]
    assert len(options) == 2 and set(v) == {True, False}, options
    return v.index(True), v.index(False)


def criteria_of(options, mapping):
    if mapping == "text":
        return list(options)
    assert len(options) <= len(KEYS)
    return {KEYS[j]: o for j, o in enumerate(options)}


class Quyet:
    def __init__(self, snap, mapping):
        import quyet
        from quyet.errors import QuestionError
        self.QuestionError = QuestionError
        self.snap, self.mapping = Path(snap), mapping
        self.m = quyet.load(str(self.snap), device="cuda")
        cfg = json.load(open(self.snap / "quyet_config.json", encoding="utf-8"))
        assert self.m.temps == dict(cfg.get("temperatures") or {}), ("nhiệt độ khác bản phát hành", self.m.temps)
        self.kind = cfg["kind"]

    def raw(self, state, q, strict=True):
        return self.m.raw_probabilities([{"state": state, "questions": {"q": q}}], strict=strict)[0]["q"]

    def score(self, r):
        try:
            if r["kind"] == "noul":
                pt, pf = self.raw(r["state"], {"type": "noul", "instructions": r["question"]})
                it, if_ = noul_map(r["options"])
                p = [0.0, 0.0]; p[it], p[if_] = float(pt), float(pf)
            else:
                if self.mapping == "text" and len(set(r["options"])) != len(r["options"]):
                    return {"probs": None, "rejected": True, "note": "nhãn trùng nhau"}
                p = [float(x) for x in self.raw(r["state"], {"type": "choice", "instructions": r["question"],
                                                             "criteria": criteria_of(r["options"], self.mapping)})]
            return {"probs": p}
        except self.QuestionError as ex:
            msg = str(ex)
            return {"probs": None, "truncated": "(strict)" in msg, "rejected": "(strict)" not in msg, "note": msg[:160]}


class Laya:
    """Xác suất CHƯA làm tròn qua đường nội bộ giống calibrate.records_from_labeled (predict công khai làm tròn 4 chữ số).
    Giới hạn nâng lên MAX_LEN / HEAD_MAX (README laya-multilingual: đọc tới 8192 token); câu vẫn bị cắt ở giới hạn nâng (state, đề, hoặc lựa chọn
    > 48 token — giới hạn cứng của gói) thì đánh dấu truncated và không tính điểm. Ghi thêm cờ 'vượt mặc định' (1024 / 256) để đếm."""
    MAX_LEN, HEAD_MAX = 8192, 512

    def __init__(self, snap, mapping):
        import laya
        from laya.common import collate_items, temp_bucket
        self.collate_items, self.temp_bucket = collate_items, temp_bucket
        self.snap, self.mapping = Path(snap), mapping
        assert (self.snap / "tokenizer").is_dir() and (self.snap / "encoder").is_dir(), "snapshot thiếu tokenizer/ hoặc encoder/"
        self.a = laya.load(str(self.snap), device="cuda", fast=False, compile=False)
        cfg = json.load(open(self.snap / "rl_agent_config.json", encoding="utf-8"))
        self.cfg = cfg
        assert list(self.a.temperature_raw) == list(cfg.get("temperature", [1.0, 1.0, 1.0])), ("nhiệt độ khác", self.a.temperature_raw)
        assert dict(self.a.temperature_by_options_raw) == dict(cfg.get("temperature_by_options") or {}), self.a.temperature_by_options_raw
        self.def_max, self.def_head = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
        self.tok = self.a.tok

    def _ntok(self, s):
        return len(self.tok(s, add_special_tokens=False)["input_ids"])

    def _head_trunc(self, qdef, head_max):
        """Tái lập phép cắt đầu câu của laya.common.build_head: lựa chọn > 48 token, ngân sách lựa chọn < 16, hoặc đề bị cắt."""
        opts = [o for o in (qdef.get("_rendered") or [])]
        lens = [self._ntok(" " + o) for o in opts]
        capped = sum(min(48, L) + 1 for L in lens)
        budget = head_max - capped
        ins_len = self._ntok("%s question: %s" % (qdef["type"], qdef["instructions"]))
        return any(L > 48 for L in lens) or budget < 16 or ins_len > max(8, budget)

    def score(self, r):
        from laya.common import render_options
        if r["kind"] == "noul":
            qdef = {"type": "noul", "instructions": r["question"]}
        else:
            if self.mapping == "text" and len(set(r["options"])) != len(r["options"]):
                return {"probs": None, "rejected": True, "note": "nhãn trùng nhau"}
            qdef = {"type": "choice", "instructions": r["question"], "criteria": criteria_of(r["options"], self.mapping)}
        self.a._check_question("q", qdef)
        internal = {"q": self.a._to_internal(qdef)}
        rendered = render_options(internal["q"])
        with torch.no_grad():
            items = self.a._encode_state(r["state"], ["q"], internal, max_len=self.MAX_LEN, head_max_len=self.HEAD_MAX)
            logits, _ = self.a._forward(self.collate_items([items], self.a.tok.pad_token_id))
            dflt = self.a._encode_state(r["state"], ["q"], internal, max_len=self.def_max, head_max_len=self.def_head)
        it = items[0]
        k = len(it["markers"]); qt = it["qtype"]
        T = self.a.temperature_by_options.get(self.temp_bucket(qt, k), self.a.temperature[qt])
        z = np.asarray(logits[0, :k], dtype=np.float64) / T
        p = np.exp(z - z.max()); p = p / p.sum()
        qd = {**qdef, "_rendered": rendered}
        trunc = bool(it["state_stats"]["truncated"]) or self._head_trunc(qd, self.HEAD_MAX)
        over_default = bool(dflt[0]["state_stats"]["truncated"]) or self._head_trunc(qd, self.def_head)
        if r["kind"] == "noul":                       # thứ tự nội bộ của laya cho noul: [false, true]
            i_t, i_f = noul_map(r["options"]); q = [0.0, 0.0]; q[i_t], q[i_f] = float(p[1]), float(p[0]); probs = q
        else:
            assert k == len(r["options"])
            probs = [float(x) for x in p]
        return {"probs": probs, "truncated": trunc, "n_ctx_tokens": len(it.get("ids", it.get("input_ids", []))) or None,
                "note": "vuot_mac_dinh" if over_default else None}
