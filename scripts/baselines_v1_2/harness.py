"""Khung chấm chung cho adapter model ngoài (bước E, DEV 1.2). Adapter chỉ cần một hàm score(row) -> dict:
    {"probs": [k số, theo ĐÚNG thứ tự row["options"]], "n_ctx_tokens": int|None, "truncated": bool, "rejected": bool, "note": str|None}
Khung lo: hai thứ tự (normal = thứ tự dev; reverse = đảo lựa chọn, target = k-1-target, adapter nhận options đã đảo), batch 1, đo độ trễ,
assert số phần tử = k và tổng ≈ 1 (trừ câu bị từ chối), ghi <out>/<order>/<name>/preds.jsonl + timing.json theo định dạng của eval_base
(id, pair_id, source, lang, kind, variant, k, target, probs, pred, correct). Câu bị cắt / bị từ chối được đánh dấu và KHÔNG tính vào điểm.
Không đọc test: từ chối file dòng có 'test' trong tên hay dòng có id chứa ':test:'."""
import json
import statistics
import time
from pathlib import Path


def read_rows(path):
    p = Path(path)
    assert "test" not in p.name.lower(), "không chấm trên test"
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    assert not any(":test:" in r["id"] for r in rows), "có dòng test"
    return rows


def reverse_row(r):
    return {**r, "options": r["options"][::-1], "target": len(r["options"]) - 1 - r["target"]}


def run(rows, score, out_root, name, orders=("normal", "reverse"), meta=None, limit_s=None, log=print):
    out_root = Path(out_root)
    res = {}
    for order in orders:
        t_start, lat, out = time.time(), [], []
        n_trunc = n_rej = 0
        for r in rows:
            rr = reverse_row(r) if order == "reverse" else r
            t0 = time.perf_counter()
            s = score(rr)
            lat.append(time.perf_counter() - t0)
            k = len(rr["options"])
            rej, trunc = bool(s.get("rejected")), bool(s.get("truncated"))
            n_rej += rej; n_trunc += trunc
            probs = s.get("probs")
            if probs is not None:
                assert len(probs) == k, (name, r["id"], probs)
                assert abs(sum(probs) - 1) < 1e-3, (name, r["id"], sum(probs))
            else:
                assert rej or trunc, (name, r["id"], "không có xác suất mà không bị loại")
            pred = max(range(k), key=lambda j: probs[j]) if probs else None
            out.append({"id": r["id"], "pair_id": r["pair_id"], "source": r["source"], "lang": r["lang"], "kind": r["kind"], "variant": r.get("variant", "orig"),
                        "k": k, "target": rr["target"], "probs": probs, "pred": pred, "correct": (pred == rr["target"]) if pred is not None else False,
                        "excluded": rej or trunc, "truncated": trunc, "rejected": rej, "n_ctx_tokens": s.get("n_ctx_tokens"), "note": s.get("note"),
                        "reversed": order == "reverse"})
            if limit_s and time.time() - t_start > limit_s:
                raise TimeoutError(f"{name} {order}: quá {limit_s} giây")
        d = out_root / order / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        ok = [x for x in out if not x["excluded"]]
        tim = {**(meta or {}), "name": name, "order": order, "rows": len(out), "n_truncated": n_trunc, "n_rejected": n_rej,
               "acc_not_excluded": sum(x["correct"] for x in ok) / max(1, len(ok)),
               "batch1_median_latency_ms": 1000 * statistics.median(lat), "batch1_p90_latency_ms": 1000 * sorted(lat)[int(0.9 * len(lat))],
               "latency_includes": "toàn bộ lời gọi score() của adapter: tách token + forward + đọc xác suất"}
        (d / "timing.json").write_text(json.dumps(tim, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{name} {order}: n={len(out)} acc(không tính câu loại)={tim['acc_not_excluded']:.4f} cắt={n_trunc} từ chối={n_rej} "
            f"median={tim['batch1_median_latency_ms']:.1f}ms")
        res[order] = tim
    return res
