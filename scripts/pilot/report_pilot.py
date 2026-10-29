"""Số liệu cho REPORT_pilot.md: accuracy (CI bootstrap theo câu), chênh lệch ghép cặp, mức sàn, decontam, kết quả soát Opus.
Chạy được bằng Python Windows (chỉ thư viện chuẩn). Ra: results/bench_candidates/pilot/{metrics_pilot.json, tables_pilot.md}.
"""
import json
import random
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "results" / "bench_candidates" / "pilot"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = {r["id"]: r for r in read(P / "pilot_rows.jsonl")}
MODELS = {"decider-2b (text)": P / "eval/decider/decider-2b-text/preds.jsonl", "decider-2b (keyed)": P / "eval/decider/decider-2b-keyed/preds.jsonl",
          "adapter đối chứng (Qwen3-1.7B)": P / "eval/control/preds.jsonl"}
CELLS = [("mlqa", "en", "orig"), ("mlqa", "vi", "orig"), ("mlqa", "vi", "nodiac"), ("gpiqa", "en", "orig"), ("gpiqa", "vi", "orig"), ("gpiqa", "vi", "nodiac")]
B, SEED = 4000, 20261014


def ci(xs):
    rng = random.Random(SEED)
    n = len(xs)
    bs = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(xs) / n, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def diff_ci(a, b):
    """a, b: dict pair_id -> 0/1 cùng tập câu; trả về trung bình (a-b) và CI bootstrap theo câu."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    m, lo, hi = ci(d)
    return {"n": len(d), "diff": m, "lo": lo, "hi": hi, "a_only": sum(x == 1 for x in d), "b_only": sum(x == -1 for x in d)}


met, lines = {"acc": {}, "diff": {}}, []
correct = {}
for name, path in MODELS.items():
    if not path.exists():
        continue
    pr = {x["id"]: x for x in read(path)}
    assert set(pr) == set(rows), (name, len(pr))
    for src, lang, var in CELLS:
        ids = [i for i, r in rows.items() if (r["source"], r["lang"], r["variant"]) == (src, lang, var)]
        c = {rows[i]["pair_id"]: int(pr[i]["pred"] == rows[i]["target"]) for i in ids}
        correct[(name, src, lang, var)] = c
        m, lo, hi = ci(list(c.values()))
        met["acc"][f"{name}|{src}|{lang}|{var}"] = {"n": len(c), "acc": m, "lo": lo, "hi": hi}
lines.append("## Accuracy (CI 95% bootstrap theo câu, 4.000 lần)\n")
lines.append("| Model | " + " | ".join(f"{s} {l}{' không dấu' if v == 'nodiac' else ''}" for s, l, v in CELLS) + " |")
lines.append("|---" * (len(CELLS) + 1) + "|")
for name in MODELS:
    if (name, "mlqa", "en", "orig") not in correct:
        continue
    cells = []
    for s, l, v in CELLS:
        a = met["acc"][f"{name}|{s}|{l}|{v}"]
        cells.append(f"{100 * a['acc']:.0f} [{100 * a['lo']:.0f}; {100 * a['hi']:.0f}]")
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
fl = json.load(open(P / "floors_pilot.json", encoding="utf-8"))
lines.append("| sàn: đoán đều | " + " | ".join(f"{100 * fl[f'{s}|{l}|{v}']['doan_deu']:.0f}" for s, l, v in CELLS) + " |")
lines.append("| sàn: vị trí phổ biến nhất (trong mẫu) | " + " | ".join(f"{100 * fl[f'{s}|{l}|{v}']['vi_tri_pho_bien_nhat_trong_mau']['acc']:.0f}" for s, l, v in CELLS) + " |")
lines.append("| sàn: lựa chọn dài nhất | " + " | ".join(f"{100 * fl[f'{s}|{l}|{v}']['dai_nhat']:.0f}" for s, l, v in CELLS) + " |")
lines.append("| sàn: câu trùng nhiều từ nhất với câu hỏi (chỉ MLQA) | " + " | ".join(
    f"{100 * fl[f'{s}|{l}|{v}']['trung_tu_cau_hoi']:.0f}" if s == "mlqa" else "—" for s, l, v in CELLS) + " |")

lines.append("\n## Chênh lệch ghép cặp (điểm %; CI 95% bootstrap theo câu; a_only/b_only = số câu chỉ một bên đúng)\n")
lines.append("| So sánh | n | chênh | CI | a đúng riêng | b đúng riêng |\n|---|---|---|---|---|---|")
names = [n for n in MODELS if (n, "mlqa", "en", "orig") in correct]
for src in ("mlqa", "gpiqa"):
    for name in names:
        for a, b, lab in ((("vi", "orig"), ("en", "orig"), "VI có dấu − EN"), (("vi", "nodiac"), ("vi", "orig"), "VI không dấu − VI có dấu")):
            d = diff_ci(correct[(name, src, *a)], correct[(name, src, *b)])
            met["diff"][f"{name}|{src}|{lab}"] = d
            lines.append(f"| {src} · {name} · {lab} | {d['n']} | {100 * d['diff']:+.0f} | [{100 * d['lo']:+.0f}; {100 * d['hi']:+.0f}] | {d['a_only']} | {d['b_only']} |")
    if len(names) >= 3:
        for l, v in (("en", "orig"), ("vi", "orig"), ("vi", "nodiac")):
            d = diff_ci(correct[(names[0], src, l, v)], correct[(names[2], src, l, v)])
            met["diff"][f"{src}|{l}|{v}|decider_text−control"] = d
            lines.append(f"| {src} {l}{' không dấu' if v == 'nodiac' else ''} · decider-2b (text) − đối chứng | {d['n']} | {100 * d['diff']:+.0f} | "
                         f"[{100 * d['lo']:+.0f}; {100 * d['hi']:+.0f}] | {d['a_only']} | {d['b_only']} |")

# ---- MLQA: model có hơn mẹo "câu trùng nhiều từ nhất với câu hỏi" không?
import re  # noqa: E402
WORD = re.compile(r"\w+", re.UNICODE)


def overlap_pick(q, opts):
    qw = {w.lower() for w in WORD.findall(q)}
    sc = [len(qw & {w.lower() for w in WORD.findall(o)}) for o in opts]
    return sc.index(max(sc))


lines.append("\n## MLQA: accuracy tách theo mẹo trùng từ (câu có nhiều từ chung với câu hỏi nhất)\n")
lines.append("| Model | biến thể | mẹo đúng: n, acc model | mẹo sai: n, acc model |\n|---|---|---|---|")
met["mlqa_overlap_split"] = {}
for name in names:
    for l, v in (("en", "orig"), ("vi", "orig"), ("vi", "nodiac")):
        ok_, bad = [], []
        for pid, c in correct[(name, "mlqa", l, v)].items():
            r = rows[f"{pid}:{l}" + (":nodiac" if v == "nodiac" else "")]
            (ok_ if overlap_pick(r["question"].split(": ", 1)[1], r["options"]) == r["target"] else bad).append(c)
        met["mlqa_overlap_split"][f"{name}|{l}|{v}"] = {"meo_dung_n": len(ok_), "meo_dung_acc": sum(ok_) / max(1, len(ok_)),
                                                         "meo_sai_n": len(bad), "meo_sai_acc": sum(bad) / max(1, len(bad))}
        lines.append(f"| {name} | {l}{' không dấu' if v == 'nodiac' else ''} | {len(ok_)}, {100 * sum(ok_) / max(1, len(ok_)):.0f} | "
                     f"{len(bad)}, {100 * sum(bad) / max(1, len(bad)):.0f} |")

# ---- soát Opus
aud = {}
for src in ("mlqa", "gpiqa"):
    scan = P / "audit" / f"out_{src}.jsonl"
    if not scan.exists():
        continue
    s = read(scan)
    vp = P / "audit" / f"verify_out_{src}.jsonl"
    ver = {x["audit_id"]: x for x in read(vp)} if vp.exists() else {}
    final = Counter()
    for x in s:
        v = ver.get(x["audit_id"])
        final[v["verdict_cuoi"] if v else x["verdict"]] += 1
    aud[src] = {"n_quet": len(s), "quet": dict(Counter(x["verdict"] for x in s)), "duong_tat_quet": sum(bool(x.get("duong_tat")) for x in s),
                "loai_khac": dict(Counter(x.get("loai_khac") for x in s if x.get("loai_khac"))),
                "ben_sai": dict(Counter(x.get("ben_sai") for x in s if x["verdict"] != "OK")),
                "n_soi_lai": len(ver), "ket_luan_soi_lai": dict(Counter(v["ket_luan"] for v in ver.values())), "cuoi": dict(final),
                "doi_dap_an_cuoi": sum(bool((ver.get(x["audit_id"]) or x).get("doi_dap_an")) for x in s if (ver.get(x["audit_id"]) or {}).get("verdict_cuoi", x["verdict"]) != "OK")}
met["audit"] = aud
met["floors"] = fl
met["decontam"] = json.load(open(P / "decontam_pilot.json", encoding="utf-8"))["tong_cau_trung"]
lines.append("\n## Soát Opus (quét → soi lại)\n")
for src, a in aud.items():
    lines.append(f"- **{src}**: quét {a['n_quet']} câu: {a['quet']}; đường tắt {a['duong_tat_quet']}; loại khác {a['loai_khac']}; bên sai {a['ben_sai']}. "
                 f"Soi lại {a['n_soi_lai']}: {a['ket_luan_soi_lai']}. Kết luận cuối: {a['cuoi']}.")
(P / "metrics_pilot.json").write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
(P / "tables_pilot.md").write_text("# Bảng số chạy thử MLQA vi và Global-PIQA vi\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
