"""Số liệu MLQA recast 2 (4 cụm từ trả lời): accuracy + CI bootstrap theo câu, chênh lệch ghép cặp, so với mức sàn và với recast 1 (pilot đợt 1).
Python Windows, thư viện chuẩn. Ra: pilot2/metrics_mlqa2.json, pilot2/tables_mlqa2.md."""
import json
import random
from pathlib import Path

B = Path(__file__).resolve().parents[2] / "results/bench_candidates"
P2 = B / "pilot2"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = {r["id"]: r for r in read(P2 / "mlqa2_rows.jsonl")}
MODELS = {"decider-2b (text)": P2 / "eval/decider/decider-2b-text/preds.jsonl", "decider-2b (keyed)": P2 / "eval/decider/decider-2b-keyed/preds.jsonl",
          "adapter đối chứng": P2 / "eval/control/preds.jsonl"}
CELLS = [("en", "orig"), ("vi", "orig"), ("vi", "nodiac")]
NB, SEED = 4000, 20261014


def ci(xs):
    rng = random.Random(SEED); n = len(xs)
    bs = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return sum(xs) / n, bs[int(0.025 * NB)], bs[int(0.975 * NB) - 1]


def dci(a, b):
    k = sorted(set(a) & set(b)); d = [a[x] - b[x] for x in k]; m, lo, hi = ci(d)
    return {"n": len(d), "diff": m, "lo": lo, "hi": hi, "a_only": d.count(1), "b_only": d.count(-1)}


met, L, corr = {"acc": {}, "diff": {}}, [], {}
for name, p in MODELS.items():
    pr = {x["id"]: x for x in read(p)}
    assert set(pr) == set(rows)
    for l, v in CELLS:
        ids = [i for i, r in rows.items() if (r["lang"], r["variant"]) == (l, v)]
        corr[(name, l, v)] = {rows[i]["pair_id"]: int(pr[i]["pred"] == rows[i]["target"]) for i in ids}
        m, lo, hi = ci(list(corr[(name, l, v)].values()))
        met["acc"][f"{name}|{l}|{v}"] = {"n": len(ids), "acc": m, "lo": lo, "hi": hi}
fl = json.load(open(P2 / "floors_mlqa2.json", encoding="utf-8"))
n = met["acc"][f"decider-2b (text)|en|orig"]["n"]
L += [f"## Accuracy MLQA recast 2 (n = {n} câu; CI 95% bootstrap theo câu)", "", "| | EN | VI có dấu | VI không dấu |", "|---|---|---|---|"]
for name in MODELS:
    L.append(f"| {name} | " + " | ".join(f"{100 * met['acc'][f'{name}|{l}|{v}']['acc']:.0f} [{100 * met['acc'][f'{name}|{l}|{v}']['lo']:.0f}; "
                                          f"{100 * met['acc'][f'{name}|{l}|{v}']['hi']:.0f}]" for l, v in CELLS) + " |")
for key, lab in (("doan_deu", "sàn: đoán đều"), ("vi_tri_pho_bien_nhat", "sàn: vị trí phổ biến nhất"), ("dai_nhat", "sàn: lựa chọn dài nhất"),
                 ("ngan_nhat", "sàn: lựa chọn ngắn nhất"), ("lua_chon_trung_tu_cau_hoi_nhieu_nhat", "sàn: lựa chọn trùng từ câu hỏi nhiều nhất"),
                 ("meo_dinh_vi_cau", "sàn: mẹo định vị câu (câu trùng từ câu hỏi nhiều nhất → lựa chọn nằm trong câu đó)"),
                 ("xuat_hien_som_nhat_trong_doan", "sàn: lựa chọn xuất hiện sớm nhất trong đoạn")):
    L.append(f"| {lab} | " + " | ".join(f"{100 * fl[f'{l}|{v}'][key]:.0f}" for l, v in CELLS) + " |")
L.append(f"| (tỉ lệ câu có ≥1 nhiễu cùng câu với đáp án) | " + " | ".join(f"{100 * fl[f'{l}|{v}']['gold_cung_cau_voi_it_nhat_1_nhieu']:.0f}" for l, v in CELLS) + " |")
L += ["", "## Chênh lệch ghép cặp (điểm %)", "", "| So sánh | n | chênh | CI | a đúng riêng | b đúng riêng |", "|---|---|---|---|---|---|"]
for name in MODELS:
    for a, b, lab in ((("vi", "orig"), ("en", "orig"), "VI − EN"), (("vi", "nodiac"), ("vi", "orig"), "không dấu − có dấu")):
        d = dci(corr[(name, *a)], corr[(name, *b)]); met["diff"][f"{name}|{lab}"] = d
        L.append(f"| {name} · {lab} | {d['n']} | {100 * d['diff']:+.0f} | [{100 * d['lo']:+.0f}; {100 * d['hi']:+.0f}] | {d['a_only']} | {d['b_only']} |")
for l, v in CELLS:
    d = dci(corr[("decider-2b (text)", l, v)], corr[("adapter đối chứng", l, v)]); met["diff"][f"decider_text−control|{l}|{v}"] = d
    L.append(f"| {l}{' không dấu' if v == 'nodiac' else ''} · decider-2b (text) − đối chứng | {d['n']} | {100 * d['diff']:+.0f} | "
             f"[{100 * d['lo']:+.0f}; {100 * d['hi']:+.0f}] | {d['a_only']} | {d['b_only']} |")
met["floors"] = fl
met["manifest"] = json.load(open(P2 / "manifest_mlqa2.json", encoding="utf-8"))
met["decontam"] = json.load(open(P2 / "decontam_mlqa2.json", encoding="utf-8"))["tong_cau_trung"]
(P2 / "metrics_mlqa2.json").write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
(P2 / "tables_mlqa2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
