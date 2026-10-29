"""Báo cáo bench 1.2 ĐÃ ĐÓNG BĂNG (data/bench_v1_2/). Python Windows, thư viện chuẩn.
- Preds DEV 1.2: dòng có nội dung trùng bản thử cuối dùng preds bản thử (results/bench_v1_2_trial/eval, chính nó = preds 1.1 + chấm lại); dòng khác
  dùng preds chấm lại (results/bench_v1_2_stage/eval_rescore). Ghi results/dev_v1_2/eval/<model>/preds.jsonl.
- Bảng điểm dev 3 cột: 1.1 / 1.2 toàn bộ / 1.2 chỉ dòng source (vi_provenance = source và gold_provenance = source), theo model × nguồn × biến thể.
Ra: results/dev_v1_2/REPORT_bench_v1_2.md, scores_v1_2.json."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B12, S, T = ROOT / "data/bench_v1_2", ROOT / "results/bench_v1_2_stage", ROOT / "results/bench_v1_2_trial"
OUT = ROOT / "results/dev_v1_2"
PROV = ("vi_provenance", "vi_revised_fields", "gold_provenance")
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
core = lambda r: {k: v for k, v in r.items() if k not in PROV}
NAME = {"belebele": "Belebele", "sib200": "SIB-200", "massive": "MASSIVE", "xcopa": "XCOPA", "vifactcheck": "ViFactCheck", "viwikifc": "ViWikiFC", "*": "Tất cả"}
MODELS = {"decider-2b-text": "decider/decider-2b-text", "decider-2b-keyed": "decider/decider-2b-keyed", "decider-ft-ep2-text": "ftd/decider-2b-text",
          "decider-ft-ep2-keyed": "ftd/decider-2b-keyed", "Qwen3-1.7B-fullvocab-s1": "control", "Qwen3-1.7B": "base", "strip-s1": "strip"}
CAT_VI = {"doi_dap_an": "đổi đáp án", "sua_cau_hoi": "sửa câu hỏi", "sua_lua_chon": "sửa lựa chọn", "sua_state": "sửa state", "loai_cau": "loại câu",
          "chi_doi_nfc": "chỉ đổi do NFC"}
CATS = list(CAT_VI)
DIF_VI = {"chuoi_mau_transport_query": "chuỗi mẫu transport_query", "nguoi_dung_trong_chat": "bản dịch / yêu cầu người dùng gửi trong chat",
          "hoan_tac_B": "hoàn tác ở màn Đúng→sai (mục B)", "quyet_dinh_tren_trang": "quyết định mới / đổi ý trên trang", "nhan_test": "quyết định nhãn test",
          "loai_do_trung_test": "loại vì trùng gần với test", "chi_provenance": "chỉ khác provenance"}

mt = json.load(open(B12 / "manifest_test_v1.2.json", encoding="utf-8"))
md = json.load(open(B12 / "manifest_dev_v1_2.json", encoding="utf-8"))
dif = json.load(open(S / "diff_vs_trial_v1_2.json", encoding="utf-8"))
br = json.load(open(S / "build_report_v1_2_trial.json", encoding="utf-8"))
leak = json.load(open(B12 / "leakage_v1_2.json", encoding="utf-8"))
CHK = json.load(open(S / "check_v1_2.json", encoding="utf-8"))
assert sha(B12 / "dev_v1_2.jsonl") == md["sha256"]["dev_v1_2.jsonl"] and sha(B12 / "test_v1.2.jsonl") == mt["sha256"]["test_v1.2.jsonl"]
dev11 = {r["id"]: r for r in read(ROOT / "data/bench_v1_1/dev_v1_1.jsonl")}
dev12 = {r["id"]: r for r in read(B12 / "dev_v1_2.jsonl")}
trial = {r["id"]: r for r in read(T / "dev_v1_2_trial.jsonl")}
from_trial = {i for i, r in dev12.items() if i in trial and core(trial[i]) == core(r)}
PAIR_SRC = {r["pair_id"]: (r["vi_provenance"] == "source" and r["gold_provenance"] == "source") for r in dev12.values() if r["lang"] == "vi" and r["variant"] == "orig"}
VARS = [("en", "orig"), ("vi", "orig"), ("vi", "nodiac")]
VAR_VI = {("en", "orig"): "EN", ("vi", "orig"): "VI có dấu", ("vi", "nodiac"): "VI không dấu"}

# ---------------------------------------------------------------- preds dev 1.2 + điểm
scores, flags, prev = {}, [], {}
pt_scores = json.load(open(T / "scores_v1_2_trial.json", encoding="utf-8"))["scores"]
for m, sub in MODELS.items():
    p11 = {x["id"]: x for x in read(ROOT / f"results/dev_v1_1/eval/{m}/preds.jsonl")}
    ptr = {x["id"]: x for x in read(T / f"eval/{m}/preds.jsonl")}
    prs = {x["id"]: x for x in read(S / f"eval_rescore/{sub}/preds.jsonl")}
    need = set(dev12) - from_trial
    assert need <= set(prs), (m, len(need - set(prs)))
    p12 = {i: (ptr[i] if i in from_trial else prs[i]) for i in dev12}
    (OUT / f"eval/{m}").mkdir(parents=True, exist_ok=True)
    with open(OUT / f"eval/{m}/preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, r in dev12.items():
            x = {k: v for k, v in p12[i].items() if k not in ("rescored",)}
            x.update(target=r["target"], correct=x["pred"] == r["target"], nguon_preds="ban_thu" if i in from_trial else "cham_lai_ban_cuoi")
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    c11 = {i: int(p11[i]["pred"] == r["target"]) for i, r in dev11.items()}
    c12 = {i: int(p12[i]["pred"] == r["target"]) for i, r in dev12.items()}
    sc = {}
    for src in sorted({r["source"] for r in dev11.values()}) + ["*"]:
        for l, v in VARS:
            ids11 = [i for i, r in dev11.items() if (src == "*" or r["source"] == src) and (r["lang"], r["variant"]) == (l, v)]
            if not ids11:
                continue
            ids12 = [i for i, r in dev12.items() if (src == "*" or r["source"] == src) and (r["lang"], r["variant"]) == (l, v)]
            ids_s = [i for i in ids12 if PAIR_SRC.get(dev12[i]["pair_id"], False)]
            ids_rem = [i for i in ids11 if i in dev12]
            a = lambda ids, c: sum(c[i] for i in ids) / len(ids) if ids else float("nan")
            sc[f"{src}|{l}|{v}"] = {"n11": len(ids11), "n12": len(ids12), "n12_source": len(ids_s), "acc11": a(ids11, c11), "acc11_con_lai": a(ids_rem, c11),
                                    "acc12": a(ids12, c12), "acc12_source": a(ids_s, c12), "acc11_tren_dong_source": a(ids_s, c11)}
            if src != "*" and abs(sc[f"{src}|{l}|{v}"]["acc12"] - sc[f"{src}|{l}|{v}"]["acc11"]) > 0.02:
                flags.append({"model": m, "nguon": src, "bien_the": f"{l}|{v}", **{k: sc[f"{src}|{l}|{v}"][k] for k in ("acc11", "acc11_con_lai", "acc12")},
                              "doi_dung_sai": sum(c11[i] != c12[i] for i in ids_rem), "bi_loai": len(ids11) - len(ids_rem)})
    scores[m] = sc
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "scores_v1_2.json").write_text(json.dumps({"scores": scores, "flags": flags, "dong_dung_preds_ban_thu": len(from_trial),
                                                   "dong_cham_lai_ban_cuoi": len(dev12) - len(from_trial)}, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- báo cáo
pct = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"
chg = read(S / "changes_v1_2_trial.jsonl")
stat = defaultdict(Counter)
for c in chg:
    for k in c["loai_thay_doi"]:
        stat[(c["split"], c["nguon"])][k] += 1
    stat[(c["split"], c["nguon"])]["tong"] += 1
pairs11 = {s: Counter(r["source"] for r in read(ROOT / f"data/bench_v1_1/{f}") if r["lang"] == "vi" and r["variant"] == "orig")
           for s, f in (("dev", "dev_v1_1.jsonl"), ("test", "test_v1.1.jsonl"))}
pv = md["provenance"]   # bản hiện hành (dev có thể đã cập nhật sau đóng băng; phần test giống hệt manifest test)
sl = pv["so_lieu_tu_log"]
go = pv["dap_an_nguoc_nguon"]
L = ["# Bench 1.2 (đã đóng băng)", "",
     "Bản 1.2 = bench 1.1 sau khi người soát: áp lớp ghi đè `data/human_edits` (đề xuất đã duyệt, sửa câu, loại câu, đổi đáp án dev, bản dịch người dùng gửi), "
     "quyết định nhãn test (`results/human_review/label_decisions_remote.jsonl`, dòng mới nhất theo cặp), chuẩn hóa NFC mọi trường tiếng Việt (bản không dấu sinh "
     "lại từ bản đã sửa). MLQA để bản 1.3, Global-PIQA không làm. Không train; không chạy model nào trên test.", "",
     "## Tóm tắt", "",
     f"- **Đóng băng** ở `data/bench_v1_2/` lúc {mt['frozen_at']}: `test_v1.2.jsonl` ({mt['counts']['rows']} dòng, chỉ đọc, sha `{mt['sha256']['test_v1.2.jsonl']}`), "
     f"`FROZEN_test_v1.2.sha256`, `manifest_test_v1.2.json` (chỉ đọc); `dev_v1_2.jsonl` ({md['counts']['rows']} dòng, sha `{md['sha256']['dev_v1_2.jsonl']}`), "
     "`manifest_dev_v1_2.json`, `leakage_v1_2.json`. Bản 1.1 giữ nguyên ở `data/bench_v1_1/`."]
for cn in md.get("cap_nhat", []):
    L.append(f"- **Cập nhật dev {cn['thoi_diem'][:10]}** (test KHÔNG đổi, vẫn `{cn['test_khong_doi'][:12]}…`): {cn['ly_do']} Cặp đổi: "
             + ", ".join(f"`{p}`" for p in cn["cap_doi"]) + f". Dev sha `{cn['sha_cu'][:12]}…` → `{cn['sha_moi'][:12]}…`; các dòng đổi đã chấm lại bằng 7 model.")
L += [
     f"- Dựng từ {sl['edits_log_ban_ghi']} bản ghi lớp ghi đè và {sl['label_decisions_remote_ban_ghi']} quyết định nhãn. `release_check`: mã {mt['release_check']['exit_code']}, "
     "0 đề xuất chờ.",
     f"- Cặp đổi so với 1.1: dev {br['n_changed_pairs']['dev']}, test {br['n_changed_pairs']['test']}. Đáp án ngược nguồn: dev {sum(g['split'] == 'dev' for g in go)}, "
     f"test {sum(g['split'] == 'test' for g in go)}.",
     f"- Rò rỉ: dev↔test {leak['dev_states_hitting_test']} / {leak['test_states_hitting_dev']}; train_v0 + val_v0 với test {leak['train_v0_vs_test']['total']}, "
     f"với dev {leak['train_v0_vs_dev_v1_2']['total']} (như 1.1). Một cặp dev (`massive:4129`) bị loại để giữ dev↔test = 0 (mục C).",
     f"- Người dùng chưa xác nhận câu nào qua nút \"Đồng ý (đã đọc)\": `user_confirmations.jsonl` {'có' if mt['xac_nhan_nguoi_dung']['co_file'] else 'chưa có'}; "
     f"đã xác nhận {mt['xac_nhan_nguoi_dung']['da_xac_nhan']}/{mt['xac_nhan_nguoi_dung']['so_cau_loai_hoac_doi_dap_an']} câu bị loại hoặc đổi đáp án.", "",
     "Điểm dev (%), trung bình trên mọi câu; \"chỉ dòng source\" = cặp có văn bản tiếng Việt và đáp án giữ như nguồn:", "",
     "| Model | EN: 1.1 / 1.2 / 1.2 source | VI có dấu: 1.1 / 1.2 / 1.2 source | VI không dấu: 1.1 / 1.2 / 1.2 source |", "|---|---|---|---|"]
for m, sc in scores.items():
    L.append(f"| {m} | " + " | ".join(f"{pct(sc[k]['acc11'])} / {pct(sc[k]['acc12'])} / {pct(sc[k]['acc12_source'])}" for k in ("*|en|orig", "*|vi|orig", "*|vi|nodiac")) + " |")
n_s = scores["decider-2b-text"]
L += ["", f"Số câu: EN {n_s['*|en|orig']['n11']} → {n_s['*|en|orig']['n12']} (source {n_s['*|en|orig']['n12_source']}); VI {n_s['*|vi|orig']['n11']} → "
      f"{n_s['*|vi|orig']['n12']} (source {n_s['*|vi|orig']['n12_source']}). Bảng theo nguồn ở mục 5. Không kết luận hướng đi.", "",
      "## 1. So với bản dựng thử cuối", "",
      f"Bản thử: dev `{dif['ban_thu']['dev'][:12]}…`, test `{dif['ban_thu']['test'][:12]}…` ({dif['ban_thu']['ban_ghi_lop_ghi_de']} bản ghi). Bản hiện hành (test trùng từng byte bản đóng băng) dựng từ "
      f"{dif['ban_dung']['ban_ghi_lop_ghi_de']} bản ghi. Mọi cặp khác bản thử đều thuộc một loại dưới đây (không có cặp không giải thích được):", "",
      "| Loại khác biệt | dev | test |", "|---|---|---|"]
for k in sorted(set(dif["dev"]["theo_loai"]) | set(dif["test"]["theo_loai"])):
    L.append(f"| {DIF_VI.get(k, k)} | {dif['dev']['theo_loai'].get(k, 0)} | {dif['test']['theo_loai'].get(k, 0)} |")
L += ["", f"- Bản dịch / yêu cầu người dùng gửi trong chat (dev): {', '.join('`' + p + '`' for p in dif['dev']['cap_theo_loai'].get('nguoi_dung_trong_chat', []))}.",
      f"- Quyết định trên trang: dev {', '.join('`' + p + '`' for p in dif['dev']['cap_theo_loai'].get('quyet_dinh_tren_trang', []))}; "
      f"test {', '.join('`' + p + '`' for p in dif['test']['cap_theo_loai'].get('quyet_dinh_tren_trang', []))}.",
      f"- Hoàn tác ở mục B: " + "; ".join(f"`{h['pair_id']}` (gỡ {', '.join(h['de_xuat'])}; kéo theo {len(h['keo_theo'])} câu)" for h in dif["dung_sai"]["hoan_tac"])
      + f". Giữ bản mới: {dif['dung_sai']['giu_moi']} câu.",
      f"- Quyết định nhãn mới sau bản thử: {', '.join(x['pair_id'] + ' ' + x['hanh_dong'] for x in dif['quyet_dinh_nhan_moi']) or 'không'} (không đổi dữ liệu vì cặp đã bị loại sẵn).",
      f"- `transport_query`: \"{dif['transport_query']['cu']}\" → \"{dif['transport_query']['moi']}\".",
      f"- Dòng dev có nội dung khác bản thử: {len(dev12) - len(from_trial)}, đã chấm lại bằng 7 model (`scripts/human_edits/rescore_dev_v1_2_final.sh`); "
      f"{len(from_trial)} dòng còn lại dùng preds của bản thử.", "",
      "## 2. Thay đổi so với 1.1 theo nguồn và loại", "",
      "Một cặp (EN + VI + VI không dấu) được đếm ở **mọi** loại nó thuộc. \"Chỉ đổi do NFC\" là cặp đổi byte nhưng chữ sau NFC không đổi.", ""]
for s in ("dev", "test"):
    L += [f"### {s}", "", "| Nguồn | Số cặp 1.1 | " + " | ".join(CAT_VI[k] for k in CATS) + " | Cặp có đổi |", "|---" * (len(CATS) + 3) + "|"]
    tot = Counter()
    for src in sorted(pairs11[s]):
        c = stat.get((s, src), Counter()); tot.update(c); tot["n"] += pairs11[s][src]
        L.append(f"| {NAME.get(src, src)} | {pairs11[s][src]} | " + " | ".join(str(c[k]) for k in CATS) + f" | {c['tong']} |")
    L += [f"| **Tổng** | {tot['n']} | " + " | ".join(str(tot[k]) for k in CATS) + f" | {tot['tong']} |", ""]
L += ["## 3. Nguồn gốc văn bản và đáp án (A)", "",
      "`gold_provenance` trên mọi dòng; `vi_provenance` + `vi_revised_fields` trên dòng tiếng Việt (NFC và chuỗi mẫu không tính là sửa). Số dòng VI có dấu:", "",
      "| Tập · nguồn | source | revised | sửa state | sửa câu hỏi | sửa lựa chọn |", "|---|---|---|---|---|---|"]
for k, v in pv["dong_vi_co_dau_theo_tap_nguon"].items():
    L.append(f"| {k.replace('|', ' · ')} | {v.get('source', 0)} | {v.get('revised', 0)} | {v.get('sua:state', 0)} | {v.get('sua:question', 0)} | {v.get('sua:option', 0)} |")
L += ["", f"Đáp án đi ngược nguồn: {len(go)} cặp:", ""] + [f"- {g['split']} `{g['pair_id']}`: {g['dap_an_nguon']} → {g['dap_an_1_2']}" for g in go]
L += ["", f"Quy trình: {pv['quy_trinh']} Cách duyệt theo lời người làm dự án: \"{pv['cach_duyet_nguoi_dung_mo_ta']}\".", "",
      "Số liệu thật từ log (manifest `provenance.so_lieu_tu_log`):", "",
      f"- Đề xuất: {sl['de_xuat_ban_ghi']} bản ghi; người dùng bấm từ xa {sl['de_xuat_tu_xa_nguoi_dung_bam']}; mang ghi chú ủy quyền {sl['de_xuat_ghi_chu_uy_quyen']}; "
      f"điều phối ghi theo lời người dùng trong chat {sl['de_xuat_theo_chat_nguoi_dung']}; khác (theo lựa chọn của người dùng trên trang) {sl['de_xuat_khac']}.",
      f"- Đề xuất đang áp: {sl['de_xuat_dang_ap']}, theo người viết: " + ", ".join(f"{k} {v}" for k, v in sorted(sl["de_xuat_dang_ap_theo_nguoi_viet"].items())) + ".",
      f"- Chuỗi mẫu: Excel {sl['chuoi_mau_excel']}, từ xa {sl['chuoi_mau_tu_xa']}, theo chat {sl['chuoi_mau_theo_chat']}. Đúng→sai: giữ bản mới {sl['dung_sai']['giu_moi']}, "
      f"hoàn tác {sl['dung_sai']['hoan_tac']}.",
      f"- Quyết định nhãn: {sl['label_decisions_remote_ban_ghi']} bản ghi (mang ghi chú ủy quyền {sl['label_decisions_ghi_chu_uy_quyen']}); mới nhất theo cặp: "
      + ", ".join(f"{k} {v}" for k, v in sl["label_decisions_moi_nhat_theo_cap"].items()) + ".",
      f"- Số do điều phối cung cấp (log không tách được): người dùng tự bấm \"{pv['so_lieu_dieu_phoi_cung_cap']['nguoi_dung_tu_bam']}\"; theo ủy quyền / theo agent "
      f"\"{pv['so_lieu_dieu_phoi_cung_cap']['theo_uy_quyen_theo_agent']}\".", "",
      "## 4. Kỹ thuật (C)", "",
      f"- Bản không dấu sinh từ bản VI hiện hành (strip_diacritics trên state, câu hỏi, lựa chọn; cùng đáp án): kiểm {CHK['nodiac_kiem']} dòng, trong đó revised "
      f"{CHK['nodiac_kiem_rieng_revised']}; không khớp: {len(CHK['nodiac_khong_khop'])}.",
      f"- Rò rỉ, đúng cách build_bench_v1_1 (khớp chính xác, MinHash ≥ 0,8, 13-gram / câu ngắn nằm trọn): dev↔test {leak['dev_states_hitting_test']} / "
      f"{leak['test_states_hitting_dev']}; train_v0 + val_v0: test {leak['train_v0_vs_test']['total']} "
      f"({', '.join(f'{k}: {v}' for k, v in leak['train_v0_vs_test']['by_kind_target_trainsource'].items())}), dev {leak['train_v0_vs_dev_v1_2']['total']}.",
      "- Lần dựng đầu phát hiện 1 trùng dev↔test (câu ngắn nằm trọn): test `massive:test:1483` \"tôi đã đặt báo thức nào\" (sau khi sửa \"lời nhắc\" → \"báo thức\" cho khớp "
      "EN \"alarms\") nằm trọn trong dev `massive:4129` \"tôi đã đặt báo thức nào ngay bây giờ\". Hai câu MASSIVE khác nhau, bản EN không trùng. Điều phối chọn loại cặp dev "
      f"(test giữ nguyên); dev MASSIVE còn {md['counts']['by_source_lang_variant']['massive|vi|orig']} cặp (1.1: 300; 5 cặp khác đã bị loại từ trước trong lớp ghi đè). "
      "`build_revised.py` giờ có chốt kiểm dev↔test (bản thử ghi và cảnh báo, bản thật chặn).", "",
      "## 5. Điểm DEV theo nguồn", "",
      "Độ đúng theo đáp án của từng bản. \"1.2 chỉ dòng source\": cặp có `vi_provenance = source` và `gold_provenance = source` (EN tính theo cặp). "
      "Ô có |1.2 − 1.1| > 2 điểm được tô đậm.", ""]
for m, sc in scores.items():
    L += [f"### {m}", "", "| Nguồn | biến thể | n 1.1 → 1.2 (source) | 1.1 | 1.2 toàn bộ | 1.2 chỉ dòng source | 1.2 − 1.1 |", "|---|---|---|---|---|---|---|"]
    for key, v in sc.items():
        src, l, var = key.split("|")
        d = 100 * (v["acc12"] - v["acc11"])
        b = "**" if abs(d) > 2 and src != "*" else ""
        L.append(f"| {NAME.get(src, src)} | {VAR_VI[(l, var)]} | {v['n11']} → {v['n12']} ({v['n12_source']}) | {pct(v['acc11'])} | {pct(v['acc12'])} | "
                 f"{pct(v['acc12_source'])} | {b}{d:+.1f}{b} |")
    L.append("")
L += ["## 6. So với điểm bản thử (toàn bộ dev, VI có dấu / không dấu)", "", "| Model | VI có dấu: thử → cuối | VI không dấu: thử → cuối | EN: thử → cuối |", "|---|---|---|---|"]
for m, sc in scores.items():
    pt = pt_scores.get(m, {})
    L.append(f"| {m} | " + " | ".join(f"{pct(pt.get(k, {}).get('acc12'))} → {pct(sc[k]['acc12'])}" for k in ("*|vi|orig", "*|vi|nodiac", "*|en|orig")) + " |")
L += ["", f"Ô (model × nguồn × biến thể) nhảy quá 2 điểm so với 1.1: {len(flags)} (danh sách ở `scores_v1_2.json`).", "",
      "## 7. Câu test đã xem", "",
      f"- `test_ids_nguoi_da_xem` (manifest test): {len(mt['test_ids_nguoi_da_xem'])} cặp ("
      + ", ".join(f"{k} {v}" for k, v in mt["test_ids_nguoi_da_xem_nguon"].items()) + ").",
      f"- `test_ids_agent_da_doc`: {mt['test_ids_agent_da_doc']}.", "",
      "## 8. File", "",
      "- `data/bench_v1_2/`: `dev_v1_2.jsonl`, `test_v1.2.jsonl` (chỉ đọc), `FROZEN_test_v1.2.sha256`, `manifest_dev_v1_2.json`, `manifest_test_v1.2.json` (chỉ đọc), `leakage_v1_2.json`.",
      "- `results/bench_v1_2_stage/`: bản dựng, `changes_v1_2_trial.jsonl`, `build_report_v1_2_trial.json`, `check_v1_2.json`, `diff_vs_trial_v1_2.json`, `rescore/`, `eval_rescore/`.",
      "- `results/dev_v1_2/`: `eval/<model>/preds.jsonl` (preds dev 1.2 đã gộp), `scores_v1_2.json`, báo cáo này.",
      "- Code: `scripts/human_edits/run_build_v1_2.sh`, `check_stage_v1_2.py`, `diff_stage_vs_trial_v1_2.py`, `rescore_dev_v1_2_final.sh`, `freeze_v1_2.py`, "
      "`report_v1_2.py`, `drop_dev_pair_via_tool.py`; `build_revised.py` (thêm chốt dev↔test)."]
(OUT / "REPORT_bench_v1_2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(json.dumps({"models": list(scores), "flags": len(flags), "from_trial": len(from_trial), "rescored": len(dev12) - len(from_trial),
                  "summary": {m: {k: round(100 * sc[k]["acc12"], 1) for k in ("*|en|orig", "*|vi|orig", "*|vi|nodiac")} for m, sc in scores.items()}}, ensure_ascii=False))
