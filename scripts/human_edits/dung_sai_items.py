"""Mục B: các cặp DEV mà từ hai model trở lên chuyển ĐÚNG → SAI ở VI có dấu (1.1 → 1.2 thử), và cặp chỉ có sửa chữ của người (state / câu hỏi /
lựa chọn; không đổi đáp án, không bị loại, không phải chỉ do chuỗi mẫu). Model = 5 model riêng: decider-2b (ánh xạ text), decider-ft-ep2 (text),
adapter đối chứng, Qwen3-1.7B gốc, strip (bản keyed của decider ghi kèm để tham khảo, không tính). Ra results/bench_v1_2_trial/dung_sai_items.json
(dữ liệu cho trang đọc kiểm và màn /dung_sai của công cụ). Python Windows."""
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/review_app"))
import edits as E  # noqa: E402
from inventory import inventory  # noqa: E402

T = ROOT / "results/bench_v1_2_trial"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
MODELS = ["decider-2b-text", "decider-ft-ep2-text", "Qwen3-1.7B-fullvocab-s1", "Qwen3-1.7B", "strip-s1"]
EXTRA = ["decider-2b-keyed", "decider-ft-ep2-keyed"]
dev11 = {r["id"]: r for r in read(ROOT / "data/bench_v1_1/dev_v1_1.jsonl")}
dev12 = {r["id"]: r for r in read(T / "dev_v1_2_trial.jsonl")}
p11 = {m: {x["id"]: x for x in read(ROOT / f"results/dev_v1_1/eval/{m}/preds.jsonl")} for m in MODELS + EXTRA}
p12 = {m: {x["id"]: x for x in read(T / f"eval/{m}/preds.jsonl")} for m in MODELS + EXTRA}
st = E.derive_state(E.read_log(ROOT / "data/human_edits/edits_log.jsonl"), inventory(ROOT / "config"))
groups = {}
for r in dev11.values():
    if r["lang"] == "vi" and r["variant"] == "orig":
        groups.setdefault(r["state"], []).append(r["pair_id"])
FIELD = {"state": "state", "question": "cau_hoi"}
items = []
for i, r12 in dev12.items():
    if not (r12["lang"] == "vi" and r12["variant"] == "orig" and r12.get("vi_provenance") == "revised" and r12.get("gold_provenance") == "source"):
        continue
    r11 = dev11[i]
    if r12["target"] != r11["target"]:
        continue
    flips = [m for m in MODELS if p11[m][i]["pred"] == r11["target"] and p12[m][i]["pred"] != r12["target"]]
    if len(flips) < 1:
        continue
    pid = r12["pair_id"]; en = dev11.get(pid + ":en")
    c = st["cau"].get(pid, {})
    applied = sorted({v.get("de_xuat_id") for f, v in c.items() if v.get("de_xuat_id")})
    manual = sorted(f for f, v in c.items() if not v.get("de_xuat_id") and f != "loai_bo" and f != "dap_an")
    sib = [q for q in groups[r11["state"]] if q != pid] if "state" in r12["vi_revised_fields"] else []
    items.append({
        "pair_id": pid, "nguon": r12["source"], "kind": r12["kind"], "dap_an": r12["target"], "truong_da_sua": r12["vi_revised_fields"],
        "vi_cu": {"state": r11["state"], "question": r11["question"], "options": r11["options"]},
        "vi_moi": {"state": r12["state"], "question": r12["question"], "options": r12["options"]},
        "en": ({"state": en["state"], "question": en["question"], "options": en["options"], "dap_an": en["target"]} if en else None),
        "du_doan": {m: {"cu": p11[m][i]["pred"], "moi": p12[m][i]["pred"], "dung_cu": p11[m][i]["pred"] == r11["target"],
                        "dung_moi": p12[m][i]["pred"] == r12["target"], "tinh_vao_dem": m in MODELS} for m in MODELS + EXTRA},
        "so_model_dung_sang_sai": len(flips), "model_dung_sang_sai": flips, "nhom": "chinh" if len(flips) >= 2 else "phu",
        "de_xuat_da_ap": applied, "sua_tay_khong_qua_de_xuat": manual, "cau_cung_doan_van": sib,
        "ty_le_giong_state": round(difflib.SequenceMatcher(None, r11["state"], r12["state"]).ratio(), 3)})
items.sort(key=lambda x: (x["nhom"] != "chinh", x["nguon"], x["pair_id"]))
(T / "dung_sai_items.json").write_text(json.dumps({"tieu_chi": __doc__.strip() + " Nhóm 'chinh': từ 2 model trở lên (đúng tiêu chí); nhóm 'phu': đúng 1 model, đưa thêm để đọc (gồm các câu người dùng nêu như xcopa:33, xcopa:79, Goma#1, Horse_riding#2).", "so_cau": len(items), "so_cau_chinh": sum(x["nhom"] == "chinh" for x in items), "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"so_cau": len(items), "chinh": sum(x["nhom"] == "chinh" for x in items), "theo_nguon": {s: sum(x["nguon"] == s for x in items) for s in sorted({x["nguon"] for x in items})},
                  "pairs_chinh": [x["pair_id"][-40:] for x in items if x["nhom"] == "chinh"], "co_cau_cung_doan": sum(bool(x["cau_cung_doan_van"]) for x in items),
                  "co_sua_tay": sum(bool(x["sua_tay_khong_qua_de_xuat"]) for x in items)}, ensure_ascii=False))
