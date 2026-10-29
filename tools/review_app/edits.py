"""Lớp ghi đè của người: data/human_edits/edits_log.jsonl (chỉ nối thêm, flush + fsync) và state.json (trạng thái hiện hành, sinh lại
từ log sau mỗi lần ghi; ghi tạm rồi os.replace). Không có gì khác ghi vào thư mục này.

Bản ghi log (mọi bản ghi có id, thoi_diem, ai, loai, hanh_dong):
  loai = "mau"     : khoa (khóa chuỗi trong inventory), hanh_dong ∈ {duyet, sua, hoan_tac}, gia_tri_cu, gia_tri_moi, ly_do, ghi_chu
  loai = "cau"     : split ∈ {dev, test} (thiếu = dev), dev_id (chỉ dev) / pair_id, nguon, truong ∈ {dap_an, cau_hoi, lua_chon:<j>, state, loai_bo},
                     hanh_dong ∈ {sua, hoan_tac}, gia_tri_cu, gia_tri_moi (cả trường), ly_do (bắt buộc khi sửa), ghi_chu;
                     khi đến từ đề xuất theo câu: de_xuat_id, thay_chuoi_con {cu, moi}
  loai = "de_xuat" : de_xuat_id, khoa (đề xuất chuỗi mẫu) hoặc split/pair_id/truong (đề xuất theo câu),
                     hanh_dong ∈ {duyet, sua_roi_duyet, tu_choi, hoan_tac}, gia_tri_moi, ghi_chu; "hoan_tac" đưa đề xuất về "đang chờ"
  loai = "chot_dot": dot (số thứ tự), tom_tat, ghi_chu
  loai = "dung_sai": pair_id (dev), hanh_dong ∈ {giu_moi, hoan_tac}, ghi_chu; khi hoan_tac: de_xuat (các đề xuất đã gỡ), keo_theo (câu cùng đoạn đổi theo)
Bản ghi mới nhất của cùng (khoa) hoặc (split, id câu, truong) là bản có hiệu lực; "hoan_tac" đưa về giá trị gốc.
Sửa ở test nằm riêng trong state["cau_test"]; các màn hình dev chỉ đọc state["cau"].
"""
import datetime
import json
import os
import threading
import uuid
from pathlib import Path

REASONS = ["sai nhãn", "lỗi dịch", "chính tả", "mơ hồ nên loại", "khác"]
_LOCK = threading.Lock()


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def read_log(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append(path, rec):
    """Nối một dòng và ép xuống đĩa. Trả về bản ghi đã thêm id/thoi_diem."""
    rec = {"id": uuid.uuid4().hex, "thoi_diem": now(), **rec}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def derive_state(log, inventory):
    """-> dict(mau={khoa: {trang_thai, gia_tri, goc, ...}}, cau={dev_id: {truong: {...}}}, cau_test={pair_id: {truong: {...}}},
    de_xuat={id: {...}}, dot=[...])."""
    base = {x["key"]: x for x in inventory}
    mau = {k: {"trang_thai": v["base_status"], "gia_tri": v["vi"], "goc": v["vi"], "trang_thai_goc": v["base_status"], "n_ban_ghi": 0} for k, v in base.items()}
    cau, cau_test, de_xuat, dot, dung_sai = {}, {}, {}, [], {}
    for r in log:
        t = r.get("loai")
        if t == "mau" and r.get("khoa") in mau:
            m = mau[r["khoa"]]; m["n_ban_ghi"] += 1
            if r["hanh_dong"] == "duyet":
                m["trang_thai"] = "đã duyệt"
            elif r["hanh_dong"] == "sua":
                m["gia_tri"] = r["gia_tri_moi"]
                m["trang_thai"] = "đã duyệt" if r["gia_tri_moi"] == m["goc"] else "đã sửa"
            elif r["hanh_dong"] == "hoan_tac":
                m["gia_tri"], m["trang_thai"] = m["goc"], m["trang_thai_goc"]
            m["cap_nhat"] = r["thoi_diem"]; m["ai"] = r.get("ai")
        elif t == "cau":
            tgt = cau_test if r.get("split") == "test" else cau
            pid = r.get("dev_id") or r["pair_id"]
            c = tgt.setdefault(pid, {})
            if r["hanh_dong"] == "hoan_tac":
                c.pop(r["truong"], None)
            else:
                c[r["truong"]] = {"gia_tri_moi": r["gia_tri_moi"], "gia_tri_cu": r.get("gia_tri_cu"), "ly_do": r.get("ly_do"), "ghi_chu": r.get("ghi_chu", ""),
                                  "thoi_diem": r["thoi_diem"], "ai": r.get("ai"), "de_xuat_id": r.get("de_xuat_id")}
            if not c:
                tgt.pop(pid, None)
            if r.get("de_xuat_id") in de_xuat:          # bản ghi do một lần duyệt đề xuất theo câu sinh ra: nhớ để hoàn tác
                de_xuat[r["de_xuat_id"]].setdefault("ap", []).append({"pair_id": pid, "truong": r["truong"], "truoc": r.get("gia_tri_cu"),
                                                                      "sau": r["gia_tri_moi"], "hanh_dong": r["hanh_dong"]})
        elif t == "de_xuat":
            if r["hanh_dong"] == "hoan_tac":
                de_xuat.pop(r["de_xuat_id"], None)
            else:
                de_xuat[r["de_xuat_id"]] = {"hanh_dong": r["hanh_dong"], "khoa": r.get("khoa"), "split": r.get("split"), "pair_id": r.get("pair_id"),
                                            "truong": r.get("truong"), "gia_tri_moi": r.get("gia_tri_moi"), "thoi_diem": r["thoi_diem"], "ai": r.get("ai")}
        elif t == "dung_sai":            # quyết định ở màn "Đúng→sai" (dev): bản ghi mới nhất của cặp có hiệu lực
            dung_sai[r["pair_id"]] = {"hanh_dong": r["hanh_dong"], "thoi_diem": r["thoi_diem"], "ai": r.get("ai"), "ghi_chu": r.get("ghi_chu", ""),
                                      "de_xuat": r.get("de_xuat", []), "keo_theo": r.get("keo_theo", [])}
        elif t == "chot_dot":
            dot.append({"dot": r["dot"], "thoi_diem": r["thoi_diem"], "tom_tat": r.get("tom_tat"), "ai": r.get("ai")})
    counts = {s: sum(1 for m in mau.values() if m["trang_thai"] == s) for s in ("chưa duyệt", "đã duyệt", "đã sửa")}
    return {"mau": mau, "cau": cau, "cau_test": cau_test, "de_xuat": de_xuat, "dot": dot, "dung_sai": dung_sai, "dem_mau": counts, "so_ban_ghi": len(log)}


def write_state(edits_dir, state):
    p = Path(edits_dir) / "state.json"
    tmp = p.with_suffix(".json.tmp")
    with _LOCK, open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def read_proposals(path):
    """config/vi_proposals.jsonl, hai loại dòng (id duy nhất trong cả file):
      đề xuất chuỗi mẫu: {id, khoa, gia_tri_de_xuat, ban_tieng_anh, ly_do, nguoi_de_xuat, thoi_diem}
      đề xuất theo câu : {id, loai:"cau", split:"dev"|"test", row_id (id dòng VI có dấu), pair_id, nguon, truong:"state"|"question"|"option:<k>",
                          gia_tri_cu (chuỗi con xuất hiện đúng 1 lần trong trường), gia_tri_de_xuat, ly_do:"loi_dich"|"chinh_ta", bang_chung_en,
                          giai_thich, nguoi_de_xuat, thoi_diem}"""
    return read_log(path)


PROPOSAL_FIELD = lambda t: "cau_hoi" if t == "question" else ("state" if t == "state" else "lua_chon:" + t.split(":", 1)[1])
PROPOSAL_REASON = {"loi_dich": "lỗi dịch", "chinh_ta": "chính tả"}


def occurrences(text, sub):
    """Số lần sub xuất hiện trong text (tính cả chồng lấn); sub rỗng -> 0."""
    if not sub:
        return 0
    n, i = 0, text.find(sub)
    while i >= 0:
        n += 1
        i = text.find(sub, i + 1)
    return n


def pending_proposals(proposals, state):
    return [p for p in proposals if p["id"] not in state["de_xuat"]]
