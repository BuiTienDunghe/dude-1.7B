"""Phần dùng chung: đường dẫn, seed, revision model, dựng prompt."""
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"

SEED = 20261012
LETTERS = "ABCDEFGHIJ"  # tối đa 10 lựa chọn

# Model ứng viên và revision ghim (do scripts/pin_models.py ghi ra).
MODELS_FILE = CONFIG / "models.json"


def model_revisions():
    return json.loads(MODELS_FILE.read_text(encoding="utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_diacritics(s):
    """Bỏ dấu tiếng Việt: đ/Đ → d/D, bỏ mọi dấu kết hợp (NFD, loại Mn), trả về NFC.
    Chuyển nguyên văn từ scripts/bench/build_bench_v1.py (hàm đã tạo bản không dấu của bench_v1)."""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s)


def strip_decision(seed, epoch, row_id, p, state_only_frac):
    """Quyết định bỏ dấu cho một dòng train ở một epoch, chỉ từ hash(seed, epoch, id) — tách khỏi mọi RNG xáo batch.
    -> None (không bỏ), "state_only" hoặc "all"."""
    if p <= 0:
        return None
    h = hashlib.sha256(f"{seed}:{epoch}:{row_id}".encode()).digest()
    u1 = int.from_bytes(h[:8], "big") / 2**64
    u2 = int.from_bytes(h[8:16], "big") / 2**64
    if u1 >= p:
        return None
    return "state_only" if u2 < state_only_frac else "all"


def apply_strip(row, mode):
    """Bỏ dấu một dòng: "all" = state, câu hỏi và lựa chọn; "state_only" = chỉ state. Target và thứ tự lựa chọn giữ nguyên."""
    if mode is None:
        return row
    out = {**row, "state": strip_diacritics(row["state"])}
    if mode == "all":
        out["question"] = strip_diacritics(row["question"])
        out["options"] = [strip_diacritics(o) for o in row["options"]]
    return out


def user_message(row, frames):
    """Nội dung lượt user: hướng dẫn, văn bản (state), câu hỏi, lựa chọn đánh chữ A, B, C..."""
    fr = frames[row["lang"]]
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return (f"{fr['instruction']}\n\n"
            f"{fr['state_label']}\n{row['state']}\n\n"
            f"{fr['question_label']} {row['question']}\n\n"
            f"{fr['options_label']}\n{opts}")


def render_prompt(row, tokenizer, frames):
    """Chuỗi prompt hoàn chỉnh qua chat template, thinking tắt; model trả lời ngay sau chuỗi này."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message(row, frames)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
