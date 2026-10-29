"""Lọc heuristic trước khi gửi Opus (giữ quy tắc (a)): câu chứa đáp án phải có ít nhất một cụm 'có vẻ cùng loại' ngoài đáp án, ở cả hai phía."""
import re
NUM = re.compile(r"\d")


def side_ok(side):
    gs, a = side["cau_chua_dap_an"], side["answer"]
    rest = gs.replace(a, " \u0000 ", 1)
    if NUM.search(a):
        return bool(NUM.search(rest.replace("\u0000", "")))
    if a[:1].isupper():
        toks = [t.strip("\"“”'‘’(),.;:") for t in rest.split()[1:] if t != "\u0000"]
        return any(t[:1].isupper() for t in toks)
    return len(rest.split()) >= 16


def ok(c):
    return side_ok(c["vi"]) and side_ok(c["en"])
