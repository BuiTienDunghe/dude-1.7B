"""Phát hiện trùng / gần trùng / chứa đoạn giữa văn bản train và văn bản của dev_v0 + phần để dành test.

Ba phép kiểm tra trên văn bản đã chuẩn hoá (NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng):
  exact   — trùng nguyên văn;
  near    — MinHash Jaccard >= 0.8 trên shingle 5 ký tự (cùng ngưỡng PhoLaya dùng khi khử rò rỉ);
  contain — train chứa một đoạn của văn bản dev/test: văn bản dài (>=13 từ) dùng chung một 13-gram từ;
            văn bản 4–12 từ phải xuất hiện nguyên vẹn (theo ranh giới từ) trong văn bản train;
            văn bản dưới 4 từ chỉ xét exact.
"""
import re
import unicodedata
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

NUM_PERM, JACCARD, NGRAM = 128, 0.8, 13


def normalize(text):
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def minhash(norm):
    m = MinHash(num_perm=NUM_PERM, seed=1)
    s = norm.replace(" ", "_")
    for i in range(max(1, len(s) - 4)):
        m.update(s[i:i + 5].encode("utf-8"))
    return m


class Index:
    def __init__(self):
        self.exact = {}
        self.lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
        self.ngrams = {}                  # 13-gram -> tag
        self.short = defaultdict(dict)    # n -> {tuple(words): tag}
        self.tags = {}
        self.n = 0

    def add(self, text, tag):
        norm = normalize(text)
        if not norm:
            return
        key = f"d{self.n}"; self.n += 1
        self.tags[key] = (tag, norm)
        self.exact.setdefault(norm, tag)
        words = norm.split()
        if len(words) >= 3:
            self.lsh.insert(key, minhash(norm))
        if len(words) >= NGRAM:
            for i in range(len(words) - NGRAM + 1):
                self.ngrams.setdefault(tuple(words[i:i + NGRAM]), tag)
        elif len(words) >= 4:
            self.short[len(words)][tuple(words)] = tag

    def check(self, text):
        """Trả về (loại, tag) của lần trùng đầu tiên, hoặc None."""
        norm = normalize(text)
        if not norm:
            return None
        if norm in self.exact:
            return ("exact", self.exact[norm])
        words = norm.split()
        if len(words) >= 3:
            m = minhash(norm)
            for key in self.lsh.query(m):
                tag, other = self.tags[key]
                if m.jaccard(minhash(other)) >= JACCARD:
                    return ("near", tag)
        for i in range(len(words) - NGRAM + 1):
            g = tuple(words[i:i + NGRAM])
            if g in self.ngrams:
                return ("contain13", self.ngrams[g])
        for n, table in self.short.items():
            for i in range(len(words) - n + 1):
                g = tuple(words[i:i + n])
                if g in table:
                    return ("contain_short", table[g])
        return None
