#!/usr/bin/env python3
"""Word Hunt (4x4 Boggle-style) solver: trie + DFS."""
import re
import sys
import time
from collections import Counter
from functools import lru_cache

SIZE = 4
CELLS = SIZE * SIZE
END = "$"
COMMON_MIN = 5.0
LESS_MIN = 3.0
TOP_N = 30
_SCORES = {3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800, 8: 2200}


def load_dictionary():
    """Lowercase alpha words from english-words 'web2'."""
    from english_words import get_english_words_set
    return set(get_english_words_set(["web2"], lower=True, alpha=True))


def neighbors():
    """8-way neighbor indices for each of the 16 cells."""
    result = []
    for i in range(CELLS):
        r, c = divmod(i, SIZE)
        result.append([
            (r + dr) * SIZE + (c + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < SIZE and 0 <= c + dc < SIZE
        ])
    return result


def filter_words(words, board):
    """Keep words of length 3-16 whose letters fit the board's letter counts."""
    avail = Counter(board.lower())
    out = []
    for w in words:
        if 3 <= len(w) <= CELLS:
            need = Counter(w)
            if all(avail[ch] >= n for ch, n in need.items()):
                out.append(w)
    return out


def build_trie(words):
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node[END] = w
    return root


def solve(board, words):
    """Return set of words from `words` found on `board`."""
    board = board.lower()
    trie = build_trie(filter_words(words, board))
    nbrs = neighbors()
    found = set()

    def dfs(i, node, mask):
        node = node.get(board[i])
        if node is None:
            return
        mask |= 1 << i
        word = node.get(END)
        if word is not None:
            found.add(word)
        for j in nbrs[i]:
            if not mask & (1 << j):
                dfs(j, node, mask)

    for i in range(CELLS):
        dfs(i, trie, 0)
    return found


def score(word):
    n = len(word)
    if n < 3:
        return 0
    if n <= 8:
        return _SCORES[n]
    return 2200 + 400 * (n - 8)


@lru_cache(maxsize=None)
def zipf(word):
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


def rank(found):
    """Return (common, less_common) sorted lists; zipf < 3 dropped."""
    common, less = [], []
    for w in found:
        z = zipf(w)
        if z >= COMMON_MIN:
            common.append((w, z))
        elif z >= LESS_MIN:
            less.append((w, z))
    key = lambda t: (-len(t[0]), -t[1], t[0])
    common.sort(key=key)
    less.sort(key=key)
    return [w for w, _ in common], [w for w, _ in less]


def parse_board(text):
    """Validate and normalise input; return 16 uppercase letters or raise ValueError."""
    s = "".join(text.split())
    if not re.fullmatch(r"[A-Za-z]{16}", s):
        raise ValueError(
            f"Board must be exactly 16 letters A-Z (got {len(s)} chars"
            + ("" if s.isalpha() else ", non-letters present") + ")."
        )
    return s.upper()


def _col(tier, i):
    if i >= len(tier):
        return ""
    w = tier[i]
    return f"{i + 1:2d}. {w.upper():<16} {score(w):>5} {zipf(w):>4.1f}"


def format_output(board, common, less, ms):
    lines = []
    for r in range(SIZE):
        lines.append(" ".join(board[r * SIZE:(r + 1) * SIZE]))
    lines.append("")
    width = 31
    lines.append(f"{'COMMON (zipf >= 5)':<{width}}  LESS COMMON (zipf 3-5)")
    lines.append(f"{'-' * width}  {'-' * width}")
    a, b = common[:TOP_N], less[:TOP_N]
    for i in range(max(len(a), len(b))):
        lines.append(f"{_col(a, i):<{width}}  {_col(b, i)}".rstrip())
    lines.append("")
    lines.append(f"Common: shown {len(a)} of {len(common)}")
    lines.append(f"Less common: shown {len(b)} of {len(less)}")
    total = sum(score(w) for w in common + less)
    lines.append(f"Total: {len(common) + len(less)} words, {total} points")
    lines.append(f"Solve time: {ms:.1f} ms")
    return "\n".join(lines)


def run(board, words):
    t = time.perf_counter()
    common, less = rank(solve(board, words))
    ms = (time.perf_counter() - t) * 1000
    print(format_output(board, common, less, ms))


def main(argv):
    if len(argv) > 1:
        try:
            board = parse_board(argv[1])
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        run(board, load_dictionary())
        return 0
    words = load_dictionary()
    while True:
        try:
            raw = input("Board: ")
        except EOFError:
            break
        if raw.strip() == "" or raw.strip().lower() == "q":
            break
        try:
            board = parse_board(raw)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        run(board, words)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
