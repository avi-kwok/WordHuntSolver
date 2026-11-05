#!/usr/bin/env python3
"""Word Hunt (4x4 Boggle-style) solver: trie + DFS."""
import re
import sys
import time
from functools import lru_cache

SIZE = 4
CELLS = SIZE * SIZE
END = "$"
COMMON_MIN = 5.0
MIN_ZIPF = 3.0
LESS_MIN = MIN_ZIPF
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


def build_trie(words):
    """Trie of words with length 3-16 (nested dicts, terminal key `$`)."""
    root = {}
    for w in words:
        if not 3 <= len(w) <= CELLS:
            continue
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node[END] = w
    return root


def solve_paths(board, trie):
    """Map each word found on `board` to its first path (cell indices 0-15)."""
    board = board.lower()
    nbrs = neighbors()
    found = {}
    path = []

    def dfs(i, node, mask):
        node = node.get(board[i])
        if node is None:
            return
        mask |= 1 << i
        path.append(i)
        word = node.get(END)
        if word is not None and word not in found:
            found[word] = list(path)
        for j in nbrs[i]:
            if not mask & (1 << j):
                dfs(j, node, mask)
        path.pop()

    for i in range(CELLS):
        dfs(i, trie, 0)
    return found


def solve(board, trie):
    """Return set of words in prebuilt `trie` found on `board`."""
    return set(solve_paths(board, trie))


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


def filter_common(words, freq_dict, min_zipf=MIN_ZIPF):
    """Keep words with zipf >= min_zipf using a wordfreq frequency dict.

    zipf = log10(freq * 1e9), so zipf >= z  <=>  freq >= 10 ** (z - 9).
    """
    cutoff = 10 ** (min_zipf - 9)
    return {w for w in words if freq_dict.get(w, 0) >= cutoff}


def rank(found):
    """Return (common, less_common); zipf picks tier (< 3 dropped).

    Each tier sorts by length desc, zipf desc, then alphabetical.
    """
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


def run(board, trie):
    t = time.perf_counter()
    common, less = rank(solve(board, trie))
    ms = (time.perf_counter() - t) * 1000
    print(format_output(board, common, less, ms))


def startup():
    """Load dictionary, build trie once, warm up wordfreq; return trie."""
    t = time.perf_counter()
    from wordfreq import get_frequency_dict
    zipf("the")
    all_words = load_dictionary()
    sized = {w for w in all_words if 3 <= len(w) <= CELLS}
    words = filter_common(sized, get_frequency_dict("en"))
    trie = build_trie(words)
    ms = (time.perf_counter() - t) * 1000
    print(f"Loaded {len(words)} of {len(sized)} words "
          f"(zipf >= {MIN_ZIPF:g}) in {ms:.0f} ms")
    return trie


def main(argv):
    if len(argv) > 1:
        try:
            board = parse_board(argv[1])
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        run(board, startup())
        return 0
    trie = startup()
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
        run(board, trie)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
