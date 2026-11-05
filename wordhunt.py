#!/usr/bin/env python3
"""Word Hunt (4x4 Boggle-style) solver: trie + DFS."""
import os
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
RARE_MIN_ZIPF = 2.5
RARE_MIN_LEN = 4
DERIVED_MIN_ZIPF = 2.0
TOP_N = 30
_SCORES = {3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800, 8: 2200}


BLOCKLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocklist.txt")


def load_dictionary():
    """Alpha words from english-words 'web2' that appear in lowercase (no proper nouns)."""
    from english_words import get_english_words_set
    return {w for w in get_english_words_set(["web2"], lower=False, alpha=True)
            if w.islower()}


def load_blocklist(path=BLOCKLIST_PATH):
    """Lowercase words from `path` (one per line; blank and # lines ignored); empty set if missing."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
    except FileNotFoundError:
        return set()
    return {ln.lower() for ln in lines if ln and not ln.startswith("#")}


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


_VOWELS = set("aeiou")


def _cons_y(w):
    return w.endswith("y") and len(w) > 1 and w[-2] not in _VOWELS


def inflections(w):
    """Every plausible suffixed spelling of `w`; the frequency check prunes."""
    c = set()
    dbl = w + w[-1]
    # -s / -es
    c.add(w + "s")
    if w.endswith(("s", "x", "z", "ch", "sh")):
        c.add(w + "es")
    if _cons_y(w):
        c.add(w[:-1] + "ies")
    # -ed
    c.add(w + "ed")
    if w.endswith("e"):
        c.add(w + "d")
    if _cons_y(w):
        c.add(w[:-1] + "ied")
    c.add(dbl + "ed")
    # -ing
    c.add(w + "ing")
    if w.endswith("e") and not w.endswith("ee"):
        c.add(w[:-1] + "ing")
    if w.endswith("ie"):
        c.add(w[:-2] + "ying")
    c.add(dbl + "ing")
    # -er / -est (and -ers)
    ers = {w + "er", dbl + "er"}
    ests = {w + "est", dbl + "est"}
    if w.endswith("e"):
        ers.add(w + "r")
        ests.add(w + "st")
    if _cons_y(w):
        ers.add(w[:-1] + "ier")
        ests.add(w[:-1] + "iest")
    c |= ers | ests | {x + "s" for x in ers}
    # -ly
    c.add(w + "ly")
    if _cons_y(w):
        c.add(w[:-1] + "ily")
    if w.endswith("le"):
        c.add(w[:-1] + "y")
    if w.endswith("ic"):
        c.add(w + "ally")
    c.discard(w)
    return c


def resolve_chains(derived_map):
    """Point every word at its root (A->B, B->C gives A->C); no self maps."""
    out = {}
    for w, b in derived_map.items():
        seen = {w}
        while b in derived_map and b not in seen:
            seen.add(b)
            b = derived_map[b]
        if b not in seen:
            out[w] = b
    return out


def generate_derived(bases, freq_dict, min_zipf=DERIVED_MIN_ZIPF):
    """Map inflected word -> base. Keeps 3-16 letter forms with zipf >= min_zipf."""
    cutoff = 10 ** (min_zipf - 9)
    best = {}
    for b in bases:
        fb = freq_dict.get(b, 0)
        for cand in inflections(b):
            if not 3 <= len(cand) <= CELLS or freq_dict.get(cand, 0) < cutoff:
                continue
            if cand not in best or fb > best[cand][0]:
                best[cand] = (fb, b)
    return resolve_chains({w: b for w, (_, b) in best.items()})


def rhyme_key(word):
    """`word` minus its leading consonant onset (its spelled rhyme ending).

    y is a consonant only at index 0; u right after a leading q joins the
    onset. A vowel-initial word is its own key; an all-consonant word too.
    """
    n = len(word)
    i = 0
    while i < n and word[i] not in _VOWELS and not (word[i] == "y" and i > 0):
        i += 1
    if i == n:
        return word
    if i and word[i - 1] == "q" and word[i] == "u":
        i += 1
    return word[i:]


def group_rows(rows):
    """Order top-level rows of one tier into contiguous rhyme groups.

    Sets row["rhyme"] and row["group_size"]. Groups: size desc, longest
    member desc, best member zipf desc, key. Rows in a group: (-len, -zipf, word).
    """
    key = lambda r: (-len(r["word"]), -zipf(r["word"]), r["word"])
    groups = {}
    for r in rows:
        groups.setdefault(rhyme_key(r["word"]), []).append(r)
    ordered = []
    for k, g in groups.items():
        g.sort(key=key)
        for r in g:
            r["rhyme"], r["group_size"] = k, len(g)
        ordered.append((-len(g), -len(g[0]["word"]),
                        -max(zipf(r["word"]) for r in g), k, g))
    ordered.sort(key=lambda t: t[:4])
    return [r for t in ordered for r in t[4]]


def rank(found, derived_map=None):
    """Return (common, less_common, rare) lists of rows.

    Row: {"word", "base", "forms", "rhyme", "group_size"}. Forms found with
    their base nest under it; a form found without its base is its own row
    with `base` set. Tier is decided by the base's zipf (< 2.5 dropped).
    Rare (2.5 <= zipf < 3) shows only words of 4+ letters; a 3-letter rare
    base counts as not found, so its forms stand alone. Tiers are grouped
    by rhyme_key over top-level rows; forms sort (-len, -zipf, word).
    """
    dm = derived_map or {}
    found = set(found)

    def usable_base(b):
        return b in found and (zipf(b) >= LESS_MIN or len(b) >= RARE_MIN_LEN)

    rows = {}
    nested = {}
    for w in found:
        root = dm.get(w)
        if root is None:
            rows[w] = {"word": w, "base": None, "forms": []}
        elif usable_base(root):
            nested.setdefault(root, []).append(w)
        else:
            rows[w] = {"word": w, "base": root, "forms": []}
    key = lambda w: (-len(w), -zipf(w), w)
    common, less, rare = [], [], []
    for w, row in rows.items():
        z = zipf(row["base"] or w)
        if z < RARE_MIN_ZIPF:
            continue
        forms = nested.get(w, [])
        if z >= COMMON_MIN:
            tier = common
        elif z >= LESS_MIN:
            tier = less
        else:
            if len(w) < RARE_MIN_LEN:
                continue
            tier = rare
            forms = [f for f in forms if len(f) >= RARE_MIN_LEN]
        row["forms"] = sorted(forms, key=key)
        tier.append(row)
    return group_rows(common), group_rows(less), group_rows(rare)


def truncate_rows(rows, limit=TOP_N):
    """First `limit` display lines (base rows + forms) of a tier.

    A group may be cut partway: its base row stays, remaining forms drop.
    Returns (rows, shown_line_count); input rows are not modified.
    """
    out, left = [], limit
    for r in rows:
        if left <= 0:
            break
        forms = r["forms"][:left - 1]
        out.append({**r, "forms": forms})
        left -= 1 + len(forms)
    return out, limit - left


def tier_words(rows):
    """Every word in a tier, forms included."""
    return [w for r in rows for w in [r["word"]] + r["forms"]]


def parse_board(text):
    """Validate and normalise input; return 16 uppercase letters or raise ValueError."""
    s = "".join(text.split())
    if not re.fullmatch(r"[A-Za-z]{16}", s):
        raise ValueError(
            f"Board must be exactly 16 letters A-Z (got {len(s)} chars"
            + ("" if s.isalpha() else ", non-letters present") + ")."
        )
    return s.upper()


def _trunc(text, n):
    return text if len(text) <= n else text[:n - 1] + "~"


def _lines(rows):
    """Display lines (31 cols) for a tier's top-level rows and their forms."""
    out = []
    for i, r in enumerate(rows):
        w = r["word"]
        label = w.upper() + (f" ({r['base']})" if r["base"] else "")
        out.append(f"{i + 1:2d}. {_trunc(label, 16):<16} {score(w):>5} {zipf(w):>4.1f}")
        for f in r["forms"]:
            out.append(f"    + {_trunc(f.upper(), 14):<14} {score(f):>5} {zipf(f):>4.1f}")
    return out


def format_output(board, common, less, rare, ms):
    lines = []
    for r in range(SIZE):
        lines.append(" ".join(board[r * SIZE:(r + 1) * SIZE]))
    lines.append("")
    width = 31
    heads = ["COMMON (zipf >= 5)", "LESS COMMON (zipf 3-5)",
             f"RARE (zipf {RARE_MIN_ZIPF:g}-{MIN_ZIPF:g}, {RARE_MIN_LEN}+)"]
    lines.append("  ".join(f"{h:<{width}}" for h in heads).rstrip())
    lines.append("  ".join("-" * width for _ in heads))
    tiers = [truncate_rows(t) for t in (common, less, rare)]
    cols = [_lines(t[0]) for t in tiers]
    for i in range(max(len(c) for c in cols)):
        cells = [c[i] if i < len(c) else "" for c in cols]
        lines.append("  ".join(f"{c:<{width}}" for c in cells).rstrip())
    lines.append("")
    for name, t, (_, n) in zip(("Common", "Less common", "Rare"),
                               (common, less, rare), tiers):
        lines.append(f"{name}: shown {n} of {len(tier_words(t))}")
    words = tier_words(common) + tier_words(less) + tier_words(rare)
    total = sum(score(w) for w in words)
    lines.append(f"Total: {len(words)} words, {total} points")
    lines.append(f"Solve time: {ms:.1f} ms")
    return "\n".join(lines)


def run(board, trie, derived_map):
    t = time.perf_counter()
    common, less, rare = rank(solve(board, trie), derived_map)
    ms = (time.perf_counter() - t) * 1000
    print(format_output(board, common, less, rare, ms))


def startup(blocklist_path=BLOCKLIST_PATH):
    """Load dictionary, build trie once, warm up wordfreq.

    Words in the blocklist are removed (exact words only) from bases and
    derived forms. Returns (trie, derived_map).
    """
    t = time.perf_counter()
    from english_words import get_english_words_set
    from wordfreq import get_frequency_dict
    zipf("the")
    freq = get_frequency_dict("en")
    all_words = load_dictionary()
    lowered = set(get_english_words_set(["web2"], lower=True, alpha=True))
    size_ok = lambda w: 3 <= len(w) <= CELLS
    proper = len(filter_common({w for w in lowered - all_words if size_ok(w)}, freq, RARE_MIN_ZIPF))
    sized = {w for w in all_words if size_ok(w)}
    bases = filter_common(sized, freq, RARE_MIN_ZIPF)
    derived_map = generate_derived(bases, freq)
    blocked = load_blocklist(blocklist_path)
    n_blocked = len((bases | set(derived_map)) & blocked)
    bases -= blocked
    derived_map = {w: b for w, b in derived_map.items() if w not in blocked}
    trie = build_trie((bases | set(derived_map)) - blocked)
    ms = (time.perf_counter() - t) * 1000
    print(f"Loaded {len(bases)} base + {len(derived_map)} derived words in {ms:.0f} ms")
    print(f"  zipf >= {MIN_ZIPF:g}; rare >= {RARE_MIN_ZIPF:g} ({RARE_MIN_LEN}+ letters); "
          f"forms >= {DERIVED_MIN_ZIPF:g}; {proper} proper nouns removed; {n_blocked} blocked")
    return trie, derived_map


def main(argv):
    if len(argv) > 1:
        try:
            board = parse_board(argv[1])
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        run(board, *startup())
        return 0
    trie, derived_map = startup()
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
        run(board, trie, derived_map)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
