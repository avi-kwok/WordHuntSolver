#!/usr/bin/env python3
"""Word Hunt (4x4 Boggle-style) solver: trie + DFS."""
import gc
import hashlib
import os
import pickle
import re
import sys
import time
from collections import defaultdict

SIZE = 4
CELLS = SIZE * SIZE
END = "$"
COMMON_MIN = 5.0
MIN_ZIPF = 3.0
LESS_MIN = MIN_ZIPF
RARE_MIN_ZIPF = 2.5
RARE_MIN_LEN = 4
DERIVED_MIN_ZIPF = 2.0
GROUP_POOL_MIN_ZIPF = 2.0  # floor for group-only near-miss words (below RARE_MIN_ZIPF)
TOP_N = 30
MIN_SHARED = 3  # shortest aligned run that makes two words match
GROUP_MAX = 6  # most top-level rows in one group (seed + picks)
# Entry-time model for ranking groups by points per second: the first
# word of a group costs FIND_FIRST_S (spot + plan it), each later word in the
# same group FIND_NEXT_S (pattern already known), plus a swipe cost per letter.
FIND_FIRST_S = 1.0   # spot + plan the first word of a group
FIND_NEXT_S = 0.4    # each later word in the same group (pattern already known)
SWIPE_PER_LETTER_S = 0.15
_SCORES = {3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800, 8: 2200}


BLOCKLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocklist.txt")
ALLOWLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allowlist.txt")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dict_cache.pkl")


class DerivedMap(dict):
    """derived_map that also carries `allowed` (allowlisted words rank() must never
    drop) and `group_only` (near-miss words that may only join a seed's group)."""
    allowed = frozenset()
    group_only = frozenset()


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


def load_allowlist(path=ALLOWLIST_PATH):
    """Same format as load_blocklist: words the solver must accept; empty set if missing."""
    return load_blocklist(path)


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


_ZIPF = {}  # word -> zipf; filled from the dictionary cache or lazily via wordfreq


def zipf(word):
    z = _ZIPF.get(word)
    if z is None:
        from wordfreq import zipf_frequency
        z = _ZIPF[word] = zipf_frequency(word, "en")
    return z


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


def _tier_idx(f):
    """0 common, 1 less common, 2 rare, from a wordfreq frequency."""
    if f >= 10 ** (COMMON_MIN - 9):
        return 0
    return 1 if f >= 10 ** (LESS_MIN - 9) else 2


def generate_derived(bases, freq_dict, min_zipf=DERIVED_MIN_ZIPF, exclude=frozenset()):
    """Map inflected word -> base. Keeps 3-16 letter forms with zipf >= min_zipf.

    Candidates in `exclude` (e.g. capitalized-only dictionary entries) are
    dropped. A candidate that is itself a base maps to root B only if B's
    tier is the same or more common than its own.
    """
    cutoff = 10 ** (min_zipf - 9)
    best = {}
    for b in bases:
        fb = freq_dict.get(b, 0)
        for cand in inflections(b):
            fc = freq_dict.get(cand, 0)
            if not 3 <= len(cand) <= CELLS or fc < cutoff or cand in exclude:
                continue
            if cand in bases and _tier_idx(fb) > _tier_idx(fc):
                continue
            if cand not in best or fb > best[cand][0]:
                best[cand] = (fb, b)
    return resolve_chains({w: b for w, (_, b) in best.items()})


def _chunk(a, b):
    """Longest same-index common run of a and b as (length, start); (0, 0) if none."""
    best = start = run = i = 0
    for x, y in zip(a, b):
        i += 1
        if x == y:
            run += 1
            if run > best:
                best, start = run, i - run
        else:
            run = 0
    return best, start


def shared_chunk(a, b):
    """Longest run (>= MIN_SHARED) of letters at the same position counted from the
    start of both words or from the end of both; '' if none. Ties: start
    alignment first, then earliest position."""
    n, i = _chunk(a, b)
    m, j = _chunk(a[::-1], b[::-1])
    if n >= MIN_SHARED and n >= m:
        return a[i:i + n]
    if m >= MIN_SHARED:
        return a[::-1][j:j + m][::-1]
    return ""


def _one_off(a, b):
    """True if a and b have the same length and differ in exactly one position."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        if x != y:
            diff += 1
            if diff > 1:
                return False
    return diff == 1


def match_len(a, b):
    """Match strength: the longer of len(shared_chunk(a, b)) and, for two
    same-length words differing in exactly one position, len(a) - 1; else 0."""
    return max(len(shared_chunk(a, b)), len(a) - 1 if _one_off(a, b) else 0)


def _shared_label(seed, word):
    """What a member shares with its seed: the chunk, else (one-letter
    difference with no 3-letter run) the seed with the differing letter as '_'."""
    c = shared_chunk(seed, word)
    if c or not _one_off(seed, word):
        return c
    return "".join(x if x == y else "_" for x, y in zip(seed, word))


def group_stats(words):
    """(points, seconds) for a group; `words` = ordered group words,
    first = best top-level word, then every other word incl. all nested forms."""
    points = sum(score(w) for w in words)
    t = sum(SWIPE_PER_LETTER_S * len(w) for w in words)
    t += FIND_FIRST_S + FIND_NEXT_S * (len(words) - 1)
    return points, t


def _build_index(words):
    """Candidate index over `words` (positions). Two rows can only match if they
    share a key: a MIN_SHARED run at the same index from the start (start[i][run])
    or from the end (end[i][run] on the reversed words), or the same word with one
    letter blanked (one_off[pattern]; the pattern's length is the word's length)."""
    start = [defaultdict(list) for _ in range(CELLS)]
    end = [defaultdict(list) for _ in range(CELLS)]
    one_off = defaultdict(list)
    m = MIN_SHARED
    for j, w in enumerate(words):
        rev = w[::-1]
        n = len(w)
        for i in range(n - m + 1):
            start[i][w[i:i + m]].append(j)
            end[i][rev[i:i + m]].append(j)
        for k in range(n):
            one_off[w[:k] + "_" + w[k + 1:]].append(j)
    return start, end, one_off


def _candidates(index, word):
    """Positions of every indexed row sharing a key with `word` (itself included)."""
    start, end, one_off = index
    rev = word[::-1]
    n = len(word)
    m = MIN_SHARED
    span = range(n - m + 1)
    found = [start[i].get(word[i:i + m], ()) for i in span]
    found += [end[i].get(rev[i:i + m], ()) for i in span]
    found += [one_off.get(word[:k] + "_" + word[k + 1:], ()) for k in range(n)]
    return set().union(*found)


def group_tiers(common, less, rare, pool=()):
    """Group top-level rows of all three tiers in ONE pass; return three lists.

    Seed = best unassigned row by (tier, -len, -zipf, word) (tier 0 common,
    1 less common, 2 rare; `pool` = group-only rows, tier 3, below rare). Its group is the seed plus up to GROUP_MAX-1
    unassigned rows of the SAME or a RARER tier with match_len(seed, row) >=
    MIN_SHARED (match against the seed only; see match_len for the
    one-letter-difference rule), except a row whose entire word is an aligned
    prefix or suffix of the seed (shared_chunk(seed, row) == row, e.g. "sis"
    for "basis"); it stays unassigned. A seed pulls rarer words up, never a
    more common word down. Picked by match_len desc, points desc, zipf desc,
    word. Pool rows are never seeds and never singles: one no seed picks is
    dropped (not in the output). The whole group goes in the SEED's column. Repeat until no rows
    are left; a seed with no match is a single. Nested forms are never
    matched or capped but count in stats.

    Sets row["group_key"] (seed word), row["shared"] (see _shared_label;
    None for the seed and singles), row["group_size"], row["group_points"],
    row["group_time"], row["group_rate"] and row["best_points"] (seed score;
    informational only). Each column orders its groups by one comparative
    sum: group_points desc (all top-level words plus all nested forms), then
    group_rate desc, size desc, seed word. Rows in a group: seed, then picks
    by (-len, -zipf, word).
    """
    zs = {}
    memo_zipf = lambda w: zs[w] if w in zs else zs.setdefault(w, zipf(w))
    key = lambda r: (-len(r["word"]), -memo_zipf(r["word"]), r["word"])
    all_rows = [r for rows in (common, less, rare, pool) for r in rows]
    tier_of = {id(r): t for t, rows in enumerate((common, less, rare, pool)) for r in rows}
    order = sorted(range(len(all_rows)), key=lambda i: (tier_of[id(all_rows[i])],) + key(all_rows[i]))
    rows_ = [all_rows[i] for i in order]
    tiers = [tier_of[id(r)] for r in rows_]
    index = _build_index([r["word"] for r in rows_])
    assigned = [False] * len(rows_)
    cols = ([], [], [], [])
    for si, seed in enumerate(rows_):
        if assigned[si]:
            continue
        assigned[si] = True
        st = tiers[si]
        if st == 3:
            continue
        sw = seed["word"]
        cands = []
        for j in _candidates(index, sw):
            if assigned[j] or tiers[j] < st:
                continue
            w = rows_[j]["word"]
            c = shared_chunk(sw, w)  # match_len(sw, w) with the chunk reused
            m = len(c)
            if m < len(sw) - 1 and _one_off(sw, w):
                m = len(sw) - 1
            if m >= MIN_SHARED and c != w:
                cands.append((m, j))
        cands.sort(key=lambda t: (-t[0], -score(rows_[t[1]]["word"]),
                                  -memo_zipf(rows_[t[1]]["word"]), rows_[t[1]]["word"]))
        for _, j in cands[:GROUP_MAX - 1]:
            assigned[j] = True
        picks = sorted((rows_[j] for _, j in cands[:GROUP_MAX - 1]), key=key)
        g = [seed] + picks
        words = []
        for r in g:
            words += [r["word"]] + list(r.get("forms", []))
        pts, secs = group_stats(words)
        rate = pts / secs
        best = score(sw)
        for r in g:
            r["group_key"], r["group_size"] = sw, len(g)
            r["shared"] = None if r is seed else _shared_label(sw, r["word"])
            r["group_points"], r["group_time"], r["group_rate"] = pts, secs, rate
            r["best_points"] = best
        cols[st].append((-pts, -rate, -len(g), sw, g))
    out = []
    for col in cols[:3]:
        col.sort(key=lambda t: t[:4])
        out.append([r for t in col for r in t[4]])
    return tuple(out)


def group_rows(rows):
    """Group rows treated as one tier (thin wrapper over group_tiers)."""
    return group_tiers(rows, [], [])[0]


def rank(found, derived_map=None):
    """Return (common, less_common, rare) lists of rows.

    Row: {"word", "base", "forms", "group_key", "group_size"}. Forms found with
    their base nest under it; a form found without its base is its own row
    with `base` set. Tier is decided by the base's zipf (< 2.5 dropped).
    Rare (2.5 <= zipf < 3) shows only words of 4+ letters; a 3-letter rare
    base counts as not found, so its forms stand alone. Top-level rows of all
    tiers are grouped in one pass (see group_tiers: a group can pull rarer
    words into the seed's column); forms
    sort (-len, -zipf, word). An allowlisted word with no derived root is
    never dropped: tier by its own zipf, Rare if below 3 (no length rule).
    """
    dm = derived_map if derived_map is not None else {}
    allowed = getattr(dm, "allowed", frozenset())
    pool_words = getattr(dm, "group_only", frozenset())
    found = set(found)

    def usable_base(b):
        return b in found and (zipf(b) >= LESS_MIN or len(b) >= RARE_MIN_LEN)

    rows = {}
    nested = {}
    pool = []
    for w in found:
        root = dm.get(w)
        if w in pool_words and w not in allowed and root is None:
            pool.append({"word": w, "base": None, "forms": []})
        elif root is None:
            rows[w] = {"word": w, "base": None, "forms": []}
        elif usable_base(root):
            nested.setdefault(root, []).append(w)
        else:
            rows[w] = {"word": w, "base": root, "forms": []}
    key = lambda w: (-len(w), -zipf(w), w)
    common, less, rare = [], [], []
    for w, row in rows.items():
        z = zipf(row["base"] or w)
        forms = nested.get(w, [])
        if w in allowed and row["base"] is None and z < MIN_ZIPF:
            row["forms"] = sorted((f for f in forms if len(f) >= RARE_MIN_LEN), key=key)
            rare.append(row)
            continue
        if z < RARE_MIN_ZIPF:
            continue
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
    return group_tiers(common, less, rare, pool)


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


def _fingerprint(blocklist_path=BLOCKLIST_PATH, allowlist_path=ALLOWLIST_PATH):
    """sha256 over everything that shapes the dictionary: this file's bytes, the
    block/allow lists and the installed english-words / wordfreq versions."""
    from importlib import metadata
    h = hashlib.sha256()

    def feed(label, data):
        h.update(label.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)

    for label, path in (("code", os.path.abspath(__file__)),
                        ("block", blocklist_path), ("allow", allowlist_path)):
        try:
            with open(path, "rb") as f:
                feed(label, f.read())
        except FileNotFoundError:
            feed(label, b"<missing>")
    for pkg in ("english-words", "wordfreq"):
        try:
            v = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v = "<none>"
        feed(pkg, v.encode())
    return h.hexdigest()


def _build(blocklist_path, allowlist_path):
    """Full dictionary pipeline. Returns (trie, derived_map, zipfs, stats)."""
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
    derived_map = generate_derived(bases, freq, exclude=lowered - all_words)
    blocked = load_blocklist(blocklist_path)
    n_blocked = len((bases | set(derived_map)) & blocked)
    bases -= blocked
    derived_map = {w: b for w, b in derived_map.items() if w not in blocked}
    allowed = {w for w in load_allowlist(allowlist_path)
               if w.isalpha() and w.isascii() and size_ok(w)} - blocked
    derived_map = DerivedMap(resolve_chains(derived_map))
    derived_map.allowed = frozenset(allowed)
    pool_cands = {w for w in all_words if RARE_MIN_LEN <= len(w) <= CELLS}
    group_only = (filter_common(pool_cands, freq, GROUP_POOL_MIN_ZIPF)
                  - filter_common(pool_cands, freq, RARE_MIN_ZIPF)
                  - bases - set(derived_map) - allowed - blocked)
    derived_map.group_only = frozenset(group_only)
    in_trie = (bases | set(derived_map) | allowed | group_only) - blocked
    trie = build_trie(in_trie)
    zipfs = {w: zipf(w) for w in in_trie if 3 <= len(w) <= CELLS}
    stats = {"bases": len(bases), "derived": len(derived_map), "proper": proper,
             "blocked": n_blocked, "allowed": len(allowed), "group_only": len(group_only)}
    return trie, derived_map, zipfs, stats


def _write_cache(path, fingerprint, payload):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(fingerprint, f, protocol=5)
        pickle.dump(payload, f, protocol=5)
    os.replace(tmp, path)


def _read_cache(path, fingerprint):
    """Payload tuple from the cache at `path`, or None if missing, stale or unreadable.
    The pickle is a trusted local file generated by `--build-cache`."""
    try:
        with open(path, "rb") as f:
            if pickle.load(f) != fingerprint:
                return None
            gc.disable()
            try:
                return pickle.load(f)
            finally:
                gc.enable()
    except Exception:
        return None


def build_cache(path=None, blocklist_path=BLOCKLIST_PATH, allowlist_path=ALLOWLIST_PATH):
    """Run the full build and write the dictionary cache (errors propagate)."""
    path = path or CACHE_PATH
    t = time.perf_counter()
    payload = _build(blocklist_path, allowlist_path)
    _write_cache(path, _fingerprint(blocklist_path, allowlist_path), payload)
    print(f"Wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) in "
          f"{time.perf_counter() - t:.1f} s")


def startup(blocklist_path=BLOCKLIST_PATH, allowlist_path=ALLOWLIST_PATH, cache_path=None):
    """Load dictionary, build trie once, warm up wordfreq.

    Words in the blocklist are removed (exact words only) from bases and
    derived forms. Allowlisted words (alpha, 3-16 letters, not blocked) are
    always added to the trie; one with no derived root is an ordinary
    top-level row. Returns (trie, derived_map); derived_map.allowed holds
    the allowlist.

    Uses the prebuilt cache (`--build-cache`) when its fingerprint matches the
    current inputs; otherwise builds from scratch and tries to refresh the cache
    (only for the default list paths; write errors are ignored).
    """
    t = time.perf_counter()
    cache_path = cache_path or CACHE_PATH
    fp = _fingerprint(blocklist_path, allowlist_path)
    payload = _read_cache(cache_path, fp)
    source = "cache"
    if payload is None:
        source = "built"
        payload = _build(blocklist_path, allowlist_path)
        if blocklist_path == BLOCKLIST_PATH and allowlist_path == ALLOWLIST_PATH:
            try:
                _write_cache(cache_path, fp, payload)
            except Exception:
                pass
    trie, derived_map, zipfs, st = payload
    _ZIPF.update(zipfs)
    ms = (time.perf_counter() - t) * 1000
    print(f"Loaded {st['bases']} base + {st['derived']} derived words in {ms:.0f} ms ({source})")
    print(f"  zipf >= {MIN_ZIPF:g}; rare >= {RARE_MIN_ZIPF:g} ({RARE_MIN_LEN}+); "
          f"forms >= {DERIVED_MIN_ZIPF:g}; {st['proper']} proper nouns; {st['blocked']} blocked; {st['allowed']} allowed; {st['group_only']} group-only")
    return trie, derived_map


def main(argv):
    if len(argv) > 1 and argv[1] == "--build-cache":
        build_cache()
        return 0
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
