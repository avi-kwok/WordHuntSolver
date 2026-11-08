# WordHuntSolver
A project made as a joke for fun. I'm actually decent at word hunt so this is not me coping with losing word hunt games!!!

Solver for 4x4 Boggle-style Word Hunt boards, with a CLI and a local web UI. Finds every dictionary word (3+ letters) traced through 8-way adjacent tiles, no tile reused, ranks them by how common they are, and groups words that are quick to enter one after another.

## Setup
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage
```
.venv/bin/python wordhunt.py SERSPATGLINESERS   # one board, then exit
.venv/bin/python wordhunt.py                    # interactive; empty line or q quits
.venv/bin/python wordhunt.py --build-cache       # rebuild dict_cache.pkl (see Startup cache)
.venv/bin/python test_wordhunt.py               # tests
```
Board = 16 letters, row-major (left to right, top to bottom), case-insensitive.

## Algorithm
1. Load the `web2` word list (english-words), keeping only words that appear in lowercase (drops proper nouns such as Yemen or Eden).
2. Startup prefilter: keep only words with zipf >= 2.5 (`RARE_MIN_ZIPF`; via wordfreq's frequency dict), add generated inflected forms, apply `blocklist.txt` / `allowlist.txt`.
3. Build one trie from words of length 3-16 (nested dicts, terminal key `$` holds the word), reused for every board. All of this is normally loaded from the prebuilt `dict_cache.pkl` (see Startup cache).
4. DFS from each cell, walking the trie in lockstep; visited cells tracked as a 16-bit bitmask; prune as soon as the trie has no child for the next letter (a typical board explores only ~500-2,000 partial paths out of ~12 million possible).

## Inflected forms
web2 lacks most plurals and inflections. At startup, for every base word (web2, 3-16 letters, zipf >= 2.5) the program generates candidate forms (-s/-es, -ed, -ing, -er/-est/-ers, -ly, with spelling variants such as e-drop, y to ies, consonant doubling) and keeps those whose wordfreq frequency gives zipf >= 2 (checked against `get_frequency_dict`, no per-word lookup). Forms are added to the trie and mapped to a base (highest-frequency base wins; chains resolve to the root). In results a found form nests under its base's row, or stands alone labelled with its base if the base is not on the board. Tier is decided by the base's zipf. Each column shows at most 30 lines (base rows plus forms, in order; a group may be cut partway); totals count every found word.

## Blocklist
`blocklist.txt` lists words the game rejects (one word per line, case-insensitive, `#` comments and blank lines ignored), e.g. apostrophe-less contractions like `dont`. Blocked words never enter the trie. Exact words only: blocking a base does not block its forms.

## Allowlist
`allowlist.txt` (same format as `blocklist.txt`: one word per line, case-insensitive, `#` comments and blank lines ignored) lists words the game accepts that the dictionary lacks. Allowlisted alphabetic 3-16 letter words always enter the trie, whatever web2 or zipf says; the blocklist wins if a word is in both. Placement: a word with a derived root behaves normally; otherwise it is an ordinary top-level base row (no prefix-nesting: lande is not a form of land, hote not of hot; it joins groups by normal matching, e.g. lande groups with land), tiered by its own zipf, Rare if below 3, never dropped by the zipf floors or the 4-letter rare rule. The startup line reports the count (`... 24 blocked; 4 allowed; N group-only`).

## Scoring
3=100, 4=400, 5=800, 6=1400, 7=1800, 8=2200, then +400 per letter beyond 8.

## Tier rules
Frequency via `wordfreq.zipf_frequency(word, 'en')`.
- COMMON: zipf >= 5.0
- LESS COMMON: 3.0 <= zipf < 5.0
- RARE: 2.5 <= zipf < 3.0, words of 4+ letters only (bases, nested forms and standalone forms alike). A 3-letter rare base counts as not found, so its 4+ letter forms appear as standalone "from base" rows.
- zipf < 2.5: dropped (not shown, not counted), except allowlisted words and group-only pool words that join a group (below)

Grouping runs once over all three tiers (see Cross-tier below); top-level rows (bases and standalone forms) are grouped by a shared aligned chunk. Two words MATCH if they share a run of `MIN_SHARED` (3) or more consecutive letters at the same position, counted from the start of both words or from the end of both (`match_len(a, b)` = longest such run, 0 if none; `shared_chunk(a, b)` = that substring, ties broken start alignment first). A row whose entire word is an aligned prefix or suffix of the seed (e.g. "sis" or "bas" for seed "basis", "ama" for "lama") never joins the seed's group; it stays unassigned and can seed or join another group. Examples: brat/brag ("bra"), bong/wrong ("ong", from the end), strange/strain ("stra"), sing/ring ("ing"); lama/ama ("ama") match as words but do not group. One-letter difference: two same-length words (3+ letters) that differ in exactly one position also match, `match_len` = len - 1 (vote/note/dote/hote, bat/cat); this still needs `match_len >= MIN_SHARED`, so 3-letter pairs like cat/cot match but cannot group; different lengths (vote/votes) or two differences never use this rule. cat/dog and brag/trap (only "ra") do not match.

Seed algorithm (one pass over Common, Less common and Rare together; `group_tiers`): sort unassigned rows by (tier, length desc, zipf desc, word); the first is the SEED. Cross-tier: candidates are unassigned rows of the SAME or a RARER tier as the seed, so a seed pulls rarer words up into its group (placed in the seed's column) but never pulls a more common word down (tether, Rare, never captures other, Common; other, as the seed, pulls tether up). Candidates are unassigned rows with `match_len(seed, row) >= MIN_SHARED` (against the seed only, never other members). Up to `GROUP_MAX - 1` (`GROUP_MAX = 6`) are picked by match_len desc, points desc, zipf desc, word. Seed plus picks form a group and are assigned; repeat until no rows are left. A seed with no candidates is a single. In a group the seed comes first, then the picks by (length desc, zipf desc, word). Nested forms stay under their base row: they are not matched and do not count toward `GROUP_MAX`, but they count in group points and time.

Group-only pool: at startup lowercase-only web2 words of 4-16 letters, not blocked, not allowlisted, not a base or derived word, with `GROUP_POOL_MIN_ZIPF` (2.0) <= zipf < 2.5 enter the trie (no inflections) and `derived_map.group_only`. They rank below Rare, can only join a group as a picked candidate (same match rules and cap), never seed, and one no seed picks is dropped from output and totals. Because pulled-up members can be longer or higher scoring than the seed, `best_points` may be below a member's own points. Group order (per column): one comparative sum. Groups sort by `group_points` desc (all top-level rows plus all nested forms), then `group_rate` desc, size desc, seed word. A single row competes on its own total (e.g. save + saved + saver = 2000 beats a 2-word group worth 1800); there is no "groups before singles" rule. `best_points` (seed score) is still set and exposed by the API but does not affect order. `group_rate = group_points / group_time` over the full group (top-level rows plus every nested form, before the 30-line cap); `group_points` is the sum of `score()`; `group_time` charges the group's first row `FIND_FIRST_S (1.0) + SWIPE_PER_LETTER_S (0.15) * len` and every other word (other rows, all nested forms) `FIND_NEXT_S (0.4) + 0.15 * len`. Rank numbers run continuously and the 30-line cap applies after grouping (a group may be cut). In the UI each group of 2+ is one black-outlined box with a header (`<points> pts · N words · rate/s`), and a single word with suffix forms gets a lighter header without a box.

Each column shows the top 30 lines; totals (words and points) cover all three tiers.

## Web UI
Local only (binds 127.0.0.1), stdlib only, no extra dependencies.
```
.venv/bin/python server.py [--port 8000]
```
Open http://127.0.0.1:8000. Enter the board in the 4x4 grid (or paste/type all 16 letters) and press Solve. Hover a result row to preview its path on the board (green tiles, green line with direction arrows, green dot on the first letter); click to pin it. On widescreen (1000px+) the lists sit on the left and the board stays visible on the right; on narrow screens everything stacks and spacing tightens. "New board" clears everything. API: `POST /api/solve` with `{"board": "<16 letters>"}` returns `common`, `less_common` and `rare` (top 30 lines each, rows carry `group_key` (the seed word), `shared` (chunk shared with the seed; null for the seed and singles), `group_size`, `group_rate`, `group_points`, `group_time`, `best_points`) with paths, plus `*_total` counts and overall totals. Tests: `.venv/bin/python test_server.py`.


## Top groups view
The results screen has a `Columns` / `Top groups` toggle. `Top groups` merges the full (untruncated) ranking of all three tiers into one list of units, one per group (multi-word group, single with forms, or plain single), sorted by group points, then rate, then seed word, and capped at 60. The API returns it as `top` (`key`, `points`, `rate`, `words`, `rows`; rows have the same shape as the column rows). On widescreen (1000px+) the page packs the units, in rank order, into viewport-height columns (2-5, about 240px each) with no page scroll and shows `showing N of M groups`; on narrow screens it is one scrolling column.

## Startup cache
`startup()` normally takes about 1.1-1.5 s (dictionary, inflections, trie). `python wordhunt.py --build-cache` runs that full pipeline once and pickles (protocol 5) the trie, `derived_map` (with `.allowed` and `.group_only`) and a zipf value for every word in the trie to `dict_cache.pkl`. With the cache, `startup()` takes about 0.1 s in a fresh process and never imports wordfreq or english-words; it prints `(cache)` or `(built)` plus the time.

The cache carries a sha256 fingerprint of `wordhunt.py`, `blocklist.txt`, `allowlist.txt` and the installed english-words and wordfreq versions. If it is missing or does not match, `startup()` falls back to the full build (and tries to refresh the file, ignoring write errors, since Vercel's filesystem is read-only). **Rebuild the cache (`python wordhunt.py --build-cache`) after editing `blocklist.txt`, `allowlist.txt` or `wordhunt.py`, otherwise startup falls back to the slow build.** `dict_cache.pkl` is committed (not gitignored) so it ships to Vercel. The pickle is a trusted local file generated by this repo; never load one from elsewhere.

Grouping (`group_tiers`) uses an index (start-aligned and end-aligned 3-letter runs, one-letter-difference patterns) so each seed only checks rows that can match; `rank()` takes about 6 ms on a typical board.

## Deploy (Vercel)
Vercel picks up the top-level `app` (WSGI callable) in `server.py` and installs `requirements.txt`. The trie loads lazily on the first request per process, so the first request after a cold start takes about 0.1 s while the cached dictionary loads (about 1.5 s if the cache is stale); warm invocations reuse it. Memory use is about 60 MB with the cache (measured locally). Not yet deployed from this repo, so the Vercel build itself is unverified; `app` is covered by `test_server.py` (called directly with a WSGI environ).

## Version history
- **v1**: Trie + DFS solver, dictionary + frequency for reasonable results
- **v2+v3**: Sorted word results + built trie at startup -> ~1.9s to ~13ms per board
- **v4**: Filter dictionary to include zipf >= 3 words only -> 17k words + ~5 ms per board
- **v5**: Simple UI
- **v6**: Grouped suffix/prefix words
- **v7**: Improved cutoff for better results
- **v8**: Excluded proper nouns + created blocklist
- **v9**: Grouped spelling rhymes for faster inputs
- **v10**: Bug fixes
- **v11**: Added points-per-second ranking for even better inputs!
- **v12**: UI restyle to larp word hunt.
- **v13**: QoL word grouping added, ready for deployment
- **v14**: Added BFS solver
- **v15**: Word path and column/board UI updates
- **v16**: Removed BFS, indexed grouping + prebuilt dictionary cache -> ~3 ms per board, ~0.1s cold start
- **v17**: Simple UI fixes
