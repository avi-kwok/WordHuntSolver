# WordHuntSolver
A project made as a joke for fun. I'm actually decent at word hunt so this is not me coping with losing word hunt games!!!

CLI solver for 4x4 Boggle-style Word Hunt boards. Finds every dictionary word (3+ letters) traced through 8-way adjacent tiles, no tile reused, and ranks them by how common they are.

## Setup
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage
```
.venv/bin/python wordhunt.py SERSPATGLINESERS   # one board, then exit
.venv/bin/python wordhunt.py                    # interactive; empty line or q quits
.venv/bin/python test_wordhunt.py               # tests
```
Board = 16 letters, row-major (left to right, top to bottom), case-insensitive.

## Algorithm
1. Load `web2` word list (english-words).
2. Build one trie at startup from words of length 3-16 (nested dicts, terminal key `$` holds the word), reused for every board.
3. Startup prefilter: keep only words with zipf >= 2.5 (`RARE_MIN_ZIPF`; via wordfreq's frequency dict) before building the trie.
4. DFS from each cell, walking the trie in lockstep; visited cells tracked as a 16-bit bitmask; prune as soon as the trie has no child for the next letter.

## Inflected forms
web2 lacks most plurals and inflections. At startup, for every base word (web2, 3-16 letters, zipf >= 2.5) the program generates candidate forms (-s/-es, -ed, -ing, -er/-est/-ers, -ly, with spelling variants such as e-drop, y to ies, consonant doubling) and keeps those whose wordfreq frequency gives zipf >= 2 (checked against `get_frequency_dict`, no per-word lookup). Forms are added to the trie and mapped to a base (highest-frequency base wins; chains resolve to the root). In results a found form nests under its base's row, or stands alone labelled with its base if the base is not on the board. Tier is decided by the base's zipf. Each column shows at most 30 lines (base rows plus forms, in order; a group may be cut partway); totals count every found word.

## Scoring
3=100, 4=400, 5=800, 6=1400, 7=1800, 8=2200, then +400 per letter beyond 8.

## Tier rules
Frequency via `wordfreq.zipf_frequency(word, 'en')`.
- COMMON: zipf >= 5.0
- LESS COMMON: 3.0 <= zipf < 5.0
- RARE: 2.5 <= zipf < 3.0, words of 4+ letters only (bases, nested forms and standalone forms alike). A 3-letter rare base counts as not found, so its 4+ letter forms appear as standalone "from base" rows.
- zipf < 2.5: dropped (not shown, not counted)

Rhyme grouping: within each tier, top-level rows (bases and standalone forms) are grouped by `rhyme_key(word)`, the word minus its leading consonant onset (`y` is a consonant only first; `u` after a leading `q` joins the onset; vowel-initial or all-consonant words keep the whole word). Groups order by estimated points per second of entry time: `group_rate = group_points / group_time`, over the full group (top-level rows plus every nested form, before the 30-line cap). `group_points` is the sum of `score()`; `group_time` charges the group's first row `FIND_FIRST_S (1.0) + SWIPE_PER_LETTER_S (0.15) * len` and every other word (other rows, all nested forms) `FIND_NEXT_S (0.4) + 0.15 * len`. Ties: best row's points desc, group size desc, key. Rows within a group: length desc, zipf desc, alphabetical. Nested forms do not change which group a row is in, only its rate. Rank numbers run continuously and the 30-line cap applies after grouping (a group may be cut). The UI shades groups of 2+ and tags the first row with its rhyme and rate (e.g. `-ONG x3 · 512/s`; single rows get just `512/s`).

Each column shows the top 30 lines; totals (words and points) cover all three tiers.

## Web UI
Local only (binds 127.0.0.1), stdlib only, no extra dependencies.
```
.venv/bin/python server.py [--port 8000]
```
Open http://127.0.0.1:8000. Enter the board in the 4x4 grid (or paste/type all 16 letters), press Solve, then click a result row to see its path drawn on the full-size board. API: `POST /api/solve` with `{"board": "<16 letters>"}` returns `common`, `less_common` and `rare` (top 30 lines each, rows carry `rhyme`, `group_size`, `group_rate`, `group_points`, `group_time`) with paths, plus `*_total` counts and overall totals. Tests: `.venv/bin/python test_server.py`.

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
