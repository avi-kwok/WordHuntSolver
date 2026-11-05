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
3. DFS from each cell, walking the trie in lockstep; visited cells tracked as a 16-bit bitmask; prune as soon as the trie has no child for the next letter.

## Scoring
3=100, 4=400, 5=800, 6=1400, 7=1800, 8=2200, then +400 per letter beyond 8.

## Tier rules
Frequency via `wordfreq.zipf_frequency(word, 'en')`.
- COMMON: zipf >= 5.0
- LESS COMMON: 3.0 <= zipf < 5.0
- zipf < 3.0: dropped (not shown, not counted)

Each tier sorts by length desc, then zipf desc, then alphabetical. Top 30 per tier shown; totals cover all words with zipf >= 3.

## Version history
- **v1**: Trie + DFS solver, dictionary + frequency for reasonable results
- **v2+v3**: Sorted word results + built trie at startup -> ~1.9s to ~13ms per board
