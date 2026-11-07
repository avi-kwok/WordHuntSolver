import os
import pickle
import tempfile
import unittest
from unittest import mock

import wordhunt as wh


class TestNeighbors(unittest.TestCase):
    def test_counts(self):
        n = wh.neighbors()
        self.assertEqual(len(n[0]), 3)
        self.assertEqual(len(n[1]), 5)
        self.assertEqual(len(n[5]), 8)


class TestScore(unittest.TestCase):
    def test_table(self):
        for length, pts in [(3, 100), (4, 400), (5, 800), (6, 1400),
                            (7, 1800), (8, 2200), (9, 2600), (10, 3000)]:
            self.assertEqual(wh.score("a" * length), pts)


class TestSolve(unittest.TestCase):
    def test_no_tile_reuse(self):
        # one 'a' only; "aba" needs two
        self.assertEqual(wh.solve("abcdefghijklmnop", wh.build_trie({"aba"})), set())

    def test_tile_reuse_ok_with_two(self):
        self.assertEqual(wh.solve("abacdefghijklmnop", wh.build_trie({"aba"})), {"aba"})

    def test_diagonal(self):
        # a(0) -> f(5) -> k(10) diagonal
        self.assertEqual(wh.solve("abcdefghijklmnop", wh.build_trie({"afk"})), {"afk"})

    def test_non_adjacent(self):
        # a(0) and d(3) not adjacent
        self.assertEqual(wh.solve("abcdefghijklmnop", wh.build_trie({"adg", "aei"})), {"aei"})

    def test_adjacent_found(self):
        self.assertEqual(wh.solve("abcdefghijklmnop", wh.build_trie({"abc", "aeb"})), {"abc", "aeb"})


class TestPaths(unittest.TestCase):
    def test_paths_valid(self):
        board = "SERSPATGLINESERS"
        paths = wh.solve_paths(board, wh.build_trie(
            {"pat", "tap", "line", "stare", "sat", "zzz"}))
        self.assertTrue(paths)
        nb = wh.neighbors()
        for w, p in paths.items():
            self.assertEqual(len(p), len(w))
            self.assertEqual(len(set(p)), len(p))
            self.assertEqual("".join(board[i] for i in p).lower(), w)
            for x, y in zip(p, p[1:]):
                self.assertIn(y, nb[x])


class TestFilterCommon(unittest.TestCase):
    def test_cutoff(self):
        freq = {"keep": 1e-6, "drop": 9.9e-7, "high": 1e-3}
        out = wh.filter_common({"keep", "drop", "high", "missing"}, freq)
        self.assertEqual(out, {"keep", "high"})


class TestRank(unittest.TestCase):
    def _rank(self, table, dm=None, found=None):
        with mock.patch.object(wh, "zipf", side_effect=table.__getitem__):
            c, l, r = wh.rank(found if found is not None else set(table), dm)
        return c, l, r

    def test_tiers(self):
        c, l, r = self._rank({"aaa": 5.0, "bbb": 4.99, "ccc": 3.0, "ddd": 2.99})
        self.assertEqual([r["word"] for r in c], ["aaa"])
        self.assertEqual([x["word"] for x in l], ["bbb", "ccc"])
        self.assertEqual(r, [])  # ddd: zipf 2.99 is rare but only 3 letters

    def test_sort(self):
        c, _, _ = self._rank({"abcd": 5.5, "zzz": 6.0, "aaa": 6.0, "bbb": 5.1, "wxyz": 5.9})
        self.assertEqual([r["word"] for r in c], ["abcd", "wxyz", "aaa", "bbb", "zzz"])  # equal rates: key decides, not zipf

    def test_same_length_zipf_decides(self):
        c, _, _ = self._rank({"zzzz": 6.0, "aaaa": 5.5})
        self.assertEqual([r["word"] for r in c], ["aaaa", "zzzz"])  # equal rate: key, not zipf

    def test_form_nested_points_raise_group_rate(self):
        t = {"plan": 5.5, "plans": 5.0, "zzzz": 5.8}
        c, _, _ = self._rank(t, {"plans": "plan"})
        self.assertEqual([r["word"] for r in c], ["plan", "zzzz"])  # 1200/2.75 beats 400/1.6
        self.assertEqual(c[0]["forms"], ["plans"])
        self.assertIsNone(c[0]["base"])

    def test_standalone_form(self):
        c, l, _ = self._rank({"plans": 4.0, "plan": 5.5}, {"plans": "plan"}, found={"plans"})
        self.assertEqual(l, [])
        self.assertEqual(c[0]["word"], "plans")
        self.assertEqual(c[0]["base"], "plan")

    def test_tier_follows_base(self):
        t = {"great": 4.9, "greatest": 6.0}
        c, l, _ = self._rank(t, {"greatest": "great"})
        self.assertEqual(c, [])
        self.assertEqual(l[0]["forms"], ["greatest"])
        t = {"big": 5.5, "bigly": 3.5}
        c, l, _ = self._rank(t, {"bigly": "big"})
        self.assertEqual(l, [])
        self.assertEqual(c[0]["forms"], ["bigly"])

    def test_forms_count_toward_30_lines(self):
        rows = [{"word": f"w{i}", "base": None, "forms": [f"w{i}s", f"w{i}d"]}
                for i in range(20)]
        out, n = wh.truncate_rows(rows)
        self.assertEqual(n, 30)
        self.assertEqual(sum(1 + len(r["forms"]) for r in out), 30)
        self.assertEqual(len(out), 10)
        self.assertEqual(len(wh.tier_words(rows)), 60)

    def test_group_cut_at_boundary(self):
        rows = [{"word": "a", "base": None, "forms": ["as"] * 28},
                {"word": "b", "base": None, "forms": ["bs", "bd", "bx"]},
                {"word": "c", "base": None, "forms": []}]
        out, n = wh.truncate_rows(rows)
        self.assertEqual(n, 30)
        self.assertEqual([r["word"] for r in out], ["a", "b"])
        self.assertEqual(len(out[0]["forms"]), 28)
        self.assertEqual(out[1]["forms"], [])
        # one more slot keeps the first form of b
        rows[0]["forms"] = ["as"] * 27
        out, n = wh.truncate_rows(rows)
        self.assertEqual(out[1]["forms"], ["bs"])
        self.assertEqual(rows[1]["forms"], ["bs", "bd", "bx"])


class TestMatch(unittest.TestCase):
    def test_examples(self):
        for a, b, chunk in [("brat", "brag", "bra"),
                            ("bong", "wrong", "ong"), ("strange", "strain", "stra"),
                            ("sing", "ring", "ing"), ("lama", "ama", "ama"),
                            ("mist", "ist", "ist")]:
            self.assertEqual(wh.shared_chunk(a, b), chunk, (a, b))
            self.assertEqual(wh.shared_chunk(b, a), chunk, (b, a))
            self.assertEqual(wh.match_len(a, b), len(chunk), (a, b))

    def test_non_match(self):
        # 2-letter runs no longer match (MIN_SHARED = 3)
        self.assertEqual(wh.MIN_SHARED, 3)
        self.assertEqual(wh.shared_chunk("brag", "trap"), "")
        self.assertEqual(wh.match_len("brag", "trap"), 0)
        self.assertEqual(wh.match_len("cats", "bits"), 0)
        # cat/dog: no letter agrees at any index, from either end
        self.assertEqual(wh.shared_chunk("cat", "dog"), "")
        self.assertEqual(wh.match_len("cat", "dog"), 0)
        # single shared letter is not a chunk
        self.assertEqual(wh.match_len("cab", "dog"), 0)
        self.assertEqual(wh.match_len("bad", "bot"), 0)

    def test_tie_prefers_start(self):
        self.assertEqual(wh.shared_chunk("abcxdef", "abcqdef"), "abc")


class TestRareAndGrouping(unittest.TestCase):
    def _rank(self, table, dm=None, found=None):
        with mock.patch.object(wh, "zipf", side_effect=table.__getitem__):
            return wh.rank(found if found is not None else set(table), dm)

    def test_rare_tier_bounds(self):
        c, l, r = self._rank({"abcd": 2.7, "efg": 2.7, "hijk": 2.49, "lmno": 3.0})
        self.assertEqual([x["word"] for x in r], ["abcd"])
        self.assertEqual([x["word"] for x in l], ["lmno"])
        self.assertEqual(c, [])

    def test_rare_forms_min_len(self):
        t = {"abcd": 2.7, "abcds": 2.0, "abcded": 2.0}
        _, _, r = self._rank(t, {"abcds": "abcd", "abcded": "abcd"})
        self.assertEqual(r[0]["forms"], ["abcded", "abcds"])

    def test_rare_three_letter_base_forms_standalone(self):
        t = {"abc": 2.7, "abcs": 2.0, "abced": 2.0}
        _, _, r = self._rank(t, {"abcs": "abc", "abced": "abc"})
        self.assertEqual([x["word"] for x in r], ["abced", "abcs"])
        for x in r:
            self.assertEqual(x["base"], "abc")
            self.assertEqual(x["forms"], [])
        # base 3 letters at rare tier is not shown even when found
        _, _, r = self._rank(t, {"abcs": "abc"}, found={"abc", "abcs"})
        self.assertEqual([x["word"] for x in r], ["abcs"])

    def test_rare_form_of_low_base_dropped(self):
        _, _, r = self._rank({"abcd": 2.4, "abcds": 3.5}, {"abcds": "abcd"})
        self.assertEqual(r, [])

    def test_group_before_single_and_seed(self):
        t = {"bong": 5.0, "dong": 5.0, "tong": 5.0, "party": 6.0, "zebra": 6.0}
        c, _, _ = self._rank(t)
        # groups of 2+ come before all singles, even 5-letter ones
        self.assertEqual([x["word"] for x in c], ["bong", "dong", "tong", "party", "zebra"])
        self.assertEqual([x["group_size"] for x in c], [3, 3, 3, 1, 1])
        self.assertEqual({x["group_key"] for x in c[:3]}, {"bong"})
        self.assertEqual([x["shared"] for x in c], [None, "ong", "ong", None, None])
        self.assertNotIn("rhyme", c[0])

    def test_seed_is_longest_then_common(self):
        t = {"cats": 6.0, "bats": 5.2, "hats": 5.5, "chats": 5.0}
        c, _, _ = self._rank(t)
        # hats is contained in chats (suffix): not in its group, trails as a single
        self.assertEqual([x["word"] for x in c], ["chats", "cats", "bats", "hats"])
        self.assertEqual([x["group_key"] for x in c], ["chats", "chats", "chats", "hats"])
        self.assertEqual([x["shared"] for x in c], [None, "ats", "ats", None])
        c, _, _ = self._rank({"stabs": 5.0, "slabs": 5.5})
        self.assertEqual(c[0]["word"], "slabs")
        self.assertEqual(c[1]["group_key"], "slabs")

    def test_cap_gives_second_group(self):
        t = {w: 5.0 for w in ["bats", "cats", "fats", "hats", "mats", "pats", "rats", "sats"]}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c],
                         ["bats", "cats", "fats", "hats", "mats", "pats", "rats", "sats"])
        self.assertEqual([x["group_size"] for x in c], [6] * 6 + [2] * 2)
        self.assertEqual([x["group_key"] for x in c], ["bats"] * 6 + ["rats"] * 2)
        self.assertEqual(wh.GROUP_MAX, 6)
        self.assertEqual(len({x["word"] for x in c}), 8)  # each word in one group

    def test_longer_match_beats_points(self):
        t = {"brand": 5.0, "stand": 5.0, "bran": 5.0}
        with mock.patch.object(wh, "GROUP_MAX", 2):
            c, _, _ = self._rank(t)
        # bran is contained in brand (prefix) so it cannot join; stand ("and") does
        self.assertEqual([x["word"] for x in c], ["brand", "stand", "bran"])
        self.assertEqual([x["group_size"] for x in c], [2, 2, 1])

    def test_nested_forms_do_not_count_toward_cap(self):
        base = ["bats", "cats", "fats", "hats", "mats", "pats"]
        t = {w: 5.0 for w in base}
        t.update({w + "x": 5.0 for w in base})
        c, _, _ = self._rank(t, {w + "x": w for w in base})
        self.assertEqual([x["group_size"] for x in c], [6] * 6)
        self.assertEqual(c[0]["group_points"], 6 * 400 + 6 * 800)
        self.assertEqual(sum(len(x["forms"]) for x in c), 6)
    def test_size_tie_longest_member(self):
        t = {"bat": 5.0, "cat": 5.0, "cuts": 5.0, "puts": 5.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["cuts", "puts", "bat", "cat"])

    def test_within_group_order_and_continuity(self):
        t = {"tong": 5.0, "bong": 5.5, "wrong": 5.0, "dong": 5.5}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["wrong", "bong", "dong", "tong"])
        self.assertEqual([x["group_size"] for x in c], [4] * 4)

    def test_group_ties_zipf_then_key(self):
        # equal rate, points, size: zipf no longer matters, seed word decides
        t = {"bats": 5.0, "cats": 5.0, "bogs": 6.0, "dogs": 6.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["bats", "cats", "bogs", "dogs"])
        t = {"bats": 5.0, "cats": 5.0, "bogs": 5.0, "dogs": 5.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["group_key"] for x in c], ["bats", "bats", "bogs", "bogs"])
    def test_nested_forms_do_not_change_group(self):
        t = {"bong": 5.0, "bongs": 5.0, "bonged": 5.0, "dong": 5.0, "tong": 5.0, "zzzz": 6.0}
        c, _, _ = self._rank(t, {"bongs": "bong", "bonged": "bong"})
        self.assertEqual([x["word"] for x in c[:3]], ["bong", "dong", "tong"])
        self.assertEqual(c[0]["forms"], ["bonged", "bongs"])
        self.assertEqual(c[0]["group_size"], 3)

    def test_standalone_form_groups_by_own_word(self):
        t = {"plans": 4.0, "plan": 5.5, "clans": 5.5}
        c, _, _ = self._rank(t, {"plans": "plan"}, found={"plans", "clans"})
        self.assertEqual([x["group_key"] for x in c], ["clans", "clans"])
        self.assertEqual(c[1]["shared"], "lans")
        self.assertEqual(c[0]["group_size"], 2)
    def test_group_rate_exact(self):
        t = {"ring": 5.0, "sing": 5.0, "rings": 5.0}
        c, _, _ = self._rank(t, {"rings": "ring"})
        self.assertEqual([x["word"] for x in c], ["ring", "sing"])
        self.assertEqual(c[0]["forms"], ["rings"])
        for x in c:
            self.assertEqual(x["group_points"], 1600)
            self.assertAlmostEqual(x["group_time"], 3.75)
            self.assertAlmostEqual(x["group_rate"], 1600 / 3.75)
            self.assertEqual(round(x["group_rate"], 2), 426.67)

    def test_higher_single_beats_pair(self):
        t = {"abcdefgh": 5.0, "bats": 5.0, "cats": 5.0}
        c, _, _ = self._rank(t)
        # single worth 2200 outranks the pair (800) on total
        self.assertEqual([x["word"] for x in c], ["abcdefgh", "bats", "cats"])
        self.assertEqual([x["best_points"] for x in c], [2200, 400, 400])
    def test_equal_rate_groups_then_single(self):
        # FIRST=1, no other cost: rate = group points. Single "zork" (1600)
        # ties the 4x400 "ats" group (1600) on total and rate; size decides.
        t = {"zork": 5.0, "bats": 5.0, "cats": 5.0, "fats": 5.0, "hats": 5.0}
        with mock.patch.object(wh, "score", side_effect=lambda w: 1600 if w == "zork" else 400), \
             mock.patch.object(wh, "FIND_FIRST_S", 1.0), \
             mock.patch.object(wh, "FIND_NEXT_S", 0.0), \
             mock.patch.object(wh, "SWIPE_PER_LETTER_S", 0.0):
            c, _, _ = self._rank(t)
        self.assertEqual([x["group_rate"] for x in c], [1600.0] * 5)
        self.assertEqual([x["word"] for x in c], ["bats", "cats", "fats", "hats", "zork"])
    def test_end_alignment_contained_word_does_not_group(self):
        # ama is a suffix of lama, ist of mist: contained words stay out
        t = {"lama": 5.0, "ama": 5.0, "mist": 5.0, "ist": 5.0}
        c, _, _ = self._rank(t)
        self.assertEqual(len(c), 4)
        self.assertEqual([x["group_size"] for x in c], [1] * 4)
        self.assertEqual(sorted(x["group_key"] for x in c), sorted(x["word"] for x in c))
        self.assertTrue(all(x["shared"] is None for x in c))

    def test_contained_word_not_in_seed_group(self):
        rows = [{"word": w} for w in ("basis", "oasis", "sis", "bas")]
        out = wh.group_rows(rows)
        by = {x["word"]: x for x in out}
        self.assertEqual(by["basis"]["group_key"], "basis")
        self.assertEqual(by["oasis"]["group_key"], "basis")
        self.assertEqual(by["basis"]["group_size"], 2)
        self.assertNotEqual(by["sis"]["group_key"], "basis")
        self.assertNotEqual(by["bas"]["group_key"], "basis")
        self.assertEqual(wh.match_len("sis", "bas"), 0)
        self.assertEqual(by["sis"]["group_size"], 1)
        self.assertEqual(by["bas"]["group_size"], 1)

    def test_lama_ama_do_not_group(self):
        out = wh.group_rows([{"word": "lama"}, {"word": "ama"}])
        self.assertEqual([x["group_size"] for x in out], [1, 1])

    def test_bigger_total_beats_higher_best(self):
        t = {"plant": 5.0, "slant": 5.0, "cogs": 5.0, "dogs": 5.0, "fogs": 5.0,
             "hogs": 5.0, "logs": 5.0, "abcde": 5.0}
        c, _, _ = self._rank(t)
        # group of 5 (2000 points) outranks the best-800 pair (1600)
        self.assertEqual([x["word"] for x in c],
                         ["cogs", "dogs", "fogs", "hogs", "logs", "plant", "slant", "abcde"])
        self.assertEqual(c[5]["best_points"], 800)
        self.assertGreater(c[0]["group_points"], c[5]["group_points"])

    def test_equal_best_bigger_group_points_first(self):
        t = {"bats": 5.0, "cats": 5.0, "dogs": 5.0, "fogs": 5.0, "logs": 5.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["dogs", "fogs", "logs", "bats", "cats"])
    def test_save_family_beats_smaller_group(self):
        # save + saved + saver = 400 + 800 + 800 = 2000 as a single row
        pts = {"plan": 1000, "clan": 800}  # 2-word group totalling 1800
        t = {"save": 5.0, "saved": 5.0, "saver": 5.0, "plan": 5.0, "clan": 5.0}
        with mock.patch.object(wh, "score", side_effect=lambda w: pts.get(w) or wh._SCORES[len(w)]):
            c, _, _ = self._rank(t, {"saved": "save", "saver": "save"})
        self.assertEqual([x["word"] for x in c], ["save", "clan", "plan"])
        self.assertEqual([x["group_points"] for x in c], [2000, 1800, 1800])
        self.assertEqual(c[0]["group_size"], 1)
        self.assertEqual(c[0]["best_points"], 400)  # still exposed

    def test_big_group_beats_single(self):
        pts = {"bats": 1000, "cats": 1000, "fats": 1000, "hats": 1000, "mats": 1000, "pats": 1100,
               "zork": 2000}  # six-word group = 6100
        t = {w: 5.0 for w in pts}
        with mock.patch.object(wh, "score", side_effect=pts.__getitem__):
            c, _, _ = self._rank(t)
        self.assertEqual([x["group_size"] for x in c], [6] * 6 + [1])
        self.assertEqual(c[0]["group_points"], 6100)
        self.assertEqual(c[-1]["word"], "zork")
        self.assertEqual(c[-1]["group_points"], 2000)

    def test_tie_on_total_broken_by_rate(self):
        # single 800 (5-letter) vs 2 x 400 group: equal total, single is faster
        t = {"abcde": 5.0, "bats": 5.0, "cats": 5.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["group_points"] for x in c], [800, 800, 800])
        self.assertEqual([x["word"] for x in c], ["abcde", "bats", "cats"])
        self.assertGreater(c[0]["group_rate"], c[1]["group_rate"])

    def test_rare_truncation_cap(self):
        rows = [{"word": f"w{i}x", "base": None, "forms": [f"w{i}xs"] * 2} for i in range(20)]
        out, n = wh.truncate_rows(rows)
        self.assertEqual(n, 30)

    def test_format_output_three_columns_and_width(self):
        def mk(p):
            return [{"word": f"{p}{i}xx", "base": None, "forms": [], "group_key": "x", "group_size": 1}
                    for i in range(40)]
        with mock.patch.object(wh, "zipf", return_value=3.0):
            out = wh.format_output("A" * 16, mk("a"), mk("b"), mk("c"), 1.0)
        self.assertIn("RARE (zipf 2.5-3, 4+)", out)
        self.assertLessEqual(max(map(len, out.splitlines())), 100)
        for name in ("Common", "Less common", "Rare"):
            self.assertIn(f"{name}: shown 30 of 40", out)
        self.assertIn("Total: 120 words", out)


class TestInflections(unittest.TestCase):
    def test_candidates(self):
        self.assertTrue({"plans", "planned", "planning"} <= wh.inflections("plan"))
        self.assertTrue({"rates", "rated", "rating", "rater", "raters"} <= wh.inflections("rate"))
        self.assertIn("cities", wh.inflections("city"))
        self.assertIn("boxes", wh.inflections("box"))
        self.assertTrue({"stopped", "stopping"} <= wh.inflections("stop"))
        self.assertTrue({"greater", "greatest", "greatly"} <= wh.inflections("great"))
        self.assertIn("seeing", wh.inflections("see"))
        self.assertIn("lying", wh.inflections("lie"))
        self.assertIn("basically", wh.inflections("basic"))
        self.assertIn("simply", wh.inflections("simple"))

    def test_generate_filter(self):
        t = 10 ** (wh.DERIVED_MIN_ZIPF - 9)
        freq = {"plans": 1e-3, "planned": 1e-4, "eated": 0, "eat": 1e-3, "eats": t,
                "eaten": t * 0.99, "eating": t * 0.99}
        d = wh.generate_derived({"plan", "eat"}, freq)
        self.assertEqual(d, {"plans": "plan", "planned": "plan", "eats": "eat"})

    def test_threshold_exact(self):
        t = 10 ** (wh.DERIVED_MIN_ZIPF - 9)
        d = wh.generate_derived({"abc"}, {"abcs": t, "abced": t * 0.99})
        self.assertEqual(d, {"abcs": "abc"})

    def test_highest_base_wins(self):
        # "boxer" from box+er and "boxe"+"r"; pick the more frequent base
        d = wh.generate_derived({"box", "boxe"}, {"box": 1e-3, "boxe": 1e-6, "boxer": 1e-5})
        self.assertEqual(d["boxer"], "box")

    def test_cap_only_excluded(self):
        t = 10 ** (wh.DERIVED_MIN_ZIPF - 9)
        freq = {"lar": 1e-6, "lars": 1e-4, "lared": t}
        d = wh.generate_derived({"lar"}, freq, exclude={"lars"})
        self.assertEqual(d, {"lared": "lar"})

    def test_base_not_captured_by_rarer_root(self):
        freq = {"splint": 10 ** (2.7 - 9), "splinter": 10 ** (3.3 - 9),
                "splints": 10 ** (2.6 - 9), "great": 10 ** (5.9 - 9),
                "greater": 10 ** (4.9 - 9)}
        d = wh.generate_derived({"splint", "splinter", "great", "greater"}, freq)
        self.assertNotIn("splinter", d)
        self.assertEqual(d["splints"], "splint")  # non-base form follows root
        self.assertEqual(d["greater"], "great")   # more common root still nests
        self.assertNotIn("great", d)

    def test_chain_with_tier_rule(self):
        freq = {"great": 1e-3, "greater": 1e-4, "greaterx": 1e-5, "greatest": 1e-5}
        d = wh.generate_derived({"great", "greater"}, freq)
        self.assertEqual(d["greatest"], "great")

    def test_chain_and_self(self):
        self.assertEqual(wh.resolve_chains({"a": "b", "b": "c"}), {"a": "c", "b": "c"})
        self.assertNotIn("x", wh.resolve_chains({"x": "x"}))
        self.assertEqual(wh.resolve_chains({"a": "b", "b": "a"}), {})

    def test_base_that_is_also_derived_maps_to_root(self):
        freq = {"great": 1e-4, "greater": 1e-5, "greatest": 1e-5, "greaterx": 1e-5}
        d = wh.generate_derived({"great", "greater"}, freq)
        self.assertEqual(d["greater"], "great")
        self.assertEqual(d["greatest"], "great")


class TestParse(unittest.TestCase):
    def test_reject(self):
        for bad in ["A" * 15, "A" * 17, "ABCDEFGHIJKLMN12"]:
            with self.assertRaises(ValueError):
                wh.parse_board(bad)

    def test_accept(self):
        self.assertEqual(wh.parse_board("  abcd efgh ijkl mnop\n"), "ABCDEFGHIJKLMNOP")


class TestProperNouns(unittest.TestCase):
    def test_dictionary(self):
        d = wh.load_dictionary()
        for w in ["yemen", "eden", "paris", "santa"]:
            self.assertNotIn(w, d, w)
        for w in ["turkey", "march", "china", "retina"]:
            self.assertIn(w, d, w)


class TestBlocklist(unittest.TestCase):
    def _tmp(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_parse(self):
        p = self._tmp("# header\n\n  Foo  \nBAR\n   # indented comment\n\t\nbaz\n")
        self.assertEqual(wh.load_blocklist(p), {"foo", "bar", "baz"})

    def test_missing(self):
        self.assertEqual(wh.load_blocklist("/nonexistent/blocklist.txt"), set())

    def test_startup_applies(self):
        board = "SERSPATGLINESERS"
        base_trie, base_dm = wh.startup(self._tmp(""))
        self.assertIn("plans", base_dm)
        found0 = wh.solve(board, base_trie)
        self.assertIn("plan", found0)
        self.assertIn("plans", found0)
        trie, dm = wh.startup(self._tmp("PLAN\nplans\n"))
        found = wh.solve(board, trie)
        for w in ["plan", "plans"]:
            self.assertNotIn(w, found)
            self.assertNotIn(w, dm)
        self.assertIn("line", found)

    def test_block_base_keeps_forms(self):
        board = "SERSPATGLINESERS"
        trie, dm = wh.startup(self._tmp("plan\n"))
        found = wh.solve(board, trie)
        self.assertNotIn("plan", found)
        self.assertIn("plans", found)


class TestAllowlist(unittest.TestCase):
    def _tmp(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_parse(self):
        p = self._tmp("# header\n\n  Lande \nFOO\n   # c\n")
        self.assertEqual(wh.load_allowlist(p), {"lande", "foo"})

    def test_missing(self):
        self.assertEqual(wh.load_allowlist("/nonexistent/allowlist.txt"), set())

    def test_shipped_file_has_lande(self):
        self.assertIn("lande", wh.load_allowlist())

    def test_non_web2_word_in_trie_and_found(self):
        board = "XYZZAAAYAAAAAAAA"
        empty = self._tmp("")
        trie, dm = wh.startup(empty, empty)
        self.assertNotIn("xyzzy", wh.solve(board, trie))
        trie, dm = wh.startup(empty, self._tmp("XYZZY\n"))
        self.assertIn("xyzzy", wh.solve(board, trie))

    def test_blocklist_beats_allowlist(self):
        board = "XYZZAAAYAAAAAAAA"
        p = self._tmp("xyzzy\n")
        trie, dm = wh.startup(p, p)
        self.assertNotIn("xyzzy", wh.solve(board, trie))

    def test_lande_groups_with_land(self):
        empty = self._tmp("")
        trie, dm = wh.startup(empty, self._tmp("lande\n"))
        self.assertNotIn("lande", dm)
        self.assertEqual(dm["lander"], "land")
        paths = wh.solve_paths("LAND" "AARE" "AAAA" "AAAA", trie)
        for w in ("land", "lande", "lander"):
            self.assertIn(w, paths)
        rows = {r["word"]: r for r in sum(wh.rank(paths, dm), [])}
        self.assertEqual(rows["land"]["forms"], ["lander"])
        self.assertEqual(rows["lande"]["group_key"], rows["land"]["group_key"])

    def test_hote_not_form_of_hot(self):
        empty = self._tmp("")
        trie, dm = wh.startup(empty, self._tmp("hote\n"))
        self.assertNotEqual(dm.get("hote"), "hot")

    def test_allowlisted_no_base_is_top_level_row(self):
        empty = self._tmp("")
        trie, dm = wh.startup(empty, self._tmp("lande\n"))
        rows = sum(wh.rank({"lande"}, dm), [])
        self.assertEqual([(r["word"], r["base"]) for r in rows], [("lande", None)])

    def test_allowlisted_never_group_only(self):
        empty = self._tmp("")
        trie, dm = wh.startup(empty, self._tmp("lande\nqzxv\n"))
        self.assertFalse({"lande", "qzxv"} & dm.group_only)

    def test_low_zipf_no_base_goes_rare(self):
        empty = self._tmp("")
        trie, dm = wh.startup(empty, self._tmp("qzxv\nzxq\n"))
        self.assertNotIn("qzxv", dm)
        c, l, r = wh.rank({"qzxv", "zxq"}, dm)
        self.assertEqual((c, l), ([], []))
        self.assertEqual(sorted(x["word"] for x in r), ["qzxv", "zxq"])  # even 3 letters
        # without the allowlist the same words are dropped
        c, l, r = wh.rank({"qzxv", "zxq"}, {})
        self.assertEqual((c, l, r), ([], [], []))


class TestOneLetterAndCrossTier(unittest.TestCase):
    def _rank(self, table, dm=None):
        with mock.patch.object(wh, "zipf", side_effect=table.__getitem__):
            return wh.rank(set(table), dm)

    def test_one_letter_matches(self):
        for a, b in [("vote", "note"), ("vote", "dote"), ("vote", "hote"),
                     ("note", "hote"), ("cat", "cot"), ("bat", "cat")]:
            self.assertGreaterEqual(wh.match_len(a, b), wh.MIN_SHARED if len(a) > 3 else 2, (a, b))
        self.assertEqual(wh.match_len("cat", "cot"), 2)
        self.assertEqual(wh.match_len("vote", "note"), 3)

    def test_one_letter_non_matches(self):
        self.assertFalse(wh._one_off("vote", "votes"))  # different length: rule does not apply
        self.assertEqual(wh.match_len("cat", "cut"), 2)
        self.assertEqual(wh.match_len("cats", "cuts"), 3)  # hamming 1 -> len-1
        self.assertEqual(wh.match_len("cats", "cuds"), 0)  # hamming 2, no run
        self.assertEqual(wh.match_len("vote", "vita"), 0)

    def test_one_letter_group_below_min_shared_len3(self):
        # 3-letter words: len-1 = 2 < MIN_SHARED, so they do not group
        c, _, _ = self._rank({"cat": 6.0, "cot": 6.0})
        self.assertEqual([x["group_size"] for x in c], [1, 1])

    def test_common_seed_pulls_rarer_variants_up(self):
        t = {"vote": 5.5, "note": 5.2, "rote": 3.5, "dote": 2.8, "hote": 2.7}
        c, l, r = self._rank(t)
        self.assertEqual(l, [])
        self.assertEqual(r, [])
        self.assertEqual({x["word"] for x in c}, set(t))
        self.assertEqual({x["group_key"] for x in c}, {"vote"})
        self.assertEqual(c[0]["word"], "vote")

    def test_rare_seed_never_pulls_common_down(self):
        t = {"tether": 2.8, "other": 5.5}
        c, l, r = self._rank(t)
        # other (common) stays in Common as the seed; the rarer tether is
        # pulled UP into its group. Nothing common moves to Rare.
        self.assertEqual([x["word"] for x in c], ["other", "tether"])
        self.assertEqual(c[1]["group_key"], "other")
        self.assertEqual(r, [])

    def test_pulled_up_word_in_one_group_and_sums(self):
        t = {"vote": 5.5, "note": 3.5, "tote": 2.8, "bats": 5.1}
        c, l, r = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["vote", "note", "tote", "bats"])
        self.assertEqual((l, r), ([], []))
        words = [x["word"] for x in c]
        self.assertEqual(len(words), len(set(words)))
        g = [x for x in c if x["group_key"] == "vote"]
        self.assertEqual(g[0]["group_points"], 3 * wh.score("vote"))
        self.assertTrue(all(x["group_points"] == g[0]["group_points"] for x in g))
        self.assertEqual(c[-1]["group_points"], wh.score("bats"))


class TestGroupOnly(unittest.TestCase):
    def _dm(self, *words):
        dm = wh.DerivedMap()
        dm.group_only = frozenset(words)
        return dm

    def _rank(self, table, dm):
        with mock.patch.object(wh, "zipf", side_effect=table.__getitem__):
            return wh.rank(set(table), dm)

    def test_joins_matching_seed(self):
        c, l, r = self._rank({"vote": 5.5, "dote": 2.23}, self._dm("dote"))
        self.assertEqual([x["word"] for x in c], ["vote", "dote"])
        self.assertEqual(c[1]["group_key"], "vote")
        self.assertEqual((l, r), ([], []))

    def test_unmatched_dropped_from_output_and_totals(self):
        c, l, r = self._rank({"vote": 5.5, "zzzz": 2.2}, self._dm("zzzz"))
        self.assertEqual([x["word"] for x in c], ["vote"])
        self.assertEqual((l, r), ([], []))
        self.assertEqual(len(wh.tier_words(c) + wh.tier_words(l) + wh.tier_words(r)), 1)

    def test_never_seeds(self):
        c, l, r = self._rank({"dote": 2.2, "hote": 2.1}, self._dm("dote", "hote"))
        self.assertEqual((c, l, r), ([], [], []))

    def test_joins_rare_seed_but_not_seeds_common_group_down(self):
        c, l, r = self._rank({"tether": 2.8, "other": 2.2}, self._dm("other"))
        self.assertEqual([x["word"] for x in r], ["tether", "other"])
        self.assertEqual(r[1]["group_key"], "tether")

    def test_cap_applies(self):
        t = {"vote": 5.5}
        t.update({w: 2.2 for w in ("note", "dote", "hote", "rote", "cote", "tote", "mote")})
        c, _, _ = self._rank(t, self._dm(*[w for w in t if w != "vote"]))
        self.assertEqual(len(c), wh.GROUP_MAX)


class TestBlockAllowTrie(unittest.TestCase):
    def test_block_and_allow(self):
        with tempfile.TemporaryDirectory() as d:
            bl, al = os.path.join(d, "b.txt"), os.path.join(d, "a.txt")
            open(bl, "w").write("sat\n")
            open(al, "w").write("zat\n")
            trie, dm = wh.startup(bl, al)
        self.assertNotIn("sat", wh.solve_paths("SATZXXXXXXXXXXXX", trie))
        self.assertIn("zat", wh.solve_paths("ZATSXXXXXXXXXXXX", trie))


class TestCache(unittest.TestCase):
    BOARD = "SERSPATGLINESERS"

    @staticmethod
    def _result(trie, dm):
        c, l, r = wh.rank(wh.solve_paths(TestCache.BOARD, trie), dm)
        return c, l, r

    def test_cached_equals_built(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.pkl")
            wh.build_cache(path)
            with mock.patch("builtins.print") as pr:
                t1, d1 = wh.startup(cache_path=path)
            self.assertIn("(cache)", pr.call_args_list[0][0][0])
            payload = wh._build(wh.BLOCKLIST_PATH, wh.ALLOWLIST_PATH)
            t2, d2 = payload[0], payload[1]
        self.assertEqual(t1, t2)
        self.assertEqual(dict(d1), dict(d2))
        self.assertEqual(d1.allowed, d2.allowed)
        self.assertEqual(d1.group_only, d2.group_only)
        self.assertEqual(self._result(t1, d1), self._result(t2, d2))

    def test_fingerprint_mismatch_rebuilds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.pkl")
            with open(path, "wb") as f:
                pickle.dump("wrong-fingerprint", f, protocol=5)
                pickle.dump("garbage-payload", f, protocol=5)
            with mock.patch("builtins.print") as pr:
                trie, dm = wh.startup(cache_path=path)
            self.assertIn("(built)", pr.call_args_list[0][0][0])
            self.assertIn("sat", wh.solve_paths("SATXXXXXXXXXXXXX", trie))
            with open(path, "rb") as f:  # refreshed with the right fingerprint
                self.assertEqual(pickle.load(f), wh._fingerprint())

    def test_unreadable_cache_rebuilds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.pkl")
            open(path, "wb").write(b"not a pickle")
            self.assertIsNone(wh._read_cache(path, "x"))
            self.assertIsNone(wh._read_cache(os.path.join(d, "missing.pkl"), "x"))

    def test_unwritable_cache_ignored(self):
        trie, dm = wh.startup(cache_path="/nonexistent-dir/c.pkl")
        self.assertTrue(trie)


class TestIntegration(unittest.TestCase):
    def test_real_dictionary(self):
        board = "SERSPATGLINESERS"
        trie, dm = wh.startup()
        paths = wh.solve_paths(board, trie)
        for w in ["pat", "tap", "sap", "line", "sat"]:
            self.assertIn(w, paths)
        common, less, rare = wh.rank(paths, dm)
        by = {r["word"]: r for r in common + less + rare}
        for base, form in [("plan", "plans"), ("part", "parts"), ("line", "lines"),
                           ("seat", "seats"), ("site", "sites")]:
            self.assertIn(form, by[base]["forms"], form)
        nb = wh.neighbors()
        for w, p in paths.items():
            self.assertEqual("".join(board[i] for i in p).lower(), w)
            self.assertEqual(len(set(p)), len(p))
            for x, y in zip(p, p[1:]):
                self.assertIn(y, nb[x])


if __name__ == "__main__":
    unittest.main()
