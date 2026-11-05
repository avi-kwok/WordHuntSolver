import os
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
        self.assertEqual([r["word"] for r in c], ["wxyz", "abcd", "aaa", "zzz", "bbb"])

    def test_same_length_zipf_decides(self):
        c, _, _ = self._rank({"zzzz": 6.0, "aaaa": 5.5})
        self.assertEqual([r["word"] for r in c], ["zzzz", "aaaa"])

    def test_form_nested_points_dont_reorder(self):
        t = {"plan": 5.5, "plans": 5.0, "zzzz": 5.8}
        c, _, _ = self._rank(t, {"plans": "plan"})
        self.assertEqual([r["word"] for r in c], ["zzzz", "plan"])
        self.assertEqual(c[1]["forms"], ["plans"])
        self.assertIsNone(c[1]["base"])

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


class TestRhymeKey(unittest.TestCase):
    def test_examples(self):
        for words, key in [(["bong", "dong", "tong", "wrong"], "ong"),
                           (["part", "start", "smart", "chart", "art", "quart"], "art"),
                           (["stranger", "danger", "ranger"], "anger"),
                           (["line", "spine", "shine"], "ine"),
                           (["style"], "yle"), (["yarn"], "arn"),
                           (["rhythm"], "ythm"), (["nth"], "nth"), (["eat"], "eat")]:
            for w in words:
                self.assertEqual(wh.rhyme_key(w), key, w)

    def test_qu_and_y(self):
        self.assertEqual(wh.rhyme_key("squat"), "at")
        self.assertEqual(wh.rhyme_key("yyy"), "yy")
        self.assertEqual(wh.rhyme_key("pst"), "pst")


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

    def test_biggest_group_first(self):
        t = {"bong": 5.0, "dong": 5.0, "tong": 5.0, "party": 6.0, "zebra": 6.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["bong", "dong", "tong", "party", "zebra"])
        self.assertEqual([x["group_size"] for x in c], [3, 3, 3, 1, 1])
        self.assertEqual({x["rhyme"] for x in c[:3]}, {"ong"})

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
        t = {"bat": 5.0, "cat": 5.0, "bog": 6.0, "dog": 6.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["word"] for x in c], ["bog", "dog", "bat", "cat"])
        t = {"bat": 5.0, "cat": 5.0, "bog": 5.0, "dog": 5.0}
        c, _, _ = self._rank(t)
        self.assertEqual([x["rhyme"] for x in c], ["at", "at", "og", "og"])

    def test_nested_forms_do_not_change_group(self):
        t = {"bong": 5.0, "bongs": 5.0, "bonged": 5.0, "dong": 5.0, "tong": 5.0, "zzzz": 6.0}
        c, _, _ = self._rank(t, {"bongs": "bong", "bonged": "bong"})
        self.assertEqual([x["word"] for x in c[:3]], ["bong", "dong", "tong"])
        self.assertEqual(c[0]["forms"], ["bonged", "bongs"])
        self.assertEqual(c[0]["group_size"], 3)

    def test_standalone_form_groups_by_own_word(self):
        t = {"plans": 4.0, "plan": 5.5, "clans": 5.5}
        c, _, _ = self._rank(t, {"plans": "plan"}, found={"plans", "clans"})
        self.assertEqual({x["rhyme"] for x in c}, {"ans"})
        self.assertEqual(c[0]["group_size"], 2)

    def test_rare_truncation_cap(self):
        rows = [{"word": f"w{i}x", "base": None, "forms": [f"w{i}xs"] * 2} for i in range(20)]
        out, n = wh.truncate_rows(rows)
        self.assertEqual(n, 30)

    def test_format_output_three_columns_and_width(self):
        def mk(p):
            return [{"word": f"{p}{i}xx", "base": None, "forms": [], "rhyme": "x", "group_size": 1}
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
