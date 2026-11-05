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
            c, l = wh.rank(found if found is not None else set(table), dm)
        return c, l

    def test_tiers(self):
        c, l = self._rank({"aaa": 5.0, "bbb": 4.99, "ccc": 3.0, "ddd": 2.99})
        self.assertEqual([r["word"] for r in c], ["aaa"])
        self.assertEqual([r["word"] for r in l], ["bbb", "ccc"])

    def test_sort(self):
        c, _ = self._rank({"abcd": 5.5, "zzz": 6.0, "aaa": 6.0, "bbb": 5.1, "wxyz": 5.9})
        self.assertEqual([r["word"] for r in c], ["wxyz", "abcd", "aaa", "zzz", "bbb"])

    def test_same_length_zipf_decides(self):
        c, _ = self._rank({"zzzz": 6.0, "aaaa": 5.5})
        self.assertEqual([r["word"] for r in c], ["zzzz", "aaaa"])

    def test_form_nested_points_dont_reorder(self):
        t = {"plan": 5.5, "plans": 5.0, "zzzz": 5.8}
        c, _ = self._rank(t, {"plans": "plan"})
        self.assertEqual([r["word"] for r in c], ["zzzz", "plan"])
        self.assertEqual(c[1]["forms"], ["plans"])
        self.assertIsNone(c[1]["base"])

    def test_standalone_form(self):
        c, l = self._rank({"plans": 4.0, "plan": 5.5}, {"plans": "plan"}, found={"plans"})
        self.assertEqual(l, [])
        self.assertEqual(c[0]["word"], "plans")
        self.assertEqual(c[0]["base"], "plan")

    def test_tier_follows_base(self):
        t = {"great": 4.9, "greatest": 6.0}
        c, l = self._rank(t, {"greatest": "great"})
        self.assertEqual(c, [])
        self.assertEqual(l[0]["forms"], ["greatest"])
        t = {"big": 5.5, "bigly": 3.5}
        c, l = self._rank(t, {"bigly": "big"})
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
        common, less = wh.rank(paths, dm)
        by = {r["word"]: r for r in common + less}
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
