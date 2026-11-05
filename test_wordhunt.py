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
        self.assertEqual(wh.solve("abcdefghijklmnop", {"aba"}), set())

    def test_tile_reuse_ok_with_two(self):
        self.assertEqual(wh.solve("abacdefghijklmnop", {"aba"}), {"aba"})

    def test_diagonal(self):
        # a(0) -> f(5) -> k(10) diagonal
        self.assertEqual(wh.solve("abcdefghijklmnop", {"afk"}), {"afk"})

    def test_non_adjacent(self):
        # a(0) and d(3) not adjacent
        self.assertEqual(wh.solve("abcdefghijklmnop", {"adg", "aei"}), {"aei"})

    def test_adjacent_found(self):
        self.assertEqual(wh.solve("abcdefghijklmnop", {"abc", "aeb"}), {"abc", "aeb"})


class TestFilter(unittest.TestCase):
    def test_filter(self):
        out = wh.filter_words(["abc", "zzz", "aab", "ab", "cab"], "abcdefghijklmnop")
        self.assertEqual(sorted(out), ["abc", "cab"])


class TestRank(unittest.TestCase):
    def _rank(self, table):
        with mock.patch.object(wh, "zipf", side_effect=table.__getitem__):
            return wh.rank(set(table))

    def test_tiers(self):
        c, l = self._rank({"aaa": 5.0, "bbb": 4.99, "ccc": 3.0, "ddd": 2.99})
        self.assertEqual(c, ["aaa"])
        self.assertEqual(l, ["bbb", "ccc"])

    def test_sort(self):
        c, _ = self._rank({"abcd": 5.5, "zzz": 6.0, "aaa": 6.0, "bbb": 5.1, "wxyz": 5.9})
        self.assertEqual(c, ["wxyz", "abcd", "aaa", "zzz", "bbb"])


class TestParse(unittest.TestCase):
    def test_reject(self):
        for bad in ["A" * 15, "A" * 17, "ABCDEFGHIJKLMN12"]:
            with self.assertRaises(ValueError):
                wh.parse_board(bad)

    def test_accept(self):
        self.assertEqual(wh.parse_board("  abcd efgh ijkl mnop\n"), "ABCDEFGHIJKLMNOP")


class TestIntegration(unittest.TestCase):
    def test_real_dictionary(self):
        found = wh.solve("SERSPATGLINESERS", wh.load_dictionary())
        for w in ["pat", "tap", "sap", "line", "sat"]:
            self.assertIn(w, found)


if __name__ == "__main__":
    unittest.main()
