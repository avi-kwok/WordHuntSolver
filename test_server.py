import json
import threading
import unittest
import urllib.error
import urllib.request

import server
import wordhunt as wh

BOARD = "SERSPATGLINESERS"
trie = None
dmap = None
srv = None
base = None


def setUpModule():
    global trie, dmap, srv, base
    trie, dmap = wh.startup()
    srv = server.make_server(trie, 0, dmap)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def tearDownModule():
    srv.shutdown()
    srv.server_close()


def req(path, data=None, method=None):
    r = urllib.request.Request(base + path, data=data, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check_paths(tc, board, rows):
    nb = wh.neighbors()
    flat = [x for row in rows for x in [row] + row.get("forms", [])]
    for row in flat:
        w, p = row["word"], row["path"]
        tc.assertEqual(len(p), len(w))
        tc.assertEqual(len(set(p)), len(p))
        tc.assertEqual("".join(board[i] for i in p).lower(), w)
        for x, y in zip(p, p[1:]):
            tc.assertIn(y, nb[x])


class TestBuildResult(unittest.TestCase):
    def test_result(self):
        r = server.build_result(BOARD, trie, dmap)
        self.assertGreater(r["total_words"], 347)
        for k in ("common", "less_common", "rare"):
            self.assertLessEqual(sum(1 + len(x["forms"]) for x in r[k]), 30)
            check_paths(self, BOARD, r[k])
            for row in r[k]:
                self.assertIn("base", row)
                self.assertIsInstance(row["rhyme"], str)
                self.assertGreaterEqual(row["group_size"], 1)
                self.assertIsInstance(row["forms"], list)
                self.assertIsInstance(row["group_rate"], float)
                self.assertGreater(row["group_points"], 0)
                self.assertGreater(row["group_time"], 0)
                self.assertAlmostEqual(row["group_rate"], row["group_points"] / row["group_time"], delta=1)
        self.assertTrue(any(x["forms"] for x in r["common"] + r["less_common"]))
        self.assertIn("rare", r)
        self.assertGreater(r["rare_total"], 0)
        self.assertEqual(r["total_words"], r["common_total"] + r["less_common_total"] + r["rare_total"])
        for row in r["rare"]:
            self.assertGreaterEqual(len(row["word"]), wh.RARE_MIN_LEN)
            for f in row["forms"]:
                self.assertGreaterEqual(len(f["word"]), wh.RARE_MIN_LEN)
        self.assertGreater(r["common_total"], 30)


class TestHTTP(unittest.TestCase):
    def test_valid(self):
        code, body = req("/api/solve", json.dumps({"board": "serspatglineserS"}).encode())
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["total_words"],
                         server.build_result(BOARD, trie, dmap)["total_words"])

    def test_invalid_board(self):
        self.assertEqual(req("/api/solve", b'{"board": "ABC"}')[0], 400)

    def test_bad_json(self):
        self.assertEqual(req("/api/solve", b"{nope")[0], 400)

    def test_oversized(self):
        self.assertEqual(req("/api/solve", b"x" * 2000)[0], 413)

    def test_index(self):
        code, body = req("/")
        self.assertEqual(code, 200)
        self.assertIn(b"<html", body.lower())

    def test_no_traversal(self):
        self.assertEqual(req("/../wordhunt.py")[0], 404)


if __name__ == "__main__":
    unittest.main()
