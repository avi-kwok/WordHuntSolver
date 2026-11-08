import io
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
                self.assertNotIn("rhyme", row)
                self.assertIsInstance(row["group_key"], str)
                self.assertIn("shared", row)
                if row["group_key"] == row["word"] or row["group_size"] == 1:
                    self.assertIsNone(row["shared"])
                else:
                    self.assertGreaterEqual(len(row["shared"]), 2)
                self.assertGreaterEqual(row["group_size"], 1)
                self.assertIsInstance(row["forms"], list)
                self.assertIsInstance(row["group_rate"], float)
                self.assertGreater(row["group_points"], 0)
                self.assertGreaterEqual(row["group_points"], row["best_points"])
                # a rarer member pulled up by a Common seed can outscore the seed
                self.assertGreaterEqual(row["group_points"], row["points"])
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


def check_top(tc, board, top):
    tc.assertLessEqual(len(top), 60)
    tc.assertGreater(len(top), 0)
    pts = [u["points"] for u in top]
    tc.assertEqual(pts, sorted(pts, reverse=True))
    tc.assertEqual(len({u["key"] for u in top}), len(top))
    for u in top:
        flat = [x for r in u["rows"] for x in [r] + r["forms"]]
        tc.assertEqual(u["points"], sum(wh.score(x["word"]) for x in flat))
        tc.assertEqual(u["words"], len(flat))
        tc.assertEqual(u["key"], u["rows"][0]["group_key"])
        check_paths(tc, board, u["rows"])


class TestTop(unittest.TestCase):
    def test_top(self):
        r = server.build_result(BOARD, trie, dmap)
        self.assertIn("top", r)
        check_top(self, BOARD, r["top"])

    def test_cap(self):
        rows = [[{"group_key": str(i), "group_points": i, "group_rate": 1.0,
                  "word": str(i), "forms": []} for i in range(100)]]
        top = server.top_units(rows)
        self.assertEqual(len(top), 60)
        self.assertEqual(top[0]["points"], 99)


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


def call_app(method, path, body=b"", length=None):
    environ = {"REQUEST_METHOD": method, "PATH_INFO": path,
               "CONTENT_LENGTH": str(len(body) if length is None else length),
               "wsgi.input": io.BytesIO(body)}
    out = {}

    def sr(status, headers):
        out["status"], out["headers"] = status, dict(headers)

    data = b"".join(server.app(environ, sr))
    return int(out["status"].split()[0]), out["headers"], data


class TestWSGI(unittest.TestCase):
    def setUp(self):
        server._STATE = (trie, dmap)  # reuse loaded trie

    def test_index(self):
        code, h, body = call_app("GET", "/")
        self.assertEqual(code, 200)
        self.assertEqual(h["Content-Type"], "text/html; charset=utf-8")
        self.assertIn(b"<html", body.lower())

    def test_valid(self):
        code, _, body = call_app("POST", "/api/solve", json.dumps({"board": BOARD}).encode())
        self.assertEqual(code, 200)
        self.assertGreater(json.loads(body)["total_words"], 0)
        self.assertNotIn("bfs", json.loads(body))
        check_top(self, BOARD, json.loads(body)["top"])

    def test_invalid_board(self):
        self.assertEqual(call_app("POST", "/api/solve", b'{"board": "ABC"}')[0], 400)

    def test_bad_json(self):
        code, _, body = call_app("POST", "/api/solve", b"{nope")
        self.assertEqual(code, 400)
        self.assertIn("error", json.loads(body))

    def test_oversized(self):
        self.assertEqual(call_app("POST", "/api/solve", b"x" * 2000)[0], 413)

    def test_traversal(self):
        self.assertEqual(call_app("GET", "/../wordhunt.py")[0], 404)

    def test_unknown(self):
        self.assertEqual(call_app("GET", "/nope")[0], 404)
        self.assertEqual(call_app("POST", "/nope", b"{}")[0], 404)

    def test_other_method(self):
        self.assertIn(call_app("PUT", "/api/solve", b"{}")[0], (404, 405))


if __name__ == "__main__":
    unittest.main()
