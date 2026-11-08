#!/usr/bin/env python3
"""Local web UI for the Word Hunt solver (stdlib only, 127.0.0.1 only)."""
import argparse
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import wordhunt as wh

MAX_BODY = 1024
INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")


TOP_UNITS = 60


def top_units(tiers, limit=TOP_UNITS):
    """Merge full per-tier row-dict lists into units (one contiguous
    group_key run each), sorted by points desc, rate desc, seed word;
    capped at `limit`."""
    units = []
    for rows in tiers:
        prev = None
        for r in rows:
            if prev is None or r["group_key"] != prev:
                prev = r["group_key"]
                units.append({"key": prev, "points": r["group_points"],
                              "rate": r["group_rate"], "words": 0, "rows": []})
            u = units[-1]
            u["rows"].append(r)
            u["words"] += 1 + len(r["forms"])
    units.sort(key=lambda u: (-u["points"], -u["rate"], u["rows"][0]["word"]))
    return units[:limit]


def build_result(board, trie, derived_map=None):
    t = time.perf_counter()
    paths = wh.solve_paths(board, trie)
    common, less, rare = wh.rank(paths, derived_map)

    def item(w):
        return {"word": w, "points": wh.score(w),
                "zipf": round(wh.zipf(w), 1), "path": paths[w]}

    def row(r):
        d = item(r["word"])
        d["base"] = r["base"]
        d["group_key"] = r["group_key"]
        d["shared"] = r["shared"]
        d["group_size"] = r["group_size"]
        d["best_points"] = r["best_points"]
        d["group_rate"] = round(r["group_rate"], 1)
        d["group_points"] = r["group_points"]
        d["group_time"] = round(r["group_time"], 2)
        d["forms"] = [item(f) for f in r["forms"]]
        return d

    def rows(tier):
        return [row(r) for r in wh.truncate_rows(tier)[0]]

    words = wh.tier_words(common) + wh.tier_words(less) + wh.tier_words(rare)
    result = {
        "board": board,
        "common": rows(common),
        "less_common": rows(less),
        "rare": rows(rare),
        "top": top_units([[row(r) for r in t] for t in (common, less, rare)]),
        "rare_total": len(wh.tier_words(rare)),
        "common_total": len(wh.tier_words(common)),
        "less_common_total": len(wh.tier_words(less)),
        "total_words": len(words),
        "total_points": sum(wh.score(w) for w in words),
    }
    result["solve_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return result


_STATE = None
_STATE_LOCK = threading.Lock()


def get_state():
    """Load (trie, derived_map) once per process, on first use."""
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = wh.startup()
    return _STATE


def _resp(code, body, ctype="application/json"):
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()
    return code, [("Content-Type", ctype), ("Content-Length", str(len(body))),
                  ("Cache-Control", "no-store")], body


def handle(method, path, length, read_body, state):
    """Shared routing. read_body(n) -> bytes; state() -> (trie, derived_map).
    Returns (status, headers, body bytes)."""
    if method == "GET":
        if path == "/":
            with open(INDEX, "rb") as f:
                return _resp(200, f.read(), "text/html; charset=utf-8")
        return _resp(404, {"error": "not found"})
    if method == "POST":
        if path != "/api/solve":
            return _resp(404, {"error": "not found"})
        if length is None:
            return _resp(400, {"error": "bad Content-Length"})
        if length > MAX_BODY:
            return _resp(413, {"error": "body too large"})
        try:
            data = json.loads(read_body(length) or b"")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            board = wh.parse_board(str(data.get("board", "")))
        except (ValueError, UnicodeDecodeError) as e:
            return _resp(400, {"error": str(e)})
        trie, derived_map = state()
        return _resp(200, build_result(board, trie, derived_map))
    return _resp(405, {"error": "method not allowed"})


def _length(value):
    try:
        n = int(value or 0)
    except ValueError:
        return None
    return None if n < 0 else n


def app(environ, start_response):
    """WSGI entry point (Vercel, gunicorn, wsgiref)."""
    length = _length(environ.get("CONTENT_LENGTH"))
    code, headers, body = handle(
        environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", "/"),
        length, lambda n: environ["wsgi.input"].read(n), get_state)
    start_response(f"{code} {HTTPStatus(code).phrase}", headers)
    return [body]


application = app


def make_handler(trie, derived_map=None):
    state = lambda: (trie, derived_map)

    class Handler(BaseHTTPRequestHandler):
        def _run(self, method):
            length = _length(self.headers.get("Content-Length"))
            if method == "POST" and length is not None and length > MAX_BODY:
                self.close_connection = True
            code, headers, body = handle(method, self.path, length,
                                         self.rfile.read, state)
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._run("GET")

        def do_POST(self):
            self._run("POST")

        def log_message(self, *args):
            pass

    return Handler


def make_server(trie, port=8000, derived_map=None):
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(trie, derived_map))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    trie, derived_map = get_state()
    srv = make_server(trie, args.port, derived_map)
    print(f"Serving on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
