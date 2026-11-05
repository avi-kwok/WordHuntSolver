#!/usr/bin/env python3
"""Local web UI for the Word Hunt solver (stdlib only, 127.0.0.1 only)."""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import wordhunt as wh

MAX_BODY = 1024
INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")


def build_result(board, trie):
    t = time.perf_counter()
    paths = wh.solve_paths(board, trie)
    common, less = wh.rank(paths)

    def rows(tier):
        return [{"word": w, "points": wh.score(w),
                 "zipf": round(wh.zipf(w), 1), "path": paths[w]}
                for w in tier[:wh.TOP_N]]

    result = {
        "board": board,
        "common": rows(common),
        "less_common": rows(less),
        "common_total": len(common),
        "less_common_total": len(less),
        "total_words": len(common) + len(less),
        "total_points": sum(wh.score(w) for w in common + less),
    }
    result["solve_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return result


def make_handler(trie):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                with open(INDEX, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/solve":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return self._send(400, {"error": "bad Content-Length"})
            if length > MAX_BODY:
                self.close_connection = True
                return self._send(413, {"error": "body too large"})
            try:
                data = json.loads(self.rfile.read(length) or b"")
                if not isinstance(data, dict):
                    raise ValueError("body must be a JSON object")
                board = wh.parse_board(str(data.get("board", "")))
            except (ValueError, UnicodeDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(200, build_result(board, trie))

        def log_message(self, *args):
            pass

    return Handler


def make_server(trie, port=8000):
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(trie))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    trie = wh.startup()
    srv = make_server(trie, args.port)
    print(f"Serving on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
