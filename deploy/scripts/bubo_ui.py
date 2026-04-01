#!/usr/bin/env python3
"""
deploy/scripts/bubo_ui.py — Bubo Adam & Eve
============================================
The Bubo Interface — local web UI with proxy.

Starts a local web server on port 3000 that:
  1. Serves bubo_ui.html (the interface)
  2. Proxies /adam/* → Adam's gateway
  3. Proxies /eve/*  → Eve's gateway

This bypasses CORS restrictions in the browser.

Usage:
  python3 deploy/scripts/bubo_ui.py

Then open: http://localhost:3000

Requires:
  BUBO_ADAM_IP — Adam's IP:port (e.g. 54.81.36.146:8443)
  BUBO_EVE_IP  — Eve's IP:port  (e.g. 34.202.59.78:8443)

Kenneth & Shannon Renshaw — Neurobotics — March 2026
"""

import os, sys, json, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ADAM_BASE = os.environ.get("BUBO_ADAM_IP", "").strip()
EVE_BASE  = os.environ.get("BUBO_EVE_IP",  "").strip()
PORT      = int(os.environ.get("BUBO_UI_PORT", "3000"))
UI_FILE   = Path(__file__).parent / "bubo_ui.html"

def proxy_request(target_base, path, method, body):
    proto = "http"
    url = f"{proto}://{target_base}{path}"
    try:
        req = urllib.request.Request(url,
            data=body if method == "POST" else None,
            headers={"Content-Type":"application/json"},
            method=method)
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 503, json.dumps({"error": str(e)}).encode()

class BuboHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default logging

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/status":
            self._serve_status()
        else:
            self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def _serve_html(self):
        if not UI_FILE.exists():
            self.send_error(404, f"bubo_ui.html not found at {UI_FILE}")
            return
        html = UI_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors()
        self.end_headers()
        self.wfile.write(html)

    def _serve_status(self):
        status = {"adam": ADAM_BASE, "eve": EVE_BASE, "port": PORT}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def _proxy(self, method):
        path = self.path
        target = None
        api_path = None

        if path.startswith("/adam/"):
            target, api_path = ADAM_BASE, path[5:]
        elif path.startswith("/eve/"):
            target, api_path = EVE_BASE, path[4:]
        else:
            self.send_error(404)
            return

        if not target:
            self.send_error(503, "Instance not configured")
            return

        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

        status, data = proxy_request(target, api_path, method, body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

def main():
    if not ADAM_BASE:
        print("WARNING: BUBO_ADAM_IP not set — Adam proxy disabled")
    if not EVE_BASE:
        print("WARNING: BUBO_EVE_IP not set — Eve proxy disabled")
    if not UI_FILE.exists():
        print(f"WARNING: bubo_ui.html not found at {UI_FILE}")

    server = HTTPServer(("127.0.0.1", PORT), BuboHandler)
    print(f"\n  🦉  Bubo Interface running at http://localhost:{PORT}")
    print(f"  Adam: http://{ADAM_BASE}" if ADAM_BASE else "  Adam: not configured")
    print(f"  Eve:  http://{EVE_BASE}"  if EVE_BASE  else "  Eve:  not configured")
    print(f"\n  Open http://localhost:{PORT} in your browser")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Bubo Interface stopped.")

if __name__ == "__main__":
    main()
