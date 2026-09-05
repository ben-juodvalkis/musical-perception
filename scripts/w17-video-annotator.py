"""W17 — mark a demonstration against the VIDEO, not just the audio.

Whether a teacher is dancing full-out or merely marking is largely a VISUAL
judgement (owner, 2026-09-05), so the annotation pass needs the picture. This
serves a small page that plays the clip, takes keyboard marks, and writes the
Audacity-format label file straight into the repo - no export dialog, no
hunting for where the save sheet put the file.

    python scripts/w17-video-annotator.py --clip barre6-frappe-demo

Then open the printed URL. Marks save to
docs/research/w17/<clip>.owner-windows.txt, which
scripts/w17-owner-annotation.py reads unchanged.
"""
from __future__ import annotations

import argparse
import http.server
import os
import re
import socketserver
import threading
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "research" / "w17"


def build(clip: str) -> tuple[str, Path]:
    import yaml
    case = yaml.safe_load((ROOT / "evals" / "cases" / f"{clip}.yaml").read_text())
    media = ROOT / case["input"]["media"]
    if not media.is_file():
        raise SystemExit(f"media not found: {media}")
    html = (ROOT / "scripts" / "w17_annotator.html").read_text()
    html = html.replace("__CLIP__", clip).replace("__VIDEO__", "/media")
    return html, media


class Handler(http.server.BaseHTTPRequestHandler):
    html = ""
    media: Path = Path()
    dest: Path = Path()

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, self.html.encode(), "text/html; charset=utf-8")
        if path == "/media":
            return self._serve_media()
        self._send(404, b"not found", "text/plain")

    def _serve_media(self):
        """Range-aware so the browser can seek; SimpleHTTPRequestHandler cannot."""
        size = self.media.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                status = 206
        end = min(end, size - 1)
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with self.media.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return                      # the browser seeked away; normal
                remaining -= len(chunk)

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/save":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        if self.dest.exists():                  # never clobber a pass silently
            bak = self.dest.with_suffix(".txt.bak")
            bak.write_text(self.dest.read_text())
        self.dest.write_text(body)
        rel = self.dest.relative_to(ROOT)
        print(f"  saved {len(body.splitlines())} mark(s) -> {rel}")
        self._send(200, str(rel).encode(), "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="barre6-frappe-demo")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    Handler.html, Handler.media = build(a.clip)
    Handler.dest = OUT / f"{a.clip}.owner-windows.txt"

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), Handler) as srv:
        url = f"http://127.0.0.1:{a.port}/"
        print(f"annotator for {a.clip}")
        print(f"  video  {Handler.media.relative_to(ROOT)}")
        print(f"  saves  {Handler.dest.relative_to(ROOT)}")
        print(f"\n  {url}\n\nCtrl-C when you are done.")
        if not a.no_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
