"""Podcast URL resolution against a local HTTP server: direct audio, RSS feed, redirects, limits."""
import http.server
import threading

import pytest

from src.asr import url_audio

MP3 = b"ID3" + b"\x00" * 2000
FEED = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel><title>Show</title>
<item><title>Ep 2: Drones in Rwanda</title><enclosure url="/ep2.mp3" type="audio/mpeg" length="2003"/></item>
<item><title>Ep 1: Batteries</title><enclosure url="/ep1.mp3" type="audio/mpeg" length="2003"/></item>
</channel></rss>"""
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>A</title>
<link rel="enclosure" href="/ep1.mp3"/></entry></feed>"""


class Handler(http.server.BaseHTTPRequestHandler):
    routes = {"/ep1.mp3": ("audio/mpeg", MP3), "/ep2.mp3": ("application/octet-stream", MP3 + b"2"),
              "/feed": ("application/rss+xml", FEED), "/atom": ("text/xml", ATOM),
              "/page": ("text/html", b"<html><body>no audio</body></html>")}

    def do_GET(self):
        if self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "/feed")
            self.end_headers()
            return
        ctype, body = self.routes.get(self.path, ("text/plain", b""))
        self.send_response(200 if self.path in self.routes else 404)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_direct_audio_link(server, tmp_path):
    got = url_audio.resolve(f"{server}/ep1.mp3", tmp_path, allow_private=True)
    assert got["kind"] == "audio" and got["path"].read_bytes() == MP3


def test_feed_takes_newest_or_named_episode_and_follows_redirects(server, tmp_path):
    newest = url_audio.resolve(f"{server}/moved", tmp_path / "a", allow_private=True)
    assert newest["kind"] == "feed" and newest["title"] == "Ep 2: Drones in Rwanda"
    assert newest["path"].read_bytes() == MP3 + b"2"
    named = url_audio.resolve(f"{server}/feed", tmp_path / "b", episode="batteries", allow_private=True)
    assert named["title"] == "Ep 1: Batteries" and named["path"].read_bytes() == MP3
    with pytest.raises(ValueError, match="No episode"):
        url_audio.resolve(f"{server}/feed", tmp_path / "c", episode="nothing like it", allow_private=True)


def test_atom_enclosures():
    assert url_audio.feed_episodes(ATOM) == [{"title": "A", "url": "/ep1.mp3"}]
    assert url_audio.feed_episodes(b"<html><body/></html>") is None


def test_size_limit(server, tmp_path):
    with pytest.raises(ValueError, match="larger than"):
        url_audio.resolve(f"{server}/ep1.mp3", tmp_path, allow_private=True, max_bytes=100)
    assert not list(tmp_path.glob("audio.*"))


def test_pages_go_to_yt_dlp(server, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(url_audio, "_yt_dlp", lambda url, dest, max_bytes: seen.append(url) or (dest / "x.m4a", "T"))
    got = url_audio.resolve(f"{server}/page", tmp_path, allow_private=True)
    assert got["kind"] == "page" and got["title"] == "T" and seen == [f"{server}/page"]


@pytest.mark.parametrize("url", ["http://127.0.0.1/x.mp3", "http://localhost:8080/feed", "http://10.1.2.3/a",
                                 "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://example.com/a"])
def test_private_and_non_http_urls_are_refused(url, tmp_path):
    with pytest.raises(ValueError):
        url_audio.resolve(url, tmp_path)
