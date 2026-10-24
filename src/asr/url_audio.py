"""Podcast URL -> local audio file + episode title, for the demo (app/gradio_app.py).

Three kinds of URL:
  - a direct audio link (.mp3, .m4a, ... or an audio/* response)
  - a podcast RSS / Atom feed: the newest episode, or the first whose title contains `episode`
  - an episode page that yt-dlp can read (YouTube, SoundCloud, Apple Podcasts, ...; optional
    dependency). DRM-protected services such as Spotify cannot be downloaded.
The demo can be shared through a public link, so only http(s) URLs that resolve to public
addresses are fetched, every redirect is checked again, and downloads stop at `max_bytes`.
"""
from __future__ import annotations

import ipaddress
import socket
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

AUDIO_EXT = (".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac", ".webm")
MAX_BYTES = 500 * 1024 * 1024
MAX_FEED_BYTES = 20 * 1024 * 1024
HEADERS = {"User-Agent": "pod2post/0.1 (podcast-to-article demo)"}


def check_public(url: str) -> None:
    """Refuse non-http(s) URLs and hosts on loopback, private or link-local networks."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Use an http:// or https:// link.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError(f"Cannot resolve {parts.hostname}.") from None
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise ValueError(f"{parts.hostname} is not a public address.")


def _get(url: str, allow_private: bool) -> requests.Response:
    """GET with redirects followed by hand, so each hop is checked."""
    for _ in range(6):
        if not allow_private:
            check_public(url)
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=30, allow_redirects=False)
        if resp.is_redirect:
            url = urljoin(url, resp.headers["location"])
            resp.close()
            continue
        resp.raise_for_status()
        return resp
    raise ValueError("Too many redirects.")


def _is_audio(url: str, content_type: str) -> bool:
    return (content_type.startswith(("audio/", "video/")) or content_type == "application/octet-stream"
            or urlparse(url).path.lower().endswith(AUDIO_EXT))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def feed_episodes(xml: bytes) -> list[dict] | None:
    """[{title, url}] in feed order (newest first in practice), for RSS <enclosure> and Atom
    rel=enclosure; None if the document is not a feed."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    if _local(root.tag) not in ("rss", "feed", "RDF"):
        return None
    episodes = []
    for item in root.iter():
        if _local(item.tag) not in ("item", "entry"):
            continue
        title, audio = "", None
        for child in item:
            name = _local(child.tag)
            if name == "title":
                title = (child.text or "").strip()
            elif name == "enclosure" and child.get("url"):
                audio = audio or child.get("url")
            elif name == "link" and child.get("rel") == "enclosure" and child.get("href"):
                audio = audio or child.get("href")
        if audio:
            episodes.append({"title": title, "url": audio})
    return episodes


def _save(resp: requests.Response, dest: Path, max_bytes: int) -> Path:
    suffix = next((e for e in AUDIO_EXT if urlparse(resp.url).path.lower().endswith(e)), ".mp3")
    path, size = dest / f"audio{suffix}", 0
    with path.open("wb") as f:
        for block in resp.iter_content(1 << 20):
            size += len(block)
            if size > max_bytes:
                resp.close()
                path.unlink()
                raise ValueError(f"The audio is larger than {max_bytes // 2**20} MB.")
            f.write(block)
    return path


def _yt_dlp(url: str, dest: Path, max_bytes: int) -> tuple[Path, str | None]:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("This is not a direct audio link or a podcast feed. Reading episode pages needs "
                           "yt-dlp (pip install yt-dlp).") from None
    opts = {"format": "bestaudio/best", "outtmpl": str(dest / "audio.%(ext)s"), "noplaylist": True,
            "playlist_items": "1", "max_filesize": max_bytes, "quiet": True, "no_warnings": True, "noprogress": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True) or {}
    files = sorted(dest.glob("audio.*"))
    if not files:
        raise RuntimeError("No downloadable audio found on that page (DRM-protected services such as Spotify "
                           f"are not supported, and files over {max_bytes // 2**20} MB are skipped).")
    entries = info.get("entries")
    return files[0], (next(iter(entries), {}) if entries else info).get("title")


def resolve(url: str, dest: Path, episode: str = "", allow_private: bool = False,
            max_bytes: int = MAX_BYTES) -> dict:
    """Download the audio behind `url` into `dest`; returns {path, title, kind, audio_url}."""
    url, dest = url.strip(), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    resp = _get(url, allow_private)
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if _is_audio(resp.url, content_type):
        return {"path": _save(resp, dest, max_bytes), "title": None, "kind": "audio", "audio_url": resp.url}
    head = resp.raw.read(MAX_FEED_BYTES + 1, decode_content=True)
    resp.close()
    episodes = feed_episodes(head) if not content_type.startswith("text/html") else None
    if episodes is not None:
        if not episodes:
            raise ValueError("That feed has no episodes with audio.")
        pick = next((e for e in episodes if episode.strip().lower() in e["title"].lower()), None) if episode.strip() \
            else episodes[0]
        if pick is None:
            raise ValueError(f"No episode in the feed has \"{episode}\" in its title.")
        audio = _get(urljoin(resp.url, pick["url"]), allow_private)
        return {"path": _save(audio, dest, max_bytes), "title": pick["title"] or None, "kind": "feed",
                "audio_url": audio.url}
    path, title = _yt_dlp(url, dest, max_bytes)
    return {"path": path, "title": title, "kind": "page", "audio_url": url}
