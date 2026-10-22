"""Read LaSOT straight from the per-class zips on HuggingFace with HTTP range requests.

Nothing is downloaded except the bytes actually needed, so a single video (or only its
annotations) can be pulled out of a 2-4 GB class archive.
"""
from __future__ import annotations

import io
import queue
import threading
import time
import zipfile
from pathlib import Path

import requests

HF_FILE = "https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/{}"
HF_TREE = "https://huggingface.co/api/datasets/l-lt/LaSOT/tree/main"
ANNOT_FILES = ("groundtruth.txt", "full_occlusion.txt", "out_of_view.txt", "nlp.txt")


def retry(request, tries: int = 7):
    """Run an HTTP request, waiting 2, 4, 8, ... s between attempts (survives short network drops)."""
    for t in range(tries):
        try:
            r = request()
            r.raise_for_status()
            return r
        except requests.RequestException:
            if t == tries - 1:
                raise
            time.sleep(2 ** (t + 1))


class HttpFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, so zipfile can read a remote archive."""

    def __init__(self, url: str, min_block: int = 256 << 10):
        self.session = requests.Session()
        r = retry(lambda: self.session.head(url, allow_redirects=True, timeout=60))
        self.url, self.size = r.url, int(r.headers["Content-Length"])
        self.pos, self.min_block = 0, min_block
        self.cache_start, self.cache = -1, b""

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        cached = self.cache_start <= self.pos and end <= self.cache_start + len(self.cache)
        if not cached:
            stop = min(max(end, self.pos + self.min_block), self.size)
            self.cache_start, self.cache = self.pos, self._get(self.pos, stop - 1)
        a = self.pos - self.cache_start
        out = self.cache[a:a + end - self.pos]
        self.pos = end
        return out

    def prefetch(self, first: int, last: int):
        """Load bytes [first, last] in one request; later reads inside that span hit the cache."""
        self.cache_start, self.cache = first, self._get(first, min(last, self.size - 1))

    def _get(self, first, last):
        headers = {"Range": f"bytes={first}-{last}"}
        return retry(lambda: self.session.get(self.url, headers=headers, timeout=600)).content


def list_classes() -> list[str]:
    files = retry(lambda: requests.get(HF_TREE, timeout=60)).json()
    return sorted(f["path"][:-4] for f in files if f["path"].endswith(".zip"))


def open_class_zip(cls: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpFile(HF_FILE.format(f"{cls}.zip")))


def frame_names(z: zipfile.ZipFile, seq: str) -> list[str]:
    return sorted(n for n in z.namelist()
                  if n.startswith(f"{seq}/img/") and n.lower().endswith(".jpg"))


def _prefetch(z: zipfile.ZipFile, infos: list[zipfile.ZipInfo]):
    # a video's frames are stored contiguously: one range request instead of thousands
    first = min(i.header_offset for i in infos)
    last = max(i.header_offset + 30 + len(i.orig_filename.encode()) + len(i.extra) + i.compress_size for i in infos)
    z.fp.prefetch(first, last + 1024)  # margin for local-header extras that differ from the central dir


def extract_frames(z: zipfile.ZipFile, seq: str, indices, out_dir: Path) -> Path:
    """Write only the given 0-based frames of one video (plus annotations) to out_dir/<seq>/.
    Frames already on disk are skipped; nearby indices are fetched with a single range request."""
    seq_dir = Path(out_dir) / seq
    (seq_dir / "img").mkdir(parents=True, exist_ok=True)
    names = frame_names(z, seq)
    for f in ANNOT_FILES:
        if not (seq_dir / f).exists() and f"{seq}/{f}" in z.NameToInfo:
            (seq_dir / f).write_bytes(z.read(f"{seq}/{f}"))
    todo = sorted({int(i) for i in indices if 0 <= i < len(names)})
    todo = [i for i in todo if not (seq_dir / "img" / names[i].rsplit("/", 1)[1]).exists()]
    # group into chunks of nearby frames so one far-away frame does not drag in the whole video
    chunks, cur = [], []
    for i in todo:
        if cur and i - cur[-1] > 64:
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    for chunk in chunks:
        infos = [z.getinfo(names[i]) for i in chunk]
        _prefetch(z, infos)
        for i in infos:
            (seq_dir / "img" / i.filename.rsplit("/", 1)[1]).write_bytes(z.read(i))
    return seq_dir


def extract_sequence(z: zipfile.ZipFile, seq: str, out_dir: Path) -> Path:
    """Write annotations and all frames of one video to out_dir/<seq>/ (frames fetched in one request)."""
    seq_dir = Path(out_dir) / seq
    (seq_dir / "img").mkdir(parents=True, exist_ok=True)
    names = set(z.namelist())
    for f in ANNOT_FILES:
        if f"{seq}/{f}" in names:
            (seq_dir / f).write_bytes(z.read(f"{seq}/{f}"))
    infos = [z.getinfo(n) for n in frame_names(z, seq)]
    if infos:
        _prefetch(z, infos)
        for i in infos:
            (seq_dir / "img" / i.filename.rsplit("/", 1)[1]).write_bytes(z.read(i))
    return seq_dir


def stream_sequences(seqs: list[str], work: Path, ahead: int = 2):
    """Yield (seq, seq_dir, error) while the next videos download in a background thread.
    At most `ahead` finished videos wait on disk; the consumer deletes each one when done."""
    q: queue.Queue = queue.Queue(maxsize=ahead)

    def producer():
        zips = {}
        for seq in seqs:
            cls = seq.rsplit("-", 1)[0]
            try:
                if cls not in zips:
                    zips = {cls: open_class_zip(cls)}  # keep one archive index in memory
                q.put((seq, extract_sequence(zips[cls], seq, work), None))
            except Exception as e:  # report and move on; a re-run retries it
                q.put((seq, None, e))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    yield from iter(q.get, None)
