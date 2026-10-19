"""PdsDownloader — parallel file retrieval from the PDS archive."""

import contextlib
import contextvars
import email.utils
import hashlib
import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .constants import APP_NAME, APP_VERSION, DEFAULT_CACHE_DIR, modernize_pds_url
from .tools import reuse_existing

log = logging.getLogger(__name__)
_USER_AGENT = f"{APP_NAME}/{APP_VERSION} (python-urllib)"

Progress = Callable[[str, int, int | None], None]
"""``progress(file name, bytes done, bytes total or None)``, called per chunk; raising aborts.

While the host is paused (throttling cooldown) it is called about once a second with
``done = -1``.
"""

_PROGRESS: contextvars.ContextVar[Progress | None] = contextvars.ContextVar("download_progress", default=None)


@contextlib.contextmanager
def report_progress(callback: Progress) -> Iterator[None]:
    """Send the progress of every [`download_file`][lunar_toolkit.downloader.download_file] in this thread /
    context to *callback*."""
    token = _PROGRESS.set(callback)
    try:
        yield
    finally:
        _PROGRESS.reset(token)


class PdsDownloader:
    """Download and cache files from the PDS archive.

    Parameters
    ----------
    cache_dir : Path
        Base directory for the local cache (default ~/.cache/lunar_toolkit).
    max_workers : int
        Thread-pool size for parallel downloads (default 8).
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        max_workers: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_workers = max_workers

    def cache_path(self, url: str) -> Path:
        """Return the local path that would be used to cache *url*."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return self.cache_dir / f"{url_hash}_{url_filename(url)}"

    def ensure_cached(
        self,
        url: str,
        force: bool = False,
        show_progress: bool = True,
        expected_md5: str | None = None,
    ) -> Path:
        """Return local path for *url*, downloading if not already cached."""
        dest = self.cache_path(url)
        if dest.exists() and not force:
            return dest
        return download_file(url, dest, show_progress=show_progress, expected_md5=expected_md5)

    def download(
        self,
        product_rows: pd.DataFrame,
        output_dir: Path,
        url_col: str = "PDS_URL",
        show_progress: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Download PDS image files in parallel.

        Parameters
        ----------
        product_rows : pd.DataFrame
            Must contain a *url_col* column (e.g. from CumIndex.lookup()).
        output_dir : Path
            Destination directory for .IMG files.
        url_col : str
            Column holding the download URL.
        show_progress : bool
            Show per-file progress bars.
        force : bool
            Download again even when ``<output_dir>/<file name>`` exists
            (by default it is reused; downloads land via ``.part`` files,
            so an existing file is complete).

        Returns
        -------
        Copy of product_rows with LOCAL_PATH and DOWNLOAD_ERROR columns added.
        """
        results = self._download_all(list(product_rows[url_col]), output_dir, show_progress, force)
        paths = {u: str(path) if path else None for u, (path, _) in results.items()}
        df = product_rows.copy()
        df["LOCAL_PATH"] = df[url_col].map(paths.get)
        df["DOWNLOAD_ERROR"] = df[url_col].map(lambda u: results.get(u, (None, None))[1])
        return df

    def download_urls(
        self,
        urls: list[str],
        output_dir: Path,
        show_progress: bool = True,
        force: bool = False,
    ) -> dict[str, Path | None]:
        """Download a plain list of URLs in parallel.

        Existing files are reused unless *force* (see
        [`download`][lunar_toolkit.downloader.PdsDownloader.download]). Returns a dict mapping each URL to its
        local path (or None on error).
        """
        path_map: dict[str, Path | None] = {}
        for url, (path, err) in self._download_all(urls, output_dir, show_progress, force).items():
            if err:
                log.error("%s: %s", url, err)
            path_map[url] = path
        return path_map

    def _download_all(
        self, urls: list[str], output_dir: Path, show_progress: bool, force: bool
    ) -> dict[str, tuple[Path | None, str | None]]:
        """``url → (local path, error message)`` for parallel downloads into *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, tuple[Path | None, str | None]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(_download_one, url, output_dir / url_filename(url), show_progress, force)
                for url in urls
            ]
            for future in as_completed(futures):
                url, path, err = future.result()
                results[url] = (path, err)
        return results


class ChecksumError(IOError):
    """Raised when a downloaded file does not match its expected MD5."""


def file_md5(path: Path, chunk: int = 8 << 20) -> str:
    """Return the hex MD5 digest of *path* (streamed, constant memory)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def url_filename(url: str) -> str:
    """File name at the end of *url*'s path (query string ignored)."""
    return Path(urllib.parse.urlparse(url).path).name


RETRY_CODES = (429, 500, 502, 503, 504)
THROTTLE_CODES = (429, 503)
MAX_PER_HOST_ENV = "LUNAR_MAX_PER_HOST"
MIN_INTERVAL_ENV = "LUNAR_MIN_INTERVAL"
MAX_WAIT_ENV = "LUNAR_MAX_WAIT"
MAX_RETRY_AFTER = 7200.0
COOLDOWN_FILE = DEFAULT_CACHE_DIR / "host_cooldowns.json"


class _Host:
    """One host's download slots, request pacing and throttling cooldown.

    The cooldown is wall-clock time shared by every worker in this process and saved
    to `COOLDOWN_FILE`, so a restarted run honours a server's earlier
    ``Retry-After`` instead of tripping its rate limit again.
    """

    def __init__(self, name: str, limit: int, min_interval: float) -> None:
        self.name = name
        self.slots = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.min_interval = min_interval
        self.last_start = 0.0
        self.cooldown_until = _saved_cooldowns().get(name, 0.0)

    def cool(self, seconds: float) -> None:
        with self.lock:
            until = time.time() + seconds
            if until > self.cooldown_until:
                self.cooldown_until = until
                _save_cooldown(self.name, until)

    def wait(self, poll: Callable[[], None] | None = None) -> None:
        """Sleep out the cooldown; with *poll*, in 1 s steps calling it (it may raise to abort)."""
        remaining = self.cooldown_until - time.time()
        if remaining > 0:
            resume = time.strftime("%H:%M:%S", time.localtime(self.cooldown_until))
            log.info("%s asked us to slow down; waiting %.0f s (until %s)", self.name, remaining, resume)
            if poll is None:
                time.sleep(remaining)
                return
            while remaining > 0:
                poll()
                time.sleep(min(1.0, remaining))
                remaining = self.cooldown_until - time.time()

    def pace(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.last_start + self.min_interval)
            self.last_start = start
        if start > now:
            time.sleep(start - now)


_HOSTS: dict[str, _Host] = {}
_HOSTS_LOCK = threading.Lock()


def _host(url: str) -> _Host:
    name = urllib.parse.urlparse(url).netloc.lower()
    with _HOSTS_LOCK:
        if name not in _HOSTS:
            limit = max(1, int(os.environ.get(MAX_PER_HOST_ENV, "3")))
            _HOSTS[name] = _Host(name, limit, float(os.environ.get(MIN_INTERVAL_ENV, "1.0")))
        return _HOSTS[name]


def _saved_cooldowns() -> dict[str, float]:
    try:
        data = json.loads(COOLDOWN_FILE.read_text())
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float)) and v > now}


def _save_cooldown(name: str, until: float) -> None:
    data = _saved_cooldowns()
    data[name] = until
    try:
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = COOLDOWN_FILE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(COOLDOWN_FILE)
    except OSError as exc:
        log.debug("Could not save host cooldowns: %s", exc)


def download_file(
    url: str,
    dest: Path,
    show_progress: bool = True,
    expected_md5: str | None = None,
    resume: bool = True,
    timeout: float = 60.0,
    retries: int = 8,
    retry_codes: tuple[int, ...] = RETRY_CODES,
    max_wait: float | None = None,
    progress: Progress | None = None,
) -> Path:
    """Download *url* to *dest*, throttled per host and retried on throttling / errors.

    Politeness: at most ``$LUNAR_MAX_PER_HOST`` (default 3) downloads run against one
    host at a time, and their requests start at least ``$LUNAR_MIN_INTERVAL`` seconds
    apart (default 1).  Throttling replies (429, 503) pause every download to that host
    for the server's ``Retry-After`` (up to 2 h; else exponential back-off) and do not
    count as failures; they give up only after *max_wait* seconds in total (default
    ``$LUNAR_MAX_WAIT``, else 6 h).  Other *retry_codes* (5xx), connection errors and
    interrupted transfers are retried *retries* times with back-off.  Each retry resumes
    the ``.part`` file; checksum mismatches are not retried.  See `_download_once`.

    *progress* (default: the [`report_progress`][lunar_toolkit.downloader.report_progress] callback, if any)
    gets the bytes done after each chunk; an exception it raises stops the download and keeps the
    ``.part`` file for a later resume.
    """
    progress = progress or _PROGRESS.get()
    url = modernize_pds_url(url)
    host = _host(url)
    budget = max_wait if max_wait is not None else float(os.environ.get(MAX_WAIT_ENV, 6 * 3600))
    deadline = time.time() + budget
    failures = throttles = 0
    poll = (lambda: progress(Path(dest).name, -1, None)) if progress else None
    while True:
        host.wait(poll)
        try:
            with host.slots:
                host.pace()
                return _download_once(url, dest, show_progress, expected_md5, resume, timeout, progress)
        except urllib.error.HTTPError as exc:
            if exc.code not in retry_codes:
                raise
            if exc.code in THROTTLE_CODES:
                wait = _retry_after(exc) or _backoff(throttles)
                throttles += 1
            else:
                failures += 1
                if failures > retries:
                    raise
                wait = _backoff(failures - 1)
            if time.time() + wait > deadline:
                raise
            log.warning("HTTP %d from %s; pausing that host %.0f s", exc.code, host.name, wait)
        except ChecksumError:
            raise
        except OSError as exc:  # URLError, timeouts, resets, incomplete transfers
            failures += 1
            if failures > retries:
                raise
            wait = _backoff(failures - 1)
            log.warning("%s for %s; retrying in %.0f s", exc, url, wait)
        host.cool(wait)


def _backoff(attempt: int) -> float:
    return min(120.0, 2.0 * 2 ** min(attempt, 10)) * random.uniform(0.75, 1.25)


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return min(MAX_RETRY_AFTER, max(0.0, seconds))


def _download_once(
    url: str,
    dest: Path,
    show_progress: bool = True,
    expected_md5: str | None = None,
    resume: bool = True,
    timeout: float = 60.0,
    progress: Progress | None = None,
) -> Path:
    """Download *url* to *dest* via ``<dest>.part`` with HTTP Range resume.

    * If a ``.part`` file exists and *resume* is set, a ``Range: bytes=N-``
      request continues it.  A ``206`` reply is appended; a ``200`` reply
      (server ignored the range) restarts from zero; ``416`` means the part
      is already complete.
    * When *expected_md5* is given the finished file is verified before the
      atomic rename; on mismatch the part file is deleted and
      [`ChecksumError`][lunar_toolkit.downloader.ChecksumError] is raised.
    * On network errors the ``.part`` file is kept so the next call resumes.
    """
    from tqdm import tqdm

    url = modernize_pds_url(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    offset = tmp.stat().st_size if (resume and tmp.exists()) else 0
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    if offset:
        req.add_header("Range", f"bytes={offset}-")

    log.info("Downloading %s%s", url, f" (resuming at {offset:,} bytes)" if offset else "")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not offset:
            raise
        resp = None  # 416: nothing left to fetch — the part file is complete

    if resp is not None:
        with resp:
            status = getattr(resp, "status", 200)
            if offset and status != 206:
                offset = 0  # server ignored Range → start over
            length = int(resp.headers.get("Content-Length", 0)) or None
            total = (length + offset) if length else None
            with (
                open(tmp, "ab" if offset else "wb") as f,
                tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    disable=not show_progress,
                    leave=True,
                ) as bar,
            ):
                chunk = 1 << 20  # 1 MB
                done = offset
                if progress:
                    progress(dest.name, done, total)
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    bar.update(len(buf))
                    done += len(buf)
                    if progress:
                        progress(dest.name, done, total)
            if total is not None and tmp.stat().st_size < total:
                raise OSError(
                    f"Incomplete download of {url}: {tmp.stat().st_size:,}/{total:,} bytes "
                    f"(partial file kept at {tmp}; re-run to resume)"
                )

    if expected_md5:
        actual = file_md5(tmp)
        if actual.lower() != expected_md5.lower():
            tmp.unlink()
            raise ChecksumError(f"MD5 mismatch for {url}: expected {expected_md5}, got {actual}")
    tmp.replace(dest)
    return dest


def _download_one(
    url: str, dest: Path, show_progress: bool, force: bool = False
) -> tuple[str, Path | None, str | None]:
    if reuse_existing(dest, force):
        return url, dest, None
    try:
        return url, download_file(url, dest, show_progress=show_progress), None
    except Exception as exc:
        return url, None, str(exc)
