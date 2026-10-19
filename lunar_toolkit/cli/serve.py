"""serve - the web app, Range-capable Parquet / image serving, a PDS image relay, status and jobs."""

from __future__ import annotations

import argparse
import hmac
import http.server
import ipaddress
import json
import math
import secrets
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any

from ._jobs import TERMINAL, JobContext, JobManager, submit
from ._status import StatusCache, StatusDirs

COMMAND = "serve"

_STATIC_DIR = Path(__file__).parent.parent / "static"
_INDEX = _STATIC_DIR / "index.html"
_BASEMAPS = _STATIC_DIR / "basemaps"

_IMAGE_TYPES = {
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
# Image hosts the page may fetch through /remote: pds.mcp.nasa.gov drops its CORS header
# whenever the request carries extra headers (browsers always send `priority`).
_RELAY_HOSTS = (
    "pds.mcp.nasa.gov",
    "pds.lroc.im-ldi.com",
    "pds.shadowcam.im-ldi.com",
    "data.darts.isas.jaxa.jp",
    "pgda.gsfc.nasa.gov",
)
# Hosts relayed only under these path prefixes (PGDA: the LOLA DEMs for the map's contours)
_RELAY_PATHS = {"pgda.gsfc.nasa.gov": "/data/LOLA_20mpp/"}
_BASEMAP_TYPES = {**_IMAGE_TYPES, ".json": "application/json"}
_MAX_LISTED_IMAGES = 2000
RELAY_MAX = 6
RELAY_WAIT = 30.0
MAX_BODY = 64 * 1024
SSE_KEEPALIVE = 15.0
SSE_MIN_INTERVAL = 0.25
_LOCAL_NAMES = ("localhost", "127.0.0.1", "::1")


def add_server_args(p: argparse.ArgumentParser, host: str, port: int) -> None:
    """Options shared by ``lunar serve`` and ``lunar app``."""
    p.add_argument(
        "--port",
        type=int,
        default=port,
        metavar="PORT",
        help=f"Port to listen on (default: {port or 'any free'}).",
    )
    p.add_argument(
        "--host",
        default=host,
        metavar="HOST",
        help=f"Address to bind (default: {host}); 0.0.0.0 serves the network (read-only there).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Directory containing parquet files. Defaults to LUNAR_DATA_DIR or the user data dir.",
    )
    p.add_argument(
        "--images",
        default=None,
        metavar="DIR",
        help="Also serve the GeoTIFF / PNG / JPEG files under DIR (with Range) for the Image view.",
    )
    p.add_argument(
        "--runs-dir",
        default=None,
        metavar="DIR",
        help="SfS runs listed in the Settings tab (default: LUNAR_RUNS_DIR).",
    )
    p.add_argument(
        "--admin-token",
        default=None,
        metavar="TOKEN",
        help="Let remote clients with this token (Authorization: Bearer TOKEN or ?token=TOKEN) "
        "start jobs and see local paths; 'auto' generates one.",
    )


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Serve the Lunar Toolkit web app with local parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar serve
  lunar serve --port 9000
  lunar serve --data-dir /mnt/lroc-data
  lunar serve --images ~/sfs_runs      # browse local GeoTIFF / COG results in the Image view
  lunar serve --host 0.0.0.0           # share on the network; remote clients are read-only
  lunar serve --host 0.0.0.0 --admin-token auto   # ... and print a token for remote admin

Admin (the Settings tab: build indexes, download, see local paths) is allowed from this
machine (loopback) and for requests carrying --admin-token; everyone else can browse,
search and view. Use `lunar app` for a desktop window.
""",
    )
    add_server_args(p, "localhost", 8080)
    p.add_argument("--no-browser", action="store_true", help="Don't open a browser tab automatically.")


def _byte_range(header: str, size: int) -> tuple[int, int] | tuple[()] | None:
    """``(start, length)`` for a single ``bytes=`` Range, ``()`` without one, None if unsatisfiable."""
    if not header.startswith("bytes="):
        return ()
    first, _, last = header[6:].split(",")[0].strip().partition("-")
    try:
        if not first:  # suffix range: the last N bytes
            n = int(last)
            start, end = max(size - n, 0), size - 1
        else:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
    except ValueError:
        return None
    if start >= size or end < start:
        return None
    return start, end - start + 1


def _image_path(root: Path, rel: str) -> Path | None:
    """The image file ``rel`` (URL-quoted, relative) under ``root``, or None if it escapes or isn't one."""
    parts = urllib.parse.unquote(rel).replace("\\", "/").split("/")
    if not rel or any(not part or part.startswith(".") for part in parts):
        return None
    base = root.resolve()
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(base) or path.suffix.lower() not in _IMAGE_TYPES or not path.is_file():
        return None
    return path


def _list_images(root: Path) -> list[str]:
    """Relative POSIX paths of the image files under ``root`` (hidden entries skipped)."""
    base = root.resolve()
    out = []
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base)
        if any(part.startswith(".") for part in rel.parts) or path.suffix.lower() not in _IMAGE_TYPES:
            continue
        if path.is_file() and path.resolve().is_relative_to(base):
            out.append(rel.as_posix())
            if len(out) >= _MAX_LISTED_IMAGES:
                break
    return out


def _report_path(runs_dir: Path, name: str) -> Path | None:
    """``<runs_dir>/<name>/report/report.html`` if *name* is a plain run dir name and it exists."""
    name = urllib.parse.unquote(name)
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    base = runs_dir.resolve()
    path = (base / name / "report" / "report.html").resolve()
    return path if path.is_relative_to(base) and path.is_file() else None


def _relay_allowed(url: str) -> bool:
    """Only https image files on the PDS / JAXA image hosts are relayed."""
    try:
        u = urllib.parse.urlsplit(url)
        port = u.port
    except ValueError:
        return False
    return (
        u.scheme == "https"
        and u.hostname in _RELAY_HOSTS
        and port is None
        and not u.username
        and Path(u.path).suffix.lower() in _IMAGE_TYPES
        and u.path.startswith(_RELAY_PATHS.get(u.hostname, "/"))
    )


class _RelayRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not _relay_allowed(newurl):
            raise urllib.error.HTTPError(newurl, 403, "redirect outside the relay hosts", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_relay_opener = urllib.request.build_opener(_RelayRedirect)


def is_loopback(addr: str) -> bool:
    """Whether client address *addr* is this machine (IPv4, IPv6 or IPv4-mapped IPv6)."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _local_host_header(host: str) -> bool:
    """A ``Host`` header naming this machine (guards loopback admin against DNS rebinding)."""
    try:
        name = urllib.parse.urlsplit("//" + host).hostname or ""
    except ValueError:
        return False
    return name in _LOCAL_NAMES or name.endswith(".localhost")


@dataclass
class ServerConfig:
    """What a :class:`LunarServer` serves and where its jobs write."""

    data_dir: Path
    explicit_dir: Path | None = None
    images_dir: Path | None = None
    runs_dir: Path | None = None
    admin_token: str | None = None
    cache_dir: Path | None = None
    download_dir: Path | None = None
    reference_dir: Path | None = None

    def __post_init__(self) -> None:
        from ..constants import DEFAULT_CACHE_DIR
        from ..reference import REFERENCE_CACHE_DIR

        self.cache_dir = self.cache_dir or DEFAULT_CACHE_DIR
        self.download_dir = self.download_dir or self.cache_dir / "downloads"
        self.reference_dir = self.reference_dir or REFERENCE_CACHE_DIR


class LunarServer(http.server.ThreadingHTTPServer):
    """The HTTP server with its job manager, status cache and relay limit."""

    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], cfg: ServerConfig, workers: int = 2) -> None:
        self.cfg = cfg
        assert cfg.cache_dir and cfg.download_dir and cfg.reference_dir
        dirs = StatusDirs(cfg.data_dir, cfg.cache_dir, cfg.download_dir, cfg.reference_dir, cfg.runs_dir)
        self.status = StatusCache(dirs)
        self.jobs = JobManager(workers, on_finish=lambda job: self.status.invalidate())
        self.job_context = JobContext(cfg.data_dir, cfg.download_dir, cfg.reference_dir)
        self.relay_slots = threading.BoundedSemaphore(RELAY_MAX)
        self.sdem_members: dict[int, list[dict[str, Any]]] = {}
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, LunarHandler)

    def close(self) -> None:
        """Cancel jobs, stop serving and close the socket."""
        self.jobs.close()
        self.shutdown()
        self.server_close()


class LunarHandler(http.server.BaseHTTPRequestHandler):
    # Keep-alive: DuckDB-WASM and the image viewer issue thousands of small Range requests
    protocol_version = "HTTP/1.1"
    server: LunarServer

    def client_ip(self) -> str:
        return self.client_address[0]

    def _query(self) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(self.path.partition("?")[2])

    def _token(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return self._query().get("token", [None])[0]

    def is_admin(self) -> bool:
        """Loopback clients (with a local ``Host``) and holders of the admin token."""
        token, given = self.server.cfg.admin_token, self._token()
        if token and given and hmac.compare_digest(given.encode(), token.encode()):
            return True
        return is_loopback(self.client_ip()) and _local_host_header(self.headers.get("Host", ""))

    def do_GET(self) -> None:
        self._route(body=True)

    def do_HEAD(self) -> None:
        self._route(body=False)

    def do_POST(self) -> None:
        path = self.path.partition("?")[0]
        if path != "/api/jobs":
            self._error(404, "not found")
            return
        if not self._check_admin(write=True):
            return
        if "application/json" not in self.headers.get("Content-Type", ""):
            self._error(415, "expected application/json")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            self._error(400 if n <= 0 else 413, "bad request body")
            return
        try:
            payload = json.loads(self.rfile.read(n))
            job = submit(self.server.jobs, payload, self.server.job_context)
        except ValueError as exc:  # also json.JSONDecodeError
            self._error(409 if "already" in str(exc) else 400, str(exc))
            return
        except RuntimeError as exc:
            self._error(503, str(exc))
            return
        self._json(job.to_dict(), code=201)

    def do_DELETE(self) -> None:
        path = self.path.partition("?")[0]
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["api", "jobs"]:
            self._error(404, "not found")
            return
        if not self._check_admin(write=True):
            return
        job = self.server.jobs.cancel(parts[2])
        if job is None:
            self._error(404, "no such job")
            return
        self._json(job.to_dict())

    def do_OPTIONS(self) -> None:
        self._error(405, "method not allowed")

    def _check_admin(self, write: bool = False) -> bool:
        if not self.is_admin():
            self._error(403, "read-only client: admin needs localhost or the --admin-token")
            return False
        origin = self.headers.get("Origin")
        if write and origin and urllib.parse.urlsplit(origin).netloc != self.headers.get("Host", ""):
            self._error(403, "cross-origin request")
            return False
        return True

    def _route(self, body: bool) -> None:
        path, _, query = self.path.partition("?")
        cfg = self.server.cfg
        if path in ("/", "/index.html"):
            self._serve(_INDEX, "text/html; charset=utf-8", body)
        elif path == "/api/config":
            from ..constants import APP_VERSION

            self._json(
                {
                    "relay": list(_RELAY_HOSTS),
                    "images": cfg.images_dir is not None,
                    "admin": self.is_admin(),
                    "jobs": True,
                    "runs": cfg.runs_dir is not None,
                    "version": APP_VERSION,
                },
                body,
            )
        elif path.startswith("/basemaps/"):
            name = path[len("/basemaps/") :]
            ctype = _BASEMAP_TYPES.get(Path(name).suffix.lower())
            if not ctype or "/" in name or name.startswith("."):
                self._send_code(404)
                return
            self._serve(_BASEMAPS / name, ctype, body)
        elif path == "/api/images":
            self._json(_list_images(cfg.images_dir) if cfg.images_dir else [], body)
        elif path.startswith("/images/") and cfg.images_dir is not None:
            img = _image_path(cfg.images_dir, path[len("/images/") :])
            if img is None:
                self._send_code(404)
                return
            self._serve(img, _IMAGE_TYPES[img.suffix.lower()], body)
        elif path == "/remote":
            self._relay(urllib.parse.parse_qs(query).get("url", [""])[0], body)
        elif path.startswith("/data/"):
            from ..constants import find_parquet

            name = path[len("/data/") :]
            if not name or "/" in name or name.startswith("."):
                self._send_code(404)
                return
            self._serve(find_parquet(name, cfg.data_dir), "application/octet-stream", body)
        elif path == "/api/status":
            refresh = self._query().get("refresh", ["0"])[0] == "1"
            self._json(self.server.status.get(self.is_admin(), refresh=refresh), body)
        elif path == "/api/regions":
            self._json(_regions(), body)
        elif path == "/api/sdem/members":
            if self._check_admin():
                self._sdem_members(self._query().get("region", [""])[0], body)
        elif path.startswith("/api/jobs"):
            self._jobs_get(path, body)
        elif path.startswith("/runs/") and path.endswith("/report.html") and cfg.runs_dir is not None:
            if not self._check_admin():
                return
            report = _report_path(cfg.runs_dir, path[len("/runs/") : -len("/report.html")])
            if report is None:
                self._send_code(404)
                return
            self._serve(report, "text/html; charset=utf-8", body)
        else:
            self._send_code(404)

    def _jobs_get(self, path: str, body: bool) -> None:
        admin = self.is_admin()
        parts = path.strip("/").split("/")[2:]
        jobs = self.server.jobs
        if not parts:
            self._json([j.to_dict(admin) for j in jobs.jobs()], body)
        elif parts == ["events"]:
            self._events(None, admin)
        elif len(parts) in (1, 2) and parts[1:] in ([], ["events"]):
            job = jobs.get(parts[0])
            if job is None:
                self._error(404, "no such job")
            elif len(parts) == 2:
                self._events(parts[0], admin)
            else:
                self._json(job.to_dict(admin), body)
        else:
            self._error(404, "not found")

    def _events(self, jid: str | None, admin: bool) -> None:
        """Server-Sent Events: ``event: jobs`` (all jobs) or ``event: job`` (one) on every change."""
        jobs = self.server.jobs
        self.send_response(200)
        self._common_headers("text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True
        version = -1
        data: Any
        try:
            while not jobs.closed:
                new = jobs.wait(version, SSE_KEEPALIVE)
                if new == version:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                version = new
                if jid is None:
                    event, data = "jobs", [j.to_dict(admin) for j in jobs.jobs()]
                    final = False
                else:
                    job = jobs.get(jid)
                    if job is None:
                        break
                    event, data, final = "job", job.to_dict(admin), job.status in TERMINAL
                self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
                if final:
                    break
                time.sleep(SSE_MIN_INTERVAL)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _sdem_members(self, region: str, body: bool) -> None:
        from .. import reference as ref

        try:
            reg = ref.get_region(region)
        except KeyError as exc:
            self._error(400, str(exc.args[0] if exc.args else exc))
            return
        cache = self.server.sdem_members
        if reg.id not in cache:
            try:
                df = ref.sdem_members(reg)
            except OSError as exc:
                self._error(502, f"could not read the zip directory: {exc}")
                return
            cache[reg.id] = [{"name": str(r["name"]), "size": int(r["size"])} for r in df.to_dict("records")]
        self._json({"region": reg.id, "zip": reg.zenodo_file, "members": cache[reg.id]}, body)

    def _serve(self, path: Path, ctype: str, body: bool = True) -> None:
        if not path.exists():
            self._send_code(404)
            return
        size = path.stat().st_size
        span = _byte_range(self.headers.get("Range", ""), size)
        if span is None:
            self._send_code(416)
            return
        start, length = span if span else (0, size)
        self.send_response(206 if span else 200)
        self._common_headers(ctype)
        self.send_header("Content-Length", str(length))
        if span:
            self.send_header("Content-Range", f"bytes {start}-{start + length - 1}/{size}")
        self.end_headers()
        if body:
            self._send_bytes(path, start, length)

    def _json(self, obj: object, body: bool = True, code: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self._common_headers("application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(data)

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, self.command != "HEAD", code)

    def _relay(self, url: str, body: bool) -> None:
        """Forward one Range request, through the downloader's per-host slots, pacing and cooldown."""
        from .. import downloader

        if not _relay_allowed(url):
            self._send_code(403)
            return
        host = downloader._host(url)
        wait = host.cooldown_until - time.time()
        if wait > 0:
            self._send_code(429, {"Retry-After": str(math.ceil(wait))})
            return
        if not self.server.relay_slots.acquire(timeout=RELAY_WAIT):
            self._send_code(503, {"Retry-After": "2"})
            return
        try:
            if not host.slots.acquire(timeout=RELAY_WAIT):
                self._send_code(503, {"Retry-After": "2"})
                return
            try:
                host.pace()
                self._relay_once(url, body, host)
            finally:
                host.slots.release()
        finally:
            self.server.relay_slots.release()

    def _relay_once(self, url: str, body: bool, host: Any) -> None:
        from .. import downloader

        headers = {"User-Agent": downloader._USER_AGENT}
        if self.headers.get("Range"):
            headers["Range"] = self.headers["Range"]
        req = urllib.request.Request(url, headers=headers, method="GET" if body else "HEAD")
        started = False
        try:
            with _relay_opener.open(req, timeout=60) as resp:
                self.send_response(resp.status)
                started = True
                self._common_headers(resp.headers.get("Content-Type", "application/octet-stream"))
                for name in ("Content-Length", "Content-Range", "ETag", "Last-Modified"):
                    if resp.headers.get(name):
                        self.send_header(name, resp.headers[name])
                if not resp.headers.get("Content-Length"):
                    self.close_connection = True
                self.end_headers()
                if body:
                    shutil.copyfileobj(resp, self.wfile, 65536)
        except urllib.error.HTTPError as e:
            if e.code in downloader.THROTTLE_CODES:
                wait = downloader._retry_after(e) or 60.0
                host.cool(wait)
                if not started:
                    self._send_code(429, {"Retry-After": str(math.ceil(wait))})
            elif not started:
                self._send_code(e.code)
        except OSError:  # upstream unreachable / timed out, or the browser went away
            self.close_connection = True
            if not started:
                self._send_code(502)

    def _common_headers(self, ctype: str) -> None:
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        # Cross-origin isolation for DuckDB-WASM; `credentialless` (not require-corp) so
        # browse images from hosts without CORP/CORS (JAXA DARTS) still load in <img>.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")

    def _send_bytes(self, path: Path, offset: int, length: int) -> None:
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                buf = f.read(min(65536, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):  # the browser aborted the read
                    self.close_connection = True
                    return
                remaining -= len(buf)

    def _send_code(self, code: int, headers: dict[str, str] | None = None) -> None:
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass  # suppress per-request noise; server start/stop is printed below


def _regions() -> dict[str, Any]:
    from .. import reference as ref

    return {
        "regions": [
            {
                "id": r.id,
                "name": r.name,
                "pgda_site": r.pgda_site,
                "zenodo_file": r.zenodo_file,
                "zenodo_size": r.zenodo_size,
            }
            for r in ref.REGIONS
        ],
        "pgda_kinds": list(ref.PGDA_KINDS),
    }


def make_server(args: argparse.Namespace) -> tuple[LunarServer, ServerConfig]:
    """Bind a :class:`LunarServer` from the shared options (ValueError for a bad option)."""
    import os

    from ..config import RUNS_DIR_ENV
    from ..constants import get_data_dir

    explicit = Path(args.data_dir).expanduser() if args.data_dir else None
    images = Path(args.images).expanduser() if args.images else None
    if images is not None and not images.is_dir():
        raise ValueError(f"--images: not a directory: {images}")
    runs = args.runs_dir or os.environ.get(RUNS_DIR_ENV)
    token = args.admin_token
    if token == "auto":
        token = secrets.token_urlsafe(18)
    elif token is not None and len(token) < 12:
        raise ValueError("--admin-token: use at least 12 characters (or 'auto')")
    cfg = ServerConfig(
        data_dir=explicit or get_data_dir(),
        explicit_dir=explicit,
        images_dir=images,
        runs_dir=Path(runs).expanduser() if runs else None,
        admin_token=token,
    )
    return LunarServer((args.host, args.port), cfg), cfg


def server_url(server: LunarServer, host: str) -> str:
    """URL for this machine's browser: wildcard binds map to localhost."""
    port = server.server_address[1]
    name = "localhost" if host in ("", "0.0.0.0", "::") else host
    return f"http://{f'[{name}]' if ':' in name else name}:{port}"


def print_banner(server: LunarServer, cfg: ServerConfig, host: str, title: str) -> str:
    url = server_url(server, host)
    print(f"{title} -> {url}")
    print(f"  data dir : {cfg.data_dir}")
    if cfg.images_dir is not None:
        print(f"  images   : {cfg.images_dir}")
    if cfg.runs_dir is not None:
        print(f"  runs dir : {cfg.runs_dir}")
    if not (is_loopback(host) or host == "localhost"):
        print("  network  : remote clients are read-only (browse, search, view)")
        if cfg.admin_token:
            print(f"  admin    : {url}/?token={cfg.admin_token}")
    return url


def run(args: argparse.Namespace) -> int:
    try:
        server, cfg = make_server(args)
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    url = print_banner(server, cfg, args.host, "Lunar Toolkit web app")
    print("  Ctrl-C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.jobs.close()
        server.server_close()
    return 0
