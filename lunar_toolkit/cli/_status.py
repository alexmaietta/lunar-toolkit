"""What ``lunar serve`` reports in ``/api/status``: indexes, caches, reference data, runs, tools."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_SCAN = 20000
MAX_LISTED = 200
CACHE_SECONDS = 30.0
_PATH_KEYS = {"path", "root", "dest", "executable"}


@dataclass(frozen=True)
class StatusDirs:
    """The directories the status page describes."""

    data_dir: Path
    cache_dir: Path
    download_dir: Path
    reference_dir: Path
    runs_dir: Path | None


def _walk(root: Path, limit: int, state: dict[str, bool] | None = None) -> Iterator[os.DirEntry]:
    """Files under *root* (no symlinks followed), stopping after *limit* entries (``state["truncated"]``)."""
    stack = [root]
    n = 0
    while stack:
        try:
            it = os.scandir(stack.pop())
        except OSError:
            continue
        with it:
            for entry in it:
                n += 1
                if n > limit:
                    if state is not None:
                        state["truncated"] = True
                    return
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        yield entry
                except OSError:
                    continue


def dir_size(root: Path, limit: int = MAX_SCAN) -> dict[str, Any]:
    """``size`` (bytes) and ``files`` under *root*, ``truncated`` when more than *limit* entries."""
    size = files = 0
    state = {"truncated": False}
    for entry in _walk(Path(root), limit, state):
        try:
            size += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        files += 1
    return {"size": size, "files": files, "truncated": state["truncated"]}


def _file(path: Path, name: str | None = None) -> dict[str, Any]:
    try:
        st = path.stat()
    except OSError:
        return {"name": name or path.name, "path": str(path), "size": None, "mtime": None}
    return {"name": name or path.name, "path": str(path), "size": st.st_size, "mtime": st.st_mtime}


def indexes(data_dir: Path) -> list[dict[str, Any]]:
    """Each dataset's Parquet: existence, size and the build sidecar (rows, release, times)."""
    from ..constants import DATASETS

    out = []
    for ds, entry in DATASETS.items():
        p = Path(data_dir) / entry["parquet"]
        meta: dict[str, Any] = {}
        try:
            meta = json.loads(p.with_name(p.stem + ".meta.json").read_text())
        except (OSError, ValueError):
            pass
        row = {"dataset": ds, "label": entry["label"], "exists": p.is_file(), **_file(p)}
        row.update({k: meta.get(k) for k in ("rows", "release", "source", "built_at", "updated_at")})
        row["volumes"] = len(meta.get("volumes") or {})
        out.append(row)
    return out


def cache_summary(cache_dir: Path) -> dict[str, Any]:
    """Top-level entries of the cache dir with their sizes."""
    items = []
    try:
        entries = sorted(Path(cache_dir).iterdir())
    except OSError:
        entries = []
    for p in entries[:MAX_LISTED]:
        if p.is_dir():
            items.append({"name": p.name + "/", "path": str(p), "dir": True, **dir_size(p)})
        else:
            items.append({**_file(p, _cache_name(p.name)), "dir": False, "files": 1})
    total = sum(i["size"] or 0 for i in items)
    return {"path": str(cache_dir), "exists": Path(cache_dir).is_dir(), "size": total, "items": items}


def _cache_name(name: str) -> str:
    """``226215a6_CUMINDEX.TAB`` -> ``CUMINDEX.TAB (226215a6)`` (PdsDownloader cache names)."""
    head, sep, rest = name.partition("_")
    if sep and len(head) == 8 and all(c in "0123456789abcdef" for c in head):
        return f"{rest} ({head})"
    return name


def downloads(download_dir: Path) -> dict[str, Any]:
    """Image files in the download dir (newest first), with ``.part`` files flagged as partial."""
    files = []
    for entry in _walk(Path(download_dir), MAX_SCAN):
        p = Path(entry.path)
        f = _file(p, p.relative_to(download_dir).as_posix())
        f["partial"] = p.suffix == ".part"
        files.append(f)
    files.sort(key=lambda f: -(f["mtime"] or 0))
    return {
        "path": str(download_dir),
        "exists": Path(download_dir).is_dir(),
        "size": sum(f["size"] or 0 for f in files),
        "count": len(files),
        "files": files[:MAX_LISTED],
    }


def reference(reference_dir: Path) -> dict[str, Any]:
    """PGDA sites / kinds and extracted Zenodo SDEM members under the reference cache."""
    from .. import reference as ref

    site_region = {r.pgda_site: f"{r.id} {r.name}" for r in ref.REGIONS if r.pgda_site}
    pgda = []
    root = Path(reference_dir) / "pgda"
    for site_dir in sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []:
        files = []
        for entry in _walk(site_dir, MAX_SCAN):
            p = Path(entry.path)
            kind = ref._classify_pgda(p.relative_to(root).as_posix())
            files.append({**_file(p, p.relative_to(site_dir).as_posix()), "kind": kind})
        kinds: dict[str, int] = {}
        for f in files:
            if f["kind"]:
                kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        pgda.append(
            {
                "site": site_dir.name,
                "region": site_region.get(site_dir.name),
                "kinds": kinds,
                "size": sum(f["size"] or 0 for f in files),
                "files": sorted(files, key=lambda f: f["name"])[:MAX_LISTED],
            }
        )
    sdem = []
    sroot = Path(reference_dir) / "sdem"
    for zdir in sorted(p for p in sroot.glob("*") if p.is_dir()) if sroot.is_dir() else []:
        members = [Path(e.path) for e in _walk(zdir, MAX_SCAN) if not e.name.endswith(".part")]
        sdem.append(
            {
                "zip": zdir.name,
                "path": str(zdir),
                "members": sorted(p.relative_to(zdir).as_posix() for p in members)[:MAX_LISTED],
                "count": len(members),
                "size": sum(p.stat().st_size for p in members if p.exists()),
            }
        )
    return {"path": str(reference_dir), "exists": Path(reference_dir).is_dir(), "pgda": pgda, "sdem": sdem}


def runs(runs_dir: Path | None) -> dict[str, Any]:
    """Run dirs with a ``provenance.json``: last run's status, times and steps, and a report flag."""
    if runs_dir is None:
        return {"configured": False, "runs": []}
    out: list[dict[str, Any]] = []
    try:
        dirs = sorted(p for p in Path(runs_dir).iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        dirs = []
    for d in dirs[:MAX_LISTED]:
        prov = d / "provenance.json"
        if not prov.is_file():
            continue
        try:
            data = json.loads(prov.read_text())
        except (OSError, ValueError):
            out.append({"name": d.name, "path": str(d), "status": "unreadable", "steps": []})
            continue
        history = data.get("runs") or [{}]
        last = history[-1]
        steps = [
            {"name": s.get("name"), "status": s.get("status"), "seconds": s.get("seconds")}
            for s in last.get("steps") or []
        ]
        out.append(
            {
                "name": d.name,
                "path": str(d),
                "status": last.get("status"),
                "started": last.get("started"),
                "finished": last.get("finished"),
                "runs": len(history),
                "steps": steps,
                "report": (d / "report" / "report.html").is_file(),
            }
        )
    out.sort(key=lambda r: r.get("started") or "", reverse=True)
    return {"configured": True, "path": str(runs_dir), "exists": Path(runs_dir).is_dir(), "runs": out}


_ENV_LOCK = threading.Lock()
_ENV: dict[str, Any] | None = None


def environment(refresh: bool = False) -> dict[str, Any]:
    """ASP / ISIS roots, ISISDATA subsets and tool versions (``sfs --version`` etc., cached)."""
    global _ENV
    with _ENV_LOCK:
        if _ENV is None or refresh:
            _ENV = _environment()
        return _ENV


def _environment() -> dict[str, Any]:
    from ..provenance import isis_paths, kernel_files, tool_versions
    from ..tools import find_tool

    paths = isis_paths()
    isisdata = paths.get("ISISDATA")
    subsets: dict[str, bool] = {}
    if isisdata:
        root = Path(isisdata)
        for key, rel in (
            ("base kernels", "base/kernels"),
            ("base dems", "base/dems"),
            ("lro kernels", "lro/kernels"),
            ("lro calibration", "lro/calibration"),
        ):
            subsets[key] = (root / rel).is_dir()
    tools = {}
    for name in (
        "sfs",
        "parallel_sfs",
        "bundle_adjust",
        "mapproject",
        "spiceinit",
        "lronac2isis",
        "gdalinfo",
    ):
        p = find_tool(name)
        tools[name] = {"found": p is not None, "path": str(p) if p else None}
    asp_root = os.environ.get("ASP_ROOT")
    return {
        "roots": [
            {"name": "ASP_ROOT", "path": asp_root, "set": bool(asp_root)},
            {"name": "ISISROOT", "path": paths.get("ISISROOT"), "set": bool(paths.get("ISISROOT"))},
            {"name": "ISISDATA", "path": isisdata, "set": bool(isisdata)},
        ],
        "isisdata": subsets,
        "kernels": kernel_files(isisdata),
        "versions": tool_versions(),
        "tools": tools,
        "checked": time.time(),
    }


def cooldowns() -> list[dict[str, Any]]:
    """Hosts paused after a throttling reply (persisted and in this process)."""
    from .. import downloader

    now = time.time()
    hosts = dict(downloader._saved_cooldowns())
    for name, h in list(downloader._HOSTS.items()):
        if h.cooldown_until > now:
            hosts[name] = max(hosts.get(name, 0.0), h.cooldown_until)
    return [
        {
            "host": h,
            "until": t,
            "until_text": time.strftime("%H:%M:%S", time.localtime(t)),
            "seconds": t - now,
        }
        for h, t in sorted(hosts.items())
        if t > now
    ]


def disks(dirs: StatusDirs) -> list[dict[str, Any]]:
    """Free / total space of the file systems holding the data, cache and runs dirs."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for label, d in (("data", dirs.data_dir), ("cache", dirs.cache_dir), ("runs", dirs.runs_dir)):
        if d is None:
            continue
        p = Path(d)
        while not p.exists() and p != p.parent:
            p = p.parent
        try:
            dev = p.stat().st_dev
            usage = shutil.disk_usage(p)
        except OSError:
            continue
        if dev in seen:
            next(o for o in out if o["dev"] == dev)["labels"].append(label)
            continue
        seen.add(dev)
        out.append({"labels": [label], "dev": dev, "path": str(p), "free": usage.free, "total": usage.total})
    for o in out:
        del o["dev"]
    return out


def redact(obj: Any) -> Any:
    """*obj* without file system paths (``path`` / ``root`` / ... keys), for read-only clients."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if k not in _PATH_KEYS and k != "kernels"}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


class StatusCache:
    """``/api/status`` documents, rebuilt at most every `CACHE_SECONDS` (or on refresh)."""

    def __init__(self, dirs: StatusDirs) -> None:
        self.dirs = dirs
        self._lock = threading.Lock()
        self._doc: dict[str, Any] | None = None
        self._at = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._doc = None

    def get(self, admin: bool, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._doc is None or refresh or time.time() - self._at > CACHE_SECONDS:
                self._doc = self._build(refresh)
                self._at = time.time()
            doc = self._doc
        live = {**doc, "cooldowns": cooldowns(), "admin": admin}
        return live if admin else redact(live)

    def _build(self, refresh: bool) -> dict[str, Any]:
        from ..constants import APP_VERSION

        d = self.dirs
        return {
            "version": APP_VERSION,
            "data": {"path": str(d.data_dir), "exists": d.data_dir.is_dir(), "indexes": indexes(d.data_dir)},
            "cache": cache_summary(d.cache_dir),
            "downloads": downloads(d.download_dir),
            "reference": reference(d.reference_dir),
            "runs": runs(d.runs_dir),
            "environment": environment(refresh),
            "disks": disks(d),
            "built": time.time(),
        }
