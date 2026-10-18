"""Run provenance for SfS work dirs (``provenance.json``): environment, inputs, steps and checks."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tools import find_tool, is_dry_run, record_commands, run, tool_env

if TYPE_CHECKING:  # pragma: no cover
    from .config import RunConfig

SCHEMA_VERSION = 1
FILE_NAME = "provenance.json"
MD5_MAX_BYTES = 2 * 1024**3

# (key, tool, args): first output line is the version.
VERSION_COMMANDS = (
    ("asp", "sfs", ["--version"]),
    ("ale", "isd_generate", ["--version"]),
    ("gdal", "gdalinfo", ["--version"]),
)
KERNEL_GLOBS = {
    "lsk": "base/kernels/lsk/naif*.tls",
    "lro_sclk": "lro/kernels/sclk/lro_clkcor_*.tsc",
}


def now() -> str:
    """UTC time as ISO 8601 (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _first_line(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else None


def tool_versions() -> dict[str, str | None]:
    """ASP, ALE, GDAL and ISIS versions (None when a tool is missing); empty under a dry run."""
    if is_dry_run():
        return {}
    out: dict[str, str | None] = {}
    for key, tool, args in VERSION_COMMANDS:
        if find_tool(tool) is None:
            out[key] = None
            continue
        res = run(tool, args, capture=True, check=False)
        out[key] = _first_line(f"{res.stdout or ''}\n{res.stderr or ''}")
    out["isis"] = isis_version(isis_paths().get("ISISROOT"))
    return out


def isis_paths() -> dict[str, str | None]:
    """``ISISROOT`` / ``ISISDATA`` as the tools see them (:func:`~lunar_toolkit.tools.tool_env`)."""
    tool = find_tool("spiceinit") or find_tool("sfs")
    env = tool_env(tool) if tool else dict(os.environ)
    return {k: env.get(k) for k in ("ISISROOT", "ISISDATA")}


def isis_version(isisroot: str | None) -> str | None:
    """First field of ``$ISISROOT/isis_version.txt`` (e.g. ``10.0.0``)."""
    if not isisroot:
        return None
    try:
        text = (Path(isisroot) / "isis_version.txt").read_text()
    except OSError:
        return None
    first = _first_line(text)
    return first.split("#")[0].strip() if first else None


def kernel_files(isisdata: str | None) -> dict[str, list[str]]:
    """Names of the leap-second and LRO clock kernels under *isisdata*."""
    if not isisdata:
        return {}
    root = Path(isisdata)
    return {k: sorted(p.name for p in root.glob(pattern)) for k, pattern in KERNEL_GLOBS.items()}


def environment() -> dict[str, Any]:
    """Package, Python, platform, tool versions, ISIS paths and kernels."""
    from .constants import APP_VERSION

    paths = isis_paths()
    return {
        "lunar_toolkit": APP_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "tools": tool_versions(),
        "isis_paths": paths,
        "kernels": kernel_files(paths.get("ISISDATA")),
    }


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """MD5 of *path*, read in *chunk*-byte pieces."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_record(
    path: Path, checksum: bool = True, known: dict[str, Any] | None = None, max_bytes: int = MD5_MAX_BYTES
) -> dict[str, Any]:
    """``path``, ``size``, ``mtime`` and ``md5`` (None if skipped) of one input file.

    The MD5 of *known* (an earlier record of the same file) is reused when size
    and mtime match, so resumed runs do not hash again.
    """
    p = Path(path)
    rec: dict[str, Any] = {"path": str(p), "size": None, "mtime": None, "md5": None}
    try:
        st = p.stat()
    except OSError:
        return rec
    rec.update(size=st.st_size, mtime=round(st.st_mtime, 3))
    if known and known.get("size") == rec["size"] and known.get("mtime") == rec["mtime"]:
        rec["md5"] = known.get("md5")
    elif checksum and p.is_file() and st.st_size <= max_bytes:
        rec["md5"] = md5sum(p)
    return rec


def read(path: Path) -> dict[str, Any]:
    """A ``provenance.json`` (an empty history when missing)."""
    p = Path(path)
    if not p.exists():
        return {"schema": SCHEMA_VERSION, "runs": []}
    return json.loads(p.read_text())


def _output_paths(result: Any) -> list[str]:
    if isinstance(result, Path):
        return [str(result)] if result.exists() else []
    if isinstance(result, dict):
        return _output_paths(list(result.values()))
    if isinstance(result, (list, tuple)):
        return [p for r in result for p in _output_paths(r)]
    path = getattr(result, "mapprojected", None) or getattr(result, "cube", None)
    return [str(path)] if isinstance(path, Path) and path.exists() else []


class Provenance:
    """Append one run (environment, config, inputs, steps) to ``<work_dir>/provenance.json``.

    Earlier runs are kept, so a resumed work dir holds its whole history.  The
    file is rewritten (atomically) after each step.
    """

    def __init__(self, work_dir: Path, config: RunConfig, inputs: Iterable[Path], checksums: bool = True):
        from .config import config_hash, to_dict

        self.path = Path(work_dir) / FILE_NAME
        self.data = read(self.path)
        known = {r["path"]: r for r in last_run(self.data).get("inputs", [])}
        self.run: dict[str, Any] = {
            "started": now(),
            "finished": None,
            "status": "running",
            **environment(),
            "config_sha256": config_hash(config),
            "config": to_dict(config),
            "inputs": [file_record(p, checksums, known.get(str(p))) for p in inputs],
            "steps": [],
        }
        self.data.setdefault("runs", []).append(self.run)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[dict[str, Any]]:
        """Record a step: times, the commands :func:`~lunar_toolkit.tools.run` executes and status.

        Set ``entry["result"]`` inside the block to list its existing output files.
        """
        entry: dict[str, Any] = {"name": name, "started": now(), "finished": None, "status": "running"}
        self.run["steps"].append(entry)
        self.save()
        t0 = time.time()
        with record_commands() as cmds:
            try:
                yield entry
            except BaseException as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            else:
                entry["status"] = "ok"
            finally:
                entry.update(
                    finished=now(),
                    seconds=round(time.time() - t0, 3),
                    commands=cmds,
                    outputs=_output_paths(entry.pop("result", None)),
                )
                self.save()

    def finish(self, status: str = "ok") -> None:
        self.run.update(finished=now(), status=status)
        self.save()


def last_run(data: dict[str, Any]) -> dict[str, Any]:
    runs = data.get("runs") or []
    return runs[-1] if runs else {}


@dataclass(frozen=True)
class Difference:
    """One mismatch between a recorded run and now: *kind* (``environment``, ``config``, ``input``)."""

    kind: str
    key: str
    recorded: Any
    current: Any

    def __str__(self) -> str:
        return f"{self.kind:11s} {self.key}: {self.recorded!r} -> {self.current!r}"


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def compare(
    recorded: dict[str, Any],
    current_env: dict[str, Any],
    current_config: dict[str, Any] | None = None,
    inputs: Sequence[Path] | None = None,
    checksums: bool = True,
) -> list[Difference]:
    """Differences between a recorded run and the current environment, config and inputs.

    *inputs* default to the recorded input paths; a file whose size or mtime
    changed is hashed again and reported only when its MD5 differs (or cannot be
    compared).
    """
    diffs: list[Difference] = []
    keys = ("lunar_toolkit", "python", "tools", "isis_paths", "kernels")
    rec_env = _flatten({k: recorded.get(k) for k in keys})
    cur_env = _flatten({k: current_env.get(k) for k in keys})
    for key in sorted(set(rec_env) | set(cur_env)):
        if rec_env.get(key) != cur_env.get(key):
            diffs.append(Difference("environment", key, rec_env.get(key), cur_env.get(key)))
    if current_config is not None:
        rec_cfg = _flatten(recorded.get("config") or {})
        cur_cfg = _flatten(current_config)
        for key in sorted(set(rec_cfg) | set(cur_cfg)):
            if rec_cfg.get(key) != cur_cfg.get(key):
                diffs.append(Difference("config", key, rec_cfg.get(key), cur_cfg.get(key)))
    known = {r["path"]: r for r in recorded.get("inputs", [])}
    paths = [str(p) for p in inputs] if inputs is not None else list(known)
    for p in paths:
        old = known.get(p)
        if old is None:
            diffs.append(Difference("input", p, None, "added"))
            continue
        new = file_record(Path(p), checksum=False)
        if new["size"] is None:
            diffs.append(Difference("input", p, old.get("md5") or old.get("size"), "missing"))
        elif (new["size"], new["mtime"]) != (old.get("size"), old.get("mtime")):
            md5 = md5sum(Path(p)) if checksums and old.get("md5") and new["size"] <= MD5_MAX_BYTES else None
            if md5 is None or md5 != old.get("md5"):
                diffs.append(Difference("input", p, old.get("md5") or old.get("size"), md5 or new["size"]))
    for p in sorted(set(known) - set(paths)):
        diffs.append(Difference("input", p, "used", "not an input now"))
    return diffs


def check(
    work_dir: Path,
    config: RunConfig | None = None,
    inputs: Sequence[Path] | None = None,
    checksums: bool = True,
) -> list[Difference]:
    """Compare the last run in ``<work_dir>/provenance.json`` with the current environment.

    Raises
    ------
    FileNotFoundError
        When the work dir has no provenance yet.
    """
    from .config import to_dict

    path = Path(work_dir) / FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {FILE_NAME} in {work_dir}")
    recorded = last_run(read(path))
    return compare(recorded, environment(), to_dict(config) if config else None, inputs, checksums)
