"""Find and run ISIS / ASP / GDAL tools; keyword options → flags; dry runs."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TOOL_ROOT_ENV_VARS = ("ASP_ROOT", "ISISROOT")

_DRY_RUN: contextvars.ContextVar[list[list[str]] | None] = contextvars.ContextVar("dry_run", default=None)
_COMMAND_LOG: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "command_log", default=None
)


class ToolError(RuntimeError):
    """An external tool is missing or exited with a non-zero status."""

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def find_tool(name: str) -> Path | None:
    """Path of executable *name*, or None when it cannot be found."""
    for var in TOOL_ROOT_ENV_VARS:
        root = os.environ.get(var)
        if root:
            p = Path(root) / "bin" / name
            if p.is_file() and os.access(p, os.X_OK):
                return p
    found = shutil.which(name)
    return Path(found) if found else None


def tool_env(tool_path: Path) -> dict[str, str]:
    """Environment for *tool_path* as a conda activation would set it.

    Unset variables come from the env's ``conda env config vars`` (stored in
    ``conda-meta/state``), else ``ISISROOT``, ``ISISDATA``, ``PROJ_DATA`` and
    ``GDAL_DATA`` default to the tool's environment prefix, so tools also run
    from an env that was not activated (ASP and GDAL fail or crash without them).
    """
    env = dict(os.environ)
    prefix = Path(tool_path).parent.parent
    for name, setting in _conda_env_vars(prefix).items():
        env.setdefault(name, setting)
    defaults = {
        "ISISROOT": (prefix, prefix / "IsisPreferences"),
        "ISISDATA": (prefix / "data", prefix / "IsisPreferences"),
        "PROJ_DATA": (prefix / "share" / "proj", prefix / "share" / "proj" / "proj.db"),
        "GDAL_DATA": (prefix / "share" / "gdal", prefix / "share" / "gdal"),
    }
    for var, (value, marker) in defaults.items():
        if not env.get(var) and marker.exists():
            env[var] = str(value)
    # ASP's Python wrappers (mapproject, parallel_*) run `env python` and
    # call sibling tools, so the tool's bin dir must lead PATH.
    bin_dir = str(Path(tool_path).parent)
    path = env.get("PATH", "")
    if path.split(os.pathsep)[0] != bin_dir:
        env["PATH"] = os.pathsep.join([bin_dir, path]) if path else bin_dir
    return env


def _conda_env_vars(prefix: Path) -> dict[str, str]:
    try:
        state = json.loads((prefix / "conda-meta" / "state").read_text())
    except (OSError, ValueError):
        return {}
    env_vars = state.get("env_vars") if isinstance(state, dict) else None
    return {str(k): str(v) for k, v in env_vars.items()} if isinstance(env_vars, dict) else {}


def format_command(argv: Sequence[str]) -> str:
    """*argv* as one command line, quoted for the platform's shell."""
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


@contextlib.contextmanager
def dry_run() -> Iterator[list[list[str]]]:
    """Record commands instead of running them.

    Yields the list the commands are appended to (tool name first)::

        with dry_run() as cmds:
            pipeline.run(...)
        for c in cmds:
            print(format_command(c))
    """
    cmds: list[list[str]] = []
    token = _DRY_RUN.set(cmds)
    try:
        yield cmds
    finally:
        _DRY_RUN.reset(token)


@contextlib.contextmanager
def record_commands() -> Iterator[list[dict[str, Any]]]:
    """Log the commands :func:`run` executes: ``argv``, ``cwd``, ``started`` (Unix time), ``seconds``, ``returncode``."""
    log_: list[dict[str, Any]] = []
    token = _COMMAND_LOG.set(log_)
    try:
        yield log_
    finally:
        _COMMAND_LOG.reset(token)


def is_dry_run() -> bool:
    return _DRY_RUN.get() is not None


def reuse_existing(path: Path, force: bool = False) -> bool:
    """Whether a step can be skipped because its output *path* already exists.

    Parameters
    ----------
    path : Path
        The step's output.
    force : bool
        Never reuse; the step is always redone.

    Returns
    -------
    bool
        True (and logged) when *path* exists and *force* is False.
    """
    if Path(path).exists() and not force:
        log.info("Reusing %s", path)
        return True
    return False


def run(
    tool: str,
    args: Sequence[Any] = (),
    *,
    cwd: Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run *tool* with *args*; raise :class:`ToolError` on failure.

    With ``capture=True`` stdout/stderr are returned instead of streamed.
    """
    argv = [str(a) for a in args]
    recorded = _DRY_RUN.get()
    if recorded is not None:
        recorded.append([tool, *argv])
        log.info("[dry-run] %s", format_command([tool, *argv]))
        return subprocess.CompletedProcess([tool, *argv], 0, "", "")

    path = find_tool(tool)
    if path is None:
        raise ToolError(f"{tool} not found; activate the ISIS / ASP environment or set ASP_ROOT / ISISROOT")
    log.info("▶  %s", tool)
    log.debug("   cmd: %s", format_command([str(path), *argv]))
    start = time.time()
    result = subprocess.run(
        [str(path), *argv], cwd=cwd, env=tool_env(path), text=True, capture_output=capture
    )
    commands = _COMMAND_LOG.get()
    if commands is not None:
        commands.append(
            {
                "argv": [str(path), *argv],
                "cwd": str(cwd) if cwd else None,
                "started": start,
                "seconds": round(time.time() - start, 3),
                "returncode": result.returncode,
            }
        )
    if check and result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        detail = ("\n" + "\n".join(tail)) if capture and tail else " Check output above."
        raise ToolError(
            f"{tool} failed (exit {result.returncode}).{detail}",
            result.returncode,
            result.stdout,
            result.stderr,
        )
    log.info("✔  %s complete", tool)
    return result


def asp_options(options: Mapping[str, Any]) -> list[str]:
    """Keyword options → ASP-style flags.

    ``tr=5`` → ``--tr 5``; ``save_sparingly=True`` → ``--save-sparingly``;
    ``False`` / ``None`` are dropped; tuples and lists expand to several
    arguments (``t_projwin=(x0, y0, x1, y1)``) while a string with spaces stays
    one argument (``remove_outliers_params="75 3 100 100"``).  Underscores
    become dashes except in GDAL-style ``t_*`` options (``--t_srs``); single
    letters get one dash (``o="run/run"`` → ``-o run/run``).
    """
    args: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        name = key if key.startswith("t_") else key.replace("_", "-")
        flag = f"-{name}" if len(name) == 1 else f"--{name}"
        args += [flag, *_values(value)]
    return args


def gdal_options(options: Mapping[str, Any]) -> list[str]:
    """Keyword options → GDAL-style flags (``a_nodata=0`` → ``-a_nodata 0``)."""
    args: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        args += [f"-{key}", *_values(value)]
    return args


def isis_params(params: Mapping[str, Any]) -> list[str]:
    """Keyword parameters → ISIS ``key=value`` arguments.

    A trailing underscore is dropped (``from_=a.cub`` → ``from=a.cub``),
    booleans become ``true`` / ``false`` and ``None`` is skipped.
    """
    args: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        args.append(f"{key.rstrip('_')}={value}")
    return args


def _values(value: Any) -> list[str]:
    if value is True:
        return []
    if isinstance(value, (list, tuple)):
        return [_fmt(v) for v in value]
    return [_fmt(value)]


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
