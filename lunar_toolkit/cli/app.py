"""app - the web app in a desktop window (pywebview, else a Chromium --app window, else a browser tab)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, cast

COMMAND = "app"
TITLE = "Lunar Toolkit"
SIZE = (1440, 900)
MIN_SIZE = (900, 600)
BACKENDS = ("auto", "webview", "chromium", "browser")

_CHROMIUM_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
    "brave-browser",
)
_MAC_APPS = ("Google Chrome", "Chromium", "Microsoft Edge", "Brave Browser")
_WIN_PATHS = (
    ("Google", "Chrome", "Application", "chrome.exe"),
    ("Microsoft", "Edge", "Application", "msedge.exe"),
    ("Chromium", "Application", "chrome.exe"),
    ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
)


def add_parser(sub: argparse._SubParsersAction) -> None:
    from .serve import add_server_args

    p = sub.add_parser(
        COMMAND,
        help="Open the Lunar Toolkit web app in a desktop window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar app
  lunar app --images ~/lunar_runs
  lunar app --host 0.0.0.0 --port 8080        # window here, read-only page for the network
  lunar app --backend browser                 # a normal browser tab instead

The server runs on this machine (a free port on 127.0.0.1 by default) and stops when
the window closes. Window backends, first available wins (--backend auto):
  webview   pywebview: Qt WebEngine on Linux (pip install "lunar-toolkit[app]",
            or PyGObject + WebKitGTK), Edge WebView2 on Windows, WebKit on macOS
  chromium  Chrome / Chromium / Edge / Brave in --app mode (own profile in the cache dir)
  browser   the default browser (Ctrl-C stops the server)
""",
    )
    add_server_args(p, "127.0.0.1", 0)
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="Window backend (default: auto, in the order above).",
    )


def webview_gui(platform: str | None = None) -> str | None:
    """The pywebview GUI to use here (``qt``, ``gtk``, ``cocoa``, ``edgechromium``), or None."""
    platform = platform or sys.platform
    if not _has("webview"):
        return None
    if platform == "darwin":
        return "cocoa" if _has("WebKit") else None
    if platform == "win32":
        return "edgechromium" if _has("clr") else None
    if _has("qtpy") and (_has("PyQt6.QtWebEngineWidgets") or _has("PySide6.QtWebEngineWidgets")):
        return "qt"
    if _has("gi") and _gtk_webkit():
        return "gtk"
    return None


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _gtk_webkit() -> bool:
    try:
        gi = importlib.import_module("gi")
        for version in ("4.1", "4.0"):
            try:
                gi.require_version("WebKit2", version)
                return True
            except ValueError:
                continue
    except ImportError:
        pass
    return False


def chromium_binary(platform: str | None = None) -> str | None:
    """A Chrome / Chromium / Edge / Brave executable, or None."""
    platform = platform or sys.platform
    if platform == "darwin":
        for app in _MAC_APPS:
            p = Path("/Applications") / f"{app}.app" / "Contents" / "MacOS" / app
            if p.is_file():
                return str(p)
    elif platform == "win32":
        roots = [os.environ.get(v) for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
        for root in filter(None, roots):
            for parts in _WIN_PATHS:
                p = Path(root, *parts)
                if p.is_file():
                    return str(p)
    for name in _CHROMIUM_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


INSTALL_HINT = 'pip install "lunar-toolkit[app]"'


def choose_backend(requested: str = "auto", platform: str | None = None) -> tuple[str, str | None]:
    """``(backend, detail)``: the pywebview GUI name, the Chromium binary, or None for the browser."""
    if requested in ("auto", "webview"):
        gui = webview_gui(platform)
        if gui:
            return "webview", gui
    if requested in ("auto", "chromium"):
        binary = chromium_binary(platform)
        if binary:
            return "chromium", binary
    return "browser", None


def _run_webview(url: str, gui: str) -> None:
    import webview

    from ..constants import DEFAULT_CACHE_DIR

    webview.create_window(TITLE, url, width=SIZE[0], height=SIZE[1], min_size=MIN_SIZE)
    storage = DEFAULT_CACHE_DIR / "app-profile" / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(gui=cast(Any, gui), private_mode=False, storage_path=str(storage))


def _run_chromium(url: str, binary: str) -> None:
    from ..constants import DEFAULT_CACHE_DIR

    profile = DEFAULT_CACHE_DIR / "app-profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        f"--app={url}",
        f"--user-data-dir={profile}",
        f"--window-size={SIZE[0]},{SIZE[1]}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    subprocess.run(cmd, check=False)


def run(args: argparse.Namespace) -> int:
    from .serve import make_server, print_banner

    backend, detail = choose_backend(args.backend)
    if args.backend == "webview" and backend != "webview":
        print(
            f"No native window toolkit for pywebview here; {INSTALL_HINT}, or use --backend chromium / browser."
        )
        return 1
    try:
        server, cfg = make_server(args)
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    url = print_banner(server, cfg, args.host, TITLE)
    thread = threading.Thread(target=server.serve_forever, name="lunar-server", daemon=True)
    thread.start()
    if backend != "webview" and args.backend == "auto":
        print(f"  note     : no native window toolkit; {INSTALL_HINT} for one")
    try:
        if backend == "webview" and detail:
            print(f"  window   : pywebview ({detail})")
            try:
                _run_webview(url, detail)
                return 0
            except Exception as exc:  # no display, broken GUI toolkit, ...
                print(f"  pywebview failed ({exc}); trying a Chromium window")
                backend, detail = choose_backend("chromium")
        if backend == "chromium" and detail:
            print(f"  window   : {Path(detail).name} --app")
            _run_chromium(url, detail)
            return 0
        print("  window   : no Chrome / Chromium / Edge either; opening a browser tab")
        print("  Ctrl-C to stop.\n")
        webbrowser.open(url)
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.close()
    return 0
