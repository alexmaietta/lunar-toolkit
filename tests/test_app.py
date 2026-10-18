"""lunar serve's admin / read-only split, jobs (progress, SSE, cancel), status, relay throttle; lunar app backends."""

import argparse
import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lunar_toolkit import downloader
from lunar_toolkit.cli import _jobs, _status, app, serve

TOKEN = "s3cret-token-for-tests"


@pytest.fixture
def server(tmp_path, built_db):
    """An in-process server on 127.0.0.1 (random port) over private data / cache / runs dirs."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "cumindex_edr.parquet").write_bytes((built_db / "cumindex_edr.parquet").read_bytes())
    (data / "cumindex_edr.meta.json").write_text(
        json.dumps({"rows": 200, "release": "LROLRC_0066C", "source": "cumindex", "built_at": "2026-01-01"})
    )
    cache = tmp_path / "cache"
    (cache / "downloads").mkdir(parents=True)
    (cache / "downloads" / "M1LE.IMG").write_bytes(b"x" * 10)
    (cache / "downloads" / "M2LE.IMG.part").write_bytes(b"x" * 3)
    (cache / "reference" / "pgda" / "Site23").mkdir(parents=True)
    (cache / "reference" / "pgda" / "Site23" / "Site23_final_adj_5mpp_surf.tif").write_bytes(b"II*\0")
    runs = tmp_path / "runs"
    (runs / "malapert" / "report").mkdir(parents=True)
    (runs / "malapert" / "report" / "report.html").write_text("<p>report</p>")
    (runs / "malapert" / "provenance.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "started": "2026-10-18T10:00:00+00:00",
                        "finished": None,
                        "status": "running",
                        "steps": [{"name": "prepare_images", "status": "running"}],
                    }
                ]
            }
        )
    )
    cfg = serve.ServerConfig(
        data_dir=data,
        runs_dir=runs,
        admin_token=TOKEN,
        cache_dir=cache,
        reference_dir=cache / "reference",
    )
    srv = serve.LunarServer(("127.0.0.1", 0), cfg)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.close()


@pytest.fixture
def remote(monkeypatch):
    """Requests look like they come from another machine."""
    monkeypatch.setattr(serve.LunarHandler, "client_ip", lambda self: "203.0.113.7")


def _req(srv, path, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(srv.base + path, data=data, method=method, headers={**h, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _json(srv, path, **kw):
    code, _, raw = _req(srv, path, **kw)
    return code, json.loads(raw) if raw else None


def _fake_job(gate: threading.Event, chunks: int = 5):
    def run(job):
        job.set_file(1, 1, "a.IMG")
        for i in range(1, chunks + 1):
            gate.wait(5)
            job.on_bytes("a.IMG", i * 100, chunks * 100)
            time.sleep(0.01)
        return {"ok": True}

    return run


def test_config_reports_admin_for_loopback_only(server, remote):
    assert _json(server, "/api/config")[1]["admin"] is False
    assert _json(server, "/api/config", headers={"Authorization": f"Bearer {TOKEN}"})[1]["admin"] is True
    assert _json(server, f"/api/config?token={TOKEN}")[1]["admin"] is True
    assert _json(server, "/api/config?token=wrong")[1]["admin"] is False


def test_loopback_is_admin_but_needs_a_local_host_header(server):
    assert _json(server, "/api/config")[1]["admin"] is True
    assert _json(server, "/api/config", headers={"Host": "evil.example:80"})[1]["admin"] is False


def test_remote_clients_cannot_start_or_cancel_jobs(server, remote):
    code, body = _json(server, "/api/jobs", method="POST", body={"kind": "build-db", "dataset": "edr"})
    assert code == 403 and "read-only" in body["error"]
    assert _req(server, "/api/jobs/1", method="DELETE")[0] == 403
    assert _req(server, "/api/sdem/members?region=11")[0] == 403
    assert _req(server, "/runs/malapert/report.html")[0] == 403
    # existing read-only features still work
    assert _req(server, "/")[0] == 200
    assert _req(server, "/data/cumindex_edr.parquet", headers={"Range": "bytes=0-3"})[2] == b"PAR1"


def test_token_header_allows_remote_admin(server, remote, monkeypatch):
    gate = threading.Event()
    monkeypatch.setitem(_jobs.KINDS, "fake", lambda p, ctx: ("Fake", "fake", _fake_job(gate)))
    code, job = _json(
        server,
        "/api/jobs",
        method="POST",
        body={"kind": "fake"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    gate.set()
    assert code == 201 and job["kind"] == "fake"
    assert _req(server, f"/runs/malapert/report.html?token={TOKEN}")[2] == b"<p>report</p>"


def test_post_needs_json_and_same_origin(server):
    code, _, _ = _req(server, "/api/jobs", method="POST", body=None, headers={"Content-Type": "text/plain"})
    assert code == 415
    code, body = _json(
        server, "/api/jobs", method="POST", body={"kind": "pgda"}, headers={"Origin": "http://evil.example"}
    )
    assert code == 403 and "cross-origin" in body["error"]
    code, body = _json(server, "/api/jobs", method="POST", body={"kind": "nope"})
    assert code == 400 and "kind" in body["error"]
    code, body = _json(server, "/api/jobs", method="POST", body={"kind": "build-db", "dataset": "xyz"})
    assert code == 400
    code, body = _json(server, "/api/jobs", method="POST", body={"kind": "download", "ids": ["../etc"]})
    assert code == 400 and "product IDs" in body["error"]
    assert _req(server, "/api/jobs", method="OPTIONS")[0] == 405


def test_status_shape_for_admin(server):
    code, st = _json(server, "/api/status")
    assert code == 200 and st["admin"] is True
    edr = next(i for i in st["data"]["indexes"] if i["dataset"] == "edr")
    assert edr["exists"] and edr["rows"] == 200 and edr["release"] == "LROLRC_0066C" and edr["size"] > 0
    assert not next(i for i in st["data"]["indexes"] if i["dataset"] == "cdr")["exists"]
    assert st["data"]["path"].endswith("data")
    files = {f["name"]: f for f in st["downloads"]["files"]}
    assert files["M1LE.IMG"]["size"] == 10 and files["M2LE.IMG.part"]["partial"]
    assert st["reference"]["pgda"][0]["site"] == "Site23"
    assert st["reference"]["pgda"][0]["kinds"] == {"surf": 1}
    run = st["runs"]["runs"][0]
    assert run["name"] == "malapert" and run["status"] == "running" and run["report"]
    assert run["steps"] == [{"name": "prepare_images", "status": "running", "seconds": None}]
    assert {"roots", "versions", "isisdata", "tools"} <= set(st["environment"])
    assert isinstance(st["cooldowns"], list) and st["disks"][0]["free"] > 0


def test_no_paths_for_read_only_clients(server, remote, tmp_path):
    downloader._host("https://pds.mcp.nasa.gov/x").cool(600)
    code, _, raw = _req(server, "/api/status")
    assert code == 200
    text = raw.decode()
    assert str(tmp_path) not in text and str(Path.home()) not in text
    st = json.loads(text)
    assert st["admin"] is False and st["data"]["indexes"][0]["rows"] == 200
    assert st["cooldowns"][0]["host"] == "pds.mcp.nasa.gov"
    assert "path" not in st["data"] and all("path" not in r for r in st["environment"]["roots"])


def test_redact_drops_path_keys():
    doc = {"path": "/x", "a": [{"path": "/y", "size": 1}], "root": "/r", "ok": {"name": "n"}}
    assert _status.redact(doc) == {"a": [{"size": 1}], "ok": {"name": "n"}}


def test_job_lifecycle_progress_and_sse(server, monkeypatch):
    gate = threading.Event()
    monkeypatch.setitem(_jobs.KINDS, "fake", lambda p, ctx: ("Fake job", "fake", _fake_job(gate)))
    code, job = _json(server, "/api/jobs", method="POST", body={"kind": "fake", "x": 1})
    assert code == 201 and job["status"] in ("queued", "running") and job["params"] == {"x": 1}
    events = []

    def listen():
        with urllib.request.urlopen(server.base + f"/api/jobs/{job['id']}/events", timeout=10) as resp:
            assert resp.headers["Content-Type"] == "text/event-stream"
            for line in resp:
                events.append(line.decode())

    t = threading.Thread(target=listen)
    t.start()
    gate.set()
    t.join(10)
    blocks = "".join(events).split("\n\n")
    data = [json.loads(b.split("data: ", 1)[1]) for b in blocks if b.startswith("event: job\n")]
    assert data and data[-1]["status"] == "done" and data[-1]["progress"] == 1.0
    assert (
        data[-1]["bytes_total"] == 500 and data[-1]["file"] == "a.IMG" and data[-1]["result"] == {"ok": True}
    )
    listing = _json(server, "/api/jobs")[1]
    assert [j["id"] for j in listing] == [job["id"]]
    # a second identical job may start once the first finished
    assert _json(server, "/api/jobs", method="POST", body={"kind": "fake"})[0] == 201


def test_jobs_stream_lists_all_jobs(server):
    lines = []
    with urllib.request.urlopen(server.base + "/api/jobs/events", timeout=10) as resp:
        lines.append(resp.readline().decode())
        lines.append(resp.readline().decode())
    assert lines == ["event: jobs\n", "data: []\n"]


def test_cancel_is_cooperative(server, monkeypatch):
    started = threading.Event()

    def run(job):
        job.set_file(1, 1, "big.IMG")
        started.set()
        for i in range(500):
            job.on_bytes("big.IMG", i, 500)
            time.sleep(0.01)
        return "finished"

    monkeypatch.setitem(_jobs.KINDS, "fake", lambda p, ctx: ("Slow", "slow", run))
    job = _json(server, "/api/jobs", method="POST", body={"kind": "fake"})[1]
    assert started.wait(5)
    code, _ = _json(server, "/api/jobs", method="POST", body={"kind": "fake"})
    assert code == 409  # same key still running
    code, body = _json(server, f"/api/jobs/{job['id']}", method="DELETE")
    assert code == 200
    for _ in range(100):
        if _json(server, f"/api/jobs/{job['id']}")[1]["status"] == "cancelled":
            break
        time.sleep(0.05)
    final = _json(server, f"/api/jobs/{job['id']}")[1]
    assert final["status"] == "cancelled" and final["result"] is None
    assert _req(server, "/api/jobs/999", method="DELETE")[0] == 404


def test_read_only_job_listing_hides_params_and_logs(server, remote, monkeypatch):
    gate = threading.Event()
    gate.set()
    monkeypatch.setitem(_jobs.KINDS, "fake", lambda p, ctx: ("Fake", "fake", _fake_job(gate)))
    _json(server, "/api/jobs", method="POST", body={"kind": "fake", "dest": "/secret/dir"},
          headers={"Authorization": f"Bearer {TOKEN}"})  # fmt: skip
    jobs = _json(server, "/api/jobs")[1]
    assert jobs and all("params" not in j and "log" not in j for j in jobs)


def test_download_progress_callback_and_cancel(tmp_path, monkeypatch):
    payload = b"z" * (3 << 20)

    class Resp(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda req, timeout=None: Resp(payload))
    seen = []
    downloader.download_file(
        "https://h.example/a.IMG", tmp_path / "a.IMG", show_progress=False, progress=lambda *a: seen.append(a)
    )
    assert seen[0] == ("a.IMG", 0, len(payload)) and seen[-1] == ("a.IMG", len(payload), len(payload))

    def stop(name, done, total):
        if done > 0:
            raise _jobs.Cancelled()

    with downloader.report_progress(stop), pytest.raises(_jobs.Cancelled):
        downloader.download_file("https://h.example/b.IMG", tmp_path / "b.IMG", show_progress=False)
    assert (tmp_path / "b.IMG.part").stat().st_size == 1 << 20  # kept for resume
    assert not (tmp_path / "b.IMG").exists()


def test_host_wait_polls_and_can_be_aborted(monkeypatch):
    h = downloader._host("https://slow.example/x")
    h.cool(30)
    calls = []

    def poll():
        calls.append(1)
        if len(calls) == 3:
            raise _jobs.Cancelled()

    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)
    with pytest.raises(_jobs.Cancelled):
        h.wait(poll)
    assert len(calls) == 3


class _Upstream(io.BytesIO):
    def __init__(self, data, status=206, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data)), "Content-Type": "image/tiff"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


PYR = "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/x/M1_pyr.tif"


def test_relay_goes_through_the_host_throttle(server, monkeypatch):
    seen = []

    class Opener:
        def open(self, req, timeout=None):
            seen.append((req.full_url, req.get_header("Range")))
            return _Upstream(b"II*\0")

    paced = []
    host = downloader._host(PYR)
    monkeypatch.setattr(host, "pace", lambda: paced.append(1))
    monkeypatch.setattr(serve, "_relay_opener", Opener())
    code, _, body = _req(server, f"/remote?url={PYR}", headers={"Range": "bytes=0-3"})
    assert code == 206 and body == b"II*\0"
    assert seen == [(PYR, "bytes=0-3")] and paced == [1]
    assert host.slots.acquire(blocking=False)  # the slot was released
    host.slots.release()


def test_relay_propagates_throttling(server, monkeypatch):
    class Opener:
        def open(self, req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {"Retry-After": "120"}, None)

    monkeypatch.setattr(serve, "_relay_opener", Opener())
    code, headers, _ = _req(server, f"/remote?url={PYR}")
    assert code == 429 and int(headers["Retry-After"]) == 120
    host = downloader._host(PYR)
    assert host.cooldown_until > time.time() + 100
    # while the host is paused the relay answers at once, without going upstream
    monkeypatch.setattr(serve, "_relay_opener", None)
    code, headers, _ = _req(server, f"/remote?url={PYR}")
    assert code == 429 and 100 < int(headers["Retry-After"]) <= 120


def test_relay_concurrency_is_capped(server, monkeypatch):
    monkeypatch.setattr(serve, "RELAY_WAIT", 0.05)
    for _ in range(serve.RELAY_MAX):
        server.relay_slots.acquire()
    try:
        code, headers, _ = _req(server, f"/remote?url={PYR}")
    finally:
        for _ in range(serve.RELAY_MAX):
            server.relay_slots.release()
    assert code == 503 and headers["Retry-After"] == "2"


def test_build_db_job_reports_stages(tmp_path, monkeypatch):
    stages = []

    class FakeIndex:
        def __init__(self, dataset, data_dir=None):
            self.dataset = dataset

        def build_db(self, force, release, volume_only, on_stage):
            stages.append((force, release, volume_only))
            on_stage("download")
            downloader._PROGRESS.get()("INDEX.TAB", 50, 100)
            on_stage("build")

        def build_info(self):
            return {"rows": 7, "release": "LROLRC_0066C"}

    monkeypatch.setattr("lunar_toolkit.lroc.cumindex.CumIndex", FakeIndex)
    mgr = _jobs.JobManager(1)
    ctx = _jobs.JobContext(tmp_path, tmp_path / "dl", tmp_path / "ref")
    job = _jobs.submit(
        mgr, {"kind": "build-db", "dataset": "edr", "release": "LROLRC_0066C", "volume_only": True}, ctx
    )
    for _ in range(200):
        if job.status in _jobs.TERMINAL:
            break
        time.sleep(0.01)
    mgr.close()
    assert job.status == "done", job.error
    assert stages == [(False, "LROLRC_0066C", True)]
    assert job.result == {"dataset": "edr", "rows": 7, "release": "LROLRC_0066C"}
    assert any("Parquet" in line for line in job.log) and not job.cancellable
    with pytest.raises(ValueError):
        _jobs.submit(mgr, {"kind": "build-db", "dataset": "edr", "release": "bad"}, ctx)


def test_report_path_stays_inside_runs_dir(tmp_path):
    (tmp_path / "r" / "report").mkdir(parents=True)
    (tmp_path / "r" / "report" / "report.html").write_text("x")
    assert serve._report_path(tmp_path, "r") == (tmp_path / "r" / "report" / "report.html").resolve()
    for bad in ("", "..", "../r", ".hidden", "r/x", "missing", "%2E%2E"):
        assert serve._report_path(tmp_path, bad) is None, bad


def test_loopback_and_host_header_helpers():
    for ip in ("127.0.0.1", "127.8.9.1", "::1", "::ffff:127.0.0.1"):
        assert serve.is_loopback(ip), ip
    for ip in ("10.0.0.2", "203.0.113.7", "::ffff:10.0.0.1", "garbage", ""):
        assert not serve.is_loopback(ip), ip
    for host in ("localhost:8080", "127.0.0.1:1", "[::1]:9", "app.localhost"):
        assert serve._local_host_header(host), host
    for host in ("evil.example", "192.168.1.5:8080", "localhost.evil.example", ""):
        assert not serve._local_host_header(host), host


def test_admin_token_auto_and_minimum_length(tmp_path, monkeypatch):
    args = argparse.Namespace(
        data_dir=str(tmp_path), images=None, runs_dir=None, admin_token="auto", host="127.0.0.1", port=0
    )
    srv, cfg = serve.make_server(args)
    srv.server_close()
    srv.jobs.close()
    assert cfg.admin_token and len(cfg.admin_token) >= 20
    args.admin_token = "short"
    with pytest.raises(ValueError):
        serve.make_server(args)


def test_backend_selection(monkeypatch):
    present: set[str] = set()
    monkeypatch.setattr(app, "_has", lambda m: m in present)
    monkeypatch.setattr(app, "_gtk_webkit", lambda: "gi" in present)
    monkeypatch.setattr(app, "chromium_binary", lambda platform=None: binary[0])
    binary = [None]
    assert app.choose_backend("auto", "linux") == ("browser", None)
    binary[0] = "/usr/bin/chromium"
    assert app.choose_backend("auto", "linux") == ("chromium", "/usr/bin/chromium")
    present |= {"webview", "gi"}
    assert app.choose_backend("auto", "linux") == ("webview", "gtk")
    present |= {"qtpy", "PyQt6.QtWebEngineWidgets"}
    assert app.choose_backend("auto", "linux") == ("webview", "qt")
    assert app.choose_backend("chromium", "linux") == ("chromium", "/usr/bin/chromium")
    assert app.choose_backend("browser", "linux") == ("browser", None)
    assert app.webview_gui("win32") is None
    present.add("clr")
    assert app.webview_gui("win32") == "edgechromium"
    present.add("WebKit")
    assert app.webview_gui("darwin") == "cocoa"
    present.discard("webview")
    assert app.webview_gui("linux") is None


def test_chromium_binary_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(app.shutil, "which", lambda n: f"/opt/{n}" if n == "chromium" else None)
    assert app.chromium_binary("linux") == "/opt/chromium"
    exe = tmp_path / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
    assert app.chromium_binary("win32") == str(exe)
    monkeypatch.setattr(app.shutil, "which", lambda n: None)
    assert app.chromium_binary("darwin") is None or app.chromium_binary("darwin").startswith("/Applications")


def test_app_falls_back_to_a_browser_tab(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(app, "choose_backend", lambda requested="auto", platform=None: ("browser", None))

    def fake_open(url):
        opened.append(url)
        raise KeyboardInterrupt  # stands in for Ctrl-C

    monkeypatch.setattr(app.webbrowser, "open", fake_open)
    args = argparse.Namespace(
        data_dir=str(tmp_path), images=None, runs_dir=None, admin_token=None, host="127.0.0.1", port=0,
        backend="auto",
    )  # fmt: skip
    assert app.run(args) == 0
    assert opened and opened[0].startswith("http://127.0.0.1:")
