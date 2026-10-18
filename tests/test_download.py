"""Resumable downloads and MD5 verification (network mocked)."""

import hashlib
import io
import urllib.error

import pytest

from lunar_toolkit import downloader
from lunar_toolkit.downloader import ChecksumError, download_file, file_md5

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes
MD5 = hashlib.md5(PAYLOAD).hexdigest()


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int, headers: dict):
        super().__init__(data)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeServer:
    """Stand-in for urllib.request.urlopen honouring Range headers."""

    def __init__(self, data: bytes, support_range: bool = True):
        self.data = data
        self.support_range = support_range
        self.requests = []

    def __call__(self, req, timeout=None):
        rng = req.get_header("Range")
        self.requests.append(rng)
        if rng and self.support_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(self.data):
                raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)
            body = self.data[start:]
            return _Resp(body, 206, {"Content-Length": str(len(body))})
        return _Resp(self.data, 200, {"Content-Length": str(len(self.data))})


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """Fresh per-host throttles, and retries never sleep for real."""
    sleeps: list[float] = []
    clock = [downloader.time.time()]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(downloader.time, "sleep", fake_sleep)
    monkeypatch.setattr(downloader.time, "time", lambda: clock[0])
    return sleeps


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer(PAYLOAD)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", srv)
    return srv


def test_fresh_download_with_md5(tmp_path, server):
    dest = download_file(
        "https://example.test/f.TAB", tmp_path / "f.TAB", show_progress=False, expected_md5=MD5
    )
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "f.TAB.part").exists()
    assert server.requests == [None]


def test_resume_from_part(tmp_path, server):
    part = tmp_path / "f.TAB.part"
    part.write_bytes(PAYLOAD[:30_000])
    dest = download_file(
        "https://example.test/f.TAB", tmp_path / "f.TAB", show_progress=False, expected_md5=MD5
    )
    assert dest.read_bytes() == PAYLOAD
    assert server.requests == ["bytes=30000-"]


def test_resume_complete_part_416(tmp_path, server):
    (tmp_path / "f.TAB.part").write_bytes(PAYLOAD)
    dest = download_file(
        "https://example.test/f.TAB", tmp_path / "f.TAB", show_progress=False, expected_md5=MD5
    )
    assert dest.read_bytes() == PAYLOAD


def test_server_ignores_range_restarts(tmp_path, monkeypatch):
    srv = FakeServer(PAYLOAD, support_range=False)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", srv)
    (tmp_path / "f.TAB.part").write_bytes(b"garbage" * 10)
    dest = download_file("https://example.test/f.TAB", tmp_path / "f.TAB", show_progress=False)
    assert dest.read_bytes() == PAYLOAD


def test_md5_mismatch_removes_part(tmp_path, server):
    with pytest.raises(ChecksumError):
        download_file(
            "https://example.test/f.TAB", tmp_path / "f.TAB", show_progress=False, expected_md5="0" * 32
        )
    assert not (tmp_path / "f.TAB").exists()
    assert not (tmp_path / "f.TAB.part").exists()


def test_interrupted_download_keeps_part(tmp_path, monkeypatch):
    class Broken(_Resp):
        def read(self, n=-1):
            chunk = super().read(min(n, 10_000) if n and n > 0 else 10_000)
            if self.tell() >= 50_000:
                raise ConnectionResetError("boom")
            return chunk

    def urlopen(req, timeout=None):
        return Broken(PAYLOAD, 200, {"Content-Length": str(len(PAYLOAD))})

    monkeypatch.setattr(downloader.urllib.request, "urlopen", urlopen)
    with pytest.raises(ConnectionResetError):
        download_file("https://example.test/f.TAB", tmp_path / "f.TAB", show_progress=False)
    assert (tmp_path / "f.TAB.part").stat().st_size > 0


def test_legacy_url_rewritten(tmp_path, server):
    from lunar_toolkit.constants import modernize_pds_url

    url = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/INDEX/INDEX.LBL"
    assert modernize_pds_url(url).startswith("https://pds.mcp.nasa.gov/data/store/img/")
    assert "/lro-l-lroc-2-edr/LROLRC_0001/" in modernize_pds_url(url)


def test_file_md5(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(PAYLOAD)
    assert file_md5(p, chunk=1000) == MD5


def test_download_urls_skips_existing_unless_force(tmp_path, server):
    from lunar_toolkit.downloader import PdsDownloader

    url = "https://example.test/f.TAB"
    (tmp_path / "f.TAB").write_bytes(b"old")
    dl = PdsDownloader(max_workers=1)
    assert dl.download_urls([url], tmp_path, show_progress=False) == {url: tmp_path / "f.TAB"}
    assert server.requests == [] and (tmp_path / "f.TAB").read_bytes() == b"old"
    dl.download_urls([url], tmp_path, show_progress=False, force=True)
    assert (tmp_path / "f.TAB").read_bytes() == PAYLOAD


def test_ode_download_force(tmp_path, monkeypatch):
    import pandas as pd

    from lunar_toolkit.ode import OdeClient

    calls = []
    monkeypatch.setattr(
        downloader.PdsDownloader,
        "download_urls",
        lambda self, urls, out, show_progress=True, force=False: calls.append((urls, force)) or {},
    )
    df = pd.DataFrame({"files": [[{"Type": "Product", "URL": "https://x/a.IMG", "FileName": "a.IMG"}]]})
    OdeClient().download(df, tmp_path, force=True)
    assert calls == [(["https://x/a.IMG"], True)]


def test_throttling_is_retried_with_retry_after(tmp_path, monkeypatch, _no_waiting):
    srv = FakeServer(PAYLOAD)
    calls = {"n": 0}

    def throttled(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, None)
        return srv(req, timeout)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", throttled)
    out = download_file("https://example.org/a.IMG", tmp_path / "a.IMG", show_progress=False)
    assert out.read_bytes() == PAYLOAD
    assert _no_waiting == [pytest.approx(3.0, abs=0.5)] * 2


def test_permanent_errors_and_exhausted_retries_raise(tmp_path, monkeypatch):
    def fail(code):
        def _open(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)

        return _open

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail(404))
    with pytest.raises(urllib.error.HTTPError):
        download_file("https://example.org/b.IMG", tmp_path / "b.IMG", show_progress=False)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail(500))
    with pytest.raises(urllib.error.HTTPError):
        download_file("https://example.org/c.IMG", tmp_path / "c.IMG", show_progress=False, retries=2)
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", fail(503)
    )  # throttling: bounded by the wait budget
    with pytest.raises(urllib.error.HTTPError):
        download_file("https://example.org/d.IMG", tmp_path / "d.IMG", show_progress=False, max_wait=1800)


def test_throttle_is_shared_per_host(monkeypatch):
    monkeypatch.setenv(downloader.MAX_PER_HOST_ENV, "2")
    a = downloader._host("https://pds.example.org/x/a.IMG")
    assert downloader._host("https://PDS.example.org/y/b.IMG") is a
    assert downloader._host("https://other.example.org/a.IMG") is not a
    assert a.slots.acquire(blocking=False) and a.slots.acquire(blocking=False)
    assert not a.slots.acquire(blocking=False)  # third concurrent download waits


def _throttle_then_ok(n, headers, srv):
    calls = {"n": 0}

    def _open(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= n:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", headers, None)
        return srv(req, timeout)

    return _open


def test_long_retry_after_is_honoured_and_persisted(tmp_path, monkeypatch, _no_waiting):
    srv = FakeServer(PAYLOAD)
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", _throttle_then_ok(1, {"Retry-After": "3600"}, srv)
    )
    download_file("https://pds.example.org/a.IMG", tmp_path / "a.IMG", show_progress=False)
    assert _no_waiting == [pytest.approx(3600, abs=5)]

    downloader._host("https://pds.example.org/x").cool(1800)
    monkeypatch.setattr(downloader, "_HOSTS", {})  # as if a new process started
    fresh = downloader._host("https://pds.example.org/b.IMG")
    assert fresh.cooldown_until == pytest.approx(downloader.time.time() + 1800, abs=1)
    fresh.wait()
    assert _no_waiting[-1] == pytest.approx(1800, abs=1)


def test_throttling_does_not_use_retries_but_respects_budget(tmp_path, monkeypatch, _no_waiting):
    srv = FakeServer(PAYLOAD)
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", _throttle_then_ok(5, {"Retry-After": "10"}, srv)
    )
    download_file("https://h.example.org/a.IMG", tmp_path / "a.IMG", show_progress=False, retries=1)
    assert len(_no_waiting) == 5
    monkeypatch.setattr(downloader, "_HOSTS", {})
    monkeypatch.setattr(downloader, "COOLDOWN_FILE", tmp_path / "other.json")
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", _throttle_then_ok(5, {"Retry-After": "3600"}, srv)
    )
    with pytest.raises(urllib.error.HTTPError):
        download_file("https://h.example.org/b.IMG", tmp_path / "b.IMG", show_progress=False, max_wait=600)


def test_retry_after_http_date():
    when = downloader.time.time() + 120
    date = downloader.email.utils.formatdate(when, usegmt=True)
    exc = urllib.error.HTTPError("u", 429, "x", {"Retry-After": date}, None)
    assert downloader._retry_after(exc) == pytest.approx(120, abs=2)
