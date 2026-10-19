"""Background jobs for ``lunar serve`` / ``lunar app``: index builds and downloads with progress."""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TERMINAL = ("done", "error", "cancelled")
MAX_IDS = 1000
_RELEASE_RE = re.compile(r"^LROLRC_\d{4}[A-Z]?$")


class Cancelled(Exception):
    """Raised inside a job's thread once the job was cancelled."""


@dataclass(frozen=True)
class JobContext:
    """Where jobs read and write: the Parquet data dir, image downloads and reference data."""

    data_dir: Path
    download_dir: Path
    reference_dir: Path


class Job:
    """One background task: status, progress (0-1), current file, bytes, rate, ETA and a log tail."""

    def __init__(self, manager: JobManager, jid: str, kind: str, title: str, key: str, params: dict) -> None:
        self._manager = manager
        self.id, self.kind, self.title, self.key, self.params = jid, kind, title, key, params
        self.status = "queued"
        self.progress = 0.0
        self.busy = False
        self.cancellable = True
        self.stage = ""
        self.file = ""
        self.file_index = self.file_count = 0
        self.bytes_done = 0
        self.bytes_total: int | None = None
        self.rate: float | None = None
        self.eta: float | None = None
        self.error: str | None = None
        self.result: Any = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.log: deque[str] = deque(maxlen=40)
        self._span = (0.0, 1.0)
        self._cancel = threading.Event()
        self._last: tuple[float, int] | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        """Raise `Cancelled` if the job was cancelled."""
        if self._cancel.is_set():
            raise Cancelled()

    def update(self, **fields: Any) -> None:
        with self._manager._cond:
            for k, v in fields.items():
                setattr(self, k, v)
            self._manager._changed()

    def info(self, msg: str) -> None:
        log.info("job %s: %s", self.id, msg)
        with self._manager._cond:
            self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
            self._manager._changed()

    def set_stage(self, stage: str, span: tuple[float, float] | None = None, busy: bool = False) -> None:
        """Start a stage; with *span*, byte progress maps onto that part of the overall bar."""
        self.check()
        if span is not None:
            self._span = span
        self.info(stage)
        self.update(stage=stage, busy=busy, progress=max(self.progress, self._span[0]))

    def set_file(self, k: int, n: int, name: str) -> None:
        """File *k* of *n* (1-based) starts; it covers ``[(k-1)/n, k/n]`` of the bar."""
        self.check()
        self._span = ((k - 1) / n, k / n)
        self._last = None
        self.update(
            file=name, file_index=k, file_count=n, bytes_done=0, bytes_total=None, rate=None, eta=None,
            progress=max(self.progress, self._span[0]), stage=f"File {k} of {n}",
        )  # fmt: skip

    def on_bytes(self, name: str, done: int, total: int | None) -> None:
        """Download progress callback
        ([`lunar_toolkit.downloader.Progress`][lunar_toolkit.downloader.Progress])."""
        self.check()
        if done < 0:
            self.update(stage="Waiting: the host asked us to slow down", rate=None, eta=None)
            return
        now = time.monotonic()
        rate = self.rate
        if self._last is None or done < self._last[1]:
            self._last = (now, done)
        elif now - self._last[0] >= 0.5:
            inst = (done - self._last[1]) / (now - self._last[0])
            rate = inst if rate is None else 0.6 * rate + 0.4 * inst
            self._last = (now, done)
        lo, hi = self._span
        frac = min(1.0, done / total) if total else 0.0
        eta = (total - done) / rate if total and rate else None
        self.update(
            file=name, bytes_done=done, bytes_total=total, rate=rate, eta=eta,
            progress=max(self.progress, lo + (hi - lo) * frac),
        )  # fmt: skip

    def to_dict(self, admin: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            k: getattr(self, k)
            for k in (
                "id", "kind", "title", "status", "progress", "busy", "cancellable", "stage", "file",
                "file_index", "file_count", "bytes_done", "bytes_total", "rate", "eta", "created",
                "started", "finished",
            )
        }  # fmt: skip
        if admin:
            d.update(params=self.params, error=self.error, result=self.result, log=list(self.log))
        else:
            d.update(error="failed" if self.error else None)
        return d


class JobManager:
    """Runs jobs in a small thread pool and lets readers wait for any change (``version``)."""

    def __init__(
        self, workers: int = 2, keep: int = 50, on_finish: Callable[[Job], None] | None = None
    ) -> None:
        self._cond = threading.Condition()
        self._on_finish = on_finish
        self._jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lunar-job")
        self._ids = itertools.count(1)
        self._keep = keep
        self.version = 0
        self.closed = False

    def _changed(self) -> None:
        self.version += 1
        self._cond.notify_all()

    def submit(self, kind: str, title: str, key: str, params: dict, func: Callable[[Job], Any]) -> Job:
        """Queue *func(job)*; ValueError when a job with the same *key* is still active."""
        with self._cond:
            if self.closed:
                raise RuntimeError("shutting down")
            for other in self._jobs.values():
                if other.key == key and other.status not in TERMINAL:
                    raise ValueError(f"{other.title} is already {other.status} (job {other.id})")
            job = Job(self, str(next(self._ids)), kind, title, key, params)
            self._jobs[job.id] = job
            done = [j for j in self._jobs.values() if j.status in TERMINAL]
            for old in done[: max(0, len(self._jobs) - self._keep)]:
                del self._jobs[old.id]
            self._changed()
        self._pool.submit(self._run, job, func)
        return job

    def _run(self, job: Job, func: Callable[[Job], Any]) -> None:
        if job.cancelled:
            return
        job.update(status="running", started=time.time())
        try:
            result = func(job)
        except Cancelled:
            job.info("Cancelled")
            job.update(status="cancelled", finished=time.time(), busy=False)
        except Exception as exc:
            log.exception("job %s (%s) failed", job.id, job.kind)
            job.info(f"Error: {exc}")
            job.update(status="error", error=str(exc) or type(exc).__name__, finished=time.time(), busy=False)
        else:
            job.update(status="done", result=result, progress=1.0, finished=time.time(), busy=False, eta=None)
        if self._on_finish:
            self._on_finish(job)

    def get(self, jid: str) -> Job | None:
        with self._cond:
            return self._jobs.get(jid)

    def jobs(self) -> list[Job]:
        with self._cond:
            return list(self._jobs.values())

    def cancel(self, jid: str) -> Job | None:
        """Ask job *jid* to stop (a queued job is cancelled at once); None if unknown."""
        with self._cond:
            job = self._jobs.get(jid)
            if job is None or job.status in TERMINAL:
                return job
            job._cancel.set()
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
            self._changed()
            return job

    def wait(self, version: int, timeout: float) -> int:
        """Block until `version` differs from *version* (or *timeout* / close); the new version."""
        with self._cond:
            self._cond.wait_for(lambda: self.version != version or self.closed, timeout)
            return self.version

    def close(self) -> None:
        """Cancel everything and stop accepting jobs (running ones stop at their next check)."""
        with self._cond:
            self.closed = True
            for job in self._jobs.values():
                if job.status not in TERMINAL:
                    job._cancel.set()
            self._changed()
        self._pool.shutdown(wait=False, cancel_futures=True)


def _flag(p: dict, name: str) -> bool:
    v = p.get(name, False)
    if not isinstance(v, bool):
        raise ValueError(f"{name} must be true or false")
    return v


def _str(p: dict, name: str, default: str | None = None) -> str | None:
    v = p.get(name, default)
    if v is not None and not isinstance(v, str):
        raise ValueError(f"{name} must be a string")
    return v.strip() if v else default


def _strings(p: dict, name: str, limit: int = 100) -> list[str]:
    v = p.get(name)
    if isinstance(v, str):
        v = v.split()
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v) or not v:
        raise ValueError(f"{name} must be a non-empty list of strings")
    if len(v) > limit:
        raise ValueError(f"at most {limit} {name}")
    return [x.strip() for x in v]


def _region(p: dict):
    from .. import reference as ref

    key = p.get("region")
    if not isinstance(key, (str, int)) or isinstance(key, bool):
        raise ValueError("region must be a region id or name")
    try:
        return ref.get_region(key)
    except KeyError as exc:
        raise ValueError(exc.args[0] if exc.args else str(exc)) from None


def _build_db(p: dict, ctx: JobContext) -> tuple[str, str, Callable[[Job], Any]]:
    from ..constants import DATASETS

    dataset = p.get("dataset")
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(DATASETS)}")
    update, force, volume_only = _flag(p, "update"), _flag(p, "force"), _flag(p, "volume_only")
    release = _str(p, "release")
    if release and not _RELEASE_RE.match(release.upper()):
        raise ValueError("release must look like LROLRC_0066C")

    def run(job: Job) -> dict[str, Any]:
        from ..downloader import report_progress
        from ..lroc.cumindex import CumIndex

        ci = CumIndex(dataset, data_dir=ctx.data_dir)

        def on_stage(stage: str) -> None:
            if stage == "download":
                job.set_stage("Downloading the index label and table", span=(0.0, 0.7))
            else:
                job.set_stage("Building the Parquet database (DuckDB)", busy=True)
                job.update(cancellable=False)

        with report_progress(job.on_bytes):
            if update:
                added = ci.update_db(on_stage=on_stage)
                job.info(f"Appended {', '.join(added)}" if added else "Already up to date")
            else:
                ci.build_db(force=force, release=release, volume_only=volume_only, on_stage=on_stage)
        info = ci.build_info() or {}
        job.info(f"{info.get('rows', 0):,} rows, release {info.get('release')}")
        return {"dataset": dataset, "rows": info.get("rows"), "release": info.get("release")}

    verb = "Update" if update else "Build"
    return f"{verb} {dataset.upper()} index", f"build-db:{dataset}", run


def _pgda(p: dict, ctx: JobContext) -> tuple[str, str, Callable[[Job], Any]]:
    from .. import reference as ref

    reg = _region(p)
    if reg.pgda_site is None:
        raise ValueError(f"Region {reg.id} {reg.name} has no PGDA LOLA 5 m site")
    kinds = _strings(p, "kinds", len(ref.PGDA_KINDS))
    bad = [k for k in kinds if k not in ref.PGDA_KINDS]
    if bad:
        raise ValueError(f"unknown PGDA kind(s) {bad}; choose from {', '.join(ref.PGDA_KINDS)}")
    clones = p.get("clones")
    if clones is not None and (not isinstance(clones, int) or isinstance(clones, bool) or clones < 1):
        raise ValueError("clones must be a positive integer")
    site = reg.pgda_site

    def run(job: Job) -> dict[str, Any]:
        from ..downloader import report_progress, url_filename

        job.set_stage("Reading the PGDA file index")
        files = [(k, u) for k, urls in ref.pgda_urls(site, kinds, clones).items() for u in urls]
        out = ctx.reference_dir / "pgda" / site
        with report_progress(job.on_bytes):
            for i, (kind, url) in enumerate(files, 1):
                name = url_filename(url)
                job.set_file(i, len(files), name)
                ref.fetch(url, out / ("Clones" if kind == "clones" else "") / name, show_progress=False)
        job.info(f"{len(files)} file(s) for {site}")
        return {"site": site, "files": len(files)}

    return f"PGDA {site} ({', '.join(kinds)})", f"pgda:{site}", run


def _sdem(p: dict, ctx: JobContext) -> tuple[str, str, Callable[[Job], Any]]:
    reg = _region(p)
    members = _strings(p, "members")

    def run(job: Job) -> dict[str, Any]:
        from .. import reference as ref
        from ..tools import reuse_existing

        job.set_stage("Reading the zip directory (HTTP Range)")
        out_dir = ctx.reference_dir / "sdem" / reg.zenodo_file.removesuffix(".zip")
        done = []
        with ref._open_zip(reg) as zf:
            names = ref.select_members(zf.namelist(), members)
            if not names:
                raise ValueError(f"No zip member matches {members}")
            for i, name in enumerate(names, 1):
                info = zf.getinfo(name)
                dest = out_dir / (name.split("/", 1)[1] if "/" in name else name)
                job.set_file(i, len(names), dest.name)
                if not reuse_existing(dest, False):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    n = 0
                    with zf.open(name) as src, open(tmp, "wb") as dst:
                        while buf := src.read(1 << 20):
                            dst.write(buf)
                            n += len(buf)
                            job.on_bytes(dest.name, n, info.file_size)
                    tmp.replace(dest)
                done.append(dest.name)
        job.info(f"{len(done)} member(s) of {reg.zenodo_file}")
        return {"region": reg.id, "files": len(done)}

    return f"SDEM {reg.name} ({', '.join(members)})", f"sdem:{reg.id}", run


def _download(p: dict, ctx: JobContext) -> tuple[str, str, Callable[[Job], Any]]:
    from ..constants import NAC_ID_RE, WAC_ID_RE

    ids = [i.upper() for i in _strings(p, "ids", MAX_IDS)]
    bad = [i for i in ids if not (NAC_ID_RE.match(i) or WAC_ID_RE.match(i))]
    if bad:
        raise ValueError(f"not LROC product IDs: {' '.join(bad[:5])}")
    product = p.get("product", "edr")
    if product not in ("edr", "cdr"):
        raise ValueError("product must be edr or cdr")
    dest_s = _str(p, "dest")
    dest = Path(dest_s).expanduser() if dest_s else ctx.download_dir
    if not dest.is_absolute():
        raise ValueError("dest must be an absolute directory")

    def run(job: Job) -> dict[str, Any]:
        from ..downloader import report_progress
        from ..lroc.nac import NacImageCollection

        job.set_stage("Looking up the product IDs")
        images = list(NacImageCollection.from_sources(ids, data_dir=ctx.data_dir))
        missing = sorted(set(ids) - {img.product_id for img in images})
        if missing:
            job.info(f"Not in the index: {' '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
        ok, failed = [], []
        with report_progress(job.on_bytes):
            for i, img in enumerate(images, 1):
                job.set_file(i, len(images), img.product_id)
                try:
                    path = img.download_cdr(dest) if product == "cdr" else img.download_edr(dest)
                except Cancelled:
                    raise
                except Exception as exc:
                    job.info(f"{img.product_id}: {exc}")
                    failed.append(img.product_id)
                else:
                    ok.append(path.name)
        job.info(f"{len(ok)} downloaded, {len(failed)} failed, {len(missing)} not found")
        if not ok and (failed or missing):
            raise RuntimeError("nothing was downloaded")
        return {"downloaded": len(ok), "failed": failed, "missing": missing}

    return f"Download {len(ids)} {product.upper()} image(s)", f"download:{product}:{' '.join(ids)}", run


KINDS: dict[str, Callable[[dict, JobContext], tuple[str, str, Callable[[Job], Any]]]] = {
    "build-db": _build_db,
    "pgda": _pgda,
    "sdem": _sdem,
    "download": _download,
}


def submit(manager: JobManager, payload: Any, ctx: JobContext) -> Job:
    """Validate a ``POST /api/jobs`` body and queue the job (ValueError for a bad request)."""
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    kind = payload.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    title, key, func = KINDS[kind](payload, ctx)
    params = {k: v for k, v in payload.items() if k != "kind"}
    return manager.submit(kind, title, key, params, func)
