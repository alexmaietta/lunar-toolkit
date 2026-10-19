"""KPLO ShadowCam: ODE file selection, download, SPICE and CSM cameras for ASP."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import asp
from ..constants import DEFAULT_PROJECTION
from ..downloader import download_file, url_filename
from ..isis import isd_generate, map_project, spiceinit
from ..tools import is_dry_run, reuse_existing, run

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..ode import OdeClient

log = logging.getLogger(__name__)

# Observation M013178466S ships M013178466SE (raw, 8-bit, 3144 samples incl.
# bias columns; ODE "RAW") and M013178466SC (calibrated, float, 3072 samples;
# ODE "CAL").  Both are ISIS cubes with the Instrument group attached.
SC_ID_RE = re.compile(r"^M\d{9}S([CE]?)$", re.IGNORECASE)

PRODUCT_SUFFIX = {"raw": "E", "cal": "C"}

# ASP 3.7 example (docs/examples/shadowcam.rst): plain ``spiceinit from=<cube>``.
SPICEINIT_PARAMS: dict[str, Any] = {"shape": "system"}

# ``cam_test`` ISIS-vs-CSM comparison from the ASP example.
CAM_TEST_SAMPLE_RATE = 50

_POINTING_RE = re.compile(rb"^\s*InstrumentPointing\s*=", re.MULTILINE)
_HISTORY_RE = re.compile(
    rb"^Object = History\s+Name\s*=\s*IsisCube\s+StartByte\s*=\s*(\d+)\s+Bytes\s*=\s*(\d+)", re.MULTILINE
)


@dataclass(frozen=True)
class ScProduct:
    """Cube and PDS4 label URLs of one ShadowCam product."""

    product_id: str
    cube_url: str
    label_url: str | None = None
    cube_kbytes: float | None = None

    @property
    def observation_id(self) -> str:
        """``M013178466S`` for ``M013178466SE`` / ``M013178466SC``."""
        return self.product_id[:-1]

    @property
    def is_calibrated(self) -> bool:
        return self.product_id.endswith("C")

    @property
    def cube_name(self) -> str:
        return f"{self.product_id}.cub"


@dataclass
class ScOutputs:
    """Files written for one product (None for skipped steps)."""

    product_id: str
    cube: Path
    camera: Path | None = None
    cam_test: Path | None = None
    geotiff: Path | None = None


def is_sc_product_id(product_id: str) -> bool:
    """True for a ShadowCam observation (``M…S``) or product (``M…SE`` / ``M…SC``) ID."""
    return bool(SC_ID_RE.match(product_id.strip()))


def product_id_for(source: str, product: str = "raw") -> str:
    """Product ID for an observation or product ID.

    Parameters
    ----------
    source : str
        ``M013178466S`` (observation), ``M013178466SE`` or ``M013178466SC``.
    product : str
        ``"raw"`` (``…SE``, the cube in ASP's example) or ``"cal"`` (``…SC``);
        only used for observation IDs.
    """
    m = SC_ID_RE.match(source.strip())
    if not m:
        raise ValueError(f"Not a ShadowCam ID: {source!r}")
    pid = source.strip().upper()
    if m.group(1):
        return pid
    try:
        return pid + PRODUCT_SUFFIX[product.lower()]
    except KeyError:
        raise ValueError(f"Unknown ShadowCam product {product!r}: {sorted(PRODUCT_SUFFIX)}") from None


def select_sc_files(files: Iterable[dict], product_id: str | None = None) -> ScProduct | None:
    """Pick the cube and label of a ShadowCam product from its ODE ``files`` list.

    Parameters
    ----------
    files : iterable of dict
        ODE product-file dicts (``FileName``, ``Type``, ``URL``, ``KBytes``).
    product_id : str, optional
        Product ID; derived from the cube file name when omitted.

    Returns
    -------
    ScProduct or None
        None when the list has no ``Product`` ``.cub`` (mosaics, DTMs).  The
        map-projected COGs and smear maps (``Referenced``) are never picked.
    """
    cube: dict | None = None
    label: dict | None = None
    for f in files or []:
        if str(f.get("Type", "")).lower() != "product" or not f.get("URL"):
            continue
        name = str(f.get("FileName") or url_filename(f["URL"])).upper()
        if name.endswith(".CUB") and cube is None:
            cube = f
        elif name.endswith(".XML") and label is None:
            label = f
    if cube is None:
        return None
    pid = (product_id or url_filename(cube["URL"]).split(".")[0]).upper()
    try:
        kbytes = float(cube["KBytes"]) if cube.get("KBytes") is not None else None
    except (TypeError, ValueError):
        kbytes = None
    return ScProduct(pid, cube["URL"], label["URL"] if label else None, kbytes)


def sc_products(df: pd.DataFrame) -> list[ScProduct]:
    """ShadowCam products with a downloadable cube from an ODE search.

    Rows that are not ``…SE`` / ``…SC`` products (mosaics, DTMs) or lack a
    cube are skipped with a warning.
    """
    out: list[ScProduct] = []
    pids = df["product_id"] if "product_id" in df else [None] * len(df)
    for pid, files in zip(pids, df.get("files", [])):
        if pid and not is_sc_product_id(pid):
            log.warning("Skipping %s: not a ShadowCam RAW / CAL product", pid)
            continue
        product = select_sc_files(files, pid)
        if product is None:
            log.warning("Skipping %s: no .cub product file", pid)
            continue
        out.append(product)
    return out


def lookup_products(
    product_ids: Iterable[str],
    client: OdeClient | None = None,
    product: str = "raw",
) -> pd.DataFrame:
    """Query ODE by observation / product ID (see
    [`product_id_for`][lunar_toolkit.shadowcam.sc.product_id_for],
    [`OdeClient.lookup`][lunar_toolkit.ode.OdeClient.lookup])."""
    from ..ode import OdeClient

    ids = [product_id_for(p, product) for p in product_ids]
    return (client or OdeClient()).lookup(ids, instrument_label="ShadowCam")


def download_product(
    product: ScProduct,
    output_dir: Path,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download one product's cube (and PDS4 label) into *output_dir*.

    Existing files are reused unless *force*; interrupted downloads resume.
    Returns the cube path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if product.label_url:
        label = output_dir / url_filename(product.label_url)
        if not reuse_existing(label, force):
            download_file(product.label_url, label, show_progress=show_progress)
    cube = output_dir / product.cube_name
    if not reuse_existing(cube, force):
        download_file(product.cube_url, cube, show_progress=show_progress)
    return cube


def has_spice(cube: Path) -> bool:
    """True when ``spiceinit`` completed on *cube*.

    A failed ``spiceinit`` can leave SPICE tables in the label, so the cube's
    History must also record a ``spiceinit`` run.
    """
    try:
        with open(cube, "rb") as f:
            head = f.read(1 << 20)
            end = head.find(b"\nEnd\n")
            label = head if end < 0 else head[:end]
            m = _HISTORY_RE.search(label)
            if not (m and _POINTING_RE.search(label)):
                return False
            f.seek(int(m.group(1)) - 1)
            return b"Object = spiceinit" in f.read(int(m.group(2)))
    except OSError:
        return False


class ScProcessor:
    """Download ShadowCam cubes and prepare them for ASP (SPICE + CSM camera).

    Follows the ASP 3.7 ShadowCam example: ``spiceinit`` on the delivered cube
    (in place), ``isd_generate -k <cube> <cube>`` for the CSM camera and,
    optionally, ``cam_test`` and ``mapproject``.  Needs ASP 3.7's custom ISIS /
    ALE build and the KPLO kernels under ``$ISISDATA/kplo/kernels`` (rsync
    from ``rsync.im-ldi.com::kplo``) plus ``$ISISDATA/base/kernels``.

    Per product *output_dir* gets ``<id>.cub``, ``<id>.json`` (when *csm*),
    ``<id>.cam_test.txt`` (when *validate*) and ``<id>.tif`` (when
    *mapproject*).  Steps whose output exists (for ``spiceinit``: a cube
    whose History records it, see [`has_spice`][lunar_toolkit.shadowcam.sc.has_spice]) are skipped unless
    *force*.

    Parameters
    ----------
    output_dir : Path
        Where cubes, labels and cameras are written.
    product : str
        ``"raw"`` (``…SE``, the cube in ASP's example) or ``"cal"`` (``…SC``,
        radiometrically calibrated, bias columns removed) for observation IDs.
    web : bool
        Use the USGS SPICE web service (it serves the KPLO kernels) in
        ``spiceinit``.  ALE's ShadowCam driver only reads NAIF kernel files,
        so ``isd_generate`` needs the local kernels either way.
    csm : bool
        Write a CSM camera with ``isd_generate``.
    validate : bool
        Run ``cam_test`` (ISIS vs CSM camera) and save its report.
    mapproject : bool
        Write a GeoTIFF: ASP ``mapproject`` onto *dem* when given, else
        ISIS ``cam2map`` onto the ``spiceinit`` shape.
    dem : Path, optional
        DEM for ASP ``mapproject``.
    resolution : float, optional
        Map resolution in m/pixel (ASP ``--tr`` / ``cam2map`` resolution).
    t_srs : str, optional
        PROJ string for ASP ``mapproject`` (default: the DEM's projection).
    projection : str
        ISIS projection for ``mosrange`` without *dem*.
    keep : bool
        Keep the ISIS map-projected cube.
    force : bool
        Redo every processing step (downloads are always reused).
    spiceinit_params : dict, optional
        Overrides merged into `SPICEINIT_PARAMS`.
    """

    def __init__(
        self,
        output_dir: Path = Path("shadowcam"),
        *,
        product: str = "raw",
        web: bool = False,
        csm: bool = True,
        validate: bool = False,
        mapproject: bool = False,
        dem: Path | None = None,
        resolution: float | None = None,
        t_srs: str | None = None,
        projection: str = DEFAULT_PROJECTION,
        keep: bool = False,
        force: bool = False,
        spiceinit_params: dict[str, Any] | None = None,
    ) -> None:
        if product.lower() not in PRODUCT_SUFFIX:
            raise ValueError(f"Unknown ShadowCam product {product!r}: {sorted(PRODUCT_SUFFIX)}")
        self.output_dir = Path(output_dir)
        self.product = product.lower()
        self.web = web
        self.csm = csm
        self.validate = validate
        self.mapproject = mapproject
        self.dem = Path(dem) if dem else None
        self.resolution = resolution
        self.t_srs = t_srs
        self.projection = projection
        self.keep = keep
        self.force = force
        self.spiceinit_params = {**SPICEINIT_PARAMS, **(spiceinit_params or {})}

    def process(self, source: ScProduct | str | Path) -> ScOutputs:
        """Download (if needed) and process one product.

        *source* is a [`ScProduct`][lunar_toolkit.shadowcam.sc.ScProduct], a local ``.cub`` path or an
        observation / product ID (looked up on ODE).
        """
        if isinstance(source, ScProduct):
            cube = download_product(source, self.output_dir)
        elif Path(source).suffix.lower() == ".cub":
            cube = Path(source)
        else:
            products = sc_products(lookup_products([str(source)], product=self.product))
            if not products:
                raise ValueError(f"No ShadowCam product {source!r} on ODE")
            cube = download_product(products[0], self.output_dir)
        return self.process_cube(cube)

    def process_batch(
        self,
        sources: Sequence[ScProduct | str | Path],
        max_workers: int = 2,
    ) -> list[ScOutputs | None]:
        """Process several products in parallel (None for failures)."""
        results: list[ScOutputs | None] = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.process, s): i for i, s in enumerate(sources)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    log.error("Failed %s: %s", _source_name(sources[i]), exc)
        return results

    def cube_path(self, product: ScProduct) -> Path:
        """Where [`process`][lunar_toolkit.shadowcam.sc.ScProcessor.process] downloads *product*'s cube."""
        return self.output_dir / product.cube_name

    def process_cube(self, cube: Path) -> ScOutputs:
        """Run the SPICE / CSM chain on a downloaded ShadowCam cube."""
        cube = Path(cube)
        pid = cube.name.split(".")[0].upper()
        if not is_dry_run():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        out = ScOutputs(pid, cube)

        if self.force or not has_spice(cube):
            spiceinit(cube, web=self.web, **self.spiceinit_params)
        else:
            log.info("Reusing SPICE in %s", cube)

        if self.csm:
            out.camera = self.output_dir / f"{pid}.json"
            if not reuse_existing(out.camera, self.force):
                isd_generate(cube, out.camera)

        if self.validate and out.camera:
            out.cam_test = self.output_dir / f"{pid}.cam_test.txt"
            if not reuse_existing(out.cam_test, self.force):
                self._cam_test(cube, out.camera, out.cam_test)

        if self.mapproject:
            out.geotiff = self.output_dir / f"{pid}.tif"
            if not reuse_existing(out.geotiff, self.force):
                self._mapproject(cube, out.camera, out.geotiff)
        return out

    def _cam_test(self, cube: Path, camera: Path, report: Path) -> None:
        args = ["--image", cube, "--cam1", cube, "--cam2", camera]
        args += ["--session1", "isis", "--session2", "csm", "--sample-rate", CAM_TEST_SAMPLE_RATE]
        res = run("cam_test", args, capture=True)
        if not is_dry_run():
            report.write_text(res.stdout or "")

    def _mapproject(self, cube: Path, camera: Path | None, tif: Path) -> None:
        if self.dem:
            asp.mapproject(self.dem, cube, tif, camera, tr=self.resolution, t_srs=self.t_srs)
            return
        map_cube = tif.with_suffix(".map.cub")
        map_project(cube, map_cube, self.projection, self.resolution, keep=self.keep)
        run("gdal_translate", ["-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", map_cube, tif])
        if not self.keep and not is_dry_run():
            map_cube.unlink(missing_ok=True)


def import_sc(cube: Path, output_dir: Path | None = None, **kwargs: Any) -> ScOutputs:
    """Process one downloaded ShadowCam cube; see [`ScProcessor`][lunar_toolkit.shadowcam.sc.ScProcessor] for
    *kwargs*."""
    cube = Path(cube)
    return ScProcessor(output_dir or cube.parent, **kwargs).process_cube(cube)


def _source_name(source: Any) -> str:
    return source.product_id if isinstance(source, ScProduct) else str(source)
