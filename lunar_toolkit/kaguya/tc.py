"""Kaguya Terrain Camera Level 2B0: ODE file selection, download and ISIS import."""

from __future__ import annotations

import gzip
import logging
import re
import shutil
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import DEFAULT_PROJECTION, GEOTIFF_OPTIONS
from ..downloader import download_file, url_filename
from ..isis import isd_generate, kaguyatc2isis, map_project, spiceinit
from ..tools import is_dry_run, reuse_existing, run

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..ode import OdeClient

log = logging.getLogger(__name__)

# Level 2B0 TC product IDs, e.g. TC1W2B0_01_02936N034E0938 (W = stereo, S = mono).
TC_ID_RE = re.compile(r"^TC[12][WS]2B0_\d{2}_\d{5}[NS]\d{3}E\d{4}$", re.IGNORECASE)

# kaguyatc2isis: keep every DN (ASP turns the special-pixel ranges off).
KAGUYATC2ISIS_PARAMS: dict[str, Any] = {
    "setnullrange": False,
    "sethrsrange": False,
    "sethisrange": False,
    "setlrsrange": False,
    "setlisrange": False,
}

# spiceinit: reconstructed CK, smithed / reconstructed SPK (ASP example).
# ``attach`` is dropped with ``web=true``, which always attaches (ISIS rejects both).
SPICEINIT_PARAMS: dict[str, Any] = {
    "attach": True,
    "cksmithed": False,
    "ckrecon": True,
    "ckpredicted": False,
    "cknadir": False,
    "spkrecon": True,
    "spkpredicted": False,
    "shape": "system",
    "startpad": 0.0,
    "endpad": 0.0,
}

_IMAGE_POINTER_RE = re.compile(r'^\s*\^IMAGE\s*=\s*\(?\s*"([^"]+)"', re.MULTILINE)


@dataclass(frozen=True)
class TcProduct:
    """Data and label URLs of one Level 2B0 TC product."""

    product_id: str
    image_url: str
    label_url: str
    image_kbytes: float | None = None

    @property
    def is_stereo(self) -> bool:
        return self.product_id[3] == "W"

    @property
    def sensor(self) -> str:
        """``"TC1"`` or ``"TC2"``."""
        return self.product_id[:3]


@dataclass
class TcOutputs:
    """Files written for one product (None for skipped steps)."""

    product_id: str
    cube: Path
    camera: Path | None = None
    map_cube: Path | None = None
    geotiff: Path | None = None


def is_tc_product_id(product_id: str) -> bool:
    """True for a Level 2B0 TC product ID that ``kaguyatc2isis`` can import."""
    return bool(TC_ID_RE.match(product_id.strip()))


def select_tc_files(files: Iterable[dict], product_id: str | None = None) -> TcProduct | None:
    """Pick the data and label files of a TC product from its ODE ``files`` list.

    Parameters
    ----------
    files : iterable of dict
        ODE product-file dicts (``FileName``, ``Type``, ``URL``, ``KBytes``).
    product_id : str, optional
        Product ID; derived from the data file name when omitted.

    Returns
    -------
    TcProduct or None
        None when the list lacks a ``.img`` / ``.img.gz`` data file or a
        ``.lbl`` label.  An uncompressed ``.img`` is preferred over ``.img.gz``.
    """
    image: dict | None = None
    label: dict | None = None
    for f in files or []:
        if str(f.get("Type", "")).lower() != "product" or not f.get("URL"):
            continue
        name = str(f.get("FileName") or url_filename(f["URL"])).upper()
        if name.endswith(".LBL") and label is None:
            label = f
        elif name.endswith(".IMG") or (name.endswith(".IMG.GZ") and image is None):
            image = f
    if image is None or label is None:
        return None
    pid = product_id or url_filename(image["URL"]).split(".")[0]
    kb = image.get("KBytes")
    try:
        kbytes = float(kb) if kb is not None else None
    except (TypeError, ValueError):
        kbytes = None
    return TcProduct(pid.upper(), image["URL"], label["URL"], kbytes)


def tc_products(df: pd.DataFrame) -> list[TcProduct]:
    """TC products with downloadable data + label files from an ODE search.

    Rows that are not Level 2B0 products (DTMs, ortho maps) or lack either
    file are skipped with a warning.
    """
    out: list[TcProduct] = []
    pids = df["product_id"] if "product_id" in df else [None] * len(df)
    for pid, files in zip(pids, df.get("files", [])):
        if pid and not is_tc_product_id(pid):
            log.warning("Skipping %s: not a Level 2B0 TC product", pid)
            continue
        product = select_tc_files(files, pid)
        if product is None:
            log.warning("Skipping %s: no .img/.lbl product files", pid)
            continue
        out.append(product)
    return out


def lookup_products(
    product_ids: Iterable[str],
    client: OdeClient | None = None,
) -> pd.DataFrame:
    """Query ODE by product ID and return the observation DataFrame (see
    [`OdeClient.lookup`][lunar_toolkit.ode.OdeClient.lookup])."""
    from ..ode import OdeClient

    return (client or OdeClient()).lookup(product_ids, instrument_label="Kaguya TC")


def image_pointer(label: Path) -> str | None:
    """File name in the label's ``^IMAGE`` pointer (None if attached / absent)."""
    m = _IMAGE_POINTER_RE.search(Path(label).read_text(encoding="latin-1", errors="replace"))
    return m.group(1) if m else None


def decompress(gz: Path, out: Path | None = None, remove: bool = False) -> Path:
    """Gunzip *gz* to *out* (default: name without ``.gz``) via a ``.part`` file."""
    gz = Path(gz)
    out = Path(out) if out else gz.with_suffix("")
    tmp = out.with_name(out.name + ".part")
    log.info("Decompressing %s", gz.name)
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    tmp.replace(out)
    if remove:
        gz.unlink()
    return out


def download_product(
    product: TcProduct,
    output_dir: Path,
    force: bool = False,
    keep_gz: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download (and gunzip) one product's label and image into *output_dir*.

    The image is written under the name in the label's ``^IMAGE`` pointer so
    ``kaguyatc2isis`` finds it next to the label.  Existing files are reused
    unless *force*.  Returns the label path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = output_dir / url_filename(product.label_url)
    if not reuse_existing(label, force):
        download_file(product.label_url, label, show_progress=show_progress)
    img = output_dir / (image_pointer(label) or f"{product.product_id}.img")
    if reuse_existing(img, force):
        return label
    remote = output_dir / url_filename(product.image_url)
    if not reuse_existing(remote, force):
        download_file(product.image_url, remote, show_progress=show_progress)
    if remote.name.lower().endswith(".gz"):
        decompress(remote, img, remove=not keep_gz)
    elif remote != img:
        remote.replace(img)
    return label


class TcProcessor:
    """Download and import Kaguya TC Level 2B0 products into ISIS.

    Each product ends up in *output_dir* as ``<id>.cub`` (spiceinit'ed),
    ``<id>.json`` (CSM camera, when *csm*) and, when *mapproject*,
    ``<id>.tif`` (`GEOTIFF_OPTIONS`), the
    same flat layout and resume / *keep* / *force* behaviour as
    [`NacProcessor`][lunar_toolkit.lroc.nac.NacProcessor].  Steps whose output
    exists are skipped unless *force*.

    Parameters
    ----------
    output_dir : Path
        Where labels, images, cubes and cameras are written.
    web : bool
        Use the USGS SPICE web service in ``spiceinit`` (no local Kaguya
        kernels needed).  Set False when ``$ISISDATA/kaguya`` is installed.
    csm : bool
        Write a CSM camera with ``isd_generate``.
    only_naif_spice : bool, optional
        Pass ``--only_naif_spice`` to ``isd_generate`` (ASP's recipe; needs
        local kernels).  Default: ``not web``.
    mapproject : bool
        Also ``cam2map`` the cube and write a GeoTIFF.
    projection : str
        ISIS projection for ``mosrange`` (default: equirectangular).
    resolution : float, optional
        Map resolution in m/pixel (default: ``mosrange``'s estimate).
    keep : bool
        Keep the PDS ``.img`` and the map-projected cube after use.
    force : bool
        Redo every step even when its output exists.
    kaguyatc2isis_params, spiceinit_params : dict, optional
        Overrides merged into `KAGUYATC2ISIS_PARAMS` /
        `SPICEINIT_PARAMS`.
    """

    def __init__(
        self,
        output_dir: Path = Path("kaguya"),
        *,
        web: bool = True,
        csm: bool = True,
        only_naif_spice: bool | None = None,
        mapproject: bool = False,
        projection: str = DEFAULT_PROJECTION,
        resolution: float | None = None,
        keep: bool = False,
        force: bool = False,
        kaguyatc2isis_params: dict[str, Any] | None = None,
        spiceinit_params: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.web = web
        self.csm = csm
        self.only_naif_spice = (not web) if only_naif_spice is None else only_naif_spice
        self.mapproject = mapproject
        self.projection = projection
        self.resolution = resolution
        self.keep = keep
        self.force = force
        self.kaguyatc2isis_params = {**KAGUYATC2ISIS_PARAMS, **(kaguyatc2isis_params or {})}
        self.spiceinit_params = {**SPICEINIT_PARAMS, **(spiceinit_params or {})}

    def process(self, source: TcProduct | str | Path) -> TcOutputs:
        """Download (if needed) and import one product.

        *source* is a [`TcProduct`][lunar_toolkit.kaguya.tc.TcProduct], a local ``.lbl`` path or a product
        ID (looked up on ODE).
        """
        if isinstance(source, TcProduct):
            label = download_product(source, self.output_dir, force=self.force)
        elif Path(source).suffix.lower() == ".lbl":
            label = Path(source)
        else:
            products = tc_products(lookup_products([str(source)]))
            if not products:
                raise ValueError(f"No Kaguya TC Level 2B0 product {source!r} on ODE")
            label = download_product(products[0], self.output_dir, force=self.force)
        return self.import_label(label)

    def process_batch(
        self,
        sources: Sequence[TcProduct | str | Path],
        max_workers: int = 2,
    ) -> list[TcOutputs | None]:
        """Process several products in parallel (None for failures)."""
        results: list[TcOutputs | None] = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.process, s): i for i, s in enumerate(sources)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    log.error("Failed %s: %s", _source_name(sources[i]), exc)
        return results

    def import_label(self, label: Path) -> TcOutputs:
        """Run the ISIS chain on a downloaded PDS3 label (image beside it)."""
        label = Path(label)
        pid = label.name.split(".")[0].upper()
        if not is_dry_run():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        cube = self.output_dir / f"{pid}.cub"
        out = TcOutputs(pid, cube)

        if not reuse_existing(cube, self.force):
            try:
                kaguyatc2isis(label, cube, **self.kaguyatc2isis_params)
                spiceinit(cube, web=self.web, **self._spiceinit_params())
            except Exception:
                cube.unlink(missing_ok=True)  # never reuse a cube without SPICE
                raise
        if not self.keep and not is_dry_run():
            self._remove_image(label)

        if self.csm:
            out.camera = self.output_dir / f"{pid}.json"
            if not reuse_existing(out.camera, self.force):
                isd_generate(cube, out.camera, only_naif_spice=self.only_naif_spice)

        if self.mapproject:
            out.geotiff = self.output_dir / f"{pid}.tif"
            map_cube = self.output_dir / f"{pid}.map.cub"
            if not reuse_existing(out.geotiff, self.force):
                self._mapproject(cube, map_cube, out.geotiff)
            if self.keep:
                out.map_cube = map_cube
        return out

    def _mapproject(self, cube: Path, map_cube: Path, tif: Path) -> None:
        map_project(cube, map_cube, self.projection, self.resolution, keep=self.keep)
        run("gdal_translate", [*GEOTIFF_OPTIONS, map_cube, tif])
        if not self.keep and not is_dry_run():
            map_cube.unlink(missing_ok=True)

    def _spiceinit_params(self) -> dict[str, Any]:
        params = dict(self.spiceinit_params)
        if self.web:
            params.pop("attach", None)
        return params

    @staticmethod
    def _remove_image(label: Path) -> None:
        try:
            name = image_pointer(label)
        except OSError:
            return
        if name:
            (label.parent / name).unlink(missing_ok=True)


def import_tc(label: Path, output_dir: Path | None = None, **kwargs: Any) -> TcOutputs:
    """Import one downloaded TC product; see [`TcProcessor`][lunar_toolkit.kaguya.tc.TcProcessor] for
    *kwargs*."""
    label = Path(label)
    return TcProcessor(output_dir or label.parent, **kwargs).import_label(label)


def _source_name(source: Any) -> str:
    return source.product_id if isinstance(source, TcProduct) else str(source)
