"""Kaguya Multiband Imager Level 2B2: ODE file selection, download and ISIS import."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import DEFAULT_PROJECTION
from ..isis import isd_generate, isis, map_project, spiceinit
from ..tools import is_dry_run, reuse_existing, run
from .tc import TcProduct, download_product, image_pointer, select_tc_files

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..ode import OdeClient

log = logging.getLogger(__name__)

# Level 2B2 MI product IDs, e.g. MVA_2B2_01_01935N340E0939 (V = VIS, N = NIR).
MI_ID_RE = re.compile(r"^M[VN][A-Z]_2B2_\d{2}_\d{5}[NS]\d{3}E\d{4}$", re.IGNORECASE)

# kaguyami2isis: keep every DN, as for TC.
KAGUYAMI2ISIS_PARAMS: dict[str, Any] = {
    "setnullrange": False,
    "sethrsrange": False,
    "sethisrange": False,
    "setlrsrange": False,
    "setlisrange": False,
}

# spiceinit: reconstructed CK / SPK, LOLA shape.
SPICEINIT_PARAMS: dict[str, Any] = {
    "ckrecon": True,
    "ckpredicted": False,
    "spkrecon": True,
    "spkpredicted": False,
    "shape": "system",
}


@dataclass(frozen=True)
class MiProduct:
    """Data and label URLs of one Level 2B2 MI product."""

    product_id: str
    image_url: str
    label_url: str
    image_kbytes: float | None = None

    @property
    def sensor(self) -> str:
        """``"VIS"`` (5 bands) or ``"NIR"`` (4 bands)."""
        return "VIS" if self.product_id[1] == "V" else "NIR"


@dataclass
class MiOutputs:
    """Files written for one product (None for skipped steps)."""

    product_id: str
    cube: Path
    camera: Path | None = None
    map_cube: Path | None = None
    geotiff: Path | None = None


def is_mi_product_id(product_id: str) -> bool:
    """True for a Level 2B2 MI product ID that ``kaguyami2isis`` can import."""
    return bool(MI_ID_RE.match(product_id.strip()))


def select_mi_files(files: Iterable[dict], product_id: str | None = None) -> MiProduct | None:
    """Pick the data and label files of an MI product from its ODE ``files`` list.

    MI products have the TC layout (detached ``.lbl`` + ``.img.gz``), so
    this follows :func:`~lunar_toolkit.kaguya.tc.select_tc_files`.

    Returns
    -------
    MiProduct or None
        None when the list lacks a ``.img`` / ``.img.gz`` data file or a
        ``.lbl`` label.
    """
    p = select_tc_files(files, product_id)
    return None if p is None else MiProduct(p.product_id, p.image_url, p.label_url, p.image_kbytes)


def mi_products(df: pd.DataFrame) -> list[MiProduct]:
    """MI Level 2B2 products with downloadable data + label files from an ODE search.

    Rows that are not Level 2B2 products (2C2, 3C5, maps) or lack either
    file are skipped with a warning.
    """
    out: list[MiProduct] = []
    pids = df["product_id"] if "product_id" in df else [None] * len(df)
    for pid, files in zip(pids, df.get("files", [])):
        if pid and not is_mi_product_id(pid):
            log.warning("Skipping %s: not a Level 2B2 MI product", pid)
            continue
        product = select_mi_files(files, pid)
        if product is None:
            log.warning("Skipping %s: no .img/.lbl product files", pid)
            continue
        out.append(product)
    return out


def lookup_mi_products(product_ids: Iterable[str], client: OdeClient | None = None) -> pd.DataFrame:
    """Query ODE by product ID and return the observation DataFrame (see :meth:`OdeClient.lookup`)."""
    from ..ode import OdeClient

    return (client or OdeClient()).lookup(product_ids, instrument_label="Kaguya MI")


def download_mi_product(
    product: MiProduct,
    output_dir: Path,
    force: bool = False,
    keep_gz: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download (and gunzip) one product's label and image; returns the label path.

    See :func:`~lunar_toolkit.kaguya.tc.download_product`.
    """
    tc = TcProduct(product.product_id, product.image_url, product.label_url, product.image_kbytes)
    return download_product(tc, output_dir, force=force, keep_gz=keep_gz, show_progress=show_progress)


class MiProcessor:
    """Download and import Kaguya MI Level 2B2 products into ISIS.

    Each product ends up in *output_dir* as ``<id>.cub`` (multi-band,
    spiceinit'ed), ``<id>.json`` (CSM camera, when *csm*) and, when
    *mapproject*, ``<id>.tif``.  Steps whose output exists are skipped
    unless *force*.

    Parameters
    ----------
    output_dir : Path
        Where labels, images, cubes and cameras are written.
    web : bool
        Use the USGS SPICE web service in ``spiceinit`` (no local Kaguya
        kernels needed).  Set False when ``$ISISDATA/kaguya`` is installed.
    csm : bool, optional
        Write a CSM camera with ``isd_generate``.  ALE only has NAIF-SPICE
        MI drivers, which need local kernels, so the default is ``not web``.
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
    kaguyami2isis_params, spiceinit_params : dict, optional
        Overrides merged into :data:`KAGUYAMI2ISIS_PARAMS` /
        :data:`SPICEINIT_PARAMS`.
    """

    def __init__(
        self,
        output_dir: Path = Path("kaguya"),
        *,
        web: bool = True,
        csm: bool | None = None,
        mapproject: bool = False,
        projection: str = DEFAULT_PROJECTION,
        resolution: float | None = None,
        keep: bool = False,
        force: bool = False,
        kaguyami2isis_params: dict[str, Any] | None = None,
        spiceinit_params: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.web = web
        self.csm = (not web) if csm is None else csm
        self.mapproject = mapproject
        self.projection = projection
        self.resolution = resolution
        self.keep = keep
        self.force = force
        self.kaguyami2isis_params = {**KAGUYAMI2ISIS_PARAMS, **(kaguyami2isis_params or {})}
        self.spiceinit_params = {**SPICEINIT_PARAMS, **(spiceinit_params or {})}

    def process(self, source: MiProduct | str | Path) -> MiOutputs:
        """Download (if needed) and import one product.

        *source* is a :class:`MiProduct`, a local ``.lbl`` path or a product
        ID (looked up on ODE).
        """
        if isinstance(source, MiProduct):
            label = download_mi_product(source, self.output_dir, force=self.force)
        elif Path(source).suffix.lower() == ".lbl":
            label = Path(source)
        else:
            products = mi_products(lookup_mi_products([str(source)]))
            if not products:
                raise ValueError(f"No Kaguya MI Level 2B2 product {source!r} on ODE")
            label = download_mi_product(products[0], self.output_dir, force=self.force)
        return self.import_label(label)

    def process_batch(
        self,
        sources: Sequence[MiProduct | str | Path],
        max_workers: int = 2,
    ) -> list[MiOutputs | None]:
        """Process several products in parallel (None for failures)."""
        results: list[MiOutputs | None] = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.process, s): i for i, s in enumerate(sources)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    src = sources[i]
                    log.error("Failed %s: %s", src.product_id if isinstance(src, MiProduct) else src, exc)
        return results

    def import_label(self, label: Path) -> MiOutputs:
        """Run the ISIS chain on a downloaded PDS3 label (image beside it)."""
        label = Path(label)
        pid = label.name.split(".")[0].upper()
        if not is_dry_run():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        cube = self.output_dir / f"{pid}.cub"
        out = MiOutputs(pid, cube)

        if not reuse_existing(cube, self.force):
            try:
                isis("kaguyami2isis", from_=label, to=cube, **self.kaguyami2isis_params)
                spiceinit(cube, web=self.web, **self.spiceinit_params)
            except Exception:
                cube.unlink(missing_ok=True)  # never reuse a cube without SPICE
                raise
        if not self.keep and not is_dry_run():
            name = image_pointer(label) if label.exists() else None
            if name:
                (label.parent / name).unlink(missing_ok=True)

        if self.csm:
            out.camera = self.output_dir / f"{pid}.json"
            if not reuse_existing(out.camera, self.force):
                isd_generate(cube, out.camera, only_naif_spice=True)

        if self.mapproject:
            out.geotiff = self.output_dir / f"{pid}.tif"
            map_cube = self.output_dir / f"{pid}.map.cub"
            if not reuse_existing(out.geotiff, self.force):
                map_project(cube, map_cube, self.projection, self.resolution, keep=self.keep)
                run(
                    "gdal_translate",
                    ["-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", map_cube, out.geotiff],
                )
                if not self.keep and not is_dry_run():
                    map_cube.unlink(missing_ok=True)
            if self.keep:
                out.map_cube = map_cube
        return out


def import_mi(label: Path, output_dir: Path | None = None, **kwargs: Any) -> MiOutputs:
    """Import one downloaded MI product; see :class:`MiProcessor` for *kwargs*."""
    label = Path(label)
    return MiProcessor(output_dir or label.parent, **kwargs).import_label(label)
