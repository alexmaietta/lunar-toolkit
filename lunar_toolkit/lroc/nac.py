"""LROC NAC images: metadata, download and the ISIS + GDAL processing pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from ..constants import DEFAULT_PROJECTION, GEOTIFF_OPTIONS, nac_id, parse_nac_id
from ..downloader import download_file, url_filename
from ..isis import calibrate_lronac, map_project
from ..tools import is_dry_run, reuse_existing, run
from .cumindex import CumIndex

log = logging.getLogger(__name__)

_T = TypeVar("_T")
_V = TypeVar("_V")

# Ordered (column_name, display_label) pairs for metadata display.
# Used by metadata_items(), the CLI query command, and the GUI.
METADATA_LABELS: list[tuple[str, str]] = [
    ("PRODUCT_ID", "Product"),
    ("DATA_SET_ID", "PDS dataset name"),
    ("VOLUME_ID", "PDS volume name"),
    ("INSTRUMENT_HOST_ID", "Instrument host"),
    ("INSTRUMENT_ID", "Instrument"),
    ("ORIGINAL_PRODUCT_ID", "Original product"),
    ("PRODUCT_VERSION_ID", "Product version"),
    ("MISSION_PHASE_NAME", "Mission phase name"),
    ("RATIONALE_DESC", "Rationale desc"),
    ("DATA_QUALITY_ID", "Data quality"),
    ("NAC_PREROLL_START_TIME", "NAC preroll start time"),
    ("START_TIME", "Start time"),
    ("STOP_TIME", "Stop time"),
    ("SPACECRAFT_CLOCK_PARTITION", "SC clock partition"),
    ("NAC_SPACECRAFT_CLOCK_PREROLL_COUNT", "NAC SC clock preroll count"),
    ("SPACECRAFT_CLOCK_START_COUNT", "SC clock start count"),
    ("SPACECRAFT_CLOCK_STOP_COUNT", "SC clock stop count"),
    ("TARGET_NAME", "Target name"),
    ("ORBIT_NUMBER", "Orbit number"),
    ("SLEW_ANGLE", "Slew angle"),
    ("ORBIT_NODE", "LRO node crossing"),
    ("LRO_FLIGHT_DIRECTION", "LRO flight direction"),
    ("NAC_LINE_EXPOSURE_DURATION", "NAC line exposure duration"),
    ("NAC_FRAME_ID", "NAC frame"),
    ("NAC_DAC_RESET", "NAC DAC reset"),
    ("NAC_CHANNEL_A_OFFSET", "NAC channel A offset"),
    ("NAC_CHANNEL_B_OFFSET", "NAC channel B offset"),
    ("INSTRUMENT_MODE_CODE", "Instrument mode code"),
    ("COMPAND_SELECT_CODE", "Compand select code"),
    ("MODE_COMPRESSION", "Mode compression"),
    ("MODE_TEST", "Mode test"),
    ("NAC_TEMPERATURE_SCS", "NAC temperature SCS"),
    ("NAC_TEMPERATURE_FPA", "NAC temperature FPA"),
    ("NAC_TEMPERATURE_FPGA", "NAC temperature FPGA"),
    ("NAC_TEMPERATURE_TELESCOPE", "NAC temperature telescope"),
    ("IMAGE_LINES", "Image lines"),
    ("LINE_SAMPLES", "Line samples"),
    ("SAMPLE_BITS", "Sample bits"),
    ("SCALED_PIXEL_WIDTH", "Scaled pixel width"),
    ("SCALED_PIXEL_HEIGHT", "Scaled pixel height"),
    ("RESOLUTION", "Resolution"),
    ("EMISSION_ANGLE", "Emission angle"),
    ("INCIDENCE_ANGLE", "Incidence angle"),
    ("PHASE_ANGLE", "Phase angle"),
    ("NORTH_AZIMUTH", "North azimuth"),
    ("SUB_SOLAR_AZIMUTH", "Sub solar azimuth"),
    ("SUB_SOLAR_LATITUDE", "Sub solar latitude"),
    ("SUB_SOLAR_LONGITUDE", "Sub solar longitude"),
    ("SUB_SPACECRAFT_LATITUDE", "Sub spacecraft latitude"),
    ("SUB_SPACECRAFT_LONGITUDE", "Sub spacecraft longitude"),
    ("SOLAR_DISTANCE", "Solar distance"),
    ("SOLAR_LONGITUDE", "Solar longitude"),
    ("CENTER_LATITUDE", "Center latitude"),
    ("CENTER_LONGITUDE", "Center longitude"),
    ("UPPER_RIGHT_LATITUDE", "Upper right latitude"),
    ("UPPER_RIGHT_LONGITUDE", "Upper right longitude"),
    ("LOWER_RIGHT_LATITUDE", "Lower right latitude"),
    ("LOWER_RIGHT_LONGITUDE", "Lower right longitude"),
    ("LOWER_LEFT_LATITUDE", "Lower left latitude"),
    ("LOWER_LEFT_LONGITUDE", "Lower left longitude"),
    ("UPPER_LEFT_LATITUDE", "Upper left latitude"),
    ("UPPER_LEFT_LONGITUDE", "Upper left longitude"),
    ("SPACECRAFT_ALTITUDE", "Spacecraft altitude"),
    ("TARGET_CENTER_DISTANCE", "Target center distance"),
    ("PDS_URL", "PDS URL"),
]


def _normalize_nac_id(product_id: str) -> str:
    """Full, upper-case NAC ID (short ``M…L`` → EDR ``M…LE``); other IDs just upper-cased."""
    return nac_id(product_id) or product_id.strip().upper()


def _raw(value: Any) -> Any:
    return value


class IndexColumn(Generic[_V]):
    """Read-only [`NacImage`][lunar_toolkit.lroc.nac.NacImage] attribute holding one CUMINDEX column.

    The value is converted by *cast*; ``None`` when the column is missing
    or null.
    """

    def __init__(self, column: str, cast: Callable[[Any], _V]) -> None:
        self.column = column
        self.cast = cast

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> IndexColumn[_V]: ...

    @overload
    def __get__(self, obj: NacImage, owner: type | None = None) -> _V | None: ...

    def __get__(self, obj: NacImage | None, owner: type | None = None) -> IndexColumn[_V] | _V | None:
        if obj is None:
            return self
        value = obj._get(self.column)
        return None if value is None else self.cast(value)

    def __repr__(self) -> str:
        return f"IndexColumn({self.column!r}, {getattr(self.cast, '__name__', self.cast)})"


class NacImage:
    """A single LROC NAC (or WAC) image identified by its PDS product ID.

    Product ID anatomy::

        M 1096757719 L E
                     │ └── type: E = EDR,  C = CDR
                     └──── side: L = left, R = right

    Short base IDs (``M1096757719L``) are automatically normalized to EDR
    (``M1096757719LE``). The correct CumIndex dataset is auto-detected from
    the type suffix, so no manual dataset selection is needed.

    Parameters
    ----------
    product_id : str
        Full or short NAC/WAC product ID, e.g. ``"M1524360231LE"``,
        ``"M1524360231L"``, or ``"M1524360231LC"``.
    cumindex : CumIndex, optional
        Pre-built CumIndex instance to reuse.  If None, one is constructed
        lazily on first metadata access (dataset auto-detected from suffix).
    _row : pd.Series, optional
        Pre-populated metadata row from a bulk ``CumIndex.get_images()`` call.
        Avoids a per-image database round-trip.
    """

    def __init__(
        self,
        product_id: str,
        cumindex: CumIndex | None = None,
        _row: pd.Series | None = None,
    ) -> None:
        self._product_id = _normalize_nac_id(product_id)
        self._cumindex = cumindex
        self.__row = _row  # pre-populated; None triggers lazy DB load

    # Identity
    @property
    def product_id(self) -> str:
        return self._product_id

    @property
    def side(self) -> str | None:
        """``"L"`` for left NAC, ``"R"`` for right NAC, ``None`` for WAC."""
        parts = parse_nac_id(self._product_id)
        return parts[1] if parts else None

    @property
    def product_type(self) -> str | None:
        """``"edr"`` for LE/RE, ``"cdr"`` for LC/RC, ``None`` for WAC."""
        parts = parse_nac_id(self._product_id)
        return parts[2] if parts else None

    @property
    def edr_id(self) -> str | None:
        """Product ID for the EDR version of this observation (e.g. ``M...LE``)."""
        return nac_id(self._product_id, product_type="edr")

    @property
    def cdr_id(self) -> str | None:
        """Product ID for the CDR version of this observation (e.g. ``M...LC``)."""
        return nac_id(self._product_id, product_type="cdr")

    @property
    def is_left(self) -> bool:
        return self.side == "L"

    @property
    def is_right(self) -> bool:
        return self.side == "R"

    @property
    def is_nac(self) -> bool:
        return self.side is not None

    # Stereo pair
    @property
    def pair_id(self) -> str | None:
        """Product ID of the stereo sibling (L↔R, same type suffix). None for WAC."""
        if self.side is None:
            return None
        return nac_id(self._product_id, side="R" if self.side == "L" else "L")

    @property
    def has_pair(self) -> bool:
        """True if the stereo sibling exists in the EDR index (DB check)."""
        pid = self.pair_id
        if pid is None:
            return False
        result = self._get_cumindex().query(
            f"SELECT count(*) AS n FROM cumindex WHERE upper(trim(PRODUCT_ID)) = '{pid}'"
        )
        return bool(result["n"].iloc[0] > 0)

    @property
    def pair(self) -> NacImage | None:
        """The stereo sibling as a NacImage, or None if not in the index."""
        pid = self.pair_id
        if pid is None:
            return None
        col = self._get_cumindex().get_images([pid])
        return col[0] if len(col) > 0 else None

    # Metadata (lazy, DB-backed)
    @property
    def _row(self) -> pd.Series:
        if self.__row is None:
            df = self._get_cumindex().lookup([self._product_id])
            if df.empty:
                ds = self.product_type or "edr"
                raise ValueError(f"Product {self._product_id!r} not found in {ds.upper()} index")
            self.__row = df.iloc[0]
        return self.__row

    def _get(self, col: str):
        val = self._row.get(col)
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    # Product / archive identity
    pds_dataset_name = IndexColumn("DATA_SET_ID", str)
    pds_volume_name = IndexColumn("VOLUME_ID", str)
    original_product_id = IndexColumn("ORIGINAL_PRODUCT_ID", str)
    product_version = IndexColumn("PRODUCT_VERSION_ID", str)
    instrument_host = IndexColumn("INSTRUMENT_HOST_ID", str)
    instrument = IndexColumn("INSTRUMENT_ID", str)
    mission_phase = IndexColumn("MISSION_PHASE_NAME", str)
    rationale_desc = IndexColumn("RATIONALE_DESC", str)
    data_quality = IndexColumn("DATA_QUALITY_ID", int)

    # Timing
    start_time = IndexColumn("START_TIME", _raw)
    stop_time = IndexColumn("STOP_TIME", _raw)
    nac_preroll_start_time = IndexColumn("NAC_PREROLL_START_TIME", _raw)
    spacecraft_clock_partition = IndexColumn("SPACECRAFT_CLOCK_PARTITION", int)
    spacecraft_clock_start_count = IndexColumn("SPACECRAFT_CLOCK_START_COUNT", str)
    spacecraft_clock_stop_count = IndexColumn("SPACECRAFT_CLOCK_STOP_COUNT", str)
    nac_spacecraft_clock_preroll_count = IndexColumn("NAC_SPACECRAFT_CLOCK_PREROLL_COUNT", str)

    # Orbit / attitude
    target_name = IndexColumn("TARGET_NAME", str)
    orbit_number = IndexColumn("ORBIT_NUMBER", int)
    slew_angle = IndexColumn("SLEW_ANGLE", float)
    """Off-nadir slew angle (degrees). Equivalent to emission_angle."""
    lro_node_crossing = IndexColumn("ORBIT_NODE", str)
    lro_flight_direction = IndexColumn("LRO_FLIGHT_DIRECTION", str)

    # NAC instrument / mode
    nac_line_exposure_duration = IndexColumn("NAC_LINE_EXPOSURE_DURATION", float)
    nac_frame = IndexColumn("NAC_FRAME_ID", str)
    nac_dac_reset = IndexColumn("NAC_DAC_RESET", int)
    nac_channel_a_offset = IndexColumn("NAC_CHANNEL_A_OFFSET", int)
    nac_channel_b_offset = IndexColumn("NAC_CHANNEL_B_OFFSET", int)
    instrument_mode_code = IndexColumn("INSTRUMENT_MODE_CODE", int)
    compand_select_code = IndexColumn("COMPAND_SELECT_CODE", int)
    mode_compression = IndexColumn("MODE_COMPRESSION", bool)
    mode_test = IndexColumn("MODE_TEST", bool)

    # NAC temperatures
    temperature_scs = IndexColumn("NAC_TEMPERATURE_SCS", float)
    temperature_fpa = IndexColumn("NAC_TEMPERATURE_FPA", float)
    temperature_fpga = IndexColumn("NAC_TEMPERATURE_FPGA", float)
    temperature_telescope = IndexColumn("NAC_TEMPERATURE_TELESCOPE", float)

    # Image dimensions
    lines = IndexColumn("IMAGE_LINES", int)
    line_samples = IndexColumn("LINE_SAMPLES", int)
    sample_bits = IndexColumn("SAMPLE_BITS", int)
    scaled_pixel_width = IndexColumn("SCALED_PIXEL_WIDTH", float)
    scaled_pixel_height = IndexColumn("SCALED_PIXEL_HEIGHT", float)
    pixel_scale = IndexColumn("RESOLUTION", float)
    """Ground-sample distance in metres/pixel."""

    # Illumination / geometry angles
    emission_angle = IndexColumn("EMISSION_ANGLE", float)
    """Off-nadir emission angle (degrees). Same as slew_angle."""
    incidence_angle = IndexColumn("INCIDENCE_ANGLE", float)
    phase_angle = IndexColumn("PHASE_ANGLE", float)
    north_azimuth = IndexColumn("NORTH_AZIMUTH", float)
    sub_solar_azimuth = IndexColumn("SUB_SOLAR_AZIMUTH", float)
    sub_solar_latitude = IndexColumn("SUB_SOLAR_LATITUDE", float)
    sub_solar_longitude = IndexColumn("SUB_SOLAR_LONGITUDE", float)
    sub_spacecraft_latitude = IndexColumn("SUB_SPACECRAFT_LATITUDE", float)
    sub_spacecraft_longitude = IndexColumn("SUB_SPACECRAFT_LONGITUDE", float)
    solar_distance = IndexColumn("SOLAR_DISTANCE", float)
    """Distance from the Sun in km."""
    solar_longitude = IndexColumn("SOLAR_LONGITUDE", float)

    # Footprint coordinates
    center_latitude = IndexColumn("CENTER_LATITUDE", float)
    center_longitude = IndexColumn("CENTER_LONGITUDE", float)
    upper_right_latitude = IndexColumn("UPPER_RIGHT_LATITUDE", float)
    upper_right_longitude = IndexColumn("UPPER_RIGHT_LONGITUDE", float)
    lower_right_latitude = IndexColumn("LOWER_RIGHT_LATITUDE", float)
    lower_right_longitude = IndexColumn("LOWER_RIGHT_LONGITUDE", float)
    lower_left_latitude = IndexColumn("LOWER_LEFT_LATITUDE", float)
    lower_left_longitude = IndexColumn("LOWER_LEFT_LONGITUDE", float)
    upper_left_latitude = IndexColumn("UPPER_LEFT_LATITUDE", float)
    upper_left_longitude = IndexColumn("UPPER_LEFT_LONGITUDE", float)

    # Spacecraft position
    spacecraft_altitude = IndexColumn("SPACECRAFT_ALTITUDE", float)
    """Spacecraft altitude above the lunar surface in km."""
    target_center_distance = IndexColumn("TARGET_CENTER_DISTANCE", float)
    """Distance from spacecraft to lunar centre in km."""

    # Display
    def metadata_items(self) -> list[tuple[str, str]]:
        """Return ``(label, value)`` pairs for all available metadata fields.

        Used by the CLI query command and the GUI — single source of truth
        for both ordering and human-readable labels.
        """
        result = []
        for col, label in METADATA_LABELS:
            val = self._get(col)
            if val is not None:
                result.append((label, str(val)))
        return result

    # URLs
    @property
    def edr_url(self) -> str | None:
        """PDS URL for the raw EDR .IMG file."""
        edr = self.edr_id
        if edr is None:
            return None
        if self.product_type == "edr":
            return self._get("PDS_URL")
        # CDR image — cross-lookup the EDR counterpart
        try:
            df = CumIndex(dataset="edr").lookup([edr])
            return df.iloc[0]["PDS_URL"] if not df.empty else None
        except FileNotFoundError:
            return None

    @property
    def cdr_url(self) -> str | None:
        """PDS URL for the calibrated CDR .IMG file (requires CDR database)."""
        cdr = self.cdr_id
        if cdr is None:
            return None
        if self.product_type == "cdr":
            return self._get("PDS_URL")
        # EDR image — cross-lookup the CDR counterpart
        try:
            df = CumIndex(dataset="cdr").lookup([cdr])
            return df.iloc[0]["PDS_URL"] if not df.empty else None
        except FileNotFoundError:
            return None

    # Single-image downloads
    def download_edr(self, output_dir: Path, force: bool = False) -> Path:
        """Download the raw EDR to ``<output_dir>/<edr_id>.IMG``."""
        return self._download(self.edr_url, self.edr_id, output_dir, force)

    def download_cdr(self, output_dir: Path, force: bool = False) -> Path:
        """Download the calibrated CDR to ``<output_dir>/<cdr_id>.IMG`` (requires the CDR database)."""
        return self._download(self.cdr_url, self.cdr_id, output_dir, force)

    def _download(self, url: str | None, pid: str | None, output_dir: Path, force: bool) -> Path:
        if not url or not pid:
            raise ValueError(f"No download URL for {pid or self._product_id} (is its index built?)")
        dest = Path(output_dir) / f"{pid}.IMG"
        if reuse_existing(dest, force):
            return dest
        return download_file(url, dest)

    # Processing
    def process(self, output_dir: Path, **kwargs: Any) -> Path:
        """Run the ISIS + GDAL pipeline on the EDR, returning the GeoTIFF.

        *kwargs* are [`NacProcessor`][lunar_toolkit.lroc.nac.NacProcessor] options (``projection``,
        ``resolution``, ``web``, ``keep``, ``force``).
        """
        url = self.edr_url
        if not url:
            raise ValueError(f"No EDR URL for {self._product_id}")
        return NacProcessor(output_dir, **kwargs).process(url)

    # Quick view
    def to_png(self, output_dir: Path, input_path: Path) -> Path:
        """Convert an image (e.g. a GeoTIFF from [`process`][lunar_toolkit.lroc.nac.NacImage.process]) to
        ``<output_dir>/<id>.png``."""
        return to_png(input_path, Path(output_dir) / f"{self._product_id}.png")

    def show(self, input_path: Path) -> None:
        """Display an image file of this product inline (Jupyter / matplotlib)."""
        img = mpimg.imread(str(input_path))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, cmap="gray")
        ax.set_title(self._product_id)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Helpers
    def _get_cumindex(self) -> CumIndex:
        if self._cumindex is None:
            dataset = self.product_type or "edr"
            self._cumindex = CumIndex(dataset=dataset)
        return self._cumindex

    def __repr__(self) -> str:
        return f"NacImage({self._product_id!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NacImage) and self._product_id == other._product_id

    def __hash__(self) -> int:
        return hash(self._product_id)


# NacImageCollection
class NacImageCollection:
    """An ordered collection of NacImages with bulk download and processing.

    Returned by ``CumIndex.get_images()``.  Behaves like a list but also
    exposes bulk operations that parallelise across images.
    """

    def __init__(self, images: list[NacImage]) -> None:
        self._images = images

    @classmethod
    def from_sources(
        cls,
        sources: list[str],
        data_dir: Path | None = None,
    ) -> NacImageCollection:
        """Build a collection from product IDs or local .IMG paths.

        Raw HTTP(S) URLs are skipped — use ``PdsDownloader`` for those.
        Automatically routes ``LE``/``RE`` → EDR index, ``LC``/``RC`` → CDR index.
        IDs not found in the index are silently excluded.
        """
        product_ids: list[str] = []
        for s in sources:
            s = s.strip()
            if s.startswith(("http://", "https://")):
                continue
            p = Path(s)
            pid = p.stem.upper() if p.exists() else _normalize_nac_id(s)
            product_ids.append(pid)

        if not product_ids:
            return cls([])

        cdr_ids = [p for p in product_ids if (parse_nac_id(p) or ("", "", ""))[2] == "cdr"]
        edr_ids = [p for p in product_ids if p not in cdr_ids]  # EDR, WAC or ambiguous

        rows: dict[str, pd.Series] = {}
        for dataset, ids in [("edr", edr_ids), ("cdr", cdr_ids)]:
            if not ids:
                continue
            try:
                ci = CumIndex(dataset=dataset, data_dir=data_dir)
                df = ci.lookup(ids)
                for _, row in df.iterrows():
                    pid = str(row.get("PRODUCT_ID", "")).strip().upper()
                    rows[pid] = row
            except FileNotFoundError:
                pass  # dataset not yet built

        images = [NacImage(pid, _row=rows[pid]) for pid in product_ids if pid in rows]
        return cls(images)

    # List interface
    def __len__(self) -> int:
        return len(self._images)

    def __iter__(self):
        return iter(self._images)

    def __getitem__(self, idx):
        return self._images[idx]

    def __repr__(self) -> str:
        return f"NacImageCollection({len(self._images)} images)"

    # Metadata
    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per image (all metadata columns)."""
        rows = [img._row for img in self._images]
        return pd.DataFrame(rows).reset_index(drop=True)

    # Bulk downloads
    def download_edrs(
        self,
        output_dir: Path,
        workers: int = 4,
        force: bool = False,
    ) -> dict[str, Path | None]:
        """Download all EDR .IMG files in parallel.

        Returns a dict mapping product_id → local Path (or None on failure).
        """
        return self._bulk_download(NacImage.download_edr, output_dir, workers, force)

    def download_cdrs(
        self,
        output_dir: Path,
        workers: int = 4,
        force: bool = False,
    ) -> dict[str, Path | None]:
        """Download all CDR .IMG files in parallel (requires CDR database)."""
        return self._bulk_download(NacImage.download_cdr, output_dir, workers, force)

    def _bulk_download(
        self, download: Callable[..., Path], output_dir: Path, workers: int, force: bool
    ) -> dict[str, Path | None]:
        return _parallel(
            {img.product_id: img for img in self._images},
            lambda img: download(img, output_dir, force=force),
            workers,
        )

    # Bulk processing
    def process(self, output_dir: Path, workers: int = 4, **kwargs: Any) -> dict[str, Path | None]:
        """Run the ISIS + GDAL pipeline on all EDRs in parallel.

        *kwargs* are [`NacProcessor`][lunar_toolkit.lroc.nac.NacProcessor] options.  Returns a dict mapping
        product_id → GeoTIFF Path (or None on failure).
        """
        urls = {img.product_id: img.edr_url for img in self._images}
        sources = [(pid, url) for pid, url in urls.items() if url]
        results: dict[str, Path | None] = dict.fromkeys(urls)
        geotiffs = NacProcessor(output_dir, **kwargs).process_batch([url for _, url in sources], workers)
        results.update((pid, tif) for (pid, _), tif in zip(sources, geotiffs))
        return results

    # Bulk PNG conversion
    def to_pngs(
        self,
        output_dir: Path,
        input_paths: dict[str, Path],
        workers: int = 4,
    ) -> dict[str, Path | None]:
        """Convert a set of TIFs to PNGs via gdal_translate in parallel.

        Parameters
        ----------
        output_dir : Path
        input_paths : dict[str, Path]
            Mapping of product_id → source TIF path (e.g. from download_cdrs()).
        workers : int

        Returns
        -------
        dict mapping product_id → PNG Path (or None on failure).
        """
        output_dir = Path(output_dir)
        return _parallel(
            {pid: (src, output_dir / f"{pid}.png") for pid, src in input_paths.items()},
            lambda paths: to_png(*paths),
            workers,
        )


class NacProcessor:
    """Download, calibrate and map-project LROC NAC EDRs into GeoTIFFs.

    Each image ends up in *output_dir* as ``<id>.tif`` (map-projected I/F,
    `GEOTIFF_OPTIONS`).  Intermediates
    (a downloaded ``<id>.IMG``, ``<id>.cal.echo.cub``, ``<id>.map.cub``) are
    written beside it and removed after success unless *keep*; after a
    failure they stay, so a re-run resumes.  Steps whose output exists are
    skipped unless *force*.

    Parameters
    ----------
    output_dir : Path
        Where downloads, cubes and GeoTIFFs are written.
    web : bool
        Use the USGS SPICE web service in ``spiceinit``.
    projection : str
        ISIS projection for ``mosrange`` (default: equirectangular).
    resolution : float, optional
        Map resolution in m/pixel (default: ``mosrange``'s estimate).
    keep : bool
        Keep the downloaded EDR and the intermediate cubes.
    force : bool
        Redo every step even when its output exists.
    """

    def __init__(
        self,
        output_dir: Path = Path("data"),
        *,
        web: bool = True,
        projection: str = DEFAULT_PROJECTION,
        resolution: float | None = None,
        keep: bool = False,
        force: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.web = web
        self.projection = projection
        self.resolution = resolution
        self.keep = keep
        self.force = force

    def process(self, source: str | Path) -> Path:
        """Process one EDR (local ``.IMG`` path or HTTP(S) URL); returns the GeoTIFF."""
        source = str(source)
        is_url = source.startswith(("http://", "https://"))
        pid = Path(url_filename(source) if is_url else source).name.split(".")[0].upper()
        if not pid:
            raise ValueError(f"Cannot derive image ID from {source!r}")
        geotiff = self.output_dir / f"{pid}.tif"
        if reuse_existing(geotiff, self.force):
            return geotiff
        if not is_dry_run():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        img = self.output_dir / f"{pid}.IMG" if is_url else Path(source)
        if is_url and not reuse_existing(img, self.force):
            download_file(source, img)
        echo = calibrate_lronac(img, self.output_dir, web=self.web, keep=self.keep, force=self.force)
        map_cub = self.output_dir / f"{pid}.map.cub"
        map_project(echo, map_cub, self.projection, self.resolution, keep=self.keep)
        run("gdal_translate", [*GEOTIFF_OPTIONS, map_cub, geotiff])
        if not self.keep and not is_dry_run():
            for f in (map_cub, echo, img if is_url else None):
                if f is not None:
                    f.unlink(missing_ok=True)
        log.info("Pipeline complete → %s", geotiff)
        return geotiff

    def process_batch(self, sources: Sequence[str | Path], max_workers: int = 4) -> list[Path | None]:
        """Process several images in parallel (None for failures)."""
        results: list[Path | None] = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.process, src): i for i, src in enumerate(sources)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    log.error("Failed %s: %s", sources[idx], exc)
        return results


def to_png(src: Path, dst: Path) -> Path:
    """Convert an image to an 8-bit, min/max-scaled PNG with ``gdal_translate``."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run("gdal_translate", ["-of", "PNG", "-ot", "Byte", "-scale", src, dst], capture=True)
    return Path(dst)


def _parallel(items: dict[str, _T], func: Callable[[_T], Path], workers: int) -> dict[str, Path | None]:
    """``{key: func(item)}`` run in a thread pool; failures are logged and map to None."""

    def call(key: str, item: _T) -> tuple[str, Path | None]:
        try:
            return key, func(item)
        except Exception as exc:
            log.error("%s: %s", key, exc)
            return key, None

    results: dict[str, Path | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(call, k, v) for k, v in items.items()]):
            key, path = future.result()
            results[key] = path
    return results
