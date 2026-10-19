"""LROC PDS CUMINDEX as a local DuckDB / Parquet database."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import (
    DATASETS,
    DEFAULT_CACHE_DIR,
    FOOTPRINT_CORNERS,
    INDEX_DIR,
    MOON_RADIUS_KM,
    NAC_PRODUCT_TYPES,
    NAC_SUFFIXES,
    PDS_STORE_URL,
    POLAR_LAT_THRESHOLD,
    VOLINDEX_LBL_FILENAME,
    VOLINDEX_TAB_FILENAME,
    find_parquet,
    is_nac_id,
    parse_nac_id,
    pds_url_from_file_spec,
)
from ..downloader import _USER_AGENT, PdsDownloader
from ..geometry import aoi_region, filter_aoi, load_aoi, query_region

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .nac import NacImageCollection

log = logging.getLogger(__name__)

# Bump when the Parquet layout changes in a way readers should know about.
SCHEMA_VERSION = 2

# Footprint columns derived at build time (all prefixed FP_).
FOOTPRINT_COLUMNS = [
    "FP_MIN_LAT",
    "FP_MAX_LAT",
    "FP_LON_WEST",
    "FP_LON_EAST",
    "FP_LON_SPAN",
    "FP_CONTAINS_POLE",
    "FP_PS_HEMI",
    "FP_PS_XMIN",
    "FP_PS_XMAX",
    "FP_PS_YMIN",
    "FP_PS_YMAX",
]


class CumIndex:
    """Query the LROC PDS CUMINDEX via DuckDB on a local Parquet database.

    Parameters
    ----------
    dataset : str
        Which PDS dataset to use: ``"edr"`` (default), ``"cdr"``, or ``"rdr"``.
    data_dir : Path, optional
        Directory containing ``cumindex_<dataset>.parquet``.  Defaults to
        ``LUNAR_DATA_DIR``, else the user data dir.
    cache_dir : Path
        Where raw CUMINDEX/INDEX .TAB / .LBL downloads are cached.
        Defaults to ``~/.cache/lunar_toolkit/``.
    release : str, optional
        PDS volume to read labels / index tables from (e.g. ``"LROLRC_0066C"``).
        Defaults to the release recorded in the build sidecar, else the
        fallback constant.  ``build_db()`` auto-discovers the newest volume.
    """

    def __init__(
        self,
        dataset: str = "edr",
        data_dir: Path | None = None,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        release: str | None = None,
    ) -> None:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r}. Valid options: {list(DATASETS)}")
        self._dataset = dataset
        self._ds = DATASETS[dataset]
        self._explicit_data_dir = Path(data_dir) if data_dir is not None else None
        self._cache_dir = Path(cache_dir)
        self._cache = PdsDownloader(self._cache_dir)
        self._release = _norm_volume(release) if release else None
        self._schema: pd.DataFrame | None = None
        self._meta: dict | None = None

    # Dataset URLs
    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def release(self) -> str:
        """Volume used for label / index URLs (no trailing slash)."""
        if self._release:
            return self._release
        side = self.build_info()
        if side and side.get("release"):
            return side["release"]
        return _norm_volume(self._ds["release"])

    def _index_base_url(self, volume: str | None = None) -> str:
        vol = _norm_volume(volume) if volume else self.release
        return self._ds["base_url"] + vol + "/" + self._ds["index_dir"]

    @property
    def lbl_url(self) -> str:
        return self._index_base_url() + self._ds["lbl_filename"]

    @property
    def tab_url(self) -> str:
        return self._index_base_url() + self._ds["tab_filename"]

    def volume_index_urls(self, volume: str) -> tuple[str, str]:
        """``(INDEX.LBL, INDEX.TAB)`` URLs for a single PDS volume."""
        base = self._index_base_url(volume)
        return base + VOLINDEX_LBL_FILENAME, base + VOLINDEX_TAB_FILENAME

    # Parquet path / sidecar
    @property
    def parquet_path(self) -> Path:
        """Path to the Parquet database for this dataset."""
        return find_parquet(self._ds["parquet"], self._explicit_data_dir)

    @property
    def meta_path(self) -> Path:
        """Build-metadata sidecar next to the Parquet (``*.meta.json``)."""
        p = self.parquet_path
        return p.with_name(p.stem + ".meta.json")

    def build_info(self) -> dict | None:
        """Return the build sidecar as a dict, or None if absent."""
        mp = self.meta_path
        if not mp.exists():
            return None
        try:
            return json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _ensure_parquet(self) -> Path:
        p = self.parquet_path
        if not p.exists():
            raise FileNotFoundError(
                f"Database not found: {p}\n"
                f"Build it with:  lunar build-db {self._dataset}\n"
                f"Or set LUNAR_DATA_DIR to a directory that contains {self._ds['parquet']}"
            )
        return p

    # Schema / metadata (from .LBL)
    @property
    def schema(self) -> pd.DataFrame:
        """Column definitions from CUMINDEX.LBL (lazy, downloads to cache)."""
        if self._schema is None:
            lbl_text = self._read_lbl(self.lbl_url)
            self._schema = _parse_pds_lbl_columns(lbl_text)
            self._meta = _parse_lbl_meta(lbl_text)
        return self._schema

    @property
    def meta(self) -> dict:
        """Top-level scalar fields from CUMINDEX.LBL (lazy)."""
        if self._meta is None:
            self._meta = _parse_lbl_meta(self._read_lbl(self.lbl_url))
        return self._meta

    def _read_lbl(self, url: str) -> str:
        return self._cache.ensure_cached(url).read_text(encoding="latin-1")

    # Volume discovery
    def discover_volumes(self) -> list[str]:
        """List this dataset's PDS volumes (sorted oldest → newest).

        Uses the S3 ``ListObjectsV2`` listing exposed by the PDS store; falls
        back to probing ``INDEX/INDEX.LBL`` of successive volume numbers.
        """
        try:
            vols = list_store_volumes(self._ds["base_url"], self._ds["volume_re"])
            if vols:
                return vols
        except (OSError, ET.ParseError, ValueError) as exc:
            log.warning("Volume listing failed (%s); probing instead", exc)
        start = self.build_info() or {}
        return probe_volumes(
            self._ds["base_url"],
            start.get("release") or _norm_volume(self._ds["release"]),
        )

    def latest_release(self) -> str:
        """Newest volume on the server (fallback: the constant release)."""
        try:
            vols = self.discover_volumes()
        except Exception as exc:  # network down etc.
            log.warning("Could not discover volumes (%s)", exc)
            vols = []
        return vols[-1] if vols else _norm_volume(self._ds["release"])

    # Build database
    def build_db(
        self,
        force: bool = False,
        release: str | None = None,
        volume_only: bool = False,
        memory_limit: str = "4GB",
        threads: int | None = None,
        wkt: bool = False,
        on_stage: Callable[[str], None] | None = None,
    ) -> Path:
        """Download the PDS index table and build the Parquet database.

        Parameters
        ----------
        force : bool
            Rebuild even if the Parquet already exists.
        release : str, optional
            Volume to build from (e.g. ``"LROLRC_0066C"``).  Default: newest
            volume discovered on the server, else the fallback constant.
        volume_only : bool
            Build from that volume's own ``INDEX/INDEX.TAB`` (one release,
            ~20 k rows) instead of the cumulative ``CUMINDEX.TAB`` (~3.8 GB).
        memory_limit : str
            DuckDB ``memory_limit``; sorts spill to a temp dir under the cache.
        threads : int, optional
            DuckDB worker threads (default: all cores).
        wkt : bool
            Also store a ``FP_WKT`` footprint polygon column.
        on_stage : callable, optional
            Called with ``"download"`` and ``"build"`` as those stages start.

        Returns
        -------
        Path to the written Parquet file.
        """
        p = self.parquet_path
        if p.exists() and not force:
            log.info("Database already exists: %s (force=True rebuilds, update_db() appends new volumes)", p)
            return p

        p.parent.mkdir(parents=True, exist_ok=True)
        clean_stale_build_files(p.parent)

        if on_stage:
            on_stage("download")
        vol = _norm_volume(release) if release else self.latest_release()
        self._release = vol
        self._schema = self._meta = None
        if volume_only:
            lbl_url, tab_url = self.volume_index_urls(vol)
        else:
            lbl_url, tab_url = self.lbl_url, self.tab_url

        lbl_text = self._read_lbl(lbl_url)
        meta = _parse_lbl_meta(lbl_text)
        columns = parse_lbl_columns(lbl_text)
        tab_path = self._cache.ensure_cached(tab_url, expected_md5=meta.get("md5_checksum"))

        if on_stage:
            on_stage("build")
        n = build_parquet(
            [(tab_path, columns)],
            p,
            temp_dir=self._cache_dir / "duckdb_tmp",
            memory_limit=memory_limit,
            threads=threads,
            wkt=wkt,
            expected_rows=meta.get("file_records") or meta.get("rows"),
        )
        now = _utcnow()
        self._write_sidecar(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": self._dataset,
                "data_set_id": self._ds["data_set_id"],
                "source": "volume_index" if volume_only else "cumindex",
                "release": vol,
                "lbl_url": lbl_url,
                "tab_url": tab_url,
                "md5": meta.get("md5_checksum"),
                "rows": n,
                "wkt": wkt,
                "volumes": {
                    vol: {
                        "kind": "volume_index" if volume_only else "cumindex",
                        "rows": n,
                        "md5": meta.get("md5_checksum"),
                        "added_at": now,
                    }
                },
                "built_at": now,
                "updated_at": now,
            }
        )
        return p

    def update_db(
        self,
        volumes: Iterable[str] | None = None,
        memory_limit: str = "4GB",
        threads: int | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Append per-volume ``INDEX.TAB`` files newer than the built release.

        Parameters
        ----------
        volumes : iterable of str, optional
            Explicit volumes to add.  Default: every discovered volume newer
            than the sidecar's ``release`` (or ``max(VOLUME_ID)`` in the
            Parquet when no sidecar exists).
        on_stage : callable, optional
            As in [`build_db`][lunar_toolkit.lroc.cumindex.CumIndex.build_db].

        Returns
        -------
        The list of volumes that were appended (empty if already up to date).
        Rows are de-duplicated on ``PRODUCT_ID`` (newest volume wins).
        """
        p = self._ensure_parquet()
        if on_stage:
            on_stage("download")
        side = self.build_info() or {}
        current = side.get("release") or self._max_volume_in_parquet()
        if volumes is None:
            if current is None:
                raise RuntimeError("Cannot determine the built release; rebuild with --force")
            found = self.discover_volumes()
            ck = _volume_key(current)
            new_vols = [v for v in found if _volume_key(v) > ck]
        else:
            new_vols = sorted({_norm_volume(v) for v in volumes}, key=_volume_key)

        if not new_vols:
            log.info("%s is up to date (release %s)", p.name, current)
            return []

        log.info("Appending %d volume(s) to %s: %s", len(new_vols), p.name, ", ".join(new_vols))
        clean_stale_build_files(p.parent)
        sources: list[tuple[Path, list[dict]]] = []
        added: dict[str, dict] = {}
        for vol in new_vols:
            lbl_url, tab_url = self.volume_index_urls(vol)
            lbl_text = self._read_lbl(lbl_url)
            meta = _parse_lbl_meta(lbl_text)
            tab = self._cache.ensure_cached(tab_url, expected_md5=meta.get("md5_checksum"))
            sources.append((tab, parse_lbl_columns(lbl_text)))
            added[vol] = {
                "kind": "volume_index",
                "rows": meta.get("file_records") or meta.get("rows"),
                "md5": meta.get("md5_checksum"),
                "added_at": _utcnow(),
            }

        if on_stage:
            on_stage("build")
        tmp = p.with_name(p.stem + ".update.parquet")
        n = build_parquet(
            sources,
            tmp,
            base_parquet=p,
            temp_dir=self._cache_dir / "duckdb_tmp",
            memory_limit=memory_limit,
            threads=threads,
            wkt=bool(side.get("wkt")),
        )
        tmp.replace(p)

        vols = dict(side.get("volumes") or {})
        vols.update(added)
        side.update(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": self._dataset,
                "data_set_id": self._ds["data_set_id"],
                "release": new_vols[-1],
                "rows": n,
                "volumes": vols,
                "updated_at": _utcnow(),
            }
        )
        side.setdefault("built_at", side["updated_at"])
        self._write_sidecar(side)
        self._release = new_vols[-1]
        return new_vols

    def _write_sidecar(self, info: dict) -> None:
        self.meta_path.write_text(json.dumps(info, indent=2, sort_keys=True))

    def _max_volume_in_parquet(self) -> str | None:
        import duckdb

        row = duckdb.sql(
            f"SELECT max(VOLUME_ID) FROM read_parquet({_sql_str(self._ensure_parquet())})"
        ).fetchone()
        return row[0] if row and row[0] else None

    # DuckDB query interface
    def _connect(self):
        import duckdb

        con = duckdb.connect()
        _register_macros(con)
        con.execute(f"CREATE VIEW cumindex AS SELECT * FROM read_parquet({_sql_str(self._ensure_parquet())})")
        return con

    def query(self, sql: str) -> pd.DataFrame:
        """Run arbitrary DuckDB SQL against the Parquet database.

        Use the alias ``cumindex`` as the table name:

            ci.query("SELECT PRODUCT_ID, EMISSION_ANGLE FROM cumindex LIMIT 10")
        """
        con = self._connect()
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def lookup(
        self,
        product_ids: list[str],
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return metadata rows for the given product IDs.

        Queries the Parquet via DuckDB — no full dataset loaded into RAM.

        Parameters
        ----------
        product_ids : list[str]
            e.g. ``["M1524360231LE", "M1512659523RE"]``
        columns : list[str], optional
            Subset of columns to return.  ``PRODUCT_ID`` and
            ``FILE_SPECIFICATION_NAME`` are always included.
        """
        ids_clean = [p.strip().upper() for p in product_ids]
        if not ids_clean:
            return self._enrich(self.query("SELECT * FROM cumindex LIMIT 0"))
        ids_sql = ", ".join(_sql_str(i) for i in ids_clean)
        col_sql = _col_select(columns)
        sql = f"SELECT {col_sql} FROM cumindex WHERE PRODUCT_ID IN ({ids_sql})"
        return self._enrich(self.query(sql))

    def search(
        self,
        lat: tuple[float, float] | None = None,
        lon: tuple[float, float] | None = None,
        point: tuple[float, float] | None = None,
        radius_km: float = 0.0,
        emission_angle: tuple[float, float] | None = None,
        incidence_angle: tuple[float, float] | None = None,
        phase_angle: tuple[float, float] | None = None,
        nac_only: bool = False,
        columns: list[str] | None = None,
        wkt: bool = False,
        limit: int | None = None,
        aoi=None,
    ) -> pd.DataFrame:
        """Return images whose *footprint* intersects a region.

        Region: either a lat/lon box (``lat=(min, max)``, ``lon=(west, east)``;
        ``west > east`` crosses the 0/360 seam, longitudes may be given in
        −180..180 or 0..360, omitted ``lon`` = all longitudes), a circle
        (``point=(lat, lon)``, ``radius_km``) or an area of interest
        (``aoi=``: GeoJSON / WKT path, WKT string or
        [`Aoi`][lunar_toolkit.geometry.Aoi]).

        Matching is bounding-box based and conservative (may include a few
        false positives, never drops a true hit for convex-ish footprints):
        the image's lat range and seam-aware longitude arc must overlap the
        query's, and for images poleward of ±60° their polar stereographic
        x/y box must overlap the query's projected box.  Footprints that
        enclose a pole are treated as spanning all longitudes.  With ``aoi``
        the bbox matches are then refined exactly: each corner footprint
        polygon must intersect the AOI polygon (in polar stereographic when
        the AOI lies poleward of ±60°), and ``limit`` applies after that.

        Example
        -------
        ::

            df = ci.search(lat=(-90, -85), nac_only=True, incidence_angle=(80, 95))
            df = ci.search(lat=(-10, 10), lon=(355, 5))         # across 0°
            df = ci.search(point=(-89.67, 129.8), radius_km=15) # Shackleton
            df = ci.search(aoi="aoi.geojson")                   # QuickMap polygon
        """
        if aoi is not None:
            if lat is not None or lon is not None or point is not None:
                raise ValueError("Pass either aoi or lat/lon/point, not both")
            aoi = load_aoi(aoi)
            region = aoi_region(aoi)
        else:
            region = query_region(lat=lat, lon=lon, point=point, radius_km=radius_km)
        conditions = [region_sql(region)]
        conditions += _angle_conditions(emission_angle, incidence_angle, phase_angle)
        if nac_only:
            conditions.append(_NAC_ONLY_SQL)

        cols = _col_select(columns, extra=FOOTPRINT_COLUMNS)
        if wkt or aoi is not None:
            cols += f", {footprint_wkt_sql(self._parquet_columns())} AS FP_WKT_QUERY"
        sql = f"SELECT {cols} FROM cumindex WHERE " + " AND ".join(conditions)
        if limit and aoi is None:
            sql += f" LIMIT {int(limit)}"
        df = self._enrich(self.query(sql))
        if aoi is not None:
            df = filter_aoi(df, aoi, wkt_column="FP_WKT_QUERY")
            if limit:
                df = df.head(int(limit))
        if "FP_WKT_QUERY" in df.columns:
            query_wkt = df.pop("FP_WKT_QUERY")
            if wkt and "FP_WKT" not in df.columns:
                df["FP_WKT"] = query_wkt
        return df

    def _parquet_columns(self) -> list[str]:
        import duckdb

        rel = duckdb.sql(f"SELECT * FROM read_parquet({_sql_str(self._ensure_parquet())}) LIMIT 0")
        return list(rel.columns)

    def get_images(self, product_ids: list[str]) -> NacImageCollection:
        """Look up product IDs and return a NacImageCollection.

        Single DuckDB query; each NacImage is pre-populated with its metadata
        row so subsequent property accesses require zero additional DB calls.

        Parameters
        ----------
        product_ids : list[str]
            e.g. ``["M1524360231LE", "M1512659523RE"]``

        Returns
        -------
        NacImageCollection
            Contains only the IDs that were found in the index, in input order.
        """
        from .nac import NacImage, NacImageCollection  # deferred — avoids circular import

        df = self.lookup(product_ids)
        row_map = {str(row["PRODUCT_ID"]): row for _, row in df.iterrows()}
        ids = [pid.strip().upper() for pid in product_ids]
        return NacImageCollection(
            [NacImage(pid, cumindex=self, _row=row_map[pid]) for pid in ids if pid in row_map]
        )

    def load(self, nrows: int | None = None) -> pd.DataFrame:
        """Load the full index into memory as a DataFrame.

        For most workflows prefer ``lookup()`` or ``search()``.
        Pass ``nrows`` for a quick preview that needs no Parquet: only the
        label and the first ``nrows`` records of the TAB are fetched (HTTP
        Range request), then parsed with the same DuckDB extraction as
        ``build_db()``.
        """
        if nrows is not None:
            import duckdb

            lbl_text = self._read_lbl(self.lbl_url)
            meta = _parse_lbl_meta(lbl_text)
            columns = parse_lbl_columns(lbl_text)
            row_bytes = meta.get("record_bytes") or meta.get("row_bytes") or 903
            full = self._cache.cache_path(self.tab_url)
            if full.exists():
                tab = full
            else:
                tab = full.with_name(full.stem + f".head{int(nrows)}" + full.suffix)
                if not tab.exists():
                    _fetch_head(self.tab_url, tab, int(nrows) * int(row_bytes))
            con = duckdb.connect()
            try:
                _register_macros(con)
                sql = _select_sql([(tab, columns)], wkt=False)
                return con.execute(f"{sql} LIMIT {int(nrows)}").df()
            finally:
                con.close()

        import pyarrow.parquet as pq

        return pq.read_table(self._ensure_parquet()).to_pandas()

    # Enrichment
    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add PDS_URL and derived convenience columns to a query result."""
        import pandas as pd

        if df.empty:
            return df

        df = df.copy()

        if "FILE_SPECIFICATION_NAME" in df.columns:
            df["PDS_URL"] = df["FILE_SPECIFICATION_NAME"].map(
                lambda p: pds_url_from_file_spec(p) if isinstance(p, str) and p else None
            )
            df["LABEL_URL"] = df["PDS_URL"].map(
                lambda u: re.sub(r"\.[A-Za-z0-9]+$", ".xml", u) if u else None
            )
        if "CENTER_LATITUDE" in df.columns:
            df["LAT"] = pd.to_numeric(df["CENTER_LATITUDE"], errors="coerce")
        if "CENTER_LONGITUDE" in df.columns:
            df["LON"] = pd.to_numeric(df["CENTER_LONGITUDE"], errors="coerce")
        if "START_TIME" in df.columns:
            df["DATE"] = pd.to_datetime(df["START_TIME"], errors="coerce").dt.date
        if "PRODUCT_ID" in df.columns:
            pid = df["PRODUCT_ID"].astype(str)
            nac = [parse_nac_id(p) if is_nac_id(p) else None for p in pid]
            df["IS_NAC"] = [t is not None for t in nac]
            df["IS_WAC"] = pid.str.contains(r"[CMUV][EC]$") & ~df["IS_NAC"]
            df["NAC_SIDE"] = [t[1] if t else None for t in nac]
            df["PRODUCT_TYPE"] = pid.str[-1:].map(NAC_PRODUCT_TYPES).where(df["IS_NAC"] | df["IS_WAC"])

        return df.reset_index(drop=True)

    def to_observations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert query results to the common multi-instrument schema."""
        from ..observation import observations_from_cumindex

        return observations_from_cumindex(df, dataset=self._dataset)


# Parquet builder (DuckDB-native)
def build_parquet(
    sources: list[tuple[Path, list[dict]]],
    parquet_path: Path,
    base_parquet: Path | None = None,
    temp_dir: Path | None = None,
    memory_limit: str = "4GB",
    threads: int | None = None,
    wkt: bool = False,
    expected_rows: int | None = None,
) -> int:
    """Build a sorted, ZSTD-compressed Parquet from fixed-width PDS TAB file(s).

    Parameters
    ----------
    sources : list of (tab_path, lbl_columns)
        Each TAB is parsed with its own label layout (record lengths differ
        between volumes) and the results are combined ``UNION ALL BY NAME``.
    base_parquet : Path, optional
        Existing database to merge with (incremental update).  Rows are then
        de-duplicated on ``PRODUCT_ID``, keeping the newest ``VOLUME_ID``.

    Returns
    -------
    int
        Number of rows written.

    Notes
    -----
    Everything happens in one ``COPY (...) TO`` statement; DuckDB spills
    the sort to *temp_dir*.
    """
    import duckdb

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = parquet_path.with_name(parquet_path.name + ".tmp")

    con = duckdb.connect()
    try:
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory={_sql_str(temp_dir)}")
        con.execute(f"SET memory_limit={_sql_str(memory_limit)}")
        con.execute("SET preserve_insertion_order=false")
        if threads:
            con.execute(f"SET threads={int(threads)}")
        _register_macros(con)

        select = _select_sql(sources, wkt=wkt, base_parquet=base_parquet)
        names = ", ".join(Path(t).name for t, _ in sources)
        base = f" + {Path(base_parquet).name}" if base_parquet else ""
        log.info("Building %s from %s%s", parquet_path.name, names, base)
        if expected_rows:
            log.info("Expected rows: %s", f"{expected_rows:,}")

        con.execute(
            f"COPY ({select} ORDER BY CENTER_LATITUDE NULLS LAST, PRODUCT_ID) "
            f"TO {_sql_str(tmp_out)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        row = con.execute(f"SELECT count(*) FROM read_parquet({_sql_str(tmp_out)})").fetchone()
        n = row[0] if row else 0
    except BaseException:
        if tmp_out.exists():
            tmp_out.unlink()
        raise
    finally:
        con.close()

    tmp_out.replace(parquet_path)
    size_mb = parquet_path.stat().st_size / 1e6
    log.info("Done — %s rows → %s (%.1f MB)", f"{n:,}", parquet_path.name, size_mb)
    if expected_rows and n != expected_rows and base_parquet is None:
        log.warning("Expected %s rows, got %s", f"{expected_rows:,}", f"{n:,}")
    return int(n)


def _tab_source_sql(tab_path: Path) -> str:
    """DuckDB table function reading a fixed-width TAB as one VARCHAR column."""
    return (
        f"read_csv({_sql_str(tab_path)}, columns={{'line': 'VARCHAR'}}, "
        f"delim=chr(31), quote='', escape='', header=false, auto_detect=false, "
        f"encoding='latin-1')"
    )


def _select_sql(
    sources: list[tuple[Path, list[dict]]],
    wkt: bool = False,
    base_parquet: Path | None = None,
) -> str:
    """Full SELECT (extraction → union → dedup → footprint columns)."""
    parts = []
    all_names: list[str] = []
    for tab, cols in sources:
        exprs = ",\n       ".join(f"{column_sql(c)} AS {_ident(c['name'])}" for c in cols)
        parts.append(f"SELECT {exprs}\nFROM {_tab_source_sql(tab)}\nWHERE length(trim(line)) > 0")
        all_names += [c["name"] for c in cols if c["name"] not in all_names]
    raw = "\nUNION ALL BY NAME\n".join(f"({p})" for p in parts)
    if base_parquet is not None:
        # Incremental update: new rows win.  An anti-join against the (small)
        # set of new product IDs is far cheaper than a window over all rows.
        raw = (
            f"WITH new_rows AS (SELECT * FROM ({raw}) QUALIFY row_number() OVER "
            f"(PARTITION BY PRODUCT_ID ORDER BY VOLUME_ID DESC NULLS LAST) = 1)\n"
            f"SELECT * FROM (SELECT COLUMNS(c -> NOT starts_with(c, 'FP_')) "
            f"FROM read_parquet({_sql_str(base_parquet)}) "
            f"WHERE PRODUCT_ID NOT IN (SELECT PRODUCT_ID FROM new_rows)) "
            f"UNION ALL BY NAME SELECT * FROM new_rows"
        )
    fp = footprint_sql(all_names, wkt=wkt)
    return f"SELECT *{', ' + fp if fp else ''} FROM ({raw}) AS t"


def column_sql(col: dict) -> str:
    """SQL expression extracting one LBL column from the ``line`` VARCHAR.

    Bytes → ``substr(line, START_BYTE, BYTES)``, trimmed of blanks and
    quotes, typed by ``DATA_TYPE`` (falling back to name hints) with
    ``TRY_CAST``; ``NOT_APPLICABLE_CONSTANT`` / ``MISSING_CONSTANT`` → NULL.
    """
    start, nbytes = int(col["start_byte"]), int(col["bytes"])
    raw = f"nullif(trim(substr(line, {start}, {nbytes}), ' \"'), '')"
    kind = _column_kind(col)
    sentinels = [
        str(col[k]).strip().strip('"')
        for k in ("not_applicable_constant", "missing_constant", "unknown_constant")
        if col.get(k) not in (None, "")
    ]

    if kind in ("BIGINT", "DOUBLE"):
        expr = f"TRY_CAST({raw} AS {kind})"
        for s in sentinels:
            try:
                v = float(s)
            except ValueError:
                continue
            lit = str(int(v)) if kind == "BIGINT" and v.is_integer() else repr(v)
            expr = f"nullif({expr}, {lit})"
        return expr
    if kind == "TIMESTAMP":
        return f"TRY_CAST({raw} AS TIMESTAMP)"

    expr = raw
    for s in sentinels:
        expr = f"nullif({expr}, {_sql_str(s)})"
    if col["name"].upper() == "PRODUCT_ID":
        expr = f"upper({expr})"
    return expr


_NUMERIC_HINTS = [
    # Schema columns
    "COLUMN_NUMBER",
    "START_BYTE",
    "BYTES",
    # Image dimensions
    "LINES",
    "LINE_SAMPLES",
    "SAMPLE_BITS",
    # Angles / azimuths
    "_ANGLE",
    "_AZIMUTH",
    "SOLAR_LONGITUDE",
    # Spatial coordinates (all corners + center + subsolar + subspacecraft)
    "_LATITUDE",
    "_LONGITUDE",
    # Distance / altitude
    "ALTITUDE",
    "DISTANCE",
    # Resolution
    "PIXEL_SCALE",
    "SCALED_PIXEL",
    "RESOLUTION",
    # Timing
    "EXPOSURE_DURATION",
    # NAC instrument / temperatures
    "TEMPERATURE",
    "DAC_RESET",
    "CHANNEL_A_OFFSET",
    "CHANNEL_B_OFFSET",
    "MODE_CODE",
    "SELECT_CODE",
    # Orbit
    "ORBIT_NUMBER",
]

_TIME_COLUMNS = {"START_TIME", "STOP_TIME", "NAC_PREROLL_START_TIME", "PRODUCT_CREATION_TIME"}


def _column_kind(col: dict) -> str:
    """DuckDB type for an LBL column: LBL DATA_TYPE first, then name hints."""
    name = col["name"].upper()
    dtype = str(col.get("data_type") or "").upper()
    if dtype in (
        "ASCII_INTEGER",
        "INTEGER",
        "MSB_INTEGER",
        "LSB_INTEGER",
        "UNSIGNED_INTEGER",
        "ASCII_UNSIGNED_INTEGER",
    ):
        return "BIGINT"
    if dtype in ("ASCII_REAL", "REAL", "FLOAT", "IEEE_REAL", "PC_REAL"):
        return "DOUBLE"
    if dtype in ("TIME", "DATE", "ASCII_TIME", "ASCII_DATE_TIME") or name in _TIME_COLUMNS:
        return "TIMESTAMP"
    if dtype in ("", "UNKNOWN") and any(h in name for h in _NUMERIC_HINTS):
        return "DOUBLE"
    return "VARCHAR"


# Footprints
_MACROS = [
    # longitude → [0, 360)
    "CREATE OR REPLACE MACRO _lon360(x) AS ((x % 360.0) + 360.0) % 360.0",
    # signed shortest longitude step a → b in [-180, 180)
    "CREATE OR REPLACE MACRO _dlon(a, b) AS ((_lon360(b) - _lon360(a) + 540.0) % 360.0) - 180.0",
    # polar stereographic radius (m) for hemisphere 'S' / 'N', true scale at the pole
    f"CREATE OR REPLACE MACRO _ps_rho(lat, hemi) AS 2.0 * {MOON_RADIUS_KM * 1000.0!r} * "
    "tan(pi() / 4.0 + (CASE WHEN hemi = 'S' THEN 1.0 ELSE -1.0 END) * radians(lat) / 2.0)",
    "CREATE OR REPLACE MACRO _ps_x(lat, lon, hemi) AS _ps_rho(lat, hemi) * sin(radians(lon))",
    "CREATE OR REPLACE MACRO _ps_y(lat, lon, hemi) AS "
    "(CASE WHEN hemi = 'S' THEN 1.0 ELSE -1.0 END) * _ps_rho(lat, hemi) * cos(radians(lon))",
    # does arc [w1, w1+s1] overlap arc [w2, w2+s2]? (degrees, s in [0, 360])
    "CREATE OR REPLACE MACRO _arc_overlap(w1, s1, w2, s2) AS "
    "(s1 >= 360 OR s2 >= 360 OR _lon360(w2 - w1) <= s1 OR _lon360(w1 - w2) <= s2)",
]


def _register_macros(con) -> None:
    for m in _MACROS:
        con.execute(m)


def _corner_pairs(names: Iterable[str]) -> list[tuple[str, str]] | None:
    have = set(names)
    pairs = [(f"{c}_LATITUDE", f"{c}_LONGITUDE") for c in FOOTPRINT_CORNERS]
    if all(a in have and b in have for a, b in pairs):
        return pairs
    return None


def footprint_sql(names: Iterable[str], wkt: bool = False) -> str:
    """Derived ``FP_*`` column expressions over an extracted-row relation.

    * ``FP_MIN_LAT`` / ``FP_MAX_LAT`` — corner (and centre) latitude range,
      extended to ±90 when the footprint encloses a pole.
    * ``FP_LON_WEST`` / ``FP_LON_EAST`` / ``FP_LON_SPAN`` — the shortest
      longitude arc covering all corners, in [0, 360); ``WEST > EAST`` means
      the arc crosses the 0/360 seam; ``SPAN = 360`` means all longitudes.
    * ``FP_CONTAINS_POLE`` — winding number of the corner ring ≠ 0.
    * ``FP_PS_*`` — polar stereographic bbox (m, R = 1737.4 km) for images
      whose centre is poleward of ±60° (``FP_PS_HEMI`` = ``'S'`` / ``'N'``).
    """
    names = list(names)
    if "CENTER_LATITUDE" not in names or "CENTER_LONGITUDE" not in names:
        return ""
    clat, clon = "CENTER_LATITUDE", "CENTER_LONGITUDE"
    pairs = _corner_pairs(names)
    if pairs:
        lats = [f"coalesce({a}, {clat})" for a, _ in pairs]
        lons = [f"coalesce({b}, {clon})" for _, b in pairs]
    else:
        lats, lons = [clat], [clon]

    l360 = "[" + ", ".join(f"_lon360({x})" for x in lons) + "]"
    l180 = "[" + ", ".join(f"(_lon360({x}) + 180.0) % 360.0 - 180.0" for x in lons) + "]"
    span0 = f"(list_max({l360}) - list_min({l360}))"
    span180 = f"(list_max({l180}) - list_min({l180}))"
    if pairs:
        ring = lons + [lons[0]]
        winding = " + ".join(f"_dlon({ring[i]}, {ring[i + 1]})" for i in range(len(lons)))
        pole = f"(abs({winding}) > 180.0)"
    else:
        pole = "false"
    west = (
        f"CASE WHEN {pole} THEN 0.0 WHEN {span0} <= {span180} THEN list_min({l360}) "
        f"ELSE _lon360(list_min({l180})) END"
    )
    span = f"CASE WHEN {pole} THEN 360.0 WHEN {span0} <= {span180} THEN {span0} ELSE {span180} END"
    lat_list = "[" + ", ".join(lats + [clat]) + "]"

    hemi = (
        f"CASE WHEN {clat} < -{POLAR_LAT_THRESHOLD} THEN 'S' WHEN {clat} > {POLAR_LAT_THRESHOLD} THEN 'N' END"
    )
    xs = "[" + ", ".join(f"_ps_x({a}, {b}, FP_PS_HEMI)" for a, b in zip(lats, lons)) + "]"
    ys = "[" + ", ".join(f"_ps_y({a}, {b}, FP_PS_HEMI)" for a, b in zip(lats, lons)) + "]"

    # Two-level: the inner block defines pole/hemi aliases, reused below via
    # DuckDB's lateral column aliases.
    exprs = [
        f"{pole} AS FP_CONTAINS_POLE",
        f"CASE WHEN FP_CONTAINS_POLE AND {clat} < 0 THEN -90.0 ELSE list_min({lat_list}) END AS FP_MIN_LAT",
        f"CASE WHEN FP_CONTAINS_POLE AND {clat} >= 0 THEN 90.0 ELSE list_max({lat_list}) END AS FP_MAX_LAT",
        f"{west} AS FP_LON_WEST",
        f"{span} AS FP_LON_SPAN",
        "CASE WHEN FP_LON_SPAN >= 360 THEN 360.0 ELSE _lon360(FP_LON_WEST + FP_LON_SPAN) END AS FP_LON_EAST",
        f"{hemi} AS FP_PS_HEMI",
        f"CASE WHEN FP_PS_HEMI IS NOT NULL THEN list_min({xs}) END AS FP_PS_XMIN",
        f"CASE WHEN FP_PS_HEMI IS NOT NULL THEN list_max({xs}) END AS FP_PS_XMAX",
        f"CASE WHEN FP_PS_HEMI IS NOT NULL THEN list_min({ys}) END AS FP_PS_YMIN",
        f"CASE WHEN FP_PS_HEMI IS NOT NULL THEN list_max({ys}) END AS FP_PS_YMAX",
    ]
    if wkt:
        exprs.append(f"{footprint_wkt_sql(names, west_expr='FP_LON_WEST')} AS FP_WKT")
    return ",\n       ".join(exprs)


def footprint_wkt_sql(names: Iterable[str], west_expr: str = "FP_LON_WEST") -> str:
    """SQL for a ``POLYGON((lon lat, ...))`` WKT of the corner ring.

    Longitudes are unwrapped relative to the footprint's western edge so the
    ring never jumps at the seam (values may exceed 360).  For pole-enclosing
    footprints the lat/lon ring is only indicative — use the ``FP_PS_*`` box.
    """
    pairs = _corner_pairs(names)
    if not pairs:
        return "NULL"
    pts = []
    for a, b in pairs + [pairs[0]]:
        lon = f"round({west_expr} + _lon360(round({b} - {west_expr}, 6)), 4)"
        pts.append(f"CAST({lon} AS VARCHAR) || ' ' || CAST(round({a}, 4) AS VARCHAR)")
    return "'POLYGON((' || " + " || ', ' || ".join(pts) + " || '))'"


# Search regions
def region_sql(region: dict) -> str:
    """WHERE clause matching ``FP_*`` columns against a
    [`query_region`][lunar_toolkit.geometry.query_region]."""
    s, n = region["lat"]
    conds = [f"FP_MAX_LAT >= {s!r}", f"FP_MIN_LAT <= {n!r}"]
    if region["span"] < 360.0:
        conds.append(f"_arc_overlap(FP_LON_WEST, FP_LON_SPAN, {region['west']!r}, {region['span']!r})")
    for hemi, (x0, x1, y0, y1) in region["ps"].items():
        conds.append(
            f"(FP_PS_HEMI IS DISTINCT FROM '{hemi}' OR (FP_PS_XMAX >= {x0!r} AND FP_PS_XMIN <= {x1!r} "
            f"AND FP_PS_YMAX >= {y0!r} AND FP_PS_YMIN <= {y1!r}))"
        )
    return "(" + " AND ".join(conds) + ")"


# Volume discovery / housekeeping
_VOLUME_RE = re.compile(r"^LROLRC_(\d{4})([A-Z]?)$")


def _norm_volume(v: str) -> str:
    return str(v).strip().strip("/").upper()


def _volume_key(v: str) -> tuple[int, str]:
    m = _VOLUME_RE.match(_norm_volume(v))
    if not m:
        return (-1, _norm_volume(v))
    return (int(m.group(1)), m.group(2))


def parse_s3_listing(xml_text: str) -> tuple[list[str], str | None]:
    """Return ``(common_prefixes, next_continuation_token)`` from ListObjectsV2 XML."""
    root = ET.fromstring(xml_text)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
    prefixes = [el.text for el in root.iter(f"{ns}Prefix") if el.text and el is not root.find(f"{ns}Prefix")]
    token = None
    if (root.findtext(f"{ns}IsTruncated") or "").lower() == "true":
        token = root.findtext(f"{ns}NextContinuationToken")
    return prefixes, token


def list_store_volumes(base_url: str, volume_re: str, timeout: float = 30.0) -> list[str]:
    """List volume directories under a dataset on the PDS S3 store."""
    if not base_url.startswith(PDS_STORE_URL):
        raise ValueError(f"Not a PDS store URL: {base_url}")
    prefix = base_url[len(PDS_STORE_URL) :]
    pat = re.compile(rf"^{re.escape(prefix)}({volume_re})/$")
    vols: list[str] = []
    token: str | None = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            q["continuation-token"] = token
        url = PDS_STORE_URL + "?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            prefixes, token = parse_s3_listing(resp.read().decode("utf-8", "replace"))
        vols += [m.group(1) for p in prefixes if (m := pat.match(p))]
        if not token:
            break
    return sorted(set(vols), key=_volume_key)


def probe_volumes(
    base_url: str,
    start: str,
    max_numbers: int = 12,
    timeout: float = 15.0,
) -> list[str]:
    """Probe ``<vol>/INDEX/INDEX.LBL`` for volumes at/after *start*.

    Volume names look like ``LROLRC_0066C`` (number + optional A/B/C… part
    letter).  Stops after two consecutive numbers without any hit.
    """
    num, _ = _volume_key(start)
    m = re.match(r"^(LROLRC_)(\d{4})", _norm_volume(start))
    if not m or num < 0:
        return [_norm_volume(start)]
    prefix = m.group(1)
    found = [_norm_volume(start)]
    misses = 0
    for n in range(num, num + max_numbers):
        hit = False
        for suffix in ("", "A", "B", "C", "D", "E"):
            vol = f"{prefix}{n:04d}{suffix}"
            if _volume_key(vol) <= _volume_key(start):
                continue
            if _url_exists(base_url + vol + "/" + INDEX_DIR + VOLINDEX_LBL_FILENAME, timeout):
                found.append(vol)
                hit = True
        if n > num:
            misses = 0 if hit else misses + 1
            if misses >= 2:
                break
    return sorted(set(found), key=_volume_key)


def _url_exists(url: str, timeout: float = 15.0) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except urllib.error.HTTPError:
        return False


def _fetch_head(url: str, dest: Path, nbytes: int, timeout: float = 60.0) -> Path:
    """Fetch the first *nbytes* of *url* (HTTP Range) into *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(nbytes)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def clean_stale_build_files(directory: Path) -> list[Path]:
    """Delete leftovers of interrupted builds (``*.build.duckdb[.wal]``, ``*.tmp``)."""
    removed: list[Path] = []
    directory = Path(directory)
    if not directory.is_dir():
        return removed
    for pattern in ("*.build.duckdb", "*.build.duckdb.wal", "*.parquet.tmp", "*.update.parquet"):
        for f in directory.glob(pattern):
            try:
                f.unlink()
                removed.append(f)
                log.info("Removed stale build file %s", f)
            except OSError:
                pass
    return removed


# LBL parsing
def parse_lbl_columns(lbl_content: str) -> list[dict]:
    """Parse ``OBJECT = COLUMN`` blocks of a PDS3 label (pure Python).

    Keys are lower-cased LBL keywords; ``column_number``, ``start_byte`` and
    ``bytes`` are ints.  Sorted by ``column_number``.
    """
    blocks = re.findall(
        r"OBJECT\s*=\s*COLUMN(.*?)END_OBJECT\s*=\s*COLUMN",
        lbl_content,
        re.DOTALL,
    )

    def _parse_value(val: str) -> str:
        val = val.strip().strip('"')
        return re.sub(r"\s*\n\s*", " ", val).strip()

    pattern = re.compile(r"([A-Z_]+)\s*=\s*(.*?)(?=\n\s*[A-Z_]+\s*=|\Z)", re.DOTALL)
    records = []
    for block in blocks:
        rec: dict[str, Any] = {
            m.group(1).strip().lower(): _parse_value(m.group(2)) for m in pattern.finditer(block)
        }
        for k in ("column_number", "start_byte", "bytes"):
            if k in rec:
                try:
                    rec[k] = int(rec[k])
                except ValueError:
                    rec[k] = None
        if rec.get("name") and rec.get("start_byte") and rec.get("bytes"):
            records.append(rec)
    return sorted(records, key=lambda r: r.get("column_number") or 0)


def _parse_pds_lbl_columns(lbl_content: str) -> pd.DataFrame:
    """``parse_lbl_columns`` as a DataFrame (the ``CumIndex.schema`` format)."""
    import pandas as pd

    df = pd.DataFrame(parse_lbl_columns(lbl_content))
    for col in ("column_number", "start_byte", "bytes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    preferred = ["name", "column_number", "data_type", "start_byte", "bytes", "format", "description"]
    ordered = [c for c in preferred if c in df.columns]
    extras = [c for c in df.columns if c not in preferred]
    return df[ordered + extras].reset_index(drop=True)


def _parse_lbl_meta(lbl_content: str) -> dict:
    """Top-level label fields (ints for sizes, strings for ids / checksum)."""
    meta: dict = {}
    for key in ("RECORD_BYTES", "FILE_RECORDS", "ROWS", "COLUMNS", "ROW_BYTES"):
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", lbl_content, re.MULTILINE)
        if m:
            meta[key.lower()] = int(m.group(1))
    for key in ("MD5_CHECKSUM", "VOLUME_ID", "DATA_SET_ID", "PRODUCT_CREATION_TIME"):
        m = re.search(rf'^\s*{key}\s*=\s*"?([^"\r\n]+?)"?\s*$', lbl_content, re.MULTILINE)
        if m:
            meta[key.lower()] = m.group(1).strip()
    return meta


# Private helpers
_NAC_ONLY_SQL = "(" + " OR ".join(f"PRODUCT_ID LIKE '%{s}'" for s in NAC_SUFFIXES) + ")"


def _angle_conditions(
    emission_angle: tuple[float, float] | None,
    incidence_angle: tuple[float, float] | None,
    phase_angle: tuple[float, float] | None,
) -> list[str]:
    out = []
    for col, rng in (
        ("EMISSION_ANGLE", emission_angle),
        ("INCIDENCE_ANGLE", incidence_angle),
        ("PHASE_ANGLE", phase_angle),
    ):
        if rng is not None:
            out.append(f"{col} BETWEEN {float(rng[0])} AND {float(rng[1])}")
    return out


def _col_select(columns: list[str] | None, extra: list[str] | None = None) -> str:
    """Build a SELECT column list, always including the key identity columns."""
    if not columns:
        return "*"
    always = {"PRODUCT_ID", "FILE_SPECIFICATION_NAME", "CENTER_LATITUDE", "CENTER_LONGITUDE", "START_TIME"}
    return ", ".join(_ident(c) for c in sorted(always | set(columns) | set(extra or [])))


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_str(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
