"""DuckDB workspace (``<work_dir>/lunar.duckdb``): image footprints, stereo overlaps and bundle-adjustment statistics."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import MOON_RADIUS_KM, parse_nac_id

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .asp import MatchOffsetStats, ResidualStats
    from .sfs import SfsConfig

log = logging.getLogger(__name__)

DB_NAME = "lunar.duckdb"
# lunar pairs --engine auto uses DuckDB from this many images up
PAIRS_ENGINE_THRESHOLD = 300
# Footprint edges are great circles (as in find_stereo_pairs), densified at this step (deg)
GREAT_CIRCLE_STEP_DEG = 0.1
# The sphere and projections of the images.footprint geometries (by images.hemi)
LONLAT_PROJ4 = f"+proj=longlat +R={MOON_RADIUS_KM * 1000:g} +no_defs"
PS_PROJ4 = {
    hemi: f"+proj=stere +lat_0={lat} +lon_0=0 +k=1 +x_0=0 +y_0=0 +R={MOON_RADIUS_KM * 1000:g} +units=m +no_defs"
    for hemi, lat in (("S", -90), ("N", 90))
}
BA_TABLES = (
    "ba_residual_stats",
    "ba_pixel_residuals",
    "ba_match_offsets",
    "ba_points",
    "ba_convergence",
    "ba_camera_offsets",
)
_R_M = MOON_RADIUS_KM * 1000.0
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_OBS_TYPES = {
    "product_id": "VARCHAR",
    "instrument": "VARCHAR",
    "product_type": "VARCHAR",
    "start_time": "TIMESTAMP",
    "center_lat": "DOUBLE",
    "center_lon": "DOUBLE",
    "min_lat": "DOUBLE",
    "max_lat": "DOUBLE",
    "lon_west": "DOUBLE",
    "lon_east": "DOUBLE",
    "footprint_wkt": "VARCHAR",
    "incidence": "DOUBLE",
    "emission": "DOUBLE",
    "phase": "DOUBLE",
    "sub_solar_azimuth": "DOUBLE",
    "north_azimuth": "DOUBLE",
    "sub_spacecraft_lat": "DOUBLE",
    "sub_spacecraft_lon": "DOUBLE",
    "spacecraft_altitude": "DOUBLE",
    "pixel_scale": "DOUBLE",
    "url": "VARCHAR",
    "label_url": "VARCHAR",
    "source": "VARCHAR",
}
_IMAGE_EXTRA = {
    "obs_key": "VARCHAR",
    "hemi": "VARCHAR",
    "cap_lat": "DOUBLE",
    "cap_lon": "DOUBLE",
    "footprint_km2": "DOUBLE",
    "xmin": "DOUBLE",
    "xmax": "DOUBLE",
    "ymin": "DOUBLE",
    "ymax": "DOUBLE",
}

_BA_DDL = [
    "CREATE TABLE IF NOT EXISTS ba_runs (run VARCHAR, prefix VARCHAR, loaded_at TIMESTAMP, files VARCHAR[])",
    "CREATE TABLE IF NOT EXISTS ba_residual_stats (run VARCHAR, stage VARCHAR, image VARCHAR, "
    "image_path VARCHAR, product_id VARCHAR, mean_px DOUBLE, median_px DOUBLE, residual_count BIGINT)",
    "CREATE TABLE IF NOT EXISTS ba_pixel_residuals (run VARCHAR, stage VARCHAR, image VARCHAR, "
    "product_id VARCHAR, norm_px DOUBLE)",
    "CREATE TABLE IF NOT EXISTS ba_match_offsets (run VARCHAR, image VARCHAR, image_path VARCHAR, "
    "product_id VARCHAR, p25 DOUBLE, p50 DOUBLE, p75 DOUBLE, p85 DOUBLE, p95 DOUBLE, match_count BIGINT)",
    "CREATE TABLE IF NOT EXISTS ba_points (run VARCHAR, stage VARCHAR, lon DOUBLE, lat DOUBLE, "
    "height_m DOUBLE, mean_residual_px DOUBLE, num_observations INTEGER, kind VARCHAR, hemi VARCHAR, "
    "x DOUBLE, y DOUBLE)",
    "CREATE TABLE IF NOT EXISTS ba_convergence (run VARCHAR, left_image VARCHAR, right_image VARCHAR, "
    "left_product_id VARCHAR, right_product_id VARCHAR, p25 DOUBLE, p50 DOUBLE, p75 DOUBLE, matches BIGINT)",
    "CREATE TABLE IF NOT EXISTS ba_camera_offsets (run VARCHAR, image VARCHAR, product_id VARCHAR, "
    "horizontal_m DOUBLE, vertical_m DOUBLE)",
    "CREATE TABLE IF NOT EXISTS rejection_params (max_offset_p85 DOUBLE, "
    "max_large_residual_fraction DOUBLE, large_residual_px DOUBLE, min_matches BIGINT)",
]

_PS_MACROS = [
    # areal scale of the polar stereographic projection is _ps_k^2 (k = 1 at the pole)
    f"CREATE OR REPLACE MACRO _ps_k(x, y) AS 1.0 + (x * x + y * y) / {4.0 * _R_M * _R_M!r}",
    f"CREATE OR REPLACE MACRO _ps_lat(x, y, hemi) AS (CASE WHEN hemi = 'S' THEN 1.0 ELSE -1.0 END) * "
    f"(degrees(2.0 * atan(sqrt(x * x + y * y) / {2.0 * _R_M!r})) - 90.0)",
    "CREATE OR REPLACE MACRO _ps_lon(x, y, hemi) AS "
    "(degrees(atan2(x, CASE WHEN hemi = 'S' THEN y ELSE -y END)) + 360.0) % 360.0",
    "CREATE OR REPLACE MACRO _az_diff(a, b) AS least(((b - a) % 360.0 + 360.0) % 360.0, "
    "360.0 - ((b - a) % 360.0 + 360.0) % 360.0)",
    "CREATE OR REPLACE MACRO _true_km2(g) AS "
    "ST_Area(g) / pow(_ps_k(ST_X(ST_Centroid(g)), ST_Y(ST_Centroid(g))), 2) / 1e6",
]

# Per-image residual stats before / after BA; {params} is a rejection_params-shaped relation.
_IMAGE_STATS_SQL = """
WITH names AS (
    SELECT DISTINCT run, image FROM (
        SELECT run, image FROM ba_residual_stats
        UNION ALL SELECT run, image FROM ba_match_offsets
        UNION ALL SELECT DISTINCT run, image FROM ba_pixel_residuals
    )
),
raw AS (
    SELECT r.run, r.image,
           count(*) FILTER (WHERE r.stage = 'final') AS raw_count,
           avg(CAST(r.norm_px > p.large_residual_px AS DOUBLE)) FILTER (WHERE r.stage = 'final')
               AS large_fraction,
           count(*) FILTER (WHERE r.stage = 'initial') AS raw_count_initial,
           avg(CAST(r.norm_px > p.large_residual_px AS DOUBLE)) FILTER (WHERE r.stage = 'initial')
               AS large_fraction_initial
    FROM ba_pixel_residuals r CROSS JOIN ({params}) p GROUP BY r.run, r.image
)
SELECT n.run, n.image, upper(split_part(n.image, '.', 1)) AS product_id,
       i.mean_px AS initial_mean, i.median_px AS initial_median, i.residual_count AS initial_count,
       f.mean_px AS final_mean, f.median_px AS final_median, f.residual_count AS final_count,
       f.median_px - i.median_px AS median_change,
       o.p25 AS offset_p25, o.p50 AS offset_p50, o.p75 AS offset_p75, o.p85 AS offset_p85,
       o.p95 AS offset_p95, o.match_count AS offset_count,
       r.raw_count, r.large_fraction, r.raw_count_initial, r.large_fraction_initial,
       c.horizontal_m AS camera_offset_horizontal_m, c.vertical_m AS camera_offset_vertical_m
FROM names n
LEFT JOIN ba_residual_stats i ON i.run = n.run AND i.image = n.image AND i.stage = 'initial'
LEFT JOIN ba_residual_stats f ON f.run = n.run AND f.image = n.image AND f.stage = 'final'
LEFT JOIN ba_match_offsets o ON o.run = n.run AND o.image = n.image
LEFT JOIN raw r ON r.run = n.run AND r.image = n.image
LEFT JOIN ba_camera_offsets c ON c.run = n.run AND c.image = n.image
"""

# Bertone et al. (2026, Sec. 2) rejection rules, first match wins (as sfs.flag_images).
_REJECTIONS_SQL = """
SELECT run, image, product_id, rule, value, threshold,
       CASE rule
           WHEN 'matches' THEN printf('%d matches', CAST(value AS BIGINT))
           WHEN 'offset' THEN printf('match offset p85 %.2f m > %g m', value, threshold)
           ELSE printf('%.0f%% of residuals > %g px', 100.0 * value, large_residual_px)
       END AS reason
FROM (
    SELECT s.*, p.large_residual_px,
           CASE WHEN s.min_count < p.min_matches THEN 'matches'
                WHEN s.offset_p85 > p.max_offset_p85 THEN 'offset'
                WHEN s.raw_count > 0 AND s.large_fraction > p.max_large_residual_fraction THEN 'residuals'
           END AS rule,
           CASE rule WHEN 'matches' THEN s.min_count WHEN 'offset' THEN s.offset_p85
                ELSE s.large_fraction END AS value,
           CASE rule WHEN 'matches' THEN p.min_matches WHEN 'offset' THEN p.max_offset_p85
                ELSE p.max_large_residual_fraction END AS threshold
    FROM (SELECT *, least(final_count, offset_count) AS min_count FROM ({stats})) s CROSS JOIN ({params}) p
)
WHERE rule IS NOT NULL
"""

_RUN_SUMMARY_SQL = """
WITH s AS (
    SELECT run,
           count(*) AS images,
           avg(initial_median) AS initial_median_mean,
           stddev_pop(initial_median) AS initial_median_std,
           median(initial_median) AS initial_median_median,
           avg(final_median) AS final_median_mean,
           stddev_pop(final_median) AS final_median_std,
           median(final_median) AS final_median_median,
           median(offset_p85) AS offset_p85_median,
           max(offset_p85) AS offset_p85_max,
           count(*) FILTER (WHERE offset_p85 > (SELECT max_offset_p85 FROM rejection_params))
               AS offset_p85_over_threshold,
           median(offset_p95) AS offset_p95_median,
           max(offset_p95) AS offset_p95_max
    FROM ba_image_stats GROUP BY run
),
rej AS (SELECT run, count(*) AS rejected FROM image_rejections GROUP BY run),
conv AS (SELECT run, count(*) AS convergence_pairs, median(p50) AS convergence_p50_median
         FROM ba_convergence GROUP BY run),
pts AS (
    SELECT run,
           count(*) FILTER (WHERE stage = 'initial') AS points_initial,
           count(*) FILTER (WHERE stage = 'final') AS points_final,
           median(mean_residual_px) FILTER (WHERE stage = 'initial') AS point_residual_median_initial,
           median(mean_residual_px) FILTER (WHERE stage = 'final') AS point_residual_median_final
    FROM ba_points GROUP BY run
)
SELECT s.*, coalesce(rej.rejected, 0) AS rejected, conv.convergence_pairs, conv.convergence_p50_median,
       pts.points_initial, pts.points_final, pts.point_residual_median_initial,
       pts.point_residual_median_final
FROM s LEFT JOIN rej USING (run) LEFT JOIN conv USING (run) LEFT JOIN pts USING (run)
ORDER BY s.run
"""

_PARAMS_VIEW = "SELECT * FROM rejection_params"
_PARAMS_MACRO = (
    "SELECT p85 AS max_offset_p85, frac AS max_large_residual_fraction, px AS large_residual_px, "
    "min_matches AS min_matches"
)


class WorkspaceError(RuntimeError):
    """The workspace cannot do what was asked (e.g. no spatial extension)."""


def workspace_path(work_dir: Path | str) -> Path:
    """``<work_dir>/lunar.duckdb``, or *work_dir* itself when it names a ``.duckdb`` file."""
    p = Path(work_dir)
    return p if p.suffix == ".duckdb" else p / DB_NAME


def spatial_available() -> bool:
    """Whether DuckDB's spatial extension loads (installing it when needed)."""
    global _SPATIAL
    if _SPATIAL is None:
        import duckdb

        con = duckdb.connect()
        try:
            _SPATIAL = _load_spatial(con) is None
        finally:
            con.close()
    return _SPATIAL


_SPATIAL: bool | None = None


def _load_spatial(con: Any) -> str | None:
    """Load (installing if needed) the spatial extension; the error message, or None."""
    import duckdb

    try:
        con.execute("LOAD spatial")
        return None
    except duckdb.Error:
        pass
    try:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
        return None
    except duckdb.Error as exc:
        return str(exc).splitlines()[0]


def _ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Not a table name: {name!r}")
    return name


def _sql_str(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _num(v: float) -> float | None:
    """*v*, None for NaN (SQL NULL)."""
    return None if v is None or math.isnan(v) else float(v)


def _product_id(image: str) -> str:
    return Path(str(image)).name.split(".")[0].upper()


class Workspace:
    """A DuckDB database of images, stereo pairs and bundle-adjustment results.

    Parameters
    ----------
    path : Path or str
        Database file (``":memory:"`` for a scratch one); a directory means
        ``<path>/lunar.duckdb``.  Parent directories are created.
    memory_limit : str
        DuckDB ``memory_limit``.
    threads : int, optional
        DuckDB ``threads`` (default: DuckDB's, all cores).
    read_only : bool
        Open an existing database read-only (no schema changes).
    spatial : bool
        Load the spatial extension (``INSTALL spatial`` once needs network; it is
        cached in ``~/.duckdb``).  Without it the bundle-adjustment tables and views
        work, while :meth:`add_observations` and :meth:`pairs` raise
        :class:`WorkspaceError`.

    Notes
    -----
    Tables: ``images`` (observation columns plus ``footprint``, a GEOMETRY in metres
    in the polar stereographic projection of ``hemi``, :data:`PS_PROJ4`),
    ``pairs`` (the last :meth:`pairs` result), ``ba_runs``, ``ba_residual_stats``,
    ``ba_pixel_residuals``, ``ba_match_offsets``, ``ba_points``, ``ba_convergence``,
    ``ba_camera_offsets`` (by ``run``) and ``rejection_params``.  Views:
    ``ba_image_stats`` (per image before / after), ``ba_run_summary`` and
    ``image_rejections`` (Bertone et al. 2026 rules with ``rejection_params``); the
    table macro ``image_rejections_at(max_offset_p85, max_large_residual_fraction,
    large_residual_px, min_matches)`` takes other thresholds.

    Example
    -------
    ::

        with Workspace("work/malapert") as ws:
            ws.add_observations(obs)
            pairs = ws.pairs(aoi={"point": (-85.9, 0.0), "radius_km": 5})
            ws.load_bundle_adjust("work/malapert/ba/run")
            ws.query("SELECT * FROM image_rejections")
    """

    def __init__(
        self,
        path: Path | str,
        *,
        memory_limit: str = "1GB",
        threads: int | None = None,
        read_only: bool = False,
        spatial: bool = True,
    ) -> None:
        import duckdb

        self.path = path if str(path) == ":memory:" else workspace_path(path)
        if not read_only and isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {"memory_limit": memory_limit}
        if threads:
            config["threads"] = int(threads)
        self.read_only = read_only
        self.con = duckdb.connect(str(self.path), read_only=read_only, config=config)
        self.con.execute("SET enable_progress_bar = false")
        self.spatial_error = _load_spatial(self.con) if spatial else "not requested"
        if not read_only:
            self._create_schema()

    @property
    def has_spatial(self) -> bool:
        return self.spatial_error is None

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def _require_spatial(self) -> None:
        if self.spatial_error is not None:
            raise WorkspaceError(
                f"DuckDB spatial extension unavailable ({self.spatial_error}). Install it once "
                "with network access: python -c \"import duckdb; duckdb.sql('INSTALL spatial')\""
            )

    def _create_schema(self) -> None:
        from .lroc.cumindex import _MACROS

        for sql in [*_MACROS, *_BA_DDL]:
            self.con.execute(sql)
        if self.has_spatial:
            for sql in _PS_MACROS:
                self.con.execute(sql)
        if not self._scalar("SELECT count(*) FROM rejection_params"):
            self.set_rejection_params()
        stats = _IMAGE_STATS_SQL.format(params=_PARAMS_VIEW)
        rejections = _REJECTIONS_SQL.format(params=_PARAMS_VIEW, stats="SELECT * FROM ba_image_stats")
        macro = _REJECTIONS_SQL.format(
            params=_PARAMS_MACRO, stats=_IMAGE_STATS_SQL.format(params=_PARAMS_MACRO)
        )
        for sql in (
            f"CREATE OR REPLACE VIEW ba_image_stats AS {stats}",
            f"CREATE OR REPLACE VIEW image_rejections AS {rejections}",
            f"CREATE OR REPLACE MACRO image_rejections_at(p85, frac, px, min_matches) AS TABLE {macro}",
            f"CREATE OR REPLACE VIEW ba_run_summary AS {_RUN_SUMMARY_SQL}",
        ):
            self.con.execute(sql)

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run SQL; returns the result as a DataFrame (empty for statements without one)."""
        import pandas as pd

        cur = self.con.execute(sql, params) if params is not None else self.con.execute(sql)
        return cur.df() if cur.description else pd.DataFrame()

    def tables(self) -> pd.DataFrame:
        """``name``, ``type`` (table / view) and ``rows`` of the workspace's tables and views."""
        import pandas as pd

        rows = self.con.execute(
            "SELECT table_name, CASE table_type WHEN 'VIEW' THEN 'view' ELSE 'table' END "
            "FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_type, table_name"
        ).fetchall()
        out = []
        for name, kind in rows:
            n = self._scalar(f'SELECT count(*) FROM "{name}"')
            out.append({"name": name, "type": kind, "rows": int(n)})
        return pd.DataFrame(out, columns=["name", "type", "rows"])

    def _scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        row = (
            self.con.execute(sql, params).fetchone()
            if params is not None
            else self.con.execute(sql).fetchone()
        )
        return None if row is None else row[0]

    def has_table(self, name: str) -> bool:
        return bool(
            self._scalar(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
                [name],
            )
        )

    def export(self, table: str, path: Path | str) -> Path:
        """Write a table or view to Parquet (``.parquet``) or CSV (anything else).

        GEOMETRY columns are written as WKT in CSV and as WKB (GeoParquet) in Parquet.
        """
        name = _ident(table)
        if not self.has_table(name):
            raise WorkspaceError(f"No table or view {name!r} in {self.path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parquet = path.suffix.lower() in (".parquet", ".pq")
        cols = self.con.execute(f'DESCRIBE "{name}"').fetchall()
        sel = ", ".join(
            f'ST_AsText("{c}") AS "{c}"' if t.startswith("GEOMETRY") and not parquet else f'"{c}"'
            for c, t, *_ in cols
        )
        fmt = "(FORMAT parquet)" if parquet else "(FORMAT csv, HEADER true)"
        self.con.execute(f'COPY (SELECT {sel} FROM "{name}") TO {_sql_str(path)} {fmt}')
        return path

    def add_observations(self, df: pd.DataFrame, *, rings: Sequence[Any] | None = None) -> int:
        """Insert or replace (by ``product_id``) observation rows in ``images``.

        Parameters
        ----------
        df : DataFrame
            :mod:`lunar_toolkit.observation` schema rows; missing columns are NULL.
        rings : sequence, optional
            Per row, :func:`lunar_toolkit.geometry.footprint_rings` output to use instead
            of the rows' own (``footprint_wkt``, else the ``min_lat`` ... ``lon_east`` box).

        Returns
        -------
        int
            Rows written.  ``footprint`` is the footprint polygon in polar stereographic
            metres (:func:`lunar_toolkit.geometry.footprint_geometry` with
            :func:`~lunar_toolkit.geometry.ps_projection` of ``hemi``, the hemisphere of
            the image centre; edges densified along great circles every
            :data:`GREAT_CIRCLE_STEP_DEG`), ``footprint_km2`` its area corrected for the projection's
            scale, ``cap_lat`` / ``cap_lon`` the centre of its vertices, ``obs_key`` the
            observation (NAC LE / RE halves share one).  ``idx`` numbers rows in insertion
            order.
        """
        import pandas as pd

        from .geometry import footprint_geometry, footprint_rings, ps_projection
        from .observation import OBSERVATION_COLUMNS, normalize_observations
        from .stereo import _cap

        self._require_spatial()
        self._create_images()
        if df is None or len(df) == 0:
            return 0
        obs = normalize_observations(df.to_dict("records"))
        if rings is None:
            rings = [footprint_rings(r, "footprint_wkt") for r in obs.to_dict("records")]
        extra: dict[str, list[Any]] = {k: [] for k in (*_IMAGE_EXTRA, "fp_wkb")}
        wkts = []
        for row, rr in zip(obs.to_dict("records"), rings):
            pid = str(row["product_id"])
            extra["obs_key"].append((parse_nac_id(pid) or (pid.upper(),))[0])
            geom = None
            cap = (math.nan, math.nan)
            if rr is not None and len(rr[0][0]) >= 3:
                c, _ = _cap([poly[0] for poly in rr])
                cap = (
                    math.degrees(math.asin(max(-1.0, min(1.0, c[2])))),
                    math.degrees(math.atan2(c[1], c[0])) % 360,
                )
            lat = row["center_lat"] if pd.notna(row["center_lat"]) else cap[0]
            hemi = None if pd.isna(lat) else ("S" if lat < 0 else "N")
            if hemi is not None and rr is not None and len(rr[0][0]) >= 3:
                geom = footprint_geometry(_great_circle_rings(rr), ps_projection(hemi))
                if geom is not None and (geom.is_empty or geom.area <= 0):
                    geom = None
            extra["hemi"].append(hemi)
            extra["cap_lat"].append(cap[0])
            extra["cap_lon"].append(cap[1])
            extra["fp_wkb"].append(None if geom is None else geom.wkb)
            if geom is not None:
                cx, cy = geom.centroid.x, geom.centroid.y
                k = 1.0 + (cx * cx + cy * cy) / (4.0 * _R_M * _R_M)
                xmin, ymin, xmax, ymax = geom.bounds
                vals = (geom.area / k**2 / 1e6, xmin, xmax, ymin, ymax)
            else:
                vals = (math.nan,) * 5
            for key, v in zip(("footprint_km2", "xmin", "xmax", "ymin", "ymax"), vals):
                extra[key].append(v)
            wkt = row.get("footprint_wkt")
            if (not isinstance(wkt, str) or not wkt.strip()) and rr is not None and len(rr[0][0]) >= 3:
                wkt = _rings_wkt(rr)
            wkts.append(wkt if isinstance(wkt, str) else None)
        frame = obs[OBSERVATION_COLUMNS].copy()
        frame["footprint_wkt"] = wkts
        frame["product_id"] = frame["product_id"].astype(str)
        for k, v in extra.items():
            frame[k] = v
        frame = frame.drop_duplicates("product_id", keep="last").reset_index(drop=True)
        start = self._scalar("SELECT coalesce(max(idx) + 1, 0) FROM images")
        frame.insert(0, "idx", np.arange(start, start + len(frame), dtype=np.int64))
        cols = ["idx", *_OBS_TYPES, *_IMAGE_EXTRA]
        sel = ", ".join(
            f"CAST({c} AS {t}) AS {c}" for c, t in {"idx": "BIGINT", **_OBS_TYPES, **_IMAGE_EXTRA}.items()
        )
        self.con.register("_new_images", frame)
        self.con.execute("BEGIN TRANSACTION")
        try:
            self.con.execute("DELETE FROM images WHERE product_id IN (SELECT product_id FROM _new_images)")
            self.con.execute(
                f"INSERT INTO images ({', '.join(cols)}, footprint, added_at) "
                f"SELECT {sel}, CASE WHEN fp_wkb IS NULL THEN NULL ELSE ST_GeomFromWKB(fp_wkb) END, "
                "current_localtimestamp() FROM _new_images"
            )
            self.con.execute("COMMIT")
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("_new_images")
        return len(frame)

    def add_index_rows(self, df: pd.DataFrame, dataset: str | None = "edr") -> int:
        """:meth:`add_observations` of ``CumIndex`` rows (footprints from ``FP_WKT``, corners or ``FP_*``)."""
        from .geometry import footprint_rings
        from .observation import observations_from_cumindex

        if df is None or len(df) == 0:
            self._require_spatial()
            self._create_images()
            return 0
        rings = [footprint_rings(r, "FP_WKT") for r in df.to_dict("records")]
        return self.add_observations(observations_from_cumindex(df, dataset=dataset), rings=rings)

    def _create_images(self) -> None:
        cols = ", ".join(f"{c} {t}" for c, t in {"idx": "BIGINT", **_OBS_TYPES, **_IMAGE_EXTRA}.items())
        self.con.execute(
            f"CREATE TABLE IF NOT EXISTS images ({cols}, footprint GEOMETRY, added_at TIMESTAMP)"
        )

    def pairs(
        self,
        aoi: Any = None,
        *,
        convergence: tuple[float, float] = (10.0, 60.0),
        ideal_convergence: tuple[float, float] = (15.0, 45.0),
        max_incidence_diff: float = 20.0,
        max_azimuth_diff: float = 30.0,
        min_overlap: float = 0.1,
        min_aoi_coverage: float = 0.0,
        max_resolution_ratio: float | None = None,
        same_instrument: bool = True,
        require_geometry: bool = False,
        min_time_sep_s: float = 60.0,
        weights: dict[str, float] | None = None,
        product_ids: Sequence[str] | None = None,
        table: str | None = "pairs",
    ) -> pd.DataFrame:
        """Rank stereo pairs of ``images`` with a SQL spatial self-join.

        Parameters are those of :func:`lunar_toolkit.stereo.find_stereo_pairs`, plus
        *product_ids* (only these images) and *table* (where the result is stored;
        None keeps it out of the database).

        Returns
        -------
        DataFrame
            :data:`lunar_toolkit.stereo.PAIR_COLUMNS`; ``index_1`` / ``index_2`` are
            ``images.idx``.

        Notes
        -----
        Candidate pairs come from a bounding-box spatial join (``ST_Intersects_Extent``,
        R-tree built by DuckDB's spatial join operator) of the footprints of each
        hemisphere, clipped to the AOI; then the attribute filters, overlap area / fraction (``ST_Area`` of
        ``ST_Intersection`` in metres, divided by the projection's areal scale at the
        centroid), AOI coverage, incidence / azimuth / emission / resolution / time
        differences and the overlap centroid are SQL.  The convergence angle and
        ``b_h`` are then computed for the candidates with numpy
        (:func:`lunar_toolkit.stereo.convergence_angles`: sub-spacecraft point +
        altitude, pushbroom-corrected), exactly as ``find_stereo_pairs``, and the
        score is :func:`lunar_toolkit.stereo.score_pairs`.  Footprint edges are great
        circles in both (densified here), so overlap areas and fractions agree with
        ``find_stereo_pairs`` to ~1e-3 relative.
        """
        import pandas as pd

        from .geometry import _polar_hemi, aoi_geometry, aoi_region, load_aoi, ps_projection
        from .stereo import PAIR_COLUMNS, convergence_angles, score_pairs

        self._require_spatial()
        self._create_images()
        lo_c, hi_c = convergence
        aoi_parts: dict[str, tuple[bytes, float]] = {}
        if aoi is not None:
            a = load_aoi(aoi)
            region = aoi_region(a)
            hemi = _polar_hemi(region) or ("S" if sum(region["lat"]) < 0 else "N")
            geom = aoi_geometry(a, ps_projection(hemi))
            k = 1.0 + (geom.centroid.x**2 + geom.centroid.y**2) / (4.0 * _R_M * _R_M)
            aoi_parts[hemi] = (geom.wkb, geom.area / k**2 / 1e6)
            hemis = [hemi]
        else:
            hemis = [
                h
                for (h,) in self.con.execute(
                    "SELECT DISTINCT hemi FROM images WHERE hemi IS NOT NULL"
                ).fetchall()
            ]

        conds = ["a.obs_key <> b.obs_key"]
        if same_instrument:
            conds.append("a.instrument IS NOT DISTINCT FROM b.instrument")
        conds += [
            f"NOT coalesce(abs(b.t - a.t) < {float(min_time_sep_s)!r}, false)",
            f"NOT coalesce(abs(b.incidence - a.incidence) > {float(max_incidence_diff)!r}, false)",
            f"NOT coalesce(_az_diff(a.az, b.az) > {float(max_azimuth_diff)!r}, false)",
            f"NOT coalesce(a.emission + b.emission < {lo_c - 5.0!r}, false)",
            f"NOT coalesce(abs(b.emission - a.emission) > {hi_c + 5.0!r}, false)",
        ]
        if require_geometry:
            conds.append("a.has_geom AND b.has_geom")
        if max_resolution_ratio is not None:
            conds.append(
                f"NOT coalesce({_ratio_sql('a.ps', 'b.ps')} > {float(max_resolution_ratio)!r}, false)"
            )

        frames = []
        for hemi in hemis:
            wkb, aoi_km2 = aoi_parts.get(hemi, (None, math.nan))
            where = [f"hemi = {_sql_str(hemi)}", "footprint IS NOT NULL"]
            params: list[Any] = []
            if product_ids is not None:
                self.con.register("_pair_ids", pd.DataFrame({"product_id": [str(p) for p in product_ids]}))
                where.append("product_id IN (SELECT product_id FROM _pair_ids)")
            g = "footprint"
            if wkb is not None:
                where.append("ST_Intersects(footprint, ST_GeomFromWKB(?))")
                g = "ST_Intersection(footprint, ST_GeomFromWKB(?))"
                params = [wkb, wkb]
            self.con.execute(
                "CREATE OR REPLACE TEMP TABLE _pair_fp AS SELECT *, _true_km2(g) AS g_km2 FROM ("
                "SELECT idx, product_id, obs_key, instrument, start_time, epoch(start_time) AS t, incidence, "
                "emission, sub_solar_azimuth AS az, pixel_scale AS ps, sub_spacecraft_lat AS sclat, "
                "sub_spacecraft_lon AS sclon, spacecraft_altitude AS alt, "
                "(sub_spacecraft_lat IS NOT NULL AND sub_spacecraft_lon IS NOT NULL "
                "AND spacecraft_altitude IS NOT NULL) AS has_geom, "
                "CASE WHEN center_lat IS NOT NULL AND center_lon IS NOT NULL THEN center_lat ELSE cap_lat END "
                "AS vlat, CASE WHEN center_lat IS NOT NULL AND center_lon IS NOT NULL THEN center_lon "
                f"ELSE cap_lon END AS vlon, {g} AS g FROM images WHERE {' AND '.join(where)}"
                ") WHERE ST_Area(g) > 0",
                params or None,
            )
            # Bounding-box prefilter: DuckDB plans its R-tree spatial join only for a bare
            # ST_Intersects[_Extent] (+ inequality) condition, so candidates go to a temp table.
            self.con.execute(
                "CREATE OR REPLACE TEMP TABLE _pair_cand AS SELECT a.idx AS i, b.idx AS j "
                "FROM _pair_fp a JOIN _pair_fp b ON ST_Intersects_Extent(a.g, b.g) AND a.idx < b.idx"
            )
            aoi_sql = f"ov_km2 / {aoi_km2!r}" if wkb is not None else "NULL::DOUBLE"
            cov_filter = f"AND aoi_coverage >= {max(min_aoi_coverage, 1e-9)!r}" if wkb is not None else ""
            self.con.execute(
                "CREATE OR REPLACE TEMP TABLE _pair_ab AS SELECT a, b FROM _pair_cand c "
                "JOIN (SELECT * EXCLUDE (g) FROM _pair_fp) a ON a.idx = c.i "
                "JOIN (SELECT * EXCLUDE (g) FROM _pair_fp) b ON b.idx = c.j"
            )
            # Each step is materialised: otherwise the a / b filters become join conditions
            # and ST_Intersection runs before them, once per reference to its result.
            self.con.execute(f"""
                CREATE OR REPLACE TEMP TABLE _pair_ov AS
                SELECT a.idx AS index_1, b.idx AS index_2, a.product_id AS product_id_1,
                       b.product_id AS product_id_2, a.instrument,
                       a.start_time AS start_time_1, b.start_time AS start_time_2,
                       abs(b.t - a.t) / 86400.0 AS dt_days,
                       abs(b.emission - a.emission) AS convergence_min,
                       a.emission + b.emission AS convergence_max,
                       a.incidence AS incidence_1, b.incidence AS incidence_2,
                       abs(b.incidence - a.incidence) AS incidence_diff,
                       a.az AS azimuth_1, b.az AS azimuth_2, _az_diff(a.az, b.az) AS azimuth_diff,
                       a.emission AS emission_1, b.emission AS emission_2,
                       a.ps AS pixel_scale_1, b.ps AS pixel_scale_2,
                       {_ratio_sql("a.ps", "b.ps")} AS resolution_ratio,
                       a.g_km2 AS g1_km2, b.g_km2 AS g2_km2,
                       a.has_geom AND b.has_geom AS has_geom,
                       a.sclat AS sclat_1, a.sclon AS sclon_1, a.alt AS alt_1,
                       b.sclat AS sclat_2, b.sclon AS sclon_2, b.alt AS alt_2,
                       a.vlat AS vlat_1, a.vlon AS vlon_1, b.vlat AS vlat_2, b.vlon AS vlon_2
                FROM _pair_ab
                WHERE {" AND ".join(conds)}
            """)
            self.con.execute(
                "CREATE OR REPLACE TEMP TABLE _pair_x AS SELECT o.*, ST_Intersection(a.g, b.g) AS ov "
                "FROM _pair_ov o JOIN _pair_fp a ON a.idx = o.index_1 JOIN _pair_fp b ON b.idx = o.index_2"
            )
            self.con.execute(
                "CREATE OR REPLACE TEMP TABLE _pair_m AS SELECT * EXCLUDE (ov, c), ST_X(c) AS cx, "
                "ST_Y(c) AS cy FROM (SELECT *, ST_Area(ov) AS ov_m2, ST_Centroid(ov) AS c FROM _pair_x "
                "WHERE NOT ST_IsEmpty(ov))"
            )
            df = self.con.execute(f"""
                SELECT * FROM (
                    SELECT *, ov_km2 / greatest(least(g1_km2, g2_km2), 1e-12) AS overlap_frac,
                           {aoi_sql} AS aoi_coverage
                    FROM (SELECT * EXCLUDE (ov_m2), ov_m2 / pow(_ps_k(cx, cy), 2) / 1e6 AS ov_km2 FROM _pair_m)
                    WHERE ov_km2 > 0
                )
                WHERE overlap_frac >= {float(min_overlap)!r} {cov_filter}
                ORDER BY index_1, index_2
            """).df()
            df["overlap_lat"] = _ps_lat(df["cx"].to_numpy(float), df["cy"].to_numpy(float), hemi)
            df["overlap_lon"] = _ps_lon(df["cx"].to_numpy(float), df["cy"].to_numpy(float), hemi)
            frames.append(df)
            for tmp in ("_pair_m", "_pair_x", "_pair_ov", "_pair_ab", "_pair_cand", "_pair_fp"):
                self.con.execute(f"DROP TABLE IF EXISTS {tmp}")
            if product_ids is not None:
                self.con.unregister("_pair_ids")

        frames = [f for f in frames if len(f)]
        if not frames:
            res = pd.DataFrame(columns=PAIR_COLUMNS)
        else:
            res = pd.concat(frames, ignore_index=True).sort_values(["index_1", "index_2"], kind="stable")
            res = res.reset_index(drop=True)
            geom = res["has_geom"].fillna(False).to_numpy(bool)
            conv = np.full(len(res), np.nan)
            bh = np.full(len(res), np.nan)
            if geom.any():
                r = res[geom]

                def col(c: str) -> np.ndarray:
                    return r[c].to_numpy(float)

                conv[geom], bh[geom] = convergence_angles(
                    (col("overlap_lat"), col("overlap_lon")),
                    (col("sclat_1"), col("sclon_1"), col("alt_1")),
                    (col("sclat_2"), col("sclon_2"), col("alt_2")),
                    (col("vlat_1"), col("vlon_1")),
                    (col("vlat_2"), col("vlon_2")),
                )
            res["convergence"], res["b_h"] = conv, bh
            res["overlap_km2"] = res["ov_km2"]
            keep = np.isnan(conv) | ((conv >= lo_c) & (conv <= hi_c))
            res = res[keep].reset_index(drop=True)
            for c in ("start_time_1", "start_time_2"):
                res[c] = pd.to_datetime(res[c])
            res["score"] = score_pairs(
                res,
                convergence=convergence,
                ideal_convergence=ideal_convergence,
                max_incidence_diff=max_incidence_diff,
                max_azimuth_diff=max_azimuth_diff,
                weights=weights,
                aoi=aoi is not None,
            )
            res = (
                res[PAIR_COLUMNS].sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
            )
        if table is not None:
            name = _ident(table)
            self.con.register("_pairs_result", res)
            try:
                self.con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _pairs_result')
            finally:
                self.con.unregister("_pairs_result")
        return res

    def set_rejection_params(self, config: SfsConfig | None = None, **overrides: Any) -> None:
        """Thresholds of the ``image_rejections`` view, from *config* (default :class:`~lunar_toolkit.sfs.SfsConfig`).

        Keyword overrides: ``max_offset_p85``, ``max_large_residual_fraction``,
        ``large_residual_px``, ``min_matches`` (None disables the first two rules).
        """
        from .sfs import SfsConfig

        cfg = config or SfsConfig()
        vals = {
            "max_offset_p85": cfg.max_offset_p85,
            "max_large_residual_fraction": cfg.max_large_residual_fraction,
            "large_residual_px": cfg.large_residual_px,
            "min_matches": cfg.min_matches,
        }
        unknown = set(overrides) - set(vals)
        if unknown:
            raise TypeError(f"Unknown rejection parameter(s): {', '.join(sorted(unknown))}")
        vals.update(overrides)
        self.con.execute("DELETE FROM rejection_params")
        self.con.execute(
            "INSERT INTO rejection_params VALUES (?, ?, ?, ?)",
            [
                vals["max_offset_p85"],
                vals["max_large_residual_fraction"],
                vals["large_residual_px"],
                int(vals["min_matches"] or 0),
            ],
        )

    def load_bundle_adjust(self, prefix: Path | str, run: str | None = None) -> dict[str, int]:
        """Load ASP ``bundle_adjust`` outputs ``<prefix>-*`` as run *run* (replacing it).

        *run* defaults to the prefix's directory name (``ba``, ``ba_refine``).  Reads,
        when present: ``-initial`` / ``-final_residuals_stats.txt`` and
        ``_raw_pixels.txt`` (``ba_residual_stats``, ``ba_pixel_residuals``),
        ``-mapproj_match_offset_stats.txt`` (``ba_match_offsets``; -1 becomes NULL),
        ``-initial`` / ``-final_residuals_pointmap.csv`` (``ba_points``, with polar
        stereographic ``x`` / ``y``), ``-convergence_angles.txt`` (``ba_convergence``)
        and ``-camera_offsets.txt`` (``ba_camera_offsets``).

        Returns
        -------
        dict
            Rows loaded per table (empty when no report exists).
        """
        from . import asp

        prefix = Path(prefix)
        run = run or prefix.parent.name or prefix.name
        f = {name: Path(f"{prefix}-{name}") for name in _BA_FILES}
        found = [p for p in f.values() if p.exists()]
        if not found:
            return {}
        stats = {
            stage: asp.read_residual_stats(f[f"{stage}_residuals_stats.txt"])
            for stage in ("initial", "final")
            if f[f"{stage}_residuals_stats.txt"].exists()
        }
        raw = {
            stage: _read_raw_pixels(f[f"{stage}_residuals_raw_pixels.txt"])
            for stage in ("initial", "final")
            if f[f"{stage}_residuals_raw_pixels.txt"].exists()
        }
        offsets = (
            asp.read_mapproj_offset_stats(f["mapproj_match_offset_stats.txt"])
            if f["mapproj_match_offset_stats.txt"].exists()
            else None
        )
        self.con.execute("BEGIN TRANSACTION")
        try:
            self._insert_ba(run, stats=stats, raw=raw, offsets=offsets)
            for stage in ("initial", "final"):
                p = f[f"{stage}_residuals_pointmap.csv"]
                if p.exists():
                    self._load_points(run, stage, p)
            if f["convergence_angles.txt"].exists():
                rows = _text_rows(self.con, f["convergence_angles.txt"], " ", 6)
                self.con.execute(
                    "INSERT INTO ba_convergence SELECT ?, parse_filename(c1), parse_filename(c2), "
                    "upper(split_part(parse_filename(c1), '.', 1)), upper(split_part(parse_filename(c2), '.', 1)), "
                    f"TRY_CAST(c3 AS DOUBLE), TRY_CAST(c4 AS DOUBLE), TRY_CAST(c5 AS DOUBLE), "
                    f"CAST(TRY_CAST(c6 AS DOUBLE) AS BIGINT) FROM ({rows}) WHERE c6 IS NOT NULL",
                    [run],
                )
            if f["camera_offsets.txt"].exists():
                rows = _text_rows(self.con, f["camera_offsets.txt"], " ", 3)
                self.con.execute(
                    "INSERT INTO ba_camera_offsets SELECT ?, parse_filename(c1), "
                    "upper(split_part(parse_filename(c1), '.', 1)), TRY_CAST(c2 AS DOUBLE), "
                    f"TRY_CAST(c3 AS DOUBLE) FROM ({rows}) WHERE c3 IS NOT NULL",
                    [run],
                )
            self.con.execute("DELETE FROM ba_runs WHERE run = ?", [run])
            self.con.execute(
                "INSERT INTO ba_runs VALUES (?, ?, current_localtimestamp(), ?)",
                [run, str(prefix), [p.name for p in found]],
            )
            self.con.execute("COMMIT")
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        return {t: int(self._scalar(f"SELECT count(*) FROM {t} WHERE run = ?", [run])) for t in BA_TABLES}

    def _insert_ba(
        self,
        run: str,
        *,
        stats: Mapping[str, Mapping[str, ResidualStats]] | None = None,
        raw: Mapping[str, pd.DataFrame] | None = None,
        offsets: Mapping[str, MatchOffsetStats] | None = None,
    ) -> None:
        """Replace run *run* with parsed reports: per stage residual stats / raw norms, offsets."""
        import pandas as pd

        for t in BA_TABLES:
            self.con.execute(f"DELETE FROM {t} WHERE run = ?", [run])
        rows = [
            (
                run,
                stage,
                Path(s.image).name,
                s.image,
                _product_id(s.image),
                _num(s.mean),
                _num(s.median),
                s.count,
            )
            for stage, by_image in (stats or {}).items()
            for s in by_image.values()
        ]
        if rows:
            self.con.executemany("INSERT INTO ba_residual_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        for stage, norms in (raw or {}).items():
            if len(norms):
                frame = pd.DataFrame(
                    {
                        "run": run,
                        "stage": stage,
                        "image": norms["image"].astype(str),
                        "product_id": [_product_id(i) for i in norms["image"]],
                        "norm_px": norms["norm_px"].astype(float),
                    }
                )
                self.con.register("_raw", frame)
                try:
                    self.con.execute("INSERT INTO ba_pixel_residuals SELECT * FROM _raw")
                finally:
                    self.con.unregister("_raw")
        orows = [
            (
                run,
                Path(o.image).name,
                o.image,
                _product_id(o.image),
                *(None if not v >= 0 else v for v in (o.p25, o.p50, o.p75, o.p85, o.p95)),
                o.count,
            )
            for o in (offsets or {}).values()
        ]
        if orows:
            self.con.executemany("INSERT INTO ba_match_offsets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", orows)

    def _load_points(self, run: str, stage: str, path: Path) -> None:
        rows = _text_rows(self.con, path, ",", 5)
        self.con.execute(
            f"""
            INSERT INTO ba_points
            SELECT run, stage, lon, lat, height_m, res, nobs, kind, hemi,
                   _ps_x(lat, lon, hemi), _ps_y(lat, lon, hemi)
            FROM (
                SELECT ? AS run, ? AS stage, TRY_CAST(c1 AS DOUBLE) AS lon, TRY_CAST(c2 AS DOUBLE) AS lat,
                       TRY_CAST(c3 AS DOUBLE) AS height_m, TRY_CAST(c4 AS DOUBLE) AS res,
                       TRY_CAST(split_part(trim(c5), ' ', 1) AS INTEGER) AS nobs,
                       CASE WHEN c5 LIKE '%GCP%' THEN 'gcp' WHEN c5 LIKE '%DEM%' THEN 'dem' ELSE 'tie' END AS kind,
                       CASE WHEN TRY_CAST(c2 AS DOUBLE) < 0 THEN 'S' ELSE 'N' END AS hemi
                FROM ({rows}) WHERE c5 IS NOT NULL
            ) WHERE lon IS NOT NULL AND lat IS NOT NULL
            """,
            [run, stage],
        )

    def runs(self) -> list[str]:
        """Loaded bundle-adjustment runs."""
        return [r for (r,) in self.con.execute("SELECT run FROM ba_runs ORDER BY loaded_at, run").fetchall()]

    def rejections(self, run: str) -> dict[str, str]:
        """``image_rejections`` of *run*: image file name -> reason."""
        rows = self.con.execute("SELECT image, reason FROM image_rejections WHERE run = ?", [run]).fetchall()
        return dict(rows)

    def ba_report(self, run: str) -> pd.DataFrame | None:
        """Per-image table of *run* in :func:`lunar_toolkit.sfs_report.read_ba_reports` form, or None."""
        if run not in self.runs():
            return None
        cols = ["image"] + [f"{k}_{s}" for k in ("initial", "final") for s in ("mean", "median", "count")]
        cols += [f"offset_p{p}" for p in (25, 50, 75, 85, 95)] + ["offset_count"]
        df = self.con.execute(
            f"SELECT {', '.join(cols)} FROM ba_image_stats WHERE run = ? AND "
            "(initial_count IS NOT NULL OR final_count IS NOT NULL OR offset_count IS NOT NULL) "
            "ORDER BY image",
            [run],
        ).df()
        for c in cols[1:]:
            df[c] = df[c].astype(float)
        return df

    def convergence(self, run: str) -> pd.DataFrame:
        """``ba_convergence`` of *run* as ``left``, ``right``, ``p25``, ``p50``, ``p75``, ``matches``."""
        return self.con.execute(
            "SELECT left_image AS left, right_image AS right, p25, p50, p75, matches "
            "FROM ba_convergence WHERE run = ?",
            [run],
        ).df()

    def point_residuals(self, run: str, stage: str = "final") -> np.ndarray:
        """``mean_residual_px`` of the *run* / *stage* residual point map."""
        rows = self.con.execute(
            "SELECT mean_residual_px FROM ba_points WHERE run = ? AND stage = ?", [run, stage]
        ).fetchnumpy()
        return np.asarray(rows["mean_residual_px"], float)

    def summary(self) -> pd.DataFrame:
        """The ``ba_run_summary`` view."""
        return self.con.execute("SELECT * FROM ba_run_summary").df()


_BA_FILES = (
    "initial_residuals_stats.txt",
    "final_residuals_stats.txt",
    "initial_residuals_raw_pixels.txt",
    "final_residuals_raw_pixels.txt",
    "mapproj_match_offset_stats.txt",
    "initial_residuals_pointmap.csv",
    "final_residuals_pointmap.csv",
    "convergence_angles.txt",
    "camera_offsets.txt",
)


def _great_circle_rings(rings: list, step_deg: float = GREAT_CIRCLE_STEP_DEG) -> list:
    """*rings* with vertices inserted along great circles, at most *step_deg* apart."""
    from .stereo import unit_vector

    out = []
    for poly in rings:
        dense = []
        for ring in poly:
            r = list(ring[:-1] if len(ring) > 1 and tuple(ring[0]) == tuple(ring[-1]) else ring)
            if len(r) < 3:
                dense.append(r)
                continue
            a = np.asarray(r, float)
            v = unit_vector(a[:, 1], a[:, 0])
            w = np.roll(v, -1, axis=0)
            theta = np.arctan2(np.linalg.norm(np.cross(v, w), axis=1), np.sum(v * w, axis=1))
            pts: list[tuple[float, float]] = []
            for k, (vk, wk, th) in enumerate(zip(v, w, theta)):
                pts.append((float(a[k, 0]), float(a[k, 1])))
                n = int(math.ceil(math.degrees(th) / step_deg))
                if n < 2 or th < 1e-12:
                    continue
                f = np.arange(1, n)[:, None] / n
                q = (np.sin((1 - f) * th) * vk + np.sin(f * th) * wk) / math.sin(th)
                lat = np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1)))
                lon = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
                # keep the ring's longitude unwrapping (values may lie outside 0..360)
                lon = a[k, 0] + (lon - a[k, 0] + 180.0) % 360.0 - 180.0
                pts += list(zip(lon.tolist(), lat.tolist()))
            dense.append(pts)
        out.append(dense)
    return out


def _ratio_sql(a: str, b: str) -> str:
    return f"(CASE WHEN {a} >= {b} THEN {a} / {b} ELSE {b} / {a} END)"


def _ps_lat(x: np.ndarray, y: np.ndarray, hemi: str) -> np.ndarray:
    sgn = 1.0 if hemi == "S" else -1.0
    return sgn * (np.degrees(2.0 * np.arctan(np.hypot(x, y) / (2.0 * _R_M))) - 90.0)


def _ps_lon(x: np.ndarray, y: np.ndarray, hemi: str) -> np.ndarray:
    return np.degrees(np.arctan2(x, y if hemi == "S" else -y)) % 360.0


def _rings_wkt(rings: list) -> str:
    from shapely.geometry import MultiPolygon, Polygon

    polys = [Polygon(r[0], [h for h in r[1:] if len(h) >= 3]) for r in rings if len(r[0]) >= 3]
    return (polys[0] if len(polys) == 1 else MultiPolygon(polys)).wkt


def _text_rows(con: Any, path: Path, delim: str, n: int) -> str:
    """SQL reading *path* as *n* VARCHAR columns ``c1``.. (``#`` lines dropped, short rows NULL-padded)."""
    cols = "{" + ", ".join(f"'c{k}': 'VARCHAR'" for k in range(1, n + 1)) + "}"
    return (
        f"SELECT * FROM read_csv({_sql_str(path)}, delim={_sql_str(delim)}, header=false, columns={cols}, "
        "null_padding=true, quote='', escape='', auto_detect=false, strict_mode=false) "
        "WHERE NOT starts_with(trim(c1), '#')"
    )


def _read_raw_pixels(path: Path) -> pd.DataFrame:
    """``image``, ``norm_px`` of a ``*_residuals_raw_pixels.txt`` (``name, count`` then ``ex, ey`` lines)."""
    import pandas as pd

    df = pd.read_csv(path, header=None, names=["a", "b"], dtype=str, skipinitialspace=True, comment=None)
    a = pd.to_numeric(df["a"], errors="coerce")
    head = a.isna().to_numpy()
    if not head.any():
        return pd.DataFrame({"image": pd.Series(dtype=str), "norm_px": pd.Series(dtype=float)})
    names = df["a"].where(head).ffill()
    keep = ~head & names.notna().to_numpy()
    ex, ey = a[keep].to_numpy(float), pd.to_numeric(df["b"][keep], errors="coerce").to_numpy(float)
    image = names[keep].map(lambda s: Path(str(s).strip()).name)
    return pd.DataFrame({"image": image.to_numpy(), "norm_px": np.hypot(ex, ey)})


def find_stereo_pairs_duckdb(
    obs_df: pd.DataFrame, aoi: Any = None, *, workspace: Workspace | None = None, **kwargs: Any
) -> pd.DataFrame:
    """:func:`lunar_toolkit.stereo.find_stereo_pairs` through :meth:`Workspace.pairs`.

    The observations are written to ``images`` of *workspace* (default: a scratch
    in-memory one) and ``index_1`` / ``index_2`` are mapped back to *obs_df* row
    labels; keyword arguments are those of ``find_stereo_pairs``.
    """
    import pandas as pd

    from .stereo import PAIR_COLUMNS

    if obs_df is None or len(obs_df) < 2:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    pids = obs_df["product_id"].astype(str)
    if pids.duplicated().any():
        raise ValueError("find_stereo_pairs_duckdb needs unique product_id values")
    label = dict(zip(pids, obs_df.index))
    ws = workspace or Workspace(":memory:")
    try:
        ws.add_observations(obs_df)
        res = ws.pairs(
            aoi, product_ids=list(pids), table="pairs" if workspace is not None else None, **kwargs
        )
    finally:
        if workspace is None:
            ws.close()
    res["index_1"] = [label[p] for p in res["product_id_1"]]
    res["index_2"] = [label[p] for p in res["product_id_2"]]
    return res


def flag_images_sql(
    residual_stats: Mapping[str, ResidualStats],
    residual_norms: Mapping[str, np.ndarray],
    offsets: Mapping[str, MatchOffsetStats],
    **thresholds: Any,
) -> dict[str, str]:
    """``image_rejections`` of parsed final-stage reports (see :func:`lunar_toolkit.sfs.flag_images`)."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "image": [n for n, v in residual_norms.items() for _ in range(len(v))],
            "norm_px": np.concatenate([np.asarray(v, float) for v in residual_norms.values()])
            if residual_norms
            else np.empty(0),
        }
    )
    with Workspace(":memory:", spatial=False, threads=1) as ws:
        ws.set_rejection_params(**thresholds)
        stats = {k: _with_image(v, k) for k, v in residual_stats.items()}
        offs = {k: _with_image(v, k) for k, v in offsets.items()}
        ws._insert_ba("_", stats={"final": stats}, raw={"final": raw}, offsets=offs)
        return ws.rejections("_")


def _with_image(obj: Any, name: str) -> Any:
    """*obj* (a frozen report row) keyed by *name*: its ``image`` replaced when the names differ."""
    import dataclasses

    return obj if Path(obj.image).name == name else dataclasses.replace(obj, image=name)
