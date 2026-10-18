"""Common observation schema shared by all search backends.

Longitudes are 0–360 east (``lon_west > lon_east`` crosses the seam); angles in degrees,
``pixel_scale`` in m/px, ``spacecraft_altitude`` in km; missing values are None."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import is_nac_id

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

OBSERVATION_COLUMNS: list[str] = [
    "product_id",
    "instrument",  # e.g. "LROC NAC", "ShadowCam", "Kaguya TC"
    "product_type",  # e.g. "EDR", "CDR", "CAL", "RAW", "DTM"
    "start_time",  # pandas Timestamp (UTC, tz-naive)
    "center_lat",
    "center_lon",
    "min_lat",
    "max_lat",
    "lon_west",
    "lon_east",
    "footprint_wkt",  # lat/lon WKT if known (ODE: 0–360 cylindrical)
    "incidence",
    "emission",
    "phase",
    "slew_angle",
    "sub_solar_azimuth",  # None when the source does not provide it
    "north_azimuth",  # PDS NORTH_AZIMUTH (image-frame), None if unknown
    "sub_spacecraft_lat",  # sub-spacecraft point at image centre time
    "sub_spacecraft_lon",
    "spacecraft_altitude",  # km above the surface
    "pixel_scale",
    "url",  # primary product file
    "label_url",  # PDS label (.xml / .lbl)
    "source",  # "cumindex" or "ode"
]


def empty_observations() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame({c: pd.Series(dtype="object") for c in OBSERVATION_COLUMNS})


def normalize_observations(records: list[dict]) -> pd.DataFrame:
    """Build an observation DataFrame from dicts (missing keys → None)."""
    import pandas as pd

    if not records:
        return empty_observations()
    df = pd.DataFrame.from_records(
        [{c: r.get(c) for c in OBSERVATION_COLUMNS} for r in records],
        columns=OBSERVATION_COLUMNS,
    )
    for c in (
        "center_lat",
        "center_lon",
        "min_lat",
        "max_lat",
        "lon_west",
        "lon_east",
        "incidence",
        "emission",
        "phase",
        "slew_angle",
        "sub_solar_azimuth",
        "pixel_scale",
        "north_azimuth",
        "sub_spacecraft_lat",
        "sub_spacecraft_lon",
        "spacecraft_altitude",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def observations_from_cumindex(df: pd.DataFrame, dataset: str | None = "edr") -> pd.DataFrame:
    """Map enriched ``CumIndex`` query results to :data:`OBSERVATION_COLUMNS`."""
    import pandas as pd

    if df is None or df.empty:
        return empty_observations()

    def g(row, col):
        v = row.get(col)
        try:
            return None if pd.isna(v) else v
        except (TypeError, ValueError):  # array-like value
            return v

    recs = []
    for _, row in df.iterrows():
        pid = str(row.get("PRODUCT_ID", ""))
        recs.append(
            {
                "product_id": pid,
                "instrument": "LROC NAC" if is_nac_id(pid) else "LROC WAC",
                "product_type": (dataset or "").upper() or None,
                "start_time": g(row, "START_TIME"),
                "center_lat": g(row, "CENTER_LATITUDE"),
                "center_lon": g(row, "CENTER_LONGITUDE"),
                "min_lat": g(row, "FP_MIN_LAT"),
                "max_lat": g(row, "FP_MAX_LAT"),
                "lon_west": g(row, "FP_LON_WEST"),
                "lon_east": g(row, "FP_LON_EAST"),
                "footprint_wkt": g(row, "FP_WKT"),
                "incidence": g(row, "INCIDENCE_ANGLE"),
                "emission": g(row, "EMISSION_ANGLE"),
                "phase": g(row, "PHASE_ANGLE"),
                "slew_angle": g(row, "SLEW_ANGLE"),
                "sub_solar_azimuth": g(row, "SUB_SOLAR_AZIMUTH"),
                "north_azimuth": g(row, "NORTH_AZIMUTH"),
                "sub_spacecraft_lat": g(row, "SUB_SPACECRAFT_LATITUDE"),
                "sub_spacecraft_lon": g(row, "SUB_SPACECRAFT_LONGITUDE"),
                "spacecraft_altitude": g(row, "SPACECRAFT_ALTITUDE"),
                "pixel_scale": g(row, "RESOLUTION") if "RESOLUTION" in row else g(row, "PIXEL_SCALE"),
                "url": g(row, "PDS_URL"),
                "label_url": g(row, "LABEL_URL"),
                "source": "cumindex",
            }
        )
    return normalize_observations(recs)
