"""App name, data directories, PDS LROC archive URLs, datasets and shared defaults."""

import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Application name used for the user data / cache directories and the HTTP
# User-Agent.  The only place the package name is spelled out at runtime.
APP_NAME = "lunar_toolkit"
try:
    APP_VERSION = version("lunar-toolkit")
except PackageNotFoundError:  # running from a source tree that was never installed
    APP_VERSION = "0.0.0"

# PDS archive roots
# The LROC SOC host (https://pds.lroc.im-ldi.com/data/<DATA_SET_ID>/...) now
# 302-redirects to the PDS Imaging Node S3 store below.  Dataset directories
# there are lowercase and drop the "-V1.0" suffix, e.g.
#   https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0066C/INDEX/
#   → https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/
#       pds4/lroc/lro-l-lroc-2-edr/LROLRC_0066C/INDEX/

LEGACY_PDS_BASE_URL = "https://pds.lroc.im-ldi.com/data/"

PDS_STORE_URL = "https://pds.mcp.nasa.gov/data/store/img/"
PDS_LROC_PREFIX = "lunar_reconnaissance_orbiter/pds4/lroc/"
PDS_BASE_URL = PDS_STORE_URL + PDS_LROC_PREFIX

EDR_DATASET = "lro-l-lroc-2-edr/"
CDR_DATASET = "lro-l-lroc-3-cdr/"
RDR_DATASET = "lro-l-lroc-5-rdr/"

EDR_BASE_URL = PDS_BASE_URL + EDR_DATASET
CDR_BASE_URL = PDS_BASE_URL + CDR_DATASET
RDR_BASE_URL = PDS_BASE_URL + RDR_DATASET

# Fallback releases — the newest volume is auto-discovered from the S3 listing
# (see cumindex.discover_volumes); these are only used when that fails.
EDR_LATEST_RELEASE = "LROLRC_0068A/"
CDR_LATEST_RELEASE = "LROLRC_1068A/"
RDR_LATEST_RELEASE = "LROLRC_2001/"

INDEX_DIR = "INDEX/"

CUMINDEX_LBL_FILENAME = "CUMINDEX.LBL"
CUMINDEX_TAB_FILENAME = "CUMINDEX.TAB"
VOLINDEX_LBL_FILENAME = "INDEX.LBL"
VOLINDEX_TAB_FILENAME = "INDEX.TAB"

# Dataset registry
# Each entry drives CumIndex URL construction and the name of the parquet file
# built by `lunar build-db <key>`.

DATASETS: dict[str, dict] = {
    "edr": {
        "label": "EDR — Experiment Data Record (raw NAC/WAC images)",
        "data_set_id": "LRO-L-LROC-2-EDR-V1.0",
        "base_url": EDR_BASE_URL,
        "release": EDR_LATEST_RELEASE,
        "volume_re": r"LROLRC_0\d{3}[A-Z]?",
        "index_dir": INDEX_DIR,
        "lbl_filename": CUMINDEX_LBL_FILENAME,
        "tab_filename": CUMINDEX_TAB_FILENAME,
        "parquet": "cumindex_edr.parquet",
    },
    "cdr": {
        "label": "CDR — Calibrated Data Record",
        "data_set_id": "LRO-L-LROC-3-CDR-V1.0",
        "base_url": CDR_BASE_URL,
        "release": CDR_LATEST_RELEASE,
        "volume_re": r"LROLRC_1\d{3}[A-Z]?",
        "index_dir": INDEX_DIR,
        "lbl_filename": CUMINDEX_LBL_FILENAME,
        "tab_filename": CUMINDEX_TAB_FILENAME,
        "parquet": "cumindex_cdr.parquet",
    },
    "rdr": {
        "label": "RDR — Reduced Data Record (WAC mosaics, DEMs, etc.)",
        "data_set_id": "LRO-L-LROC-5-RDR-V1.0",
        "base_url": RDR_BASE_URL,
        "release": RDR_LATEST_RELEASE,
        "volume_re": r"LROLRC_2\d{3}[A-Z]?",
        "index_dir": INDEX_DIR,
        "lbl_filename": CUMINDEX_LBL_FILENAME,
        "tab_filename": CUMINDEX_TAB_FILENAME,
        "parquet": "cumindex_rdr.parquet",
    },
}

# PDS URL helpers
_LEGACY_DS_RE = re.compile(r"^(LRO-L-LROC-\d-[A-Z]+)-V\d+\.\d+$", re.IGNORECASE)


def pds_url_from_file_spec(file_spec: str) -> str:
    """Map a CUMINDEX ``FILE_SPECIFICATION_NAME`` to a download URL.

    ``LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/DATA/.../M101013931LE.IMG`` →
    ``<PDS_BASE_URL>lro-l-lroc-2-edr/LROLRC_0001/DATA/.../M101013931LE.IMG``.
    Paths without a dataset prefix are joined to ``PDS_BASE_URL`` unchanged.
    """
    spec = str(file_spec).strip().lstrip("/")
    head, _, rest = spec.partition("/")
    m = _LEGACY_DS_RE.match(head)
    if m and rest:
        return PDS_BASE_URL + m.group(1).lower() + "/" + rest
    return PDS_BASE_URL + spec


def modernize_pds_url(url: str) -> str:
    """Rewrite a legacy ``pds.lroc.im-ldi.com/data/`` URL to the PDS store host.

    Other URLs are returned unchanged.  (The legacy host still redirects, but
    resolving up front saves a round-trip and keeps Range requests simple.)
    """
    if url.startswith(LEGACY_PDS_BASE_URL):
        return pds_url_from_file_spec(url[len(LEGACY_PDS_BASE_URL) :])
    return url


# Data directory resolution
# Environment variable overriding the data directory.
DATA_DIR_ENV_VAR = "LUNAR_DATA_DIR"


def _user_data_dir(app_name: str = APP_NAME) -> Path:
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir(app_name, appauthor=False))
    except ImportError:  # pragma: no cover - platformdirs is a declared dependency
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        return Path(base) / app_name


def get_data_dir() -> Path:
    """Directory where Parquet databases are stored.

    ``LUNAR_DATA_DIR`` if set, else the per-user data directory
    (``platformdirs.user_data_dir``), e.g. ``~/.local/share/lunar_toolkit``
    on Linux (honours ``XDG_DATA_HOME``).
    """
    env = os.environ.get(DATA_DIR_ENV_VAR)
    return Path(env) if env else _user_data_dir()


def find_parquet(name: str, data_dir: Path | None = None) -> Path:
    """Path of Parquet database *name* in *data_dir* (default:
    [`get_data_dir`][lunar_toolkit.constants.get_data_dir])."""
    return Path(data_dir or get_data_dir()) / name


# User cache (raw TAB / LBL downloads)
DEFAULT_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / APP_NAME

# Misc
DEFAULT_PROJECTION = "equirectangular"

# gdal_translate flags for map-projected GeoTIFFs: native values, lossless and tiled
GEOTIFF_OPTIONS = ["-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER"]

# Mean lunar radius (km) used for polar stereographic footprints and searches
MOON_RADIUS_KM = 1737.4

# |latitude| above which footprints also get polar stereographic coordinates
POLAR_LAT_THRESHOLD = 60.0

# Index columns <corner>_LATITUDE / <corner>_LONGITUDE of a footprint, in ring order
FOOTPRINT_CORNERS = ("UPPER_LEFT", "UPPER_RIGHT", "LOWER_RIGHT", "LOWER_LEFT")

# NAC product IDs: <observation><side L/R><type E = EDR, C = CDR>; the short
# base form (M1096757719L) omits the type
NAC_ID_RE = re.compile(r"^([A-Z]\d{9,12})([LR])([EC]?)$", re.IGNORECASE)
WAC_ID_RE = re.compile(r"^[A-Z]\d{9,12}[CMUV]E$", re.IGNORECASE)
NAC_PRODUCT_TYPES = {"E": "edr", "C": "cdr"}
NAC_SUFFIXES = ("LE", "RE", "LC", "RC")


def parse_nac_id(product_id: str) -> tuple[str, str, str] | None:
    """Split a NAC product ID into ``(observation, side, type)``.

    ``"M1096757719LE"`` → ``("M1096757719", "L", "edr")``; a short base ID
    (``"M1096757719L"``) gets type ``"edr"``.  None for anything else (WAC,
    ShadowCam, …).
    """
    m = NAC_ID_RE.match(str(product_id).strip())
    if m is None:
        return None
    obs, side, kind = m.groups()
    return obs.upper(), side.upper(), NAC_PRODUCT_TYPES[(kind or "E").upper()]


def nac_id(product_id: str, side: str | None = None, product_type: str | None = None) -> str | None:
    """Full NAC product ID of *product_id*, optionally for another side / type.

    ``nac_id("M1096757719L")`` → ``"M1096757719LE"``;
    ``nac_id("M1096757719LE", side="R", product_type="cdr")`` → ``"M1096757719RC"``.
    None when *product_id* is not a NAC ID.
    """
    parts = parse_nac_id(product_id)
    if parts is None:
        return None
    obs, own_side, own_type = parts
    letter = {v: k for k, v in NAC_PRODUCT_TYPES.items()}[(product_type or own_type).lower()]
    return f"{obs}{(side or own_side).upper()}{letter}"


def is_nac_id(product_id: str) -> bool:
    """True for a full NAC product ID (``LE`` / ``RE`` / ``LC`` / ``RC``)."""
    m = NAC_ID_RE.match(str(product_id).strip())
    return bool(m and m.group(3))


# SfS pipeline option values (lunar_toolkit.sfs.SfsConfig), here so the CLI stays light
REGISTRATION_MODES = ("stereo", "residuals", "none")
SELECTION_METHODS = ("coverage", "kmedoids", "image_subset", "all")
CAMERA_SOURCES = ("ba", "published")
SFS_CAMERA_MODELS = ("csm", "isis")
