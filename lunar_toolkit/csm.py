"""Read, write and resample USGS CSM linescan cameras (ISD and model state) to match ISIS ``reduce`` / ``crop``."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import isis
from .tools import is_dry_run

LINESCAN = "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL"


@dataclass
class CsmCamera:
    """A CSM camera file: an ALE / ``isd_generate`` ISD or a USGS CSM model state.

    Attributes
    ----------
    data : dict
        The parsed JSON (ISD keys such as ``image_lines`` or state keys such
        as ``m_nLines``).
    kind : {"isd", "state"}
    header : str or None
        For a state, the model-name line that precedes the JSON in
        ``getModelState()`` output and ASP ``.adjusted_state.json`` files
        (``None`` when the file is bare JSON).
    """

    data: dict[str, Any]
    kind: str
    header: str | None = None

    @property
    def model(self) -> str:
        key = "name_model" if self.kind == "isd" else "m_modelName"
        return str(self.data.get(key, self.header or ""))

    @property
    def size(self) -> tuple[int, int]:
        """``(samples, lines)`` of the image the camera describes."""
        if self.kind == "isd":
            return int(self.data["image_samples"]), int(self.data["image_lines"])
        return int(self.data["m_nSamples"]), int(self.data["m_nLines"])


def parse_camera(text: str) -> CsmCamera:
    """Parse a CSM camera from text (ISD JSON, or a model state with or without its name line)."""
    stripped = text.lstrip()
    header = None
    if not stripped.startswith("{"):
        first, _, rest = stripped.partition("\n")
        header, stripped = first.strip(), rest
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("CSM camera JSON must be an object")
    if "m_nLines" in data:
        return CsmCamera(data, "state", header)
    if "image_lines" in data:
        return CsmCamera(data, "isd", None)
    raise ValueError("Not a CSM ISD (image_lines) or model state (m_nLines)")


def read_camera(path: Path) -> CsmCamera:
    """Read a CSM camera file (``.json`` ISD, model state or ASP ``.adjusted_state.json``)."""
    return parse_camera(Path(path).read_text())


def format_camera(cam: CsmCamera) -> str:
    """Text of *cam* in its original layout (state name line kept)."""
    body = json.dumps(cam.data, indent=2)
    return f"{cam.header}\n{body}\n" if cam.header else body + "\n"


def write_camera(cam: CsmCamera, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_camera(cam))
    return path


def transform_camera(
    cam: CsmCamera,
    sample_scale: float,
    sample_offset: float,
    line_scale: float,
    line_offset: float,
    size: tuple[int, int],
) -> CsmCamera:
    """Camera for an image whose pixels map affinely onto *cam*'s image.

    CSM image coordinates put the top-left pixel edge at 0 (first pixel
    centre at 0.5).  A point at ``(x, y)`` of the new image is
    ``(sample_scale * x + sample_offset, line_scale * y + line_offset)`` of
    the old one.  Across track this scales the detector sample summing and
    shifts the starting detector sample; along track it scales the
    integration times (and the line summing, which converts the residual
    detector-line offset into image lines in ``groundToImage``) and moves the
    integration start lines so that line ``y`` is exposed when old line
    ``line_scale * y + line_offset`` was.

    Parameters
    ----------
    cam : CsmCamera
        A USGS linescan ISD or state.
    sample_scale, sample_offset, line_scale, line_offset : float
        The mapping above (ISIS ``reduce`` by *f*: scales *f*, offsets 0;
        ISIS ``crop`` from sample *s*, line *l*: scales 1, offsets
        ``s - 1``, ``l - 1``).
    size : (samples, lines)
        Size of the new image.

    Returns
    -------
    CsmCamera
        A new camera of the same kind.
    """
    if cam.model != LINESCAN:
        raise ValueError(f"Only {LINESCAN} cameras are supported, not {cam.model or 'unknown'}")
    if sample_scale <= 0 or line_scale <= 0:
        raise ValueError("Scales must be positive")
    d = json.loads(json.dumps(cam.data))
    ns, nl = int(size[0]), int(size[1])

    def new_lines(lines: list[float]) -> list[float]:
        return [(v - 0.5 - line_offset) / line_scale + 0.5 for v in lines]

    if cam.kind == "isd":
        ss = float(d.get("detector_sample_summing", 1.0))
        d["starting_detector_sample"] = float(d.get("starting_detector_sample", 0.0)) + sample_offset * ss
        _isd_summing(d, "sample", ss * sample_scale)
        _isd_summing(d, "line", float(d.get("detector_line_summing", 1.0)) * line_scale)
        rates = d["line_scan_rate"]
        d["line_scan_rate"] = [
            [new_lines([float(r[0])])[0], float(r[1]), float(r[2]) * line_scale] for r in rates
        ]
        d["image_samples"], d["image_lines"] = ns, nl
        return CsmCamera(d, "isd")

    ss = float(d["m_detectorSampleSumming"])
    d["m_startingDetectorSample"] = float(d["m_startingDetectorSample"]) + sample_offset * ss
    d["m_detectorSampleSumming"] = ss * sample_scale
    d["m_detectorLineSumming"] = float(d["m_detectorLineSumming"]) * line_scale
    d["m_intTimeLines"] = new_lines([float(v) for v in d["m_intTimeLines"]])
    d["m_intTimes"] = [float(v) * line_scale for v in d["m_intTimes"]]
    d["m_nSamples"], d["m_nLines"] = ns, nl
    if "m_gsd" in d:
        d["m_gsd"] = float(d["m_gsd"]) * math.sqrt(sample_scale * line_scale)
        if "m_halfSwath" in d:
            d["m_halfSwath"] = d["m_gsd"] * ns / 2.0
    starts, times, lines = d["m_intTimeStartTimes"], d["m_intTimes"], d["m_intTimeLines"]
    if "m_halfTime" in d and starts:
        d["m_halfTime"] = (starts[-1] - starts[0] + times[-1] * (nl - lines[-1])) / 2.0
    return CsmCamera(d, "state", cam.header)


def _isd_summing(d: dict[str, Any], axis: str, summing: float) -> None:
    """Set an ISD detector summing; usgscsm reads it as an integer, so a fractional
    remainder ``k`` goes into that axis' focal2pixel row, detector centre and
    starting detector pixel (all divided by ``k``, which leaves focal-plane
    coordinates unchanged)."""
    n = max(1, round(summing))
    d[f"detector_{axis}_summing"] = n
    k = summing / n
    if abs(k - 1.0) < 1e-12:
        return
    key = "lines" if axis == "line" else "samples"
    d[f"focal2pixel_{key}"] = [float(v) / k for v in d[f"focal2pixel_{key}"]]
    centre = dict(d.get("detector_center", {}))
    centre[axis] = float(centre.get(axis, 0.0)) / k
    d["detector_center"] = centre
    d[f"starting_detector_{axis}"] = float(d.get(f"starting_detector_{axis}", 0.0)) / k


def reduce_camera(
    camera: Path | CsmCamera,
    factor: float,
    out_json: Path | None = None,
    size: tuple[int, int] | None = None,
) -> CsmCamera:
    """CSM camera of an image averaged by *factor* (ISIS ``reduce``, ``mode=scale``).

    ``reduce`` output pixel *k* (0-based) averages input pixels
    ``k*f .. k*f + f - 1``, so in CSM coordinates (pixel edges at integers)
    the mapping is a pure scale by *f*: summing and integration times grow
    by *f*, sizes shrink by *f*.

    Parameters
    ----------
    camera : Path or CsmCamera
        ISD or state of the full-resolution image.
    factor : float
        ``sscale`` = ``lscale``.
    out_json : Path, optional
        Where to write the result (same format as the input).
    size : (samples, lines), optional
        Size of the reduced image; default ``ceil(size / factor)``.
    """
    cam = camera if isinstance(camera, CsmCamera) else read_camera(camera)
    if size is None:
        ns, nl = cam.size
        size = (math.ceil(ns / factor), math.ceil(nl / factor))
    out = transform_camera(cam, factor, 0.0, factor, 0.0, size)
    if out_json is not None:
        write_camera(out, out_json)
    return out


def read_isis_label(cube: Path, max_bytes: int = 1 << 22) -> str:
    """The attached PVL label of an ISIS cube (text up to the closing ``End``)."""
    with open(cube, "rb") as f:
        head = f.read(max_bytes).decode("latin-1")
    m = re.search(r"^End\s*$", head, flags=re.M)
    return head[: m.end()] if m else head


def _group(label: str, name: str) -> dict[str, str] | None:
    m = re.search(rf"^\s*Group\s*=\s*{name}\s*$(.*?)^\s*End_Group", label, flags=re.M | re.S)
    if not m:
        return None
    return dict(re.findall(r"^\s*(\w+)\s*=\s*(\S+)", m.group(1), flags=re.M))


@dataclass(frozen=True)
class CubeGeometry:
    """Size of an ISIS cube and its pixel mapping onto the original (alpha) image.

    ``alpha = scale * beta + offset`` per axis in CSM coordinates (pixel
    edges at integers), from the cube's ``AlphaCube`` group (identity when
    absent).
    """

    samples: int
    lines: int
    alpha_samples: int
    alpha_lines: int
    sample_scale: float = 1.0
    sample_offset: float = 0.0
    line_scale: float = 1.0
    line_offset: float = 0.0

    @classmethod
    def from_label(cls, label: str) -> CubeGeometry:
        dims = _group(label, "Dimensions")
        if not dims:
            raise ValueError("No Dimensions group in the ISIS label")
        ns, nl = int(dims["Samples"]), int(dims["Lines"])
        a = _group(label, "AlphaCube")
        if not a:
            return cls(ns, nl, ns, nl)
        s0, s1 = float(a["AlphaStartingSample"]), float(a["AlphaEndingSample"])
        l0, l1 = float(a["AlphaStartingLine"]), float(a["AlphaEndingLine"])
        bs, bl = int(a["BetaSamples"]), int(a["BetaLines"])
        return cls(
            ns,
            nl,
            int(a["AlphaSamples"]),
            int(a["AlphaLines"]),
            (s1 - s0) / bs,
            s0 - 0.5,
            (l1 - l0) / bl,
            l0 - 0.5,
        )

    @classmethod
    def from_cube(cls, cube: Path) -> CubeGeometry:
        return cls.from_label(read_isis_label(cube))


def camera_for_cube(cam: CsmCamera, cube: CubeGeometry, source: CubeGeometry | None = None) -> CsmCamera:
    """Re-map *cam* so it describes *cube* pixel for pixel.

    *cam* is taken to describe the original (alpha) image when its size
    matches ``cube.alpha_*`` (cameras from ``isd_generate`` on the
    uncropped cube, as in the ASP LRONAC example), else the *source* cube
    that *cube* was derived from (``reduce`` / ``crop`` of a cube whose own
    camera was written for it).
    """
    size = cam.size
    ss, so, ls, lo = cube.sample_scale, cube.sample_offset, cube.line_scale, cube.line_offset
    if size != (cube.alpha_samples, cube.alpha_lines):
        if source is None or size != (source.samples, source.lines):
            raise ValueError(
                f"Camera size {size} matches neither the alpha image "
                f"{(cube.alpha_samples, cube.alpha_lines)} nor the source cube"
            )
        # cube -> alpha, then alpha -> source (inverse of the source's AlphaCube)
        ss, so = ss / source.sample_scale, (so - source.sample_offset) / source.sample_scale
        ls, lo = ls / source.line_scale, (lo - source.line_offset) / source.line_scale
    return transform_camera(cam, ss, so, ls, lo, (cube.samples, cube.lines))


def isis_reduce_with_csm(
    cube: Path,
    camera: Path,
    factor: int,
    out_dir: Path,
    algorithm: str = "average",
    force: bool = False,
) -> tuple[Path, Path]:
    """ISIS ``reduce`` a cube by *factor* and write the matching CSM camera.

    Parameters
    ----------
    cube : Path
        Input ISIS cube (may be a crop: its ``AlphaCube`` is honoured).
    camera : Path
        CSM ISD or state of the original image or of *cube*.
    factor : int
        ``sscale`` = ``lscale``.
    out_dir : Path
        Receives ``<cube stem>.red<factor>.cub`` and ``<camera stem>.red<factor>.json``
        (same CSM format as *camera*).
    algorithm : str
        ``reduce`` algorithm (``average`` as in Bertone et al. 2026, App. A).
    force : bool
        Redo existing outputs.

    Returns
    -------
    (reduced_cube, reduced_camera)
    """
    out_dir = Path(out_dir)
    stem = Path(cube).stem
    red = out_dir / f"{stem}.red{factor}.cub"
    out_cam = out_dir / f"{Path(camera).stem}.red{factor}.json"  # named after its source camera
    if force or not red.exists():
        if not is_dry_run():
            out_dir.mkdir(parents=True, exist_ok=True)
        isis.reduce(cube, red, factor, algorithm=algorithm)
    if is_dry_run():
        return red, out_cam
    if force or not out_cam.exists():
        geom = CubeGeometry.from_cube(red)
        write_camera(camera_for_cube(read_camera(camera), geom, CubeGeometry.from_cube(cube)), out_cam)
    return red, out_cam
