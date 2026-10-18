"""Shape-from-shading DEMs from LRO NAC images with ISIS and ASP (Bertone et al. 2026 workflow)."""

from __future__ import annotations

import csv
import dataclasses
import importlib
import logging
import math
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import asp, isis
from .constants import (
    CAMERA_SOURCES,
    MOON_RADIUS_KM,
    REGISTRATION_MODES,
    SELECTION_METHODS,
    SFS_CAMERA_MODELS,
)
from .geometry import load_aoi, ps_xy
from .selection import select_kmedoids
from .stereo import azimuth_difference
from .stereo_dem import stereo_dem
from .tools import ToolError, is_dry_run, reuse_existing

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .config import RunConfig
    from .provenance import Provenance
    from .workspace import Workspace

log = logging.getLogger(__name__)

# ASP "Bundle adjustment" recipe for many LRO NAC images under varied illumination
# (docs/sfs_usage.rst; --match-first-to-last: near the poles the Sun loops the sky).
ASP_BUNDLE_ADJUST: dict[str, Any] = {
    "ip_per_image": 30000,
    "overlap_limit": 100,
    "num_iterations": 100,
    "num_passes": 2,
    "min_matches": 1,
    "max_pairwise_matches": 2000,
    "camera_weight": 0,
    "robust_threshold": 2,
    "tri_weight": 0.05,
    "tri_robust_threshold": 0.05,
    "remove_outliers_params": "75.0 3.0 100 100",
    "save_intermediate_cameras": True,
    "match_first_to_last": True,
    "forced_triangulation_distance": 100000,
    "min_triangulation_angle": 1e-10,
    "datum": "D_MOON",
}

# ASP "Registration refinement" recipe (bundle_adjust --heights-from-dem, reusing the matches).
ASP_REFINE: dict[str, Any] = {
    "max_pairwise_matches": 2000,
    "match_first_to_last": True,
    "min_matches": 1,
    "num_iterations": 100,
    "num_passes": 2,
    "camera_weight": 0,
    "save_intermediate_cameras": True,
    "heights_from_dem_uncertainty": 20.0,
    "forced_triangulation_distance": 100000,
    "min_triangulation_angle": 1e-10,
    "remove_outliers_params": "75.0 3.0 100 100",
    "parameter_tolerance": 1e-20,
    "datum": "D_MOON",
}

# ASP "Running SfS in parallel" recipe.
ASP_SFS: dict[str, Any] = {
    "tile_size": 200,
    "padding": 50,
    "shadow_threshold": 0.005,
    "crop_input_images": True,
    "blending_dist": 10,
    "min_blend_size": 50,
    "allow_borderline_data": True,
    "low_light_threshold": 0.03,
    "smoothness_weight": 0.08,
    "initial_dem_constraint_weight": 0.0025,
    "reflectance_type": 1,
    "max_iterations": 5,
    "save_sparingly": True,
}

# Bertone et al. (2026): match each image against the next 200 in Sun-azimuth order whose
# footprints overlap (``auto_overlap_percent`` → ``--auto-overlap-params "<dem> 15"``), and
# soft-anchor cameras at 100 m horizontal / vertical (App. A); intrinsics stay fixed
# (ASP's default without --solve-intrinsics).
_UNCERTAINTY = {"camera_position_uncertainty": "100,100"}
BERTONE_BUNDLE_ADJUST: dict[str, Any] = {
    **ASP_BUNDLE_ADJUST,
    # The next 200 images in Sun-azimuth order (Bertone et al. 2026, §2).  Footprint-based
    # pairs (`auto_overlap_percent`) need cameras that fall mostly on the prior DEM, which
    # a crop does not satisfy; set it as an override for full-region runs.
    "overlap_limit": 200,
    **_UNCERTAINTY,
}
BERTONE_REFINE: dict[str, Any] = {**ASP_REFINE, **_UNCERTAINTY}
# §2: mu = 0.05–0.1, lambda = 1e-3, five iterations; App. A: shadow threshold 0.002, albedo fixed.
BERTONE_SFS: dict[str, Any] = {
    **ASP_SFS,
    "shadow_threshold": 0.002,
    "smoothness_weight": 0.08,
    "initial_dem_constraint_weight": 1e-3,
    "max_iterations": 5,
}

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "bertone": {"bundle_adjust": BERTONE_BUNDLE_ADJUST, "refine": BERTONE_REFINE, "sfs": BERTONE_SFS},
    "asp": {"bundle_adjust": ASP_BUNDLE_ADJUST, "refine": ASP_REFINE, "sfs": ASP_SFS},
}


_OVERLAP_OPTIONS = ("overlap_list", "auto_overlap_params", "auto_overlap_buffer", "overlap_limit")


@dataclass
class SfsConfig:
    """Settings for :class:`SfsPipeline`.

    Parameters
    ----------
    resolution : float
        SfS DEM grid spacing (m/px).
    projwin : tuple, optional
        ``(ulx, uly, lrx, lry)`` crop of the prior DEM, in its projected units.
    preset : {"bertone", "asp"}
        Option defaults: Bertone et al. (2026) or the ASP SfS recipe (:data:`PRESETS`).
    ba_resolution : float, optional
        Grid spacing of the images mapprojected for bundle-adjustment matching and
        stereo registration (default 1 m, NAC native; None = *resolution*).
    reduce : int, "auto" or None
        ISIS ``reduce`` factor bringing the cubes to *resolution* before SfS.
        ``"auto"``: ``round(resolution / native)`` per image (native = observation
        ``pixel_scale``, else *native_resolution*).  Reduced cubes use ISIS cameras
        (with ``--use-approx-camera-models``); bundle adjustment still uses CSM.
    native_resolution : float
        Assumed image m/px when the observation table does not give it.
    csm : bool
        CSM cameras (7–15× faster than ISIS) for mapproject, bundle_adjust and stereo.
    web_spice : bool
        ``spiceinit web=true`` (USGS SPICE web service) instead of local kernels.
    registration : {"stereo", "residuals", "none"}
        Registering the adjusted cameras to the prior DEM (translation only).
        ``"stereo"``: DEMs from up to *stereo_pairs* pairs, mosaicked and
        ``pc_align``-ed (falls back to ``"residuals"`` without pairs);
        ``"residuals"``: the bundle-adjustment triangulated points.
    stereo_pairs : int
        Maximum stereo pairs for ``"stereo"`` registration.
    stereo_convergence : (min, max)
        Convergence-angle limits (deg) for those pairs.
    max_displacement : float
        ``pc_align --max-displacement`` (m).
    refine : bool
        Re-run bundle adjustment on the kept, registered cameras with
        ``--heights-from-dem`` (reusing the matches).
    max_offset_p85 : float, optional
        Drop images whose 85th-percentile mapprojected match offset (m) is larger.
    max_large_residual_fraction : float, optional
        Drop images with more than this fraction of reprojection errors over
        *large_residual_px* pixels.
    min_matches : int
        Drop images with fewer inlier matches.
    selection : {"coverage", "image_subset", "all"}
        SfS image selection: greedy per-pixel illumination coverage
        (:func:`select_by_illumination`), ASP ``image_subset`` per Sun-azimuth bin,
        or all kept images.
    max_shadow_fraction : float, optional
        Images at least this shadowed (below the SfS shadow threshold) are dropped.
    selection_views, selection_azimuth_sep : int, float
        Illumination directions wanted per pixel and their minimum separation (deg).
    max_sfs_images : int, optional
        Cap on the selected images (``"coverage"`` / ``"kmedoids"`` selection).
    selection_clusters : int
        Medoids seeding ``"kmedoids"`` selection (Bertone et al. 2026, §2: ~30).
    sfs_cameras : {"csm", "isis"}
        Camera models for SfS on reduced cubes: CSM cameras re-mapped to the reduced
        image (:func:`lunar_toolkit.csm.isis_reduce_with_csm`, as in Bertone et al.
        2026, App. A) or ISIS cameras with ``--use-approx-camera-models``.
    cameras_from : {"ba", "published"}
        ``"published"`` uses the bundle-adjusted cameras of Bertone et al. (2026) for
        *published_region* (:func:`lunar_toolkit.reference.published_cameras`) and skips
        bundle adjustment, registration and refinement; images without one are removed.
    published_region : str or int, optional
        Region of the published cameras (default: from ``region_id`` ``A3NN``).
    max_sfs_slew : float, optional
        Only images with ``|slew| <`` this (deg) are selected for SfS (Bertone et al.
        2026, §2: nadir images); slewed images still join bundle adjustment and stereo
        registration.  Unknown slews count as nadir; ``None`` disables the cut.
    selection_grid : int
        Cells along the longer side of the selection grid.
    region_id : str, optional
        ``ROIID`` for product file names (default: the work directory name).
    ensemble_members : int
        SfS runs on random subsets of the kept images (sharing at most 40% of their
        images) over an *ensemble_crop* m square at the DEM centre, for the ensemble
        uncertainty maps (Bertone et al. 2026, §3.2); 0 skips this.
    ensemble_size : int, optional
        Images per ensemble member (default: as many as selected).
    ensemble_crop : float
        Side of the ensemble crop (m).
    seed : int
        Random seed of the ensemble image subsets.
    processes, threads : int, optional
        Passed to ``parallel_bundle_adjust``, ``parallel_stereo``, ``parallel_sfs`` and ``mapproject``.
    bundle_adjust_options, refine_options, sfs_options : dict
        Overrides merged onto the preset; a value of ``None`` drops a default.
    """

    resolution: float = 5.0
    projwin: tuple[float, float, float, float] | None = None
    preset: str = "bertone"
    ba_resolution: float | None = 1.0
    reduce: int | str | None = "auto"
    native_resolution: float = 1.0
    csm: bool = True
    web_spice: bool = True
    registration: str = "stereo"
    stereo_pairs: int = 30
    stereo_convergence: tuple[float, float] = (5.0, 60.0)
    max_displacement: float = 400.0
    refine: bool = True
    max_offset_p85: float | None = 1.8
    max_large_residual_fraction: float | None = 0.85
    large_residual_px: float = 1.0
    min_matches: int = 1
    selection: str = "coverage"
    max_shadow_fraction: float | None = 0.95
    selection_views: int = 3
    selection_azimuth_sep: float = 30.0
    max_sfs_images: int | None = None
    selection_clusters: int = 30
    sfs_cameras: str = "csm"
    cameras_from: str = "ba"
    published_region: str | int | None = None
    max_sfs_slew: float | None = 5.0
    selection_grid: int = 250
    region_id: str | None = None
    ensemble_members: int = 0
    ensemble_size: int | None = None
    ensemble_crop: float = 2500.0
    seed: int = 0
    processes: int | None = None
    threads: int | None = None
    bundle_adjust_options: dict[str, Any] = field(default_factory=dict)
    refine_options: dict[str, Any] = field(default_factory=dict)
    sfs_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("preset", self.preset, tuple(PRESETS)),
            ("registration", self.registration, REGISTRATION_MODES),
            ("selection", self.selection, SELECTION_METHODS),
            ("sfs_cameras", self.sfs_cameras, SFS_CAMERA_MODELS),
            ("cameras_from", self.cameras_from, CAMERA_SOURCES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}, not {value!r}")
        if isinstance(self.reduce, str) and self.reduce != "auto":
            raise ValueError(f"reduce must be an integer, 'auto' or None, not {self.reduce!r}")

    def options(self, step: str) -> dict[str, Any]:
        """Preset options for ``"bundle_adjust"``, ``"refine"`` or ``"sfs"`` with overrides applied."""
        extra = {
            "bundle_adjust": self.bundle_adjust_options,
            "refine": self.refine_options,
            "sfs": self.sfs_options,
        }[step]
        return {**PRESETS[self.preset][step], **extra}


@dataclass
class SfsImage:
    """One input image: cube, optional CSM camera, Sun angles and derived files."""

    product_id: str
    cube: Path
    camera: Path | None = None
    sun_azimuth: float | None = None
    sun_elevation: float | None = None
    pixel_scale: float | None = None
    mapprojected: Path | None = None
    shadow_fraction: float | None = None
    slew_angle: float | None = None


@dataclass(frozen=True)
class RemovedImage:
    """An image dropped by a quality filter: *stage* (``ba``, ``refine``, ``selection``) and why."""

    image: SfsImage
    stage: str
    reason: str


class SfsPipeline:
    """Build an SfS DEM over a region from NAC images and a prior DEM.

    :meth:`run` follows Bertone et al. (2026, PSJ 7:118) and the ASP large-area SfS recipe:

    1. calibrate the images and write CSM cameras (``images/``);
    2. crop the prior DEM and resample it to the target grid (``dem/``);
    3. sort by Sun azimuth (``sfs --query``);
    4. mapproject at native resolution (``prj/``) and bundle-adjust with matches between
       the mapprojected images (``ba/``);
    5. drop images with no matches, large mapprojected match offsets or mostly large
       residuals (``selection/removed.csv``);
    6. register the cameras to the prior DEM, translation only, from stereo DEMs
       (``stereo/``) or the BA points; apply to all cameras (``align/``, ``ba_align/``);
    7. refine with ``--heights-from-dem`` and filter again (``ba_refine/``);
    8. mapproject at the target resolution (``prj_sfs/``), write max-lit QA mosaics,
       all images and per 30° Sun-azimuth bin (``qa/``);
    9. select an illumination-diverse subset (``selection/``);
    10. reduce the selected cubes to the target resolution and run ``parallel_sfs`` (``sfs/``);
    11. products (``products/`` via :mod:`lunar_toolkit.sfs_products`, else hillshade + diff).

    Steps whose outputs exist are skipped, so a rerun resumes.

    Parameters
    ----------
    work_dir : Path
        Directory for all intermediate and final files.
    prior_dem : Path
        Reference terrain, e.g. a LOLA polar stereographic LDEM GeoTIFF.
    config : SfsConfig, optional
    force : bool
        Recompute steps even when their outputs exist.
    observations : DataFrame, optional
        Observation table (:mod:`lunar_toolkit.observation` schema) with the input
        images, for native pixel scales and stereo pairs; looked up in the local
        EDR ``CumIndex`` when needed and not given.
    aoi : optional
        Region the stereo pairs must overlap in: anything
        :func:`lunar_toolkit.stereo.find_stereo_pairs` takes (dict, GeoJSON / WKT).
    data_dir : Path, optional
        ``CumIndex`` data directory.
    run_config : RunConfig, optional
        The full run settings (:mod:`lunar_toolkit.config`) written to ``config.yaml``
        and ``provenance.json``; default: *work_dir*, *prior_dem*, the sources and *config*.
    provenance : bool
        Write ``config.yaml`` and append this run to ``provenance.json`` (never in a dry run).

    Attributes
    ----------
    all_images, images : list of SfsImage
        Every input / those passing the quality filters (Sun-azimuth order).
    removed : list of RemovedImage
    selected, unselected : list of SfsImage
        The SfS inputs and the remaining good images (e.g. for an uncertainty ensemble);
        both carry ``mapprojected`` (``prj_sfs/``) and ``sun_azimuth``.
    dem : Path
        The cropped prior DEM on the target grid.

    Example
    -------
    ::

        cfg = SfsConfig(resolution=5, projwin=(-10000, 50000, 0, 40000))
        p = SfsPipeline("work/malapert", "LDEM_87S_5MPP_ADJ.TIF", cfg)
        dem = p.run(["raw/M1234567890LE.IMG", ...])
    """

    def __init__(
        self,
        work_dir: Path,
        prior_dem: Path,
        config: SfsConfig | None = None,
        force: bool = False,
        *,
        observations: pd.DataFrame | None = None,
        aoi: Any = None,
        data_dir: Path | None = None,
        run_config: RunConfig | None = None,
        provenance: bool = True,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.prior_dem = Path(prior_dem)
        self.config = config or SfsConfig()
        self.force = force
        self.observations = observations
        self.aoi = aoi
        self.data_dir = data_dir
        self.all_images: list[SfsImage] = []
        self.images: list[SfsImage] = []
        self.removed: list[RemovedImage] = []
        self.selected: list[SfsImage] = []
        self.unselected: list[SfsImage] = []
        self.dem = self.work_dir / "dem" / f"prior-{self.config.resolution:g}m.tif"
        self._registered = False
        self.run_config = run_config
        self.provenance = provenance
        self._prov: Provenance | None = None

    def run(self, sources: Sequence[Path]) -> Path:
        """All steps; returns the final SfS DEM.

        Writes ``config.yaml`` and appends a run to ``provenance.json`` first (see
        :meth:`start_provenance`), then records each step there.
        """
        sources = [Path(s) for s in sources]
        self.start_provenance(sources)
        step = self._step
        try:
            step("prepare_images", self.prepare_images, sources)
            step("prepare_dem", self.prepare_dem)
            step("sort_by_sun", self.sort_by_sun)
            step(
                "mapproject",
                self.mapproject,
                self.images,
                "prj",
                self.config.ba_resolution or self.config.resolution,
            )
            if self.config.cameras_from == "published":
                step("published_cameras", self.use_published_cameras)
            else:
                step("bundle_adjust", self.bundle_adjust)
                step("filter_ba", self.filter_images, self.ba_prefix, "ba")
                step("register", self.register)
                if self.config.refine:
                    step("refine", self.refine)
                    step("filter_refine", self.filter_images, self.refine_prefix, "refine")
            step("mapproject_for_sfs", self.mapproject_for_sfs)
            step("qa_mosaics", self.qa_mosaics)
            step("select_images", self.select_images)
            dem = step("shape_from_shading", self.shape_from_shading)
            step("products", self.products, dem)
            if self.config.ensemble_members:
                step("ensemble", self.ensemble)
        except BaseException:
            if self._prov:
                self._prov.finish("error")
            raise
        if self._prov:
            self._prov.finish("ok")
        return dem

    def effective_config(self, sources: Sequence[Path] = ()) -> RunConfig:
        """:attr:`run_config` with this pipeline's work dir, prior DEM and :class:`SfsConfig`."""
        from .config import RunConfig

        if self.run_config is None:
            return RunConfig(
                work_dir=self.work_dir,
                dem=str(self.prior_dem),
                sources=[str(s) for s in sources],
                sfs=self.config,
            )
        return dataclasses.replace(
            self.run_config, work_dir=self.work_dir, dem=str(self.prior_dem), sfs=self.config
        )

    def start_provenance(self, sources: Sequence[Path]) -> Path | None:
        """Write ``config.yaml`` and start a ``provenance.json`` run; returns the config path.

        Inputs are *sources* and the prior DEM.  Nothing is written in a dry run
        or with ``provenance=False``.
        """
        if is_dry_run() or not self.provenance:
            return None
        from .config import dump_config
        from .provenance import Provenance

        cfg = self.effective_config(sources)
        path = self.work_dir / "config.yaml"
        dump_config(cfg, path)
        self._prov = Provenance(self.work_dir, cfg, [*sources, self.prior_dem], checksums=cfg.checksums)
        return path

    def _step(self, name: str, func: Any, *args: Any) -> Any:
        if self._prov is None:
            return func(*args)
        with self._prov.step(name) as entry:
            entry["result"] = result = func(*args)
        return result

    @property
    def ba_prefix(self) -> Path:
        return self.work_dir / "ba" / "run"

    @property
    def align_prefix(self) -> Path:
        return self.work_dir / "align" / "run"

    @property
    def ba_align_prefix(self) -> Path:
        return self.work_dir / "ba_align" / "run"

    @property
    def refine_prefix(self) -> Path:
        return self.work_dir / "ba_refine" / "run"

    @property
    def sfs_camera_prefix(self) -> Path:
        """Adjustments renamed for the reduced ISIS cubes."""
        return self.work_dir / "sfs_cameras" / "run"

    @property
    def sfs_prefix(self) -> Path:
        return self.work_dir / "sfs" / "run"

    @property
    def sfs_dem(self) -> Path:
        return Path(f"{self.sfs_prefix}-DEM-final.tif")

    @property
    def camera_prefix(self) -> Path | None:
        """Adjustments of the final (registered, refined) cameras; None for published cameras."""
        if self.config.cameras_from == "published":
            return None
        if self.config.refine:
            return self.refine_prefix
        return self.ba_align_prefix if self.config.registration != "none" else self.ba_prefix

    @property
    def shadow_threshold(self) -> float:
        return float(self.config.options("sfs").get("shadow_threshold") or 0.0)

    def _skip(self, path: Path) -> bool:
        return reuse_existing(path, self.force)

    def _parallel_opts(self) -> dict[str, Any]:
        return {"processes": self.config.processes, "threads": self.config.threads}

    def _cameras(self, images: Sequence[SfsImage]) -> list[Path]:
        return [im.camera for im in images if im.camera] if self.config.csm else []

    def _map_path(self, subdir: str, im: SfsImage) -> Path:
        return self.work_dir / subdir / f"{im.product_id}.map.tif"

    def prepare_images(self, sources: Sequence[Path]) -> list[SfsImage]:
        """Calibrate EDRs (``.IMG``) or take ready cubes; write CSM cameras."""
        cfg = self.config
        scales = _by_id(self.observations, "pixel_scale")
        slews = _by_id(self.observations, "slew_angle")
        self.all_images = []
        for src in map(Path, sources):
            cube, camera = isis.prepare_lronac(
                src, self.work_dir / "images", csm=cfg.csm, web=cfg.web_spice, force=self.force
            )
            pid = src.name.split(".")[0]
            self.all_images.append(
                SfsImage(
                    pid, cube, camera, pixel_scale=scales.get(pid.upper()), slew_angle=slews.get(pid.upper())
                )
            )
        self.images = list(self.all_images)
        self._record_observations()
        return self.images

    def _record_observations(self) -> None:
        """Write the given observation rows to the workspace ``images`` table (not in a dry run)."""
        if is_dry_run() or self.observations is None or not len(self.observations):
            return
        from .workspace import WorkspaceError

        try:
            with self.workspace() as ws:
                ws.add_observations(self.observations)
        except WorkspaceError as exc:
            log.warning("Workspace images not written: %s", exc)

    def prepare_dem(self) -> Path:
        """Crop the prior DEM to ``projwin`` and resample to the target grid."""
        cfg = self.config
        crop = self.work_dir / "dem" / "prior.tif"
        if not self._skip(crop):
            asp.gdal_translate(self.prior_dem, crop, projwin=cfg.projwin)
        if not self._skip(self.dem):
            asp.gdal_translate(crop, self.dem, tr=cfg.resolution, r="cubicspline")
        return self.dem

    def sort_by_sun(self) -> list[SfsImage]:
        """Fill Sun azimuth / elevation and sort the images by azimuth.

        Bundle adjustment matches neighbours in this order, and ``parallel_sfs``
        expects it.
        """
        angles = asp.sun_angles(
            self.dem,
            [im.cube for im in self.images],
            self._cameras(self.images),
            out_prefix=self.work_dir / "query" / "run",
        )
        by_image = {Path(a.image).name: a for a in angles}
        for im in self.images:
            a = by_image.get(im.cube.name)
            if a:
                im.sun_azimuth, im.sun_elevation = a.azimuth, a.elevation
        if angles:
            self.images.sort(key=lambda im: (im.sun_azimuth is None, im.sun_azimuth or 0.0))
        elif not is_dry_run():
            log.warning("sfs --query reported no Sun angles; keeping input order")
        return self.images

    def mapproject(
        self, images: Sequence[SfsImage], subdir: str, resolution: float, **options: Any
    ) -> list[Path]:
        """Mapproject *images* onto the prior DEM (cropped to ``projwin``) into ``<subdir>/``."""
        cfg = self.config
        win = cfg.projwin
        outs = []
        for im in images:
            out = self._map_path(subdir, im)
            if not self._skip(out):
                asp.mapproject(
                    self.dem,
                    im.cube,
                    out,
                    im.camera if cfg.csm else None,
                    tr=resolution,
                    t_projwin=(win[0], win[3], win[2], win[1]) if win else None,
                    threads=cfg.threads,
                    **options,
                )
            outs.append(out)
        return outs

    def bundle_adjust(self) -> Path:
        """``parallel_bundle_adjust`` with matches from the mapprojected images.

        ``--mapproj-dem`` writes the per-image match-offset report used by
        :meth:`filter_images`.
        """
        done = Path(f"{self.ba_prefix}-final_residuals_stats.txt")
        if self._skip(done):
            return self.ba_prefix
        opts = self._ba_options(self.config.options("bundle_adjust"))
        return asp.bundle_adjust(
            [im.cube for im in self.images],
            self.ba_prefix,
            self._cameras(self.images),
            mapprojected=[self._map_path("prj", im) for im in self.images],
            dem=self.dem,
            parallel=True,
            mapproj_dem=self.dem,
            **{**self._parallel_opts(), **opts},
        )

    def _ba_options(self, opts: dict[str, Any]) -> dict[str, Any]:
        """Preset bundle-adjust options made valid for ASP.

        ``auto_overlap_percent`` becomes ``--auto-overlap-params "<dem> <pct>"``, and of
        ASP's mutually exclusive pair-selection options only the most specific is kept
        (overlap list, footprint overlap, overlap buffer, then consecutive-image limit).
        """
        opts = dict(opts)
        pct = opts.pop("auto_overlap_percent", None)
        if pct is not None and "auto_overlap_params" not in opts:
            opts["auto_overlap_params"] = f"{self.dem} {pct:g}"
        given = [k for k in _OVERLAP_OPTIONS if opts.get(k) not in (None, False)]
        for key in given[1:]:
            log.info(
                "bundle_adjust: --%s overrides --%s; dropping the latter",
                given[0].replace("_", "-"),
                key.replace("_", "-"),
            )
            del opts[key]
        return opts

    def workspace(self, read_only: bool = False) -> Workspace:
        """The work dir's DuckDB workspace (``lunar.duckdb``, :mod:`lunar_toolkit.workspace`)."""
        from .workspace import Workspace

        return Workspace(self.work_dir, threads=self.config.threads, read_only=read_only)

    def filter_images(self, prefix: Path, stage: str) -> list[RemovedImage]:
        """Drop images that ASP's bundle-adjustment reports flag (see :func:`flag_images`).

        Loads ``<prefix>-*`` reports (residual stats and raw pixels, match offsets, point
        maps, convergence angles) into the workspace as run ``<prefix>``'s directory name
        (``ba``, ``ba_refine``) and removes the images of its ``image_rejections`` view,
        with this config's thresholds.  Removed images are logged and appended to
        ``selection/removed.csv``.
        """
        cfg = self.config
        files = {
            k: Path(f"{prefix}-{name}")
            for k, name in (
                ("stats", "final_residuals_stats.txt"),
                ("raw", "final_residuals_raw_pixels.txt"),
                ("offsets", "mapproj_match_offset_stats.txt"),
            )
        }
        if not any(p.exists() for p in files.values()):
            if not is_dry_run():
                log.warning("No bundle-adjustment reports under %s; keeping all images", prefix)
            return []
        with self.workspace() as ws:
            ws.set_rejection_params(cfg)
            ws.load_bundle_adjust(prefix, run=Path(prefix).parent.name)
            flags = ws.rejections(Path(prefix).parent.name)
        removed = [
            RemovedImage(im, stage, flags[im.cube.name]) for im in self.images if im.cube.name in flags
        ]
        return self._remove(removed)

    def _remove(self, removed: list[RemovedImage]) -> list[RemovedImage]:
        if not removed:
            return []
        gone = {id(r.image) for r in removed}
        self.images = [im for im in self.images if id(im) not in gone]
        self.removed += removed
        for r in removed:
            log.info("Removed %s (%s): %s", r.image.product_id, r.stage, r.reason)
        log.info("%d images removed at %s, %d remain", len(removed), removed[0].stage, len(self.images))
        _write_csv(
            self.work_dir / "selection" / "removed.csv",
            ["product_id", "stage", "reason"],
            [[r.image.product_id, r.stage, r.reason] for r in self.removed],
        )
        return removed

    def register(self) -> Path:
        """Register the adjusted cameras to the prior DEM (translation only); returns their prefix."""
        mode = self.config.registration
        if mode == "none":
            return self.ba_prefix
        transform = Path(f"{self.align_prefix}-transform.txt")
        if not self._skip(transform):
            if mode != "stereo" or self._stereo_mosaic() is None:
                self._align_residuals()
            else:
                self._align_stereo()
        marker = Path(f"{self.ba_align_prefix}-applied-transform.txt")
        if not self._skip(marker):
            asp.apply_transform(
                [im.cube for im in self.images],
                transform,
                self.ba_align_prefix,
                self._cameras(self.images),
                input_adjustments_prefix=self.ba_prefix,
            )
            if not is_dry_run():
                marker.write_text(transform.read_text())
        return self.ba_align_prefix

    def _align_residuals(self) -> Path:
        return asp.pc_align(
            self.dem,
            Path(f"{self.ba_prefix}-final_residuals_pointmap.csv"),
            self.align_prefix,
            max_displacement=self.config.max_displacement,
            compute_translation_only=True,
            csv_format="1:lon 2:lat 3:height_above_datum",
            save_transformed_source_points=True,
        )

    def _align_stereo(self) -> Path:
        return asp.pc_align(
            self.dem,
            self.work_dir / "stereo" / "mosaic.tif",
            self.align_prefix,
            max_displacement=self.config.max_displacement,
            compute_translation_only=True,
        )

    def stereo_pairs(self) -> pd.DataFrame:
        """Up to ``stereo_pairs`` kept-image pairs ranked by :func:`~lunar_toolkit.stereo.find_stereo_pairs`."""
        from .stereo import find_stereo_pairs

        obs = self._observations()
        kept = {im.product_id.upper() for im in self.images}
        obs = obs[obs["product_id"].astype(str).str.upper().isin(kept)]
        pairs = find_stereo_pairs(obs, aoi=self.aoi, convergence=self.config.stereo_convergence)
        return pairs.head(self.config.stereo_pairs)

    def _stereo_mosaic(self) -> Path | None:
        """Stereo DEMs (triangulated with the ``ba/`` cameras) mosaicked; None without any."""
        cfg = self.config
        mosaic = self.work_dir / "stereo" / "mosaic.tif"
        if self._skip(mosaic):
            return mosaic
        try:
            pairs = self.stereo_pairs()
        except (OSError, KeyError, ValueError) as exc:
            log.warning("No observations for stereo registration (%s); using BA points", exc)
            return None
        by_id = {im.product_id.upper(): im for im in self.images}
        dems = []
        for _, pair in pairs.iterrows():
            a, b = by_id[str(pair["product_id_1"]).upper()], by_id[str(pair["product_id_2"]).upper()]
            try:
                dem = stereo_dem(
                    a.cube,
                    b.cube,
                    self.work_dir / "stereo" / f"{a.product_id}__{b.product_id}",
                    resolution=cfg.resolution,
                    prior_dem=self.dem,
                    align=False,
                    csm=cfg.csm,
                    cameras=(a.camera, b.camera) if cfg.csm and a.camera and b.camera else None,
                    mapprojected=(self._map_path("prj", a), self._map_path("prj", b)),
                    bundle_adjust_prefix=self.ba_prefix,
                    processes=cfg.processes,
                    threads=cfg.threads,
                    force=self.force,
                )
            except ToolError as exc:
                log.warning("Stereo %s / %s failed: %s", a.product_id, b.product_id, exc)
                continue
            if is_dry_run() or dem.exists():
                dems.append(dem)
        if not dems:
            log.warning("No stereo DEMs for registration; using BA points")
            return None
        return asp.dem_mosaic(dems, mosaic)

    def _observations(self) -> pd.DataFrame:
        if self.observations is None:
            from .lroc.cumindex import CumIndex

            ci = CumIndex("edr", data_dir=self.data_dir)
            self.observations = ci.to_observations(ci.lookup([im.product_id for im in self.all_images]))
        return self.observations

    def refine(self) -> Path:
        """Bundle-adjust the kept, registered cameras again with ``--heights-from-dem``.

        Reuses the ``ba/`` matches (``--match-files-prefix``); ``--mapproj-dem`` gives
        the match-offset report for a second :meth:`filter_images`.
        """
        done = Path(f"{self.refine_prefix}-final_residuals_stats.txt")
        if self._skip(done):
            return self.refine_prefix
        start = self.ba_align_prefix if self.config.registration != "none" else self.ba_prefix
        opts = self._ba_options(self.config.options("refine"))
        return asp.bundle_adjust(
            [im.cube for im in self.images],
            self.refine_prefix,
            self._cameras(self.images),
            match_files_prefix=self.ba_prefix,
            input_adjustments_prefix=start,
            heights_from_dem=self.dem,
            mapproj_dem=self.dem,
            threads=self.config.threads,
            **opts,
        )

    def mapproject_for_sfs(self) -> list[Path]:
        """Mapproject the kept images at the target resolution with the final cameras (``prj_sfs/``)."""
        outs = self.mapproject(
            self.images, "prj_sfs", self.config.resolution, bundle_adjust_prefix=self.camera_prefix
        )
        for im, out in zip(self.images, outs):
            im.mapprojected = out
        return outs

    def qa_mosaics(self, bin_width: float = 30.0) -> dict[str, Path]:
        """Max-lit mosaics of all kept images and per Sun-azimuth bin (``qa/``).

        Ghosting in these shows misregistered images (Bertone et al. 2026, App. A).
        Bins are named by their start, e.g. ``max_lit_az030.tif`` for 30–60°.
        """
        qa = self.work_dir / "qa"
        maps = [im.mapprojected or self._map_path("prj_sfs", im) for im in self.images]
        groups: dict[str, list[Path]] = {"max_lit": maps}
        for im, m in zip(self.images, maps):
            if im.sun_azimuth is not None:
                start = int((im.sun_azimuth % 360.0) // bin_width * bin_width)
                groups.setdefault(f"max_lit_az{start:03d}", []).append(m)
        out = {}
        for name, paths in groups.items():
            out[name] = qa / f"{name}.tif"
            if paths and not self._skip(out[name]):
                asp.dem_mosaic(paths, out[name], max=True)
        return out

    def select_images(self) -> list[SfsImage]:
        """Split the kept images into SfS inputs and the rest (``selection/``).

        Images at least ``max_shadow_fraction`` shadowed are removed first.  Writes
        ``selection.csv`` (every kept image: azimuth, shadow fraction, mapprojected path,
        selected) plus ``selected.txt`` / ``unselected.txt`` (one product ID per line).
        """
        cfg = self.config
        out_dir = self.work_dir / "selection"
        lit = self._lit_masks()
        if lit is not None and cfg.max_shadow_fraction is not None:
            shadowed = [
                RemovedImage(im, "selection", f"{im.shadow_fraction:.1%} shadowed")
                for im in self.images
                if im.shadow_fraction is not None and im.shadow_fraction >= cfg.max_shadow_fraction
            ]
            gone = {id(r.image) for r in shadowed}
            lit = lit[[k for k, im in enumerate(self.images) if id(im) not in gone]]
            self._remove(shadowed)
        eligible = self._sfs_eligible()
        if cfg.selection == "all" or lit is None:
            chosen = eligible
        elif cfg.selection == "coverage":
            az = np.array([np.nan if im.sun_azimuth is None else im.sun_azimuth for im in self.images])
            picked, views = select_by_illumination(
                lit[eligible],
                az[eligible],
                views=cfg.selection_views,
                azimuth_sep=cfg.selection_azimuth_sep,
                max_images=cfg.max_sfs_images,
            )
            chosen = [eligible[k] for k in picked]
            covered = lit[eligible].any(axis=0)
            if covered.any():
                log.info(
                    "Selected %d / %d nadir images; lit cells with >=%d directions: %.1f%%",
                    len(chosen),
                    len(eligible),
                    cfg.selection_views,
                    100 * np.mean(views[covered] >= cfg.selection_views),
                )
        elif cfg.selection == "kmedoids":
            az = np.array([np.nan if im.sun_azimuth is None else im.sun_azimuth for im in self.images])
            res = select_kmedoids(
                lit[eligible],
                az[eligible],
                cfg.selection_clusters,
                views=cfg.selection_views,
                azimuth_sep=cfg.selection_azimuth_sep,
                max_images=cfg.max_sfs_images,
                seed=cfg.seed,
            )
            chosen = sorted(eligible[k] for k in res.chosen)
            log.info(
                "k-medoids selected %d / %d nadir images (%d medoids); uncovered %.2f%%, poorly lit %.2f%%",
                len(chosen),
                len(eligible),
                len(res.medoids),
                100 * res.uncovered,
                100 * res.poorly_lit,
            )
        else:
            chosen = sorted(set(self._image_subset()) & set(eligible))
        chosen_set = set(chosen)
        self.selected = [im for k, im in enumerate(self.images) if k in chosen_set]
        self.unselected = [im for k, im in enumerate(self.images) if k not in chosen_set]
        _write_csv(
            out_dir / "selection.csv",
            ["product_id", "sun_azimuth", "sun_elevation", "shadow_fraction", "mapprojected", "selected"],
            [
                [
                    im.product_id,
                    im.sun_azimuth,
                    im.sun_elevation,
                    im.shadow_fraction,
                    im.mapprojected,
                    k in chosen_set,
                ]
                for k, im in enumerate(self.images)
            ],
        )
        for name, ims in (("selected", self.selected), ("unselected", self.unselected)):
            _write_lines(out_dir / f"{name}.txt", [im.product_id for im in ims])
        return self.selected

    def _sfs_eligible(self) -> list[int]:
        """Indices of the kept images allowed into SfS (``|slew| < max_sfs_slew``)."""
        limit = self.config.max_sfs_slew
        if limit is None:
            return list(range(len(self.images)))
        if any(im.slew_angle is None for im in self.images):
            try:
                obs = self._observations()
            except (OSError, KeyError, ValueError) as exc:
                log.warning("No slew angles (%s); all images count as nadir", exc)
                obs = None
            if obs is not None and "slew_angle" in obs:
                slews = dict(zip(obs["product_id"].astype(str).str.upper(), obs["slew_angle"]))
                for im in self.images:
                    v = slews.get(im.product_id.upper())
                    if im.slew_angle is None and v is not None and np.isfinite(v):
                        im.slew_angle = float(v)
        eligible = [
            k for k, im in enumerate(self.images) if im.slew_angle is None or abs(im.slew_angle) < limit
        ]
        if len(eligible) < len(self.images):
            log.info(
                "%d slewed images (|slew| >= %g deg) kept out of SfS", len(self.images) - len(eligible), limit
            )
        return eligible

    def _lit_masks(self) -> np.ndarray | None:
        """``(images, cells)`` lit mask on a common coarse grid; fills ``shadow_fraction``."""
        bounds = asp.raster_bounds(self.dem)
        if bounds is None or not self.images:
            return None
        ulx, uly, lrx, lry = bounds
        n = self.config.selection_grid
        w, h = abs(lrx - ulx), abs(uly - lry)
        size = (n, max(1, round(n * h / w))) if w >= h else (max(1, round(n * w / h)), n)
        thr = self.shadow_threshold
        masks = []
        for im in self.images:
            r = asp.read_raster(
                im.mapprojected or self._map_path("prj_sfs", im), projwin=bounds, outsize=size
            )
            v = r.data.ravel() if r is not None else np.full(size[0] * size[1], np.nan)
            valid = np.isfinite(v) & (v > -1e30)
            lit = valid & (v > thr)
            im.shadow_fraction = float(1.0 - lit.sum() / valid.sum()) if valid.any() else 1.0
            masks.append(lit)
        return np.array(masks)

    def _image_subset(self, bin_width: float = 30.0) -> list[int]:
        """ASP ``image_subset`` in each Sun-azimuth bin; returns indices of the union."""
        bounds = asp.raster_bounds(self.dem)
        win = (bounds[0], bounds[3], bounds[2], bounds[1]) if bounds else None
        bins: dict[int, list[int]] = {}
        for k, im in enumerate(self.images):
            bins.setdefault(int(((im.sun_azimuth or 0.0) % 360.0) // bin_width), []).append(k)
        by_path = {str(im.mapprojected): k for k, im in enumerate(self.images)}
        chosen: list[int] = []
        for b, idx in sorted(bins.items()):
            subset = asp.image_subset(
                [self.images[k].mapprojected or self._map_path("prj_sfs", self.images[k]) for k in idx],
                self.work_dir / "selection" / f"image_subset_az{int(b * bin_width):03d}.txt",
                threshold=self.shadow_threshold,
                t_projwin=win,
                threads=self.config.threads,
            )
            chosen += [by_path[p] for p, _ in subset if p in by_path]
        return sorted(set(chosen))

    def reduce_factor(self, im: SfsImage) -> int:
        """ISIS ``reduce`` factor for *im* (1 = keep native resolution)."""
        cfg = self.config
        if cfg.reduce is None:
            return 1
        if cfg.reduce == "auto":
            native = im.pixel_scale or cfg.native_resolution
            return max(1, int(round(cfg.resolution / native)))
        return max(1, int(cfg.reduce))

    def sfs_inputs(self, images: Sequence[SfsImage]) -> tuple[list[Path], list[Path], Path | None, bool]:
        """Cubes, cameras and adjustment prefix for SfS on *images*.

        Cubes needing it are ``reduce``-d to the target resolution (Bertone et al. 2026,
        App. A: SfS's own resampling leaves artifacts).  With ``sfs_cameras="csm"`` each
        CSM camera is re-mapped to its reduced cube (:func:`csm.isis_reduce_with_csm`);
        otherwise ISIS cameras are used.  The final ``.adjust`` files (resolution
        independent) are copied under :attr:`sfs_camera_prefix` with the new names;
        published cameras need none.  Returns ``(cubes, cameras, prefix, isis_cameras)``.
        """
        from . import csm

        cfg = self.config
        factors = [self.reduce_factor(im) for im in images]
        prefix = self.camera_prefix
        if cfg.csm and all(f == 1 for f in factors):
            return [im.cube for im in images], self._cameras(images), prefix, False
        use_csm = cfg.csm and cfg.sfs_cameras == "csm"
        cubes: list[Path] = []
        cameras: list[Path] = []
        for im, f in zip(images, factors):
            cube, camera = im.cube, im.camera if use_csm else None
            if f > 1 and use_csm and im.camera is not None:
                cube, camera = csm.isis_reduce_with_csm(
                    im.cube, im.camera, f, self.work_dir / "images", force=self.force
                )
            elif f > 1:
                cube = self.work_dir / "images" / f"{im.product_id}.red{f}.cub"
                if not self._skip(cube):
                    isis.reduce(im.cube, cube, f)
            if prefix is not None:
                src = asp.bundle_adjust_file_name(prefix, im.cube, im.camera if cfg.csm else None)
                dst = asp.bundle_adjust_file_name(self.sfs_camera_prefix, cube, camera)
                if not is_dry_run() and not self._skip(dst):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            cubes.append(cube)
            if camera is not None:
                cameras.append(camera)
        out_prefix = self.sfs_camera_prefix if prefix is not None else None
        return cubes, cameras, out_prefix, not use_csm

    def use_published_cameras(self) -> list[SfsImage]:
        """Replace the cameras with the published Bertone et al. (2026) adjusted ones.

        Each published camera (a CSM state of the reduced full image) is re-mapped to
        the image of its cube (:func:`reference.published_camera_native`) and written to
        ``images/<id>.published.json``; images without one are removed.
        """
        from . import csm
        from . import reference as ref

        region = self.config.published_region
        if region is None:
            rid = self.config.region_id or ""
            if not (rid.startswith("A3") and rid[2:].isdigit()):
                raise ValueError("cameras_from: published needs published_region (or region_id A3NN)")
            region = int(rid[2:])
        out_dir = self.work_dir / "images"
        if is_dry_run():
            for im in self.images:
                im.camera = out_dir / f"{im.product_id}.published.json"
            return self.images
        published = ref.published_cameras(
            region, self.work_dir / "published", images=[im.product_id for im in self.images]
        )
        missing = [
            RemovedImage(im, "published", "no published camera")
            for im in self.images
            if im.product_id not in published
        ]
        self._remove(missing)
        for im in self.images:
            camera = out_dir / f"{im.product_id}.published.json"
            if not self._skip(camera):
                geom = csm.CubeGeometry.from_cube(im.cube)
                native = (geom.alpha_samples, geom.alpha_lines)
                csm.write_camera(ref.published_camera_native(published[im.product_id], native), camera)
            im.camera = camera
        return self.images

    def shape_from_shading(
        self,
        images: Sequence[SfsImage] | None = None,
        prefix: Path | None = None,
        dem: Path | None = None,
        **options: Any,
    ) -> Path:
        """``parallel_sfs`` on the target-grid prior DEM; returns ``<prefix>-DEM-final.tif``.

        *images* default to :attr:`selected` (else all kept images), *prefix* to
        ``sfs/run`` and *dem* (input DEM, e.g. a crop) to :attr:`dem`.  *options*
        override the configured SfS options.
        """
        images = list(images) if images is not None else (self.selected or self.images)
        prefix = Path(prefix) if prefix else self.sfs_prefix
        out = Path(f"{prefix}-DEM-final.tif")
        if self._skip(out):
            return out
        cubes, cameras, ba_prefix, isis_cameras = self.sfs_inputs(images)
        opts: dict[str, Any] = {**self.config.options("sfs"), **self._parallel_opts()}
        if isis_cameras:
            opts["use_approx_camera_models"] = True
        opts.update(options)
        return asp.sfs(
            dem or self.dem, cubes, prefix, cameras, parallel=True, bundle_adjust_prefix=ba_prefix, **opts
        )

    def products(self, dem: Path) -> dict[str, Path]:
        """Standard products via :func:`lunar_toolkit.sfs_products.make_products` when available.

        Otherwise (module missing or its dependencies not installed) a hillshade of the
        SfS DEM and its difference from the prior.
        """
        try:
            products = importlib.import_module(f"{__package__}.sfs_products")
        except ImportError as exc:
            log.info("sfs_products unavailable (%s); writing hillshade and diff only", exc)
        else:
            height_error = Path(f"{self.sfs_prefix}-height-error.tif")
            return products.make_products(
                dem,
                self.dem,
                self.work_dir / "products",
                [
                    (im.mapprojected, im.sun_azimuth, im.pixel_scale)
                    for im in self.selected
                    if im.mapprojected and im.sun_azimuth is not None
                ],
                region_id=self.config.region_id or self.work_dir.name,
                resolution=self.config.resolution,
                height_error=height_error if height_error.exists() else None,
                shadow_threshold=self.shadow_threshold,
                force=self.force,
            )
        out = {"dem": dem}
        shade = Path(f"{self.sfs_prefix}-DEM-final_hillshade.tif")
        out["hillshade"] = shade if self._skip(shade) else asp.hillshade(dem, shade)
        diff_prefix = Path(f"{self.sfs_prefix}-prior")
        diff = Path(f"{diff_prefix}-diff.tif")
        out["diff"] = diff if self._skip(diff) else asp.geodiff(dem, self.dem, diff_prefix)
        return out

    def ensemble(self) -> dict[str, Path]:
        """SfS ensemble over a central crop and its spread (``ensemble/``, ``products/``).

        Members are :func:`~lunar_toolkit.sfs_products.ensemble_subsets` of the kept
        images; returns :func:`~lunar_toolkit.sfs_products.ensemble_uncertainty`'s maps.
        """
        from . import sfs_products

        cfg = self.config
        bounds = asp.raster_bounds(self.dem) or cfg.projwin
        if bounds is None:
            log.warning("Ensemble skipped: DEM extent unknown (dry run without projwin)")
            return {}
        ens = self.work_dir / "ensemble"
        crop = ens / "prior.tif"
        if not self._skip(crop):
            centre = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
            asp.gdal_translate(self.dem, crop, projwin=sfs_products.crop_window(*centre, cfg.ensemble_crop))
        size = min(cfg.ensemble_size or len(self.selected) or len(self.images), len(self.images))
        subsets = sfs_products.ensemble_subsets(self.images, cfg.ensemble_members, size, seed=cfg.seed)
        members = [
            self.shape_from_shading(subset, ens / f"m{k:02d}" / "run", dem=crop)
            for k, subset in enumerate(subsets)
        ]
        _write_lines(ens / "members.txt", [" ".join(im.product_id for im in subset) for subset in subsets])
        if is_dry_run():
            return {}
        return sfs_products.ensemble_uncertainty(
            members,
            self.work_dir / "products",
            region_id=cfg.region_id or self.work_dir.name,
            resolution=cfg.resolution,
            reference=crop,
            force=self.force,
        )


def flag_images(
    residual_stats: dict[str, asp.ResidualStats],
    residual_norms: dict[str, np.ndarray],
    offsets: dict[str, asp.MatchOffsetStats],
    *,
    max_offset_p85: float | None = 1.8,
    max_large_residual_fraction: float | None = 0.85,
    large_residual_px: float = 1.0,
    min_matches: int = 1,
) -> dict[str, str]:
    """Images to drop after bundle adjustment (Bertone et al. 2026, §2), keyed by file name.

    An image is flagged when it has fewer than *min_matches* residuals or match offsets,
    an 85th-percentile mapprojected match offset over *max_offset_p85* m, or more than
    *max_large_residual_fraction* of its reprojection errors over *large_residual_px*.
    The value is the reason.  The rules are the workspace's ``image_rejections`` view
    (:mod:`lunar_toolkit.workspace`), evaluated in an in-memory database.
    """
    from .workspace import flag_images_sql

    return flag_images_sql(
        residual_stats,
        residual_norms,
        offsets,
        max_offset_p85=max_offset_p85,
        max_large_residual_fraction=max_large_residual_fraction,
        large_residual_px=large_residual_px,
        min_matches=min_matches,
    )


def select_by_illumination(
    lit: np.ndarray,
    azimuth: np.ndarray,
    *,
    views: int = 3,
    azimuth_sep: float = 30.0,
    max_images: int | None = None,
    min_gain: int = 1,
) -> tuple[list[int], np.ndarray]:
    """Greedy per-pixel illumination-diverse image selection on a raster grid.

    The raster analogue of :func:`lunar_toolkit.selection.select_for_coverage`, after
    Bertone et al. (2026, §2): a lit cell is *helped* by an image lighting it if it has
    fewer than *views* illumination directions so far and the image's Sun azimuth is at
    least *azimuth_sep* from each of them.  Each step takes the image helping most
    cells, until none helps *min_gain* cells or *max_images* are chosen.

    Parameters
    ----------
    lit : (n_images, n_cells) bool array
    azimuth : (n_images,) array
        Sun azimuths (deg); NaN never helps.

    Returns
    -------
    (chosen, count)
        Image indices in pick order, and the directions reached per cell.
    """
    lit = np.asarray(lit, bool)
    az = np.asarray(azimuth, float)
    n, cells = lit.shape
    dirs = np.full((cells, views), np.nan)
    count = np.zeros(cells, int)
    available = np.isfinite(az)
    chosen: list[int] = []
    while available.any() and (max_images is None or len(chosen) < max_images):
        open_cells = count < views
        best, best_cells = -1, np.empty(0, int)
        for k in np.flatnonzero(available):
            idx = np.flatnonzero(lit[k] & open_cells)
            if len(idx) <= len(best_cells):
                continue
            d = azimuth_difference(dirs[idx], az[k])
            idx = idx[np.all(np.isnan(d) | (d >= azimuth_sep), axis=1)]
            if len(idx) > len(best_cells):
                best, best_cells = int(k), idx
        if best < 0 or len(best_cells) < min_gain:
            break
        dirs[best_cells, count[best_cells]] = az[best]
        count[best_cells] += 1
        available[best] = False
        chosen.append(best)
    return chosen, count


def prefilter_candidates(
    df: pd.DataFrame, max_slew: float | None = None, max_resolution: float | None = 1.5
) -> pd.DataFrame:
    """Images finer than *max_resolution* m/px (and, with *max_slew*, ``|slew| <`` it).

    Works on ``CumIndex`` rows (``SLEW_ANGLE``, ``RESOLUTION``) or observation rows
    (``slew_angle``, ``pixel_scale``).  Rows with unknown values are dropped.  The
    nadir cut of Bertone et al. (2026, §2) belongs to SfS selection
    (:attr:`SfsConfig.max_sfs_slew`), so searches keep slewed images by default: they
    give the stereo convergence the registration step needs.
    """
    import pandas as pd

    keep = pd.Series(True, index=df.index)
    slew_col = next((c for c in ("SLEW_ANGLE", "slew_angle") if c in df), None)
    if max_slew is not None:
        if slew_col:
            keep &= pd.to_numeric(df[slew_col], errors="coerce").abs() < max_slew
        else:
            log.warning("No slew column; slew filter skipped")
    res_col = next((c for c in ("RESOLUTION", "pixel_scale", "PIXEL_SCALE") if c in df), None)
    if max_resolution is not None and res_col:
        keep &= pd.to_numeric(df[res_col], errors="coerce") < max_resolution
    return df[keep]


def cap_candidates(
    obs: pd.DataFrame,
    aoi: Any,
    max_images: int,
    *,
    max_sfs_slew: float | None = 5.0,
    stereo_pairs: int = 30,
    stereo_convergence: tuple[float, float] = (5.0, 60.0),
    views: int = 3,
    azimuth_sep: float = 30.0,
) -> pd.DataFrame:
    """At most *max_images* candidate images, chosen before anything is downloaded.

    Keeps both images of the best stereo pairs (up to half the budget; they feed the
    registration step), then nadir images (``|slew| < max_sfs_slew``) chosen for
    illumination coverage over *aoi* (:func:`~lunar_toolkit.selection.select_for_coverage`:
    *views* directions at least *azimuth_sep* apart per cell), then the finest remaining
    nadir images.  Returns the chosen rows in their original order.
    """
    from .selection import select_for_coverage

    if len(obs) <= max_images:
        return obs
    ids = obs["product_id"].astype(str).str.upper()
    chosen: list[str] = []

    def take(new: Sequence[str], limit: int) -> None:
        for pid in new:
            if len(chosen) >= limit:
                return
            if pid not in chosen:
                chosen.append(pid)

    if stereo_pairs:
        pairs = _candidate_pairs(obs, aoi, stereo_convergence)
        stereo_ids = [
            str(v).upper()
            for pair in pairs.head(stereo_pairs).itertuples()
            for v in (pair.product_id_1, pair.product_id_2)
        ]
        take(stereo_ids, max_images // 2)
    slew = _numeric(obs.get("slew_angle"))
    nadir = obs[(slew.abs() < max_sfs_slew) | slew.isna()] if max_sfs_slew is not None else obs
    nadir = nadir[~nadir["product_id"].astype(str).str.upper().isin(chosen)]
    if len(nadir):
        cover = select_for_coverage(
            nadir, aoi, azimuth_sep_deg=azimuth_sep, views_per_cell=views, max_images=max_images - len(chosen)
        )
        take(cover["product_id"].astype(str).str.upper().tolist(), max_images)
        rest = nadir.assign(_res=_numeric(nadir.get("pixel_scale"))).sort_values("_res")
        take(rest["product_id"].astype(str).str.upper().tolist(), max_images)
    log.info("Capped %d candidates to %d (stereo pairs + illumination coverage)", len(obs), len(chosen))
    return obs[ids.isin(set(chosen)).to_numpy()]


def _candidate_pairs(obs: pd.DataFrame, aoi: Any, convergence: tuple[float, float]) -> pd.DataFrame:
    from .stereo import find_stereo_pairs
    from .workspace import PAIRS_ENGINE_THRESHOLD, find_stereo_pairs_duckdb, spatial_available

    if len(obs) >= PAIRS_ENGINE_THRESHOLD and spatial_available():
        return find_stereo_pairs_duckdb(obs, aoi, convergence=convergence)
    return find_stereo_pairs(obs, aoi=aoi, convergence=convergence)


def _numeric(values: Any) -> pd.Series:
    import pandas as pd

    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce")


def search_images(
    aoi: Any,
    *,
    max_slew: float | None = None,
    max_resolution: float | None = 1.5,
    max_images: int | None = None,
    config: SfsConfig | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """NAC EDRs over *aoi* from the local ``CumIndex``, pre-filtered and capped; observation schema.

    *config* supplies the SfS slew cut and stereo settings used by :func:`cap_candidates`.
    """
    from .lroc.cumindex import CumIndex

    cfg = config or SfsConfig()
    ci = CumIndex("edr", data_dir=data_dir)
    df = ci.search(aoi=aoi, nac_only=True, wkt=True)
    kept = prefilter_candidates(df, max_slew, max_resolution)
    log.info("%d NAC images over the AOI, %d after the resolution cut", len(df), len(kept))
    obs = ci.to_observations(kept)
    if max_images is not None:
        obs = cap_candidates(
            obs,
            aoi,
            max_images,
            max_sfs_slew=cfg.max_sfs_slew,
            stereo_pairs=cfg.stereo_pairs if cfg.registration == "stereo" else 0,
            stereo_convergence=cfg.stereo_convergence,
            views=cfg.selection_views,
            azimuth_sep=cfg.selection_azimuth_sep,
        )
    return obs


@dataclass(frozen=True)
class PolarStereographic:
    """A polar stereographic projection (sphere, true scale at the pole) from a PROJ string."""

    hemisphere: str
    lon_0: float = 0.0
    radius: float = MOON_RADIUS_KM * 1000.0
    x_0: float = 0.0
    y_0: float = 0.0

    @classmethod
    def from_proj4(cls, proj4: str) -> PolarStereographic:
        """Parse e.g. ``+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m``.

        Raises
        ------
        ValueError
            For anything else (other projections, oblique stereographic, scale ≠ 1, non-metre units).
        """
        p: dict[str, str] = dict(re.findall(r"\+(\w+)(?:=(\S+))?", proj4))
        lat_0 = float(p.get("lat_0", "nan"))
        k = float(p.get("k", p.get("k_0", "1")))
        lat_ts = float(p["lat_ts"]) if "lat_ts" in p else lat_0
        sphere_a = p.get("a") if p.get("a") == p.get("b", p.get("a")) else None
        radius = float(p.get("R") or sphere_a or "nan")
        if (
            p.get("proj") not in ("stere", "ups")
            or not math.isclose(abs(lat_0), 90.0)
            or not math.isclose(k, 1.0)
            or not math.isclose(lat_ts, lat_0)
            or p.get("units", "m") != "m"
            or not math.isfinite(radius)
        ):
            raise ValueError(
                f"Expected a spherical polar stereographic projection in metres, got {proj4!r}; "
                "pass --projwin in the DEM's own units instead"
            )
        return cls(
            "S" if lat_0 < 0 else "N",
            float(p.get("lon_0", 0.0)),
            radius,
            float(p.get("x_0", 0.0)),
            float(p.get("y_0", 0.0)),
        )

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        x, y = ps_xy(lat, lon - self.lon_0, self.hemisphere)
        scale = self.radius / (MOON_RADIUS_KM * 1000.0)
        return x * scale + self.x_0, y * scale + self.y_0


def aoi_projwin(aoi: Any, dem: Path, snap: float | None = None) -> tuple[float, float, float, float]:
    """``(ulx, uly, lrx, lry)`` bounding *aoi* in the polar stereographic projection of *dem*.

    *aoi* is anything :func:`lunar_toolkit.geometry.load_aoi` takes (GeoJSON / WKT file,
    WKT string).  The window is widened to multiples of *snap* (m) when given.
    """
    proj = PolarStereographic.from_proj4(asp.srs_proj4(dem))
    pts = [proj.xy(lat, lon) for poly in load_aoi(aoi).polygons for lon, lat in poly[0]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if snap:
        x0, y0 = math.floor(x0 / snap) * snap, math.floor(y0 / snap) * snap
        x1, y1 = math.ceil(x1 / snap) * snap, math.ceil(y1 / snap) * snap
    return x0, y1, x1, y0


def _by_id(observations: pd.DataFrame | None, column: str) -> dict[str, float]:
    """Finite values of a numeric observation *column* keyed by upper-case product ID."""
    if observations is None or column not in observations:
        return {}
    import pandas as pd

    values = pd.to_numeric(observations[column], errors="coerce")
    ids = observations["product_id"].astype(str).str.upper()
    return {i: float(v) for i, v in zip(ids, values) if np.isfinite(v)}


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    if is_dry_run():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if v is None else v for v in r] for r in rows])


def _write_lines(path: Path, lines: list[str]) -> None:
    if not is_dry_run():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{ln}\n" for ln in lines))
