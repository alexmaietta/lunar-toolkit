"""CSM camera files: parsing, reduce / crop re-mapping (offline) and cam_test against ISIS (pytest -m tools)."""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES

from lunar_toolkit import csm
from lunar_toolkit.tools import dry_run, run

EXAMPLE = Path(os.environ.get("LUNAR_ASP_EXAMPLE", "LRONAC_example"))


def _state(**kw):
    d = {
        "m_modelName": csm.LINESCAN,
        "m_nLines": 52224,
        "m_nSamples": 5064,
        "m_intTimeLines": [0.5, 20000.5],
        "m_intTimeStartTimes": [-16.4651, -3.8540],
        "m_intTimes": [0.00063, 0.00064],
        "m_detectorSampleSumming": 1.0,
        "m_detectorLineSumming": 1.0,
        "m_startingDetectorSample": 0.0,
        "m_startingDetectorLine": 0.0,
        "m_detectorSampleOrigin": 2547.5,
        "m_detectorLineOrigin": 0.0,
        "m_iTransS": [0.0, 0.0, 142.857],
        "m_iTransL": [0.0, 142.857, 0.0],
        "m_gsd": 1.02,
        "m_halfSwath": 2588.0,
        "m_halfTime": 16.46,
    }
    d.update(kw)
    return csm.CsmCamera(d, "state", csm.LINESCAN)


def _isd(**kw):
    d = {
        "name_model": csm.LINESCAN,
        "image_lines": 52224,
        "image_samples": 5064,
        "line_scan_rate": [[0.5, -16.4651, 0.00063]],
        "detector_sample_summing": 1,
        "detector_line_summing": 1,
        "starting_detector_sample": 0,
        "starting_detector_line": 0,
        "detector_center": {"line": 0, "sample": 2547.5},
        "focal2pixel_lines": [0, 142.857, 0],
        "focal2pixel_samples": [0, 0, 142.857],
    }
    d.update(kw)
    return csm.CsmCamera(d, "isd")


def _forward(cam, line, sample):
    """usgscsm linescan image -> (exposure time, focal-plane x, y), as in getImageTime / losToEcf."""
    d = cam.data
    if cam.kind == "isd":  # usgscsm reads the ISD summing as an integer
        rates = np.array(d["line_scan_rate"], float)
        lines, starts, times = rates[:, 0], rates[:, 1], rates[:, 2]
        ss, sl = int(d["detector_sample_summing"]), int(d["detector_line_summing"])
        s0, l0 = d["starting_detector_sample"], d["starting_detector_line"]
        so, lo = d["detector_center"]["sample"], d["detector_center"]["line"]
        its, itl = d["focal2pixel_samples"], d["focal2pixel_lines"]
    else:
        lines, starts, times = (
            np.array(d[k], float) for k in ("m_intTimeLines", "m_intTimeStartTimes", "m_intTimes")
        )
        ss, sl = d["m_detectorSampleSumming"], d["m_detectorLineSumming"]
        s0, l0 = d["m_startingDetectorSample"], d["m_startingDetectorLine"]
        so, lo = d["m_detectorSampleOrigin"], d["m_detectorLineOrigin"]
        its, itl = d["m_iTransS"], d["m_iTransL"]
    k = max(int(np.searchsorted(lines, line, side="right")) - 1, 0)
    t = starts[k] + times[k] * (line - lines[k] + 0.5)
    t1 = 0.0 * sl + l0 - lo - itl[0]
    t2 = sample * ss + s0 - so - its[0]
    x, y = np.linalg.solve([[itl[1], itl[2]], [its[1], its[2]]], [t1, t2])
    return t, x, y


@pytest.mark.parametrize("make", [_state, _isd])
@pytest.mark.parametrize(
    "scale, offset",
    [
        ((4.0, 4.0), (0.0, 0.0)),
        ((5.0, 5.0), (0.0, 0.0)),
        ((1.0, 1.0), (300.0, 1200.0)),
        ((4.0013, 3.9984), (300.0, 1200.0)),
    ],
)
def test_transform_preserves_geometry(make, scale, offset):
    cam = make()
    new = csm.transform_camera(cam, scale[0], offset[0], scale[1], offset[1], (750, 626))
    assert new.size == (750, 626) and new.kind == cam.kind
    for y, x in [(0.5, 0.5), (100.25, 333.7), (620.0, 749.5)]:
        assert _forward(new, y, x) == pytest.approx(
            _forward(cam, scale[1] * y + offset[1], scale[0] * x + offset[0]), rel=1e-12, abs=1e-12
        )
    assert cam.data == make().data  # input untouched


def test_reduce_camera_state_and_isd(tmp_path):
    red = csm.reduce_camera(_state(), 4, tmp_path / "s.json")
    d = red.data
    assert (d["m_nSamples"], d["m_nLines"]) == (1266, 13056)
    assert d["m_detectorSampleSumming"] == d["m_detectorLineSumming"] == 4.0
    assert d["m_intTimes"] == pytest.approx([0.00252, 0.00256])
    assert d["m_intTimeLines"] == pytest.approx([0.5, 5000.5])
    assert d["m_gsd"] == pytest.approx(4.08) and d["m_halfSwath"] == pytest.approx(4.08 * 1266 / 2)
    back = csm.read_camera(tmp_path / "s.json")
    assert back.header == csm.LINESCAN and back.data == d
    isd = csm.reduce_camera(_isd(), 5, tmp_path / "i.json", size=(1000, 1000))
    assert isd.data["detector_sample_summing"] == 5 and isd.data["image_lines"] == 1000
    assert isd.data["line_scan_rate"] == [[0.5, -16.4651, pytest.approx(0.00315)]]
    assert json.loads((tmp_path / "i.json").read_text())["image_samples"] == 1000


def test_fractional_isd_summing_goes_into_focal2pixel():
    new = csm.transform_camera(_isd(), 4.0013, 0.0, 4.0, 0.0, (10, 10))
    d = new.data
    assert d["detector_sample_summing"] == 4 and isinstance(d["detector_sample_summing"], int)
    k = 4.0013 / 4
    assert d["focal2pixel_samples"][2] == pytest.approx(142.857 / k)
    assert d["detector_center"]["sample"] == pytest.approx(2547.5 / k)


def test_parse_errors_and_formats():
    with pytest.raises(ValueError, match="Not a CSM"):
        csm.parse_camera('{"a": 1}')
    bare = csm.parse_camera(json.dumps(_state().data))
    assert bare.header is None and csm.format_camera(bare).startswith("{")
    frame = _state(m_modelName="USGS_ASTRO_FRAME_SENSOR_MODEL")
    with pytest.raises(ValueError, match="Only"):
        csm.reduce_camera(frame, 2)
    with pytest.raises(ValueError, match="positive"):
        csm.transform_camera(_state(), 0, 0, 1, 0, (1, 1))


def test_published_camera_fixture_parses():
    cam = csm.read_camera(FIXTURES / "A311_GLDACAM_005_M176388393LE.json")
    assert cam.kind == "state" and cam.header == csm.LINESCAN and cam.model == csm.LINESCAN
    assert cam.size == (158, 960)


LABEL = """Object = IsisCube
  Object = Core
    Group = Dimensions
      Samples = {ns}
      Lines   = {nl}
      Bands   = 1
    End_Group
  End_Object
{alpha}End_Object
End
"""

ALPHA = """  Group = AlphaCube
    AlphaSamples        = 5064
    AlphaLines          = 52224
    AlphaStartingSample = 300.5
    AlphaStartingLine   = 1200.5
    AlphaEndingSample   = 3301.5
    AlphaEndingLine     = 3703.5
    BetaSamples         = {bs}
    BetaLines           = {bl}
  End_Group
"""


def test_cube_geometry_from_label(tmp_path):
    plain = csm.CubeGeometry.from_label(LABEL.format(ns=5000, nl=5000, alpha=""))
    assert plain == csm.CubeGeometry(5000, 5000, 5000, 5000)
    red = csm.CubeGeometry.from_label(LABEL.format(ns=750, nl=626, alpha=ALPHA.format(bs=750, bl=626)))
    assert (red.sample_scale, red.sample_offset) == pytest.approx((3001 / 750, 300.0))
    assert (red.line_scale, red.line_offset) == pytest.approx((2503 / 626, 1200.0))
    cube = tmp_path / "x.cub"
    cube.write_bytes(LABEL.format(ns=750, nl=626, alpha=ALPHA.format(bs=750, bl=626)).encode() + b"\0" * 64)
    assert csm.CubeGeometry.from_cube(cube) == red
    with pytest.raises(ValueError):
        csm.CubeGeometry.from_label("Object = IsisCube\nEnd\n")


def test_camera_for_cube_alpha_and_source_frames():
    red = csm.CubeGeometry.from_label(LABEL.format(ns=750, nl=626, alpha=ALPHA.format(bs=750, bl=626)))
    sub = csm.CubeGeometry.from_label(LABEL.format(ns=3001, nl=2503, alpha=ALPHA.format(bs=3001, bl=2503)))
    full = _state()  # describes the alpha image
    a = csm.camera_for_cube(full, red, sub)
    sub_cam = csm.camera_for_cube(full, sub)  # a camera written for the crop itself
    b = csm.camera_for_cube(sub_cam, red, sub)
    for y, x in [(0.5, 0.5), (300.0, 700.0)]:
        assert _forward(a, y, x) == pytest.approx(_forward(b, y, x), abs=1e-9)
        assert _forward(a, y, x) == pytest.approx(
            _forward(full, red.line_scale * y + 1200, red.sample_scale * x + 300), abs=1e-9
        )
    with pytest.raises(ValueError, match="matches neither"):
        csm.camera_for_cube(_state(m_nLines=7), red, sub)


def test_isis_reduce_with_csm_dry_run(tmp_path):
    with dry_run() as cmds:
        cube, cam = csm.isis_reduce_with_csm(tmp_path / "a.cub", tmp_path / "a.json", 4, tmp_path / "out")
    assert cube == tmp_path / "out" / "a.red4.cub" and cam == tmp_path / "out" / "a.red4.json"
    assert cmds[0][0] == "reduce" and "sscale=4" in cmds[0] and "algorithm=average" in cmds[0]
    assert not cam.exists()


def _pixel_diffs(out: str) -> dict[str, float]:
    block = out.split("cam1 to cam2 pixel diff", 1)[1]
    return {k: float(v) for k, v in re.findall(r"(Median|Max):\s*(\S+)", block)[:2]}


@pytest.mark.tools("reduce", "cam_test")
@pytest.mark.parametrize("factor", [4, 5])
def test_reduced_csm_matches_reduced_isis(tmp_path, monkeypatch, factor):
    """CSM camera of an ISIS-reduced cube vs the ISIS camera of that cube (cam_test, ~0.03 px max)."""
    cube, camera = EXAMPLE / "M181058717LE_crop.cub", EXAMPLE / "M181058717LE.json"
    if not cube.is_file():
        pytest.skip(f"ASP LRONAC example not found at {EXAMPLE} (set LUNAR_ASP_EXAMPLE)")
    monkeypatch.chdir(tmp_path)
    red, red_cam = csm.isis_reduce_with_csm(cube, camera, factor, tmp_path)
    assert csm.CubeGeometry.from_cube(red).samples == 5000 // factor
    args = ["--image", red, "--cam1", red, "--cam2", red_cam, "--session1", "isis", "--session2", "csm"]
    out = run("cam_test", [*args, "--sample-rate", 50], capture=True).stdout
    diffs = _pixel_diffs(out)
    assert diffs["Median"] < 0.01 and diffs["Max"] < 0.1
