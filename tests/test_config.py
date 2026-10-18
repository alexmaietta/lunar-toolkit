"""YAML run config, region resolution, provenance and the sfs --config / --dump-config / --check CLI (offline)."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from lunar_toolkit import cli, config, provenance, reference, sfs, sfs_products
from lunar_toolkit.config import ConfigError, RunConfig, dump_config, from_dict, load_config, merge, resolve
from lunar_toolkit.sfs import SfsConfig, SfsPipeline
from lunar_toolkit.tools import dry_run, record_commands, run

EXAMPLES = Path(__file__).parent.parent / "examples"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_round_trip(tmp_path):
    cfg = RunConfig(
        work_dir=tmp_path / "w",
        dem="ldem.tif",
        sources=["M1LE", "M2RE"],
        crop_center=(-85.9, -0.2),
        sfs=SfsConfig(
            resolution=2.0,
            projwin=(0.0, 10.0, 10.0, 0.0),
            preset="asp",
            reduce=None,
            seed=7,
            sfs_options={"smoothness_weight": 0.05, "float_albedo": True},
            bundle_adjust_options={"overlap_limit": None},
        ),
    )
    text = dump_config(cfg, tmp_path / "c.yaml")
    assert "projwin: [0.0, 10.0, 10.0, 0.0]" in text and "sources:\n- M1LE\n" in text
    assert text.isascii()
    back = load_config(tmp_path / "c.yaml")
    assert back == cfg
    assert dump_config(back) == text
    assert set(yaml.safe_load(text)) == set(config.KEYS)


def test_defaults_and_empty_file(tmp_path):
    assert load_config(_write(tmp_path / "e.yaml", "")) == RunConfig()
    cfg = load_config(
        _write(tmp_path / "a.yaml", "resolution: 2\nreduce: 3\nprojwin: [1, 2, 3, 4]\nregion: 11\n")
    )
    assert cfg.sfs.resolution == 2.0 and isinstance(cfg.sfs.resolution, float)
    assert cfg.sfs.reduce == 3 and cfg.sfs.projwin == (1.0, 2.0, 3.0, 4.0) and cfg.region == 11


@pytest.mark.parametrize(
    "text, match",
    [
        ("resoluton: 5\n", "unknown key 'resoluton' \\(did you mean 'resolution'\\?\\)"),
        ("bogus: 1\n", "unknown key 'bogus'$"),
        ("resolution: fast\n", "resolution: expected number"),
        ("csm: 1\n", "csm: expected boolean"),
        ("stereo_pairs: 2.5\n", "stereo_pairs: expected integer"),
        ("projwin: [1, 2, 3]\n", "projwin: expected list of 4 numbers or null"),
        ("sources: M1LE\n", "sources: expected list of strings"),
        ("sfs_options: [1]\n", "sfs_options: expected mapping"),
        ("preset: fancy\n", "preset must be one of bertone, asp"),
        ("reduce: sometimes\n", "reduce must be an integer"),
        ("- a\n- b\n", "expected a mapping"),
        ("a: [\n", "invalid YAML"),
    ],
)
def test_validation_errors(tmp_path, text, match):
    path = _write(tmp_path / "bad.yaml", text)
    with pytest.raises(ConfigError, match=match) as exc:
        load_config(path)
    assert str(path) in str(exc.value)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read config"):
        load_config(tmp_path / "nope.yaml")


def test_merge_options_and_dashes():
    cfg = from_dict({"sfs_options": {"max-iterations": 3, "smoothness_weight": 0.1}, "work-dir": "w"})
    assert cfg.work_dir == Path("w") and cfg.sfs.sfs_options == {
        "max_iterations": 3,
        "smoothness_weight": 0.1,
    }
    cfg = merge(cfg, {"sfs_options": {"max_iterations": 9}, "resolution": 1})
    assert cfg.sfs.sfs_options == {"max_iterations": 9, "smoothness_weight": 0.1}
    assert cfg.sfs.options("sfs")["max_iterations"] == 9 and cfg.sfs.resolution == 1.0


@pytest.fixture
def fake_pgda(tmp_path, monkeypatch):
    calls = []

    def download(site, kinds, out_dir, *a, **kw):
        calls.append((site, list(kinds), Path(out_dir)))
        return {k: [_write(Path(out_dir) / site / f"{site}_final_adj_5mpp_{k}.tif", "x")] for k in kinds}

    monkeypatch.setattr(reference, "download_pgda", download)
    return calls


def test_resolve_region_crop(tmp_path, fake_pgda):
    cfg = RunConfig(region="malapert", crop_km=2.5, pgda_dir=tmp_path / "pgda")
    out = resolve(cfg)
    box = reference.crop_window("malapert", 2.5)
    assert out.sfs.projwin == box.projwin and out.sfs.region_id == "A311"
    assert out.dem == str(tmp_path / "pgda" / "Site23" / "Site23_final_adj_5mpp_surf.tif")
    assert fake_pgda == [("Site23", ["surf"], tmp_path / "pgda")]
    assert resolve(out) == out
    wkt = config.search_aoi(out)
    assert wkt.startswith("POLYGON((") and config.search_aoi(RunConfig(aoi="a.geojson")) == "a.geojson"


def test_resolve_region_keeps_explicit_values(tmp_path, fake_pgda):
    cfg = RunConfig(region=11, dem="mine.tif", sfs=SfsConfig(projwin=(0, 1, 1, 0), region_id="X"))
    out = resolve(cfg)
    assert out.dem == "mine.tif" and out.sfs.projwin == (0, 1, 1, 0) and out.sfs.region_id == "X"
    full = resolve(RunConfig(region="Site23", pgda_dir=tmp_path), download=False)
    assert full.sfs.projwin == reference.get_region(11).projwin and not fake_pgda
    assert not (tmp_path / "Site23").exists()


@pytest.mark.parametrize(
    "cfg, match",
    [
        (RunConfig(crop_km=2.5), "need a region"),
        (RunConfig(dem="pgda"), "needs a region"),
        (RunConfig(region="atlantis"), "region: No region matches"),
    ],
)
def test_resolve_errors(cfg, match):
    with pytest.raises(ConfigError, match=match):
        resolve(cfg, download=False)


def test_load_sources(tmp_path):
    f = _write(tmp_path / "list.txt", "# ids\nM3LE\n\n M4RE \n")
    assert config.load_sources(RunConfig(sources=["M1LE"], from_file=f)) == ["M1LE", "M3LE", "M4RE"]
    with pytest.raises(ConfigError, match="from_file"):
        config.load_sources(RunConfig(from_file=tmp_path / "missing.txt"))


@pytest.mark.parametrize("name", ["malapert_crop.yaml", "asp_recipe.yaml"])
def test_examples_validate(name, tmp_path):
    cfg = load_config(EXAMPLES / name)
    out = resolve(merge(cfg, {"pgda_dir": tmp_path}), download=False)
    assert out.dem and out.sfs.projwin
    assert all(line.isascii() for line in (EXAMPLES / name).read_text().splitlines())


def test_malapert_example_is_the_bertone_crop():
    cfg = load_config(EXAMPLES / "malapert_crop.yaml")
    assert cfg.region == "malapert" and cfg.crop_km == 2.5 and cfg.sfs.preset == "bertone"


def _cli(*argv):
    return cli.main(["sfs", *map(str, argv)])


def test_cli_dump_config_overrides(tmp_path, capsys, fake_pgda):
    path = _write(
        tmp_path / "run.yaml",
        f"work_dir: {tmp_path / 'w'}\ndem: d.tif\nresolution: 5\nsources: [M1LE]\nsfs_options: {{max_iterations: 3}}\n",
    )
    assert (
        _cli("--config", path, "--res", 2, "--no-csm", "--sfs-opt", "smoothness_weight=0.1", "--dump-config")
        == 0
    )
    cfg = from_dict(yaml.safe_load(capsys.readouterr().out))
    assert cfg.sfs.resolution == 2.0 and cfg.sfs.csm is False and cfg.dem == "d.tif"
    assert cfg.sfs.sfs_options == {"max_iterations": 3, "smoothness_weight": 0.1}
    assert cfg.sources == ["M1LE"] and cfg.sfs.web_spice is True
    assert _cli("--config", path, "M9LE", "--dump-config") == 0
    assert from_dict(yaml.safe_load(capsys.readouterr().out)).sources == ["M9LE"]
    assert not (tmp_path / "w").exists() and not fake_pgda


def test_cli_dump_config_region(tmp_path, capsys, fake_pgda):
    assert _cli("--region", "malapert", "--crop-km", 2.5, "--dump-config") == 0
    cfg = from_dict(yaml.safe_load(capsys.readouterr().out))
    assert cfg.sfs.projwin == reference.crop_window(11, 2.5).projwin and cfg.dem.endswith(
        "Site23_final_adj_5mpp_surf.tif"
    )
    assert not fake_pgda


def test_cli_config_errors(tmp_path, capsys):
    path = _write(tmp_path / "run.yaml", "resolutoin: 5\n")
    assert _cli("--config", path, "--dump-config") == 1
    assert "did you mean 'resolution'" in capsys.readouterr().err
    assert _cli("M1LE", "--work-dir", tmp_path / "w") == 1
    assert "No prior DEM" in capsys.readouterr().err


def test_cli_dry_run_from_config(tmp_path, capsys, fake_pgda):
    path = _write(
        tmp_path / "run.yaml",
        f"work_dir: {tmp_path / 'w'}\nregion: malapert\ncrop_km: 2.5\npgda_dir: {tmp_path / 'pgda'}\n"
        "sources: [M1234567890LE]\nregistration: residuals\nreduce: null\n",
    )
    assert _cli("--config", path, "--dry-run", "--sfs-opt", "max_iterations=2") == 0
    lines = capsys.readouterr().out.splitlines()
    sfs_line = next(ln for ln in lines if ln.startswith("parallel_sfs"))
    assert "--max-iterations 2" in sfs_line
    win = reference.crop_window(11, 2.5).projwin
    assert any(ln.startswith("gdal_translate") and f"-projwin {win[0]:g}" in ln for ln in lines)
    assert not fake_pgda and not (tmp_path / "w").exists() and not (tmp_path / "pgda").exists()


def test_pipeline_writes_config_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "tool_versions", lambda: {"asp": "ASP 3.7.0", "gdal": None})
    src = _write(tmp_path / "A.IMG", "image bytes")
    dem = _write(tmp_path / "ldem.tif", "dem bytes")
    rc = RunConfig(region="malapert", sources=["A.IMG"], sfs=SfsConfig(seed=3))
    p = SfsPipeline(tmp_path / "w", dem, SfsConfig(seed=3), run_config=rc)
    with dry_run():
        assert p.start_provenance([src]) is None
    assert not (tmp_path / "w").exists()
    assert p.start_provenance([src]) == tmp_path / "w" / "config.yaml"
    cfg = load_config(tmp_path / "w" / "config.yaml")
    assert (
        cfg.dem == str(dem)
        and cfg.work_dir == tmp_path / "w"
        and cfg.region == "malapert"
        and cfg.sfs.seed == 3
    )
    assert p._step("echo", lambda x: x, src) == src
    with pytest.raises(RuntimeError):
        p._step("boom", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    p._prov.finish("error")
    data = json.loads((tmp_path / "w" / "provenance.json").read_text())
    run_ = data["runs"][-1]
    assert data["schema"] == 1 and run_["status"] == "error" and run_["tools"]["asp"] == "ASP 3.7.0"
    assert run_["config_sha256"] == config.config_hash(cfg)
    md5 = provenance.md5sum(src)
    assert run_["inputs"][0] == {
        "path": str(src),
        "size": 11,
        "mtime": run_["inputs"][0]["mtime"],
        "md5": md5,
    }
    assert [(s["name"], s["status"]) for s in run_["steps"]] == [("echo", "ok"), ("boom", "error")]
    assert run_["steps"][0]["outputs"] == [str(src)] and "bad" in run_["steps"][1]["error"]
    SfsPipeline(tmp_path / "w", dem, SfsConfig(), run_config=rc).start_provenance([src])
    assert len(json.loads((tmp_path / "w" / "provenance.json").read_text())["runs"]) == 2


def test_seed_reaches_ensemble(tmp_path, monkeypatch):
    seen = {}
    real = sfs_products.ensemble_subsets

    def spy(pool, n, size, **kw):
        seen.update(kw)
        return real(pool, n, size, **kw)

    monkeypatch.setattr(sfs_products, "ensemble_subsets", spy)
    p = SfsPipeline(
        tmp_path / "w", tmp_path / "d.tif", SfsConfig(ensemble_members=2, seed=42, projwin=(0, 9, 9, 0))
    )
    p.images = [sfs.SfsImage(f"I{k}", tmp_path / f"I{k}.cub") for k in range(4)]
    with dry_run():
        p.ensemble()
    assert seen["seed"] == 42


def _fake_tools(tmp_path: Path) -> Path:
    prefix = tmp_path / "env"
    for name, text in (
        ("sfs", "NASA Ames Stereo Pipeline 3.7.0\\n  Build ID: abc"),
        ("isd_generate", "ale version 1.2.0"),
        ("gdalinfo", "GDAL 3.12.2"),
    ):
        _write(prefix / "bin" / name, f'#!/bin/sh\nprintf "{text}\\n"\n').chmod(0o755)
    (prefix / "IsisPreferences").touch()
    _write(prefix / "isis_version.txt", "10.0.0 # Public version number\n2026-06-09 # Release date\n")
    _write(prefix / "data" / "base" / "kernels" / "lsk" / "naif0012.tls", "")
    _write(prefix / "data" / "lro" / "kernels" / "sclk" / "lro_clkcor_2025351_v00.tsc", "")
    return prefix


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX shell scripts")
def test_environment_from_fake_tools(tmp_path, monkeypatch):
    prefix = _fake_tools(tmp_path)
    monkeypatch.setenv("ASP_ROOT", str(prefix))
    for var in ("ISISROOT", "ISISDATA"):
        monkeypatch.delenv(var, raising=False)
    env = provenance.environment()
    assert env["tools"] == {
        "asp": "NASA Ames Stereo Pipeline 3.7.0",
        "ale": "ale version 1.2.0",
        "gdal": "GDAL 3.12.2",
        "isis": "10.0.0",
    }
    assert env["isis_paths"] == {"ISISROOT": str(prefix), "ISISDATA": str(prefix / "data")}
    assert env["kernels"] == {"lsk": ["naif0012.tls"], "lro_sclk": ["lro_clkcor_2025351_v00.tsc"]}
    with dry_run() as cmds:
        assert provenance.tool_versions() == {}
    assert cmds == []
    with record_commands() as log:
        run("gdalinfo", ["--version"], capture=True)
    assert log[0]["argv"] == [str(prefix / "bin" / "gdalinfo"), "--version"] and log[0]["returncode"] == 0


def test_file_record_reuses_md5(tmp_path):
    f = _write(tmp_path / "a.bin", "abc")
    rec = provenance.file_record(f)
    assert rec["md5"] == "900150983cd24fb0d6963f7d28e17f72"
    assert provenance.file_record(f, known={**rec, "md5": "cached"})["md5"] == "cached"
    assert provenance.file_record(f, checksum=False)["md5"] is None
    assert provenance.file_record(f, max_bytes=2)["md5"] is None
    assert provenance.file_record(tmp_path / "none")["size"] is None


def test_check_reports_differences(tmp_path, monkeypatch, capsys):
    versions = {"asp": "ASP 3.7.0"}
    monkeypatch.setattr(provenance, "tool_versions", lambda: dict(versions))
    src = _write(tmp_path / "A.IMG", "image")
    dem = _write(tmp_path / "ldem.tif", "dem")
    rc = RunConfig(work_dir=tmp_path / "w", dem=str(dem), sources=[str(src)])
    SfsPipeline(tmp_path / "w", dem, rc.sfs, run_config=rc).start_provenance([src])
    run_yaml = tmp_path / "w" / "config.yaml"
    assert provenance.check(tmp_path / "w", load_config(run_yaml), [src, dem]) == []
    assert _cli("--config", run_yaml, "--check") == 0
    assert "No differences" in capsys.readouterr().out

    versions["asp"] = "ASP 3.8.0"
    src.write_text("other image bytes")
    diffs = provenance.check(tmp_path / "w", merge(rc, {"resolution": 2}), [src, dem])
    kinds = {(d.kind, d.key) for d in diffs}
    assert kinds == {("environment", "tools.asp"), ("config", "resolution"), ("input", str(src))}
    assert _cli("--config", run_yaml, "--res", 2, "--check") == 1
    out = capsys.readouterr().out
    assert "3 differences" in out and "tools.asp" in out and "resolution: 5.0 -> 2.0" in out

    dem.unlink()
    assert any(d.current == "missing" for d in provenance.check(tmp_path / "w"))
    with pytest.raises(FileNotFoundError):
        provenance.check(tmp_path / "elsewhere")
    assert _cli("--work-dir", tmp_path / "elsewhere", "--dem", dem, "--check") == 1


def test_runs_dir_places_relative_work_dirs(tmp_path, monkeypatch):
    from lunar_toolkit import config as cfgmod

    monkeypatch.delenv(cfgmod.RUNS_DIR_ENV, raising=False)
    cfg = cfgmod.from_dict({"work_dir": "crop", "dem": str(tmp_path / "d.tif")})
    assert Path(cfgmod.resolve(cfg).work_dir) == Path("crop")
    monkeypatch.setenv(cfgmod.RUNS_DIR_ENV, str(tmp_path / "env_runs"))
    assert Path(cfgmod.resolve(cfg).work_dir) == tmp_path / "env_runs" / "crop"
    cfg = cfgmod.from_dict(
        {"work_dir": "crop", "runs_dir": str(tmp_path / "runs"), "dem": str(tmp_path / "d.tif")}
    )
    assert Path(cfgmod.resolve(cfg).work_dir) == tmp_path / "runs" / "crop"
    cfg = cfgmod.from_dict(
        {"work_dir": str(tmp_path / "abs"), "runs_dir": str(tmp_path / "runs"), "dem": "x.tif"}
    )
    assert Path(cfgmod.resolve(cfg).work_dir) == tmp_path / "abs"
