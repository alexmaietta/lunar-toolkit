# Install

```bash
pip install lunar-toolkit              # or from a checkout: pip install -e .  /  uv sync
pip install "lunar-toolkit[app]"       # + Qt WebEngine for the `lunar app` window on Linux
pip install -e ".[dev]"                # + pytest, mypy, ruff, pre-commit
pip install -e ".[docs]"               # + mkdocs, mkdocstrings
```

Python 3.10+; the package is pure Python (one wheel for every platform). Image processing and DEMs (`lunar process`, `kaguya`, `shadowcam`, `stereo`, `sfs`) also need ISIS and the Ames Stereo Pipeline, installed as their own docs describe (see [DEMs with ISIS + ASP](dems.md#setup)), or the Docker image below.

## Docker

A self-contained image has ISIS 10 + ASP 3.7 (from `docker/asp-env.lock`, the exact package list of a verified env), a minimal ISISDATA for LRO NAC (`docker/isisdata.sh`) and lunar_toolkit in its own Python 3.13 venv. Pull the published image from the GitHub Container Registry (`latest` and the release versions, `main` for the newest Dockerfile), or build it from the repository.

```bash
docker pull ghcr.io/alexmaietta/lunar-toolkit:latest
docker build -t ghcr.io/alexmaietta/lunar-toolkit .   # or build it yourself
docker build -t ghcr.io/alexmaietta/lunar-toolkit --build-arg WITH_SHAPE_MODEL=1 .   # + 2 GB ISIS LOLA shape model
```

Run it with volumes for the indexes, the runs and the cache.

```bash
docker run --rm -v "$PWD/runs:/runs" -v lunar-data:/data ghcr.io/alexmaietta/lunar-toolkit \
    sfs --config /opt/lunar_toolkit/examples/malapert_crop.yaml
docker run --rm -p 8080:8080 -v lunar-data:/data ghcr.io/alexmaietta/lunar-toolkit serve --host 0.0.0.0 --no-browser
```

The volumes are `/data` (index Parquet files, `LUNAR_DATA_DIR`), `/runs` (run directories, `LUNAR_RUNS_DIR`) and `/cache` (downloads and reference data, `XDG_CACHE_HOME`), and the entry point is `lunar`. The image is about 12 GB and linux/amd64 only. Its ISISDATA has no SPICE kernels for LRO, so `spiceinit` uses the USGS SPICE web service (`web=yes`, the toolkit's default) and needs network access.

## Data locations

| What | Default | Override |
|---|---|---|
| Index Parquet files (+ `*.meta.json`) | `~/.local/share/lunar_toolkit` | `LUNAR_DATA_DIR`, `--data-dir` |
| Raw TAB / LBL, downloads, reference data | `~/.cache/lunar_toolkit` | `XDG_CACHE_HOME`, `--cache-dir` |
| Run directories (relative `work_dir`) | current directory | `runs_dir:` in a config, `LUNAR_RUNS_DIR` |
| ISIS data | `$ISISDATA` | conda env vars or the environment |

## Downloads

PDS files come from `https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/` (URLs on the old `pds.lroc.im-ldi.com/data/` host, still returned by ODE, are rewritten). Downloads resume from `.part` files with HTTP Range and are MD5-checked when the label carries a checksum.

PDS hosts rate-limit aggressively, so downloads are polite and patient: at most `LUNAR_MAX_PER_HOST` (default 3) run per host, requests start at least `LUNAR_MIN_INTERVAL` seconds apart (default 1), and a 429 / 503 pauses every download to that host for the server's `Retry-After` (up to 2 h) instead of retrying. The pause is saved in `~/.cache/lunar_toolkit/host_cooldowns.json`, so a restarted run honours it too; downloads give up only after `LUNAR_MAX_WAIT` seconds of waiting (default 6 h). Other errors and interrupted transfers are retried with back-off.
