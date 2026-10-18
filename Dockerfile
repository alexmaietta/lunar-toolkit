FROM mambaorg/micromamba:1.5.10 AS asp

USER root
COPY docker/asp-env.lock /tmp/asp-env.lock
RUN micromamba create -y -p /opt/asp -f /tmp/asp-env.lock \
    && micromamba clean -a -y \
    && rm /tmp/asp-env.lock

FROM asp AS isisdata

ARG WITH_SHAPE_MODEL=0
COPY docker/isisdata.sh /tmp/isisdata.sh
RUN PATH=/opt/asp/bin:$PATH bash /tmp/isisdata.sh /opt/isisdata /opt/asp/etc/isis/rclone.conf "$WITH_SHAPE_MODEL"

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
COPY --from=asp /opt/asp /opt/asp
COPY --from=isisdata /opt/isisdata /opt/isisdata

ENV ASP_ROOT=/opt/asp \
    ISISROOT=/opt/asp \
    ISISDATA=/opt/isisdata \
    PROJ_DATA=/opt/asp/share/proj \
    GDAL_DATA=/opt/asp/share/gdal \
    LUNAR_DATA_DIR=/data \
    LUNAR_RUNS_DIR=/runs \
    XDG_CACHE_HOME=/cache \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/venv/bin:/opt/asp/bin:$PATH

WORKDIR /opt/lunar_toolkit
COPY pyproject.toml README.md LICENSE ./
COPY lunar_toolkit ./lunar_toolkit
COPY examples ./examples
RUN uv venv --python 3.13 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache . \
    && /opt/venv/bin/python -c "import duckdb; duckdb.sql('INSTALL spatial')"

RUN useradd --create-home --uid 1000 lunar \
    && mkdir -p /data /runs /cache /work \
    && cp -r /root/.duckdb /home/lunar/.duckdb \
    && chown -R lunar:lunar /data /runs /cache /work /home/lunar/.duckdb
USER lunar
WORKDIR /work
VOLUME ["/data", "/runs", "/cache"]
EXPOSE 8080

ENTRYPOINT ["lunar"]
CMD ["--help"]
