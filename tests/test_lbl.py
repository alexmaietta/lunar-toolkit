"""PDS3 label parsing and column-type mapping."""

from lunar_toolkit.lroc.cumindex import (
    _column_kind,
    _parse_lbl_meta,
    _parse_pds_lbl_columns,
    column_sql,
    parse_lbl_columns,
)


def test_parse_columns(lbl_text):
    cols = parse_lbl_columns(lbl_text)
    assert len(cols) == 83
    names = [c["name"] for c in cols]
    assert names[0] == "VOLUME_ID"
    assert names[-1] == "LRO_FLIGHT_DIRECTION"
    pid = next(c for c in cols if c["name"] == "PRODUCT_ID")
    assert (pid["start_byte"], pid["bytes"], pid["data_type"]) == (123, 13, "CHARACTER")
    # multi-line DESCRIPTION is collapsed, not split into bogus keys
    fsn = next(c for c in cols if c["name"] == "FILE_SPECIFICATION_NAME")
    assert "relative to a PDS volume" in fsn["description"]


def test_corner_columns_present(lbl_text):
    names = {c["name"] for c in parse_lbl_columns(lbl_text)}
    for corner in ("UPPER_LEFT", "UPPER_RIGHT", "LOWER_RIGHT", "LOWER_LEFT"):
        assert f"{corner}_LATITUDE" in names
        assert f"{corner}_LONGITUDE" in names


def test_meta(lbl_text):
    meta = _parse_lbl_meta(lbl_text)
    assert meta["record_bytes"] == 903
    assert meta["file_records"] == 4142412
    assert meta["columns"] == 83
    assert meta["md5_checksum"] == "a45d60e14a7fac114820f51bbe89b9ed"
    assert meta["data_set_id"] == "LRO-L-LROC-2-EDR-V1.0"


def test_schema_dataframe(lbl_text):
    df = _parse_pds_lbl_columns(lbl_text)
    assert list(df.columns[:6]) == ["name", "column_number", "data_type", "start_byte", "bytes", "format"]
    assert df["column_number"].is_monotonic_increasing


def test_column_kinds(lbl_text):
    cols = {c["name"]: c for c in parse_lbl_columns(lbl_text)}
    assert _column_kind(cols["ORBIT_NUMBER"]) == "BIGINT"
    assert _column_kind(cols["CENTER_LATITUDE"]) == "DOUBLE"
    assert _column_kind(cols["START_TIME"]) == "TIMESTAMP"
    assert _column_kind(cols["SPACECRAFT_CLOCK_START_COUNT"]) == "VARCHAR"
    assert _column_kind({"name": "EMISSION_ANGLE"}) == "DOUBLE"  # name hint fallback


def test_column_sql_sentinel(lbl_text):
    cols = {c["name"]: c for c in parse_lbl_columns(lbl_text)}
    sql = column_sql(cols["CENTER_LATITUDE"])
    assert "substr(line, 808, 6)" in sql
    assert "999.99" in sql  # NOT_APPLICABLE_CONSTANT → NULL
    assert column_sql(cols["PRODUCT_ID"]).startswith("upper(")
