"""tool_env picks up a conda env's config vars when the env is not activated."""

import json

from lunar_toolkit.tools import tool_env


def test_conda_env_vars(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "conda-meta").mkdir()
    (prefix / "IsisPreferences").touch()
    data = tmp_path / "isisdata"
    state = {"env_vars": {"ISISDATA": str(data), "ISISROOT": str(prefix)}}
    (prefix / "conda-meta" / "state").write_text(json.dumps(state))
    for var in ("ISISROOT", "ISISDATA"):
        monkeypatch.delenv(var, raising=False)
    env = tool_env(prefix / "bin" / "spiceinit")
    assert env["ISISDATA"] == str(data)
    assert env["ISISROOT"] == str(prefix)

    monkeypatch.setenv("ISISDATA", "/explicit")
    assert tool_env(prefix / "bin" / "spiceinit")["ISISDATA"] == "/explicit"


def test_broken_state_file(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "conda-meta").mkdir()
    (prefix / "conda-meta" / "state").write_text("not json")
    monkeypatch.delenv("ISISDATA", raising=False)
    assert "ISISDATA" not in tool_env(prefix / "bin" / "x")
