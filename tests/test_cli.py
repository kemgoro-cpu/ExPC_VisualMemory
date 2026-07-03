import pytest

from visual_memory import __version__, cli
from visual_memory.config import Settings


def test_cli_runs_local_server_without_opening_browser(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    calls = {}
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    monkeypatch.setattr(cli, "create_app", lambda value: {"settings": value})
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))

    cli.main(["--data-dir", str(settings.data_dir), "--port", "9001", "--no-open"])

    assert calls["app"]["settings"] is settings
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9001


def test_cli_rejects_non_local_bind_and_reports_version(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda _path: Settings(data_dir=tmp_path / "data"))
    with pytest.raises(SystemExit, match="Refusing to bind outside localhost"):
        cli.main(["--host", "0.0.0.0", "--no-open"])
    with pytest.raises(SystemExit) as version:
        cli.build_parser().parse_args(["--version"])
    assert version.value.code == 0
    assert __version__ in capsys.readouterr().out
