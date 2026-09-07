"""Runtime directory and listener configuration should match the running server."""
from pathlib import Path

import pytest

from pyscf_webui import app, visualization


def test_custom_run_directory_configures_files_and_viewers(tmp_path, monkeypatch):
    base_paths = []
    real_blocks = app.working_directory_blocks

    def record_base_path(base_path):
        base_paths.append(Path(base_path))
        return real_blocks(base_path)

    monkeypatch.setattr(app, "working_directory_blocks", record_base_path)
    monkeypatch.setattr(app, "calculation_tab_content", lambda *args: None)
    monkeypatch.setattr(app, "conformer_generation_tab_content", lambda *args: None)
    monkeypatch.setattr(app, "result_tab_content", lambda *args: None)
    monkeypatch.setattr(app.gr, "mount_gradio_app", lambda host, blocks, path: host)
    monkeypatch.setattr(visualization, "_STATIC_DIR", visualization._STATIC_DIR)
    run_dir = tmp_path / "separate-run-dir"
    host = app.build_app(run_dir)

    assert base_paths == [run_dir / "data"]
    static_route = next(route for route in host.routes if route.path == "/static")
    assert Path(static_route.app.directory) == run_dir / "static"
    assert Path(visualization._STATIC_DIR) == run_dir / "static"
    assert (run_dir / "data").is_dir()


def test_port_search_stops_at_last_tcp_port(monkeypatch):
    class OccupiedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 65535)
            raise OSError("occupied")

    monkeypatch.setattr(app.socket, "socket", lambda *args: OccupiedSocket())
    with pytest.raises(OSError, match="No available TCP port"):
        app.find_available_port(65535)


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_invalid_starting_port_has_actionable_error(port):
    with pytest.raises(ValueError, match="between 1 and 65535"):
        app.find_available_port(port)
