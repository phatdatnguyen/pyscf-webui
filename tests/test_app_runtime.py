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
    mounted_options = {}

    def mount_gradio(host, blocks, **options):
        mounted_options.update(options)
        return host

    monkeypatch.setattr(app.gr, "mount_gradio_app", mount_gradio)
    monkeypatch.setattr(visualization, "_STATIC_DIR", visualization._STATIC_DIR)
    run_dir = tmp_path / "separate-run-dir"
    host = app.build_app(run_dir)

    assert base_paths == [run_dir / "data"]
    static_route = next(route for route in host.routes if route.path == "/static")
    assert Path(static_route.app.directory) == run_dir / "static"
    assert Path(visualization._STATIC_DIR) == run_dir / "static"
    assert (run_dir / "data").is_dir()
    assert mounted_options == {"path": "/", "css_paths": app._STYLES_PATH}


def test_gradio_mount_serves_styles_editor_and_transient_views(tmp_path, monkeypatch):
    """Exercise the real mounted app: Gradio 6 owns CSS at mount time, not Blocks."""
    import asyncio
    import httpx
    from pyscf_webui import molecular_editor

    editor_dir = tmp_path / "installed-editor"
    editor_dir.mkdir()
    (editor_dir / "index.html").write_text("<html><body>Molecular editor</body></html>")
    (editor_dir / "editor.js").write_text("window.editorLoaded = true;")
    monkeypatch.setattr(molecular_editor, "KETCHER_DIR", editor_dir)
    monkeypatch.setattr(visualization, "_STATIC_DIR", visualization._STATIC_DIR)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    run_dir = tmp_path / "run"

    host = app.build_app(run_dir)

    async def check_routes():
        async with host.router.lifespan_context(host):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=host),
                                         base_url="http://testserver") as client:
                assert (await client.get("/")).status_code == 200
                config_response = await client.get("/config")
                assert config_response.status_code == 200
                config = config_response.json()
                assert app._STYLES_PATH.read_text() in config["css"]
                assert any(component["props"].get("elem_id") == "molecular-editor"
                           for component in config["components"])
                editor = await client.get("/molecule-editor/")
                assert editor.text == (editor_dir / "index.html").read_text()
                script = await client.get("/molecule-editor/editor.js")
                assert script.text == "window.editorLoaded = true;"
                (run_dir / "static" / "view.html").write_text("<html>Structure view</html>")
                viewer = await client.get("/static/view.html")
                assert viewer.text == "<html>Structure view</html>"

    asyncio.run(check_routes())


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
