"""The standalone editor installs offline-ready files without risking existing data."""
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile

import pytest

from pyscf_webui import install_molecular_editor as installer


@pytest.fixture
def release(monkeypatch):
    files = {name: ("asset " + name).encode() for name in installer.REQUIRED_FILES}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            if name not in installer.NOTICE_SHA256:
                archive.writestr(name, content)
    archive_bytes = buffer.getvalue()
    monkeypatch.setattr(installer, "ARCHIVE_SHA256", hashlib.sha256(archive_bytes).hexdigest())
    monkeypatch.setattr(installer, "NOTICE_SHA256", {
        name: hashlib.sha256(files[name]).hexdigest() for name in installer.NOTICE_SHA256
    })
    requests = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        content = (archive_bytes if request.full_url == installer.ARCHIVE_URL
                   else files[request.full_url.rsplit("/", 1)[-1]])
        return io.BytesIO(content)

    monkeypatch.setattr(installer, "urlopen", urlopen)
    return files, requests


def test_installs_and_verifies_assets_and_notices(tmp_path, release):
    files, requests = release
    destination = tmp_path / "ketcher"
    assert installer.install_ketcher(destination) == destination
    for name, content in files.items():
        assert (destination / name).read_bytes() == content
    assert installer.installation_is_current(destination)
    marker = json.loads((destination / installer.MARKER).read_text())
    assert marker["archive_url"] == installer.ARCHIVE_URL
    assert marker["version"] == installer.KETCHER_VERSION
    assert set(marker["files"]) == set(files)
    assert len(requests) == 3  # standalone archive plus its root license and notice
    assert list(tmp_path.iterdir()) == [destination]


def test_verified_install_is_idempotent_and_needs_no_network(tmp_path, release, monkeypatch):
    destination = installer.install_ketcher(tmp_path / "ketcher")
    monkeypatch.setattr(installer, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert installer.install_ketcher(destination) == destination


def test_checksum_failure_leaves_no_partial_install(tmp_path, release, monkeypatch):
    monkeypatch.setattr(installer, "ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        installer.install_ketcher(tmp_path / "ketcher")
    assert list(tmp_path.iterdir()) == []


def test_modified_install_is_preserved(tmp_path, release):
    destination = installer.install_ketcher(tmp_path / "ketcher")
    (destination / "index.html").write_text("user's changed installation")
    assert not installer.installation_is_current(destination)
    with pytest.raises(RuntimeError, match="no files were changed"):
        installer.install_ketcher(destination)
    assert (destination / "index.html").read_text() == "user's changed installation"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "C:escape",
                                  "file:stream", "folder\\escape", "", "./", "x\0y"])
def test_archive_paths_reject_escape_and_platform_specific_names(name):
    with pytest.raises(ValueError, match="Unsafe editor archive path"):
        installer.safe_relative_path(name)


def test_extraction_rejects_symlinks(tmp_path):
    archive_path = tmp_path / "release.zip"
    entry = zipfile.ZipInfo("link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(entry, "../../outside")
    with pytest.raises(ValueError, match="Unsupported editor archive entry"):
        installer.extract_archive(archive_path, tmp_path / "staging")
    assert not (tmp_path / "staging").exists()


def test_installation_marker_cannot_point_outside_bundle(tmp_path, release):
    destination = installer.install_ketcher(tmp_path / "ketcher")
    marker_path = destination / installer.MARKER
    marker = json.loads(marker_path.read_text())
    marker["files"]["../outside"] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    assert not installer.installation_is_current(destination)


def test_asset_directory_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSCF_WEBUI_EDITOR_DIR", str(tmp_path / "custom"))
    assert installer.default_editor_directory() == tmp_path / "custom"


def test_linux_uses_user_data_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("PYSCF_WEBUI_EDITOR_DIR", raising=False)
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert installer.default_editor_directory() == (
        tmp_path / "pyscf-webui" / "ketcher" / installer.KETCHER_VERSION)


def test_cli_uses_requested_destination(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "editor"
    calls = []
    monkeypatch.setattr(installer.sys, "argv", ["pyscf-webui-install-editor",
                                               "--destination", str(destination)])
    monkeypatch.setattr(installer, "install_ketcher", lambda path: calls.append(path) or path)
    installer.main()
    assert calls == [Path(destination)]
    assert "Verified Ketcher" in capsys.readouterr().out
