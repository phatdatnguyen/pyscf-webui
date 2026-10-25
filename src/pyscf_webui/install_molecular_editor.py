"""Install the pinned, self-contained Ketcher editor (stdlib only).

Run once with ``pyscf-webui-install-editor``. The application then serves
the editor locally, without a CDN, Node build, or separate chemistry service.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from urllib.request import Request, urlopen
import zipfile


KETCHER_VERSION = "3.18.0"


def default_editor_directory():
    """Keep the downloaded bundle outside both the package and transient viewers."""
    override = os.environ.get("PYSCF_WEBUI_EDITOR_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "pyscf-webui" / "ketcher" / KETCHER_VERSION


KETCHER_DIR = default_editor_directory()
ARCHIVE_URL = (
    "https://github.com/epam/ketcher/releases/download/"
    f"v{KETCHER_VERSION}/ketcher-standalone-{KETCHER_VERSION}.zip"
)
ARCHIVE_SHA256 = "484e7f10a0e74808ae5f43f0e6448db6f4d884bdd2189aa271108474167a3e7e"
NOTICE_SHA256 = {
    "LICENSE": "3d54b234fa88c11d1d96380ff5fe29f3c5d8e1078ef914dfe8185ce4b1fbeac7",
    "NOTICE": "ab82986438fc66fa2b40836415b734ff018b1e42ee3a18434b27bdf1c42eb368",
}
MARKER = "installation.json"
REQUIRED_FILES = (
    "index.html", "static/js/main.e47c48ad.js", "static/css/main.96b87be0.css",
    "LICENSE", "NOTICE",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, destination, expected_sha256):
    request = Request(url, headers={"User-Agent": "pyscf-webui editor installer"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    if file_sha256(destination) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {url}; installation cancelled.")


def safe_relative_path(name):
    path = PurePosixPath(name)
    # Colons also reject Windows drive paths and alternate data streams, even when
    # validating an archive on a Unix host before a later Windows installation.
    if (not name or "\\" in name or "\x00" in name or ":" in name or path.is_absolute()
            or ".." in path.parts or not path.parts):
        raise ValueError(f"Unsafe editor archive path: {name!r}")
    return Path(*path.parts)


def extract_archive(archive_path, destination):
    """Extract regular files only; never permit links or paths outside staging."""
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative_path = safe_relative_path(entry.filename)
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported editor archive entry: {entry.filename!r}")
            output_path = destination / relative_path
            if entry.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, output_path.open("xb") as output:
                    shutil.copyfileobj(source, output)


def installation_is_current(destination):
    """Check both pinned provenance and the files recorded at installation."""
    try:
        if destination.is_symlink():
            return False
        marker = json.loads((destination / MARKER).read_text(encoding="utf-8"))
        if (marker["version"] != KETCHER_VERSION
                or marker["archive_url"] != ARCHIVE_URL
                or marker["archive_sha256"] != ARCHIVE_SHA256):
            return False
        files = marker["files"]
        if not isinstance(files, dict) or not set(REQUIRED_FILES).issubset(files):
            return False
        for name, expected in files.items():
            path = destination / safe_relative_path(name)
            if not path.resolve().is_relative_to(destination.resolve()):
                return False
            if not path.is_file() or file_sha256(path) != expected:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def install_ketcher(destination=KETCHER_DIR):
    """Install atomically into a new directory; preserve existing directories."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        if installation_is_current(destination):
            return destination
        raise RuntimeError(
            f"{destination} already exists and does not match verified Ketcher "
            f"{KETCHER_VERSION}. Move it aside before reinstalling; no files were changed."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ketcher-install-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        archive = staging / "release.zip"
        download_verified(ARCHIVE_URL, archive, ARCHIVE_SHA256)
        editor = staging / "editor"
        editor.mkdir()
        extract_archive(archive, editor)
        # Release bundles retain their dependency notices but omit these root files.
        for name, expected in NOTICE_SHA256.items():
            if not (editor / name).exists():
                url = f"https://raw.githubusercontent.com/epam/ketcher/v{KETCHER_VERSION}/{name}"
                download_verified(url, editor / name, expected)
        if not all((editor / name).is_file() for name in REQUIRED_FILES):
            raise ValueError("The Ketcher archive is missing required editor assets.")
        marker = {
            "version": KETCHER_VERSION,
            "archive_url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "files": {
                path.relative_to(editor).as_posix(): file_sha256(path)
                for path in sorted(editor.rglob("*")) if path.is_file()
            },
        }
        (editor / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"{destination} was created during installation; left it unchanged.")
        editor.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=KETCHER_DIR,
        help="Asset directory (set PYSCF_WEBUI_EDITOR_DIR to the same path when running the app).",
    )
    args = parser.parse_args()
    try:
        path = install_ketcher(args.destination.expanduser().resolve())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"Editor installation failed: {exc}") from exc
    print(f"Verified Ketcher {KETCHER_VERSION}: {path}")


if __name__ == "__main__":
    main()
