"""Native Windows must not require extensions with no Windows distributions."""
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
import pytest
import tomlkit


def dependencies_for(platform):
    metadata = tomlkit.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    environment = dict(default_environment(), sys_platform=platform)
    dependencies = [Requirement(item) for item in metadata["project"]["dependencies"]]
    return {item.name: item for item in dependencies
            if item.marker is None or item.marker.evaluate(environment)}


def test_native_windows_uses_pyscf_with_windows_wheels_without_dispersion_extension():
    dependencies = dependencies_for("win32")
    assert "2.15.1" in dependencies["pyscf"].specifier
    assert "2.14.0" not in dependencies["pyscf"].specifier
    assert "pyscf-dispersion" not in dependencies
    assert "pyscf-properties" in dependencies
    assert "gpu4pyscf-cuda12x" not in dependencies


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_non_windows_installation_keeps_dispersion_and_existing_pyscf_support(platform):
    dependencies = dependencies_for(platform)
    assert "pyscf-dispersion" in dependencies
    assert "2.14.0" in dependencies["pyscf"].specifier
    assert "pyscf-properties" in dependencies
