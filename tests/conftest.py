"""Shared fixtures and capability markers.

Most of this suite runs without PySCF: the mapping layer, the file-naming helpers and the
Result tab are all engine-free by design, and keeping their tests engine-free is what lets
them run in a bare checkout. Tests that genuinely need the engine are marked ``pyscf``,
and those that need a working GPU are marked ``gpu``.

Availability is probed in a **subprocess**, for the same reason the web server does it that
way: importing cupy here would create a CUDA context in the pytest process, and a driver
fault would take the whole run down rather than skipping a few tests.
"""
import subprocess
import sys

import pytest


def _module_importable(module: str) -> bool:
    try:
        return subprocess.run([sys.executable, "-c", f"import {module}"],
                              capture_output=True, timeout=300).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


PYSCF_AVAILABLE = _module_importable("pyscf")
GPU_AVAILABLE = _module_importable("gpu4pyscf.dft")

needs_pyscf = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="pyscf is not installed")
needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="gpu4pyscf is not usable here")


@pytest.fixture
def water_result():
    """A minimal completed result file, shaped like what the runner writes."""
    return {
        "version": 2,
        "status": "completed",
        "error": None,
        "engine": "pyscf",
        "device": "cpu",
        "calculation_type": "Single-Point",
        "method_type": "DFT",
        "functional": "B3LYP",
        "xc": "B3LYP",
        "basis_set": "6-31G(d)",
        "reference": "RHF",
        "charge": 0,
        "multiplicity": 1,
        "structure_file": "water.xyz",
        "solvent": None,
        "log_file": "sp.log",
        "symbols": ["O", "H", "H"],
        "geometry_angstrom": [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
        "energy_hartree": -76.4068,
        "dipole_debye": 2.07,
        "dipole_vector_debye": [0.0, 0.0, -2.07],
        "mo_energies_hartree": [-19.1, -1.0, -0.5, -0.4, -0.3, 0.1, 0.2],
        "homo_index": 4,
        "n_alpha": 5,
        "n_beta": 5,
        "wavefunction_file": "sp.chk",
        "duration_seconds": 2.0,
    }
