"""Explicit dispersion choices must affect the energy or fail before calculation."""
import builtins
import json
from types import SimpleNamespace

import pytest

from pyscf_webui import engine, runner

from conftest import needs_pyscf


@pytest.mark.parametrize("functional,selected,expected", [
    ("B3LYP", "d3bj", ("B3LYP", "d3bj")),
    ("PBE0", "D4", ("PBE0", "d4")),
    ("B3LYP-D3BJ", "", ("B3LYP", "d3bj")),
    ("PBE-D3ZERO", "d3zero", ("PBE", "d3zero")),
    ("CAM-B3LYP", "", ("CAM-B3LYP", None)),
])
def test_explicit_and_legacy_dispersion_choices(functional, selected, expected):
    assert engine.dispersion_settings({
        "method_type": "DFT", "functional": functional, "dispersion": selected,
    }) == expected


def test_conflicting_dispersion_choices_are_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        engine.dispersion_settings({
            "method_type": "TD-DFT", "functional": "B3LYP-D3BJ", "dispersion": "d4",
        })


@pytest.mark.parametrize("method_type", ["MP2", "CCSD", "CCSD(T)"])
def test_post_hf_jobs_do_not_silently_use_hf_dispersion_parameters(method_type):
    with pytest.raises(ValueError, match="not supported"):
        engine.dispersion_settings({"method_type": method_type, "dispersion": "d3bj"})


def test_hf_uses_explicit_dispersion_without_a_stale_dft_functional():
    assert engine.dispersion_settings({
        "method_type": "HF", "functional": "B3LYP-D4", "dispersion": "d3bj",
    }) == (None, "d3bj")


def _hide_backend(monkeypatch, error_type=ModuleNotFoundError):
    original_import = builtins.__import__

    def without_dispersion(name, *args, **kwargs):
        if name == "pyscf.dispersion":
            raise error_type("The dispersion extension cannot be loaded")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_dispersion)


@pytest.mark.parametrize("platform,remedy", [("win32", "WSL2/Linux"), ("linux", "pip install")])
@pytest.mark.parametrize("error_type", [ModuleNotFoundError, OSError])
def test_missing_or_broken_backend_has_platform_specific_remedy(monkeypatch, platform,
                                                               remedy, error_type):
    _hide_backend(monkeypatch, error_type)
    monkeypatch.setattr(engine, "sys", SimpleNamespace(platform=platform))
    with pytest.raises(RuntimeError, match=remedy) as error:
        engine._require_dispersion_backend("d3bj")
    assert "D3BJ" in str(error.value)
    assert "pyscf-dispersion" in str(error.value)
    assert isinstance(error.value.__cause__, error_type)


@pytest.fixture
def cpu_spec():
    return {
        "calculation_type": "Single-Point", "method_type": "DFT", "functional": "B3LYP",
        "basis_set": "sto-3g", "geometry": "O 0 0 0; H 0 0 0.96; H 0 0.92 -0.24",
        "charge": 0, "multiplicity": 1, "reference": "RHF", "n_threads": 1,
        "verbose": 0, "density_fit": False, "grid_level": 0,
        "save_wavefunction": False, "save_molden": False, "save_arrays": False,
    }


@pytest.fixture
def available_backend():
    pytest.importorskip("pyscf.dispersion.dftd3")
    pytest.importorskip("pyscf.dispersion.dftd4")


@needs_pyscf
@pytest.mark.parametrize("dispersion", ["d3bj", "d3zero", "d4"])
@pytest.mark.parametrize("method_type", ["DFT", "HF"])
def test_selected_correction_changes_the_scf_energy(cpu_spec, available_backend,
                                                  dispersion, method_type):
    from pyscf import lib

    lib.num_threads(1)
    cpu_spec["method_type"] = method_type
    mol = engine.build_mol(cpu_spec, verbose=0)
    uncorrected = engine.build_scf(cpu_spec, mol, use_gpu=False)
    baseline_energy = engine.run_scf(uncorrected)
    corrected = engine.build_scf(dict(cpu_spec, dispersion=dispersion), mol, use_gpu=False)
    corrected_energy = engine.run_scf(corrected)
    correction = float(corrected.get_dispersion())

    assert corrected.disp == dispersion
    assert correction < 0
    assert corrected_energy - baseline_energy == pytest.approx(correction, abs=1e-9)


@needs_pyscf
def test_runner_records_selected_correction(cpu_spec, available_backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYSCF_TMPDIR", str(tmp_path))
    cpu_spec["dispersion"] = "d3bj"
    job_path = tmp_path / "corrected.job.json"
    job_path.write_text(json.dumps(cpu_spec))

    assert runner.run_job(str(job_path)) == 0
    result = json.loads((tmp_path / "corrected.result.json").read_text())
    assert result["status"] == "completed"
    assert result["dispersion"] == "d3bj"
    assert result["xc"] == "B3LYP"


@needs_pyscf
@pytest.mark.parametrize("dispersion,expected_exit", [("", 0), ("d3bj", 1)])
def test_no_backend_allows_plain_dft_but_never_silently_drops_correction(
        cpu_spec, tmp_path, monkeypatch, dispersion, expected_exit):
    # Load PySCF before simulating an absent optional extension, as it would be in
    # a Windows install where its own optional-import guard has already run.
    from pyscf import dft  # noqa: F401

    _hide_backend(monkeypatch)
    monkeypatch.setenv("PYSCF_TMPDIR", str(tmp_path))
    cpu_spec["dispersion"] = dispersion
    job_path = tmp_path / "optional.job.json"
    job_path.write_text(json.dumps(cpu_spec))

    assert runner.run_job(str(job_path)) == expected_exit
    result = json.loads((tmp_path / "optional.result.json").read_text())
    assert result["dispersion"] == (dispersion or None)
    if dispersion:
        assert result["status"] == "failed"
        assert "pyscf-dispersion" in result["error"]
        assert "energy_hartree" not in result
    else:
        assert result["status"] == "completed"
        assert result["energy_hartree"] < 0
