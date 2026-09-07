"""Regression coverage for scientific result integrity and optional GPU properties."""
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pyscf_webui import engine, runner

from conftest import needs_pyscf


def test_restricted_open_shell_density_preserves_unpaired_electrons():
    coefficients, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(5, 5)))
    occupations = np.array([2., 2., 1., 1., 0.])
    alpha, beta = engine.density_matrices(coefficients, occupations)

    assert np.trace(alpha) == pytest.approx(4)
    assert np.trace(beta) == pytest.approx(2)
    np.testing.assert_allclose(
        alpha - beta, coefficients[:, 2:4] @ coefficients[:, 2:4].T, atol=1e-14
    )
    np.testing.assert_allclose(
        alpha + beta, (coefficients * occupations) @ coefficients.T, atol=1e-14
    )


@pytest.mark.parametrize("unrestricted", [False, True])
def test_complex_orbitals_produce_hermitian_density(unrestricted):
    coefficients = np.array([[1., 1.j], [1.j, 1.]]) / np.sqrt(2)
    occupations = np.array([2., 0.])
    if unrestricted:
        coefficients = np.stack([coefficients, coefficients])
        occupations = np.array([[1., 0.], [1., 0.]])
    alpha, beta = engine.density_matrices(coefficients, occupations)

    np.testing.assert_allclose(alpha, alpha.conj().T)
    np.testing.assert_allclose(beta, beta.conj().T)
    assert np.trace(alpha + beta) == pytest.approx(2.)


def test_unconverged_td_roots_are_rejected():
    td = SimpleNamespace(kernel=Mock(), e=np.array([0.1, 0.2]),
                         converged=np.array([True, False]))
    mf = SimpleNamespace(TDDFT=Mock(return_value=td))

    with pytest.raises(engine.ConvergenceError, match="TD-DFT did not converge"):
        engine.build_tddft(mf, {"n_states": 2})


def test_emission_root_must_exist_in_the_returned_states():
    td = SimpleNamespace(kernel=Mock(), e=np.array([0.1]), converged=np.array([True]))
    mf = SimpleNamespace(TDDFT=Mock(return_value=td))

    with pytest.raises(ValueError, match="root 2.*only 1 states"):
        engine.build_tddft(mf, {"n_states": 2, "root": 2})


@needs_pyscf
def test_unconverged_ccsd_never_runs_triples(monkeypatch):
    from pyscf import cc

    solver = SimpleNamespace(converged=False, e_tot=-1., ccsd_t=Mock(return_value=-0.1))
    constructor = Mock(return_value=SimpleNamespace(run=Mock(return_value=solver)))
    monkeypatch.setattr(cc, "CCSD", constructor)

    with pytest.raises(engine.ConvergenceError, match="CCSD did not converge"):
        engine.correlated_energy({"method_type": "CCSD(T)"}, SimpleNamespace())
    solver.ccsd_t.assert_not_called()


@needs_pyscf
@pytest.mark.parametrize("missing", ["oscillator", "rotatory"])
def test_unavailable_transition_strengths_are_unknown_not_zero(monkeypatch, missing):
    from pyscf.tdscf import rhf

    td = SimpleNamespace(e=np.array([0.1, 0.2]), oscillator_strength=Mock(
        side_effect=RuntimeError("property unavailable") if missing == "oscillator" else None,
        return_value=np.array([0., 0.3]),
    ))
    monkeypatch.setattr(rhf, "transition_dipole", Mock(
        side_effect=RuntimeError("property unavailable") if missing == "rotatory" else None,
        return_value=np.array([[0., 0., 0.], [1., 0., 0.]]),
    ))
    monkeypatch.setattr(rhf, "transition_magnetic_dipole", Mock(
        return_value=np.array([[0., 0., 0.], [1., 0., 0.]]),
    ))

    fragment, arrays = engine.excitation_results(td)

    missing_key = f"{missing}_strength"
    present_key = "rotatory_strength" if missing == "oscillator" else "oscillator_strength"
    assert all(entry[missing_key] is None for entry in fragment["excitations"])
    assert missing_key not in arrays
    assert fragment["excitations"][0][present_key] == 0.
    assert fragment["excitations"][1][present_key] > 0.


@needs_pyscf
@pytest.mark.parametrize("failure_stage", ["hessian", "dipole"])
def test_gpu_ir_failure_keeps_frequencies_and_reuses_completed_hessian(monkeypatch, failure_stage):
    from pyscf.hessian import thermo

    mol = SimpleNamespace(natm=2)
    hessian = np.ones((2, 2, 3, 3))
    gpu_hessian = SimpleNamespace(de=np.empty((0, 0, 3, 3)))

    def compute_hessian():
        if failure_stage == "hessian":
            raise RuntimeError("GPU Hessian unavailable")
        gpu_hessian.de = hessian
        return hessian

    gpu_hessian.kernel = Mock(side_effect=compute_hessian)
    cpu_hessian = SimpleNamespace(kernel=Mock(return_value=hessian))
    cpu_mf = SimpleNamespace(mol=mol, Hessian=Mock(return_value=cpu_hessian))
    mf = SimpleNamespace(mol=mol, Hessian=Mock(return_value=gpu_hessian),
                         to_cpu=Mock(return_value=cpu_mf))

    def compute_ir(mf, hess):
        hess.kernel()
        raise RuntimeError("GPU dipole response unavailable")

    gpu_package = ModuleType("gpu4pyscf")
    properties = ModuleType("gpu4pyscf.properties")
    properties.ir = SimpleNamespace(eval_ir_freq_intensity=compute_ir)
    monkeypatch.setitem(sys.modules, "gpu4pyscf", gpu_package)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.properties", properties)
    monkeypatch.setattr(engine, "_supports_gpu_ir", lambda mf: True)
    monkeypatch.setattr(engine, "_thermochemistry", lambda *args: {})
    harmonic = Mock(return_value={
        "freq_wavenumber": np.array([1000.]), "freq_au": np.array([0.01]),
        "norm_mode": np.zeros((1, 2, 3)), "reduced_mass": np.array([1.]),
    })
    monkeypatch.setattr(thermo, "harmonic_analysis", harmonic)

    fragment, arrays = engine.frequency_analysis(mf, {}, use_gpu=True)

    assert fragment["frequency"]["frequencies_cm1"] == [1000.]
    assert fragment["frequency"]["ir_intensities_km_mol"] is None
    np.testing.assert_array_equal(arrays["hessian"], hessian)
    if failure_stage == "dipole":
        gpu_hessian.kernel.assert_called_once()
        mf.to_cpu.assert_not_called()
    else:
        mf.to_cpu.assert_called_once()
        cpu_hessian.kernel.assert_called_once()


@pytest.fixture
def job(tmp_path, monkeypatch):
    # These tests exercise real CPU SCF through run_job, without creating a CUDA context
    # for the runner's best-effort cleanup or leaving scratch configuration behind.
    monkeypatch.setattr(runner, "_release_gpu_memory", lambda: None)
    monkeypatch.setenv("PYSCF_TMPDIR", str(tmp_path))
    spec = {
        "calculation_type": "Single-Point", "method_type": "HF", "basis_set": "sto-3g",
        "geometry": "H 0 0 0; H 0 0 0.74", "charge": 0, "multiplicity": 1,
        "reference": "RHF", "n_threads": 1, "verbose": 0,
        "save_wavefunction": False, "save_molden": False, "save_arrays": False,
    }

    def run(**updates):
        spec.update(updates)
        path = tmp_path / "test.job.json"
        path.write_text(json.dumps(spec))
        exit_code = runner.run_job(str(path))
        return exit_code, json.loads((tmp_path / "test.result.json").read_text())

    return run


@needs_pyscf
def test_runner_rejects_unconverged_scf(job):
    exit_code, result = job(
        geometry="O 0 0 0; H 0 0 0.96; H 0 0.92 -0.24", scf_max_cycle=1,
    )
    assert exit_code == 1
    assert result["status"] == "failed"
    assert result["converged"] is False
    assert "SCF did not converge" in result["error"]
    assert "energy_hartree" not in result


@needs_pyscf
def test_cpu_job_does_not_initialize_cuda_for_cleanup(job, monkeypatch):
    cleanup = Mock(side_effect=RuntimeError("CUDA unavailable"))
    monkeypatch.setattr(runner, "_release_gpu_memory", cleanup)

    exit_code, result = job()

    assert exit_code == 0
    assert result["status"] == "completed"
    cleanup.assert_not_called()


def test_unknown_calculation_type_is_rejected_before_running_engine(job):
    exit_code, result = job(calculation_type="TD-DFT")
    assert exit_code == 1
    assert result["status"] == "failed"
    assert "Unknown calculation type" in result["error"]


@pytest.mark.parametrize("contents", ["{incomplete", "[]"])
def test_invalid_job_spec_still_produces_a_failed_result(tmp_path, monkeypatch, contents):
    monkeypatch.setattr(runner, "_release_gpu_memory", lambda: None)
    path = tmp_path / "damaged.job.json"
    path.write_text(contents)

    assert runner.run_job(str(path)) == 1
    result = json.loads((tmp_path / "damaged.result.json").read_text())
    assert result["status"] == "failed"
    assert result["error"]


@needs_pyscf
def test_runner_keeps_finished_optimization_when_xyz_writes_fail(job, monkeypatch):
    monkeypatch.setattr(
        runner, "_optimize", lambda mf, spec: (True, mf.mol, [-1.], [[[0., 0., 0.]]])
    )
    monkeypatch.setattr(runner, "_write_xyz", Mock(side_effect=PermissionError("read only")))
    monkeypatch.setattr(
        runner, "_write_trajectory_xyz", Mock(side_effect=PermissionError("read only"))
    )

    exit_code, result = job(calculation_type="Geometry Optimization")

    assert exit_code == 0
    assert result["status"] == "completed"
    assert result["energy_hartree"] == pytest.approx(-1.1167593, abs=1e-6)
    assert "optimized_xyz_file" not in result
    assert "trajectory_xyz_file" not in result


@needs_pyscf
def test_frequency_result_records_unconverged_geometry(job, monkeypatch):
    monkeypatch.setattr(runner, "_optimize", lambda mf, spec: (False, mf.mol, [-1.], []))

    exit_code, result = job(calculation_type="Frequency")

    assert exit_code == 0
    assert result["optimization"]["converged"] is False
    assert result["optimization"]["energies_hartree"] == [-1.]
    assert len(result["frequency"]["frequencies_cm1"]) == 1


@needs_pyscf
def test_geometry_optimizer_rejects_unconverged_electronic_gradients():
    pytest.importorskip("geometric")
    from pyscf import gto, lib, scf

    lib.num_threads(1)
    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0 0.92 -0.24", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.max_cycle = 1

    with pytest.raises(RuntimeError, match="Nuclear gradients.*not converged"):
        runner._optimize(mf, {"geom_maxiter": 1})
