"""Regressions for shared-directory runs and native-process cleanup."""
import inspect
import subprocess
import sys

import pytest

from pyscf_webui import calculation


@pytest.fixture
def run_job(tmp_path, monkeypatch):
    values = dict(
        working_directory_path=str(tmp_path), structure_file_name="input.xyz",
        calculation_type="Single-Point", use_mm=False, force_field="MMFF", max_iters=200,
        method_type="HF", functional="B3LYP", basis_set="sto-3g", reference="RHF",
        charge=0, multiplicity=1, geom_maxiter=50, g_convergence="GAU",
        temperature=298.15, pressure=1, n_states=1, tda=False, root=1,
        use_solvation=False, solvent_model="SMD", solvent="Water", n_threads=1,
        memory_gb=1, use_gpu=False, density_fit=False, dispersion="", grid_level=3,
        scf_conv_tol=1e-9, compute_j_couplings=False, j_coupling_nuclei="1H",
        job_name="test",
    )
    monkeypatch.setattr(calculation, "mol_from_structure_file", lambda path: object())
    monkeypatch.setattr(calculation, "pyscf_atom_spec", lambda mol: "H 0 0 0; H 0 0 0.74")
    monkeypatch.setattr(calculation, "_POLL_INTERVAL", 0)

    def run(**overrides):
        return calculation.on_run_calculation(**(values | overrides))

    yield run
    assert not calculation._processes
    assert not calculation._active_directories
    assert not calculation._stopped


class Process:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture
def process(monkeypatch):
    process = Process()
    monkeypatch.setattr(calculation.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(calculation, "_terminate_process_tree",
                        lambda child: setattr(child, "returncode", -15))
    return process


def test_duplicate_run_preserves_active_job_and_log(run_job, process, tmp_path):
    first = run_job()
    try:
        next(first)
        job_before = (tmp_path / "test.job.json").read_bytes()
        (tmp_path / "test.log").write_text("SCF still running\n")
        # A path alias must contend for the same directory reservation.
        duplicate = list(run_job(working_directory_path=str(tmp_path / "alias" / ".."),
                                 functional="PBE", method_type="DFT"))
        assert "already running" in duplicate[-1][0]
        assert (tmp_path / "test.job.json").read_bytes() == job_before
        assert (tmp_path / "test.log").read_text() == "SCF still running\n"
        assert not (tmp_path / "test.log.bak").exists()
        assert calculation._lock.acquire(blocking=False)
        calculation._lock.release()
    finally:
        first.close()


def test_directory_is_reserved_during_preparation(run_job, process, monkeypatch):
    duplicate_status = []

    def load(path):
        duplicate_status.extend(list(run_job(job_name="competing")))
        return object()

    monkeypatch.setattr(calculation, "mol_from_structure_file", load)
    first = run_job()
    try:
        next(first)
        assert "already running" in duplicate_status[-1][0]
    finally:
        first.close()


def test_different_directories_can_run_together(run_job, process, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    first, second = run_job(), run_job(working_directory_path=str(other))
    try:
        assert "Running" in next(first)[0]
        assert "Running" in next(second)[0]
        assert len(calculation._processes) == 2
    finally:
        first.close()
        second.close()


def test_failed_preparation_releases_directory(run_job, process, monkeypatch):
    def fail(path):
        raise ValueError("invalid structure")

    monkeypatch.setattr(calculation, "mol_from_structure_file", fail)
    assert "invalid structure" in list(run_job())[-1][0]
    assert "invalid structure" in list(run_job())[-1][0]


def test_stop_during_preparation_prevents_launch(run_job, tmp_path, monkeypatch):
    def load(path):
        assert "Stopping" in calculation.on_stop_calculation(str(tmp_path))
        return object()

    monkeypatch.setattr(calculation, "mol_from_structure_file", load)
    monkeypatch.setattr(calculation.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("launched a cancelled calculation"))
    assert "stopped before launch" in list(run_job())[-1][0]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("force_field", ["MMFF", "UFF"])
@pytest.mark.parametrize("parameters_available,status", [(False, 0), (True, 1)])
def test_failed_mm_cleanup_cannot_launch_qm(run_job, monkeypatch, force_field,
                                          parameters_available, status):
    monkeypatch.setattr(calculation.AllChem, force_field + "HasAllMoleculeParams",
                        lambda mol: parameters_available)
    monkeypatch.setattr(calculation.AllChem, force_field + "OptimizeMolecule",
                        lambda *a, **k: status)
    monkeypatch.setattr(calculation.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("launched despite failed MM cleanup"))
    message = list(run_job(use_mm=True, force_field=force_field))[-1][0]
    assert "Could not start" in message
    assert "pre-optimization" in message


@pytest.mark.parametrize("overrides", [
    {"job_name": "../outside"}, {"job_name": "/tmp/outside"},
    {"job_name": "nested\\outside"}, {"structure_file_name": "../outside.xyz"},
])
def test_paths_cannot_escape_working_directory(run_job, tmp_path, overrides):
    assert "Could not start" in list(run_job(**overrides))[-1][0]
    assert list(tmp_path.iterdir()) == []


def test_output_symlink_cannot_overwrite_external_file(run_job, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside.json")
    outside.write_text("keep me")
    (tmp_path / "test.job.json").symlink_to(outside)
    assert "Could not start" in list(run_job())[-1][0]
    assert outside.read_text() == "keep me"


def test_rerun_archives_old_result_before_child_can_be_stopped(run_job, process, tmp_path):
    (tmp_path / "test.result.json").write_text('{"status": "completed"}')
    first = run_job()
    try:
        next(first)
        assert not (tmp_path / "test.result.json").exists()
        assert (tmp_path / "test.result.json.bak").read_text() == '{"status": "completed"}'
    finally:
        first.close()


def test_closing_stream_terminates_real_child(run_job, monkeypatch):
    popen = subprocess.Popen
    children = []

    def start(*args, **kwargs):
        child = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(calculation.subprocess, "Popen", start)
    stream = run_job()
    try:
        next(stream)
        assert children[0].poll() is None
        stream.close()
        assert children[0].poll() is not None
    finally:
        stream.close()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_stop_uses_same_directory_key(run_job, process, tmp_path):
    stream = run_job()
    try:
        next(stream)
        assert "Stopping" in calculation.on_stop_calculation(str(tmp_path / "alias" / ".."))
        assert process.poll() is not None
        assert "stopped" in list(stream)[-1][0]
    finally:
        stream.close()


def test_run_event_allows_independent_directories():
    import gradio as gr

    with gr.Blocks() as blocks:
        calculation.calculation_tab_content(gr.State(), gr.State(), gr.Markdown())
    handlers = [fn for fn in blocks.fns.values() if fn.fn is calculation.on_run_calculation]
    assert len(handlers) == 1
    assert handlers[0].concurrency_limit is None
    # Preserve Gradio's positional handler contract after factoring process ownership.
    assert list(inspect.signature(calculation.on_run_calculation).parameters) == list(
        inspect.signature(calculation._run_reserved_calculation).parameters)
