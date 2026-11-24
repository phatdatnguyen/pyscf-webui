"""The runner and file viewers use UTF-8 even on legacy Windows code pages."""
import builtins
import subprocess
import sys

import pytest

from pyscf_webui import calculation, result, working_directory
from pyscf_webui.utils import write_json


def test_calculation_log_uses_utf8_despite_parent_stdio_encoding(tmp_path, monkeypatch):
    directory = tmp_path / "tính toán"
    directory.mkdir()
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setattr(calculation, "mol_from_structure_file", lambda path: object())
    monkeypatch.setattr(calculation, "pyscf_atom_spec", lambda mol: "H 0 0 0; H 0 0 0.74")
    monkeypatch.setattr(calculation, "_POLL_INTERVAL", 0.01)
    popen = subprocess.Popen
    message = "Calculation in tính toán: ΔE = 0.01"

    def launch(command, **kwargs):
        return popen([sys.executable, "-c", f"print({message!r})"], **kwargs)

    monkeypatch.setattr(calculation.subprocess, "Popen", launch)
    updates = list(calculation.on_run_calculation(
        working_directory_path=str(directory), structure_file_name="input.xyz",
        calculation_type="Single-Point", use_mm=False, force_field="MMFF", max_iters=200,
        method_type="HF", functional="B3LYP", basis_set="sto-3g", reference="RHF",
        charge=0, multiplicity=1, geom_maxiter=50, g_convergence="GAU",
        temperature=298.15, pressure=1, n_states=1, tda=False, root=1,
        use_solvation=False, solvent_model="SMD", solvent="Water", n_threads=1,
        memory_gb=1, use_gpu=False, density_fit=False, dispersion="", grid_level=3,
        scf_conv_tol=1e-9, compute_j_couplings=False, j_coupling_nuclei="H", job_name="test",
    ))
    assert "Calculation finished" in updates[-1][0]
    assert message in updates[-1][1]
    assert message in (directory / "test.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("output", ["print('Đã tạo cube')", "import sys; sys.stdout.buffer.write(b'\\xff')"])
def test_cubegen_output_cannot_break_viewer_decoding(tmp_path, water_result, monkeypatch, output):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("PYTHONUTF8", "0")
    write_json(str(tmp_path / "sp.result.json"), water_result)
    loaded = result.on_load_result_file(str(tmp_path), "sp.result.json")[1]
    run = subprocess.run

    def run_cube(command, **kwargs):
        return run([sys.executable, "-c", output], **kwargs)

    monkeypatch.setattr(result.subprocess, "run", run_cube)
    monkeypatch.setattr(result, "render_cube_html", lambda *a, **k: "rendered cube")
    assert result.on_visualize(str(tmp_path), "sp.result.json", loaded, "MO 1",
                               "blue", "red", 0.8, 0.02, 0.2) == "rendered cube"


def test_text_viewer_preserves_utf8_on_a_legacy_locale(tmp_path, monkeypatch):
    def locale_open(path, mode, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        return builtins.open(path, mode, **kwargs)

    monkeypatch.setattr(working_directory, "open", locale_open, raising=False)
    monkeypatch.setattr(working_directory.gr, "Warning", lambda message: None)
    path = tmp_path / "calculation.log"
    original, edited = "Tính toán: ΔE = 0.01", "Tính toán: ΔE = 0.02"
    path.write_text(original, encoding="utf-8")
    viewer, _, loaded_path = working_directory.on_view_text_file(str(tmp_path), path.name)
    assert viewer["value"] == original
    working_directory.on_save_text_file(str(tmp_path), loaded_path, edited)
    assert path.read_text(encoding="utf-8") == edited
