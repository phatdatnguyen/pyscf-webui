"""Reject invalid output destinations before running conformer generation."""
import pytest
from rdkit import Chem

from pyscf_webui import conformer_generation as conformers


def generate(directory, name="conformer", file_type="xyz"):
    return conformers.on_generate_conformers(
        directory, "CCO", 0, 1, 1, 0.1, 0.5, name, file_type,
    )


@pytest.mark.parametrize("name", ["", "../outside", "/tmp/outside", "nested/name"])
def test_conformer_prefix_cannot_escape_working_directory(tmp_path, monkeypatch, name):
    def must_not_embed(*args):
        pytest.fail("An invalid output name must be rejected before embedding")

    monkeypatch.setattr(conformers, "generate_unique_conformers", must_not_embed)
    status, files, table = generate(str(tmp_path), name)
    assert "color:red" in status
    assert "file name" in status
    assert files == []
    assert table.empty


def test_conformer_output_does_not_follow_external_symlink(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    outside = tmp_path / "outside.xyz"
    outside.write_text("keep me")
    (working / "conformer_1.xyz").symlink_to(outside)
    status, _, table = generate(str(working))
    assert "outside the working directory" in status
    assert table.empty
    assert outside.read_text() == "keep me"


def test_generation_requires_open_directory():
    status, files, table = generate(None)
    assert "open a working directory" in status
    assert files == []
    assert table.empty


def test_empty_sketch_clears_smiles():
    assert conformers.on_smiles_from_editor({"smiles": "", "load_id": None}) == ""


def test_only_converged_finite_energies_are_accepted(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    monkeypatch.setattr(conformers.AllChem, "MMFFGetMoleculeProperties", lambda mol: object())
    monkeypatch.setattr(
        conformers.AllChem, "MMFFOptimizeMoleculeConfs",
        lambda *args, **kwargs: [(0, -2.0), (1, -9.0), (-1, -10.0), (0, float("nan"))],
    )
    assert conformers.optimize_conformers(mol) == [-2.0, None, None, None]


def test_failed_minimization_never_writes_a_conformer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        conformers, "optimize_conformers",
        lambda mol: [None] * mol.GetNumConformers(),
    )
    status, files, table = generate(str(tmp_path))
    assert "No converged conformers" in status
    assert "color:red" in status
    assert files == []
    assert table.empty


def test_missing_force_field_parameters_report_error(tmp_path):
    status, files, table = conformers.on_generate_conformers(
        str(tmp_path), "[U]", 0, 1, 1, 0.1, 0.5, "uranium", "xyz",
    )
    assert "parameters for every atom" in status
    assert files == []
    assert table.empty
