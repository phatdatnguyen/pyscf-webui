"""Coordinate-file and spectrum regressions that do not require a chemistry engine."""
import numpy as np
import pytest

from pyscf_webui.utils import (
    _multiplet_lines,
    generate_nmr_spectrum_interactive,
    get_files_in_working_directory,
    mol_from_xyz_file,
)


def test_incomplete_xyz_trajectory_cannot_mix_frames(tmp_path):
    path = tmp_path / "trajectory.xyz"
    path.write_text("2\ncomplete frame\nH 0 0 0\nH 0 0 0.7\n2\nincomplete frame\nH 1 0 0\n")
    with pytest.raises(ValueError, match="Incomplete or malformed XYZ frame"):
        mol_from_xyz_file(str(path))


def test_xyz_reader_uses_final_frames_own_atom_count(tmp_path):
    path = tmp_path / "frames.xyz"
    path.write_text("2\nfirst frame\nH 0 0 0\nH 0 0 0.7\n1\nC 10 20 30\nHe 1 2 3\n")
    molecule = mol_from_xyz_file(str(path))
    assert molecule.GetNumAtoms() == 1
    assert molecule.GetAtomWithIdx(0).GetSymbol() == "He"
    assert np.allclose(molecule.GetConformer().GetPositions(), [[1, 2, 3]])


def test_xyz_reader_preserves_app_charge_and_multiplicity_header(tmp_path):
    path = tmp_path / "charged.xyz"
    path.write_text("-1 2\nO 0 0 0\nH 0 0 0.97\n")
    molecule, charge, multiplicity = mol_from_xyz_file(str(path), True)
    assert (charge, multiplicity) == (-1, 2)
    assert molecule.GetNumAtoms() == 2


def test_equivalent_nmr_partners_use_binomial_lines_without_duplicates():
    peak = {"shift": 5.0, "couplings": [{"n": 16, "j_hz": 10.0}]}
    lines = _multiplet_lines(peak, 100.0)
    assert len(lines) == 17
    assert sum(weight for _, weight in lines) == pytest.approx(1.0)
    assert sum(shift * weight for shift, weight in lines) == pytest.approx(5.0)
    assert lines[0] == pytest.approx((4.2, 1 / 65536))
    assert lines[8] == pytest.approx((5.0, 12870 / 65536))


@pytest.mark.parametrize("frequency", [0.0, -400.0, float("nan"), float("inf")])
def test_nmr_rejects_invalid_spectrometer_frequencies(frequency):
    with pytest.raises(ValueError, match="positive finite"):
        generate_nmr_spectrum_interactive([{"shift": 5.0}], "H", spectrometer_mhz=frequency)


def test_file_list_can_refresh_after_working_directory_was_removed(tmp_path):
    directory = tmp_path / "removed"
    directory.mkdir()
    directory.rmdir()
    assert get_files_in_working_directory(str(directory)) == []
    directory.write_text("a file replaced the directory")
    assert get_files_in_working_directory(str(directory)) == []
