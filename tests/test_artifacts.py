"""The files the runner generates.

PySCF produces no output files of its own, so everything the Result tab and external
programs consume is written by this app. These tests cover the writers and the one reader
that has to agree with them.
"""
import os

import numpy as np
import pytest

from pyscf_webui import engine
from pyscf_webui.visualization import (
    ELECTROSTATIC_POTENTIAL,
    SPIN_DENSITY,
    TOTAL_DENSITY,
    cube_file_name,
    cube_files_for_selection,
)

from conftest import needs_pyscf


class TestCubeNaming:
    """The runner and the viewer must agree on every file name, exactly."""

    @pytest.mark.parametrize("task,expected", [
        ("density", "density.cube"),
        ("spin", "spin_density.cube"),
        ("esp", "esp.cube"),
    ])
    def test_task_names(self, task, expected):
        assert cube_file_name(task) == expected

    def test_orbital_names_are_zero_padded(self):
        # Unpadded names sort MO 10 before MO 2 in a directory listing.
        assert cube_file_name("orbital", 5) == "mo_0005.cube"
        assert cube_file_name("orbital", 12) == "mo_0012.cube"

    def test_lookup_round_trips_through_the_writer(self, tmp_path):
        """The producer/consumer contract: whatever the runner writes, the viewer finds."""
        for selection, task, orbital in [
            (TOTAL_DENSITY, "density", None),
            (SPIN_DENSITY, "spin", None),
            (ELECTROSTATIC_POTENTIAL, "esp", None),
            ("MO 7", "orbital", 7),
        ]:
            path = tmp_path / cube_file_name(task, orbital)
            path.write_text("cube")
            found = cube_files_for_selection(str(tmp_path), selection)
            assert found == [str(path)], selection

    def test_missing_file_yields_no_candidates(self, tmp_path):
        assert cube_files_for_selection(str(tmp_path), TOTAL_DENSITY) == []


class TestDensityMatrices:
    def test_restricted_total_density_traces_to_electron_count(self):
        # Two doubly-occupied orbitals in an orthonormal basis -> 4 electrons.
        mo_coeff = np.eye(4)
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        dm_a, dm_b = engine.density_matrices(mo_coeff, mo_occ)
        assert np.trace(dm_a + dm_b) == pytest.approx(4.0)

    def test_restricted_spin_density_is_zero(self):
        dm_a, dm_b = engine.density_matrices(np.eye(4), np.array([2.0, 2.0, 0.0, 0.0]))
        assert np.allclose(dm_a - dm_b, 0.0)

    def test_unrestricted_spin_density_traces_to_the_spin_excess(self):
        mo_coeff = np.stack([np.eye(4), np.eye(4)])
        mo_occ = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        dm_a, dm_b = engine.density_matrices(mo_coeff, mo_occ)
        assert np.trace(dm_a + dm_b) == pytest.approx(5.0)
        assert np.trace(dm_a - dm_b) == pytest.approx(1.0)


class TestFrequencyMolden:
    """PySCF has no writer for this format, so the output is checked structurally."""

    @needs_pyscf
    def test_sections_and_counts(self, tmp_path):
        from pyscf import gto

        mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467",
                    basis="sto-3g", verbose=0)
        arrays = {
            "freq_wavenumber": np.array([1700.0, 3700.0, 3800.0]),
            "norm_mode": np.zeros((3, 3, 3)),
            "ir_intensity": np.array([80.0, 2.0, 20.0]),
        }
        path = tmp_path / "f.freq.molden"
        engine.save_frequency_molden(mol, arrays, str(path))
        text = path.read_text()

        for section in ("[Molden Format]", "[FREQ]", "[FR-COORD]",
                        "[FR-NORM-COORD]", "[INT]"):
            assert section in text, section
        assert text.count("vibration ") == 3
        # [FR-COORD] is in Bohr, not the angstrom used everywhere else in this app.
        coord_line = text.split("[FR-COORD]")[1].splitlines()[1].split()
        assert float(coord_line[3]) == pytest.approx(0.117 / 0.52917721092, rel=1e-4)

    @needs_pyscf
    def test_intensity_section_omitted_when_unavailable(self, tmp_path):
        from pyscf import gto

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        arrays = {"freq_wavenumber": np.array([4400.0]), "norm_mode": np.zeros((1, 2, 3))}
        path = tmp_path / "f.freq.molden"
        engine.save_frequency_molden(mol, arrays, str(path))
        assert "[INT]" not in path.read_text()


class TestCheckpointRoundTrip:
    @needs_pyscf
    def test_chkfile_reloads_without_rerunning_the_scf(self, tmp_path):
        """This is what lets the orbital viewer work after the fact."""
        from pyscf import gto, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        path = str(tmp_path / "job.chk")
        engine.save_chkfile(mol, mf, path)
        loaded_mol, mo_coeff, mo_occ, mo_energy = engine.load_chkfile(path)

        assert loaded_mol.natm == mol.natm
        assert loaded_mol.nao == mol.nao
        assert np.allclose(mo_coeff, mf.mo_coeff)
        assert np.allclose(mo_occ, mf.mo_occ)
        assert np.allclose(mo_energy, mf.mo_energy)

    @needs_pyscf
    def test_molden_written_for_a_restricted_wavefunction(self, tmp_path):
        from pyscf import gto, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        path = str(tmp_path / "job.molden")
        engine.save_molden(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy, path)
        text = open(path).read()
        assert "[Molden Format]" in text
        assert "[MO]" in text

    @needs_pyscf
    def test_molden_writes_both_spins_for_an_unrestricted_wavefunction(self, tmp_path):
        """pyscf.tools.molden.dump_scf misses this case for gpu4pyscf UKS objects -- its
        isinstance check is against pyscf's own UHF -- and silently writes alpha only.
        save_molden branches on the array shape instead, so both blocks are always there."""
        from pyscf import gto, scf

        mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
        mf = scf.UHF(mol)
        mf.kernel()
        path = str(tmp_path / "job.molden")
        engine.save_molden(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy, path)
        text = open(path).read()
        assert "Alpha" in text
        assert "Beta" in text


class TestArraySidecar:
    def test_writes_each_array_as_a_dataset(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = str(tmp_path / "job.arrays.h5")
        engine.save_arrays({"hessian": np.zeros((2, 2, 3, 3)),
                            "freq_wavenumber": np.array([1.0, 2.0])}, path)
        with h5py.File(path) as handle:
            assert handle["hessian"].shape == (2, 2, 3, 3)
            assert handle["freq_wavenumber"].shape == (2,)

    def test_no_file_when_there_is_nothing_to_store(self, tmp_path):
        path = str(tmp_path / "job.arrays.h5")
        assert engine.save_arrays({}, path) is None
        assert not os.path.exists(path)
