"""The Result tab's contract with the result file.

The tab returns a flat positional tuple that must stay index-aligned with the component
list it feeds. Nothing but a test catches a drift there: adding a component and forgetting
a return value shifts every later value into the wrong widget, and the page still renders.
"""

from pyscf_webui import result as result_tab


def test_output_count_matches_the_blank_tuple():
    """``_blank_outputs`` is the failure path; it must have the same shape as the
    success path or an error would itself error."""
    assert len(result_tab._blank_outputs("x")) == result_tab.RESULT_OUTPUT_COUNT


def test_loader_returns_the_declared_number_of_outputs(tmp_path, water_result):
    from pyscf_webui.utils import write_json

    write_json(str(tmp_path / "sp.result.json"), water_result)
    outputs = result_tab.on_load_result_file(str(tmp_path), "sp.result.json")
    assert len(outputs) == result_tab.RESULT_OUTPUT_COUNT


def test_failed_result_is_explained_not_hidden(tmp_path, water_result):
    from pyscf_webui.utils import write_json

    water_result.update(status="failed", error="RuntimeError: SCF did not converge")
    write_json(str(tmp_path / "bad.result.json"), water_result)
    outputs = result_tab.on_load_result_file(str(tmp_path), "bad.result.json")
    assert len(outputs) == result_tab.RESULT_OUTPUT_COUNT
    assert "SCF did not converge" in str(outputs[0])


def test_version_1_psi4_result_still_renders(tmp_path, water_result):
    """A working directory copied over from the Psi4 app must not break the tab."""
    from pyscf_webui.utils import write_json

    legacy = dict(water_result)
    legacy["version"] = 1
    legacy["calculation_type"] = "Emission (EOM-CCSD)"
    legacy["wavefunction_file"] = "sp.npy"
    for key in ("engine", "device", "xc"):
        legacy.pop(key, None)
    write_json(str(tmp_path / "old.result.json"), legacy)
    outputs = result_tab.on_load_result_file(str(tmp_path), "old.result.json")
    assert len(outputs) == result_tab.RESULT_OUTPUT_COUNT
    assert "red" not in str(outputs[0])


class TestFrequencyTable:
    def test_includes_intensities_when_present(self):
        frame = result_tab._frequency_dataframe({
            "frequencies_cm1": [1700.0, 3700.0],
            "ir_intensities_km_mol": [80.0, 2.0],
        })
        assert list(frame.columns) == ["Frequency (cm-1)", "IR intensity (km/mol)"]

    def test_drops_the_column_rather_than_filling_zeros(self):
        """Zeros would read as "IR-inactive", a claim about the molecule rather than
        about the calculation. Open-shell and CPU runs have no intensities at all."""
        frame = result_tab._frequency_dataframe({
            "frequencies_cm1": [1700.0, 3700.0],
            "ir_intensities_km_mol": None,
        })
        assert list(frame.columns) == ["Frequency (cm-1)"]
        assert len(frame) == 2


class TestVisualizationChoices:
    def test_spin_density_hidden_for_a_closed_shell_result(self, water_result):
        from pyscf_webui.visualization import SPIN_DENSITY

        assert SPIN_DENSITY not in result_tab._visualization_choices(water_result)

    def test_spin_density_offered_for_an_open_shell_result(self, water_result):
        from pyscf_webui.visualization import SPIN_DENSITY

        water_result["n_alpha"], water_result["n_beta"] = 5, 4
        assert SPIN_DENSITY in result_tab._visualization_choices(water_result)

    def test_one_orbital_choice_per_molecular_orbital(self, water_result):
        choices = result_tab._visualization_choices(water_result)
        orbitals = [c for c in choices if c.startswith("MO ")]
        assert len(orbitals) == len(water_result["mo_energies_hartree"])


class TestEmissionTable:
    def test_reports_the_stokes_shift_when_available(self):
        frame = result_tab._emission_dataframe({
            "root": 1, "emission_energy_ev": 3.1, "wavelength_nm": 397.7,
            "absorption_wavelength_nm": 319.0, "stokes_shift_cm1": 6209.0,
            "oscillator_strength": 0.0,
            "excited_energy_hartree": -114.0, "ground_energy_at_excited_geometry_hartree": -114.1,
        })
        assert "Stokes shift (cm-1)" in frame.columns
        assert "Oscillator strength" in frame.columns

    def test_omits_optional_columns_for_an_older_result(self):
        frame = result_tab._emission_dataframe({
            "root": 1, "emission_energy_ev": 3.1, "wavelength_nm": 397.7,
            "excited_energy_hartree": -114.0, "ground_energy_at_excited_geometry_hartree": -114.1,
        })
        assert "Stokes shift (cm-1)" not in frame.columns
