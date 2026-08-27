"""The per-calculation-type menus.

These sweep every choice the UI offers against the engine that has to accept it. They are
seconds-fast and they catch the failure mode that is otherwise only found by a user, part
way into a job: an entry in a dropdown that the engine refuses.
"""
import pytest

from pyscf_webui.engine import SOLVENT_DB
from pyscf_webui.utils import (
    CALCULATION_PROFILES,
    CALCULATION_TYPES,
    EMISSION,
    EMISSION_METHOD_TYPE,
    NMR,
    NMR_FUNCTIONALS,
    calculation_profile,
    method_uses_functional,
    resolve_choice,
)

from conftest import needs_gpu, needs_pyscf

ALL_FUNCTIONALS = sorted({f for p in CALCULATION_PROFILES.values() for f in p.functionals})
ALL_BASIS_SETS = sorted({b for p in CALCULATION_PROFILES.values() for b in p.basis_sets})


class TestProfileIntegrity:
    def test_every_calculation_type_has_a_profile(self):
        for calculation_type in CALCULATION_TYPES:
            assert calculation_type in CALCULATION_PROFILES, calculation_type

    def test_every_default_is_in_its_own_menu(self):
        """A default that is not on offer would leave the dropdown showing a value its
        own choices do not contain."""
        for calculation_type, profile in CALCULATION_PROFILES.items():
            assert profile.default_functional in profile.functionals, calculation_type
            assert profile.default_basis in profile.basis_sets, calculation_type
            assert profile.default_method_type in profile.method_types, calculation_type

    def test_no_menu_is_empty(self):
        for calculation_type, profile in CALCULATION_PROFILES.items():
            assert profile.functionals, calculation_type
            assert profile.basis_sets, calculation_type
            assert profile.method_types, calculation_type


class TestResolveChoice:
    def test_a_still_valid_choice_survives_a_type_switch(self):
        profile = calculation_profile(NMR)
        assert resolve_choice("B3LYP", profile.functionals, profile.default_functional) == "B3LYP"

    def test_an_invalid_choice_falls_back_to_the_type_default(self):
        """CAM-B3LYP is fine for TD-DFT and impossible for NMR, which is exactly the
        case this exists for: without the fallback it would reach the engine and be
        refused there."""
        profile = calculation_profile(NMR)
        assert resolve_choice("CAM-B3LYP", profile.functionals,
                              profile.default_functional) == profile.default_functional

    def test_a_custom_value_is_not_silently_kept(self):
        profile = calculation_profile(NMR)
        assert resolve_choice("not-a-functional", profile.functionals,
                              profile.default_functional) == profile.default_functional


class TestNmrRestrictions:
    """NMR is the most constrained calculation type, and the constraints are checkable."""

    @needs_pyscf
    def test_no_nmr_functional_is_range_separated(self):
        """gpu4pyscf's shielding code asserts ``omega == 0.0``, so a range-separated entry
        here would abort every job that selected it. Asserted rather than trusted, because
        the menu is hand-written and this is invisible on inspection."""
        from pyscf.dft import libxc

        offenders = [f for f in NMR_FUNCTIONALS if libxc.rsh_coeff(f)[0] != 0.0]
        assert not offenders, f"range-separated functionals offered for NMR: {offenders}"

    def test_nmr_offers_only_dft(self):
        # HF has no ``_numint`` for the shielding code to read; the correlated methods have
        # no shielding implementation at all.
        assert calculation_profile(NMR).method_types == ["DFT"]

    @needs_pyscf
    def test_a_range_separated_functional_is_refused_with_a_reason(self):
        from pyscf_webui.engine import nmr_unavailable_reason

        spec = {"method_type": "DFT", "reference": "RHF", "functional": "CAM-B3LYP"}
        reason = nmr_unavailable_reason(spec, use_gpu=True)
        assert reason and "range-separated" in reason

    @needs_pyscf
    def test_a_supported_functional_is_allowed(self):
        from pyscf_webui.engine import nmr_unavailable_reason

        spec = {"method_type": "DFT", "reference": "RHF", "functional": "B3LYP"}
        assert nmr_unavailable_reason(spec, use_gpu=True) is None

    @pytest.mark.parametrize("spec,fragment", [
        ({"method_type": "HF", "reference": "RHF", "functional": "B3LYP"}, "DFT"),
        ({"method_type": "DFT", "reference": "UHF", "functional": "B3LYP"}, "restricted"),
    ])
    def test_unsupported_combinations_are_refused(self, spec, fragment):
        from pyscf_webui.engine import nmr_unavailable_reason

        reason = nmr_unavailable_reason(spec, use_gpu=True)
        assert reason and fragment in reason

    def test_cpu_is_refused_because_there_is_no_fallback(self):
        from pyscf_webui.engine import nmr_unavailable_reason

        spec = {"method_type": "DFT", "reference": "RHF", "functional": "B3LYP"}
        reason = nmr_unavailable_reason(spec, use_gpu=False)
        assert reason and "GPU" in reason


@needs_pyscf
def test_every_functional_parses():
    from pyscf.dft import libxc

    for functional in ALL_FUNCTIONALS:
        libxc.parse_xc(functional)  # raises if libxc does not know the name


@needs_pyscf
def test_no_functional_carries_a_dispersion_suffix():
    """A "-D3BJ" entry in a menu would run without dispersion and say nothing."""
    from pyscf_webui.engine import split_dispersion

    for functional in ALL_FUNCTIONALS:
        assert split_dispersion(functional)[1] is None, functional


@needs_pyscf
@pytest.mark.parametrize("element", ["H", "C", "N", "O", "Si"])
def test_every_basis_set_loads(element):
    """Every offered basis must cover the common elements, since one basis is applied to
    the whole molecule. This is what excluded cc-pCVTZ (no hydrogen) and 6-311+G(2d,p)
    (no third row) from the NMR menu."""
    from pyscf.gto.basis import load

    for basis in ALL_BASIS_SETS:
        assert load(basis, element), f"{basis} has no {element}"


@needs_pyscf
def test_ui_solvent_labels_all_have_engine_entries():
    from pyscf_webui.calculation import SOLVENTS

    missing = [s for s in SOLVENTS if s not in SOLVENT_DB]
    assert not missing, f"solvents offered in the UI with no engine mapping: {missing}"


@needs_gpu
def test_smd_keys_and_dielectrics_match_the_engine_database():
    """The two halves of SOLVENT_DB must both refer to the same real liquid.

    The SMD key and the dielectric constant are transcribed by hand, so this pins them to
    the source of truth. A drifted eps would make the PCM and SMD results for "the same"
    solvent quietly disagree.
    """
    from gpu4pyscf.solvent.smd import solvent_db

    for label, (eps, smd_name) in SOLVENT_DB.items():
        assert smd_name in solvent_db, f"{label}: unknown SMD solvent {smd_name!r}"
        assert eps == pytest.approx(solvent_db[smd_name][5], rel=1e-3), label


class TestFunctionalVisibility:
    """Every method type that consumes a functional must also *show* the control.

    Emission is the case that broke: it runs TD-DFT and its whole result depends on the
    functional, but its method type is spelled "TD-DFT", so a bare ``== "DFT"`` check hid
    the dropdown and left the user unable to see or change it.
    """

    @pytest.mark.parametrize("method_type", ["DFT", EMISSION_METHOD_TYPE])
    def test_functional_families_show_the_control(self, method_type):
        from pyscf_webui.calculation import on_method_type_change

        assert method_uses_functional(method_type)
        assert on_method_type_change(method_type)["visible"] is True

    @pytest.mark.parametrize("method_type", ["HF", "MP2", "CCSD", "CCSD(T)"])
    def test_other_families_hide_it(self, method_type):
        from pyscf_webui.calculation import on_method_type_change

        assert not method_uses_functional(method_type)
        assert on_method_type_change(method_type)["visible"] is False

    def test_every_profile_offering_a_functional_menu_can_show_it(self):
        """A profile whose default method type hides the functional control would offer a
        functional menu the user cannot reach."""
        for calculation_type, profile in CALCULATION_PROFILES.items():
            if len(profile.functionals) > 1:
                assert method_uses_functional(profile.default_method_type), calculation_type

    def test_emission_is_locked_to_its_one_method_type(self):
        assert calculation_profile(EMISSION).method_types == [EMISSION_METHOD_TYPE]
