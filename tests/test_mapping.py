"""The engine-neutral mapping layer: the things that are silently wrong when wrong.

Every test here guards a translation that fails *quietly* -- producing a plausible number
from the wrong physics rather than an error. That is the whole reason these are tests and
not review comments.
"""
import pytest

from pyscf_webui import engine
from pyscf_webui.runner import job_base_name
from pyscf_webui.utils import (
    G_CONVERGENCE_KWARGS,
    build_job_spec,
    pyscf_method,
    pyscf_spin,
    result_base_name,
)

from conftest import needs_pyscf


class TestSpin:
    """``mol.spin`` is 2S, but the UI collects a multiplicity (2S+1)."""

    @pytest.mark.parametrize("multiplicity,expected_spin", [(1, 0), (2, 1), (3, 2), (6, 5)])
    def test_multiplicity_becomes_two_s(self, multiplicity, expected_spin):
        assert pyscf_spin(multiplicity) == expected_spin

    def test_singlet_is_not_spin_one(self):
        # The specific mistake this guards: passing the multiplicity straight through
        # would run every singlet as a triplet, converge happily, and report a wrong
        # energy with no error anywhere.
        assert pyscf_spin(1) == 0


class TestDispersion:
    """A dispersion suffix on a functional name is silently ignored by libxc."""

    @pytest.mark.parametrize("label,xc,disp", [
        ("B3LYP-D3BJ", "B3LYP", "d3bj"),
        ("B3LYP-D3ZERO", "B3LYP", "d3zero"),
        ("PBE-D4", "PBE", "d4"),
        ("B3LYP-D3", "B3LYP", "d3"),
        ("B3LYP", "B3LYP", None),
        ("CAM-B3LYP", "CAM-B3LYP", None),
    ])
    def test_suffix_is_split_off(self, label, xc, disp):
        assert engine.split_dispersion(label) == (xc, disp)

    def test_longest_suffix_wins(self):
        # "-D3" is a prefix of "-D3BJ"; checking the short one first would leave "BJ"
        # glued to the functional name and produce an unknown-functional error later.
        assert engine.split_dispersion("B3LYP-D3BJ")[0] == "B3LYP"

    def test_cam_b3lyp_is_not_mistaken_for_a_correction(self):
        # A hyphen in the functional name itself must not be treated as a suffix.
        assert engine.split_dispersion("CAM-B3LYP") == ("CAM-B3LYP", None)

    @needs_pyscf
    def test_libxc_really_does_ignore_the_suffix(self):
        """The premise of the whole split. If libxc ever starts honouring the suffix,
        this test fails and the split becomes redundant rather than load-bearing."""
        from pyscf.dft import libxc

        assert libxc.parse_xc("B3LYP-D3BJ") == libxc.parse_xc("B3LYP")


class TestBaseNames:
    """``os.path.splitext`` mishandles the double suffixes these files use."""

    @pytest.mark.parametrize("name,base", [
        ("opt.job.json", "opt"),
        ("opt.result.json", "opt"),
        ("my.calc.v2.result.json", "my.calc.v2"),
    ])
    def test_result_base_name(self, name, base):
        assert result_base_name(name) == base

    def test_job_base_name_strips_both_suffixes(self):
        # Getting this wrong names every artifact "opt.job.log", "opt.job.chk", ...
        assert job_base_name("/tmp/wd/opt.job.json") == "opt"


class TestJobSpec:
    def test_density_fitting_defaults_on_for_dft_and_off_for_hf(self):
        dft = build_job_spec(**_spec_kwargs(method_type="DFT"))
        hf = build_job_spec(**_spec_kwargs(method_type="HF"))
        assert dft["density_fit"] is True
        assert hf["density_fit"] is False

    def test_explicit_density_fit_overrides_the_default(self):
        spec = build_job_spec(**_spec_kwargs(method_type="DFT", density_fit=False))
        assert spec["density_fit"] is False

    def test_gradio_floats_are_coerced(self):
        # Gradio sliders hand back floats even for integer-valued controls.
        spec = build_job_spec(**_spec_kwargs(charge=-1.0, multiplicity=2.0, n_threads=4.0))
        assert isinstance(spec["charge"], int)
        assert isinstance(spec["multiplicity"], int)
        assert isinstance(spec["n_threads"], int)

    @pytest.mark.parametrize("method_type,expected", [
        ("HF", "scf"), ("DFT", "dft"), ("TD-DFT", "dft"),
        ("MP2", "mp2"), ("CCSD", "ccsd"), ("CCSD(T)", "ccsd(t)"),
    ])
    def test_method_key(self, method_type, expected):
        assert pyscf_method(method_type, "B3LYP") == expected


def _spec_kwargs(**overrides):
    kwargs = dict(
        calculation_type="Single-Point", geometry="O 0 0 0", structure_file="w.xyz",
        method_type="DFT", functional="B3LYP", basis_set="6-31G(d)", reference="RHF",
        charge=0, multiplicity=1, n_threads=1, memory_gb=1.0,
    )
    kwargs.update(overrides)
    return kwargs


class TestConvergenceCriteria:
    def test_every_option_maps_to_geometric_kwargs(self):
        for name, kwargs in G_CONVERGENCE_KWARGS.items():
            assert "convergence_set" in kwargs, name

    def test_qchem_and_molpro_are_flags_not_sets(self):
        # geomeTRIC has no "QCHEM" set; it has a qccnv flag that changes how the standard
        # criteria are combined. Passing the name as a set would raise a KeyError inside
        # the optimizer, several minutes into a job.
        assert G_CONVERGENCE_KWARGS["QCHEM"]["qccnv"] is True
        assert G_CONVERGENCE_KWARGS["MOLPRO"]["molcnv"] is True

    def test_sets_are_ones_geometric_actually_defines(self):
        pytest.importorskip("geometric")
        from geometric.params import OptParams

        for name, kwargs in G_CONVERGENCE_KWARGS.items():
            # OptParams raises on an unknown set name, so constructing it is the check.
            OptParams(**kwargs)
