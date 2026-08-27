"""NMR peak grouping, referencing and spectrum simulation.

The engine half (gpu4pyscf's shielding call) is covered by the GPU integration test at the
bottom; everything above it is pure and runs anywhere.
"""
import pytest

from pyscf_webui import result as result_tab
from pyscf_webui.utils import (
    assign_multiplets,
    build_nmr_peaks,
    equivalent_nucleus_groups,
    format_couplings_by_atom,
    generate_nmr_spectrum_interactive,
    mol_from_symbols_and_coords,
    multiplicity_label,
)

from conftest import needs_gpu

# Shieldings shaped like the runner's output: an ethanol-ish set with three H environments.
SHIELDINGS = [
    {"index": 0, "symbol": "C", "isotropic_ppm": 160.0, "anisotropy_ppm": 30.0},
    {"index": 1, "symbol": "C", "isotropic_ppm": 130.0, "anisotropy_ppm": 40.0},
    {"index": 2, "symbol": "H", "isotropic_ppm": 31.0, "anisotropy_ppm": 8.0},
    {"index": 3, "symbol": "H", "isotropic_ppm": 31.0, "anisotropy_ppm": 8.0},
    {"index": 4, "symbol": "H", "isotropic_ppm": 31.0, "anisotropy_ppm": 8.0},
    {"index": 5, "symbol": "H", "isotropic_ppm": 28.0, "anisotropy_ppm": 9.0},
]


class TestPeakGrouping:
    def test_shift_is_reference_minus_shielding(self):
        """Sign convention: a *less* shielded nucleus appears further downfield, so the
        subtraction runs reference - sigma and not the other way round."""
        peaks = build_nmr_peaks(SHIELDINGS, "H", reference_ppm=32.0, equivalence=None)
        shifts = sorted(p["shift"] for p in peaks)
        assert shifts == pytest.approx([1.0, 1.0, 1.0, 4.0])

    def test_without_equivalence_every_nucleus_is_its_own_peak(self):
        peaks = build_nmr_peaks(SHIELDINGS, "H", 32.0, equivalence=None)
        assert len(peaks) == 4
        assert all(p["count"] == 1 for p in peaks)

    def test_equivalent_nuclei_merge_into_one_peak(self):
        equivalence = {2: "methyl", 3: "methyl", 4: "methyl", 5: "hydroxyl"}
        peaks = build_nmr_peaks(SHIELDINGS, "H", 32.0, equivalence)
        assert len(peaks) == 2
        methyl = next(p for p in peaks if p["count"] == 3)
        assert methyl["atom_idxs"] == [2, 3, 4]
        assert methyl["shift"] == pytest.approx(1.0)

    def test_a_merged_peak_averages_its_members(self):
        shieldings = [
            {"index": 0, "symbol": "H", "isotropic_ppm": 30.0},
            {"index": 1, "symbol": "H", "isotropic_ppm": 32.0},
        ]
        peaks = build_nmr_peaks(shieldings, "H", 32.0, {0: "g", 1: "g"})
        assert peaks[0]["shielding"] == pytest.approx(31.0)

    def test_peaks_are_ordered_downfield_first(self):
        peaks = build_nmr_peaks(SHIELDINGS, "H", 32.0, equivalence=None)
        shifts = [p["shift"] for p in peaks]
        assert shifts == sorted(shifts, reverse=True)

    def test_other_elements_are_ignored(self):
        peaks = build_nmr_peaks(SHIELDINGS, "C", 190.0, equivalence=None)
        assert len(peaks) == 2
        assert {p["atom_idxs"][0] for p in peaks} == {0, 1}

    def test_absent_element_gives_no_peaks(self):
        assert build_nmr_peaks(SHIELDINGS, "F", 100.0, None) == []


class TestEquivalence:
    def test_toluene_methyl_protons_are_one_group(self):
        """The three methyl H must merge, and the ortho and meta pairs must each merge --
        that is what turns 8 protons into 4 signals."""
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
        groups = equivalent_nucleus_groups(mol)
        hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
        sizes = sorted(
            sum(1 for h in hydrogens if groups[h] == groups[i]) for i in hydrogens)
        # 8 protons: 1 para (group of 1), 2 ortho + 2 meta (two groups of 2),
        # 3 methyl (one group of 3). Each proton reports its own group's size.
        assert sizes == [1, 2, 2, 2, 2, 3, 3, 3]

    def test_benzene_protons_are_all_equivalent(self):
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        groups = equivalent_nucleus_groups(mol)
        hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
        assert len({groups[h] for h in hydrogens}) == 1

    def test_works_on_a_molecule_rebuilt_from_a_result_file(self):
        """The real call path: a bond-free geometry out of result.json. RDKit's ranking
        needs a populated valence cache, which such a molecule does not have."""
        symbols = ["C", "H", "H", "H", "H"]
        coords = [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                  [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]
        mol = mol_from_symbols_and_coords(symbols, coords)
        groups = equivalent_nucleus_groups(mol)
        assert len({groups[i] for i in (1, 2, 3, 4)}) == 1  # methane's four H

    def test_failure_degrades_to_no_grouping(self):
        class Broken:
            def UpdatePropertyCache(self, strict=True):
                raise RuntimeError("no")

        assert equivalent_nucleus_groups(Broken()) == {}


class TestSpectrum:
    def test_axis_runs_downfield_left(self):
        peaks = build_nmr_peaks(SHIELDINGS, "H", 32.0, None)
        figure = generate_nmr_spectrum_interactive(peaks, "H")
        assert figure.layout.xaxis.autorange == "reversed"

    def test_one_annotation_per_peak(self):
        peaks = build_nmr_peaks(SHIELDINGS, "H", 32.0, None)
        figure = generate_nmr_spectrum_interactive(peaks, "H")
        assert len(figure.layout.annotations) == len(peaks)

    def test_peak_height_tracks_nucleus_count(self):
        """A methyl singlet must be three times a methine, or the spectrum misleads."""
        equivalence = {2: "m", 3: "m", 4: "m", 5: "o"}
        peaks = build_nmr_peaks(SHIELDINGS, "H", 32.0, equivalence)
        figure = generate_nmr_spectrum_interactive(peaks, "H", linewidth_hz=1.0)
        y = figure.data[0].y
        x = figure.data[0].x
        import numpy as np

        methyl = next(p for p in peaks if p["count"] == 3)
        other = next(p for p in peaks if p["count"] == 1)
        h_methyl = y[int(np.argmin(np.abs(x - methyl["shift"])))]
        h_other = y[int(np.argmin(np.abs(x - other["shift"])))]
        assert h_methyl == pytest.approx(3 * h_other, rel=0.05)

    def test_absent_element_gives_no_figure(self):
        assert generate_nmr_spectrum_interactive([], "F") is None


class TestResultTab:
    def test_nmr_outputs_hidden_when_the_result_has_none(self, water_result):
        visible = result_tab._nmr_outputs(water_result, "32.19", "190.41", True)[0]
        assert visible["visible"] is False

    def test_nmr_outputs_render_when_present(self, water_result):
        water_result["nmr"] = {"shieldings": SHIELDINGS}
        water_result["symbols"] = ["C", "C", "H", "H", "H", "H"]
        water_result["geometry_angstrom"] = [[0, 0, 0], [1.5, 0, 0], [0, 1, 0],
                                             [0, -1, 0], [-1, 0, 0], [2.5, 0, 0]]
        visible, per_atom, h_df, h_fig, c_df, c_fig = result_tab._nmr_outputs(
            water_result, "32.0", "190.0", False)
        assert visible["visible"] is True
        assert len(per_atom) == len(SHIELDINGS)
        assert "Chemical shift (ppm)" in per_atom.columns
        assert h_df is not None and c_df is not None
        assert h_fig is not None and c_fig is not None

    def test_a_blank_reference_box_falls_back_to_the_default(self, water_result):
        water_result["nmr"] = {"shieldings": SHIELDINGS}
        per_atom = result_tab._nmr_outputs(water_result, "", "", False)[1]
        shift = float(per_atom.loc[per_atom["Symbol"] == "H", "Chemical shift (ppm)"].iloc[0])
        assert shift == pytest.approx(result_tab.DEFAULT_H_REFERENCE - 31.0, abs=1e-6)

    def test_elements_without_a_reference_keep_their_shielding(self):
        """Nitrogen has no TMS reference, but its shielding is still a real result and
        should not vanish from the table."""
        frame = result_tab._nmr_dataframe(
            [{"index": 0, "symbol": "N", "isotropic_ppm": 250.0, "anisotropy_ppm": 1.0}],
            {"H": 32.0, "C": 190.0})
        assert len(frame) == 1
        assert frame["Shielding (ppm)"].iloc[0] == "250.0000"
        assert frame["Chemical shift (ppm)"].iloc[0] == ""


@needs_gpu
def test_tms_shielding_matches_literature():
    """The reference compound, end to end through the engine.

    TMS at B3LYP/6-31G(d) should give sigma(13C) ~ 190 ppm and sigma(1H) ~ 32 ppm. Getting
    these right is what makes every chemical shift this app reports meaningful.
    """
    import numpy as np
    from pyscf import gto

    from gpu4pyscf.dft import RKS

    from pyscf_webui import engine

    mol = gto.M(atom="""Si 0.0 0.0 0.0
        C 1.09 1.09 1.09
        C -1.09 -1.09 1.09
        C -1.09 1.09 -1.09
        C 1.09 -1.09 -1.09
        H 1.74 0.49 1.74
        H 1.74 1.74 0.49
        H 0.49 1.74 1.74
        H -1.74 -0.49 1.74
        H -1.74 -1.74 0.49
        H -0.49 -1.74 1.74
        H -1.74 0.49 -1.74
        H -1.74 1.74 -0.49
        H -0.49 1.74 -1.74
        H 1.74 -0.49 -1.74
        H 1.74 -1.74 -0.49
        H 0.49 -1.74 -1.74""", basis="6-31G(d)", verbose=0)
    mf = RKS(mol, xc="b3lyp").density_fit()
    mf.kernel()

    spec = {"functional": "B3LYP", "basis_set": "6-31G(d)"}
    fragment, arrays = engine.nmr_shielding(mf, spec)
    shieldings = fragment["nmr"]["shieldings"]

    carbon = np.mean([e["isotropic_ppm"] for e in shieldings if e["symbol"] == "C"])
    hydrogen = np.mean([e["isotropic_ppm"] for e in shieldings if e["symbol"] == "H"])
    assert carbon == pytest.approx(190.4, abs=2.0)
    assert hydrogen == pytest.approx(32.2, abs=0.5)
    assert arrays["nmr_shielding_total"].shape == (mol.natm, 3, 3)


# Ethanol, atom order C0(CH3) C1(CH2) O2 H3,4,5(CH3) H6,7(CH2) H8(OH). These are the real
# couplings computed by this app at B3LYP/6-31G(d,p); the three CH3-CH2 vicinal values
# differ because a single frozen conformer has three different dihedrals.
ETHANOL_EQUIVALENCE = {3: "me", 4: "me", 5: "me", 6: "ch2", 7: "ch2", 8: "oh"}
ETHANOL_SHIELDINGS = [
    {"index": i, "symbol": "H", "isotropic_ppm": v}
    for i, v in [(3, 30.9), (4, 30.9), (5, 30.9), (6, 28.05), (7, 28.05), (8, 30.76)]
]


def _c(a, b, j, sa="H", sb="H"):
    return {"atom_a": a, "symbol_a": sa, "atom_b": b, "symbol_b": sb, "j_hz": j}


ETHANOL_COUPLINGS = [
    _c(5, 4, -14.87), _c(5, 3, -14.66), _c(4, 3, -13.17),   # geminal, within CH3
    _c(7, 6, -12.59),                                        # geminal, within CH2
    _c(8, 6, 12.16), _c(6, 4, 11.07), _c(7, 3, 10.63),
    _c(7, 4, 3.63), _c(6, 3, 3.60), _c(6, 5, 1.66),
    _c(8, 5, -1.58), _c(8, 7, 1.04),
]


def _ethanol_peaks():
    peaks = build_nmr_peaks(ETHANOL_SHIELDINGS, "H", 32.19, ETHANOL_EQUIVALENCE)
    assign_multiplets(peaks, ETHANOL_COUPLINGS, "H")
    return {ETHANOL_EQUIVALENCE[p["atom_idxs"][0]]: p for p in peaks}


class TestMultiplets:
    def test_ethanol_gives_the_textbook_pattern(self):
        """CH3 split by 2 CH2 protons is a triplet; CH2 split by 3 CH3 protons a quartet.
        Both also see the OH, so each picks up one extra doublet."""
        peaks = _ethanol_peaks()
        assert multiplicity_label(peaks["ch2"]["couplings"]) == "dq"
        assert multiplicity_label(peaks["me"]["couplings"]) == "td"
        assert multiplicity_label(peaks["oh"]["couplings"]) == "tq"

    def test_multiplicity_comes_from_partner_group_size(self):
        """Not from how many couplings happened to clear the threshold -- that is what
        makes a frozen geometry still read as a clean triplet."""
        me = _ethanol_peaks()["me"]
        to_ch2 = next(c for c in me["couplings"] if c["n"] == 2)
        assert to_ch2["partner_idxs"] == (6, 7)

    def test_coupling_to_a_group_is_the_average_of_its_pairs(self):
        """The three CH3-CH2 vicinal couplings are 11.07, 3.60 and 1.66 Hz for one proton
        and 10.63, 3.63 and ~0 for the other; averaging recovers the ~6 Hz the observed
        triplet is built from."""
        me = _ethanol_peaks()["me"]
        to_ch2 = next(c for c in me["couplings"] if c["n"] == 2)
        assert to_ch2["j_hz"] == pytest.approx(6.1, abs=0.5)

    def test_equivalent_nuclei_do_not_split_each_other(self):
        """The -14 Hz geminal couplings inside the methyl group are real but unobservable:
        mutually equivalent nuclei produce no splitting."""
        me = _ethanol_peaks()["me"]
        partners = {i for c in me["couplings"] for i in c["partner_idxs"]}
        assert partners.isdisjoint({3, 4, 5})

    def test_negligible_couplings_are_dropped(self):
        peaks = build_nmr_peaks(ETHANOL_SHIELDINGS, "H", 32.19, ETHANOL_EQUIVALENCE)
        assign_multiplets(peaks, [_c(6, 3, 0.2)], "H", j_threshold=0.5)
        assert all(not p["couplings"] for p in peaks)

    def test_heteronuclear_couplings_do_not_split(self):
        """A 1H signal is not split by the carbon it sits on -- 13C is 1% abundant, and
        those satellites are not what a multiplicity means."""
        peaks = build_nmr_peaks(ETHANOL_SHIELDINGS, "H", 32.19, ETHANOL_EQUIVALENCE)
        assign_multiplets(peaks, [_c(3, 0, 125.0, sa="H", sb="C")], "H")
        assert all(not p["couplings"] for p in peaks)

    def test_no_couplings_means_every_signal_is_a_singlet(self):
        peaks = build_nmr_peaks(ETHANOL_SHIELDINGS, "H", 32.19, ETHANOL_EQUIVALENCE)
        assign_multiplets(peaks, [], "H")
        assert all(multiplicity_label(p["couplings"]) == "s" for p in peaks)


class TestMultipletSpectrum:
    def test_a_triplet_has_three_lines_in_the_right_ratio(self):
        import numpy as np

        peaks = [{"atom_idxs": [0], "count": 1, "shielding": 0.0, "shift": 5.0,
                  "couplings": [{"partner_idxs": (1, 2), "n": 2, "j_hz": 10.0}]}]
        figure = generate_nmr_spectrum_interactive(
            peaks, "H", spectrometer_mhz=100.0, linewidth_hz=0.5, points=40000)
        y = np.asarray(figure.data[0].y)
        # Local maxima above a small floor.
        peaks_found = [i for i in range(1, len(y) - 1)
                       if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 0.05]
        assert len(peaks_found) == 3
        heights = sorted(y[i] for i in peaks_found)
        assert heights[2] == pytest.approx(2 * heights[0], rel=0.1)  # 1:2:1

    def test_splitting_scales_with_spectrometer_frequency(self):
        """J is fixed in hertz, so in ppm a multiplet narrows as the field rises -- which
        is the whole reason the frequency is a control rather than a constant."""
        import numpy as np

        def spread(mhz):
            peaks = [{"atom_idxs": [0], "count": 1, "shielding": 0.0, "shift": 5.0,
                      "couplings": [{"partner_idxs": (1,), "n": 1, "j_hz": 10.0}]}]
            fig = generate_nmr_spectrum_interactive(
                peaks, "H", spectrometer_mhz=mhz, linewidth_hz=0.5, points=40000)
            y, x = np.asarray(fig.data[0].y), np.asarray(fig.data[0].x)
            found = [x[i] for i in range(1, len(y) - 1)
                     if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 0.05]
            return max(found) - min(found)

        assert spread(100.0) == pytest.approx(4 * spread(400.0), rel=0.1)

    def test_multiplicity_appears_in_the_peak_label(self):
        peaks = _ethanol_peaks()
        figure = generate_nmr_spectrum_interactive(list(peaks.values()), "H")
        labels = [a.text for a in figure.layout.annotations]
        assert any("(td)" in label for label in labels)


class TestCouplingText:
    def test_both_partners_are_annotated(self):
        text = format_couplings_by_atom([_c(3, 6, 7.0)])
        assert "H6" in text[3]
        assert "H3" in text[6]

    def test_a_pair_listed_twice_is_not_duplicated(self):
        text = format_couplings_by_atom([_c(3, 6, 7.0), _c(6, 3, 7.0)])
        assert text[3].count("Hz") == 1

    def test_strongest_coupling_comes_first(self):
        text = format_couplings_by_atom([_c(3, 6, 2.0), _c(3, 7, 12.0)])
        assert text[3].startswith("12.00")

    def test_heteronuclear_couplings_are_kept_in_the_table(self):
        """Unlike the spectrum, the table shows C-H couplings: they are useful numbers."""
        text = format_couplings_by_atom([_c(3, 0, 125.0, sa="H", sb="C")])
        assert "C0" in text[3]


class TestResultTabCouplings:
    def test_j_column_appears_only_when_couplings_exist(self, water_result):
        without = result_tab._nmr_dataframe(ETHANOL_SHIELDINGS, {"H": 32.19}, None)
        assert "J-couplings" not in without.columns
        with_j = result_tab._nmr_dataframe(
            ETHANOL_SHIELDINGS, {"H": 32.19}, format_couplings_by_atom(ETHANOL_COUPLINGS))
        assert "J-couplings" in with_j.columns

    def test_multiplicity_columns_appear_only_when_split(self):
        peaks = list(_ethanol_peaks().values())
        plain = result_tab._nmr_peak_dataframe(peaks, "H", with_couplings=False)
        assert "Multiplicity" not in plain.columns
        split = result_tab._nmr_peak_dataframe(peaks, "H", with_couplings=True)
        assert "Multiplicity" in split.columns and "J (Hz)" in split.columns

    def test_carbon_stays_proton_decoupled(self, water_result):
        """13C is reported decoupled, so it must show singlets even with couplings present."""
        water_result["nmr"] = {
            "shieldings": ETHANOL_SHIELDINGS + [
                {"index": 0, "symbol": "C", "isotropic_ppm": 160.0}],
            "couplings": ETHANOL_COUPLINGS + [_c(3, 0, 125.0, sa="H", sb="C")],
        }
        water_result["symbols"] = ["C"] + ["H"] * 6
        water_result["geometry_angstrom"] = [[0, 0, 0]] + [[1.0 * i, 0, 0] for i in range(1, 7)]
        _v, _per_atom, h_df, _hf, c_df, _cf = result_tab._nmr_outputs(
            water_result, "32.19", "190.41", False)
        assert "Multiplicity" in h_df.columns
        assert "Multiplicity" not in c_df.columns
