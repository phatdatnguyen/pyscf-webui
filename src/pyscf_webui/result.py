"""Result tab: load a result file and render whichever results it contains.

The data source is the ``.result.json`` the runner wrote, not the engine log. Because this
app owns both the producer and the consumer, there is no output-parsing layer to go stale
-- and some of what is shown (normal-mode IR intensities, TD-DFT rotatory strengths) is
only available from the live wavefunction anyway, never from the printed output.

A single loader (:func:`on_load_result_file`) reads the file and conditionally reveals
accordions -- Energy/MOs, Geometry Optimization, Frequency/IR, Absorption/ECD, and
Orbitals & Density -- based on what the calculation actually produced.
"""
import os
import shutil
import subprocess
import sys

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import (
    RESULT_SUFFIX,
    assign_multiplets,
    build_nmr_peaks,
    format_couplings_by_atom,
    equivalent_nucleus_groups,
    generate_absorption_emission_spectrum_interactive,
    generate_ecd_spectrum_interactive,
    generate_ir_spectrum_interactive,
    generate_nmr_spectrum_interactive,
    get_files_in_working_directory,
    mol_from_symbols_and_coords,
    multiplicity_label,
    read_json,
    result_base_name,
    write_json,
)
from .visualization import (
    CUBE_TASKS,
    ELECTROSTATIC_POTENTIAL,
    SPIN_DENSITY,
    TOTAL_DENSITY,
    render_cube_html,
)


def on_working_directory_file_list_change(working_directory_file_list):
    """Repopulate the result-file dropdown with the result files in the directory."""
    result_file_names = sorted(
        (f for f in (working_directory_file_list or []) if f.endswith(RESULT_SUFFIX)),
        key=str.lower,
    )
    return gr.update(
        choices=result_file_names,
        value=result_file_names[0] if result_file_names else None,
        interactive=True,
    )


def _optimization_figure(energies):
    """Plotly trace of SCF energy against optimization step.

    Plotly rather than matplotlib so the y-axis can be inspected interactively: the
    interesting part of an optimization is the last few microhartree, which is invisible
    on a static plot scaled to the first step's drop.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=list(range(1, len(energies) + 1)), y=energies,
        mode="lines+markers", name="Energy",
    ))
    figure.update_layout(
        title="Optimization energy",
        xaxis_title="Step",
        yaxis_title="Energy (hartree)",
        template="plotly_white",
    )
    # Absolute electronic energies are large numbers whose variation is tiny; an offset
    # axis would hide the convergence behaviour this plot exists to show.
    figure.update_yaxes(tickformat=".6f")
    return figure


def _mo_dataframe(result):
    """Molecular-orbital energy table, with the HOMO and LUMO labelled."""
    energies = result.get("mo_energies_hartree") or []
    if not energies:
        return None
    homo_index = result.get("homo_index")
    homo_index = -1 if homo_index is None else int(homo_index)

    rows = []
    for index, energy in enumerate(energies):
        if index == homo_index:
            label = f"MO {index + 1} (HOMO)"
        elif index == homo_index + 1:
            label = f"MO {index + 1} (LUMO)"
        else:
            label = f"MO {index + 1}"
        rows.append({
            "Molecular orbital": label,
            "Energy (hartree)": "{:.6f}".format(energy),
            "Energy (eV)": "{:.4f}".format(energy * 27.211386245988),
        })
    return pd.DataFrame(rows, columns=["Molecular orbital", "Energy (hartree)", "Energy (eV)"])


def _visualization_choices(result):
    """Selections for the orbital viewer, given what the wavefunction contains."""
    choices = [TOTAL_DENSITY, ELECTROSTATIC_POTENTIAL]
    # A spin density is identically zero for a closed-shell system, so it is only offered
    # when the calculation actually had unpaired electrons.
    if result.get("n_alpha") != result.get("n_beta"):
        choices.append(SPIN_DENSITY)
    n_orbitals = len(result.get("mo_energies_hartree") or [])
    choices.extend(f"MO {i + 1}" for i in range(n_orbitals))
    return choices


def _frequency_dataframe(frequency):
    """Vibrational frequency / IR intensity table.

    The intensity column is dropped rather than zero-filled when the calculation could not
    produce intensities (see :func:`pyscf_webui.engine.frequency_analysis`): a column of
    zeros would read as "these modes are IR-inactive", which is a statement about the
    molecule rather than about the calculation.
    """
    frequencies = frequency.get("frequencies_cm1") or []
    columns = {"Frequency (cm-1)": ["{:.2f}".format(f) for f in frequencies]}
    intensities = frequency.get("ir_intensities_km_mol")
    if intensities:
        columns["IR intensity (km/mol)"] = ["{:.4f}".format(i) for i in intensities]
    return pd.DataFrame(columns)


def _thermo_dataframe(thermo):
    """Thermochemistry summary, converted from hartree to kcal/mol for display."""
    if not thermo:
        return None
    hartree_to_kcal = 627.5094740631

    def kcal(key):
        value = thermo.get(key)
        return "" if value is None else "{:.4f}".format(value * hartree_to_kcal)

    entropy = thermo.get("entropy_cal_mol_k")
    return pd.DataFrame([{
        "Zero-point energy (kcal/mol)": kcal("zpve_hartree"),
        "Thermal energy (kcal/mol)": kcal("thermal_energy_hartree"),
        "Enthalpy (kcal/mol)": kcal("enthalpy_hartree"),
        "Entropy (cal/mol/K)": "" if entropy is None else "{:.4f}".format(entropy),
        "Gibbs free energy (kcal/mol)": kcal("gibbs_hartree"),
        "T (K)": "{:.2f}".format(thermo.get("temperature_k", 0.0)),
    }])


def _excitation_dataframe(excitations):
    """Excited-state table for the TD-DFT accordion."""
    return pd.DataFrame([{
        "State": state["index"],
        "Wavelength (nm)": "" if state.get("wavelength_nm") is None else "{:.2f}".format(state["wavelength_nm"]),
        "Energy (eV)": "{:.4f}".format(state["energy_ev"]),
        "Oscillator strength": "{:.6f}".format(state["oscillator_strength"]),
        "Rotatory strength": "{:.6f}".format(state["rotatory_strength"]),
        "Symmetry": state.get("symmetry", ""),
    } for state in excitations])


def _emission_figure(emission):
    """One broadened emission band at the computed emission wavelength.

    A single band rather than a stick spectrum, because Kasha's rule puts essentially all
    emission in the lowest excited state. The y-axis is relative: the position and width
    of the band carry the physics, not its height. (Unlike the Psi4 version, an oscillator
    strength for the emitting state *is* available now that this is a TD-DFT calculation,
    but with one band to draw there is nothing to weight it against.)
    """
    wavelength = emission.get("wavelength_nm")
    if not wavelength:
        return None
    return generate_absorption_emission_spectrum_interactive([wavelength], [1.0])


def _emission_dataframe(emission):
    """Single-row summary of the emission energy and the states it came from."""
    wavelength = emission.get("wavelength_nm")
    absorption = emission.get("absorption_wavelength_nm")
    stokes = emission.get("stokes_shift_cm1")
    oscillator = emission.get("oscillator_strength")
    row = {
        "Emitting state": f"S{emission.get('root', 1)}",
        "Wavelength (nm)": "" if wavelength is None else "{:.2f}".format(wavelength),
        "Emission energy (eV)": "{:.4f}".format(emission["emission_energy_ev"]),
    }
    if absorption is not None:
        row["Absorption (nm)"] = "{:.2f}".format(absorption)
    if stokes is not None:
        row["Stokes shift (cm-1)"] = "{:.1f}".format(stokes)
    if oscillator is not None:
        row["Oscillator strength"] = "{:.6f}".format(oscillator)
    row["E(excited, relaxed) (hartree)"] = "{:.8f}".format(emission["excited_energy_hartree"])
    row["E(ground, same geometry) (hartree)"] = "{:.8f}".format(
        emission["ground_energy_at_excited_geometry_hartree"])
    return pd.DataFrame([row])


# Nominal spectrometer frequency, used only to turn a coupling in hertz into a splitting
# in ppm. It changes how a multiplet looks, exactly as it does on a real instrument, but
# not the underlying shifts or coupling constants.
DEFAULT_SPECTROMETER_MHZ = 400.0

# Absolute shielding of TMS, the reference compound for 1H and 13C, computed with this app.
# A shielding reference is strongly method- and basis-dependent, so the defaults are taken
# at **the level the NMR menu itself defaults to** -- otherwise the very first spectrum a
# user produces is systematically wrong, and by a lot: the same molecule at B3LYP/6-31G(d)
# gives 190.41 for carbon, 9.5 ppm away from the pcseg-2 value.
TMS_REFERENCES = {
    "B3LYP/pcseg-2": (31.57, 180.96),
    "B3LYP/6-31G(d)": (32.19, 190.41),
}
DEFAULT_H_REFERENCE, DEFAULT_C_REFERENCE = TMS_REFERENCES["B3LYP/pcseg-2"]


def _parse_reference(value, default):
    """Read a reference-shielding textbox, falling back on blank or unparseable input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nmr_dataframe(shieldings, references, coupling_text=None):
    """Per-nucleus shielding table, with a chemical shift where a reference exists.

    Nuclei of an element with no reference (N, O, F, Si...) keep their absolute shielding
    and get a blank shift, rather than being dropped from the table entirely -- the
    shielding is a real computed result and the user may have a reference of their own.

    The J-couplings column appears only when couplings were computed, and unlike the
    simulated spectrum it keeps heteronuclear pairs: a C-H coupling is a useful number to
    read even though it does not split a 1H multiplet.
    """
    columns = ["Index", "Symbol", "Shielding (ppm)", "Anisotropy (ppm)",
               "Chemical shift (ppm)"]
    rows = []
    for entry in shieldings:
        symbol = entry.get("symbol", "")
        shielding = entry.get("isotropic_ppm")
        reference = references.get(symbol)
        row = {
            "Index": entry.get("index"),
            "Symbol": symbol,
            "Shielding (ppm)": "" if shielding is None else "{:.4f}".format(shielding),
            "Anisotropy (ppm)": "" if entry.get("anisotropy_ppm") is None
                                else "{:.4f}".format(entry["anisotropy_ppm"]),
            "Chemical shift (ppm)": "" if (reference is None or shielding is None)
                                    else "{:.4f}".format(reference - shielding),
        }
        if coupling_text is not None:
            row["J-couplings"] = coupling_text.get(entry.get("index"), "")
        rows.append(row)
    if coupling_text is not None:
        columns.append("J-couplings")
    return pd.DataFrame(rows, columns=columns)


def _nmr_peak_dataframe(peaks, element_symbol, with_couplings=False):
    """Averaged-signal table: one row per group of equivalent nuclei.

    This is the table that reads like a reported spectrum -- shift, integration,
    multiplicity, J -- so the multiplicity and J columns appear together or not at all.
    """
    rows = []
    for peak in peaks:
        row = {
            f"Atoms ({element_symbol})": ", ".join(str(i) for i in peak["atom_idxs"]),
            f"# {element_symbol}": peak["count"],
            "Chemical shift (ppm)": "{:.3f}".format(peak["shift"]),
        }
        if with_couplings:
            couplings = peak.get("couplings") or []
            row["Multiplicity"] = multiplicity_label(couplings)
            row["J (Hz)"] = ", ".join("{:.2f}".format(c["j_hz"]) for c in couplings)
        row["Shielding (ppm)"] = "{:.3f}".format(peak["shielding"])
        rows.append(row)
    return pd.DataFrame(rows)


def _nmr_outputs(result, h_reference, c_reference, average_equivalent,
                 spectrometer_mhz=DEFAULT_SPECTROMETER_MHZ):
    """Build every NMR output from a loaded result.

    Returns ``(visible, per_atom_df, h_df, h_figure, c_df, c_figure)``. Referencing and
    multiplet construction both happen here rather than in the runner, so changing a
    reference, the spectrometer frequency or the averaging toggle re-renders everything
    without re-running the calculation.
    """
    nmr = result.get("nmr")
    shieldings = (nmr or {}).get("shieldings") or []
    if not shieldings:
        return gr.update(visible=False), None, None, None, None, None

    references = {"H": _parse_reference(h_reference, DEFAULT_H_REFERENCE),
                  "C": _parse_reference(c_reference, DEFAULT_C_REFERENCE)}
    couplings = (nmr or {}).get("couplings") or []
    frequency = _parse_reference(spectrometer_mhz, DEFAULT_SPECTROMETER_MHZ)

    equivalence = None
    if average_equivalent:
        try:
            molecule = mol_from_symbols_and_coords(result["symbols"],
                                                   result["geometry_angstrom"])
            equivalence = equivalent_nucleus_groups(molecule) or None
        except Exception as exc:
            print(f"Could not group equivalent nuclei ({exc}); showing every nucleus.")

    per_atom = _nmr_dataframe(shieldings, references,
                              format_couplings_by_atom(couplings) if couplings else None)
    tables_and_plots = []
    for symbol in ("H", "C"):
        peaks = build_nmr_peaks(shieldings, symbol, references[symbol], equivalence)
        # 13C is reported proton-decoupled, so it is left as singlets even when couplings
        # are available -- which is also why only homonuclear pairs are passed in.
        split = bool(couplings) and symbol == "H"
        if split:
            assign_multiplets(peaks, couplings, symbol)
        tables_and_plots.append(
            _nmr_peak_dataframe(peaks, symbol, with_couplings=split) if peaks else None)
        tables_and_plots.append(generate_nmr_spectrum_interactive(
            peaks, symbol, spectrometer_mhz=frequency))
    return (gr.update(visible=True), per_atom, *tables_and_plots)


# Number of values :func:`on_load_result_file` returns, i.e. the length of
# ``_result_outputs`` in :func:`result_tab_content`. The two must stay index-aligned;
# ``test_handlers.py`` asserts it rather than leaving it to review.
RESULT_OUTPUT_COUNT = 29


def _blank_outputs(status):
    """The failure-path return: every accordion hidden, nothing rendered."""
    return (
        status, None,
        gr.update(visible=False), "", "", None,
        gr.update(visible=False), None,
        gr.update(visible=False), None, None, None, "",
        gr.update(visible=False), None, None, gr.update(interactive=False),
        gr.update(visible=False), gr.update(choices=[], value=None), None,
        gr.update(visible=False), None, None,
        gr.update(visible=False), None, None, None, None, None,
    )


def on_load_result_file(working_directory_path, result_file_name,
                        h_reference=DEFAULT_H_REFERENCE, c_reference=DEFAULT_C_REFERENCE,
                        average_equivalent=True,
                        spectrometer_mhz=DEFAULT_SPECTROMETER_MHZ):
    """Load a ``.result.json`` and produce every Result-tab output.

    Returns a positional tuple aligned index-for-index with ``_result_outputs`` in
    :func:`result_tab_content` -- keep the two in sync when changing either. On error,
    returns a matching tuple that hides every accordion and reports the error.
    """
    if not working_directory_path or not result_file_name:
        gr.Warning("Please select a result file.")
        return _blank_outputs("")

    try:
        result_path = os.path.join(working_directory_path, result_file_name)
        result = read_json(result_path)
    except Exception as exc:
        return _blank_outputs(f"<span style='color:red;'>Error loading result file: {exc}</span>")

    if result.get("status") != "completed":
        error = result.get("error") or "the calculation did not finish"
        return _blank_outputs(
            f"<span style='color:red;'>This calculation failed: {error}</span>"
        )

    try:
        # --- Energy / molecular orbitals -----------------------------------------
        energy = result.get("energy_hartree")
        energy_text = "" if energy is None else "{:.8f} hartree".format(energy)
        dipole = result.get("dipole_debye")
        dipole_text = "" if dipole is None else "{:.4f} Debye".format(dipole)
        mo_dataframe = _mo_dataframe(result)
        show_energy = energy is not None

        # --- Geometry optimization ------------------------------------------------
        optimization = result.get("optimization")
        show_optimization = bool(optimization and optimization.get("energies_hartree"))
        optimization_figure = (
            _optimization_figure(optimization["energies_hartree"]) if show_optimization else None
        )

        # --- Frequency / IR --------------------------------------------------------
        frequency = result.get("frequency")
        show_frequency = bool(frequency and frequency.get("frequencies_cm1"))
        if show_frequency:
            frequency_dataframe = _frequency_dataframe(frequency)
            # Imaginary modes are stored as negative numbers. Only the real modes are
            # broadened into a spectrum -- an imaginary frequency has no absorption band,
            # and passing it through would put a phantom peak at a mirrored wavenumber.
            frequencies = np.asarray(frequency["frequencies_cm1"], dtype=float)
            raw_intensities = frequency.get("ir_intensities_km_mol")
            real = frequencies > 0
            notes = []
            if raw_intensities:
                intensities = np.asarray(raw_intensities, dtype=float)
                ir_spectrum = generate_ir_spectrum_interactive(
                    frequencies[real], intensities[real], width=12.0, transmittance=True
                )
            else:
                # No intensities means no spectrum: the frequency table and the
                # thermochemistry are still valid and still shown.
                ir_spectrum = None
                notes.append(
                    "<span style='color:orange;'>IR intensities are unavailable for this "
                    "calculation, so no spectrum is shown. They require a closed-shell "
                    "reference running on the GPU -- PySCF has no CPU implementation of "
                    "dipole derivatives.</span>"
                )
            thermo_dataframe = _thermo_dataframe(result.get("thermochemistry"))
            n_imaginary = int(frequency.get("n_imaginary", 0))
            if n_imaginary:
                notes.append(
                    f"<span style='color:orange;'>{n_imaginary} imaginary frequency/frequencies: "
                    "this geometry is a saddle point, not a minimum, so the thermochemistry "
                    "below is not valid for a stable species.</span>"
                )
            frequency_note = "<br>".join(notes)
        else:
            frequency_dataframe = ir_spectrum = thermo_dataframe = None
            frequency_note = ""

        # --- Absorption / ECD ------------------------------------------------------
        excitations = result.get("excitations") or []
        # A state with no wavelength (a non-positive root) cannot be plotted.
        plottable = [s for s in excitations if s.get("wavelength_nm")]
        show_excitations = bool(plottable)
        if show_excitations:
            excitation_dataframe = _excitation_dataframe(excitations)
            wavelengths = [s["wavelength_nm"] for s in plottable]
            absorption_spectrum = generate_absorption_emission_spectrum_interactive(
                wavelengths, [s["oscillator_strength"] for s in plottable]
            )
            # ECD is only meaningful when something is actually chiral; an achiral
            # molecule gives rotatory strengths that are all zero and a flat line.
            has_ecd = any(abs(s["rotatory_strength"]) > 1e-10 for s in plottable)
        else:
            excitation_dataframe = absorption_spectrum = None
            has_ecd = False

        # --- Emission --------------------------------------------------------------
        emission = result.get("emission")
        show_emission = bool(emission and emission.get("wavelength_nm"))
        emission_dataframe = _emission_dataframe(emission) if show_emission else None
        emission_figure = _emission_figure(emission) if show_emission else None

        # --- Orbitals & density ----------------------------------------------------
        show_orbitals = bool(result.get("wavefunction_file"))
        visualization_choices = _visualization_choices(result) if show_orbitals else []

        status = f"<span style='color:green;'>Loaded {result_file_name}.</span>"
        return (
            status, result,
            gr.update(visible=show_energy), energy_text, dipole_text, mo_dataframe,
            gr.update(visible=show_optimization), optimization_figure,
            gr.update(visible=show_frequency), frequency_dataframe, ir_spectrum, thermo_dataframe, frequency_note,
            gr.update(visible=show_excitations), excitation_dataframe, absorption_spectrum, gr.update(interactive=has_ecd),
            gr.update(visible=show_orbitals),
            gr.update(choices=visualization_choices,
                      value=visualization_choices[0] if visualization_choices else None),
            None,
            gr.update(visible=show_emission), emission_dataframe, emission_figure,
            *_nmr_outputs(result, h_reference, c_reference, average_equivalent,
                          spectrometer_mhz),
        )
    except Exception as exc:
        return _blank_outputs(f"<span style='color:red;'>Error rendering result: {exc}</span>")


def on_show_ecd_spectrum(result):
    """Build the ECD spectrum from the loaded result's rotatory strengths."""
    excitations = (result or {}).get("excitations") or []
    plottable = [s for s in excitations if s.get("wavelength_nm")]
    if not plottable:
        gr.Warning("This result has no excited states to plot.")
        return None
    return generate_ecd_spectrum_interactive(
        [s["wavelength_nm"] for s in plottable],
        [s["rotatory_strength"] for s in plottable],
    )


def on_visualization_change(selection):
    """Give density and orbital isosurfaces different default isolevels.

    An electron density is large and positive everywhere near the nuclei, so a contour
    around 0.05 traces the molecular surface; an MO amplitude is much smaller and signed,
    so the same value would render almost nothing. The old version of this app hid the
    slider entirely for MOs and hardcoded +/-2, which is far outside the range any orbital
    reaches -- the surface it drew came from the fallback, not from the number.
    """
    is_density = selection in (TOTAL_DENSITY, SPIN_DENSITY, ELECTROSTATIC_POTENTIAL)
    return gr.update(value=0.05 if is_density else 0.02)


def on_visualize(working_directory_path, result_file_name, result, selection,
                 color1, color2, opacity, isolevel, grid_spacing):
    """Render the selected density or orbital as an nglview isosurface.

    Runs ``pyscf.tools.cubegen`` in a child process against the checkpoint saved by the
    calculation, so any orbital can be inspected after the fact without re-running the
    calculation or generating every cube up front.
    """
    if not result or not result.get("wavefunction_file"):
        gr.Warning("This result has no saved wavefunction to visualize.")
        return None
    if not selection:
        gr.Warning("Please choose something to visualize.")
        return None

    base_name = result_base_name(result_file_name)
    try:
        if selection.startswith("MO "):
            # "MO 7" -> orbital 7, 1-based. Unlike Psi4, which asked for the alpha and
            # beta index of the same orbital as a positive/negative pair, cubegen takes a
            # single coefficient column.
            task, orbital = "orbital", int(selection.split()[1])
        else:
            # The total and spin densities are distinct requests here, not one task whose
            # two output files are told apart afterwards.
            task, orbital = CUBE_TASKS[selection], None

        # One output directory per result, wiped before each render: cube files are large
        # and accumulate quickly, and a stale file from a previous selection is exactly
        # what the old version's "newest matching file" glob used to pick up by mistake.
        output_directory = os.path.join(working_directory_path, "cubes", base_name)
        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory, ignore_errors=True)
        os.makedirs(output_directory, exist_ok=True)

        spec_path = os.path.join(working_directory_path, base_name + ".cube.json")
        # Absolute throughout: the child runs *in* the working directory, so any path
        # relative to it would be resolved against itself a second time.
        write_json(spec_path, {
            "wavefunction_path": os.path.abspath(
                os.path.join(working_directory_path, result["wavefunction_file"])),
            "tasks": [task],
            "orbital": orbital,
            "grid_spacing": float(grid_spacing),
            "output_directory": os.path.abspath(output_directory),
        })

        completed = subprocess.run(
            [sys.executable, "-u", "-m", "pyscf_webui.runner", "cubegen",
             os.path.abspath(spec_path)],
            cwd=working_directory_path, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            gr.Warning(f"Could not generate the cube file:\n{completed.stdout[-2000:]}")
            return None

        return render_cube_html(
            result, output_directory, selection,
            color1=color1, color2=color2, opacity=opacity, isolevel=isolevel,
        )
    except Exception as exc:
        gr.Warning(f"Visualization error: {exc}")
        return None


def on_export_data(working_directory_path, file_name, dataframe):
    """Export a result DataFrame to ``<file_name>.csv`` in the working directory."""
    try:
        file_path = os.path.join(working_directory_path, file_name + ".csv")
        dataframe.to_csv(file_path, encoding="utf-8", index=False)
        return ("<span style='color:green;'>Data exported successfully.</span>",
                get_files_in_working_directory(working_directory_path))
    except Exception as exc:
        return (f"<span style='color:red;'>Error exporting data: {exc}</span>",
                get_files_in_working_directory(working_directory_path))


def result_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    """Build the "Result" tab (all result accordions) and wire its events.

    The output list wired to the load handler must stay index-aligned with the return
    tuple of :func:`on_load_result_file`.
    """
    with gr.Tab("Result") as result_tab:
        with gr.Row():
            with gr.Column(scale=1):
                result_file_dropdown = gr.Dropdown(label="Calculation result", choices=[""], value="", interactive=False)
            with gr.Column(scale=1):
                load_button = gr.Button("Load", variant="primary")
                result_state = gr.State()
        with gr.Accordion(label="Energy", visible=False) as energy_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    energy_textbox = gr.Textbox(label="Total energy", value="")
                    dipole_textbox = gr.Textbox(label="Dipole moment", value="")
                with gr.Column(scale=1):
                    mo_dataframe = gr.DataFrame(label="Molecular orbitals", max_height=360)
        with gr.Accordion(label="Geometry Optimization", visible=False) as optimization_accordion:
            optimization_plot = gr.Plot(label="Energy per step")
        with gr.Accordion(label="Frequency", visible=False) as frequency_accordion:
            frequency_note_markdown = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    frequency_dataframe = gr.Dataframe(label="Vibrational frequencies", max_height=360)
                    frequency_filename_textbox = gr.Textbox(label="File name", value="ir_data")
                    export_frequency_button = gr.Button(value="Export")
                with gr.Column(scale=2):
                    ir_spectrum_plot = gr.Plot(label="IR spectrum")
            thermo_dataframe = gr.Dataframe(label="Thermochemistry")
        with gr.Accordion(label="Absorption / ECD Spectrum", visible=False) as excitation_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    ecd_spectrum_button = gr.Button("Show ECD spectrum", interactive=False)
                    excitation_filename_textbox = gr.Textbox(label="File name", value="excitation_data")
                    export_excitation_button = gr.Button(value="Export")
                with gr.Column(scale=2):
                    excitation_dataframe = gr.Dataframe(label="Excited states", max_height=360)
            absorption_plot = gr.Plot(label="UV-Vis absorption spectrum")
            ecd_plot = gr.Plot(label="ECD spectrum")
        with gr.Accordion(label="Emission", visible=False) as emission_accordion:
            emission_dataframe_component = gr.Dataframe(label="Emission")
            emission_plot = gr.Plot(label="Emission spectrum")
        with gr.Accordion(label="NMR", visible=False) as nmr_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    nmr_dataframe = gr.Dataframe(label="Shielding per nucleus", max_height=420)
                    nmr_filename_textbox = gr.Textbox(label="File name", value="nmr_data")
                    export_nmr_button = gr.Button("Export")
                with gr.Column(scale=2):
                    gr.Markdown(
                        "Chemical shift is `reference - shielding`. The defaults are TMS at "
                        "**B3LYP/pcseg-2**, the level the NMR menu defaults to. A shielding "
                        "reference depends strongly on functional and basis, so for any "
                        "other level run TMS through this app and paste its isotropic "
                        "shieldings here.\n\n"
                        + "  \n".join(
                            f"- TMS at {level}: \u00b9H {h}, \u00b9\u00b3C {c}"
                            for level, (h, c) in TMS_REFERENCES.items())
                    )
                    with gr.Row():
                        h_reference_textbox = gr.Textbox(
                            label="1H reference shielding (ppm)", value=str(DEFAULT_H_REFERENCE))
                        c_reference_textbox = gr.Textbox(
                            label="13C reference shielding (ppm)", value=str(DEFAULT_C_REFERENCE))
                    spectrometer_slider = gr.Slider(
                        label="Spectrometer frequency (MHz)", value=DEFAULT_SPECTROMETER_MHZ,
                        minimum=60, maximum=1200, step=10,
                        info="Converts J in hertz to a splitting in ppm. Affects how "
                             "multiplets look, not the shifts or coupling constants.")
                    average_equivalent_checkbox = gr.Checkbox(
                        label="Average symmetry-equivalent nuclei", value=True,
                        info="Groups nuclei the molecular graph cannot distinguish into one "
                             "signal, as a real spectrum shows them. Topological only: "
                             "diastereotopic protons are merged.")
                    gr.Markdown(
                        "*¹H multiplets are shown only when the calculation was run with "
                        "**Compute J-couplings** enabled; otherwise every signal is a "
                        "singlet. ¹³C is always shown proton-decoupled.*"
                    )
                    nmr_1h_plot = gr.Plot(label="1H NMR spectrum")
                    nmr_1h_dataframe = gr.Dataframe(label="Averaged 1H signals")
                    nmr_13c_plot = gr.Plot(label="13C NMR spectrum")
                    nmr_13c_dataframe = gr.Dataframe(label="Averaged 13C signals")
        with gr.Accordion(label="Orbitals & Density", visible=False) as orbital_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    visualization_dropdown = gr.Dropdown(label="Visualization", choices=[], value=None)
                    with gr.Row():
                        color1_picker = gr.ColorPicker(label="Positive lobe", value="#0000ff")
                        color2_picker = gr.ColorPicker(label="Negative lobe", value="#ff0000")
                    with gr.Row():
                        opacity_slider = gr.Slider(label="Opacity", value=0.8, minimum=0, maximum=1, step=0.01)
                        isolevel_slider = gr.Slider(label="Isolevel", value=0.05, minimum=0.001, maximum=1, step=0.001)
                    grid_spacing_slider = gr.Slider(label="Grid spacing (bohr)", value=0.2, minimum=0.05, maximum=0.5, step=0.05,
                                                    info="Smaller is smoother but slower and much larger on disk.")
                    visualize_button = gr.Button("Visualize", variant="primary")
                with gr.Column(scale=2):
                    visualization_html = gr.HTML()

    _result_outputs = [
        status_markdown, result_state,
        energy_accordion, energy_textbox, dipole_textbox, mo_dataframe,
        optimization_accordion, optimization_plot,
        frequency_accordion, frequency_dataframe, ir_spectrum_plot, thermo_dataframe, frequency_note_markdown,
        excitation_accordion, excitation_dataframe, absorption_plot, ecd_spectrum_button,
        orbital_accordion, visualization_dropdown, visualization_html,
        emission_accordion, emission_dataframe_component, emission_plot,
        nmr_accordion, nmr_dataframe,
        nmr_1h_dataframe, nmr_1h_plot, nmr_13c_dataframe, nmr_13c_plot,
    ]

    # The NMR panel re-renders from the already-loaded result rather than re-reading the
    # file, so changing a reference or the averaging toggle is instant. Shared by Load and
    # by the panel's own controls, hence the single input list.
    _load_inputs = [working_directory_path_state, result_file_dropdown,
                    h_reference_textbox, c_reference_textbox, average_equivalent_checkbox,
                    spectrometer_slider]

    working_directory_file_list_state.change(
        on_working_directory_file_list_change, working_directory_file_list_state, result_file_dropdown)
    load_button.click(on_load_result_file, _load_inputs, _result_outputs)
    # Unlike the ORCA app this is modelled on, these do not need a separate Refresh button:
    # each control re-runs the loader itself, so a changed reference takes effect at once.
    for _nmr_control in (h_reference_textbox, c_reference_textbox):
        _nmr_control.submit(on_load_result_file, _load_inputs, _result_outputs)
    average_equivalent_checkbox.change(on_load_result_file, _load_inputs, _result_outputs)
    spectrometer_slider.release(on_load_result_file, _load_inputs, _result_outputs)
    export_nmr_button.click(
        on_export_data, [working_directory_path_state, nmr_filename_textbox, nmr_dataframe],
        [status_markdown, working_directory_file_list_state])
    ecd_spectrum_button.click(on_show_ecd_spectrum, result_state, ecd_plot)
    visualization_dropdown.change(on_visualization_change, visualization_dropdown, isolevel_slider)
    visualize_button.click(
        on_visualize,
        [working_directory_path_state, result_file_dropdown, result_state, visualization_dropdown,
         color1_picker, color2_picker, opacity_slider, isolevel_slider, grid_spacing_slider],
        visualization_html,
    )
    export_frequency_button.click(
        on_export_data, [working_directory_path_state, frequency_filename_textbox, frequency_dataframe],
        [status_markdown, working_directory_file_list_state])
    export_excitation_button.click(
        on_export_data, [working_directory_path_state, excitation_filename_textbox, excitation_dataframe],
        [status_markdown, working_directory_file_list_state])

    return result_tab
