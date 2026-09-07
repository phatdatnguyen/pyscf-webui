"""Core, UI-free engine for pyscf-webui.

Groups three responsibilities used by the Gradio tabs:

1. Molecule I/O (``mol_from_*``, ``add_bonds``, ``conformer_to_xyz_file``).
2. Job specs: the small JSON contract between the Calculation tab (which writes a
   spec) and :mod:`pyscf_webui.runner` (which executes it in a child process) and the
   Result tab (which reads the results the runner wrote back).
3. Spectrum simulation (IR, UV-Vis absorption, ECD) as Plotly figures.

Everything here is pure Python with no Gradio dependency -- and, importantly, **no
``pyscf``, ``gpu4pyscf`` or ``cupy`` import**: this module is loaded by the web server,
which must start and remain usable where PySCF is missing and must never hold a CUDA
context it does not use. The engine is imported only inside the runner subprocess.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import plotly.graph_objects as go

# Suffixes of the two JSON artifacts the runner contract is built on. A job spec is what
# the Calculation tab writes and the runner consumes; a result file is what the runner
# writes and the Result tab consumes. They are distinct suffixes (rather than one
# directory of ``.json``) so the file browser can label them and each tab's dropdown can
# filter to just the kind it accepts.
JOB_SUFFIX = ".job.json"
RESULT_SUFFIX = ".result.json"

# Calculation types. Module-level constants because the Calculation tab, the runner and
# the Result tab all branch on them and a typo in any one of the three would silently
# mis-route a job.
SINGLE_POINT = "Single-Point"
GEOMETRY_OPTIMIZATION = "Geometry Optimization"
FREQUENCY = "Frequency"
TDDFT = "Time-Dependent Density Functional Theory"
EMISSION = "Emission (TD-DFT)"
NMR = "NMR Spectrum"
CALCULATION_TYPES = [SINGLE_POINT, GEOMETRY_OPTIMIZATION, FREQUENCY, TDDFT, EMISSION, NMR]

def get_files_in_working_directory(working_directory_path: str) -> list[str]:
    """Return the file names in ``working_directory_path``, or ``[]`` if it is unset.

    Windows ``Zone.Identifier`` alternate-data-stream files (created when files are
    downloaded) are filtered out so they never appear in the UI file list.

    Subdirectories are excluded: the runner keeps its scratch in ``scratch/`` and
    cube files in ``cubes/`` inside the working directory, and listing those as if they
    were files would offer them to the viewers and the delete button, neither of which
    can handle a directory.

    The ``None`` guard matters: the shared path ``gr.State`` starts empty and is reset
    to ``None`` when a directory fails to open, and ``os.listdir(None)`` quietly lists
    the *current* directory instead of raising — which surfaced as callers building
    paths out of a ``None`` directory.
    """
    if not working_directory_path:
        return []
    try:
        files = os.listdir(working_directory_path)
    except (FileNotFoundError, NotADirectoryError):
        # Handlers also refresh this list while reporting a failed file operation;
        # a directory moved or deleted in another session must not break that path.
        return []
    return [f for f in files
            if not f.endswith('Zone.Identifier')
            and os.path.isfile(os.path.join(working_directory_path, f))]

def conformer_to_xyz_file(mol: Chem.Mol, conf_id: int, file_path: str,
                          charge: int = 0, multiplicity: int = 1) -> None:
    """Write one conformer of ``mol`` to an XYZ file.

    The first line is ``"<charge> <multiplicity>"`` (the convention this app's XYZ
    reader expects) rather than the standard atom count, followed by one
    ``"<symbol> <x> <y> <z>"`` line per atom taken from conformer ``conf_id``.
    """
    # Get atom information
    atoms = mol.GetAtoms()
    xyz_lines = []
    for atom in atoms:
        pos = mol.GetConformer(conf_id).GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}")

    # Construct the XYZ string
    xyz_string = f"{charge} {multiplicity}\n" + "\n".join(xyz_lines)
    
    with open(file_path, 'w') as file:
        file.write(xyz_string)

def add_bonds(mol: Chem.Mol, bond_factor: float = 1.25) -> Chem.Mol:
    """Return a copy of ``mol`` with bonds inferred from interatomic distances.

    Structure files such as XYZ carry only atoms and coordinates, so RDKit sees no
    connectivity. Two atoms are joined by a single bond when their distance is below
    ``(r_cov_i + r_cov_j) * bond_factor`` using RDKit's covalent radii. Bond orders
    are not perceived (all bonds are single); this is enough for visualization and
    for feeding coordinates into a quantum-chemistry calculation.

    Formal charges are assigned to hypervalent main-group atoms so the result passes
    ``Chem.SanitizeMol`` — currently 4-coordinate boron (see ``_assign_formal_charges``),
    e.g. the ``BF2`` bridge in BODIPY dyes. These are per-atom bookkeeping charges only;
    the overall charge used for the calculation is set separately by the user.
    """
    # Create a new empty molecule
    mol_new = Chem.RWMol()

    # Add atoms
    for atom in mol.GetAtoms():
        mol_new.AddAtom(atom)

    # Add conformer
    conf = mol.GetConformer()
    mol_new.AddConformer(conf)

    # Add bonds based on covalent radii
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j)))
            pt = Chem.GetPeriodicTable()
            r_cov_i = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetSymbol())
            r_cov_j = pt.GetRcovalent(mol.GetAtomWithIdx(j).GetSymbol())
            if dist < (r_cov_i + r_cov_j) * bond_factor:
                mol_new.AddBond(i, j, Chem.BondType.SINGLE)

    # Reconcile valences that would otherwise fail sanitization (e.g. borate boron).
    _assign_formal_charges(mol_new)

    # Convert to a regular Mol object
    mol_new = mol_new.GetMol()

    return mol_new

# Neutral bonding valence for main-group elements whose over-coordination (with the
# all-single-bond perception used here) implies a formal charge. degree - valence gives
# the charge: 4-coordinate boron -> -1 (borate), a 4th bond on N -> +1 (ammonium), etc.
_NEUTRAL_VALENCE = {'B': 3, 'N': 3, 'O': 2}

def _assign_formal_charges(mol: Chem.RWMol) -> None:
    """Assign formal charges so over-coordinated main-group atoms pass sanitization.

    Only atoms bonded to MORE neighbors than their neutral valence are touched, and the
    charge is the difference (boron gains electrons -> negative; N/O lose -> positive).
    This fixes structures like BODIPY's 4-coordinate borate boron, which RDKit otherwise
    rejects with "Explicit valence for atom B, 4, is greater than permitted."
    """
    for atom in mol.GetAtoms():
        neutral = _NEUTRAL_VALENCE.get(atom.GetSymbol())
        if neutral is None:
            continue
        excess = atom.GetDegree() - neutral
        if excess > 0:
            # Group 13 (B) is electron-deficient: an extra bond makes it anionic;
            # groups 15/16 (N, O) become cationic.
            sign = -1 if atom.GetSymbol() == 'B' else 1
            atom.SetFormalCharge(sign * excess)

def mol_from_xyz_file(file_path: str, return_charge_and_multiplicity: bool = False):
    """Read an XYZ file into a bond-free RDKit ``Mol`` with one conformer.

    Handles both flavours the app deals with:

    * The first line is ``"<charge> <multiplicity>"`` (what
      :func:`conformer_to_xyz_file` writes).
    * Standard XYZ, whose first line is an atom count. These may hold **many frames**
      concatenated (an optimization trajectory), in which case the *last* frame is
      returned -- for a trajectory that is the converged geometry, which is what someone
      clicking the file in the browser wants to see. Reading every line as an atom would
      otherwise fuse all the frames into one nonsense molecule.

    Anything else defaults to charge/multiplicity ``0``/``1``. Only lines with exactly
    four whitespace-separated fields (symbol + x/y/z) are treated as atoms.

    Returns the ``Mol`` alone, or ``(mol, charge, multiplicity)`` when
    ``return_charge_and_multiplicity`` is True.
    """
    with open(file_path, 'r') as file:
        xyz_string = file.read()

    lines = xyz_string.split('\n')
    charge, multiplicity = 0, 1
    first_fields = lines[0].split()
    if len(first_fields) == 2:
        try:
            charge, multiplicity = int(first_fields[0]), int(first_fields[1])
        except ValueError:
            charge, multiplicity = 0, 1
    elif len(first_fields) == 1:
        try:
            n_atoms = int(first_fields[0])
        except ValueError:
            n_atoms = None
        if n_atoms is not None:
            # Read frame boundaries, including the mandatory comment line. Taking
            # the last N four-field lines can silently splice two geometries when a
            # trajectory's final frame is incomplete, or misread a comment as an atom.
            cursor = 0
            last_frame = None
            while cursor < len(lines):
                if not lines[cursor].strip():
                    cursor += 1
                    continue
                try:
                    n_atoms = int(lines[cursor].strip())
                except ValueError as exc:
                    raise ValueError(f"Invalid XYZ atom count on line {cursor + 1}.") from exc
                if n_atoms <= 0:
                    raise ValueError("An XYZ frame must contain at least one atom.")
                start, end = cursor + 2, cursor + 2 + n_atoms
                frame = lines[start:end]
                if len(frame) != n_atoms or any(len(line.split()) != 4 for line in frame):
                    raise ValueError(f"Incomplete or malformed XYZ frame on line {cursor + 1}.")
                last_frame = frame
                cursor = end
            lines = [""] + last_frame

    # Create a new empty molecule
    mol = Chem.RWMol()

    # Parse the atomic coordinates and add atoms
    elements = []
    for line in lines[1:]:
        if len(line.split())==4:
            parts = line.split()
            element = parts[0].capitalize()
            atom = Chem.Atom(element)
            mol.AddAtom(atom)
            elements.append(element)
        else:
            continue

    # Add 3D coordinates to the molecule
    conf = Chem.Conformer(mol.GetNumAtoms())
    atom_idx = 0
    for line in lines[1:]:
        if len(line.split())==4:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            conf.SetAtomPosition(atom_idx, (x, y, z))
            atom_idx += 1
        else:
            continue
    mol.AddConformer(conf)
                
    # Convert to a regular Mol object
    mol = mol.GetMol()
    
    if return_charge_and_multiplicity:
        return mol, charge, multiplicity
    else:
        return mol

def mol_from_structure_file(file_path: str) -> Chem.Mol:
    """Load any structure file the UI accepts into a sanitized ``Mol`` with a conformer.

    Shared by the structure viewer and the Calculation tab so that both see the same
    atoms in the same order. That ordering is load-bearing: the viewer labels each atom
    with its index, and :func:`pyscf_atom_spec` emits the coordinate block in that
    same index order, so two different loaders here would make the on-screen labels
    disagree with the geometry the engine actually receives.

    Raises ``ValueError`` for an unsupported extension rather than returning ``None``,
    which would otherwise surface as an opaque RDKit error further down.
    """
    if file_path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=False)
    elif file_path.endswith('.mol'):
        mol = Chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
    elif file_path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(file_path, sanitize=False, removeHs=False)
    elif file_path.endswith('.xyz'):
        mol = add_bonds(mol_from_xyz_file(file_path))
    else:
        raise ValueError(f"unsupported structure file type: {os.path.basename(file_path)}")

    if mol is None:
        raise ValueError(f"could not read structure file: {os.path.basename(file_path)}")

    Chem.SanitizeMol(mol)
    # Display/use the file's REAL coordinates. Only embed a fresh conformer if the file
    # somehow carried none — re-embedding otherwise would discard the parsed (e.g.
    # QM-optimized) geometry and, because add_bonds perceives every bond as single,
    # pucker planar/aromatic systems into a bogus sp3-looking shape.
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)

    return mol

def lorentzian_ir(wavenumber, position, intensity, width: float = 10):
    """Lorentzian band shape (peak height = ``intensity`` at ``position``).

    ``wavenumber`` may be a scalar or a numpy array (vectorized over the x-axis).
    ``width`` is the half-width at half-maximum in cm^-1.
    """
    return intensity * (width ** 2) / ((wavenumber - position) ** 2 + width ** 2)

def generate_ir_spectrum_interactive(
    frequencies,
    intensities,
    width: float = 10.0,
    points: int = 4000,
    plot_range: tuple[float, float] | None = None,
    normalize: bool = True,
    transmittance: bool = True,
) -> go.Figure | None:
    """Build an interactive IR spectrum from vibrational frequencies + IR intensities.

    Each mode is broadened with a Lorentzian (:func:`lorentzian_ir`) and summed onto
    a wavenumber grid. Optionally normalized and converted to transmittance
    (``1 - absorbance``, with the x-axis reversed per IR convention). Returns a Plotly
    figure, or ``None`` when no frequencies are supplied.
    """
    # Safety check
    if frequencies is None or len(frequencies) == 0:
        print("No IR frequencies detected.")
        return None

    # Define plotting range
    if plot_range:
        min_wn, max_wn = plot_range
    else:
        min_wn = np.min(frequencies) - 100
        max_wn = np.max(frequencies) + 100

    x = np.linspace(min_wn, max_wn, points)
    spectrum = np.zeros_like(x)

    # Generate Lorentzian broadened spectrum
    annotations = []
    for freq, inten in zip(frequencies, intensities):
        spectrum += lorentzian_ir(x, freq, inten, width)

        annotations.append({
            "freq": freq,
            "intensity": inten,
            "text": f"{freq:.1f}"
        })

    # Normalize if requested
    if normalize and np.max(spectrum) != 0:
        spectrum /= np.max(spectrum)

    # Convert to transmittance if requested
    if transmittance:
        spectrum = 1 - spectrum  # simple inversion

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name="IR Spectrum"
    ))

    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann["freq"],
            y=max(spectrum),
            text=ann["text"],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            textangle=-90,
            font=dict(color='red', size=10)
        )

    # Layout
    fig.update_layout(
        title="IR Spectrum",
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            autorange="reversed"  # Important for IR
        ),
        yaxis=dict(
            title="Transmittance" if transmittance else "Intensity (a.u.)"
        ),
        showlegend=False
    )

    # Apply manual range if provided
    if plot_range:
        fig.update_xaxes(range=plot_range[::-1])  # reverse manually

    return fig

def gaussian(x, x0, fwhm: float):
    """Unit-height Gaussian centered at ``x0`` with the given full width at half max.

    ``x`` may be a scalar or numpy array (vectorized over the x-axis).
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

def generate_absorption_emission_spectrum_interactive(
    wavelengths, oscs, points: int = 10000,
    plot_range: tuple[float, float] | None = None,
) -> go.Figure | None:
    """Build an interactive UV-Vis spectrum from transition wavelengths + oscillator strengths.

    Each transition is broadened with a Gaussian whose width grows with its oscillator
    strength, weighted by that strength, and summed onto a wavelength grid. Returns a
    normalized Plotly figure, or ``None`` when no transitions are supplied.
    """
    # Check if wavelengths and oscs are valid
    if wavelengths is None or len(wavelengths) == 0:
        print("No absorption peak detected.")
        return
    
    # Create the chemical shift axis
    if plot_range:
        min_wavelength, max_wavelength = plot_range
    else:
        min_wavelength = np.min(wavelengths) - 10
        max_wavelength = np.max(wavelengths) + 10
    
    x = np.linspace(min_wavelength, max_wavelength, points)
    
    # Initialize the spectrum
    spectrum = np.zeros_like(x)
    
    # Generate annotations for each peak with Gaussian broadening
    annotations = []
    base_fwhm = 10.0
    for wavelength, osc in zip(wavelengths, oscs):
        fwhm = base_fwhm * (1 + osc)   # stronger osc → broader peak
        spectrum += osc * gaussian(x, wavelength, fwhm)
        annotations.append({
            'wavelength': wavelength,
            'oscilation strength': oscs,
            'text': f"{wavelength:.2f}",
        })

    # Normalize the spectrum (guard against all-zero oscillator strengths)
    max_intensity = np.max(spectrum)
    if max_intensity != 0:
        spectrum /= max_intensity

    # Create the Plotly figure
    fig = go.Figure()

    # Add the spectrum trace
    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name='Spectrum'
    ))
    
    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann['wavelength'],
            y=1.05,
            text=ann['text'],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            yshift=10,
            textangle=-90,
            font=dict(color='red'),
        )

    fig.update_layout(
        xaxis=dict(
            title='Wavelength (nm)',
        ),
        yaxis=dict(
            title='Relative Intensity (a.u.)',
        ),
        title='Spectrum',
        showlegend=False,
    )

    # Adjust plot range if specified
    if plot_range:
        fig.update_xaxes(range=plot_range)

    return fig

def generate_ecd_spectrum_interactive(
    wavelengths, rotatory_strengths, points: int = 10000,
    plot_range: tuple[float, float] | None = None,
    fwhm_wavenumber: float = 3000.0,
) -> go.Figure | None:
    """Build an interactive ECD spectrum from wavelengths + rotatory strengths.

    Differs from :func:`generate_absorption_emission_spectrum_interactive` in two ways
    that matter physically:

    * Rotatory strengths are **signed**. They are summed with their sign so positive and
      negative Cotton effects appear above and below zero, and neighbouring bands of
      opposite sign partially cancel. Taking magnitudes here would destroy the
      information that makes ECD useful (two enantiomers differ only by the sign of
      every R, so their spectra are mirror images).
    * Broadening is done in **wavenumber space**, where a Gaussian band shape is the
      physically appropriate model, and only the display axis is wavelength. Broadening
      with a constant width directly in nm would make bands artificially narrow at short
      wavelengths and wide at long ones.

    ``fwhm_wavenumber`` is the band width in cm^-1 (~3000 cm^-1 is a conventional
    starting point for ECD). The curve is scaled by its largest absolute value, so the
    y-axis is relative and signed, in [-1, 1]. Returns ``None`` when no transitions are
    supplied.
    """
    if wavelengths is None or len(wavelengths) == 0:
        print("No ECD transitions to plot.")
        return None

    wavelengths = np.asarray(wavelengths, dtype=float)
    rotatory_strengths = np.asarray(rotatory_strengths, dtype=float)

    if plot_range:
        min_wavelength, max_wavelength = plot_range
    else:
        min_wavelength = np.min(wavelengths) - 50
        max_wavelength = np.max(wavelengths) + 50
    # The nm -> cm^-1 conversion below divides by the axis, so it must stay positive.
    min_wavelength = max(float(min_wavelength), 1.0)
    max_wavelength = max(float(max_wavelength), min_wavelength + 1.0)

    x = np.linspace(min_wavelength, max_wavelength, points)
    grid_wavenumbers = 1e7 / x

    spectrum = np.zeros_like(x)
    annotations = []
    for wavelength, rotatory_strength in zip(wavelengths, rotatory_strengths):
        if wavelength <= 0:
            continue
        spectrum += rotatory_strength * gaussian(grid_wavenumbers, 1e7 / wavelength, fwhm_wavenumber)
        annotations.append({
            'wavelength': wavelength,
            'rotatory_strength': rotatory_strength,
            'text': f"{wavelength:.2f}",
        })

    # Scale by the largest absolute value so the sign is preserved.
    max_intensity = np.max(np.abs(spectrum))
    if max_intensity != 0:
        spectrum /= max_intensity

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name='ECD',
    ))
    # ECD crosses zero, so the baseline is a meaningful reference and is drawn in.
    fig.add_hline(y=0, line_width=1, line_color='grey')

    # Annotate each transition on the side its own sign points to.
    for ann in annotations:
        above = ann['rotatory_strength'] >= 0
        fig.add_annotation(
            x=ann['wavelength'],
            y=1.05 if above else -1.05,
            text=ann['text'],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40 if above else 40,
            yshift=10 if above else -10,
            textangle=-90,
            font=dict(color='red' if above else 'blue'),
        )

    fig.update_layout(
        xaxis=dict(title='Wavelength (nm)'),
        # Symmetric range so a mirror-image (opposite enantiomer) spectrum is directly
        # comparable, and so the zero crossing sits in the middle.
        yaxis=dict(title='Relative Δε (a.u.)', range=[-1.3, 1.3]),
        title='ECD Spectrum',
        showlegend=False,
    )

    if plot_range:
        fig.update_xaxes(range=[min_wavelength, max_wavelength])

    return fig

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Job specs
#
# The Calculation tab never builds an *input file* -- there is nothing for a user to
# author or edit. It writes a machine-readable job spec and the runner turns that into
# direct PySCF API calls. The spec exists only because the calculation runs in a child
# process (so it can be cancelled, and so a CUDA abort cannot take the web server down);
# it is an implementation detail that happens to be visible in the file browser, not a
# format anyone is expected to write by hand.
#
# Everything below is static data and pure functions. Nothing here may import the engine.

def lorentzian_nmr(x, x0, gamma):
    """Unit-height Lorentzian line centred at ``x0`` with half-width ``gamma`` (ppm)."""
    return (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)


def equivalent_nucleus_groups(mol) -> dict[int, int]:
    """Map each atom index to a symmetry class, so equivalent nuclei share one signal.

    A real spectrum shows one averaged peak for the three protons of a methyl group, not
    three. RDKit's canonical ranking with ``breakTies=False`` returns exactly that
    partition -- atoms that the graph automorphism cannot distinguish get the same rank --
    which is why this is four lines rather than the hand-rolled bond perception plus Morgan
    refinement an app starting from a bond-free geometry would need.

    The equivalence is *topological*: it does not know about diastereotopicity or hindered
    rotation, so it can merge two protons that a real spectrum resolves. That is the usual
    trade-off for this kind of prediction and matches what the ORCA app does.
    """
    from rdkit import Chem

    # The molecule is rebuilt from a result file's bare coordinates, so its valence cache
    # was never populated; CanonicalRankAtoms needs it and otherwise raises a
    # "getNumImplicitHs() called without preceding call to calcImplicitValence()"
    # pre-condition violation. strict=False because a QM geometry can legitimately contain
    # a hypervalent or radical centre that strict sanitization would reject outright.
    try:
        mol.UpdatePropertyCache(strict=False)
        return {index: rank for index, rank
                in enumerate(Chem.CanonicalRankAtoms(mol, breakTies=False))}
    except Exception as exc:
        # Losing symmetry detection costs averaging, not correctness: every nucleus simply
        # becomes its own peak. Better than failing to display a finished calculation.
        print(f"Could not determine equivalent nuclei ({exc}); "
              f"every nucleus will be shown separately.")
        return {}


def build_nmr_peaks(shieldings, element_symbol, reference_ppm, equivalence=None):
    """Group shieldings for one element into peaks and convert them to chemical shifts.

    Returns a list of ``{"atom_idxs", "count", "shift", "shielding"}`` sorted by descending
    shift, which is the order an NMR spectrum is conventionally read in.

    The shift convention is ``delta = sigma_reference - sigma``: a *less* shielded nucleus
    appears further downfield, so the subtraction runs this way round and not the other.
    """
    groups: dict[Any, dict[str, Any]] = {}
    for entry in shieldings:
        if entry.get("symbol") != element_symbol:
            continue
        index = int(entry["index"])
        shielding = float(entry["isotropic_ppm"])
        # Without an equivalence map every nucleus is its own peak. Rounding the shielding
        # as a fallback key would merge accidental near-coincidences, which is a different
        # (and wrong) claim from "these are symmetry-equivalent".
        key = equivalence.get(index, ("atom", index)) if equivalence else ("atom", index)
        group = groups.setdefault(key, {"atom_idxs": [], "shieldings": []})
        group["atom_idxs"].append(index)
        group["shieldings"].append(shielding)

    peaks = []
    for group in groups.values():
        mean_shielding = sum(group["shieldings"]) / len(group["shieldings"])
        peaks.append({
            "atom_idxs": sorted(group["atom_idxs"]),
            "count": len(group["atom_idxs"]),
            "shielding": mean_shielding,
            "shift": reference_ppm - mean_shielding,
        })
    peaks.sort(key=lambda peak: -peak["shift"])
    return peaks


# Coupling to n equivalent partners splits a signal into n+1 lines. Beyond a septet the
# conventional letters run out, so anything larger is reported as an unresolved multiplet.
_MULTIPLICITY_LETTER = {1: "d", 2: "t", 3: "q", 4: "p", 5: "h", 6: "hept"}


def assign_multiplets(peaks, couplings, element_symbol, j_threshold=0.5):
    """Attach a coupling list to each peak, in place, and return the peaks.

    The modelling decision that makes a single static geometry produce a textbook spectrum:
    two groups of equivalent nuclei A and B are treated to first order, so every nucleus of
    A is split by B into ``size(B) + 1`` lines with **one** coupling constant equal to the
    *average* of all pairwise A-B couplings.

    Both halves of that matter. Real ethanol shows a clean CH3 triplet because rotation
    averages the three CH3-CH2 dihedrals; a frozen geometry instead gives couplings of
    ~11, ~3.6 and ~1.7 Hz for the three protons, and averaging them recovers the ~5 Hz the
    triplet is actually built from. And taking the multiplicity from the partner group's
    *size* rather than from how many couplings happen to exceed the threshold is what keeps
    a CH3-CH2 pair reading as a triplet/quartet.

    Only **homonuclear** couplings split: a 1H spectrum is split by other protons, not by
    the carbons they sit on (in practice 13C is 1% abundant, so those satellites are not
    what anyone means by the multiplicity).
    """
    pairwise = {}
    for coupling in couplings or []:
        if (coupling.get("symbol_a") != element_symbol
                or coupling.get("symbol_b") != element_symbol):
            continue
        a, b = int(coupling["atom_a"]), int(coupling["atom_b"])
        pairwise[(a, b)] = pairwise[(b, a)] = float(coupling["j_hz"])

    for peak in peaks:
        members = peak["atom_idxs"]
        peak_couplings = []
        for other in peaks:
            if other is peak:
                continue  # mutually equivalent nuclei do not split each other
            partners = other["atom_idxs"]
            values = [pairwise[(a, b)] for a in members for b in partners
                      if (a, b) in pairwise]
            if not values:
                continue
            average = sum(values) / len(values)
            if abs(average) < j_threshold:
                continue  # below what a spectrometer would resolve
            peak_couplings.append({"partner_idxs": tuple(partners),
                                   "n": len(partners), "j_hz": average})
        peak_couplings.sort(key=lambda c: -abs(c["j_hz"]))
        peak["couplings"] = peak_couplings
    return peaks


def multiplicity_label(couplings) -> str:
    """Human-readable multiplicity for a peak: ``s``, ``d``, ``t``, ``dd``, ..."""
    if not couplings:
        return "s"
    return "".join(_MULTIPLICITY_LETTER.get(c["n"], "m") for c in couplings)


def format_couplings_by_atom(couplings, decimals=2) -> dict[int, str]:
    """Per-nucleus coupling text for the shielding table, keyed by atom index.

    Both partners of a pair are annotated, and heteronuclear couplings are kept here even
    though they do not split the simulated spectrum -- a C-H coupling is a real, useful
    number to read off the table.
    """
    by_atom: dict[int, list[tuple[float, str]]] = {}
    seen = set()
    for coupling in couplings or []:
        a, b = int(coupling["atom_a"]), int(coupling["atom_b"])
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        if pair in seen:
            continue  # a pair listed from both directions must not be shown twice
        seen.add(pair)
        value = float(coupling["j_hz"])
        by_atom.setdefault(a, []).append((value, f"{coupling['symbol_b']}{b}"))
        by_atom.setdefault(b, []).append((value, f"{coupling['symbol_a']}{a}"))

    formatted = {}
    for atom_index, entries in by_atom.items():
        entries.sort(key=lambda entry: -abs(entry[0]))
        formatted[atom_index] = ", ".join(
            f"{value:.{decimals}f} Hz ({partner})" for value, partner in entries)
    return formatted


def _multiplet_lines(peak, spectrometer_mhz):
    """Split one peak into its first-order multiplet lines as ``(shift_ppm, weight)``.

    Equivalent partners give n+1 lines with binomial weights. Enumerating every
    individual spin configuration instead produces 2**n duplicate lines and can exhaust
    memory before a spectrum is drawn, even though its observable multiplet is small.
    """
    lines = [(peak["shift"], 1.0)]
    for coupling in peak.get("couplings") or []:
        n = coupling["n"]
        j_ppm = coupling["j_hz"] / spectrometer_mhz
        components = [((k - n / 2) * j_ppm, math.comb(n, k) / 2**n)
                      for k in range(n + 1)]
        lines = [(position + offset, weight * fraction)
                 for position, weight in lines for offset, fraction in components]
    return lines


def generate_nmr_spectrum_interactive(peaks, element_symbol, spectrometer_mhz=400.0,
                                      linewidth_hz=1.0, points=20000, plot_range=None):
    """Simulated NMR spectrum for one element as a Plotly figure.

    Peak *area* is proportional to the number of equivalent nuclei, so integrations read
    correctly whether or not a signal is split. Every peak is annotated with the atom
    indices it came from -- the same 0-based indices the structure viewer labels.

    When :func:`assign_multiplets` has run, signals are split into first-order multiplets;
    otherwise every signal is a singlet. The spectrometer frequency is what converts a
    coupling in hertz into a splitting in ppm, so the same molecule looks different at
    400 and 600 MHz exactly as it does on a real instrument.

    Returns ``None`` when the molecule contains none of this element, so the caller can
    hide the plot rather than render an empty axis.
    """
    if not peaks:
        return None

    if not np.isfinite(spectrometer_mhz) or spectrometer_mhz <= 0:
        raise ValueError("Spectrometer frequency must be a positive finite number.")

    linewidth_ppm = max(linewidth_hz, 1e-6) / max(spectrometer_mhz, 1e-6)
    all_lines = [(peak, _multiplet_lines(peak, spectrometer_mhz)) for peak in peaks]
    positions = [position for _peak, lines in all_lines for position, _w in lines]
    if plot_range is None:
        span = max(positions) - min(positions)
        margin = max(0.5, 0.1 * span) if span else 0.5
        plot_range = (min(positions) - margin, max(positions) + margin)
    x = np.linspace(plot_range[0], plot_range[1], int(points))

    spectrum = np.zeros_like(x)
    for peak, lines in all_lines:
        for position, weight in lines:
            spectrum += peak["count"] * weight * lorentzian_nmr(x, position, linewidth_ppm)
    peak_max = spectrum.max()
    if peak_max > 0:
        spectrum = spectrum / peak_max

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=spectrum, mode="lines",
                                line=dict(color="black"), name="Spectrum"))
    for peak in peaks:
        # Annotate at the group's centre of gravity rather than at a line, so a symmetric
        # multiplet gets one label between its lines instead of one per line.
        index = int(np.argmin(np.abs(x - peak["shift"])))
        label = f"{element_symbol}-" + ",".join(str(i) for i in peak["atom_idxs"])
        suffix = multiplicity_label(peak.get("couplings"))
        if suffix != "s":
            label += f" ({suffix})"
        figure.add_annotation(x=peak["shift"], y=spectrum[index], text=label,
                              showarrow=True, arrowhead=1, ax=0, ay=-40, yshift=10,
                              font=dict(color="red"))
    figure.update_layout(
        # An NMR axis runs right to left: shielded nuclei (small delta) sit on the right.
        xaxis=dict(autorange="reversed", title="Chemical shift (ppm)"),
        yaxis=dict(title="Relative intensity"),
        title=f"{element_symbol} NMR spectrum ({spectrometer_mhz:g} MHz)",
        template="plotly_white",
        showlegend=False,
    )
    return figure


# ---------------------------------------------------------------------------
# What each calculation type can actually run
#
# The menus adapt to the calculation type, because the constraints genuinely differ:
# NMR shielding refuses range-separated functionals outright, TD-DFT wants them, and
# geometry optimization has no correlated-method path in this app at all. Encoding that
# as a table rather than as branching inside the event handler keeps the policy in one
# reviewable place and lets a test check every entry against the engine.
# ---------------------------------------------------------------------------

# Method families offered in the UI. "DFT" is the only one that consumes a functional.
METHOD_TYPES = ["HF", "DFT", "MP2", "CCSD", "CCSD(T)"]

# Emission needs the gradient of an *excited* state. Psi4 had only EOM-CCSD for that, at
# N^6 cost; PySCF has no EOM-CCSD gradient at all but does have analytic TD-DFT
# excited-state gradients on both CPU and GPU, so that is what this calculation type now
# uses -- cheaper, and consistent with the TD-DFT tab beside it.
EMISSION_METHOD_TYPE = "TD-DFT"

# Note on older result files: the Psi4 ancestor of this app spelled this calculation type
# "Emission (EOM-CCSD)", but no mapping is needed to read its output. The Result tab
# branches on which result *keys* are present, never on calculation_type, so a working
# directory copied across from it still renders correctly.

# Analytic Hessians and TD-DFT exist for restricted and unrestricted references only --
# neither pyscf.hessian nor pyscf.tdscf ships an ROHF/ROKS module, and neither does
# gpu4pyscf. Caught in the UI rather than by an exception deep inside the child process.
ROHF_UNSUPPORTED_TYPES = {FREQUENCY, TDDFT, EMISSION, NMR}

# NMR shielding is the most constrained thing this app runs: gpu4pyscf's implementation is
# cupy-only (PySCF has no CPU equivalent installed), indexes a 1-D occupation array so it
# needs a restricted reference, and reads mf.xc so it needs a KS object. Surfaced as
# constants because both the UI and the runner have to enforce them.
NMR_REQUIRES_GPU = True
NMR_METHOD_TYPES = ["DFT"]

# Functionals, all verified to resolve with pyscf.dft.libxc.parse_xc in this build.
# Dispersion is a separate control, so no name here carries a "-D3BJ" suffix (libxc
# silently ignores such a suffix, which would run the job without the correction).
GENERAL_FUNCTIONALS = [
    "B3LYP", "PBE0", "PBE", "BLYP", "BP86", "TPSS", "TPSSh", "SCAN", "r2SCAN",
    "M06-2X", "M06-L", "B97-D", "CAM-B3LYP", "wB97X-D", "wB97X-V", "wB97M-V",
]
# Excitation energies depend strongly on the functional, and range-separated hybrids are
# the defensible default (they fix the charge-transfer underestimation that plagues global
# hybrids), so the menu leads with them.
TDDFT_FUNCTIONALS = [
    "CAM-B3LYP", "wB97X-D", "wB97X-V", "wB97M-V", "PBE0", "B3LYP", "M06-2X", "TPSSh",
]
# NMR excludes every range-separated functional: gpu4pyscf's shielding code asserts
# omega == 0.0 and would otherwise abort mid-job. KT2 and KT3 lead because Keal and Tozer
# designed them specifically for shielding constants. test_options.py asserts this list
# contains nothing range-separated, so it cannot silently drift.
NMR_FUNCTIONALS = [
    "KT2", "KT3", "B3LYP", "PBE0", "BP86", "TPSS", "B97-D", "PBE", "BLYP", "OLYP",
]

# Basis sets, all verified to load for H, B, C, N, O, F, Si, P, S and Cl.
GENERAL_BASIS_SETS = [
    "STO-3G", "3-21G", "6-31G", "6-31G(d)", "6-31G(d,p)", "6-31+G(d,p)",
    "6-311G(d,p)", "6-311+G(d,p)", "cc-pVDZ", "cc-pVTZ", "cc-pVQZ",
    "aug-cc-pVDZ", "aug-cc-pVTZ", "def2-SVP", "def2-TZVP", "def2-TZVPP", "def2-QZVP",
]
# Excited states are diffuse, so a basis without augmentation systematically overestimates
# excitation energies. Only sets with diffuse functions (or def2 triple-zeta) are offered.
TDDFT_BASIS_SETS = [
    "6-31+G(d,p)", "6-311+G(d,p)", "aug-cc-pVDZ", "aug-cc-pVTZ",
    "def2-SVP", "def2-TZVP", "def2-TZVPP",
]
# Jensen's segmented polarization-consistent sets are built for this property and are the
# analogue of the pcSseg-* sets an ORCA user would reach for; PySCF ships pcseg but not
# pcSseg. Two obvious candidates are deliberately absent: 6-311+G(2d,p) has no Si/P/S/Cl in
# this build, and cc-pCVTZ has no hydrogen -- both unusable when one basis covers every atom.
NMR_BASIS_SETS = [
    "pcseg-1", "pcseg-2", "pcseg-3", "aug-pcseg-1", "aug-pcseg-2",
    "6-311+G(d,p)", "cc-pVTZ", "aug-cc-pVTZ", "def2-TZVP", "def2-TZVPP", "def2-QZVP",
]


class CalculationProfile:
    """What one calculation type offers, and what it falls back to.

    Attribute access rather than a bare dict so a typo raises instead of silently
    returning ``None`` into a Gradio dropdown.
    """

    __slots__ = ("method_types", "functionals", "basis_sets",
                 "default_functional", "default_basis", "default_method_type")

    def __init__(self, method_types, functionals, basis_sets,
                 default_functional, default_basis, default_method_type="DFT"):
        self.method_types = method_types
        self.functionals = functionals
        self.basis_sets = basis_sets
        self.default_functional = default_functional
        self.default_basis = default_basis
        self.default_method_type = default_method_type


# Method types are the app's real capability, not an aspiration. Optimization and frequency
# are HF/DFT only because the runner hands the SCF object straight to geomeTRIC and to
# mf.Hessian(); the correlated methods exist only on the single-point path.
CALCULATION_PROFILES: dict[str, CalculationProfile] = {
    SINGLE_POINT: CalculationProfile(
        METHOD_TYPES, GENERAL_FUNCTIONALS, GENERAL_BASIS_SETS, "B3LYP", "6-31G(d)"),
    GEOMETRY_OPTIMIZATION: CalculationProfile(
        ["HF", "DFT"], GENERAL_FUNCTIONALS, GENERAL_BASIS_SETS, "B3LYP", "6-31G(d)"),
    FREQUENCY: CalculationProfile(
        ["HF", "DFT"], GENERAL_FUNCTIONALS, GENERAL_BASIS_SETS, "B3LYP", "6-31G(d)"),
    TDDFT: CalculationProfile(
        ["DFT"], TDDFT_FUNCTIONALS, TDDFT_BASIS_SETS, "CAM-B3LYP", "def2-TZVP"),
    EMISSION: CalculationProfile(
        [EMISSION_METHOD_TYPE], TDDFT_FUNCTIONALS, TDDFT_BASIS_SETS,
        "CAM-B3LYP", "def2-SVP", default_method_type=EMISSION_METHOD_TYPE),
    NMR: CalculationProfile(
        NMR_METHOD_TYPES, NMR_FUNCTIONALS, NMR_BASIS_SETS, "B3LYP", "pcseg-2"),
}

DEFAULT_BASIS_SET = "6-31G(d)"


def calculation_profile(calculation_type: str) -> CalculationProfile:
    """Profile for ``calculation_type``, falling back to the single-point one."""
    return CALCULATION_PROFILES.get(calculation_type, CALCULATION_PROFILES[SINGLE_POINT])


def resolve_choice(current, choices, default):
    """Keep ``current`` if the new menu still offers it, otherwise use ``default``.

    Switching calculation type should not silently discard a deliberate choice that remains
    valid -- but it must not leave an invalid one selected either, because these dropdowns
    accept custom values and a stale entry would otherwise reach the engine and fail there
    (an NMR job left on CAM-B3LYP, say, which the shielding code refuses outright).
    """
    return current if current in choices else default


# geomeTRIC's named convergence criteria, mapped to the kwargs geometric_solver.kernel
# forwards to it. The first seven are sets defined in geometric/params.py; QCHEM and
# MOLPRO are not sets there but flags that change how the criteria are combined, which is
# why they pair a set with a boolean. Psi4's CFOUR has no geomeTRIC equivalent and is gone.
G_CONVERGENCE_KWARGS: dict[str, Any] = {
    "GAU": {"convergence_set": "GAU"},
    "GAU_LOOSE": {"convergence_set": "GAU_LOOSE"},
    "GAU_TIGHT": {"convergence_set": "GAU_TIGHT"},
    "GAU_VERYTIGHT": {"convergence_set": "GAU_VERYTIGHT"},
    "TURBOMOLE": {"convergence_set": "TURBOMOLE"},
    "NWCHEM_LOOSE": {"convergence_set": "NWCHEM_LOOSE"},
    "INTERFRAG_TIGHT": {"convergence_set": "INTERFRAG_TIGHT"},
    "QCHEM": {"convergence_set": "GAU", "qccnv": True},
    "MOLPRO": {"convergence_set": "GAU", "molcnv": True},
}
G_CONVERGENCES = list(G_CONVERGENCE_KWARGS)
DEFAULT_G_CONVERGENCE = "GAU"

# Continuum solvation models. SMD is the default: it is parameterised for solvation free
# energies rather than electrostatics alone, and gpu4pyscf implements it with gradients
# and Hessians. The PCM variants are offered because solvated TD-DFT is PCM-only.
SOLVENT_MODELS = ["SMD", "IEF-PCM", "C-PCM"]
DEFAULT_SOLVENT_MODEL = "SMD"

# Empirical dispersion corrections, as pyscf-dispersion spells them. This is a control of
# its own rather than a "-D3BJ" suffix on the functional name, because libxc silently
# ignores such a suffix: parse_xc("B3LYP-D3BJ") returns plain B3LYP, so a functional list
# carrying the suffix would quietly run every job without the correction it advertises.
DISPERSIONS = [("None", ""), ("D3(BJ)", "d3bj"), ("D3(zero)", "d3zero"), ("D4", "d4")]

# DFT integration grid. PySCF's default is 3; finer grids are worth offering because a
# coarse grid is a common cause of small spurious imaginary frequencies.
GRID_LEVELS = [("Coarse (3)", 3), ("Default (4)", 4), ("Fine (5)", 5)]
DEFAULT_GRID_LEVEL = 4

# Which nuclei to compute J-couplings between. Cost is driven by the number of distinct
# nuclei -- one set of CPHF solves each -- so this is the user's main lever: hydrogen alone
# is about twice as fast as hydrogen plus carbon, and only H-H couplings split a 1H
# spectrum anyway. Carbon is worth the extra time when the per-nucleus J table is the point.
J_COUPLING_NUCLEI_CHOICES = [("1H only (faster)", "H"), ("1H and 13C", "H,C")]
DEFAULT_J_COUPLING_NUCLEI = "H"

SCF_CONV_TOLS = [1e-8, 1e-9, 1e-10]
DEFAULT_SCF_CONV_TOL = 1e-9


def method_uses_functional(method_type: str) -> bool:
    """Whether ``method_type`` is driven by an exchange-correlation functional.

    Emission is the reason this is a function rather than ``== "DFT"`` written out at each
    site: it runs TD-DFT, so it very much has a functional, but its method type is spelled
    "TD-DFT". The rule was previously repeated at seven places and three of them omitted
    that case, which hid the functional dropdown on the one calculation type whose result
    depends on it most.
    """
    return method_type in ("DFT", EMISSION_METHOD_TYPE)


def pyscf_spin(multiplicity: int) -> int:
    """Convert a spin multiplicity to PySCF's ``spin``, which is 2S rather than 2S+1.

    Isolated into a named function used in exactly one place, so this off-by-one can only
    ever be wrong once. Getting it wrong does not raise -- it runs a different spin state
    and returns a perfectly plausible number.
    """
    return int(multiplicity) - 1


def pyscf_method(method_type: str, functional: str) -> str:
    """Return the engine-side method key for a UI method family.

    Unlike Psi4, PySCF does not take a method *name*: the family selects a class and the
    functional is set on the object. So this maps to a short key the runner branches on,
    and the functional travels in its own spec field.
    """
    return {"HF": "scf", "DFT": "dft", "TD-DFT": "dft", "MP2": "mp2",
            "CCSD": "ccsd", "CCSD(T)": "ccsd(t)"}.get(method_type, "scf")


def pyscf_atom_spec(mol: Chem.Mol) -> str:
    """Build the atom specification PySCF's ``gto.M(atom=...)`` parses.

    Atom order follows ``mol``'s atom order, which is the order the structure viewer
    labels on screen.

    Three things the Psi4 version of this function had to encode are simply absent here.
    Charge and multiplicity are separate ``Mole`` fields rather than a header line, so they
    cannot drift out of step with the spec's own fields. ``no_reorient``/``no_com`` are
    unnecessary because PySCF only shifts and rotates a molecule when symmetry is enabled,
    and the runner builds every ``Mole`` with ``symmetry=False`` -- so the coordinates the
    engine works in are the ones the user saw, and every geometry written back out stays
    in the input frame. And ``force_c1`` for excited states is moot for the same reason:
    with no irreps, there is no per-irrep root to lose track of.
    """
    conformer = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {position.x:>14.8f} "
                     f"{position.y:>14.8f} {position.z:>14.8f}")
    return "\n".join(lines)


def build_job_spec(
    *,
    calculation_type: str,
    geometry: str,
    structure_file: str,
    method_type: str,
    functional: str,
    basis_set: str,
    reference: str,
    charge: int,
    multiplicity: int,
    n_threads: int,
    memory_gb: float,
    use_gpu: bool = False,
    density_fit: bool | None = None,
    dispersion: str = "",
    grid_level: int = DEFAULT_GRID_LEVEL,
    scf_conv_tol: float = DEFAULT_SCF_CONV_TOL,
    scf_max_cycle: int = 100,
    verbose: int = 4,
    geom_maxiter: int = 50,
    g_convergence: str = DEFAULT_G_CONVERGENCE,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    n_states: int = 10,
    tda: bool = False,
    root: int = 1,
    use_solvation: bool = False,
    solvent_model: str = DEFAULT_SOLVENT_MODEL,
    solvent: str = "Water",
    compute_j_couplings: bool = False,
    j_coupling_nuclei: str = "H",
    save_wavefunction: bool = True,
    save_molden: bool = True,
    save_arrays: bool = True,
) -> dict[str, Any]:
    """Assemble the JSON-serializable job spec handed to the runner.

    Keyword-only because the argument list is long and homogeneous (several ints, several
    strings) -- positional calls would be easy to mis-order and hard to spot.
    """
    if density_fit is None:
        # Density fitting is the default for DFT, as it was in the Psi4 version via
        # SCF_TYPE=DF, and is much of what makes the GPU path worth having.
        density_fit = method_uses_functional(method_type)
    return {
        "version": 2,
        "calculation_type": calculation_type,
        "geometry": geometry,
        "structure_file": structure_file,
        "method_type": method_type,
        "method": pyscf_method(method_type, functional),
        "functional": functional,
        "basis_set": basis_set,
        "reference": reference,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "n_threads": int(n_threads),
        "memory_gb": float(memory_gb),
        "use_gpu": bool(use_gpu),
        "density_fit": bool(density_fit),
        "dispersion": dispersion or "",
        "grid_level": int(grid_level),
        "scf_conv_tol": float(scf_conv_tol),
        "scf_max_cycle": int(scf_max_cycle),
        "verbose": int(verbose),
        "geom_maxiter": int(geom_maxiter),
        "g_convergence": g_convergence,
        "temperature": float(temperature),
        "pressure": float(pressure),
        "n_states": int(n_states),
        "tda": bool(tda),
        "root": int(root),
        "use_solvation": bool(use_solvation),
        "solvent_model": solvent_model,
        "solvent": solvent,
        "compute_j_couplings": bool(compute_j_couplings),
        "j_coupling_nuclei": j_coupling_nuclei,
        "save_wavefunction": bool(save_wavefunction),
        "save_molden": bool(save_molden),
        "save_arrays": bool(save_arrays),
    }


def write_json(file_path: str, payload: dict[str, Any]) -> None:
    """Write ``payload`` as UTF-8 JSON.

    ``default=str`` is a backstop: the runner builds results out of numpy scalars in a
    few places, and a stray ``np.float64`` would otherwise abort a finished calculation
    at the very last step, throwing away hours of compute over a formatting detail.
    """
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(file_path: str) -> dict[str, Any]:
    """Read a UTF-8 JSON file written by :func:`write_json`."""
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def result_base_name(file_name: str) -> str:
    """Strip the ``.result.json`` / ``.job.json`` suffix from a job artifact name.

    ``os.path.splitext`` would only remove ``.json`` and leave the ``.result`` behind, so
    the sibling ``.log`` / ``.npy`` of a job could not be located from its result file.
    """
    for suffix in (RESULT_SUFFIX, JOB_SUFFIX):
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return os.path.splitext(file_name)[0]


def mol_from_symbols_and_coords(symbols, coords) -> Chem.Mol:
    """Build a bonded RDKit ``Mol`` from element symbols and angstrom coordinates.

    Used to rebuild the molecule from a result file for the orbital viewer, so the atoms
    drawn are in exactly the frame the cube grid was computed on. Bonds are perceived by
    :func:`add_bonds` because a result file stores geometry only, not connectivity.
    """
    mol = Chem.RWMol()
    for symbol in symbols:
        mol.AddAtom(Chem.Atom(str(symbol).capitalize()))

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    mol.AddConformer(conformer)

    return add_bonds(mol.GetMol())
