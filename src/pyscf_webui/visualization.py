"""nglview rendering, shared by the structure viewer and the orbital/density viewer.

nglview is a Jupyter widget, so it cannot be embedded in a Gradio page directly. Both
viewers therefore take the same route: build the widget, write it to a standalone HTML
file under ``static/``, and return an ``<iframe>`` pointing at it. The iframe URL carries
a timestamp because the browser would otherwise serve the previous render from cache --
the file name never changes.
"""
import contextlib
import os
import time

import nglview

from .utils import mol_from_symbols_and_coords

# Where the generated viewer HTML goes. Relative to the process working directory, which
# is what app.py mounts at /static.
_STATIC_DIR = os.path.join(".", "static")

# The non-orbital selections the viewer understands. Defined here rather than in the
# Result tab because this module owns both halves of the mapping: which cubegen task a
# selection needs, and which file name that task produces.
TOTAL_DENSITY = "Electron density"
SPIN_DENSITY = "Spin density"
ELECTROSTATIC_POTENTIAL = "Electrostatic potential"

# Which cubegen task each non-orbital selection needs. Psi4's cubeprop conflated the two
# densities -- both selections asked for the DENSITY task and were told apart only by
# which of the files it happened to emit was read afterwards. pyscf.tools.cubegen is
# handed an explicit density matrix, so the total and spin densities are genuinely
# distinct requests and cannot be confused.
CUBE_TASKS = {
    TOTAL_DENSITY: "density",
    SPIN_DENSITY: "spin",
    ELECTROSTATIC_POTENTIAL: "esp",
}

# cubegen writes wherever it is told, so this module chooses the names rather than having
# to reverse-engineer them. That is what lets the lookup below be exact.
_TASK_FILE_NAMES = {
    "density": "density.cube",
    "spin": "spin_density.cube",
    "esp": "esp.cube",
}


def cube_file_name(task: str, orbital: int | None = None) -> str:
    """File name the runner writes, and the viewer reads, for ``task``.

    Shared by both halves so the producer and the consumer cannot disagree: the runner
    imports this to name its output, and :func:`cube_files_for_selection` uses it to find
    that output again.
    """
    if task == "orbital":
        # Zero-padded so a directory listing sorts in orbital order rather than
        # lexicographically putting MO 10 before MO 2.
        return f"mo_{int(orbital):04d}.cube"
    return _TASK_FILE_NAMES[task]


def _iframe(src: str, width: int = 600, height: int = 600) -> str:
    """Cache-busted iframe pointing at a file under ``/static``."""
    return (f'<iframe src="{src}?ts={int(time.time())}" height="{height}" width="{width}" '
            f'title="NGL View" style="border:none;"></iframe>')


@contextlib.contextmanager
def _transient_widgets():
    """Unregister any widgets created inside the block once it finishes.

    This is not optional housekeeping. ``nglview.write_html`` ends up in ipywidgets'
    ``embed_data``, which -- given no explicit state -- serializes **every widget still
    registered in the process**, not just the one it was handed. Each render would
    therefore embed all of its predecessors: the file grows by about a full view per
    render (roughly 1 MB with real cube data) and carries stale isosurfaces from earlier
    selections into the current one. In a long-lived server that compounds without bound.

    The widgets are unregistered rather than closed. ``close()`` severs the comm while
    leaving the object referenced, and the *next* render then fails serializing a widget
    whose ``model_id`` no longer resolves.

    The registry is a private ipywidgets detail, so a version that moves it degrades to
    the old accumulating behaviour rather than breaking the viewer.
    """
    try:
        from ipywidgets.widgets import widget as widget_module
        registry = widget_module._instances
    except Exception:  # pragma: no cover - depends on the ipywidgets version
        registry = None

    before = set(registry) if registry is not None else set()
    try:
        yield
    finally:
        if registry is not None:
            for key in set(registry) - before:
                registry.pop(key, None)


def _write_view(view, relative_path: str) -> str:
    """Write ``view`` to ``static/<relative_path>`` and return the iframe HTML."""
    output_path = os.path.join(_STATIC_DIR, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    nglview.write_html(output_path, [view])
    return _iframe("/static/" + relative_path.replace(os.sep, "/"))


def render_structure_html(mol) -> str:
    """Render ``mol`` with per-atom index labels and return the embedding iframe.

    The labels are what let a user map what they see to the geometry the engine receives: this
    is the same loader the Calculation tab uses, so atom *i* here is atom *i* there.

    ``labelType="text"`` with an explicit ``labelText`` is what makes this work.
    ``labelType="atomname"`` reads the names out of the PDB block ``show_rdkit``
    generates, and RDKit numbers those per element (C1, C2, O1, ...) -- so the third atom
    of ethanol came out as "O1", the first oxygen, rather than "O3".
    """
    with _transient_widgets():
        view = nglview.show_rdkit(mol)
        atom_labels = [f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in mol.GetAtoms()]
        view.add_representation("label", labelType="text", labelText=atom_labels,
                                color="black", showBackground=False)
        return _write_view(view, "structure.html")


def cube_files_for_selection(output_directory: str, selection: str) -> list[str]:
    """Locate the cube file ``cubegen`` produced for ``selection``.

    An exact lookup, because :func:`cube_file_name` chose the name in the first place.
    Psi4's cubeprop named orbital cubes ``Psi_a_<n>_<n>-<irrep>.cube``, embedding an irrep
    label this module could not predict, so the old version of this function had to glob
    and hope the directory held exactly one candidate.
    """
    if selection.startswith("MO "):
        name = cube_file_name("orbital", int(selection.split()[1]))
    else:
        task = CUBE_TASKS.get(selection)
        name = cube_file_name(task) if task else None
    if name is None:
        return []
    path = os.path.join(output_directory, name)
    return [path] if os.path.isfile(path) else []


def render_cube_html(result, output_directory: str, selection: str,
                     color1: str = "#0000ff", color2: str = "#ff0000",
                     opacity: float = 0.8, isolevel: float = 0.05) -> str:
    """Render a cube-file isosurface over the molecule and return the embedding iframe.

    The molecule is rebuilt from the result's own geometry rather than from the original
    structure file, so the atoms are in the same frame as the cube grid. (The Mole is built
    with symmetry=False, so PySCF never reorients or re-centres it and the two frames
    coincide -- but taking coordinates from the result keeps that true if it ever changes.)

    Signed quantities -- molecular orbitals and the electrostatic potential -- are drawn
    as two surfaces at ``+isolevel`` and ``-isolevel`` in the two chosen colours; a
    density is unsigned and gets a single surface.

    ``isolevelType="value"`` is essential and easy to miss. NGL's surface representation
    defaults to ``isolevelType="sigma"``, which multiplies the given number by the RMS of
    the grid rather than using it directly -- so a chemist's 0.02 a.u. contour silently
    becomes 0.02 sigma, roughly 0.0008 a.u., and renders a huge diffuse blob that clips
    against the edge of the grid box (which is what makes the two lobes of a symmetric
    orbital come out visibly unequal). Setting the type to "value" makes the number mean
    the wavefunction amplitude it appears to mean, and makes +/-isolevel enclose exactly
    mirrored volumes.
    """
    cube_files = cube_files_for_selection(output_directory, selection)
    if not cube_files:
        raise FileNotFoundError(f"No cube file was produced for '{selection}'.")

    mol = mol_from_symbols_and_coords(result["symbols"], result["geometry_angstrom"])
    cube_path = cube_files[0]
    # A total electron density is the only unsigned quantity here. Orbitals and the ESP
    # obviously change sign, and so does a spin density -- alpha minus beta goes negative
    # wherever spin polarization is inverted, which is exactly the interesting part.
    signed = selection != TOTAL_DENSITY

    with _transient_widgets():
        view = nglview.show_rdkit(mol)
        view.add_component(cube_path)
        view.component_1.update_surface(opacity=opacity, color=color1,
                                        isolevelType="value", isolevel=abs(isolevel))
        if signed:
            view.add_component(cube_path)
            view.component_2.update_surface(opacity=opacity, color=color2,
                                            isolevelType="value", isolevel=-abs(isolevel))
        view.camera = "orthographic"

        return _write_view(view, os.path.join("cubes", "view.html"))
