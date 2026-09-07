"""PySCF execution, in a child process.

This module and :mod:`pyscf_webui.engine` are the **only** places ``pyscf`` /
``gpu4pyscf`` are imported. The web server never imports them; the Calculation tab
launches this module as

    python -u -m pyscf_webui.runner run <job.json>
    python -u -m pyscf_webui.runner cubegen <cube.json>
    python -u -m pyscf_webui.runner gpu-probe

and waits on the process. Running out-of-process buys three things that an in-process
thread cannot provide:

* **Cancellation.** An SCF or a Hessian is a long-running native call with no interrupt
  hook, so the only reliable Stop is to kill the process tree.
* **Crash isolation.** A CUDA out-of-memory or an illegal memory access aborts the
  process outright. In-process, that would take the whole web server with it.
* **Clean state.** cupy's device memory pool, ``lib.num_threads`` and ``PYSCF_TMPDIR``
  are process-global, so successive runs in one process leak into each other. A fresh
  child cannot.

The child communicates only through files in the working directory: the job spec going
in, and ``<base>.log`` (the engine's own output), ``<base>.result.json`` (structured
results) plus ``.chk``/``.molden``/``.h5``/``.xyz`` artifacts coming back. Exit code 0
means the result file holds a completed calculation; non-zero means it holds
``status: "failed"`` with an error message. A result file is written in **both** cases,
so the Result tab can always explain what happened rather than showing an empty dropdown.

Unlike Psi4, PySCF writes no output file of its own -- it prints to a stream. The parent
therefore points this child's stdout **and** stderr at ``<base>.log``, and this module
runs with ``-u``. That single arrangement captures all three writers at once: PySCF's
``lib.logger`` (which flushes after every record), geomeTRIC's optimizer table (which
goes to ``sys.stderr``), and any Python traceback. The result is a live, human-readable
log the Calculation tab can tail, without any module here having to manage a file handle.

Engine imports deliberately happen inside the functions rather than at module scope, so
the module (and its file-naming helpers) can be imported and unit-tested in an
environment where PySCF is not installed.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

from .utils import (
    CALCULATION_TYPES,
    EMISSION,
    FREQUENCY,
    NMR,
    GEOMETRY_OPTIMIZATION,
    G_CONVERGENCE_KWARGS,
    JOB_SUFFIX,
    RESULT_SUFFIX,
    TDDFT,
    method_uses_functional,
    read_json,
    write_json,
)

# Result schema version. Bumped from the Psi4 app's 1 because the engine fields are new
# and ``wavefunction_file`` now names a PySCF chkfile rather than a Psi4 .npy. Every key
# the Result tab reads kept its name, so a version 1 file still renders.
RESULT_VERSION = 2


def job_base_name(job_path: str) -> str:
    """Strip the job-spec suffix from ``job_path`` to get the artifact base name.

    ``os.path.splitext`` alone would turn ``opt.job.json`` into ``opt.job``, so every
    artifact of that run would be named ``opt.job.log``, ``opt.job.chk`` and so on.
    """
    name = os.path.basename(job_path)
    for suffix in (JOB_SUFFIX, ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _write_xyz(file_path: str, symbols, coords, charge: int, multiplicity: int) -> None:
    """Write a single geometry in this app's XYZ convention.

    The first line is ``"<charge> <multiplicity>"`` rather than an atom count, matching
    :func:`pyscf_webui.utils.conformer_to_xyz_file`, so an optimized geometry can be fed
    straight back into a follow-up calculation with its charge and spin state intact.
    """
    lines = [f"{int(charge)} {int(multiplicity)}"]
    for symbol, (x, y, z) in zip(symbols, coords):
        lines.append(f"{symbol} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_trajectory_xyz(file_path: str, symbols, frames) -> None:
    """Write an optimization trajectory as a standard multi-frame XYZ file.

    Standard format (atom count + comment line per frame) rather than this app's own
    convention, because the point of a trajectory file is to open it in an external
    viewer that can animate the frames. The in-app reader understands it too and shows
    the final frame.
    """
    lines = []
    for step, coords in enumerate(frames):
        lines.append(str(len(symbols)))
        lines.append(f"Step {step}")
        for symbol, (x, y, z) in zip(symbols, coords):
            lines.append(f"{symbol} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _atom_symbols(mol) -> list[str]:
    """Element symbols in PySCF's atom order (which is the input order)."""
    return [mol.atom_symbol(i) for i in range(mol.natm)]


def _geometry_angstrom(mol) -> list[list[float]]:
    """Current geometry of ``mol`` as a plain nested list, in angstrom."""
    from pyscf import lib

    return [[float(x) * lib.param.BOHR for x in row] for row in mol.atom_coords()]


def _print_header(spec: dict[str, Any], device: str) -> None:
    """Write a summary block at the top of the log.

    PySCF prints plenty about the basis and each SCF cycle but nothing about *the job*.
    Echoing the request makes ``<base>.log`` a self-contained record -- which is the whole
    point of generating it, rather than leaving the user to cross-reference the job spec.
    """
    import pyscf

    lines = [
        "=" * 78,
        "  pyscf-webui calculation",
        "=" * 78,
        f"  calculation type : {spec.get('calculation_type')}",
        f"  method           : {spec.get('method_type')}"
        + (f" / {spec.get('functional')}"
           if method_uses_functional(spec.get("method_type")) else ""),
        f"  basis set        : {spec.get('basis_set')}",
        f"  reference        : {spec.get('reference')}",
        f"  charge / mult    : {spec.get('charge')} / {spec.get('multiplicity')}",
        f"  density fitting  : "
        f"{spec.get('density_fit', method_uses_functional(spec.get('method_type')))}",
        "  solvation        : "
        + (f"{spec.get('solvent_model', 'SMD')} / {spec.get('solvent')}"
           if spec.get("use_solvation") else "none"),
        f"  device           : {device}",
        f"  pyscf            : {pyscf.__version__}",
    ]
    if device != "cpu":
        try:
            import gpu4pyscf

            lines.append(f"  gpu4pyscf        : {gpu4pyscf.__version__}")
        except Exception:
            pass
    lines += ["=" * 78, ""]
    print("\n".join(lines), flush=True)


def _optimize(mf, spec: dict[str, Any]):
    """Run a geomeTRIC optimization, returning ``(converged, mol_eq, energies, frames)``.

    ``frames`` are in angstrom. The callback fires inside geomeTRIC's engine, where
    ``coords`` is in **Bohr** -- the previous version of this app wrote optimizer frames
    out unconverted, making every structure it produced 1.89x too large, so the conversion
    happens here, once.

    ``kernel`` returns an unconverged geometry when ``geom_maxiter`` is exhausted, so
    those steps can still be reported. Its separate ``assert_convergence`` switch checks
    the electronic solver behind EACH gradient; disabling it would let unconverged SCF
    or TD-DFT solutions steer the optimization.
    """
    import numpy
    from pyscf import lib
    from pyscf.geomopt import geometric_solver

    energies: list[float] = []
    frames: list = []

    def callback(envs):
        energies.append(float(envs["energy"]))
        coords = numpy.asarray(envs["coords"], dtype=float).reshape(-1, 3) * lib.param.BOHR
        frames.append(coords.tolist())

    kwargs = dict(G_CONVERGENCE_KWARGS.get(spec.get("g_convergence", "GAU"),
                                           {"convergence_set": "GAU"}))
    converged, mol_eq = geometric_solver.kernel(
        mf, maxsteps=int(spec.get("geom_maxiter", 50)), callback=callback,
        assert_convergence=True, **kwargs
    )
    return bool(converged), mol_eq, energies, frames


def _reconverge_at(engine, spec: dict[str, Any], mol_eq, use_gpu: bool):
    """Rebuild and re-converge the SCF at an optimized geometry.

    This is **not** optional bookkeeping. ``geometric_solver.kernel`` optimizes a *copy* of
    the molecule and leaves the method object pointing at the original geometry -- its own
    docstring says "method.mol should be still in its original geometry". Skipping this
    step would give a result file whose energy came from the optimization but whose
    orbitals, dipole, checkpoint and cube grids all silently describe the **input**
    structure.

    The method is rebuilt rather than ``mf.reset()``-ed so that density fitting, the
    solvent wrapper and the DFT grids are all reconstructed consistently for the new
    geometry instead of being carried across in a half-updated state.
    """
    mf = engine.build_scf(spec, mol_eq, use_gpu)
    energy = engine.run_scf(mf)
    return mf, energy


def run_job(job_path: str) -> int:
    """Execute the job described by ``job_path``; return the process exit code.

    Always writes ``<base>.result.json``, whether the calculation succeeded or failed, so
    the Result tab has something to explain.
    """
    from . import engine

    working_directory = os.path.dirname(os.path.abspath(job_path))
    base_name = job_base_name(job_path)
    log_path = os.path.join(working_directory, base_name + ".log")
    result_path = os.path.join(working_directory, base_name + RESULT_SUFFIX)

    result: dict[str, Any] = {
        "version": RESULT_VERSION,
        "status": "failed",
        "error": None,
        "engine": "pyscf",
        "log_file": os.path.basename(log_path),
    }

    started = time.time()
    arrays: dict[str, Any] = {}
    use_gpu = False
    try:
        # Read and validate inside the result-file guard, so a damaged job spec or an
        # unwritable scratch directory still produces an explanation in the Result tab.
        spec = read_json(job_path)
        if not isinstance(spec, dict):
            raise ValueError("The job specification must be a JSON object.")
        result.update({key: spec.get(key) for key in (
            "calculation_type", "method_type", "method", "functional", "basis_set",
            "reference", "charge", "multiplicity", "structure_file",
        )})
        result["solvent"] = spec.get("solvent") if spec.get("use_solvation") else None
        result["solvent_model"] = (
            spec.get("solvent_model") if spec.get("use_solvation") else None
        )
        calculation_type = spec.get("calculation_type")
        if calculation_type not in CALCULATION_TYPES:
            raise ValueError(f"Unknown calculation type {calculation_type!r}.")

        # PySCF reads lib.param.TMPDIR at import time, so this must precede its import.
        # Scratch and geomeTRIC's temporary directory stay inside the working directory.
        scratch_dir = os.path.join(working_directory, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        os.environ["PYSCF_TMPDIR"] = scratch_dir

        import pyscf
        from pyscf import lib

        lib.num_threads(int(spec.get("n_threads", 1)))
        result["engine_version"] = pyscf.__version__

        use_gpu = bool(spec.get("use_gpu", False)) and engine.gpu_available()
        if spec.get("use_gpu") and not use_gpu:
            print("Warning: a GPU run was requested but gpu4pyscf is unavailable; "
                  "falling back to the CPU.", flush=True)
        device = engine.device_name() if use_gpu else "cpu"
        result["device"] = device
        _print_header(spec, device)

        mol = engine.build_mol(spec, verbose=int(spec.get("verbose", 4)))
        mf = engine.build_scf(spec, mol, use_gpu)

        xc_used, dispersion_used = engine.split_dispersion(spec.get("functional") or "")
        uses_functional = method_uses_functional(spec.get("method_type"))
        result["xc"] = xc_used if uses_functional else None
        result["dispersion"] = dispersion_used
        result["density_fit"] = bool(spec.get("density_fit", uses_functional))

        td = None
        if calculation_type == GEOMETRY_OPTIMIZATION:
            converged, mol_eq, energies, frames = _optimize(mf, spec)
            mf, energy = _reconverge_at(engine, spec, mol_eq, use_gpu)
            mol = mf.mol
            result["optimization"] = {
                "energies_hartree": energies,
                "n_steps": len(energies),
                "converged": converged,
            }
            trajectory_path = os.path.join(working_directory, base_name + "_trajectory.xyz")
            try:
                _write_trajectory_xyz(trajectory_path, _atom_symbols(mol), frames)
                result["trajectory_xyz_file"] = os.path.basename(trajectory_path)
            except Exception as exc:
                print(f"Could not write the optimization trajectory ({exc}).", file=sys.stderr)
            arrays["opt_energies"] = energies

        elif calculation_type == FREQUENCY:
            # A frequency analysis is only meaningful at a stationary point, so the
            # geometry is optimized first and the Hessian taken there. Taking it at an
            # arbitrary input geometry is what produces spurious imaginary modes and
            # renders the thermochemistry meaningless.
            converged, mol_eq, energies, _frames = _optimize(mf, spec)
            mf, energy = _reconverge_at(engine, spec, mol_eq, use_gpu)
            mol = mf.mol
            result["optimization"] = {
                "energies_hartree": energies,
                "n_steps": len(energies),
                "converged": converged,
            }
            arrays["opt_energies"] = energies
            if not converged:
                print("Warning: the geometry optimization did not converge; the "
                      "frequencies below are not those of a stationary point.", flush=True)
            fragment, frequency_arrays = engine.frequency_analysis(mf, spec, use_gpu)
            result.update(fragment)
            arrays.update(frequency_arrays)

        elif calculation_type == TDDFT:
            energy = engine.run_scf(mf)
            td = engine.build_tddft(mf, spec)
            fragment, excitation_arrays = engine.excitation_results(td)
            result.update(fragment)
            arrays.update(excitation_arrays)

        elif calculation_type == NMR:
            # A property of the structure as given -- no optimization, matching how every
            # other code presents an NMR job. Refused up front rather than part-way in:
            # every unsupported combination otherwise fails from inside the library with
            # an exception that says nothing about what the user should change.
            reason = engine.nmr_unavailable_reason(spec, use_gpu)
            if reason:
                raise RuntimeError(reason)
            energy = engine.run_scf(mf)
            fragment, nmr_arrays = engine.nmr_shielding(mf, spec)
            arrays.update(nmr_arrays)
            if spec.get("compute_j_couplings"):
                # Guarded separately from the shielding: J-couplings are the expensive,
                # optional half of an NMR job and run on the CPU, so losing them must not
                # discard shieldings that already succeeded.
                try:
                    couplings, j_arrays = engine.spin_spin_couplings(mf, spec)
                    fragment["nmr"]["couplings"] = couplings
                    arrays.update(j_arrays)
                except Exception as exc:
                    print(f"Warning: J-couplings failed ({type(exc).__name__}: {exc}); "
                          f"reporting shieldings only.", file=sys.stderr)
            result.update(fragment)

        elif calculation_type == EMISSION:
            mf, energy, emission_fragment = _run_emission(engine, spec, mf, use_gpu)
            mol = mf.mol
            result.update(emission_fragment)

        else:  # SINGLE_POINT
            energy = engine.run_scf(mf)
            if spec.get("method_type") in ("MP2", "CCSD", "CCSD(T)"):
                energy = engine.correlated_energy(spec, mf)

        result["converged"] = bool(getattr(mf, "converged", True))
        symbols = _atom_symbols(mol)
        result["symbols"] = symbols
        result["geometry_angstrom"] = _geometry_angstrom(mol)
        result["energy_hartree"] = float(energy)
        result.update(engine.scf_summary(mf))

        if calculation_type in (GEOMETRY_OPTIMIZATION, FREQUENCY, EMISSION):
            optimized_path = os.path.join(working_directory, base_name + "_optimized.xyz")
            # Charge and multiplicity are taken from the Mole the calculation actually
            # ran with rather than from the spec, so the saved structure can never be
            # labelled with a spin state the numbers do not belong to.
            try:
                _write_xyz(optimized_path, symbols, result["geometry_angstrom"],
                           int(mol.charge), int(mol.spin) + 1)
                result["optimized_xyz_file"] = os.path.basename(optimized_path)
            except Exception as exc:
                print(f"Could not write the optimized geometry ({exc}).", file=sys.stderr)

        _write_artifacts(engine, spec, mol, mf, energy, arrays,
                         working_directory, base_name, result)

        result["status"] = "completed"
        exit_code = 0
    except Exception as exc:
        if isinstance(exc, engine.ConvergenceError):
            result["converged"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        # Creating a CUDA context solely to clean up a CPU job can fail on a broken
        # optional GPU stack before the otherwise successful result gets written.
        if use_gpu:
            _release_gpu_memory()

    result["duration_seconds"] = time.time() - started
    write_json(result_path, result)
    print(f"\nResult written to {os.path.basename(result_path)} "
          f"({result['status']}, {result['duration_seconds']:.1f} s)", flush=True)
    return exit_code


def _run_emission(engine, spec: dict[str, Any], mf, use_gpu: bool):
    """Optimize the excited state, then read the emission off that geometry.

    Psi4 used EOM-CCSD here because it was the only excited-state gradient it had. PySCF
    has none at all for EOM-CCSD, but it *does* have analytic TD-DFT excited-state
    gradients on both the CPU (``pyscf.grad.tdrks``) and the GPU
    (``gpu4pyscf.grad.tdrks``) -- so this branch is both cheaper and better matched to the
    TD-DFT tab next to it.

    What is computed: the excited state is relaxed, then *both* surfaces are evaluated at
    that relaxed geometry. The gap between them is the vertical emission, and the
    difference between it and the vertical absorption at the ground-state geometry is the
    Stokes shift.
    """
    root = int(spec.get("root", 1))

    engine.run_scf(mf)
    td = engine.build_tddft(mf, spec)
    absorption_hartree = float(engine.to_numpy(td.e)[root - 1])

    # gpu4pyscf's excited-state gradient scanner asserts that it is running on the GPU, so
    # the CPU route has to be a genuinely separate object graph rather than a to_cpu() of
    # this one. build_scf already returned CPU classes when use_gpu is false, so simply
    # taking the gradient of the TD object we have does the right thing on both paths.
    scanner = td.nuc_grad_method().as_scanner(state=root)
    converged, mol_eq, energies, _frames = _optimize(scanner, spec)
    if not converged:
        print(f"Warning: the excited-state optimization did not converge in "
              f"{spec.get('geom_maxiter', 50)} steps.", flush=True)

    mf, ground_energy = _reconverge_at(engine, spec, mol_eq, use_gpu)
    td = engine.build_tddft(mf, spec)
    excitation_hartree = float(engine.to_numpy(td.e)[root - 1])
    excited_energy = ground_energy + excitation_hartree

    wavenumber = excitation_hartree * engine.HARTREE2WAVENUMBER
    absorption_wavenumber = absorption_hartree * engine.HARTREE2WAVENUMBER
    try:
        oscillator = float(
            engine.to_numpy(td.oscillator_strength(gauge="length"))[root - 1])
    except Exception:
        oscillator = None

    fragment = {
        "emission": {
            "root": root,
            "converged": converged,
            "excited_energy_hartree": excited_energy,
            "ground_energy_at_excited_geometry_hartree": ground_energy,
            "emission_energy_hartree": excitation_hartree,
            "emission_energy_ev": excitation_hartree * engine.HARTREE2EV,
            "wavenumber_cm1": wavenumber,
            "wavelength_nm": (1.0e7 / wavenumber) if wavenumber > 0 else None,
            "oscillator_strength": oscillator,
            "absorption_wavelength_nm": (
                (1.0e7 / absorption_wavenumber) if absorption_wavenumber > 0 else None),
            "stokes_shift_cm1": absorption_wavenumber - wavenumber,
        }
    }
    # The excited state is the one that was optimized, so it is the energy the result
    # file reports as "the" energy of this job.
    return mf, excited_energy, fragment


def _write_artifacts(engine, spec, mol, mf, energy, arrays, working_directory,
                     base_name, result) -> None:
    """Write the checkpoint, Molden and HDF5 artifacts for a finished job.

    Each write is independently guarded. None of these files is needed for the result
    itself, so a failure to produce one -- a full disk, an exotic basis Molden cannot
    represent -- degrades that one feature rather than discarding a calculation that has
    otherwise succeeded. The Psi4 version made the same choice for its wavefunction file.
    """
    if spec.get("save_wavefunction", True):
        try:
            result["wavefunction_file"] = engine.save_chkfile(
                mol, mf, os.path.join(working_directory, base_name + ".chk"))
            result["wavefunction_format"] = "pyscf-chkfile"
        except Exception as exc:
            print(f"Could not save the wavefunction ({exc}); orbital visualization "
                  f"will be unavailable for this calculation.", file=sys.stderr)
            result["wavefunction_file"] = None

    if spec.get("save_molden", True):
        try:
            result["molden_file"] = engine.save_molden(
                mol, engine.to_numpy(mf.mo_coeff), engine.to_numpy(mf.mo_occ),
                engine.to_numpy(mf.mo_energy),
                os.path.join(working_directory, base_name + ".molden"))
        except Exception as exc:
            print(f"Could not write the Molden file ({exc}).", file=sys.stderr)

    if "norm_mode" in arrays:
        try:
            result["frequency_molden_file"] = engine.save_frequency_molden(
                mol, arrays, os.path.join(working_directory, base_name + ".freq.molden"))
        except Exception as exc:
            print(f"Could not write the normal-mode file ({exc}).", file=sys.stderr)

    if arrays and spec.get("save_arrays", True):
        try:
            result["arrays_file"] = engine.save_arrays(
                arrays, os.path.join(working_directory, base_name + ".arrays.h5"))
        except Exception as exc:
            print(f"Could not write the array sidecar ({exc}).", file=sys.stderr)


def _release_gpu_memory() -> None:
    """Best-effort release of cupy's device memory pool.

    Stands in for ``psi4.core.clean()``. The process is about to exit anyway, so this
    matters mainly when a run fails early and something else on the machine wants the
    device back promptly.
    """
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass  # never mask the real outcome of the job


def run_cubegen(spec_path: str) -> int:
    """Generate cube files from a saved wavefunction.

    Kept separate from :func:`run_job` so orbitals can be explored *after* a calculation
    has finished, without paying to generate every cube up front or re-running the
    calculation just to get the wavefunction back.

    Two things the Psi4 version had to work around are simply absent here: the electrostatic
    potential needs no auxiliary basis rebuilt onto the wavefunction, and the output files
    are named by this module rather than by the engine, so the viewer can look them up
    exactly instead of globbing for an irrep label it cannot predict.
    """
    from .visualization import cube_file_name

    spec = read_json(spec_path)
    try:
        from pyscf.tools import cubegen

        from . import engine

        mol, mo_coeff, mo_occ, _mo_energy = engine.load_chkfile(spec["wavefunction_path"])
        dm_alpha, dm_beta = engine.density_matrices(mo_coeff, mo_occ)

        output_directory = spec["output_directory"]
        os.makedirs(output_directory, exist_ok=True)
        # cubegen's `resolution` is in Bohr, which is the same unit Psi4's
        # CUBIC_GRID_SPACING used, so the existing slider carries over unchanged.
        resolution = float(spec.get("grid_spacing", 0.2))

        for task in spec["tasks"]:
            path = os.path.join(output_directory, cube_file_name(task, spec.get("orbital")))
            if task == "density":
                cubegen.density(mol, path, dm_alpha + dm_beta, resolution=resolution)
            elif task == "spin":
                cubegen.density(mol, path, dm_alpha - dm_beta, resolution=resolution)
            elif task == "esp":
                cubegen.mep(mol, path, dm_alpha + dm_beta, resolution=resolution)
            elif task == "orbital":
                import numpy

                coefficients = numpy.asarray(mo_coeff)
                if coefficients.ndim == 3:  # unrestricted: show the alpha orbital
                    coefficients = coefficients[0]
                index = int(spec["orbital"]) - 1
                cubegen.orbital(mol, path, coefficients[:, index], resolution=resolution)
            else:
                raise ValueError(f"unknown cube task {task!r}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def probe_gpu() -> int:
    """Exit 0 if gpu4pyscf can run here, 1 otherwise.

    Exists as a subcommand so the web server can decide whether to offer the GPU without
    importing cupy into its own process: a CUDA initialisation failure must not be able to
    take the UI down, and the server should not hold a device context it never uses.
    """
    from . import engine

    if engine.gpu_available():
        print(engine.device_name())
        return 0
    return 1


_MODES = {"run": run_job, "cubegen": run_cubegen}


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``run`` / ``cubegen`` / ``gpu-probe`` to the matching handler."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "gpu-probe":
        return probe_gpu()
    if len(argv) != 2 or argv[0] not in _MODES:
        print("usage: python -m pyscf_webui.runner {run|cubegen} <spec.json>\n"
              "       python -m pyscf_webui.runner gpu-probe", file=sys.stderr)
        return 2
    mode, spec_path = argv
    return _MODES[mode](spec_path)


if __name__ == "__main__":
    sys.exit(main())
