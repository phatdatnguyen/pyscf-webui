"""Calculation tab: pick a structure, choose a method, run PySCF.

Unlike the ORCA app this is modelled on, there is **no input-file step**. ORCA is driven
by a text input file, so building one and running it are two separate user actions; PySCF
is driven by its Python API, so there is nothing for a user to author. One Run button
takes the structure and the settings on screen straight to a calculation.

The calculation itself happens in a child process (:mod:`pyscf_webui.runner`) rather than
on the Gradio worker thread. The job spec written into the working directory is the
argument to that child -- an implementation detail that happens to be visible in the file
browser, not a format anyone is expected to edit.

Because the child is a real OS process, Stop can genuinely stop it, and the handler
streams the growing PySCF log back to the browser while it runs.
"""
import math
import multiprocessing
import os
import subprocess
import sys
import threading
import time

import gradio as gr
import psutil
from rdkit.Chem import AllChem

from .utils import (
    CALCULATION_TYPES,
    DEFAULT_G_CONVERGENCE,
    DEFAULT_GRID_LEVEL,
    DEFAULT_SCF_CONV_TOL,
    DEFAULT_SOLVENT_MODEL,
    DISPERSIONS,
    EMISSION,
    FREQUENCY,
    GEOMETRY_OPTIMIZATION,
    DEFAULT_J_COUPLING_NUCLEI,
    GRID_LEVELS,
    G_CONVERGENCES,
    JOB_SUFFIX,
    J_COUPLING_NUCLEI_CHOICES,
    NMR,
    NMR_REQUIRES_GPU,
    ROHF_UNSUPPORTED_TYPES,
    SCF_CONV_TOLS,
    SINGLE_POINT,
    SOLVENT_MODELS,
    TDDFT,
    build_job_spec,
    calculation_profile,
    method_uses_functional,
    get_files_in_working_directory,
    mol_from_structure_file,
    pyscf_atom_spec,
    resolve_choice,
    write_json,
)

# Physical RAM in whole GB, and the share of it a calculation may claim. Unlike Psi4's
# set_memory, PySCF's mol.max_memory is *advisory* -- it sizes blocking loops rather than
# capping the process -- so this is a hint, not a limit. GPU memory is separate and not
# controlled from here.
_TOTAL_MEMORY_GB = max(1, math.floor(psutil.virtual_memory().total / (1024 ** 3)))
MAX_MEMORY_GB = max(1, math.floor(_TOTAL_MEMORY_GB * 0.8))
DEFAULT_MEMORY_GB = max(1, min(4, MAX_MEMORY_GB))
# PySCF parallelises with OpenMP threads, so the full logical CPU count is usable. It
# still matters on the GPU path: integral prescreening and any correlated step run on the
# host. (The ORCA app halves this only because ORCA refuses more MPI ranks than cores.)
MAX_THREADS = max(1, multiprocessing.cpu_count())
DEFAULT_THREADS = max(1, MAX_THREADS // 2)

# The functional and basis menus are no longer fixed: each calculation type advertises its
# own via CALCULATION_PROFILES in utils, because the constraints really do differ (NMR
# refuses range-separated functionals, TD-DFT wants them). The initial values below are the
# single-point profile's; on_change_calculation_type rebuilds them on every switch.
_INITIAL_PROFILE = calculation_profile(SINGLE_POINT)
FUNCTIONALS = _INITIAL_PROFILE.functionals
BASIS_SETS = _INITIAL_PROFILE.basis_sets
DEFAULT_BASIS_SET = _INITIAL_PROFILE.default_basis
# Solvents both continuum models can describe. The engine holds the dielectric constants
# and the SMD database keys; the UI only needs the labels.
SOLVENTS = ["Water", "Methanol", "Ethanol", "Acetonitrile", "Acetone", "DMSO",
            "Dichloromethane", "Chloroform", "Tetrahydrofuran", "Toluene", "Benzene",
            "Cyclohexane", "N-heptane", "Carbon tetrachloride", "Aniline",
            "Chlorobenzene", "Nitromethane", "1,2-Dichloroethane"]
RESTRICTED_REFERENCES = ["RHF", "ROHF", "UHF"]
# A closed-shell (restricted) reference cannot represent an open-shell state, so anything
# with unpaired electrons must use an unrestricted or restricted-open reference.
OPEN_SHELL_REFERENCES = ["UHF", "ROHF"]


def _gpu_is_available() -> bool:
    """Whether the child process can run on the GPU.

    Probed in a subprocess, never by importing cupy here. The web server must start on a
    machine with no CUDA, and a driver or runtime fault during ``import gpu4pyscf`` would
    otherwise take the whole UI down instead of just disabling one checkbox.
    """
    try:
        return subprocess.run(
            [sys.executable, "-m", "pyscf_webui.runner", "gpu-probe"],
            capture_output=True, timeout=180).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


GPU_AVAILABLE = _gpu_is_available()

# Default job/artifact name per calculation type, so the name field tracks the choice.
_DEFAULT_JOB_NAMES = {
    SINGLE_POINT: "single_point",
    GEOMETRY_OPTIMIZATION: "geometry_optimization",
    FREQUENCY: "frequency",
    TDDFT: "tddft",
    EMISSION: "emission",
    NMR: "nmr_spectrum",
}

# How often the run handler re-reads the log to stream progress, in seconds. Short enough
# to feel live, long enough not to spend the worker thread's time on stat() calls.
_POLL_INTERVAL = 0.5
# Tail of the log shown while a calculation runs. The full file is always one click
# away in the working-directory viewer; streaming all of it would push megabytes into the
# browser for a long CCSD run.
_LOG_TAIL_CHARS = 8000

# Running child processes, keyed by working directory rather than held in a single global
# as the ORCA app does. Two browser sessions working in two different directories can
# then run at the same time, and Stop kills the right one.
_processes: dict[str, subprocess.Popen] = {}
_stopped: set[str] = set()
_lock = threading.Lock()


def _status(message: str, color: str) -> str:
    """Colored status span, the convention every handler in this app reports through."""
    return f"<span style='color:{color};'>{message}</span>"


def _read_log_tail(log_path: str) -> str:
    """Return the last :data:`_LOG_TAIL_CHARS` characters of ``log_path``, or ``""``."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            if size > _LOG_TAIL_CHARS:
                handle.seek(size - _LOG_TAIL_CHARS)
            return handle.read()
    except OSError:
        return ""


def on_working_directory_file_list_change(working_directory_file_list):
    """Repopulate the input-structure dropdown from the working directory's files.

    The file-list state is ``None`` until a directory is opened, so it is normalised to an
    empty list here. Trajectory files are excluded: they hold many frames, and feeding one
    to a calculation would silently use only its final geometry, which is never what
    someone picking a "structure" intends.
    """
    working_directory_file_list = working_directory_file_list or []
    structure_file_names = sorted(
        (f for f in working_directory_file_list
         if f.endswith(('.xyz', '.pdb', '.mol', '.mol2')) and not f.endswith('_trajectory.xyz')),
        key=str.lower,
    )
    return gr.update(
        choices=structure_file_names,
        value=structure_file_names[0] if structure_file_names else None,
        interactive=True,
    )


def on_change_calculation_type(calculation_type, method_type, functional, basis_set):
    """React to the calculation-type radio changing.

    Rebuilds the three method menus from that type's :data:`CALCULATION_PROFILES` entry and
    toggles the control groups that only apply to some types. The profile is consulted
    rather than branched on, so what each calculation type supports is stated once, as data,
    in one place a chemist can review.

    Selections are preserved when the new type still offers them and replaced by that type's
    default when it does not. Silently keeping an invalid one matters here: these dropdowns
    accept custom values, so a range-separated functional left over from TD-DFT would reach
    an NMR job and be refused by the engine.
    """
    profile = calculation_profile(calculation_type)
    method_value = resolve_choice(method_type, profile.method_types,
                                  profile.default_method_type)
    functional_value = resolve_choice(functional, profile.functionals,
                                      profile.default_functional)
    basis_value = resolve_choice(basis_set, profile.basis_sets, profile.default_basis)

    is_tddft = calculation_type == TDDFT
    is_emission = calculation_type == EMISSION
    show_optimizer = calculation_type in (GEOMETRY_OPTIMIZATION, FREQUENCY, EMISSION)
    # Emission is the one type with no method choice at all: it optimizes an excited state,
    # which here means TD-DFT, so the dropdown is locked rather than merely filtered.
    method_interactive = not is_emission

    return (
        _DEFAULT_JOB_NAMES.get(calculation_type, "calculation"),
        gr.update(choices=profile.functionals, value=functional_value),
        gr.update(choices=profile.basis_sets, value=basis_value),
        gr.update(visible=show_optimizer),
        gr.update(visible=calculation_type == FREQUENCY),
        gr.update(visible=is_tddft),
        gr.update(choices=profile.method_types, value=method_value,
                  interactive=method_interactive),
        gr.update(visible=is_emission),
        gr.update(visible=is_emission),
        gr.update(visible=calculation_type == NMR),
        gr.update(visible=calculation_type == NMR),
    )


def on_method_type_change(method_type):
    """Show the functional dropdown only for the families that consume a functional.

    Emission reports its method type as "TD-DFT" rather than "DFT", so a bare
    ``== "DFT"`` here hid the control on the one calculation type whose entire result is
    determined by the functional. :func:`method_uses_functional` states the rule once.
    """
    return gr.update(visible=method_uses_functional(method_type))


def on_j_coupling_change(compute_j_couplings):
    """Show the nuclei picker only when J-couplings are being computed."""
    return gr.update(visible=bool(compute_j_couplings))


def on_multiplicity_change(multiplicity, reference):
    """Restrict the reference to open-shell options when the state has unpaired electrons.

    A restricted (RHF) reference forces electrons into doubly-occupied orbitals, which
    simply cannot represent a doublet or triplet -- PySCF rejects the combination. Rather
    than letting the user discover that from a crashed job, the menu drops the restricted
    option and switches to UHF as soon as the multiplicity leaves 1.
    """
    if int(multiplicity) > 1:
        value = reference if reference in OPEN_SHELL_REFERENCES else "UHF"
        return gr.update(choices=OPEN_SHELL_REFERENCES, value=value)
    return gr.update(choices=RESTRICTED_REFERENCES, value=reference or "RHF")


def on_mm_checkbox_change(use_mm):
    """Show/hide the force-field and max-iterations controls with the MM checkbox."""
    return gr.update(visible=use_mm), gr.update(visible=use_mm)


def on_solvation_checkbox_change(use_solvation):
    """Show/hide the solvation model and solvent pickers with the solvation checkbox."""
    return gr.update(visible=use_solvation), gr.update(visible=use_solvation)


def on_calculation_started():
    """Grey out the Run button for the duration of a calculation.

    Chained *ahead* of :func:`on_run_calculation` rather than folded into it: that handler
    is a generator whose first yield only reaches the browser once the child has been
    launched, and the button needs to go dead the moment the click lands.
    """
    return gr.update(interactive=False)


def on_calculation_finished():
    """Re-enable the Run button once no calculation is running.

    Chained with ``.then`` (not ``.success``) so the button comes back even when the run
    failed or was stopped, which would otherwise leave it stuck.
    """
    return gr.update(interactive=True)


def on_run_calculation(working_directory_path, structure_file_name, calculation_type,
                       use_mm, force_field, max_iters,
                       method_type, functional, basis_set, reference,
                       charge, multiplicity,
                       geom_maxiter, g_convergence, temperature, pressure,
                       n_states, tda, root,
                       use_solvation, solvent_model, solvent,
                       n_threads, memory_gb, use_gpu,
                       density_fit, dispersion, grid_level, scf_conv_tol,
                       compute_j_couplings, j_coupling_nuclei, job_name):
    """Run PySCF on the selected structure, streaming its log as it goes.

    A generator handler: it yields ``(status, log_tail, file_list)`` every
    :data:`_POLL_INTERVAL` seconds while the child runs, so the user watches the SCF
    iterate instead of staring at a frozen page. The final yield reports the outcome.

    Errors are reported in the status span rather than raised, matching every other
    handler in this app.
    """
    if not working_directory_path:
        gr.Warning("Please open a working directory first.")
        return
    if not structure_file_name:
        gr.Warning("Please select an input structure.")
        return
    if not job_name or not job_name.strip():
        gr.Warning("Please give the calculation a name.")
        return
    if calculation_type == NMR and NMR_REQUIRES_GPU and not use_gpu:
        # There is no CPU shielding implementation to fall back to, so this cannot be a
        # warning-and-continue the way the Hessian fallback is.
        gr.Warning("NMR shielding requires the GPU. Tick \"Use GPU\" under Resources, or "
                   "pick a different calculation type.")
        return
    if reference == "ROHF" and calculation_type in ROHF_UNSUPPORTED_TYPES:
        # Neither pyscf.hessian nor pyscf.tdscf ships an ROHF/ROKS module, and neither
        # does gpu4pyscf. Caught here so the user is told immediately rather than after
        # a geometry optimization has already been paid for.
        gr.Warning(f"{calculation_type} is not available with an ROHF reference. "
                   f"Use UHF instead.")
        return

    job_name = job_name.strip()
    file_list = get_files_in_working_directory(working_directory_path)

    try:
        # Same loader the structure viewer uses, so the atom indices labelled on screen
        # are the ones PySCF receives.
        structure_path = os.path.join(working_directory_path, structure_file_name)
        mol = mol_from_structure_file(structure_path)

        if use_mm:
            # A cheap force-field cleanup before an expensive QM calculation: a strained
            # embedding costs optimizer steps, and MMFF removes most of that strain for
            # a fraction of a second of work.
            if force_field == "MMFF":
                AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))

        spec = build_job_spec(
            calculation_type=calculation_type,
            geometry=pyscf_atom_spec(mol),
            structure_file=structure_file_name,
            method_type=method_type,
            functional=functional,
            basis_set=basis_set,
            reference=reference,
            charge=charge,
            multiplicity=multiplicity,
            n_threads=n_threads,
            memory_gb=memory_gb,
            use_gpu=bool(use_gpu),
            density_fit=bool(density_fit),
            dispersion=dispersion,
            grid_level=grid_level,
            scf_conv_tol=scf_conv_tol,
            geom_maxiter=geom_maxiter,
            g_convergence=g_convergence,
            temperature=temperature,
            # The slider is in atm because that is how chemists quote standard state;
            # the thermochemistry code wants pascal.
            pressure=float(pressure) * 101325.0,
            n_states=n_states,
            tda=tda,
            root=root,
            use_solvation=use_solvation,
            solvent_model=solvent_model,
            solvent=solvent,
            # Only an NMR job knows what to do with these, and carrying them on every other
            # spec would imply they mean something there.
            compute_j_couplings=bool(compute_j_couplings) and calculation_type == NMR,
            j_coupling_nuclei=j_coupling_nuclei,
        )

        job_path = os.path.join(working_directory_path, job_name + JOB_SUFFIX)
        log_path = os.path.join(working_directory_path, job_name + ".log")
        write_json(job_path, spec)

        # Opening the log below truncates it, so preserve a previous run's output first:
        # an accidental click on Run would otherwise destroy it. os.replace overwrites an
        # older backup atomically and works on Windows too.
        backed_up = False
        if os.path.isfile(log_path):
            os.replace(log_path, log_path + ".bak")
            backed_up = True
    except Exception as exc:
        yield _status(f"Could not start the calculation: {exc}", "red"), "", file_list
        return

    with _lock:
        existing = _processes.get(working_directory_path)
        if existing is not None and existing.poll() is None:
            yield (_status("A calculation is already running in this working directory.", "red"),
                   "", file_list)
            return

    # PySCF writes no output file of its own -- it prints to a stream -- so the log *is*
    # the child's stdout. Pointing both streams at <name>.log captures all three writers
    # at once: PySCF's logger (which flushes after every record), geomeTRIC's optimizer
    # table (which goes to stderr), and any Python traceback. That is also why Psi4-era
    # separate .runner.log is gone: with one stream there is nowhere else for a crash
    # message to hide, and the traceback is recorded in the result file besides.
    started = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                # -u so the log is unbuffered and the tail below is genuinely live; a
                # buffered child would show nothing until it exited.
                # The job path must be absolute: working directories are named relatively
                # ("./data/wd"), and the child is started *in* that directory, so a
                # relative path would be resolved a second time against itself.
                [sys.executable, "-u", "-m", "pyscf_webui.runner", "run",
                 os.path.abspath(job_path)],
                cwd=working_directory_path,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            with _lock:
                _processes[working_directory_path] = process
                _stopped.discard(working_directory_path)

            try:
                while process.poll() is None:
                    elapsed = time.time() - started
                    yield (_status(f"Running {calculation_type}... ({elapsed:.0f} s)", "black"),
                           _read_log_tail(log_path),
                           file_list)
                    time.sleep(_POLL_INTERVAL)
                return_code = process.returncode
            finally:
                with _lock:
                    _processes.pop(working_directory_path, None)
                    stopped = working_directory_path in _stopped
                    _stopped.discard(working_directory_path)
    except Exception as exc:
        yield _status(f"Error running calculation: {exc}", "red"), "", get_files_in_working_directory(working_directory_path)
        return

    duration = time.time() - started
    file_list = get_files_in_working_directory(working_directory_path)
    log_tail = _read_log_tail(log_path)

    if stopped:
        yield (_status(f"Calculation stopped after {duration:.1f} s. Partial output in {job_name}.log.", "red"),
               log_tail, file_list)
        return
    if return_code != 0:
        # The runner writes a result file even when it fails, so point at the message it
        # recorded rather than making the user go hunting through two log files.
        detail = f"See {job_name}.log for the traceback."
        yield (_status(f"Calculation failed (exit code {return_code}). {detail}", "red"),
               log_tail, file_list)
        return

    message = f"Calculation finished ({duration:.1f} s)."
    if backed_up:
        message += f" Previous log saved as {job_name}.log.bak."
    yield _status(message, "green"), log_tail, file_list


def on_stop_calculation(working_directory_path):
    """Terminate the child running in ``working_directory_path``, if any.

    PySCF spawns OpenMP worker threads and cupy holds a device context that only process
    death reliably releases, so the whole process
    tree is signalled (SIGTERM first, SIGKILL for anything still alive after 5 s);
    terminating only the process we launched could leave the cores busy. The run handler
    reports the outcome once ``poll()`` returns, so this only needs to acknowledge the
    request.
    """
    with _lock:
        process = _processes.get(working_directory_path)
        if process is None or process.poll() is not None:
            gr.Warning("No calculation is running.")
            return ""
        _stopped.add(working_directory_path)

    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        targets = []

    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass

    return _status("Stopping calculation...", "red")


def calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    """Build the "Calculation" tab and wire its events.

    Takes the shared path/file-list states and status line; returns the tab component.
    """
    with gr.Tab("Calculation") as calculation_tab:
        with gr.Accordion("Resources", open=False):
            with gr.Row():
                n_threads_slider = gr.Slider(label="Number of threads", value=DEFAULT_THREADS,
                                             minimum=1, maximum=MAX_THREADS, step=1,
                                             info="PySCF parallelises with OpenMP threads.")
                memory_slider = gr.Slider(label="Memory (GB)", value=DEFAULT_MEMORY_GB,
                                          minimum=1, maximum=MAX_MEMORY_GB, step=1,
                                          info=f"Advisory budget PySCF uses to size its blocking loops, not a hard cap; this machine has {_TOTAL_MEMORY_GB} GB.")
                gpu_checkbox = gr.Checkbox(
                    label="Use GPU", value=GPU_AVAILABLE, interactive=GPU_AVAILABLE,
                    info=("SCF, DFT, gradients and Hessians run on the GPU via gpu4pyscf. "
                          "MP2, CCSD and CCSD(T) always finish on the CPU."
                          if GPU_AVAILABLE else
                          "No usable gpu4pyscf installation was found; running on the CPU. "
                          "See the README for the CUDA runtime requirement."))
        with gr.Accordion("Calculation"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_structure_file_dropdown = gr.Dropdown(label="Input structure", choices=[""], value="", interactive=False)
                with gr.Column(scale=4):
                    calculation_type_radio = gr.Radio(label="Type of calculation", value=SINGLE_POINT, choices=CALCULATION_TYPES)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        mm_checkbox = gr.Checkbox(label="Pre-optimize geometry with molecular mechanics", value=False)
                    with gr.Row():
                        force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], visible=False)
                        max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1, visible=False)
                    with gr.Row():
                        solvation_checkbox = gr.Checkbox(label="Solvation", value=False,
                                                         info="Continuum solvation. SMD is parameterised for solvation free energies; the PCM variants model electrostatics only.")
                    with gr.Row():
                        solvent_model_radio = gr.Radio(label="Solvation model", value=DEFAULT_SOLVENT_MODEL,
                                                       choices=SOLVENT_MODELS, visible=False)
                        solvent_dropdown = gr.Dropdown(label="Solvent", value="Water", choices=SOLVENTS, visible=False)
                    with gr.Group(visible=False) as optimizer_group:
                        geom_maxiter_slider = gr.Slider(label="Max optimization steps", value=50, minimum=1, maximum=500, step=1)
                        g_convergence_dropdown = gr.Dropdown(label="Convergence criteria",
                                                             value=DEFAULT_G_CONVERGENCE,
                                                             choices=G_CONVERGENCES,
                                                             info="geomeTRIC convergence criteria sets.")
                    with gr.Group(visible=False) as thermo_group:
                        temperature_slider = gr.Slider(label="Temperature (K)", value=298.15, minimum=1, maximum=1000, step=0.05)
                        pressure_slider = gr.Slider(label="Pressure (atm)", value=1.0, minimum=0.1, maximum=100, step=0.1)
                with gr.Column(scale=1):
                    with gr.Row():
                        method_type_dropdown = gr.Dropdown(label="Type of method",
                                                           value=_INITIAL_PROFILE.default_method_type,
                                                           choices=_INITIAL_PROFILE.method_types)
                        reference_dropdown = gr.Dropdown(label="Reference", value="RHF", choices=RESTRICTED_REFERENCES)
                    with gr.Row():
                        functional_dropdown = gr.Dropdown(label="Functional", value="B3LYP", choices=FUNCTIONALS, allow_custom_value=True)
                        basis_set_dropdown = gr.Dropdown(label="Basis set", value=DEFAULT_BASIS_SET, choices=BASIS_SETS, allow_custom_value=True)
                    with gr.Row():
                        charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                        multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1,
                                                            choices=[("Singlet", 1), ("Doublet", 2), ("Triplet", 3),
                                                                     ("Quartet", 4), ("Quintet", 5), ("Sextet", 6)])
                    with gr.Group(visible=False) as excited_state_group:
                        n_states_slider = gr.Slider(label="Number of excited states", value=10, minimum=1, maximum=50, step=1)
                        tda_checkbox = gr.Checkbox(label="Tamm-Dancoff approximation (TDA)", value=False,
                                                   info="Cheaper and more stable than full TDDFT/RPA, at some cost in accuracy.")
                    with gr.Group(visible=False) as emission_group:
                        root_slider = gr.Slider(label="Excited state to optimize", value=1, minimum=1, maximum=10, step=1,
                                                info="1 = S1. By Kasha's rule emission comes from S1, so leave this at 1 unless you know otherwise.")
                    with gr.Accordion("Method details", open=False):
                        density_fit_checkbox = gr.Checkbox(
                            label="Density fitting", value=True,
                            info="Much faster, and the default for DFT. Turn it off for a "
                                 "reference calculation in a small basis, where the fitting "
                                 "error is not negligible.")
                        dispersion_dropdown = gr.Dropdown(
                            label="Dispersion correction", value="", choices=DISPERSIONS,
                            info="Applied separately from the functional: libxc silently "
                                 "ignores a '-D3BJ' suffix on a functional name.")
                        grid_level_dropdown = gr.Dropdown(
                            label="DFT grid", value=DEFAULT_GRID_LEVEL, choices=GRID_LEVELS,
                            info="A coarse grid is a common cause of small spurious "
                                 "imaginary frequencies.")
                        scf_conv_tol_dropdown = gr.Dropdown(
                            label="SCF convergence", value=DEFAULT_SCF_CONV_TOL,
                            choices=SCF_CONV_TOLS)
                with gr.Column(scale=1):
                    job_name_textbox = gr.Textbox(label="Calculation name", value=_DEFAULT_JOB_NAMES[SINGLE_POINT],
                                                  info="Names the .log, result and wavefunction files.")
                    emission_warning_markdown = gr.Markdown(visible=False, value=(
                        "**Emission optimizes the excited state.** It relaxes the chosen "
                        "root on the TD-DFT excited-state surface, then evaluates both "
                        "surfaces at that geometry. Each step costs a TD-DFT gradient, so "
                        "expect it to take several times a ground-state optimization. "
                        "Excited-state roots can also cross during the optimization -- if "
                        "the result looks wrong, check the state ordering in the log."
                    ))
                    with gr.Group(visible=False) as nmr_group:
                        j_coupling_checkbox = gr.Checkbox(
                            label="Compute J-couplings", value=False,
                            info="Indirect spin-spin couplings, which turn the simulated "
                                 "1H signals into real multiplets. Runs on the CPU (there "
                                 "is no GPU implementation) and is the expensive part of "
                                 "an NMR job: seconds for ethanol, minutes for a "
                                 "drug-sized molecule.")
                        j_nuclei_dropdown = gr.Dropdown(
                            label="Couple which nuclei", value=DEFAULT_J_COUPLING_NUCLEI,
                            choices=J_COUPLING_NUCLEI_CHOICES, visible=False,
                            info="Cost scales with the number of nuclei involved. Only "
                                 "1H-1H couplings split a 1H spectrum; add 13C when the "
                                 "per-nucleus J table is what you are after.")
                    nmr_note_markdown = gr.Markdown(visible=False, value=(
                        "Requires the GPU and a closed-shell DFT method; range-separated "
                        "functionals are not supported."
                    ))
                    run_button = gr.Button("Run", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
        with gr.Accordion("PySCF output", open=True):
            live_output_textarea = gr.TextArea(label="PySCF output (live)", lines=18,
                                               elem_id="textfile_viewer", interactive=False)

        working_directory_file_list_state.change(
            on_working_directory_file_list_change,
            working_directory_file_list_state,
            input_structure_file_dropdown,
        )
        calculation_type_radio.change(
            on_change_calculation_type,
            [calculation_type_radio, method_type_dropdown, functional_dropdown,
             basis_set_dropdown],
            [job_name_textbox, functional_dropdown, basis_set_dropdown, optimizer_group,
             thermo_group, excited_state_group, method_type_dropdown, emission_group,
             emission_warning_markdown, nmr_note_markdown, nmr_group],
        )
        method_type_dropdown.change(on_method_type_change, method_type_dropdown, functional_dropdown)
        j_coupling_checkbox.change(on_j_coupling_change, j_coupling_checkbox, j_nuclei_dropdown)
        multiplicity_dropdown.change(on_multiplicity_change, [multiplicity_dropdown, reference_dropdown], reference_dropdown)
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox,
                                  [solvent_model_radio, solvent_dropdown])

        run_button.click(on_calculation_started, None, run_button) \
                  .then(on_run_calculation,
                        [working_directory_path_state, input_structure_file_dropdown, calculation_type_radio,
                         mm_checkbox, force_field_dropdown, max_iters_slider,
                         method_type_dropdown, functional_dropdown, basis_set_dropdown, reference_dropdown,
                         charge_slider, multiplicity_dropdown,
                         geom_maxiter_slider, g_convergence_dropdown, temperature_slider, pressure_slider,
                         n_states_slider, tda_checkbox, root_slider,
                         solvation_checkbox, solvent_model_radio, solvent_dropdown,
                         n_threads_slider, memory_slider, gpu_checkbox,
                         density_fit_checkbox, dispersion_dropdown, grid_level_dropdown,
                         scf_conv_tol_dropdown, j_coupling_checkbox, j_nuclei_dropdown,
                         job_name_textbox],
                        [status_markdown, live_output_textarea, working_directory_file_list_state]) \
                  .then(on_calculation_finished, None, run_button)
        # concurrency_limit=None: without it this event would queue behind the running
        # calculation it is meant to interrupt, and only fire once PySCF had finished.
        stop_button.click(on_stop_calculation, working_directory_path_state, status_markdown, concurrency_limit=None)

    return calculation_tab
