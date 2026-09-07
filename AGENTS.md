# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

A Gradio web UI wrapping [PySCF](https://pyscf.org/) and
[gpu4pyscf](https://github.com/pyscf/gpu4pyscf) for computational chemistry workflows:
conformer generation, then Single-Point / Geometry Optimization / Frequency / TD-DFT /
Emission / NMR calculations, then result visualization (energies, MOs, IR and UV-Vis/ECD
spectra, NMR spectra, and orbital/density isosurfaces).

PySCF is driven through its Python API. There is deliberately no "generate an input file,
then run it" step: the Calculation tab takes the structure and the settings on screen
straight to a calculation.

This app is a port of a Psi4 sibling (`../psi4-webui`), which in turn descends from an
ORCA one (`../orca-webui`). Comments occasionally reference those apps to explain why
something is the way it is; that history is deliberate, not stale.

## Packaging & layout

A pip-installable package (import name `pyscf_webui`) using a **src layout**: all modules
live under `src/pyscf_webui/`. Metadata is in `pyproject.toml` (setuptools backend);
`styles.css` ships as package data. The `pyscf-webui` console command maps to
`pyscf_webui.app:main`, which serves on the first free port at/after 7860.

`gpu4pyscf` is an optional extra, not a hard dependency — the CUDA-12 wheels would make
the package uninstallable on a machine without one. Everything except GPU acceleration
works without it.

## Commands

```bash
python3 -m venv pyscf-env && source pyscf-env/bin/activate
pip install -e ".[dev]"          # add ".[gpu]" for gpu4pyscf + the CUDA runtime wheels
pip install --no-deps gradio_molecule2d   # optional 2D sketcher; it pins gradio<5

pyscf-webui                      # or: python -m pyscf_webui.app

pytest                           # whole suite (~7 s)
pytest tests/test_nmr.py         # one module
pytest tests/test_nmr.py::TestPeakGrouping::test_shift_is_reference_minus_shielding
pytest -k equivalence            # by name
ruff check --select F,E9 src/ tests/      # real errors only; the repo is clean at this level
```

Tests that need PySCF or a GPU **skip themselves**. There are no pytest markers: `conftest.py`
probes availability in a *subprocess* and exposes `needs_pyscf` / `needs_gpu` skipif
decorators. The subprocess matters — importing cupy in the pytest process would create a
CUDA context, and a driver fault would take the whole run down instead of skipping a few
tests.

`data/` (working directories) and `static/` (transient viewer HTML) are created relative to
the **current working directory**, never the install location.

## Architecture

FastAPI host with a Gradio Blocks UI mounted at `/` and a `StaticFiles` mount at `/static`,
assembled in `app.py`'s `build_app()`. The `/static` mount exists only because nglview
renders to standalone HTML that has to be embedded in an `<iframe>`.

### The execution model — the load-bearing design decision

Calculations run in a **child process**, launched as
`python -u -m pyscf_webui.runner run <job.json>`. `runner.py` and `engine.py` are the only
modules that import `pyscf` / `gpu4pyscf` / `cupy`, and every such import is
**function-local** so the naming helpers stay importable without the engine. Three reasons,
all of which an in-process thread fails:

1. **Cancellation.** An SCF or a Hessian is a long native call with no interrupt hook. The
   only reliable Stop is killing the process tree.
2. **Crash isolation.** A CUDA out-of-memory or illegal access aborts the process. In-process
   that takes down the web server and every browser session with it.
3. **Clean state.** cupy's device memory pool, `lib.num_threads` and `PYSCF_TMPDIR` are
   process-global.

Two consequences worth knowing:

- **Paths handed to the child must be absolute.** Working directories are named relatively
  (`./data/wd`) and the child is started *in* that directory, so a relative path gets
  resolved against itself twice.
- **The runner always writes a result file**, success or failure, so the Result tab can
  explain what happened. Exit 0 = completed, 1 = the calculation raised, 2 = bad usage.

Subcommands: `run <job.json>`, `cubegen <cube.json>`, and `gpu-probe` (exit 0 if gpu4pyscf
is usable — this is how the web server decides whether to offer the GPU without importing
cupy into its own process).

`engine.py` vs `runner.py`: the engine knows about PySCF, the runner knows about the
job/result file contract. New physics goes in the engine; new artifacts go in the runner.

### The log is the child's stdout — this is why PySCF gets an output file

PySCF writes no output file of its own; it prints to a stream. Rather than manage a file
handle, `calculation.py` points the child's **stdout and stderr both** at `<base>.log` and
runs it with `-u`. That one arrangement captures all three writers at once: PySCF's
`lib.logger` (which flushes after every record), geomeTRIC's optimizer table (which goes to
`sys.stderr`), and any Python traceback. The 0.5 s tail in the Calculation tab is then
genuinely live with no extra machinery, and there is no separate `.runner.log`.

**Do not set `mol.output = <path>`.** `Mole.build()` opens it in `'w'` mode, so anything
that rebuilds a Mole mid-job would silently truncate the log the browser is tailing.
`runner._print_header` adds a summary block at the top so the file reads as a real output
file rather than an incidental dump.

### The result JSON is the contract

`result.py` reads `<name>.result.json` — never the log. There is not a single regex in the
repo. Because this app owns both producer and consumer there is no output-parsing layer to
go stale, and some of what is displayed (IR intensities, TD-DFT rotatory strengths, NMR
shieldings) exists only on the live wavefunction and is never printed. Units are canonical
in the file (hartree, Å, cm⁻¹, km/mol, Debye, ppm); display conversion happens in the UI.

The schema is `"version": 2`; version 1 (Psi4-era) files still render, because the Result
tab branches on which result *keys* are present, never on `calculation_type`.

When changing the schema, change `runner.py` and `result.py` together and update
`tests/conftest.py`'s fixtures — those fixtures are the executable spec.

### Per-job artifacts

PySCF produces no files, so everything is generated explicitly (`runner._write_artifacts`):
`.log`, `.result.json`, `.job.json`, `.chk` (HDF5 wavefunction, the replacement for Psi4's
`.npy`), `.molden`, `.freq.molden` (hand-written — PySCF has no writer), `.arrays.h5` (bulk
arrays kept out of the JSON), `_optimized.xyz`, `_trajectory.xyz`, `cubes/<name>/*.cube`.

Every artifact write is **independently guarded**: none is needed for the result itself, so
a failure degrades that one feature rather than discarding a finished calculation.

Two XYZ flavours, both deliberate: structure files start with `"<charge> <multiplicity>"`
so an optimized geometry feeds straight back in; trajectories use **standard** XYZ so
external viewers animate them.

### One rule, one place

`utils.method_uses_functional(method_type)` answers "does this family consume a functional?"
It exists because the rule was previously written out at seven call sites and three of them
spelled it `== "DFT"`, which is wrong for **Emission**: that type runs TD-DFT and its entire
result depends on the functional, but its method type is the string `"TD-DFT"`. The visible
symptom was the functional dropdown disappearing on the one calculation type that needs it
most. Any new site that asks this question should call the predicate rather than restate it;
`tests/test_options.py::TestFunctionalVisibility` guards it.

### Calculation profiles drive the menus

`utils.CALCULATION_PROFILES` maps each calculation type to its method types, functionals,
basis sets and per-type defaults. The constraints genuinely differ — NMR cannot use a
range-separated functional, TD-DFT wants one — so this is a data table rather than branching
inside the handler, and `tests/test_options.py` sweeps every entry against the engine.

`resolve_choice(current, choices, default)` is the switch policy: keep the user's selection
when the new type still offers it, else fall back to that type's default. Never silently
keep an invalid one — these dropdowns accept custom values, so a stale entry would reach the
engine and fail there.

When adding a calculation type: add the constant to `utils.CALCULATION_TYPES`, add a
profile, add a `_DEFAULT_JOB_NAMES` entry, add the runner branch, and add the Result-tab
outputs.

### Working-directory model

Everything is organized around a working directory: a subfolder of `data/` selected in the
left column. State flows through `gr.State` objects created in `working_directory_blocks()`:

- `working_directory_path_state` — the current path.
- `working_directory_file_list_state` — **the central event bus.** Any handler that changes
  files on disk returns a fresh `get_files_in_working_directory(...)` into it; its `.change`
  fires each tab's `on_working_directory_file_list_change`, which is how every dropdown
  stays in sync.
- `status_markdown` — a shared status line; handlers return coloured HTML spans.

### UI structure (three tabs, one file each)

- `conformer_generation.py` — SMILES or sketch → RDKit oversampled embedding → MMFF/UFF
  minimize → dedupe by energy **and** RMSD. Engine-agnostic; untouched by the PySCF port.
- `calculation.py` — structure + method + type → job spec → run the child, streaming its
  log. Holds the process registry and the profile-driven menu handler.
- `result.py` — loads a result file and conditionally reveals accordions based on what the
  calculation produced. `on_load_result_file` returns one positional tuple that must stay
  index-aligned with `_result_outputs`; `RESULT_OUTPUT_COUNT` (currently 30) and
  `tests/test_result_tab.py` enforce it. **Adding a component means editing four places.**
- `visualization.py` — nglview rendering, shared by the structure and cube viewers. Owns the
  cube filename contract (`cube_file_name`), which the runner imports so producer and
  consumer cannot disagree.
- `utils.py` / `engine.py` — the UI-free layers. `utils.py` deliberately imports no engine,
  so the server starts (and every mapping test runs) without PySCF or CUDA.

Process registry: `_processes` is keyed by **working directory**, so two sessions in two
directories can run at once and Stop hits the right one.
`stop_button.click(..., concurrency_limit=None)` is load-bearing — without it the Stop event
queues behind the run it is meant to interrupt.

### nglview rendering

Two non-obvious details:

- **`isolevelType="value"`.** NGL's surface representation defaults to `"sigma"`, which
  multiplies the given number by the RMS of the grid, silently turning a 0.02 a.u. contour
  into ~0.0008 a.u.
- **Transient widgets must be unregistered.** `nglview.write_html` reaches ipywidgets'
  `embed_data`, which serializes *every widget still registered in the process*. Without
  `_transient_widgets()` each render embeds all its predecessors. Unregister, do not
  `close()` — closing severs the comm while leaving the object referenced, and the next
  render fails.

## PySCF / gpu4pyscf gotchas that shaped this code

Each was verified against this venv (pyscf 2.14.0, gpu4pyscf 1.8.1) and most are enforced by
a test:

- **`mol.spin` is 2S, not the multiplicity.** Psi4 took 2S+1. Getting it wrong does not
  raise — it runs a different spin state and returns a plausible number. Isolated in
  `utils.pyscf_spin` so it can only ever be wrong once.
- **libxc silently ignores a dispersion suffix.** `parse_xc("B3LYP-D3BJ")` returns *exactly*
  the same tuple as `parse_xc("B3LYP")`, so `mf.xc = "B3LYP-D3BJ"` runs plain B3LYP with no
  correction and no error. Dispersion is a separate control and `engine.split_dispersion`
  splits the suffix off; no functional in any menu carries one.
- **`geometric_solver.kernel` leaves `mf.mol` at the *original* geometry** (its own docstring
  says so). `runner._reconverge_at` rebuilds and re-converges at `mol_eq`; without it every
  downstream property, checkpoint and cube would describe the input structure.
- **The geomeTRIC callback's `coords` are in Bohr.** Multiply by `lib.param.BOHR` exactly
  once. The Psi4 ancestor got this wrong and wrote every structure 1.89× too large.
  `calc_new` also fires once per energy+gradient *evaluation*, not per accepted step, so
  rejected line-search points appear in the trajectory.
- **`symmetry=False` everywhere.** This replaces Psi4's `no_reorient`/`no_com` (PySCF only
  reorients when symmetry is on) *and* removes the per-irrep root bookkeeping that forced
  Psi4 to drop symmetry for excited states. It also means excited states have no irrep
  labels, so the `symmetry` field is `""`.
- **No CPU IR intensities.** `pyscf.hessian.thermo.harmonic_analysis` carries a literal
  `#TODO: IR intensity`. The only implementation is `gpu4pyscf.properties.ir`, which asserts
  a restricted reference. When unavailable, `ir_intensities_km_mol` is `None` and the UI
  omits the column — never zero-fills it, which would read as "IR-inactive".
- **gpu4pyscf 1.8.1 cannot do an unrestricted DF Hessian** (`assert wv.flags.c_contiguous`
  in `hessian/uks.py`). `engine._hessian_with_cpu_fallback` catches *any* GPU Hessian failure
  and retries on CPU, written against the failure rather than a list of known-bad combinations.
- **`thermo.rotational_symmetry_number` can raise** for a point group it does not tabulate —
  after the Hessian has been paid for. `_thermochemistry` wraps it.
- **`harmonic_analysis` returns complex frequencies**; imaginary modes live in the imaginary
  part. Collapse with `freq.real - abs(freq.imag)` so they come back negative.
- **PySCF has no rotatory-strength function.** ECD is assembled from `transition_dipole` and
  `transition_magnetic_dipole` as `R = 500 * Σ μ·m`, the factor taken from
  `gpu4pyscf.tdscf.spectralib` where it reproduces Gaussian 16. Verified: R ≡ 0 for achiral
  water, mixed signs for chiral H₂O₂. The overall *sign* has not been checked against a
  literature spectrum for a known enantiomer.
- **TD-DFT properties need host arrays** — `engine.excitation_results` moves a GPU solver
  across with `to_cpu()` first.
- **`mf.dip_moment(unit="Debye")` returns Debye directly** — no au conversion, unlike Psi4.
- **`cubegen`'s `resolution` is in Bohr**, matching Psi4's `CUBIC_GRID_SPACING`, so the grid
  slider carries over unchanged. `cubegen.mep` is CPU-only and slow (*ngrid × nao²*).

### NMR specifics

`gpu4pyscf.properties.shielding` is the only implementation in this stack, and it is
narrow — **GPU + restricted + non-range-separated DFT**:

- range-separated functionals hit a bare `assert omega == 0.0`
- HF fails with `AttributeError: '_numint'`
- open-shell fails with a tuple-unpacking `ValueError`
- there is no usable CPU path. `pyscf.prop.nmr` **is** installed (pulled in for
  J-couplings) but is incompatible with pyscf 2.14: it dies on
  `assert blksize % BLKSIZE == 0` inside `pyscf/dft/numint.py`. Its `ssc` sibling is
  unaffected, which is why J-couplings work on the CPU and shielding does not.

`engine.nmr_unavailable_reason` checks all of these **up front** with an actionable message,
because every one otherwise surfaces from library internals as something unreadable. The NMR
functional menu excludes range-separated entries and a test asserts that with
`libxc.rsh_coeff(f)[0] == 0.0`, so it cannot drift.

**J-couplings come from a different package on a different device.** gpu4pyscf has no
spin–spin module, so `engine.spin_spin_couplings` uses `pyscf.prop.ssc` (the
`pyscf-properties` dependency) on the **CPU**, moving a GPU mean field across with
`to_cpu()` first. It is opt-in (`compute_j_couplings` in the job spec) because it is the
expensive half of an NMR job — ethanol doubles, a drug-sized molecule runs to minutes — and
it is guarded separately in the runner so a J failure never discards finished shieldings.

Cost is driven by the number of *distinct* nuclei (one set of CPHF solves each), not the
pair count, which is why `j_coupling_nuclei` restricts `nuc_pair` by element. Restricting
pairs changes results only at the ~1e-4 Hz level (CPHF convergence noise), so it is safe.

`kernel()` returns raw tensors in atomic units; the conversion to hertz lives only in that
method's *logging* path, so `spin_spin_couplings` reproduces it
(`K = au2Hz · nuclear_magneton² · trace/3`, then `J = K · g(A) · g(B)`).

**Multiplets are built in `utils.assign_multiplets`**, and the two modelling choices there
are what make a single static geometry produce a textbook spectrum: the coupling between two
equivalent groups is the **average** of all their pairwise couplings, and the multiplicity
comes from the partner group's **size** rather than from how many couplings cleared the
threshold. Mutually equivalent nuclei do not split each other, and only homonuclear pairs
split — ¹³C is always rendered proton-decoupled.

`result.TMS_REFERENCES` must contain an entry for the NMR profile's own default level, and
`DEFAULT_H_REFERENCE`/`DEFAULT_C_REFERENCE` are taken from it. A reference at a different
level than the job is a silent 9.5 ppm error on ¹³C, so the two defaults are tied together
rather than chosen independently. Adding a level means running TMS through the app and
reading the isotropic shieldings off the result.

The runner stores **absolute shieldings and raw couplings only**. Referencing is a display concern: shift is
`reference − shielding`, applied in `result._nmr_outputs` — as are multiplet construction
and the spectrometer frequency — so changing a reference, the field or the averaging toggle
re-renders without re-running. Defaults are TMS at B3LYP/pcseg-2 (σ¹H 31.57, σ¹³C 180.96),
matching the NMR profile's own default level; `result.TMS_REFERENCES` holds the tabulated
levels.

Equivalent nuclei are grouped with RDKit `CanonicalRankAtoms(breakTies=False)`. The molecule
is rebuilt from the result's bare geometry, so `UpdatePropertyCache(strict=False)` is
required first or the ranking raises. The equivalence is topological: diastereotopic protons
are merged.

## Conventions

- **Handler signature:** every UI callback is a top-level `on_*` function taking Gradio
  component values positionally and returning positionally into the outputs list wired at the
  bottom of the tab's `*_tab_content` / `*_blocks` builder. Keep the two index-aligned.
- **Errors surface to the user, never raise:** handlers wrap work in `try/except` and return
  a red status span (and a refreshed file list) rather than propagating.
- **Comments explain *why*, especially where the code guards a silent failure.** Much of this
  codebase's value is in the reasons — a change that removes a guard should remove its
  comment too, and one that keeps the guard should keep the reason current.
- `data/`, `static/`, `pyscf-env/` and generated artifacts (`*.chk`, `*.molden`, `*.h5`,
  `*.cube`, `*.log`) are gitignored — never commit them.
- `orca-webui/` at the repo root is a reference copy of the sibling app's source, not part of
  this package. It is currently **not** gitignored.
