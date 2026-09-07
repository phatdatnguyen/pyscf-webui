"""PySCF / gpu4pyscf method construction and result extraction.

Imported only by :mod:`pyscf_webui.runner`, which runs in a child process -- never by the
web server. Every ``pyscf`` / ``gpu4pyscf`` / ``cupy`` import is therefore function-local,
so this module can be imported and unit-tested (and its pure helpers used) in an
environment where neither is installed, and so the web server never pays the cost of
creating a CUDA context.

The division of labour with :mod:`pyscf_webui.runner` is: this module knows about PySCF,
the runner knows about the job/result file contract.
"""
from __future__ import annotations

import os
from typing import Any

# Hartree-based conversions, duplicated from pyscf.data.nist so that the pure helpers here
# stay importable without PySCF. Values match CODATA 2018, which is what PySCF uses.
HARTREE2EV = 27.211386245988
HARTREE2WAVENUMBER = 219474.6313632
HARTREE2KCALMOL = 627.5094740631

# Named solvents offered in the UI, mapped to what each solvation model needs:
#   (dielectric constant for PCM, solvent key in the SMD database).
# The two names diverge often enough -- "DMSO" against "dimethylsulfoxide", "Carbon
# tetrachloride" against "carbon tetrachloride" -- that deriving one from the other at
# runtime would fail silently for a third of the list, so both are written out.
SOLVENT_DB: dict[str, tuple[float, str]] = {
    "Water": (78.3553, "water"),
    "Methanol": (32.613, "methanol"),
    "Ethanol": (24.852, "ethanol"),
    "Acetonitrile": (35.688, "acetonitrile"),
    "Acetone": (20.493, "acetone"),
    "DMSO": (46.826, "dimethylsulfoxide"),
    "Dichloromethane": (8.93, "dichloromethane"),
    "Chloroform": (4.711, "chloroform"),
    "Tetrahydrofuran": (7.4257, "tetrahydrofuran"),
    "Toluene": (2.3741, "toluene"),
    "Benzene": (2.2706, "benzene"),
    "Cyclohexane": (2.0165, "cyclohexane"),
    "N-heptane": (1.9113, "n-heptane"),
    "Carbon tetrachloride": (2.228, "carbon tetrachloride"),
    "Aniline": (6.8882, "aniline"),
    "Chlorobenzene": (5.6968, "chlorobenzene"),
    "Nitromethane": (36.562, "nitromethane"),
    "1,2-Dichloroethane": (10.125, "1,2-dichloroethane"),
}

# Dispersion suffixes the UI may append to a functional name. Ordered longest-first: a
# plain endswith loop would otherwise strip "-D3" off "B3LYP-D3BJ" and leave "BJ" behind.
_DISPERSION_SUFFIXES = ("-D3BJ", "-D3ZERO", "-D4", "-D3")


class ConvergenceError(RuntimeError):
    """A solver returned approximate results without satisfying its convergence test."""


def split_dispersion(functional: str) -> tuple[str, str | None]:
    """Split ``"B3LYP-D3BJ"`` into ``("B3LYP", "d3bj")``.

    This is not cosmetic. ``libxc.parse_xc("B3LYP-D3BJ")`` returns *exactly* the same
    functional tuple as ``parse_xc("B3LYP")`` -- the suffix is silently ignored -- so
    passing the UI label straight through to ``mf.xc`` runs plain B3LYP with no dispersion
    and no error of any kind. The correction has to be requested separately via ``mf.disp``.
    """
    upper = functional.upper()
    for suffix in _DISPERSION_SUFFIXES:
        if upper.endswith(suffix):
            return functional[: -len(suffix)], suffix[1:].lower()
    return functional, None


def to_numpy(array):
    """Return ``array`` as a numpy array, copying off the GPU first if it lives there.

    gpu4pyscf hands back ``cupy`` arrays from the same attributes PySCF fills with numpy
    ones. Everything downstream of a calculation -- JSON, HDF5, Molden, cube grids -- needs
    host memory, so this is the single conversion point rather than a ``.get()`` sprinkled
    through every extractor.
    """
    import numpy

    if array is None:
        return None
    try:
        import cupy

        if isinstance(array, cupy.ndarray):
            return cupy.asnumpy(array)
    except ImportError:
        pass
    return numpy.asarray(array)


def gpu_available() -> bool:
    """Whether a usable gpu4pyscf stack is present.

    Deliberately imports rather than probing ``nvidia-smi``: a visible device is not
    enough, because gpu4pyscf also needs the CUDA 12 runtime wheels and fails at import
    with ``Failure finding "libnvJitLink.so"`` when they are missing -- a state in which
    ``import cupy`` still succeeds and still reports a device.
    """
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        from gpu4pyscf.dft import RKS  # noqa: F401

        return True
    except Exception:
        return False


def device_name() -> str:
    """Human-readable name of GPU 0, for the log header and the result file."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return "GPU"


# ---------------------------------------------------------------------------
# Building the calculation
# ---------------------------------------------------------------------------

def build_mol(spec: dict[str, Any], verbose: int = 4):
    """Build the PySCF ``Mole`` described by ``spec``.

    Two things here are easy to get wrong and produce plausible-looking wrong numbers
    rather than an error:

    * ``spin`` is **2S**, not the multiplicity. Psi4 takes 2S+1, so a straight copy of the
      old spec field would silently run every triplet as a quintet.
    * ``symmetry=False`` is deliberate. It keeps PySCF from reorienting the molecule onto
      its principal axes -- which is what the old ``no_reorient``/``no_com`` flags bought,
      and what keeps cube grids and written-out geometries in the frame the user saw -- and
      it removes the per-irrep excited-root bookkeeping that forced the Psi4 version to
      drop symmetry for excited-state optimizations anyway.
    """
    from pyscf import gto

    multiplicity = int(spec["multiplicity"])
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")

    return gto.M(
        atom=spec["geometry"],
        basis=spec["basis_set"],
        charge=int(spec["charge"]),
        spin=multiplicity - 1,
        unit="Angstrom",
        symmetry=False,
        verbose=verbose,
        max_memory=float(spec.get("memory_gb", 4)) * 1024.0,
    )


def build_scf(spec: dict[str, Any], mol, use_gpu: bool):
    """Build the (unconverged) SCF object for ``spec``.

    Returns the mean-field object only; the caller runs ``kernel()``. Correlated methods
    are layered on top of the SCF this returns.
    """
    from .utils import method_uses_functional

    method_type = spec["method_type"]
    reference = spec.get("reference", "RHF")
    # Only DFT-family jobs consume a functional. MP2/CCSD/CCSD(T) are built on an HF
    # reference; TD-DFT and the excited-state optimization are built on a KS reference.
    scf_is_dft = method_uses_functional(method_type)

    if use_gpu:
        from gpu4pyscf import dft as dft_module, scf as scf_module
    else:
        from pyscf import dft as dft_module, scf as scf_module

    if scf_is_dft:
        classes = {"RHF": dft_module.RKS, "ROHF": dft_module.ROKS, "UHF": dft_module.UKS}
    else:
        classes = {"RHF": scf_module.RHF, "ROHF": scf_module.ROHF, "UHF": scf_module.UHF}
    if reference not in classes:
        raise ValueError(f"unknown reference {reference!r}; expected one of {sorted(classes)}")

    mf = classes[reference](mol)

    if scf_is_dft:
        xc, dispersion = split_dispersion(spec.get("functional") or "B3LYP")
        mf.xc = xc
        if dispersion:
            mf.disp = dispersion
        mf.grids.level = int(spec.get("grid_level", 3))

    # Density fitting is the default for DFT (as it was in the Psi4 version, via
    # SCF_TYPE=DF) and is much of what makes the GPU path worth having; the conventional
    # algorithm is kept for HF and the correlated methods so small reference calculations
    # do not carry density-fitting error.
    if spec.get("density_fit", scf_is_dft):
        mf = mf.density_fit()

    if spec.get("use_solvation"):
        mf = _attach_solvent(mf, spec)

    mf.conv_tol = float(spec.get("scf_conv_tol", 1e-9))
    mf.max_cycle = int(spec.get("scf_max_cycle", 100))
    return mf


def _attach_solvent(mf, spec: dict[str, Any]):
    """Wrap ``mf`` in the requested continuum solvation model."""
    name = spec.get("solvent") or "Water"
    if name not in SOLVENT_DB:
        raise ValueError(f"unknown solvent {name!r}; expected one of {sorted(SOLVENT_DB)}")
    eps, smd_name = SOLVENT_DB[name]

    if str(spec.get("solvent_model", "SMD")).upper() == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = smd_name
    else:
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps = eps
    return mf


def correlated_energy(spec: dict[str, Any], mf) -> float:
    """Run the post-HF correlation step on top of a converged ``mf``.

    Always on the CPU. gpu4pyscf ships DF-MP2 and an in-core restricted CCSD, but no
    CCSD(T) and no open-shell coupled cluster, so routing the whole family through PySCF
    keeps one code path instead of four. The SCF -- which dominates the cost at the
    molecule sizes this UI targets -- has already run on whichever device the user chose.
    """
    method_type = spec["method_type"]
    mf_cpu = mf.to_cpu() if hasattr(mf, "to_cpu") else mf

    if method_type == "MP2":
        from pyscf import mp

        return float(mp.MP2(mf_cpu).run().e_tot)

    from pyscf import cc

    ccsd = cc.CCSD(mf_cpu).run()
    if not ccsd.converged:
        raise ConvergenceError(
            "CCSD did not converge; its energy and any (T) correction cannot be reported "
            "as a completed calculation. Adjust the method or SCF reference and retry."
        )
    if method_type == "CCSD(T)":
        return float(ccsd.e_tot + ccsd.ccsd_t())
    return float(ccsd.e_tot)


# ---------------------------------------------------------------------------
# Reading results off a converged calculation
# ---------------------------------------------------------------------------

def run_scf(mf) -> float:
    """Run SCF and reject unconverged orbitals before computing derived properties."""
    energy = float(to_numpy(mf.kernel()))
    if not mf.converged:
        # PySCF returns an energy even when max_cycle is exhausted. Treating that as
        # success can silently feed an unconverged wavefunction into TD-DFT or NMR.
        raise ConvergenceError(
            f"SCF did not converge within {mf.max_cycle} cycles. Increase the SCF cycle "
            "limit or adjust the reference, method, or starting geometry and retry."
        )
    return energy


def scf_summary(mf) -> dict[str, Any]:
    """Orbital energies, occupations and the dipole -- fields every result carries."""
    import numpy

    mo_energy = to_numpy(mf.mo_energy)
    mo_occ = to_numpy(mf.mo_occ)

    if mo_energy.ndim == 2:  # unrestricted: (2, nmo)
        alpha_energies = mo_energy[0]
        n_alpha = int((mo_occ[0] > 0).sum())
        n_beta = int((mo_occ[1] > 0).sum())
    else:
        alpha_energies = mo_energy
        # Restricted orbitals hold two electrons each, so the alpha count is the number of
        # orbitals with any occupation; an ROHF singly-occupied orbital is counted here and
        # not in n_beta, which is what makes n_alpha - 1 the highest occupied alpha index.
        n_alpha = int((mo_occ > 0).sum())
        n_beta = int((mo_occ > 1).sum())

    summary: dict[str, Any] = {
        "mo_energies_hartree": [float(e) for e in alpha_energies],
        # 0-based index of the highest occupied alpha orbital. -1 when there are no
        # electrons at all, which the Result tab reads as "no HOMO to label".
        "homo_index": n_alpha - 1,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
    }

    try:
        # PySCF returns Debye directly when asked, so unlike Psi4 there is no au-to-Debye
        # conversion to get wrong. This also prints the dipole into the log, which is
        # wanted here: the log is a generated artifact, not a byproduct.
        vector = numpy.asarray(to_numpy(mf.dip_moment(unit="Debye")), dtype=float).ravel()
        summary["dipole_debye"] = float(numpy.linalg.norm(vector))
        summary["dipole_vector_debye"] = [float(v) for v in vector]
    except Exception as exc:
        print(f"Warning: dipole moment unavailable ({exc}).")
        summary["dipole_debye"] = None
        summary["dipole_vector_debye"] = None

    return summary


def frequency_analysis(mf, spec: dict[str, Any], use_gpu: bool) -> tuple[dict[str, Any], dict]:
    """Hessian, normal modes, IR intensities and thermochemistry.

    Returns ``(result_fragment, arrays)`` -- the scalars for ``result.json`` and the bulky
    arrays for the HDF5 sidecar and the Molden normal-mode file.

    IR intensities are the awkward part. ``pyscf.hessian.thermo.harmonic_analysis`` carries
    a literal ``#TODO: IR intensity``, so on the CPU there simply are none; the only
    implementation in this stack is ``gpu4pyscf.properties.ir``, which asserts a
    *restricted* reference. When intensities are unavailable the frequencies are still
    returned and ``ir_intensities_km_mol`` is ``None``, which the Result tab renders as a
    stick spectrum with a warning rather than failing an otherwise-complete calculation.
    """
    import numpy
    from pyscf.hessian import thermo

    mol = mf.mol
    intensities = None
    reference_frequencies = None

    if use_gpu and _supports_gpu_ir(mf):
        hessian_obj = None
        try:
            from gpu4pyscf.properties import ir as gpu_ir

            hessian_obj = mf.Hessian()
            reference_frequencies, intensities = gpu_ir.eval_ir_freq_intensity(mf, hessian_obj)
            reference_frequencies = numpy.asarray(
                to_numpy(reference_frequencies), dtype=float).ravel()
            intensities = numpy.asarray(to_numpy(intensities), dtype=float).ravel()
        except Exception as exc:
            # IR also solves a dipole response after the Hessian. Failure in that optional
            # property must not discard the geometry or the already-computed frequencies.
            print(f"Warning: GPU IR intensities unavailable ({type(exc).__name__}: {exc}); "
                  "reporting frequencies without intensities.")
            reference_frequencies = intensities = None

        # eval_ir_freq_intensity leaves a completed Hessian on the object even if the
        # later dipole response fails. An uncomputed GPU Hessian starts with an EMPTY
        # .de array, so checking only for None would accept it as a usable Hessian.
        hessian = to_numpy(getattr(hessian_obj, "de", None))
        if hessian is None or hessian.shape != (mol.natm, mol.natm, 3, 3):
            mf, hessian = _hessian_with_cpu_fallback(mf)
    else:
        if use_gpu:
            print("Note: IR intensities are only implemented for a restricted reference; "
                  "reporting frequencies without intensities.")
        mf, hessian = _hessian_with_cpu_fallback(mf)

    harmonic = thermo.harmonic_analysis(mol, hessian)
    frequencies = harmonic["freq_wavenumber"]
    if numpy.iscomplexobj(frequencies):
        # harmonic_analysis stores imaginary modes as complex numbers; the rest of this app
        # (and every other QC code) expects them as negative reals. This is the same
        # collapse thermo.dump_normal_mode performs before printing.
        frequencies = frequencies.real - abs(frequencies.imag)
    frequencies = numpy.asarray(frequencies, dtype=float).ravel()

    intensities = _align_intensities(frequencies, reference_frequencies, intensities)

    temperature = float(spec.get("temperature", 298.15))
    pressure = float(spec.get("pressure", 101325.0))

    fragment = {
        "frequency": {
            "frequencies_cm1": [float(f) for f in frequencies],
            "ir_intensities_km_mol": (
                [float(i) for i in intensities] if intensities is not None else None
            ),
            # A true minimum has none, a transition state exactly one. Surfaced so the
            # Result tab can warn rather than leaving the user to spot a minus sign.
            "n_imaginary": int((frequencies < 0).sum()),
        },
        "thermochemistry": _thermochemistry(mf, harmonic, temperature, pressure),
    }

    arrays = {
        "hessian": numpy.asarray(hessian, dtype=float),
        "freq_wavenumber": frequencies,
        "norm_mode": numpy.asarray(numpy.real(harmonic["norm_mode"]), dtype=float),
        "reduced_mass": numpy.asarray(numpy.real(harmonic["reduced_mass"]), dtype=float),
    }
    if intensities is not None:
        arrays["ir_intensity"] = intensities
    return fragment, arrays


def _align_intensities(frequencies, reference_frequencies, intensities):
    """Reorder ``intensities`` onto ``frequencies``, or drop them if they cannot be matched.

    The two arrays come from independent projections of the translations and rotations, so
    their mode ordering is not guaranteed to agree. A silent mismatch would give every peak
    some other mode's height -- a wrong spectrum that still looks entirely plausible -- so
    the pairing is verified rather than assumed, and abandoned rather than guessed at.
    """
    import numpy

    if intensities is None:
        return None
    if intensities.size != frequencies.size:
        print(f"Warning: got {intensities.size} IR intensities for {frequencies.size} "
              f"normal modes; dropping them rather than mis-assigning peaks.")
        return None

    order = numpy.argsort(reference_frequencies)
    target = numpy.argsort(frequencies)
    if not numpy.allclose(reference_frequencies[order], frequencies[target], atol=1.0):
        print("Warning: IR intensities could not be matched to the normal modes; "
              "dropping them rather than mis-assigning peaks.")
        return None

    aligned = numpy.empty_like(intensities)
    aligned[target] = intensities[order]
    return aligned


def _hessian_with_cpu_fallback(mf):
    """Compute the Hessian, retrying on the CPU if the GPU implementation fails.

    Returns ``(mf, hessian)`` -- the method object too, because a fallback replaces it
    with its CPU counterpart and the caller needs that for the thermochemistry.

    gpu4pyscf's coverage is narrower than PySCF's and the gaps are not always declared: in
    1.8.1, an unrestricted DF Hessian dies on a bare ``assert wv.flags.c_contiguous`` deep
    inside ``hessian/uks.py``. Falling back costs a repeat of the Hessian on the CPU, which
    is far better than discarding a converged geometry optimization over an upstream bug --
    and the fallback is written against the failure rather than against a list of known-bad
    combinations, so it also covers whatever else is missing.
    """
    try:
        return mf, to_numpy(mf.Hessian().kernel())
    except Exception as exc:
        if not hasattr(mf, "to_cpu"):
            raise
        print(f"Warning: the GPU Hessian failed ({type(exc).__name__}: {exc}); "
              f"retrying on the CPU. This is slower but gives the same result.")
        mf_cpu = mf.to_cpu()
        return mf_cpu, to_numpy(mf_cpu.Hessian().kernel())


def _supports_gpu_ir(mf) -> bool:
    """Whether ``gpu4pyscf.properties.ir`` will accept ``mf``.

    It asserts a restricted reference. ``RKS`` subclasses ``gpu4pyscf.scf.hf.RHF`` so
    restricted DFT passes, but UHF/UKS does not -- and an ``AssertionError`` raised from
    inside a two-hour Hessian is a poor way to discover that.
    """
    try:
        from gpu4pyscf.scf.hf import RHF

        return isinstance(mf, RHF)
    except Exception:
        return False


def _thermochemistry(mf, harmonic, temperature: float, pressure: float) -> dict[str, Any]:
    """Standard-state thermochemistry from the harmonic frequencies."""
    import numpy
    from pyscf.hessian import thermo

    result: dict[str, Any] = {"temperature_k": temperature, "pressure_pa": pressure}
    try:
        # thermo() reads only .mol and .e_tot off the model it is handed, so a plain shim
        # keeps this working for GPU objects whose e_tot may be a device scalar.
        class _Model:
            mol = mf.mol
            e_tot = float(to_numpy(mf.e_tot))

        data = thermo.thermo(_Model(), harmonic["freq_au"], temperature, pressure)
    except Exception as exc:
        print(f"Warning: thermochemistry unavailable ({exc}).")
        for key in ("zpve_hartree", "thermal_energy_hartree", "enthalpy_hartree",
                    "gibbs_hartree", "entropy_cal_mol_k"):
            result[key] = None
        return result

    def _value(key):
        entry = data.get(key)
        if entry is None:
            return None
        value = entry[0] if isinstance(entry, tuple) else entry
        return float(numpy.real(value))

    result["zpve_hartree"] = _value("ZPE")
    result["thermal_energy_hartree"] = _value("E_tot")
    result["enthalpy_hartree"] = _value("H_tot")
    result["gibbs_hartree"] = _value("G_tot")

    entropy = _value("S_tot")  # Eh/K
    if entropy is not None:
        result["entropy_cal_mol_k"] = entropy * HARTREE2KCALMOL * 1000.0
    elif (result["enthalpy_hartree"] is not None and result["gibbs_hartree"] is not None
          and temperature > 0):
        # S = (H - G) / T, the fallback the Psi4 version used, in cal/(mol K).
        result["entropy_cal_mol_k"] = (
            (result["enthalpy_hartree"] - result["gibbs_hartree"]) / temperature
            * HARTREE2KCALMOL * 1000.0
        )
    else:
        result["entropy_cal_mol_k"] = None
    return result



def nmr_unavailable_reason(spec: dict[str, Any], use_gpu: bool) -> str | None:
    """Why an NMR job cannot run, or ``None`` if it can.

    Checked up front, because every one of these failures otherwise surfaces from deep
    inside a library as something unactionable: a bare ``AssertionError`` with no message
    for a range-separated functional, an ``AttributeError`` about ``_numint`` for HF, and a
    tuple-unpacking ``ValueError`` for an open-shell reference.
    """
    if not use_gpu:
        return ("NMR shielding requires the GPU. gpu4pyscf's implementation is the only one "
                "in this stack -- PySCF itself has no CPU equivalent installed -- so there "
                "is nothing to fall back to. (J-couplings, by contrast, are CPU-only.)")
    from .utils import method_uses_functional

    if not method_uses_functional(spec.get("method_type")):
        return (f"NMR shielding requires a DFT method; {spec.get('method_type')} has no "
                f"exchange-correlation functional for the shielding code to read.")
    if spec.get("reference", "RHF") != "RHF":
        return ("NMR shielding requires a restricted (RHF) reference; the implementation "
                "assumes doubly-occupied orbitals.")

    functional = spec.get("functional") or ""
    xc, _dispersion = split_dispersion(functional)
    try:
        from pyscf.dft import libxc

        if libxc.rsh_coeff(xc)[0] != 0.0:
            return (f"{functional} is a range-separated functional, which the NMR shielding "
                    f"code does not support. Use a global hybrid or a pure functional "
                    f"(B3LYP, PBE0, KT2, ...).")
    except Exception:
        pass  # an unknown name will fail more informatively when the Mole is built
    return None


def nmr_shielding(mf, spec: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    """GIAO NMR shielding tensors for every nucleus.

    Returns ``(result_fragment, arrays)``. The fragment carries **absolute** shieldings
    only -- no chemical shifts. Referencing is deliberately left to the Result tab: a shift
    is just ``reference - sigma``, and keeping the reference out of the stored result means
    it can be changed and the table re-rendered without re-running the calculation.

    ``eval_shielding`` returns the diamagnetic and paramagnetic contributions separately,
    each as an ``(natm, 3, 3)`` tensor in ppm. Both are kept in the HDF5 sidecar; the table
    needs only the isotropic average and the anisotropy.
    """
    import numpy
    from gpu4pyscf.properties import shielding as gpu_shielding

    dia, para = gpu_shielding.eval_shielding(mf)
    dia = numpy.asarray(to_numpy(dia), dtype=float)
    para = numpy.asarray(to_numpy(para), dtype=float)
    total = dia + para

    mol = mf.mol
    entries = []
    for index in range(mol.natm):
        tensor = total[index]
        isotropic = float(numpy.trace(tensor) / 3.0)
        # The shielding tensor is not symmetric in general, and only its symmetric part is
        # observable; the anisotropy is defined from the principal values of that part.
        eigenvalues = numpy.sort(numpy.linalg.eigvalsh((tensor + tensor.T) / 2.0))
        anisotropy = float(eigenvalues[2] - (eigenvalues[0] + eigenvalues[1]) / 2.0)
        entries.append({
            "index": index,
            "symbol": mol.atom_symbol(index),
            "isotropic_ppm": isotropic,
            "anisotropy_ppm": anisotropy,
        })

    fragment = {
        "nmr": {
            "shieldings": entries,
            # Named so the Result tab can say what the numbers are without guessing, and so
            # a future J-coupling array has an obvious place to sit beside them.
            "gauge": "GIAO",
            "functional": spec.get("functional"),
            "basis_set": spec.get("basis_set"),
        }
    }
    arrays = {
        "nmr_shielding_total": total,
        "nmr_shielding_dia": dia,
        "nmr_shielding_para": para,
    }
    return fragment, arrays


# Which nuclei J-couplings may be computed between. The cost is driven by the number of
# distinct atoms involved -- one set of CPHF solves each -- not by the pair count, so this
# choice is the main lever a user has over how long an NMR job takes.
J_COUPLING_NUCLEI = {
    "H": ("H",),
    "H,C": ("H", "C"),
}
DEFAULT_J_COUPLING_NUCLEI = "H"


def spin_spin_couplings(mf, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    """Indirect nuclear spin-spin coupling constants J, in Hz.

    Returns ``(couplings, arrays)``. Each entry is one nucleus pair with its isotropic J.

    **CPU only.** gpu4pyscf has no spin-spin module at all; ``pyscf.prop.ssc`` is the only
    implementation in this stack and is pure PySCF. A GPU mean field is therefore moved to
    the host first -- which converts the converged orbitals rather than re-running the SCF,
    so the expensive part still ran on the device.

    Cost scales with the number of *distinct* nuclei requested, because a CPHF equation is
    solved per atom per perturbation. Restricting to hydrogen is roughly twice as fast as
    hydrogen plus carbon on a small aromatic, and the difference grows.
    """
    import numpy
    from pyscf.data import nist
    from pyscf.data.gyro import get_nuc_g_factor

    try:
        from pyscf.prop import ssc as ssc_module
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "J-couplings need the pyscf-properties package (pyscf.prop.ssc), which is not "
            "installed. Run `pip install pyscf-properties`, or clear the "
            "'Compute J-couplings' checkbox to get shieldings only."
        ) from exc

    mf_cpu = mf.to_cpu() if hasattr(mf, "to_cpu") else mf
    mol = mf_cpu.mol
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    elements = J_COUPLING_NUCLEI.get(
        spec.get("j_coupling_nuclei", DEFAULT_J_COUPLING_NUCLEI),
        J_COUPLING_NUCLEI[DEFAULT_J_COUPLING_NUCLEI])
    pairs = [(i, j) for i in range(mol.natm) for j in range(i)
             if symbols[i] in elements and symbols[j] in elements]
    if not pairs:
        print(f"No {'/'.join(elements)} nucleus pairs to couple; skipping J-couplings.")
        return [], {}

    # RKS/RHF chosen by whether the reference carries a functional, the same split
    # build_scf made. Only closed-shell references are offered for NMR, so there is no
    # unrestricted branch to pick here.
    is_dft = hasattr(mf_cpu, "xc")
    solver = ssc_module.RKS(mf_cpu) if is_dft else ssc_module.RHF(mf_cpu)
    solver.nuc_pair = pairs
    print(f"Computing J-couplings for {len(pairs)} nucleus pairs "
          f"({len({a for p in pairs for a in p})} distinct nuclei) on the CPU...",
          flush=True)
    tensors = solver.kernel()

    # kernel() returns the raw coupling tensors in atomic units; the conversion to hertz
    # is done in its *logging* path, so it has to be repeated here rather than read off.
    # Reproduced from pyscf/prop/ssc/rhf.py so the numbers match what PySCF prints.
    nuclear_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
    au_to_hz = (nist.HARTREE2J / nist.PLANCK) * nuclear_magneton ** 2
    reduced = au_to_hz * numpy.einsum("kii->k", tensors) / 3
    # J = K * g(A) * g(B): the reduced coupling times both nuclear g factors, which is what
    # makes J isotope-specific while K is not.
    g_factors = numpy.array([get_nuc_g_factor(s) for s in symbols])

    couplings = []
    for index, (i, j) in enumerate(pairs):
        couplings.append({
            "atom_a": i, "symbol_a": symbols[i],
            "atom_b": j, "symbol_b": symbols[j],
            "j_hz": float(reduced[index] * g_factors[i] * g_factors[j]),
        })
    arrays = {"j_coupling_tensors": numpy.asarray(tensors, dtype=float),
              "j_coupling_pairs": numpy.asarray(pairs, dtype=int)}
    return couplings, arrays

def build_tddft(mf, spec: dict[str, Any]):
    """Construct and solve the TD-DFT problem for ``spec``, returning the solver object."""
    n_states = int(spec.get("n_states", 10))
    root = int(spec.get("root", 1))
    if n_states < 1 or root < 1:
        raise ValueError("The number of excited states and the excited-state root must be >= 1.")
    td = mf.TDA() if spec.get("tda") else mf.TDDFT()
    # Asking for fewer states than the root being followed would make the Emission branch
    # index past the end of the solution vector.
    td.nstates = max(n_states, root)
    td.kernel()
    if not to_numpy(td.converged).all():
        raise ConvergenceError(
            "TD-DFT did not converge for every computed excited state. Try TDA or adjust "
            "the method, basis, or number of requested states and retry."
        )
    if len(td.e) < root:
        raise ValueError(
            f"Excited-state root {root} was requested, but the solver returned only "
            f"{len(td.e)} states. Choose a lower root or a larger basis."
        )
    return td


def excitation_results(td) -> tuple[dict[str, Any], dict]:
    """Tabulate excited states: energies, oscillator strengths, rotatory strengths.

    PySCF has ``transition_dipole`` and ``transition_magnetic_dipole`` but no rotatory
    strength function, so ``R`` is assembled here. The factor of 500 and the sign
    convention come from ``gpu4pyscf.tdscf.spectralib``, which uses them to reproduce
    Gaussian 16's ECD output -- worth matching, since that is what users compare against.
    """
    import numpy
    from pyscf.tdscf.rhf import transition_dipole, transition_magnetic_dipole

    # The transition-moment helpers work on host arrays, so a GPU solver is moved across
    # first. oscillator_strength is inherited from the CPU implementation either way.
    td_host = td
    if type(td).__module__.startswith("gpu4pyscf"):
        try:
            td_host = td.to_cpu()
        except Exception as exc:
            print(f"Note: could not move the TD-DFT solver to the host ({exc}); "
                  f"computing properties on the device object.")

    energies = numpy.asarray(to_numpy(td_host.e), dtype=float).ravel()
    try:
        strengths = numpy.asarray(
            to_numpy(td_host.oscillator_strength(gauge="length")), dtype=float).ravel()
    except Exception as exc:
        print(f"Warning: oscillator strengths unavailable ({exc}).")
        strengths = None

    try:
        electric = numpy.asarray(to_numpy(transition_dipole(td_host)), dtype=float)
        magnetic = numpy.asarray(to_numpy(transition_magnetic_dipole(td_host)), dtype=float)
        rotatory = 500.0 * numpy.einsum("ix,ix->i", electric, magnetic)
    except Exception as exc:
        print(f"Warning: rotatory strengths unavailable ({exc}); the ECD plot will be "
              f"disabled for this calculation.")
        electric = magnetic = None
        rotatory = None

    excitations = []
    for index, energy in enumerate(energies, start=1):
        wavenumber = float(energy) * HARTREE2WAVENUMBER
        excitations.append({
            "index": index,
            "energy_hartree": float(energy),
            "energy_ev": float(energy) * HARTREE2EV,
            "wavenumber_cm1": wavenumber,
            # 1e7 nm per cm; a non-positive root would be unphysical, so guard the divide.
            "wavelength_nm": (1.0e7 / wavenumber) if wavenumber > 0 else None,
            # Missing transition properties are unknown, not physically zero (a forbidden
            # transition or an achiral response). The table and plots distinguish these.
            "oscillator_strength": (
                float(strengths[index - 1]) if strengths is not None else None),
            "rotatory_strength": (
                float(rotatory[index - 1]) if rotatory is not None else None),
            "spin": "",
            # PySCF runs without symmetry here (see build_mol), so there are no irrep
            # labels to report. The column stays in the schema so the Result tab and any
            # older Psi4 result file keep the same shape.
            "symmetry": "",
        })

    arrays: dict[str, Any] = {
        "excitation_energies_hartree": energies,
    }
    if strengths is not None:
        arrays["oscillator_strength"] = strengths
    if rotatory is not None:
        arrays["rotatory_strength"] = rotatory
    if electric is not None:
        arrays["transition_dipole"] = electric
        arrays["transition_magnetic_dipole"] = magnetic
    return {"excitations": excitations}, arrays


# ---------------------------------------------------------------------------
# Artifact writers
#
# PySCF produces no files of its own beyond what it prints, so everything the Result tab
# or an external viewer might want has to be written out here explicitly.
# ---------------------------------------------------------------------------

def save_chkfile(mol, mf, path: str) -> str:
    """Save the converged wavefunction as a PySCF HDF5 checkpoint.

    This is the replacement for Psi4's ``wfn.to_file()`` ``.npy``: it is what lets the
    Orbitals & Density viewer generate cubes later without re-running the SCF.

    Written field by field rather than by relying on ``mf.chkfile`` auto-save, because the
    fields are pulled through :func:`to_numpy` on the way -- a gpu4pyscf run would
    otherwise try to write cupy arrays into HDF5 -- and because a solvent- or
    density-fitting-wrapped object does not reliably carry the auto-save through.
    """
    from pyscf.lib import chkfile

    chkfile.save_mol(mol, path)
    chkfile.save(path, "scf", {
        "e_tot": float(to_numpy(mf.e_tot)),
        "mo_energy": to_numpy(mf.mo_energy),
        "mo_coeff": to_numpy(mf.mo_coeff),
        "mo_occ": to_numpy(mf.mo_occ),
    })
    return os.path.basename(path)


def load_chkfile(path: str):
    """Reload ``(mol, mo_coeff, mo_occ, mo_energy)`` written by :func:`save_chkfile`."""
    from pyscf.lib import chkfile

    mol = chkfile.load_mol(path)
    scf_data = chkfile.load(path, "scf")
    return (mol, to_numpy(scf_data["mo_coeff"]), to_numpy(scf_data["mo_occ"]),
            to_numpy(scf_data["mo_energy"]))


def density_matrices(mo_coeff, mo_occ):
    """Alpha and beta density matrices from MO coefficients and occupations.

    Returns ``(dm_alpha, dm_beta)``. Closed-shell restricted orbitals give identical
    densities; restricted open-shell orbitals put the singly occupied electrons in alpha,
    matching PySCF's positive-spin convention. The total is always ``dm_alpha + dm_beta``
    and the spin density always ``dm_alpha - dm_beta``.
    """
    import numpy

    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)
    if mo_coeff.ndim == 3:  # unrestricted
        dm_alpha = (mo_coeff[0] * mo_occ[0]).dot(mo_coeff[0].conj().T)
        dm_beta = (mo_coeff[1] * mo_occ[1]).dot(mo_coeff[1].conj().T)
        return dm_alpha, dm_beta
    # Halving ROHF/ROKS occupations would erase the spin density entirely while still
    # producing the correct total density, making a plausible but wrong cube file.
    occ_alpha = numpy.minimum(mo_occ, 1.0)
    occ_beta = numpy.maximum(mo_occ - 1.0, 0.0)
    dm_alpha = (mo_coeff * occ_alpha).dot(mo_coeff.conj().T)
    dm_beta = (mo_coeff * occ_beta).dot(mo_coeff.conj().T)
    return dm_alpha, dm_beta


def save_molden(mol, mo_coeff, mo_occ, mo_energy, path: str) -> str:
    """Write the orbitals in Molden format.

    Nothing in this app reads it back -- it exists so that a finished job opens directly in
    Avogadro, Jmol, Multiwfn or GaussView. It costs one file write and makes every
    calculation portable out of the tool.
    """
    import numpy
    from pyscf.tools import molden

    unrestricted = numpy.asarray(mo_coeff).ndim == 3
    with open(path, "w") as handle:
        molden.header(mol, handle, ignore_h=True)
        if unrestricted:
            molden.orbital_coeff(mol, handle, mo_coeff[0], spin="Alpha",
                                 ene=mo_energy[0], occ=mo_occ[0], ignore_h=True)
            molden.orbital_coeff(mol, handle, mo_coeff[1], spin="Beta",
                                 ene=mo_energy[1], occ=mo_occ[1], ignore_h=True)
        else:
            molden.orbital_coeff(mol, handle, mo_coeff, spin="Alpha",
                                 ene=mo_energy, occ=mo_occ, ignore_h=True)
    return os.path.basename(path)


def save_frequency_molden(mol, arrays, path: str) -> str:
    """Write normal modes in Molden's vibration format.

    PySCF has no writer for this, so the sections are emitted by hand. The payoff is that
    the modes animate in any Molden-aware viewer, which is the usual way to find out what
    an imaginary frequency actually is. ``[FR-COORD]`` is in Bohr, unlike the angstrom used
    everywhere else in this app.
    """
    import numpy

    frequencies = numpy.asarray(arrays["freq_wavenumber"], dtype=float).ravel()
    modes = numpy.asarray(arrays["norm_mode"], dtype=float)
    coords = mol.atom_coords()  # Bohr
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    lines = ["[Molden Format]", "[FREQ]"]
    lines += [f"{f:12.4f}" for f in frequencies]
    lines.append("[FR-COORD]")
    for symbol, (x, y, z) in zip(symbols, coords):
        lines.append(f"{symbol:<3s} {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.append("[FR-NORM-COORD]")
    for index, mode in enumerate(modes, start=1):
        lines.append(f"vibration {index}")
        for dx, dy, dz in mode:
            lines.append(f"{dx:14.8f} {dy:14.8f} {dz:14.8f}")
    intensities = arrays.get("ir_intensity")
    if intensities is not None:
        lines.append("[INT]")
        lines += [f"{float(i):12.4f}" for i in numpy.asarray(intensities).ravel()]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return os.path.basename(path)


def save_arrays(arrays: dict, path: str) -> str | None:
    """Store bulky arrays in HDF5 alongside the result file.

    Hessians, normal modes and transition moments are megabytes of numbers that no part of
    the UI plots directly. Keeping them out of ``result.json`` is what lets that file stay
    small enough to open and read by hand, which is most of its value.
    """
    if not arrays:
        return None
    import h5py
    import numpy

    with h5py.File(path, "w") as handle:
        for key, value in arrays.items():
            if value is None:
                continue
            handle.create_dataset(key, data=numpy.asarray(value))
    return os.path.basename(path)
