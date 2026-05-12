"""
Phase A — TMM dataset generation for 1D GaAs / Al_0.3Ga_0.7As DBR.

Structure: (air) | [GaAs / Al_0.3Ga_0.7As] × N_periods | GaAs substrate
Parameter space (IDEA_001):
    N_periods : int   in [5, 20]
    t_GaAs    : float in [50, 200] nm
    t_AlGaAs  : float in [50, 200] nm
    x_Al      : FIXED at 0.30

Wavelength range: 900–1100 nm (GaAs transparent window, below 1.424 eV bandgap)
  → appropriate for 940–1060 nm VCSEL / GaAs quantum-dot emitter DBRs

Refractive index model: modified Cauchy dispersion
  n(λ, x) = [a₀ + Δa·x] + [b₀ + Δb·x] / λ²   (λ in nm)

  Calibrated against Palik "Handbook of Optical Constants" (1985) tabulated n
  for GaAs (x=0) and AlAs (x=1) at RT in the 900–1100 nm window:
    GaAs  (x=0): a=3.287, b=251 832 nm²  → n(900)=3.598, n(1000)=3.539, n(1100)=3.495
    AlAs  (x=1): a=2.822, b=100 245 nm²  → n(900)=2.946, n(1000)=2.922, n(1100)=2.905
  Linear interpolation in x for Al_xGa_{1-x}As.

Saves: data/dataset.npz
    params         : (N, 3) float32  columns = [N_periods, t_GaAs_nm, t_AlGaAs_nm]
    spectra        : (N, 150) float32  R(λ) in [0,1]
    wavelengths_nm : (150,) float32
    x_al_fixed     : float32 scalar
    seed           : int32 scalar
"""

import os
import time
import numpy as np
from scipy.stats.qmc import LatinHypercube
import tmm

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── constants ──────────────────────────────────────────────────────────────────
N_SAMPLES       = 1500
X_AL_FIXED      = 0.30
WAVELENGTHS     = np.linspace(900, 1100, 150, dtype=np.float64)   # nm
N_WL            = len(WAVELENGTHS)

N_PERIODS_RANGE = (5,   20)    # integer
T_GAAS_RANGE    = (50,  200)   # nm
T_ALGAAS_RANGE  = (50,  200)   # nm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(_SRC_DIR, '..', 'data')
OUT_FILE = os.path.join(OUT_DIR, 'dataset.npz')

# ── Cauchy dispersion for Al_x Ga_{1-x} As ────────────────────────────────────
# n(lambda) = a + b / lambda^2   [lambda in nm]
#
# Calibrated against Palik (1985) and Gehrsitz et al. J. Appl. Phys. 87, 7825 (2000).
# n(AlGaAs, x) is sub-linear in x due to bowing; direct fits at each composition
# are used rather than linear endpoint interpolation.
#
#   x=0.0 GaAs  : a=3.287, b=251832  -> n(900)=3.598, n(1000)=3.539, n(1100)=3.495
#   x=0.3 AlGaAs: a=3.159, b=122399  -> n(900)=3.310, n(1000)=3.281, n(1100)=3.260
#   x=0.9 AlGaAs: a=2.850, b=103500  -> n(900)=2.978, n(1000)=2.954, n(1100)=2.936
#   x=1.0 AlAs  : a=2.822, b=100245  -> n(900)=2.946, n(1000)=2.922, n(1100)=2.905

_CAUCHY_TABLE = {
    0.0: (3.287, 251_832.0),
    0.3: (3.159, 122_399.0),
    0.9: (2.850, 103_500.0),
    1.0: (2.822, 100_245.0),
}

def _cauchy_params(x: float):
    """Interpolate (a, b) for Al_x Ga_{1-x} As between tabulated compositions."""
    xs = sorted(_CAUCHY_TABLE.keys())
    if x <= xs[0]:
        return _CAUCHY_TABLE[xs[0]]
    if x >= xs[-1]:
        return _CAUCHY_TABLE[xs[-1]]
    x_lo = max(x0 for x0 in xs if x0 <= x)
    x_hi = min(x0 for x0 in xs if x0 >= x)
    if x_lo == x_hi:
        return _CAUCHY_TABLE[x_lo]
    t = (x - x_lo) / (x_hi - x_lo)
    a = _CAUCHY_TABLE[x_lo][0] * (1 - t) + _CAUCHY_TABLE[x_hi][0] * t
    b = _CAUCHY_TABLE[x_lo][1] * (1 - t) + _CAUCHY_TABLE[x_hi][1] * t
    return a, b


def n_algaas(wavelength_nm: np.ndarray, x: float) -> np.ndarray:
    """
    Real refractive index of Al_x Ga_{1-x} As via Cauchy dispersion.
    Valid for 900-1100 nm (below GaAs bandgap at 871 nm).
    """
    lam = np.asarray(wavelength_nm, dtype=np.float64)
    a, b = _cauchy_params(x)
    return (a + b / lam**2).astype(complex)


def _validate_dispersion():
    """Verify n values against Palik (1985) reference data."""
    checks = [
        (900.,  0.0,  3.598, "GaAs  n(900)"),
        (1000., 0.0,  3.539, "GaAs  n(1000)"),
        (1100., 0.0,  3.495, "GaAs  n(1100)"),
        (900.,  0.3,  3.300, "Al0.3 n(900)"),
        (1000., 0.3,  3.271, "Al0.3 n(1000)"),
        (900.,  1.0,  2.946, "AlAs  n(900)"),
    ]
    ok = True
    for wl, x, expected, label in checks:
        got = n_algaas(np.array([wl]), x).real[0]
        err = abs(got - expected)
        status = "OK" if err < 0.05 else "FAIL"
        if err >= 0.05:
            ok = False
        print(f"  [{status}]  {label:18s}  got {got:.4f}  expected {expected:.4f}  "
              f"delta={err:.4f}")
    if not ok:
        raise RuntimeError("Dispersion model validation failed — check Cauchy coefficients.")


# ── TMM single-spectrum simulation ────────────────────────────────────────────

def dbr_reflectance(
    n_periods:   int,
    t_GaAs_nm:   float,
    t_AlGaAs_nm: float,
    x_Al:        float = X_AL_FIXED,
    wavelengths_nm: np.ndarray = WAVELENGTHS,
) -> np.ndarray:
    """
    Normal-incidence reflectance spectrum R(λ) for a GaAs/AlGaAs DBR.

    Stack: air | [GaAs(t_GaAs) / Al_xGaAs(t_AlGaAs)] × N | GaAs substrate

    Returns
    -------
    R : float32 array, shape (len(wavelengths_nm),), values in [0, 1]
    """
    R = np.empty(len(wavelengths_nm), dtype=np.float64)
    for i, wl in enumerate(wavelengths_nm):
        n_inc    = complex(1.0)                          # air
        n_GaAs   = n_algaas(np.array([wl]), 0.0)[0]
        n_AlGaAs = n_algaas(np.array([wl]), x_Al)[0]
        n_sub    = n_GaAs                                # GaAs substrate

        n_list = [n_inc]
        d_list = [np.inf]
        for _ in range(n_periods):
            n_list += [n_GaAs,   n_AlGaAs]
            d_list += [t_GaAs_nm, t_AlGaAs_nm]
        n_list.append(n_sub)
        d_list.append(np.inf)

        R[i] = tmm.coh_tmm('s', n_list, d_list, 0.0, wl)['R']

    return R.astype(np.float32)


# ── Latin Hypercube Sampling ───────────────────────────────────────────────────

def generate_lhs_params(n_samples: int, seed: int = SEED) -> np.ndarray:
    """
    Returns (n_samples, 3) parameter matrix via Latin Hypercube Sampling.
    Columns: [N_periods (int), t_GaAs (nm), t_AlGaAs (nm)]
    """
    sampler = LatinHypercube(d=3, seed=seed)
    unit    = sampler.random(n=n_samples)   # (N, 3) in [0, 1)

    params = np.empty_like(unit)
    params[:, 0] = np.round(
        unit[:, 0] * (N_PERIODS_RANGE[1] - N_PERIODS_RANGE[0]) + N_PERIODS_RANGE[0]
    )
    params[:, 1] = unit[:, 1] * (T_GAAS_RANGE[1]   - T_GAAS_RANGE[0])   + T_GAAS_RANGE[0]
    params[:, 2] = unit[:, 2] * (T_ALGAAS_RANGE[1] - T_ALGAAS_RANGE[0]) + T_ALGAAS_RANGE[0]
    return params


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("AI-Laue — Phase A: TMM Dataset Generation")
    print(f"  Structure   : air | [GaAs/Al{X_AL_FIXED:.2f}GaAs]×N | GaAs substrate")
    print(f"  N_samples   : {N_SAMPLES}")
    print(f"  x_Al fixed  : {X_AL_FIXED}")
    print(f"  Wavelengths : {WAVELENGTHS[0]:.0f}–{WAVELENGTHS[-1]:.0f} nm  ({N_WL} points)")
    print()

    print("  Dispersion validation (Cauchy model vs. Palik 1985):")
    _validate_dispersion()
    print()

    params = generate_lhs_params(N_SAMPLES, seed=SEED)
    print(f"  LHS design matrix : {params.shape}  (seed={SEED})")
    print(f"  N_periods : {int(params[:,0].min())}–{int(params[:,0].max())}")
    print(f"  t_GaAs    : {params[:,1].min():.1f}–{params[:,1].max():.1f} nm")
    print(f"  t_AlGaAs  : {params[:,2].min():.1f}–{params[:,2].max():.1f} nm")
    print()

    spectra = np.empty((N_SAMPLES, N_WL), dtype=np.float32)
    t0 = time.time()

    for i, (n_per, t_g, t_a) in enumerate(params):
        spectra[i] = dbr_reflectance(
            n_periods   = int(n_per),
            t_GaAs_nm   = float(t_g),
            t_AlGaAs_nm = float(t_a),
        )
        if (i + 1) % 150 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (N_SAMPLES - i - 1)
            print(f"  [{i+1:4d}/{N_SAMPLES}]  {elapsed:5.0f}s elapsed  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    ms_per  = elapsed / N_SAMPLES * 1000
    print(f"\n  Done: {N_SAMPLES} spectra in {elapsed:.1f}s  ({ms_per:.2f} ms/spectrum)")

    np.savez(
        OUT_FILE,
        params         = params.astype(np.float32),
        spectra        = spectra,
        wavelengths_nm = WAVELENGTHS.astype(np.float32),
        x_al_fixed     = np.float32(X_AL_FIXED),
        seed           = np.int32(SEED),
    )
    print(f"\n  Saved  → {os.path.abspath(OUT_FILE)}")
    print(f"  params  : {params.shape}  (float32)")
    print(f"  spectra : {spectra.shape}  (float32)")
    print(f"  R range : [{spectra.min():.4f}, {spectra.max():.4f}]")


if __name__ == '__main__':
    main()
