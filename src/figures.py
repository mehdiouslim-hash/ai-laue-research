"""
Phase C -- Publication figures for AI-Laue IDEA_001.

Generates 5 figures (PDF + PNG, 300 DPI):
  fig1.pdf  -- Parity plot: GP predicted vs. TMM R at 3 representative wavelengths
  fig2.pdf  -- MAE heatmap over {t_GaAs, t_AlGaAs} space (fixed N_periods=10)
  fig3.pdf  -- Full spectrum comparison: 12 test samples, GP mean +/- 1-sigma vs TMM
  fig4.pdf  -- Stopband centre wavelength + peak reflectance scatter (GP vs TMM)
  fig5.pdf  -- Learning curve: RMSE vs N_train (subsampled GP)

All figures: 300 DPI, no title, 12pt axis labels, viridis/seismic palette.
"""

import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_SRC_DIR, '..', 'data',   'dataset.npz')
PRED_FILE = os.path.join(_SRC_DIR, '..', 'output', 'predictions.npz')
METR_FILE = os.path.join(_SRC_DIR, '..', 'output', 'metrics.json')
FIG_DIR   = os.path.join(_SRC_DIR, '..', 'output', 'figures')
SEED      = 42

os.makedirs(FIG_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':        12,
    'axes.labelsize':   13,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  10,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

def savefig(name: str):
    path_pdf = os.path.join(FIG_DIR, name + '.pdf')
    path_png = os.path.join(FIG_DIR, name + '.png')
    plt.savefig(path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}.pdf / .png")


def load_data():
    d   = np.load(DATA_FILE)
    p   = np.load(PRED_FILE)
    wl  = p['wavelengths_nm']
    return d, p, wl


# ── Figure 1: Parity plot ──────────────────────────────────────────────────────

def fig1_parity(p, wl):
    """GP predicted R vs TMM ground truth at 3 wavelengths, coloured by N_periods."""
    target_wls = [950., 1000., 1050.]
    wl_idx     = [int(np.argmin(np.abs(wl - t))) for t in target_wls]
    labels     = [f'{wl[i]:.0f} nm' for i in wl_idx]

    Y_true = p['Y_test']
    Y_pred = p['Y_test_gp']
    N_per  = p['X_test'][:, 0]   # N_periods column

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    cmap = plt.cm.viridis
    norm = Normalize(vmin=5, vmax=20)

    for ax, idx, lbl in zip(axes, wl_idx, labels):
        sc = ax.scatter(Y_true[:, idx], Y_pred[:, idx],
                        c=N_per, cmap=cmap, norm=norm,
                        s=18, alpha=0.7, linewidths=0)
        lo = min(Y_true[:, idx].min(), Y_pred[:, idx].min()) - 0.02
        hi = max(Y_true[:, idx].max(), Y_pred[:, idx].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=5)
        rmse_v = float(np.sqrt(np.mean((Y_true[:, idx] - Y_pred[:, idx])**2)))
        ax.text(0.05, 0.93, f'RMSE={rmse_v:.4f}',
                transform=ax.transAxes, fontsize=9, va='top')
        ax.set_xlabel('TMM reflectance $R$')
        ax.set_ylabel('GP prediction $\hat{R}$')
        ax.set_title(lbl, fontsize=12)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes, label='$N_\\mathrm{periods}$',
                 shrink=0.85, pad=0.02)
    savefig('fig1_parity')


# ── Figure 2: MAE heatmap ─────────────────────────────────────────────────────

def fig2_mae_heatmap(p):
    """Mean absolute error over {t_GaAs, t_AlGaAs} space."""
    X = p['X_test']            # (N, 3): N_periods, t_GaAs, t_AlGaAs
    Y_true = p['Y_test']
    Y_pred = p['Y_test_gp']
    mae_per_sample = np.mean(np.abs(Y_true - Y_pred), axis=1)

    t_GaAs   = X[:, 1]
    t_AlGaAs = X[:, 2]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(t_GaAs, t_AlGaAs,
                    c=mae_per_sample, cmap='plasma',
                    s=30, alpha=0.85, linewidths=0)
    fig.colorbar(sc, ax=ax, label='MAE in $R$')
    ax.set_xlabel('$t_\\mathrm{GaAs}$ (nm)')
    ax.set_ylabel('$t_\\mathrm{AlGaAs}$ (nm)')
    savefig('fig2_mae_heatmap')


# ── Figure 3: Spectrum comparison ─────────────────────────────────────────────

def fig3_spectra(p, wl):
    """12 random test samples: GP mean +/- 1-sigma vs TMM."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(p['Y_test']), size=12, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharey=False)
    axes_flat = axes.flatten()

    for k, i in enumerate(idx):
        ax = axes_flat[k]
        y_true = p['Y_test'][i]
        y_mean = p['Y_test_gp'][i]
        y_std  = p['Y_test_gp_std'][i]
        n_per  = int(p['X_test'][i, 0])
        t_g    = p['X_test'][i, 1]
        t_a    = p['X_test'][i, 2]

        ax.fill_between(wl, y_mean - y_std, y_mean + y_std,
                        alpha=0.25, color='tab:orange', label='GP $\pm 1\sigma$')
        ax.plot(wl, y_true, 'k-',    lw=1.2, label='TMM')
        ax.plot(wl, y_mean, '-',     lw=1.0, color='tab:orange', label='GP mean')
        ax.set_xlim(wl[0], wl[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.04, 0.95,
                f'N={n_per}\n$t_G$={t_g:.0f}\n$t_A$={t_a:.0f}',
                transform=ax.transAxes, fontsize=7, va='top')
        if k >= 8:
            ax.set_xlabel('Wavelength (nm)', fontsize=10)
        if k % 4 == 0:
            ax.set_ylabel('$R$', fontsize=10)

    # shared legend on first axis
    axes_flat[0].legend(fontsize=7, loc='upper right')
    savefig('fig3_spectra')


# ── Figure 4: Stopband centre + peak reflectance scatter ──────────────────────

def _stopband_props(spectra: np.ndarray, wl: np.ndarray):
    """Return (lambda_centre, R_peak) for each spectrum."""
    idx_peak  = np.argmax(spectra, axis=1)
    R_peak    = spectra[np.arange(len(spectra)), idx_peak]
    lam_centre = wl[idx_peak]
    return lam_centre, R_peak


def fig4_stopband(p, wl):
    """Scatter: GP-predicted vs TMM stopband centre and peak reflectance."""
    lam_true, R_true = _stopband_props(p['Y_test'],    wl)
    lam_pred, R_pred = _stopband_props(p['Y_test_gp'], wl)

    rmse_lam = float(np.sqrt(np.mean((lam_true - lam_pred)**2)))
    rmse_R   = float(np.sqrt(np.mean((R_true   - R_pred  )**2)))
    r2_lam   = 1 - np.sum((lam_true-lam_pred)**2) / np.sum((lam_true-np.mean(lam_true))**2)
    r2_R     = 1 - np.sum((R_true-R_pred)**2)     / np.sum((R_true-np.mean(R_true))**2)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, yt, yp, rmse_v, r2_v, xlab, ylab in [
        (axes[0], lam_true, lam_pred, rmse_lam, r2_lam,
         'TMM $\\lambda_\\mathrm{centre}$ (nm)', 'GP $\\hat{\\lambda}_\\mathrm{centre}$ (nm)'),
        (axes[1], R_true,   R_pred,   rmse_R,   r2_R,
         'TMM $R_\\mathrm{peak}$', 'GP $\\hat{R}_\\mathrm{peak}$'),
    ]:
        ax.scatter(yt, yp, s=20, alpha=0.6, linewidths=0, color='tab:blue')
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = (hi - lo) * 0.03
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], 'k--', lw=1.0)
        ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
        ax.text(0.05, 0.93, f'RMSE={rmse_v:.3f}\n$R^2$={r2_v:.4f}',
                transform=ax.transAxes, fontsize=9, va='top')
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    plt.tight_layout()
    savefig('fig4_stopband')


# ── Figure 5: Learning curve ──────────────────────────────────────────────────

def fig5_learning_curve():
    """RMSE vs N_train for GP (PCA) and RF; re-trains at each subset size."""
    d = np.load(DATA_FILE)
    X_all = d['params'].astype(np.float32)
    Y_all = d['spectra'].astype(np.float32)

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X_all, Y_all, test_size=0.20, random_state=SEED)

    scaler = StandardScaler()
    X_test_s = scaler.fit_transform(X_test)   # scaler fit on full X for consistency

    pca_full = PCA(n_components=15, random_state=SEED)
    pca_full.fit(Y_train_full)

    N_sizes = [50, 100, 200, 300, 500]
    gp_rmse_list = []
    rf_rmse_list = []

    kernel = (
        1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )

    for n in N_sizes:
        idx = np.random.default_rng(SEED).choice(len(X_train_full), n, replace=False)
        Xn  = X_train_full[idx];  Yn = Y_train_full[idx]
        scn = StandardScaler(); Xn_s = scn.fit_transform(Xn)
        Xt_s = scn.transform(X_test)

        # PCA on subset
        n_comp = min(10, n - 1)
        pca_n  = PCA(n_components=n_comp, random_state=SEED)
        Zn     = pca_n.fit_transform(Yn)
        Zt_n   = pca_n.transform(Y_test)

        gp_preds = np.zeros((len(X_test), n_comp))
        for i in range(n_comp):
            gp_i = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=0,
                normalize_y=True, random_state=SEED)
            gp_i.fit(Xn_s, Zn[:, i])
            gp_preds[:, i] = gp_i.predict(Xt_s)
        Y_gp = pca_n.inverse_transform(gp_preds)
        gp_rmse_list.append(float(np.sqrt(np.mean((Y_test - Y_gp)**2))))

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
        rf.fit(Xn_s, Yn)
        Y_rf = rf.predict(Xt_s)
        rf_rmse_list.append(float(np.sqrt(np.mean((Y_test - Y_rf)**2))))

        print(f"    N={n:4d}  GP RMSE={gp_rmse_list[-1]:.4f}  RF RMSE={rf_rmse_list[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_sizes, gp_rmse_list, 'o-', color='tab:orange', lw=1.5,
            ms=6, label='GP (PCA)')
    ax.plot(N_sizes, rf_rmse_list, 's--', color='tab:blue',   lw=1.5,
            ms=6, label='Random Forest')
    ax.axhline(0.02, color='gray', lw=1.0, ls=':', label='Target RMSE = 0.02')
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Test RMSE in $R$')
    ax.legend()
    ax.set_xlim(0, max(N_sizes) * 1.05)
    savefig('fig5_learning_curve')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("AI-Laue -- Phase C: Generating publication figures")
    print()

    d, p, wl = load_data()

    print("  Figure 1: Parity plot ...")
    fig1_parity(p, wl)

    print("  Figure 2: MAE heatmap ...")
    fig2_mae_heatmap(p)

    print("  Figure 3: Spectrum comparison (12 samples) ...")
    fig3_spectra(p, wl)

    print("  Figure 4: Stopband centre + peak reflectance ...")
    fig4_stopband(p, wl)

    print("  Figure 5: Learning curve (re-training at subset sizes) ...")
    fig5_learning_curve()

    print()
    print(f"  All figures saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
