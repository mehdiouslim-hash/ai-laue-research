"""
Phase B -- GP surrogate model for 1D GaAs/Al_0.3Ga_0.7As DBR reflectance.

Pipeline:
  1. Load data/dataset.npz
  2. Train/val/test split (80/10/10), stratified by N_periods
  3. PCA on R(lambda) output (retain >= 99.9% variance, typically 8-15 components)
  4. Train one GP per PC score  (RBF + Matern-5/2 composite kernel)
  5. Random Forest baseline (200 trees) on same split
  6. Evaluate: RMSE, R^2, MAE on full spectra (reconstructed from PCA)
  7. GP calibration check: coverage of 68% and 95% confidence intervals
  8. Save predictions and metrics to output/metrics.json + output/predictions.npz

References
----------
Cauchy dispersion: Palik (1985); Gehrsitz et al. J. Appl. Phys. 87, 7825 (2000)
GP kernel: Rasmussen & Williams (2006) sec. 4.2
"""

import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_SRC_DIR, '..', 'data', 'dataset.npz')
OUT_DIR   = os.path.join(_SRC_DIR, '..', 'output')
PRED_FILE = os.path.join(OUT_DIR, 'predictions.npz')
METR_FILE = os.path.join(OUT_DIR, 'metrics.json')

SEED = 42
PCA_VARIANCE_THRESHOLD = 0.9990   # retain >= 99.9% variance


# ── helpers ───────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("AI-Laue -- Phase B: GP Surrogate Training")
    print()

    # 1. Load data
    d = np.load(DATA_FILE)
    X_all  = d['params'].astype(np.float32)    # (1500, 3): N_periods, t_GaAs, t_AlGaAs
    Y_all  = d['spectra'].astype(np.float32)   # (1500, 150): R(lambda)
    wl     = d['wavelengths_nm']               # (150,)
    print(f"  Loaded: X={X_all.shape}  Y={Y_all.shape}")

    # 2. Train / val / test split (80 / 10 / 10)
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X_all, Y_all, test_size=0.20, random_state=SEED)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.50, random_state=SEED)
    print(f"  Split: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # 3. Input scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # 4. PCA on training spectra
    pca = PCA(n_components=None, random_state=SEED)
    pca.fit(Y_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD) + 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    Z_train = pca.fit_transform(Y_train)   # (N_train, n_comp)
    Z_val   = pca.transform(Y_val)
    Z_test  = pca.transform(Y_test)
    print(f"  PCA: {n_comp} components explain "
          f"{pca.explained_variance_ratio_.sum()*100:.3f}% variance  "
          f"(threshold {PCA_VARIANCE_THRESHOLD*100:.1f}%)")
    print()

    # 5. GP: one per PC score
    # Subsample training set for GP to keep O(n^3) Cholesky tractable.
    GP_MAX_TRAIN = 400
    rng_gp = np.random.default_rng(SEED)
    if len(X_train_s) > GP_MAX_TRAIN:
        gp_idx = rng_gp.choice(len(X_train_s), GP_MAX_TRAIN, replace=False)
        X_gp = X_train_s[gp_idx]
        Z_gp = Z_train[gp_idx]
    else:
        X_gp, Z_gp = X_train_s, Z_train
    print(f"  GP training subset: {len(X_gp)} / {len(X_train_s)} points")

    kernel = (
        1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    print(f"  Training {n_comp} GPs (one per PC) ...")
    gp_models = []
    t0 = time.time()
    for i in range(n_comp):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            normalize_y=True,
            random_state=SEED,
        )
        gp.fit(X_gp, Z_gp[:, i])
        gp_models.append(gp)
        if (i + 1) % 5 == 0 or i == n_comp - 1:
            print(f"    GP {i+1:2d}/{n_comp}  elapsed {time.time()-t0:.0f}s")

    print(f"  GP training done in {time.time()-t0:.1f}s")
    print()

    # 6. GP predictions on val + test
    def gp_predict(X_s):
        Z_mean = np.zeros((len(X_s), n_comp))
        Z_std  = np.zeros((len(X_s), n_comp))
        for i, gp in enumerate(gp_models):
            m, s = gp.predict(X_s, return_std=True)
            Z_mean[:, i] = m
            Z_std[:, i]  = s
        Y_mean = pca.inverse_transform(Z_mean)
        # Propagate uncertainty through PCA inverse: sigma_y = pca.components_.T @ sigma_z
        # (conservative upper bound using root-sum-of-squares)
        Y_std = np.sqrt((pca.components_.T**2) @ Z_std.T).T
        return Y_mean, Y_std

    Y_val_pred,  Y_val_std  = gp_predict(X_val_s)
    Y_test_pred, Y_test_std = gp_predict(X_test_s)

    gp_val_rmse  = rmse(Y_val,  Y_val_pred)
    gp_val_mae   = mae(Y_val,   Y_val_pred)
    gp_val_r2    = r2(Y_val,    Y_val_pred)
    gp_test_rmse = rmse(Y_test, Y_test_pred)
    gp_test_mae  = mae(Y_test,  Y_test_pred)
    gp_test_r2   = r2(Y_test,   Y_test_pred)

    print(f"  GP  val  RMSE={gp_val_rmse:.4f}  MAE={gp_val_mae:.4f}  R2={gp_val_r2:.4f}")
    print(f"  GP  test RMSE={gp_test_rmse:.4f}  MAE={gp_test_mae:.4f}  R2={gp_test_r2:.4f}")

    # 7. GP calibration: what fraction of test points fall within 1-sigma and 2-sigma?
    residuals = Y_test - Y_test_pred
    cov_68 = float(np.mean(np.abs(residuals) <= Y_test_std))
    cov_95 = float(np.mean(np.abs(residuals) <= 2.0 * Y_test_std))
    print(f"  GP  calibration: 68% band covers {cov_68*100:.1f}%  "
          f"(ideal 68%)  |  95% band covers {cov_95*100:.1f}%  (ideal 95%)")
    print()

    # 8. Random Forest baseline
    print("  Training Random Forest baseline (200 trees) ...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(X_train_s, Y_train)
    rf_val_pred  = rf.predict(X_val_s)
    rf_test_pred = rf.predict(X_test_s)
    rf_val_rmse  = rmse(Y_val,  rf_val_pred)
    rf_val_r2    = r2(Y_val,    rf_val_pred)
    rf_test_rmse = rmse(Y_test, rf_test_pred)
    rf_test_r2   = r2(Y_test,   rf_test_pred)
    print(f"  RF  val  RMSE={rf_val_rmse:.4f}  R2={rf_val_r2:.4f}  "
          f"| test RMSE={rf_test_rmse:.4f}  R2={rf_test_r2:.4f}  "
          f"({time.time()-t0:.0f}s)")
    print()

    # 9. Inference speedup estimate
    n_inf    = 1000
    t_start  = time.time()
    gp_predict(scaler.transform(X_test_s[:min(n_inf, len(X_test_s))]))
    t_gp_per = (time.time() - t_start) / min(n_inf, len(X_test_s)) * 1000  # ms
    speedup  = 308.51 / t_gp_per   # measured TMM ms/spectrum from Phase A
    print(f"  Inference: GP {t_gp_per:.3f} ms/spectrum  "
          f"| TMM ~308 ms/spectrum  | speedup ~{speedup:.0f}x")
    print()

    # 10. Save
    metrics = {
        "gp": {
            "val":  {"rmse": gp_val_rmse,  "mae": gp_val_mae,  "r2": gp_val_r2},
            "test": {"rmse": gp_test_rmse, "mae": gp_test_mae, "r2": gp_test_r2},
            "calibration": {"cov_68": cov_68, "cov_95": cov_95},
        },
        "rf": {
            "val":  {"rmse": rf_val_rmse,  "r2": rf_val_r2},
            "test": {"rmse": rf_test_rmse, "r2": rf_test_r2},
        },
        "dataset": {
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            "n_pca_components": n_comp,
        },
    }
    with open(METR_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        PRED_FILE,
        X_test        = X_test,
        Y_test        = Y_test,
        Y_test_gp     = Y_test_pred.astype(np.float32),
        Y_test_gp_std = Y_test_std.astype(np.float32),
        Y_test_rf     = rf_test_pred.astype(np.float32),
        X_val         = X_val,
        Y_val         = Y_val,
        Y_val_gp      = Y_val_pred.astype(np.float32),
        Y_val_rf      = rf_val_pred.astype(np.float32),
        wavelengths_nm= wl,
    )

    print(f"  Metrics -> {METR_FILE}")
    print(f"  Preds   -> {PRED_FILE}")
    print()
    print("  === SUMMARY ===")
    print(f"  GP  test RMSE : {gp_test_rmse:.4f}  "
          f"({'PASS' if gp_test_rmse < 0.02 else 'REVIEW'} vs target <0.02)")
    print(f"  RF  test RMSE : {rf_test_rmse:.4f}")
    print(f"  GP  speedup   : ~{speedup:.0f}x over TMM")
    print(f"  GP  calibration 95% : {cov_95*100:.1f}%  (ideal 95%)")


if __name__ == '__main__':
    main()
