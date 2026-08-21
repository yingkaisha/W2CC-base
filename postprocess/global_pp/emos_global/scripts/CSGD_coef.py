import os
import sys
import time
import zarr
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func, gammainc, gammaincc, beta as beta_func
from scipy.stats import gamma as gamma_dist
from tqdm import tqdm

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('lat_i', help='lat_i')
args = vars(parser.parse_args())

# ================================================ #
# -------------------- config -------------------- #

lat_i = int(args['lat_i'])
VAR_CESM = 'PRECT'
VAR_ERA5 = 'total_precipitation'
unit_CESM = 60 * 60 * 24 * 1000 # m/s to mm/day
unit_ERA5 = 1000  # m/day to mm/day

WINDOW = 0        # sliding window half-width (days) around each lead_time
                  #   0 → fit per lead_time only
                  #  15 → ±15 days pooled (~52 × 31 ≈ 1600 samples)
MIN_SAMPLES = 20  # minimum valid (init_time × window) samples to fit
EPS = 1e-6        # floor for predicted variance
SAVE_PATH = f'/glade/derecho/scratch/ksha/EPRI_data/EMOS/{VAR_CESM}/emos_coef_lat_ind_{lat_i}.zarr'

def csgd_params_from_moments(mu, sigma, delta):
    """
    Convert (mu, sigma, delta) to Gamma (shape, scale).
    mu must be > delta for valid parameters.
    """
    mu_shifted = np.maximum(mu - delta, EPS)
    k     = (mu_shifted / np.maximum(sigma, EPS)) ** 2      # shape
    theta = np.maximum(sigma, EPS) ** 2 / mu_shifted         # scale
    return k, theta
 
 
def crps_csgd(mu, sigma, delta, obs):
    """
    Closed-form CRPS for the Censored Shifted Gamma Distribution.
 
    Parameters
    ----------
    mu, sigma, delta : array-like, CSGD location, scale, shift
    obs              : array-like, observed precipitation (>= 0)
 
    Returns
    -------
    crps : array, same shape as obs
 
    Based on Scheuerer & Hamill (2015) Eq. 3-5, using the identity
    for CRPS of a censored distribution at zero:
 
        CRPS = (y - delta) * (2*F_y - 1)
             - (mu - delta) * (2*F_0*F_y_k   - F_0^2 - 2*F_y_kp1 + 1)
             + (mu - delta) * B(0.5, k) / (k * pi)    (Gamma normalization term)
 
    where F_y = CDF at y, F_0 = CDF at 0, and F_y_k, F_y_kp1 are regularised
    incomplete gamma values at shape k and k+1.
    """
    k, theta = csgd_params_from_moments(mu, sigma, delta)
 
    # Standardise: z = (value - delta) / theta
    y = np.asarray(obs, dtype=np.float64)
    z_obs = np.maximum(y - delta, 0.0) / np.maximum(theta, EPS)
    z_zero = np.maximum(-delta, 0.0) / np.maximum(theta, EPS)
 
    # Regularised lower incomplete gamma: P(k, x) = gammainc(k, x)
    F_obs  = gammainc(k, z_obs)           # CDF at obs
    F_zero = gammainc(k, z_zero)          # CDF at 0  (= P(Y <= 0))
 
    # Terms for the CRPS formula
    # Term 1: observation part
    term1 = (y - delta) * (2.0 * F_obs - 1.0)
 
    # Clamp negative part (censoring at zero)
    term1 = term1 + delta * (2.0 * F_zero - 1.0)
    # Simpler: for censored, replace y with max(y, 0) in the formula
    # and add the censoring correction
 
    # Term 2: mean of the distribution * calibration factor
    # E[X] for Gamma(k, theta) = k * theta = (mu - delta)
    mu_shifted = k * theta
 
    # For two iid draws X1, X2 ~ Gamma(k, theta):
    # E|X1 - X2| = 2 * theta * k * Beta(k, 0.5) / sqrt(pi)
    # But with censoring we need the full expression.
 
    # Use the direct CRPS formula for censored Gamma from Scheuerer & Hamill:
    #   CRPS = y*(2*Fy - 1) - mu_s*(2*F0*Pk(z_y) - F0^2 + Pk+1(z_y)... )
    # where Pk means regularised incomplete gamma at shape k.
 
    # Simplified stable implementation via the Baran & Nemoda (2016) form:
    F_obs_kp1  = gammainc(k + 1.0, z_obs)
    F_zero_kp1 = gammainc(k + 1.0, z_zero)
 
    # Beta function term: B(0.5, k) = Gamma(0.5)*Gamma(k) / Gamma(k+0.5)
    log_B = (np.lgamma(0.5) + np.lgamma(k) - np.lgamma(k + 0.5))
    B_half_k = np.exp(log_B)
 
    crps = (
        np.maximum(y, 0.0) - np.maximum(y, 0.0) * 2.0 * F_obs
        + mu_shifted * (2.0 * F_obs_kp1 - 1.0)
        - mu_shifted * F_zero * (2.0 * F_zero_kp1 / np.maximum(F_zero, EPS) - F_zero)
        # ↑ careful: when F_zero ~ 0, this term vanishes anyway
    )
    # Correct version using the full closed form:
    crps = (
        np.maximum(y, 0.0) * (2.0 * F_obs - 1.0)
        - mu_shifted * (
            2.0 * F_obs_kp1
            - 2.0 * F_zero * F_zero_kp1
            + F_zero ** 2
            - B_half_k / np.sqrt(np.pi)
        )
    )
 
    # Censoring correction: add back the point mass contribution
    # For obs = 0 cases, the formula above already handles it through F_obs = F_zero
    # Final CRPS should be non-negative
    crps = np.maximum(crps, 0.0)
 
    return crps.astype(np.float32)
 
 
def mean_crps_csgd(params, ens_mean, ens_std, obs, min_mu_shift=0.1):
    """
    Objective function: mean CRPS over training samples.
 
    params: [a0, a1, b0, b1, delta]
    """
    a0, a1, b0, b1, delta = params
 
    mu    = a0 + a1 * ens_mean
    sigma = np.abs(b0 + b1 * ens_std)
    sigma = np.maximum(sigma, EPS)
 
    # Ensure mu > delta for valid Gamma params
    mu = np.maximum(mu, delta + min_mu_shift)
 
    try:
        c = crps_csgd(mu, sigma, delta, obs)
        result = np.nanmean(c)
        if np.isnan(result) or np.isinf(result):
            return 1e10
        return result
    except Exception:
        return 1e10
 
 
# ════════════════════════════════════════════════════════════════════════
# Noleap calendar mapping (same as Gaussian version)
# ════════════════════════════════════════════════════════════════════════
 
def build_date_indices(init_years, n_lead, target_times):
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(target_times)}
    indices = np.full((len(init_years), n_lead), -1, dtype=np.int32)
    for i, year in enumerate(init_years):
        year = int(year)
        noleap_dates = []
        for y in range(year, year + 10):
            days = pd.date_range(f'{y}-01-01', f'{y}-12-31', freq='D')
            days = days[~((days.month == 2) & (days.day == 29))]
            noleap_dates.extend(days.tolist())
        for L in range(min(n_lead, len(noleap_dates))):
            d = noleap_dates[L]
            if d in time_to_idx:
                indices[i, L] = time_to_idx[d]
    return indices
 
 
# ════════════════════════════════════════════════════════════════════════
# Initial guess from method of moments
# ════════════════════════════════════════════════════════════════════════
 
def moment_based_init(ens_mean, ens_std, obs):
    """
    Quick method-of-moments initial guess for [a0, a1, b0, b1, delta].
    Uses OLS for the mean equation and a simple ratio for the spread.
    """
    valid = ~np.isnan(obs) & ~np.isnan(ens_mean) & (ens_mean >= 0)
    if valid.sum() < 5:
        return [0.0, 1.0, 0.1, 1.0, -0.1]
 
    em = ens_mean[valid]
    es = ens_std[valid]
    ob = obs[valid]
 
    # OLS for mean: obs ≈ a0 + a1 * ens_mean
    em_bar, ob_bar = em.mean(), ob.mean()
    cov = ((em - em_bar) * (ob - ob_bar)).mean()
    var = ((em - em_bar) ** 2).mean()
    a1 = cov / max(var, EPS)
    a0 = ob_bar - a1 * em_bar
 
    # Spread: match residual std to b0 + b1 * ens_std
    resid_std = np.abs(ob - (a0 + a1 * em)).std()
    es_bar = es.mean()
    b1 = resid_std / max(es_bar, EPS) * 0.5
    b0 = max(resid_std * 0.5, EPS)
 
    # Shift: small negative value (allows censoring near zero)
    delta = -max(0.1 * ob_bar, 0.01)
 
    return [a0, max(a1, EPS), b0, b1, delta]
 
 
# ════════════════════════════════════════════════════════════════════════
# Fit EMOS-CSGD: loop over lat, then (lead_time, lon) per grid point
# ════════════════════════════════════════════════════════════════════════
 
def fit_one_gridpoint(em_v, es_v, ob_v):
    """
    Fit CSGD-EMOS for one (lead_time, lon) cell.
    Returns (params, crps_value) or (None, None) if fitting fails.
    """
    ob_v = np.maximum(ob_v, 0.0)
    ob_mean = max(ob_v.mean(), EPS)
    ob_std  = max(ob_v.std(), EPS)

    # ── Physical bounds ──
    bounds = [
        (-2.0 * ob_std, 2.0 * ob_mean),       # a0
        (EPS, 3.0),                             # a1
        (EPS, 3.0 * ob_std),                    # b0
        (EPS, 5.0),                             # b1
        (-2.0 * ob_std, -EPS),                  # delta
    ]

    # ── Multiple initial guesses ──
    x0_mom = moment_based_init(em_v, es_v, ob_v)

    rng = np.random.default_rng(42)
    x0_list = [x0_mom]
    for _ in range(3):
        perturbed = [
            np.clip(x0_mom[i] * rng.uniform(0.5, 1.5), bounds[i][0], bounds[i][1])
            for i in range(5)
        ]
        x0_list.append(perturbed)
    x0_list.append([0.0, 1.0, 0.5 * ob_std, 1.0, -0.5 * ob_std])
    x0_list.append([0.1 * ob_mean, 0.8, 0.2 * ob_std, 0.5, -0.1 * ob_mean])

    # ── Multi-start L-BFGS-B ──
    best_result = None
    best_cost = 1e10

    for x0 in x0_list:
        x0_clipped = [np.clip(x0[i], bounds[i][0], bounds[i][1]) for i in range(5)]
        try:
            result = minimize(
                mean_crps_csgd, x0_clipped,
                args=(em_v, es_v, ob_v),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
        except Exception:
            continue

    # ── Nelder-Mead polish ──
    if best_result is not None:
        try:
            result_nm = minimize(
                mean_crps_csgd, best_result.x,
                args=(em_v, es_v, ob_v),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8},
            )
            if result_nm.fun < best_cost:
                best_cost = result_nm.fun
                best_result = result_nm
        except Exception:
            pass

    if best_result is not None and best_cost < 1e9:
        return list(best_result.x), best_cost
    return None, None


def fit_emos_csgd(ds_input, ds_target, date_idx, var=VAR_CESM,
                  window=WINDOW, min_samples=MIN_SAMPLES):
    """
    Fit CSGD-EMOS coefficients per (lead_time, lat, lon).

    Returns xr.Dataset with variables a0, a1, b0, b1, delta,
    each of shape (lead_time, lon).
    """
    init_years = ds_input.init_time.values
    n_init = len(init_years)
    n_lead = ds_input.sizes['lead_time']
    n_lon  = ds_input.sizes['lon']

    # Precompute ensemble stats (lazy / dask-backed)
    ens_mean_da = ds_input[var].mean(dim='member')
    ens_std_da  = ds_input[var].std(dim='member', ddof=1)

    # Load target
    target_data = ds_target[var].values.astype(np.float32)
    target_data = np.maximum(target_data, 0.0)

    # Allocate coefficient arrays
    shape = (n_lead, n_lon)
    coeffs = {name: np.full(shape, np.nan, dtype=np.float32)
              for name in ['a0', 'a1', 'b0', 'b1', 'delta']}

    # Load this lat row: (n_init, n_lead, n_lon)
    X_mean = ens_mean_da.values.astype(np.float32)
    X_std  = ens_std_da.values.astype(np.float32)

    # Build matched obs
    Y = np.full_like(X_mean, np.nan)
    for i in range(n_init):
        mask = date_idx[i] >= 0
        Y[i, mask, :] = target_data[date_idx[i, mask], :]

    # Censor ensemble below zero
    X_mean = np.maximum(X_mean, 0.0)
    X_std  = np.maximum(X_std, EPS)

    # Optional sliding window pooling
    if window > 0:
        X_mean_pool, X_std_pool, Y_pool = [], [], []
        for dL in range(-window, window + 1):
            if dL == 0:
                X_mean_pool.append(X_mean)
                X_std_pool.append(X_std)
                Y_pool.append(Y)
            else:
                pad = np.full((n_init, abs(dL), n_lon), np.nan, dtype=np.float32)
                if dL > 0:
                    X_mean_pool.append(np.concatenate([pad, X_mean[:, :-dL, :]], axis=1))
                    X_std_pool.append(np.concatenate([pad, X_std[:, :-dL, :]], axis=1))
                    Y_pool.append(np.concatenate([pad, Y[:, :-dL, :]], axis=1))
                else:
                    X_mean_pool.append(np.concatenate([X_mean[:, -dL:, :], pad], axis=1))
                    X_std_pool.append(np.concatenate([X_std[:, -dL:, :], pad], axis=1))
                    Y_pool.append(np.concatenate([Y[:, -dL:, :], pad], axis=1))

        X_mean = np.concatenate(X_mean_pool, axis=0)
        X_std  = np.concatenate(X_std_pool,  axis=0)
        Y      = np.concatenate(Y_pool,      axis=0)

    # ── Per grid point optimization ──
    for lt in tqdm(range(n_lead), desc='lead_time'):
        for lo in range(n_lon):

            em = X_mean[:, lt, lo]
            es = X_std[:, lt, lo]
            ob = Y[:, lt, lo]

            valid = ~np.isnan(ob) & ~np.isnan(em)
            if valid.sum() < min_samples:
                continue

            em_v = em[valid]
            es_v = es[valid]
            ob_v = np.maximum(ob[valid], 0.0)

            params, cost = fit_one_gridpoint(em_v, es_v, ob_v)

            if params is not None:
                a0, a1, b0, b1, delta = params
                coeffs['a0'][lt, lo] = a0
                coeffs['a1'][lt, lo] = max(a1, EPS)
                coeffs['b0'][lt, lo] = b0
                coeffs['b1'][lt, lo] = b1
                coeffs['delta'][lt, lo] = delta

    # Package as xr.Dataset
    coords = {
        'lead_time': ds_input.lead_time.values,
        'lat':       ds_input.lat.values,
        'lon':       ds_input.lon.values,
    }
    dims = ('lead_time', 'lon')
    ds_emos = xr.Dataset(
        {name: (dims, arr) for name, arr in coeffs.items()},
        coords=coords,
    )
    ds_emos.attrs['description'] = (
        'CSGD-EMOS coefficients for precipitation. '
        'mu = a0 + a1*ens_mean, sigma = |b0 + b1*ens_std|, '
        'Gamma shape k = ((mu-delta)/sigma)^2, scale theta = sigma^2/(mu-delta). '
        'Censored at zero.'
    )
    ds_emos.attrs['window_halfwidth'] = window
    ds_emos.attrs['min_samples'] = min_samples
    return ds_emos


# ================================================ #
# --------------------- data --------------------- #
list_input = []
for year in range(1958, 2010):
    fn_CESM = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year}-11-01_daily_ensemble.zarr'
    ds_CESM = xr.open_zarr(fn_CESM)[[VAR_CESM,]].sel(time=slice(f"{year+1}-01-01", f"{year+10}-12-31"))
    ds_CESM = ds_CESM.rename({'time': 'lead_time'})
    ds_CESM['lead_time'] = np.arange(3650) # 10 non-leap year, 365 day on each 
    list_input.append(ds_CESM.isel(lat=lat_i))

ds_input = xr.concat(list_input, dim='init_time')
ds_input = ds_input.assign_coords({'init_time': np.arange(1958+1, 2010+1)})

list_target = []
for year in range(1958, 2026):
    fn_ERA5 = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
    ds_ERA5 = xr.open_zarr(fn_ERA5)[[VAR_ERA5,]].rename({VAR_ERA5: VAR_CESM})    
    list_target.append(ds_ERA5.isel(lat=lat_i))

ds_target = xr.concat(list_target, dim='time')

ds_input[VAR_CESM] = ds_input[VAR_CESM] * unit_CESM
ds_target[VAR_CESM] = ds_target[VAR_CESM] * unit_ERA5

# Date mapping
init_years = ds_input.init_time.values
target_times_pd = pd.DatetimeIndex(ds_target.time.values)
date_idx = build_date_indices(init_years, 3650, target_times_pd)

# CSGD fitting
ds_emos = fit_emos_csgd(ds_input, ds_target, date_idx, var=VAR_CESM, window=WINDOW)
ds_emos = ds_emos.assign_coords({'lat': ds_input.lat.values})

print(f"\nSaving EMOS coefficients to {SAVE_PATH} …")
ds_emos.to_zarr(SAVE_PATH, mode='w')
print("Done.\n")








