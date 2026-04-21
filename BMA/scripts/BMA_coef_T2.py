import os
import sys
import time
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.stats import norm

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('lat_i', help='lat_i')
args = vars(parser.parse_args())

# ================================================ #
# -------------------- config -------------------- #
 
lat_i  = int(args['lat_i'])
# lead_0 = 0    # int(args['lead_0'])
# lead_1 = 100  # int(args['lead_1'])
 
VAR_CESM  = 'TREFHT'
VAR_ERA5  = '2m_temperature'
unit_CESM = 1
unit_ERA5 = 1
 
WINDOW      = 0     # sliding window half-width (days) around each lead_time
                    #   0 → fit per lead_time only
                    #  15 → ±15 calendar days pooled
MIN_SAMPLES = 20    # minimum valid samples to fit
EPS         = 1e-6  # floor for predicted variance
 
SAVE_PATH = (
    f'/glade/derecho/scratch/ksha/EPRI_data/BMA/{VAR_CESM}/'
    f'bma_coef_lat_ind_{lat_i}.zarr'
)
# bma_coef_lat_ind_{lat_i}_lead_ind_{lead_0}_{lead_1}.zarr
# ================================================ #
# --------------------- data --------------------- #
 
list_input = []
for year in range(1958, 2010):
    fn_CESM = (
        f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/'
        f'SMYLE_{year}-11-01_daily_ensemble.zarr'
    )
    ds_CESM = xr.open_zarr(fn_CESM)[[VAR_CESM]].sel(
        time=slice(f"{year+1}-01-01", f"{year+10}-12-31")
    )
    ds_CESM = ds_CESM.rename({'time': 'lead_time'})
    ds_CESM['lead_time'] = np.arange(3650)   # 10 × 365 (no-leap)
    list_input.append(ds_CESM.isel(lat=lat_i))
 
ds_input = xr.concat(list_input, dim='init_time')
ds_input = ds_input.assign_coords({'init_time': np.arange(1959, 2011)})
 
list_target = []
for year in range(1958, 2026):
    fn_ERA5 = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
    ds_ERA5 = xr.open_zarr(fn_ERA5)[[VAR_ERA5]].rename({VAR_ERA5: VAR_CESM})
    list_target.append(ds_ERA5.isel(lat=lat_i))
 
ds_target = xr.concat(list_target, dim='time')
 
ds_input[VAR_CESM]  = ds_input[VAR_CESM]  * unit_CESM
ds_target[VAR_CESM] = ds_target[VAR_CESM] * unit_ERA5
 
 
# ================================================ #
# ----------- lazy handles + indexing ------------ #
 
# Ensemble dimension (whatever the non-init / non-lead / non-lon dim is called).
ens_dim = [d for d in ds_input[VAR_CESM].dims
           if d not in ('init_time', 'lead_time', 'lon')][0]
 
K      = ds_input.sizes[ens_dim]
n_init = ds_input.sizes['init_time']
n_lead = ds_input.sizes['lead_time']
n_lon  = ds_input.sizes['lon']
init_years = ds_input['init_time'].values.astype(int)
 
# Transpose so ensemble is the *last* non-lon axis — this way
# fcst_pool[..., j].reshape(-1, K) gives (N_pool, K) correctly.
fcst_lazy = ds_input[VAR_CESM].transpose('init_time', 'lead_time', ens_dim, 'lon')
 
# Target: (n_time, n_lon) for one latitude — fine in memory.
target_time_index = pd.DatetimeIndex(ds_target['time'].values)
target_vals       = ds_target[VAR_CESM].values          # (n_time, n_lon)

# ================================================ #
# ---- CESM no-leap → ERA5 Gregorian mapping ----- #
 
# CESM uses a 365-day (no-leap) calendar, so lead_time index i corresponds to
# day-of-year i in a non-leap year.  Build a lookup (0..364) → (month, day).
_month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_doy_to_md = np.array(
    [(m, d) for m, ml in enumerate(_month_lengths, start=1)
            for d in range(1, ml + 1)]
)                                                       # (365, 2)
assert _doy_to_md.shape == (365, 2)
 
# Per-lead (year_offset, month, day) under the no-leap calendar.
_lead_year_offset = np.arange(n_lead) // 365
_lead_month       = _doy_to_md[np.arange(n_lead) % 365, 0]
_lead_day         = _doy_to_md[np.arange(n_lead) % 365, 1]
 
# Used by the WINDOW>0 pool-selection logic.
day_of_year = np.arange(n_lead) % 365

def get_obs_for_pool(pool_indices):
    """
    Return (n_init, len(pool), n_lon) ERA5 observations aligned to CESM lead
    indices, with NaN where the Gregorian target date is missing.  Calendar-
    aware: CESM no-leap leads are mapped to Gregorian (year, month, day) so
    Feb-29 misalignment does not occur.
    """
    year_offsets = _lead_year_offset[pool_indices]
    months       = _lead_month[pool_indices]
    days         = _lead_day[pool_indices]
 
    target_years = init_years[:, None] + year_offsets[None, :]
    months_b     = np.broadcast_to(months, target_years.shape)
    days_b       = np.broadcast_to(days,   target_years.shape)
 
    flat_dates = pd.to_datetime({
        'year':  target_years.ravel(),
        'month': months_b.ravel(),
        'day':   days_b.ravel(),
    })
    flat   = target_time_index.get_indexer(flat_dates)
    idx_2d = flat.reshape(n_init, len(pool_indices))
 
    obs = np.full((n_init, len(pool_indices), n_lon), np.nan)
    for i in range(n_init):
        good = idx_2d[i] >= 0
        if good.any():
            obs[i, good, :] = target_vals[idx_2d[i, good], :]
    return obs
 
 
# ================================================ #
# ------ exchangeable-member BMA components ------ #
 
def bias_correct_pooled(X, y):
    """
    Single (a, b) shared across exchangeable ensemble members, fit by OLS on
    the pooled (f_{n,k}, y_n) pairs.  Model: y_n = a + b * f_{n,k} + noise.
    """
    f_flat = X.ravel()
    y_flat = np.broadcast_to(y[:, None], X.shape).ravel()
    f_m, y_m = f_flat.mean(), y_flat.mean()
    b = np.dot(f_flat - f_m, y_flat - y_m) / (np.dot(f_flat - f_m, f_flat - f_m) + EPS)
    a = y_m - b * f_m
    return a, b
 
 
def bma_em_exchangeable(X, y, n_iter=200, tol=1e-6):
    """
    BMA EM with equal weights (w_k = 1/K) and a single shared variance sigma^2
    across exchangeable ensemble members (Fraley, Raftery & Gneiting 2010).
 
    Returns sigma2 (float).
    """
    N, K = X.shape
 
    sigma2 = max(np.mean((y[:, None] - X) ** 2), EPS)
 
    for _ in range(n_iter):
        # E-step: equal priors cancel out in the responsibility normalization.
        logp = norm.logpdf(y[:, None], loc=X, scale=np.sqrt(sigma2))
        logp -= logp.max(axis=1, keepdims=True)
        z = np.exp(logp)
        z /= z.sum(axis=1, keepdims=True)
 
        # M-step: single pooled variance.
        resid2     = (y[:, None] - X) ** 2
        sigma2_new = max((z * resid2).sum() / N, EPS)
 
        if abs(sigma2_new - sigma2) < tol * max(sigma2, EPS):
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new
 
    return sigma2

# ================================================ #
# ------------------- main loop ------------------ #
list_lt_coef = []
 
for lt in range(n_lead):
    # ---- pool selection ----
    if WINDOW == 0:
        pool = np.array([lt])
    else:
        doy  = day_of_year[lt]
        dist = np.minimum(np.abs(day_of_year - doy),
                          365 - np.abs(day_of_year - doy))
        pool = np.where(dist <= WINDOW)[0]
 
    # ---- load ONLY the pooled slice (lazy → small in-memory) ----
    fcst_pool = fcst_lazy.isel(lead_time=pool).values    # (n_init, n_pool, K, n_lon)
    obs_pool  = get_obs_for_pool(pool)                   # (n_init, n_pool, n_lon)
 
    # ---- per-longitude fit ----
    w_lt  = np.full((n_lon, K), np.nan)
    s2_lt = np.full((n_lon, K), np.nan)
    a_lt  = np.full((n_lon, K), np.nan)
    b_lt  = np.full((n_lon, K), np.nan)
 
    for j in range(n_lon):
        # fcst_pool is (n_init, n_pool, K, n_lon) thanks to the transpose above.
        f_pool = fcst_pool[:, :, :, j].reshape(-1, K)    # (n_init * n_pool, K)
        y_pool = obs_pool[:, :, j].ravel()
 
        valid  = np.isfinite(y_pool) & np.all(np.isfinite(f_pool), axis=1)
        f_pool, y_pool = f_pool[valid], y_pool[valid]
 
        if len(y_pool) < MIN_SAMPLES:
            continue
 
        # Pooled bias correction: one (a, b) shared across all K members.
        a, b = bias_correct_pooled(f_pool, y_pool)
        X_bc = a + b * f_pool
 
        # Equal-weight, shared-variance BMA.
        sigma2 = bma_em_exchangeable(X_bc, y_pool)
 
        # Replicate shared values across the member axis so the output schema
        # (lead_time, lon, member) stays unchanged for downstream consumers.
        a_lt[j, :]  = a
        b_lt[j, :]  = b
        w_lt[j, :]  = 1.0 / K
        s2_lt[j, :] = sigma2
 
    # ---- append this lead_time's results ----
    ds_lt = xr.Dataset(
        {
            'weight':    (['lead_time', 'lon', 'member'], w_lt[None, ...]),
            'sigma2':    (['lead_time', 'lon', 'member'], s2_lt[None, ...]),
            'intercept': (['lead_time', 'lon', 'member'], a_lt[None, ...]),
            'slope':     (['lead_time', 'lon', 'member'], b_lt[None, ...]),
        },
        coords={
            'lead_time': [lt],
            'lon':       ds_input['lon'].values,
            'lat':       ds_input['lat'].values,
            'member':    np.arange(K),
        },
    )
    list_lt_coef.append(ds_lt)
 
    del fcst_pool, obs_pool

ds_out = xr.concat(list_lt_coef, dim='lead_time')
ds_out = ds_out.chunk({'lead_time': -1, 'lon': -1, 'member': -1})
ds_out.to_zarr(SAVE_PATH, mode='w', compute=True)
print(SAVE_PATH)
