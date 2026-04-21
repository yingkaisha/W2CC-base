import os
import sys
import time
import dask
import zarr
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('lat_i', help='lat_i')
args = vars(parser.parse_args())

# ================================================ #
# -------------------- config -------------------- #
lat_i = int(args['lat_i'])
VAR = VAR_CESM = 'PRECT'
WINDOW = 0        # sliding window half-width (days) around each lead_time
                  #   0 → fit per lead_time only
                  #  15 → ±15 days pooled (~52 × 31 ≈ 1600 samples)
MIN_SAMPLES = 20  # minimum valid (init_time × window) samples to fit
EPS = 1e-6        # floor for predicted variance
SAVE_DIR = f'/glade/derecho/scratch/ksha/EPRI_data/QM/{VAR_CESM}/'
SAVE_PATH = f'/glade/derecho/scratch/ksha/EPRI_data/EMOS/{VAR_CESM}_QM/emos_coef_lat_ind_{lat_i}.zarr'

# ================================================ #
# --------------------- data --------------------- #
fn_input  = f'{SAVE_DIR}/mapped_input_lat_ind_{lat_i}.zarr'
ds_input = xr.open_zarr(fn_input)

fn_target = f'{SAVE_DIR}/mapped_target_lat_ind_{lat_i}.zarr'
ds_target = xr.open_zarr(fn_target)
# ================================================ #
# Noleap calendar → real-date index mapping

def build_date_indices(init_years, n_lead, target_times):
    """
    Map every (init_time, lead_time) pair to an integer index into the
    target (ERA5) time axis.  Returns -1 where no matching date exists.
    
    The CESM2 noleap calendar has exactly 365 days per year (no Feb 29).
    For each model day we find the corresponding real-calendar date,
    then look it up in the ERA5 time coordinate.
    """
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(target_times)}
    indices = np.full((len(init_years), n_lead), -1, dtype=np.int32)
 
    for i, year in enumerate(init_years):
        year = int(year)
        # Build the 3650-day noleap calendar for this initialisation
        noleap_dates = []
        for y in range(year, year + 10):
            days = pd.date_range(f'{y}-01-01', f'{y}-12-31', freq='D')
            days = days[~((days.month == 2) & (days.day == 29))]  # drop leap day
            noleap_dates.extend(days.tolist())
 
        for L in range(min(n_lead, len(noleap_dates))):
            d = noleap_dates[L]
            if d in time_to_idx:
                indices[i, L] = time_to_idx[d]
 
    return indices

print("Building noleap → real-calendar date mapping …")
init_years = ds_input.init_time.values           # [1959, 1960, …, 2010]
n_init     = len(init_years)
n_lead     = len(ds_input.lead_time)              # 3650
n_lon      = len(ds_input.lon)
 
target_times_pd = pd.DatetimeIndex(ds_target.time.values)
date_idx = build_date_indices(init_years, n_lead, target_times_pd)
# date_idx shape: (n_init, n_lead) — value = position in target time axis
 
# Quick sanity check
n_valid_total = (date_idx >= 0).sum()
print(f"  Matched {n_valid_total} of {n_init * n_lead} (init, lead) pairs to ERA5 dates.\n")

# ================================================ #
# Precompute ensemble mean and variance
ens_mean_da = ds_input[VAR].mean(dim='member')            # (init_time, lead_time, lat, lon)
ens_var_da  = ds_input[VAR].var(dim='member', ddof=1)     # (init_time, lead_time, lat, lon)

print("Loading ERA5 target into memory …")
target_data = ds_target[VAR].values.astype(np.float32)    # (time, lat, lon)
print(f"  target_data shape: {target_data.shape}\n")

# ================================================ #
# Fit EMOS

# For each latitude row we load (n_init, n_lead, n_lon) arrays and solve
# two OLS problems (mean, variance) in closed form.
a_coeff = np.full((n_lead, n_lon), np.nan, dtype=np.float32)
b_coeff = np.full_like(a_coeff, np.nan)
c_coeff = np.full_like(a_coeff, np.nan)
d_coeff = np.full_like(a_coeff, np.nan)

def _ols_slope_intercept(X, Y, valid, n_valid):
    """
    Vectorised OLS  Y = a + b·X  along axis 0, with NaN masking.
    X, Y   : (N, *spatial)
    valid  : boolean mask, same shape
    n_valid: (*spatial) count of valid samples per column
    Returns a, b with shape (*spatial).
    """
    X = X.copy(); Y = Y.copy()
    X[~valid] = 0.0; Y[~valid] = 0.0
 
    # Means (only over valid entries)
    nv = np.maximum(n_valid, 1).astype(np.float32)
    X_bar = X.sum(axis=0) / nv
    Y_bar = Y.sum(axis=0) / nv
 
    # Centred (invalid entries contribute 0)
    Xd = X - X_bar[None, ...]
    Yd = Y - Y_bar[None, ...]
    Xd[~valid] = 0.0; Yd[~valid] = 0.0
 
    cov_xy = (Xd * Yd).sum(axis=0) / np.maximum(nv - 1, 1)
    var_x  = (Xd ** 2).sum(axis=0)  / np.maximum(nv - 1, 1)
 
    b = np.where(var_x > EPS, cov_xy / var_x, 1.0)
    a = Y_bar - b * X_bar
    return a, b
 
print(f"Fitting EMOS for latitude row {lat_i}") 
# ── Load ensemble stats for this lat row ──
# Shape: (n_init, n_lead, n_lon)
X  = ens_mean_da.values.astype(np.float32)
S2 = ens_var_da.values.astype(np.float32)

# ── Build matched observation array ──
# Y[i, L, :] = ERA5 on the date corresponding to (init_i, lead_L)
Y = np.full_like(X, np.nan)
for i in range(n_init):
    mask = date_idx[i] >= 0                    # (n_lead,) bool
    Y[i, mask, :] = target_data[date_idx[i, mask], :]

# ── Apply sliding window (pool nearby lead_times) ──
if WINDOW > 0:
    # Expand dims: replicate each sample across the window so that
    # lead_time L gets training data from [L-W, L+W].
    # We create views indexed by (init × window, lead, lon).
    X_pool  = []
    S2_pool = []
    Y_pool  = []
    for dL in range(-WINDOW, WINDOW + 1):
        # shifted arrays — pad edges with NaN
        if dL == 0:
            X_pool.append(X);  S2_pool.append(S2);  Y_pool.append(Y)
        else:
            pad = np.full((n_init, abs(dL), n_lon), np.nan, dtype=np.float32)
            if dL > 0:
                X_pool.append(np.concatenate([pad, X[:, :-dL, :]], axis=1))
                S2_pool.append(np.concatenate([pad, S2[:, :-dL, :]], axis=1))
                Y_pool.append(np.concatenate([pad, Y[:, :-dL, :]], axis=1))
            else:
                X_pool.append(np.concatenate([X[:, -dL:, :], pad], axis=1))
                S2_pool.append(np.concatenate([S2[:, -dL:, :], pad], axis=1))
                Y_pool.append(np.concatenate([Y[:, -dL:, :], pad], axis=1))

    # Stack along a new "sample" axis: (n_init * n_window, n_lead, n_lon)
    X  = np.concatenate(X_pool,  axis=0)
    S2 = np.concatenate(S2_pool, axis=0)
    Y  = np.concatenate(Y_pool,  axis=0)

# ── Valid mask & counts ──
valid   = ~np.isnan(Y) & ~np.isnan(X)           # (samples, n_lead, n_lon)
n_valid = valid.sum(axis=0)                      # (n_lead, n_lon)

# ── OLS for mean:  Y = a + b · X ──
a_lt, b_lt = _ols_slope_intercept(X, Y, valid, n_valid)

# ── Predicted mean → residual squared ──
mu_pred  = a_lt[None, ...] + b_lt[None, ...] * X
resid_sq = (Y - mu_pred) ** 2
resid_sq[~valid] = np.nan

# ── OLS for variance:  resid² = c + d · S² ──
valid_r = valid & ~np.isnan(resid_sq) & ~np.isnan(S2)
n_valid_r = valid_r.sum(axis=0)
c_lt, d_lt = _ols_slope_intercept(S2, resid_sq, valid_r, n_valid_r)

# Ensure predicted variance stays positive
c_lt = np.maximum(c_lt, EPS)

# ── Store only where we had enough training samples ──
enough = n_valid >= MIN_SAMPLES
a_coeff[:, :] = np.where(enough, a_lt, np.nan)
b_coeff[:, :] = np.where(enough, b_lt, np.nan)
c_coeff[:, :] = np.where(enough, c_lt, np.nan)
d_coeff[:, :] = np.where(enough, d_lt, np.nan)

# ================================================ #
# Save coefficients

coords = {
    'lead_time': ds_input.lead_time.values,
    'lat':       ds_input.lat.values,
    'lon':       ds_input.lon.values,
}
dims = ('lead_time', 'lon')

ds_emos = xr.Dataset({
    'a': (dims, a_coeff),
    'b': (dims, b_coeff),
    'c': (dims, c_coeff),
    'd': (dims, d_coeff),
}, coords=coords)

ds_emos.attrs['description'] = (
    f'EMOS coefficients for CESM2-SMYLE TREFHT and lat index {lat_i}'
    'mu = a + b*ens_mean, sigma2 = max(c + d*ens_var, eps).'
)
ds_emos.attrs['window_halfwidth'] = WINDOW
ds_emos.attrs['min_samples']      = MIN_SAMPLES
 
print(f"\nSaving EMOS coefficients to {SAVE_PATH} …")
ds_emos.to_zarr(SAVE_PATH, mode='w')
print("Done.\n")



