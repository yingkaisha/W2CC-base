import os
import sys
import time
import dask
import zarr
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import interp1d

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('lat_i', help='lat_i')
args = vars(parser.parse_args())

# ================================================ #
# -------------------- config -------------------- #

lat_i = int(args['lat_i'])
VAR = VAR_CESM = 'PRECT'
VAR_ERA5 = 'total_precipitation'
unit_CESM = 60*60*24 * 1000 # m/s to mm/day
unit_ERA5 = 1000  # m/day to mm/day
verif_years = np.arange(2010, 2020)

WINDOW = 0
MIN_SAMPLES = 20
EPS = 1e-6
N_QUANTILES = 200   # number of quantile levels for the mapping table
# Day-of-year index for each lead time
N_DOY = 365
DOY_WINDOW = 15  # ±15 days around each doy for pooling

QM_DIR = f'/glade/derecho/scratch/ksha/EPRI_data/QM/{VAR_CESM}'
SAVE_QM_CESM   = f'{QM_DIR}/qm_cesm_lat_ind_{lat_i}.zarr'
SAVE_QM_ERA5   = f'{QM_DIR}/qm_era5_lat_ind_{lat_i}.zarr'

SAVE_DIR = f'/glade/derecho/scratch/ksha/EPRI_data/QM_PRED/{VAR_CESM}'
SAVE_MAPPED_INPUT  = f'{SAVE_DIR}/pred_input_lat_ind_{lat_i}.zarr'
SAVE_MAPPED_TARGET = f'{SAVE_DIR}/pred_target_lat_ind_{lat_i}.zarr'

# ================================================ #
# --------------------- data --------------------- #
list_input = []
for year in verif_years:
    fn_CESM = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year}-11-01_daily_ensemble.zarr'
    ds_CESM = xr.open_zarr(fn_CESM)[[VAR_CESM,]].sel(time=slice(f"{year+1}-01-01", f"{year+10}-12-31"))
    ds_CESM = ds_CESM.rename({'time': 'lead_time'})
    ds_CESM['lead_time'] = np.arange(3650) # 10 non-leap year, 365 day on each 
    list_input.append(ds_CESM.isel(lat=lat_i))

ds_input = xr.concat(list_input, dim='init_time')
ds_input = ds_input.assign_coords({'init_time': verif_years+1})

list_target = []
for year in range(verif_years[0], 2026):
    fn_ERA5 = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
    ds_ERA5 = xr.open_zarr(fn_ERA5)[[VAR_ERA5,]].rename({VAR_ERA5: VAR_CESM})    
    list_target.append(ds_ERA5.isel(lat=lat_i))

ds_target = xr.concat(list_target, dim='time')

ds_input[VAR_CESM] = ds_input[VAR_CESM] * unit_CESM
ds_target[VAR_CESM] = ds_target[VAR_CESM] * unit_ERA5

ds_qm_cesm = xr.open_zarr(SAVE_QM_CESM)
ds_qm_era5 = xr.open_zarr(SAVE_QM_ERA5)

# ════════════════════════════════════════════════════════════════════════
# Noleap calendar mapping
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

init_years = ds_input.init_time.values
n_init = len(init_years)
n_lead = ds_input.sizes['lead_time']
n_lon  = ds_input.sizes['lon']
n_member = ds_input.sizes['member']

target_times_pd = pd.DatetimeIndex(ds_target.time.values)
date_idx = build_date_indices(init_years, n_lead, target_times_pd)

def extrapolate_upper_tail(q_values):
    """
    Estimate the quantile at level 1.0 by linearly extrapolating
    from the two highest quantile levels (0.9925 and 0.9975).
    Appends the extrapolated value and returns extended arrays.
    """
    v1 = q_values[-2]  # quantile at 0.9925
    v2 = q_values[-1]  # quantile at 0.9975
    l1 = q_levels[-2]
    l2 = q_levels[-1]

    # Linear extrapolation to level 1.0
    slope = (v2 - v1) / (l2 - l1)
    v_max = v2 + slope * (1.0 - l2)
    v_max = max(v_max, v2)  # ensure monotonicity

    q_ext = np.append(q_values, np.float32(v_max))
    l_ext = np.append(q_levels, 1.0)
    return q_ext, l_ext

def precip_to_gaussian(values, q_values, wet_frac):
    """
    Map precipitation values to Gaussian space.

    Strategy for zero-inflated precipitation:
      1. Compute empirical CDF rank for each value
      2. For zeros: assign a random uniform rank in [0, 1-wet_frac]
         (jittered to avoid a point mass in Gaussian space)
      3. For positives: interpolate within the empirical quantile table
         (with upper tail extrapolated to quantile level 1.0)
      4. Map uniform ranks to Gaussian via norm.ppf
    """
    n = len(values)
    gaussian = np.full(n, np.nan, dtype=np.float32)

    if np.all(np.isnan(q_values)):
        return gaussian

    valid = ~np.isnan(values)
    vals = np.maximum(values[valid], 0.0)

    # Extend quantile table with extrapolated upper tail
    q_ext, l_ext = extrapolate_upper_tail(q_values)

    # Build interpolator: precipitation → quantile level (uniform CDF)
    q_unique, idx_unique = np.unique(q_ext, return_index=True)
    ql_unique = l_ext[idx_unique]

    if len(q_unique) < 2:
        gaussian[valid] = 0.0
        return gaussian

    interp_func = interp1d(
        q_unique, ql_unique,
        kind='linear',
        bounds_error=False,
        fill_value=(ql_unique[0], ql_unique[-1]),
    )

    # Map to uniform CDF ranks
    u = interp_func(vals)

    # Handle zeros: jitter within [eps, dry_frac] to spread the point mass
    dry_frac = 1.0 - wet_frac
    if dry_frac > 0:
        is_zero = vals <= 0.0
        rng = np.random.default_rng(42)
        u[is_zero] = rng.uniform(EPS, max(dry_frac, 2 * EPS), size=is_zero.sum())

    # Clip to avoid infinite Gaussian values
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    # Uniform → Gaussian
    gaussian[valid] = norm.ppf(u).astype(np.float32)

    return gaussian


def gaussian_to_precip(gaussian_values, q_values, wet_frac):
    """
    Inverse transform: Gaussian space → precipitation.
    Uses extrapolated upper tail for values beyond the training range.
    """
    shape = gaussian_values.shape
    g_flat = gaussian_values.ravel()
    precip = np.full_like(g_flat, np.nan, dtype=np.float32)

    if np.all(np.isnan(q_values)):
        return precip.reshape(shape)

    valid = ~np.isnan(g_flat)

    # Gaussian → uniform
    u = norm.cdf(g_flat[valid])

    # Censor: if u <= dry_frac, output zero
    dry_frac = 1.0 - wet_frac
    result = np.zeros_like(u)

    wet_mask = u > dry_frac
    if wet_mask.any():
        # Extend quantile table with extrapolated upper tail
        q_ext, l_ext = extrapolate_upper_tail(q_values)

        q_unique, idx_unique = np.unique(q_ext, return_index=True)
        ql_unique = l_ext[idx_unique]

        if len(q_unique) >= 2:
            inv_func = interp1d(
                ql_unique, q_unique,
                kind='linear',
                bounds_error=False,
                fill_value=(q_unique[0], q_unique[-1]),
            )
            result[wet_mask] = np.maximum(inv_func(u[wet_mask]), 0.0)

    precip[valid] = result.astype(np.float32)
    return precip.reshape(shape)

# ════════════════════════════════════════════════════════════════════════
# Load quantile tables (fitted on training period)
# ════════════════════════════════════════════════════════════════════════

cesm_qtable  = ds_qm_cesm['quantile_values'].values   # (365, n_lon, N_QUANTILES)
cesm_wetfrac = ds_qm_cesm['wet_fraction'].values       # (365, n_lon)
era5_qtable  = ds_qm_era5['quantile_values'].values
era5_wetfrac = ds_qm_era5['wet_fraction'].values
q_levels     = ds_qm_cesm['quantile'].values

doy = np.arange(n_lead) % N_DOY

# ════════════════════════════════════════════════════════════════════════
# Load raw data into memory
# ════════════════════════════════════════════════════════════════════════

print("Loading data into memory …")
input_data  = ds_input[VAR_CESM].values.astype(np.float32)   # (init, member, lead, lon)
target_data = ds_target[VAR_CESM].values.astype(np.float32)  # (time, lon)

input_data  = np.maximum(input_data, 0.0)
target_data = np.maximum(target_data, 0.0)

# ════════════════════════════════════════════════════════════════════════
# Transform CESM verification data → Gaussian using training quantiles
# ════════════════════════════════════════════════════════════════════════

print("Transforming CESM verification data to Gaussian space …")
input_gaussian = np.full_like(input_data, np.nan)

for lt in tqdm(range(n_lead), desc='CESM → Gaussian'):
    d = doy[lt]
    for lo in range(n_lon):
        qt = cesm_qtable[d, lo, :]
        wf = cesm_wetfrac[d, lo]
        if np.isnan(wf):
            continue
        vals = input_data[:, :, lt, lo].ravel()
        mapped = precip_to_gaussian(vals, qt, wf)
        input_gaussian[:, :, lt, lo] = mapped.reshape(n_init, n_member)

# ════════════════════════════════════════════════════════════════════════
# Transform ERA5 verification data → Gaussian (matched to init/lead)
# ════════════════════════════════════════════════════════════════════════

print("Transforming ERA5 verification data to Gaussian space …")
target_gaussian_matched = np.full(
    (n_init, n_lead, n_lon), np.nan, dtype=np.float32
)

for lt in tqdm(range(n_lead), desc='ERA5 → Gaussian (matched)'):
    d = doy[lt]
    time_indices = date_idx[:, lt]
    valid_mask = time_indices >= 0
    if not valid_mask.any():
        continue
    valid_inits = np.where(valid_mask)[0]
    valid_times = time_indices[valid_mask]

    for lo in range(n_lon):
        qt = era5_qtable[d, lo, :]
        wf = era5_wetfrac[d, lo]
        if np.isnan(wf):
            continue
        vals = target_data[valid_times, lo]
        mapped = precip_to_gaussian(vals, qt, wf)
        target_gaussian_matched[valid_inits, lt, lo] = mapped

# ════════════════════════════════════════════════════════════════════════
# Save transformed verification datasets
# ════════════════════════════════════════════════════════════════════════

print("Saving Gaussian-transformed CESM verification …")
ds_input_gauss = ds_input.copy(deep=False)
ds_input_gauss[VAR_CESM] = xr.DataArray(
    input_gaussian,
    dims=ds_input[VAR_CESM].dims,
    coords=ds_input[VAR_CESM].coords,
)
ds_input_gauss.attrs['transform'] = 'NQT applied using training-period quantile tables'
ds_input_gauss.to_zarr(SAVE_MAPPED_INPUT, mode='w')

print("Saving Gaussian-transformed ERA5 verification …")
ds_target_gauss = xr.Dataset(
    {
        VAR_CESM: (['init_time', 'lead_time', 'lon'], target_gaussian_matched),
    },
    coords={
        'init_time':  ds_input.init_time.values,
        'lead_time':  ds_input.lead_time.values,
        'lon':        ds_input.lon.values,
        'lat':        ds_input.lat.values,
    },
)
ds_target_gauss.attrs['transform'] = 'NQT applied using training-period quantile tables (matched to init/lead)'
ds_target_gauss.to_zarr(SAVE_MAPPED_TARGET, mode='w')

print("\nDone. Verification datasets ready for Gaussian EMOS evaluation.")

