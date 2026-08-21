
import os
import sys
import time
import random
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.stats import norm

VAR = VAR_CESM = 'PRECT'
VAR_ERA5 = 'total_precipitation'
unit_CESM = 60*60*24 * 1000 # m/s to mm/day
unit_ERA5 = 1000  # m/day to mm/day
EPS = 1e-6        # floor for predicted variance
verif_years = np.arange(2010, 2020)
N_ens = 50

# ================================================ #
# --------------------- data --------------------- #
# CESM Raw
list_input = []
for year in verif_years:
    fn_CESM = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year}-11-01_daily_ensemble.zarr'
    ds_CESM = xr.open_zarr(fn_CESM)[[VAR_CESM,]].sel(time=slice(f"{year+1}-01-01", f"{year+10}-12-31"))
    ds_CESM = ds_CESM.rename({'time': 'lead_time'})
    ds_CESM['lead_time'] = np.arange(3650) # 10 non-leap year, 365 day on each 
    list_input.append(ds_CESM)

ds_input = xr.concat(list_input, dim='init_time')
ds_input = ds_input.assign_coords({'init_time': verif_years+1})
ds_input[VAR_CESM] = ds_input[VAR_CESM] * unit_CESM

# ERA5 target
list_target = []
for year in range(verif_years[0], 2026):
    fn_ERA5 = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
    ds_ERA5 = xr.open_zarr(fn_ERA5)[[VAR_ERA5,]].rename({VAR_ERA5: VAR_CESM})    
    list_target.append(ds_ERA5)

ds_target = xr.concat(list_target, dim='time')
ds_target[VAR_CESM] = ds_target[VAR_CESM] * unit_ERA5

fn_list = sorted(glob(
    '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/QM_EMOS_dressed_N1_*.zarr'
))
list_EMOS = []
for fn in fn_list:
    list_EMOS.append(xr.open_zarr(fn))
ds_EMOS = xr.concat(list_EMOS, dim='member')
ds_EMOS['init_time'] = verif_years + 1

# Ensure consistent dimension order
ds_input = ds_input.transpose('init_time', 'member', 'lead_time', 'lat', 'lon')
ds_EMOS  = ds_EMOS.transpose('init_time', 'member', 'lead_time', 'lat', 'lon')

print(ds_EMOS)
print(ds_input)

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

print("Building date index mapping …")
init_years = ds_input.init_time.values
n_verif = len(init_years)
n_lead  = ds_input.sizes['lead_time']
n_lat   = ds_input.sizes['lat']
n_lon   = ds_input.sizes['lon']

target_times_pd = pd.DatetimeIndex(ds_target.time.values)
date_idx = build_date_indices(init_years, n_lead, target_times_pd)

n_valid_total = (date_idx >= 0).sum()
print(f"  Matched {n_valid_total:,} of {n_verif * n_lead:,} (init, lead) pairs.\n")

# ════════════════════════════════════════════════════════════════════════
# Load arrays into memory
# ════════════════════════════════════════════════════════════════════════

print("Loading ERA5 target into memory …")
target_vals = np.maximum(ds_target[VAR_CESM].values.astype(np.float32), 0.0)

# Keep ds_input and ds_EMOS as lazy (dask-backed from open_zarr)
n_member_raw  = ds_input.sizes['member']
n_member_emos = ds_EMOS.sizes['member']

print(f"  Raw:  {n_verif} inits × {n_member_raw} members × {n_lead} leads × {n_lat} lat × {n_lon} lon")
print(f"  EMOS: {n_verif} inits × {n_member_emos} members × {n_lead} leads × {n_lat} lat × {n_lon} lon")
print(f"  Loading per-init to save memory.\n")


# ════════════════════════════════════════════════════════════════════════
# Vectorized empirical CRPS
# ════════════════════════════════════════════════════════════════════════

def empirical_crps_vectorized(ensemble, obs):
    """
    Vectorized CRPS along axis 0 (member dimension).

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Using the sorted-ensemble identity:
        E|X - X'| = (2/n^2) * sum_i (2i - n - 1) * x_(i)

    Parameters
    ----------
    ensemble : (n_members, *spatial)
    obs      : (*spatial)

    Returns
    -------
    crps : (*spatial), same shape as obs
    """
    ens = np.sort(ensemble, axis=0)
    n = ens.shape[0]

    # E|X - y|
    term1 = np.mean(np.abs(ens - obs[None, ...]), axis=0)

    # E|X - X'| via sorted identity
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    shape = [n] + [1] * (ens.ndim - 1)
    weights = weights.reshape(shape)
    term2 = np.sum(weights * ens, axis=0) / (n * n)

    return term1 - term2

# ════════════════════════════════════════════════════════════════════════
# Compute CRPS
# ════════════════════════════════════════════════════════════════════════
print("Computing CRPS …")

raw_crps  = np.full((n_verif, n_lead, n_lat, n_lon), np.nan, dtype=np.float32)
emos_crps = np.full_like(raw_crps, np.nan)

for ii in tqdm(range(n_verif), desc='CRPS by init'):

    # ── Build obs: (n_lead, n_lat, n_lon) ──
    obs = np.full((n_lead, n_lat, n_lon), np.nan, dtype=np.float32)
    valid_leads = date_idx[ii] >= 0
    lead_idx = np.where(valid_leads)[0]
    time_idx = date_idx[ii, lead_idx]
    obs[lead_idx, :, :] = target_vals[time_idx, :, :]

    valid = ~np.isnan(obs)

    # ── Load only this init's ensembles ──
    raw_ii = np.maximum(
        ds_input[VAR_CESM].isel(init_time=ii).values.astype(np.float32), 0.0
    )  # (member, lead, lat, lon)

    emos_ii = np.maximum(
        ds_EMOS[VAR_CESM].isel(init_time=ii).values.astype(np.float32), 0.0
    )  # (member, lead, lat, lon)

    # ── Vectorized CRPS ──
    r = empirical_crps_vectorized(raw_ii, obs)
    r[~valid] = np.nan
    raw_crps[ii] = r

    e = empirical_crps_vectorized(emos_ii, obs)
    e[~valid] = np.nan
    emos_crps[ii] = e

    # Free memory
    del raw_ii, emos_ii
    
# ════════════════════════════════════════════════════════════════════════
# Summary statistics
# ════════════════════════════════════════════════════════════════════════

n_valid = np.sum(~np.isnan(raw_crps))
r_mean = np.nanmean(raw_crps)
e_mean = np.nanmean(emos_crps)

print(f"\n{'='*60}")
print(f"Real-space CRPS verification ({n_valid:,} valid cells)")
print(f"{'='*60}")
print(f"  Raw CESM   : {r_mean:.4f} mm/day")
print(f"  EMOS       : {e_mean:.4f} mm/day")
print(f"  Improvement: {100 * (r_mean - e_mean) / r_mean:.1f}%\n")

# Per forecast year
print("Per forecast year:")
for yr in range(10):
    lt_lo = yr * 365
    lt_hi = (yr + 1) * 365
    r_yr = np.nanmean(raw_crps[:, lt_lo:lt_hi, :, :])
    e_yr = np.nanmean(emos_crps[:, lt_lo:lt_hi, :, :])
    imp = 100 * (r_yr - e_yr) / r_yr if r_yr > 0 else 0
    print(f"  Year {yr+1:2d} (lead {lt_lo:4d}–{lt_hi:4d}): "
          f"raw={r_yr:.4f}  emos={e_yr:.4f}  gain={imp:+.1f}%")

# Per init year
print("\nPer initialisation:")
for i, yr in enumerate(init_years):
    r_i = np.nanmean(raw_crps[i])
    e_i = np.nanmean(emos_crps[i])
    imp = 100 * (r_i - e_i) / r_i if r_i > 0 else 0
    print(f"  Init {yr}: raw={r_i:.4f}  emos={e_i:.4f}  gain={imp:+.1f}%")


# ════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════

ds_verif = xr.Dataset(
    {
        "CRPS_CESM": (["init_time", "lead_time", "lat", "lon"], raw_crps),
        "CRPS_EMOS": (["init_time", "lead_time", "lat", "lon"], emos_crps),
    },
    coords={
        "init_time": verif_years,
        "lead_time": np.arange(3650),
        "lat": ds_input.lat.values,
        "lon": ds_input.lon.values,
    },
)
ds_verif.attrs['description'] = (
    f'Empirical CRPS in mm/day for {VAR_CESM}. '
    f'Raw: {n_member_raw} CESM members, EMOS: {n_member_emos} dressed members. '
    f'Verification years: {verif_years[0]}–{verif_years[-1]}.'
)

fn_save = (
    f'/glade/derecho/scratch/ksha/EPRI_data/PP_verif/'
    f'TP_QM_EMOS_{verif_years[0]}_{verif_years[-1]}.zarr'
)
print(f"\nSaving to {fn_save} …")
ds_verif.to_zarr(fn_save, mode='w')
print("Done.")

