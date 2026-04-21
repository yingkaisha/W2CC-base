import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from tqdm import tqdm
from scipy.stats import norm

VAR = VAR_CESM = 'TREFHT'
VAR_ERA5 = '2m_temperature'
EPS = 1e-6        # floor for predicted variance
verif_years = np.arange(2010, 2020)

lat_ind = np.arange(192)

list_coef = []
for lat_i in lat_ind:
    fn = f'/glade/derecho/scratch/ksha/EPRI_data/BMA/{VAR_CESM}/bma_coef_lat_ind_{lat_i}.zarr'
    list_coef.append(xr.open_zarr(fn))

ds_bma = xr.concat(list_coef, dim='lat')
ds_bma = ds_bma.transpose('member', 'lead_time', 'lat', 'lon')

list_input = []
for year in verif_years:
    fn_CESM = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year}-11-01_daily_ensemble.zarr'
    ds_CESM = xr.open_zarr(fn_CESM)[[VAR_CESM,]].sel(time=slice(f"{year+1}-01-01", f"{year+10}-12-31"))
    ds_CESM = ds_CESM.rename({'time': 'lead_time'})
    ds_CESM['lead_time'] = np.arange(3650) # 10 non-leap year, 365 day on each 
    list_input.append(ds_CESM)

ds_input = xr.concat(list_input, dim='init_time')
ds_input = ds_input.assign_coords({'init_time': verif_years+1})

list_target = []
for year in range(2010, 2026):
    fn_ERA5 = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
    ds_ERA5 = xr.open_zarr(fn_ERA5)[[VAR_ERA5,]].rename({VAR_ERA5: VAR_CESM})    
    list_target.append(ds_ERA5)

ds_target = xr.concat(list_target, dim='time')
ds_bma['member'] = ds_input['member'].values


# fcst = ds_input[VAR_CESM]                                   # (init, lead, member, lat, lon)
# mu_k = ds_bma['intercept'] + ds_bma['slope'] * fcst         # broadcasts on lead/lat/lon/member
# bma_mu = (ds_bma['weight'] * mu_k).sum('member')            # (init, lead, lat, lon)
 
# within  = (ds_bma['weight'] * ds_bma['sigma2']).sum('member')
# between = (ds_bma['weight'] * (mu_k - bma_mu) ** 2).sum('member')
# bma_var = within + between
# bma_sigma = np.sqrt(np.maximum(bma_var, EPS))
 
# ds_calib = xr.Dataset(
#     {
#         'bma_mu':    bma_mu.astype(np.float32),
#         'bma_sigma': bma_sigma.astype(np.float32),
#     }
# )

# ds_calib = ds_calib.load()
# fn = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/T2_BMA_calib.zarr'
# ds_calib.to_zarr(fn, mode='w')

fn = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/T2_BMA_calib.zarr'
ds_calib = xr.open_zarr(fn)

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
n_lat      = len(ds_input.lat)
n_lon      = len(ds_input.lon)
 
target_times_pd = pd.DatetimeIndex(ds_target.time.values)
date_idx = build_date_indices(init_years, n_lead, target_times_pd)
# date_idx shape: (n_init, n_lead) — value = position in target time axis
 
# Quick sanity check
n_valid_total = (date_idx >= 0).sum()
print(f"  Matched {n_valid_total} of {n_init * n_lead} (init, lead) pairs to ERA5 dates.\n")


# ════════════════════════════════════════════════════════════════════════
# Step 6 (optional): Quick verification — CRPS comparison
# ════════════════════════════════════════════════════════════════════════
 
def crps_gaussian(mu, sigma, obs):
    """Closed-form CRPS for N(mu, sigma²) vs scalar obs (vectorised)."""
    z = (obs - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    
def crps_verif(ds_calib, ds_input, ds_target, date_idx, var='TREFHT', init_idx=None):
    """
    Vectorized CRPS verification for one or more initializations.

    Parameters
    ----------
    init_idx : int, list/array of ints, or None
        Which initialization(s) to evaluate.
        None  → all initializations in ds_input.
        int   → single initialization; returns 3D arrays.
        list  → multiple initializations; returns 4D arrays.

    Returns
    -------
    emos_crps : np.ndarray
    """
    n_lead = ds_input.sizes['lead_time']
    n_lat  = ds_input.sizes['lat']
    n_lon  = ds_input.sizes['lon']

    target_vals = ds_target[var].values.astype(np.float32)  # (time, lat, lon)

    # ── Normalize init_idx ──
    squeeze = False
    if init_idx is None:
        init_idx = np.arange(ds_input.sizes['init_time'])
    elif isinstance(init_idx, (int, np.integer)):
        init_idx = [init_idx]
        squeeze = True
        
    init_idx = np.asarray(init_idx)
    n_sel = len(init_idx)

    # ── Allocate output ──
    emos_crps  = np.full((n_sel, n_lead, n_lat, n_lon), np.nan, dtype=np.float32)
    
    for j, ii in enumerate(init_idx):
        # ── Obs: (lead_time, lat, lon) ──
        obs = np.full((n_lead, n_lat, n_lon), np.nan, dtype=np.float32)
        valid_leads = date_idx[ii] >= 0
        lead_idx    = np.where(valid_leads)[0]
        time_idx    = date_idx[ii, lead_idx]
        obs[lead_idx, :, :] = target_vals[time_idx, :, :]

        # ── EMOS calibrated: (lead_time, lat, lon) ──
        emos_mu  = ds_calib['bma_mu'].isel(init_time=ii).values.astype(np.float32)
        emos_sig = ds_calib['bma_sigma'].isel(init_time=ii).values.astype(np.float32)
        emos_sig = np.maximum(emos_sig, EPS)
        
        emos_crps[j] = crps_gaussian(emos_mu, emos_sig, obs)

        # Mask missing obs
        invalid = np.isnan(obs)
        emos_crps[j][invalid] = np.nan

    # ── Summary ──
    emos_mean = np.nanmean(emos_crps)
    print(f"EMOS         : {emos_mean:.4f}")
    
    # ── Squeeze back to 3D for single init ──
    if squeeze:
        emos_crps = emos_crps[0]
        
    return emos_crps

bma_crps = crps_verif(ds_calib, ds_input, ds_target, date_idx, var='TREFHT')

ds_verif = xr.Dataset(
    {
        "CRPS_BMA": (["init_time", "lead_time", "lat", "lon"], bma_crps),
    },
    coords={
        "init_time": verif_years,
        "lead_time": np.arange(3650),
        "lat": ds_input.lat.values,
        "lon": ds_input.lon.values,
    },
)

fn_save = f'/glade/derecho/scratch/ksha/EPRI_data/PP_verif/T2_BMA_{verif_years[0]}_{verif_years[-1]}.zarr'
ds_verif.to_zarr(fn_save, mode='w')







