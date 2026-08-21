
import os
import sys
import time
import zarr
import random
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.stats import norm
from scipy.interpolate import interp1d

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

VAR = VAR_CESM = 'PRECT'
N_ens = 1 # 20 # draw 20 members
EPS = 1e-6        # floor for predicted variance
verif_years = np.arange(2010, 2020)
SAVE_DIR = f'/glade/derecho/scratch/ksha/EPRI_data/QM_PRED/{VAR_CESM}/'
QM_DIR = f'/glade/derecho/scratch/ksha/EPRI_data/QM/{VAR_CESM}'

lat_ind = np.arange(192)

list_coef = []
for lat_i in lat_ind:
    fn = f'/glade/derecho/scratch/ksha/EPRI_data/EMOS/{VAR_CESM}_QM/emos_coef_lat_ind_{lat_i}.zarr'
    list_coef.append(xr.open_zarr(fn))

ds_emos = xr.concat(list_coef, dim='lat')
ds_emos = ds_emos.load()
ds_emos = ds_emos.transpose('lead_time', 'lat', 'lon')

list_input = []
# list_target = []
for lat_i in np.arange(192):
    fn_input  = f'{SAVE_DIR}/pred_input_lat_ind_{lat_i}.zarr'
    ds_CESM = xr.open_zarr(fn_input)
    list_input.append(ds_CESM)

    # fn_target = f'{SAVE_DIR}/pred_target_lat_ind_{lat_i}.zarr'
    # ds_ERA5 = xr.open_zarr(fn_target)
    # list_target.append(ds_ERA5)

ds_input = xr.concat(list_input, dim='lat')
# ds_target = xr.concat(list_target, dim='lat')

ds_input = ds_input.transpose('init_time', 'member', 'lead_time', 'lat', 'lon')
# ds_target = ds_target.transpose('init_time', 'lead_time', 'lat', 'lon')

fn = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/QM_EMOS_calib.zarr'
ds_calib = xr.open_zarr(fn)

print("Loading ERA5 quantile tables for all latitudes …")
n_init = len(verif_years)
n_lead = 3650
n_lat  = 192
n_lon  = len(ds_input.lon.values)
N_DOY  = 365
N_QUANTILES = 200
doy = np.arange(n_lead) % N_DOY

# era5_qtable: (365, n_lat, n_lon, N_QUANTILES)
# era5_wetfrac: (365, n_lat, n_lon)
list_qt = []
list_wf = []
for lat_i in tqdm(range(n_lat), desc='Loading QM tables'):
    fn = f'{QM_DIR}/qm_era5_lat_ind_{lat_i}.zarr'
    ds_qm = xr.open_zarr(fn)
    list_qt.append(ds_qm['quantile_values'].values)    # (365, n_lon, N_QUANTILES)
    list_wf.append(ds_qm['wet_fraction'].values)        # (365, n_lon)

era5_qtable  = np.stack(list_qt, axis=1)    # (365, n_lat, n_lon, N_QUANTILES)
era5_wetfrac = np.stack(list_wf, axis=1)    # (365, n_lat, n_lon)

q_levels = ds_qm['quantile'].values

print(f"  era5_qtable  shape: {era5_qtable.shape}")
print(f"  era5_wetfrac shape: {era5_wetfrac.shape}")


print(f"Ensemble dressing with {N_ens} members …")
emos_mu_vals  = ds_calib['emos_mu'].values.astype(np.float32)
emos_sig_vals = np.maximum(ds_calib['emos_sigma'].values.astype(np.float32), EPS)

# Allocate output
dressed = np.full(
    (n_init, N_ens, n_lead, n_lat, n_lon), np.nan, dtype=np.float32
)

rng = np.random #.default_rng(42)
# Loop over lat to manage memory
for lat_i in tqdm(range(n_lat), desc='Dressing'):

    # Draw Gaussian samples: (n_init, N_ens, n_lead, n_lon)
    mu_lat  = emos_mu_vals[:, :, lat_i, :]     # (n_init, n_lead, n_lon)
    sig_lat = emos_sig_vals[:, :, lat_i, :]

    z = rng.normal(
        loc=mu_lat[:, None, :, :],             # (n_init, 1, n_lead, n_lon)
        scale=sig_lat[:, None, :, :],
        size=(n_init, N_ens, n_lead, n_lon),
    )

    # Inverse NQT per (lead, lon) using doy
    for lt in range(n_lead):
        d = doy[lt]
        for lo in range(n_lon):
            qt = era5_qtable[d, lat_i, lo, :]
            wf = era5_wetfrac[d, lat_i, lo]
            if np.isnan(wf):
                continue

            # All inits × all members for this (lead, lon): flatten → transform → reshape
            g_flat = z[:, :, lt, lo].ravel()           # (n_init * N_ens,)
            p_flat = gaussian_to_precip(g_flat, qt, wf)
            dressed[:, :, lt, lat_i, lo] = p_flat.reshape(n_init, N_ens)

# ════════════════════════════════════════════════════════════════════════
# Package and save
# ════════════════════════════════════════════════════════════════════════

print("Packaging dressed ensemble …")
ds_dressed = xr.Dataset(
    {
        VAR_CESM: (
            ['init_time', 'member', 'lead_time', 'lat', 'lon'],
            dressed,
        ),
    },
    coords={
        'init_time':  verif_years+1,
        'member':     np.arange(N_ens),
        'lead_time':  np.arange(n_lead),
        'lat':        ds_input.lat.values,
        'lon':        ds_input.lon.values,
    },
)
ds_dressed.attrs['description'] = (
    f'EMOS-calibrated ensemble (N={N_ens}) for {VAR_CESM} in mm/day. '
    'Generated by drawing from N(emos_mu, emos_sigma^2) in NQT-Gaussian space '
    'and inverse-transforming via ERA5 quantile tables.'
)
ds_dressed.attrs['units'] = 'mm/day'
ds_dressed.attrs['n_members'] = N_ens

num = random.randint(100, 999)
fn_out = f'/glade/derecho/scratch/ksha/EPRI_data/PP_calib/QM_EMOS_dressed_N{N_ens}_{num}.zarr'
print(f"Saving to {fn_out} …")
ds_dressed.to_zarr(fn_out, mode='w')
print("Done.")















