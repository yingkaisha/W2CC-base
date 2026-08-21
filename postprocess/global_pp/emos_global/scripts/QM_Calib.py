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
from scipy.interpolate import interp1d

VAR = VAR_CESM = 'PRECT'
VAR_ERA5 = 'total_precipitation'
unit_CESM = 60*60*24 * 1000 # m/s to mm/day
unit_ERA5 = 1000  # m/day to mm/day
EPS = 1e-6        # floor for predicted variance
verif_years = np.arange(2010, 2020)
N_ens = 50

QM_DIR = f'/glade/derecho/scratch/ksha/EPRI_data/QM/{VAR_CESM}'
fn = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/QM_EMOS_calib.zarr'
ds_calib = xr.open_zarr(fn)

n_init = len(verif_years)
n_lead = 3650
n_lat  = 192
n_lon  = len(ds_calib.lon.values)
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

def extrapolate_upper_tail(q_values, q_levels_in):
    """
    Linearly extrapolate from the two highest quantiles to level 1.0.
    """
    v1, v2 = q_values[-2], q_values[-1]
    l1, l2 = q_levels_in[-2], q_levels_in[-1]
    slope = (v2 - v1) / max(l2 - l1, EPS)
    v_max = max(v2 + slope * (1.0 - l2), v2)
    return np.append(q_values, np.float32(v_max)), np.append(q_levels_in, 1.0)


def gaussian_to_precip(gaussian_values, q_values, wet_frac):
    """
    Inverse NQT: Gaussian space → precipitation (mm/day).

    Steps:
      1. Gaussian → uniform via norm.cdf
      2. If uniform rank ≤ dry_frac → zero precipitation
      3. If uniform rank > dry_frac → interpolate through ERA5
         quantile table (with extrapolated upper tail)

    Parameters
    ----------
    gaussian_values : 1D array of Gaussian-space values
    q_values        : (N_QUANTILES,) ERA5 quantile table
    wet_frac        : scalar, fraction of wet days in training

    Returns
    -------
    precip : 1D array, same length, in mm/day
    """
    shape = gaussian_values.shape
    g_flat = gaussian_values.ravel()
    precip = np.full_like(g_flat, np.nan, dtype=np.float32)

    if np.all(np.isnan(q_values)):
        return precip.reshape(shape)

    valid = ~np.isnan(g_flat)
    if not valid.any():
        return precip.reshape(shape)

    # Gaussian → uniform
    u = norm.cdf(g_flat[valid])

    # Dry fraction threshold
    dry_frac = 1.0 - wet_frac
    result = np.zeros_like(u, dtype=np.float32)

    # Only interpolate for wet values
    wet_mask = u > dry_frac
    if wet_mask.any():
        # Extract wet portion of quantile table
        wet_q_mask = q_values > 0.0
        if wet_q_mask.sum() >= 2:
            q_wet = q_values[wet_q_mask]
            l_wet = q_levels[wet_q_mask]

            # Extend upper tail
            q_ext, l_ext = extrapolate_upper_tail(q_wet, l_wet)

            q_unique, idx_unique = np.unique(q_ext, return_index=True)
            ql_unique = l_ext[idx_unique]

            if len(q_unique) >= 2:
                # Rescale wet uniform ranks from [dry_frac, 1] → [l_min, l_max]
                l_min = ql_unique[0]
                l_max = ql_unique[-1]
                u_rescaled = l_min + (l_max - l_min) * ((u[wet_mask] - dry_frac) / max(1.0 - dry_frac, EPS))
                inv_func = interp1d(
                    ql_unique, q_unique,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(q_unique[0], q_unique[-1]),
                )
                result[wet_mask] = np.maximum(inv_func(u_rescaled), 0.0)
            else:
                result[wet_mask] = q_values[wet_q_mask].mean()
        else:
            # Too few wet quantiles — use the single wet value
            if wet_q_mask.any():
                result[wet_mask] = q_values[wet_q_mask][0]

    precip[valid] = result.astype(np.float32)
    return precip.reshape(shape)


emos_precip = np.full((n_init, n_lead, n_lat, n_lon), np.nan, dtype=np.float32)

for lat_i in tqdm(range(n_lat), desc='Inverse NQT'):

    # Load only this latitude slice: (n_init, n_lead, n_lon)
    mu_lat = ds_calib['emos_mu'].isel(lat=lat_i).values.astype(np.float32)

    for lt in range(n_lead):
        d = doy[lt]
        for lo in range(n_lon):
            qt = era5_qtable[d, lat_i, lo, :]
            wf = era5_wetfrac[d, lat_i, lo]
            if np.isnan(wf):
                continue

            g_vals = mu_lat[:, lt, lo]       # (n_init,)
            valid = ~np.isnan(g_vals)
            if not valid.any():
                continue

            p_vals = gaussian_to_precip(g_vals[valid], qt, wf)
            emos_precip[valid, lt, lat_i, lo] = p_vals

    del mu_lat

ds_calib['emos_PRECT'] = xr.DataArray(
    emos_precip,
    dims=['init_time', 'lead_time', 'lat', 'lon'],
    coords=ds_calib['emos_mu'].coords,
)
ds_calib['emos_PRECT'].attrs['units'] = 'mm/day'
ds_calib['emos_PRECT'].attrs['description'] = (
    'EMOS calibrated mean inverse-transformed from Gaussian to precipitation '
    'via ERA5 quantile tables. Censored at zero.'
)

save_name = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/QM_EMOS_calib_PRECT.zarr'
print(f"Saving to {save_name} …")
ds_calib.to_zarr(save_name, mode='w')
print("Done.")



















