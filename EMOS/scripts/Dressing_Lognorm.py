
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
N_ens = 1 # quick test on 1 member

lat_ind = np.arange(192)

list_coef = []
list_missing = []
for lat_i in lat_ind:
    try:
        fn = f'/glade/derecho/scratch/ksha/EPRI_data/EMOS/{VAR_CESM}_lognorm/emos_coef_lat_ind_{lat_i}.zarr'
        list_coef.append(xr.open_zarr(fn))
    except:
        list_missing.append(lat_ind)

ds_emos = xr.concat(list_coef, dim='lat')
ds_emos = ds_emos.load()
ds_emos = ds_emos.transpose('lead_time', 'lat', 'lon')

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
ds_input[VAR_CESM] = ds_input[VAR_CESM]**0.25

# def apply_emos(ds_input, ds_emos, var=VAR_CESM):
#     """
#     Apply fitted EMOS to ensemble forecasts.
    
#     Returns an xr.Dataset with:
#         emos_mu    : calibrated predictive mean   (init_time, lead_time, lat, lon)
#         emos_sigma : calibrated predictive std     (init_time, lead_time, lat, lon)
#     """
#     ens_mean = ds_input[var].mean(dim='member')
#     ens_var  = ds_input[var].var(dim='member', ddof=1)
 
#     a = ds_emos['a']  # (lead_time, lat, lon)
#     b = ds_emos['b']
#     c = ds_emos['c']
#     d = ds_emos['d']
 
#     mu    = a + b * ens_mean                         # broadcast over init_time
#     sigma = np.sqrt(np.maximum(c + d * ens_var, EPS))
 
#     return xr.Dataset({
#         'emos_mu':    mu,
#         'emos_sigma': sigma,
#     })
    
# print("Applying EMOS to the training forecasts (for verification) …")
# ds_calib = apply_emos(ds_input, ds_emos, var=VAR)
# ds_calib = ds_calib.transpose('init_time', 'lead_time', 'lat', 'lon')
# ds_calib = ds_calib.load()
# fn = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/PRECT_lognorm_EMOS_calib.zarr'
# ds_calib.to_zarr(fn, mode='w')

fn = '/glade/derecho/scratch/ksha/EPRI_data/PP_calib/PRECT_lognorm_EMOS_calib.zarr'
ds_calib = xr.open_zarr(fn)

print(f"Ensemble dressing with {N_ens} members …")

n_init = len(verif_years)
n_lead = 3650
n_lat  = ds_calib.sizes['lat']
n_lon  = ds_calib.sizes['lon']

emos_mu  = ds_calib['emos_mu'].values.astype(np.float32)
emos_sig = np.maximum(ds_calib['emos_sigma'].values.astype(np.float32), EPS)
# shape: (n_init, n_lead, n_lat, n_lon)

rng = np.random.default_rng(42)

# Draw Gaussian samples in quad-root space, then invert to mm/day
# Process per init to manage memory
dressed = np.full((n_init, N_ens, n_lead, n_lat, n_lon), np.nan, dtype=np.float32)

for ii in tqdm(range(n_init), desc='Dressing'):
    # (N_ens, lead, lat, lon)
    z = rng.normal(
        loc=emos_mu[ii][None, ...],
        scale=emos_sig[ii][None, ...],
        size=(N_ens, n_lead, n_lat, n_lon),
    )

    # Censor negatives in quad-root space, then raise to 4th power
    z = np.maximum(z, 0.0)
    dressed[ii] = z ** 4   # back to mm/day

# ── Package ──
ds_dressed = xr.Dataset(
    {
        VAR_CESM: (['init_time', 'member', 'lead_time', 'lat', 'lon'], dressed),
    },
    coords={
        'init_time':  verif_years + 1,
        'member':     np.arange(N_ens),
        'lead_time':  np.arange(n_lead),
        'lat':        ds_calib.lat.values,
        'lon':        ds_calib.lon.values,
    },
)
ds_dressed.attrs['description'] = (
    f'EMOS-calibrated ensemble (N={N_ens}) for {VAR_CESM} in mm/day. '
    'Drawn from N(emos_mu, emos_sigma^2) in quad-root space, '
    'censored at zero, then raised to the 4th power.'
)
ds_dressed.attrs['units'] = 'mm/day'

# ── Sanity checks ──
print("\nSanity checks:")
print(f"  Negative values: {100 * np.nanmean(dressed < 0):.4f}%")
print(f"  Zero precip:     {100 * np.nanmean(dressed == 0):.1f}%")
print(f"  NaN fraction:    {100 * np.nanmean(np.isnan(dressed)):.1f}%")
print(f"  Global mean:     {np.nanmean(dressed):.3f} mm/day")
print(f"  99th percentile: {np.nanpercentile(dressed, 99):.2f} mm/day")

num = random.randint(100, 999)
fn_out = f'/glade/derecho/scratch/ksha/EPRI_data/PP_calib/Lognorm_EMOS_dressed_N{N_ens}_{num}.zarr'
print(f"Saving to {fn_out} …")
ds_dressed.to_zarr(fn_out, mode='w')
print("Done.")



































