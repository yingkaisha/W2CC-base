
import os
import zarr
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd 

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind', help='stn_ind')
args = vars(parser.parse_args())

ind = int(args['ind'])

station_names = ['Pituffik', 'Fairbanks', 'Guam', 'Yuma_PG' ,'Fort_Bragg']

def detrend_linear_doy(da: xr.DataArray, dim: str = "time", keep_mean: bool = True) -> xr.DataArray:
    t = da[dim]
    doy = t.dt.dayofyear
    x = t.dt.year.astype("float64")  # predictor

    # If you have missing values, this keeps x/y stats consistent
    mask = da.notnull()

    # Groupwise means (per DOY)
    x_mean = x.where(mask).groupby(doy).mean(dim, skipna=True)       # dims: dayofyear
    y_mean = da.groupby(doy).mean(dim, skipna=True)                  # dims: dayofyear + (other dims)

    # Broadcast group means back onto the time axis
    x_anom = x - x_mean.sel(dayofyear=doy)                           # dims: time
    y_anom = da - y_mean.sel(dayofyear=doy)                          # dims: time + (other dims)

    # Least-squares slope per DOY: slope = cov(x,y)/var(x)
    numer = (x_anom * y_anom).where(mask).groupby(doy).mean(dim, skipna=True)
    denom = (x_anom ** 2).where(mask).groupby(doy).mean(dim, skipna=True)
    slope = numer / denom                                            # dims: dayofyear + (other dims)

    if keep_mean:
        # Remove slope only; preserves the DOY mean exactly
        return da - slope.sel(dayofyear=doy) * x_anom
    else:
        # Remove full fitted line (slope + intercept)
        intercept = y_mean - slope * x_mean
        fit = slope.sel(dayofyear=doy) * x + intercept.sel(dayofyear=doy)
        return da - fit

def annual_metrics(ds_in, suffix):
    """
    Compute yearly min/max/mean and 30-day rolling-mean max, then rename + suffix.
    Minimal change: same variables as your original.
    """
    g = ds_in.groupby("time.year")
    ds_max  = g.max("time", skipna=True)
    ds_min  = g.min("time", skipna=True)
    ds_mean = g.mean("time", skipna=True)
    ds_30d = (
        ds_in.rolling(time=30, min_periods=30).mean()
            .groupby("time.year").max("time", skipna=True)
    )

    ds_min = ds_min.rename({'TREFHTMN': 'TREFHTMN_min', 'TREFHT': 'TREFHT_min'})[['TREFHTMN_min', 'TREFHT_min']]
    ds_max = ds_max.rename({'PRECT': 'PRECT_max', 'TREFHTMX': 'TREFHTMX_max', 'TREFHT': 'TREFHT_max'})[['PRECT_max', 'TREFHTMX_max', 'TREFHT_max']]
    ds_30d = ds_30d.rename({'TREFHT': 'TREFHT_30d', 'PRECT': 'PRECT_30d'})[['TREFHT_30d', 'PRECT_30d']]
    ds_mean = ds_mean.rename({'PRECT': 'PRECT_mean', 'TREFHT': 'TREFHT_mean'})[['PRECT_mean', 'TREFHT_mean']]

    ds_out = xr.merge([ds_min, ds_max, ds_30d, ds_mean])
    return ds_out.rename({v: f"{v}_{suffix}" for v in ds_out.data_vars})

stn = station_names[ind]

base_dir = f'/glade/derecho/scratch/ksha/EPRI_data/METRICS_STN/{stn}/'

# ========================== #
# get data
list_ds = []
for year in range(1958, 2025):
    fn = f'/glade/campaign/ral/hap/ksha/EPRI_data/ERA5_daily/{stn}_{year}.zarr'
    ds = xr.open_zarr(fn)
    list_ds.append(ds)
    
ds_all = xr.concat(list_ds, dim='time')
ds_all = ds_all[['PRECT', 'TREFHT', 'TREFHTMX', 'TREFHTMN']]

ds_all['PRECT'] = ds_all['PRECT'] * 1000  # mm per day
ds_all = ds_all.chunk({"time": -1})

# ========================== #
# get anomaly (vectorized)
clim = ds_all.groupby("time.dayofyear").mean("time", keep_attrs=True)
ds_all_anom = ds_all.groupby("time.dayofyear") - clim

# ========================== #
# get detrend data (your function; keep loop minimal)
ds_all_detrend = ds_all.copy()
for v in ds_all.data_vars:
    ds_all_detrend[v] = detrend_linear_doy(ds_all[v], dim="time", keep_mean=False)

# ======================= #
# metrics (compute via helper; avoids repeating logic)
ds_metrics_default = annual_metrics(ds_all, "default")
ds_metrics_anom    = annual_metrics(ds_all_anom, "anom")
ds_metrics_detrend = annual_metrics(ds_all_detrend, "detrend")

# ========================== #
# save
ds_final = xr.merge([ds_metrics_default, ds_metrics_anom, ds_metrics_detrend])

save_name = base_dir + 'metrics.zarr'
ds_final.to_zarr(save_name, mode='w')
print(save_name)




