import os
import yaml
import copy
import time
import numpy as np
import pandas as pd
import xarray as xr

from xclim.indices import (
    standardized_precipitation_evapotranspiration_index,
    water_budget,
)

from xclim.indices.stats import (
    standardized_index_fit_params, 
    standardized_index
)

import warnings
warnings.filterwarnings("ignore", message="Converting a CFTimeIndex.*noleap.*")

from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind_lat', help='ind_lat')
args = vars(parser.parse_args())

ind_lat = int(args['ind_lat'])

def process_vars(data):
    """
    Preprocess data variables for SPEI calculation.
    """
    
    data["time"] = (pd.to_datetime(data["time"]).to_numpy().astype("datetime64[D]"))
    first_date = data.time.min().dt.strftime("%Y-%m-%d").values.item()
    
    if not first_date.endswith("-01-01"):
        first_year = int(first_date[:4])
        data = data.sel(time=slice(str(first_year + 1), None))

    precip = data["precip"]
    tmin = data["tmin"]
    tmax = data["tmax"]
    
    return precip, tmin, tmax
    
def calc_spei_and_params(
    precip, tmin, tmax, 
    agg_freq, start_date, end_date, 
    lat=None, dist="fisk", method="ML"):
    
    # --- Daily water budget for full period
    wb = water_budget(pr=precip, tasmin=tmin, tasmax=tmax, method="HG85", lat=lat)
    wb.attrs["units"] = "kg m-2 s-1"

    # --- Fit params on the calibration period (monthly aggregation + rolling handled inside)
    params = standardized_index_fit_params(
        wb.sel(time=slice(start_date, end_date)),
        freq="MS",           # aggregate daily WB to monthly
        window=agg_freq,     # e.g., 12 for SPEI-12
        dist=dist,           # "fisk" = 3-parameter log-logistic
        method=method,       # "PWM" (L-moments) is robust; "ML" for MLE
    )

    # --- Apply those params to the full record to get SPEI (ensures consistency)
    spei = standardized_index(
        wb,
        freq="MS",
        window=agg_freq,
        dist=dist,
        method=method,
        params=params,       # << use fixed calibration parameters
        zero_inflated=False,    # for water balance, not zero-inflated
        fitkwargs=None,         # or {}
        cal_start=None,         # not needed when params are supplied
        cal_end=None,   
    )

    # spei = standardized_precipitation_evapotranspiration_index(
    #     wb=wb, freq="MS", window=agg_freq, dist=dist, method=method, params=params
    # )
    
    if np.nanmin(spei.values) < -3:
        spei = spei.where(spei >= -3, np.nan).interpolate_na("time")

    return params, spei

def noleap_to_gregorian_add_leap(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """
    Convert cftime.DatetimeNoLeap time coord to pandas DatetimeIndex (Gregorian)
    and add Feb 29 for leap years by reindexing to a complete daily index and
    linearly filling inserted dates.
    """
    # 1) CFTimeIndex -> pandas DatetimeIndex (drops Feb 29 by definition)
    cft = ds.indexes[time_dim]                 # xarray.coding.cftimeindex.CFTimeIndex
    pd_idx = cft.to_datetimeindex()           # pandas.DatetimeIndex
    ds = ds.assign_coords({time_dim: pd_idx})

    # 2) Build full daily Gregorian index (includes Feb 29 when applicable)
    full_idx = pd.date_range(pd_idx[0], pd_idx[-1], freq="D")

    # 3) Reindex to insert missing days (Feb 29 becomes NaN rows)
    ds2 = ds.reindex({time_dim: full_idx})

    # 4) Fill inserted NaNs by linear interpolation in time
    #    (works for numeric variables; keeps non-numeric as-is)
    num_vars = [v for v in ds2.data_vars if ds2[v].dtype.kind in "fiu"]
    ds2[num_vars] = ds2[num_vars].interpolate_na(time_dim, method="linear")

    return ds2

def fill_nan_linear_2d(a):
    a = np.asarray(a, dtype=float)
    ny, nx = a.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    mask = np.isfinite(a)
    pts = np.column_stack((yy[mask], xx[mask]))   # (row, col) for valid points
    vals = a[mask]

    # Linear interpolation inside the convex hull of valid points
    filled = a.copy()
    filled[~mask] = griddata(pts, vals, (yy[~mask], xx[~mask]), method="linear")

    # Optional: fill any remaining NaNs (outside convex hull) with nearest
    if np.any(~np.isfinite(filled)):
        filled[~np.isfinite(filled)] = griddata(pts, vals, (yy[~np.isfinite(filled)], xx[~np.isfinite(filled)]),
                                                method="nearest")
    return filled

dict_loc = {
    'Pituffik': (76.4, -68.575),
    'Fairbanks': (64.75, -147.4),
    'Guam': (13.475, 144.75),
    'Yuma_PG': (33.125, -114.125),
    'Fort_Bragg': (35.05, -79.115),
}
keys = list(dict_loc.keys())

key = 'Yuma_PG'
dir_stn = '/glade/derecho/scratch/ksha/EPRI_data/METRICS/Yuma_PG/'
base_dir = '/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/'

ds_example = xr.open_zarr(base_dir+'/Yuma_PG/SMYLE_Yuma_PG_2019.zarr')
lat = ds_example['lat'].values
lon = ds_example['lon'].values

Nx = len(lat)
Ny = len(lon)

SPEI = np.empty((10, 12*(2020-1959+1), Ny, 2))
SPEI[...] = np.nan
# 62 years x 12 month

for ind_lon in range(Ny):
    for lead_year in range(0, 10):
          
        start_date = f"{1959+lead_year}-01-01T00"
        end_date = f"{2020+lead_year}-12-31T23"
        
        ds_collection = []
        
        for year_init in range(1958, 2020, 1):
            
            year_start = year_init + 1 + lead_year
            time_start = f'{year_start}-01-01T00'
            time_end = f'{year_start}-12-31T00'
            
            fn_CESM = base_dir + f'/{key}/SMYLE_{key}_{year_init}.zarr'
            ds_CESM = xr.open_zarr(fn_CESM)[['TREFHTMN', 'TREFHTMX', 'PRECT']].isel(lon=ind_lon, lat=ind_lat)
            
            ds_CESM = ds_CESM.sel(time=slice(time_start, time_end))
            ds_collection.append(ds_CESM)
        
        ds_all = xr.concat(ds_collection, dim='time')
        ds_all = ds_all.load()
        
        cft = ds_all.indexes['time']
        pd_idx = cft.to_datetimeindex()
        ds_all = ds_all.assign_coords({'time': pd_idx})
        
        lat_ref = ds_all['lat'].values
        lat_mid = lat_ref # lat_ref[ind_lat]
        time_vals = ds_all['time']
        
        tmin = ds_all['TREFHTMN'].values
        tmax = ds_all['TREFHTMX'].values
        precip = ds_all['PRECT'].values
        
        ds = xr.Dataset(
            {
                "precip": (("time",), precip*1e3, {"units": "kg m-2 s-1"}),
                "tmin":   (("time",), tmin-273.15, {"units": "degC"}),
                "tmax":   (("time",), tmax-273.15, {"units": "degC"}),
            },
            coords={"time": time_vals, "lat": lat_mid}
        )
        
        for v in ("precip", "tmin", "tmax"):
            
            ds[v] = ds[v].assign_coords(lat=lat_mid)
            
            ds[v]["lat"].attrs = {
                "standard_name": "latitude",
                "units": "degrees_north", "axis": "Y"
            }
        
        precip, tmin, tmax = process_vars(ds)

        # ---------------------------------- #
        # 24 month lagged SPEI
        params, spei = calc_spei_and_params(
            precip, tmin, tmax, 
            agg_freq=24, 
            start_date=start_date,
            end_date=end_date,
            lat=precip["lat"],
            dist="fisk", method="ML"
        )
        
        SPEI[lead_year, :, ind_lon, 0] = spei.values

        # ---------------------------------- #
        # 48 month lagged SPEI
        params, spei = calc_spei_and_params(
            precip, tmin, tmax, 
            agg_freq=48, 
            start_date=start_date,
            end_date=end_date,
            lat=precip["lat"],
            dist="fisk", method="ML"
        )
        
        SPEI[lead_year, :, ind_lon, 1] = spei.values

save_name = dir_stn + f'temp_np/SPEI_{ind_lat}.npy'
np.save(save_name, SPEI)
print(save_name)






