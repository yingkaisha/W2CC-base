import os
import math
import yaml
import copy
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from xclim.indices import (
    standardized_precipitation_evapotranspiration_index,
    water_budget,
)

from xclim.indices.stats import (
    standardized_index_fit_params, 
    standardized_index
)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind_lat', help='ind_lat')
args = vars(parser.parse_args())

import warnings
warnings.filterwarnings("ignore", message="Converting a CFTimeIndex.*noleap.*")

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


SPEI = np.empty((12*(2020-1959+1), 288))
SPEI[...] = np.nan
base_dir = '/glade/derecho/scratch/ksha/EPRI_data/CESM2_grid/SPEI/'

for lead_year in range(10):
    for N_ens in range(20):
        
        start_date = f"{1959+lead_year}-01-01T00"
        end_date = f"{2020+lead_year}-12-31T23"
        
        save_name = base_dir + f'CESM_SPEI_lat{ind_lat}_lead{lead_year}_mem{N_ens}.npy'
        file_path = Path(save_name)
        
        if file_path.exists():
            print("File exists")
            # move to the next file
            continue
        else:
            print(f'missing {save_name}')
            
        ds_collection = []
        
        for year_init in range(1958, 2020, 1):
            
            year_start = year_init + 1 + lead_year
            time_start = f'{year_start}-01-01T00'
            time_end = f'{year_start}-12-31T00'
            
            fn = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year_init}-11-01_daily_ensemble.zarr'
            ds_CESM = xr.open_zarr(fn)[['TREFHTMN', 'TREFHTMX', 'PRECT']].isel(member=N_ens, lat=ind_lat)
            ds_CESM = ds_CESM.sel(time=slice(time_start, time_end))
            ds_collection.append(ds_CESM)
        
        ds_all = xr.concat(ds_collection, dim='time')
        ds_all = ds_all.load()
        
        cft = ds_all.indexes['time']
        pd_idx = cft.to_datetimeindex()
        ds_all = ds_all.assign_coords({'time': pd_idx})
        
        lat_ref = ds_all['lat'].values
        lat_mid = lat_ref # lat_ref[ind_lat]
        time = ds_all['time']
        
        for ind_lon in range(288):
            
            # print(ind_lon)
            ds_subset = ds_all.isel(lon=ind_lon)
            
            tmin = ds_subset['TREFHTMN'].values
            tmax = ds_subset['TREFHTMX'].values
            precip = ds_subset['PRECT'].values
            
            ds = xr.Dataset(
                {
                    "precip": (("time",), precip*1e3, {"units": "kg m-2 s-1"}),
                    "tmin":   (("time",), tmin-273.15, {"units": "degC"}),
                    "tmax":   (("time",), tmax-273.15, {"units": "degC"}),
                },
                coords={"time": time, "lat": lat_mid}
            )
            
            for v in ("precip", "tmin", "tmax"):
                
                ds[v] = ds[v].assign_coords(lat=lat_mid)
                
                ds[v]["lat"].attrs = {
                    "standard_name": "latitude",
                    "units": "degrees_north", "axis": "Y"
                }
            
            precip, tmin, tmax = process_vars(ds)
            # precip["time"] = pd.to_datetime(precip["time"]).to_numpy().astype("datetime64[D]")
            
            params, spei = calc_spei_and_params(
                precip, tmin, tmax, 
                agg_freq=1, 
                start_date=start_date,
                end_date=end_date,
                lat=precip["lat"],
                dist="fisk", method="ML"
            )
            
            SPEI[:, ind_lon] = spei.values
            
        print(save_name)
        np.save(save_name, SPEI)

