import os
import math
import yaml
import copy
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


# =================================================================== #
# Prepare data

varnames = [
    "2m_temperature", 
    "maximum_2m_temperature_since_previous_post_processing", 
    "minimum_2m_temperature_since_previous_post_processing", 
    "total_precipitation"
]

ds_collection = []
for year in range(1958, 2026):
    fn_ERA5 = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
    ds_ERA5 = xr.open_zarr(fn_ERA5)[varnames]
    ds_ERA5 = ds_ERA5.rename(
        {
            'total_precipitation': "PRECT", 
            "2m_temperature": "TREFHT", 
            "maximum_2m_temperature_since_previous_post_processing": "TREFHTMAX", 
            "minimum_2m_temperature_since_previous_post_processing": "TREFHTMIN"
        }
    )

    ds_ERA5["TREFHT"] -= 273.15
    ds_ERA5["TREFHTMAX"] -= 273.15
    ds_ERA5["TREFHTMIN"] -= 273.15
    ds_ERA5["PRECT"] = ds_ERA5["PRECT"]*1e3/86400
    
    ds_collection.append(ds_ERA5)
    
ds_all = xr.concat(ds_collection, dim="time", combine_attrs="override")

time = ds_all['time'].values
start_date = "1958-01-01T00"
end_date = "2025-12-31T23"

lat_ref = ds_all['lat'].values

# =================================================================== #
# SPEI compute

SPEI = np.zeros((12*(2025-1958+1), 288))

# ====================================== #
# An example grid

for ind_lon in range(288):
    print(f'{ind_lat}, {ind_lon}')
    
    lat_mid = lat_ref[ind_lat]
    
    ds = ds_all.isel(lon=ind_lon, lat=ind_lat)
    precip = ds["PRECT"].values
    t2m_max = ds["TREFHTMAX"].values
    t2m_min = ds["TREFHTMIN"].values
    
    # ================================================= #
    # pack ds
    ds = xr.Dataset(
        {
            "precip": (("time",), precip, {"units": "kg m-2 s-1"}),
            "tmin":   (("time",), t2m_min, {"units": "degC"}),
            "tmax":   (("time",), t2m_max, {"units": "degC"}),
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


save_name = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/SPEI/ERA5_SPEI_lat{ind_lat}'
np.save(save_name, SPEI)






