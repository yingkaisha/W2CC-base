
import os
import yaml
import copy
import time
import numpy as np
import pandas as pd
import xarray as xr

save_dir = f'/glade/derecho/scratch/ksha/EPRI_data/METRICS_GLOBE/'
base_dir = '/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/'

var_map = ["TREFHT", "TREFHTMX", "PRECT"]
list_per_verif = []

# cache: init_file -> ensemble-mean dataset (lazy dask ok)
_cache = {}

for verif_year in range(1968, 2021):

    list_per_lead = []
    for lead_year in range(10):
        
        init_year = verif_year - lead_year
        fn_CESM = base_dir + f"SMYLE_{init_year-1}-11-01_daily_ensemble.zarr"

        if fn_CESM not in _cache:
            ds0 = xr.open_zarr(fn_CESM)[var_map]
            _cache[fn_CESM] = ds0.mean("member")

        ds_CESM = _cache[fn_CESM].sel(time=slice(f"{verif_year}-01-01", f"{verif_year}-12-31"))
        
        ds_max = ds_CESM[["PRECT", "TREFHTMX"]].max("time", skipna=True).rename(
            {"PRECT": "PRECT_max", "TREFHTMX": "TREFHTMX_max"}
        )
        ds_mean = ds_CESM[["PRECT", "TREFHT"]].mean("time", skipna=True).rename(
            {"PRECT": "PRECT_mean", "TREFHT": "TREFHT_mean"}
        )

        ds_merge = xr.merge([ds_max, ds_mean]).expand_dims(lead_year=[lead_year])
        list_per_lead.append(ds_merge)

    ds_per_verif = xr.concat(list_per_lead, dim="lead_year").expand_dims(valid_year=[verif_year])
    list_per_verif.append(ds_per_verif)

ds_all = xr.concat(list_per_verif, dim="valid_year")
ds_all = ds_all.chunk({"valid_year": -1, "lead_year": -1, "lat": 192, "lon": 288})

save_name = save_dir + 'CESM_minmax.zarr'
ds_all.to_zarr(save_name, mode='w')
print(save_name)

