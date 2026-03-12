
import os
import sys
import time
import dask
import zarr

import numpy as np
import xesmf as xe
import xarray as xr

import pandas as pd
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

varnames = [
    'total_column_water',
    'surface_pressure',
    '2m_temperature',
    'minimum_2m_temperature_since_previous_post_processing', 
    'maximum_2m_temperature_since_previous_post_processing', 
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
    'surface_solar_radiation_downwards',
    'surface_thermal_radiation_downwards'
]

ERA5_1h = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token='anon'),)[varnames]

ds = ERA5_1h.sel(time=slice(f'{year-1}-12-01T00', f'{year}-12-31T23'))
ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

ds_tp = ds[['total_precipitation',]]
ds_ave = ds[list(set(varnames)-set(['total_precipitation',]))]

time_start = '{}-12-31T00'.format(year-1)
time_start_save = '{}-01-01T00'.format(year)
time_end = '{}-12-31T23'.format(year)

ds_tp = ds_tp.sel(time=slice(time_start, time_end))
ds_tp = ds_tp.shift(time=-1)
ds_tp = ds_tp.resample(time='24h').sum()
ds_tp['time'] = ds_tp['time'] + pd.Timedelta(hours=24)
ds_tp = ds_tp.sel(time=slice(time_start_save, time_end))

ds_ave = ds_ave.sel(time=slice(time_start, time_end))
ds_ave = ds_ave.resample(time='24h').mean()
ds_ave = ds_ave.sel(time=slice(time_start_save, time_end))
ds_final = xr.merge([ds_tp, ds_ave])

fn_CESM = '/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_1958-11-01_daily_ensemble.zarr'
ds_CESM = xr.open_zarr(fn_CESM)
# ds_CESM['lon'] = (ds_CESM['lon']  + 180) % 360 - 180
regridder = xe.Regridder(ds_final, ds_CESM, method='bilinear')
ds_interp = regridder(ds_final)

ds_interp = ds_interp.chunk({"time": 32, "lat": 192, "lon": 288})

# ==================================================================== #
# encoding
dict_encoding = {}
varnames = list(ds_interp.keys())
varname_4D = []

chunk_size_3d = dict(chunks=(32, 192, 288))
chunk_size_4d = dict(chunks=(32, 12, 192, 288))
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# ==================================================================== #
# save
save_name = f'/glade/derecho/scratch/ksha/EPRI_data/ERA5_grid/ERA5_{year}.zarr'
ds_interp.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)


