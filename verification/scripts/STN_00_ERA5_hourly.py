
import os
import sys
import time
import dask
import zarr

import xesmf as xe
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])


# ========================================================= #
# station info
dict_loc = {
    'Pituffik': (76.4, -68.575),
    'Fairbanks': (64.75, -147.4),
    'Guam': (13.475, 144.75),
    'Yuma_PG': (33.125, -114.125),
    'Fort_Bragg': (35.05, -79.115),
}

varnames = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
    'minimum_2m_temperature_since_previous_post_processing', 
    'maximum_2m_temperature_since_previous_post_processing', 
    'surface_solar_radiation_downwards'
]

ERA5_1h = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token='anon'),)[varnames]

ERA5_select = ERA5_1h.sel(time=slice(f'{year}-01-01T00', f'{year}-12-31T23'))

ERA5_select = ERA5_select.assign_coords(longitude = (((ERA5_select.longitude + 180) % 360) - 180))
ERA5_select = ERA5_select.sortby("longitude")

save_name = '/glade/campaign/ral/hap/ksha/EPRI_data/ERA5_hourly/{}_{}.zarr'

for stn, (lat_mid, lon_mid) in dict_loc.items():
    
    save_name_ = save_name.format(stn, year)
    file_path = Path(save_name_)
    
    if file_path.exists():
        print("File exists")
        # move to the next stn
        continue
    else:
        print('missing')
        
        subset = ERA5_select.sel(latitude=lat_mid, longitude=lon_mid, method='nearest')
        
        subset.to_zarr(save_name_)
        print(f'Save to {save_name_}')

