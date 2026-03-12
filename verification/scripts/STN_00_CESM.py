
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

year_init = int(args['year'])


# ========================================================= #
# station info
dict_loc = {
    'Pituffik': (76.4, -68.575),
    'Fairbanks': (64.75, -147.4),
    'Guam': (13.475, 144.75),
    'Yuma_PG': (33.125, -114.125),
    'Fort_Bragg': (35.05, -79.115),
}

ds_static = xr.open_zarr('/glade/derecho/scratch/ksha/EPRI_data/static/static.zarr')
lon = ds_static['lon'].values
lat = ds_static['lat'].values

varnames = ['FLDS', 'FSDS', 'PRECSC', 'PRECSL', 'PRECT', 'PSL', 'QREFHT', 'TMQ', 'TREFHT', 'TREFHTMN', 'TREFHTMX', 'U10']

# ============================================================================= #
year_start = year_init + 1
year_end = year_init + 10

fn = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year_init}-11-01_daily_ensemble.zarr'
ds = xr.open_zarr(fn)[varnames]
ds = ds.sel(time=slice(f'{year_start}-01-01T00', f'{year_end}-12-31T00'))

ds = ds.assign_coords(lon = (((ds.lon + 180) % 360) - 180))
ds = ds.sortby("lon")

for stn, (lat_mid, lon_mid) in dict_loc.items():
    
    subset = ds.sel(lat=lat_mid, lon=lon_mid, method='nearest')
    subset = subset.mean(('member',))
    
    L = len(subset['time'])
    subset = subset.chunk({'time': L})
    
    save_name = f'/glade/campaign/ral/hap/ksha/EPRI_data/CESM_SMYLE_STN/{stn}_{year_init}.zarr'        
    subset.to_zarr(save_name, mode='w', consolidated=True, compute=True)
    print(save_name)

