
import os
import sys
import time
import dask
import zarr
import xesmf as xe
import numpy as np
import xarray as xr
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

ds_static = xr.open_zarr('/glade/derecho/scratch/ksha/EPRI_data/static/static.zarr')
lon = ds_static['lon'].values
lat = ds_static['lat'].values

dict_loc = {
    'Pituffik': (76.4, -68.575),
    'Fairbanks': (64.75, -147.4),
    'Guam': (13.475, 144.75),
    'Yuma_PG': (33.125, -114.125),
    'Fort_Bragg': (35.05, -79.115),
}

stn_pad = 10
keys = list(dict_loc.keys())

base_dir = '/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/'
years = np.arange(1958, 2020)

def _wrap_lon_180(lon):
    """Map longitude to [-180, 180). Works for DataArray/ndarray/scalar."""
    return ((lon + 180) % 360) - 180

def _infer_lat_lon_names(ds):
    # Adjust candidates if your ERA5 zarr uses different naming
    lat_name = 'lat' if 'lat' in ds.coords else ('latitude' if 'latitude' in ds.coords else None)
    lon_name = 'lon' if 'lon' in ds.coords else ('longitude' if 'longitude' in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"Cannot find lat/lon coords. Found coords: {list(ds.coords)}")
    return lat_name, lon_name

for i, key in enumerate(keys):
    stn_lat, stn_lon = dict_loc[key]
    lon_min, lon_max = stn_lon - stn_pad, stn_lon + stn_pad
    lat_min, lat_max = stn_lat - stn_pad, stn_lat + stn_pad

    out_dir = os.path.join(base_dir, key)
    os.makedirs(out_dir, exist_ok=True)

    fn = os.path.join(base_dir, f'SMYLE_{year}-11-01_daily_ensemble.zarr')
    ds = xr.open_zarr(fn)
    ds = ds.mean(['member'])
    
    lat_name, lon_name = _infer_lat_lon_names(ds)

    # ---- longitude handling (supports 0..360 or -180..180, and dateline crossing) ----
    lon = ds[lon_name]
    lon_is_360 = (lon.min() >= 0) and (lon.max() > 180)

    if lon_is_360:
        # Keep in [0, 360)
        lon_min_mod = lon_min % 360
        lon_max_mod = lon_max % 360

        if lon_min_mod <= lon_max_mod:
            lon_mask = (lon >= lon_min_mod) & (lon <= lon_max_mod)
        else:
            # crosses 0 meridian in 0..360 system
            lon_mask = (lon >= lon_min_mod) | (lon <= lon_max_mod)
    else:
        # Use [-180, 180)
        lon_wrapped = _wrap_lon_180(lon)
        lon_min_w = _wrap_lon_180(lon_min)
        lon_max_w = _wrap_lon_180(lon_max)

        if lon_min_w <= lon_max_w:
            lon_mask = (lon_wrapped >= lon_min_w) & (lon_wrapped <= lon_max_w)
        else:
            # crosses dateline in -180..180 system
            lon_mask = (lon_wrapped >= lon_min_w) | (lon_wrapped <= lon_max_w)

    lat_mask = (ds[lat_name] >= lat_min) & (ds[lat_name] <= lat_max)

    # ---- subset ----
    ds_sub = ds.sel({lat_name: lat_mask, lon_name: lon_mask})

    # Ensure coords are monotonically increasing for nicer downstream behavior
    if ds_sub[lat_name][0] > ds_sub[lat_name][-1]:
        ds_sub = ds_sub.sortby(lat_name)

    # ---- save ----
    save_name = os.path.join(out_dir, f'SMYLE_{key}_{year}.zarr')
    print(save_name)
    ds_sub.to_zarr(save_name, mode='w')


