
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

# ==================================================================== #
# CESM variables
varname_dict = {
    'PSL': 'Sea level pressure',
    'TREFHT': '2-m Air Temperature [K]',
    'TREFHTMN': '2-m Air Temperature [K]',
    'TREFHTMX': '2-m Air Temperature [K]',
    'QREFHT': '2-m Specific Humidity [kg/kg]',
    'PRECT': 'Total Precipitation [m/s]',
    'PRECSC': 'Convective snow rate (water equivalent) [m/s]',
    'PRECSL': 'Large-scale (stable) snow rate (water equivalent) [m/s]',
    'TMQ': 'Total (vertically integrated) precipitable water [kg/m2]',
    'FLDS': 'Downwelling longwave flux at surface [W/m2]',
    'FSDS': 'Downwelling solar flux at surface [W/m2]',
    'U10': '10-m wind speed [m/s]',
    'Z500': 'Geopotential Z at  500 mbar pressure surface'
}

# ==================================================================== #
# file path
base_case = 'b.e21.BSMYLE.f09_g17'
base_dir_DP = '/glade/campaign/cesm/development/espwg/CESM2-DP/timeseries/'
base_dir_SMYLE = '/glade/campaign/cesm/development/espwg/SMYLE/archive/'

# ==================================================================== #
# DP collection loop
members = np.arange(11, 31, 1)
open_kwargs = dict(engine="netcdf4", chunks={"time": 6, "lev": 12, "lat": 192, "lon": 288})

ds_full_DP = []

for i_member, mem in enumerate(members):

    ds_collect = []
    
    str_member = f'{mem:03d}'
    fn_daily = base_dir_DP + f'{base_case}.{year}-11.{str_member}/' + '/atm/proc/tseries/day_1/'
    
    for varname in list(varname_dict.keys()):
        fn_var = glob(fn_daily + f'*{varname}.*')[0]
        ds_collect.append(xr.open_dataset(fn_var, **open_kwargs)[[varname,]])
    
    ds_full_DP.append(xr.merge(ds_collect))

lat = ds_full_DP[0]['lat'].values
lon = ds_full_DP[0]['lon'].values

for ds in ds_full_DP:
    ds['lat'] = lat
    ds['lon'] = lon

# ==================================================================== #
# SMYLE collection loop
members = np.arange(11, 31, 1)
open_kwargs = dict(engine="netcdf4", chunks={"time": 6, "lev": 12, "lat": 192, "lon": 288})

ds_full_SMYLE = []

for i_member, mem in enumerate(members):

    ds_collect = []
    
    str_member = f'{mem:03d}'
    fn_daily = base_dir_SMYLE + f'{base_case}.{year}-11.{str_member}/' + '/atm/proc/tseries/day_1/'
    
    for varname in list(varname_dict.keys()):
        fn_var = glob(fn_daily + f'*{varname}.*')[0]
        ds_collect.append(xr.open_dataset(fn_var, **open_kwargs)[[varname,]])
    
    ds_full_SMYLE.append(xr.merge(ds_collect))

lat = ds_full_SMYLE[0]['lat'].values
lon = ds_full_SMYLE[0]['lon'].values

for ds in ds_full_SMYLE:
    ds['lat'] = lat
    ds['lon'] = lon

# ==================================================================== #
# merge

ds_DP = xr.concat(ds_full_DP, dim=xr.DataArray(members, dims="member", name="member"),)
ds_DP = ds_DP.assign_coords(member=("member", members))
ds_DP = ds_DP.sel(time=slice(f'{year+2}-11-01T00', f'{year+10}-12-31T00'))

ds_SMYLE = xr.concat(ds_full_SMYLE, dim=xr.DataArray(members, dims="member", name="member"),)
ds_SMYLE = ds_SMYLE.assign_coords(member=("member", members))
ds_SMYLE = ds_SMYLE.sel(time=slice(f'{year}-11-01T00', f'{year+2}-10-31T00'))

ds_year = xr.concat([ds_SMYLE, ds_DP], dim='time')
ds_year = ds_year.chunk({"member": 20, "time": 6, "lat": 192, "lon": 288})
# ==================================================================== #
# encoding
dict_encoding = {}
varnames = list(ds_year.keys())
varname_4D = []

chunk_size_3d = dict(chunks=(20, 6, 192, 288)) # member, time, lat, lon
chunk_size_4d = dict(chunks=(20, 6, 12, 192, 288)) # member, time, lev, lat, lon
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# ==================================================================== #
# save
save_name = f'/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/SMYLE_{year}-11-01_daily_ensemble.zarr'
ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)


