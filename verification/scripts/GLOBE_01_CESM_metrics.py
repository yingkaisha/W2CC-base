
import os
import yaml
import copy
import time
import zarr
import numpy as np
import pandas as pd
import xarray as xr

save_dir = f'/glade/derecho/scratch/ksha/EPRI_data/METRICS_GLOBE/'
base_dir = '/glade/derecho/scratch/ksha/EPRI_data/CESM2_SMYLE/'

var_map = ["TREFHT", "TREFHTMX", "PRECT"]

# Two caches: one for ensemble mean, one for full member dimension
_cache_mean = {}
_cache_member = {}

list_per_verif_mean = []
list_per_verif_member = []

for verif_year in range(1968, 2021):
    list_per_lead_mean = []
    list_per_lead_member = []

    for lead_year in range(10):
        init_year = verif_year - lead_year
        fn_CESM = base_dir + f"SMYLE_{init_year-1}-11-01_daily_ensemble.zarr"

        # Open once per init file; cache both mean and member-resolved versions
        if fn_CESM not in _cache_mean:
            ds0 = xr.open_zarr(fn_CESM)[var_map]
            ds0['PRECT'] = ds0['PRECT'] * 60 * 60 * 24 * 1000  # mm/day
            _cache_member[fn_CESM] = ds0
            _cache_mean[fn_CESM] = ds0.mean("member")

        time_slice = slice(f"{verif_year}-01-01", f"{verif_year}-12-31")

        # ----- ensemble mean -----
        ds_CESM_mean = _cache_mean[fn_CESM].sel(time=time_slice)
        ds_max_m = ds_CESM_mean[["PRECT", "TREFHTMX"]].max("time", skipna=True).rename(
            {"PRECT": "PRECT_max", "TREFHTMX": "TREFHTMX_max"}
        )
        ds_mean_m = ds_CESM_mean[["PRECT", "TREFHT"]].mean("time", skipna=True).rename(
            {"PRECT": "PRECT_mean", "TREFHT": "TREFHT_mean"}
        )
        ds_merge_m = xr.merge([ds_max_m, ds_mean_m]).expand_dims(lead_year=[lead_year])
        list_per_lead_mean.append(ds_merge_m)

        # ----- per member -----
        ds_CESM_mem = _cache_member[fn_CESM].sel(time=time_slice)
        ds_max_e = ds_CESM_mem[["PRECT", "TREFHTMX"]].max("time", skipna=True).rename(
            {"PRECT": "PRECT_max", "TREFHTMX": "TREFHTMX_max"}
        )
        ds_mean_e = ds_CESM_mem[["PRECT", "TREFHT"]].mean("time", skipna=True).rename(
            {"PRECT": "PRECT_mean", "TREFHT": "TREFHT_mean"}
        )
        ds_merge_e = xr.merge([ds_max_e, ds_mean_e]).expand_dims(lead_year=[lead_year])
        list_per_lead_member.append(ds_merge_e)

    ds_per_verif_mean = xr.concat(list_per_lead_mean, dim="lead_year").expand_dims(valid_year=[verif_year])
    ds_per_verif_member = xr.concat(list_per_lead_member, dim="lead_year").expand_dims(valid_year=[verif_year])

    list_per_verif_mean.append(ds_per_verif_mean)
    list_per_verif_member.append(ds_per_verif_member)

# ---- Assemble raw arrays ----
ds_mean_raw = xr.concat(list_per_verif_mean, dim="valid_year")
ds_member_raw = xr.concat(list_per_verif_member, dim="valid_year")

# Rechunk for the regression step (small along valid_year, spatial chunks intact)
ds_mean_raw = ds_mean_raw.chunk({"valid_year": -1, "lead_year": -1, "lat": 192, "lon": 288})
ds_member_raw = ds_member_raw.chunk({"valid_year": -1, "lead_year": -1, "member": -1, "lat": 192, "lon": 288})


# ---- Linear detrending ----
# Strategy: fit slope/intercept on the ENSEMBLE MEAN along valid_year for each lead_year.
# Apply the same (slope, intercept) to (a) the ensemble mean and (b) every member.
# This preserves ensemble spread (common-trend principle).

def fit_trend(da, dim="valid_year"):
    """
    Fit y = slope * x + intercept along `dim` using xarray's polyfit.
    Returns slope, intercept with same non-`dim` dims as da.
    """
    coeffs = da.polyfit(dim=dim, deg=1, skipna=True)["polyfit_coefficients"]
    # polyfit returns degree=1 (slope) and degree=0 (intercept) along 'degree' dim
    slope = coeffs.sel(degree=1).drop_vars("degree")
    intercept = coeffs.sel(degree=0).drop_vars("degree")
    return slope, intercept


def apply_trend(da, slope, intercept, dim="valid_year"):
    """Reconstruct the fitted trend line along `dim`."""
    x = da[dim].astype("float64")
    return slope * x + intercept


# Fit per variable, per lead_year, on the ensemble mean
varnames = ["PRECT_max", "TREFHTMX_max", "PRECT_mean", "TREFHT_mean"]

slopes = {}
intercepts = {}
for v in varnames:
    s, b = fit_trend(ds_mean_raw[v], dim="valid_year")
    slopes[v] = s          # dims: (lead_year, lat, lon)
    intercepts[v] = b      # dims: (lead_year, lat, lon)

# Build detrended ensemble mean
ds_mean_detrended = xr.Dataset()
for v in varnames:
    trend_line = apply_trend(ds_mean_raw[v], slopes[v], intercepts[v])
    ds_mean_detrended[v] = ds_mean_raw[v] - trend_line

# Build detrended per-member (same slope/intercept, broadcast across member)
ds_member_detrended = xr.Dataset()
for v in varnames:
    trend_line = apply_trend(ds_member_raw[v], slopes[v], intercepts[v])
    # trend_line has no 'member' dim; xarray broadcasts automatically
    ds_member_detrended[v] = ds_member_raw[v] - trend_line


# ---- Pack everything into a single ds_all ----
# Naming convention:
#   <var>_mean_raw         ensemble mean, raw
#   <var>_mean_detrended   ensemble mean, detrended
#   <var>_member_raw       per-member, raw
#   <var>_member_detrended per-member, detrended
#   <var>_slope            linear trend slope (per year units)
#   <var>_intercept        linear trend intercept

ds_all = xr.Dataset()
for v in varnames:
    ds_all[f"{v}_mean_raw"] = ds_mean_raw[v]
    ds_all[f"{v}_mean_detrended"] = ds_mean_detrended[v]
    ds_all[f"{v}_member_raw"] = ds_member_raw[v]
    ds_all[f"{v}_member_detrended"] = ds_member_detrended[v]
    ds_all[f"{v}_slope"] = slopes[v]
    ds_all[f"{v}_intercept"] = intercepts[v]

# Attributes for documentation
ds_all.attrs["detrend_method"] = (
    "Linear regression on ensemble mean across valid_year, "
    "per lead_year and gridpoint. Same slope/intercept applied to "
    "ensemble mean and every member (common-trend principle)."
)
ds_all.attrs["valid_year_range"] = "1968-2020"
ds_all.attrs["lead_year_range"] = "0-9"
ds_all.attrs["units_PRECT"] = "mm/day"
ds_all.attrs["units_TREFHT"] = "K"
ds_all.attrs["units_TREFHTMX"] = "K"

# Optional: write to disk
ds_all = ds_all.chunk({'valid_year': 1, 'lead_year': 1, 'lat': -1, 'lon': -1, 'member': -1,})

compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

# Build per-variable encoding based on each variable's actual dimensions
dict_encoding = {}

for var in ds_all.data_vars:
    
    dims = ds_all[var].dims
    chunks = []
    
    for d in dims:
        size = ds_all[var].sizes[d]
        if d in ("valid_year", "lead_year"):
            chunks.append(1)
        else:  # lat, lon, member, degree, etc.
            chunks.append(size)
            
    dict_encoding[var] = {"compressor": compressor, "chunks": tuple(chunks),}

ds_all.to_zarr(save_dir + "SMYLE_Annual_Metrics.zarr", mode="w", encoding=dict_encoding, consolidated=True,)





