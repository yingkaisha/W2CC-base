# Repository for the ESTCP W2CC Project

Analysis code for the ESTCP **Weather-to-Climate Continuum (W2CC)** project. The work has two
halves:

1. **Predictability analysis** — how much skill the CESM2 SMYLE / Decadal-Prediction (DP) system
   has at annual-to-decadal lead times, globally and at five DoD installations.
2. **Statistical post-processing** — improving that skill with EMOS, BMA, quantile mapping, and
   Analog Ensemble (AnEn), again both globally and at the five installations.

* Notes: [Google Doc](https://docs.google.com/document/d/1BDNNlFBagJSeBX4Juo0ELJ-_Ii5h5F4TcxALod5HGBU/edit?usp=sharing)
* ESTCP Metrics: [epri_statistical_forecasts_metrics.xlsx](https://github.com/yingkaisha/W2CC-base/blob/main/epri_statistical_forecasts_metrics.xlsx)

---

## Contents

- [Experiment setup](#experiment-setup)
- [Repository layout](#repository-layout)
- [`libs/` — shared Python modules](#libs--shared-python-modules)
- [`verification/` — predictability analysis](#verification--predictability-analysis)
- [`postprocess/` — statistical post-processing](#postprocess--statistical-post-processing)
- [`visualization/` — summary plots](#visualization--summary-plots)
- [`qsub/`, `figures/`, and top-level files](#qsub-figures-and-top-level-files)
- [Data on disk](#data-on-disk)
- [Running the workflows](#running-the-workflows)
- [File-naming conventions](#file-naming-conventions)

---

## Experiment setup

**Forecast system.** CESM2 `b.e21.BSMYLE.f09_g17`, initialized on **1 November** of each year from
1958 to ~2019, 20 ensemble members (`011`–`030`), daily output on the f09 grid
(**192 lat × 288 lon**). Each initialization is stitched from two archives:

| Segment | Source | Time span |
|---|---|---|
| Years 0–2 | `/glade/campaign/cesm/development/espwg/SMYLE/archive/` | Nov(*y*) → Oct(*y*+2) |
| Years 2–10 | `/glade/campaign/cesm/development/espwg/CESM2-DP/timeseries/` | Nov(*y*+2) → Dec(*y*+10) |

Merged into one Zarr store per initialization year. On a `noleap` calendar, each forecast is
**3650 days = 10 × 365**, indexed as `lead_time` 0–3649 or aggregated to `lead_year` 0–9.

**Variables carried through the workflow:** `PSL`, `TREFHT`, `TREFHTMN`, `TREFHTMX`, `QREFHT`,
`PRECT`, `PRECSC`, `PRECSL`, `TMQ`, `FLDS`, `FSDS`, `U10`, `Z500`, plus `SST` from the ocean
component.

**Reference / verification target.** ARCO ERA5
(`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`), conservatively regridded
to the CESM f09 grid with xESMF, or extracted at the nearest grid point for station work.

**Five ESTCP installations** (lat, lon), used consistently across the whole repo:

| Key | Site | Latitude | Longitude |
|---|---|---|---|
| `Pituffik` | Pituffik | 76.400 | −68.575 |
| `Fairbanks` | Fairbanks | 64.750 | −147.400 |
| `Guam` | Guam | 13.475 | 144.750 |
| `Yuma_PG` | Yuma | 33.125 | −114.125 |
| `Fort_Bragg` | Fort Bragg | 35.050 | −79.115 |

**Site-specific metrics.** Beyond generic annual mean/min/max, each site has metrics tied to SPEI (3/9/24/48-month lags) for drought at Guam, Fort Bragg, and Yuma; multi-day maximum
precipitation for Fort Bragg and Yuma; extreme-precipitation counts (`days_above_999p`) at Yuma; melting degree days (`MDD`) and freeze–thaw day counts (`FT_days`) at Pituffik; and 1 h/1 d/7 d/30 d
temperature extremes at Fairbanks.

---

## Repository layout

```
W2CC-base/
├── libs/              shared Python modules (imported via sys.path from notebooks)
├── verification/      predictability analysis of raw CESM2-SMYLE
│   ├── scripts/       PBS-launched batch jobs (data packing, metrics, SPEI)
│   ├── global_verif/  global-scale gathering, metrics, ACC, plots
│   ├── mean_grid/     regional boxes around each installation
│   ├── mean_stn/      single-grid-point (station) time series
│   └── qsub_*.ipynb   PBS script generators
├── postprocess/       EMOS / BMA / QM / AnEn calibration
│   ├── data_prep/     station + member metric assembly for post-processing
│   ├── global_pp/     global gridded post-processing
│   │   ├── emos_global/{*.ipynb, scripts/}
│   │   └── bma_global/{*.ipynb, scripts/}
│   └── STN_*.ipynb    station-scale post-processing and verification
├── visualization/     final figure notebooks
├── figures/           PNG output (git-ignored)
├── qsub/              generated PBS job scripts + logs (git-ignored)
├── TEST.ipynb         scratch notebook for inspecting Zarr stores
├── LICENSE            MIT, © 2026 Yingkai (Kyle) Sha
└── README.md
```

Notebooks import the shared modules with `sys.path.insert(0, os.path.realpath('../libs/'))`, so
they expect to be run from their own directory (one level below the repo root — deeper notebooks
such as those in `global_pp/*/` therefore need the path adjusted).

---

## `libs/` — shared Python modules

| Module | Purpose |
|---|---|
| `graph_utils.py` | Plot helpers: `lg_box`/`lg_clean` (legend styling), `precip_cmap` (NCL-style precip colormap), `ax_decorate`, `ax_decorate_box`, `cmap_combine`, `string_partial_format` (multi-color inline text), `ksha_color_set_summon`, `xcolor`. Imported as `gu` in nearly every plotting notebook. |
| `verif_utils.py` | Verification plumbing: `create_dir`, `lead_to_index`, `get_forward_data` / `get_forward_data_netCDF4`, `accum_6h_24h`, `get_nc_files`, `ds_subset_everything`, `process_file_group`, `process_file_group_safe`. |
| `score_utils.py` | `bootstrap_confidence_intervals` (+ numba `bootstrap_core`) and `zonal_energy_spectrum_sph` (spherical-harmonic spectra via `pyshtools`). |
| `seeps_utils.py` | SEEPS precipitation score: `SEEPSThreshold` (heavy/light thresholds + dry fraction), `Region`, `Metric`, `SpatialSEEPS`. |
| `interp_utils.py` | Regridding on `xarray.Dataset`s, adapted from WeatherBench2: `Grid`, `Regridder`, and `NearestRegridder` / `BilinearRegridder` / `ConservativeRegridder`. |
| `physics_utils.py` | Physical diagnostics: `grid_area`, `pressure_integral`, `compute_divergence`, `weighted_sum`. |
| `solar_utils.py` | Solar forcing via `pvlib`: `era5_tsi_data`, `get_tsi`, `get_toa_radiation`, `get_solar_radiation_loc`, `get_solar_index`. |
| `preprocess_utils.py` | ERA5 preprocessing: `get_forward_data`, `zscore_var`, `residual_zscore_var` (pooled mean/variance across yearly files). |
| `plevel_utils.py` | Numba-jitted hybrid-σ → pressure-level interpolation: `interp1d_extrap`, `interp_WRF_to_plevels`, `interp_GP_to_plevels`, `interp_T_to_plevels`. |
| `plevel_utils-Copy1.py` | Variant of the above with the newer names `interp_hybrid_to_pressure_levels`, `interp_geopotential_to_pressure_levels`, `interp_temperature_to_pressure_levels`. |
| `plevel_utils_old.py` | Older copy of `plevel_utils.py`. |

The three `plevel_utils*` files are near-duplicates kept for provenance; `plevel_utils.py` is the
one imported. Nothing in the current W2CC workflow uses pressure-level interpolation.

---

## `verification/` — predictability analysis

Skill assessment of the **raw** (uncalibrated) SMYLE/DP ensemble against ERA5.

### `verification/qsub_*.ipynb` — PBS job generators

These write `.sh` files into `qsub/` that run the batch scripts in `verification/scripts/`. Edit
the year/index range in the notebook, execute, then submit the generated scripts.

| Notebook | Generates jobs for |
|---|---|
| `qsub_DATA00.ipynb` | CESM-DP packing and SMYLE subsetting (older paths under `EPRI-prep/`). |
| `qsub_GRID.ipynb` | Global packing (`SMYLE.sh`), ARCO ERA5 gathering, CESM2 regional-grid subsetting. |
| `qsub_STN.ipynb` | Station extraction — CESM ensemble mean, CESM per-member, ERA5 hourly, ERA5 metrics. |

### `verification/scripts/` — batch jobs

Every script takes one CLI argument (`year` or `ind_lat`) so a year range or the 192 latitude bands
can be fanned out across independent PBS jobs.

| Script | Arg | What it does | Output |
|---|---|---|---|
| `GLOBE_00_FULL_packing.py` | `year` | Merges the 20 SMYLE + DP members into a single 10-year daily global Zarr (zstd-compressed, chunked `member×time×lat×lon`). | `EPRI_data/CESM2_SMYLE/SMYLE_{year}-11-01_daily_ensemble.zarr` |
| `GLOBE_00_gather_CESM_OCN.py` | `year` | Same, for ocean `SST`. | `EPRI_data/CESM2_SMYLE_OCN/SMYLE_{year}-11-01_daily_ensemble.zarr` |
| `GLOBE_00_gather_ERA5.py` | `year` | Pulls 13 ARCO ERA5 variables (incl. 500 hPa geopotential), regrids to the CESM grid, writes one year. | `EPRI_data/ERA5_grid/ERA5_{year}.zarr` |
| `GLOBE_01_CESM_metrics.py` | — | Annual global metrics (`PRECT_max/mean`, `TREFHTMX_max`, `TREFHT_mean`) per `valid_year` × `lead_year`, for both the ensemble mean and the member-resolved ensemble. Unit-converts `PRECT` to mm/day. | `EPRI_data/METRICS_GLOBE/SMYLE_Annual_Metrics.zarr` |
| `GLOBE_01_CESM_metrics_old.py` | — | Earlier ensemble-mean-only version of the above. |  |
| `GLOBE_01_SPEI_CESM.py` | `ind_lat` | Global SPEI from CESM (`xclim` water budget + standardized index), one latitude band per job, looped over all 10 lead years × 20 members. | `EPRI_data/CESM2_grid/SPEI/CESM_SPEI_lat{i}_lead{l}_mem{n}.npy` |
| `GLOBE_01_SPEI_ERA5.py` | `ind_lat` | Same for ERA5. | `EPRI_data/ERA5_grid/SPEI/ERA5_SPEI_lat{i}` |
| `GRID_00_CESM2.py` | `year` | Cuts a ±10-grid-point box around each of the five sites out of the global SMYLE Zarr. | `SMYLE_{site}_{year}.zarr` |
| `GRID_{Fort_Bragg,Guam,Yuma}_SPEI.py` | `ind_lat` | Site-domain SPEI, including interpolation over gaps (`LinearNDInterpolator`/`NearestNDInterpolator`). | `EPRI_data/METRICS/{site}/temp_np/SPEI_{ind_lat}.npy` |
| `STN_00_CESM.py` | `year` | Nearest-grid-point extraction of 12 variables at the five sites, ensemble mean. | `EPRI_data/CESM_SMYLE_STN/{site}_{year}.zarr` |
| `STN_00_CESM_MEM.py` | `year` | Same, keeping the `member` dimension. | `EPRI_data/CESM_SMYLE_STN_MEMBER/{site}_{year}.zarr` |
| `STN_00_ERA5_hourly.py` | `year` | Hourly ARCO ERA5 at the five sites. | `EPRI_data/ERA5_hourly/{site}_{year}.zarr` |
| `STN_01_ERA5_metrics.py` | `ind` (site index 0–4) | Annual ERA5 station metrics with day-of-year-wise linear detrending (`detrend_linear_doy`). | `EPRI_data/METRICS_STN/{site}/metrics.zarr` |
| `x_CESM_Global_Verif_var.py` | `year` | Deprecated per-year global verification with grid-point detrending. |  |
| `old/CESM_00_DP_packing_old.py` | `year` | Superseded by `GLOBE_00_FULL_packing.py` (DP only, no SMYLE segment). |  |
| `old/CESM_01_stn_subset.py` | `year` | Superseded box-subset around stations. |  |
| `old/CESM_02_stn_clim.py` | `year` | Daily climatology by forecast year on a circular 365-day window, pooled over `(init, member)`. |  |
| `old/CESM_03_stn_anom_clim.py` | `year` | Anomaly version of the above. |  |

### `verification/global_verif/` — global skill

| Notebook | Purpose |
|---|---|
| `GLOBE_00_gather_CESM_FULL_packing.ipynb` | Interactive development of `GLOBE_00_FULL_packing.py` — inspects the DP/SMYLE file layout and the merge. |
| `GLOBE_00_gather_CESM_OCN_data.ipynb` | Same, for the ocean (SST) collection. |
| `GLOBE_00_gather_ERA5.ipynb` | Development of the ARCO ERA5 regridding, plus land–sea mask prep into `static/static.zarr`. |
| `GLOBE_01_CESM2_metrics.ipynb` | Interannual and decadal (10-year) global metrics from CESM, raw and detrended. Source of `CESM_minmax*.zarr`. |
| `GLOBE_01_ERA5_metrics.ipynb` | Matching ERA5 metrics (`ERA5_minmax*.zarr`), raw and detrended. |
| `GLOBE_02_ACC.ipynb` | Anomaly correlation coefficient between CESM and ERA5 metrics, with a t-test `corr_pvalue` helper. Covers interannual and decadal, raw and detrended. Writes `CESM_minmax_ACC*.zarr`. |
| `GLOBE_03_PLOT.ipynb` | Global ACC maps by lead year (raw, detrended, and post-processed EMOS/BMA overlays). |
| `GLOBE_03_PLOT_old.ipynb` | Earlier version reading `CESM2_detrend/ACC_annual.zarr`. |
| `GLOBE_04_PLOT_decadal.ipynb` | Decadal-mean ACC maps. |
| `OCN_00_SST_clim.ipynb` | SST climatology from the ocean SMYLE Zarr stores. |
| `OCN_01_ENSO_signal.ipynb` | Niño-region ENSO index from CESM SST; saved as `CESM_OCN/ENSO_index.npy` and later used as an AnEn predictor. |

### `verification/mean_grid/` — regional boxes around each installation

A repeating three-stage pattern per site: `_CESM` (compute CESM metrics) → `_ERA5` (matching ERA5
metrics) → `_ACC` (skill scores), then a `GRID_03_*_PLOT` notebook for maps.

| Notebook(s) | Purpose |
|---|---|
| `GRID_00_Domain_size.ipynb` | Defines and plots the analysis box around each site on the CESM grid. |
| `GRID_01_CESM2.ipynb` | Subsets the global SMYLE Zarr to the five regional boxes. |
| `GRID_01_ERA5.ipynb` | Same for ERA5, with xESMF regridding onto the box grids. |
| `GRID_02_Fairbanks_{CESM,ERA5,ACC}.ipynb` | TMAX-focused metrics and skill. |
| `GRID_02_Fort_Bragg_{CESM,ERA5,ACC}.ipynb` | Max total precipitation (raw + detrended) and SPEI. |
| `GRID_02_Guam_{CESM,ERA5,ACC}.ipynb` | SPEI. |
| `GRID_02_Pituffik_{CESM,ERA5,ACC}.ipynb` | Melting degree days and freeze–thaw days; the `_ACC` notebook adds a SEDI score (`_sedi_from_hits_falsealarms`) for these binary-style events. |
| `GRID_02_Yuma_{CESM,ERA5,ACC}.ipynb` | Max total precipitation, extreme-precip counts, and SPEI. |
| `GRID_03_{site}_PLOT.ipynb` | Cartopy maps of the site metrics, with `smooth_preserve_scale` (variance-preserving smoothing) and shared geo-decoration. |
| `GRID_04_GBI.ipynb` | Greenland Blocking Index from CESM Z500 → `METRICS/GBI.zarr` (a Pituffik predictor). |
| `GRID_04_NAO_CESM.ipynb` | NAO via rotated EOF (`xeofs`) on CESM, daily and monthly → `NAO_CESM_REOF.zarr`, `NAO_mon.zarr`. |
| `GRID_04_NAO_ERA5.ipynb` | Same for ERA5; pickles the fitted EOF model. |
| `GRID_04_NAO_verif.ipynb` | Verifies CESM NAO against ERA5, including lag-1 autocorrelation and significance testing. |
| `eof_model.pkl` | Pickled EOF/REOF model saved by the NAO notebooks so CESM is projected onto the ERA5 modes. |

### `verification/mean_stn/` — single-grid-point station series

| Notebook | Purpose |
|---|---|
| `STN_00_CESM2.ipynb` | Nearest-grid-point CESM extraction at the five sites (interactive twin of `STN_00_CESM.py`). |
| `STN_00_ERA5_gather_data.ipynb` | Collects hourly ARCO ERA5 at the sites; also documents the AnEn input format. |
| `STN_00_ERA5_hourly_to_daily.ipynb` | Hourly → daily aggregation, renaming ERA5 variables to CESM names (`2m_temperature` → `TREFHT`, etc.). |
| `STN_01_CESM2_daily_metrics.ipynb` | Annual CESM station metrics, with and without detrending; `time_to_lead_and_stack` converts calendar time to lead time and stacks initializations. |
| `STN_01_ERA5_daily_metrics.ipynb` | Matching ERA5 annual metrics (`annual_metrics`: yearly min/max/mean plus 30-day rolling-mean max). |
| `STN_02_AnEn_data_prep.ipynb` | Assembles the AnEn input files in `(site, lead_time, gen_date)` layout, merging GBI and ENSO predictors → `EPRI_AnEn/input_AnEn_{CESM,ERA5}_*.nc`. |
| `STN_03_debug.ipynb` | QC: cross-checks station extractions against the corresponding grid-box values. |
| `WORKFLOW.ipynb` | End-to-end run-through of the station pipeline in one notebook. |
| `x_STN_02_CESM2_SPEI.ipynb` | Deprecated station-level SPEI (superseded by the `data_prep` metrics notebooks). |

---

## `postprocess/` — statistical post-processing

Calibration of the raw ensemble. Four methods appear throughout:

- **EMOS** — nonhomogeneous Gaussian regression: `μ = a + b·x̄`, `σ² = c² + d²·s²ₓ`, with the four
  parameters fitted per `(variable, lead)` by minimizing closed-form CRPS (L-BFGS-B on standardized
  predictors). A `_Trends` variant adds a linear time term centered on `T_CENTER`.
- **BMA** — Bayesian Model Averaging, EM-fitted member weights with a Gaussian kernel.
- **QM / NQT** — quantile mapping and normal quantile transform, used for precipitation so that
  Gaussian machinery can be applied in transformed space and inverted afterwards.
- **CSGD / lognormal** — censored shifted gamma and lognormal predictive distributions, the
  direct-precipitation alternatives to QM.

Calibrated distributions are turned into ensembles by **dressing** (drawing `N_DRESS = 50` samples
from the predictive distribution) so that CRPS and ACC can be computed the same way for raw and
post-processed forecasts.

### `postprocess/data_prep/` — inputs for station post-processing

| Notebook | Purpose |
|---|---|
| `STN_00_CESM2.ipynb` | Station extraction with the `member` dimension retained → `CESM_SMYLE_STN_MEMBER/`. |
| `STN_01_CESM_ERA5_simple_metrics.ipynb` | "Simple" annual metrics (mean / min / max / 30-day) for both CESM and ERA5, plus the ENSO index. |
| `STN_02_CESM_MEMBER_simple_metrics.ipynb` | Member-resolved version, with a QC section; writes the AnEn baseline NetCDF. |
| `STN_03_CESM_ESTCP_metrics.ipynb` | The mission-relevant ESTCP metrics from CESM (SPEI, multi-day precip maxima, MDD, freeze–thaw, extreme counts) → `METRICS/{site}/CESM_STN.zarr`. |
| `STN_03_ERA5_ESTCP_metrics.ipynb` | The same metrics from ERA5 → `METRICS/{site}/ERA5_STN.zarr`. |
| `STN_04_CESM_add_all_metrics.ipynb` | Renames and merges every site's simple + ESTCP metrics into the two master files consumed by all post-processing: `METRICS/STN_CESM_ALL_20260604.zarr` and `METRICS/STN_ERA5_ALL_20260604.zarr`. |

### `postprocess/STN_*.ipynb` — station-scale post-processing

Each `STN_00_*` notebook trains on 1959–1999, applies to 2000–2020, and writes a calibrated
distribution (`Calib*.zarr`, `_mu` / `_sigma` per variable) and a dressed 50-member ensemble
(`Dress*.zarr`) under `EPRI_data/PP_calib/{METHOD}/`.

| Notebook | Method | Outputs |
|---|---|---|
| `STN_00_EMOS_Gaussian_Vars.ipynb` | 4-parameter Gaussian EMOS on the ~30 approximately-Gaussian variables. | `EMOS/Calib_4param.zarr`, `EMOS/Dress_4param.zarr` |
| `STN_00_EMOS_Gaussian_Vars_Trends.ipynb` | Same plus a linear trend term. | `EMOS/Calib_Trend.zarr`, `EMOS/Dress_Trend.zarr` |
| `STN_00_EMOS_QM_Vars.ipynb` | NQT → EMOS in Gaussian space → inverse transform, for precipitation-like variables. | `EMOS/Calib_QM.zarr`, `EMOS/Dress_QM.zarr` |
| `STN_00_BMA_Gaussian_Vars.ipynb` | Gaussian BMA. | `BMA/Calib.zarr`, `BMA/Dress.zarr` |
| `STN_00_BMA_QM_Vars.ipynb` | NQT + Gaussian BMA for precipitation. | `BMA/Calib_QM.zarr`, `BMA/Dress_QM.zarr` |
| `STN_00_AnEn_Vars.ipynb` | Analog Ensemble on the SMYLE ensemble-mean predictors; the analog count *K* is chosen per `(variable, lead)` by LOOCV fair-CRPS over `K ∈ {5, 10, 20}`. | `AnEn/Dress.zarr`, `K_opt` |
| `STN_01_EMOS_PP_esnemble_verif.ipynb` | Merges the three EMOS products, verifies against ERA5 (2000–2025), and saves the per-variable figures in `figures/`. | figures |
| `STN_01_BMA_PP_esnemble_verif.ipynb` | Same for BMA. | `BMA/verif_scores.zarr` |
| `STN_01_AnEn_PP_esnemble_verif.ipynb` | Same for AnEn. | `AnEn/verif_scores.zarr` |
| `PLOT_Scores.ipynb` | Side-by-side EMOS / BMA / AnEn score comparison from the three `verif_scores.zarr` files. | figures |
| `x_STN_BMA_Gaussian_all.ipynb` | Deprecated all-in-one BMA notebook (BMA core functions, NQT, dressing) built on the older `EPRI_AnEn/*.nc` inputs. |  |
| `x_STN_EMOS_QM_Vars_Trends.ipynb` | Deprecated NQT-EMOS-with-trend experiment. |  |

Scores reported throughout: **CRPS** (fair/ensemble and closed-form), **CRPSS** against the raw
ensemble, and **ACC** for the raw and calibrated forecasts.

### `postprocess/global_pp/` — global gridded post-processing

Global fitting is embarrassingly parallel over latitude, so the coefficient scripts take a
`lat_i ∈ [0, 192)` argument and each write `..._lat_ind_{lat_i}.zarr`; the verification stage
concatenates all 192 bands along `lat`. Training uses initializations 1958–2009 and verification
2010–2019.

#### `global_pp/emos_global/`

| Notebook | Purpose |
|---|---|
| `EMOS_Gaussian_workflow.ipynb` | Reference implementation of gridded Gaussian EMOS for `TREFHT`, developed on a single latitude band. |
| `EMOS_Gaussian_workflow_TP.ipynb` | Lognormal EMOS variant for `PRECT`. |
| `EMOS_Gaussian_workflow_QM.ipynb` | EMOS applied after quantile mapping, for `PRECT`. |
| `EMOS_CSGD_workflow.ipynb` | Censored shifted gamma distribution EMOS for precipitation, after Scheuerer & Hamill (2015), *Mon. Wea. Rev.* **143**, 4578–4596. |
| `Quantile_Mapping_TP_opt.ipynb` | Builds the CESM and ERA5 quantile tables (200 levels, ±15-day day-of-year pooling) and the mapped training arrays. |
| `Quantile_Mapping_TP_infer.ipynb` | Applies those tables to the verification years. |
| `Dressing_EMOS_Gaussian_TP.ipynb` | Draws dressed ensembles from the lognormal EMOS predictive distributions. |
| `Dressing_EMOS_Gaussian_QM.ipynb` | Same for the QM pipeline, with upper-tail extrapolation past the 0.9975 quantile. |
| `VERIF_EMOS_Gaussian_{T2,TMAX,TMIN}.ipynb` | Assemble the 192 coefficient bands, generate calibrated forecasts, score CRPS, write `PP_calib/{VAR}_EMOS_calib.zarr` and `PP_verif/{VAR}_EMOS_2010_2019.zarr`. |
| `VERIF_EMOS_Gaussian_TP_lognorm.ipynb` | Precipitation verification, lognormal route. |
| `VERIF_EMOS_Gaussian_TP_QM.ipynb` | Precipitation verification, QM route. |
| `VERIF_EMOS_Gaussian_TP_QM_inverse.ipynb` | Inverse-transforms QM-space output back to mm/day. |
| `EMOS_CSGD_verif.ipynb` | CSGD verification. |
| `PLOT_EMOS_{T2,TMAX,TMIN,TP_QM}.ipynb` | Global maps of raw-vs-EMOS skill per variable. |
| `VERIF_GLOBE_ACC.ipynb` | ACC comparison of raw CESM, EMOS, and BMA at interannual and decadal scales, raw and detrended. |
| `qsub.ipynb` | Generates the 192-job PBS arrays for EMOS coefficients, quantile mapping, CRPS, and CSGD. |

| Script | Arg | Purpose |
|---|---|---|
| `EMOS_coef_{T2,TMAX,TMIN}.py` | `lat_i` | Fit Gaussian EMOS coefficients for `TREFHT` / `TREFHTMX` / `TREFHTMN`. |
| `EMOS_coef_TP.py` | `lat_i` | Lognormal EMOS coefficients for `PRECT`. |
| `EMOS_coef_QM.py` | `lat_i` | EMOS coefficients on quantile-mapped precipitation. |
| `CSGD_coef.py` | `lat_i` | CSGD parameter fitting for precipitation (moment-matched gamma init, `scipy.optimize.minimize`). |
| `quantile_opt.py` | `lat_i` | Build CESM/ERA5 quantile tables and mapped training arrays. |
| `quantile_infer.py` | `lat_i` | Apply the tables to the verification period. |
| `QM_Calib.py` | — | Assemble the global QM tables and produce calibrated precipitation fields. |
| `Dressing_QM.py` | — | Dressed ensembles from the QM pipeline (with `extrapolate_upper_tail`, `gaussian_to_precip`). |
| `Dressing_Lognorm.py` | — | Dressed ensembles from the lognormal pipeline. |
| `CRPS_TP_QM.py` / `CRPS_TP_lognorm.py` | — | CRPS verification for the two precipitation routes (50-member dressed ensembles). |

Shared configuration in these scripts: `WINDOW` (day-of-year pooling half-width, 0 = fit per lead
time), `MIN_SAMPLES = 20`, `EPS = 1e-6` (variance floor), unit conversions `PRECT` m/s → mm/day
(`×86400×1000`) and ERA5 `total_precipitation` m/day → mm/day (`×1000`).

#### `global_pp/bma_global/`

| Notebook | Purpose |
|---|---|
| `BMA_Gaussian_workflow.ipynb` | Reference gridded Gaussian BMA implementation (EM loop) for `TREFHT`. |
| `BMA_Gaussian_workflow_QM.ipynb` | BMA on quantile-mapped precipitation. |
| `VERIF_BMA_Gaussian_TMAX.ipynb` | Assembles the 192 bands, produces calibrated forecasts, scores CRPS → `PP_calib/T2_BMA_calib.zarr`, `PP_verif/T2_BMA_2010_2019.zarr`. |
| `PLOT_BMA_T2.ipynb` | Global BMA-vs-EMOS skill maps for 2-m temperature. |
| `qsub.ipynb` | Generates the 192-job PBS arrays for the BMA scripts. |

| Script | Arg | Purpose |
|---|---|---|
| `BMA_coef_{T2,TMAX,TMIN}.py` | `lat_i` | BMA weights and variance per latitude band for `TREFHT` / `TREFHTMX` / `TREFHTMN`. |
| `BMA_coef_QM.py` | `lat_i` | BMA on quantile-mapped `PRECT`. |
| `BMA_VERIF_{T2,TMAX,TMIN}.py` | — | Batch versions of the verification notebooks. |

> **Note:** the two `qsub.ipynb` notebooks still write `scripts_loc` as
> `/glade/u/home/ksha/W2CC-base/{EMOS,BMA}/scripts/`. Those directories were reorganized into
> `postprocess/global_pp/{emos_global,bma_global}/scripts/`, so update the path before regenerating
> job scripts.

---

## `visualization/` — summary plots

| Notebook | Purpose |
|---|---|
| `PLOT_Annual_Metrics_raw.ipynb` | Global maps of raw annual-metric skill by lead year, from `METRICS_GLOBE/SMYLE_Verif.zarr`. |
| `PLOT_Annual_Metrics_detrend.ipynb` | Same for detrended metrics. |
| `PLOT_STN_verif.ipynb` | Station verification figure set — raw vs. EMOS vs. BMA per variable; writes `figures/{variable}.png`. |
| `x_PLOT_Annual_Metrics_Old.ipynb` | Deprecated earlier version of the annual-metric figures. |

---

## `qsub`, `figures`, and top-level files

- **`qsub/`** — generated PBS scripts and their `.log` / `.err` output. `SMYLE.sh` is a
  representative job: casper queue, project `P48500028`, 12 CPUs, 256 GB, 24 h walltime, activates
  the `credit` conda environment and runs a script from `verification/scripts/`. This directory is
  git-ignored; regenerate it from the `qsub_*.ipynb` notebooks.
- **`figures/`** — 63 PNGs written by `PLOT_STN_verif.ipynb` and
  `STN_01_EMOS_PP_esnemble_verif.ipynb`, named `{site}_{metric}.png` (e.g.
  `Fort_Bragg_SPEI_48_min.png`, `Pituffik_MDD.png`). Also git-ignored.
- **`TEST.ipynb`** — scratch notebook that opens one example store from each data collection
  (`CESM2_grid`, `CESM2_SMYLE`, `CESM_SMYLE_STN_MEMBER`, `ERA5_grid`, `METRICS`, `METRICS_STN`,
  `METRICS_STN_MEMBER`) to check dimensions and variable names. Useful as a quick data-layout
  reference.
- **`LICENSE`** — MIT, © 2026 Yingkai (Kyle) Sha.

---

## Data on disk

Nothing large is stored in the repo; everything lives on GLADE scratch/campaign under
`EPRI_data/`.

| Path | Contents |
|---|---|
| `scratch/ksha/EPRI_data/CESM2_SMYLE/` | Packed global daily SMYLE+DP ensembles, one Zarr per initialization year. |
| `scratch/ksha/EPRI_data/CESM2_SMYLE_OCN/` | Ocean (SST) equivalents. |
| `scratch/ksha/EPRI_data/ERA5_grid/` | ARCO ERA5 regridded to the CESM grid, one Zarr per year (+ `SPEI/`). |
| `scratch/ksha/EPRI_data/static/static.zarr` | Grid coordinates and land–sea mask. |
| `scratch/ksha/EPRI_data/CESM_SMYLE_STN_MEMBER/`, `campaign/.../CESM_SMYLE_STN/` | Station extractions, with and without the member dimension. |
| `campaign/ral/hap/ksha/EPRI_data/ERA5_{hourly,daily}/` | Station ERA5. |
| `scratch/ksha/EPRI_data/METRICS/` | Per-site metrics and the `STN_{CESM,ERA5}_ALL_20260604.zarr` master files. |
| `scratch/ksha/EPRI_data/METRICS_GLOBE/` | Global annual/decadal metrics and ACC. |
| `scratch/ksha/EPRI_data/{EMOS,BMA,QM,QM_PRED}/` | Per-latitude post-processing coefficients and quantile tables. |
| `scratch/ksha/EPRI_data/PP_calib/`, `PP_verif/` | Calibrated forecasts, dressed ensembles, and verification scores. |
| `scratch/ksha/EPRI_AnEn/` | AnEn input/output NetCDFs. |

Dated filenames (`..._20260604.zarr`) mark regenerated versions; the post-processing notebooks
currently read the `20260604` master files.

---

## Running the workflows

Everything runs on Derecho/Casper under the `credit` conda environment. The general order:

**Predictability (raw skill)**

1. `verification/qsub_GRID.ipynb` → submit → `GLOBE_00_FULL_packing.py`, `GLOBE_00_gather_ERA5.py`
   (packs CESM and ERA5 into yearly Zarr stores).
2. `verification/qsub_STN.ipynb` → submit → `STN_00_CESM*.py`, `STN_00_ERA5_hourly.py`
   (station extractions).
3. `global_verif/GLOBE_01_*` and `GLOBE_02_ACC.ipynb` for global metrics and skill;
   `mean_grid/GRID_02_*` and `mean_stn/STN_01_*` for regional and station metrics.
4. `GLOBE_03_PLOT.ipynb`, `GRID_03_*_PLOT.ipynb`, `visualization/PLOT_Annual_Metrics_*.ipynb`.

**Post-processing**

1. `postprocess/data_prep/STN_00` → `STN_04` to build the two master metric files.
2. Station: `postprocess/STN_00_{EMOS,BMA,AnEn}_*.ipynb` → `STN_01_*_verif.ipynb` →
   `PLOT_Scores.ipynb`.
3. Global: `global_pp/*/qsub.ipynb` → submit the 192 per-latitude coefficient jobs →
   `VERIF_*.ipynb` → `PLOT_*.ipynb`.

Batch scripts and notebooks are deliberately near-duplicates: prototype interactively in the
notebook, then run the matching script across years or latitudes under PBS.

---

## File-naming conventions

| Pattern | Meaning |
|---|---|
| `NN_` numeric prefix (`GLOBE_00_`, `STN_01_`, `GRID_02_`) | Execution order within a folder. |
| `GLOBE_` / `GRID_` / `STN_` | Global · regional box · single grid point. |
| `x_` prefix | Deprecated or experimental; git-ignored (`x_*` in `.gitignore`) but kept on disk. |
| `_old` suffix | Superseded version retained for reference. |
| `T2` / `TMAX` / `TMIN` / `TP` | `TREFHT` · `TREFHTMX` · `TREFHTMN` · `PRECT` (paired with the ERA5 names `2m_temperature`, `maximum/minimum_2m_temperature_since_previous_post_processing`, `total_precipitation`). |
| `simple_` in a metric name | Generic annual mean/min/max metric, as opposed to a mission-specific ESTCP metric. |
| `_ensmean` suffix | Ensemble-mean predictor (AnEn inputs). |
| `Calib*` / `Dress*` | Calibrated predictive distribution (`_mu`, `_sigma`) vs. dressed 50-member ensemble. |

