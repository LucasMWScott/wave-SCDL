# Feature Normalization Reference

This document answers one question directly: which input and target features
use which normalization method.

Unless noted otherwise, this reflects the current default training path in
[configs/training.yaml](../configs/training.yaml)
and the saved preprocessing metadata in
`data/processed/thesis_set_cnn2/point_centric_metadata.json`.

## Quick Summary

| Feature family | Current method | Fit scope | Where defined |
| --- | --- | --- | --- |
| Offshore dynamic magnitudes | `zscore` | train timesteps only | `normalization.default_method` |
| Offshore dynamic directions | degrees -> `sin/cos`, then no scaler | n/a | `direction_vars` |
| Site-dynamic sidecar magnitudes | `zscore` | train timesteps and train sites only | metadata `site_dynamic_scaler` |
| Site-dynamic sidecar directions | `sin/cos`, then no scaler | n/a | generated features |
| Multi-source dynamic magnitudes | `zscore` | train timesteps and train sites only | metadata `source_dynamic_scaler` |
| Multi-source dynamic directions | `sin/cos`, then no scaler | n/a | generated features |
| Source geometry | already bounded / normalized by construction | n/a | `src/multi_source.py` |
| Static cyclical geometry | degrees -> `sin/cos`, then no scaler | n/a | `src/preprocessing/normalize.py` |
| Static heavy-tailed features | `log1p` -> `MinMaxScaler([0, 1])` | train sites only | static ColumnTransformer |
| Static spatial derivatives | `RobustScaler` | train sites only | static ColumnTransformer |
| Static ratios | `MinMaxScaler([0, 1])` | train sites only | static ColumnTransformer |
| Static other numeric features | median impute -> `MinMaxScaler([0, 1])` | train sites only | static ColumnTransformer |
| Bathy `depth` (v2) | clip to `depth_clip_m`, then divide by train max | train-site wet cells only | `build_bathy_patches.py` |
| Bathy `slope_magnitude` / `distance_to_land` / `curvature_laplacian` | robust scaling with median and `(p95 - p05)` | train-site wet cells only | `build_bathy_patches.py` |
| Bathy masks | no scaler | n/a | `build_bathy_patches.py` |
| Direct targets | `StandardScaler` on all six stored channels | train rows and train sites only | metadata `target_scaler` |
| Transfer targets | `StandardScaler` on `log_hs_ratio`, `tp_delta` only | train rows and train sites only | metadata `transfer_target_scaler` |

## Dynamic Inputs

### 1. Legacy offshore sequence: `X_dynamic`

For each offshore point, the current preprocessing keeps these magnitude
features and applies the global dynamic method from
`configs/training.yaml -> normalization.default_method`:

- `hs`
- `tp`
- `tm1`
- `tm2`
- `tmp`
- `hs_sea`
- `tp_sea`
- `hs_swell`
- `tp_swell`
- `wind_speed_10m`

With the current config, that method is `zscore`.

These direction features are converted from degrees to paired `sin/cos`
channels and are then excluded from dynamic scaling:

- `Pdir`
- `thq`
- `thq_sea`
- `thq_swell`
- `wind_direction_10m`

So the stored features look like:

- `Pdir_sin`, `Pdir_cos`
- `thq_sin`, `thq_cos`
- `thq_sea_sin`, `thq_sea_cos`
- `thq_swell_sin`, `thq_swell_cos`
- `wind_direction_10m_sin`, `wind_direction_10m_cos`

In the saved metadata this appears under:

- `normalization.dynamic_scaler.columns`: scaled magnitude columns
- `normalization.dynamic_scaler.skipped_circular_columns`: unscaled `sin/cos`
  columns

### 2. Runtime seasonal channels

At dataset runtime, the loader appends:

- `time_sin`
- `time_cos`

These are already bounded in `[-1, 1]` and are not scaled again.

### 3. Site-dynamic sidecar: `X_dynamic_sitewise`

Current site-dynamic magnitude features are:

- `wave_fetch_aligned_m`
- `wave_fetch_aligned_ratio`
- `wave_slope_aligned`
- `wave_laplacian_aligned`
- `wave_min_depth_aligned_m`
- `wave_blocked_sector_fraction_pm30`
- `wave_open_sector_fraction_pm30`
- `local_wind_fetch_aligned_m`
- `local_wind_fetch_aligned_ratio`
- `local_windsea_proxy_u2_fetch`
- `local_windsea_proxy_u2_fetch_ratio`
- `local_windsea_proxy_3h_mean`
- `local_windsea_proxy_6h_mean`
- `local_windsea_proxy_12h_mean`
- `local_wind_speed_10m`

Current site-dynamic circular features are:

- `local_wind_dir_sin`
- `local_wind_dir_cos`

Only the magnitude features are scaled. With the current config they use
`zscore`.

### 4. Multi-source dynamic tensor: `X_dynamic_sources`

Current scaled magnitude features are:

- `hs`
- `tp`
- `tm1`
- `tm2`
- `tmp`
- `hs_sea`
- `tp_sea`
- `hs_swell`
- `tp_swell`
- `wind_speed_10m`
- `fetch_at_swell_direction_m`
- `blocking_at_swell_direction`
- `slope_at_swell_direction`
- `fetch_at_windwave_direction_m`
- `blocking_at_windwave_direction`
- `slope_at_windwave_direction`
- `fetch_at_local_wind_direction_m`
- `blocking_at_local_wind_direction`
- `slope_at_local_wind_direction`
- `local_wind_speed_10m`

Current unscaled circular features are:

- `Pdir_sin`, `Pdir_cos`
- `thq_sin`, `thq_cos`
- `thq_sea_sin`, `thq_sea_cos`
- `thq_swell_sin`, `thq_swell_cos`
- `wind_direction_10m_sin`, `wind_direction_10m_cos`
- `local_wind_dir_sin`, `local_wind_dir_cos`

### 5. Source geometry tensor: `source_geometry`

These inputs do not use the dynamic scaler:

- `distance_m_norm`: distance divided by `max_distance_km_error * 1000`, then
  clipped to `[0, 1]`
- `bearing_sin`, `bearing_cos`: already bounded circular encoding
- `inverse_distance_weight`: already normalized to sum to 1 across sources
- `source_rank_1`, `source_rank_2`, `source_rank_3`: one-hot indicators

## Static Inputs

Static features are transformed with the strict sklearn `ColumnTransformer`
implemented in
[src/preprocessing/normalize.py](../src/preprocessing/normalize.py).
It is fit on train sites only.

All non-cyclical static groups use median imputation before scaling.

`<sector>` below means one of:
`N`, `NNE`, `NE`, `ENE`, `E`, `ESE`, `SE`, `SSE`, `S`, `SSW`, `SW`, `WSW`,
`W`, `WNW`, `NW`, `NNW`.

### 1. Cyclical: degrees -> `sin/cos`, no scaler

- `static_final_approach_deg`
- `static_net_deflection_deg`
- `static_nearest_shore_normal_deg`
- `static_signed_curvature_deg`

### 2. Heavy-tailed: `log1p` -> `MinMaxScaler([0, 1])`

- `path_length_m`
- `path_bottleneck_m`
- `path_direct_distance_m`
- `ray_fetch_<sector>_m`

### 3. Spatial derivatives: `RobustScaler`

- `static_local_depth_m`
- `static_nearest_shore_steepness`
- `ray_max_slope_<sector>`
- `ray_max_laplacian_<sector>`
- `ray_min_depth_<sector>_m`

### 4. Ratios: `MinMaxScaler([0, 1])`

- `funneling_ratio`
- `choke_out_ratio`
- `static_porosity_500m`
- `static_porosity_1km`
- `static_porosity_2km`
- `static_porosity_5km`
- `static_porosity_10km`
- `fetch_max_over_mean`
- `fetch_std_m`
- `fetch_cv`
- `fetch_directional_entropy`
- `fetch_resultant_length`
- `open_sector_fraction`
- `closed_sector_fraction`
- `open_sector_width_deg`
- `dominant_fetch_direction_sin`
- `dominant_fetch_direction_cos`
- `fjordness_land_blocking_component`
- `fjordness_low_porosity_component`
- `fjordness_closed_sector_component`
- `fjordness_anisotropy_component`
- `fjordness_low_fetch_component`
- `fjordness_route_complexity_component`
- `fjordness_score`
- `site_regime_open`
- `site_regime_transition`
- `site_regime_fjord`
- `path_tortuosity_ratio`
- `bottleneck_to_path_ratio`
- `bottleneck_to_fetch_ratio`
- `funneling_log`

### 5. Other numeric: median impute -> `MinMaxScaler([0, 1])`

- `path_point_count`
- `snap_distance_m`
- `static_tortuosity_sum`
- `static_dist_to_coast_m`
- `local_depth_m`
- `local_breaking_hs_cap`
- `local_breaking_cap_valid`
- `ray_fetch_min_m`
- `ray_fetch_mean_m`
- `ray_fetch_max_m`
- `ray_hit_land_fraction`
- `ray_count`

Important detail:

- `local_depth_m` and `local_breaking_hs_cap` exist twice in practice:
  normalized versions inside the static vector, and raw versions in
  `point_centric_physics.npz` for physics-aware losses.

## Bathymetry Inputs

Bathymetry normalization is implemented in
[src/preprocess/build_bathy_patches.py](../src/preprocess/build_bathy_patches.py)
and is fit on train-site wet cells only.

### Saved v2 artifact channels

The v2 artifact stores six channels:

| Channel | Method |
| --- | --- |
| `depth` | clip to `depth_clip_m`, then divide by train wet-cell max (`unit_interval_train_max`) |
| `land_sea_mask` | no scaler |
| `slope_magnitude` | robust scaling using median and `p95 - p05` |
| `distance_to_land` | robust scaling using median and `p95 - p05` |
| `curvature_laplacian` | robust scaling using median and `p95 - p05` |
| `shallow_breaking_mask` | no scaler |

### Current training config

The current model request in
[configs/training.yaml](../configs/training.yaml)
uses only:

- `depth`
- `land_sea_mask`
- `slope_magnitude`

So those are the only bathy channels consumed by the current default run,
even though the saved artifact contains six.

### Legacy v1 artifact

Older v1 bathy artifacts use:

- `depth_log1p_norm`: `log1p` depth, then standard scaling
- `wet_mask`: no scaler

## Targets

Target scaling uses the helper in
[src/preprocessing/normalize.py](../src/preprocessing/normalize.py)
and is fit on train rows and train sites only.

Non-finite values are replaced with train-column medians before scaling.

### 1. Direct target tensor: `Y_targets`

The stored direct target channels are:

- `hs`
- `tp`
- `dir_sin`
- `dir_cos`
- `dp_sin`
- `dp_cos`

All six use `StandardScaler`.

That means the target directions are treated differently from the input
directions:

- input directions: `sin/cos`, then no scaler
- target directions: `sin/cos`, then `StandardScaler`

### 2. Physical targets: `Yphysical__<site>`

These are stored in raw units / raw degrees:

- `hs`
- `tp`
- `dir`
- `dp`

They are used to build transfer targets and for reconstruction, but the saved
raw physical arrays themselves are not normalized.

### 3. Transfer targets: `Ytransfer__<site>`

Current transfer target columns are:

- `log_hs_ratio`
- `tp_delta`
- `dir_delta_deg`
- `dp_delta_deg`

In the current `transfer_representation=legacy` path:

- `log_hs_ratio`: `StandardScaler`
- `tp_delta`: `StandardScaler`
- `dir_delta_deg`: left unscaled in raw degrees
- `dp_delta_deg`: left unscaled in raw degrees

### 4. Reference targets: `Yreference__<site>`

Reference columns are:

- `ref_hs`
- `ref_tp`
- `ref_dir`
- `ref_dp`

These are stored in raw physical units / raw degrees and are not scaled.

## Where To Verify This In Artifacts

If you want to confirm the exact saved behavior for a dataset, check
`point_centric_metadata.json` under:

- `normalization.dynamic_scaler`
- `normalization.site_dynamic_scaler`
- `normalization.source_dynamic_scaler`
- `normalization.target_scaler`
- `normalization.transfer_target_scaler`
- `normalization.static_scaler`

For bathymetry, check `point_centric_X_bathy.npz -> normalization_metadata`.
