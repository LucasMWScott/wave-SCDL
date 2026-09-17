# Static Feature Computation Methods

## Purpose
This document explains how the site-level static features in `master_static_features.csv` are computed by the current geometry pipeline. It is the implementation-oriented companion to `Static_Features_summary.md`, which remains the quicker feature dictionary.

The descriptions below are based directly on:

- [geometric_builder/src/build_features.py](../geometric_builder/src/build_features.py)
- [geometric_builder/src/fetch_router.py](../geometric_builder/src/fetch_router.py)
- [geometric_builder/src/ray_caster.py](../geometric_builder/src/ray_caster.py)

## Pipeline Overview
The master static table is assembled in four stages:

1. Bathymetry subgrid generation
   - `generate_bathy_field.run_generate_bathy_field(...)` builds a projected local bathymetry grid, typically in `EPSG:32633`.
   - The downstream geometry code expects a structured raster with `x`, `y`, and `z`, and optionally a `land_mask`.
2. Offshore-to-nearshore routing
   - `fetch_router.run_fetch_routing(...)` builds a cost surface over water cells and routes from the offshore curtain to each nearshore site using `skimage.graph.MCP_Geometric`.
   - This stage produces the route-level descriptors such as path length, bottleneck width, funneling, tortuosity, and approach angle.
3. Directional ray casting
   - `ray_caster.run_ray_casting(...)` casts many rays from each snapped nearshore site across the bathymetry grid.
   - This stage produces the per-ray fetch, slope, Laplacian, minimum depth, porosity, shoreline, and local depth descriptors.
4. Site-level merge and derived summaries
   - `build_features.build_master_static_features(...)` aggregates per-ray records to one row per site, merges them with the route table, and adds the derived anisotropy, path-ratio, fjordness, and local-breaking columns.

The final artifact is one row per nearshore site in `master_static_features.csv`.

## Libraries and Numerical Tools
The static-feature pipeline relies on the following libraries and core methods:

- `numpy`: array math, trigonometry, reductions, clipping, `log`, `log1p`, `hypot`
- `pandas`: tabular joins, grouping, aggregation, CSV I/O
- `scipy.ndimage.distance_transform_edt`: metric distance-to-coast and nearest-land lookup
- `scipy.ndimage.map_coordinates`: nearest-neighbor sampling for land interception and bilinear interpolation of raster fields along rays
- `scipy.ndimage.convolve`: Laplacian kernels on the bathymetry grid
- `scipy.spatial.cKDTree`: snapping sites or curtain pixels to the nearest valid water cell
- `skimage.graph.MCP_Geometric`: bottleneck-aware shortest-path routing across the water grid
- `skimage.draw.line_nd`: rasterization of the offshore boundary curtain
- `skimage.draw.disk`: circular neighborhoods for porosity
- `skimage.measure.approximate_polygon`: route smoothing before turn-angle calculations
- `pyproj.Transformer`: `lon/lat -> EPSG:32633` projected coordinates
- `xarray`: reading NetCDF bathymetry grids when the input is not NPZ

## Grid Conventions and Preprocessing

### Projected coordinates and site indexing
- Both routing and ray casting transform the site YAML coordinates from `EPSG:4326` into the configured projected CRS, default `EPSG:32633`.
- Grid coordinates are converted to raster row/column indices by nearest-neighbor lookup with `xy_to_rc(...)`.
- Sites are then snapped to the nearest valid water cell using `cKDTree`.
- Routing snaps to the current `routing_water` mask.
- Ray casting snaps to the broader `water_mask`.

### Land/water inference and vertical convention
- If the bathymetry NPZ includes `land_mask`, that mask is used directly.
- Otherwise `infer_water_land_masks(...)` infers the sign convention from finite `z` values:
  - if more than 90% of finite values are positive, the grid is treated as `positive_down_depth`, with land defined by `z <= 0`
  - otherwise the grid is treated as `signed_elevation`, with land defined by `z >= 0`
- NaNs are always treated as land.
- Positive water depth is obtained by `elevation_to_positive_depth(...)`:
  - `depth = z` for `positive_down_depth`
  - `depth = -z` otherwise
  - finite depths are clipped to `>= 0`

### Raster spacing and smoothing tolerance
- `dx` and `dy` are the median spacings of the projected `x` and `y` coordinates.
- Routing computes:

```text
avg_spacing_m = max(1.0, 0.5 * (|dx| + |dy|))
path_smoothing_tolerance_m = PATH_SMOOTH_TOLERANCE_CELLS * avg_spacing_m
```

- `PATH_SMOOTH_TOLERANCE_CELLS = 1.5`.
- This smoothing affects only the angle-based path features. Path length, path point count, and bottleneck features are computed from the unsmoothed routed path.

### Ray geometry and compass sectors
- Rays are cast at evenly spaced compass angles over `[0, 360)`.
- In the end-to-end builder, the default ray angle step is `10.0` degrees unless overridden on the CLI.
- Standalone `ray_caster.py` uses `5.625` degrees by default, so the number of rays per site depends on how the script is run.
- Sector summary columns always use the fixed 16-sector compass partition:
  `N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW`.
- Each sector therefore spans:

```text
SECTOR_WIDTH_DEG = 360 / 16 = 22.5 degrees
```

## Where Each Column Comes From

| Origin | Columns |
|---|---|
| Metadata / indexing | `site_name`, `site_lat`, `site_lon`, `site_x`, `site_y`, `site_row`, `site_col`, `reachable` |
| Routing-derived | `path_length_m`, `path_point_count`, `snap_distance_m`, `path_bottleneck_m`, `funneling_ratio`, `choke_out_ratio`, `static_tortuosity_sum`, `static_signed_curvature_deg`, `static_net_deflection_deg`, `static_final_approach_deg` |
| Ray-derived, aggregated to site level | `static_porosity_*`, `static_dist_to_coast_m`, `static_local_depth_m`, `local_depth_m`, `local_breaking_cap_valid`, `static_nearest_shore_steepness`, `static_nearest_shore_normal_deg`, `ray_fetch_min_m`, `ray_fetch_mean_m`, `ray_fetch_max_m`, `ray_fetch_{sector}_m`, `ray_max_slope_{sector}`, `ray_max_laplacian_{sector}`, `ray_min_depth_{sector}_m`, `ray_hit_land_fraction`, `ray_count` |
| Derived during site-level merge | `fetch_max_over_mean`, `fetch_std_m`, `fetch_cv`, `fetch_directional_entropy`, `fetch_resultant_length`, `open_sector_fraction`, `closed_sector_fraction`, `open_sector_width_deg`, `dominant_fetch_direction_sin`, `dominant_fetch_direction_cos`, `path_direct_distance_m`, `path_tortuosity_ratio`, `bottleneck_to_path_ratio`, `bottleneck_to_fetch_ratio`, `funneling_log`, `fjordness_*_component`, `fjordness_score`, `site_regime_*`, `local_breaking_hs_cap` |

## Feature Families

### 1. Site metadata and indexing

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `site_name`, `site_lat`, `site_lon` | `fetch_router.load_sites_yaml`, `_route_record_from_solution` | site | Copied from the nearshore site YAML entry. |
| `site_x`, `site_y` | `_route_record_from_solution` | site | Projected coordinates from the transformed nearshore YAML point. These are not the snapped cell-center coordinates used internally by the ray records. |
| `site_row`, `site_col` | `_route_record_from_solution` | site | Raster indices of the snapped nearshore site in the routing grid. |
| `reachable` | `run_fetch_routing`, `_route_record_from_solution` | site | `True` when the MCP cumulative cost at the snapped site is finite. |

### 2. Routing and path features

#### Offshore curtain construction
- `build_curtain(...)` converts the offshore source points to raster indices.
- It connects the two endpoint offshore pixels using `skimage.draw.line_nd`.
- If the offshore points collapse to one grid cell, the code instead chooses the two most separated offshore pixels.
- The orchestrator supports two curtain modes via `static_features.route_curtain.mode`:
  - `global`: one curtain built from the full offshore source set
  - `local_k_nearest`: a site-specific curtain built from the nearshore site's selected multi-source offshore neighbors
- The default is `global`. The `local_k_nearest` mode requires `use_multi_source_config=true` and a compatible enabled multi-source training config so static routing matches the dynamic source selection logic.

#### Offshore ghost-water masking
- `mask_ghost_offshore_water(...)` estimates the offshore curtain tangent using SVD/PCA on the offshore point cloud.
- A normal vector is constructed from that tangent.
- Water cells on the offshore side of the curtain line, opposite the nearshore centroid or site, are marked as ghost water and removed from routing.
- This prevents routes from taking physically unrealistic paths farther offshore than the forcing curtain.

#### Bottleneck-aware routing cost
- `build_dynamic_cost_matrix(...)` computes the water-domain distance transform:

```text
distance_to_coast_m = EDT(water_mask, sampling=(dy, dx))
```

- The routing cost is then:

```text
cost = base_water_cost + bottleneck_weight / (distance_to_coast_m + bottleneck_epsilon_m)
```

- Current defaults are:
  - `base_water_cost = 1.0`
  - `bottleneck_weight = 500.0`
  - `bottleneck_epsilon_m = 1.0`
- Narrow channels have smaller `distance_to_coast_m`, so they incur a larger penalty.
- `MCP_Geometric(..., fully_connected=True)` is used so diagonal moves are allowed with metric sampling in meters.

#### Path extraction, orientation, and smoothing
- `mcp.find_costs(starts=curtain_pixels)` computes cumulative cost from the curtain.
- `mcp.traceback(target_rc)` extracts the least-cost path for each reachable site.
- `orient_path_rc_toward_site(...)` ensures the final vertex is the nearshore site end of the path.
- `rc_path_to_xy(...)` converts row/column vertices to projected `x/y`.
- `smooth_path_xy(...)` simplifies the raster staircase path with `approximate_polygon(...)` while preserving endpoints.
- Angle features are computed from the smoothed path; width and length features use the unsmoothed routed path.

#### Route feature formulas

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `path_length_m` | `path_length_m`, `_route_record_from_solution` | route/site | Sum of Euclidean segment lengths in projected space: `sum_i sqrt((dy_i * dy)^2 + (dx_i * dx)^2)`. |
| `path_point_count` | `_route_record_from_solution` | route/site | Number of routed row/column vertices in the unsmoothed path. |
| `snap_distance_m` | `_route_record_from_solution` | route/site | Distance between the original nearshore raster index and the snapped routing-water index. |
| `path_bottleneck_m` | `compute_path_width_features` | route/site | `min(2 * distance_to_coast_m(p_i))` along the routed path. |
| `funneling_ratio` | `compute_path_width_features` | route/site | `max(path_width_m) / local_site_width_m`, where `path_width_m = 2 * distance_to_coast_m` and `local_site_width_m = 2 * distance_to_coast_m(site)`. |
| `choke_out_ratio` | `compute_path_width_features` | route/site | `path_bottleneck_m / local_site_width_m`. |
| `static_tortuosity_sum` | `compute_path_angle_features` | route/site | Sum of absolute wrapped turn increments on the smoothed path. If segment headings are `theta_i`, then `delta_i = atan2(sin(theta_i - theta_{i-1}), cos(theta_i - theta_{i-1}))`, and `static_tortuosity_sum = sum_i |degrees(delta_i)|`. |
| `static_signed_curvature_deg` | `compute_path_angle_features` | route/site | `sum_i degrees(delta_i)` over the smoothed path. |
| `static_net_deflection_deg` | `compute_path_angle_features` | route/site | Compass bearing from the first to the last smoothed path point. |
| `static_final_approach_deg` | `compute_path_angle_features`, `terminal_segment_start` | route/site | Compass bearing of the last `500 m` of the smoothed path into the site. |

### 3. Porosity, shoreline, and local depth

#### Porosity
- `compute_static_site_features(...)` evaluates circular neighborhoods around the snapped site using `skimage.draw.disk`.
- The radii are fixed at:
  - `500 m`
  - `1 km`
  - `2 km`
  - `5 km`
  - `10 km`
- Each radius is converted to cells with:
- Each radius is converted to cells with:

```text
avg_spacing_m = max(1.0, 0.5 * (|dx| + |dy|))
radius_cells = max(1, round(radius_m / avg_spacing_m))
```

- Porosity is then:

```text
porosity = N_water / N_total
```

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `static_porosity_500m`, `static_porosity_1km`, `static_porosity_2km`, `static_porosity_5km`, `static_porosity_10km` | `compute_static_site_features` | site | Mean of `water_mask` inside the disk around the snapped site. |

#### Distance to coast and nearest-shore descriptors
- `distance_transform_edt(..., return_indices=True)` is run on the water mask to obtain:
  - metric distance from each water cell to the nearest land cell
  - the row/column of that nearest land cell
- The shoreline gradient is sampled at that nearest land cell using the bathymetry gradients `dz_dx` and `dz_dy`.

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `static_dist_to_coast_m` | `compute_static_site_features` | site | `distance_to_coast_m(site)` from the EDT. |
| `static_nearest_shore_steepness` | `compute_static_site_features` | site | `sqrt((dz/dx)^2 + (dz/dy)^2)` at the nearest land cell. |
| `static_nearest_shore_normal_deg` | `compute_static_site_features`, `to_compass_degrees` | site | Compass direction of the shoreline gradient vector using `atan2` and conversion to compass degrees. |

#### Local depth
- `static_local_depth_m` is sampled from the raw bathymetry grid `z_grid` at the snapped site.
- If that location is non-finite, the code replaces it with the nearest-filled bathymetry value from `z_filled`.
- `local_depth_m` is then converted to positive water depth via `elevation_to_positive_depth(...)`.

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `static_local_depth_m` | `compute_static_site_features` | site | Raw `z_grid[row_site, col_site]`, with nearest finite fill if needed. |
| `local_depth_m` | `compute_static_site_features`, `elevation_to_positive_depth` | site | Positive depth after vertical-sign normalization and clipping to `>= 0`. |
| `local_breaking_cap_valid` | `compute_static_site_features` | site | `1` when the snapped site is in water and `local_depth_m > 0`; else `0`. |

### 4. Directional ray features

#### Base raster derivatives
- `compute_base_matrices(...)` fills NaNs in `z_grid` using nearest finite neighbors.
- It then computes:

```text
dz_dy, dz_dx = gradient(z_filled, dy, dx)
slope = sqrt((dz/dx)^2 + (dz/dy)^2)
laplacian = d2z/dx2 + d2z/dy2
```

- The Laplacian is implemented with two 3x3 finite-difference kernels convolved along `x` and `y`.

#### Per-ray sampling
- `cast_single_ray(...)` samples a ray from a snapped site at angle `angle_deg`.
- Sample positions are generated every `ray_step_m` until `max_ray_m`.
- Land interception is tested with nearest-neighbor sampling of the land mask.
- Bathymetry, slope, and Laplacian profiles are sampled over the water section only using bilinear interpolation (`order=1`).

#### Fetch
- If land is hit, fetch is the distance to the first land sample.
- Otherwise fetch is the last sampled distance, typically bounded by `max_ray_m` or the grid edge.

#### Per-ray formulas

| Quantity in `ray_features.csv` | Source functions | Level | Computation |
|---|---|---|---|
| `fetch_m` | `cast_single_ray` | ray | Distance from site to first land interception, or to the end of the sampled ray if no land is hit. |
| `hit_land` | `cast_single_ray` | ray | Boolean first-contact land flag. |
| `max_slope`, `mean_slope` | `cast_single_ray` | ray | Max and mean of interpolated slope values along the water portion of the ray. |
| `mean_laplacian`, `max_laplacian`, `max_abs_laplacian` | `cast_single_ray` | ray | Summaries of interpolated Laplacian values along the water portion of the ray. |
| `min_depth_m` | `cast_single_ray` | ray | Minimum positive depth along the water portion of the ray after sign normalization. |

#### Site-level aggregation from rays
- `build_sector_site_features(...)` maps each ray angle into one of the 16 compass sectors.
- It then aggregates:
  - mean fetch per sector
  - max slope per sector
  - max Laplacian per sector
  - minimum depth per sector
- `build_ray_site_summary(...)` collapses the per-ray CSV to one row per site.

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `ray_fetch_min_m` | `build_ray_site_summary` | site | Minimum of per-ray `fetch_m` over the site. |
| `ray_fetch_mean_m` | `build_ray_site_summary` | site | Mean of per-ray `fetch_m` over the site. |
| `ray_fetch_max_m` | `build_ray_site_summary` | site | Maximum of per-ray `fetch_m` over the site. |
| `ray_fetch_{sector}_m` | `build_sector_site_features` | sector/site | Mean `fetch_m` of all rays assigned to that sector. |
| `ray_max_slope_{sector}` | `build_sector_site_features` | sector/site | Maximum per-ray `max_slope` in that sector. |
| `ray_max_laplacian_{sector}` | `build_sector_site_features` | sector/site | Maximum per-ray `max_laplacian` in that sector. |
| `ray_min_depth_{sector}_m` | `build_sector_site_features` | sector/site | Minimum per-ray `min_depth_m` in that sector. |
| `ray_hit_land_fraction` | `build_ray_site_summary` | site | Mean of `hit_land` after conversion to numeric `0/1`. |
| `ray_count` | `build_ray_site_summary` | site | Number of ray rows for the site. |

### 5. Fetch anisotropy summaries
- `add_fetch_anisotropy_features(...)` consumes the 16-column sector fetch vector:

```text
F = [ray_fetch_N_m, ray_fetch_NNE_m, ..., ray_fetch_NNW_m]
```

- Let `F_i` be the sector fetch values and `N = 16`.

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `fetch_max_over_mean` | `add_fetch_anisotropy_features`, `_safe_divide` | site | `max(F_i) / mean(F_i)`. |
| `fetch_std_m` | `add_fetch_anisotropy_features` | site | Standard deviation of the 16 sector fetch values. |
| `fetch_cv` | `add_fetch_anisotropy_features`, `_safe_divide` | site | `std(F_i) / mean(F_i)`. |
| `fetch_directional_entropy` | `add_fetch_anisotropy_features` | site | With `p_i = F_i / sum(F_i)`, entropy is `H = -sum_i p_i log(p_i)`, then normalized by `log(16)` and clipped to `[0, 1]`. |
| `fetch_resultant_length` | `add_fetch_anisotropy_features`, `_safe_divide` | site | Compute sector angles `theta_i`, then `R = sqrt((sum_i F_i sin(theta_i))^2 + (sum_i F_i cos(theta_i))^2) / sum_i F_i`, clipped to `[0, 1]`. |
| `open_sector_fraction` | `add_fetch_anisotropy_features` | site | Fraction of sectors with `F_i >= 5000 m`. |
| `closed_sector_fraction` | `add_fetch_anisotropy_features` | site | Fraction of sectors with `F_i <= 500 m`. |
| `open_sector_width_deg` | `add_fetch_anisotropy_features`, `_circular_run_width` | site | Largest contiguous circular run of open sectors times `22.5` degrees. |
| `dominant_fetch_direction_sin` | `add_fetch_anisotropy_features` | site | `sin(theta_max)` where `theta_max` is the compass angle of the sector with maximum fetch. |
| `dominant_fetch_direction_cos` | `add_fetch_anisotropy_features` | site | `cos(theta_max)` where `theta_max` is the compass angle of the sector with maximum fetch. |

Important details:

- `open_sector_fraction` and `closed_sector_fraction` use fixed code thresholds:
  - `OPEN_FETCH_THRESHOLD_M = 5000.0`
  - `CLOSED_FETCH_THRESHOLD_M = 500.0`
- `open_sector_width_deg` wraps around north correctly by duplicating the boolean sector mask and searching for the longest contiguous run.
- `dominant_fetch_direction_*` uses `np.nanargmax`, so ties resolve to the first maximum in sector order.

### 6. Path geometry ratios
- `add_path_geometry_ratio_features(...)` adds scale-normalized path metrics after the route and ray tables have already been merged.
- Straight-line distance is computed from the first and last points of the stored `path_xy` polyline from the routing payload.

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `path_direct_distance_m` | `add_path_geometry_ratio_features` | site | Straight-line Euclidean distance between the route start and end points. |
| `path_tortuosity_ratio` | `add_path_geometry_ratio_features`, `_safe_divide` | site | `path_length_m / path_direct_distance_m`. |
| `bottleneck_to_path_ratio` | `add_path_geometry_ratio_features`, `_safe_divide` | site | `path_bottleneck_m / path_length_m`. |
| `bottleneck_to_fetch_ratio` | `add_path_geometry_ratio_features`, `_safe_divide` | site | `path_bottleneck_m / ray_fetch_mean_m`. |
| `funneling_log` | `add_path_geometry_ratio_features` | site | `log1p(max(funneling_ratio, 0))`. |

### 7. Fjordness and regime features
- `add_fjordness_features(...)` builds a continuous enclosure score from six components and then bins that score into three regimes.
- The internal helper `_minmax_component(values, inverse=False)` works as follows:
  - convert to `float64`
  - if `inverse=True`, replace each finite value with `1 / max(value, 1e-6)`
  - replace non-finite values with the median of the finite values
  - min-max scale to `[0, 1]`
  - if there is no finite spread, return zeros

This means fjordness is not an absolute physical index. It is a dataset-relative composite defined by the distribution of the current site set.

#### Fjordness components

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `fjordness_land_blocking_component` | `add_fjordness_features`, `_minmax_component` | site | Min-max scaled `ray_hit_land_fraction`. |
| `fjordness_low_porosity_component` | `add_fjordness_features`, `_minmax_component` | site | Min-max scaled `1 - static_porosity_5km`. |
| `fjordness_closed_sector_component` | `add_fjordness_features`, `_minmax_component` | site | `0.5 * (scaled(closed_sector_fraction) + scaled(1 - open_sector_fraction))`. |
| `fjordness_anisotropy_component` | `add_fjordness_features`, `_minmax_component` | site | Min-max scaled `fetch_max_over_mean`. |
| `fjordness_low_fetch_component` | `add_fjordness_features`, `_minmax_component(..., inverse=True)` | site | Inverse min-max scaled `ray_fetch_mean_m`, so smaller mean fetch gives a larger fjordness contribution. |
| `fjordness_route_complexity_component` | `add_fjordness_features`, `_minmax_component` | site | Mean of three scaled terms: `static_tortuosity_sum`, `funneling_log`, and inverse-scaled `path_bottleneck_m`. |

#### Fjordness score and regimes

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `fjordness_score` | `add_fjordness_features` | site | Arithmetic mean of the six fjordness component columns, then clipped to `[0, 1]`. |
| `site_regime_open` | `add_fjordness_features` | site | `1` if `fjordness_score < 0.35`, else `0`. |
| `site_regime_transition` | `add_fjordness_features` | site | `1` if `0.35 <= fjordness_score < 0.65`, else `0`. |
| `site_regime_fjord` | `add_fjordness_features` | site | `1` if `fjordness_score >= 0.65`, else `0`. |

Current regime thresholds:

- `FJORDNESS_OPEN_MAX = 0.35`
- `FJORDNESS_TRANSITION_MAX = 0.65`

### 8. Local breaking features
- `add_local_breaking_features(...)` uses `local_depth_m` and the site validity flag to create a depth-limited significant-wave-height cap.
- The logic is:

```text
valid = isfinite(local_depth_m) and local_depth_m > 0
if local_breaking_cap_valid already exists:
    valid = valid and (local_breaking_cap_valid > 0)
```

- If breaking is enabled:

```text
local_breaking_hs_cap = gamma * local_depth_m
```

- Otherwise the cap column is filled with `NaN`.
- Current defaults in `build_features.py` are:
  - `DEFAULT_BREAKING_ENABLED = True`
  - `DEFAULT_BREAKING_GAMMA = 0.78`
- The preprocess config can override these under `static_features.breaking`.

| Columns | Source functions | Level | Computation |
|---|---|---|---|
| `local_breaking_cap_valid` | `compute_static_site_features`, `add_local_breaking_features` | site | Binary validity flag for whether the local breaking cap should be trusted. |
| `local_breaking_hs_cap` | `add_local_breaking_features` | site | `gamma * local_depth_m` when valid and enabled, else `NaN`. |

## Important Implementation Caveats

### Fjordness is relative to the current site set
- The fjordness components use per-dataset min-max scaling.
- Adding or removing sites changes the scaling range and can therefore change every site's fjordness component values and `fjordness_score`.

### Angular resolution matters
- `ray_count`, `ray_fetch_*`, `ray_hit_land_fraction`, and the sector summaries depend on the chosen ray angle step.
- The sector columns are always 16-direction summaries, but those summaries are built from however many raw rays were cast.

### Grid spacing matters
- Porosity disks are approximated in raster cells using average grid spacing.
- Distance-to-coast, path length, bottleneck width, and ray fetch all depend on the raster resolution and spacing.

### Path smoothing is selective
- Route smoothing affects only:
  - `static_tortuosity_sum`
  - `static_signed_curvature_deg`
  - `static_net_deflection_deg`
  - `static_final_approach_deg`
- It does not affect:
  - `path_length_m`
  - `path_point_count`
  - `path_bottleneck_m`
  - `funneling_ratio`
  - `choke_out_ratio`

### Safe division returns `NaN`
- `_safe_divide(...)` returns `NaN` when the denominator is zero or non-finite.
- This affects:
  - `fetch_max_over_mean`
  - `fetch_cv`
  - `fetch_resultant_length`
  - `path_tortuosity_ratio`
  - `bottleneck_to_path_ratio`
  - `bottleneck_to_fetch_ratio`

### Raw route/ray outputs vs merged columns
- `routing_features.pkl` stores route geometry and route QA records.
- `ray_features.csv` stores per-ray records plus sector-level columns repeated across the site's rays.
- `master_static_features.csv` is the site-level merged and derived table that the model consumes.

### Static vs dynamic geometry-interaction features
- The static features described here depend only on the fixed site geometry and bathymetry.
- Features such as `wave_fetch_aligned_m`, `wave_slope_aligned`, `wind_fetch_aligned_m`, and windsea proxies are not part of `master_static_features.csv`.
- Those are dynamic features because they combine static geometry with time-varying offshore wave or wind direction during point-centric preprocessing.

## Validation Notes
The builder validates several invariants in `validate_master_static_features(...)`:

- all required derived columns exist
- no numeric column contains `inf`
- all `ray_min_depth_{sector}_m` values are non-negative
- `fjordness_score` stays in `[0, 1]`
- `site_regime_open + site_regime_transition + site_regime_fjord = 1` for every site
- `local_depth_m` is non-negative where finite
- `local_breaking_cap_valid` contains only `0/1`

Those checks are useful when updating the geometry pipeline, because they encode the current expectations of the static-feature contract.
