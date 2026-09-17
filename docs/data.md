# Data and artifact contracts

## Raw inputs

Site YAML contains ordered `offshore_sites` and `nearshore_sites` lists. Each entry requires `name` (unique string), `lat`, and `lon` (numeric geographic degrees, EPSG:4326). Nearshore entries may contain `paired_offshore` (list of source names). `backend.nora3_params_dir` and `backend.norac_params_dir` identify local CSV directories. The synthetic generator supplies a minimal complete example. Real site configurations are in `configs/sites*.yaml`.

Parameter files are recursively discovered by site name in their filenames; the point-centric loader uses token-aware matching to avoid confusing numbered sites. CSV headers must include the configured `split.datetime_column` (default `time`) and `data.offshore_vars` / `data.nearshore_vars`. Comma-separated numeric fields are expected; lines beginning `#` are ignored. Parsed invalid timestamps are dropped, rows are sorted, and duplicate timestamps keep the last row. Multiple matching files are concatenated in sorted filename order. Offshore timelines are intersected; targets are reindexed to the aligned timeline. No general resampling is performed.

Supply a consistent timestamp convention, preferably naive UTC ISO timestamps (the synthetic case is hourly). The implementation does not infer the originating timezone. Date-range bounds normalize timezone-aware bounds to naive time; do not mix local time and UTC files. The sequence validator checks regular spacing inferred from the prepared timeline, not a universally hardcoded hourly frequency. Set history length with `data.sequence_window`.

Common offshore variables are `hs`, `tp`, `tm1`, `tm2`, `tmp`, `Pdir`, `thq`, `hs_sea`, `tp_sea`, `thq_sea`, `hs_swell`, `tp_swell`, `thq_swell`, `wind_speed_10m`, and `wind_direction_10m`. Required subsets depend on the configuration: weighted partitioned transfer requires sea/swell height, period, and direction fields. Heights are meters; periods seconds; wind speed m/s; directions degrees. Nearshore targets are ordered `hs`, `tp`, `dir`, `dp`.

NORA3 `Pdir`, `thq`, `thq_sea`, and `thq_swell` are shifted by +180 degrees modulo 360 before encoding. Wind directions are not shifted. Circular features use sine then cosine of the supplied angle. Compass geometry bearings use degrees; do not substitute mathematical counterclockwise-from-east angles. Validate external products' direction conventions before adapting them.

## Geometry boundary

`master_static_features.csv` is keyed by `site_name`, one row per nearshore site. It includes raw site coordinates, routing reachability and lengths in meters, turn/approach angles in degrees, depth, shoreline derivatives, 16 compass-sector fetch/slope/Laplacian groups, porosity, anisotropy, fjordness, and path ratios. The implemented synthetic run has 124 ordered columns. The executable schema is `geometric_builder.src.features.MASTER_COLUMNS`; [geometry feature formulas](geometry_features.md) provide the scientific reference. Use saved transformed feature names, not the CSV's apparent numeric columns, as the model schema.

Bathymetry XYZ inputs are whitespace-separated numeric rows read by numpy.loadtxt, with at least three columns: projected x/y and elevation/depth in meters. Default projected CRS is EPSG:32633; coordinates must match the configured EPSG. Prepared grid NPZ stores one-dimensional `x`, `y`, two-dimensional `z` in `(y,x)` order, boolean `land_mask` (true means land), sample counts, and metadata describing vertical convention. Missing raster cells and land are handled by the existing generator/mask rules. Do not negate an already positive-down dataset without checking its metadata. The ML patch `land_sea_mask` is a *wet* indicator (1 water), the opposite polarity of grid `land_mask`.

## Prepared arrays

Let N be sites, T timestamps, K sources, L history, and F features. Numeric model tensors are float32; stored timestamps and names are strings; split indices are integers.

| File | Principal contract |
|---|---|
| `point_centric_X_dynamic.npz` | `X_dynamic` [T,F], timestamps, train/val/test index vectors |
| `point_centric_X_dynamic_sources.npz` | `X_dynamic_sources` [N,T,K,F], site/source ordering and timestamps |
| `point_centric_Y_targets.npz` | `Y__<site>` [T,6] in hs,tp,dir_sin,dir_cos,dp_sin,dp_cos order |
| `point_centric_X_static.npz` | `Xstatic__<site>` transformed static vectors, when enabled |
| `point_centric_source_geometry.npz` | source geometry [N,K,G], when enabled |
| `point_centric_X_bathy.npz` | `X_bathy` [N,C,H,W], target sites, channel names, normalization metadata |
| `point_centric_metadata.json` | feature names/order, fitting scopes, scaler state, target representation and provenance |
| `point_centric_source_metadata.json` | ordered selected sources, distances, bearings, weights per target site |

Optional physical, transfer, reference, site-dynamic, local-wind, and breaking-physics sidecars are produced according to enabled branches. Preserve the whole prepared directory with its run. Do not infer artifact availability from a historical filename list. Physical values and transfer references remain separate from normalized model inputs; losses, reconstruction, and samplers use them where configured.

Nearest sources use haversine distance (Earth radius 6,371,000 m), deterministic distance/name ranking, configured K and inverse-distance power. Site order follows site YAML; source rank order follows saved metadata. Geometry local curtains use this same implementation. Source geometry is ordered normalized distance, bearing sine/cosine, inverse-distance weight, then K one-hot rank channels.

Each sample uses rows `[t-L+1:t+1]` with target at `t`. Seasonal `time_sin/time_cos` are added at runtime using day-of-year and 365.25 days; they are absent from persisted dynamic arrays. Window eligibility, finite targets, site exclusions, sample filters, and split boundaries are enforced by the dataset. Nonfinite dynamic/static values use existing median/zero fallback handling. A fully missing reference feature may fall back beyond the training subset; this is retained and listed as a scientific review item rather than changed silently.

## Fitting and split scope

Static column groups use cyclic encoding, median/constant imputation, log1p plus MinMax scaling for heavy-tailed features, RobustScaler for derivatives, and MinMax for ratios/other numeric columns. Fit uses training sites only. Dynamic magnitude statistics use the configured normalization method and fitting rows. Target StandardScaler uses training sites and fitting rows; circular target components remain circular. Supported general methods are `zscore`, `minmax`, `robust`, `global_maxabs`, and `samplewise_l2`.

In temporal mode, fitting rows are the training prefix. In legacy site-heldout mode, fitting uses the full timeline of training sites. In `shared_recent` site-heldout mode, fitting uses only the earlier training window; validation and test sites use the same recent temporal window. Their *sites* remain disjoint, so `split.val` and `split.test` both describe that recent fraction rather than adding independently to the training fraction. This distinction is intentional and must not be silently reconciled.

Bathymetry v1 uses standardized log1p depth plus wet mask. V2 supports configured channel order/subsets: depth, wet mask, slope magnitude, distance to land, curvature Laplacian, shallow breaking mask. V2 depth uses training wet-cell maximum scaling with clipping; derivative/distance statistics retain their implemented robust transforms. Statistics are fitted on training-site wet patches. `--normalization-reference` reuses a training bathymetry NPZ for new sites and validates channel/version compatibility.

Restore static transforms with `restore_static_transformer_artifacts`; do not refit on inference sites. Keep scaler metadata, channel names and feature lists with the checkpoint. Older metadata missing fitted state may require the original training data through the recovery command.
