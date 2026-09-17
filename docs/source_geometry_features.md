# Source Geometry Features

This document explains what `use_geometry` means in `configs/training.yaml`,
which features it enables, and how those features are used at runtime.

## Short answer

`data.use_geometry` does **not** mean "use all geometry/static features".

It is a runtime alias for:

```yaml
model:
  coastal_transformer:
    multi_source:
      use_geometry_features: true
```

That flag enables the optional multi-source `source_geometry` tensor used by
the offshore-source encoder.

It is separate from:

- `data.use_static_features`: enables the site-level static coastal geometry
  vector from `point_centric_X_static.npz`
- `data.use_bathymetry`: enables the bathymetry raster branch

## What artifact it controls

When enabled, preprocessing/runtime uses:

- `point_centric_source_geometry.npz`

This contains a site-indexed tensor with shape:

- `[N_sites, K, Dg]`

where:

- `N_sites` = number of nearshore target sites
- `K` = number of selected offshore source points per site
- `Dg` = number of geometry features per source

At batch time the model receives:

- `source_geometry: [B, K, Dg]`

## Which features are included

The features are built in [src/multi_source.py](../src/multi_source.py)
by `build_source_geometry_array(...)`.

Base features per offshore source:

1. `distance_m_norm`
2. `bearing_sin`
3. `bearing_cos`
4. `inverse_distance_weight`

Plus one-hot rank indicators:

5. `source_rank_1`
6. `source_rank_2`
7. `source_rank_3`

So in the current `thesis_set_1` processed dataset, `Dg = 7` because
`K = 3`.

The generated metadata confirms this in
`data/processed/thesis_set_1/point_centric_metadata.json`.

## What each feature means

- `distance_m_norm`: source-to-target distance divided by the configured
  maximum allowed distance error scale, then clipped to `[0, 1]`
- `bearing_sin`, `bearing_cos`: circular encoding of the source-to-target
  bearing in degrees
- `inverse_distance_weight`: precomputed distance-based weight from the source
  selection stage
- `source_rank_i`: which of the `K` selected offshore sources this row refers
  to

If `data.multi_source.k_nearest` changes, the number of `source_rank_*`
features changes too.

## What it does not include

`use_geometry` does **not** turn on the site-level static geometry feature set
from `master_static_features.csv`.

Those static features are controlled by `data.use_static_features` and include
things like:

- route/path descriptors such as `path_length_m` and `path_bottleneck_m`
- directional fetch rays such as `ray_fetch_NE_m`
- local depth and sheltering descriptors

Those live in `point_centric_X_static.npz`, not `point_centric_source_geometry.npz`.

## How the model uses it

In the multi-source encoder
([src/models/coastal_transformer.py](../src/models/coastal_transformer.py)),
the source geometry features are projected and added to each source token
before aggregation across offshore sources.

Two aggregation modes exist:

- `attention`: learned attention over offshore sources
- `weighted_pool`: uses `source_geometry[..., 3]`, which is
  `inverse_distance_weight`

## Precedence rule

Runtime resolves the flag in this order:

1. `model.coastal_transformer.multi_source.use_geometry_features`
2. `data.use_geometry`

So the model-side config takes precedence if both are present.

## Current dataset example

For `data/processed/thesis_set_1`:

- `source_geometry_shape = [310, 3, 7]`
- `source_geometry_feature_names = [`
  `distance_m_norm, bearing_sin, bearing_cos, inverse_distance_weight,`
  `source_rank_1, source_rank_2, source_rank_3]`

See
`data/processed/thesis_set_1/point_centric_metadata.json`
and
`data/processed/thesis_set_1/point_centric_metadata.json`.
