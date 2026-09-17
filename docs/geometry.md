# Independent geometry generation

Install `python -m pip install -e ".[geometry]"`. Geometry does not import Torch, models, trainers, or loss functions. Its only shared component with ML is neutral source selection/configuration.

The stages are XYZ bathymetry rasterization, boundary-curtain routing, directional ray casting, and site aggregation. For a complete small run:

```sh
python -m coastal_wave demo --output examples/geometry-run
```

For research inputs, start from the supplied site/preprocess/training YAML files and explicitly select their paths:

```sh
python -m coastal_wave geometry --sites configs/sites.yaml --preprocess-config configs/preprocess.yaml --training-config configs/training.yaml --bathy-dir data/raw/bathy
```

See `python -m coastal_wave geometry --help` for output paths, EPSG, raster resolution, padding, bottleneck cost, ray step and maximum range. Individual commands `bathy-grid`, `route`, and `rays` expose the original stages. Their defaults are retained: standalone routing defaults to global curtains, while the supplied preprocessing config selects local K-nearest curtains. Choose explicitly for comparisons.

CLI path arguments are relative to the calling working project (or absolute). Set `COASTAL_WAVE_PROJECT` when selecting a project from another directory; explicit absolute CLI paths are preferable for automation. YAML path fields with `path_base: config` resolve relative to their own file. The editable root configs use this rule; retained `FINAL_RUNS_V2` configs preserve their historical path behavior and should be run from the repository root. The default command values describe the checkout's `configs` and `data` layout; installed wheels do not bundle research configs or datasets.

`static_features.route_curtain.mode=global` uses the global offshore curtain; `local_k_nearest` builds site-local curtains and requires `data.multi_source.enabled=true` in the supplied training/source-selection config. K, candidate filtering, weights and warning/error distance thresholds must match ML source selection. `padding_m` changes the cropped domain and therefore routing/ray boundary behavior; retain it when comparing runs.

Outputs include full/local grid NPZs, routing pickle, ray CSV, route-curtain QA CSV, and `master_static_features.csv`. Distances stay in meters and angles in degrees until ML preprocessing. Routing pickle contains diagnostic paths and tables; only load trusted project artifacts. The static master table is the ML boundary. Features combining fetch geometry with time-varying wave/wind direction remain in `src.multi_source` and point-centric preprocessing.

The table validates one row per site and schema/order, but synthetic checks do not establish scientific fidelity for a real bathymetric domain. [Feature formulas](geometry_features.md) and [the normalization reference](feature_normalization_reference.md) give details.
