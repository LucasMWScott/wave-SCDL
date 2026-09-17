# Evaluation and new-site inference

For prepared validation/test data, use `evaluate` with the original training config, checkpoint and prepared artifacts. It reloads the checkpoint, reuses target/scaler metadata, reconstructs physical units, normalizes circular pairs and exports metrics/predictions. The synthetic quick-start verifies this route, including checkpoint reload.

The existing target-free adapter is `python -m coastal_wave infer --config examples/inference.yaml --device cpu`. Edit the example paths to real artifacts first. It needs a trusted checkpoint; matching training configuration; `training_run_metadata.json` beside the checkpoint; the original `point_centric_metadata.json` and saved source/target/static normalization state; new site/source YAML; raw NORA3 CSVs; static table where enabled; and bathymetry patches where enabled. Training metadata may contain absolute runtime paths: when moving a run, update the referenced prepared directory to its new location without changing schemas or statistics.

Static transforms and source magnitude normalization are restored from the saved training state. Never fit replacement scalers on new-site observations. Bathymetry generation must use `bathy-patches --normalization-reference TRAINING_NPZ` with the original channel order and statistics.

Raw inference windows include history through the prediction timestamp and seasonal channels. Output CSV rows identify site and timestamp; predicted Hs is meters, Tp seconds, and mean/peak direction degrees modulo 360. Reference wave fields are physical quantities separate from normalized inputs and are required for transfer reconstruction.

## Preserved adapter restrictions

This adapter originated for a particular transfer model. It is not a general deployment API for every supported training configuration:

- Raw mode hardcodes three nearest wave sources, weight power 1, 80/150 km thresholds, and an initial minimum of 31 aligned timestamps. Actual window length is read later from training metadata.
- It requires configured local wind points and sea/swell partition fields. Local wind selection uses squared latitude/longitude distance; this differs from the training helper's geodesic selection.
- Enabled bathymetry currently requires the five channels depth, land_sea_mask, slope_magnitude, curvature_laplacian, distance_to_land in that order; six-channel training configurations cannot be assumed compatible.
- Prepared mode unconditionally expects source, static, bathymetry, geometry and explicit reference artifacts, and slices a final 30-step window. It repeats that final-window prediction across the supplied timestamps. This existing behavior is unsuitable for a time-resolved forecast product and is documented rather than silently fixed.
- Reference construction and output decoding have transfer-specific assumptions. Physical-head models should use evaluation on correctly prepared data until a separately reviewed general target-free adapter is implemented.

New sites are supported only within these schema and branch constraints, with valid matching geometry, source fields, fitted transformations and checkpoint inputs. Broader geographic or scientific generalization was not validated. The refactor tests exercise transform restoration, transfer reconstruction and checkpoint loading; they do not certify the raw adapter on new real sites.
