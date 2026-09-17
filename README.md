# Coastal wave downscaling

Generate site-level coastal geometry and train a coastal-conditioned wave model using offshore wave/wind histories. Geometry runs independently of PyTorch; research data and trained checkpoints are supplied separately.

> **Research code.** This repository contains the reproducible software workflow, not the NORA3/NORAC observations, bathymetry files, or trained models used in the research runs.

For more help, please contact lucas.scott@rogers.com.

## Contents

- [Install](#install)
- [Full workflow instructions](#full-workflow-instructions)
- [Project layout](#project-layout)
- [Documentation](#documentation)

## What this repository provides

| Stage | Purpose | Main command |
| --- | --- | --- |
| Geometry | Build grids, routes, rays, and static site features from bathymetry | `python -m coastal_wave geometry` |
| Preprocessing | Create aligned point-centric arrays and saved normalization state | `python -m coastal_wave preprocess --build-point-centric` |
| Training | Fit a configured coastal transformer or LSTM experiment | `python -m coastal_wave train` |
| Evaluation | Restore a checkpoint and write predictions and metrics | `python -m coastal_wave evaluate` |
| Analysis | Inspect data, model behavior, and evaluation outputs in notebooks | `jupyter lab` |

## Install

Python 3.10 or newer is required; validation used Python 3.12 with CPU PyTorch 2.5.1. From a WSL terminal in this directory:

```sh
python -m venv .venv
```

Activate the environment:

```sh
source .venv/bin/activate
```

Then:

```sh
python -m pip install -e ".[geometry,ml,dev]"
```

## Full workflow instructions

### 1. Install the project

Run these commands from the repository root. Use `cpu` in the commands below if you do not have a compatible CUDA PyTorch installation; use `auto` to select CUDA when it is available.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[geometry,ml,notebooks,dev]"
```

Check that the command interface is available:

```bash
python -m coastal_wave --help
```

### 2. Create your three editable configuration files

Do not edit `configs/FINAL_RUNS_V2` unless you are deliberately replaying a historical final run. Copy the editable templates and give the copies a run-specific name:

```bash
cp configs/sites.yaml configs/my_sites.yaml
cp configs/preprocess.yaml configs/my_preprocess.yaml
cp configs/training.yaml configs/my_training.yaml
```

In `configs/my_preprocess.yaml`, set `paths.sites_yaml: my_sites.yaml` and choose a run directory, for example `paths.processed_dir: ../data/processed/my_run`. 
In `configs/my_training.yaml`, set `data.sites_config: my_sites.yaml`, `data.point_centric_dir: ../data/processed/my_run`, and set `data.static_features_csv` to the static table you will create below. These editable files use `path_base: config`, so `../data/...` is relative to the `configs` directory.

### 3. Supply and configure the raw data

Create this directory structure, or change the matching paths under `backend` in `configs/my_sites.yaml`:

```text
data/
  raw/
    bathy/
      *.xyz
    nora3/
      params/
        nora3_grid_<id>.csv
        nora3_wind_grid_<id>.csv
    norac/
      params/
        norac_grid_<id>.csv
  processed/
  results/
```

Bathymetry files are whitespace-separated XYZ text files with at least three numeric columns: projected easting `x`, projected northing `y`, and elevation/depth `z`, all in metres. Their CRS must match `bathy.epsg` in `my_preprocess.yaml` (the supplied template uses EPSG:32633). The grid loader infers positive-down depth versus signed elevation from the values; check the generated metadata before changing the vertical convention. The original project uses Kartverket-50m bathymetry data, but any XYZ bathymetry data will work.

Each NORA3 and NORAC parameter CSV needs a `time` column plus the fields enabled in `data.offshore_vars` or `data.nearshore_vars` in `my_training.yaml`. Use ISO-like timestamps in one consistent timezone. For the supplied default model, offshore CSVs need wave variables such as `hs`, `tp`, `Pdir`, `thq`, sea/swell partition fields when transfer targets are enabled, and wind fields if enabled. Nearshore CSVs need `hs`, `tp`, `dir`, and `dp`. Heights are metres, periods are seconds, wind speed is m/s, and directions are degrees. File names must contain the matching site name, such as `nora3_grid_24.csv` or `norac_grid_111.csv`.

NOTE: Although the original project uses NORA3 wave and wind data with NORAC wave data, you are able to use any type of training data so long as it matches the same timeseries layout. Some slight tweaking may be required, but can easily be done using agents. 

In `configs/my_sites.yaml`, set the wave param dir and the name and coords of your offshore + nearshore sites:

```yaml
backend:
  nora3_params_dir: ../data/raw/nora3/params
  norac_params_dir: ../data/raw/norac/params

offshore_sites:
  - name: nora3_grid_24
    lat: 63.23
    lon: 7.27
  - name: nora3_wind_grid_20
    lat: 63.23
    lon: 7.30

nearshore_sites:
  - name: norac_grid_111
    lat: 63.20
    lon: 7.25
```

Every site needs a unique `name`, `lat`, and `lon` in WGS84 degrees. Nearshore names must match both the NORAC filename and the static-feature `site_name`. Offshore wave sources use names without `wind`; offshore local-wind sources include `wind` in their name. Keep this ordering and naming consistent across all files. 

If you want to run training or inference on non-NORA data, follow the same procedure. Specify your data point coordinates + file names, as well as your nearshore coordinates + filenames. To run inference on non-trained sites (like validating a SWAN run), simply provide your coordinates of interest. 

It is easy to create a script to automatically generate sites.yaml configs using a downloaded list of offshore + nearshore data points.


### 4. Choose preprocessing settings

Edit `configs/my_preprocess.yaml` before running geometry. The most important settings are:

- `bathy.resolution_m`, `padding_m`, and `epsg` control the raster used for routing and ray casting.
- `static_features.route_curtain.mode` selects `global` or `local_k_nearest`. Local-nearest routing requires `data.multi_source.enabled: true` in `my_training.yaml`.
- `static_features.master_out_path` is the generated `master_static_features.csv` consumed by training.
- `bathy.channels`, `patch_size`, and `depth_clip_m` control optional bathymetry inputs.
- `circular.input_degrees` must match the direction units in your CSVs.

Set `data.sequence_window`, `date_range`, `train_sites`, `validation_sites`, and `test_sites` in `configs/my_training.yaml`. Validation and test site lists must be disjoint from training sites. If you use held-out sites with `split.site_holdout_temporal_mode: shared_recent`, use the documented shared-recent fractions. The model uses the final timestamp of each history window as its target timestamp.

The `train_sites`, `validation_sites`, and `test_sites` configs are lists that specify which my_sites.yaml sites are used for training, validation, or testing. 

### 5. Generate geometry and static features

This consumes your site YAML and XYZ bathymetry and writes grids, routing diagnostics, rays, and the static table. Run it from the repository root:

```bash
python -m coastal_wave geometry --sites configs/my_sites.yaml --preprocess-config configs/my_preprocess.yaml --training-config configs/my_training.yaml --bathy-dir data/raw/bathy
```

Confirm that the generated path matches `data.static_features_csv` in `configs/my_training.yaml`. `master_static_features.csv` has one row per nearshore site. Check the logged reachable-site count before training.

### 6. Build optional bathymetry patches

Skip this step when `data.use_bathymetry: false`. When it is true, build patches after geometry and set the resulting NPZ path in the training configuration where required by your run:

```bash
python -m coastal_wave bathy-patches --sites configs/my_sites.yaml --preprocess-config configs/my_preprocess.yaml --out data/processed/my_run/point_centric_X_bathy.npz
```

For inference at new sites, reuse the training patch normalization with `--normalization-reference data/processed/my_run/point_centric_X_bathy.npz`; do not fit bathymetry statistics from new-site data.

### 7. Build the point-centric dataset

First inspect file discovery. It should report non-zero file counts for the expected sites:

```bash
python -m coastal_wave preprocess --sites configs/my_sites.yaml --list
```

Then create the prepared arrays and metadata:

```bash
python -m coastal_wave preprocess --sites configs/my_sites.yaml --training-config configs/my_training.yaml --preprocess-config configs/my_preprocess.yaml --build-point-centric
```

This writes `point_centric_X_dynamic.npz`, targets, source metadata, static vectors, and optional source-geometry/bathymetry/physics sidecars under `data/processed/my_run`. Keep this complete directory with its model checkpoint. The preprocessing state is fitted on the configured training subset and must be reused at evaluation and inference.

### 8. Set training options and train

In `configs/my_training.yaml`, change these groups deliberately:

- `data.use_static_features`, `use_geometry`, `use_bathymetry`, and `multi_source` select input branches.
- `data.targets` selects physical, transfer, or combined target behavior. Preserve its matching CSV fields and saved reference sidecars.
- `model.coastal_transformer` controls encoder type, dimensions, layers, heads, static branch, bathymetry branch, and source geometry.
- `training.batch_size`, `epochs`, `optimizer`, scheduler, sampler, loss, and seed control the optimization run.
- `logging.output_dir` and `checkpoint_name` choose where checkpoints and metrics are written.

Run training:

```bash
python -m coastal_wave train --config configs/my_training.yaml --device auto
```

The console prints the selected checkpoint path. Training writes a checkpoint, `training_run_metadata.json`, `observed_model_io.json`, history, and metric summaries in `logging.output_dir`. Do not change feature order, history length, target representation, fitted preprocessing state, or enabled branches when evaluating that checkpoint.

### 9. Evaluate the trained model

Use the same training configuration and prepared-data directory. Replace the checkpoint path with the path printed by training:

```bash
python -m coastal_wave evaluate --config configs/my_training.yaml --checkpoint results/my_run/model.pt --split test --device auto --out-dir results/my_run/evaluation
```

Choose `train`, `val`, or `test` with `--split`. Evaluation restores preprocessing metadata, reconstructs physical units, normalizes circular predictions, and writes prediction CSV/NetCDF files and metric summaries. Use the test split once for final reporting; tune settings with training/validation data only.

### 10. Visualize and inspect results

Launch Jupyter from the repository root:

```bash
jupyter lab
```

Use the notebooks in this order: `00_synthetic_geometry` for a no-data check; `01`–`06` for raw/prepared data QA; `07`–`10` for model and checkpoint checks; `11_evaluation_overview`, `12_site_performance`, and `13_direction_errors` for results; and `14`–`18` for optional explainability and failure analysis. In each notebook, set its `RESULTS_DIR`, `SPLIT`, and configuration variables to your run directory. The evaluation command’s prediction CSV is the main input for the result notebooks.

### 11. Reproduce an archived final run

The historical final configurations remain in `configs/FINAL_RUNS_V2`. They retain their original relative-path behavior, so run them from the repository root and ensure their referenced external data paths exist:

```bash
python -m coastal_wave train --config configs/FINAL_RUNS_V2/4_trans_static_cross.yaml --device auto
```

Commands also work outside the checkout when config arguments are absolute. The editable `training.yaml`, `preprocess.yaml`, `sites.yaml`, and `tuning.yaml` use `path_base: config`, so their paths resolve relative to the declaring file. `FINAL_RUNS_V2` remains unmarked to retain its historical path behavior; run those configurations from the repository root unless you migrate their paths deliberately.

## Project layout

- `geometric_builder/src/`: bathymetry, routing, ray casting, and static aggregation; no ML imports.
- `coastal_wave/common/`: shared deterministic source selection and configuration paths.
- `src/`: ML data preparation, datasets, models, losses, training, evaluation, inference, and diagnostics. This import name is retained for artifact and research-code compatibility.
- `tuning/`: Optuna objective; `configs/`: the editable workflow configs and the archived `FINAL_RUNS_V2` configurations.
- `notebooks/`: an ordered set of preparation, QA, evaluation, and interpretation notebooks.
- `tests/` and `examples/fixtures/`: regression checks and a labeled synthetic reference table.

## Documentation

Read [data and artifacts](docs/data.md), [geometry](docs/geometry.md), [training and tuning](docs/training.md), [inference](docs/inference.md), [notebooks](docs/notebooks.md), and the [refactor report](docs/refactor_report.md). Every workflow has help: `python -m coastal_wave train --help`.

Supply bathymetry, site coordinates, NORA3/NORAC parameter CSVs, and any historical run artifacts through configuration. This repository does not download or redistribute them. Missing research artifacts account for explicit test skips. See the report for baseline failures, inference restrictions, and what was actually verified.

## License

Released under the [MIT License](LICENSE).

## Release checks

Before opening a pull request or publishing a change, install the full local
development environment and run:

```bash
python -m pip install -e ".[geometry,ml,notebooks,dev]"
python -m ruff check . --no-cache
python -m pytest -q
python -m build --wheel --no-isolation
```

Generated data, checkpoints, results, build products, and local environments
are excluded by `.gitignore` and should not be committed.
