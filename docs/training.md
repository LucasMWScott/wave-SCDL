# Preprocessing, training, evaluation and tuning

Install the geometry and ML extras. Real runs need local parameter CSVs, geometry products, and sufficient RAM/storage for prepared arrays. CPU is supported; select a compatible GPU PyTorch build yourself for large experiments.

1. Configure site/source identities, CSV directories and target site partitions.
2. Generate static geometry where static, source geometry, directional exposure or physics branches require it.
3. Copy a research configuration and preserve its scientific choices explicitly: source K, target representation, history length, feature list/order, bathymetry channels, loss, optimizer, sampling, and split mode.
4. Build a fresh prepared directory, then train against it.

```sh
python -m coastal_wave preprocess --sites configs/sites.yaml --training-config configs/training.yaml --preprocess-config configs/preprocess.yaml --build-point-centric
python -m coastal_wave train --config configs/training.yaml --device cpu
python -m coastal_wave evaluate --config configs/training.yaml --checkpoint results/run/model.pt --split test --device cpu
```

The checkpoint in the last command is an example runtime path: use the actual `logging.output_dir` / `checkpoint_name` printed by training. Do not evaluate a checkpoint with an unrelated config or prepared directory.

The factory creates `CoastalConditionedTransformer`, with configurable Transformer or LSTM sequence encoder, static branch, bathymetry CNN, multi-source processing, task decoder and optional expert heads. In physical mode the ordinary heads regress Hs and classify Tp/direction/Dp; distribution recovery yields physical predictions. Transfer modes predict four residual quantities (`log_hs_ratio`, `tp_delta`, `dir_delta_deg`, `dp_delta_deg`) with legacy or bounded residual-correction representation. `physical_and_transfer` combines configured loss pathways. The six-column legacy target artifact is not a statement that every model has a six-scalar output head.

Model parameters and state-dictionary keys were retained. Losses (including hybrid, transfer, expert and breaking options), optimizer/scheduler choices, PCGrad, sampling/subsampling, branch ablations, augmentation and evaluation metrics remain implemented in their original ML modules. Existing config aliases keep their warnings and precedence. Read [configuration reference](config_reference.md) and executable `resolve_config` for detailed switches. Defaults vary across research configurations; do not combine settings merely because they have similar names.

Training writes checkpoint(s), history, physical-unit prediction/metric exports, `training_run_metadata` and `observed_model_io` manifests. Keep JSON metadata and the complete prepared dataset alongside checkpoints; reconstructing a model needs architecture and input schema as well as weights. Only trusted legacy pickle artifacts should be loaded. Python compatibility modules preserve the historical normalization class paths; model modules still live under `src.models`.

Batch plans remain supported by `train --batch-config PATH` (see help for the exact available argument). Tuning uses:

```sh
python -m coastal_wave tune --config configs/training.yaml --tuning-config configs/tuning.yaml --device cpu --n-trials 1
```

Optuna storage and result paths must point to writable locations. Existing study databases are intentionally not included. Trial export preserves the study's parameter replay rather than relying on potentially stale best-parameter files. No full optimization study was performed for the refactor.

For a different dataset, keep its site identifiers consistent across YAML, CSV filenames, static rows and sidecars; update feature names and physical conventions explicitly; regenerate geometry and prepared data; and train a new checkpoint when the input/output schema changes. Merely changing the config passed to an old checkpoint is not a supported schema migration.

Troubleshooting: missing CSVs usually indicate a directory or filename/site mismatch; missing static features indicate disabled/incomplete geometry; site leakage errors require disjoint site partitions. Do not bypass those checks. For historical temporal-only configurations rejected by the current trainer's site-overlap check, see the preserved behavior notes in the refactor report.
