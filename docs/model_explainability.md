# Model Explainability Guide

## Artifacts
The explainability notebooks expect the standard project result layout:

- `training_run_metadata.json`
- `observed_model_io.json`
- `predictions_<split>.csv` or `predictions_<split>.nc`
- point-centric preprocessing artifacts referenced from `data.point_centric_dir`

They reuse the current training and evaluation contracts instead of introducing a separate result format.

## Notebook Suite
- `notebooks/analysis_05_static_explainability.ipynb`
  Explains static site descriptors with grouped static SHAP, grouped permutation importance, thesis-ready ALE summaries, counterfactual curves, and static embedding plots. Its saved artifacts live under `<RESULTS_DIR>/static_explainability/`.
- `notebooks/analysis_06_temporal_explainability.ipynb`
  Explains dynamic forcing, lags, and sources with integrated gradients, temporal occlusion, feature-family occlusion, and attention summaries.
- `notebooks/analysis_07_cross_branch_explainability.ipynb`
  Compares branch importance and modality contributions, supports bathymetry attribution when enabled, and can compare multiple ablation runs.
- `notebooks/analysis_08_failure_mode_explainability.ipynb`
  Ranks failing sites, computes error-focused attributions, and produces rule-based per-site diagnosis summaries.

Most notebook outputs are written under:

- `<RESULTS_DIR>/explainability/`

The static explainability notebook now writes its saved figures and tables under:

- `<RESULTS_DIR>/static_explainability/`

Each notebook starts with the same editable config block and prints a run summary before computing outputs.

## Method Notes
- Static SHAP is computed against a context-averaged multimodal prediction target so static features are interpreted under representative dynamic forcing rather than in isolation.
- Dynamic explanations use integrated gradients and occlusion as the primary sensitivity tools.
- Attention should be treated as routing evidence, not proof of causal importance. Agreement or disagreement with gradients and occlusion is usually more informative than the attention map alone.
- Direction heads always use circular recovery and circular error handling. Raw logits are only used when the chosen explainability quantity explicitly requests logits, probabilities, or entropy.

## Branch and Ablation Compatibility
The shared diagnostics layer inspects the loaded run and adapts automatically to:

- static enabled or disabled
- bathymetry enabled or disabled
- source geometry enabled or disabled
- single-source or multi-source dynamic inputs
- physical or `physical_and_transfer` target mode
- feature subsets removed by ablation

When a branch is unavailable, the notebooks write a note and skip only the affected sections.

## Optional Dependencies
The notebook suite supports optional explainability packages:

- `captum`
- `shap`
- `umap-learn`

The code degrades gracefully when one is unavailable:

- missing `shap`: static SHAP falls back to a documented surrogate while grouped permutation importance still runs
- missing `umap-learn`: embedding plots fall back to PCA-only
- missing `captum`: the suite still works because the integrated gradients path is implemented directly in the diagnostics module

## Shared API
The reusable helpers live under `src.diagnostics` and are intended to keep the notebooks thin:

- results loading and manifest resolution
- prediction recovery
- branch and feature inference
- sampling and caching
- grouped static and dynamic explainability utilities
- plotting helpers
