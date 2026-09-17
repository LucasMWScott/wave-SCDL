# Notebook use and audit

Install `python -m pip install -e ".[geometry,ml,notebooks]"`, then launch `jupyter lab` from the project root. Select the environment where the project is installed. Imports use the installed package; no sys.path editing or notebook-local fake src package is needed.

Start with `notebooks/00_synthetic_geometry.ipynb`. It generates its own inputs in a temporary directory, runs the actual geometry pipeline, displays three site summaries, and cleans up. It was executed in a fresh kernel; saved outputs remain cleared.

The numbered workflow is deliberate:

1. `00_synthetic_geometry` runs without research data.
2. `01_preprocess_walkthrough` through `06_bathymetry_patches` inspect raw data, prepared tensors, routing, and bathymetry products.
3. `07_model_smoke_test` through `10_model_attention` check a model/checkpoint and its runtime contract.
4. `11_evaluation_overview` through `13_direction_errors` visualize predictions and site-level performance.
5. `14_static_explainability` through `18_worst_sites` are optional post-training interpretation notebooks.

Research notebooks require the external artifacts named in their configuration cells. Prepared-data QA needs matching `point_centric_*.npz` and metadata; routing/ray QA needs grid NPZ, routing pickle, rays, and sites; evaluation needs a selected results directory and prediction CSV; explainability also needs a checkpoint and training artifacts. No research result directories or trained checkpoints were copied, so only the synthetic notebook was executed. All retained notebooks were parsed, cleared, and had saved outputs removed.

The project retains no duplicate analysis variants, exploratory satellite/correlation exports, old categorical QA alternatives, or obsolete generated notebook copies. The complete removal and retention rationale is in [the refactor report](refactor_report.md).
