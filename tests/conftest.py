"""Separate optional research artifacts from portable regression tests."""

from pathlib import Path
import sys
import pytest


def pytest_collection_modifyitems(items):
    missing_results = not (
        Path(__file__).resolve().parents[1] / "results/prodset_win_tuned_1"
    ).exists()
    artifact_tests = {
        "test_grouped_regime_helpers_run_on_non_expert_checkpoint",
        "test_join_prediction_static_and_attribution_summaries_uses_head_specific_columns",
        "test_load_prediction_frame_derives_direction_degrees",
        "test_load_results_bundle_supports_current_run_layout",
        "test_resolve_active_branches_matches_current_manifest",
    }
    windows_handles = {
        "test_export_trial_config_writes_exact_replay_and_ignores_stale_best_params",
        "test_exported_config_matches_build_trial_config_on_behavior_fields",
        "test_save_best_params_writes_study_specific_file",
        "test_point_centric_preprocess_records_nearest_local_wind_metadata_and_feature_shapes",
        "test_point_centric_preprocess_shared_recent_uses_train_only_temporal_scaler_fit",
    }
    for item in items:
        if missing_results and item.name in artifact_tests:
            item.add_marker(
                pytest.mark.skip(
                    reason="Requires external results/prodset_win_tuned_1 checkpoint and artifacts"
                )
            )
        if item.name == "test_loader_aligns_physics_and_dataset_returns_cap_keys":
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason="Baseline: runtime branch settings omit physics cap keys; see refactor report",
                )
            )
        if sys.platform == "win32" and item.name in windows_handles:
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    raises=PermissionError,
                    reason="Baseline Windows cleanup: open NumPy/Optuna file handles",
                )
            )
