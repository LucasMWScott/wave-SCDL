"""Geometry/ML boundary and portable artifact regression checks."""

import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from coastal_wave.common.config import read_config
from coastal_wave.common.sources import build_k_nearest_source_metadata
from coastal_wave.demo import create_inputs, run_geometry

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_matches_reference_and_has_no_torch_dependency(tmp_path):
    create_inputs(tmp_path)
    path = run_geometry(tmp_path)
    actual = pd.read_csv(path)
    expected = pd.read_csv(ROOT / "examples/fixtures/synthetic_master_expected.csv")
    # Projection libraries may differ in their last few floating-point bits.
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-8)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import geometric_builder.src.build_features; assert 'torch' not in sys.modules",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_config_paths_are_owned_by_declaring_file(tmp_path):
    parent = tmp_path / "base"
    parent.mkdir()
    (parent / "train.yaml").write_text(
        "path_base: config\ndata:\n  point_centric_dir: arrays\n", encoding="utf-8"
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "path_base: config\nextends: base/train.yaml\nlogging:\n  output_dir: output\n",
        encoding="utf-8",
    )
    cfg = read_config(child)
    assert Path(cfg["data"]["point_centric_dir"]) == parent / "arrays"
    assert Path(cfg["logging"]["output_dir"]) == tmp_path / "output"


def test_source_ties_and_site_order():
    sites = [{"name": "b", "lat": 60, "lon": 5}, {"name": "a", "lat": 60, "lon": 5}]
    sources = [{"name": "z", "lat": 60, "lon": 5}, {"name": "a", "lat": 60, "lon": 5}]
    result, _ = build_k_nearest_source_metadata(sites, sources, k_nearest=2)
    assert result["target_sites"] == ["b", "a"]
    assert result["source_names"] == [["a", "z"], ["a", "z"]]
    np.testing.assert_array_equal(result["weights"], [[0.5, 0.5], [0.5, 0.5]])


def test_saved_static_state_reused_without_refitting():
    from src.preprocessing.normalize import (
        fit_master_static_feature_transformer,
        transform_master_static_features,
        restore_static_transformer_artifacts,
    )

    frame = pd.DataFrame(
        {
            "site_name": ["train_a", "train_b", "test"],
            "path_length_m": [10.0, 20.0, 10000.0],
            "static_final_approach_deg": [0.0, 90.0, 180.0],
        }
    )
    art = fit_master_static_feature_transformer(frame, ["train_a", "train_b"])
    restored = restore_static_transformer_artifacts(json.loads(json.dumps(art.to_metadata())))
    before = transform_master_static_features(frame, art)
    after = transform_master_static_features(frame, restored)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    changed = frame.copy()
    changed.loc[2, "path_length_m"] = 1e8
    refit = fit_master_static_feature_transformer(changed, ["train_a", "train_b"])
    pd.testing.assert_frame_equal(
        before.iloc[:2], transform_master_static_features(changed, refit).iloc[:2], check_exact=True
    )
    # Protocol-zero GLOBAL reference emitted by older serialized transformers.
    legacy = pickle.loads(b"cpoint_centric_normalize\nCyclicalDegreesTransformer\n.")
    from src.preprocessing.normalize import CyclicalDegreesTransformer

    assert legacy is CyclicalDegreesTransformer


def test_checkpoint_parameter_names_and_reload(tmp_path):
    import torch
    from src.models import build_model_from_config

    cfg = {
        "data": {"use_static_features": False},
        "model": {
            "coastal_transformer": {
                "model_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.0,
            }
        },
    }
    torch.manual_seed(42)
    first = build_model_from_config(cfg, 4, 0, 6).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(first.state_dict(), checkpoint)
    second = build_model_from_config(cfg, 4, 0, 6).eval()
    second.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    x = torch.randn(2, 4, 4)
    with torch.no_grad():
        a, b = first(x), second(x)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
