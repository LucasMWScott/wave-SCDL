"""Generate explicitly synthetic coastal inputs and run the geometry workflow."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def create_inputs(output: Path) -> dict:
    """Create a 41 by 41 bathymetry tile and hourly toy wave records."""
    from pyproj import Transformer

    output = output.resolve()
    (output / "bathy").mkdir(parents=True, exist_ok=True)
    (output / "nora3").mkdir(exist_ok=True)
    (output / "norac").mkdir(exist_ok=True)
    x = 300000.0 + np.arange(41) * 100
    y = 6600000.0 + np.arange(41) * 100
    xx, yy = np.meshgrid(x, y)
    z = -10.0 - (xx - x[0]) / 200
    z[:, :3] = 2.0
    np.savetxt(
        output / "bathy" / "synthetic.xyz",
        np.column_stack([xx.ravel(), yy.ravel(), z.ravel()]),
        fmt="%.3f",
    )
    to_geo = Transformer.from_crs(32633, 4326, always_xy=True)

    def site(name, col, row):
        lon, lat = to_geo.transform(x[col], y[row])
        return {"name": name, "lon": float(lon), "lat": float(lat)}

    offshore = [site(f"nora3_grid_{i}", 33, row) for i, row in enumerate([8, 20, 32])]
    nearshore = [
        site(f"synthetic_site_{i}", col, row)
        for i, (col, row) in enumerate([(8, 12), (12, 20), (16, 28)])
    ]
    for entry in nearshore:
        entry["paired_offshore"] = [offshore[0]["name"]]
    sites = {
        "path_base": "config",
        "backend": {"nora3_params_dir": "nora3", "norac_params_dir": "norac"},
        "offshore_sites": offshore,
        "nearshore_sites": nearshore,
    }
    prep = {
        "path_base": "config",
        "paths": {"processed_dir": "processed"},
        "static_features": {
            "master_out_path": "master_static_features.csv",
            "route_curtain": {
                "mode": "local_k_nearest",
                "use_multi_source_config": True,
                "padding_m": 300,
                "qa_out_path": "route_curtain_qa.csv",
            },
        },
        "circular": {"input_degrees": True},
    }
    train = {
        "path_base": "config",
        "data": {
            "sites_config": "sites.yaml",
            "point_centric_dir": "processed",
            "static_features_csv": "master_static_features.csv",
            "sequence_window": 4,
            "num_workers": 0,
            "pin_memory": False,
            "train_sites": ["synthetic_site_0"],
            "validation_sites": ["synthetic_site_1"],
            "test_sites": ["synthetic_site_2"],
            "use_static_features": False,
            "use_bathymetry": False,
            "use_geometry": False,
            "use_offshore_wind_features": False,
            "use_local_wind_features": False,
            "multi_source": {
                "enabled": True,
                "k_nearest": 3,
                "source_type": "nora3_wave",
                "allow_padding": False,
            },
            "offshore_vars": ["hs", "tp", "Pdir", "thq"],
            "nearshore_vars": ["hs", "tp", "dir", "dp"],
            "direction_vars": ["Pdir", "thq", "dir", "dp"],
            "targets": {"mode": "physical"},
        },
        "model": {
            "architecture": "coastal_transformer",
            "coastal_transformer": {
                "use_static": False,
                "model_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.0,
                "multi_source": {"enabled": True, "use_geometry_features": False},
                "bathy": {"enabled": False},
            },
        },
        "split": {
            "train": 0.6,
            "val": 0.4,
            "test": 0.4,
            "site_holdout_temporal_mode": "shared_recent",
        },
        "normalization": {"default_method": "zscore"},
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "max_train_batches": 2,
            "max_val_batches": 2,
            "seed": 42,
            "optimizer": {"type": "adam", "lr": 0.001},
            "loss": {"type": "blueprint_hybrid"},
            "progress_bar": {"enabled": False},
        },
        "logging": {"output_dir": "results", "checkpoint_name": "demo.pt"},
    }
    t = np.arange(96, dtype=float)
    dates = pd.date_range("2020-01-01", periods=len(t), freq="h")
    for i, entry in enumerate(offshore):
        pd.DataFrame(
            {
                "time": dates,
                "hs": 1.5 + 0.3 * np.sin(t / 8 + i),
                "tp": 6 + np.sin(t / 10),
                "Pdir": (250 + t) % 360,
                "thq": (240 + t) % 360,
            }
        ).to_csv(output / "nora3" / (entry["name"] + ".csv"), index=False)
    for i, entry in enumerate(nearshore):
        pd.DataFrame(
            {
                "time": dates,
                "hs": 0.7 + 0.2 * np.sin(t / 8 + i),
                "tp": 5 + np.sin(t / 10),
                "dir": (70 + t) % 360,
                "dp": (60 + t) % 360,
            }
        ).to_csv(output / "norac" / (entry["name"] + ".csv"), index=False)
    for name, payload in [("sites", sites), ("preprocess", prep), ("training", train)]:
        (output / (name + ".yaml")).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    return {"output": output}


def run_geometry(output: Path) -> Path:
    """Run real geometry algorithms on synthetic bathymetry and sites."""
    from geometric_builder.src.build_features import run_pipeline

    return run_pipeline(
        project_root=output,
        sites_rel="sites.yaml",
        preprocess_config_rel="preprocess.yaml",
        training_config_rel="training.yaml",
        bathy_dir_rel="bathy",
        bathy_rel=None,
        bathy_out_dir_rel="grids",
        full_name="full.npz",
        subgrid_name="subgrid.npz",
        routing_output_rel="routes.pkl",
        ray_output_rel="rays.csv",
        master_output_rel="master_static_features.csv",
        target_epsg=32633,
        padding_m=300,
        resolution_m=100,
        bottleneck_weight=1.0,
        bottleneck_epsilon_m=1.0,
        angle_step_deg=22.5,
        ray_step_m=100,
        max_ray_m=2000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory to avoid overwriting an existing run")
    create_inputs(output)
    if not args.inputs_only:
        print(run_geometry(output))
    print("Synthetic example only; these values are not scientific validation.")
