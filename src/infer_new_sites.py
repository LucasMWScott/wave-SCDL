"""Target-free forward adapter for a trained point-centric model.

The adapter deliberately validates raw NORA3 delivery before constructing any
dataset.  A prepared-artifact mode is also provided for smoke tests and for
deployments that have already run the project preprocessing functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config_resolution import resolve_config
from src.runtime_paths import normalize_runtime_config_paths
from src.models import build_model_from_config
from src.point_centric_pipeline import find_param_files
from src.point_centric_pipeline import (
    load_site_timeseries,
    _build_multisource_dynamic_artifacts,
    _load_master_static_frame,
    apply_nora3_wave_direction_offset,
)
from src.multi_source import build_k_nearest_source_metadata
from src.preprocessing.normalize import (
    restore_static_transformer_artifacts,
    transform_master_static_features,
)
from src.preprocessing.transfer_targets import circular_weighted_mean_deg
from src.preprocessing.transfer_targets import reconstruct_physical_from_transfer
from src.evaluate import _extract_state_dict, _load_checkpoint_compat
from src.transfer_runtime import load_transfer_scaler_stats, decode_transfer_predictions_numpy


def _read_yaml(path):
    """Read configuration paths relative to their declaring file."""
    from coastal_wave.common.config import read_config

    return read_config(path)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _required_sources(sites_yaml: Path) -> tuple[list[str], list[str], list[str]]:
    sites = _read_yaml(sites_yaml)
    wave = [
        str(x["name"])
        for x in sites.get("offshore_sites", [])
        if str(x.get("name", "")).startswith("nora3_grid_")
    ]
    wind = [
        str(x["name"])
        for x in sites.get("offshore_sites", [])
        if str(x.get("name", "")).startswith("nora3_wind_grid_")
    ]
    targets = [str(x["name"]) for x in sites.get("nearshore_sites", [])]
    return wave, wind, targets


def _validate_source_files(
    params_dir: Path, sites_yaml: Path
) -> tuple[list[str], list[str], list[str]]:
    wave, wind, targets = _required_sources(sites_yaml)
    missing = {}
    for name in wave + wind:
        files = find_param_files(name, str(params_dir))
        if not files:
            missing[name] = str(params_dir / f"*{name}*.csv")
    if missing:
        details = ", ".join(
            f"{name} (expected {pattern})" for name, pattern in sorted(missing.items())
        )
        raise FileNotFoundError(
            "Target-free inference cannot start because required NORA3 parameter CSVs are missing: "
            + details
        )
    return wave, wind, targets


def _build_raw_inputs(cfg: dict, pc_meta: dict, train_meta: dict):
    sites_path = Path(cfg["sites_config"])
    sites = _read_yaml(sites_path)
    targets = [dict(x) for x in sites.get("nearshore_sites", [])]
    data_cfg = (train_meta.get("config", {}) or {}).get("data", {}) or {}
    waves = [
        dict(x)
        for x in sites.get("offshore_sites", [])
        if str(x.get("name", "")).startswith("nora3_grid_")
    ]
    winds = [
        dict(x)
        for x in sites.get("offshore_sites", [])
        if str(x.get("name", "")).startswith("nora3_wind_grid_")
    ]
    params = Path(cfg["nora3_params_dir"])
    if not params.is_absolute():
        params = Path.cwd() / params
    wave_frames = {
        str(e["name"]): apply_nora3_wave_direction_offset(
            load_site_timeseries(str(e["name"]), str(params)),
            data_cfg.get("offshore_vars", []),
        )
        for e in waves
    }
    wind_frames = {str(e["name"]): load_site_timeseries(str(e["name"]), str(params)) for e in winds}
    frames = [f for f in list(wave_frames.values()) + list(wind_frames.values()) if not f.empty]
    if not frames:
        raise ValueError("No readable NORA3 parameter rows were found")
    index = frames[0].index
    for frame in frames[1:]:
        index = index.intersection(frame.index)
    index = index.sort_values()
    if len(index) < 31:
        raise ValueError(f"Only {len(index)} aligned timestamps available; sequence length is 30")
    offshore_entries = waves
    near_entries = targets
    source_meta, warnings = build_k_nearest_source_metadata(
        nearshore_entries=near_entries,
        offshore_entries=offshore_entries,
        k_nearest=3,
        source_type="nora3_wave",
        weight_power=1.0,
        max_distance_km_warn=80.0,
        max_distance_km_error=150.0,
        allow_padding=False,
    )
    # Resolve each target's nearest configured local-wind point.
    local_by_site = {}
    for target in targets:
        nearest = min(
            winds,
            key=lambda w: (
                (float(target["lat"]) - float(w["lat"])) ** 2
                + (float(target["lon"]) - float(w["lon"])) ** 2
            ),
        )
        local_by_site[str(target["name"])] = wind_frames[str(nearest["name"])]
    training_pc = Path(
        str((train_meta.get("config", {}) or {}).get("data", {}).get("point_centric_dir", ""))
    )
    if not training_pc.is_absolute():
        training_pc = Path.cwd() / training_pc
    expected = list(pc_meta.get("source_feature_names", []))
    include_local_direction_features = any(
        str(name).startswith(("fetch_at_", "blocking_at_", "slope_at_")) for name in expected
    )
    include_local_wind_features = any(
        str(name) in {"local_wind_speed_10m", "local_wind_dir_sin", "local_wind_dir_cos"}
        for name in expected
    )
    use_source_geometry_features = bool(pc_meta.get("source_geometry_feature_names", []))
    needs_static = bool(pc_meta.get("static_feature_names", [])) or include_local_direction_features
    raw_static = None
    transformed = pd.DataFrame({"site_name": [str(x["name"]) for x in targets]})
    if needs_static:
        static_csv = Path(cfg["static_features_csv"])
        if not static_csv.is_absolute():
            static_csv = Path.cwd() / static_csv
        raw_static = _load_master_static_frame(str(static_csv), [str(x["name"]) for x in targets])
        static_meta = _load_json(training_pc / "point_centric_metadata.json")["normalization"][
            "static_scaler"
        ]
        # Match training's raw-column exclusions before feeding static direction features.
        ignored = set(
            (train_meta.get("config", {}) or {}).get("data", {}).get("static_ignore_columns", [])
            or []
        )
        raw_static = raw_static.drop(columns=[c for c in ignored if c in raw_static.columns])
        static_art = restore_static_transformer_artifacts(static_meta)
        transformed = transform_master_static_features(raw_static, static_art)
    source_scaler = (
        _load_json(training_pc / "point_centric_metadata.json")
        .get("normalization", {})
        .get("source_dynamic_scaler", {})
        or {}
    )
    if "mean" not in source_scaler:
        recovered = Path(
            cfg.get("source_normalization_metadata", "missing_source_normalization_metadata.json")
        )
        if recovered.exists():
            source_scaler = (
                _load_json(recovered).get("normalization", {}).get("source_dynamic_scaler", {})
                or {}
            )
    prep_cfg = _read_yaml(Path(cfg["preprocess_config"]))
    source_norm = __import__(
        "src.preprocessing.normalize", fromlist=["select_dynamic_columns_for_scaling"]
    )
    scale_columns = source_norm.select_dynamic_columns_for_scaling(expected)
    if len(source_scaler.get("mean", [])) != len(scale_columns):
        available_columns = [str(name) for name in source_scaler.get("columns", [])]
        missing_stats = [name for name in scale_columns if name not in available_columns]
        if missing_stats:
            raise ValueError(
                "Source normalization statistics do not cover the checkpoint input features: "
                f"{missing_stats}"
            )
        stats_indices = [available_columns.index(name) for name in scale_columns]
        source_scaler = {
            **source_scaler,
            "columns": scale_columns,
            "mean": [source_scaler["mean"][idx] for idx in stats_indices],
            "std": [source_scaler["std"][idx] for idx in stats_indices],
        }
    xsrc, names, geom, geom_names, _ = _build_multisource_dynamic_artifacts(
        source_metadata=source_meta,
        prepared_offshore_by_site=wave_frames,
        local_wind_by_site=local_by_site,
        master_static_df=raw_static,
        aligned_index=index,
        offshore_vars=data_cfg.get("offshore_vars", []),
        direction_vars=set(data_cfg.get("direction_vars", [])),
        auto_dynamic_direction_vars=set(),
        input_degrees=True,
        norm_mod=source_norm,
        method="zscore",
        method_cfg={},
        feature_range=(0.0, 1.0),
        split_fit_idx=np.arange(len(index)),
        normalization_train_sites=[str(x["name"]) for x in targets],
        max_distance_km_error=150.0,
        use_source_geometry_features=use_source_geometry_features,
        include_local_direction_features=include_local_direction_features,
        include_local_wind_features=include_local_wind_features,
        scaler_override=source_scaler,
    )
    if names != expected:
        raise ValueError(f"Source feature order mismatch: got {names}, expected {expected}")
    return xsrc, geom, transformed, source_meta, index, targets


def load_trained_model(
    config_path: Path, checkpoint_path: Path, training_metadata_path: Path, device: torch.device
):
    """Recreate the architecture from the training config and load strictly."""
    cfg = normalize_runtime_config_paths(
        resolve_config(_read_yaml(config_path)), config_path=config_path
    )
    meta = _load_json(training_metadata_path)
    pc_dir = Path(str((meta.get("config", {}) or {}).get("data", {}).get("point_centric_dir", "")))
    if not pc_dir.is_absolute():
        pc_dir = Path.cwd() / pc_dir
    pc_meta = _load_json(pc_dir / "point_centric_metadata.json")
    dynamic_dim = int(pc_meta["X_dynamic_shape"][1])
    # Prefer the checkpoint's projection width when older metadata predates
    # two persisted source channels.
    source_dim = len(pc_meta.get("source_feature_names", [])) or None
    static_dim = len(pc_meta.get("static_feature_names", []))
    geometry_dim = len(pc_meta.get("source_geometry_feature_names", [])) or None
    checkpoint = _load_checkpoint_compat(checkpoint_path, device)
    state = _extract_state_dict(checkpoint)
    projection = state.get("multi_source_encoder.source_proj.weight")
    if projection is not None:
        source_dim = int(projection.shape[1])
    model = build_model_from_config(
        config=cfg,
        dynamic_input_dim=dynamic_dim,
        static_input_dim=static_dim,
        output_dim=4,
        dynamic_feature_names=pc_meta.get("dynamic_feature_names"),
        source_dynamic_input_dim=source_dim,
        source_geometry_input_dim=geometry_dim,
        source_feature_names=pc_meta.get("source_feature_names"),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, pc_meta


def _predict_prepared(
    model, prepared_dir: Path, output: Path, batch_size: int, device: torch.device
) -> pd.DataFrame:
    """Run target-free inference on already-prepared tensors (no Y is read)."""
    metadata = _load_json(prepared_dir / "point_centric_metadata.json")
    source = np.load(prepared_dir / "point_centric_X_dynamic_sources.npz", allow_pickle=False)
    static = np.load(prepared_dir / "point_centric_X_static.npz", allow_pickle=True)
    bathy = np.load(prepared_dir / "point_centric_X_bathy.npz", allow_pickle=True)
    geometry = np.load(prepared_dir / "point_centric_source_geometry.npz", allow_pickle=False)
    xsrc = source["X_dynamic_sources"]
    names = [str(x) for x in source["target_sites"].tolist()]
    timestamps = pd.to_datetime(source["timestamps"].astype(str))
    static_by_site = {
        str(k).split("Xstatic__", 1)[1]: static[k]
        for k in static.files
        if k.startswith("Xstatic__")
    }
    bathy_by_site = {
        str(k).split("Xbathy__", 1)[1]: bathy[k] for k in bathy.files if k.startswith("Xbathy__")
    }
    geom = geometry["source_geometry"]
    print(
        f"dynamic/source shape={xsrc.shape} static sites={len(static_by_site)} bathy sites={len(bathy_by_site)}"
    )
    rows = []
    for start in range(0, len(names), batch_size):
        end = min(start + batch_size, len(names))
        site_batch = names[start:end]
        # Prepared tensors are [site,time,...]; use the final sequence window.
        seq = torch.from_numpy(xsrc[start:end, -30:]).float().to(device)
        st = torch.from_numpy(np.stack([static_by_site[s] for s in site_batch])).float().to(device)
        bt = torch.from_numpy(np.stack([bathy_by_site[s] for s in site_batch])).float().to(device)
        gg = torch.from_numpy(geom[start:end]).float().to(device)
        with torch.inference_mode():
            pred = model(None, x_static=st, x_bathy=bt, x_dynamic_sources=seq, source_geometry=gg)
        if not isinstance(pred, dict) or not all(
            k in pred for k in ("log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg")
        ):
            raise RuntimeError(
                "Prepared inference requires the transfer-output model representation"
            )
        values = np.column_stack(
            [
                pred[k].detach().cpu().numpy().reshape(-1)
                for k in ("log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg")
            ]
        )
        # A target-free prepared artifact must carry references explicitly.
        ref_path = prepared_dir / "point_centric_reference_inputs.npz"
        if not ref_path.exists():
            raise FileNotFoundError(
                "Missing point_centric_reference_inputs.npz required to decode transfer outputs"
            )
        ref = np.load(ref_path, allow_pickle=False)["reference"]
        physical = reconstruct_physical_from_transfer(
            values, ref[start:end], tp_min=0.5, tp_max=30.0
        )
        for i, site in enumerate(site_batch):
            for j, ts in enumerate(timestamps):
                rows.append(
                    {
                        "timestamp": ts,
                        "site_id": site,
                        "pred_hs": physical[i, 0],
                        "pred_tp": physical[i, 1],
                        "pred_mean_direction": physical[i, 2] % 360.0,
                        "pred_peak_direction": physical[i, 3] % 360.0,
                    }
                )
    result = pd.DataFrame(rows).sort_values(["site_id", "timestamp"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--prepared-dir",
        default=None,
        help="Optional target-free prepared tensor directory for smoke tests",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = _read_yaml(cfg_path)
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(1)
    checkpoint = Path(cfg["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    train_cfg = Path(cfg["training_config"])
    if not train_cfg.is_absolute():
        train_cfg = Path.cwd() / train_cfg
    model_dir = checkpoint.parent
    model, _, _ = load_trained_model(
        train_cfg, checkpoint, model_dir / "training_run_metadata.json", device
    )
    if args.prepared_dir:
        result = _predict_prepared(
            model,
            Path(args.prepared_dir),
            Path(cfg.get("output_dir", "results/inference")) / "predictions.csv",
            args.batch_size,
            device,
        )
        print(f"wrote {len(result)} rows")
        return
    params_dir = Path(cfg["nora3_params_dir"])
    if not params_dir.is_absolute():
        params_dir = Path.cwd() / params_dir
    _validate_source_files(params_dir, Path(cfg["sites_config"]))
    train_meta = _load_json(model_dir / "training_run_metadata.json")
    training_pc = Path(
        str((train_meta.get("config", {}) or {}).get("data", {}).get("point_centric_dir", ""))
    )
    if not training_pc.is_absolute():
        training_pc = Path.cwd() / training_pc
    pc_meta = _load_json(training_pc / "point_centric_metadata.json")
    xsrc, geom, static_df, source_meta, index, targets = _build_raw_inputs(cfg, pc_meta, train_meta)
    expected_source_dim = int(model.source_dynamic_input_dim)
    if xsrc.shape[-1] + 2 == expected_source_dim:
        dates = pd.to_datetime(index)
        phase = 2.0 * np.pi * (dates.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
        seasonal = np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)
        seasonal = np.broadcast_to(
            seasonal[None, :, None, :], (xsrc.shape[0], xsrc.shape[1], xsrc.shape[2], 2)
        )
        xsrc = np.concatenate([xsrc, seasonal], axis=-1)
    if xsrc.shape[-1] != expected_source_dim:
        raise ValueError(
            f"Built source tensor has {xsrc.shape[-1]} features but checkpoint expects {expected_source_dim}"
        )
    # Match the runtime dataset's finite-value contract using feature medians.
    flat = xsrc.reshape(-1, xsrc.shape[-1])
    for j in range(flat.shape[1]):
        bad = ~np.isfinite(flat[:, j])
        if bad.any():
            vals = flat[~bad, j]
            flat[bad, j] = np.median(vals) if vals.size else 0.0
    xsrc = flat.reshape(xsrc.shape).astype(np.float32, copy=False)
    if geom is not None:
        geom = np.nan_to_num(geom.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    use_static_features = bool(getattr(model, "use_static_features", False))
    use_bathymetry = bool(getattr(model, "use_bathymetry", False))
    use_source_geometry_features = bool(getattr(model, "use_source_geometry_features", False))
    if use_static_features:
        static_df.iloc[:, 1:] = static_df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        static_map = {str(x): i for i, x in enumerate(static_df["site_name"].tolist())}
    else:
        static_map = {}
    bathy_npz = None
    bathy_map = {}
    bathy_order = []
    if use_bathymetry:
        bathy_npz = np.load(Path(cfg["bathymetry_patches"]), allow_pickle=True)
        expected_bathy_channels = int(model.bathy_encoder.in_channels)
        bathy_names = [str(x) for x in bathy_npz["channel_names"].tolist()]
        expected_bathy_names = [
            "depth",
            "land_sea_mask",
            "slope_magnitude",
            "curvature_laplacian",
            "distance_to_land",
        ]
        if expected_bathy_channels != len(expected_bathy_names):
            raise ValueError(
                f"Unsupported checkpoint bathymetry channel count: {expected_bathy_channels}"
            )
        missing_bathy = [x for x in expected_bathy_names if x not in bathy_names]
        if missing_bathy:
            raise ValueError(
                f"Haugesund bathymetry is missing checkpoint channels: {missing_bathy}"
            )
        bathy_order = [bathy_names.index(x) for x in expected_bathy_names]
        if bathy_names != expected_bathy_names:
            print(f"reordering bathymetry channels {bathy_names} -> {expected_bathy_names}")
        bathy_sites = [str(x) for x in bathy_npz["target_sites"].tolist()]
        bathy_map = {s: i for i, s in enumerate(bathy_sites)}
    seq_len = int((train_meta.get("config", {}) or {}).get("data", {}).get("sequence_window", 30))
    transfer_stats = load_transfer_scaler_stats(str(training_pc))
    static_shape = (len(targets), static_df.shape[1] - 1) if use_static_features else None
    bathy_shape = bathy_npz["X_bathy"].shape if bathy_npz is not None else None
    print(
        f"dynamic shape={xsrc.shape} static shape={static_shape} bathymetry shape={bathy_shape} sequence_length={seq_len}"
    )
    rows = []
    for site_i, target in enumerate(targets):
        site = str(target["name"])
        if use_static_features and site not in static_map:
            raise ValueError(f"Missing static row for {site}")
        if use_bathymetry and site not in bathy_map:
            raise ValueError(f"Missing bathymetry row for {site}")
        src_names = [str(x) for x in source_meta["source_names"][site_i]]
        frames = [
            apply_nora3_wave_direction_offset(
                load_site_timeseries(s, str(params_dir)),
                (train_meta.get("config", {}) or {}).get("data", {}).get("offshore_vars", []),
            ).reindex(index)
            for s in src_names
        ]
        parts = []
        for frame in frames:
            se = np.square(frame["hs_swell"].to_numpy(float))
            we = np.square(frame["hs_sea"].to_numpy(float))
            total = np.where(se + we > 0, se + we, 1.0)
            parts.append(
                np.column_stack(
                    [
                        np.sqrt(se + we),
                        (
                            se * frame["tp_swell"].to_numpy(float)
                            + we * frame["tp_sea"].to_numpy(float)
                        )
                        / total,
                        frame["thq_swell"].to_numpy(float),
                        frame["thq_sea"].to_numpy(float),
                    ]
                )
            )
        stack = np.stack(parts, axis=1)
        weights = np.broadcast_to(
            np.asarray(source_meta["weights"][site_i], float)[None, :], stack[:, :, 0].shape
        )
        ref = np.column_stack(
            [
                np.sum(stack[:, :, 0] * weights, 1),
                np.sum(stack[:, :, 1] * weights, 1),
                circular_weighted_mean_deg(stack[:, :, 2], weights, 1),
                circular_weighted_mean_deg(stack[:, :, 3], weights, 1),
            ]
        )
        times = list(range(seq_len - 1, len(index)))
        windows = np.stack([xsrc[site_i, t - seq_len + 1 : t + 1] for t in times])
        for start in range(0, len(times), args.batch_size):
            end = min(start + args.batch_size, len(times))
            seq = torch.from_numpy(windows[start:end]).float().to(device)
            static = None
            if use_static_features:
                static = (
                    torch.from_numpy(
                        np.repeat(
                            static_df.iloc[static_map[site], 1:].to_numpy(float)[None],
                            end - start,
                            axis=0,
                        )
                    )
                    .float()
                    .to(device)
                )
            bathy = None
            if use_bathymetry:
                bathy = (
                    torch.from_numpy(
                        np.repeat(
                            bathy_npz["X_bathy"][bathy_map[site], bathy_order][None],
                            end - start,
                            axis=0,
                        )
                    )
                    .float()
                    .to(device)
                )
            geometry = None
            if use_source_geometry_features:
                geometry = (
                    torch.from_numpy(np.repeat(geom[site_i][None], end - start, axis=0))
                    .float()
                    .to(device)
                )
            with torch.inference_mode():
                pred = model(
                    None,
                    x_static=static,
                    x_bathy=bathy,
                    x_dynamic_sources=seq,
                    source_geometry=geometry,
                )
            scaled = np.column_stack(
                [
                    pred[k].detach().cpu().numpy().reshape(-1)
                    for k in ("log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg")
                ]
            )
            physical = reconstruct_physical_from_transfer(
                decode_transfer_predictions_numpy(scaled, transfer_stats),
                ref[times[start:end]],
                tp_min=0.5,
                tp_max=30.0,
            )
            for j, t in enumerate(times[start:end]):
                rows.append(
                    {
                        "timestamp": index[t],
                        "site_id": site,
                        "latitude": target["lat"],
                        "longitude": target["lon"],
                        "pred_hs": physical[j, 0],
                        "pred_tp": physical[j, 1],
                        "pred_mean_direction": physical[j, 2] % 360.0,
                        "pred_peak_direction": physical[j, 3] % 360.0,
                        "ref_hs": ref[t, 0],
                        "ref_tp": ref[t, 1],
                        "ref_mean_direction": ref[t, 2],
                        "ref_peak_direction": ref[t, 3],
                        "wave_sources": "|".join(src_names),
                    }
                )
    result = pd.DataFrame(rows).sort_values(["site_id", "timestamp"])
    if not np.isfinite(
        result[["pred_hs", "pred_tp", "pred_mean_direction", "pred_peak_direction"]].to_numpy(float)
    ).all():
        raise ValueError("Non-finite prediction generated")
    out = Path(
        cfg.get(
            "output_csv",
            Path(cfg.get("output_dir", "results/inference")) / "18_cnn5chan_v1_predictions.csv",
        )
    )
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"wrote {len(result)} rows to {out}")


if __name__ == "__main__":
    main()
