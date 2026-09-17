#!/usr/bin/env python3
"""Optuna entrypoint for isolated hyperparameter tuning."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna
import torch
import yaml
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import TrialState


REPO_ROOT = Path(__file__).resolve().parents[1]

from tuning.objective import WaveTuningObjective


logger = logging.getLogger(__name__)


def sanitize_study_name(study_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(study_name).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "optuna_study"


def read_yaml(path):
    """Read YAML with inherited, config-relative paths."""
    from src.config_loader import read_yaml_config

    return read_yaml_config(path)


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_storage(storage_value: str, repo_root: Path) -> Tuple[str, Path | None]:
    """Resolve storage setting to an Optuna storage URL and local DB path if sqlite."""
    storage_raw = str(storage_value).strip()

    if storage_raw.startswith("sqlite:///"):
        local_part = storage_raw[len("sqlite:///") :]
        db_path = Path(local_part)
        if not db_path.is_absolute():
            db_path = (repo_root / db_path).resolve()
            return f"sqlite:///{db_path}", db_path
        return storage_raw, db_path

    if storage_raw.startswith("sqlite:////"):
        local_part = storage_raw[len("sqlite:////") :]
        db_path = Path("/" + local_part)
        return storage_raw, db_path

    # Treat plain path as sqlite file path.
    db_path = (repo_root / storage_raw).resolve()
    return f"sqlite:///{db_path}", db_path


def build_sampler(cfg: Dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_cfg = cfg.get("sampler", {}) or {}
    kind = str(sampler_cfg.get("type", "TPESampler")).lower()

    if kind == "tpesampler":
        return TPESampler(
            seed=sampler_cfg.get("seed", None),
            n_startup_trials=int(sampler_cfg.get("n_startup_trials", 10)),
            multivariate=bool(sampler_cfg.get("multivariate", True)),
            group=bool(sampler_cfg.get("group", False)),
        )

    if kind == "randomsampler":
        return RandomSampler(seed=sampler_cfg.get("seed", None))

    raise ValueError("sampler.type must be one of: TPESampler, RandomSampler")


def build_pruner(cfg: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_cfg = cfg.get("pruner", {}) or {}
    kind = str(pruner_cfg.get("type", "MedianPruner")).lower()

    if kind == "medianpruner":
        return MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 1)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )

    if kind == "successivehalvingpruner":
        return SuccessiveHalvingPruner(
            min_resource=int(pruner_cfg.get("min_resource", 1)),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 4)),
            min_early_stopping_rate=int(pruner_cfg.get("min_early_stopping_rate", 0)),
        )

    if kind == "hyperbandpruner":
        return HyperbandPruner(
            min_resource=int(pruner_cfg.get("min_resource", 1)),
            max_resource=pruner_cfg.get("max_resource", "auto"),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
        )

    if kind == "noppruner":
        return NopPruner()

    raise ValueError(
        "pruner.type must be one of: MedianPruner, SuccessiveHalvingPruner, HyperbandPruner, NopPruner"
    )


def save_optuna_visualizations(study: optuna.Study, out_dir: Path) -> None:
    """Save Optuna built-in visualizations as HTML files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
    }

    for name, builder in plots.items():
        out_path = out_dir / f"{name}.html"
        try:
            fig = builder(study)
            fig.write_html(str(out_path), include_plotlyjs="cdn")
            logger.info("Saved visualization: %s", out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s visualization: %s", name, exc)


def save_best_params(study: optuna.Study, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials found. best_params.yaml was not generated.")
        return

    best = study.best_trial
    resolved_params = best.user_attrs.get("resolved_params", None)
    payload = {
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_value": float(best.value),
        "params": dict(best.params),
        "resolved_params": resolved_params if resolved_params is not None else dict(best.params),
        "user_attrs": dict(best.user_attrs),
    }

    generic_out_path = out_dir / "best_params.yaml"
    write_yaml(generic_out_path, payload)
    logger.info(
        "Saved best params: %s (last-run study only; use the study-specific file for reproducible replay)",
        generic_out_path,
    )

    study_slug = sanitize_study_name(study.study_name)
    study_out_path = out_dir / f"{study_slug}_best_params.yaml"
    write_yaml(study_out_path, payload)
    logger.info("Saved study-specific best params: %s", study_out_path)


def log_trial_result(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == TrialState.FAIL])

    if trial.state == TrialState.COMPLETE:
        metric = trial.user_attrs.get("objective_metric", "objective")
        best_epoch = trial.user_attrs.get("best_epoch", "n/a")
        logger.info(
            "Study progress | trial=%d complete | value=%.6f | metric=%s | best_epoch=%s | complete=%d pruned=%d failed=%d",
            trial.number,
            float(trial.value),
            metric,
            best_epoch,
            completed,
            pruned,
            failed,
        )
        return

    if trial.state == TrialState.PRUNED:
        logger.info(
            "Study progress | trial=%d pruned | complete=%d pruned=%d failed=%d",
            trial.number,
            completed,
            pruned,
            failed,
        )
        return

    if trial.state == TrialState.FAIL:
        logger.warning(
            "Study progress | trial=%d failed | reason=%s | complete=%d pruned=%d failed=%d",
            trial.number,
            trial.user_attrs.get("failure_reason", "unknown"),
            completed,
            pruned,
            failed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna tuning for coastal-transformer wave model"
    )
    parser.add_argument("--config", default="configs/training.yaml", help="Base training config")
    parser.add_argument(
        "--tuning-config", default="configs/tuning.yaml", help="Optuna tuning config"
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = read_yaml(args.config)
    tuning_cfg = read_yaml(args.tuning_config)

    study_cfg = tuning_cfg.get("study", {}) or {}
    study_name = str(study_cfg.get("name", "wave_downscaling_optuna"))
    direction = str(study_cfg.get("direction", "minimize"))
    load_if_exists = bool(study_cfg.get("load_if_exists", True))

    n_trials = args.n_trials if args.n_trials is not None else tuning_cfg.get("n_trials", 30)
    timeout = args.timeout if args.timeout is not None else tuning_cfg.get("timeout", None)

    storage_raw = study_cfg.get("storage", "tuning/results/study.db")
    storage_url, db_path = resolve_storage(storage_raw, REPO_ROOT)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path(
        (tuning_cfg.get("logging", {}) or {}).get(
            "output_dir", Path(args.tuning_config).resolve().parent / "results"
        )
    ).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    sampler = build_sampler(tuning_cfg)
    pruner = build_pruner(tuning_cfg)
    device = resolve_device(args.device)

    logger.info("Starting Optuna study '%s'", study_name)
    logger.info("Storage: %s", storage_url)
    logger.info("Device: %s", device)
    logger.info("Trials: %s | Timeout: %s", n_trials, timeout)

    objective = WaveTuningObjective(
        base_config=base_cfg,
        tuning_config=tuning_cfg,
        device=device,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=None if n_trials is None else int(n_trials),
        timeout=None if timeout is None else int(timeout),
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[log_trial_result],
    )

    save_best_params(study, results_dir)
    save_optuna_visualizations(study, results_dir)

    logger.info("Tuning complete. Results written to %s", results_dir)


if __name__ == "__main__":
    main()
