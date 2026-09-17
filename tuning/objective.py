"""Optuna objective for validation-driven coastal model tuning."""

from __future__ import annotations

import copy
import gc
import logging
from typing import Any, Dict, Mapping

import optuna
import torch

try:
    from preprocessing.transfer_targets import resolve_targets_config
except Exception:
    from src.preprocessing.transfer_targets import resolve_targets_config

try:
    from config_resolution import resolve_config
except Exception:
    from src.config_resolution import resolve_config

try:
    from data_pipeline import (
        apply_runtime_static_ablation,
        build_split_dataloader,
        load_point_centric_arrays,
    )
    from losses import build_loss_from_config
    from models import build_model_from_config
    from train import (
        _resolve_runtime_bathy_request,
        build_optimizer,
        build_scheduler,
        compute_physical_score,
        resolve_breaking_physics_config,
        resolve_validation_every_n_epochs,
        resolve_selection_config,
        run_epoch,
        set_random_seed,
        should_run_validation_epoch,
    )
    from training.pcgrad_bridge import create_pcgrad_optimizer
except Exception:
    from src.data_pipeline import (
        apply_runtime_static_ablation,
        build_split_dataloader,
        load_point_centric_arrays,
    )
    from src.losses import build_loss_from_config
    from src.models import build_model_from_config
    from src.train import (
        _resolve_runtime_bathy_request,
        build_optimizer,
        build_scheduler,
        compute_physical_score,
        resolve_breaking_physics_config,
        resolve_validation_every_n_epochs,
        resolve_selection_config,
        run_epoch,
        set_random_seed,
        should_run_validation_epoch,
    )
    from src.training.pcgrad_bridge import create_pcgrad_optimizer


logger = logging.getLogger(__name__)


def _merge_nested(base: dict, overrides: Mapping[str, Any]) -> dict:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_nested(base[key], value)
        elif isinstance(value, Mapping):
            base[key] = _merge_nested({}, value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_nested(config: dict, path: str, value: Any) -> None:
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        raise ValueError("Config path must not be empty")

    cursor = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = copy.deepcopy(value)


def _get_nested(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in [part for part in str(path).split(".") if part]:
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _choice_label(choice: Any) -> str:
    if isinstance(choice, (list, tuple)):
        return ",".join(str(item) for item in choice)
    return str(choice)


def _resolve_search_groups(
    group_cfg: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> Dict[str, bool]:
    resolved: Dict[str, bool] = {}
    for name, raw_spec in (group_cfg or {}).items():
        if isinstance(raw_spec, bool):
            resolved[str(name)] = bool(raw_spec)
            continue

        spec = dict(raw_spec or {})
        enabled = bool(spec.get("enabled", True))
        for requirement in spec.get("requires", []) or []:
            req = dict(requirement or {})
            path = str(req.get("path", "")).strip()
            expected = req.get("equals", True)
            actual = _get_nested(config, path, default=None)
            if actual != expected:
                enabled = False
                break
        resolved[str(name)] = enabled
    return resolved


def _sample_param(trial: optuna.trial.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    kind = str(spec.get("type", "categorical")).strip().lower()

    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        if bool(spec.get("log", False)):
            return int(trial.suggest_int(name, low, high, log=True))
        return int(trial.suggest_int(name, low, high, step=int(spec.get("step", 1))))

    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if bool(spec.get("log", False)):
            return float(trial.suggest_float(name, low, high, log=True))
        step = spec.get("step", None)
        if step is None:
            return float(trial.suggest_float(name, low, high))
        return float(trial.suggest_float(name, low, high, step=float(step)))

    if kind == "categorical":
        choices = list(spec.get("choices", []) or [])
        if not choices:
            raise ValueError(f"search_space.{name}.choices must not be empty")
        return trial.suggest_categorical(name, choices)

    if kind == "bool":
        return bool(trial.suggest_categorical(name, [True, False]))

    if kind == "int_list":
        raw_choices = list(spec.get("choices", []) or [])
        if not raw_choices:
            raise ValueError(f"search_space.{name}.choices must not be empty")
        label_to_value = {
            _choice_label(choice): [int(item) for item in choice] for choice in raw_choices
        }
        label = trial.suggest_categorical(name, list(label_to_value.keys()))
        return label_to_value[str(label)]

    raise ValueError(
        f"Unsupported search-space type '{kind}' for parameter '{name}'. "
        "Supported: int, float, categorical, bool, int_list."
    )


def build_trial_config(
    base_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any],
    trial: optuna.trial.Trial,
) -> tuple[dict, Dict[str, Any], Dict[str, bool]]:
    config = resolve_config(copy.deepcopy(dict(base_config)))

    loop_cfg = dict((tuning_config.get("training_loop", {}) or {}))
    trial_overrides = dict((tuning_config.get("trial_overrides", {}) or {}))
    if trial_overrides:
        _merge_nested(config, trial_overrides)

    training_cfg = config.setdefault("training", {})
    early_cfg = training_cfg.setdefault("early_stopping", {})
    progress_cfg = training_cfg.setdefault("progress_bar", {})
    data_cfg = config.setdefault("data", {})

    if "epochs_per_trial" in loop_cfg:
        training_cfg["epochs"] = int(loop_cfg["epochs_per_trial"])
    if "max_train_batches" in loop_cfg:
        training_cfg["max_train_batches"] = (
            None if loop_cfg["max_train_batches"] is None else int(loop_cfg["max_train_batches"])
        )
    if "max_val_batches" in loop_cfg:
        training_cfg["max_val_batches"] = (
            None if loop_cfg["max_val_batches"] is None else int(loop_cfg["max_val_batches"])
        )
    if "validation_every_n_epochs" in loop_cfg:
        training_cfg["validation_every_n_epochs"] = int(loop_cfg["validation_every_n_epochs"])
    if "gradient_clip_norm" in loop_cfg:
        training_cfg["gradient_clip_norm"] = float(loop_cfg["gradient_clip_norm"])
    if "early_stop_patience" in loop_cfg:
        early_cfg["enabled"] = True
        early_cfg["patience"] = int(loop_cfg["early_stop_patience"])
    dataloader_cfg = dict((loop_cfg.get("dataloader", {}) or {}))
    if "num_workers" in dataloader_cfg:
        data_cfg["num_workers"] = int(dataloader_cfg["num_workers"])
    if "pin_memory" in dataloader_cfg:
        data_cfg["pin_memory"] = bool(dataloader_cfg["pin_memory"])

    progress_cfg["enabled"] = False
    progress_cfg["train"] = False
    progress_cfg["val"] = False

    search_groups = _resolve_search_groups(tuning_config.get("search_groups", {}), config)
    sampled_params: Dict[str, Any] = {}
    for name, raw_spec in (tuning_config.get("search_space", {}) or {}).items():
        spec = dict(raw_spec or {})
        group_name = str(spec.get("group", "")).strip()
        if group_name and not search_groups.get(group_name, False):
            continue

        value = _sample_param(trial, str(name), spec)
        sampled_params[str(name)] = copy.deepcopy(value)

        target_paths = list(spec.get("paths", []) or [])
        if not target_paths:
            raw_path = spec.get("path", None)
            if raw_path is not None:
                target_paths = [raw_path]
        if not target_paths:
            raise ValueError(f"search_space.{name} must define path or paths")

        for path in target_paths:
            _set_nested(config, str(path), value)

    config = resolve_config(config)
    return config, sampled_params, search_groups


def _infer_metric_mode(metric_name: str) -> str:
    normalized = str(metric_name).strip().lower()
    if normalized.endswith("_r2") or normalized == "physical_score_max":
        return "max"
    return "min"


def _is_improved(candidate: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "max":
        return (candidate - best) > min_delta
    return (best - candidate) > min_delta


def _worst_value(mode: str) -> float:
    return 1.0e12 if str(mode).strip().lower() == "min" else -1.0e12


class WaveTuningObjective:
    """Validation-driven Optuna objective using the training runtime stack."""

    def __init__(
        self,
        base_config: Mapping[str, Any],
        tuning_config: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.base_config = copy.deepcopy(dict(base_config))
        self.tuning_config = copy.deepcopy(dict(tuning_config))
        self.device = device
        self._arrays_cache: dict[tuple[str, tuple[str, ...] | None, int | None], Any] = {}
        logging_cfg = dict((self.tuning_config.get("logging", {}) or {}))
        self.log_epoch_interval = max(1, int(logging_cfg.get("epoch_interval", 1)))
        self.log_params = bool(logging_cfg.get("show_sampled_params", True))

    def _load_arrays(self, config: Mapping[str, Any]):
        requested_bathy_channels, requested_bathy_in_channels = _resolve_runtime_bathy_request(
            dict(config)
        )
        data_dir = str(((config.get("data", {}) or {}).get("point_centric_dir", "")))
        cache_key = (
            data_dir,
            None
            if requested_bathy_channels is None
            else tuple(str(name) for name in requested_bathy_channels),
            requested_bathy_in_channels,
        )
        if cache_key not in self._arrays_cache:
            self._arrays_cache[cache_key] = load_point_centric_arrays(
                data_dir,
                bathy_channels=requested_bathy_channels,
                bathy_in_channels=requested_bathy_in_channels,
            )
        return copy.deepcopy(self._arrays_cache[cache_key])

    def _resolve_objective_config(
        self,
        config: Mapping[str, Any],
        selection_weights: Mapping[str, float],
    ) -> tuple[str, str, Dict[str, float]]:
        objective_cfg = dict((self.tuning_config.get("objective", {}) or {}))
        metric = str(objective_cfg.get("metric", "val_loss")).strip()
        mode = str(objective_cfg.get("mode", _infer_metric_mode(metric))).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError("objective.mode must be 'min' or 'max'")

        weights_cfg = objective_cfg.get("physical_score_weights", None)
        if weights_cfg is None:
            weights = {str(key): float(value) for key, value in selection_weights.items()}
        else:
            weights = {str(key): float(value) for key, value in dict(weights_cfg).items()}
        return metric, mode, weights

    @staticmethod
    def _compute_objective_value(
        metric_name: str,
        val_loss: float,
        val_metrics: Mapping[str, float],
        physical_score_weights: Mapping[str, float],
    ) -> float:
        normalized = str(metric_name).strip().lower()
        if normalized == "val_loss":
            return float(val_loss)
        if normalized == "physical_score":
            return float(compute_physical_score(dict(val_metrics), dict(physical_score_weights)))
        if normalized.startswith("val_"):
            normalized = normalized[4:]
        if normalized not in val_metrics:
            raise KeyError(
                f"Objective metric '{metric_name}' was not found in validation metrics. "
                f"Available keys: {sorted(val_metrics)}"
            )
        return float(val_metrics[normalized])

    def __call__(self, trial: optuna.trial.Trial) -> float:
        selection_mode = "min"
        objective_mode = "min"

        model = None
        loss_fn = None
        optimizer = None
        scheduler = None
        pcgrad_optimizer = None

        try:
            config, sampled_params, active_groups = build_trial_config(
                self.base_config, self.tuning_config, trial
            )
            trial.set_user_attr("resolved_params", copy.deepcopy(sampled_params))
            trial.set_user_attr("active_groups", active_groups)

            training_cfg = config.get("training", {}) or {}
            data_cfg = config.get("data", {}) or {}
            deterministic = bool(training_cfg.get("deterministic", False))
            seed = training_cfg.get("seed", None)
            if seed is not None:
                set_random_seed(int(seed) + int(trial.number), deterministic=deterministic)

            if self.log_params:
                logger.info("Trial %d started | params=%s", trial.number, sampled_params)
            else:
                logger.info("Trial %d started", trial.number)

            arrays = apply_runtime_static_ablation(self._load_arrays(config))
            train_loader, train_ds = build_split_dataloader(
                arrays,
                config,
                split_name="train",
                shuffle=True,
                apply_train_sample_subsampling=True,
            )
            val_loader, val_ds = build_split_dataloader(
                arrays, config, split_name="val", shuffle=False
            )

            train_sites = set(getattr(train_ds, "sites", []) or [])
            val_sites = set(getattr(val_ds, "sites", []) or [])
            train_val_overlap = sorted(train_sites.intersection(val_sites))
            if train_val_overlap:
                raise RuntimeError(
                    "Split leakage detected in tuning: train/val site sets overlap. "
                    f"overlap={train_val_overlap}. "
                    "Use data.validation_sites for strict held-out validation."
                )

            if len(train_ds) == 0:
                raise RuntimeError("Training dataset is empty for the sampled trial config.")
            if len(val_ds) == 0:
                raise RuntimeError("Validation dataset is empty for the sampled trial config.")

            logger.info(
                "Trial %d data | train_samples=%d val_samples=%d train_sites=%d val_sites=%d",
                trial.number,
                len(train_ds),
                len(val_ds),
                len(getattr(train_ds, "sites", []) or []),
                len(getattr(val_ds, "sites", []) or []),
            )

            sample = train_ds[0]
            dynamic_input_dim = int(sample["x_dynamic"].shape[-1]) if "x_dynamic" in sample else 0
            source_dynamic_input_dim = (
                int(sample["x_dynamic_sources"].shape[-1])
                if "x_dynamic_sources" in sample
                else None
            )
            source_geometry_input_dim = (
                int(sample["source_geometry"].shape[-1]) if "source_geometry" in sample else None
            )
            static_input_dim = int(sample["x_static"].shape[-1]) if "x_static" in sample else 0
            output_dim = 4 if isinstance(sample["y"], dict) else int(sample["y"].shape[-1])

            model = build_model_from_config(
                config=config,
                dynamic_input_dim=dynamic_input_dim,
                static_input_dim=static_input_dim,
                output_dim=output_dim,
                dynamic_feature_names=getattr(arrays, "dynamic_feature_names", None),
                source_dynamic_input_dim=source_dynamic_input_dim,
                source_geometry_input_dim=source_geometry_input_dim,
                source_feature_names=getattr(arrays, "source_feature_names", None),
            ).to(self.device)

            loss_fn = build_loss_from_config(config)
            loss_trainable_params = [param for param in loss_fn.parameters() if param.requires_grad]
            optimizer, _optimizer_kind = build_optimizer(
                model,
                config,
                extra_parameters=loss_trainable_params,
            )
            scheduler, scheduler_kind, plateau_monitor = build_scheduler(optimizer, config)
            selection_monitor, selection_mode, selection_weights = resolve_selection_config(config)
            objective_metric, objective_mode, objective_weights = self._resolve_objective_config(
                config, selection_weights
            )
            targets_cfg = resolve_targets_config(data_cfg)
            breaking_cfg = resolve_breaking_physics_config(config)

            pcgrad_cfg = training_cfg.get("pcgrad", {}) or {}
            if bool(pcgrad_cfg.get("enabled", False)):
                try:
                    import torch_optimizer as _torch_optimizer  # noqa: F401
                except Exception as exc:
                    raise ImportError(
                        "training.pcgrad.enabled=true requires torch-optimizer. "
                        "Install dependency and retry."
                    ) from exc
                if not hasattr(loss_fn, "task_losses"):
                    raise ValueError(
                        "training.pcgrad.enabled=true requires a loss module with task_losses(pred, target)."
                    )
                pcgrad_optimizer = create_pcgrad_optimizer(
                    optimizer=optimizer,
                    reduction=str(pcgrad_cfg.get("reduction", "mean")),
                )

            epochs = int(training_cfg.get("epochs", 10))
            grad_clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
            max_train_batches = training_cfg.get("max_train_batches", None)
            max_val_batches = training_cfg.get("max_val_batches", None)
            validation_every_n_epochs = resolve_validation_every_n_epochs(config)
            max_train_batches = None if max_train_batches is None else int(max_train_batches)
            max_val_batches = None if max_val_batches is None else int(max_val_batches)

            early_cfg = training_cfg.get("early_stopping", {}) or {}
            early_enabled = bool(early_cfg.get("enabled", False))
            early_patience = int(early_cfg.get("patience", 10))
            early_min_delta = float(early_cfg.get("min_delta", 0.0))

            best_objective = _worst_value(objective_mode)
            best_epoch = 0
            best_val_loss = _worst_value("min")
            best_val_metrics: Dict[str, float] = {}
            best_selection_score = _worst_value(selection_mode)
            early_best_score = _worst_value(selection_mode)
            early_bad_epochs = 0

            for epoch in range(1, epochs + 1):
                train_loss, _train_metrics, _train_components = run_epoch(
                    model=model,
                    loader=train_loader,
                    loss_fn=loss_fn,
                    device=self.device,
                    optimizer=optimizer,
                    grad_clip_norm=grad_clip_norm,
                    pcgrad_optimizer=pcgrad_optimizer,
                    max_batches=max_train_batches,
                    point_centric_dir=data_cfg.get("point_centric_dir", ""),
                    transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
                    transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
                    epoch_index=epoch,
                    total_epochs=epochs,
                    progress_cfg={
                        "enabled": False,
                        "train": False,
                        "val": False,
                        "leave": False,
                        "update_interval": 1,
                    },
                    progress_phase="train",
                    breaking_enabled=bool(breaking_cfg.get("enabled", False)),
                    breaking_cap_key=str(breaking_cfg.get("cap_key", "local_breaking_hs_cap")),
                    breaking_valid_key=str(
                        breaking_cfg.get("valid_key", "local_breaking_cap_valid")
                    ),
                    breaking_penalty_weight=float(breaking_cfg.get("penalty_weight", 0.02)),
                    batch_observer=None,
                )
                validation_ran = should_run_validation_epoch(
                    epoch, epochs, validation_every_n_epochs
                )
                val_loss = None
                val_metrics: Dict[str, float] = {}
                objective_value = None
                selection_score = None
                if validation_ran:
                    val_loss, val_metrics, _val_components = run_epoch(
                        model=model,
                        loader=val_loader,
                        loss_fn=loss_fn,
                        device=self.device,
                        optimizer=None,
                        grad_clip_norm=0.0,
                        pcgrad_optimizer=None,
                        max_batches=max_val_batches,
                        point_centric_dir=data_cfg.get("point_centric_dir", ""),
                        transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
                        transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
                        epoch_index=epoch,
                        total_epochs=epochs,
                        progress_cfg={
                            "enabled": False,
                            "train": False,
                            "val": False,
                            "leave": False,
                            "update_interval": 1,
                        },
                        progress_phase="val",
                        breaking_enabled=bool(breaking_cfg.get("enabled", False)),
                        breaking_cap_key=str(breaking_cfg.get("cap_key", "local_breaking_hs_cap")),
                        breaking_valid_key=str(
                            breaking_cfg.get("valid_key", "local_breaking_cap_valid")
                        ),
                        breaking_penalty_weight=float(breaking_cfg.get("penalty_weight", 0.02)),
                        batch_observer=None,
                    )

                    selection_score = (
                        float(val_loss)
                        if selection_monitor == "val_loss"
                        else compute_physical_score(val_metrics, selection_weights)
                    )
                    objective_value = self._compute_objective_value(
                        objective_metric,
                        val_loss,
                        val_metrics,
                        objective_weights,
                    )

                if scheduler is not None:
                    if scheduler_kind == "plateau":
                        if plateau_monitor == "val_loss":
                            if validation_ran and val_loss is not None:
                                scheduler.step(val_loss)
                        else:
                            scheduler.step(train_loss)
                    else:
                        scheduler.step()

                if (
                    validation_ran
                    and objective_value is not None
                    and val_loss is not None
                    and selection_score is not None
                ):
                    trial.report(float(objective_value), step=epoch)
                    if (epoch % self.log_epoch_interval) == 0 or epoch == 1 or epoch == epochs:
                        logger.info(
                            "Trial %d | Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | objective(%s)=%.6f",
                            trial.number,
                            epoch,
                            epochs,
                            float(train_loss),
                            float(val_loss),
                            objective_metric,
                            float(objective_value),
                        )
                    if trial.should_prune():
                        trial.set_user_attr("pruned_epoch", int(epoch))
                        trial.set_user_attr("last_val_loss", float(val_loss))
                        trial.set_user_attr(
                            "last_val_metrics", {str(k): float(v) for k, v in val_metrics.items()}
                        )
                        logger.info(
                            "Trial %d pruned at epoch %d | val_loss=%.6f | objective(%s)=%.6f",
                            trial.number,
                            epoch,
                            float(val_loss),
                            objective_metric,
                            float(objective_value),
                        )
                        raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")

                    if _is_improved(objective_value, best_objective, objective_mode):
                        best_objective = float(objective_value)
                        best_epoch = int(epoch)
                        best_val_loss = float(val_loss)
                        best_val_metrics = {
                            str(key): float(value) for key, value in val_metrics.items()
                        }

                    if _is_improved(selection_score, best_selection_score, selection_mode):
                        best_selection_score = float(selection_score)

                    if early_enabled:
                        if _is_improved(
                            selection_score,
                            early_best_score,
                            selection_mode,
                            min_delta=early_min_delta,
                        ):
                            early_best_score = float(selection_score)
                            early_bad_epochs = 0
                        else:
                            early_bad_epochs += 1
                            if early_bad_epochs >= early_patience:
                                break
                elif (epoch % self.log_epoch_interval) == 0 or epoch == 1 or epoch == epochs:
                    logger.info(
                        "Trial %d | Epoch %d/%d | train_loss=%.6f | val=skipped | validation_every_n_epochs=%d",
                        trial.number,
                        epoch,
                        epochs,
                        float(train_loss),
                        int(validation_every_n_epochs),
                    )

            if best_epoch == 0:
                raise RuntimeError(
                    "Validation never ran successfully during tuning. "
                    "Check training.validation_every_n_epochs and epochs_per_trial."
                )

            trial.set_user_attr("best_epoch", int(best_epoch))
            trial.set_user_attr("best_val_loss", float(best_val_loss))
            trial.set_user_attr("best_val_metrics", best_val_metrics)
            trial.set_user_attr("objective_metric", str(objective_metric))
            trial.set_user_attr("best_objective_value", float(best_objective))
            trial.set_user_attr("best_selection_score", float(best_selection_score))
            logger.info(
                "Trial %d complete | best_epoch=%d | best_val_loss=%.6f | best_objective(%s)=%.6f",
                trial.number,
                best_epoch,
                float(best_val_loss),
                objective_metric,
                float(best_objective),
            )
            return float(best_objective)

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("failure_reason", str(exc))
            logger.exception("Trial %d failed", trial.number)
            return _worst_value(objective_mode)
        finally:
            del model
            del loss_fn
            del optimizer
            del scheduler
            del pcgrad_optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
