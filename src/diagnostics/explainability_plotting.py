"""Plotting helpers for explainability notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
except Exception:
    shap = None


THESIS_FONT_SIZES = {
    "title": 14,
    "label": 11,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}


def _finalize(fig, title: str | None = None):
    if title:
        fig.suptitle(title, fontsize=THESIS_FONT_SIZES["title"])
    fig.tight_layout()
    return fig


def _save(fig, out_path: str | Path | None) -> None:
    if out_path is None:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def save_figure_variants(
    fig,
    out_stem: str | Path | None,
    *,
    dpi: int = 300,
    close: bool = False,
) -> list[Path]:
    """Save both PNG and PDF copies for thesis-ready figures."""
    if out_stem is None:
        return []
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    if close:
        plt.close(fig)
    return outputs


def _apply_thesis_axis_style(ax, *, grid_axis: str = "x") -> None:
    ax.tick_params(axis="both", labelsize=THESIS_FONT_SIZES["tick"])
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)


def plot_grouped_shap_bar(
    df: pd.DataFrame,
    *,
    value_col: str = "mean_abs_value",
    label_col: str = "group",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df) + 1)))
    ordered = df.sort_values(value_col, ascending=True)
    ax.barh(ordered[label_col], ordered[value_col], color="#2b6cb0")
    ax.set_xlabel(value_col.replace("_", " ").title(), fontsize=THESIS_FONT_SIZES["label"])
    ax.set_ylabel("")
    _apply_thesis_axis_style(ax, grid_axis="x")
    _save(_finalize(fig, title), out_path)
    return fig


def plot_grouped_shap_beeswarm(
    values: pd.DataFrame,
    *,
    group_col: str = "group",
    shap_col: str = "value",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * values[group_col].nunique() + 1)))
    groups = list(values[group_col].dropna().unique())
    for idx, group in enumerate(groups):
        subset = values.loc[values[group_col] == group, shap_col].to_numpy(dtype=float)
        if subset.size == 0:
            continue
        jitter = np.linspace(-0.18, 0.18, num=subset.size) if subset.size > 1 else np.array([0.0])
        ax.scatter(subset, np.full(subset.size, idx, dtype=float) + jitter, s=12, alpha=0.5)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_yticks(np.arange(len(groups), dtype=float))
    ax.set_yticklabels(groups)
    ax.set_xlabel(shap_col.replace("_", " ").title(), fontsize=THESIS_FONT_SIZES["label"])
    _apply_thesis_axis_style(ax, grid_axis="x")
    _save(_finalize(fig, title), out_path)
    return fig


def plot_grouped_shap_distribution(
    values: pd.DataFrame,
    *,
    group_order: Sequence[str],
    display_name_map: dict[str, str] | None = None,
    group_col: str = "group",
    shap_col: str = "value",
    title: str | None = None,
    x_label: str = "Grouped SHAP value",
    ax=None,
):
    """Signed grouped-SHAP distribution plot without a feature-value color scale."""
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.55 * len(group_order) + 1.5)))

    display_name_map = dict(display_name_map or {})
    for idx, group in enumerate(group_order):
        subset = values.loc[values[group_col] == group, shap_col].to_numpy(dtype=float)
        subset = subset[np.isfinite(subset)]
        if subset.size == 0:
            continue
        order = np.argsort(subset, kind="mergesort")
        jitter_sorted = (
            np.linspace(-0.22, 0.22, num=subset.size) if subset.size > 1 else np.array([0.0])
        )
        jitter = np.zeros_like(jitter_sorted)
        jitter[order] = jitter_sorted
        ax.scatter(
            subset,
            np.full(subset.size, idx, dtype=float) + jitter,
            s=16,
            alpha=0.55,
            color="#1f4e79",
            edgecolors="none",
        )

    labels = [display_name_map.get(group, group) for group in group_order]
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_yticks(np.arange(len(group_order), dtype=float))
    ax.set_yticklabels(labels, fontsize=THESIS_FONT_SIZES["tick"])
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=THESIS_FONT_SIZES["label"])
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=THESIS_FONT_SIZES["title"])
    _apply_thesis_axis_style(ax, grid_axis="x")
    return created_fig if created_fig is not None else ax.figure


def plot_top_static_shap_beeswarm(
    shap_values,
    feature_values,
    feature_names,
    target_name: str,
    output_dir: str | Path,
    *,
    base_values=None,
    max_display: int = 10,
    readable_name_map: dict[str, str] | None = None,
    x_label: str | None = None,
    title: str | None = None,
    feature_value_label: str = "Feature value",
):
    """Render and save a standard SHAP beeswarm for selected static features."""
    if shap is None:
        raise RuntimeError("shap is required for static SHAP beeswarm plotting")

    shap_array = np.asarray(shap_values, dtype=float)
    feature_array = np.asarray(feature_values, dtype=float)
    name_map = {str(key): str(value) for key, value in dict(readable_name_map or {}).items()}
    raw_names = [str(name) for name in feature_names]
    names = [name_map.get(name, name) for name in raw_names]

    if shap_array.ndim != 2:
        raise ValueError(f"shap_values must be 2D, got shape {shap_array.shape}")
    if feature_array.ndim != 2:
        raise ValueError(f"feature_values must be 2D, got shape {feature_array.shape}")
    if shap_array.shape != feature_array.shape:
        raise ValueError(
            "shap_values and feature_values must have identical shape; "
            f"got {shap_array.shape} vs {feature_array.shape}"
        )
    if shap_array.shape[1] != len(names):
        raise ValueError(
            "feature_names length must match the feature dimension; "
            f"got {len(names)} names for {shap_array.shape[1]} features"
        )

    base_array = None
    if base_values is not None:
        base_array = np.asarray(base_values, dtype=float).reshape(-1)
        if base_array.size == 1:
            base_array = np.repeat(base_array, repeats=shap_array.shape[0])
        if base_array.shape[0] != shap_array.shape[0]:
            raise ValueError(
                "base_values length must match the sample dimension; "
                f"got {base_array.shape[0]} for {shap_array.shape[0]} samples"
            )

    display_count = int(min(max_display, shap_array.shape[1]))
    fig_height = max(5.5, 0.44 * display_count + 2.2)
    fig = plt.figure(figsize=(10.5, fig_height))
    ax = fig.add_subplot(111)
    plt.sca(ax)

    explanation = shap.Explanation(
        values=shap_array,
        data=feature_array,
        feature_names=names,
        base_values=base_array,
    )
    shap.plots.beeswarm(explanation, max_display=display_count, show=False)

    fig = plt.gcf()
    ax = fig.axes[0]
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(
        x_label or f"SHAP value (impact on predicted {target_name})",
        fontsize=THESIS_FONT_SIZES["label"],
    )
    ax.set_title(
        title or f"Static-feature SHAP contributions: {target_name}",
        fontsize=THESIS_FONT_SIZES["title"],
    )
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=THESIS_FONT_SIZES["tick"])

    if len(fig.axes) > 1:
        colorbar_ax = fig.axes[-1]
        colorbar_ax.set_ylabel(feature_value_label, fontsize=THESIS_FONT_SIZES["label"])
        colorbar_ax.tick_params(labelsize=THESIS_FONT_SIZES["tick"])

    fig.tight_layout()
    return save_figure_variants(fig, Path(output_dir), close=True)


def plot_static_shap_beeswarm(
    shap_values,
    feature_values,
    feature_names,
    target_name: str,
    output_dir: str | Path,
    *,
    base_values=None,
    max_display: int = 20,
    feature_value_label: str = "Feature value",
):
    """Backward-compatible wrapper for static SHAP beeswarm plotting."""
    return plot_top_static_shap_beeswarm(
        shap_values,
        feature_values,
        feature_names,
        target_name,
        output_dir,
        base_values=base_values,
        max_display=max_display,
        feature_value_label=feature_value_label,
    )


def plot_grouped_shap_importance_heatmap(
    importance_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    target_order: Sequence[str],
    display_name_map: dict[str, str] | None = None,
    target_display_map: dict[str, str] | None = None,
    value_col: str = "normalized_mean_abs_shap",
    title: str | None = None,
    colorbar_label: str = "Normalized mean |grouped SHAP|",
    ax=None,
):
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.5 * len(group_order) + 1.75)))

    display_name_map = dict(display_name_map or {})
    target_display_map = dict(target_display_map or {})
    pivot = (
        importance_df.pivot(index="feature_group", columns="target", values=value_col)
        .reindex(index=list(group_order), columns=list(target_order))
        .fillna(0.0)
    )
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")

    ax.set_xticks(np.arange(len(target_order)))
    ax.set_xticklabels(
        [target_display_map.get(target, target) for target in target_order],
        fontsize=THESIS_FONT_SIZES["tick"],
    )
    ax.set_yticks(np.arange(len(group_order)))
    ax.set_yticklabels(
        [display_name_map.get(group, group) for group in group_order],
        fontsize=THESIS_FONT_SIZES["tick"],
    )
    if title:
        ax.set_title(title, fontsize=THESIS_FONT_SIZES["title"])
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=THESIS_FONT_SIZES["label"])
    cbar.ax.tick_params(labelsize=THESIS_FONT_SIZES["tick"])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "#f7fafc" if luminance < 0.5 else "#111827"
            outline_color = "#111827" if luminance < 0.5 else "#ffffff"
            text = ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=THESIS_FONT_SIZES["annotation"],
                fontweight="semibold",
                color=text_color,
            )
            text.set_path_effects(
                [pe.withStroke(linewidth=1.25, foreground=outline_color, alpha=0.9)]
            )

    ax.set_xlabel("Target", fontsize=THESIS_FONT_SIZES["label"])
    ax.set_ylabel("")
    return created_fig if created_fig is not None else ax.figure


def plot_ale_curve(
    df: pd.DataFrame,
    *,
    x_col: str = "feature_value",
    y_col: str = "ale",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df[x_col], df[y_col], color="#1a202c", linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=THESIS_FONT_SIZES["label"])
    ax.set_ylabel(y_col.upper(), fontsize=THESIS_FONT_SIZES["label"])
    _apply_thesis_axis_style(ax, grid_axis="both")
    _save(_finalize(fig, title), out_path)
    return fig


def plot_ale_with_histogram(
    ale_df: pd.DataFrame,
    *,
    feature_label: str,
    title: str | None = None,
    x_col: str = "bin_center",
    y_col: str = "ale_value",
    lower_col: str = "lower_confidence",
    upper_col: str = "upper_confidence",
    count_col: str = "n_unique_sites",
    left_col: str = "bin_left",
    right_col: str = "bin_right",
    y_label: str = "ALE",
    count_label: str = "Number of sites",
    ax_main=None,
    ax_hist=None,
):
    created_fig = None
    if ax_main is None or ax_hist is None:
        created_fig, (ax_main, ax_hist) = plt.subplots(
            2,
            1,
            figsize=(8.2, 5.6),
            sharex=True,
            gridspec_kw={"height_ratios": [4.0, 1.15], "hspace": 0.05},
        )

    x = ale_df[x_col].to_numpy(dtype=float)
    y = ale_df[y_col].to_numpy(dtype=float)
    lower = (
        ale_df[lower_col].to_numpy(dtype=float)
        if lower_col in ale_df.columns
        else np.full_like(y, np.nan)
    )
    upper = (
        ale_df[upper_col].to_numpy(dtype=float)
        if upper_col in ale_df.columns
        else np.full_like(y, np.nan)
    )
    counts = (
        ale_df[count_col].to_numpy(dtype=float) if count_col in ale_df.columns else np.ones_like(y)
    )
    analysis_kind = (
        str(ale_df.get("analysis_kind", pd.Series(["continuous"])).iloc[0]).strip().lower()
    )

    if analysis_kind == "discrete":
        ax_main.errorbar(
            x,
            y,
            yerr=np.vstack(
                [
                    np.where(np.isfinite(lower), y - lower, np.nan),
                    np.where(np.isfinite(upper), upper - y, np.nan),
                ]
            ),
            fmt="o-",
            color="#1f4e79",
            ecolor="#9ecae1",
            elinewidth=1.6,
            capsize=3.0,
            linewidth=1.8,
            markersize=5.5,
        )
    else:
        ax_main.plot(x, y, color="#1f4e79", linewidth=2.2)
        if np.isfinite(lower).any() and np.isfinite(upper).any():
            ax_main.fill_between(x, lower, upper, color="#9ecae1", alpha=0.35, linewidth=0.0)
    ax_main.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax_main.set_ylabel(y_label, fontsize=THESIS_FONT_SIZES["label"])
    if title:
        ax_main.set_title(title, fontsize=THESIS_FONT_SIZES["title"])
    _apply_thesis_axis_style(ax_main, grid_axis="both")

    if analysis_kind == "discrete":
        if x.size > 1:
            min_gap = np.min(np.diff(np.sort(np.unique(x))))
            width = max(min_gap * 0.7, np.finfo(float).eps)
        else:
            width = 0.8
        ax_hist.bar(x, counts, width=width, color="#9fbad0", edgecolor="white", linewidth=0.8)
    elif left_col in ale_df.columns and right_col in ale_df.columns:
        left = ale_df[left_col].to_numpy(dtype=float)
        right = ale_df[right_col].to_numpy(dtype=float)
        widths = np.maximum(right - left, np.finfo(float).eps)
        ax_hist.bar(
            left,
            counts,
            width=widths,
            align="edge",
            color="#9fbad0",
            edgecolor="white",
            linewidth=0.8,
        )
    else:
        ax_hist.bar(x, counts, width=0.8, color="#9fbad0", edgecolor="white", linewidth=0.8)
    ax_hist.set_ylabel(count_label, fontsize=THESIS_FONT_SIZES["label"])
    ax_hist.set_xlabel(feature_label, fontsize=THESIS_FONT_SIZES["label"])
    ax_hist.tick_params(axis="both", labelsize=THESIS_FONT_SIZES["tick"])
    _apply_thesis_axis_style(ax_hist, grid_axis="y")

    if left_col in ale_df.columns and right_col in ale_df.columns and analysis_kind != "discrete":
        left = ale_df[left_col].to_numpy(dtype=float)
        right = ale_df[right_col].to_numpy(dtype=float)
        if np.isfinite(left).any() and np.isfinite(right).any():
            ax_main.set_xlim(np.nanmin(left), np.nanmax(right))

    return created_fig if created_fig is not None else ax_main.figure


def plot_temporal_feature_lag_heatmap(
    matrix: np.ndarray,
    *,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(np.asarray(matrix, dtype=float), aspect="auto", cmap="coolwarm")
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax, shrink=0.85)
    _save(_finalize(fig, title), out_path)
    return fig


def plot_temporal_source_lag_heatmap(
    matrix: np.ndarray,
    *,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
):
    return plot_temporal_feature_lag_heatmap(
        matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        out_path=out_path,
    )


def plot_attention_by_head(
    attention: np.ndarray,
    *,
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(np.asarray(attention, dtype=float), aspect="auto", cmap="viridis")
    ax.set_xlabel("Context Token")
    ax.set_ylabel("Head")
    fig.colorbar(im, ax=ax, shrink=0.85)
    _save(_finalize(fig, title), out_path)
    return fig


def plot_attention_regime_comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_title: str = "Left",
    right_title: str = "Right",
    out_path: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arr, ttl in zip(axes, [left, right], [left_title, right_title]):
        im = ax.imshow(np.asarray(arr, dtype=float), aspect="auto", cmap="viridis")
        ax.set_title(ttl)
        ax.set_xlabel("Context Token")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Head")
    _save(_finalize(fig), out_path)
    return fig


def plot_branch_ablation_bars(
    df: pd.DataFrame,
    *,
    x_col: str = "branch",
    y_col: str = "mean_absolute_change",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ordered = df.sort_values(y_col, ascending=False)
    ax.bar(ordered[x_col], ordered[y_col], color="#dd6b20")
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35)
    _save(_finalize(fig, title), out_path)
    return fig


def plot_static_embedding_space(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    color_col: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col is not None and color_col in df.columns:
        scatter = ax.scatter(
            df[x_col], df[y_col], c=df[color_col], cmap="viridis", s=28, alpha=0.85
        )
        fig.colorbar(scatter, ax=ax, shrink=0.85)
    else:
        ax.scatter(df[x_col], df[y_col], s=28, alpha=0.85)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    _save(_finalize(fig, title), out_path)
    return fig


def plot_sitewise_error_vs_analog_distance_grid(
    df: pd.DataFrame,
    *,
    target_order: Sequence[str] = ("hs", "tp", "dir", "dp"),
    distance_col: str = "mean_training_analogue_distance",
    error_col: str = "normalized_rmse",
    split_col: str = "split",
    target_col: str = "target",
    target_label_map: dict[str, str] | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
):
    if df.empty:
        raise ValueError("Cannot plot sitewise error vs analogue distance with an empty dataframe")
    if (
        distance_col not in df.columns
        or error_col not in df.columns
        or target_col not in df.columns
    ):
        raise ValueError(
            f"Expected columns '{distance_col}', '{error_col}', and '{target_col}' in dataframe"
        )

    label_map = {
        "hs": "Hs",
        "tp": "Tp",
        "dir": "Mean direction",
        "dp": "Peak direction",
    }
    label_map.update({str(key): str(value) for key, value in dict(target_label_map or {}).items()})
    split_colors = {
        "train": "#2b6cb0",
        "val": "#dd6b20",
        "test": "#2f855a",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    axes_flat = axes.ravel()

    for ax, target in zip(axes_flat, target_order):
        subset = df.loc[df[target_col].astype(str) == str(target)].copy()
        subset[distance_col] = pd.to_numeric(subset[distance_col], errors="coerce")
        subset[error_col] = pd.to_numeric(subset[error_col], errors="coerce")
        subset = subset.dropna(subset=[distance_col, error_col])
        if subset.empty:
            ax.set_visible(False)
            continue

        if split_col in subset.columns:
            split_values = subset[split_col].fillna("unknown").astype(str)
        else:
            split_values = pd.Series(["all"] * len(subset), index=subset.index)

        for split_name in split_values.drop_duplicates().tolist():
            split_df = subset.loc[split_values == split_name]
            handle = ax.scatter(
                split_df[distance_col],
                split_df[error_col],
                s=46,
                alpha=0.82,
                color=split_colors.get(str(split_name), "#6b7280"),
                edgecolors="white",
                linewidths=0.5,
                label=str(split_name),
            )
            if str(split_name) not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(str(split_name))

        x = subset[distance_col].to_numpy(dtype=float)
        y = subset[error_col].to_numpy(dtype=float)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]
        corr_text = "r = n/a"
        if x.size >= 2:
            x_span = float(np.nanmax(x) - np.nanmin(x))
            y_span = float(np.nanmax(y) - np.nanmin(y))
            if x_span > 0.0:
                slope, intercept = np.polyfit(x, y, deg=1)
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                ax.plot(
                    x_line, slope * x_line + intercept, color="#1a202c", linewidth=1.4, alpha=0.9
                )
            if x_span > 0.0 and y_span > 0.0:
                corr_text = f"r = {np.corrcoef(x, y)[0, 1]:.2f}"
        corr_text = f"{corr_text}\nn = {x.size}"

        ax.text(
            0.03,
            0.97,
            corr_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=THESIS_FONT_SIZES["annotation"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#d1d5db",
            },
        )
        ax.set_title(
            label_map.get(str(target), str(target)), fontsize=THESIS_FONT_SIZES["title"], loc="left"
        )
        ax.set_xlabel(
            "Mean distance to nearest training analogues", fontsize=THESIS_FONT_SIZES["label"]
        )
        ax.set_ylabel("Site-wise normalized RMSE", fontsize=THESIS_FONT_SIZES["label"])
        _apply_thesis_axis_style(ax, grid_axis="both")

    for ax in axes_flat[len(tuple(target_order)) :]:
        ax.set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=max(1, len(legend_labels)),
            frameon=False,
            fontsize=THESIS_FONT_SIZES["legend"],
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=THESIS_FONT_SIZES["title"])
        if legend_handles:
            fig.subplots_adjust(top=0.88)
    _save(fig, out_path)
    return fig


def plot_bathy_attribution_map(
    attribution: np.ndarray,
    *,
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.asarray(attribution, dtype=float), cmap="coolwarm")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    _save(_finalize(fig, title), out_path)
    return fig


def plot_good_bad_attribution_delta(
    df: pd.DataFrame,
    *,
    label_col: str = "group",
    delta_col: str = "delta",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df) + 1)))
    ordered = df.sort_values(delta_col, ascending=True)
    colors = ["#c53030" if value < 0 else "#2f855a" for value in ordered[delta_col]]
    ax.barh(ordered[label_col], ordered[delta_col], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(delta_col.replace("_", " ").title())
    _save(_finalize(fig, title), out_path)
    return fig


def plot_site_regime_dominance_bars(
    df: pd.DataFrame,
    *,
    site_col: str = "site",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    ordered = df.sort_values(
        ["site_dominant_regime", "local_windsea_fraction", "swell_fraction"],
        ascending=[True, False, False],
    )
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(ordered) + 1.5)))
    y = np.arange(len(ordered))
    swell = ordered.get("swell_fraction", pd.Series(np.zeros(len(ordered))))
    local = ordered.get("local_windsea_fraction", pd.Series(np.zeros(len(ordered))))
    mixed = ordered.get("mixed_fraction", pd.Series(np.zeros(len(ordered))))
    ax.barh(y, swell, color="#2b6cb0", label="swell")
    ax.barh(y, local, left=swell, color="#dd6b20", label="local_windsea")
    ax.barh(y, mixed, left=swell + local, color="#718096", label="mixed")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered[site_col].astype(str))
    ax.set_xlabel("Fraction of samples")
    ax.legend(loc="lower right")
    _save(_finalize(fig, title), out_path)
    return fig


def plot_site_regime_score_heatmap(
    df: pd.DataFrame,
    *,
    site_col: str = "site",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    ordered = df.sort_values(
        ["site_dominant_regime", "mean_local_windsea_score", "mean_swell_score"],
        ascending=[True, False, False],
    )
    matrix = ordered[
        [
            "mean_swell_score",
            "mean_local_windsea_score",
            "swell_fraction",
            "local_windsea_fraction",
            "mixed_fraction",
        ]
    ].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ordered) + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered[site_col].astype(str))
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        ["mean_swell", "mean_local_windsea", "swell_frac", "local_frac", "mixed_frac"],
        rotation=35,
        ha="right",
    )
    fig.colorbar(im, ax=ax, shrink=0.85)
    _save(_finalize(fig, title), out_path)
    return fig


def plot_regime_share_scatter(
    df: pd.DataFrame,
    *,
    x_col: str = "open_sector_fraction",
    y_col: str = "local_windsea_fraction",
    label_col: str = "site",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    color_map = {"swell": "#2b6cb0", "local_windsea": "#dd6b20", "mixed": "#718096"}
    colors = [color_map.get(value, "#4a5568") for value in df.get("site_dominant_regime", [])]
    ax.scatter(df[x_col], df[y_col], c=colors, s=60, alpha=0.85)
    for row in df.itertuples(index=False):
        if hasattr(row, x_col) and hasattr(row, y_col) and hasattr(row, label_col):
            ax.text(
                getattr(row, x_col),
                getattr(row, y_col),
                str(getattr(row, label_col)),
                fontsize=8,
                alpha=0.75,
            )
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    _save(_finalize(fig, title), out_path)
    return fig


def plot_directional_alignment_summary(
    df: pd.DataFrame,
    *,
    dir_delta_col: str = "dir_abs_error_deg",
    dp_delta_col: str = "dp_abs_error_deg",
    title: str | None = None,
    out_path: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    color_map = {"swell": "#2b6cb0", "local_windsea": "#dd6b20", "mixed": "#718096"}
    colors = [color_map.get(value, "#4a5568") for value in df.get("dominant_regime", [])]
    axes[0].scatter(
        df.get("delta_target_vs_swell_dir_deg", 0.0),
        df.get(dir_delta_col, 0.0),
        c=colors,
        s=28,
        alpha=0.7,
    )
    axes[0].set_xlabel("Target vs swell dir delta (deg)")
    axes[0].set_ylabel("Dir abs error (deg)")
    axes[1].scatter(
        df.get("delta_target_vs_local_wind_dir_deg", 0.0),
        df.get(dp_delta_col, 0.0),
        c=colors,
        s=28,
        alpha=0.7,
    )
    axes[1].set_xlabel("Target vs local wind dir delta (deg)")
    axes[1].set_ylabel("Dp abs error (deg)")
    _save(_finalize(fig, title), out_path)
    return fig
