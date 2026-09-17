"""Point-centric dataset API shim."""

from __future__ import annotations

try:
    from src.data_pipeline import (
        PointCentricArrays,
        PointCentricWindowDataset,
        build_split_endpoints,
        resolve_output_indices,
        resolve_sequence_length,
        select_sites,
    )
except Exception:
    from data_pipeline import (
        PointCentricArrays,
        PointCentricWindowDataset,
        build_split_endpoints,
        resolve_output_indices,
        resolve_sequence_length,
        select_sites,
    )

__all__ = [
    "PointCentricArrays",
    "PointCentricWindowDataset",
    "build_split_endpoints",
    "resolve_output_indices",
    "resolve_sequence_length",
    "select_sites",
]
