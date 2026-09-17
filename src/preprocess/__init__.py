"""Preprocessing helper package for point-centric dataset builders."""

from src.preprocessing.discover import discover_site_files, read_sites_yaml

__all__ = ["discover_site_files", "read_sites_yaml"]


def __getattr__(name):
    """Preserve discovery helpers previously exposed by preprocess.py."""
    from src.preprocessing import discover

    return getattr(discover, name)
