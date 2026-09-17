"""Grid and site preprocessing utilities for coastal geometry workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import CRS, Transformer


class GridManager:
    """Manage site and bathymetry preprocessing for geometry/model pipelines.

    Parameters
    ----------
    target_epsg:
        Projected EPSG code used by the bathymetry grid (e.g., 32633 for UTM33N).
    bathymetry_filename:
        XYZ filename located in ``data/bathy``.
    padding_meters:
        Padding distance applied to all sides of the site bounding box.
    config_relative_path:
        Relative path from project root to the YAML site configuration.
    bathy_relative_dir:
        Relative directory from project root that contains bathymetry files.
    output_relative_path:
        Relative path from project root where the padded NetCDF is exported.
    project_root:
        Optional explicit project root path. If omitted, inferred as parent of ``src``.
    """

    def __init__(
        self,
        target_epsg: int,
        bathymetry_filename: str,
        padding_meters: float = 20_000.0,
        config_relative_path: str = "configs/sites.yaml",
        bathy_relative_dir: str = "data/bathy",
        output_relative_path: str = "data/bathy/subgrid_padded.nc",
        project_root: Optional[Path] = None,
    ) -> None:
        self.project_root = (
            project_root.resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        self.config_path = self.project_root / config_relative_path
        self.bathy_dir = self.project_root / bathy_relative_dir
        self.bathymetry_path = self.bathy_dir / bathymetry_filename
        self.output_path = self.project_root / output_relative_path

        self.padding_meters = float(padding_meters)
        self.target_epsg = int(target_epsg)
        self.target_crs = CRS.from_epsg(self.target_epsg)

        self.sites_df: Optional[pd.DataFrame] = None
        self.bathy_grid: Optional[xr.DataArray] = None
        self.local_sub_grid: Optional[xr.DataArray] = None

    def parse_configuration(self) -> pd.DataFrame:
        """Read ``configs/sites.yaml`` and return all offshore/nearshore site points.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns: ``site_name``, ``site_group``, ``lat``, ``lon``.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        KeyError
            If ``offshore_sites`` or ``nearshore_sites`` keys are missing.
        ValueError
            If a site is missing ``name``, ``lat`` or ``lon``.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Site config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        required_top_keys = ("offshore_sites", "nearshore_sites")
        for key in required_top_keys:
            if key not in config:
                raise KeyError(f"Missing required key '{key}' in {self.config_path}")

        records: List[Dict[str, object]] = []
        for site_group in required_top_keys:
            for site in config.get(site_group, []):
                name = site.get("name")
                lat = site.get("lat")
                lon = site.get("lon")

                if name is None or lat is None or lon is None:
                    raise ValueError(
                        f"Site in '{site_group}' missing one of required fields: name/lat/lon"
                    )

                records.append(
                    {
                        "site_name": str(name),
                        "site_group": site_group,
                        "lat": float(lat),
                        "lon": float(lon),
                    }
                )

        sites_df = pd.DataFrame.from_records(records)
        self.sites_df = sites_df
        return sites_df

    def transform_site_coordinates(self, source_epsg: int = 4326) -> pd.DataFrame:
        """Project site coordinates from geographic CRS to the target projected CRS.

        Parameters
        ----------
        source_epsg:
            EPSG code for source coordinates in YAML. Defaults to EPSG:4326.

        Returns
        -------
        pandas.DataFrame
            Site DataFrame with added columns ``x`` and ``y`` in projected meters.
        """
        if self.sites_df is None:
            self.parse_configuration()

        if self.sites_df is None:
            raise RuntimeError("Site parsing failed; sites_df is empty.")

        transformer = Transformer.from_crs(
            CRS.from_epsg(int(source_epsg)), self.target_crs, always_xy=True
        )
        x_values, y_values = transformer.transform(
            self.sites_df["lon"].to_numpy(), self.sites_df["lat"].to_numpy()
        )

        transformed_df = self.sites_df.copy()
        transformed_df["x"] = x_values
        transformed_df["y"] = y_values
        self.sites_df = transformed_df
        return transformed_df

    def load_and_grid_bathymetry(self) -> xr.DataArray:
        """Load XYZ bathymetry and rasterize it into a dense 2D grid.

        The XYZ file is treated as a point cloud on a regular grid. Instead of slower
        interpolation, this method uses coordinate indexing to place values directly
        into a dense matrix for computational efficiency.

        Returns
        -------
        xarray.DataArray
            A 2D DataArray named ``depth`` with dims ``(y, x)``.

        Raises
        ------
        FileNotFoundError
            If the XYZ file is missing.
        ValueError
            If the XYZ file cannot be parsed as three numeric columns.
        """
        if not self.bathymetry_path.exists():
            raise FileNotFoundError(f"Bathymetry file not found: {self.bathymetry_path}")

        xyz_df = pd.read_csv(
            self.bathymetry_path,
            sep=r"\s+",
            header=None,
            names=["x", "y", "z"],
            dtype=np.float64,
            engine="c",
        )

        if xyz_df.empty:
            raise ValueError(f"Bathymetry file is empty: {self.bathymetry_path}")

        x_coords = np.sort(xyz_df["x"].unique())
        y_coords = np.sort(xyz_df["y"].unique())

        z_grid = np.full((y_coords.size, x_coords.size), np.nan, dtype=np.float32)
        x_idx = np.searchsorted(x_coords, xyz_df["x"].to_numpy())
        y_idx = np.searchsorted(y_coords, xyz_df["y"].to_numpy())
        z_grid[y_idx, x_idx] = xyz_df["z"].to_numpy(dtype=np.float32)

        grid = xr.DataArray(
            z_grid,
            coords={"y": y_coords, "x": x_coords},
            dims=("y", "x"),
            name="depth",
            attrs={
                "long_name": "Bathymetry depth",
                "units": "m",
                "grid_mapping": f"EPSG:{self.target_epsg}",
            },
        )

        self.bathy_grid = grid
        return grid

    def generate_padded_sub_grid(self) -> xr.DataArray:
        """Crop bathymetry to a padded bounding box around all projected sites.

        Returns
        -------
        xarray.DataArray
            Cropped local sub-grid with the same ``(y, x)`` dimensions.
        """
        if self.sites_df is None or not {"x", "y"}.issubset(self.sites_df.columns):
            self.transform_site_coordinates()

        if self.bathy_grid is None:
            self.load_and_grid_bathymetry()

        if self.sites_df is None or self.bathy_grid is None:
            raise RuntimeError("Required data are missing for sub-grid generation.")

        min_x = float(self.sites_df["x"].min()) - self.padding_meters
        max_x = float(self.sites_df["x"].max()) + self.padding_meters
        min_y = float(self.sites_df["y"].min()) - self.padding_meters
        max_y = float(self.sites_df["y"].max()) + self.padding_meters

        local_sub_grid = self.bathy_grid.sel(x=slice(min_x, max_x), y=slice(min_y, max_y))

        if local_sub_grid.sizes.get("x", 0) == 0 or local_sub_grid.sizes.get("y", 0) == 0:
            raise ValueError(
                "Sub-grid is empty. Check target EPSG, padding, and site/bathymetry extents."
            )

        local_sub_grid = local_sub_grid.copy()
        local_sub_grid.attrs.update(
            {
                "padding_meters": self.padding_meters,
                "bbox_min_x": min_x,
                "bbox_max_x": max_x,
                "bbox_min_y": min_y,
                "bbox_max_y": max_y,
                "target_epsg": self.target_epsg,
            }
        )

        self.local_sub_grid = local_sub_grid
        return local_sub_grid

    def export_sub_grid(self) -> Path:
        """Write the padded sub-grid to NetCDF at ``data/bathy/subgrid_padded.nc``.

        Returns
        -------
        pathlib.Path
            Full output path of the generated NetCDF file.
        """
        if self.local_sub_grid is None:
            self.generate_padded_sub_grid()

        if self.local_sub_grid is None:
            raise RuntimeError("No sub-grid available for export.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = self.local_sub_grid.to_dataset(name="depth")
        dataset.attrs.update(
            {
                "title": "Padded local bathymetry sub-grid",
                "source_xyz": str(self.bathymetry_path.relative_to(self.project_root)),
                "config_yaml": str(self.config_path.relative_to(self.project_root)),
                "target_epsg": self.target_epsg,
            }
        )
        dataset.to_netcdf(self.output_path)
        return self.output_path

    def run(self) -> Path:
        """Execute the full preprocessing workflow and export a NetCDF sub-grid.

        Returns
        -------
        pathlib.Path
            Path to ``data/bathy/subgrid_padded.nc``.
        """
        self.parse_configuration()
        self.transform_site_coordinates()
        self.load_and_grid_bathymetry()
        self.generate_padded_sub_grid()
        return self.export_sub_grid()

    def get_projected_sites(self) -> pd.DataFrame:
        """Return parsed + projected site coordinates.

        Returns
        -------
        pandas.DataFrame
            Columns include ``site_name``, ``site_group``, ``lat``, ``lon``, ``x``, ``y``.
        """
        if self.sites_df is None or not {"x", "y"}.issubset(self.sites_df.columns):
            self.transform_site_coordinates()

        if self.sites_df is None:
            raise RuntimeError("Projected site data unavailable.")
        return self.sites_df.copy()


__all__ = ["GridManager"]
