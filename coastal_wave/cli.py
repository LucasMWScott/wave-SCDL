"""Command dispatcher; import each optional workflow only when selected."""

import argparse
import importlib
import logging
import sys

COMMANDS = {
    "geometry": "geometric_builder.src.build_features",
    "route": "geometric_builder.src.fetch_router",
    "rays": "geometric_builder.src.ray_caster",
    "bathy-grid": "geometric_builder.src.generate_bathy_field",
    "bathy-patches": "src.preprocess.build_bathy_patches",
    "preprocess": "src.preprocessing.discover",
    "train": "src.train",
    "evaluate": "src.evaluate",
    "infer": "src.infer_new_sites",
    "tune": "src.run_tuning",
    "demo": "coastal_wave.demo",
}


def main():
    parser = argparse.ArgumentParser(description="Coastal geometry and wave downscaling workflows")
    parser.add_argument("workflow", choices=sorted(COMMANDS))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.argv = [f"coastal-wave {args.workflow}", *rest]
    importlib.import_module(COMMANDS[args.workflow]).main()
