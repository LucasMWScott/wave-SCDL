"""Test training batch config."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.runtime_paths import REPO_ROOT, normalize_runtime_config_paths
from src.training_batch import load_batch_training_plan


class RuntimeConfigPathResolutionTests(unittest.TestCase):
    def test_normalize_runtime_config_paths_resolves_repo_relative_inputs_from_subfolder_config(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_runtime_cfg_") as tmpdir:
            base = Path(tmpdir).resolve()
            rel_base = base.relative_to(REPO_ROOT)
            case_dir = base / "configs" / "training_cases"
            case_dir.mkdir(parents=True, exist_ok=True)

            sites_path = base / "shared" / "sites.yaml"
            static_path = base / "shared" / "master_static_features.csv"
            ablation_path = base / "shared" / "ablation.yaml"
            point_centric_dir = base / "data" / "point_centric_case"
            point_centric_dir.mkdir(parents=True, exist_ok=True)
            sites_path.parent.mkdir(parents=True, exist_ok=True)
            sites_path.write_text("nearshore_sites: []\n", encoding="utf-8")
            static_path.write_text("feature\n", encoding="utf-8")
            ablation_path.write_text("enabled: false\n", encoding="utf-8")

            config = {
                "data": {
                    "sites_config": str(rel_base / "shared" / "sites.yaml"),
                    "static_features_csv": str(rel_base / "shared" / "master_static_features.csv"),
                    "static_ablation_config": str(rel_base / "shared" / "ablation.yaml"),
                    "point_centric_dir": str(rel_base / "data" / "point_centric_case"),
                },
                "logging": {
                    "output_dir": "results/training_case_batch",
                },
            }

            resolved = normalize_runtime_config_paths(
                config,
                config_path=case_dir / "training_case1.yaml",
            )

            self.assertEqual(resolved["data"]["sites_config"], str(sites_path.resolve()))
            self.assertEqual(resolved["data"]["static_features_csv"], str(static_path.resolve()))
            self.assertEqual(
                resolved["data"]["static_ablation_config"], str(ablation_path.resolve())
            )
            self.assertEqual(
                resolved["data"]["point_centric_dir"], str(point_centric_dir.resolve())
            )
            self.assertEqual(
                resolved["logging"]["output_dir"],
                str((REPO_ROOT / "results" / "training_case_batch").resolve()),
            )

    def test_normalize_runtime_config_paths_honors_explicit_config_relative_output_dir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_runtime_cfg_") as tmpdir:
            case_dir = Path(tmpdir).resolve() / "configs" / "training_cases"
            case_dir.mkdir(parents=True, exist_ok=True)
            config_path = case_dir / "training_case2.yaml"

            resolved = normalize_runtime_config_paths(
                {"logging": {"output_dir": "./results/local_case"}},
                config_path=config_path,
            )

            self.assertEqual(
                resolved["logging"]["output_dir"],
                str((case_dir / "results" / "local_case").resolve()),
            )


class TrainingBatchPlanTests(unittest.TestCase):
    def test_load_batch_training_plan_resolves_relative_cases_and_skips_disabled_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_batch_plan_") as tmpdir:
            base = Path(tmpdir).resolve()
            batch_dir = base / "configs" / "training_cases"
            batch_dir.mkdir(parents=True, exist_ok=True)

            case1 = batch_dir / "training_case1.yaml"
            case2 = batch_dir / "training_case2.yaml"
            case1.write_text("logging:\n  output_dir: results/case1\n", encoding="utf-8")
            case2.write_text("logging:\n  output_dir: results/case2\n", encoding="utf-8")

            batch_config = batch_dir / "train_batch.yaml"
            batch_config.write_text(
                "\n".join(
                    [
                        "name: nightly",
                        "cases:",
                        "  - training_case1.yaml",
                        "  - path: training_case2",
                        "    enabled: true",
                        "  - path: training_case3.yaml",
                        "    enabled: false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload, case_paths = load_batch_training_plan(batch_config)

            self.assertEqual(payload["name"], "nightly")
            self.assertEqual(case_paths, [case1.resolve(), case2.resolve()])


if __name__ == "__main__":
    unittest.main()
