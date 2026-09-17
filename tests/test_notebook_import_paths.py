"""Test notebook import paths."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
class NotebookImportPathTests(unittest.TestCase):
    def _run_import_check(self, cwd: Path) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from notebooks import multisource_notebook_helpers as helpers; "
                    "from src.data_pipeline import load_point_centric_arrays; "
                    "from src.diagnostics import load_results_bundle; "
                    "assert (helpers.REPO_ROOT / 'src').is_dir(); "
                    "assert callable(load_point_centric_arrays); "
                    "assert callable(load_results_bundle)"
                ),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"Import check failed for cwd={cwd}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )

    def test_imports_work_from_repo_root(self) -> None:
        self._run_import_check(REPO_ROOT)

if __name__ == "__main__":
    unittest.main()
