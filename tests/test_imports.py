"""Import smoke test: every bench module compiles, and imports unless an optional or private dependency is missing."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"

# Modules that run their work at import time (writes under bench/ or subprocess pipelines). They are compiled only.
RUNS_ON_IMPORT = {"baselines/oracle", "baselines/run_all", "ai/ablation_jobs"}
# Top-level packages that the quick path does not need. A module that needs one of them is skipped, not failed.
OPTIONAL_ROOTS = {
    "pycalphad",
    "tinydb",
    "sympy",
    "cv2",
    "pdfplumber",
    "camelot",
    "pytesseract",
    "google",
    "thermo_extractor",
    "thermo_table_reader",
    "text_reader",
    "text_reader_v6",
    "assessed_pd",
    "gemini_table_reader",
}


def _module_names(bench_dir):
    """Return the dotted-path-style names of every bench module, sorted, with no __pycache__ entries."""
    names = []
    for path in sorted(bench_dir.rglob("*.py")):
        if "__pycache__" not in path.parts:
            names.append(path.relative_to(bench_dir).with_suffix("").as_posix())
    return names


def pytest_generate_tests(metafunc):
    """Parametrise the module tests over every bench module."""
    names = _module_names(BENCH)
    if "compile_name" in metafunc.fixturenames:
        metafunc.parametrize("compile_name", names)
    if "import_name" in metafunc.fixturenames:
        metafunc.parametrize("import_name", [n for n in names if n not in RUNS_ON_IMPORT])


def test_module_compiles(compile_name, bench_dir):
    """Every bench module, including those that run work on import, is valid Python source."""
    path = bench_dir / f"{compile_name}.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_module_imports_or_skips_on_missing_optional_dependency(import_name):
    """Every bench module imports; a missing optional or private dependency skips the module instead of failing it."""
    try:
        importlib.import_module(import_name.replace("/", "."))
    except ModuleNotFoundError as exc:
        root = (exc.name or "").split(".")[0]
        if root in OPTIONAL_ROOTS:
            pytest.skip(f"optional or private dependency missing: {exc.name}")
        raise


def test_run_on_import_modules_are_known(bench_dir):
    """The modules excluded from the import check still exist, so the exclusion list cannot go stale silently."""
    for name in RUNS_ON_IMPORT:
        assert (bench_dir / f"{name}.py").is_file(), name
