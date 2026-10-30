"""Golden-hash regression: generate.py output for two systems at the fixed seed, as recorded on this tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SEED = 20261022
SYSTEMS = ["AgCu-06He", "BiSn-08Li"]
DOC_IDS = ["syn-AgCu-06He-v0", "syn-BiSn-08Li-v0"]
# sha256 of the canonical JSON of each text output, recorded from this tree (binary digests removed).
GOLDEN = {
    "docs/MANIFEST.json": "c82699a312133aceb4727a045a80669c15f5696226c2777c8a46e997d1e820dd",
    "truth/syn-AgCu-06He-v0.json": "6c16f5e3d9ff9d15f2eb7030e78eeeeed7d0e95410fb3b671f64c6f329b3972d",
    "truth/syn-BiSn-08Li-v0.json": "b527466ad19fefa5a08577f804f0dbff034cd0e81104e31e3385a3d5e07b4599",
}
FIGURE_FILES = ["vector.pdf", "image_only.pdf", "figs/fig1.pdf", "figs/fig2.pdf", "figs/fig3.pdf", "figs/fig4.pdf"]


def _strip_binary_digests(node, under_renderings=False):
    """Remove the digests of rasters and PDFs (library-dependent bytes) and the manifest's per-truth digests."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "truth_sha256" or (key == "sha256" and under_renderings):
                continue
            out[key] = _strip_binary_digests(value, under_renderings or key == "renderings")
        return out
    if isinstance(node, list):
        return [_strip_binary_digests(value, under_renderings) for value in node]
    return node


def canonical_digest(path):
    """Return the sha256 of a JSON file re-serialised canonically, after removing binary digests."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    text = json.dumps(_strip_binary_digests(data), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def generated(tmp_path_factory, bench_dir):
    """Run generate.py on a copy of the code (no docs, truth or outputs) and return the copy's bench directory."""
    for module in ("numpy", "matplotlib", "pymupdf", "PIL", "scipy"):
        pytest.importorskip(module)
    root = tmp_path_factory.mktemp("golden")
    shutil.copytree(bench_dir, root / "bench", ignore=shutil.ignore_patterns("docs", "truth", "outputs", "__pycache__", "work"))
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
    cmd = [sys.executable, "bench/generate.py", "--seed", str(SEED), "--systems", *SYSTEMS, "--jobs", "1"]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return root / "bench"


@pytest.mark.slow
class TestGeneratorGolden:
    """generate.py reproduces the recorded text outputs for the fixed seed and two systems."""

    @pytest.mark.parametrize("rel", sorted(GOLDEN))
    def test_text_output_digest_matches_recorded(self, generated, rel):
        """The canonical JSON of each text output hashes to the value recorded from this tree."""
        assert canonical_digest(generated / rel) == GOLDEN[rel]

    def test_manifest_lists_each_requested_system(self, generated):
        """The manifest records the seed and exactly one document per requested system."""
        manifest = json.loads((generated / "docs" / "MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["seed"] == SEED
        assert [d["doc_id"] for d in manifest["documents"]] == DOC_IDS
        assert all(d["pages"] == 4 for d in manifest["documents"])

    @pytest.mark.parametrize("doc_id", DOC_IDS)
    def test_figures_and_pages_written(self, generated, doc_id):
        """Each document writes its vector and image-only PDFs, its four figure PDFs and its raster pages."""
        doc = generated / "docs" / doc_id
        for name in FIGURE_FILES:
            assert (doc / name).stat().st_size > 0, name
        assert len(list((doc / "clean").glob("page-*.png"))) == 4
        assert len(list((doc / "degraded").glob("page-*.jpg"))) == 4
