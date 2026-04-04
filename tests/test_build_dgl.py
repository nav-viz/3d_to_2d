"""
tests/test_build_dgl.py
=======================
Unit tests for the DGL heterograph builder.

Tests cover:
  - MFCAD_H5_Dataset: schema integrity, feature shapes, label shapes
  - build_from_h5: single-sample graph structure
  - _build_heterograph: edge type partitioning logic
  - build_from_neo4j: graph construction from stub topology

Run:
    pytest tests/test_build_dgl.py -v
    pytest tests/ --cov=graph --cov-report=term-missing
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_h5_file(tmp_path: Path) -> Path:
    """Create a minimal synthetic H5 file for testing."""
    import h5py
    h5_path = tmp_path / "sample.h5"

    N = 10    # faces
    M = 15    # edges

    rng = np.random.default_rng(42)
    src = rng.integers(0, N, size=M)
    dst = rng.integers(0, N, size=M)
    A_idx = np.stack([src, dst], axis=0)

    E1 = rng.integers(0, 2, size=M, dtype=np.int32)
    E2 = rng.integers(0, 2, size=M, dtype=np.int32)
    E3 = rng.integers(0, 2, size=M, dtype=np.int32)

    with h5py.File(h5_path, "w") as fh:
        grp = fh.create_group("part_0001")
        grp.create_dataset("V_1",       data=rng.random((N, 10), dtype="f4"))
        grp.create_dataset("Y",         data=rng.integers(0, 25, size=N, dtype="i4"))
        grp.create_dataset("A_1_idx",   data=A_idx.astype(np.int64))
        grp.create_dataset("A_1_values",data=np.ones(M, dtype="f4"))
        grp.create_dataset("E_1",       data=E1)
        grp.create_dataset("E_2",       data=E2)
        grp.create_dataset("E_3",       data=E3)

    return h5_path


@pytest.fixture
def stub_topology() -> dict:
    """Minimal topology dict (simulates parser output without OCC)."""
    N = 6
    faces = [
        {"id": i, "surface_type": "Plane", "area": float(i + 1), "occ_face": None}
        for i in range(N)
    ]
    edges = [
        {"id": i, "occ_edge": None, "convexity": "unknown", "face_ids": []}
        for i in range(8)
    ]
    return {
        "shape": None,
        "faces": faces,
        "edges": edges,
        "vertices": [],
        "num_faces": N,
        "num_edges": 8,
        "num_vertices": 0,
        "source_file": "stub.step",
    }


# ---------------------------------------------------------------------------
# Tests: MFCAD_H5_Dataset
# ---------------------------------------------------------------------------

class TestMFCADH5Dataset:
    """Tests for ``graph.mfcad_h5_dataset.MFCAD_H5_Dataset``."""

    def test_len(self, sample_h5_file: Path) -> None:
        """Dataset should have one sample per top-level H5 key."""
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        ds = MFCAD_H5_Dataset(sample_h5_file, normalise=False)
        assert len(ds) == 1, f"Expected 1 sample, got {len(ds)}"

    def test_getitem_shapes(self, sample_h5_file: Path) -> None:
        """Graph and label shapes must be consistent with H5 data."""
        import torch
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        ds = MFCAD_H5_Dataset(sample_h5_file, normalise=False)
        g, labels = ds[0]
        assert labels.dtype == torch.int64, "Labels must be int64"
        assert labels.shape[0] == 10, "Expected 10 face labels"

    def test_normalise(self, sample_h5_file: Path) -> None:
        """Normalised dataset should compute stats without error."""
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        ds = MFCAD_H5_Dataset(sample_h5_file, normalise=True)
        assert ds._mean is not None
        assert ds._std is not None
        assert ds._mean.shape == (10,)

    def test_max_samples(self, sample_h5_file: Path) -> None:
        """max_samples should limit dataset size."""
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        ds = MFCAD_H5_Dataset(sample_h5_file, normalise=False, max_samples=0)
        assert len(ds) == 0

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing H5 file."""
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        with pytest.raises(FileNotFoundError):
            MFCAD_H5_Dataset("non_existent.h5")


# ---------------------------------------------------------------------------
# Tests: _build_heterograph
# ---------------------------------------------------------------------------

class TestBuildHeterograph:
    """Tests for the static ``_build_heterograph`` method."""

    def test_basic_structure(self, sample_h5_file: Path) -> None:
        """Graph must have 'face' nodes with 'feat' data."""
        try:
            import dgl
        except ImportError:
            pytest.skip("DGL not installed")
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        ds = MFCAD_H5_Dataset(sample_h5_file, normalise=False)
        g, _ = ds[0]
        assert "face" in g.ntypes
        assert "feat" in g.nodes["face"].data
        assert g.nodes["face"].data["feat"].shape == (10, 10)

    def test_edge_types_present(self, sample_h5_file: Path) -> None:
        """At least one edge type must be present."""
        try:
            import dgl
        except ImportError:
            pytest.skip("DGL not installed")
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
        ds = MFCAD_H5_Dataset(sample_h5_file, normalise=False)
        g, _ = ds[0]
        assert len(g.etypes) > 0, "Graph must have at least one edge type"

    def test_all_edge_masks(self) -> None:
        """Edge masks should partition edges into correct types."""
        try:
            import dgl
        except ImportError:
            pytest.skip("DGL not installed")
        import torch
        from graph.mfcad_h5_dataset import MFCAD_H5_Dataset

        N, M = 5, 6
        src = np.array([0, 1, 1, 2, 3, 4], dtype=np.int64)
        dst = np.array([1, 0, 2, 3, 4, 0], dtype=np.int64)
        A_idx  = np.stack([src, dst])
        A_vals = np.ones(M, dtype=np.float32)
        E1 = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        E2 = np.array([0, 1, 0, 0, 0, 0], dtype=np.int32)
        E3 = np.array([0, 0, 1, 0, 0, 0], dtype=np.int32)
        V1 = np.zeros((N, 10), dtype=np.float32)

        g = MFCAD_H5_Dataset._build_heterograph(A_idx, A_vals, E1, E2, E3, V1)
        canonical = {(s, r, d) for s, r, d in g.canonical_etypes}
        assert ("face", "convex",  "face") in canonical
        assert ("face", "concave", "face") in canonical
        assert ("face", "smooth",  "face") in canonical
        # Remaining 3 edges → adj
        assert ("face", "adj",     "face") in canonical


# ---------------------------------------------------------------------------
# Tests: build_from_neo4j
# ---------------------------------------------------------------------------

class TestBuildFromNeo4j:
    """Tests for Neo4j-sourced graph construction with stub client."""

    def test_stub_topology_no_error(self, stub_topology: dict) -> None:
        """build_from_neo4j must succeed with stub topology + no Neo4j."""
        try:
            import dgl
        except ImportError:
            pytest.skip("DGL not installed")

        from graph.build_dgl import build_from_neo4j
        from kg.neo4j_client import Neo4jClient

        cfg = {
            "graph": {"face_feat_dim": 10, "edge_feat_dim": 0,
                      "normalise_features": False},
        }

        # Use stub client (Neo4j not required)
        client = Neo4jClient.__new__(Neo4jClient)
        client._driver = None

        g = build_from_neo4j(
            model_id="test_model_001",
            client=client,
            topology=stub_topology,
            cfg=cfg,
        )
        assert "face" in g.ntypes
        assert g.num_nodes("face") == 6


# ---------------------------------------------------------------------------
# Tests: view selector
# ---------------------------------------------------------------------------

class TestViewSelector:
    def test_returns_views(self) -> None:
        from reasoning.view_selector import select_views
        cfg = {
            "view_selection": {
                "max_views": 4,
                "min_confidence": 0.4,
                "view_candidates": ["front", "top", "right", "isometric"],
                "simple_max_faces": 50,
                "medium_max_faces": 300,
            }
        }
        preds = [
            {"face_id": 0, "feature_type": "Through_Hole", "confidence": 0.9},
            {"face_id": 1, "feature_type": "Stock",         "confidence": 0.95},
        ]
        views = select_views(preds, cfg, num_faces=10)
        assert len(views) >= 1
        assert views[0].name in ["front", "top", "right", "isometric"]


# ---------------------------------------------------------------------------
# Tests: DimensionPlacer
# ---------------------------------------------------------------------------

class TestDimensionPlacer:
    def test_greedy_places_all(self) -> None:
        from drawing.dimension_placer import DimensionSpec, place_dimensions_greedy
        dims = [
            DimensionSpec("LINEAR", (0, 0), (50, 0), 50.0),
            DimensionSpec("DIAMETER", (10, 10), (30, 10), 20.0),
        ]
        placed = place_dimensions_greedy(dims, bbox=(0, 0, 100, 80))
        assert all(d.placed for d in placed)

    def test_resolve_collisions(self) -> None:
        from drawing.dimension_placer import DimensionSpec, resolve_collisions
        d1 = DimensionSpec("LINEAR", (0, 0), (50, 0), 50.0,
                           text_point=(25.0, 90.0), placed=True)
        d2 = DimensionSpec("LINEAR", (0, 0), (50, 0), 50.0,
                           text_point=(25.0, 91.0), placed=True)
        resolved = resolve_collisions([d1, d2], clearance=8.0)
        dy = abs(resolved[1].text_point[1] - resolved[0].text_point[1])
        assert dy >= 8.0 - 1e-6, f"Clearance violated: dy={dy}"
