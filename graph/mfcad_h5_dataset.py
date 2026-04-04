"""
graph/mfcad_h5_dataset.py
=========================
Enhanced MFCAD++ H5 dataset loader with proper normalisation and heterograph building.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from loguru import logger

try:
    import dgl
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False
    logger.error("DGL not installed. Run: pip install dgl==1.1.2")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 25
FACE_FEAT_DIM = 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MFCAD_H5_Dataset(Dataset):
    """
    Loads MFCAD++ H5 files and returns DGL heterographs.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        normalise: bool = True,
        cache: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        self.normalise = normalise
        self.cache = cache
        self._graph_cache: dict[int, tuple] = {}

        # Enumerate part keys
        with h5py.File(self.h5_path, "r") as fh:
            self._keys: list[str] = list(fh.keys())

        if max_samples is not None:
            self._keys = self._keys[:max_samples]

        logger.info(
            f"MFCAD_H5_Dataset: {len(self._keys)} samples from {self.h5_path.name}"
        )

        # Detect actual feature dimension from first sample
        with h5py.File(self.h5_path, "r") as fh:
            sample_key = self._keys[0]
            self.feat_dim = fh[sample_key]["V_1"].shape[1]
            logger.info(f"Detected face feature dimension: {self.feat_dim}")

        # Compute normalisation stats
        if normalise:
            self._mean, self._std = self._compute_stats()
        else:
            self._mean = np.zeros(self.feat_dim, dtype=np.float32)
            self._std = np.ones(self.feat_dim, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> tuple:
        if self.cache and idx in self._graph_cache:
            return self._graph_cache[idx]

        key = self._keys[idx]
        with h5py.File(self.h5_path, "r") as fh:
            grp = fh[key]

            # Node features
            V1 = np.array(grp["V_1"], dtype=np.float32)

            # Labels — support both "Y" and "labels"
            if "labels" in grp:
                Y = np.array(grp["labels"], dtype=np.int64)
            elif "Y" in grp:
                Y = np.array(grp["Y"], dtype=np.int64)
            else:
                raise KeyError(f"No label dataset found in group '{key}'. "
                               f"Available: {list(grp.keys())}")
            print(f"Sample {key}: {V1.shape[0]} faces, labels shape = {Y.shape}")
            # Adjacency
            A1_idx = np.array(grp["A_1_idx"], dtype=np.int64)
            A1_vals = np.array(grp["A_1_values"], dtype=np.float32)

            # Edge masks — optional (E_1, E_2, E_3)
            if "E_1" in grp and "E_2" in grp and "E_3" in grp:
                E1 = np.array(grp["E_1"], dtype=np.int32)
                E2 = np.array(grp["E_2"], dtype=np.int32)
                E3 = np.array(grp["E_3"], dtype=np.int32)
            else:
                # Fallback: all edges treated as generic "adj"
                M = A1_idx.shape[1] if len(A1_idx.shape) > 1 else len(A1_idx[0])
                E1 = np.zeros(M, dtype=np.int32)
                E2 = np.zeros(M, dtype=np.int32)
                E3 = np.zeros(M, dtype=np.int32)
                logger.debug(f"Group {key}: No E_1/E_2/E_3 found → using all edges as 'adj'")

        # Normalise features
        V1 = (V1 - self._mean) / (self._std + 1e-8)

        # Build graph
        g = self._build_heterograph(A1_idx, A1_vals, E1, E2, E3, V1)
        labels = torch.from_numpy(Y)

        result = (g, labels)
        if self.cache:
            self._graph_cache[idx] = result
        return result

    @staticmethod
    def _build_heterograph(
        A1_idx: np.ndarray,
        A1_vals: np.ndarray,
        E1: np.ndarray,
        E2: np.ndarray,
        E3: np.ndarray,
        V1: np.ndarray,
    ) -> "dgl.DGLHeteroGraph":
        if not _DGL_AVAILABLE:
            return {"V1": torch.from_numpy(V1), "A1_idx": torch.from_numpy(A1_idx)}

        src = A1_idx[0] if len(A1_idx.shape) > 1 else A1_idx[:, 0]
        dst = A1_idx[1] if len(A1_idx.shape) > 1 else A1_idx[:, 1]

        # If no edge type masks, treat everything as "adj"
        if E1.sum() == 0 and E2.sum() == 0 and E3.sum() == 0:
            data = {("face", "adj", "face"): (torch.from_numpy(src).long(), 
                                             torch.from_numpy(dst).long())}
        else:
            # Original partitioning logic...
            e1_mask = E1.astype(bool)
            e2_mask = E2.astype(bool) & ~e1_mask
            e3_mask = E3.astype(bool) & ~e1_mask & ~e2_mask
            e_adj   = ~e1_mask & ~e2_mask & ~e3_mask

            def _edges(mask):
                return (torch.from_numpy(src[mask]).long(), 
                        torch.from_numpy(dst[mask]).long())

            data: dict[tuple, tuple] = {}
            if e1_mask.any(): data[("face", "convex", "face")] = _edges(e1_mask)
            if e2_mask.any(): data[("face", "concave", "face")] = _edges(e2_mask)
            if e3_mask.any(): data[("face", "smooth", "face")] = _edges(e3_mask)
            if e_adj.any():   data[("face", "adj", "face")] = _edges(e_adj)

            if not data:
                data[("face", "adj", "face")] = (torch.arange(V1.shape[0]), 
                                                 torch.arange(V1.shape[0]))

        g = dgl.heterograph(data, num_nodes_dict={"face": V1.shape[0]})
        g.nodes["face"].data["feat"] = torch.from_numpy(V1)
        return g

    def _compute_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and std across all faces in the dataset."""
        logger.info("Computing feature normalisation statistics...")

        n_total = 0
        running_sum = np.zeros(self.feat_dim, dtype=np.float64)
        running_sum2 = np.zeros(self.feat_dim, dtype=np.float64)

        with h5py.File(self.h5_path, "r") as fh:
            for key in self._keys:
                V1 = np.array(fh[key]["V_1"], dtype=np.float64)
                running_sum += V1.sum(axis=0)
                running_sum2 += (V1 ** 2).sum(axis=0)
                n_total += V1.shape[0]

        mean = (running_sum / n_total).astype(np.float32)
        var = (running_sum2 / n_total - mean.astype(np.float64) ** 2)
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        logger.debug(f"Feature mean={mean[:3]}…  std={std[:3]}…")
        return mean, std


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    h5_path: Union[str, Path],
    batch_size: int = 4,          # Changed default to 4 (safe now)
    shuffle: bool = True,
    normalise: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> "torch.utils.data.DataLoader":
    """
    Returns DataLoader that properly handles variable-sized graphs.
    """
    ds = MFCAD_H5_Dataset(
        h5_path, normalise=normalise, max_samples=max_samples
    )

    if _DGL_AVAILABLE:
        from dgl.dataloading import GraphDataLoader
        return GraphDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda x: x,   # Critical: manual batch handling
        )
    else:
        from torch.utils.data import DataLoader
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda x: x,
        )