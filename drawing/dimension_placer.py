"""
drawing/dimension_placer.py
============================
Dimension extraction and placement engine.

Two strategies are supported (configured via ``drawing.placement_strategy``):

1. **CSP** (Constraint Satisfaction Problem via ``python-constraint``):
   - Variables: dimension position offsets (x, y)
   - Domain:    discrete candidate positions around the view bounding box
   - Constraints: no overlap, arrow length within range, readable clearance
   - Timeout: falls back to greedy if CSP solver exceeds ``csp_timeout``

2. **Greedy**:
   - Sort dimensions by type priority (overall → feature → tolerance)
   - Place each dimension at the first non-colliding position

Dimension types supported
--------------------------
- LINEAR  : point-to-point distance
- RADIUS  : arc/circle radius annotation
- DIAMETER: hole diameter annotation
- ANGULAR : angle between two straight edges (stub)

Extension point
---------------
Add GD&T symbol placement by extending ``_place_gdt`` below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from utils.logger import logger

try:
    from constraint import Problem, AllDifferentConstraint
    _CSP_AVAILABLE = True
except ImportError:
    _CSP_AVAILABLE = False
    logger.warning(
        "python-constraint not installed – using greedy placement only. "
        "'pip install python-constraint'"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionSpec:
    """A single dimension annotation."""
    dim_type: str          # "LINEAR" | "RADIUS" | "DIAMETER" | "ANGULAR"
    p1: tuple[float, float]   # first reference point (2D)
    p2: tuple[float, float]   # second reference point (for LINEAR/ANGULAR)
    value: float              # dimension value (mm or degrees)
    unit: str = "mm"
    text_override: str = ""   # leave empty to auto-format
    # Placement (filled after placement algorithm)
    text_point: tuple[float, float] = field(default=(0.0, 0.0))
    offset: float = 8.0       # mm offset from entity
    placed: bool = False
    feature_type: str = "Generic"   # origin feature type for filtering

    @property
    def display_text(self) -> str:
        if self.text_override:
            return self.text_override
        unit_sym = "°" if self.unit == "deg" else " mm"
        prefix = "⌀" if self.dim_type == "DIAMETER" else (
            "R" if self.dim_type == "RADIUS" else ""
        )
        return f"{prefix}{self.value:.2f}{unit_sym}"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_dimensions(
    predictions: list[dict[str, Any]],
    topology: dict[str, Any],
) -> list[DimensionSpec]:
    """
    Extract dimension candidates from GNN predictions + B-Rep topology.

    This creates one or more ``DimensionSpec`` objects per predicted feature.
    For Holes/Bosses → DIAMETER, Pockets → LINEAR (depth + width), etc.
    """
    dims: list[DimensionSpec] = []
    faces = topology.get("faces", [])

    for pred in predictions:
        ftype = pred.get("feature_type", "Stock")
        fid   = pred.get("face_id", 0)
        conf  = pred.get("confidence", 0.0)
        params = pred.get("params", {})

        if conf < 0.3 or ftype in ("Stock", "Low_Confidence", "Unknown"):
            continue

        face_data = faces[fid] if fid < len(faces) else {}
        area = face_data.get("area", 0.0)

        if "Hole" in ftype or "Boss" in ftype:
            radius = float(params.get("radius", 5.0))
            # Stub reference points (replace with actual projected circle centre)
            cx, cy = float(fid * 15 % 200), float(fid * 10 % 160)
            dims.append(DimensionSpec(
                dim_type="DIAMETER",
                p1=(cx - radius, cy),
                p2=(cx + radius, cy),
                value=radius * 2,
                feature_type=ftype,
                offset=10.0,
            ))
            # Depth
            depth = float(params.get("depth", 10.0))
            if depth > 0:
                dims.append(DimensionSpec(
                    dim_type="LINEAR",
                    p1=(cx, cy), p2=(cx, cy - depth),
                    value=depth,
                    feature_type=ftype,
                    text_override=f"↧{depth:.2f}",
                ))

        elif "Pocket" in ftype or "Slot" in ftype or "Step" in ftype:
            w = max(float(area) ** 0.5, 5.0)
            bx, by = float(fid * 15 % 200), float(fid * 10 % 160)
            dims.append(DimensionSpec(
                dim_type="LINEAR",
                p1=(bx, by), p2=(bx + w, by),
                value=w,
                feature_type=ftype,
            ))

    logger.info(f"Dimension extraction: {len(dims)} dimensions found.")
    return dims


# ---------------------------------------------------------------------------
# Placement – CSP
# ---------------------------------------------------------------------------

def place_dimensions_csp(
    dims: list[DimensionSpec],
    bbox: tuple[float, float, float, float],
    timeout: float = 5.0,
) -> list[DimensionSpec]:
    """
    Use CSP to find non-overlapping dimension text positions.

    Falls back to greedy if CSP times out or is unavailable.
    """
    if not _CSP_AVAILABLE or not dims:
        return place_dimensions_greedy(dims, bbox)

    xmin, ymin, xmax, ymax = bbox
    margin = 15.0
    step   = 12.0   # grid step in mm

    # Candidate y-offsets: above and below the bounding box
    candidates = [
        ymax + margin + i * step for i in range(len(dims))
    ] + [
        ymin - margin - i * step for i in range(len(dims))
    ]

    problem = Problem()
    for i in range(len(dims)):
        problem.addVariable(f"y_{i}", candidates)

    # Constraint: all y-positions differ by at least ``step``
    # (proxy for no-overlap in vertical arrangement)
    def no_overlap(*args: float) -> bool:
        vals = sorted(args)
        return all(abs(vals[j+1] - vals[j]) >= step * 0.9 for j in range(len(vals)-1))

    var_names = [f"y_{i}" for i in range(len(dims))]
    problem.addConstraint(no_overlap, var_names)

    t0 = time.time()
    solution = None
    try:
        solution = problem.getSolution()
    except Exception:
        pass
    elapsed = time.time() - t0

    if solution is None or elapsed >= timeout:
        logger.warning(
            f"CSP timeout ({elapsed:.2f}s) or no solution – falling back to greedy."
        )
        return place_dimensions_greedy(dims, bbox)

    # Apply solution
    for i, dim in enumerate(dims):
        mid_x = (dim.p1[0] + dim.p2[0]) / 2
        dim.text_point = (mid_x, solution[f"y_{i}"])
        dim.placed = True

    logger.info(f"CSP placement: {len(dims)} dimensions placed in {elapsed:.2f}s")
    return dims


# ---------------------------------------------------------------------------
# Placement – Greedy
# ---------------------------------------------------------------------------

def place_dimensions_greedy(
    dims: list[DimensionSpec],
    bbox: tuple[float, float, float, float],
) -> list[DimensionSpec]:
    """
    Greedy dimension placement: stack dimensions above the bounding box.
    Each successive dimension is placed one step higher.
    """
    xmin, ymin, xmax, ymax = bbox
    margin = 15.0
    step   = 12.0
    used_y: set[float] = set()

    for i, dim in enumerate(dims):
        mid_x = (dim.p1[0] + dim.p2[0]) / 2
        y_candidate = ymax + margin + i * step
        # Simple collision: bump if already used
        while round(y_candidate, 1) in used_y:
            y_candidate += step
        used_y.add(round(y_candidate, 1))
        dim.text_point = (mid_x, y_candidate)
        dim.placed = True

    logger.info(f"Greedy placement: {len(dims)} dimensions placed.")
    return dims


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------

def resolve_collisions(
    dims: list[DimensionSpec],
    clearance: float = 8.0,
) -> list[DimensionSpec]:
    """
    Post-placement collision resolution: nudge overlapping text positions
    vertically until clearance is satisfied.
    """
    dims_sorted = sorted(dims, key=lambda d: d.text_point[1])
    for i in range(1, len(dims_sorted)):
        prev = dims_sorted[i - 1]
        curr = dims_sorted[i]
        dy = curr.text_point[1] - prev.text_point[1]
        if dy < clearance:
            shift = clearance - dy
            curr.text_point = (curr.text_point[0], curr.text_point[1] + shift)
    return dims_sorted


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def place_and_resolve(
    dims: list[DimensionSpec],
    bbox: tuple[float, float, float, float],
    cfg: dict[str, Any],
) -> list[DimensionSpec]:
    """
    Main entry point: run the configured placement strategy + collision fix.
    """
    strategy = cfg["drawing"].get("placement_strategy", "csp")
    timeout  = cfg["drawing"].get("csp_timeout", 5.0)

    if strategy == "csp":
        dims = place_dimensions_csp(dims, bbox, timeout=timeout)
    else:
        dims = place_dimensions_greedy(dims, bbox)

    dims = resolve_collisions(dims)
    return dims
