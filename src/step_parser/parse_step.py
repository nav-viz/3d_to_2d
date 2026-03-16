"""STEP file parser with OCC fallback to synthetic data."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional pythonOCC import
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    OCC_AVAILABLE = True
    logger.info("pythonOCC is available – using native STEP parser.")
except ImportError:
    OCC_AVAILABLE = False
    logger.info("pythonOCC not available – using synthetic fallback parser.")


@dataclass
class Vertex:
    """A single 3-D vertex in the CAD model."""
    id: int
    x: float
    y: float
    z: float

    @property
    def coords(self) -> Tuple[float, float, float]:
        """Return vertex coordinates as a tuple."""
        return (self.x, self.y, self.z)


@dataclass
class Edge:
    """A topological edge connecting two vertices."""
    id: int
    start_vertex_id: int
    end_vertex_id: int
    edge_type: str = "LINE"  # LINE | ARC | SPLINE
    length: float = 0.0
    mid_point: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))


@dataclass
class Face:
    """A topological face of the solid."""
    id: int
    surface_type: str = "PLANE"  # PLANE | CYLINDER | CONE | SPHERE | TORUS | BSPLINE
    area: float = 0.0
    normal: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 1.0))
    center: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    vertices: List[int] = field(default_factory=list)
    edges: List[int] = field(default_factory=list)
    curvature: float = 0.0
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(
        default_factory=lambda: ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    )


@dataclass
class Surface:
    """Parametric surface data associated with a Face."""
    face_id: int
    surface_type: str = "PLANE"
    params: dict = field(default_factory=dict)


@dataclass
class ParsedCADModel:
    """
    Container for all topology and geometry extracted from a STEP file (or
    synthetic generator).
    """
    faces: List[Face] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)
    surfaces: List[Surface] = field(default_factory=list)
    source_file: str = ""
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(
        default_factory=lambda: ((0.0, 0.0, 0.0), (100.0, 100.0, 50.0))
    )
    metadata: dict = field(default_factory=dict)


class StepParser:
    """
    Parses a STEP file into a :class:`ParsedCADModel`.

    When pythonOCC is available the real geometry is read; otherwise a
    synthetic model with a plausible topology is generated so the remainder
    of the pipeline can run without any CAD kernel installed.
    """

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> ParsedCADModel:
        """
        Parse a STEP file (or generate synthetic data when OCC is absent).

        Args:
            file_path: Path to the .stp / .step file.  When empty or the file
                       cannot be found the synthetic generator is used.

        Returns:
            A fully populated :class:`ParsedCADModel`.
        """
        if OCC_AVAILABLE and file_path:
            try:
                return self._parse_with_occ(file_path)
            except Exception as exc:
                self._log.warning("OCC parsing failed (%s); using synthetic fallback.", exc)

        if file_path:
            self._log.info("Using synthetic fallback for: %s", file_path)
        else:
            self._log.info("No file path provided – generating synthetic model.")
        return self._synthetic_model(file_path)

    # ------------------------------------------------------------------
    # OCC parser (only reached when OCC_AVAILABLE is True)
    # ------------------------------------------------------------------

    def _parse_with_occ(self, file_path: str) -> ParsedCADModel:  # pragma: no cover
        """Parse a STEP file using pythonOCC."""
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEPControl_Reader could not read '{file_path}'")
        reader.TransferRoots()
        shape = reader.OneShape()

        faces: List[Face] = []
        edges: List[Edge] = []
        vertices: List[Vertex] = []

        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
        from OCC.Core.GProp import GProp_GProps

        face_id = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_shape = explorer.Current()
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face_shape, props)
            area = props.Mass()
            cog = props.CentreOfMass()
            faces.append(Face(
                id=face_id,
                surface_type="PLANE",
                area=float(area),
                center=(float(cog.X()), float(cog.Y()), float(cog.Z())),
            ))
            face_id += 1
            explorer.Next()

        self._log.info("OCC: extracted %d faces.", len(faces))
        model = ParsedCADModel(faces=faces, edges=edges, vertices=vertices, source_file=file_path)
        model.metadata["source"] = "occ"
        return model

    # ------------------------------------------------------------------
    # Synthetic model
    # ------------------------------------------------------------------

    def _synthetic_model(self, source_file: str = "") -> ParsedCADModel:
        """Generate a deterministic synthetic :class:`ParsedCADModel`."""
        rng = random.Random(42)

        # Vertices of a simple box with a cylindrical hole
        raw_vertices = [
            (0, 0, 0), (100, 0, 0), (100, 80, 0), (0, 80, 0),
            (0, 0, 30), (100, 0, 30), (100, 80, 30), (0, 80, 30),
        ]
        vertices = [Vertex(i, x, y, z) for i, (x, y, z) in enumerate(raw_vertices)]

        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),   # top
            (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
        ]
        edges = []
        for eid, (s, t) in enumerate(box_edges):
            vs, vt = raw_vertices[s], raw_vertices[t]
            length = sum((a - b) ** 2 for a, b in zip(vs, vt)) ** 0.5
            mid = tuple((a + b) / 2 for a, b in zip(vs, vt))
            edges.append(Edge(eid, s, t, "LINE", length, mid))  # type: ignore[arg-type]

        # Box faces
        face_defs = [
            ("PLANE", [0, 1, 2, 3], [0, 1, 2, 3], (0.0, 0.0, -1.0), (50, 40, 0)),
            ("PLANE", [4, 5, 6, 7], [4, 5, 6, 7], (0.0, 0.0, 1.0), (50, 40, 30)),
            ("PLANE", [0, 1, 5, 4], [0, 8, 4, 9], (0.0, -1.0, 0.0), (50, 0, 15)),
            ("PLANE", [1, 2, 6, 5], [1, 9, 5, 10], (1.0, 0.0, 0.0), (100, 40, 15)),
            ("PLANE", [2, 3, 7, 6], [2, 10, 6, 11], (0.0, 1.0, 0.0), (50, 80, 15)),
            ("PLANE", [3, 0, 4, 7], [3, 11, 7, 8], (-1.0, 0.0, 0.0), (0, 40, 15)),
            ("CYLINDER", [], [], (0.0, 0.0, 1.0), (50, 40, 15)),  # hole
        ]
        faces = []
        for fid, (stype, vids, eids, normal, center) in enumerate(face_defs):
            area = rng.uniform(800, 8000) if stype == "PLANE" else rng.uniform(200, 600)
            curvature = 0.0 if stype == "PLANE" else rng.uniform(0.01, 0.1)
            faces.append(Face(
                id=fid,
                surface_type=stype,
                area=area,
                normal=normal,  # type: ignore[arg-type]
                center=center,  # type: ignore[arg-type]
                vertices=vids,
                edges=eids,
                curvature=curvature,
                bounding_box=((0.0, 0.0, 0.0), (100.0, 80.0, 30.0)),
            ))

        surfaces = [Surface(f.id, f.surface_type) for f in faces]

        model = ParsedCADModel(
            faces=faces,
            edges=edges,
            vertices=vertices,
            surfaces=surfaces,
            source_file=source_file,
            bounding_box=((0.0, 0.0, 0.0), (100.0, 80.0, 30.0)),
            metadata={"source": "synthetic", "complexity": 1},
        )
        self._log.info(
            "Synthetic model: %d faces, %d edges, %d vertices.",
            len(faces), len(edges), len(vertices),
        )
        return model
