"""
drawing/dxf_writer.py
=====================
DXF export engine using ezdxf.

Produces ISO-compliant DXF R2010 files with:
  - Proper layer structure (VISIBLE, HIDDEN, DIM, GDT, BORDER, TITLE)
  - One modelspace viewport per view
  - Dimension entities (linear, radial, diameter)
  - Basic GD&T feature control frames (as MTEXT blocks)
  - Title block with part name, scale, date, revision

Extension points
----------------
- Add more GD&T symbols in ``_add_gdt_annotation``
- Implement proper ISO sheet templates via ezdxf layouts
- Add surface finish symbols per face feature type
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any
from loguru import logger

try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    from ezdxf import colors as dxf_colors
    _EZDXF_AVAILABLE = True
except ImportError:
    _EZDXF_AVAILABLE = False
    logger.error(
        "ezdxf not installed. 'pip install ezdxf==1.1.4'"
    )

from drawing.projection import ProjectionResult, Polyline
from drawing.dimension_placer import DimensionSpec


# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

LAYERS = {
    "VISIBLE": {"color": 7,  "linetype": "CONTINUOUS", "lineweight": 35},
    "HIDDEN":  {"color": 8,  "linetype": "DASHED",     "lineweight": 18},
    "CENTER":  {"color": 1,  "linetype": "CENTER",      "lineweight": 9},
    "DIM":     {"color": 2,  "linetype": "CONTINUOUS",  "lineweight": 13},
    "GDT":     {"color": 4,  "linetype": "CONTINUOUS",  "lineweight": 13},
    "BORDER":  {"color": 5,  "linetype": "CONTINUOUS",  "lineweight": 50},
    "TITLE":   {"color": 7,  "linetype": "CONTINUOUS",  "lineweight": 25},
    "HATCH":   {"color": 254,"linetype": "CONTINUOUS",  "lineweight": 9},
}

# Sheet sizes in mm (width, height)
SHEET_SIZES: dict[str, tuple[float, float]] = {
    "A0": (1189, 841),
    "A1": (841, 594),
    "A2": (594, 420),
    "A3": (420, 297),
    "A4": (297, 210),
}

# View layout: (col, row) multiplier for offset within the sheet
VIEW_LAYOUT: dict[str, tuple[int, int]] = {
    "front":     (1, 1),
    "top":       (1, 2),
    "right":     (2, 1),
    "left":      (0, 1),
    "bottom":    (1, 0),
    "isometric": (2, 2),
}


# ---------------------------------------------------------------------------
# Main DXF writer
# ---------------------------------------------------------------------------

class DXFWriter:
    """
    Composes a multi-view technical drawing as a DXF file.

    Parameters
    ----------
    cfg : dict  – full pipeline config
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        if not _EZDXF_AVAILABLE:
            raise ImportError("ezdxf is required for DXF export.")
        self.cfg = cfg
        dc = cfg["drawing"]
        self.sheet_size = dc.get("sheet_size", "A2")
        self.scale_str  = dc.get("scale", "1:1")
        self.text_h     = dc.get("dimension_text_height", 3.5)
        self.arrow_sz   = dc.get("arrow_size", 2.5)
        self.gdt_h      = dc.get("gdt_text_height", 2.5)
        self.lw_vis     = dc.get("line_weight_visible", 0.35)
        self.lw_hid     = dc.get("line_weight_hidden", 0.18)

        self._doc: Any = None
        self._msp: Any = None   # modelspace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, part_name: str = "CAD_PART") -> None:
        """Initialise a new DXF document."""
        self._doc = ezdxf.new(dxfversion="R2010", setup=True)
        self._msp = self._doc.modelspace()
        self._setup_layers()
        self._setup_dimstyle()
        self._draw_border(part_name)
        logger.debug(f"DXF document created: part_name={part_name}")

    def add_projection(
        self,
        result: ProjectionResult,
        offset: tuple[float, float] = (0, 0),
    ) -> None:
        """
        Add one projection view (visible + hidden lines) to modelspace.

        Parameters
        ----------
        result : ProjectionResult from ``drawing.projection``
        offset : (x, y) translation to place the view on the sheet
        """
        ox, oy = offset

        for pl in result.visible_lines:
            self._draw_polyline(pl, ox, oy, layer="VISIBLE")
        for pl in result.hidden_lines:
            self._draw_polyline(pl, ox, oy, layer="HIDDEN")

        # View label
        h = self.text_h * 0.9
        label_y = oy - 8.0
        self._msp.add_text(
            result.view_name.upper(),
            dxfattribs={
                "layer": "TITLE",
                "height": h,
                "insert": (ox + (result.bounding_box[2] - result.bounding_box[0]) / 2, label_y),
                "halign": 1,  # centre
            }
        )

    def add_dimension(
        self,
        dim: DimensionSpec,
        offset: tuple[float, float] = (0, 0),
    ) -> None:
        """Add a single dimension entity to modelspace."""
        if not dim.placed:
            return
        ox, oy = offset
        tp = (dim.text_point[0] + ox, dim.text_point[1] + oy)
        p1 = (dim.p1[0] + ox, dim.p1[1] + oy)
        p2 = (dim.p2[0] + ox, dim.p2[1] + oy)

        if dim.dim_type == "LINEAR":
            self._msp.add_linear_dim(
                base=tp,
                p1=p1,
                p2=p2,
                dimstyle="ISO-25",
                override={"dimtxt": self.text_h, "dimasz": self.arrow_sz},
                dxfattribs={"layer": "DIM"},
            ).render()

        elif dim.dim_type in ("DIAMETER", "RADIUS"):
            cx = (p1[0] + p2[0]) / 2
            cy = (p1[1] + p2[1]) / 2
            radius = math.dist(p1, p2) / 2
            prefix = "⌀" if dim.dim_type == "DIAMETER" else "R"
            diag_pt = (cx + radius * 0.7071, cy + radius * 0.7071)
            self._msp.add_radius_dim(
                center=(cx, cy),
                radius=radius,
                angle=45,
                dimstyle="ISO-25",
                override={"dimtxt": self.text_h, "dimasz": self.arrow_sz,
                          "dimpost": f"{prefix}<>"},
                dxfattribs={"layer": "DIM"},
            ).render()

    def add_gdt_annotation(
        self,
        position: tuple[float, float],
        symbol: str,
        tolerance: str,
        datum: str = "",
    ) -> None:
        """
        Add a basic GD&T feature control frame as MTEXT.

        Extension point: generate proper ISO 1101 GD&T frames here.
        """
        frame_text = f"|{symbol}|{tolerance}|{datum}|"
        self._msp.add_mtext(
            frame_text,
            dxfattribs={
                "layer": "GDT",
                "char_height": self.gdt_h,
                "insert": position,
            }
        )

    def save(self, output_path: str | Path) -> None:
        """Write the DXF document to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._doc.saveas(str(output_path))
        logger.success(f"DXF saved → {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_layers(self) -> None:
        """Create all drawing layers."""
        for name, props in LAYERS.items():
            try:
                self._doc.layers.new(
                    name=name,
                    dxfattribs={
                        "color": props["color"],
                        "linetype": props["linetype"],
                        "lineweight": props["lineweight"],
                    }
                )
            except ezdxf.DXFTableEntryError:
                pass   # layer already exists

    def _setup_dimstyle(self) -> None:
        """Configure the ISO-25 dimension style."""
        try:
            dimstyle = self._doc.dimstyles.get("ISO-25")
            dimstyle.dxf.dimtxt  = self.text_h
            dimstyle.dxf.dimasz  = self.arrow_sz
            dimstyle.dxf.dimexe  = 2.0
            dimstyle.dxf.dimexo  = 1.5
            dimstyle.dxf.dimgap  = 1.0
        except Exception:
            pass

    def _draw_polyline(
        self,
        pl: Polyline,
        ox: float,
        oy: float,
        layer: str,
    ) -> None:
        """Add a polyline to modelspace with the given offset."""
        if len(pl.points) < 2:
            return
        translated = [(x + ox, y + oy) for x, y in pl.points]
        self._msp.add_lwpolyline(
            translated,
            dxfattribs={"layer": layer}
        )

    def _draw_border(self, part_name: str) -> None:
        """Draw outer sheet border and basic title block."""
        W, H = SHEET_SIZES.get(self.sheet_size, (420, 297))
        margin = 15.0
        inner_left   = margin
        inner_bottom = margin
        inner_right  = W - margin
        inner_top    = H - margin

        # Outer border
        border = [
            (inner_left,  inner_bottom),
            (inner_right, inner_bottom),
            (inner_right, inner_top),
            (inner_left,  inner_top),
            (inner_left,  inner_bottom),
        ]
        self._msp.add_lwpolyline(border, dxfattribs={"layer": "BORDER"})

        # Title block (simplified bottom-right box)
        tb_h = 30.0
        tb_left = inner_right - 180.0
        tb_box = [
            (tb_left, inner_bottom),
            (inner_right, inner_bottom),
            (inner_right, inner_bottom + tb_h),
            (tb_left, inner_bottom + tb_h),
            (tb_left, inner_bottom),
        ]
        self._msp.add_lwpolyline(tb_box, dxfattribs={"layer": "BORDER"})

        # Title text
        today = datetime.date.today().isoformat()
        texts = [
            (f"{part_name}",                    3.5, (tb_left + 5, inner_bottom + 20)),
            (f"Scale: {self.scale_str}",         2.5, (tb_left + 5, inner_bottom + 12)),
            (f"Date: {today}",                   2.5, (tb_left + 5, inner_bottom + 6)),
            ("CAD Automation Pipeline v0.1",     2.0, (tb_left + 5, inner_bottom + 2)),
        ]
        for txt, h, pos in texts:
            self._msp.add_text(
                txt,
                dxfattribs={"layer": "TITLE", "height": h, "insert": pos}
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def export_dxf(
    projections: list[ProjectionResult],
    dimensions: list[DimensionSpec],
    predictions: list[dict[str, Any]],
    output_path: str | Path,
    cfg: dict[str, Any],
    part_name: str = "CAD_PART",
) -> None:
    """
    Full DXF export: compose all views, dimensions, and GD&T annotations.

    Parameters
    ----------
    projections  : list of ProjectionResult (one per selected view)
    dimensions   : placed DimensionSpec objects
    predictions  : GNN predictions (for GD&T decisions)
    output_path  : output ``.dxf`` file path
    cfg          : full pipeline config
    part_name    : displayed in title block
    """
    writer = DXFWriter(cfg)
    writer.create(part_name=part_name)

    W, H = SHEET_SIZES.get(cfg["drawing"]["sheet_size"], (594, 420))
    margin = 15.0
    view_w = (W - 2 * margin) / 3
    view_h = (H - 2 * margin - 30) / 3   # 30 = title block height

    # ── Place views ───────────────────────────────────────────────────────────
    placed_views: set[str] = set()
    for proj in projections:
        col, row = VIEW_LAYOUT.get(proj.view_name, (0, 0))
        ox = margin + col * view_w
        oy = margin + 30 + row * view_h   # +30 for title block
        writer.add_projection(proj, offset=(ox, oy))
        placed_views.add(proj.view_name)

    # ── Place dimensions on the front view ───────────────────────────────────
    front_col, front_row = VIEW_LAYOUT.get("front", (1, 1))
    dim_offset = (
        margin + front_col * view_w,
        margin + 30 + front_row * view_h,
    )
    for dim in dimensions:
        writer.add_dimension(dim, offset=dim_offset)

    # ── GD&T annotations (minimal) ────────────────────────────────────────────
    _add_gdt_for_holes(writer, predictions, dim_offset)

    writer.save(output_path)


def _add_gdt_for_holes(
    writer: DXFWriter,
    predictions: list[dict],
    offset: tuple[float, float],
) -> None:
    """
    Add cylindricity GD&T annotation for high-confidence holes.

    Extension point: expand with flatness, perpendicularity, etc.
    """
    ox, oy = offset
    y_pos = oy + 5.0
    for pred in predictions:
        if (
            "Hole" in pred.get("feature_type", "")
            and pred.get("confidence", 0.0) > 0.75
        ):
            cx = ox + float(pred.get("face_id", 0)) * 15 % 150
            writer.add_gdt_annotation(
                position=(cx, y_pos),
                symbol="⌀",   # cylindricity symbol stub
                tolerance="0.05",
                datum="A",
            )
            y_pos += 10.0
