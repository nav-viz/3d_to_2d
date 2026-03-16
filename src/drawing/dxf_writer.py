"""DXF writer using ezdxf."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import ezdxf
from ezdxf import units
from ezdxf.document import Drawing as EzdxfDoc

from src.dimensioning.dimension_generator import DimensionCandidate
from src.drawing.view_projection import ProjectedView
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# DXF layer names
LAYER_VISIBLE = "VISIBLE"
LAYER_HIDDEN = "HIDDEN"
LAYER_CENTER = "CENTER"
LAYER_DIMENSIONS = "DIMENSIONS"
LAYER_TITLEBLOCK = "TITLEBLOCK"
LAYER_BORDER = "BORDER"


class DXFWriter:
    """
    Creates a DXF drawing document from projected views and dimension candidates.

    Uses ezdxf to produce an ISO-compliant drawing with:
    - Proper layer structure
    - Orthographic view projections
    - Linear and diameter dimension annotations
    - Title block with part metadata
    """

    def __init__(self, font_size: float = settings.FONT_SIZE) -> None:
        self._font_size = font_size
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_drawing(
        self,
        views: Dict[str, ProjectedView],
        dimensions: List[DimensionCandidate],
        metadata: Optional[Dict] = None,
        sheet_size: str = "A3",
    ) -> EzdxfDoc:
        """
        Build a complete DXF document.

        Args:
            views: Projected views keyed by view name.
            dimensions: Optimised dimension candidates.
            metadata: Optional part metadata (title, drawing number …).
            sheet_size: ISO sheet size string.

        Returns:
            An :class:`ezdxf.document.Drawing` ready to save.
        """
        doc = ezdxf.new(dxfversion="R2010")
        doc.units = units.MM
        msp = doc.modelspace()
        self._setup_layers(doc)

        width, height = settings.SHEET_SIZES.get(sheet_size, (420, 297))

        # Sheet border
        self._draw_border(msp, width, height)
        # Title block
        self._draw_title_block(msp, width, height, metadata or {})
        # Orthographic views
        for vname, view in views.items():
            self._draw_view(msp, view)
        # Dimensions
        self._draw_dimensions(msp, dimensions)

        self._log.info(
            "DXF document created: %d views, %d dimensions.",
            len(views),
            len(dimensions),
        )
        return doc

    def save(self, doc: EzdxfDoc, path: str) -> None:
        """
        Save a DXF document to *path*.

        Args:
            doc: The ezdxf document object.
            path: Output file path (will create parent directories).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        doc.saveas(path)
        self._log.info("DXF saved to '%s'.", path)

    # ------------------------------------------------------------------
    # Layer setup
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_layers(doc: EzdxfDoc) -> None:
        """Define standard drawing layers."""
        layer_defs = [
            (LAYER_VISIBLE, 7, "CONTINUOUS"),     # white / black
            (LAYER_HIDDEN, 8, "DASHED"),           # grey
            (LAYER_CENTER, 1, "CENTER"),            # red center lines
            (LAYER_DIMENSIONS, 3, "CONTINUOUS"),   # green
            (LAYER_TITLEBLOCK, 5, "CONTINUOUS"),   # blue
            (LAYER_BORDER, 7, "CONTINUOUS"),
        ]
        for name, color, linetype in layer_defs:
            layer = doc.layers.new(name=name)
            layer.color = color
            try:
                if linetype != "CONTINUOUS":
                    doc.linetypes.get(linetype)  # ensure it exists
                layer.linetype = linetype
            except Exception:
                pass  # linetype not defined; use default

    # ------------------------------------------------------------------
    # Border and title block
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_border(msp, width: int, height: int) -> None:
        """Draw the sheet border rectangle."""
        margin = 10
        points = [
            (margin, margin),
            (width - margin, margin),
            (width - margin, height - margin),
            (margin, height - margin),
            (margin, margin),
        ]
        msp.add_lwpolyline(points, dxfattribs={"layer": LAYER_BORDER})

    @staticmethod
    def _draw_title_block(msp, width: int, height: int, metadata: Dict) -> None:
        """Draw a simple title block in the lower-right corner."""
        tb_x = width - 160
        tb_y = 10
        tb_w = 150
        tb_h = 50

        # Title block border
        corners = [
            (tb_x, tb_y),
            (tb_x + tb_w, tb_y),
            (tb_x + tb_w, tb_y + tb_h),
            (tb_x, tb_y + tb_h),
            (tb_x, tb_y),
        ]
        msp.add_lwpolyline(corners, dxfattribs={"layer": LAYER_TITLEBLOCK})

        # Text entries
        title = metadata.get("title", "UNTITLED")
        drw_no = metadata.get("drawing_number", "DRW-000")
        revision = metadata.get("revision", "A")
        date = metadata.get("date", "")

        texts = [
            (title[:30], tb_x + 5, tb_y + 35, 3.5),
            (f"DRW: {drw_no}", tb_x + 5, tb_y + 22, 2.5),
            (f"REV: {revision}", tb_x + 5, tb_y + 14, 2.5),
            (f"DATE: {date}", tb_x + 5, tb_y + 6, 2.0),
            ("HYBRID AI CAD GENERATOR", tb_x + 5, tb_y + 42, 2.0),
        ]
        for text, x, y, height_mm in texts:
            msp.add_text(
                text,
                dxfattribs={
                    "layer": LAYER_TITLEBLOCK,
                    "height": height_mm,
                    "insert": (x, y),
                },
            )

    # ------------------------------------------------------------------
    # View drawing
    # ------------------------------------------------------------------

    def _draw_view(self, msp, view: ProjectedView) -> None:
        """Render all lines and arcs of a projected view."""
        ox, oy = view.origin
        scale = view.scale

        for line in view.lines:
            sx = ox + line.start[0] * scale
            sy = oy + line.start[1] * scale
            ex = ox + line.end[0] * scale
            ey = oy + line.end[1] * scale
            layer = LAYER_HIDDEN if line.line_type == "HIDDEN" else LAYER_VISIBLE
            msp.add_line((sx, sy), (ex, ey), dxfattribs={"layer": layer})

        for arc in view.arcs:
            cx = ox + arc.center[0] * scale
            cy = oy + arc.center[1] * scale
            r = arc.radius * scale
            if arc.start_angle == 0.0 and arc.end_angle == 360.0:
                msp.add_circle(
                    (cx, cy), r,
                    dxfattribs={"layer": LAYER_VISIBLE}
                )
                # Center lines
                cl = r + 3.0
                msp.add_line(
                    (cx - cl, cy), (cx + cl, cy),
                    dxfattribs={"layer": LAYER_CENTER},
                )
                msp.add_line(
                    (cx, cy - cl), (cx, cy + cl),
                    dxfattribs={"layer": LAYER_CENTER},
                )
            else:
                msp.add_arc(
                    (cx, cy), r,
                    arc.start_angle,
                    arc.end_angle,
                    dxfattribs={"layer": LAYER_VISIBLE},
                )

        # View label
        msp.add_text(
            view.view_name,
            dxfattribs={
                "layer": LAYER_DIMENSIONS,
                "height": self._font_size,
                "insert": (ox, oy - 8),
            },
        )

    # ------------------------------------------------------------------
    # Dimension drawing
    # ------------------------------------------------------------------

    def _draw_dimensions(
        self, msp, dimensions: List[DimensionCandidate]
    ) -> None:
        """Render dimension annotations on the DIMENSIONS layer."""
        for dim in dimensions:
            self._draw_single_dim(msp, dim)

    def _draw_single_dim(self, msp, dim: DimensionCandidate) -> None:
        """Draw a single dimension candidate as text + extension lines."""
        sx, sy = dim.start_point
        ex, ey = dim.end_point

        # Extension lines
        msp.add_line(
            (sx, sy), (ex, ey),
            dxfattribs={"layer": LAYER_DIMENSIONS},
        )

        # Dimension text at midpoint
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        text = dim.text or f"{dim.value:.2f}"
        msp.add_text(
            text,
            dxfattribs={
                "layer": LAYER_DIMENSIONS,
                "height": self._font_size,
                "insert": (mx, my + self._font_size * 1.2),
            },
        )
