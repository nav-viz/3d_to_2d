"""PNG renderer for DXF files."""
from __future__ import annotations

import math
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try ezdxf matplotlib backend
try:
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger.info("ezdxf matplotlib backend available.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.info("ezdxf matplotlib backend not available – using basic PNG renderer.")

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


class PNGRenderer:
    """
    Renders a DXF file to a PNG image.

    Tries the ezdxf matplotlib backend first; falls back to a simple
    numpy-based placeholder when matplotlib is not available.
    """

    def __init__(self, dpi: int = 150, bg_color: str = "white") -> None:
        self._dpi = dpi
        self._bg_color = bg_color
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, dxf_path: str, output_path: str) -> bool:
        """
        Render a DXF file to PNG.

        Args:
            dxf_path: Path to the source DXF file.
            output_path: Destination PNG file path.

        Returns:
            ``True`` when the file was written successfully.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if MATPLOTLIB_AVAILABLE and EZDXF_AVAILABLE:
            return self._render_matplotlib(dxf_path, output_path)
        return self._render_fallback(dxf_path, output_path)

    # ------------------------------------------------------------------
    # Matplotlib backend
    # ------------------------------------------------------------------

    def _render_matplotlib(self, dxf_path: str, output_path: str) -> bool:
        """Use ezdxf's Matplotlib backend for high-quality rendering."""
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            fig = plt.figure(figsize=(16, 12), facecolor=self._bg_color)
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            backend = MatplotlibBackend(ax)
            frontend = Frontend(ctx, backend)
            frontend.draw_layout(msp, finalize=True)
            fig.savefig(output_path, dpi=self._dpi, bbox_inches="tight")
            plt.close(fig)
            self._log.info("PNG rendered (matplotlib) → '%s'", output_path)
            return True
        except Exception as exc:
            self._log.warning("Matplotlib render failed (%s); using fallback.", exc)
            return self._render_fallback(dxf_path, output_path)

    # ------------------------------------------------------------------
    # Fallback renderer
    # ------------------------------------------------------------------

    def _render_fallback(self, dxf_path: str, output_path: str) -> bool:
        """
        Create a simple placeholder PNG by parsing DXF line entities.

        Reads LINE and CIRCLE entities from the DXF modelspace and draws them
        on a white canvas using NumPy array operations.
        """
        try:
            if EZDXF_AVAILABLE:
                return self._numpy_render_from_dxf(dxf_path, output_path)
            return self._create_blank_png(output_path)
        except Exception as exc:
            self._log.error("Fallback render failed: %s", exc)
            return self._create_blank_png(output_path)

    def _numpy_render_from_dxf(self, dxf_path: str, output_path: str) -> bool:
        """Parse DXF and draw entities to a PNG using NumPy + basic imaging."""
        import numpy as np
        try:
            doc = ezdxf.readfile(dxf_path)
        except Exception as exc:
            self._log.warning("Could not read DXF '%s': %s", dxf_path, exc)
            return self._create_blank_png(output_path)

        width_px, height_px = 1200, 900
        canvas = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

        lines = []
        circles = []
        for entity in doc.modelspace():
            if entity.dxftype() == "LINE":
                s = entity.dxf.start
                e = entity.dxf.end
                lines.append(((s.x, s.y), (e.x, e.y)))
            elif entity.dxftype() == "CIRCLE":
                c = entity.dxf.center
                circles.append((c.x, c.y, entity.dxf.radius))

        # Determine coordinate range
        all_x = [p[0] for seg in lines for p in seg] + [c[0] for c in circles]
        all_y = [p[1] for seg in lines for p in seg] + [c[1] for c in circles]
        if not all_x:
            return self._create_blank_png(output_path)

        mx, Mx = min(all_x), max(all_x)
        my, My = min(all_y), max(all_y)
        rng_x = max(Mx - mx, 1.0)
        rng_y = max(My - my, 1.0)

        def world_to_pixel(x, y):
            px = int((x - mx) / rng_x * (width_px - 40) + 20)
            py = int(height_px - 20 - (y - my) / rng_y * (height_px - 40))
            return max(0, min(px, width_px - 1)), max(0, min(py, height_px - 1))

        def draw_line_bresenham(canvas, x0, y0, x1, y1, color=(0, 0, 0)):
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            while True:
                if 0 <= x0 < width_px and 0 <= y0 < height_px:
                    canvas[y0, x0] = color
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

        for (sx, sy), (ex, ey) in lines:
            px0, py0 = world_to_pixel(sx, sy)
            px1, py1 = world_to_pixel(ex, ey)
            draw_line_bresenham(canvas, px0, py0, px1, py1)

        for cx, cy, r in circles:
            pcx, pcy = world_to_pixel(cx, cy)
            pr = int(r / rng_x * (width_px - 40))
            pr = max(2, min(pr, 200))
            for angle_deg in range(0, 360, 2):
                angle = math.radians(angle_deg)
                px = int(pcx + pr * math.cos(angle))
                py = int(pcy + pr * math.sin(angle))
                if 0 <= px < width_px and 0 <= py < height_px:
                    canvas[py, px] = (0, 0, 200)

        self._save_png(canvas, output_path)
        self._log.info("PNG rendered (NumPy fallback) → '%s'", output_path)
        return True

    def _create_blank_png(self, output_path: str) -> bool:
        """Write a blank white PNG as a last-resort fallback."""
        import numpy as np
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 240
        self._save_png(canvas, output_path)
        self._log.info("Blank PNG placeholder written → '%s'", output_path)
        return True

    @staticmethod
    def _save_png(canvas, output_path: str) -> None:
        """Save a NumPy RGB array as a PNG file using a minimal writer."""
        import struct
        import zlib
        height, width, _ = canvas.shape

        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            body = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
            return length + body + crc

        # PNG signature
        signature = b"\x89PNG\r\n\x1a\n"
        # IHDR
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr = png_chunk(b"IHDR", ihdr_data)
        # IDAT
        raw_rows = b""
        for row in canvas:
            raw_rows += b"\x00" + bytes(row.flatten().tolist())
        compressed = zlib.compress(raw_rows, 6)
        idat = png_chunk(b"IDAT", compressed)
        # IEND
        iend = png_chunk(b"IEND", b"")

        with open(output_path, "wb") as f:
            f.write(signature + ihdr + idat + iend)
