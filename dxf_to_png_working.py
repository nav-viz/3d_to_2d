#!/usr/bin/env python3
"""
dxf_to_png_svg.py - Convert DXF to PNG via SVG + Pillow with timestamped output
"""

import argparse
from pathlib import Path
from datetime import datetime

import ezdxf
from PIL import Image, ImageDraw, ImageFont


def get_timestamp() -> str:
    """Return current timestamp in YYYYMMDD_HHMMSS format"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dxf_to_png(dxf_path: str | Path, png_path: str | Path | None = None, dpi: int = 300):
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        print(f"❌ DXF not found: {dxf_path}")
        return

    timestamp = get_timestamp()
    base_name = dxf_path.stem

    # Auto-generate timestamped PNG if none provided
    if png_path is None:
        png_path = dxf_path.parent / f"{base_name}_{timestamp}.png"
    else:
        png_path = Path(png_path)
        # If user gave a directory, append timestamped name
        if png_path.is_dir():
            png_path = png_path / f"{base_name}_{timestamp}.png"

    print(f"Loading DXF: {dxf_path.name} → Generating {png_path.name}")

    # Read DXF
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Export to SVG (timestamped too)
    svg_path = dxf_path.with_name(f"{base_name}_{timestamp}.svg")

    try:
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.svg import SVGBackend
        from ezdxf.addons.drawing import layout as dxf_layout
        from ezdxf.math import BoundingBox2d

        ctx = RenderContext(doc)
        backend = SVGBackend()
        frontend = Frontend(ctx, backend)

        for entity in msp:
            try:
                frontend.draw_entity(entity)
            except Exception:
                continue

        # ezdxf 1.x: finalize is on the backend, get_string needs a Page
        backend.finalize()
        page = dxf_layout.Page(594, 420)                       # A2 in mm
        render_box = BoundingBox2d([(0, 0), (594, 420)])
        svg_string = backend.get_string(page, render_box=render_box)

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_string)

        print(f"SVG exported: {svg_path.name}")

    except Exception as e:
        print(f"SVG export failed: {e}")

    # Convert SVG → PNG
    try:
        from cairosvg import svg2png
        with open(svg_path, "rb") as f:
            svg_data = f.read()

        png_bytes = svg2png(bytestring=svg_data, dpi=dpi)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        print(f"✅ High-quality PNG created: {png_path} @ {dpi} DPI (timestamp: {timestamp})")

    except ImportError:
        print("cairosvg not installed → using basic fallback")
        img = Image.new("RGB", (1200, 800), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), f"DXF: {dxf_path.name}\nTimestamp: {timestamp}\n\nPNG conversion fallback", fill="red")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(png_path)
        print(f"Basic PNG saved: {png_path}")

    except Exception as e:
        print(f"Final conversion failed: {e}")
        img = Image.new("RGB", (1200, 800), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), f"DXF: {dxf_path.name}\nTimestamp: {timestamp}\n\nError during conversion", fill="red")
        img.save(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DXF to PNG via SVG with timestamp")
    parser.add_argument("dxf_file", help="Input DXF file")
    parser.add_argument("-o", "--output", help="Output PNG path (or directory)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution")

    args = parser.parse_args()

    dxf_to_png(
        dxf_path=args.dxf_file,
        png_path=args.output,
        dpi=args.dpi,
    )