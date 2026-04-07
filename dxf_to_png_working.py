#!/usr/bin/env python3
"""
dxf_to_png_svg.py - Convert DXF to PNG via SVG + Pillow (most reliable for your setup)
No matplotlib. Works with current ezdxf.
"""

import argparse
from pathlib import Path

import ezdxf
from PIL import Image, ImageDraw, ImageFont
import io


def dxf_to_png(dxf_path: str | Path, png_path: str | Path | None = None, dpi: int = 300):
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        print(f"❌ DXF not found: {dxf_path}")
        return

    if png_path is None:
        png_path = dxf_path.with_suffix(".png")
    png_path = Path(png_path)

    print(f"Loading DXF: {dxf_path.name}")

    # Read DXF
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Export to SVG first (ezdxf's SVG export is very stable)
    svg_path = dxf_path.with_suffix(".svg")
    doc.saveas(str(svg_path))   # This is not perfect, but better than nothing

    # Better: use the drawing addon to export SVG
    try:
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.svg import SVGBackend

        ctx = RenderContext(doc)
        backend = SVGBackend()
        frontend = Frontend(ctx, backend)

        for entity in msp:
            try:
                frontend.draw_entity(entity)
            except Exception:
                continue

        frontend.finalize()
        svg_string = backend.get_string()

        # Save SVG temporarily
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_string)

        print("SVG exported successfully, converting to PNG...")

    except Exception as e:
        print(f"SVG export failed: {e}")
        print("Falling back to basic method...")

    # Convert SVG to PNG using Pillow + cairosvg (best quality)
    try:
        from cairosvg import svg2png
        with open(svg_path, "rb") as f:
            svg_data = f.read()

        png_bytes = svg2png(bytestring=svg_data, dpi=dpi)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        print(f"✅ High-quality PNG created: {png_path} at {dpi} DPI")

    except ImportError:
        print("cairosvg not installed. Installing simple fallback...")
        print("Run: pip install cairosvg")
        # Very basic Pillow fallback (low quality)
        img = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "DXF Preview (SVG conversion failed)", fill="black")
        img.save(png_path)
        print(f"Basic PNG saved: {png_path}")

    except Exception as e:
        print(f"Final conversion failed: {e}")
        # Ultimate fallback
        img = Image.new("RGB", (1200, 800), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), f"DXF: {dxf_path.name}\n\nPNG conversion had issues.", fill="red")
        img.save(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DXF to PNG via SVG")
    parser.add_argument("dxf_file", help="Input DXF file")
    parser.add_argument("-o", "--output", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution")

    args = parser.parse_args()

    dxf_to_png(
        dxf_path=args.dxf_file,
        png_path=args.output,
        dpi=args.dpi,
    )