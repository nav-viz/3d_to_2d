"""
run_pipeline.py – root-level entry point for the CAD Drawing Generator.

Prompts the user for a STEP file path (or uses the synthetic demo when no
path is provided) then runs the full pipeline.
"""
from __future__ import annotations

import sys

from src.pipeline import CADPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Interactive pipeline entry point."""
    print("=" * 60)
    print("  Hybrid AI CAD Drawing Generator")
    print("=" * 60)

    try:
        step_file = input(
            "Enter STEP file path (or press Enter for demo): "
        ).strip()
    except EOFError:
        step_file = ""

    if not step_file:
        logger.info("No path provided – running synthetic demo pipeline.")

    pipeline = CADPipeline()
    result = pipeline.run(step_file_path=step_file)

    print("\n=== OUTPUT FILES ===")
    print(f"  DXF   : {result['dxf_path']}")
    print(f"  PNG   : {result['png_path']}")
    print(f"  JSON  : {result['features_json']}")
    print(f"  Graph : {result['graph_json']}")
    print(f"  Time  : {result['elapsed_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
