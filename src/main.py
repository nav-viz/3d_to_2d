"""Entry point for the CAD Drawing Generator."""
from __future__ import annotations

import argparse
import sys

from src.pipeline import CADPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid AI CAD Drawing Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.main --step-file part.step\n"
            "  python -m src.main  # runs synthetic demo\n"
        ),
    )
    parser.add_argument(
        "--step-file",
        default="",
        help="Path to the STEP file.  Omit to run the synthetic demo.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    step_file = args.step_file

    if not step_file:
        logger.info("No --step-file provided.  Running synthetic demo pipeline.")

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
