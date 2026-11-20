"""Write PR curves, ROC curves and confusion matrices to reports/figures/ from run folders.

Usage: python scripts/make_figures.py --runs runs --out reports/figures
"""

import argparse
import logging

from cvx.report.figures import write_figures


def main() -> int:
    """Parse arguments and write the figures (all logic lives in cvx.report.figures)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", help="Folder holding the run folders")
    parser.add_argument("--out", default="reports/figures", help="Output folder")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    write_figures(args.runs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
