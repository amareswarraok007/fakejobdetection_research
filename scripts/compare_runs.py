"""Check that two runs of the same config produced the same metrics (PRD M8).

Usage: python scripts/compare_runs.py runs/e0_reproduce/<stored> runs/e0_reproduce/<fresh>
Exit code 0 = identical (within 1e-9), 1 = differences listed.
"""

import argparse
import logging
import sys

from cvx.report.compare import compare_runs


def main() -> int:
    """Compare the two runs and print the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    found = compare_runs(args.run_a, args.run_b, args.tolerance)
    for line in found:
        logging.info("DIFFERENT %s", line)
    logging.info("Runs match." if not found else f"{len(found)} difference(s).")
    return 0 if not found else 1


if __name__ == "__main__":
    sys.exit(main())
