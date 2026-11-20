"""Write Markdown result tables to reports/tables/ from run folders.

Usage: python scripts/make_tables.py --runs runs --out reports/tables
"""

import argparse
import logging

from cvx.report.tables import write_tables


def main() -> int:
    """Parse arguments and write the tables (all logic lives in cvx.report.tables)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", help="Folder holding the run folders")
    parser.add_argument("--out", default="reports/tables", help="Output folder")
    parser.add_argument(
        "--paper", default="docs/career_verify_reported.yaml", help="Owner-filled paper values"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    write_tables(args.runs, args.out, args.paper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
