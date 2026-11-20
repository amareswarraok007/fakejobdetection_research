"""Export chosen models from a run folder into app/model_store (PRD Section 10.6).

Usage:
  python scripts/export_model.py --run runs/e4_ablations/<timestamp> \\
      --models lr_full,lr_text,xgb_full,xgb_text --out app/model_store \\
      --data data/fake_job_postings.csv
"""

import argparse
import logging

from cvx.artifacts.save import export_models


def main() -> int:
    """Parse arguments and export (all logic lives in cvx.artifacts.save)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Complete run folder with models/")
    parser.add_argument(
        "--models", required=True, help="Comma-separated ids, e.g. lr_full,xgb_text"
    )
    parser.add_argument("--out", default="app/model_store", help="Model store folder")
    parser.add_argument("--data", default="data/fake_job_postings.csv", help="Dataset CSV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export_models(
        args.run, [m.strip() for m in args.models.split(",") if m.strip()], args.out, args.data
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
