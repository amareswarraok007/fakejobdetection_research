# CareerVerify-XAI (CVX)

A leakage-aware, explainable fake job posting detector built on the EMSCAD dataset.
The full specification is in [docs/PRD.md](docs/PRD.md); agent rules are in `CLAUDE.md`.

> This README is a stub for milestone M1. The full README (Colab steps, experiments,
> model export, app) is written in milestone M8.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
python -m nltk.downloader stopwords wordnet   # needed by the P-CV preprocessing
```

Put the Kaggle file `fake_job_postings.csv` in `data/`.

## Tests and lint

```bash
pytest -q
ruff check src tests
```

## Data profile

```bash
python -m cvx.data.profile --config configs/base.yaml
```

Writes `reports/data_profile.json` (every statistic in PRD Section 6.3), computed from the CSV.
