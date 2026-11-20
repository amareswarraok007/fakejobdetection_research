# CareerVerify-XAI (CVX)

A leakage-aware, explainable fake job posting detector built on the EMSCAD dataset.
The full specification is in [docs/PRD.md](docs/PRD.md); agent rules are in `CLAUDE.md`.

Contents: 1 lock file · 2 laptop setup · 3 tests · 4 data profile · 5 experiments in Colab ·
6 reports · 7 export models and run the app · 8 reproducibility check · 9 project map.

## 1. Create the lock file (first time only, in Colab)

`requirements-lock.txt` holds the exact library versions that Colab and the laptop both install.
Colab runs Linux and the laptop runs Windows, so a full `pip freeze` would list Linux-only packages
(Colab tools, NVIDIA GPU libraries) that cannot install on Windows. This cell pins **only** the
packages in `requirements.in` plus `numpy`, `scipy`, `joblib` and `threadpoolctl`.

In a Colab notebook, first clone the repo and move into it:

```python
!git clone https://github.com/<your-account>/<your-repo>.git
%cd <your-repo>
```

Then run this cell:

```python
# Lock-file cell: installs requirements.in and writes requirements-lock.txt with exact
# versions of ONLY the packages in requirements.in plus numpy, scipy, joblib, threadpoolctl.
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

assert sys.version_info[:2] == (3, 13), f"Expected Python 3.13, got {sys.version}"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.in"], check=True)

EXTRA = ["numpy", "scipy", "joblib", "threadpoolctl"]
names = []
for line in Path("requirements.in").read_text(encoding="utf-8").splitlines():
    name = re.split(r"[\s#;\[<>=!~]", line.strip(), maxsplit=1)[0]  # drop comments/extras
    if name:
        names.append(name)


def norm(name: str) -> str:
    """Standard package-name form, so 'PyYAML' and 'pyyaml' count as the same package."""
    return re.sub(r"[-_.]+", "-", name).lower()


pins, seen = [], set()
for name in names + EXTRA:
    if norm(name) not in seen:
        seen.add(norm(name))
        pins.append(f"{name}=={version(name)}")

banned = [p for p in pins if norm(p).startswith(("google-colab", "colab", "nvidia"))]
assert not banned, f"Colab/NVIDIA packages must not be locked: {banned}"

Path("requirements-lock.txt").write_text(
    "# Generated in Colab by the lock-file cell in README.md. Do not edit by hand.\n"
    f"# Python {sys.version.split()[0]}\n" + "\n".join(pins) + "\n",
    encoding="utf-8",
)
print(Path("requirements-lock.txt").read_text(encoding="utf-8"))
```

Download the file (Colab Files panel, or `from google.colab import files; files.download("requirements-lock.txt")`),
put it in the repository root on the laptop and commit it. If Colab offers "Restart session" after
the install, restart before running experiments.

Note: only the listed packages are pinned. Their own dependencies (for example `numba` for shap,
`scikit-image` for lime) are chosen by pip on each machine and may differ slightly.

## 2. Laptop setup (Windows, PowerShell)

```powershell
cd C:\Users\amare\Desktop\REsearchFakeJob
python --version                                # must print Python 3.13.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1                    # prompt now starts with (.venv)
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m nltk.downloader stopwords wordnet     # data needed by the P-CV preprocessing
python -m pytest -q
python -m ruff check src tests
```

- If `Activate.ps1` is blocked ("running scripts is disabled"), run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
- Activate the environment (`.\.venv\Scripts\Activate.ps1`) in every new terminal before working.
- In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv\Scripts\python.exe`.

Linux/macOS: `source .venv/bin/activate` instead of `Activate.ps1`.

Put the Kaggle file `fake_job_postings.csv` in `data/`.

## 3. Tests and lint

```bash
pytest -q
ruff check src tests scripts app
```

## 4. Data profile

```bash
python -m cvx.data.profile --config configs/base.yaml
```

Writes `reports/data_profile.json` (every statistic in PRD Section 6.3), computed from the CSV.

## 5. Experiments in Colab (E0–E8)

Open `notebooks/colab_runner.ipynb` in Colab (Runtime → CPU) and run the cells top to bottom.
Cell 1 clones the private repo using the `GITHUB_TOKEN` saved in Colab Secrets (never printed),
cell 2 installs the lock file, cell 3 checks the dataset hash.

| Cell | Experiment | Config | Needs |
| --- | --- | --- | --- |
| 4 | E0 reproduce Career Verify | `configs/e0_reproduce.yaml` | dataset |
| 5 | E1 reported vs realistic (balancing before the split) | `configs/e1_leakage.yaml` | dataset |
| 6 | E2 corrected split (P-CV, P-CVX) | `configs/e2_protocol.yaml` | dataset |
| 7 | E3 five models | `configs/e3_models.yaml` | dataset |
| 8 | E4 ablations, saves the app models | `configs/e4_ablations.yaml` | dataset |
| 10 | export E4 models to `app/model_store` (and a copy on Drive) | — | complete E4 run |
| 11 | E5 explanations | `configs/e5_explain.yaml` | exported models |
| 12 | E6 faithfulness (main contribution) | `configs/e6_faithfulness.yaml` | exported models |
| 13 | E7 shortcut audit + `errors.csv` | `configs/e7_audit.yaml` | exported models |
| 14 | E8 rule hit rates | `configs/e8_rules.yaml` | dataset |

Each cell runs, for example:

```bash
python -m cvx.run --config configs/e0_reproduce.yaml     --data /content/drive/MyDrive/FakeJobProject/data/raw/fake_job_postings.csv     --runs-dir /content/drive/MyDrive/FakeJobProject/cvx_runs
```

Each run writes `<runs-dir>/<experiment_id>/<UTC timestamp>/` with `config.yaml`, `env.json`,
`metrics.json`, `run.log`, `figures/` and the experiment's own files (`cv_results.csv` and
`predictions.csv` for E0–E4, `models/` for E4, `explanations.jsonl` for E5, `per_posting.csv`
for E6, `errors.csv` for E7). A run is finished only when `env.json` says `"status": "complete"`
(cells 9 and 15 list them). Copy the complete run folders to `runs/` on the laptop.

## 6. Reports (laptop)

Fill in `docs/career_verify_reported.yaml` from the paper (replace every `FILL`), then:

```bash
python scripts/make_tables.py --runs runs --out reports/tables
python scripts/make_figures.py --runs runs --out reports/figures
```

Both scripts use the newest complete run of each experiment. Every table names the run
folder and git commit it came from. After E7, fill in the `cause` column of
`runs/e7_audit/<timestamp>/errors.csv` by hand.

## 7. Export the models and run the app (laptop)

Download the Drive folder `FakeJobProject/model_store` (written by cell 10) into
`app/model_store/`. Its smoke scores were computed in Colab, so the app's startup check also
proves the laptop gives the same scores. (Or export on the laptop from the copied E4 run:)

```bash
python scripts/export_model.py --run runs/e4_ablations/<timestamp>     --models lr_text,lr_full,xgb_text,xgb_full --out app/model_store --data data/fake_job_postings.csv
streamlit run app/main.py
```

The app opens at http://localhost:8501. It refuses to start if a model file's SHA-256, a
library version or a smoke score does not match `app/model_store/model_card.json`.
Pages: **Analyze** (full posting or pasted message → risk level, score, reasons, highlighted
words, top-10 SHAP chart, safety rules, optional LIME) and **About**.
Nothing you type is stored; the app runs offline and only on `localhost`.

## 8. Reproducibility check

From a fresh clone (following sections 2–5), re-run E0 and compare it with the stored run:

```bash
python -m cvx.run --config configs/e0_reproduce.yaml --runs-dir runs_check
python scripts/compare_runs.py runs/e0_reproduce/<stored timestamp> runs_check/e0_reproduce/<new timestamp>
```

`Runs match.` means every number in `metrics.json` is identical (within 1e-9).

## 9. Project map

| Folder | What it holds |
| --- | --- |
| `src/cvx/` | all ML code: data, features, models, evaluate, explain, xai_eval, audit, rules, artifacts, report, `service.py` (the app's functions) |
| `configs/` | one YAML per experiment (`base.yaml`, `e0`–`e8`) |
| `notebooks/colab_runner.ipynb` | thin Colab runner (no ML code) |
| `scripts/` | `make_tables.py`, `make_figures.py`, `export_model.py`, `compare_runs.py` |
| `app/` | Streamlit pages (`main.py` = Analyze, `pages/1_About.py`) |
| `runs/`, `data/`, `app/model_store/` | outputs, raw CSV, exported models (not in git) |
| `reports/` | generated tables and figures (in git) |
| `docs/` | PRD, paper values, literature notes |
