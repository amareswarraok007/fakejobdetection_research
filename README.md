# CareerVerify-XAI (CVX)

A leakage-aware, explainable fake job posting detector built on the EMSCAD dataset.
The full specification is in [docs/PRD.md](docs/PRD.md); agent rules are in `CLAUDE.md`.

> This README is a stub for milestone M1. The full README (Colab steps, experiments,
> model export, app) is written in milestone M8.

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
ruff check src tests scripts
```

## 4. Data profile

```bash
python -m cvx.data.profile --config configs/base.yaml
```

Writes `reports/data_profile.json` (every statistic in PRD Section 6.3), computed from the CSV.

## 5. Experiments (Colab) and reports (laptop)

Experiments E0–E4 run in Colab with `notebooks/colab_runner.ipynb` (one cell per experiment).
Each cell runs, for example:

```bash
python -m cvx.run --config configs/e0_reproduce.yaml \
    --data /content/drive/MyDrive/FakeJobProject/data/raw/fake_job_postings.csv \
    --runs-dir /content/drive/MyDrive/FakeJobProject/cvx_runs
```

Each run writes `<runs-dir>/<experiment_id>/<UTC timestamp>/` with `config.yaml`, `env.json`,
`metrics.json`, `cv_results.csv`, `predictions.csv`, `figures/`, `run.log` (and `models/` for E4).
A run is finished only when `env.json` says `"status": "complete"`.

Copy the complete run folders to `runs/` on the laptop, fill in `docs/career_verify_reported.yaml`
from the paper, then:

```bash
python scripts/make_tables.py --runs runs --out reports/tables
python scripts/make_figures.py --runs runs --out reports/figures
```

Both scripts use the newest complete run of each experiment.
