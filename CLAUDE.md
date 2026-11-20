# CLAUDE.md — CareerVerify-XAI (CVX)

This file gives coding agents the rules for this repository. Copy it to `AGENTS.md` unchanged for Google Antigravity and Gemini.
The full specification is in `docs/PRD.md`. If this file and the PRD disagree, stop and ask the owner.

## 1. What this project is

- A fake job posting detector built on the EMSCAD dataset (Kaggle "Real / Fake Job Posting Prediction").
- Base paper: "Career Verify – A Fake Job Posting Detection" (IEEE AIDE 2026), which uses TF-IDF + Logistic Regression + Flask.
- This project: (1) reproduces Career Verify, (2) re-evaluates it without leakage, (3) adds metadata features and stronger models, (4) explains predictions with SHAP and LIME, (5) **tests whether those explanations are faithful** (main contribution), and (6) ships a Streamlit app (changed from Flask on 20 Nov 2026).
- The owner is an MTech student who is learning. Explain your work in simple words and define new terms.

## 2. Where to find things in docs/PRD.md

| Need | PRD section |
| --- | --- |
| Dataset columns, statistics, preprocessing P-CV and P-CVX, metrics | 6 |
| Pipeline phases and metadata features | 7 |
| Experiments E0–E8 (exact settings) | 8 |
| Repository layout, configs, run folders, Colab workflow, model artifacts | 10 |
| Web app, safety rules R1–R7, risk level | 11 |
| Milestones M1–M8 and acceptance criteria | 12 |

## 3. Rules you must always follow

1. **One milestone at a time.** Build only the milestone the owner names (PRD §12). When it is done, stop and report using the template in §8 below. Do not start the next milestone until the owner says so.
2. **Never invent numbers.** No metric, count, score or result may be typed by hand anywhere (code, README, docs, templates, comments). Results come only from run folders produced by `python -m cvx.run`, and tables only from `scripts/make_tables.py`. Where a number is not known yet, write `<<from runs/...>>`.
3. **Never invent data.** Do not create synthetic job postings to "make code run" on real experiments. Tiny hand-written fixtures are allowed only inside `tests/`.
4. **Never invent references.** Do not add citations or paper titles. Literature is the owner's work.
5. **All ML logic lives in `src/cvx/`.** Notebooks, scripts and the app only call functions from `cvx`. Never copy cleaning or feature code into another place.
6. **Do not edit `runs/` or `reports/` by hand.** They are outputs.
7. **Seed 42 everywhere.** The test split is scored once, after tuning. Never tune on the test split.
8. **Do not change scope.** Do not add or remove datasets, models, metrics, features or experiments without asking.
9. **Pickle safety.** Load `.pkl` files only from `app/model_store/` and only if their SHA-256 matches `model_card.json`. Never download or load a model from the internet or another repository.
10. **Versions.** Install only from `requirements-lock.txt`. Never upgrade or pin a library yourself. If something needs a new library, ask first.
11. **Missing file?** If the dataset, a run folder or the lock file is missing, stop and tell the owner exactly what is needed.
12. **No secrets in code.** Any secret comes from an environment variable (for example `CVX_SECRET`), never from code. Never log posting text.

## 4. Commands

```bash
# setup (laptop; same Python minor version as Colab)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
pip install -e .

# quality
pytest -q
ruff check src tests app scripts

# experiments (normally run in Colab via notebooks/colab_runner.ipynb)
python -m cvx.run --config configs/e0_reproduce.yaml

# reports and export
python scripts/make_tables.py --runs runs --out reports/tables
python scripts/make_figures.py --runs runs --out reports/figures
python scripts/export_model.py --run runs/e4_ablations/<timestamp> --models lr_full,lr_text,xgb_full,xgb_text --out app/model_store

# app
streamlit run app/main.py
```

## 5. Repository map

```text
configs/        one YAML per experiment (base.yaml + e0..e8)
src/cvx/        the package: config, run, data/, features/, models/, evaluate/,
                explain/, xai_eval/, audit/, rules/, artifacts/, schema.py,
                service.py (the only functions the app calls; PRD §11.6)
notebooks/      colab_runner.ipynb only (thin: clone, install, mount Drive, run configs)
scripts/        make_tables.py, make_figures.py, export_model.py
app/            Streamlit app (main.py, pages/, db.py, model_store/); no ML logic
.streamlit/     config.toml (localhost only, usage stats off, error details hidden)
runs/           experiment outputs (gitignored)
reports/        generated tables and figures (committed)
data/           raw CSV (gitignored)
tests/          pytest, mirrors src/cvx/
docs/PRD.md     full specification
```

## 6. Fixed definitions (do not reinterpret)

- **Label:** `fraudulent`; positive class = 1 = fake.
- **P-CV:** Career Verify preprocessing (lowercase, remove stop words, numbers, punctuation and special characters, NLTK WordNet lemmatisation, join title + company_profile + description + requirements + benefits). Used only in E0, E1, E2a.
- **P-CVX:** our preprocessing (strip HTML, mask URL, email and phone placeholders as `urltoken`, `emailtoken`, `phonetoken`, lowercase, keep "no" and "not", no lemmatisation, join the same five fields with labels).
- **Corrected split:** remove exact duplicates, group near-duplicates (TF-IDF cosine ≥ 0.95), then grouped stratified 70/15/15. No group may appear in two splits.
- **Primary metrics:** fake-class F1 and PR-AUC. Accuracy is reported but never used to choose a model.
- **Thresholds:** `t_f1` = best F1 on validation; `t_r90` = lowest threshold with recall ≥ 0.90 on validation.
- **Linear SHAP (LR):** φᵢ = wᵢ × (xᵢ − mean of xᵢ in training), log-odds scale. Must match `shap.LinearExplainer` within 1e-6.
- **Tree SHAP (XGBoost):** `shap.TreeExplainer`; base value + Σφ must equal the model margin within 1e-4.
- **Faithfulness:** comp@k = p(x) − p(x without top-k words); suff@k = p(x) − p(only top-k words); AOPC = mean comp@k over k ∈ {1, 3, 5, 10, 20}; p = probability of the predicted label; removing a word deletes every occurrence; metadata stays fixed; baseline = k random words, 5 draws.
- **Risk level:** model level and rules level computed separately; overall = the higher one. Never add scores together.

## 7. Coding conventions

- Python with type hints on every function and a docstring that says what it does, its inputs and outputs.
- Small, pure functions where possible; `pathlib` for paths; `logging`, not `print`, inside the package.
- Comments explain **why** in simple English.
- Every module in `src/cvx/` has a test file in `tests/`. Core maths (metrics, split, SHAP, faithfulness) is tested on tiny hand-calculated examples.
- App tests use a tiny model trained inside the test, never real artifacts.
- Keep functions under about 50 lines; split them if they grow.
- Run `pytest` and `ruff` before reporting a milestone as done.

## 8. How to report after each milestone

```text
Milestone: M<n> — <name>
Files created/changed: <list>
What each new module does (simple words): <2–3 lines each>
Acceptance criteria (PRD §12): <each box: done / not done + reason>
Tests: <command> → <result copied from the terminal>
What the owner must do next: <e.g. run configs/e0_reproduce.yaml in Colab>
Questions or assumptions: <list every setting you had to assume>
```

## 9. Things that are out of scope unless the owner asks

BERT or other transformers, OCR, batch CSV upload, user accounts, cloud deployment, scraping job sites, WHOIS lookups, any external API.
