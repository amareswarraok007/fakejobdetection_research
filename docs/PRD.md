# CareerVerify-XAI — Project Proposal and PRD

Nov 20, 2026 · @AMARESWAR RAO K

## 0. How to use this document with a coding agent

This document is both the project proposal (sections 1–9: title, abstract, literature plan, dataset, methodology, work plan) and the build specification for a coding agent (sections 10–12). It replaces the earlier JobShield PRD. Start a new, empty repository named `careerverify-xai`.

### 0.1 Files to put in the new repository

| File | What it is | Read by |
| --- | --- | --- |
| `docs/PRD.md` | This document, exported as Markdown | You and the agent |
| `CLAUDE.md` | Short rules file for the agent (sent separately) | Claude Code |
| `AGENTS.md` | An exact copy of `CLAUDE.md` | Google Antigravity and Gemini. Antigravity's docs say it reads `AGENTS.md` or `GEMINI.md` from the workspace folder ([Antigravity rules docs](https://antigravity.google/docs/rules.md)) |

Keep `CLAUDE.md` short. One guide reports a limit of about 12,000 characters per Antigravity rule file, so the long detail lives here in `docs/PRD.md` and the rules file points to it.

### 0.2 Order of work

1. Create the empty repo, add the three files above, and push to GitHub.
2. Give the agent the first prompt below. It builds milestone M1 only, then stops.
3. You review, run the tests, and type "continue with M2". Repeat for each milestone in Section 12.
4. Training runs in Google Colab Pro. The agent writes the code; you run it and copy the results back (Section 10.5).

```text
Read CLAUDE.md and docs/PRD.md completely.
Build milestone M1 only (Section 12 of the PRD). Do not start M2.
Before writing code, list every file you will create and ask me any questions.
Never invent numbers; every result must come from a file the code produced.
```

Learning tip: after each milestone, ask the agent to explain each new file in simple words, and re-type the key functions yourself (cleaning, splitting, SHAP wrapper, faithfulness metric). You must be able to explain them in your reviews and viva.

## 1. Project title

**Recommended title:** CareerVerify-XAI: A Leakage-Aware, Explainable Fake Job Posting Detector with Faithfulness-Evaluated SHAP and LIME Explanations

It names the base paper, the two main fixes (honest evaluation and explanations) and the research contribution (testing whether the explanations are true).

| Option | Title | Best when |
| --- | --- | --- |
| A (recommended) | CareerVerify-XAI: A Leakage-Aware, Explainable Fake Job Posting Detector with Faithfulness-Evaluated SHAP and LIME Explanations | Project report and review presentations |
| B | Beyond 97% Accuracy: Reproducing and Correcting a TF-IDF Logistic Regression Fake Job Detector | The evaluation-correction finding turns out to be the strongest result |
| C | Are the Explanations True? Faithfulness Testing of SHAP and LIME for Online Recruitment Fraud Detection | Writing a conference paper focused on the XAI evaluation |

Short name used in code and this document: **CVX**.

## 2. Abstract

This is the proposal-stage abstract. It is written in the future tense and states no results. After the experiments, replace the last two sentences with the measured numbers from `reports/`.

> Fake job and internship postings cause financial loss and identity theft, especially among students and fresh graduates. Career Verify (IEEE AIDE 2026) detects such postings with TF-IDF features and Logistic Regression and reports 97% accuracy on the Employment Scam Aegean Dataset (EMSCAD; 17,880 postings, 866 fraudulent). However, its reported confusion matrix is inconsistent with the dataset's class balance, it uses only text fields, it reports accuracy as its main measure, and it offers no per-posting explanation. This project will first reproduce Career Verify, then re-evaluate it under a leakage-aware protocol that removes duplicate postings before stratified splitting and reports fraud-class F1, PR-AUC and confidence intervals. It will extend the detector with metadata features and an XGBoost model, and explain every prediction with SHAP and LIME, translated into plain-English reasons and complemented by rule-based safety checks for payment and personal-data requests. Unlike prior work that only displays explanations, the project will measure whether they are faithful, using comprehensiveness, sufficiency and AOPC against a random-word baseline, together with SHAP–LIME agreement and LIME stability. SHAP will also be used to audit the model for shortcut features. The complete system will be delivered as a reproducible pipeline and a Streamlit web application. We expect to show how much evaluation choices change reported performance, and which explanation method better reflects the model's real decisions.

**Keywords:** fake job detection, online recruitment fraud, explainable AI, SHAP, LIME, explanation faithfulness, data leakage, TF-IDF, XGBoost.

## 3. Base paper and problem statement

The base paper is Career Verify, a simple and recent IEEE paper whose method can be reproduced in a day and whose limitations are clear and checkable. Each limitation below becomes one contribution in Section 4.

### 3.1 Base paper

Bhagyashree M and Pallavi Shetty, "Career Verify – A Fake Job Posting Detection", 2026 International Conference on Artificial Intelligence and Data Engineering (AIDE), IEEE, pp. 822–826, DOI 10.1109/AIDE69088.2026.11544522.

| Part | What Career Verify does |
| --- | --- |
| Dataset | EMSCAD from Kaggle, about 17,880 postings |
| Split | 80% train, 10% validation, 10% test |
| Preprocessing | Lowercase; remove stop words, special characters, numbers and punctuation; tokenise and lemmatise; impute or drop missing values; join title, company profile, description, requirements and benefits into one text field |
| Features | TF-IDF on the joined text only (settings not reported) |
| Model | Logistic Regression, L2 penalty, class weighting, C and max iterations tuned |
| Other models | Naive Bayes, SVM and Random Forest "tested", with no numbers given |
| Metrics | Accuracy, precision, recall, F1, confusion matrix |
| Reported result | Accuracy 97%; fake class precision 96%, recall 95%, F1 95% |
| Deployment | Flask web app; Windows 10, 8 GB RAM, no GPU |

### 3.2 Verified limitations

| ID | Limitation | Evidence from the paper |
| --- | --- | --- |
| L1 | Reported test results do not fit the dataset | The confusion matrix (TN 1,892, TP 450, FP 18, FN 22) sums to 2,382 test postings, but 10% of 17,880 is about 1,788. It contains 472 fake postings (19.8%), while the dataset is about 4.8% fake, so a 10% test set would hold about 87. Accuracy computed from the matrix is 98.3%, not the stated 97%. A likely cause is oversampling before splitting, which leaks information; this is tested in experiment E1. |
| L2 | Accuracy-led evaluation | Accuracy is the headline number on a dataset where predicting "genuine" for everything already scores about 95%. No PR-AUC, cross-validation or confidence intervals. |
| L3 | No duplicate check | Duplicate or near-duplicate postings can sit in both training and test sets. Not discussed. |
| L4 | Claims interpretability, gives no per-posting explanation | Only a global feature-importance graph is mentioned. A job seeker is not told why a posting was flagged. |
| L5 | Text only | Metadata (company logo, screening questions, salary, employment type) is ignored on purpose. |
| L6 | Baselines without numbers | NB, SVM and RF are described in words only. |
| L7 | No error or shortcut analysis | No study of which postings are misclassified or whether the model relies on accidental cues. |
| L8 | Reference problems | In-text citation numbers do not match the reference list (for example "Kumar et al. \[1\]" vs reference \[1\] by Jindal and Kumar). Do not reuse its reference list. |

### 3.3 Problem statement

Lightweight fake-job detectors such as Career Verify report very high accuracy, but their evaluation may be inflated by leakage and duplicate postings, and they give job seekers no trustworthy reason for a decision. Where explanation tools such as SHAP and LIME are used in this area, their output is shown but rarely tested. The problem this project addresses is to build a fake job detector whose performance is measured honestly and whose explanations are shown, measurably, to reflect what actually drives each prediction.

## 4. Objectives and contributions

The project's main contribution is C4, testing whether explanations are faithful. C1 and C2 make the comparison with Career Verify honest, and C3, C5 and C6 make the system useful to a job seeker.

### 4.1 Objectives

1. **O1** Reproduce Career Verify exactly as described, as the baseline.
2. **O2** Re-evaluate it with a leakage-aware protocol and imbalance-appropriate metrics.
3. **O3** Improve detection with metadata features and a stronger model, with full baseline comparisons.
4. **O4** Explain every prediction with SHAP and LIME in a form a job seeker can read.
5. **O5** Measure the faithfulness, agreement and stability of those explanations, and use them to audit for shortcuts.
6. **O6** Deliver a reproducible pipeline and a working web application.

### 4.2 Contributions mapped to Career Verify's limitations

| ID | Contribution | Fixes | Evidence produced |
| --- | --- | --- | --- |
| C1 | Reproduction plus a leakage test: Career Verify's pipeline as described, then the same pipeline with oversampling before splitting, to test whether that explains its numbers | L1 | E0 vs E1 table |
| C2 | Leakage-aware evaluation: duplicate removal, grouped stratified split, fraud-class F1, PR-AUC, 5-fold CV, bootstrap confidence intervals, McNemar's test | L1, L2, L3 | E2 table and curves |
| C3 | Fair model comparison (NB, linear SVM, RF, LR, XGBoost) with text-only vs text + metadata | L5, L6 | E3 and E4 tables |
| C4 | **Faithfulness evaluation of SHAP and LIME**: comprehensiveness, sufficiency and AOPC against random-word removal, plus agreement, stability and runtime | L4 | E6 tables, curves and Wilcoxon tests |
| C5 | Per-posting explanations with plain-English reasons, plus rule-based safety checks for scam signals the dataset lacks | L4 | Web app and E8 rule hit rates |
| C6 | SHAP-based shortcut audit and error analysis | L7 | E7 audit and masking results |

### 4.3 Research questions

- **RQ1** Does Career Verify's reported performance hold when the pipeline is reproduced, and does oversampling before splitting explain the gap?
- **RQ2** How much do duplicate removal and grouped splitting change measured performance?
- **RQ3** Do metadata features and XGBoost improve on TF-IDF + Logistic Regression under the corrected protocol?
- **RQ4** Are SHAP explanations more faithful than LIME explanations, and how much do the two agree?
- **RQ5** Does the model rely on shortcut features, and what happens to performance when they are removed?

## 5. Literature review plan

The review will cover 20–25 peer-reviewed papers in six themes and end with a gap statement that leads directly to contributions C1–C6. The seed papers below were found on IEEE Xplore and publisher pages at abstract level; read the full text before citing any number. This section is written by the owner, not the coding agent.

### 5.1 Search strategy

| Item | Plan |
| --- | --- |
| Databases | IEEE Xplore (through SRM access), Scopus, SpringerLink, ScienceDirect, ACM Digital Library, Google Scholar for forward and backward citations |
| Years | 2017 (EMSCAD release) to 2026 |
| Search strings | ("fake job" OR "fraudulent job" OR "employment scam" OR "online recruitment fraud") AND ("machine learning" OR "deep learning" OR "NLP"); then each of those AND ("explainable" OR "SHAP" OR "LIME"); separately ("faithfulness" OR "comprehensiveness") AND ("explanation" AND "text classification") |
| Include | Peer-reviewed; describes method, data and metrics; job-fraud task, or XAI evaluation method used in this project |
| Exclude | No method detail; duplicate versions of the same work; journals not indexed in Scopus or Web of Science, unless they are the base paper |
| Record keeping | Count records found, screened, read in full and included, and draw a PRISMA-style flow diagram |

### 5.2 Themes and seed papers

| Theme | What to learn | Seed papers |
| --- | --- | --- |
| T1 Dataset and classical ML | How EMSCAD was built; what TF-IDF + LR, SVM, ensembles achieve | [Vidros et al. 2017, Future Internet (EMSCAD)](https://mdpi.com/1999-5903/9/1/6); Career Verify (base); [ORFDetector, IC3 2019](https://ieeexplore.ieee.org/document/8844879/); [ORF Detection using ANN](https://ieeexplore.ieee.org/abstract/document/9636978/); [Web App with Ensemble Classifiers](https://ieeexplore.ieee.org/document/10170567/); [Salloum et al., LR with text + meta-features](https://salford-repository.worktribe.com/output/3541625/analysis-of-fraudulent-job-postings-using-machine-learning) |
| T2 Deep learning and transformers | What BERT-type models add and cost | [Akram et al., IEEE Access 2024, DOI 10.1109/ACCESS.2024.3435670](https://doaj.org/article/aa7a0146d8724e668e9340a26bdcc1bd); [Fraud-BERT (Springer)](https://link.springer.com/article/10.1007/s10791-025-09502-8); [DistilBERT + XGBoost](https://ieeexplore.ieee.org/document/11085449/); [GRU on EMSCAD](https://ieeexplore.ieee.org/document/9929928/) |
| T3 Metadata and behavioural features | Which non-text fields help | [Behavioral Features Extraction](https://ieeexplore.ieee.org/document/8969842/); [Fraudulent job types, AI & Society 2022](https://link.springer.com/doi/10.1007/s00146-022-01469-0) |
| T4 XAI for job fraud | How SHAP and LIME are currently used here | [XAI-Recruit (Bi-LSTM + LIME)](https://ieeexplore.ieee.org/document/11507788/); [ORF with Transformers and SHAP](https://ieeexplore.ieee.org/abstract/document/11436921/) |
| T5 XAI methods and their evaluation | Theory of SHAP and LIME; how to test faithfulness | Lundberg and Lee 2017 (SHAP); Ribeiro et al. 2016 (LIME); DeYoung et al. 2020 (ERASER, comprehensiveness and sufficiency) |
| T6 Evaluation pitfalls | Why leakage and imbalance inflate results | Kapoor and Narayanan 2023, "Leakage and the reproducibility crisis in machine-learning-based science" (Patterns); Chawla et al. 2002 (SMOTE) |

T5 and T6 entries are standard papers listed from memory; confirm their details before citing.

### 5.3 Extraction table (one row per paper)

| Author, year, venue | Dataset | Split; duplicates handled? | Features | Model | Metrics reported | XAI method | XAI evaluated? | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (fill) |  |  |  |  |  |  |  |  |

Fill this as you read. The "Split; duplicates handled?" and "XAI evaluated?" columns are the ones that will show the gap.

### 5.4 Chapter outline

1. Online recruitment fraud and the EMSCAD dataset
2. Classical machine learning approaches (including Career Verify)
3. Deep learning and transformer approaches
4. Metadata and behavioural features
5. Explainable AI in fraud and job-scam detection
6. Evaluating explanations: faithfulness metrics
7. Evaluation pitfalls: imbalance, oversampling and leakage
8. Comparison table and research gap

### 5.5 Expected gap (confirm after the review)

Most studies on EMSCAD report accuracy-led results without describing duplicate handling, and studies that use SHAP or LIME display explanations without testing their faithfulness. No reviewed study is expected to combine a reproduced baseline, a leakage-aware evaluation and a faithfulness comparison of SHAP and LIME. Change this statement if the review finds such a study.

## 6. Dataset identification

The project uses one public benchmark, EMSCAD, the same data as Career Verify, so the comparison is like for like. A small self-collected Indian test set is optional and used only to test generalisation, never for training.

### 6.1 Sources

| Role | Source | Access |
| --- | --- | --- |
| Primary data | [Real / Fake Job Posting Prediction, Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), file `fake_job_postings.csv` | Free Kaggle login; browser download or Kaggle API |
| Original description | [Vidros, Kolias, Kambourakis, Akoglu, "Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset", Future Internet 9(1), 2017](https://mdpi.com/1999-5903/9/1/6) | Open access; cite this as the dataset paper |
| Optional external test set | 30–50 job messages collected by the team from WhatsApp, Telegram, email and portals, labelled by two people | Personal data removed before use; never used for training |

Check the licence shown on the Kaggle page and cite both the Kaggle page and Vidros et al.

### 6.2 Description

Each row is one job posting with 18 columns: free text, metadata and the label.

| Column | Type | Used in Career Verify reproduction | Used in CVX |
| --- | --- | --- | --- |
| `job_id` | ID | No | No (row key only) |
| `title`, `company_profile`, `description`, `requirements`, `benefits` | Text | Yes, joined | Yes, joined |
| `location`, `department`, `industry` | Text / category | No | Shortcut audit only (E7) |
| `salary_range` | Text range | No | As "salary given: yes/no" |
| `telecommuting`, `has_company_logo`, `has_questions` | 0/1 | No | Yes |
| `employment_type`, `required_experience`, `required_education`, `function` | Category | No | Yes, one-hot |
| `fraudulent` | 0/1 label | Label | Label |

### 6.3 Statistics

Reported by the dataset authors and many users: 17,880 postings, of which 866 (about 4.8%) are fraudulent, posted between 2012 and 2014; the CSV is about 50 MB. The pipeline must recompute every number below itself and save it to `reports/data_profile.json`. Nothing is hard-coded.

- Row count, class counts and fraud rate, before and after duplicate removal
- Exact and near-duplicate counts (Section 6.4)
- Missing-value rate per column, for fake and genuine postings separately
- Fraud rate for each metadata value (logo yes/no, questions yes/no, remote yes/no, salary given yes/no)
- Text length distribution, fake vs genuine
- Top 10 locations and industries among fake postings (input to the shortcut audit)
- Vocabulary size after cleaning

### 6.4 Preprocessing plan

There are two preprocessing pipelines: one copies Career Verify, the other is ours.

| Step | P-CV (reproduction of Career Verify) | P-CVX (this project) |
| --- | --- | --- |
| Missing values | Empty string for text | Empty string for text, `Unknown` for categories |
| Cleaning | Lowercase; remove stop words, numbers, punctuation, special characters | Lowercase; strip HTML; replace the dataset's URL, email and phone placeholders with `urltoken`, `emailtoken`, `phonetoken`; keep words, including "no" and "not" |
| Lemmatisation | NLTK WordNet lemmatiser | None, so explanation words stay readable |
| Text field | Title, company profile, description, requirements, benefits joined | Same five, joined with field labels: `title: … \| company: …` |
| Duplicates | Not handled (as in the paper) | Exact duplicates removed; near-duplicates (TF-IDF cosine ≥ 0.95) grouped |
| Split | Stratified random 80/10/10, seed 42 | Grouped stratified 70/15/15 (`StratifiedGroupKFold`), seed 42; each duplicate group stays in one split |
| Imbalance | Class weights | Class weights; SMOTE only in an ablation, inside training folds |
| Features | TF-IDF (scikit-learn defaults, because the paper gives no settings) | TF-IDF 1–2 grams, `min_df=3`, `max_df=0.9`, 20,000 features, `sublinear_tf=True`, plus metadata features (Section 7) |

All cleaning code lives in one module, used by both training and the web app.

### 6.5 Evaluation metrics

Detection metrics (positive class = fake):

| Metric | Formula or meaning | Role |
| --- | --- | --- |
| Precision | TP / (TP + FP) | How many flagged postings are really fake |
| Recall | TP / (TP + FN) | How many fake postings are caught |
| F1 (fake class) | 2 × precision × recall / (precision + recall) | **Primary metric** |
| PR-AUC | Area under the precision–recall curve | **Primary threshold-free metric** |
| ROC-AUC, macro-F1, accuracy | Standard | Reported for comparison with papers only |
| Confusion matrix | TP, FP, TN, FN counts | Required for every model, to avoid Career Verify's inconsistency |
| 95% bootstrap CI | 1,000 resamples of the test set | Shows uncertainty |
| McNemar's test | Compares two models' errors on the same test set | Shows whether a difference is real |

Explanation metrics (defined fully in Section 8, E6): comprehensiveness@k, sufficiency@k, AOPC, SHAP–LIME top-k Jaccard and Spearman correlation, LIME stability across seeds, seconds per explanation.

System metrics: prediction latency, explanation latency and a successful startup self-check (Section 10).

## 7. Proposed methodology

The method runs two tracks on the same data: a baseline track that reproduces Career Verify exactly, and a CVX track with the corrected protocol and stronger features. Both feed one evaluation. The explainability track then explains the CVX models and tests those explanations, and the final models are deployed in a Streamlit app.

&#91;embedded content: proposed methodology · baseline track, CVX track, explainability track, deployment\]

### 7.1 Phases

1. **Phase 0 — Data profiling.** Load EMSCAD, compute every statistic in Section 6.3, find duplicates.
2. **Phase 1 — Baseline track (E0, E1).** P-CV preprocessing, random 80/10/10 split, TF-IDF + Logistic Regression, as Career Verify describes; then the same with oversampling before the split, to test the leakage explanation.
3. **Phase 2 — CVX track (E2–E4).** P-CVX preprocessing, duplicate removal and grouped stratified split, TF-IDF plus metadata features, five models tuned by grouped cross-validation, thresholds chosen on validation.
4. **Phase 3 — Evaluation and comparison.** Fraud-class F1, PR-AUC, confidence intervals and McNemar's test; one table showing Career Verify reported vs reproduced vs corrected vs CVX best.
5. **Phase 4 — Explainability (E5–E7).** Exact linear SHAP for LR, TreeExplainer for XGBoost and LIME for both; plain-English reasons; faithfulness, agreement and stability tests; shortcut audit and error analysis.
6. **Phase 5 — Deployment.** Export the chosen models with a model card, load them safely in the Streamlit app, add safety rules and risk levels (Section 11).
7. **Phase 6 — Reporting.** Generate every table and figure from run folders; write the report and paper.

### 7.2 Metadata features (CVX track)

| Feature | How it is computed |
| --- | --- |
| `has_company_logo`, `has_questions`, `telecommuting` | Dataset 0/1 columns |
| `salary_given` | 1 if `salary_range` is not empty |
| `company_profile_empty`, `requirements_empty`, `benefits_empty` | 1 if that field is empty after cleaning |
| `log_text_length` | log(1 + characters in the joined text) |
| `n_url`, `n_email`, `n_phone` | Counts of `urltoken`, `emailtoken`, `phonetoken` |
| `money_mentions` | Count of money patterns (`$`, `₹`, `rs.`, `inr`, "per week", "per day") |
| `exclamation_count`, `uppercase_ratio` | Computed on raw text before lowercasing |
| `employment_type`, `required_experience`, `required_education`, `function` | One-hot; `function` keeps its top 20 values plus `Other`; missing = `Unknown` |

Numeric features are scaled with `StandardScaler(with_mean=False)` so they combine with sparse TF-IDF inside one scikit-learn Pipeline. Both a text-only and a full (text + metadata) version of each final model are exported, because many real messages (for example on WhatsApp) have no metadata.

### 7.3 Text version of the pipeline (for the coding agent)

`data.load` → `data.profile` → `data.clean` (`p_cv` or `p_cvx`) → `data.dedup` → `data.split` → `features.build` → `models.train` (tuning, thresholds) → `evaluate.metrics` and `evaluate.stats` → `explain.*` → `xai_eval.*` and `audit.*` → `artifacts.save` → app (`schema.build_input` → Pipeline → `explain` → `rules.safety` → risk level).

## 8. Experiments

Nine experiments, E0–E8, each answer one research question and write their results to `runs/<experiment_id>/`. Every run uses seed 42, and the test split is scored only after all tuning is finished.

### 8.1 Summary

| ID | Name | Answers | Main output |
| --- | --- | --- | --- |
| E0 | Reproduce Career Verify | RQ1 | Metrics and confusion matrix next to the paper's reported numbers |
| E1 | Leakage test: oversampling before splitting (E1a, E1b) vs inside training only (E1c) | RQ1 | Test-set fake share; reported (E1a/E1b) vs realistic (E1c) PR-AUC, ROC-AUC and fake-class recall |
| E2 | Leakage-aware protocol | RQ2 | Metrics with duplicates removed and grouped split |
| E3 | Model comparison | RQ3 | NB, linear SVM, RF, LR, XGBoost table |
| E4 | Feature and imbalance ablations | RQ3 | Text-only vs text + metadata; class weights vs SMOTE |
| E5 | Explanation generation | RQ4 | Per-posting SHAP and LIME; global SHAP ranking |
| E6 | Faithfulness evaluation | RQ4 | Comprehensiveness, sufficiency, AOPC, agreement, stability, runtime |
| E7 | Shortcut audit and error analysis | RQ5 | Top-30 SHAP audit, masking re-train, error table |
| E8 | Safety rules and external test | C5 | Rule hit rates; optional Indian test set results |

### 8.2 Details

**E0 — Reproduce Career Verify.** Preprocessing P-CV, TF-IDF with scikit-learn defaults, Logistic Regression (L2, `class_weight='balanced'`, C tuned on validation from {0.1, 1, 10}; speed mode §9.1, originally {0.01, 0.1, 1, 10}), stratified random 80/10/10 split. Report every metric and the confusion matrix beside the paper's reported values, and list every setting the paper did not specify and the value chosen.

**E1 — Leakage test.** Same as E0, but balance the whole dataset before splitting: E1a with random oversampling, E1b with SMOTE (TF-IDF is fitted on all postings first, because SMOTE needs numbers). E1c is the correct control: the E0 split, with random oversampling inside the training split only, so validation and test keep their natural prevalence. Report the test-set fake share and all metrics, and label E1a/E1b vs E1c as **reported vs realistic performance**, not as pure leakage, because the test prevalence differs (about half fake in E1a/E1b, the natural rate in E1c). For this comparison report PR-AUC, ROC-AUC and fake-class recall, and the differences E1a − E1c and E1b − E1c. If metrics jump towards the paper's numbers, that supports the leakage explanation for L1. Report it as a likely cause, not as proof.

**E2 — Leakage-aware protocol.** E2a: P-CV preprocessing with the corrected split (duplicates removed, grouped stratified 70/15/15). E2b: P-CVX preprocessing with the same split. Compare E0 → E2a (effect of the split) and E2a → E2b (effect of preprocessing). Include 5-fold grouped cross-validation on the training split and bootstrap confidence intervals on the test split.

**E3 — Model comparison** (P-CVX, text only, corrected split). Multinomial NB (alpha in {0.1, 0.5, 1}); linear SVM (C in {0.1, 1, 10}, probabilities from `CalibratedClassifierCV` with sigmoid calibration; in grouped experiments its internal folds are group-aware); Random Forest (300 trees, `class_weight='balanced_subsample'`); Logistic Regression (C in {0.1, 1, 10}; speed mode §9.1, originally {0.1, 0.3, 1, 3, 10}); XGBoost (`tree_method='hist'`, learning rate 0.05, depth in {4, 6, 8}, `scale_pos_weight` = genuine/fake ratio in training, early stopping on validation, up to 600 trees). Tune by grouped 5-fold CV PR-AUC. Thresholds are chosen on validation: `t_f1` (best F1) and `t_r90` (highest threshold whose validation recall is still ≥ 0.90). Report test results at 0.5, `t_f1` and `t_r90`, plus McNemar's test between LR and XGBoost.

**E4 — Ablations.** Text-only vs text + metadata for LR and XGBoost. Class weights vs SMOTE inside training folds (imbalanced-learn Pipeline) for LR.

**E5 — Explanations.** For LR, the exact SHAP value of feature i is wᵢ × (xᵢ − training mean of xᵢ), on the log-odds scale; check it against `shap.LinearExplainer` on 20 postings. For XGBoost, use `shap.TreeExplainer` and check that the base value plus the SHAP values equals the model output to within 1e-4. For LIME, use `LimeTextExplainer` with the metadata held fixed. Save global mean-absolute SHAP rankings for both models.

**E6 — Faithfulness** (main contribution). Sample: 100 test postings (up to 50 true positives and 50 true negatives; speed mode §9.1), the same for every method. SHAP's word and two-word scores are converted to per-word scores (a two-word phrase is split equally between its words). Removing a word deletes every occurrence of it; metadata stays fixed.

```latex
\text{comp}@k = p(x) - p(x \setminus \text{top-}k) \qquad \text{(higher = more faithful)}
```

```latex
\text{suff}@k = p(x) - p(\text{top-}k \text{ only}) \qquad \text{(lower = more faithful)}
```

```latex
\text{AOPC} = \frac{1}{|K|}\sum_{k \in K} \text{comp}@k, \quad K = \{1, 3, 5, 10, 20\}
```

Here p(x) is the probability of the predicted label. Baseline: k random words from the same posting, averaged over 5 draws. Agreement: SHAP vs LIME top-5 and top-10 Jaccard, Spearman correlation over the union of their words, and the share of shared words with the same direction. Stability: LIME with 3 seeds (speed mode §9.1), mean pairwise top-5 Jaccard (SHAP is deterministic here, so its stability is 1 by construction). Runtime: mean and 95th-percentile seconds per explanation. Tests: paired Wilcoxon signed-rank on per-posting comp@5 for SHAP vs random, LIME vs random and SHAP vs LIME, separately for LR and XGBoost. Limitation to state: deleting words creates unnatural text; the random baseline partly controls for this.

**E7 — Shortcut audit and errors.** Review the global top-30 SHAP features for place names, company names and other accidental cues. E7a: add `location` and `industry` and retrain; if F1 jumps and these columns dominate SHAP, report a location shortcut. E7b: mask flagged tokens with `maskedtoken`, retrain, and report the change. Export 20 false negatives and 20 false positives with their top-5 SHAP features to `errors.csv`, and group them by cause by hand.

**E8 — Rules and external test.** Report how often rules R1, R2 and R5–R7 (Section 11.3) fire on fake vs genuine EMSCAD postings. R3 and R4 need email and website fields the dataset lacks, so state that they are not validated. Optional: run the final model and the rules on the self-collected Indian test set and report the change in performance (domain shift).

## 9. Work plan

The plan takes 10 weeks of part-time work, with three review points with the guide. Week numbers are relative; set real dates once the semester calendar is fixed. Milestones M1–M8 are defined in Section 12.

| Week | Phase | Main tasks | Deliverable | Milestone |
| --- | --- | --- | --- | --- |
| 1 | Setup and literature | Search and screen papers; create repo with `CLAUDE.md`, `AGENTS.md` and `docs/PRD.md`; package skeleton and tests | Proposal (this doc); M1 code | M1 |
| 2 | Data and reproduction | Data profile, both preprocessing pipelines, splits; run E0 and E1 | `data_profile.json`; E0 and E1 results | M2 |
| 3 | Corrected evaluation | E2 and E3 in Colab; continue the literature extraction table | E2 and E3 tables | M3 |
| 4 | Ablations | E4; choose the final model and thresholds | **Review 1: baseline and corrected results** | M3 |
| 5 | Explanation code | SHAP, LIME, reasons and faithfulness modules with tests (E5) | Explanation module; unit tests pass | M4 |
| 6 | XAI evaluation | Run E6 in Colab; run E7 audit and error analysis | E6 and E7 results | M5 |
| 7 | Web app core | Model-loading checks, Analyze page with its result (SHAP, rules, risk level) | Working app | M6 |
| 8 | Web app complete | LIME on demand, About page, E8 rule hit rates | **Review 2: demo and XAI results** | M7 |
| 9 | Reporting | Auto-generated result tables and figures; write report chapters | Draft report | M8 |
| 10 | Buffer and finish | Fix review comments; paper draft; slides | **Review 3: final report, paper draft, demo** | — |

The literature review runs alongside weeks 1–3 and must be finished before Review 1. If the schedule slips, cut in this order: the external Indian test set, then the SMOTE ablation, then the LIME stability runs. E0–E3, E6 and the app core are never cut.

### 9.1 Speed mode (approved by the owner on 20 Nov 2026)

These decisions replace the matching parts of Sections 8, 11 and 12.

- **Batches.** Milestones are built in three batches, and the agent stops after each batch for the owner's Colab run: M2+M3, then M4+M5, then M6+M7+M8.
- **Small grids.** At most 3 values per hyperparameter. Logistic Regression uses C in {0.1, 1, 10} in every experiment (E0–E4); the other grids in E3 already have 3 values or fewer.
- **Colab budget.** `colab_runner.ipynb` has one cell per experiment, and all experiments of a batch must run in one Colab CPU session in under 1 hour.
- **LIME.** 100 test postings and 3 seeds (E6 sample and stability).
- **DistilBERT.** No new DistilBERT training (stretch item S4 dropped). The DistilBERT result from the owner's earlier notebook used a different split, so it never appears in the M3 comparison tables or McNemar tests. It may only be reported as a separate side experiment, next to its own Logistic Regression baseline trained on that same split.
- **App.** Streamlit has only the Analyze and About pages. There is no History, Model or Baseline page, no `app/db.py` and no SQLite history.
- **Unchanged.** All tests, ruff, and the rule that no number is ever typed by hand.

## 10. New repository and software architecture

The new repository `careerverify-xai` is built around one installable Python package, `cvx`, driven by one config file per experiment. Colab notebooks, scripts and the web app all call the same package, so code is never copied between them and every result can be traced to a config and a run folder.

### 10.1 Design rules

1. **One package.** All logic lives in `src/cvx/`. Notebooks and the app contain no ML logic, only calls.
2. **Config-driven runs.** Each experiment is a YAML file in `configs/`. One command runs it: `python -m cvx.run --config configs/e3_models.yaml`.
3. **Self-contained run folders.** Each run writes `runs/<experiment_id>/<timestamp>/` with the config used, environment versions, metrics, predictions, figures and models.
4. **Generated reports.** Tables and figures in `reports/` are produced only by `scripts/make_tables.py` and `scripts/make_figures.py` from run folders. Nobody types a result by hand.
5. **Tests first for core maths.** Metrics, splitting, SHAP formulas and faithfulness functions are tested on tiny hand-made examples before running on real data.

### 10.2 Folder layout

```text
careerverify-xai/
├── CLAUDE.md                 # agent rules (Claude Code)
├── AGENTS.md                 # identical copy (Antigravity / Gemini)
├── README.md
├── .streamlit/
│   └── config.toml           # localhost only, usage statistics off, error details hidden
├── pyproject.toml            # package "cvx", installed with: pip install -e .
├── requirements.in           # top-level dependencies, no versions
├── requirements-lock.txt     # exact versions frozen from Colab (Section 10.5)
├── docs/
│   ├── PRD.md                # this document
│   ├── career_verify_reported.yaml  # paper's reported values, filled in by the owner
│   └── literature/           # owner's notes and extraction table (not agent work)
├── configs/
│   ├── base.yaml             # paths, seed, split ratios, TF-IDF settings
│   ├── e0_reproduce.yaml
│   ├── e1_leakage.yaml
│   ├── e2_protocol.yaml
│   ├── e3_models.yaml
│   ├── e4_ablations.yaml
│   ├── e5_explain.yaml
│   ├── e6_faithfulness.yaml
│   ├── e7_audit.yaml
│   └── e8_rules.yaml
├── src/cvx/
│   ├── config.py             # load + validate YAML (dataclasses)
│   ├── run.py                # CLI: python -m cvx.run --config ...
│   ├── runio.py              # run folders: config.yaml, env.json, CSV/JSON, run.log, models/
│   ├── data/      load.py, profile.py, clean.py (p_cv, p_cvx), dedup.py, split.py,
│   │              prepare.py (split plan, model data), resample.py (E1, SMOTE)
│   ├── features/  text.py, meta.py, build.py (ColumnTransformer)
│   ├── models/    zoo.py (NB, SVM, RF, LR, XGB), train.py (tuning, thresholds)
│   ├── evaluate/  metrics.py, stats.py (bootstrap, McNemar, Wilcoxon), plots.py
│   ├── explain/   shap_linear.py, shap_tree.py, lime_text.py, reasons.py, highlight.py
│   ├── xai_eval/  faithfulness.py, agreement.py, stability.py
│   ├── audit/     shortcuts.py, errors.py
│   ├── rules/     safety.py (R1-R7)
│   ├── artifacts/ save.py, load.py (hash, version, smoke checks)
│   ├── report/    runs.py, tables.py, figures.py (used by scripts/make_*.py)
│   ├── schema.py             # build_input(form) -> DataFrame for the app
│   └── service.py            # analyze(), lime_compare(), health(): what the app calls
├── notebooks/
│   └── colab_runner.ipynb    # thin: clone, install, mount Drive, run configs
├── scripts/   make_tables.py, make_figures.py, export_model.py
├── app/
│   ├── main.py               # Streamlit entry point: Analyze page (st.form + result)
│   ├── pages/                # 1_About.py (speed mode: no History, Model or Baseline pages)
│   └── model_store/          # exported model files (gitignored)
├── runs/                     # experiment outputs (gitignored, synced from Drive)
├── reports/                  # generated tables and figures (committed)
├── data/                     # raw CSV (gitignored)
└── tests/                    # pytest, mirrors src/cvx/
```

### 10.3 Example config

```yaml
# configs/e3_models.yaml
experiment_id: e3_models
extends: base.yaml
preprocessing: p_cvx
split: {method: grouped_stratified, ratios: [0.70, 0.15, 0.15], near_dup_threshold: 0.95}
features: {text: true, metadata: false}
models: [nb, linear_svm, rf, lr, xgb]
tuning: {cv_folds: 5, scoring: average_precision}
thresholds: [0.5, t_f1, t_r90]
stats: {bootstrap: 1000, mcnemar: [lr, xgb]}
```

### 10.4 What each run folder contains

`config.yaml` (as run), `env.json` (Python and library versions, git commit, dataset SHA-256), `metrics.json`, `cv_results.csv`, `predictions.csv` (test-set row id, label, score), `figures/`, `run.log`, and `models/` when the config asks for export. `env.json` is written at the start with `"status": "running"` and changed to `"complete"` only after every file is written, so an interrupted run is easy to spot.

### 10.5 Colab workflow

1. Open `notebooks/colab_runner.ipynb` in Colab Pro. Cell 1 clones the repo, mounts Drive and prints Python and library versions.
2. First time only: the owner runs the lock-file cell from the README in Colab. It installs `requirements.in` and writes `requirements-lock.txt` with exact versions of only the packages in `requirements.in` plus numpy, scipy, joblib and threadpoolctl (no Colab or NVIDIA packages, because Colab is Linux and the laptop is Windows). The owner commits that file. From then on, Colab and the laptop both install exactly that file, and both use Python 3.13.
3. Cell 2 installs the lock file and the package (`pip install -e .`). Later cells each run one config, reading the CSV from `MyDrive/FakeJobProject/data/raw/fake_job_postings.csv` and writing to `MyDrive/FakeJobProject/cvx_runs/`.
4. The owner downloads the run folders into `runs/` locally, then runs `scripts/make_tables.py` and `scripts/make_figures.py`.
5. A CPU runtime is enough for every experiment; a GPU is not needed.

### 10.6 Model artifact contract (.pkl)

`scripts/export_model.py` copies the chosen models from a run into `app/model_store/`:

| File | Contents |
| --- | --- |
| `<model_id>.pkl` | Whole scikit-learn Pipeline (features + model), saved with `joblib.dump(compress=3)` |
| `<model_id>_booster.json` | XGBoost `save_model` backup (XGBoost models only) |
| `<model_id>_feature_means.npy` | Training means, used by the exact linear SHAP formula |
| `<model_id>_features.json` | Readable feature names in column order |
| `model_card.json` | Per model: SHA-256 of each file, `t_f1`, `t_r90`, test metrics copied from the run, `input_columns`, library versions, dataset hash, seed, git commit |
| `smoke_samples.json` | 5 fixed postings with each model's exact score from the training run |

The app loads its models once per process, in a function cached with `st.cache_resource`. That function (1) loads only files listed in `model_card.json`, (2) checks each SHA-256, (3) refuses if the scikit-learn, XGBoost or NumPy major or minor version differs, and (4) re-scores the smoke samples and refuses if any score differs by more than 1e-6. "Refuses" means every page shows the error with `st.error` and calls `st.stop()`, so no prediction is ever made with a model that failed a check. Never load a .pkl from the internet or another repository, because loading a pickle can run code.

### 10.7 Tech stack

| Layer | Tools |
| --- | --- |
| Language and packaging | Python (same minor version as Colab), `pyproject.toml`, pip |
| Data and ML | pandas, numpy, scipy, scikit-learn, xgboost, imbalanced-learn, nltk (P-CV only) |
| Explainability | shap, lime |
| Statistics and plots | scipy.stats, matplotlib |
| Config | PyYAML with dataclasses |
| Web | Streamlit (`st.form` for input, `st.pyplot` for matplotlib charts, `st.cache_resource` for model loading, `st.session_state` for the latest result); no database (speed mode §9.1) |
| Testing and quality | pytest, ruff |
| Compute | Google Colab Pro (CPU runtime), Google Drive |

Exact versions come only from `requirements-lock.txt`; the agent never picks version numbers.

## 11. Web application requirements

> **Web app: Streamlit (changed from Flask on 20 Nov 2026)**

The app is a local Streamlit website that runs on the laptop with `streamlit run app/main.py`. Career Verify uses a local Flask website; both are local web apps on a normal laptop, so the deployment setting is comparable. A user pastes a posting and gets a score, a risk level, highlighted words, plain-English reasons, separate rule-based safety warnings and an optional LIME comparison. Every number shown comes from the model or from `reports/`.

The app files contain no ML logic. They call `src/cvx/service.py` (11.6), which calls the rest of the `cvx` package.

### 11.1 Pages

Streamlit lists the pages in the sidebar; there are no URL routes to design. Streamlit re-runs the page script after every click, so the latest analysis is kept in `st.session_state` and stays on screen until the next one.

| Page | File | Shows |
| --- | --- | --- |
| Analyze | `app/main.py` | Input form (11.2) built with `st.form`. After submit, on the same page: score, risk level, reasons, highlighted text, top-10 SHAP chart (`st.pyplot`), safety checks, and an "Explain with LIME" button outside the form |
| About | `app/pages/1_About.py` | What the tool can and cannot do, dataset age, the "not proof" warning |

Every page's sidebar has a "System check" box showing the result of `service.health()`: loaded models, library versions and the smoke-check result.

### 11.2 Input modes

1. **Full posting:** title\*, company name, company description, description\*, requirements, benefits, salary range; dropdowns for employment type, experience, education and function (each with Unknown); yes/no for logo, screening questions and remote. Uses the text + metadata model.
2. **Text only:** one large box for a message from WhatsApp, email and so on. Uses the text-only model.
3. Both modes have optional recruiter email and company website fields (rules only), and a model choice (Auto, LR, XGBoost). Maximum 20,000 characters. Fields marked \* are required; they are checked after submit and a missing one is shown with `st.error`.

The mode is chosen with a radio button above the form. Widgets inside an `st.form` do not update the page until the form is submitted, so a mode switch inside the form would not change which fields are shown.

### 11.3 Safety rules (not machine learning)

These live in `src/cvx/rules/safety.py` and are shown in a separate box titled "Safety checks (rules, not AI)". Each fired rule returns its ID, severity, message and matched words.

| ID | Rule | Severity |
| --- | --- | --- |
| R1 | Payment request: registration, processing, training or security fee; deposit; "refundable"; UPI, Google Pay, PhonePe, Paytm | High |
| R2 | Sensitive data request: Aadhaar, PAN, bank account, OTP, CVV, passport copy | High |
| R3 | Recruiter email from a free provider (gmail, yahoo, outlook, hotmail, rediffmail, ymail, protonmail) | Medium |
| R4 | Recruiter email domain differs from company website domain | Medium |
| R5 | Contact only through WhatsApp or Telegram | Medium |
| R6 | Guaranteed job, no interview, or unrealistic pay such as "earn ₹5,000 per day from home" | Medium |
| R7 | Urgency pressure: "immediate joining", "limited seats", "apply within 24 hours" | Low |

### 11.4 Risk level

The model level and the rules level are computed separately and the higher one is shown as Overall. Scores are never added together.

| Source | Low | Medium | High |
| --- | --- | --- | --- |
| Model | score < `t_r90` | `t_r90` ≤ score < `t_f1` | score ≥ `t_f1` |
| Rules | none, or only Low rules | one Medium rule | any High rule, or two or more Medium rules |

Example display: "Model: Medium · Safety rules: High (asks for a registration fee) · Overall: High".

### 11.5 Result details

- **Highlighting:** red for words pushing towards fake, green towards genuine, strength by absolute SHAP value. Each highlight also has a +/− sign and a tooltip, so colour is never the only signal. The highlighted HTML is built by `cvx.explain.highlight` from text escaped with `html.escape`, then shown with `st.markdown(..., unsafe_allow_html=True)`.
- **Reasons:** templates filled from the top-5 SHAP features, for example "The phrase ‘registration fee’ appears much more often in fake postings in the training data" or "The posting has no company logo".
- **LIME:** runs only when the button is clicked (`num_samples=1000`), inside `st.spinner`, and shows its top-10 words next to SHAP's, with their top-5 Jaccard overlap. It uses the input kept in `st.session_state`.
- **Fixed caution:** "These reasons show what the model learned from older public job postings. They are not proof. Verify the company yourself."

### 11.6 App functions (no storage)

There is no HTTP API. The pages call three plain Python functions in `src/cvx/service.py`. They do not import Streamlit, so they can be tested directly with pytest.

| Function | Returns | Called by |
| --- | --- | --- |
| `analyze(form, mode, model_choice, models)` | Score, label, thresholds, SHAP list, highlights, reasons, rules, risk level | Analyze page, on form submit |
| `lime_compare(form, mode, model_id, shap_words, models)` | LIME top-10 words and top-5 Jaccard with SHAP | "Explain with LIME" button |
| `health(models)` | Loaded models, versions, smoke-check result | Sidebar "System check" on every page |

`models` is the object returned by the cached loading function (Section 10.6).

No storage (speed mode §9.1): there is no database and no history. The latest input and result live only in `st.session_state` for the current browser session and are never written to disk. Data never leaves the laptop.

### 11.7 Non-functional requirements

- SHAP result under 2 seconds and LIME under 15 seconds on a normal laptop CPU. Models are loaded once per process (`st.cache_resource`), so these times do not include loading.
- Works offline: Streamlit serves its own front-end files from the laptop; no external links, fonts or CDNs. Usage statistics are off (`browser.gatherUsageStats = false` in `.streamlit/config.toml`).
- Security:
  - Posting text is passed through `html.escape` before any output that uses `unsafe_allow_html=True`. This replaces the template auto-escaping a Flask app would have.
  - The server listens on localhost only (`server.address = "localhost"`).
  - Python error details are hidden from the page (`client.showErrorDetails`).
  - The app needs no secret. If one is ever added, it comes from an environment variable (for example `CVX_SECRET`), never from code.
  - Posting text is never written to logs.
- Accessibility: every widget has a label, keyboard navigation works, colours meet WCAG AA contrast, and every `st.pyplot` chart is followed by the same numbers in a table, because screen readers cannot read chart images.

## 12. Milestones and acceptance criteria

The agent builds eight milestones in order and stops after each for the owner to review. A milestone is done only when every box below is ticked and `pytest` passes.

| Milestone | Builds | Owner runs | Week |
| --- | --- | --- | --- |
| M1 | Repo, package skeleton, config loader, data modules | pytest locally | 1 |
| M2 | Features, model zoo, training, metrics, CLI, Colab runner, E0 and E1 configs | E0, E1 in Colab | 2 |
| M3 | E2–E4 configs, statistics, table and figure scripts | E2–E4 in Colab | 3–4 |
| M4 | Explanation module (E5) | pytest; E5 in Colab | 5 |
| M5 | Faithfulness, agreement, stability, audit (E6, E7) | E6, E7 in Colab | 6 |
| M6 | Model export, safe loading, Streamlit core, safety rules | App on laptop | 7 |
| M7 | LIME in app, Model, Baseline and About pages, E8 | App; E8 in Colab | 8 |
| M8 | Reports, README, reproducibility check | Fresh-clone test | 9 |

Speed mode (§9.1): built in three batches, M2+M3, M4+M5 and M6+M7+M8, with a stop for the owner's Colab run after each batch. Every acceptance box below still applies.

### M1 — Foundation

- [ ] Folder layout matches Section 10.2; `pip install -e .` works; `.gitignore` excludes `data/`, `runs/`, `app/model_store/`, `app/data/`
- [ ] `config.py` loads YAML with `extends` and rejects unknown keys
- [ ] `data/load.py` stops with a clear message if the CSV is missing; `profile.py` writes every statistic in Section 6.3
- [ ] `clean.py` implements both P-CV and P-CVX (Section 6.4), each with tests
- [ ] `dedup.py` and `split.py` implement exact and near-duplicate grouping and the grouped stratified split; a test proves no group crosses splits

### M2 — Training core and reproduction

- [ ] `features/` builds text-only and text + metadata transformers; metadata features as in Section 7
- [ ] `models/zoo.py` has NB, linear SVM, RF, LR and XGBoost with the grids in E3; `train.py` tunes by grouped CV and picks `t_f1` and `t_r90` on validation
- [ ] `evaluate/metrics.py` computes every metric in Section 6.5 and tests check them against hand-calculated examples
- [ ] `python -m cvx.run --config configs/e0_reproduce.yaml` writes a complete run folder (Section 10.4)
- [ ] `colab_runner.ipynb` runs E0 and E1 end to end in Colab

### M3 — Corrected evaluation

- [ ] E2, E3 and E4 configs run; bootstrap CIs and McNemar's test included
- [ ] `make_tables.py` writes Markdown tables to `reports/tables/` from run folders, including "Career Verify reported vs E0 reproduced vs E2 corrected"
- [ ] `make_figures.py` writes PR curves, ROC curves and confusion matrices to `reports/figures/`

### M4 — Explanations

- [ ] Linear SHAP formula matches `shap.LinearExplainer` within 1e-6 on 20 postings (test)
- [ ] TreeExplainer additivity holds within 1e-4 (test)
- [ ] LIME wrapper keeps metadata fixed and recomputes text-based counts (test)
- [ ] `reasons.py` and `highlight.py` produce sentences and character spans; global SHAP top-30 saved

### M5 — XAI evaluation and audit

- [ ] comp@k, suff@k and AOPC tested on a toy model where the right answer is known
- [ ] E6 writes per-posting results, a summary with Wilcoxon tests, and comprehensiveness-vs-k curves for SHAP, LIME and random
- [ ] The M4+M5 batch runs in one Colab CPU session in under 1 hour; if slower, LIME `num_samples` is reduced and the setting recorded
- [ ] E7 writes the audit, E7a and E7b results, and `errors.csv`

### M6 — App core

- [ ] `export_model.py` produces every file in Section 10.6
- [ ] The cached model loader (`st.cache_resource`) refuses on a hash, version or smoke-score mismatch: the app shows the error and stops (`st.stop()`); `service.health()` reports the check in the sidebar
- [ ] The Analyze page works in both input modes: submitting the `st.form` shows score, risk level, reasons, highlights, SHAP chart (`st.pyplot`) and safety checks, and the result stays on screen across reruns (`st.session_state`)
- [ ] `.streamlit/config.toml` sets localhost only, usage statistics off and error details hidden
- [ ] Rules R1–R7 each have a test that fires and one that does not
- [ ] SHAP result under 2 seconds; `service.py` is tested directly with pytest and the Analyze page with Streamlit's `AppTest`; app tests use a tiny model trained inside the test, never real results

### M7 — App complete

- [ ] LIME button works under 15 seconds with `st.spinner` and the SHAP comparison
- [ ] The About page exists in `app/pages/` (speed mode: no History, Model or Baseline pages, no SQLite)
- [ ] Works offline; keyboard and screen-reader basics checked; every chart has a matching table
- [ ] E8 rule hit rates written to `reports/`

### M8 — Reporting and reproducibility

- [ ] README covers install, Colab steps, running each experiment, exporting models, running the app and tests
- [ ] A fresh clone, following only the README, re-runs E0 and gets the same metrics as the stored run
- [ ] Every number in `reports/` traces to a run folder

### Stretch items (only after M8)

S1 external Indian test set (E8 optional part); S2 screenshot input with OCR; S3 batch CSV upload; S4 DistilBERT model with SHAP: dropped in speed mode (§9.1); no new DistilBERT training.

## 13. Risks, limitations and ethics

The biggest risks are an agent inventing numbers, version mismatch between Colab and the laptop, and honest results looking weaker than published ones. Each has a planned response below.

### 13.1 Project risks

| Risk | Effect | Response |
| --- | --- | --- |
| Agent invents results or fake data | Report contains numbers nobody produced | `CLAUDE.md` rule; all tables generated from run folders; owner checks one number per table against `metrics.json` |
| Library versions differ between Colab and laptop | Model fails to load or scores change | `requirements-lock.txt`, hash, version and smoke checks at startup |
| E0 does not reproduce 97% | Looks like a failed reproduction | Expected; E1 tests the likely cause. Report reproduction gaps openly with every assumed setting |
| Corrected F1 lower than published papers | Reviewer doubts the work | Show E0 → E1 → E2 side by side; the gap is a finding, not a weakness |
| Colab disconnects | Lost runs | Each experiment writes to Drive as it finishes and can be re-run from its config alone |
| E6 too slow | Delays | Reduce LIME `num_samples` or sample size, and record the change |
| Owner cannot explain agent-written code | Problems in reviews and viva | Ask the agent to explain each file; re-type key functions; read the glossary |

### 13.2 Limitations to state in the report

- EMSCAD postings are from 2012–2014 and mostly from outside India; today's WhatsApp, Telegram and UPI scams may look different.
- Rules R3 and R4 cannot be validated on EMSCAD because it has no email or website columns.
- Word-deletion faithfulness tests create unnatural text; the random baseline only partly controls for this.
- SHAP and LIME explain the model, not the truth. A faithful explanation of a wrong model is still wrong.
- Career Verify does not report several settings (TF-IDF parameters, exact split method), so E0 uses documented assumptions.

### 13.3 Ethics

- The tool is a research prototype. It warns users that results are not proof and must not be used to accuse a real employer.
- The optional Indian test set must have names, phone numbers, emails and links removed before storage, and it is never published.
- The app stores no analyses: inputs and results live only in the current browser session.
- Credit Career Verify fairly: report its results as stated, and describe the inconsistencies as observations from its own numbers, not as accusations.

## 14. Glossary

| Term | Meaning in plain words |
| --- | --- |
| Ablation | Remove or change one part to see how much it matters |
| AOPC | Average comprehensiveness over several values of k |
| Base paper | The published paper this project reproduces and improves (Career Verify) |
| Bootstrap CI | Resample the test set many times and take the middle 95% of the metric values |
| Class weights | Make mistakes on the rare class (fake) count more during training |
| Comprehensiveness | Drop in confidence when the explanation's top words are removed; higher = more faithful |
| Config-driven run | An experiment fully described by a YAML file, so it can be re-run exactly |
| Data leakage | Information from the test set reaching training, making scores look better than they are |
| Duplicate / near-duplicate | Postings with the same or almost the same text (cosine similarity ≥ 0.95) |
| EMSCAD | Employment Scam Aegean Dataset, the public fake job dataset |
| F1 | 2 × precision × recall / (precision + recall) |
| Faithfulness | Whether an explanation reflects what really drove the model |
| Grouped stratified split | Keeps the fake/genuine ratio equal in each split and keeps each duplicate group together |
| Jaccard | Shared items ÷ all distinct items in two sets |
| LIME | Explains one prediction by removing words at random and fitting a simple model to the changes |
| Logistic Regression | Linear model that weighs each feature and outputs a probability |
| McNemar's test | Checks whether two models make significantly different mistakes |
| Oversampling | Copying or synthesising rare-class examples to balance classes; leaks if done before splitting |
| PR-AUC | Area under the precision–recall curve; best single summary for rare classes |
| Precision / Recall | Share of flagged postings that are fake / share of fake postings that are caught |
| Reproduction | Re-implementing a paper's method to check its reported results |
| SHAP | Gives each feature a fair share of the push from the average prediction to this one |
| Shortcut | A feature that works on this dataset by accident and will not generalise |
| Smoke test | Quick check that loaded models give the expected scores |
| SMOTE | Creates synthetic rare-class examples by blending real ones |
| Sufficiency | Drop in confidence when only the top words are kept; lower = more faithful |
| TF-IDF | Word score: frequent in this posting, rare across all postings |
| Threshold (`t_f1`, `t_r90`) | Score cut-off for "fake": best-F1 cut-off, and the highest cut-off that still gives at least 90% recall on validation |
| Wilcoxon test | Compares paired values (two methods on the same postings) without assuming a normal distribution |
| XGBoost | Many small decision trees, each correcting the previous ones |

## 15. References

### 15.1 Base paper and dataset

1. Bhagyashree M and P. Shetty, "Career Verify – A Fake Job Posting Detection," 2026 International Conference on Artificial Intelligence and Data Engineering (AIDE), IEEE, pp. 822–826, DOI 10.1109/AIDE69088.2026.11544522.
2. S. Vidros, C. Kolias, G. Kambourakis and L. Akoglu, "Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset," Future Internet, vol. 9, no. 1, 2017. [Link](https://mdpi.com/1999-5903/9/1/6)
3. [Real / Fake Job Posting Prediction dataset, Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

### 15.2 Related work (abstracts seen; read full texts before citing numbers)

4. [XAI-Recruit: Explainable AI for Job Recruitment Fraud Detection Using Deep Learning (IEEE)](https://ieeexplore.ieee.org/document/11507788/)
5. [Online Recruitment Fraud Detection Using Transformer Models With SHAP Explainability (IEEE)](https://ieeexplore.ieee.org/abstract/document/11436921/)
6. N. Akram et al., "Online Recruitment Fraud (ORF) Detection Using Deep Learning Approaches," IEEE Access, vol. 12, pp. 109388–109408, 2024, DOI 10.1109/ACCESS.2024.3435670. [Link](https://doaj.org/article/aa7a0146d8724e668e9340a26bdcc1bd)
7. [Fraud-BERT (Springer)](https://link.springer.com/article/10.1007/s10791-025-09502-8)
8. [AI-Powered Detection of Fraudulent Job Listings using DistilBERT and XGBoost (IEEE)](https://ieeexplore.ieee.org/document/11085449/)
9. [ORFDetector: Ensemble Learning Based Online Recruitment Fraud Detection, IC3 2019 (IEEE)](https://ieeexplore.ieee.org/document/8844879/)
10. [Online Recruitment Fraud Detection using ANN (IEEE)](https://ieeexplore.ieee.org/abstract/document/9636978/)
11. [Web App For Predicting Fake Job Posts Using Ensemble Classifiers (IEEE)](https://ieeexplore.ieee.org/document/10170567/)
12. [Detecting Scam in Online Job Vacancy Using Behavioral Features Extraction (IEEE)](https://ieeexplore.ieee.org/document/8969842/)
13. [Recruitment Scam Detection Using Gated Recurrent Unit (IEEE)](https://ieeexplore.ieee.org/document/9929928/)
14. [A Machine Learning Approach to Detecting Fraudulent Job Types, AI & Society 2022 (Springer)](https://link.springer.com/doi/10.1007/s00146-022-01469-0)
15. [S. Salloum et al., Analysis of Fraudulent Job Postings Using Machine Learning](https://salford-repository.worktribe.com/output/3541625/analysis-of-fraudulent-job-postings-using-machine-learning)

### 15.3 Method references (from memory; confirm details before citing)

16. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017 (SHAP).
17. S. M. Lundberg et al., "From Local Explanations to Global Understanding with Explainable AI for Trees," Nature Machine Intelligence, 2020 (TreeExplainer).
18. M. T. Ribeiro, S. Singh and C. Guestrin, "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016 (LIME).
19. J. DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP Models," ACL 2020 (comprehensiveness and sufficiency).
20. S. Kapoor and A. Narayanan, "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science," Patterns, 2023.
21. N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, 2002.
22. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.

### 15.4 Tools

23. [Google Antigravity rules documentation](https://antigravity.google/docs/rules.md) (AGENTS.md / GEMINI.md)
