# Career Verify reported vs E0 reproduced vs E2 corrected

Logistic Regression, threshold 0.5. Career Verify column: <<fill docs/career_verify_reported.yaml>>. Runs: `e0_reproduce/20261120-012520`, `e2_protocol/20261120-012656`.

| Metric | Career Verify (reported) | E0 reproduced | E2a corrected (P-CV) | E2b corrected (P-CVX) |
| --- | --- | --- | --- | --- |
| Accuracy | <<fill docs/career_verify_reported.yaml>> | 0.986 | 0.977 | 0.978 |
| Precision (fake) | <<fill docs/career_verify_reported.yaml>> | 0.844 | 0.857 | 0.837 |
| Recall (fake) | <<fill docs/career_verify_reported.yaml>> | 0.874 | 0.745 | 0.795 |
| F1 (fake) | <<fill docs/career_verify_reported.yaml>> | 0.859 | 0.797 | 0.815 |
| PR-AUC | <<fill docs/career_verify_reported.yaml>> | 0.942 | 0.827 | 0.846 |
| ROC-AUC | <<fill docs/career_verify_reported.yaml>> | 0.991 | 0.949 | 0.957 |
| TN | <<fill docs/career_verify_reported.yaml>> | 1687 | 2480 | 2475 |
| FP | <<fill docs/career_verify_reported.yaml>> | 14 | 20 | 25 |
| FN | <<fill docs/career_verify_reported.yaml>> | 11 | 41 | 33 |
| TP | <<fill docs/career_verify_reported.yaml>> | 76 | 120 | 128 |
| Test rows | <<fill docs/career_verify_reported.yaml>> | 1788 | 2661 | 2661 |
| Test fake share | <<fill docs/career_verify_reported.yaml>> | 0.049 | 0.061 | 0.061 |
