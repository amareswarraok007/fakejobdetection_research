# Career Verify reported vs E0 reproduced vs E2 corrected

Logistic Regression, threshold 0.5. Career Verify column: Bhagyashree M and P. Shetty, Career Verify, IEEE AIDE 2026, pp. 822-826 (from docs/PRD.md §3.1-3.2). Runs: `e0_reproduce/20261120-012520`, `e2_protocol/20261120-012656`.

| Metric | Career Verify (reported) | E0 reproduced | E2a corrected (P-CV) | E2b corrected (P-CVX) |
| --- | --- | --- | --- | --- |
| Accuracy | 0.970 | 0.986 | 0.977 | 0.978 |
| Precision (fake) | 0.960 | 0.844 | 0.857 | 0.837 |
| Recall (fake) | 0.950 | 0.874 | 0.745 | 0.795 |
| F1 (fake) | 0.950 | 0.859 | 0.797 | 0.815 |
| PR-AUC | not reported | 0.942 | 0.827 | 0.846 |
| ROC-AUC | not reported | 0.991 | 0.949 | 0.957 |
| TN | 1892 | 1687 | 2480 | 2475 |
| FP | 18 | 14 | 20 | 25 |
| FN | 22 | 11 | 41 | 33 |
| TP | 450 | 76 | 120 | 128 |
| Test rows | 2382 | 1788 | 2661 | 2661 |
| Test fake share | 0.198 | 0.049 | 0.061 | 0.061 |
