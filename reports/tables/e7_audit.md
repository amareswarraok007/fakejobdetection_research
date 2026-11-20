# e7_audit

Source: `e7_audit/20261120-062514` · git `2cb11a5e0b6cca94dd2fba7315d5e7ad611ef492`

## lr_text

| Re-train | Δ PR-AUC | Δ ROC-AUC | Δ F1 @ t_f1 | Share of top features from audit columns |
| --- | --- | --- | --- | --- |
| E7a + audit columns | -0.017 | -0.013 | -0.030 | 0.667 |
| E7b masked tokens | -0.001 | -0.002 | 0.009 | 0.000 |

Flagged and masked tokens: `industry`, `new`, `mobile`
