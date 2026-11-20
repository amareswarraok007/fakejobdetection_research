# e2_protocol

Source: `e2_protocol/20261120-012656` · git `0b9c99c6d4f454a178bf2ec30929cf48275d6815` · dataset sha256 `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`

## Threshold-free test metrics

| Variant | Model | Test rows | Test fake share | PR-AUC [95% CI] | ROC-AUC [95% CI] |
| --- | --- | --- | --- | --- | --- |
| e2a_p_cv | lr | 2661 | 0.061 | 0.827 [0.774, 0.876] | 0.949 [0.929, 0.967] |
| e2b_p_cvx | lr | 2661 | 0.061 | 0.846 [0.797, 0.891] | 0.957 [0.938, 0.973] |

## Test metrics at threshold `0.5`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2a_p_cv | lr | 0.500 | 0.857 [0.801, 0.912] | 0.745 [0.678, 0.814] | 0.797 [0.745, 0.844] | 0.893 | 0.977 | 2480 | 20 | 41 | 120 |
| e2b_p_cvx | lr | 0.500 | 0.837 [0.781, 0.890] | 0.795 [0.734, 0.856] | 0.815 [0.771, 0.859] | 0.902 | 0.978 | 2475 | 25 | 33 | 128 |

## Test metrics at threshold `t_f1`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2a_p_cv | lr | 0.704 | 0.894 [0.840, 0.943] | 0.683 [0.614, 0.755] | 0.775 [0.721, 0.823] | 0.881 | 0.976 | 2487 | 13 | 51 | 110 |
| e2b_p_cvx | lr | 0.583 | 0.873 [0.821, 0.922] | 0.770 [0.707, 0.835] | 0.818 [0.772, 0.859] | 0.904 | 0.979 | 2482 | 18 | 37 | 124 |

## Test metrics at threshold `t_r90`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2a_p_cv | lr | 0.021 | 0.179 [0.154, 0.206] | 0.963 [0.929, 0.988] | 0.302 [0.267, 0.339] | 0.568 | 0.731 | 1791 | 709 | 6 | 155 |
| e2b_p_cvx | lr | 0.022 | 0.188 [0.161, 0.214] | 0.901 [0.854, 0.942] | 0.311 [0.272, 0.347] | 0.582 | 0.758 | 1873 | 627 | 16 | 145 |
