# e1_leakage

Source: `e1_leakage/20261120-012548` · git `0b9c99c6d4f454a178bf2ec30929cf48275d6815` · dataset sha256 `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`

## Threshold-free test metrics

| Variant | Model | Test rows | Test fake share | PR-AUC [95% CI] | ROC-AUC [95% CI] |
| --- | --- | --- | --- | --- | --- |
| e1a_oversample_before_split | lr | 3403 | 0.500 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| e1b_smote_before_split | lr | 3403 | 0.500 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| e1c_oversample_train_only | lr | 1788 | 0.049 | 0.941 [0.902, 0.975] | 0.991 [0.980, 0.998] |

## Test metrics at threshold `0.5`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1a_oversample_before_split | lr | 0.500 | 0.992 [0.988, 0.996] | 1.000 [1.000, 1.000] | 0.996 [0.994, 0.998] | 0.996 | 0.996 | 1689 | 13 | 0 | 1701 |
| e1b_smote_before_split | lr | 0.500 | 0.995 [0.991, 0.998] | 0.999 [0.998, 1.000] | 0.997 [0.995, 0.999] | 0.997 | 0.997 | 1693 | 9 | 1 | 1700 |
| e1c_oversample_train_only | lr | 0.500 | 0.852 [0.767, 0.926] | 0.862 [0.785, 0.935] | 0.857 [0.793, 0.908] | 0.925 | 0.986 | 1688 | 13 | 12 | 75 |

## Test metrics at threshold `t_f1`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1a_oversample_before_split | lr | 0.737 | 0.996 [0.993, 0.999] | 0.999 [0.997, 1.000] | 0.998 [0.996, 0.999] | 0.998 | 0.998 | 1696 | 6 | 2 | 1699 |
| e1b_smote_before_split | lr | 0.656 | 0.996 [0.993, 0.999] | 0.999 [0.997, 1.000] | 0.998 [0.996, 0.999] | 0.998 | 0.998 | 1696 | 6 | 2 | 1699 |
| e1c_oversample_train_only | lr | 0.596 | 0.892 [0.818, 0.952] | 0.851 [0.770, 0.926] | 0.871 [0.814, 0.921] | 0.932 | 0.988 | 1692 | 9 | 13 | 74 |

## Test metrics at threshold `t_r90`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1a_oversample_before_split | lr | 0.950 | 0.999 [0.998, 1.000] | 0.899 [0.885, 0.913] | 0.946 [0.939, 0.954] | 0.949 | 0.949 | 1701 | 1 | 172 | 1529 |
| e1b_smote_before_split | lr | 0.952 | 1.000 [1.000, 1.000] | 0.917 [0.903, 0.930] | 0.957 [0.949, 0.964] | 0.958 | 0.959 | 1702 | 0 | 141 | 1560 |
| e1c_oversample_train_only | lr | 0.224 | 0.713 [0.628, 0.793] | 0.943 [0.890, 0.988] | 0.812 [0.750, 0.865] | 0.900 | 0.979 | 1668 | 33 | 5 | 82 |

## Reported vs realistic performance

| Reported | Realistic | Test fake share reported | Test fake share realistic | Test fake share difference | PR-AUC reported | PR-AUC realistic | PR-AUC difference | ROC-AUC reported | ROC-AUC realistic | ROC-AUC difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1a_oversample_before_split/lr | e1c_oversample_train_only/lr | 0.500 | 0.049 | 0.451 | 1.000 | 0.941 | 0.059 | 1.000 | 0.991 | 0.009 |
| e1b_smote_before_split/lr | e1c_oversample_train_only/lr | 0.500 | 0.049 | 0.451 | 1.000 | 0.941 | 0.059 | 1.000 | 0.991 | 0.009 |

| Reported | Realistic | Threshold | Recall (fake) reported | Recall (fake) realistic | Recall (fake) difference |
| --- | --- | --- | --- | --- | --- |
| e1a_oversample_before_split/lr | e1c_oversample_train_only/lr | 0.5 | 1.000 | 0.862 | 0.138 |
| e1a_oversample_before_split/lr | e1c_oversample_train_only/lr | t_f1 | 0.999 | 0.851 | 0.148 |
| e1a_oversample_before_split/lr | e1c_oversample_train_only/lr | t_r90 | 0.899 | 0.943 | -0.044 |
| e1b_smote_before_split/lr | e1c_oversample_train_only/lr | 0.5 | 0.999 | 0.862 | 0.137 |
| e1b_smote_before_split/lr | e1c_oversample_train_only/lr | t_f1 | 0.999 | 0.851 | 0.148 |
| e1b_smote_before_split/lr | e1c_oversample_train_only/lr | t_r90 | 0.917 | 0.943 | -0.025 |
