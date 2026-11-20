# e0_reproduce

Source: `e0_reproduce/20261120-012520` · git `0b9c99c6d4f454a178bf2ec30929cf48275d6815` · dataset sha256 `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`

## Threshold-free test metrics

| Variant | Model | Test rows | Test fake share | PR-AUC [95% CI] | ROC-AUC [95% CI] |
| --- | --- | --- | --- | --- | --- |
| main | lr | 1788 | 0.049 | 0.942 [0.903, 0.975] | 0.991 [0.982, 0.998] |

## Test metrics at threshold `0.5`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | lr | 0.500 | 0.844 [0.759, 0.916] | 0.874 [0.800, 0.944] | 0.859 [0.795, 0.911] | 0.926 | 0.986 | 1687 | 14 | 11 | 76 |

## Test metrics at threshold `t_f1`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | lr | 0.712 | 0.923 [0.862, 0.975] | 0.828 [0.744, 0.907] | 0.873 [0.814, 0.925] | 0.933 | 0.988 | 1695 | 6 | 15 | 72 |

## Test metrics at threshold `t_r90`

| Variant | Model | Threshold | Precision | Recall | F1 (fake) | Macro-F1 | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | lr | 0.300 | 0.717 [0.629, 0.798] | 0.931 [0.874, 0.979] | 0.810 [0.748, 0.865] | 0.899 | 0.979 | 1669 | 32 | 6 | 81 |
