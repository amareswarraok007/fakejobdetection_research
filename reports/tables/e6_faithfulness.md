# e6_faithfulness

Source: `e6_faithfulness/20261120-062207` · git `2cb11a5e0b6cca94dd2fba7315d5e7ad611ef492`

## lr_text: faithfulness (100 postings)

| Method | AOPC | comp@1 | comp@3 | comp@5 | comp@10 | comp@20 | suff@1 | suff@3 | suff@5 | suff@10 | suff@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shap | 0.206 | 0.038 | 0.095 | 0.149 | 0.272 | 0.476 | 0.142 | 0.004 | -0.028 | -0.046 | -0.052 |
| lime | 0.149 | 0.038 | 0.095 | 0.148 | 0.266 | 0.199 | 0.114 | -0.019 | -0.038 | -0.050 | 0.039 |
| random | 0.011 | 0.002 | 0.004 | 0.007 | 0.013 | 0.029 | 0.415 | 0.400 | 0.391 | 0.371 | 0.333 |

## lr_text: Wilcoxon signed-rank tests

| Pair | Metric | n | Median difference | p-value |
| --- | --- | --- | --- | --- |
| shap vs random | comp@5 | 100 | 0.068 | 0.0000 |
| lime vs random | comp@5 | 100 | 0.067 | 0.0000 |
| shap vs lime | comp@5 | 100 | 0.000 | 0.7507 |

## lr_text: agreement, stability, runtime

| Measure | Value |
| --- | --- |
| jaccard_5 | 0.542 |
| jaccard_10 | 0.509 |
| spearman | 0.710 |
| same_direction | 1.000 |
| LIME stability (top-5 Jaccard) | 0.561 |
| shap seconds (mean / p95) | 0.048 / 0.074 |
| lime seconds (mean / p95) | 0.595 / 1.269 |
