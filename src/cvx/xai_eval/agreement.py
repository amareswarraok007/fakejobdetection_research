"""How much do SHAP and LIME agree on one posting? (PRD E6)

- Top-k Jaccard: shared words / all distinct words in the two top-k lists (k = 5, 10),
  where top-k = largest absolute effect.
- Spearman correlation of the two methods' scores over the union of their top-10 words
  (a word missing from one method gets 0 there).
- Same direction: among words in both top-10 lists, the share that push the same way.
"""

import math
from collections.abc import Iterable
from typing import Any

from scipy.stats import spearmanr

from cvx.explain.core import Explanation, top_words


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """|A ∩ B| / |A ∪ B|; 1.0 when both are empty."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def agreement(first: Explanation, second: Explanation, k_union: int = 10) -> dict[str, Any]:
    """Jaccard@5, Jaccard@10, Spearman over the top-10 union, and same-direction share."""
    top_a, top_b = top_words(first, k_union), top_words(second, k_union)
    union = sorted(set(top_a) | set(top_b))
    a_scores = [first.words.get(w, 0.0) for w in union]
    b_scores = [second.words.get(w, 0.0) for w in union]
    rho = spearmanr(a_scores, b_scores).statistic if len(union) > 1 else math.nan
    shared = set(top_a) & set(top_b)
    same = [(first.words[w] > 0) == (second.words[w] > 0) for w in shared]
    return {
        "jaccard_5": jaccard(top_words(first, 5), top_words(second, 5)),
        "jaccard_10": jaccard(top_a, top_b),
        "spearman": None if math.isnan(rho) else float(rho),
        "same_direction": float(sum(same) / len(same)) if same else None,
    }
