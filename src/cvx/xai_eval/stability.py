"""Stability of LIME across random seeds (PRD E6; speed mode: 3 seeds).

LIME is random: run it with different seeds and see whether the top-5 words stay the
same (mean pairwise top-5 Jaccard; 1 = identical every time). SHAP is deterministic
here, so its stability is 1 by construction and is not measured.
"""

from itertools import combinations

import numpy as np

from cvx.explain.core import Explanation, top_words
from cvx.xai_eval.agreement import jaccard


def mean_pairwise_jaccard(word_lists: list[list[str]]) -> float:
    """Average Jaccard over every pair of lists (1.0 for fewer than two lists)."""
    pairs = list(combinations(word_lists, 2))
    return float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else 1.0


def lime_stability(explanations: list[Explanation], k: int = 5) -> float:
    """Mean pairwise top-k Jaccard of LIME explanations of ONE posting with different seeds."""
    return mean_pairwise_jaccard([top_words(e, k) for e in explanations])
