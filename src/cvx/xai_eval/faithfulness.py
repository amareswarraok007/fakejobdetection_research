"""Faithfulness of an explanation (PRD E6; CLAUDE.md §6) — the main contribution.

Idea: if an explanation's top words really drive the prediction, then
- REMOVING them should lower the model's confidence a lot (comprehensiveness, higher = better):
      comp@k = p(x) - p(x without its top-k words)
- KEEPING ONLY them should keep most of the confidence (sufficiency, lower = better):
      suff@k = p(x) - p(only the top-k words)
- AOPC = mean of comp@k over k in {1, 3, 5, 10, 20}.
p = probability of the PREDICTED label. Removing a word deletes every occurrence.
Baseline: the same measures for k RANDOM words from the posting (5 draws, averaged).
If a posting has fewer than k eligible words, all of them are used.
"""

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cvx.explain.lime_text import unlabelled_tokens

K_VALUES = (1, 3, 5, 10, 20)
# Same rule as TfidfVectorizer's default tokens: words of 2+ letters/digits.
MIN_WORD_LENGTH = 2


def eligible_words(text: str) -> list[str]:
    """Distinct words in the posting that the model can see (labels excluded), in order."""
    seen: dict[str, None] = {}
    for token in unlabelled_tokens(text):
        if len(token) >= MIN_WORD_LENGTH:
            seen.setdefault(token, None)
    return list(seen)


def remove_words(text: str, words: Sequence[str]) -> str:
    """Delete every whole-word occurrence of each word (field labels are kept)."""
    if not words:
        return text
    pattern = re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, words)) + r")(?![\w:])")
    return " ".join(pattern.sub(" ", text).split())


def keep_only(text: str, words: Sequence[str]) -> str:
    """Only the given words, in their original order and with their repeats."""
    keep = set(words)
    return " ".join(token for token in unlabelled_tokens(text) if token in keep)


def label_probability(p_fake: np.ndarray, label: int) -> np.ndarray:
    """Probability of ``label`` (1 = fake, 0 = genuine)."""
    return p_fake if label == 1 else 1.0 - p_fake


def texts_for_ranking(text: str, ranking: Sequence[str], ks: Sequence[int]) -> list[str]:
    """[removed top-k for each k] + [kept top-k for each k] (2 * len(ks) texts)."""
    removed = [remove_words(text, ranking[:k]) for k in ks]
    kept = [keep_only(text, ranking[:k]) for k in ks]
    return removed + kept


def scores_from_probs(p_label_x: float, probs: np.ndarray, ks: Sequence[int]) -> dict[str, Any]:
    """comp@k, suff@k and AOPC from the probabilities of the edited texts."""
    n = len(ks)
    comp = {int(k): float(p_label_x - probs[i]) for i, k in enumerate(ks)}
    suff = {int(k): float(p_label_x - probs[n + i]) for i, k in enumerate(ks)}
    return {"comp": comp, "suff": suff, "aopc": float(np.mean(list(comp.values())))}


def random_rankings(words: Sequence[str], draws: int, seed: int) -> list[list[str]]:
    """``draws`` random orders of the posting's words (the random-word baseline)."""
    rng = np.random.default_rng(seed)
    return [list(rng.permutation(list(words))) for _ in range(draws)]


def evaluate_posting(
    predict: Callable[[list[str]], np.ndarray],
    text: str,
    label: int,
    rankings: dict[str, list[str]],
    random_draws: list[list[str]],
    ks: Sequence[int] = K_VALUES,
) -> dict[str, Any]:
    """Faithfulness of each method's ranking for one posting, plus the random baseline.

    ``predict`` maps texts to p(fake) with everything else fixed; ``label`` is the
    predicted label; ``rankings`` maps method -> words ordered by importance towards it.
    All edited texts are scored in ONE call. Output: {method: scores, "random": averaged}.
    """
    batches = [[text]] + [texts_for_ranking(text, r, ks) for r in rankings.values()]
    batches += [texts_for_ranking(text, r, ks) for r in random_draws]
    flat = [t for batch in batches for t in batch]
    probs = label_probability(predict(flat), label)
    p_x, cursor, size = float(probs[0]), 1, 2 * len(ks)
    result: dict[str, Any] = {"p_label": p_x}
    for method in rankings:
        result[method] = scores_from_probs(p_x, probs[cursor : cursor + size], ks)
        cursor += size
    draws = [
        scores_from_probs(p_x, probs[cursor + i * size : cursor + (i + 1) * size], ks)
        for i in range(len(random_draws))
    ]
    result["random"] = _average(draws, ks)
    return result


def _average(draws: list[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    """Mean comp@k, suff@k and AOPC over the random draws."""
    return {
        "comp": {int(k): float(np.mean([d["comp"][int(k)] for d in draws])) for k in ks},
        "suff": {int(k): float(np.mean([d["suff"][int(k)] for d in draws])) for k in ks},
        "aopc": float(np.mean([d["aopc"] for d in draws])),
    }
