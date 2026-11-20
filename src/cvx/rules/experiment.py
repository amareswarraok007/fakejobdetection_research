"""E8 — how often the safety rules fire on fake vs genuine EMSCAD postings (PRD §8.2).

Only the text rules R1, R2, R5, R6 and R7 can be checked: EMSCAD has no recruiter email
or company website columns, so R3 and R4 are reported as "not validated".
Text = the five raw text fields joined (the same text the app's rules read).
"""

from typing import Any

import pandas as pd

from cvx.data.load import LABEL_COLUMN
from cvx.data.profile import raw_joined_text
from cvx.rules.safety import TEXT_RULES, rules_level, text_rule


def _rates(fired: pd.Series, is_fake: pd.Series) -> dict[str, Any]:
    """Hit counts and rates among fake and genuine postings."""
    fake, genuine = fired[is_fake], fired[~is_fake]
    return {
        "fake_hits": int(fake.sum()),
        "fake_rate": float(fake.mean()) if len(fake) else None,
        "genuine_hits": int(genuine.sum()),
        "genuine_rate": float(genuine.mean()) if len(genuine) else None,
    }


def run_e8(postings: pd.DataFrame) -> dict[str, Any]:
    """Rule hit rates per class, for each text rule, any rule, and each rules level."""
    texts = raw_joined_text(postings)
    is_fake = postings[LABEL_COLUMN] == 1
    hits = {rule: texts.map(lambda t, r=rule: text_rule(r, t)) for rule in TEXT_RULES}
    results = {rule: _rates(series.notna(), is_fake) for rule, series in hits.items()}
    any_rule = pd.concat([s.notna() for s in hits.values()], axis=1).any(axis=1)
    levels = pd.Series(
        [
            rules_level([h for h in (hits[rule].iloc[i] for rule in TEXT_RULES) if h is not None])
            for i in range(len(postings))
        ],
        index=postings.index,
    )
    return {
        "rows": int(len(postings)),
        "fake": int(is_fake.sum()),
        "rules": results,
        "any_rule": _rates(any_rule, is_fake),
        "rules_level": {
            level: _rates(levels == level, is_fake) for level in ("Low", "Medium", "High")
        },
        "not_validated": ["R3", "R4"],
        "note": "EMSCAD has no recruiter email or website columns, so R3 and R4 cannot be checked.",
    }
