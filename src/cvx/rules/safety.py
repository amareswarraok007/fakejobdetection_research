"""Safety rules R1-R7 (PRD Section 11.3). These are fixed rules, NOT machine learning.

Each rule looks for a scam signal that EMSCAD does not label well (payment requests,
personal-data requests, free-mail recruiters …). A fired rule returns its id,
severity, a message and the exact words it matched.

Risk level (PRD §11.4; CLAUDE.md §6): the model level and the rules level are
computed SEPARATELY and the overall level is the higher of the two. Scores are never
added together.
"""

import re
from dataclasses import dataclass

LEVELS = ("Low", "Medium", "High")
FREE_MAIL = ("gmail", "yahoo", "outlook", "hotmail", "rediffmail", "ymail", "protonmail")

_PATTERNS: dict[str, tuple[str, str, re.Pattern[str]]] = {
    "R1": (
        "High",
        "Asks for a payment (fee, deposit or a payment app)",
        re.compile(
            r"\b(?:registration|processing|training|security)\s+(?:fee|fees|charge|charges|deposit)\b"
            r"|\bdeposit\b|\brefundable\b|\bupi\b|\bgoogle\s*pay\b|\bgpay\b|\bphone\s*pe\b|\bpaytm\b",
            re.IGNORECASE,
        ),
    ),
    "R2": (
        "High",
        "Asks for sensitive personal data",
        re.compile(
            r"\baadhaa?r\b|\bpan\s*(?:card|number|no\.?)\b|\bbank\s+account\b|\botp\b|\bcvv\b"
            r"|\bpassport\s+(?:copy|scan|photo)\b",
            re.IGNORECASE,
        ),
    ),
    "R5": (
        "Medium",
        "Asks you to make contact through WhatsApp or Telegram",
        re.compile(r"\bwhats\s?app\b|\btelegram\b", re.IGNORECASE),
    ),
    "R6": (
        "Medium",
        "Promises a guaranteed job, no interview, or unrealistic pay",
        re.compile(
            r"\b(?:100\s?%\s*)?(?:job\s+)?guarantee(?:d)?\s+(?:job|placement|selection)\b"
            r"|\bjob\s+guarantee(?:d)?\b|\bno\s+interview\b|\bwithout\s+(?:any\s+)?interview\b"
            r"|\bearn\s+(?:up\s+to\s+)?(?:rs\.?|₹|inr|\$)\s?[\d,]+\s*(?:/|per|a|every)\s*(?:day|week)\b",
            re.IGNORECASE,
        ),
    ),
    "R7": (
        "Low",
        "Puts you under time pressure",
        re.compile(
            r"\bimmediate\s+joining\b|\bjoin\s+immediately\b|\blimited\s+(?:seats|slots|vacancies)\b"
            r"|\bapply\s+within\s+\d+\s*(?:hours|hrs|days)\b|\burgent(?:ly)?\s+hiring\b",
            re.IGNORECASE,
        ),
    ),
}
TEXT_RULES = ("R1", "R2", "R5", "R6", "R7")  # need only the posting text (E8 checks these)


@dataclass(frozen=True)
class RuleHit:
    """One fired rule."""

    rule_id: str
    severity: str
    message: str
    matched: tuple[str, ...]


def text_rule(rule_id: str, text: str) -> RuleHit | None:
    """Run one text rule; None if it does not fire."""
    severity, message, pattern = _PATTERNS[rule_id]
    found = tuple(dict.fromkeys(m.group(0).strip() for m in pattern.finditer(text or "")))
    return RuleHit(rule_id, severity, message, found) if found else None


def _domain(value: str) -> str:
    """Lower-case domain of an email address or website (without 'www.')."""
    value = value.strip().lower()
    value = value.split("@", 1)[1] if "@" in value else re.sub(r"^[a-z]+://", "", value)
    value = value.split("/", 1)[0].split(":", 1)[0]
    return value[4:] if value.startswith("www.") else value


def email_rules(email: str | None, website: str | None) -> list[RuleHit]:
    """R3 (free-mail recruiter) and R4 (email domain differs from the company website)."""
    hits = []
    if email and "@" in email:
        domain = _domain(email)
        provider = domain.split(".", 1)[0]
        if provider in FREE_MAIL:
            hits.append(
                RuleHit("R3", "Medium", "Recruiter email is from a free provider", (domain,))
            )
        if website and website.strip():
            site = _domain(website)
            if not (domain == site or domain.endswith("." + site) or site.endswith("." + domain)):
                message = "Recruiter email domain differs from the company website"
                hits.append(RuleHit("R4", "Medium", message, (domain, site)))
    return hits


def check_all(text: str, email: str | None = None, website: str | None = None) -> list[RuleHit]:
    """Every rule that fires for a posting (R1-R7), in rule order."""
    hits = [hit for rule in TEXT_RULES if (hit := text_rule(rule, text)) is not None]
    return sorted(hits + email_rules(email, website), key=lambda h: int(h.rule_id[1:]))


def rules_level(hits: list[RuleHit]) -> str:
    """High: any High rule or 2+ Medium rules. Medium: one Medium. Low: otherwise."""
    severities = [hit.severity for hit in hits]
    if "High" in severities or severities.count("Medium") >= 2:
        return "High"
    return "Medium" if "Medium" in severities else "Low"


def model_level(score: float, t_f1: float, t_r90: float) -> str:
    """High: score >= t_f1. Medium: t_r90 <= score < t_f1. Low: score < t_r90."""
    if score >= t_f1:
        return "High"
    return "Medium" if score >= t_r90 else "Low"


def overall_level(model: str, rules: str) -> str:
    """The higher of the two levels (never a sum of scores)."""
    return max(model, rules, key=LEVELS.index)
