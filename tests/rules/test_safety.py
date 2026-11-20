"""PRD M6 check: rules R1-R7 each have a test that fires and one that does not; risk levels."""

import pytest

from cvx.rules.safety import check_all, model_level, overall_level, rules_level, text_rule


@pytest.mark.parametrize(
    ("rule", "fires", "silent"),
    [
        ("R1", "Pay the registration fee by UPI or Paytm", "The company pays for your training"),
        (
            "R2",
            "Send your Aadhaar and PAN card with bank account details",
            "Bring a company panel id",
        ),
        ("R5", "Message us on WhatsApp for the next step", "Send your CV to our careers page"),
        ("R6", "Earn ₹5,000 per day from home, no interview", "Interviews are held every Monday"),
        ("R7", "Immediate joining, limited seats, apply within 24 hours", "Start date in March"),
    ],
)
def test_text_rules(rule: str, fires: str, silent: str) -> None:
    hit = text_rule(rule, fires)
    assert hit is not None and hit.rule_id == rule and hit.matched
    assert text_rule(rule, silent) is None


def test_r3_free_email() -> None:
    assert [h.rule_id for h in check_all("", "hr.team@gmail.com", None)] == ["R3"]
    assert check_all("", "hr@acme.com", None) == []


def test_r4_domain_mismatch() -> None:
    assert "R4" in [
        h.rule_id for h in check_all("", "jobs@acme-careers.net", "https://www.acme.com")
    ]
    assert check_all("", "jobs@careers.acme.com", "https://acme.com/jobs") == []


def test_rules_level() -> None:
    def hits(text: str, email: str | None = None) -> list:
        return check_all(text, email, None)

    assert rules_level(hits("Normal posting")) == "Low"
    assert rules_level(hits("Immediate joining")) == "Low"  # only a Low rule
    assert rules_level(hits("Message on WhatsApp")) == "Medium"  # one Medium
    assert rules_level(hits("Message on WhatsApp", "x@gmail.com")) == "High"  # two Medium
    assert rules_level(hits("Pay the processing fee")) == "High"  # any High


def test_model_and_overall_level() -> None:
    assert model_level(0.9, t_f1=0.6, t_r90=0.2) == "High"
    assert model_level(0.4, t_f1=0.6, t_r90=0.2) == "Medium"
    assert model_level(0.1, t_f1=0.6, t_r90=0.2) == "Low"
    assert overall_level("Medium", "High") == "High" and overall_level("Low", "Low") == "Low"
