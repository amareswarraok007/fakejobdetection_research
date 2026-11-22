"""Offline CSS and safe HTML presentation helpers for the Streamlit UI.

Pure presentation only — zero ML logic, zero external URLs or CDNs (PRD §11.7).
Every dynamic string inserted into HTML is passed through ``html.escape``.
"""

import html
from typing import Any

RISK_PALETTE: dict[str, dict[str, str]] = {
    "High": {
        "bg": "#FEF2F2",
        "border": "#DC2626",
        "text": "#991B1B",
        "badge_bg": "#DC2626",
        "label": "HIGH RISK — Strong fraud signals detected",
    },
    "Medium": {
        "bg": "#FFFBEB",
        "border": "#D97706",
        "text": "#92400E",
        "badge_bg": "#D97706",
        "label": "MEDIUM RISK — Verify employer carefully before applying",
    },
    "Low": {
        "bg": "#F0FDF4",
        "border": "#16A34A",
        "text": "#166534",
        "badge_bg": "#16A34A",
        "label": "LOW RISK — No strong fraud signals detected",
    },
}

APP_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}
.cvx-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 62%, #2563EB 100%);
    color: #FFFFFF;
    padding: 1.6rem 1.85rem;
    border-radius: 14px;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
}
.cvx-hero h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}
.cvx-hero p {
    margin: 0.45rem 0 0.9rem 0;
    font-size: 0.96rem;
    color: #DBEAFE;
    line-height: 1.45;
}
.cvx-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
.cvx-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.24);
    color: #F8FAFC;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
}
.cvx-section-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    margin: 0.2rem 0 0.5rem 0;
}
.cvx-verdict {
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    border-left: 6px solid;
    margin: 0.35rem 0 1rem 0;
}
.cvx-verdict-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.cvx-badge {
    display: inline-block;
    color: #FFFFFF;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.24rem 0.7rem;
    border-radius: 999px;
    text-transform: uppercase;
}
.cvx-gauge-wrap {
    margin-top: 0.85rem;
}
.cvx-gauge-track {
    position: relative;
    height: 10px;
    background: #E2E8F0;
    border-radius: 999px;
    overflow: hidden;
}
.cvx-gauge-fill {
    height: 100%;
    border-radius: 999px;
}
.cvx-gauge-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.35rem;
}
.cvx-highlight-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    line-height: 1.85;
    font-size: 0.96rem;
    color: #0F172A;
    max-height: 360px;
    overflow-y: auto;
}
.cvx-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 0.55rem;
}
.cvx-rule-item {
    border: 1px solid #E2E8F0;
    border-left: 4px solid;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.55rem;
    background: #FFFFFF;
}
.cvx-rule-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
}
.cvx-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 0.08rem 0.45rem;
    font-size: 0.78rem;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    margin-right: 0.3rem;
}
.cvx-sidebar-brand {
    background: #0F172A;
    color: #FFFFFF;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.85rem;
}
.cvx-sidebar-brand strong {
    display: block;
    font-size: 1rem;
}
.cvx-sidebar-brand span {
    font-size: 0.78rem;
    color: #93C5FD;
}
.cvx-status-ok {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.65rem;
}
</style>
"""


def hero_html(title: str, subtitle: str, pills: tuple[str, ...]) -> str:
    """Top gradient hero banner (all inputs escaped)."""
    chips = "".join(f'<span class="cvx-pill">{html.escape(p)}</span>' for p in pills)
    return (
        '<div class="cvx-hero">'
        f"<h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(subtitle)}</p>"
        f'<div class="cvx-pills">{chips}</div>'
        "</div>"
    )


def verdict_html(result: dict[str, Any]) -> str:
    """Color-coded risk verdict card with probability gauge bar."""
    risk = result["risk"]
    overall = risk["overall"]
    pal = RISK_PALETTE.get(overall, RISK_PALETTE["Low"])
    score = float(result["score"])
    pct = max(0.0, min(100.0, score * 100.0))
    thresholds = result["thresholds"]
    t_r90 = float(thresholds.get("t_r90", 0.0))
    t_f1 = float(thresholds.get("t_f1", 0.5))
    meta = html.escape(
        f"Model: {risk['model']}  ·  Safety rules: {risk['rules']}  ·  "
        f"Active model: {result['model_id']}"
    )
    box_style = f"background:{pal['bg']};border-left-color:{pal['border']}"
    return (
        f'<div class="cvx-verdict" style="{box_style}">'
        '<div class="cvx-verdict-row">'
        f'<div style="color:{pal["text"]};font-weight:700;font-size:1.05rem">'
        f'{html.escape(pal["label"])}</div>'
        f'<span class="cvx-badge" style="background:{pal["badge_bg"]}">'
        f"{html.escape(overall)} Risk</span>"
        "</div>"
        f'<div style="color:#334155;font-size:0.86rem;margin-top:0.35rem">{meta}</div>'
        '<div class="cvx-gauge-wrap">'
        '<div class="cvx-gauge-track">'
        f'<div class="cvx-gauge-fill" style="width:{pct:.1f}%;background:{pal["border"]}"></div>'
        "</div>"
        '<div class="cvx-gauge-meta">'
        f"<span>Fake probability: <strong>{pct:.1f}%</strong> ({score:.3f})</span>"
        f"<span>Thresholds: t_r90 = {t_r90:.3f} · t_f1 = {t_f1:.3f}</span>"
        "</div></div></div>"
    )


def rule_card_html(hit: dict[str, Any]) -> str:
    """One styled safety-rule card with severity pill and matched-term chips."""
    sev = str(hit["severity"])
    pal = RISK_PALETTE.get(sev, RISK_PALETTE["Low"])
    msg = html.escape(str(hit["message"]))
    chips = "".join(
        f'<span class="cvx-chip">{html.escape(str(m))}</span>' for m in hit["matched"]
    )
    return (
        f'<div class="cvx-rule-item" style="border-left-color:{pal["border"]}">'
        '<div class="cvx-rule-head">'
        f'<span class="cvx-badge" style="background:{pal["badge_bg"]}">'
        f"{html.escape(str(hit['rule_id']))} · {html.escape(sev)}</span>"
        f'<strong style="color:#0F172A;font-size:0.9rem">{msg}</strong>'
        "</div>"
        f'<div style="margin-top:0.3rem;font-size:0.8rem;color:#475569">Matched: {chips}</div>'
        "</div>"
    )
