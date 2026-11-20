"""Highlight explained words inside a posting's text (PRD §11.5).

``word_spans`` finds every whole-word, case-insensitive occurrence of each explained
word and returns character positions. ``render_html`` turns the text into HTML in
which the posting text is ALWAYS escaped first (so text such as ``<script>`` can never
run in the app), and each highlight carries a +/- sign and a tooltip, so colour is
never the only signal (accessibility).
"""

import html
import re
from dataclasses import dataclass

# Placeholder tokens made by cleaning never appear in the original text as words.
PLACEHOLDERS = {"urltoken", "emailtoken", "phonetoken"}


@dataclass(frozen=True)
class Span:
    """One highlighted occurrence: text[start:end] is ``word``; score > 0 means fake."""

    start: int
    end: int
    word: str
    score: float


def word_spans(text: str, word_scores: dict[str, float]) -> list[Span]:
    """All occurrences of the scored words in ``text`` (no overlaps, sorted by start)."""
    spans: list[Span] = []
    taken = [False] * len(text)
    # Longer words first so that, e.g., 'payment' wins over a shorter overlapping word.
    for word, score in sorted(word_scores.items(), key=lambda item: (-len(item[0]), item[0])):
        if not word or word in PLACEHOLDERS or score == 0:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            if not any(taken[match.start() : match.end()]):
                spans.append(Span(match.start(), match.end(), word, float(score)))
                taken[match.start() : match.end()] = [True] * (match.end() - match.start())
    return sorted(spans, key=lambda span: span.start)


def _mark(text: str, span: Span, strongest: float) -> str:
    """One <mark>: red for fake, green for genuine, opacity by strength, sign + tooltip."""
    fake = span.score > 0
    strength = 0.25 + 0.6 * (abs(span.score) / strongest if strongest else 0)
    colour = f"rgba(220, 38, 38, {strength:.2f})" if fake else f"rgba(22, 163, 74, {strength:.2f})"
    sign = "+" if fake else "−"
    direction = "towards fake" if fake else "towards genuine"
    tip = html.escape(f"{span.word}: {span.score:+.3f} ({direction})", quote=True)
    word = html.escape(text[span.start : span.end])
    return (
        f'<mark title="{tip}" style="background:{colour};color:#111;border-radius:3px">'
        f"{word}<sup>{sign}</sup></mark>"
    )


def render_html(text: str, spans: list[Span]) -> str:
    """Escaped text with highlighted spans (safe for ``unsafe_allow_html=True``)."""
    strongest = max((abs(span.score) for span in spans), default=0.0)
    parts, position = [], 0
    for span in spans:
        parts.append(html.escape(text[position : span.start]))
        parts.append(_mark(text, span, strongest))
        position = span.end
    parts.append(html.escape(text[position:]))
    return "".join(parts).replace("\n", "<br>")
