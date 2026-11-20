"""Streamlit app tests with AppTest and a tiny model store trained inside the tests."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture
def app(tiny_store: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The Analyze page pointed at the tiny model store."""
    monkeypatch.setenv("CVX_MODEL_STORE", str(tiny_store))
    return AppTest.from_file(str(APP / "main.py"), default_timeout=120)


def test_text_mode_analysis_and_lime(app: AppTest) -> None:
    app.run()
    assert not app.exception
    app.radio[0].set_value("text").run()
    app.text_area[0].input("Pay the registration fee by UPI and message us on WhatsApp")
    app.button[0].click().run()  # the form's Analyze button
    assert not app.exception
    result = app.session_state["result"]
    assert result["risk"]["rules"] == "High"
    assert any("Overall risk" in s.value for s in app.subheader)
    lime_button = [b for b in app.button if b.label == "Explain with LIME"][0]
    lime_button.click().run()
    assert not app.exception and app.session_state["lime"]["lime"]


def test_missing_required_field_shows_error(app: AppTest) -> None:
    app.run()
    app.button[0].click().run()  # full mode, nothing filled in
    assert any("title" in e.value for e in app.error)


def test_refuses_to_start_without_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CVX_MODEL_STORE", str(tmp_path / "nothing"))
    page = AppTest.from_file(str(APP / "main.py"), default_timeout=60).run()
    assert any("cannot start" in e.value for e in page.error)


def test_about_page() -> None:
    page = AppTest.from_file(str(APP / "pages" / "1_About.py")).run()
    assert not page.exception and "not proof" in page.warning[0].value
