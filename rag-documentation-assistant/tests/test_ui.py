"""Streamlit UI smoke tests — no API key, no network needed."""

import pytest

from app import config

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture()
def ui(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", str(tmp_path / "ui_fb.db"))
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_ui_has_four_tabs(ui):
    assert [t.label for t in ui.tabs] == ["Ask", "Ingest", "Documents", "Feedback"]


def test_ask_without_api_key_shows_clear_error(ui):
    ui.text_area(key="question_input").set_value("What is dependency injection?")
    ui.button(key="ask_button").click().run()
    assert not ui.exception
    assert any("GEMINI_API_KEY" in e.value for e in ui.error)


def test_feedback_submit_from_ui(ui):
    ui.text_input(key="fb_question").set_value("What is DI?")
    ui.radio(key="fb_rating").set_value("positive")
    ui.button(key="fb_submit_btn").click().run()
    assert not ui.exception
    assert any("Feedback saved" in s.value for s in ui.success)


def test_documents_tab_renders(ui):
    # With an empty index it shows the empty-state caption, no crash.
    assert not ui.exception
