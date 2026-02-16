import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_popover_markup_is_scrollable_and_accessible():
    for rel in (
        "webapp/templates/index.html",
        "webapp/templates/stock.html",
        "webapp/templates/risk.html",
    ):
        text = _read(rel)
        assert "groq-popover" in text
        assert "explain-popover" in text
        assert 'role="dialog"' in text
        assert 'aria-hidden="true"' in text
        assert 'tabindex="0"' in text


def test_popover_css_has_required_scroll_rules():
    css = _read("webapp/static/css/style.css")
    assert ".groq-popover" in css
    assert ".explain-popover" in css
    assert "max-height: 360px" in css
    assert "overflow-y: auto" in css
    assert "-webkit-overflow-scrolling: touch" in css
    assert "pointer-events: auto" in css


def test_js_popover_handlers_cover_hover_and_keyboard():
    app_js = _read("webapp/static/js/app.js")
    stock_js = _read("webapp/static/js/stock.js")
    risk_js = _read("webapp/static/js/risk.js")

    for js in (app_js, stock_js, risk_js):
        assert "mouseenter" in js
        assert "mouseleave" in js
        assert "focus" in js or "focusin" in js
        assert "keydown" in js


@pytest.mark.skipif(
    os.getenv("RUN_UI_E2E") != "1", reason="Set RUN_UI_E2E=1 to run browser E2E."
)
def test_playwright_popover_e2e_placeholder():
    pytest.importorskip("playwright.sync_api")
    pytest.skip(
        "Install Playwright browsers and run against a local server to execute full pointer/scroll E2E checks."
    )
