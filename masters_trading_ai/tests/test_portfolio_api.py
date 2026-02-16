from pathlib import Path

from webapp import server


def test_portfolio_add_merge_and_fetch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
    monkeypatch.setattr(server, "ticker_names", {"ABC.NS": "ABC"})

    with server.app.test_client() as client:
        r1 = client.post("/api/portfolio", json={"ticker": "ABC.NS", "quantity": 2, "entry_price": 100})
        assert r1.status_code == 200
        p1 = r1.get_json()
        assert p1["count"] == 1
        assert p1["holdings"][0]["quantity"] == 2
        assert p1["holdings"][0]["entry_price"] == 100

        # Merge with weighted average entry
        r2 = client.post("/api/portfolio", json={"ticker": "ABC.NS", "quantity": 1, "entry_price": 130})
        assert r2.status_code == 200
        p2 = r2.get_json()
        assert p2["count"] == 1
        assert p2["holdings"][0]["quantity"] == 3
        assert p2["holdings"][0]["entry_price"] == 110.0

        r3 = client.get("/api/portfolio?ticker=ABC.NS")
        assert r3.status_code == 200
        p3 = r3.get_json()
        assert p3["count"] == 1
        assert p3["holdings"][0]["ticker"] == "ABC.NS"


def test_portfolio_delete(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "PORTFOLIO_FILE", tmp_path / "portfolio.json")

    with server.app.test_client() as client:
        client.post("/api/portfolio", json={"ticker": "A.NS", "quantity": 1, "entry_price": 10})
        client.post("/api/portfolio", json={"ticker": "B.NS", "quantity": 1, "entry_price": 20})
        out = client.delete("/api/portfolio?ticker=A.NS")
        assert out.status_code == 200
        payload = out.get_json()
        tickers = {h["ticker"] for h in payload["holdings"]}
        assert tickers == {"B.NS"}
