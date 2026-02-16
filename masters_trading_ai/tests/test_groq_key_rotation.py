import time
from types import SimpleNamespace

from webapp import groq_explainer as ge


def _reset_groq_state():
    ge._groq_keys = ["key-1", "key-2"]
    ge._active_key_index = 0
    ge._key_cooldown_until = {}
    ge._key_last_429_at = {}
    ge._degraded_until = 0.0
    ge._degraded_reason = ""
    ge._last_429_at = 0.0
    ge._last_error = ""
    ge._last_success_at = 0.0
    ge._last_call_time = 0.0
    ge.GROQ_KEY_ROTATION_ENABLED = True
    ge.MIN_CALL_INTERVAL = 0.0


def _fake_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_groq_rotates_to_next_key_on_429(monkeypatch):
    _reset_groq_state()

    monkeypatch.setattr(ge, "_get_cached", lambda _key: None)
    monkeypatch.setattr(ge, "_set_cache", lambda _key, _text: None)
    monkeypatch.setattr(ge, "_reserve_rate_slot", lambda *_args, **_kwargs: True)

    def _client_for_key(api_key: str):
        class _Client:
            def __init__(self, key: str):
                self.key = key
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                if self.key == "key-1":
                    raise RuntimeError("429 rate limit")
                return _fake_response("ok-from-key-2")

        return _Client(api_key)

    monkeypatch.setattr(ge, "_get_client_for_key", _client_for_key)

    out = ge._call_groq("rotation-test")
    assert out == "ok-from-key-2"
    assert ge._key_cooldown_until.get(0, 0) > time.time()
    status = ge.get_groq_system_status()
    assert status["degraded_mode"] is False
    assert status["key_pool_size"] == 2


def test_groq_enters_degraded_mode_when_all_keys_429(monkeypatch):
    _reset_groq_state()

    monkeypatch.setattr(ge, "_get_cached", lambda _key: None)
    monkeypatch.setattr(ge, "_set_cache", lambda _key, _text: None)
    monkeypatch.setattr(ge, "_reserve_rate_slot", lambda *_args, **_kwargs: True)

    def _client_for_key(_api_key: str):
        class _Client:
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            @staticmethod
            def _create(**_kwargs):
                raise RuntimeError("429 token quota exceeded")

        return _Client()

    monkeypatch.setattr(ge, "_get_client_for_key", _client_for_key)

    out = ge._call_groq("all-429-test")
    assert "rate-limited" in out.lower()
    status = ge.get_groq_system_status()
    assert status["degraded_mode"] is True
    assert status["degraded_reason"] in {"upstream_429", "all_keys_cooling_down"}
