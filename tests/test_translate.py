import threading
from unittest.mock import patch

import pytest

from engine import translate as tr


# ── _resolve_key / using_openrouter ─────────────────────────────────────────

def test_resolve_key_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    assert tr._resolve_key("explicit-key") == "explicit-key"


def test_resolve_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    assert tr._resolve_key(None) == "env-key"
    assert tr._resolve_key("") == "env-key"
    assert tr._resolve_key("   ") == "env-key"


def test_resolve_key_empty_when_neither_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert tr._resolve_key(None) == ""


def test_resolve_key_strips_whitespace(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert tr._resolve_key("  key-with-spaces  ") == "key-with-spaces"


def test_using_openrouter_true_and_false(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert tr.using_openrouter("some-key") is True
    assert tr.using_openrouter(None) is False
    assert tr.using_openrouter("") is False


# ── translate_one dispatch ───────────────────────────────────────────────────

def test_translate_one_uses_openrouter_when_key_present(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("engine.translate.openrouter_backend.translate_one", return_value="translated") as mock_or, \
         patch("engine.translate.local_backend.translate_batch") as mock_local:
        result = tr.translate_one("source", "api-key", "model-x")
    assert result == "translated"
    mock_or.assert_called_once()
    mock_local.assert_not_called()


def test_translate_one_uses_local_backend_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("engine.translate.openrouter_backend.translate_one") as mock_or, \
         patch("engine.translate.local_backend.translate_batch", return_value=["local translation"]) as mock_local:
        result = tr.translate_one("source", None, "model-x")
    assert result == "local translation"
    mock_or.assert_not_called()
    mock_local.assert_called_once_with(["source"])


# ── translate_segments ───────────────────────────────────────────────────────

def _segments(*sources):
    return [{"source": s, "target": ""} for s in sources]


def test_translate_segments_skips_already_translated(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    segments = [{"source": "a", "target": ""}, {"source": "b", "target": "already done"}]
    with patch("engine.translate.local_backend.translate_batch", return_value=["A"]) as mock_local:
        result, errors = tr.translate_segments(segments, None, "model")
    assert result[0]["target"] == "A"
    assert result[1]["target"] == "already done"
    assert errors == []
    mock_local.assert_called_once_with(["a"])


def test_translate_segments_skips_empty_source():
    segments = [{"source": "", "target": ""}, {"source": "  ", "target": ""}]
    with patch("engine.translate.local_backend.translate_batch") as mock_local:
        result, errors = tr.translate_segments(segments, None, "model")
    mock_local.assert_not_called()
    assert errors == []


def test_translate_segments_local_backend_path(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    segments = _segments("a", "b")
    with patch("engine.translate.local_backend.translate_batch", return_value=["A", "B"]):
        result, errors = tr.translate_segments(segments, None, "model")
    assert [s["target"] for s in result] == ["A", "B"]
    assert errors == []


def test_translate_segments_local_backend_error_recorded(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    segments = _segments("a", "b")
    with patch("engine.translate.local_backend.translate_batch", side_effect=RuntimeError("model crashed")):
        result, errors = tr.translate_segments(segments, None, "model")
    assert len(errors) == 2
    assert "model crashed" in errors[0]
    assert all(s["target"] == "" for s in result)


def test_translate_segments_openrouter_batch_success():
    segments = _segments("a", "b")
    with patch("engine.translate.openrouter_backend.translate_batch", return_value=["A", "B"]) as mock_batch:
        result, errors = tr.translate_segments(segments, "key", "model")
    assert [s["target"] for s in result] == ["A", "B"]
    assert errors == []
    mock_batch.assert_called_once()


def test_translate_segments_openrouter_batch_falls_back_to_per_segment():
    segments = _segments("a", "b")
    with patch("engine.translate.openrouter_backend.translate_batch", return_value=None), \
         patch("engine.translate.openrouter_backend.translate_one", side_effect=["A", "B"]) as mock_one:
        result, errors = tr.translate_segments(segments, "key", "model")
    assert [s["target"] for s in result] == ["A", "B"]
    assert errors == []
    assert mock_one.call_count == 2


def test_translate_segments_per_segment_fallback_records_errors_and_continues():
    segments = _segments("a", "b")
    with patch("engine.translate.openrouter_backend.translate_batch", return_value=None), \
         patch("engine.translate.openrouter_backend.translate_one", side_effect=[RuntimeError("bad"), "B"]):
        result, errors = tr.translate_segments(segments, "key", "model")
    assert result[0]["target"] == ""  # failed segment left untouched
    assert result[1]["target"] == "B"
    assert len(errors) == 1
    assert "Segment 1" in errors[0]


def test_translate_segments_respects_stop_event_between_batches():
    stop = threading.Event()
    segments = _segments(*[f"seg{i}" for i in range(60)])  # 60 > _BATCH_SIZE(25), needs 3 batches

    call_count = {"n": 0}

    def fake_batch(texts, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            stop.set()
        return [t.upper() for t in texts]

    with patch("engine.translate.openrouter_backend.translate_batch", side_effect=fake_batch):
        result, errors = tr.translate_segments(segments, "key", "model", stop=stop)

    assert call_count["n"] == 1  # stopped before the second batch
    assert result[0]["target"] == "SEG0"
    assert result[59]["target"] == ""


def test_translate_segments_progress_callback_invoked_per_batch():
    segments = _segments(*[f"seg{i}" for i in range(30)])
    progress_calls = []

    with patch("engine.translate.openrouter_backend.translate_batch", side_effect=lambda texts, *a, **kw: texts):
        tr.translate_segments(
            segments, "key", "model",
            progress_callback=lambda i, total: progress_calls.append((i, total)),
        )

    assert progress_calls == [(0, 30), (25, 30)]


def test_translate_segments_passes_recent_translations_as_preceding_context():
    segments = _segments(*[f"seg{i}" for i in range(30)])
    seen_preceding = []

    def fake_batch(texts, api_key, model, template, preceding=None):
        seen_preceding.append(preceding)
        return texts

    with patch("engine.translate.openrouter_backend.translate_batch", side_effect=fake_batch):
        tr.translate_segments(segments, "key", "model")

    assert seen_preceding[0] is None  # nothing translated yet for the first batch
    assert seen_preceding[1] == ["seg22", "seg23", "seg24"]  # last 3 of the first batch
