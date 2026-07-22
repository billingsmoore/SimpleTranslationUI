"""Unit tests for engine.usage_store — usage telemetry written to the
billingsmoore/stui-usage HuggingFace dataset (private). All huggingface_hub
calls are mocked; no test in this file talks to the network.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from engine import usage_store


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")


class TestNoToken:
    def test_log_event_returns_false_without_network_call(self):
        with patch("huggingface_hub.HfApi") as mock_api:
            assert usage_store.log_event("upload", "session-1", {"foo": "bar"}) is False
            mock_api.assert_not_called()


class TestLogEvent:
    def test_no_session_id_returns_false_without_network_call(self, with_token):
        with patch("huggingface_hub.HfApi") as mock_api:
            assert usage_store.log_event("upload", "", {"foo": "bar"}) is False
            mock_api.assert_not_called()

    def test_writes_single_commit_with_expected_fields(self, with_token):
        mock_api_instance = MagicMock()
        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            ok = usage_store.log_event("translate", "session-1", {"backend": "local_cpu", "segment_count": 2})

        assert ok is True
        mock_api_instance.create_repo.assert_called_once_with(
            repo_id=usage_store.DATASET_REPO_ID, repo_type="dataset", private=True, exist_ok=True,
        )
        mock_api_instance.create_commit.assert_called_once()
        _, kwargs = mock_api_instance.create_commit.call_args
        assert kwargs["repo_id"] == usage_store.DATASET_REPO_ID
        assert kwargs["repo_type"] == "dataset"

        operations = kwargs["operations"]
        assert len(operations) == 1
        op = operations[0]
        assert op.path_in_repo.startswith("events/session-1/")
        assert op.path_in_repo.endswith(".json")

        payload = json.loads(op.path_or_fileobj.decode("utf-8"))
        assert payload["event"] == "translate"
        assert payload["session_id"] == "session-1"
        assert payload["backend"] == "local_cpu"
        assert payload["segment_count"] == 2
        assert "timestamp" in payload

    def test_network_error_fails_open(self, with_token):
        mock_api_instance = MagicMock()
        mock_api_instance.create_commit.side_effect = Exception("network error")
        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            assert usage_store.log_event("edit", "session-1", {}) is False
