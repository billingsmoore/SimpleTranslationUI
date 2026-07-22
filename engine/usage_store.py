"""Usage telemetry, written as one event per file to the billingsmoore/stui-usage
HuggingFace dataset (private). Tracks uploads, translations, and edits so we can
study how the app is actually used; users can opt out per-session in Settings.

Each event is an independent commit (no shared index to read-modify-write),
so concurrent sessions never race each other. Fails open (returns False /
no-op and logs a warning) if HF_TOKEN isn't set, the dataset repo doesn't
exist yet, or the network call fails — usage logging must never break the
translation workflow.
"""

import json
import os
import uuid
from datetime import datetime, timezone

DATASET_REPO_ID = "billingsmoore/stui-usage"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def _ensure_dataset(token: str) -> None:
    from huggingface_hub import HfApi
    HfApi(token=token).create_repo(
        repo_id=DATASET_REPO_ID, repo_type="dataset", private=True, exist_ok=True,
    )


def log_event(event_type: str, session_id: str, payload: dict) -> bool:
    """Write one usage event. Returns True on success, False if skipped/failed."""
    token = _hf_token()
    if not token or not session_id:
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        _ensure_dataset(token)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = {
            "event": event_type,
            "session_id": session_id,
            "timestamp": now,
            **payload,
        }
        path_in_repo = f"events/{session_id}/{now}_{event_type}_{uuid.uuid4().hex[:8]}.json"

        HfApi(token=token).create_commit(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                ),
            ],
            commit_message=f"{event_type} event ({session_id})",
        )
        return True
    except Exception as e:
        print(f"[WARN] Could not log usage event {event_type!r}: {e}")
        return False
