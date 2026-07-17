import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import handlers as h

SAMPLES_DIR = Path(__file__).parent.parent / "samples"


def _segments(*sources):
    return [{"source": s, "target": ""} for s in sources]


# ── state / pagination ──────────────────────────────────────────────────────

def test_make_state_defaults():
    state = h._make_state()
    assert state == {"page": 0, "segments": [], "has_translation": False, "label": "output"}


def test_save_page_edits_writes_current_page_slots():
    state = h._make_state()
    state["segments"] = _segments("a", "b", "c")
    state["page"] = 0
    sources = ["A", "B", "C"] + [""] * (h.MAX_SLOTS - 3)
    targets = ["ta", "tb", "tc"] + [""] * (h.MAX_SLOTS - 3)
    state = h._save_page_edits(state, sources, targets)
    assert state["segments"] == [
        {"source": "A", "target": "ta"},
        {"source": "B", "target": "tb"},
        {"source": "C", "target": "tc"},
    ]


def test_save_page_edits_respects_page_offset():
    state = h._make_state()
    state["segments"] = _segments(*[f"s{i}" for i in range(h.MAX_SLOTS + 2)])
    state["page"] = 1
    sources = ["edited0", "edited1"] + [""] * (h.MAX_SLOTS - 2)
    targets = ["t0", "t1"] + [""] * (h.MAX_SLOTS - 2)
    state = h._save_page_edits(state, sources, targets)
    assert state["segments"][h.MAX_SLOTS]["source"] == "edited0"
    assert state["segments"][h.MAX_SLOTS + 1]["source"] == "edited1"
    assert state["segments"][0]["source"] == "s0"  # page 0 untouched


def test_load_page_empty_state_hides_everything():
    state = h._make_state()
    result = h._load_page(state)
    _, nav, prev, next_, *rest = result
    assert nav["value"] == ""
    assert prev["interactive"] is False
    assert next_["interactive"] is False
    groups = rest[:h.MAX_SLOTS]
    assert all(g["visible"] is False for g in groups)


def test_load_page_shows_correct_slot_count_and_nav_text():
    state = h._make_state()
    state["segments"] = _segments("a", "b", "c")
    result = h._load_page(state)
    _, nav, prev, next_, *rest = result
    assert "All 3 segments" in nav["value"]
    assert prev["interactive"] is False
    assert next_["interactive"] is False
    groups = rest[:h.MAX_SLOTS]
    sources = rest[h.MAX_SLOTS:2 * h.MAX_SLOTS]
    assert [g["visible"] for g in groups[:3]] == [True, True, True]
    assert all(g["visible"] is False for g in groups[3:])
    assert [s["value"] for s in sources[:3]] == ["a", "b", "c"]


def test_load_page_pagination_across_multiple_pages():
    total = h.MAX_SLOTS + 5
    state = h._make_state()
    state["segments"] = _segments(*[f"s{i}" for i in range(total)])
    state["page"] = 0

    result = h._load_page(state)
    _, nav, prev, next_, *_rest = result
    assert "Page 1 of 2" in nav["value"]
    assert prev["interactive"] is False
    assert next_["interactive"] is True

    state["page"] = 1
    result = h._load_page(state)
    new_state, nav, prev, next_, *rest = result
    assert "Page 2 of 2" in nav["value"]
    assert prev["interactive"] is True
    assert next_["interactive"] is False
    groups = rest[:h.MAX_SLOTS]
    assert [g["visible"] for g in groups[:5]] == [True] * 5
    assert all(g["visible"] is False for g in groups[5:])


def test_load_page_clamps_page_when_segments_shrank():
    state = h._make_state()
    state["segments"] = _segments("a")
    state["page"] = 5  # stale page index from before segments were replaced
    new_state, *_ = h._load_page(state)
    assert new_state["page"] == 0


def test_handle_prev_and_next_navigate_and_save(monkeypatch):
    total = h.MAX_SLOTS + 3
    state = h._make_state()
    state["segments"] = _segments(*[f"s{i}" for i in range(total)])
    state["page"] = 0
    slot_values = [""] * h.MAX_SLOTS + [""] * h.MAX_SLOTS

    result = h._handle_next(state, *slot_values)
    new_state = result[0]
    assert new_state["page"] == 1

    result = h._handle_prev(new_state, *slot_values)
    new_state = result[0]
    assert new_state["page"] == 0


def test_handle_next_does_not_advance_past_last_page():
    state = h._make_state()
    state["segments"] = _segments("a", "b")
    state["page"] = 0
    slot_values = [""] * h.MAX_SLOTS * 2
    result = h._handle_next(state, *slot_values)
    assert result[0]["page"] == 0


def test_handle_prev_does_not_go_below_zero():
    state = h._make_state()
    state["segments"] = _segments("a", "b")
    state["page"] = 0
    slot_values = [""] * h.MAX_SLOTS * 2
    result = h._handle_prev(state, *slot_values)
    assert result[0]["page"] == 0


# ── file loading ─────────────────────────────────────────────────────────

def test_load_source_none_hides_editor():
    state = h._make_state()
    result = h._load_source(None, state)
    assert result[-1]["visible"] is False


def test_load_source_txt_populates_segments():
    state = h._make_state()
    path = str(SAMPLES_DIR / "sample_tibetan.txt")
    result = h._load_source(path, state)
    new_state = result[0]
    assert len(new_state["segments"]) > 0
    assert new_state["label"] == "sample_tibetan"
    assert new_state["has_translation"] is False
    assert result[-1]["visible"] is True


def test_load_source_bad_file_hides_editor_and_does_not_raise(tmp_path):
    bad_path = tmp_path / "bad.rtf"
    bad_path.write_text("nope", encoding="utf-8")
    state = h._make_state()
    result = h._load_source(str(bad_path), state)
    assert result[-1]["visible"] is False
    assert state["segments"] == []


def test_load_resume_json_merges_targets_by_position():
    state = h._make_state()
    state["segments"] = _segments("a", "b", "c")
    resume_content = json.dumps([
        {"source": "a", "target": "A"},
        {"source": "b", "target": ""},
        {"source": "c", "target": "C"},
    ])
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(resume_content)
        path = f.name

    h._load_resume_json(path, state)
    assert [s["target"] for s in state["segments"]] == ["A", "", "C"]
    assert state["has_translation"] is True


def test_load_resume_json_no_file_or_no_segments_is_noop():
    state = h._make_state()
    result = h._load_resume_json(None, state)
    assert result[0] == state

    state2 = h._make_state()
    state2["segments"] = []
    result = h._load_resume_json("irrelevant.json", state2)
    assert result[0]["segments"] == []


# ── downloads ────────────────────────────────────────────────────────────

def test_make_downloads_writes_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(h.tempfile, "gettempdir", lambda: str(tmp_path))
    segments = [{"source": "src1", "target": "tgt1"}]
    paths = h._make_downloads(segments, "mylabel")
    assert paths["txt"] and Path(paths["txt"]).read_text(encoding="utf-8") == "tgt1\n"
    assert paths["docx"] and Path(paths["docx"]).exists()
    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8")) == segments
    assert paths["source_txt"] and Path(paths["source_txt"]).name == "mylabel_source.txt"


def test_make_downloads_omits_empty_content():
    segments = [{"source": "", "target": ""}]
    paths = h._make_downloads(segments, "mylabel")
    assert paths["txt"] is None
    assert paths["source_txt"] is None


def test_handle_download_click_lists_available_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(h.tempfile, "gettempdir", lambda: str(tmp_path))
    state = h._make_state()
    state["segments"] = [{"source": "src", "target": "tgt"}]
    # _handle_download_click saves current textbox contents before exporting, so the
    # slot values passed in must reflect the segment we just seeded above.
    sources = ["src"] + [""] * (h.MAX_SLOTS - 1)
    targets = ["tgt"] + [""] * (h.MAX_SLOTS - 1)
    slot_values = sources + targets
    new_state, checklist, get_files_btn, download_zip = h._handle_download_click(state, *slot_values)
    assert "Translation (TXT)" in checklist["choices"]
    assert "Translation (DOCX)" in checklist["choices"]
    assert "JSON (for resume)" in checklist["choices"]
    assert "Source text (TXT)" in checklist["choices"]
    assert get_files_btn["visible"] is True
    assert download_zip["visible"] is False


def test_handle_get_files_builds_zip_with_selected_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(h.tempfile, "gettempdir", lambda: str(tmp_path))
    state = h._make_state()
    state["segments"] = [{"source": "src", "target": "tgt"}]
    state["label"] = "lbl"

    zip_update, checklist_update, btn_update = h._handle_get_files(state, ["Translation (TXT)"])
    assert zip_update["visible"] is True
    zip_path = Path(zip_update["value"])
    assert zip_path.exists()

    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["lbl.txt"]


def test_handle_get_files_no_selection_hides_download():
    state = h._make_state()
    state["segments"] = [{"source": "src", "target": "tgt"}]
    zip_update, checklist_update, btn_update = h._handle_get_files(state, [])
    assert zip_update["visible"] is False


def test_handle_save_persists_edits_and_shows_status():
    state = h._make_state()
    state["segments"] = _segments("a")
    slot_values = ["edited"] + [""] * (h.MAX_SLOTS - 1) + ["translated"] + [""] * (h.MAX_SLOTS - 1)
    new_state, status = h._handle_save(state, *slot_values)
    assert new_state["segments"][0] == {"source": "edited", "target": "translated"}
    assert status["value"] == "Saved."


# ── translation handlers ──────────────────────────────────────────────────

def test_translate_one_updates_state_and_returns_text():
    state = h._make_state()
    state["segments"] = _segments("source text")
    state["page"] = 0
    with patch.object(h, "_engine_translate_one", return_value="translated!") as mock_translate:
        new_state, update = h._translate_one(state, 0, "source text", "api-key", "model-x")
    assert update["value"] == "translated!"
    assert new_state["segments"][0]["target"] == "translated!"
    mock_translate.assert_called_once_with("source text", "api-key", "model-x")


def test_translate_one_records_error_in_target():
    state = h._make_state()
    state["segments"] = _segments("source text")
    with patch.object(h, "_engine_translate_one", side_effect=RuntimeError("network down")):
        new_state, update = h._translate_one(state, 0, "source text", "api-key", "model-x")
    assert "Translation error" in update["value"]
    assert "network down" in new_state["segments"][0]["target"]


def test_translate_one_out_of_range_slot_does_not_crash():
    state = h._make_state()
    state["segments"] = _segments("only one")
    state["page"] = 0
    with patch.object(h, "_engine_translate_one", return_value="x"):
        new_state, update = h._translate_one(state, 5, "whatever", None, "model")
    assert update["value"] == "x"
    assert new_state["segments"] == _segments("only one")  # untouched, index out of range


def test_translate_all_reports_completion_status():
    state = h._make_state()
    state["segments"] = _segments("a", "b")
    # _translate_all saves current textbox contents before translating, so the slot
    # values passed in must reflect the segments we just seeded above.
    sources = ["a", "b"] + [""] * (h.MAX_SLOTS - 2)
    targets = [""] * h.MAX_SLOTS
    slot_values = sources + targets

    def fake_translate_segments(segments, api_key, model, progress_callback=None, stop=None):
        for seg in segments:
            seg["target"] = seg["source"].upper()
        return segments, []

    with patch.object(h, "translate_segments", side_effect=fake_translate_segments), \
         patch.object(h, "using_openrouter", return_value=True):
        results = list(h._translate_all(state, "api-key", "model-x", *slot_values))

    final = results[-1]
    status = final[-1]
    assert "Translated 2 segment(s) via OpenRouter." in status["value"]
    final_state = final[0]
    assert [s["target"] for s in final_state["segments"]] == ["A", "B"]


def test_translate_all_reports_errors_in_status():
    state = h._make_state()
    state["segments"] = _segments("a", "b")
    slot_values = [""] * h.MAX_SLOTS * 2

    def fake_translate_segments(segments, api_key, model, progress_callback=None, stop=None):
        return segments, ["Segment 1: boom"]

    with patch.object(h, "translate_segments", side_effect=fake_translate_segments), \
         patch.object(h, "using_openrouter", return_value=False):
        results = list(h._translate_all(state, None, "model-x", *slot_values))

    status = results[-1][-1]
    assert "local CPU model" in status["value"]
    assert "1 error(s)" in status["value"]


def test_translate_all_cancel_sets_stop_event():
    state = h._make_state()
    state["segments"] = _segments(*[f"s{i}" for i in range(3)])
    slot_values = [""] * h.MAX_SLOTS * 2

    stop_seen = {}

    def fake_translate_segments(segments, api_key, model, progress_callback=None, stop=None):
        stop_seen["stop"] = stop
        time.sleep(0.05)
        return segments, []

    with patch.object(h, "translate_segments", side_effect=fake_translate_segments):
        gen = h._translate_all(state, "key", "model", *slot_values)
        next(gen)  # drive it far enough to start the background thread
        for _ in gen:
            pass

    assert stop_seen["stop"] is not None
    assert not stop_seen["stop"].is_set()  # ran to completion, was never told to stop


# ── prompt management ──────────────────────────────────────────────────────

def test_read_save_reset_prompt_roundtrip(monkeypatch, tmp_path):
    import engine.prompt as prompt_module
    prompt_path = tmp_path / "translation_prompt.txt"
    prompt_path.write_text("default", encoding="utf-8")
    monkeypatch.setattr(prompt_module, "PROMPT_PATH", prompt_path)
    monkeypatch.setattr(prompt_module, "_DEFAULT_PROMPT", "default")

    assert h._read_prompt() == "default"

    status = h._save_prompt("edited")
    assert status["value"] == "Prompt saved."
    assert h._read_prompt() == "edited"

    box_update, status_update = h._reset_prompt()
    assert box_update["value"] == "default"
    assert status_update["value"] == "Prompt reset to default."
