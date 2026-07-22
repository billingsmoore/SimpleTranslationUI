"""UI event handlers and state management for SimpleTranslationUI."""

import math
import os
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

import gradio as gr

from engine import (
    export_to_docx,
    export_to_json,
    export_to_txt,
    load_source_file,
    log_event,
    parse_json,
    read_prompt,
    reset_prompt,
    save_prompt,
    translate_one as _engine_translate_one,
    translate_segments,
    using_openrouter,
)

MAX_SLOTS = 25


# ── State ─────────────────────────────────────────────────────────────────────

def _make_state() -> dict:
    return {
        "page": 0,
        "segments": [],
        "has_translation": False,
        "label": "output",
        "session_id": uuid.uuid4().hex,
        "usage_tracking_enabled": True,
    }


def _log(state: dict, event_type: str, payload: dict) -> None:
    if not state.get("usage_tracking_enabled", True):
        return
    log_event(event_type, state.get("session_id", ""), payload)


def _set_usage_tracking(state: dict, enabled: bool):
    state["usage_tracking_enabled"] = enabled
    return state


def _save_page_edits(state: dict, sources: list, targets: list) -> dict:
    segments = state["segments"]
    start = state["page"] * MAX_SLOTS
    for i in range(MAX_SLOTS):
        idx = start + i
        if idx >= len(segments):
            break
        segments[idx]["source"] = sources[i] or ""
        segments[idx]["target"] = targets[i] or ""
    state["segments"] = segments
    return state


# ── Page rendering ────────────────────────────────────────────────────────────

def _load_page(state: dict) -> tuple:
    segments = state["segments"]
    total = len(segments)
    page = state["page"]

    if total == 0:
        return (
            state,
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            *[gr.update(visible=False)] * MAX_SLOTS,
            *[gr.update(value="")] * MAX_SLOTS,
            *[gr.update(value="")] * MAX_SLOTS,
        )

    total_pages = math.ceil(total / MAX_SLOTS)
    state["page"] = min(page, total_pages - 1)
    start = state["page"] * MAX_SLOTS
    end = min(start + MAX_SLOTS, total)
    n = end - start

    if total_pages > 1:
        nav = f"<p>Page {state['page'] + 1} of {total_pages} · Segments {start + 1}–{end} of {total}</p>"
    else:
        nav = f"<p>All {total} segments</p>"

    groups, sources, targets = [], [], []
    for i in range(MAX_SLOTS):
        if i < n:
            seg = segments[start + i]
            groups.append(gr.update(visible=True))
            sources.append(gr.update(value=seg.get("source", "")))
            targets.append(gr.update(value=seg.get("target", "")))
        else:
            groups.append(gr.update(visible=False))
            sources.append(gr.update(value=""))
            targets.append(gr.update(value=""))

    return (
        state,
        gr.update(value=nav),
        gr.update(interactive=(state["page"] > 0)),
        gr.update(interactive=(state["page"] < total_pages - 1)),
        *groups, *sources, *targets,
    )


# ── File loading ──────────────────────────────────────────────────────────────

def _load_source(source_file, state):
    if source_file is None:
        return (*_load_page(state), gr.update(visible=False))

    path = source_file if isinstance(source_file, str) else source_file.name
    label = Path(path).stem

    try:
        segments = load_source_file(path)
    except Exception as e:
        print(f"[WARN] Failed to load source file: {e}")
        return (*_load_page(state), gr.update(visible=False))

    state["segments"] = segments
    state["page"] = 0
    state["label"] = label
    state["has_translation"] = False

    _log(state, "upload", {
        "filename": Path(path).name,
        "file_type": Path(path).suffix.lstrip("."),
        "segment_count": len(segments),
        "segments": [{"source": seg.get("source", "")} for seg in segments],
    })

    return (*_load_page(state), gr.update(visible=True))


def _load_resume_json(resume_file, state):
    """Merge translations from a previously-downloaded JSON export back in,
    matched by segment position (assumes the same source file/segmentation)."""
    if resume_file is None or not state.get("segments"):
        return _load_page(state)

    path = resume_file if isinstance(resume_file, str) else resume_file.name
    with open(path, "rb") as f:
        content = f.read()

    try:
        resumed = parse_json(content)
    except Exception as e:
        print(f"[WARN] Failed to parse resume JSON: {e}")
        return _load_page(state)

    for seg, r in zip(state["segments"], resumed):
        if r.get("target"):
            seg["target"] = r["target"]

    state["has_translation"] = any(seg.get("target") for seg in state["segments"])
    state["page"] = 0

    _log(state, "resume", {
        "filename": Path(path).name,
        "segment_count": len(state["segments"]),
        "segments": [
            {"source": seg.get("source", ""), "target": seg.get("target", "")}
            for seg in state["segments"]
        ],
    })

    return _load_page(state)


# ── Downloads ─────────────────────────────────────────────────────────────────

def _make_downloads(segments: list[dict], label: str) -> dict:
    tmp = tempfile.gettempdir()

    def _write_text(content, ext):
        if not content.strip():
            return None
        path = os.path.join(tmp, f"{label}.{ext}")
        Path(path).write_text(content, encoding="utf-8")
        return path

    def _write_bytes(content, ext):
        if not content:
            return None
        path = os.path.join(tmp, f"{label}.{ext}")
        Path(path).write_bytes(content)
        return path

    def _write_named(content, filename):
        if not content.strip():
            return None
        path = os.path.join(tmp, filename)
        Path(path).write_text(content, encoding="utf-8")
        return path

    return {
        "txt": _write_text(export_to_txt(segments, "target"), "txt"),
        "docx": _write_bytes(export_to_docx(segments, "target"), "docx"),
        "json": _write_text(export_to_json(segments), "json"),
        "source_txt": _write_named(export_to_txt(segments, "source"), f"{label}_source.txt"),
    }


_FORMAT_META = [
    ("txt", "Translation (TXT)"),
    ("docx", "Translation (DOCX)"),
    ("json", "JSON (for resume)"),
    ("source_txt", "Source text (TXT)"),
]


def _handle_save(state, *slot_values):
    sources = list(slot_values[:MAX_SLOTS])
    targets = list(slot_values[MAX_SLOTS:])
    state = _save_page_edits(state, sources, targets)

    n = min(MAX_SLOTS, len(state["segments"]) - state["page"] * MAX_SLOTS)
    _log(state, "edit", {
        "segment_count": max(n, 0),
        "segments": [{"source": sources[i], "target": targets[i]} for i in range(max(n, 0))],
    })

    return state, gr.update(value="Saved.", visible=True)


def _handle_download_click(state, *slot_values):
    sources = list(slot_values[:MAX_SLOTS])
    targets = list(slot_values[MAX_SLOTS:])
    state = _save_page_edits(state, sources, targets)

    paths = _make_downloads(state["segments"], state.get("label", "output"))
    available = [label for key, label in _FORMAT_META if paths.get(key)]

    return (
        state,
        gr.update(choices=available, value=available, visible=True),
        gr.update(visible=True),
        gr.update(value=None, visible=False),
    )


def _handle_get_files(state, selected_labels):
    segments = state["segments"]
    label = state.get("label", "output")
    paths = _make_downloads(segments, label)
    path_by_label = {fmt_label: paths[key] for key, fmt_label in _FORMAT_META}

    chosen = [path_by_label[l] for l in (selected_labels or []) if path_by_label.get(l)]
    if not chosen:
        return gr.update(value=None, visible=False), gr.update(visible=False), gr.update(visible=False)

    zip_path = os.path.join(tempfile.gettempdir(), f"{label}_export.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in chosen:
            zf.write(path, arcname=os.path.basename(path))

    _log(state, "download", {
        "formats": list(selected_labels or []),
        "segment_count": len(segments),
        "segments": [
            {"source": seg.get("source", ""), "target": seg.get("target", "")}
            for seg in segments
        ],
    })

    return gr.update(value=zip_path, visible=True), gr.update(visible=False), gr.update(visible=False)


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_one(state: dict, slot_idx: int, source: str, api_key: str, model: str):
    try:
        text = _engine_translate_one(source, api_key, model)
    except Exception as e:
        text = f"[Translation error: {e}]"

    actual_idx = state["page"] * MAX_SLOTS + slot_idx
    segments = state["segments"]
    if actual_idx < len(segments):
        segments[actual_idx]["target"] = text
        state["segments"] = segments

    is_openrouter = using_openrouter(api_key)
    _log(state, "translate", {
        "backend": "openrouter" if is_openrouter else "local_cpu",
        "model": model if is_openrouter else "billingsmoore/mlotsawa-ground-base",
        "prompt": read_prompt() if is_openrouter else None,
        "segment_count": 1,
        "segments": [{"source": source, "target": text}],
    })

    return state, gr.update(value=text)


def _translate_all(state, api_key, model, *slot_values):
    sources = list(slot_values[:MAX_SLOTS])
    targets = list(slot_values[MAX_SLOTS:])
    state = _save_page_edits(state, sources, targets)

    is_openrouter = using_openrouter(api_key)
    backend = "OpenRouter" if is_openrouter else "local CPU model (mlotsawa-ground-base)"

    stop = threading.Event()
    result = [None, None]
    progress = [0, 0]

    def _run():
        def _on_progress(i, total):
            progress[0] = i
            progress[1] = total
        s, e = translate_segments(
            state["segments"], api_key, model,
            progress_callback=_on_progress, stop=stop,
        )
        result[0] = s
        result[1] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
            if thread.is_alive():
                i, total = progress
                prog = f" ({i + 1}/{total})" if total > 0 else ""
                yield (*_load_page(state), gr.update(value=f"Translating via {backend}…{prog}", visible=True))
    except GeneratorExit:
        stop.set()
        raise

    segments = result[0] if result[0] is not None else state["segments"]
    errors = result[1] or []
    state["segments"] = segments
    state["has_translation"] = True
    state["page"] = 0

    count = len(segments) - len(errors)
    status = f"Translated {count} segment(s) via {backend}."
    if errors:
        status += f" {len(errors)} error(s)."

    _log(state, "translate", {
        "backend": "openrouter" if is_openrouter else "local_cpu",
        "model": model if is_openrouter else "billingsmoore/mlotsawa-ground-base",
        "prompt": read_prompt() if is_openrouter else None,
        "segment_count": count,
        "segments": [{"source": seg.get("source", ""), "target": seg.get("target", "")} for seg in segments],
    })

    yield (*_load_page(state), gr.update(value=status, visible=True))


def _handle_cancel_translate(state: dict):
    _log(state, "translate_cancel", {})
    return state


# ── Prompt management ─────────────────────────────────────────────────────────

def _read_prompt() -> str:
    return read_prompt()


def _save_prompt(state: dict, text: str):
    save_prompt(text)
    _log(state, "prompt_save", {"prompt": text})
    return state, gr.update(value="Prompt saved.")


def _reset_prompt(state: dict):
    text = reset_prompt()
    _log(state, "prompt_reset", {})
    return state, gr.update(value=text), gr.update(value="Prompt reset to default.")


# ── Navigation ────────────────────────────────────────────────────────────────

def _handle_prev(state, *slot_values):
    sources = list(slot_values[:MAX_SLOTS])
    targets = list(slot_values[MAX_SLOTS:])
    state = _save_page_edits(state, sources, targets)
    if state["page"] > 0:
        state["page"] -= 1
    _log(state, "navigate", {"direction": "prev", "page": state["page"]})
    return _load_page(state)


def _handle_next(state, *slot_values):
    sources = list(slot_values[:MAX_SLOTS])
    targets = list(slot_values[MAX_SLOTS:])
    state = _save_page_edits(state, sources, targets)
    total = len(state["segments"])
    total_pages = math.ceil(total / MAX_SLOTS) if total else 1
    if state["page"] < total_pages - 1:
        state["page"] += 1
    _log(state, "navigate", {"direction": "next", "page": state["page"]})
    return _load_page(state)
