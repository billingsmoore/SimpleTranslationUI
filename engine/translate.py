"""Backend dispatcher: OpenRouter if an API key is available (user-supplied or
OPENROUTER_API_KEY env var), otherwise the local CPU model. Callers don't need
to know which backend actually ran."""

import os

from . import local_backend
from . import openrouter_backend
from .prompt import read_prompt

CURATED_MODELS = openrouter_backend.CURATED_MODELS
DEFAULT_MODEL = openrouter_backend.DEFAULT_MODEL
list_model_choices = openrouter_backend.list_model_choices
_BATCH_SIZE = openrouter_backend._BATCH_SIZE


def _resolve_key(api_key: str | None) -> str:
    return (api_key or "").strip() or os.environ.get("OPENROUTER_API_KEY", "")


def using_openrouter(api_key: str | None) -> bool:
    return bool(_resolve_key(api_key))


def translate_one(text: str, api_key: str | None, model: str) -> str:
    key = _resolve_key(api_key)
    if key:
        return openrouter_backend.translate_one(text, key, model, read_prompt())
    return local_backend.translate_batch([text])[0]


def translate_segments(
    segments: list[dict], api_key: str | None, model: str,
    progress_callback=None, stop=None,
) -> tuple[list[dict], list[str]]:
    """Translate all segments with empty targets. Returns (segments, errors)."""
    key = _resolve_key(api_key)
    template = read_prompt()
    errors = []

    translatable = [
        (i, seg.get("source", "").strip())
        for i, seg in enumerate(segments)
        if not seg.get("target", "").strip() and seg.get("source", "").strip()
    ]
    total = len(translatable)
    print(f"[INFO] Translating {total} segment(s) via {'OpenRouter' if key else 'local CPU model'}.")

    if not key:
        # No GPU to exploit here, so there's no throughput win from batching on CPU
        # (a single sequence's matmuls already use all available threads) — batching
        # would only add padding waste and delay every result in a chunk until the
        # slowest one finishes. One at a time also means the UI can show each
        # translation the moment it's ready instead of waiting on a whole chunk.
        for i, (idx, text) in enumerate(translatable):
            if stop is not None and stop.is_set():
                break
            if progress_callback:
                progress_callback(i, total)
            try:
                segments[idx]["target"] = local_backend.translate_batch([text])[0]
            except Exception as e:
                print(f"[WARN] Local translation failed for segment {idx + 1}: {e}")
                errors.append(f"Segment {idx + 1}: {e}")
        return segments, errors

    recent: list[str] = []
    for batch_start in range(0, total, _BATCH_SIZE):
        if stop is not None and stop.is_set():
            break
        batch = translatable[batch_start:batch_start + _BATCH_SIZE]
        indices = [idx for idx, _ in batch]
        texts = [txt for _, txt in batch]

        if progress_callback:
            progress_callback(batch_start, total)

        preceding = recent[-3:] if recent else None
        translations = openrouter_backend.translate_batch(texts, key, model, template, preceding=preceding)
        if translations is None:
            translations = []
            for idx, text in zip(indices, texts):
                if stop is not None and stop.is_set():
                    break
                try:
                    t = openrouter_backend.translate_one(text, key, model, template)
                except Exception as e:
                    print(f"[WARN] Segment {idx + 1} failed: {e}")
                    errors.append(f"Segment {idx + 1}: {e}")
                    t = ""
                translations.append(t)
                recent.append(t)
            for idx, t in zip(indices, translations):
                if t:
                    segments[idx]["target"] = t
            continue

        for idx, t in zip(indices, translations):
            segments[idx]["target"] = t
        recent.extend(translations)

    return segments, errors
