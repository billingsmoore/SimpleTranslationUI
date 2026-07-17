"""OpenRouter translation backend. Used only when an OpenRouter API key is
available (user-supplied or OPENROUTER_API_KEY env var) — otherwise
engine.translate falls back to the local CPU model."""

import re
import time

import requests

BASE_URL = "https://openrouter.ai/api/v1"

# Small hand-picked set of models known to work well for this kind of
# instruction-following translation task. These are pinned to the top of the
# UI dropdown (ahead of the full live-fetched catalog) and also used as
# OpenRouter's server-side model fallback chain for a given request.
CURATED_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]

DEFAULT_MODEL = CURATED_MODELS[0]

_TIMEOUT = 120
_MODELS_TIMEOUT = 10
_BACKOFF_BASE = 2
# Kept small (rather than e.g. one batch per page) so the UI can show translated
# segments as each batch finishes instead of the whole page appearing at once.
_BATCH_SIZE = 5

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _is_retryable(exc) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "rate limit", "timed out", "timeout", "connection"))


def fetch_available_models(timeout: int = _MODELS_TIMEOUT) -> list[dict]:
    """Raw model list from OpenRouter's public /models endpoint (no key required)."""
    resp = requests.get(f"{BASE_URL}/models", timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _is_text_model(model: dict) -> bool:
    arch = model.get("architecture") or {}
    modality = arch.get("modality") or ""
    output_modalities = arch.get("output_modalities") or []
    return modality.endswith("->text") or "text" in output_modalities


def list_model_choices(timeout: int = _MODELS_TIMEOUT) -> list[str]:
    """Curated models first (in preferred order), then the rest of OpenRouter's
    text-output catalog alphabetically. Falls back to just the curated list if
    the live fetch fails (offline, OpenRouter down, etc.)."""
    try:
        models = fetch_available_models(timeout=timeout)
    except Exception as e:
        print(f"[WARN] Could not fetch OpenRouter model list, using curated defaults only: {e}")
        return list(CURATED_MODELS)

    ids = {m["id"] for m in models if _is_text_model(m)}
    curated_present = [m for m in CURATED_MODELS if m in ids]
    rest = sorted(ids.difference(curated_present))
    return curated_present + rest


def _call_openrouter(api_key: str, model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # OpenRouter tries "models" in order server-side if the primary errors out,
    # so we don't need to re-implement model-swapping client-side.
    fallback_models = [model] + [m for m in CURATED_MODELS if m != model]
    body = {
        "models": fallback_models,
        "messages": [{"role": "user", "content": prompt}],
    }

    last_exc = None
    for attempt in range(4):
        try:
            resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=body, timeout=_TIMEOUT)
        except requests.RequestException as e:
            last_exc = e
            print(f"[WARN] OpenRouter request attempt {attempt + 1}/4 failed: {e}")
        else:
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            last_exc = RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code not in _RETRYABLE_STATUS:
                raise last_exc
            print(f"[WARN] OpenRouter attempt {attempt + 1}/4 failed: {last_exc}")
        if attempt < 3:
            time.sleep(_BACKOFF_BASE ** attempt)
    raise last_exc


def _build_batch_prompt(template: str, texts: list[str], preceding: list[str] | None = None) -> str:
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
    context_block = ""
    if preceding:
        context_block = (
            "\n\nThe following lines were translated immediately before this batch. "
            "Do not begin any line in this batch with the same opening word as any of these:\n"
            + "\n".join(f"- {c}" for c in preceding) + "\n"
        )
    return (
        template.rstrip()
        + context_block
        + "\n\nTranslate each of the following numbered source lines into English, "
          "following all instructions above. Output ONLY a numbered list, one line per "
          'input, format "<number>. <translation>". No other text.\n\n'
          "--- SOURCE LINES TO TRANSLATE ---\n"
        + numbered
    )


def _parse_batch_response(response_text: str, expected: int) -> list[str] | None:
    results = []
    for line in response_text.strip().splitlines():
        m = re.match(r"^\d+\.\s*(.*)", line.strip())
        if m:
            results.append(m.group(1).strip())
    return results if len(results) == expected else None


def translate_one(text: str, api_key: str, model: str, template: str) -> str:
    prompt = (
        template.rstrip()
        + f"\n\nTranslate the following into English, following all instructions "
          f"above. Output ONLY the translation, no other text.\n\n{text}"
    )
    return _call_openrouter(api_key, model, prompt).strip()


def translate_batch(
    texts: list[str], api_key: str, model: str, template: str, preceding: list[str] | None = None,
) -> list[str] | None:
    if not texts:
        return []
    prompt = _build_batch_prompt(template, texts, preceding=preceding)
    try:
        content = _call_openrouter(api_key, model, prompt)
        return _parse_batch_response(content.strip(), len(texts))
    except Exception as e:
        print(f"[WARN] OpenRouter batch translation failed: {e}")
        return None
