"""Gemini translation backend. Used only when a Gemini API key is available
(user-supplied or GEMINI_API_KEY env var) — otherwise engine.translate falls
back to the local CPU model."""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

_GEMINI_TIMEOUT = 120
_BACKOFF_BASE = 2
_BATCH_SIZE = 25


def _is_retryable(exc) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "rate limit", "quota", "503", "500", "unavailable"))


def _call_gemini(client, model: str, prompt: str):
    from google.genai import types
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    queue = [model] + [m for m in FALLBACK_CHAIN if m != model]
    last_exc = None
    for candidate in queue:
        if candidate != model:
            print(f"[WARN] Gemini model '{model}' failed. Trying '{candidate}'.")
        for attempt in range(4):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        client.models.generate_content,
                        model=candidate, contents=prompt, config=config,
                    )
                    return future.result(timeout=_GEMINI_TIMEOUT)
            except _FuturesTimeout:
                last_exc = TimeoutError(f"Gemini '{candidate}' timed out after {_GEMINI_TIMEOUT}s")
                print(f"[WARN] {last_exc} (attempt {attempt + 1}/4)")
            except Exception as e:
                last_exc = e
                if not _is_retryable(e):
                    break
                print(f"[WARN] Gemini '{candidate}' attempt {attempt + 1}/4 failed: {e}")
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
    from google import genai
    client = genai.Client(api_key=api_key)
    prompt = (
        template.rstrip()
        + f"\n\nTranslate the following into English, following all instructions "
          f"above. Output ONLY the translation, no other text.\n\n{text}"
    )
    response = _call_gemini(client, model, prompt)
    return response.text.strip()


def translate_batch(
    texts: list[str], api_key: str, model: str, template: str, preceding: list[str] | None = None,
) -> list[str] | None:
    if not texts:
        return []
    from google import genai
    client = genai.Client(api_key=api_key)
    prompt = _build_batch_prompt(template, texts, preceding=preceding)
    try:
        response = _call_gemini(client, model, prompt)
        return _parse_batch_response(response.text.strip(), len(texts))
    except Exception as e:
        print(f"[WARN] Gemini batch translation failed: {e}")
        return None
