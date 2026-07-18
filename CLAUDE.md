# CLAUDE.md

This file guides Claude Code (or any future session) working in this
repository. This project is a **standalone, simplified fork** of
`TranslationUI` (from the `Garchen` monorepo) — it is intentionally
detached: no submodule relationship, no shared git history, no HuggingFace
dataset dependencies. Everything needed to understand and run it lives here.

## What this app is

A minimal Gradio app for translating Tibetan Buddhist texts into English.
Upload a `.txt`, `.docx`, or `.pdf` file → it's split into segments →
translate each segment (or all at once) → edit inline → download the result.

Compared to the original `TranslationUI` this app deliberately drops:
- Video/audio playback and Google Drive import (no media, ever)
- Subtitle formats (SRT/VTT) and timestamp-based segments — plain text in, plain text out
- Multi-language target selection (English only)
- Per-language prompt files and a separate post-edit pass (one prompt, no post-edit)
- HuggingFace-dataset-backed project storage and versioned prompt history
  (no `HF_TOKEN`, no `datasets`/`huggingface_hub` write dependency — state
  lives only in the browser session; prompt edits are a local file)

## Architecture

```
app.py                    Gradio UI (Blocks), wires handlers to components
handlers.py                Event handlers + in-memory session state (gr.State)
styles.py                  Minimal CSS (no floating video player JS anymore)
translation_prompt.txt     The one editable translation prompt (Garchen Rinpoche voice/glossary)
engine/
  chunking.py               Plain-text -> segment list (paragraph, then shad/whitespace splitting)
  formats.py                Read .txt/.docx/.pdf; write .txt/.docx/.json
  prompt.py                 Read/write/reset translation_prompt.txt (local file only)
  translate.py              Backend dispatcher (OpenRouter vs local CPU model) + batching/progress
  openrouter_backend.py      OpenRouter API calls, retries, server-side model fallback, batch prompt building
  local_backend.py           CPU fallback: AutoModelForSeq2SeqLM "billingsmoore/mlotsawa-ground-base"
```

## Translation backend selection

`engine/translate.py::_resolve_key()` decides per-call:
- An OpenRouter API key was typed into the UI, **or** `OPENROUTER_API_KEY` is set (env var / `.env`)
  → use OpenRouter, with the prompt in `translation_prompt.txt` (editable in Settings) and the model
  picked in the dropdown. The dropdown is populated by `engine/openrouter_backend.py::list_model_choices()`,
  which live-fetches OpenRouter's `/models` catalog and puts `CURATED_MODELS` (a small hand-picked,
  known-good-for-translation set) at the top, followed by the rest of the text-output catalog
  alphabetically — falls back to just `CURATED_MODELS` if the fetch fails (offline, OpenRouter down).
  `CURATED_MODELS` also doubles as the `models` fallback array OpenRouter is given per request, so it
  retries server-side against another good model if the chosen one errors out.
- No key anywhere → use the local CPU model `billingsmoore/mlotsawa-ground-base` via
  `AutoModelForSeq2SeqLM.generate()` directly (see `engine/local_backend.py`). This is a plain finetuned
  T5 seq2seq model — it does **not** read `translation_prompt.txt` at all. First call downloads the
  model (~a few hundred MB) and is slow on CPU; it stays cached in memory for the life of the process.
  This model was introduced in ["Optimizing T5 for Lightweight Tibetan-English Translation"](https://doi.org/10.21203/rs.3.rs-7409829/v1)
  (Moore & Lauren, 2025); its training/eval code is at
  [optimizing-t5-tibetan-english-mt](https://github.com/billingsmoore/optimizing-t5-tibetan-english-mt).

**Why not `pipeline("translation", ...)`:** that's the model card's documented usage, but the
`"translation"` pipeline task was removed in `transformers>=5` (raises `Unknown task translation`).
Instead `local_backend.py` reads the model's own `model.config.task_specific_params["translation_bo_to_en"]`
(prefix `"translate Tibetan to English: "`, `num_beams=4`, `max_length=300`) and calls `generate()`
directly — verified to reproduce the model card's example output exactly, and works on any transformers
version that ships `AutoModelForSeq2SeqLM`/`AutoTokenizer`. Don't "fix" this back to `pipeline(...)`
without first checking what transformers version is actually installed.

## Segmentation

`engine/chunking.py::split_into_segments()`: splits on blank-line paragraph breaks, then (for any
paragraph longer than `max_chars`, default 400) on Tibetan sentence-ending shad marks (`།` `༎`), then
falls back to plain whitespace word-chunking for any single "sentence" still too long. There is no
timestamp concept anywhere in this app — segments are just `{"source": str, "target": str}` dicts.

## Running it

```bash
pip install -r requirements.txt
python app.py
```
Opens on `http://localhost:7860`. Put `OPENROUTER_API_KEY=...` in a `.env` file to default to
OpenRouter without typing a key into the UI each time.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/` covers chunking, file I/O (txt/docx/pdf load+export, JSON resume),
prompt storage, the OpenRouter backend (model curation/fetch-fallback, request
retry/fallback logic, batch prompt building/parsing — all with `requests`
mocked, no real network calls or API key needed), the backend dispatcher, the
local CPU backend (with `torch`/`transformers` stubbed via `sys.modules` so
the suite doesn't require installing those heavy deps), the Gradio event
handlers, and a smoke test that the Blocks graph in `app.py` builds without
error. `samples/sample_tibetan.{txt,docx,pdf}` are real Tibetan-script fixture
files (a well-known refuge/bodhicitta verse, the four immeasurables, and a
dedication verse) used by the format-loading tests and useful for manually
exercising the upload flow.

A `pre-push` git hook (`.githooks/pre-push`) runs the full suite and blocks
the push if anything fails. It's already installed into `.git/hooks/pre-push`
in this working copy; on a fresh clone, install it with either
`git config core.hooksPath .githooks` or `cp .githooks/pre-push .git/hooks/`.

## Conventions to keep

- No comments explaining *what* code does — only *why*, when non-obvious (mirrors the parent repo's style).
- Don't reintroduce HuggingFace dataset persistence, subtitle timing, or multi-language prompt files —
  those were removed on purpose to keep this app simple. If a future ask needs them, treat it as a
  deliberate scope expansion, not a bug to fix.
- This repo has no relationship to `Garchen`'s git history — it's fine to `git init` here and commit
  independently.

## Remotes

Two remotes, same dual-remote convention as `Garchen`: `origin` = GitHub
(https://github.com/billingsmoore/SimpleTranslationUI, source of truth) and `space` = the deployed
Hugging Face Space (https://huggingface.co/spaces/billingsmoore/SimpleTranslationUI). Push to `origin`
for normal commits; push to `space` (`git push space main:main`) to actually redeploy the Space.
