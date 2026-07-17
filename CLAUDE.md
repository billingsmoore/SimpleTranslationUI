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
  translate.py              Backend dispatcher (Gemini vs local CPU model) + batching/progress
  gemini_backend.py          Gemini API calls, retries, fallback model chain, batch prompt building
  local_backend.py           CPU fallback: AutoModelForSeq2SeqLM "billingsmoore/mlotsawa-ground-base"
```

## Translation backend selection

`engine/translate.py::_resolve_key()` decides per-call:
- A Gemini API key was typed into the UI, **or** `GEMINI_API_KEY` is set (env var / `.env`) → use Gemini,
  with the prompt in `translation_prompt.txt` (editable in Settings) and the model picked in the dropdown
  (`FALLBACK_CHAIN` in `engine/gemini_backend.py`; falls through the chain on failure).
- No key anywhere → use the local CPU model `billingsmoore/mlotsawa-ground-base` via
  `AutoModelForSeq2SeqLM.generate()` directly (see `engine/local_backend.py`). This is a plain finetuned
  T5 seq2seq model — it does **not** read `translation_prompt.txt` at all. First call downloads the
  model (~a few hundred MB) and is slow on CPU; it stays cached in memory for the life of the process.

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
Opens on `http://localhost:7860`. Put `GEMINI_API_KEY=...` in a `.env` file to default to Gemini
without typing a key into the UI each time.

## Conventions to keep

- No comments explaining *what* code does — only *why*, when non-obvious (mirrors the parent repo's style).
- Don't reintroduce HuggingFace dataset persistence, subtitle timing, or multi-language prompt files —
  those were removed on purpose to keep this app simple. If a future ask needs them, treat it as a
  deliberate scope expansion, not a bug to fix.
- This repo has no relationship to `Garchen`'s git history — it's fine to `git init` here and commit
  independently.
