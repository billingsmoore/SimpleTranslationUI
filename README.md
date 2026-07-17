---
title: SimpleTranslationUI
emoji: 📜
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# SimpleTranslationUI

A stripped-down Tibetan Buddhist translation editor. Upload a `.txt`, `.docx`,
or `.pdf` file — it's automatically split into segments, translated, and
editable before download.

No video/audio, no subtitle timing, no multi-language prompts, no saved
projects — just upload, translate, edit, download.

## Translation backend

- If you provide a **Gemini API key** (in the UI, or via a `GEMINI_API_KEY`
  env var / `.env` file), translation uses Gemini with the editable prompt
  in the Settings panel (`translation_prompt.txt`).
- Otherwise it falls back to a **local CPU model**,
  [billingsmoore/mlotsawa-ground-base](https://huggingface.co/billingsmoore/mlotsawa-ground-base),
  loaded via `transformers`. The prompt box has no effect on this backend —
  it's a plain seq2seq translation model, not an instruction-following LLM.
  First use downloads the model (a few hundred MB) and is slow; keep it running.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Optionally set `GEMINI_API_KEY` in a `.env` file to default to Gemini without
typing a key into the UI each time.

## Downloads

After translating, Download offers whichever of these have content:
- Translation as `.txt`
- Translation as `.docx`
- `.json` (source + target per segment — re-upload via "Resume" to merge
  translations back in if you re-upload the same source file)
- Source text as `.txt`
