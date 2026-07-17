"""Local-only prompt storage. Just one prompt file, no per-language
selection, no versioned dataset — edit the box, Save writes the file,
Reset restores what shipped with this app."""

from pathlib import Path

PROMPT_PATH = Path(__file__).parent.parent / "translation_prompt.txt"

_DEFAULT_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def read_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else _DEFAULT_PROMPT


def save_prompt(text: str) -> None:
    PROMPT_PATH.write_text(text, encoding="utf-8")


def reset_prompt() -> str:
    PROMPT_PATH.write_text(_DEFAULT_PROMPT, encoding="utf-8")
    return _DEFAULT_PROMPT
