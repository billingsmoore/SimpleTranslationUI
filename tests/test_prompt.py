import importlib

import engine.prompt as prompt_module


def _reload_with_prompt_path(monkeypatch, tmp_path, initial_text):
    prompt_path = tmp_path / "translation_prompt.txt"
    prompt_path.write_text(initial_text, encoding="utf-8")
    monkeypatch.setattr(prompt_module, "PROMPT_PATH", prompt_path)
    monkeypatch.setattr(prompt_module, "_DEFAULT_PROMPT", initial_text)
    return prompt_path


def test_read_prompt_returns_file_contents(monkeypatch, tmp_path):
    _reload_with_prompt_path(monkeypatch, tmp_path, "default prompt text")
    assert prompt_module.read_prompt() == "default prompt text"


def test_save_prompt_persists_to_file(monkeypatch, tmp_path):
    path = _reload_with_prompt_path(monkeypatch, tmp_path, "default prompt text")
    prompt_module.save_prompt("edited prompt text")
    assert path.read_text(encoding="utf-8") == "edited prompt text"
    assert prompt_module.read_prompt() == "edited prompt text"


def test_reset_prompt_restores_default_and_writes_file(monkeypatch, tmp_path):
    path = _reload_with_prompt_path(monkeypatch, tmp_path, "default prompt text")
    prompt_module.save_prompt("edited prompt text")
    result = prompt_module.reset_prompt()
    assert result == "default prompt text"
    assert path.read_text(encoding="utf-8") == "default prompt text"


def test_read_prompt_falls_back_to_default_when_file_missing(monkeypatch, tmp_path):
    path = _reload_with_prompt_path(monkeypatch, tmp_path, "default prompt text")
    path.unlink()
    assert prompt_module.read_prompt() == "default prompt text"


def test_default_prompt_file_exists_and_is_nonempty():
    # sanity check on the real shipped translation_prompt.txt, independent of monkeypatching above
    importlib.reload(prompt_module)
    assert prompt_module.PROMPT_PATH.exists()
    assert len(prompt_module.read_prompt().strip()) > 0
