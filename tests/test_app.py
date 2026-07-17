"""Smoke test: the Gradio Blocks graph wires up without error. This is what
would have caught a stale gr.Dropdown choices/value reference or a mismatched
inputs/outputs list length after the Gemini -> OpenRouter rename."""

from unittest.mock import patch

from app import build_app


def test_build_app_does_not_raise():
    with patch("app.list_model_choices", return_value=["model/a", "model/b"]), \
         patch("app.DEFAULT_MODEL", "model/a"):
        demo = build_app()
    assert demo is not None


def test_build_app_falls_back_gracefully_if_model_fetch_fails():
    with patch("app.list_model_choices", return_value=["model/a"]), \
         patch("app.DEFAULT_MODEL", "model/a"):
        demo = build_app()
    assert demo is not None


def test_download_file_hidden_until_get_files_clicked():
    with patch("app.list_model_choices", return_value=["model/a"]), \
         patch("app.DEFAULT_MODEL", "model/a"):
        demo = build_app()
    download_components = [
        b for b in demo.blocks.values()
        if b.__class__.__name__ == "File" and b.label == "Download"
    ]
    assert len(download_components) == 1
    assert download_components[0].visible is False
