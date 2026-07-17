from unittest.mock import MagicMock, patch

import pytest
import requests

from engine import openrouter_backend as ob


# ── prompt building / response parsing (pure functions) ────────────────────

def test_build_batch_prompt_numbers_lines_in_order():
    prompt = ob._build_batch_prompt("TEMPLATE", ["first", "second", "third"])
    assert "1. first" in prompt
    assert "2. second" in prompt
    assert "3. third" in prompt
    assert prompt.startswith("TEMPLATE")


def test_build_batch_prompt_includes_preceding_context_when_given():
    prompt = ob._build_batch_prompt("TEMPLATE", ["text"], preceding=["prev1", "prev2"])
    assert "prev1" in prompt
    assert "prev2" in prompt
    assert "translated immediately before" in prompt


def test_build_batch_prompt_omits_context_block_when_no_preceding():
    prompt = ob._build_batch_prompt("TEMPLATE", ["text"], preceding=None)
    assert "translated immediately before" not in prompt


def test_parse_batch_response_happy_path():
    response = "1. hello\n2. world\n3. foo bar"
    assert ob._parse_batch_response(response, 3) == ["hello", "world", "foo bar"]


def test_parse_batch_response_wrong_count_returns_none():
    response = "1. hello\n2. world"
    assert ob._parse_batch_response(response, 3) is None


def test_parse_batch_response_ignores_non_numbered_lines():
    response = "Sure, here you go:\n1. hello\n2. world\nThanks!"
    assert ob._parse_batch_response(response, 2) == ["hello", "world"]


def test_parse_batch_response_empty_input():
    assert ob._parse_batch_response("", 0) == []
    assert ob._parse_batch_response("", 1) is None


# ── model listing / curation ────────────────────────────────────────────────

def _model(id_, modality="text->text", output_modalities=None):
    return {
        "id": id_,
        "architecture": {
            "modality": modality,
            "output_modalities": output_modalities if output_modalities is not None else ["text"],
        },
    }


def test_is_text_model_accepts_text_to_text():
    assert ob._is_text_model(_model("a/a", modality="text->text"))


def test_is_text_model_accepts_multimodal_input_text_output():
    assert ob._is_text_model(_model("a/a", modality="text+image->text"))


def test_is_text_model_rejects_image_output():
    m = _model("a/a", modality="text->image", output_modalities=["image"])
    assert not ob._is_text_model(m)


def test_list_model_choices_puts_curated_first_then_rest_alphabetically():
    fetched = [
        _model("zzz/last"),
        _model(ob.CURATED_MODELS[1]),
        _model("aaa/first"),
        _model(ob.CURATED_MODELS[0]),
    ]
    with patch.object(ob, "fetch_available_models", return_value=fetched):
        choices = ob.list_model_choices()
    assert choices == [ob.CURATED_MODELS[0], ob.CURATED_MODELS[1], "aaa/first", "zzz/last"]


def test_list_model_choices_drops_curated_models_not_in_catalog():
    fetched = [_model("aaa/first")]
    with patch.object(ob, "fetch_available_models", return_value=fetched):
        choices = ob.list_model_choices()
    assert choices == ["aaa/first"]
    for curated in ob.CURATED_MODELS:
        assert curated not in choices


def test_list_model_choices_filters_out_non_text_models():
    fetched = [
        _model(ob.CURATED_MODELS[0]),
        _model("image/only", modality="text->image", output_modalities=["image"]),
    ]
    with patch.object(ob, "fetch_available_models", return_value=fetched):
        choices = ob.list_model_choices()
    assert "image/only" not in choices


def test_list_model_choices_falls_back_to_curated_on_fetch_failure():
    with patch.object(ob, "fetch_available_models", side_effect=requests.RequestException("network down")):
        choices = ob.list_model_choices()
    assert choices == list(ob.CURATED_MODELS)


def test_fetch_available_models_hits_public_models_endpoint():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": [{"id": "a/a"}]}
    with patch.object(ob.requests, "get", return_value=fake_resp) as mock_get:
        result = ob.fetch_available_models(timeout=5)
    mock_get.assert_called_once_with(f"{ob.BASE_URL}/models", timeout=5)
    fake_resp.raise_for_status.assert_called_once()
    assert result == [{"id": "a/a"}]


# ── _call_openrouter (network layer) ────────────────────────────────────────

def _http_response(status_code, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_body is not None:
        resp.json.return_value = json_body
    return resp


def test_call_openrouter_success_returns_message_content():
    ok = _http_response(200, {"choices": [{"message": {"content": "translated text"}}]})
    with patch.object(ob.requests, "post", return_value=ok) as mock_post:
        result = ob._call_openrouter("key123", "some/model", "prompt text")
    assert result == "translated text"

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer key123"
    assert kwargs["json"]["models"][0] == "some/model"
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "prompt text"}]


def test_call_openrouter_includes_curated_models_as_serverside_fallback():
    ok = _http_response(200, {"choices": [{"message": {"content": "x"}}]})
    with patch.object(ob.requests, "post", return_value=ok) as mock_post:
        ob._call_openrouter("key", ob.CURATED_MODELS[2], "prompt")
    sent_models = mock_post.call_args.kwargs["json"]["models"]
    assert sent_models[0] == ob.CURATED_MODELS[2]
    assert set(sent_models) == set(ob.CURATED_MODELS)
    assert len(sent_models) == len(ob.CURATED_MODELS)  # no duplicate of the primary model


def test_call_openrouter_retries_on_retryable_status_then_succeeds():
    bad = _http_response(429, text="rate limited")
    good = _http_response(200, {"choices": [{"message": {"content": "ok"}}]})
    with patch.object(ob.requests, "post", side_effect=[bad, good]), patch.object(ob.time, "sleep"):
        result = ob._call_openrouter("key", "model", "prompt")
    assert result == "ok"


def test_call_openrouter_raises_immediately_on_non_retryable_status():
    unauthorized = _http_response(401, text="invalid api key")
    with patch.object(ob.requests, "post", return_value=unauthorized) as mock_post:
        with pytest.raises(RuntimeError, match="401"):
            ob._call_openrouter("bad-key", "model", "prompt")
    assert mock_post.call_count == 1


def test_call_openrouter_exhausts_retries_and_raises_last_error():
    bad = _http_response(503, text="unavailable")
    with patch.object(ob.requests, "post", return_value=bad), patch.object(ob.time, "sleep"):
        with pytest.raises(RuntimeError, match="503"):
            ob._call_openrouter("key", "model", "prompt")


def test_call_openrouter_retries_on_network_exception():
    good = _http_response(200, {"choices": [{"message": {"content": "recovered"}}]})
    with patch.object(
        ob.requests, "post", side_effect=[requests.ConnectionError("boom"), good]
    ), patch.object(ob.time, "sleep"):
        result = ob._call_openrouter("key", "model", "prompt")
    assert result == "recovered"


# ── translate_one / translate_batch ─────────────────────────────────────────

def test_translate_one_strips_whitespace_from_response():
    with patch.object(ob, "_call_openrouter", return_value="  translated  \n"):
        result = ob.translate_one("source text", "key", "model", "template")
    assert result == "translated"


def test_translate_batch_empty_texts_short_circuits_without_network():
    with patch.object(ob, "_call_openrouter") as mock_call:
        result = ob.translate_batch([], "key", "model", "template")
    assert result == []
    mock_call.assert_not_called()


def test_translate_batch_happy_path():
    with patch.object(ob, "_call_openrouter", return_value="1. one\n2. two"):
        result = ob.translate_batch(["a", "b"], "key", "model", "template")
    assert result == ["one", "two"]


def test_translate_batch_returns_none_on_parse_mismatch():
    with patch.object(ob, "_call_openrouter", return_value="1. only one line"):
        result = ob.translate_batch(["a", "b"], "key", "model", "template")
    assert result is None


def test_translate_batch_returns_none_on_network_error():
    with patch.object(ob, "_call_openrouter", side_effect=RuntimeError("boom")):
        result = ob.translate_batch(["a", "b"], "key", "model", "template")
    assert result is None
