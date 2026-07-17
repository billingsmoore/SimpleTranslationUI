"""Tests for engine/local_backend.py, stubbing out torch/transformers so this
suite doesn't require installing the heavy real dependencies (torch is a
multi-GB download and isn't needed to verify this module's own glue logic:
prompt prefixing, device selection, and decode handling)."""

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stubbed_torch_and_transformers(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSeq2SeqLM = MagicMock()
    fake_transformers.AutoTokenizer = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    return fake_torch, fake_transformers


@pytest.fixture(autouse=True)
def reset_model_cache(monkeypatch):
    import engine.local_backend as lb
    monkeypatch.setattr(lb, "_model", None)
    monkeypatch.setattr(lb, "_tokenizer", None)
    yield


def _make_fake_model_and_tokenizer(decoded_outputs):
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.to.return_value = encoded
    tokenizer.return_value = encoded
    tokenizer.decode.side_effect = decoded_outputs

    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = list(range(len(decoded_outputs)))  # dummy token id "rows"

    return model, tokenizer


def test_translate_batch_empty_list_short_circuits(stubbed_torch_and_transformers):
    import engine.local_backend as lb
    assert lb.translate_batch([]) == []


def test_translate_batch_prefixes_and_decodes(stubbed_torch_and_transformers, monkeypatch):
    import engine.local_backend as lb

    model, tokenizer = _make_fake_model_and_tokenizer(["translation one", "translation two"])
    monkeypatch.setattr(lb, "_load", lambda: (model, tokenizer))

    result = lb.translate_batch(["first source", "second source"])

    assert result == ["translation one", "translation two"]
    called_texts = tokenizer.call_args[0][0]
    assert called_texts == [
        "translate Tibetan to English: first source",
        "translate Tibetan to English: second source",
    ]


def test_translate_batch_uses_generation_settings_from_docstring(stubbed_torch_and_transformers, monkeypatch):
    import engine.local_backend as lb

    model, tokenizer = _make_fake_model_and_tokenizer(["out"])
    monkeypatch.setattr(lb, "_load", lambda: (model, tokenizer))

    lb.translate_batch(["text"])

    _, kwargs = model.generate.call_args
    assert kwargs["max_length"] == 300
    assert kwargs["num_beams"] == 4
    assert kwargs["early_stopping"] is True


def test_translate_batch_uses_cpu_when_no_cuda(stubbed_torch_and_transformers, monkeypatch):
    import engine.local_backend as lb

    model, tokenizer = _make_fake_model_and_tokenizer(["out"])
    monkeypatch.setattr(lb, "_load", lambda: (model, tokenizer))

    lb.translate_batch(["text"])

    model.to.assert_called_with("cpu")


def test_translate_batch_uses_cuda_when_available(stubbed_torch_and_transformers, monkeypatch):
    import engine.local_backend as lb

    stubbed_torch_and_transformers[0].cuda.is_available = lambda: True
    model, tokenizer = _make_fake_model_and_tokenizer(["out"])
    monkeypatch.setattr(lb, "_load", lambda: (model, tokenizer))

    lb.translate_batch(["text"])

    model.to.assert_called_with("cuda")


def test_load_caches_model_and_tokenizer_across_calls(stubbed_torch_and_transformers):
    import engine.local_backend as lb

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys.modules["transformers"], "AutoModelForSeq2SeqLM",
            MagicMock(from_pretrained=MagicMock(return_value=fake_model)),
        )
        mp.setattr(
            sys.modules["transformers"], "AutoTokenizer",
            MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)),
        )
        model1, tok1 = lb._load()
        model2, tok2 = lb._load()

    assert model1 is model2
    assert tok1 is tok2
    fake_model.eval.assert_called_once()
