"""CPU translation backend using billingsmoore/mlotsawa-ground-base
(a Tibetan->English seq2seq T5 model). Used whenever no Gemini API key is
available. The model is loaded once, lazily, on first use.

Generation uses the model's own task_specific_params["translation_bo_to_en"]
(prefix "translate Tibetan to English: ", num_beams=4, max_length=300) rather
than transformers' old pipeline("translation", ...) task, which was removed
in transformers>=5 (raises "Unknown task translation"). Calling generate()
directly with the model's documented settings works on any transformers
version that ships AutoModelForSeq2SeqLM/AutoTokenizer.

This backend ignores the editable translation prompt — it's a plain
seq2seq model, not an instruction-following LLM.

translate_batch is decorated with @spaces.GPU so this runs on HF ZeroGPU
Spaces (which refuse to boot a gradio-sdk app with no @spaces.GPU function
at all); the decorator is a no-op outside a ZeroGPU Space.
"""

import spaces

MODEL_ID = "billingsmoore/mlotsawa-ground-base"
_PREFIX = "translate Tibetan to English: "

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f"[INFO] Loading local CPU translation model: {MODEL_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _tokenizer


@spaces.GPU
def translate_batch(texts: list[str]) -> list[str]:
    if not texts:
        return []
    import torch
    model, tokenizer = _load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(
        [_PREFIX + t for t in texts],
        return_tensors="pt", padding=True, truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300, num_beams=4, early_stopping=True)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
