"""Translation engine for SimpleTranslationUI.

Public surface used by handlers.py / app.py.
"""

from .chunking import split_into_segments
from .formats import (
    load_source_file,
    export_to_txt,
    export_to_docx,
    export_to_json,
    parse_json,
)
from .prompt import read_prompt, save_prompt, reset_prompt, PROMPT_PATH
from .usage_store import log_event
from .translate import (
    CURATED_MODELS,
    DEFAULT_MODEL,
    list_model_choices,
    translate_one,
    translate_segments,
    using_openrouter,
)
