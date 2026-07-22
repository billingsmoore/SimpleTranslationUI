"""SimpleTranslationUI — plain-text Tibetan Buddhist translation editor."""

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from engine import DEFAULT_MODEL, list_model_choices

from styles import CSS
from handlers import (
    MAX_SLOTS,
    _handle_cancel_translate,
    _handle_download_click,
    _handle_get_files,
    _handle_next,
    _handle_prev,
    _handle_save,
    _load_resume_json,
    _load_source,
    _make_state,
    _read_prompt,
    _reset_prompt,
    _save_prompt,
    _set_usage_tracking,
    _translate_all,
    _translate_one,
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SimpleTranslationUI") as demo:
        app_state = gr.State(_make_state())

        gr.Markdown(
            "# SimpleTranslationUI\nTibetan Buddhist Translation Editor\n\n"
            "**How to use:** Upload a source document below — it's automatically split into "
            "segments (by paragraph, then by sentence-ending shad marks or whitespace for long "
            "paragraphs). Click **Translate All** to translate every segment, or translate one "
            "segment at a time with its own **Translate** button. Edit any translation inline, "
            "then **Download** the result as text, Word, or JSON.\n\n"
            "**Choosing a model:** By default (no API key) translation runs on a local CPU "
            "model ([billingsmoore/mlotsawa-ground-base]"
            "(https://huggingface.co/billingsmoore/mlotsawa-ground-base)) — free, runs on this "
            "machine, but a plain translation model, not an instruction-following AI (it ignores "
            "the translation prompt below) and generally lower translation quality. For higher-quality, instruction-"
            "following translation, add an **OpenRouter API key** in Settings and pick a model "
            "from the dropdown — this calls OpenRouter's API (usage is billed to your OpenRouter "
            "account) and uses the editable translation prompt.\n\n"
            "Usage of this space may be monitored for research purposes. You can opt out of this "
            "in Settings."
        )

        with gr.Row():
            source_input = gr.File(
                label="Source Document (.txt, .docx, .pdf)",
                file_types=[".txt", ".docx", ".pdf"],
                scale=1,
            )
            resume_input = gr.File(
                label="Resume: Previously downloaded JSON (optional)",
                file_types=[".json"],
                scale=1,
            )

        with gr.Accordion("Settings", open=False):
            usage_tracking_checkbox = gr.Checkbox(
                label="Allow usage tracking for research purposes",
                value=True,
            )
            with gr.Row():
                openrouter_api_key = gr.Textbox(
                    label="OpenRouter API Key (optional)",
                    placeholder="sk-or-… — leave blank to use the local CPU model instead",
                    type="password",
                    scale=2,
                )
                openrouter_model = gr.Dropdown(
                    choices=list_model_choices(), value=DEFAULT_MODEL,
                    label="OpenRouter Model (only used if a key is given)",
                    info="A few recommended models are pinned at the top; the rest of "
                         "OpenRouter's catalog is below — type to filter.",
                    scale=1,
                )

            gr.Markdown(
                "### Translation Prompt\n"
                "Only used when translating with OpenRouter — the local CPU fallback "
                "(billingsmoore/mlotsawa-ground-base) is a plain translation model and ignores this. "
                "Saved directly to `translation_prompt.txt` in this app's folder."
            )
            prompt_box = gr.Textbox(label="Translation Prompt", lines=12, value=_read_prompt())
            with gr.Row():
                save_prompt_btn = gr.Button("Save", variant="primary", size="sm", scale=0, min_width=80)
                reset_prompt_btn = gr.Button("Reset to Default", size="sm", scale=0, min_width=130)
            prompt_status = gr.Markdown("")

        with gr.Row():
            translate_all_btn = gr.Button("Translate All", variant="primary", scale=2)
            cancel_btn = gr.Button("Cancel", variant="stop", scale=1)
        status_output = gr.Textbox(label="Status", lines=2, interactive=False, visible=False)

        with gr.Column(visible=False) as editor_col:
            gr.Markdown("### Segments")

            with gr.Row():
                prev_btn = gr.Button("◀ Previous", interactive=False)
                nav_label = gr.HTML("")
                next_btn = gr.Button("Next ▶", interactive=False)

            with gr.Row():
                save_btn = gr.Button("Save", variant="secondary")
                download_btn = gr.Button("Download", variant="secondary")
            with gr.Row():
                download_checklist = gr.CheckboxGroup(
                    label="Files to include", choices=[], value=[], visible=False, scale=2,
                )
                get_files_btn = gr.Button("Get Files", variant="primary", scale=0, min_width=100, visible=False)
            download_zip = gr.File(label="Download", visible=False, interactive=False)

            slot_groups, slot_sources, slot_targets, slot_translate_btns = [], [], [], []

            for _ in range(MAX_SLOTS):
                with gr.Group(visible=False) as grp:
                    with gr.Row():
                        tlb = gr.Button("Translate", size="sm", scale=0, min_width=90)
                    with gr.Row(equal_height=True):
                        src = gr.Textbox(label="Source", lines=3, scale=1, elem_classes=["src-box"])
                        tgt = gr.Textbox(label="Translation", lines=3, scale=1, elem_classes=["tgt-box"])
                slot_groups.append(grp)
                slot_sources.append(src)
                slot_targets.append(tgt)
                slot_translate_btns.append(tlb)

        page_outputs = [
            app_state, nav_label, prev_btn, next_btn,
            *slot_groups, *slot_sources, *slot_targets,
        ]
        edit_inputs = [app_state, *slot_sources, *slot_targets]
        load_outputs = [*page_outputs, editor_col]
        translate_all_outputs = [*page_outputs, status_output]

        source_input.change(fn=_load_source, inputs=[source_input, app_state], outputs=load_outputs)
        resume_input.change(fn=_load_resume_json, inputs=[resume_input, app_state], outputs=page_outputs)

        prev_btn.click(_handle_prev, inputs=edit_inputs, outputs=page_outputs)
        next_btn.click(_handle_next, inputs=edit_inputs, outputs=page_outputs)

        translate_event = translate_all_btn.click(
            _translate_all,
            inputs=[app_state, openrouter_api_key, openrouter_model, *slot_sources, *slot_targets],
            outputs=translate_all_outputs,
        )
        cancel_btn.click(_handle_cancel_translate, inputs=[app_state], outputs=[app_state], cancels=[translate_event])

        save_btn.click(_handle_save, inputs=edit_inputs, outputs=[app_state, status_output])
        download_btn.click(
            _handle_download_click,
            inputs=edit_inputs,
            outputs=[app_state, download_checklist, get_files_btn, download_zip],
        )
        get_files_btn.click(
            _handle_get_files,
            inputs=[app_state, download_checklist],
            outputs=[download_zip, download_checklist, get_files_btn],
        )

        for i in range(MAX_SLOTS):
            slot_sources[i].blur(_handle_save, inputs=edit_inputs, outputs=[app_state, status_output])
            slot_targets[i].blur(_handle_save, inputs=edit_inputs, outputs=[app_state, status_output])

        for i in range(MAX_SLOTS):
            slot_translate_btns[i].click(
                fn=lambda: gr.update(value="", placeholder="Translating…"),
                inputs=[],
                outputs=[slot_targets[i]],
            ).then(
                fn=lambda state, source, key, model, i=i: _translate_one(state, i, source, key, model),
                inputs=[app_state, slot_sources[i], openrouter_api_key, openrouter_model],
                outputs=[app_state, slot_targets[i]],
            )

        save_prompt_btn.click(_save_prompt, inputs=[app_state, prompt_box], outputs=[app_state, prompt_status])
        reset_prompt_btn.click(_reset_prompt, inputs=[app_state], outputs=[app_state, prompt_box, prompt_status])

        usage_tracking_checkbox.change(
            _set_usage_tracking, inputs=[app_state, usage_tracking_checkbox], outputs=[app_state],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(css=CSS, ssr_mode=False, server_name="0.0.0.0", server_port=7860)
else:
    demo.css = CSS
