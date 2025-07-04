from typing import Dict
import gradio as gr

from constants import characters, comfyui_input
from ui.layout import CharacterManagerLayout


def bind_events(components):
    _bind_character_change(components)


def on_character_change(character: str):
    char_key = character.replace(" ", "_").lower()

    return characters[char_key]


def _bind_character_change(components: Dict[str, gr.Component]):
    input_keys = ["character_select"]
    output_keys = ["current_fields"]

    components["character_select"].change(
        on_character_change,
        inputs=[components[x] for x in input_keys],
        outputs=[components[x] for x in output_keys],
    )
