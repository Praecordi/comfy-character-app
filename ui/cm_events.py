from typing import Dict
import gradio as gr

from constants import characters, comfyui_input


def bind_events(components):
    _bind_character_change(components)


def on_character_change(character: str):
    char_key = character.replace(" ", "_").lower()

    char_dict = characters[char_key]

    face_images = char_dict.pop("face_reference", [])
    if isinstance(face_images, list):
        face_images = [str(comfyui_input / ref) for ref in face_images]
    else:
        face_images = [str(comfyui_input / face_images)]

    face_prompt = char_dict.pop("face", "")
    skin_prompt = char_dict.pop("skin", "")
    hair_prompt = char_dict.pop("hair", "")
    eyes_prompt = char_dict.pop("eyes", "")

    return char_dict, face_images, face_prompt, skin_prompt, hair_prompt, eyes_prompt


def _bind_character_change(components: Dict[str, gr.Component]):
    input_keys = ["character_select"]
    output_keys = [
        "current_fields",
        "face_images",
        "face_prompt",
        "skin_prompt",
        "hair_prompt",
        "eyes_prompt",
    ]

    components["character_select"].change(
        on_character_change,
        inputs=[components[x] for x in input_keys],
        outputs=[components[x] for x in output_keys],
    )
