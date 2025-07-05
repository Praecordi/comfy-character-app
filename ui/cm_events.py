from typing import Dict
from copy import deepcopy
import gradio as gr

import constants
from ui.utils import make_key, make_name


def load_initial():
    character_choices = [
        char.replace("_", " ").title() for char in constants.characters.keys()
    ]

    return gr.update(value=character_choices[0])


def on_character_change(character: str):
    char_key = make_key(character)

    char_dict = deepcopy(constants.characters[char_key])

    face_images = char_dict.pop("face_reference", [])
    if isinstance(face_images, list):
        face_images = [str(constants.comfyui_input / ref) for ref in face_images]
    else:
        face_images = [str(constants.comfyui_input / face_images)]

    face_prompt = char_dict.pop("face", "")
    skin_prompt = char_dict.pop("skin", "")
    hair_prompt = char_dict.pop("hair", "")
    eyes_prompt = char_dict.pop("eyes", "")

    return char_dict, face_images, face_prompt, skin_prompt, hair_prompt, eyes_prompt


def reset(character):
    constants.reset_characters()
    new_choices = [
        char.replace("_", " ").title() for char in constants.characters.keys()
    ]

    return gr.update(value=character, choices=new_choices), *on_character_change(
        character
    )


def save():
    constants.save_characters()


def add_character(character):
    char_key = make_key(character)

    if char_key in constants.characters.keys():
        gr.Warning("Character already exists", duration=10)
        return gr.update(), gr.update()

    if char_key == "":
        gr.Warning("Please provide a character name")
        return gr.update(), gr.update()

    char_dict = {
        "face_reference": [],
        "face": "",
        "skin": "",
        "hair": "",
        "eyes": "",
    }

    constants.characters[char_key] = char_dict

    new_choices = [
        char.replace("_", " ").title() for char in constants.characters.keys()
    ]

    return gr.update(value=""), gr.update(
        value=make_name(char_key), choices=new_choices
    )


def add_field(field_name, current_fields, character):
    char_key = make_key(character)
    field_key = make_key(field_name)

    if char_key not in constants.characters:
        gr.Warning(f"Character {character} does not exist...")
        return gr.update(value=""), gr.update("")

    if field_key in constants.characters[char_key]:
        gr.Warning(f"Attribute {field_name} already exists...")
        return gr.update(value=""), current_fields

    if field_key == "":
        gr.Warning("Please provide non-empty field")
        return gr.update(value=""), current_fields

    current_fields[field_key] = ""

    constants.characters[char_key][field_key] = ""

    return gr.update(value=""), current_fields


def delete_field(attribute, current_fields, character):
    char_key = make_key(character)
    attr_key = make_key(attribute)

    if char_key not in constants.characters:
        gr.Warning(f"Character {character} doesn't exist")
        return current_fields

    if attr_key in constants.characters[char_key]:
        del constants.characters[char_key][attr_key]
        del current_fields[attr_key]

    return current_fields


def bind_events(block, components):
    _bind_character_change(components)
    _bind_buttons(components)

    block.load(load_initial, outputs=[components["character_select"]])


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


def _bind_buttons(components: Dict[str, gr.Component]):
    reset_output_keys = [
        "character_select",
        "current_fields",
        "face_images",
        "face_prompt",
        "skin_prompt",
        "hair_prompt",
        "eyes_prompt",
    ]
    components["reset_btn"].click(
        reset,
        inputs=[components["character_select"]],
        outputs=[components[x] for x in reset_output_keys],
    )

    components["new_character"].submit(
        add_character,
        inputs=[components["new_character"]],
        outputs=[components["new_character"], components["character_select"]],
    )

    components["new_character_btn"].click(
        add_character,
        inputs=[components["new_character"]],
        outputs=[components["new_character"], components["character_select"]],
    )

    components["new_field"].submit(
        add_field,
        inputs=[
            components["new_field"],
            components["current_fields"],
            components["character_select"],
        ],
        outputs=[components["new_field"], components["current_fields"]],
    )

    components["add_field_btn"].click(
        add_field,
        inputs=[
            components["new_field"],
            components["current_fields"],
            components["character_select"],
        ],
        outputs=[components["new_field"], components["current_fields"]],
    )

    components["save_btn"].click(save)
