from typing import Dict
from copy import deepcopy
import shutil
from pathlib import Path
import gradio as gr

import constants
from ui.utils import make_key, make_name


def _make_fields(character):
    char = deepcopy(character)

    for key in ["face_reference", "face", "skin", "hair", "eyes"]:
        del char[key]

    return char


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
    char_key = make_key(character)
    res = constants.reset_character(char_key)

    if res:
        return gr.update(value=character), *on_character_change(character)
    else:
        new_choices = [make_name(char) for char in constants.characters.keys()]

        return gr.update(
            value=new_choices[0], choices=new_choices
        ), *on_character_change(new_choices[0])


def save(character, face_gallery):
    if not character or character == "":
        gr.Warning("Please select a character first")
        return gr.update()

    char_key = make_key(character)
    if char_key not in constants.characters:
        gr.Warning(f"Character {character} not found")
        return gr.update()

    processed_files = []
    face_refs = []

    if face_gallery:
        for file_path in face_gallery:
            src = Path(file_path[0])

            if src.parent == constants.comfyui_input:
                processed_files.append(str(src))
                face_refs.append(src.name)
                continue

            dest_name = src.name
            dest_path = constants.comfyui_input / dest_name
            counter = 1

            while dest_path.exists():
                stem = src.stem
                suffix = src.suffix
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = constants.comfyui_input / dest_name
                counter += 1

            shutil.move(str(src), str(dest_path))
            processed_files.append(str(dest_path))
            face_refs.append(dest_name)

        constants.characters[char_key]["face_reference"] = face_refs
    else:
        constants.characters[char_key]["face_reference"] = []

    constants.save_character(char_key)

    return processed_files


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


def remove_character(character):
    char_key = make_key(character)

    constants.delete_character(char_key)

    new_choices = [
        char.replace("_", " ").title() for char in constants.characters.keys()
    ]

    return gr.update(value=new_choices[0], choices=new_choices)


def add_field(attribute, character):
    char_key = make_key(character)
    attr_key = make_key(attribute)

    if char_key not in constants.characters:
        gr.Warning(f"Character {character} does not exist...")
        return gr.update(value=""), gr.update("")

    if attr_key in constants.characters[char_key]:
        gr.Warning(f"Attribute {attribute} already exists...")
        return gr.update(value=""), gr.update()

    if attr_key == "":
        gr.Warning("Please provide non-empty field")
        return gr.update(value=""), gr.update()

    constants.characters[char_key][attr_key] = ""

    fields = _make_fields(constants.characters[char_key])

    return gr.update(value=""), fields


def delete_field(attribute, character):
    char_key = make_key(character)
    attr_key = make_key(attribute)

    if char_key not in constants.characters:
        gr.Warning(f"Character {character} doesn't exist")
        return gr.update()

    if attr_key in constants.characters[char_key]:
        del constants.characters[char_key][attr_key]

    fields = _make_fields(constants.characters[char_key])

    return fields


def update_field(attribute, value, character):
    char_key = make_key(character)

    if char_key not in constants.characters:
        gr.Warning(f"Character {character} not found")
        return

    attr_key = make_key(attribute)
    constants.characters[char_key][attr_key] = value


def bind_events(block, components):
    _bind_character_change(components)
    _bind_buttons(components)
    _bind_attribute_changes(components)

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

    components["save_btn"].click(
        save,
        inputs=[components["character_select"], components["face_images"]],
        outputs=[components["face_images"]],
    )

    components["new_character"].submit(
        add_character,
        inputs=[components["new_character"]],
        outputs=[components["new_character"], components["character_select"]],
    )

    components["add_character_btn"].click(
        add_character,
        inputs=[components["new_character"]],
        outputs=[components["new_character"], components["character_select"]],
    )

    components["remove_character_btn"].click(
        remove_character,
        inputs=[components["character_select"]],
        outputs=[components["character_select"]],
    )

    components["new_field"].submit(
        add_field,
        inputs=[components["new_field"], components["character_select"]],
        outputs=[components["new_field"], components["current_fields"]],
    )

    components["add_field_btn"].click(
        add_field,
        inputs=[components["new_field"], components["character_select"]],
        outputs=[components["new_field"], components["current_fields"]],
    )


def _bind_attribute_changes(components: Dict[str, gr.Component]):
    components["face_prompt"].change(
        lambda value, char: update_field("face", value, char),
        inputs=[components["face_prompt"], components["character_select"]],
    )

    components["skin_prompt"].change(
        lambda value, char: update_field("skin", value, char),
        inputs=[components["skin_prompt"], components["character_select"]],
    )

    components["hair_prompt"].change(
        lambda value, char: update_field("hair", value, char),
        inputs=[components["hair_prompt"], components["character_select"]],
    )

    components["eyes_prompt"].change(
        lambda value, char: update_field("eyes", value, char),
        inputs=[components["eyes_prompt"], components["character_select"]],
    )
