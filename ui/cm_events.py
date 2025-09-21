from typing import Dict
from copy import deepcopy
import shutil
from pathlib import Path
import gradio as gr

from constants import characters, comfyui_input, save_character
from utils import make_key, make_name


def _make_fields(character):
    char = deepcopy(character)

    for key in ["face_reference", "face", "skin", "hair", "eyes"]:
        del char[key]

    return char


def load_initial():
    character_choices = [make_name(char) for char in characters.keys()]

    return (
        character_choices[0],
        characters,
        *on_character_change(character_choices[0], characters),
    )


def on_character_change(character: str, character_state: dict):
    char_key = make_key(character)

    char_dict = deepcopy(character_state[char_key])

    face_images = char_dict.pop("face_reference", [])
    if isinstance(face_images, list):
        face_images = [str(comfyui_input / ref) for ref in face_images]
    else:
        face_images = [str(comfyui_input / face_images)]

    base_prompt = char_dict.pop("base", "")
    face_prompt = char_dict.pop("face", "")
    skin_prompt = char_dict.pop("skin", "")
    hair_prompt = char_dict.pop("hair", "")
    eyes_prompt = char_dict.pop("eyes", "")

    return (
        char_dict,
        face_images,
        base_prompt,
        face_prompt,
        skin_prompt,
        hair_prompt,
        eyes_prompt,
    )


def reset(character, current_character_state, character_state):
    char_key = make_key(character)
    if char_key not in character_state:
        del current_character_state[char_key]
    else:
        current_character_state[char_key] = deepcopy(character_state[char_key])

    if char_key in current_character_state:
        return (
            gr.update(value=character),
            current_character_state,
            *on_character_change(character, current_character_state),
        )
    else:
        new_choices = [make_name(char) for char in character_state.keys()]

        return (
            gr.update(value=new_choices[0], choices=new_choices),
            current_character_state,
            *on_character_change(new_choices[0], current_character_state),
        )


def save(character, current_character_state, character_state, face_gallery):
    if not character or character == "":
        gr.Warning("Please select a character first")
        return gr.update(), current_character_state

    char_key = make_key(character)
    if char_key not in current_character_state:
        gr.Warning(f"Character {character} not found")
        return gr.update(), current_character_state

    processed_files = []
    face_refs = []

    if face_gallery:
        for file_path in face_gallery:
            src = Path(file_path[0])

            if src.parent.absolute == comfyui_input.absolute:
                processed_files.append(str(src))
                face_refs.append(src.name)
                continue

            dest_name = src.name
            dest_path = comfyui_input / dest_name
            counter = 1

            while not dest_path.exists():
                stem = src.stem
                suffix = src.suffix
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = comfyui_input / dest_name
                counter += 1

            shutil.move(str(src), str(dest_path))
            processed_files.append(str(dest_path))
            face_refs.append(dest_name)

        current_character_state[char_key]["face_reference"] = face_refs
    else:
        current_character_state[char_key]["face_reference"] = []

    character_state[char_key] = current_character_state[char_key]

    save_character(character_state)

    return processed_files, current_character_state, character_state


def add_character(character, current_character_state):
    char_key = make_key(character)

    if char_key in current_character_state.keys():
        gr.Warning("Character already exists", duration=10)
        return gr.update(), gr.update()

    if char_key == "":
        gr.Warning("Please provide a character name")
        return gr.update(), gr.update()

    char_dict = {
        "face_reference": [],
        "base": "",
        "face": "",
        "skin": "",
        "hair": "",
        "eyes": "",
    }

    current_character_state[char_key] = char_dict

    new_choices = [
        char.replace("_", " ").title() for char in current_character_state.keys()
    ]

    return (
        gr.update(value=""),
        current_character_state,
        gr.update(value=make_name(char_key), choices=new_choices),
    )


def remove_character(character, current_character_state, character_state):
    char_key = make_key(character)

    del current_character_state[char_key]
    del character_state[char_key]

    save_character(character_state)

    new_choices = [char.replace("_", " ").title() for char in character_state.keys()]

    return (
        gr.update(value=new_choices[0], choices=new_choices),
        current_character_state,
        character_state,
    )


def add_field(attribute, character, current_character_state):
    char_key = make_key(character)
    attr_key = make_key(attribute)

    if char_key not in current_character_state:
        gr.Warning(f"Character {character} does not exist...")
        return gr.update(value=""), gr.update(""), current_character_state

    if attr_key in current_character_state:
        gr.Warning(f"Attribute {attribute} already exists...")
        return gr.update(value=""), gr.update(), current_character_state

    if attr_key == "":
        gr.Warning("Please provide non-empty field")
        return gr.update(value=""), gr.update(), current_character_state

    current_character_state[char_key][attr_key] = ""

    fields = _make_fields(current_character_state[char_key])

    return gr.update(value=""), fields, current_character_state


def delete_field(attribute, character, current_character_state):
    char_key = make_key(character)
    attr_key = make_key(attribute)

    if char_key not in current_character_state:
        gr.Warning(f"Character {character} doesn't exist")
        return gr.update(), current_character_state

    if attr_key in current_character_state[char_key]:
        del current_character_state[char_key][attr_key]

    fields = _make_fields(current_character_state[char_key])

    return fields, current_character_state


def update_field(attribute, value, character, current_character_state):
    char_key = make_key(character)

    if char_key not in current_character_state:
        gr.Warning(f"Character {character} not found")
        return

    attr_key = make_key(attribute)
    current_character_state[char_key][attr_key] = value

    return current_character_state


def bind_events(block, components):
    _bind_character_change(components)
    _bind_buttons(components)
    _bind_attribute_changes(components)

    initial_output_keys = [
        "character_select",
        "current_character_state",
        "current_fields",
        "cm_face_images",
        "cm_base_prompt",
        "cm_face_prompt",
        "cm_skin_prompt",
        "cm_hair_prompt",
        "cm_eyes_prompt",
    ]

    block.load(load_initial, outputs=[components[x] for x in initial_output_keys])


def _bind_character_change(components: Dict[str, gr.Component]):
    input_keys = ["character_select", "current_character_state"]
    output_keys = [
        "current_fields",
        "cm_face_images",
        "cm_base_prompt",
        "cm_face_prompt",
        "cm_skin_prompt",
        "cm_hair_prompt",
        "cm_eyes_prompt",
    ]

    components["character_select"].change(
        on_character_change,
        inputs=[components[x] for x in input_keys],
        outputs=[components[x] for x in output_keys],
    )

    components["character_state"].change(
        on_character_change,
        inputs=[components[x] for x in input_keys],
        outputs=[components[x] for x in output_keys],
    )


def _bind_buttons(components: Dict[str, gr.Component]):
    reset_output_keys = [
        "character_select",
        "current_character_state",
        "current_fields",
        "cm_face_images",
        "cm_base_prompt",
        "cm_face_prompt",
        "cm_skin_prompt",
        "cm_hair_prompt",
        "cm_eyes_prompt",
    ]
    components["reset_btn"].click(
        reset,
        inputs=[
            components["character_select"],
            components["current_character_state"],
            components["character_state"],
        ],
        outputs=[components[x] for x in reset_output_keys],
    )

    components["save_btn"].click(
        save,
        inputs=[
            components["character_select"],
            components["current_character_state"],
            components["character_state"],
            components["cm_face_images"],
        ],
        outputs=[
            components["cm_face_images"],
            components["current_character_state"],
            components["character_state"],
        ],
    )

    components["new_character"].submit(
        add_character,
        inputs=[components["new_character"], components["current_character_state"]],
        outputs=[
            components["new_character"],
            components["current_character_state"],
            components["character_select"],
        ],
    )

    components["add_character_btn"].click(
        add_character,
        inputs=[components["new_character"], components["current_character_state"]],
        outputs=[
            components["new_character"],
            components["current_character_state"],
            components["character_select"],
        ],
    )

    components["remove_character_btn"].click(
        remove_character,
        inputs=[
            components["character_select"],
            components["current_character_state"],
            components["character_state"],
        ],
        outputs=[
            components["character_select"],
            components["current_character_state"],
            components["character_state"],
        ],
    )

    components["new_field"].submit(
        add_field,
        inputs=[
            components["new_field"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[
            components["new_field"],
            components["current_fields"],
            components["current_character_state"],
        ],
    )

    components["add_field_btn"].click(
        add_field,
        inputs=[
            components["new_field"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[
            components["new_field"],
            components["current_fields"],
            components["current_character_state"],
        ],
    )


def _bind_attribute_changes(components: Dict[str, gr.Component]):
    components["cm_base_prompt"].change(
        lambda value, char, char_state: update_field("base", value, char, char_state),
        inputs=[
            components["cm_base_prompt"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[components["current_character_state"]],
    )

    components["cm_face_prompt"].change(
        lambda value, char, char_state: update_field("face", value, char, char_state),
        inputs=[
            components["cm_face_prompt"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[components["current_character_state"]],
    )

    components["cm_skin_prompt"].change(
        lambda value, char, char_state: update_field("skin", value, char, char_state),
        inputs=[
            components["cm_skin_prompt"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[components["current_character_state"]],
    )

    components["cm_hair_prompt"].change(
        lambda value, char, char_state: update_field("hair", value, char, char_state),
        inputs=[
            components["cm_hair_prompt"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[components["current_character_state"]],
    )

    components["cm_eyes_prompt"].change(
        lambda value, char, char_state: update_field("eyes", value, char, char_state),
        inputs=[
            components["cm_eyes_prompt"],
            components["character_select"],
            components["current_character_state"],
        ],
        outputs=[components["current_character_state"]],
    )
