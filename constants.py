import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

comfyui_installation = Path(config["comfyui_installation"])
comfyui_server_url: str = config["comfyui_server_url"]
characters_config = Path(config["characters_config"])
app_constants: Dict[str, str] = config["app_constants"]

comfyui_input = comfyui_installation / "input"
comfyui_output = comfyui_installation / "output"
comfyui_temp = comfyui_installation / "temp"

with open(characters_config, encoding="utf-8") as f:
    original_characters = json.load(f)

characters = deepcopy(original_characters)


def reset_character(key):
    global characters
    if key not in original_characters:
        del characters[key]
        return None
    else:
        characters[key] = deepcopy(original_characters[key])
        return key


def reset_all_characters():
    global characters
    characters = deepcopy(original_characters)


def save_character(key):
    global characters, original_characters
    original_characters[key] = characters[key]

    with open(characters_config, mode="w", encoding="utf-8") as f:
        json.dump(original_characters, f, indent=2)


def save_all_characters():
    global characters, original_characters
    original_characters = deepcopy(characters)

    with open(characters_config, mode="w", encoding="utf-8") as f:
        json.dump(original_characters, f, indent=2)


def delete_character(key):
    global characters, original_characters
    del characters[key]
    del original_characters[key]

    with open(characters_config, mode="w", encoding="utf-8") as f:
        json.dump(original_characters, f, indent=2)
