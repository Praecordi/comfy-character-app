import json
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict

with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

comfyui_installation = Path(config["comfyui_installation"])
comfyui_server_url: str = config["comfyui_server_url"]
characters_config = Path(config["characters_config"])
app_constants: Dict[str, str] = config["app_constants"]

comfyui_input = comfyui_installation / "input"
comfyui_output = comfyui_installation / "output"

characters = OmegaConf.load(characters_config)