import json
from pathlib import Path
from omegaconf import OmegaConf

with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

comfyui_installation = Path(config["comfyui_installation"])
comfyui_server_url = config["comfyui_server_url"]
characters_config = config["characters_config"]
app_constants = config["app_constants"]

comfyui_input = comfyui_installation / "input"
comfyui_output = comfyui_installation / "output"

characters = OmegaConf.load(characters_config)

from comfy_script.runtime import load

load(comfyui_server_url)
from comfy_script.runtime.nodes import Checkpoints, CRAspectRatio, UpscaleModels


checkpoints = {str(x): x for x in Checkpoints}
resolutions = {str(x): x for x in CRAspectRatio.aspect_ratio}
upscalers = {str(x): x for x in UpscaleModels}
