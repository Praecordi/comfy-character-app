from ui import UI
from comfy_nodes import *

from constants import comfyui_input, comfyui_output

# Process checkpoint names (unchanged)
ui_checkpoints = [
    (str(x)[str(x).find("/") + 1 :].replace(".safetensors", ""), str(x))
    for x in Checkpoints
    if str(x).startswith("sdxl") or str(x).startswith("pony")
]

# Process resolutions (unchanged)
ui_resolutions = [
    (str(x).replace("SDXL - ", ""), str(x))
    for x in CRAspectRatio.aspect_ratio
    if str(x).startswith("SDXL - ")
]

# Process upscalers (unchanged)
ui_upscalers = [(str(x).split(".")[0], str(x)) for x in UpscaleModels]

if __name__ == "__main__":
    ui = UI(ui_checkpoints, ui_resolutions, ui_upscalers)
    demo = ui.create_ui()

    demo.launch(allowed_paths=[str(comfyui_input), str(comfyui_output)])
