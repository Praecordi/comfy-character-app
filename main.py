from ui import UI
from constants import checkpoints, resolutions, upscalers, comfyui_input, comfyui_output

# Process checkpoint names (unchanged)
ui_checkpoints = [
    (x[x.find("/") + 1 :].replace(".safetensors", ""), x)
    for x in checkpoints
    if x.startswith("sdxl") or x.startswith("pony")
]

# Process resolutions (unchanged)
ui_resolutions = [
    (x.replace("SDXL - ", ""), x) for x in resolutions if x.startswith("SDXL - ")
]

# Process upscalers (unchanged)
ui_upscalers = [(x.split(".")[0], x) for x in upscalers]

if __name__ == "__main__":
    ui = UI(ui_checkpoints, ui_resolutions, ui_upscalers)
    demo = ui.create_ui()

    demo.launch(allowed_paths=[str(comfyui_input), str(comfyui_output)])
