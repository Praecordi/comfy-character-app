import gradio as gr

from ui import UI
from comfy_nodes import Checkpoints, CRAspectRatio, UpscaleModels, Loras

from constants import comfyui_input, comfyui_output, comfyui_temp

ui_checkpoints = [
    (str(x).replace(".safetensors", ""), str(x))
    for x in Checkpoints
    if str(x).lower().startswith("sdxl") or str(x).lower().startswith("pony")
]

ui_loras = [
    (str(x).replace(".safetensors", ""), str(x))
    for x in Loras
    if str(x).lower().startswith("sdxl") or str(x).lower().startswith("pony")
]

ui_resolutions = [
    (str(x).replace("SDXL - ", ""), str(x))
    for x in CRAspectRatio.aspect_ratio
    if str(x).startswith("SDXL - ")
]

ui_upscalers = [(str(x).split(".")[0], str(x)) for x in UpscaleModels]

if __name__ == "__main__":
    gr.set_static_paths([comfyui_input, comfyui_output, comfyui_temp])

    ui = UI(ui_checkpoints, ui_loras, ui_resolutions, ui_upscalers)
    demo = ui.create_ui()

    demo.launch(allowed_paths=[str(comfyui_input), str(comfyui_output)])
