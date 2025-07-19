# Character Generation Workflow for ComfyUI with ComfyScript

<div align="center">
  <image src="./res/teaser.webp" alt="Output Example">
  <figcaption>Example of output for sample character.
  
  Base Image &rarr; Latent Upscale &rarr; Inpainting Steps (Face, Skin, Hair, Eyes) &rarr; Image Upscale &rarr; Remove Background</figcaption>
</div>

This project provides a Gradio-based interface for generating consistent character images using ComfyUI. It features a multi-step generation process with specialized controls for character attributes, style transfer, and detail enhancement.

## Key Features

<div align="center">
  <image src="./res/ui1.webp" alt="UI Screenshot with Preview">
  <figcaption>Output UI with live preview pane</figcaption>
</div>

<div align="center">
  <image src="./res/ui2.webp" alt="Options UI">
  <figcaption>Available options for generation</figcaption>
</div>

<div align="center">
  <image src="./res/ui3.webp" alt="Character Manager UI">
  <figcaption>In-build character manager</figcaption>
</div>

- 🧑‍🎨 Character-focused generation workflow
- 🎨 Style transfer with reference images
- 🧬 Multi-step detail enhancement (face, skin, hair, eyes)
- 🔍 Florence2-based image captioning (for quick prompt generation)
- 🧑‍💼 Integrated character manager for organizing presets and traits
- 🖼️ Option to inpaint existing images instead of generating from base
- ‍♂️ Multi-stage upscaling (latent and final)
- ⚙️ Customizable generation steps
- 💾 Persistent UI state across sessions
- 🖼️ Real-time preview during generation

## Requirements

- Python 3.8+
- ComfyUI installation
- Gradio
- ComfyScript (See https://github.com/Chaoses-Ib/ComfyScript for installation)

### Custom Nodes

- [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI Impact Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack)
- [ComfyUI Inspire Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)
- [ComfyUI InstantID](https://github.com/cubiq/ComfyUI_InstantID)
- [ComfyUI Essentials](https://github.com/cubiq/ComfyUI_essentials)
- [ComfyUI Segment Anything](https://github.com/storyicon/comfyui_segment_anything)
- [Comfyroll Studio](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)
- [ComfyUI Neural Network Latent Upscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale)
- [ComfyUI ReActor](https://github.com/Gourieff/ComfyUI-ReActor)
- [ComfyUI Detail Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
- [ComfyUI Florence2](https://github.com/kijai/ComfyUI-Florence2)
- [ComfyScript](https://github.com/Chaoses-Ib/ComfyScript)

### Required Models

- [Xinsir's ControlNet Union Promax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors?download=true) in `ComfyUI/models/controlnet`
- [InstantID model](https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true) in `ComfyUI/models/instantid`
- [InstantID ControlNet](https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true) in `ComfyUI/models/controlnet`
- [VIT-H SAM Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) in `ComfyUI/models/sams`
- [LCM Lora](https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true) in `ComfyUI/models/loras`
- Any of the [Turbo Lora](https://huggingface.co/shiroppo/sd_xl_turbo_lora/tree/main) in `ComfyUI/models/loras`
- [DPO Turbo Lora](https://huggingface.co/radames/sdxl-turbo-DPO-LoRA/resolve/main/pytorch_lora_weights-sdxl-turbo-comfyui.safetensors?download=true) in `ComfyUI/models/loras`

> [!NOTE]
> Make sure to place them in the proper folder. You can rename any of these files, but make sure to update the config with the correct names.

## Installation

1. Navigate to your ComfyUI folder and activate your ComfyUI virtual environment

```bash
cd path/to/comfyui
source .venv/bin/activate
```

or

```cmd
cd path/to/comfyui
.venv/Scripts/activate
```

2. Clone this repository:

```bash
cd path/to/where/you/want/to/install
git clone https://github.com/Praecordi/comfy-character-app.git
cd comfy-character-app/
```

Note: You don't need to do this in ComfyUI's custom_nodes folder.

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Create a `config.json` or copy and rename the `config.example.json`

```json
{
  "comfyui_installation": "/path/to/your/comfyui",
  "comfyui_server_url": "http://127.0.0.1:8188",
  "characters_config": "./characters.json",
  "app_constants": {
    "union_controlnet": "your_controlnet.safetensors",
    "instantid_model": "your_instantid.bin",
    "instantid_controlnet": "your_instantid_controlnet.safetensors",
    "lcm_lora": "your_lcm_lora.safetensors",
    "turbo_lora": "your_turbo_lora.safetensors",
    "dpo_turbo_lora": "your_dpo_turbo_lora.safetensors"
  }
}
```

> [!TIP]
> All the values in `app_constants` are their respective name in ComfyUI. So if your controlnets are in a subdirectory in your controlnet folder, then it. For reference, look at how it is written in a `Load ControlNet Model` node or similarly for other values.

4. Create your character definitions in your `.json` file. See `character.example.json` file for an example/template.

## Usage

Run the application

```bash
python main.py
```

### Character Configuration

The JSON file defines all your characters. See `character.example.json` for a template. The required components for each character include:

- `base`: Defining characteristic: "warrior", "man", "woman", "eldritch monster"
- `face`: Description of character's face
- `skin`: Description of character's skin
- `hair`: Description of character's hair
- `eyes`: Description of character's eyes
- `face_reference`: Path (or paths) to reference face image

Additional attributes can be added which can then be used in the prompt template.

### Workflow Steps

The current generation process (each of which can be toggled) includes:

1. Base Generation
2. Iterative Latent Upscale
3. Face Detail Enhancement
4. Skin Detail Enhancement
5. Hair Detail Enhancement
6. Eyes Detail Enhancement
7. Image Upscale
8. Background Removal

> [!NOTE]
> The app currently only works with the SDXL architecture. It also assumes that sdxl checkpoints are placed in a folder called `sdxl/` and pony checkpoints in a folder called `pony/` in your models folder.
>
> Furthermore, it detects whether the checkpoint is a Lightning, Hyper, Turbo model using its name. If the checkpoint contains "Lightning", "Hyper4S" (4-step Hyper models), "Hyper8S" (8-step Hyper models) or "Turbo" then it will be handled differently.

> [!WARNING]
> I've only been able to test this on Ubuntu

### Character Manager

The Character Manager tab provides a UI-based editor for managing your character JSON configurations. From this tab, you can:

- View and switch between characters
- Edit prompts and attributes
- Assign or preview reference images
- Save/load configurations persistently

Changes made through the Character Manager are reflected in the current session and saved to the character config file. This dramatically speeds up iteration, especially when balancing multiple characters across a dataset.

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements!
