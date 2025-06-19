# Character Generation Workflow for ComfyUI with ComfyScript

This project provides a Gradio-based interface for generating consistent character images using ComfyUI. It features a multi-step generation process with specialized controls for character attributes, style transfer, and detail enhancement.

## Key Features

- 🧑‍🎨 Character-focused generation workflow
- 🎨 Style transfer with reference images
- 🔍 Detailed enhancement of facial features, hair, and eyes
- ‍♂️ Multi-stage upscaling (latent and image)
- ⚙️ Customizable generation steps
- 🖼️ Real-time preview during generation
- 💾 Persistent UI state across sessions

## Requirements

- Python 3.8+
- ComfyUI installation
- Gradio
- ComfyScript (See https://github.com/Chaoses-Ib/ComfyScript for installation)
- OmegaConf

### Custom Nodes

- ComfyUI Impact Pack (https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- ComfyUI Impact Subpack (https://github.com/ltdrdata/ComfyUI-Impact-Subpack)
- ComfyUI Inspire Pack (https://github.com/ltdrdata/ComfyUI-Inspire-Pack)
- ComfyUI InstantID (https://github.com/cubiq/ComfyUI_InstantID)
- ComfyUI Essentials (https://github.com/cubiq/ComfyUI_essentials)
- Comfyroll Studio (https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)
- ComfyUI Neural Network Latent Upscale (https://github.com/Ttl/ComfyUi_NNLatentUpscale)
- Facerestore CF (https://github.com/mav-rik/facerestore_cf)
- Detail Daemon (https://github.com/Jonseed/ComfyUI-Detail-Daemon)

## Installation

1. Activate your ComfyUI virtual environment

```bash
source .venv/bin/activate
```

or

```cmd
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
    "dpo_turbo_lora": "your_dpo_turbo_lora.safetensors",
    "hair_seg_model": "your_hair_seg_model.pt"
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

- `face_prompt`: Description of character's face
- `hair_prompt`: Description of character's hair
- `eyes_prompt`: Description of character's eyes
- `face-reference`: Path to reference face image

Additional attributes can be added which can then be used in the prompt template.

### Workflow Steps

The current generation process (each of which can be toggled) includes:

1. Base Generation
2. Iterative Latent Upscale
3. Face Detail Enhancement
4. Hair Detail Enhancement
5. Eyes Detail Enhancement
6. Image Upscale
7. Background Removal

> [!NOTE]
> The app currently only works with the SDXL architecture. It also assumes that sdxl checkpoints are placed in a folder called `sdxl/` and pony checkpoints in a folder called `pony/` in your models folder.
>
> Furthermore, it detects whether the checkpoint is a Lightning, Hyper, Turbo model using its name. If the checkpoint contains "Lightning", "Hyper4S" (4-step Hyper models), "Hyper8S" (8-step Hyper models) or "Turbo" then it will be handled differently.

> [!WARNING]
> I've only been able to test this on Ubuntu

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements!
