from typing import Dict
import gradio as gr

from comfy_nodes import queue
import constants

from ui import PREVIEW_REFRESH_RATE
from ui.runner import WorkflowRunner
from ui.utils import make_character_description, make_name, make_key

from workflow.steps import get_steps


def bind_events(
    block,
    components: Dict[str, gr.Component],
    runner: WorkflowRunner,
    checkpoints,
    resolutions,
):
    _bind_character_change(components)
    _bind_checkpoint_change(components)
    _bind_image_triggers(components)
    _bind_buttons(components, runner)
    _bind_local_storage(block, components, checkpoints, resolutions)
    _bind_preview_refresh(components, runner)


def on_character_change(character):
    if character == "Custom":
        return (
            gr.update(value="", interactive=True),
            gr.update(value="", interactive=True),
            gr.update(value="", interactive=True),
            gr.update(value="", interactive=True),
            gr.update(value=None, interactive=True),
            gr.update(value=""),
        )
    else:
        char_key = make_key(character)
        if char_key not in constants.characters:
            char_key = list(constants.characters.keys())[0]

        if isinstance(constants.characters[char_key]["face_reference"], list):
            reference = [
                str(constants.comfyui_input / ref)
                for ref in constants.characters[char_key]["face_reference"]
            ]
        else:
            reference = [
                str(
                    constants.comfyui_input
                    / constants.characters[char_key]["face_reference"]
                )
            ]

        return (
            gr.update(value=constants.characters[char_key]["face"], interactive=False),
            gr.update(value=constants.characters[char_key]["skin"], interactive=False),
            gr.update(value=constants.characters[char_key]["hair"], interactive=False),
            gr.update(value=constants.characters[char_key]["eyes"], interactive=False),
            gr.update(
                value=reference,
                interactive=False,
            ),
            gr.update(value=make_character_description(make_name(char_key))),
        )


def on_checkpoint_change(checkpoint):
    disable = any(x in checkpoint for x in ["Lightning", "Hyper4S", "Hyper8S", "Turbo"])
    if disable:
        return gr.update(value="none", interactive=False)
    else:
        return gr.update(interactive=True)


def on_image_change(image):
    if image is None:
        return gr.update(value=70, interactive=False)
    else:
        return gr.update(interactive=True)


def _bind_character_change(components: Dict[str, gr.Component]):
    input_keys = ["character"]
    output_keys = [
        "face_prompt",
        "skin_prompt",
        "hair_prompt",
        "eyes_prompt",
        "face_images",
        "character_description",
    ]

    components["character"].change(
        on_character_change,
        inputs=[components[x] for x in input_keys],
        outputs=[components[x] for x in output_keys],
    )


def _bind_checkpoint_change(components: Dict[str, gr.Component]):
    input_keys = ["checkpoint"]
    output_keys = ["fewsteplora"]

    components["checkpoint"].change(
        on_checkpoint_change,
        inputs=[components[x] for x in input_keys],
        outputs=[components[x] for x in output_keys],
    )


def _bind_image_triggers(components: Dict[str, gr.Component]):
    cn_inputs = ["controlnet_image"]
    cn_outputs = ["controlnet_strength"]

    style_inputs = ["style_image"]
    style_outputs = ["style_strength"]

    components["controlnet_image"].change(
        on_image_change,
        inputs=[components[x] for x in cn_inputs],
        outputs=[components[x] for x in cn_outputs],
    )

    components["style_image"].change(
        on_image_change,
        inputs=[components[x] for x in style_inputs],
        outputs=[components[x] for x in style_outputs],
    )

    def on_style_enable(enabled):
        if enabled:
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)

    components["enable_style"].change(
        on_style_enable,
        inputs=[components["enable_style"]],
        outputs=[components["style_prompt"]],
    )


def _bind_buttons(components: Dict[str, gr.Component], runner: WorkflowRunner):
    generate_inputs = [
        "input_image",
        "checkpoint",
        "fewsteplora",
        "resolution",
        "upscaler",
        "enable_style",
        "style_prompt",
        "use_detail_daemon",
        "process_controller",
        "latent_scale",
        "latent_adherence",
        "image_scale",
        "image_adherence",
        "base_seed",
        "perturb_seed",
        "controlnet_image",
        "controlnet_strength",
        "style_image",
        "style_strength",
        "face_prompt",
        "skin_prompt",
        "hair_prompt",
        "eyes_prompt",
        "face_images",
        "swap_method",
        "positive_prompt",
        "negative_prompt",
        "character",
    ]
    generate_outputs = ["output", "output_text"]

    generate_caption_inputs = ["input_image"]
    generate_caption_outputs = ["positive_prompt"]

    async def generate(*args):
        async for res in runner.generate(dict(zip(generate_inputs, args))):
            yield res

    async def generate_caption(*args):
        return await runner.generate_caption(dict(zip(generate_caption_inputs, args)))

    components["generate"].click(
        generate,
        inputs=[components[x] for x in generate_inputs],
        outputs=[components[x] for x in generate_outputs],
    )

    components["interrupt"].click(runner.interrupt)

    components["auto_caption"].click(
        generate_caption,
        inputs=[components[x] for x in generate_caption_inputs],
        outputs=[components[x] for x in generate_caption_outputs],
    )


def _bind_local_storage(
    block,
    components: Dict[str, gr.Component],
    checkpoints,
    resolutions,
):
    persist_components = [
        "checkpoint",
        "fewsteplora",
        "resolution",
        "upscaler",
        "enable_style",
        "style_prompt",
        "use_detail_daemon",
        "process_controller",
        "latent_scale",
        "latent_adherence",
        "image_scale",
        "image_adherence",
        "base_seed",
        "perturb_seed",
        "positive_prompt",
        "negative_prompt",
        "character",
        "swap_method",
    ]
    output_components = [
        "input_image",
        "checkpoint",
        "fewsteplora",
        "resolution",
        "upscaler",
        "enable_style",
        "style_prompt",
        "use_detail_daemon",
        "process_controller",
        "latent_scale",
        "latent_adherence",
        "image_scale",
        "image_adherence",
        "base_seed",
        "perturb_seed",
        "controlnet_image",
        "controlnet_strength",
        "style_image",
        "style_strength",
        "face_prompt",
        "skin_prompt",
        "hair_prompt",
        "eyes_prompt",
        "face_images",
        "swap_method",
        "positive_prompt",
        "negative_prompt",
        "character",
        "character_description",
    ]

    def save_state(*args):
        return dict(zip(persist_components, args))

    def load_state(state):
        state = state or {}
        checkpoint = state.get("checkpoint", checkpoints[0][1])
        character = state.get(
            "character", list(constants.characters.keys())[0].capitalize()
        )

        char_tuple = on_character_change(character)
        disable = any(
            x in checkpoint for x in ["Lightning", "Hyper4S", "Hyper8S", "Turbo"]
        )
        fewsteplora = (
            state.get("fewsteplora", "none")
            if not disable
            else gr.update(value="none", interactive=not disable)
        )
        return [
            None,
            checkpoint,
            fewsteplora,
            state.get("resolution", resolutions[0][1]),
            state.get("upscaler", "None"),
            state.get("enable_style", True),
            state.get("style_prompt", ""),
            state.get("use_detail_daemon", False),
            state.get("process_controller", get_steps()),
            state.get("latent_scale", 1.6),
            state.get("latent_adherence", 0.2),
            state.get("image_scale", 1.3),
            state.get("image_adherence", 0.9),
            state.get("base_seed", -1),
            state.get("perturb_seed", -1),
            gr.update(),
            gr.update(value=70, interactive=False),
            gr.update(),
            gr.update(value=70, interactive=False),
            char_tuple[0],
            char_tuple[1],
            char_tuple[2],
            char_tuple[3],
            char_tuple[4],
            state.get("swap_method", "instantid"),
            state.get("positive_prompt", ""),
            state.get("negative_prompt", ""),
            character,
            make_character_description(character),
        ]

    for key in persist_components:
        components[key].change(
            save_state,
            inputs=[components[x] for x in persist_components],
            outputs=[components["browser_state"]],
        )

    block.load(
        load_state,
        inputs=[components["browser_state"]],
        outputs=[components[x] for x in output_components],
    )
    block.unload(lambda: queue.cancel_all())


def _bind_preview_refresh(components: Dict[str, gr.Component], runner: WorkflowRunner):
    def get_preview():
        image, msg = runner.get_preview()

        if msg == "stopped":
            return components["preview"]
        else:
            if image:
                return gr.update(value=image, visible=True)
            else:
                return gr.update()

    components["preview"].attach_load_event(get_preview, every=PREVIEW_REFRESH_RATE)
