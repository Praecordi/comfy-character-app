from typing import Dict
import gradio as gr

from comfy_nodes import queue
from constants import characters, comfyui_input

from ui import PREVIEW_REFRESH_RATE
from ui.runner import WorkflowRunner
from ui.utils import make_character_description

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
            gr.update(value=None, interactive=True),
            gr.update(value=""),
        )
    else:
        char_key = character.lower()
        if isinstance(characters[char_key]["face_reference"], list):
            reference = [
                str(comfyui_input / ref)
                for ref in characters[char_key]["face_reference"]
            ]
        else:
            reference = [str(comfyui_input / characters[char_key]["face_reference"])]

        return (
            gr.update(value=characters[char_key]["face"], interactive=False),
            gr.update(value=characters[char_key]["hair"], interactive=False),
            gr.update(value=characters[char_key]["eyes"], interactive=False),
            gr.update(
                value=reference,
                interactive=False,
            ),
            gr.update(value=make_character_description(character)),
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
        "base_seed",
        "perturb_seed",
        "controlnet_image",
        "controlnet_strength",
        "style_image",
        "style_strength",
        "face_prompt",
        "hair_prompt",
        "eyes_prompt",
        "face_images",
        "swap_method",
        "positive_prompt",
        "negative_prompt",
        "character",
    ]
    generate_outputs = ["output", "output_text"]

    def generate(*args):
        for res in runner.generate(dict(zip(generate_inputs, args))):
            yield res

    components["generate"].click(
        generate,
        inputs=[components[x] for x in generate_inputs],
        outputs=[components[x] for x in generate_outputs],
    )

    components["interrupt"].click(runner.interrupt)


def _bind_local_storage(
    block,
    components: Dict[str, gr.Component],
    checkpoints,
    resolutions,
):
    persist_components = [
        "input_image",
        "checkpoint",
        "fewsteplora",
        "resolution",
        "upscaler",
        "enable_style",
        "style_prompt",
        "use_detail_daemon",
        "process_controller",
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
        "base_seed",
        "perturb_seed",
        "controlnet_image",
        "controlnet_strength",
        "style_image",
        "style_strength",
        "face_prompt",
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
        character = state.get("character", list(characters.keys())[0].capitalize())

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
            state.get("input_image", None),
            checkpoint,
            fewsteplora,
            state.get("resolution", resolutions[0][1]),
            state.get("upscaler", "None"),
            state.get("enable_style", True),
            state.get("style_prompt", ""),
            state.get("use_detail_daemon", False),
            state.get("process_controller", get_steps()),
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
