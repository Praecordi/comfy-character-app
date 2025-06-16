from typing import List, Tuple

import gradio as gr
from comfy_script.runtime.data import ImageBatchResult
import queue as std_queue
from threading import Thread, Event

from constants import characters, comfyui_input
from workflow import CharacterWorkflow


def make_output_text(resolved_prompts, base_seed, perturb_seed):
    text = """Base Seed: {bseed}
Perturb Seed: {pseed}
Positive Prompt: {pprompt}
Negative Prompt: {nprompt}
Hair Prompt: {hprompt}
Eyes Prompt: {eprompt}""".format(
        bseed=base_seed,
        pseed=perturb_seed,
        pprompt=resolved_prompts["positive"],
        nprompt=resolved_prompts["negative"],
        hprompt=resolved_prompts["hair"],
        eprompt=resolved_prompts["eyes"],
    )

    return text


def make_character_description(character):
    char_key = character.lower()
    char_dict = characters[char_key]

    desc = "\n".join([f"- **{{{key}}}**" for key in char_dict.keys()])

    return desc


class UI:
    def __init__(
        self,
        checkpoints: List[Tuple[str, str]],
        resolutions: List[Tuple[str, str]],
        upscalers: List[Tuple[str, str]],
    ):
        self.checkpoints = checkpoints
        self.resolutions = resolutions
        self.upscalers = upscalers
        self.components = {}

        self.preview_queue = std_queue.Queue()
        self.stop_event = Event()
        self.preview_process_thread = None

    def create_ui(self):
        with gr.Blocks(
            title="Character Generator", head_paths=["attention.html"]
        ) as demo:
            gr.Markdown("""# Character Generator""")
            with gr.Column():
                self._create_output_panel()
                with gr.Row():
                    with gr.Column(scale=2):
                        self._create_advanced_settings()
                    with gr.Column(scale=3):
                        self._create_main_content()

            self._setup_event_handlers(demo)

        return demo

    def _create_advanced_settings(self):
        with gr.Group():
            self.checkpoint = gr.Dropdown(
                label="Checkpoint",
                choices=self.checkpoints,
            )

            self.fewsteplora = gr.Radio(
                label="Few Step LoRA", choices=["none", "lcm", "turbo", "dpo_turbo"]
            )

            self.resolution = gr.Dropdown(
                label="Resolution",
                choices=self.resolutions,
            )

            self.upscaler = gr.Dropdown(label="Upscaler", choices=self.upscalers)

            self.style_prompt = gr.Textbox(
                label="Style Prompt",
                lines=4,
                placeholder="Enter style prompt to append to positive prompt...",
                interactive=True,
            )

            with gr.Row():
                with gr.Column():
                    self.cn_image = gr.Image(
                        label="Control Net Image",
                        type="filepath",
                        sources=["upload"],
                        show_download_button=False,
                        show_fullscreen_button=False,
                        show_share_button=False,
                    )
                    self.cn_strength = gr.Slider(
                        minimum=0, maximum=100, step=1, label="Control Net Strength"
                    )

                with gr.Column():
                    self.style_image = gr.Image(
                        label="Style Image",
                        type="filepath",
                        sources=["upload"],
                        show_download_button=False,
                        show_fullscreen_button=False,
                        show_share_button=False,
                    )

                    self.style_strength = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        label="Style Strength",
                    )

    def _create_custom_character_settings(self):
        with gr.Accordion("Custom Character", open=False):
            with gr.Row():
                with gr.Column():
                    self.hair_prompt = gr.Textbox(
                        label="Hair Prompt",
                        lines=4,
                        placeholder="Enter hair prompt here...",
                        interactive=True,
                        elem_classes=["attention-editable"],
                    )

                    self.eyes_prompt = gr.Textbox(
                        label="Eyes Prompt",
                        lines=4,
                        placeholder="Enter eyes prompt here",
                        interactive=True,
                        elem_classes=["attention-editable"],
                    )

                self.face_image = gr.Image(
                    label="Face Image",
                    type="filepath",
                    interactive=True,
                    sources=["upload"],
                    show_download_button=False,
                    show_fullscreen_button=False,
                    show_share_button=False,
                )

    def _create_output_panel(self):
        with gr.Accordion("Output", open=False):
            with gr.Group():
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        self.output_gallery = gr.Gallery(
                            label="Output Images",
                            format="png",
                            columns=5,
                            object_fit="contain",
                            preview=True,
                            interactive=False,
                            show_label=True,
                            height="auto",
                            show_share_button=False,
                            show_download_button=True,
                            show_fullscreen_button=True,
                        )

                        self.output_text = gr.Textbox(
                            label="Resolved Text", interactive=False, lines=7
                        )

                    self.preview = gr.Image(
                        label="Preview",
                        interactive=False,
                        visible=False,
                        scale=1,
                    )

    def _create_main_content(self):
        self.positive_prompt = gr.Textbox(
            label="Positive Prompt",
            lines=4,
            placeholder="Enter positive prompt here...",
            elem_classes=["attention-editable"],
        )

        self.negative_prompt = gr.Textbox(
            label="Negative Prompt",
            lines=4,
            placeholder="Enter negative prompt here...",
            elem_classes=["attention-editable"],
        )

        character_choices = [char.capitalize() for char in characters.keys()] + [
            "Custom"
        ]

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    self.character = gr.Dropdown(
                        label="Character", choices=character_choices, scale=2
                    )
                    with gr.Accordion(label="Available Keys", open=False):
                        self.character_description = gr.Markdown(padding=True)

            with gr.Column(scale=3):
                self._create_custom_character_settings()

        self.process_controller = gr.CheckboxGroup(
            label="Process Controllers",
            choices=CharacterWorkflow.get_steps(),
        )

        with gr.Row():
            self.base_seed = gr.Number(
                label="Base Seed",
                precision=0,
            )

            self.perturb_seed = gr.Number(
                label="Perturb Seed",
                precision=0,
            )

        with gr.Row():
            self.generate_btn = gr.Button("Generate")
            self.interrupt_btn = gr.Button("Interrupt")
            self.generate_all_btn = gr.Button("Generate All")

    def _setup_event_handlers(self, block):
        self.character.change(
            self.on_character_change,
            inputs=[self.character],
            outputs=[
                self.hair_prompt,
                self.eyes_prompt,
                self.face_image,
                self.character_description,
            ],
        )

        self.checkpoint.change(
            self.on_checkpoint_change,
            inputs=[self.checkpoint],
            outputs=[self.fewsteplora],
        )

        self.cn_image.change(
            self.on_image_change, inputs=[self.cn_image], outputs=[self.cn_strength]
        )

        self.style_image.change(
            self.on_image_change,
            inputs=[self.style_image],
            outputs=[self.style_strength],
        )

        all_input_components = [
            self.checkpoint,
            self.fewsteplora,
            self.resolution,
            self.upscaler,
            self.style_prompt,
            self.process_controller,
            self.base_seed,
            self.perturb_seed,
            self.cn_image,
            self.cn_strength,
            self.style_image,
            self.style_strength,
            self.hair_prompt,
            self.eyes_prompt,
            self.face_image,
            self.positive_prompt,
            self.negative_prompt,
            self.character,
        ]

        self.generate_btn.click(
            self.generate,
            inputs=all_input_components,
            outputs=[self.output_gallery, self.output_text],
        )

        self.interrupt_btn.click(self.interrupt)

        self.generate_all_btn.click(self.generate_all)

        self.local_storage = gr.BrowserState(
            storage_key="ccw-ui-state", secret="ccw_secret"
        )

        persist_components = [
            self.checkpoint,
            self.fewsteplora,
            self.resolution,
            self.upscaler,
            self.style_prompt,
            self.process_controller,
            self.base_seed,
            self.perturb_seed,
            self.positive_prompt,
            self.negative_prompt,
            self.character,
        ]

        def save_state(*args):
            return {
                "checkpoint": args[0],
                "fewsteplora": args[1],
                "resolution": args[2],
                "upscaler": args[3],
                "style_prompt": args[4],
                "process_controller": args[5],
                "base_seed": args[6],
                "perturb_seed": args[7],
                "positive_prompt": args[8],
                "negative_prompt": args[9],
                "character": args[10],
            }

        for comp in persist_components:
            comp.change(
                save_state,
                inputs=persist_components,
                outputs=self.local_storage,
            )

        def load_state(state):
            state = state or {}
            checkpoint = state.get("checkpoint", self.checkpoints[0][1])
            character = state.get("character", list(characters.keys())[0].capitalize())

            char_tuple = self.on_character_change(character)
            disable = any(
                x in checkpoint for x in ["Lightning", "Hyper4S", "Hyper8S", "Turbo"]
            )
            fewsteplora = (
                state.get("fewsteplora", "")
                if not disable
                else gr.update(value="none", interactive=not disable)
            )
            return [
                checkpoint,
                fewsteplora,
                state.get("resolution", self.resolutions[0][1]),
                state.get("upscaler", self.upscalers[0][1]),
                state.get("style_prompt", ""),
                state.get("process_controller", CharacterWorkflow.get_steps()),
                state.get("base_seed", -1),
                state.get("perturb_seed", -1),
                gr.update(),
                gr.update(value=70, interactive=False),
                gr.update(),
                gr.update(value=70, interactive=False),
                char_tuple[0],
                char_tuple[1],
                char_tuple[2],
                state.get("positive_prompt", ""),
                state.get("negative_prompt", ""),
                character,
                make_character_description(character),
            ]

        other_components = [self.character_description]

        block.load(
            load_state,
            inputs=self.local_storage,
            outputs=all_input_components + other_components,
        )
        block.unload(lambda: CharacterWorkflow.cancel_all())

        self.preview.attach_load_event(self.get_next_preview, every=0.5)

    def start_preview_processor(self):
        self.stop_event.clear()
        self.preview_process_thread = Thread(target=None, daemon=True)
        self.preview_process_thread.start()

    def stop_preview_processor(self):
        self.stop_event.set()
        if self.preview_process_thread:
            self.preview_process_thread.join(timeout=1.0)

    def get_next_preview(self):
        if self.stop_event.is_set():
            return self.preview
        try:
            image = self.preview_queue.get_nowait()
            return gr.update(value=image, visible=True)
        except std_queue.Empty:
            return gr.update()

    def on_character_change(self, character):
        if character == "Custom":
            return (
                gr.update(value="", interactive=True),
                gr.update(value="", interactive=True),
                gr.update(value=None, interactive=True),
                gr.update(value=""),
            )
        else:
            char_key = character.lower()
            return (
                gr.update(value=characters[char_key]["hair"], interactive=False),
                gr.update(value=characters[char_key]["eyes"], interactive=False),
                gr.update(
                    value=str(comfyui_input / characters[char_key]["face"]),
                    interactive=False,
                ),
                gr.update(value=make_character_description(character)),
            )

    def on_checkpoint_change(self, checkpoint):
        disable = any(
            x in checkpoint for x in ["Lightning", "Hyper4S", "Hyper8S", "Turbo"]
        )
        if disable:
            return gr.update(value="none", interactive=False)
        else:
            return gr.update(interactive=True)

    def on_image_change(self, image):
        if image is None:
            return gr.update(value=70, interactive=False)
        else:
            return gr.update(interactive=True)

    def generate(
        self,
        checkpoint,
        fewsteplora,
        resolution,
        upscaler,
        style_prompt,
        process_controller,
        base_seed,
        perturb_seed,
        cn_image,
        cn_strength,
        style_image,
        style_strength,
        hair_prompt,
        eyes_prompt,
        face_image,
        pos_prompt,
        neg_prompt,
        character,
    ):
        self.start_preview_processor()

        self.preview_queue = std_queue.Queue()

        def handle_preview(image):
            self.preview_queue.put(image)

        wf = CharacterWorkflow(
            checkpoint,
            fewsteplora,
            resolution,
            upscaler,
            style_prompt,
            base_seed,
            perturb_seed,
            cn_image,
            cn_strength,
            style_image,
            style_strength,
            hair_prompt,
            eyes_prompt,
            face_image,
            pos_prompt,
            neg_prompt,
            character,
            preview_callback=handle_preview,
        )

        res_agg = []

        out_text = make_output_text(wf.resolved_prompts, wf.base_seed, wf.perturb_seed)

        yield gr.update(), out_text

        try:
            for result, label in wf.generate(process_controller):
                if isinstance(result[0], ImageBatchResult):
                    image = result[0].wait()
                    image_size = image[0].size

                    res_agg += [
                        (image[0], f"{label} ({image_size[0]} x {image_size[1]})")
                    ]

                    yield gr.update(value=res_agg), gr.update()
        except IndexError:
            yield gr.update(value=res_agg), gr.update()
        finally:
            self.stop_preview_processor()

    def interrupt(self):
        CharacterWorkflow.cancel_current()

    def generate_all(self):
        pass
