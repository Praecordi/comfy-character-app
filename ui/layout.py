import gradio as gr

from constants import characters
from workflow.steps import get_steps


class MainLayout:
    @staticmethod
    def create_seeds():
        with gr.Row():
            base_seed = gr.Number(
                label="Base Seed",
                precision=0,
            )

            perturb_seed = gr.Number(
                label="Perturb Seed",
                precision=0,
            )

        return {"base_seed": base_seed, "perturb_seed": perturb_seed}

    @staticmethod
    def create_main_prompts():
        positive_prompt = gr.Textbox(
            label="Positive Prompt",
            lines=4,
            placeholder="Enter positive prompt here...",
            elem_classes=["attention-editable"],
        )

        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            lines=4,
            placeholder="Enter negative prompt here...",
            elem_classes=["attention-editable"],
        )

        return {"positive_prompt": positive_prompt, "negative_prompt": negative_prompt}

    @staticmethod
    def create_buttons():
        with gr.Row():
            with gr.Column():
                generate_btn = gr.Button("Generate")
                generate_all_btn = gr.Button("Generate All")

            with gr.Column():
                interrupt_btn = gr.Button("Interrupt")
                interrupt_all_btn = gr.Button("Interrupt All")

        return {
            "generate": generate_btn,
            "generate_all": generate_all_btn,
            "interrupt": interrupt_btn,
            "interrupt_all": interrupt_all_btn,
        }

    @staticmethod
    def create_controllers():
        process_controller = gr.CheckboxGroup(
            label="Process Controllers",
            choices=get_steps(),
        )

        return {"process_controller": process_controller}

    @staticmethod
    def create_output_panel():
        with gr.Group():
            with gr.Accordion("Output", open=False):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        output_gallery = gr.Gallery(
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

                        output_text = gr.Textbox(
                            interactive=False, lines=7, show_label=False
                        )

                    preview = gr.Image(
                        label="Preview",
                        interactive=False,
                        visible=False,
                        scale=1,
                    )

        return {
            "output": output_gallery,
            "output_text": output_text,
            "preview": preview,
            # "preview_text": preview_text,
        }

    @staticmethod
    def create_character_settings():
        character_choices = [char.capitalize() for char in characters.keys()] + [
            "Custom"
        ]

        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    character = gr.Dropdown(
                        label="Character", choices=character_choices, scale=2
                    )
                    with gr.Accordion(label="Available Keys", open=False):
                        character_description = gr.Markdown(padding=True)

                with gr.Column(scale=3):
                    with gr.Accordion("Custom Character", open=False):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                face_prompt = gr.Textbox(
                                    label="Face Prompt",
                                    lines=4,
                                    placeholder="Enter face prompt here...",
                                    interactive=True,
                                    elem_classes=["attention-editable"],
                                )
                                hair_prompt = gr.Textbox(
                                    label="Hair Prompt",
                                    lines=4,
                                    placeholder="Enter hair prompt here...",
                                    interactive=True,
                                    elem_classes=["attention-editable"],
                                )
                                eyes_prompt = gr.Textbox(
                                    label="Eyes Prompt",
                                    lines=4,
                                    placeholder="Enter eyes prompt here",
                                    interactive=True,
                                    elem_classes=["attention-editable"],
                                )

                            with gr.Column(scale=3):
                                face_image = gr.Image(
                                    label="Face Image",
                                    type="filepath",
                                    interactive=True,
                                    sources=["upload"],
                                    show_download_button=False,
                                    show_fullscreen_button=False,
                                    show_share_button=False,
                                )
                                use_instantid = gr.Checkbox(label="Use InstantID")

        return {
            "character": character,
            "character_description": character_description,
            "face_prompt": face_prompt,
            "hair_prompt": hair_prompt,
            "eyes_prompt": eyes_prompt,
            "face_image": face_image,
            "use_instantid": use_instantid,
        }

    @staticmethod
    def create_settings(checkpoints, resolutions, upscalers):
        with gr.Group():
            checkpoint = gr.Dropdown(
                label="Checkpoint",
                choices=checkpoints,
            )

            fewsteplora = gr.Radio(
                label="Few Step LoRA", choices=["none", "lcm", "turbo", "dpo_turbo"]
            )

            resolution = gr.Dropdown(
                label="Resolution",
                choices=resolutions,
            )

            upscaler = gr.Dropdown(label="Upscaler", choices=["None"] + upscalers)

            style_prompt = gr.Textbox(
                label="Style Prompt",
                lines=4,
                placeholder="Enter style prompt to append to positive prompt...",
                interactive=True,
            )

            with gr.Row():
                with gr.Column():
                    cn_image = gr.Image(
                        label="Control Net Image",
                        type="filepath",
                        sources=["upload"],
                        show_download_button=False,
                        show_fullscreen_button=False,
                        show_share_button=False,
                    )
                    cn_strength = gr.Slider(
                        minimum=0, maximum=100, step=1, label="Control Net Strength"
                    )

                with gr.Column():
                    style_image = gr.Image(
                        label="Style Image",
                        type="filepath",
                        sources=["upload"],
                        show_download_button=False,
                        show_fullscreen_button=False,
                        show_share_button=False,
                    )

                    style_strength = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        label="Style Strength",
                    )

        return {
            "checkpoint": checkpoint,
            "fewsteplora": fewsteplora,
            "resolution": resolution,
            "upscaler": upscaler,
            "style_prompt": style_prompt,
            "controlnet_image": cn_image,
            "controlnet_strength": cn_strength,
            "style_image": style_image,
            "style_strength": style_strength,
        }

    @staticmethod
    def create(checkpoints, resolutions, upscalers):
        components = {}

        out_comps = MainLayout.create_output_panel()

        with gr.Row():
            with gr.Column(scale=2):
                settings_comps = MainLayout.create_settings(
                    checkpoints, resolutions, upscalers
                )

            with gr.Column(scale=3):
                prompt_comps = MainLayout.create_main_prompts()

                seed_comps = MainLayout.create_seeds()

                character_comps = MainLayout.create_character_settings()

                controller_comps = MainLayout.create_controllers()

                button_comps = MainLayout.create_buttons()

        components = {
            **out_comps,
            **settings_comps,
            **prompt_comps,
            **seed_comps,
            **character_comps,
            **controller_comps,
            **button_comps,
        }

        return components
