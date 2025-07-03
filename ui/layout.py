import gradio as gr

from constants import characters
from workflow.steps import get_steps


class MainLayout:
    @staticmethod
    def create_seeds():
        with gr.Group():
            gr.Markdown("Seeds Panel", container=True)
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
        with gr.Row(equal_height=True):
            positive_prompt = gr.Textbox(
                label="Positive Prompt",
                lines=4,
                placeholder="Enter positive prompt here...",
                elem_classes=["attention-editable"],
                scale=6,
            )

            auto_caption = gr.Button("Caption Input Image", variant="primary", scale=1)

        with gr.Row(equal_height=True):
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=4,
                placeholder="Enter negative prompt here...",
                elem_classes=["attention-editable"],
                scale=6,
            )

            gr.Button("Invisible Button", scale=1, elem_id="invisible", interactive=False)

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "auto_caption": auto_caption,
        }

    @staticmethod
    def create_buttons():
        with gr.Row():
            generate_btn = gr.Button(
                "Generate (Ctrl-ENTER)", variant="primary", elem_id="generate-btn"
            )
            interrupt_btn = gr.Button(
                "Interrupt (Ctrl-Shift-ENTER)", variant="stop", elem_id="interrupt-btn"
            )

        return {
            "generate": generate_btn,
            "interrupt": interrupt_btn,
        }

    @staticmethod
    def create_controllers():
        with gr.Accordion("Process Controller", open=True, elem_id="proc-acrdn"):
            process_controller = gr.CheckboxGroup(
                label="Process Controllers",
                choices=get_steps(),
            )

            with gr.Row():
                with gr.Column():
                    latent_upscale_scale = gr.Slider(
                        minimum=1,
                        maximum=2,
                        step=0.05,
                        label="Latent Upscale Scale",
                        show_reset_button=False,
                    )
                    latent_upscale_adherence = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        label="Latent Upscale Adherence",
                        show_reset_button=False,
                    )

                with gr.Column():
                    image_upscale_scale = gr.Slider(
                        minimum=1,
                        maximum=2,
                        step=0.05,
                        label="Image Upscale Scale",
                        show_reset_button=False,
                    )
                    image_upscale_adherence = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        label="Image Upscale Adherence",
                        show_reset_button=False,
                    )

        return {
            "process_controller": process_controller,
            "latent_scale": latent_upscale_scale,
            "latent_adherence": latent_upscale_adherence,
            "image_scale": image_upscale_scale,
            "image_adherence": image_upscale_adherence,
        }

    @staticmethod
    def create_output_panel():
        with gr.Group():
            with gr.Accordion("Output (Alt-O)", open=False, elem_id="output-acrdn"):
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
            gr.Markdown("Character Settings", container=True)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    character = gr.Dropdown(
                        label="Character", choices=character_choices
                    )
                with gr.Column(scale=3):
                    swap_method = gr.Radio(
                        choices=[
                            ("Use InstantID", "instantid"),
                            ("Use ReActor", "reactor"),
                            ("Use Prompt Only", "prompt"),
                        ],
                        label="Face Swap method",
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Accordion(label="Available Keys", open=False):
                        character_description = gr.Markdown(padding=True)

                with gr.Column(scale=3):
                    with gr.Accordion("Character Details", open=False):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                face_prompt = gr.Textbox(
                                    label="Face Prompt",
                                    lines=4,
                                    placeholder="Enter face prompt here...",
                                    interactive=True,
                                    elem_classes=["attention-editable"],
                                )
                                skin_prompt = gr.Textbox(
                                    label="Skin Prompt",
                                    lines=4,
                                    placeholder="Enter skin prompt here...",
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
                                face_images = gr.Gallery(
                                    label="Face Images",
                                    type="filepath",
                                    file_types=["image"],
                                    show_download_button=False,
                                    show_fullscreen_button=False,
                                    show_share_button=False,
                                )

        return {
            "character": character,
            "character_description": character_description,
            "face_prompt": face_prompt,
            "skin_prompt": skin_prompt,
            "hair_prompt": hair_prompt,
            "eyes_prompt": eyes_prompt,
            "face_images": face_images,
            "swap_method": swap_method,
        }

    @staticmethod
    def create_settings(checkpoints, resolutions, upscalers):
        with gr.Accordion(
            label="Optional Input Image (Alt-I)", open=False, elem_id="input-acrdn"
        ):
            input_image = gr.Image(
                label="Input Image",
                type="filepath",
                sources=["upload"],
                show_download_button=False,
                show_fullscreen_button=False,
                show_share_button=False,
                height=512,
            )

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

        use_detail_daemon = gr.Checkbox(label="Use Detail Daemon")

        with gr.Group():
            gr.Markdown("Control Net and Style Settings", container=True)
            enable_style = gr.Checkbox(label="Enable Style Prompt")

            style_prompt = gr.Textbox(
                label="Style Prompt",
                lines=4,
                placeholder="Enter style prompt to append to positive prompt...",
                elem_classes=["attention-editable"],
                interactive=True,
            )

            with gr.Row(equal_height=True):
                with gr.Accordion(
                    label="Control Net Image (Alt-C)", open=False, elem_id="cn-acrdn"
                ):
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

                with gr.Accordion(
                    label="Style Image (Alt-S)", open=False, elem_id="style-acrdn"
                ):
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
            "input_image": input_image,
            "checkpoint": checkpoint,
            "fewsteplora": fewsteplora,
            "resolution": resolution,
            "upscaler": upscaler,
            "enable_style": enable_style,
            "style_prompt": style_prompt,
            "use_detail_daemon": use_detail_daemon,
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
