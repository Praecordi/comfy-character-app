import gradio as gr

from constants import characters
from utils import make_name
from workflow.steps import get_steps
from ui.cm_events import delete_field, update_field


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

            gr.Button(
                "Invisible Button", scale=1, elem_id="invisible", interactive=False
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                enable_style = gr.Checkbox(label="Enable Style Prompt")

                style_prompt = gr.Textbox(
                    label="Style Prompt",
                    lines=4,
                    placeholder="Enter style prompt to append to positive prompt...",
                    elem_classes=["attention-editable"],
                    interactive=True,
                )

            gr.Button(
                "Invisible Button", scale=1, elem_id="invisible", interactive=False
            )

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "auto_caption": auto_caption,
            "enable_style": enable_style,
            "style_prompt": style_prompt,
        }

    @staticmethod
    def create_buttons():
        with gr.Row():
            queue_btn = gr.Button(
                "Queue (Ctrl-ENTER)", variant="primary", elem_id="generate-btn"
            )
            interrupt_btn = gr.Button(
                "Interrupt (Ctrl-Shift-ENTER)", variant="stop", elem_id="interrupt-btn"
            )

        return {
            "queue": queue_btn,
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
        gallery_index = gr.State(value=0)
        gallery_state = gr.State(value=[])

        def get_gallery(state):
            result = []
            for tup in state:
                result.append((tup[0], tup[1]))
            return result

        def get_text(state, index):
            if index >= len(state):
                return ""
            else:
                return state[index][2]

        with gr.Group():
            with gr.Accordion("Output (Alt-O)", open=False, elem_id="output-acrdn"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        output_gallery = gr.Gallery(
                            value=get_gallery,
                            inputs=[gallery_state],
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

                        with gr.Row():
                            output_text = gr.Textbox(
                                value=get_text,
                                inputs=[gallery_state, gallery_index],
                                interactive=False,
                                lines=7,
                                show_label=False,
                                scale=8,
                            )

                            reset_gallery_btn = gr.Button("Reset Gallery", scale=1)

                    preview = gr.Image(
                        label="Preview",
                        interactive=False,
                        visible=False,
                        scale=1,
                    )

        return {
            "gallery_index": gallery_index,
            "gallery_state": gallery_state,
            "reset_gallery_btn": reset_gallery_btn,
            "output": output_gallery,
            "output_text": output_text,
            "preview": preview,
        }

    @staticmethod
    def create_character_settings():
        character_choices = [make_name(char) for char in characters.keys()] + ["Custom"]

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
                with gr.Column(scale=1), gr.Accordion(
                    label="Available Keys", open=False
                ):
                    character_description = gr.Markdown(padding=True)

                with gr.Column(scale=4), gr.Accordion(
                    "Character Details", open=False
                ), gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        base_prompt = gr.Textbox(
                            label="Base Prompt",
                            lines=2,
                            interactive=False,
                            elem_classes=["attention-editable"],
                        )
                        face_prompt = gr.Textbox(
                            label="Face Prompt",
                            lines=2,
                            interactive=False,
                            elem_classes=["attention-editable"],
                        )
                        skin_prompt = gr.Textbox(
                            label="Skin Prompt",
                            lines=2,
                            interactive=False,
                            elem_classes=["attention-editable"],
                        )
                        hair_prompt = gr.Textbox(
                            label="Hair Prompt",
                            lines=2,
                            interactive=False,
                            elem_classes=["attention-editable"],
                        )
                        eyes_prompt = gr.Textbox(
                            label="Eyes Prompt",
                            lines=2,
                            interactive=False,
                            elem_classes=["attention-editable"],
                        )

                    with gr.Column(scale=2):
                        face_images = gr.Gallery(
                            label="Face Images",
                            type="filepath",
                            file_types=["image"],
                            show_download_button=False,
                            show_fullscreen_button=False,
                            show_share_button=False,
                            interactive=False,
                        )

        return {
            "character": character,
            "character_description": character_description,
            "base_prompt": base_prompt,
            "face_prompt": face_prompt,
            "skin_prompt": skin_prompt,
            "hair_prompt": hair_prompt,
            "eyes_prompt": eyes_prompt,
            "face_images": face_images,
            "swap_method": swap_method,
        }

    @staticmethod
    def create_settings(checkpoints, loras, resolutions, upscalers):
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

        with gr.Group():
            checkpoint = gr.Dropdown(
                label="Checkpoint",
                choices=checkpoints,
            )

            hook_checkpoint = gr.Dropdown(
                label="Hook Checkpoint",
                choices=["None"] + checkpoints,
            )

            with gr.Row():
                hook_start = gr.Slider(0, 1.0, 0, step=0.1, label="Hook Start Strength")
                hook_end = gr.Slider(0, 1.0, 1.0, step=0.1, label="Hook End Strength")

            lora_options = gr.Dropdown(
                choices=loras,
                label="LoRAs",
                multiselect=True,
            )

            lora_state = gr.State({})

            with gr.Accordion(
                "LoRA Settings (Alt-L)", open=False, elem_id="lora-acrdn"
            ):

                @gr.render(
                    inputs=[lora_options, lora_state], triggers=[lora_options.change]
                )
                def render_lora_sliders(selected, state):
                    sliders = {}
                    with gr.Column():
                        for i, lkey in enumerate(selected):
                            sliders[lkey] = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=state.get(lkey, 0.8),
                                label=f"{lkey}",
                                interactive=True,
                                key=i,
                            )

                            def create_change_handler(lora_name):

                                def update_slider(slider, state):
                                    state[lora_name] = slider
                                    return state

                                return update_slider

                            sliders[lkey].change(
                                create_change_handler(lkey),
                                inputs=[sliders[lkey], lora_state],
                                outputs=[lora_state],
                            )

            lora_options.change(
                lambda selected, current_state: {
                    lora: current_state.get(lora, 0.8) for lora in selected
                },
                inputs=[lora_options, lora_state],
                outputs=[lora_state],
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
            "hook_checkpoint": hook_checkpoint,
            "hook_start": hook_start,
            "hook_end": hook_end,
            "lora_options": lora_options,
            "loras": lora_state,
            "fewsteplora": fewsteplora,
            "resolution": resolution,
            "upscaler": upscaler,
            "use_detail_daemon": use_detail_daemon,
            "controlnet_image": cn_image,
            "controlnet_strength": cn_strength,
            "style_image": style_image,
            "style_strength": style_strength,
        }

    @staticmethod
    def create(checkpoints, loras, resolutions, upscalers):
        components = {}

        gr.Markdown("## Generator")

        out_comps = MainLayout.create_output_panel()

        with gr.Row():
            with gr.Column(scale=2):
                settings_comps = MainLayout.create_settings(
                    checkpoints, loras, resolutions, upscalers
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


class CharacterManagerLayout:
    @staticmethod
    def create_face_field():
        with gr.Group():
            with gr.Row(equal_height=True):
                gr.Textbox(
                    "Face References", label="Attribute", interactive=False, scale=2
                )

                char_gallery = gr.Gallery(
                    label="Face Images",
                    type="filepath",
                    file_types=["image"],
                    columns=4,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    show_share_button=False,
                    interactive=True,
                    scale=8,
                )

        return char_gallery

    @staticmethod
    def create_panel():
        character_choices = [make_name(char) for char in characters.keys()]

        current_fields = gr.State(value={})
        current_character_state = gr.State(value={})

        with gr.Row():
            with gr.Column():
                character_select = gr.Dropdown(
                    label="Select Character",
                    choices=character_choices,
                    value=None,
                    interactive=True,
                )
                with gr.Row():
                    save_btn = gr.Button("Save Character", variant="primary")
                    reset_btn = gr.Button("Reset Character", variant="secondary")

            with gr.Column():
                new_char = gr.Textbox(
                    label="New Character Name",
                    placeholder="Enter character name...",
                )

                with gr.Row():
                    add_char = gr.Button("Add Character", variant="primary")
                    remove_char = gr.Button("Remove Character", variant="stop")

        with gr.Column():
            face_images = CharacterManagerLayout.create_face_field()

            base_prompt = CharacterManagerLayout.create_field(
                "base", "", removable=False
            )["value"]
            face_prompt = CharacterManagerLayout.create_field(
                "face", "", removable=False
            )["value"]
            skin_prompt = CharacterManagerLayout.create_field(
                "skin", "", removable=False
            )["value"]
            hair_prompt = CharacterManagerLayout.create_field(
                "hair", "", removable=False
            )["value"]
            eyes_prompt = CharacterManagerLayout.create_field(
                "eyes", "", removable=False
            )["value"]

        gr.Markdown("### Other Fields")

        @gr.render(
            inputs=current_fields,
            triggers=[character_select.change, current_fields.change],
        )
        def render_fields(fields):
            with gr.Column():
                for i, (key, value) in enumerate(fields.items()):
                    comps = CharacterManagerLayout.create_field(key, value, key=i)

                    comps["value"].change(
                        update_field,
                        inputs=[
                            comps["attribute"],
                            comps["value"],
                            character_select,
                            current_character_state,
                        ],
                        outputs=[current_character_state],
                    )

                    comps["button"].click(
                        delete_field,
                        inputs=[
                            comps["attribute"],
                            character_select,
                            current_character_state,
                        ],
                        outputs=[current_fields, current_character_state],
                    )

        with gr.Row():
            with gr.Group():
                new_field = gr.Textbox(
                    label="New Attribute", placeholder="Enter new attribute..."
                )
                add_field_btn = gr.Button("Add Attribute")

        return {
            "character_select": character_select,
            "new_character": new_char,
            "add_character_btn": add_char,
            "remove_character_btn": remove_char,
            "cm_face_images": face_images,
            "cm_base_prompt": base_prompt,
            "cm_face_prompt": face_prompt,
            "cm_skin_prompt": skin_prompt,
            "cm_hair_prompt": hair_prompt,
            "cm_eyes_prompt": eyes_prompt,
            "new_field": new_field,
            "add_field_btn": add_field_btn,
            "save_btn": save_btn,
            "reset_btn": reset_btn,
            "current_fields": current_fields,
            "current_character_state": current_character_state,
        }

    @staticmethod
    def create_field(
        field_name,
        field_value,
        removable=True,
        key=None,
        field_name_params={},
        field_value_params={},
    ):
        if key is not None:
            attr_params = {"key": f"attr_{key}", "preserved_by_key": ["label"]}
            val_params = {"key": f"val_{key}", "preserved_by_key": ["label"]}
            btn_params = {"key": f"btn_{key}"}
        else:
            attr_params, val_params, btn_params = {}, {}, {}

        components = {}
        field_name = make_name(field_name)
        fn_params = {
            "label": "Attribute",
            "value": field_name,
            "scale": 2,
            "interactive": False,
            **attr_params,
            **field_name_params,
        }
        fv_params = {
            "label": "Value",
            "value": field_value,
            "scale": 8,
            "interactive": True,
            "elem_classes": ["attention-editable"],
            **val_params,
            **field_value_params,
        }
        b_params = {"value": "Delete", "variant": "stop", "scale": 1, **btn_params}
        with gr.Group():
            with gr.Row(equal_height=True):
                components["attribute"] = gr.Textbox(**fn_params)
                components["value"] = gr.Textbox(**fv_params)
                if removable:
                    components["button"] = gr.Button(**b_params)
                else:
                    components["button"] = None

        return components

    @staticmethod
    def create():
        components = {}

        gr.Markdown("## Character Manager")

        main_comp = CharacterManagerLayout.create_panel()

        components = {**main_comp}

        return components
