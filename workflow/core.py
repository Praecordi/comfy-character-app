from typing import List, Tuple, Iterator

import comfy_nodes as csn
from constants import app_constants
from workflow.steps import _STEP_REGISTRY, WorkflowStep
from workflow.state import WorkflowContext, WorkflowState
from workflow.utils import (
    clean_prompt_string,
    build_conditioning_prompt,
    substitute_character_tokens,
)


class CharacterWorkflow:
    def __init__(self, ui_state: dict):
        self.ctx = WorkflowContext()

        self.base_steps = {
            "base_gen": 30,
            "latent_upscale": (20, 10),
            "detail_face": 15,
            "detail_hair": (15, 10),
            "detail_eyes": (15, 10),
            "image_upscale": (15, 10),
        }
        self.base_cfg = {
            "base_gen": 8,
            "latent_upscale": (10, 4),
            "detail_face": 4,
            "detail_hair": (6, 4),
            "detail_eyes": (6, 4),
            "image_upscale": (6, 4),
        }

        self.preview_callback = ui_state["preview_callback"]
        self._init_models(
            ui_state["checkpoint"], ui_state["fewsteplora"], ui_state["resolution"]
        )
        self._init_style_adapter(ui_state["style_image"], ui_state["style_strength"])
        self._init_controlnet_and_upscale(ui_state["upscaler"])
        self._init_prompts(
            ui_state["positive_prompt"],
            ui_state["negative_prompt"],
            ui_state["style_prompt"],
            ui_state["face_prompt"],
            ui_state["hair_prompt"],
            ui_state["eyes_prompt"],
            ui_state["character"],
            ui_state["checkpoint"].startswith("pony"),
        )
        self._init_input_images(
            ui_state["controlnet_image"],
            ui_state["controlnet_strength"],
            ui_state["face_image"],
        )

        self.ctx = self.ctx.update(
            base_seed=ui_state["base_seed"],
            perturb_seed=ui_state["perturb_seed"],
            use_instantid=ui_state["use_instantid"],
        )

    def _init_models(self, checkpoint, fewsteplora, resolution):
        model, clip, vae = csn.CheckpointLoaderSimple(checkpoint)
        if "Lightning" in checkpoint:
            sampler = csn.Samplers.dpmpp_2s_ancestral
            scheduler = csn.Schedulers.normal
            steps, cfg = self._generate_scaled_config(6, 2)
        elif "Hyper4S" in checkpoint:
            sampler = csn.Samplers.dpmpp_2s_ancestral
            scheduler = csn.Schedulers.normal
            steps, cfg = self._generate_scaled_config(6, 2)
        elif "Hyper8S" in checkpoint:
            sampler = csn.Samplers.dpmpp_2s_ancestral
            scheduler = csn.Schedulers.normal
            steps, cfg = self._generate_scaled_config(10, 2)
        elif "Turbo" in checkpoint:
            sampler = csn.Samplers.dpmpp_2s_ancestral
            scheduler = csn.Schedulers.normal
            steps, cfg = self._generate_scaled_config(10, 2)
        else:
            if fewsteplora in ["lcm", "turbo", "dpo_turbo"]:
                lora_map = {
                    "lcm": app_constants["lcm_lora"],
                    "turbo": app_constants["turbo_lora"],
                    "dpo_turbo": app_constants["dpo_turbo_lora"],
                }
                model, clip = csn.LoraLoader(
                    model=model,
                    clip=clip,
                    lora_name=lora_map[fewsteplora],
                    strength_model=1,
                    strength_clip=1,
                )
                sampler = csn.Samplers.lcm
                scheduler = csn.Schedulers.sgm_uniform
                steps, cfg = self.generate_scaled_config(8, 2)
            else:
                sampler = csn.Samplers.dpmpp_2m_sde_gpu
                scheduler = csn.Schedulers.karras
                steps, cfg = self._generate_scaled_config(30, 8)

        instantid = csn.InstantIDModelLoader(
            instantid_file=app_constants["instantid_model"]
        )
        faceanalysis = csn.InstantIDFaceAnalysis(provider="CPU")
        instantid_cn = csn.ControlNetLoader(
            control_net_name=app_constants["instantid_controlnet"]
        )

        self.ctx = self.ctx.update(
            model=model,
            clip=clip,
            vae=vae,
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            resolution=resolution,
            instantid=instantid,
            faceanalysis=faceanalysis,
            instantid_cn=instantid_cn,
        )

    def _init_style_adapter(self, style_image, style_strength):
        if style_image is not None:
            s_image, _ = csn.LoadImage(style_image)
            s_strength = style_strength

            model, ipadapter = csn.IPAdapterUnifiedLoader(
                model=self.ctx.model,
                preset=csn.IPAdapterUnifiedLoader.preset.STANDARD_medium_strength,
            )

            s_negative = csn.IPAdapterNoise(
                type=csn.IPAdapterNoise.type.shuffle,
                strength=0.4,
                blur=10,
                image_optional=s_image,
            )

            model = csn.IPAdapterPreciseStyleTransfer(
                model=model,
                ipadapter=ipadapter,
                image=s_image,
                weight=s_strength,
                style_boost=1,
                combine_embeds=csn.IPAdapterPreciseStyleTransfer.combine_embeds.concat,
                start_at=0,
                end_at=1,
                embeds_scaling=csn.IPAdapterPreciseStyleTransfer.embeds_scaling.K_V,
                image_negative=s_negative,
            )

            self.ctx = self.ctx.update(
                style_image=s_image, style_strength=s_strength, model=model
            )
        else:
            self.ctx = self.ctx.update(style_image=None, style_strength=None)

    def _init_controlnet_and_upscale(self, upscaler):
        cn = csn.ControlNetLoader(control_net_name=app_constants["union_controlnet"])
        cn = csn.SetUnionControlNetType(
            control_net=cn, type=csn.SetUnionControlNetType.type.auto
        )

        upscale_model_name = upscaler
        if not upscaler == "None":
            upscale_model = csn.UpscaleModelLoader(model_name=upscale_model_name)
        else:
            upscale_model = None

        self.ctx = self.ctx.update(
            cn=cn, upscale_model_name=upscale_model_name, upscale_model=upscale_model
        )

    def _init_prompts(
        self,
        pos_prompt,
        neg_prompt,
        style_prompt,
        face_prompt,
        hair_prompt,
        eyes_prompt,
        character,
        apply_scores,
    ):
        pos = clean_prompt_string(pos_prompt)
        neg = clean_prompt_string(neg_prompt)

        if style_prompt:
            pos += clean_prompt_string(style_prompt)

        if apply_scores:
            pos = ["score_9", "score_8", "score_7"] + pos
            neg = ["score_1", "score_2", "score_3"] + neg

        pos = ", ".join(pos)
        neg = ", ".join(neg)

        pos = substitute_character_tokens(pos, character)
        neg = substitute_character_tokens(neg, character)

        face = build_conditioning_prompt(face_prompt or "", style_prompt, apply_scores)
        hair = build_conditioning_prompt(hair_prompt or "", style_prompt, apply_scores)
        eyes = build_conditioning_prompt(eyes_prompt or "", style_prompt, apply_scores)

        def encode(text: str) -> csn.Conditioning:
            return csn.CLIPTextEncodeSDXL(
                clip=self.ctx.clip,
                width=4096,
                height=4096,
                crop_w=0,
                crop_h=0,
                target_width=4096,
                target_height=4096,
                text_g=text,
                text_l=text,
            )

        pos_cond = encode(pos)
        neg_cond = encode(neg)
        face_cond = encode(face)
        hair_cond = encode(hair)
        eyes_cond = encode(eyes)

        self.ctx = self.ctx.update(
            positive_prompt=pos,
            negative_prompt=neg,
            face_prompt=face,
            hair_prompt=hair,
            eyes_prompt=eyes,
            positive_conditioning=pos_cond,
            negative_conditioning=neg_cond,
            face_conditioning=face_cond,
            hair_conditioning=hair_cond,
            eyes_conditioning=eyes_cond,
        )

    def _init_input_images(self, cn_image, cn_strength, face_image):
        if cn_image is not None:
            cn_im, _ = csn.LoadImage(cn_image)
            cn_st = cn_strength
        else:
            cn_im = None
            cn_st = None

        if face_image is not None:
            face_im, _ = csn.LoadImage(face_image)
        else:
            face_im = None

        self.ctx = self.ctx.update(
            cn_image=cn_im, cn_strength=cn_st, face_image=face_im
        )

    def _generate_scaled_config(self, n, k):
        new_steps = {}
        for key, val in self.base_steps.items():
            if isinstance(val, tuple):
                scaled_tuple = tuple(round(1 + (x - 1) * (n - 1) / 29) for x in val)
                new_steps[key] = scaled_tuple
            else:
                scaled_val = round(1 + (val - 1) * (n - 1) / 29)
                new_steps[key] = scaled_val

        new_cfg = {}
        for key, val in self.base_cfg.items():
            if isinstance(val, tuple):
                scaled_tuple = tuple(round(1 + (x - 1) * (k - 1) / 7, 2) for x in val)
                new_cfg[key] = scaled_tuple
            else:
                scaled_val = round(1 + (val - 1) * (k - 1) / 7, 2)
                new_cfg[key] = scaled_val

        return new_steps, new_cfg

    def iter_wf(self, controller: List[str]) -> Iterator[WorkflowStep]:
        steps: List[Tuple[int, WorkflowStep]] = []
        for cls in _STEP_REGISTRY.values():
            if cls.metadata.default_enabled or cls.metadata.label in controller:
                step_instance = cls(self.ctx)
                steps.append((cls.metadata.order, step_instance))

        for _, step in sorted(steps, key=lambda x: x[0]):
            yield step

    def generate(self, controller) -> Iterator[Tuple[csn.Image, str]]:
        results: List[Tuple[csn.Image, str]] = []
        with csn.Workflow(queue=False) as wf:
            _, _, _, _, _, latent, _ = csn.CRAspectRatio(
                aspect_ratio=self.ctx.resolution
            )
            image = csn.VAEDecode(latent, self.ctx.vae)

            state = WorkflowState(latent=latent, image=image)

            for step in self.iter_wf(controller):
                state = step.run(state)
                results.append((csn.PreviewImage(state.image), step.metadata.label))

        wf.queue()

        for result, label in results:
            result.task.add_preview_callback(self.preview_callback)
            yield result.wait(), label
            result.task.remove_preview_callback(self.preview_callback)
