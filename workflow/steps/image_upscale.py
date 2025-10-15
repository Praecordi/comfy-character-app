from comfy_nodes import *

from utils import scale_steps, scale_cfg
from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class ImageUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(
        label="Image Upscale",
        order=6,
        parameters={
            "concat_conditioning": {
                "type": "checkboxgroup",
                "label": "Concat Conditioning",
                "choices": ["Skin", "Face", "Hair", "Eyes"],
                "value": [],
            },
            "image_scale": {
                "minimum": 1,
                "maximum": 2,
                "step": 0.05,
                "value": 1.25,
                "label": "Image Upscale Scale",
                "type": "slider",
            },
            "image_adherence": {
                "minimum": 0,
                "maximum": 1,
                "step": 0.05,
                "value": 0.8,
                "label": "Image Upscale Adherence",
                "type": "slider",
            },
            "cfg": {"type": "number", "label": "CFG", "value": 8},
            "use_instantid": {
                "type": "checkbox",
                "value": True,
                "label": "Use InstantID",
            },
        },
    )

    def _init(
        self,
        concat_conditioning=None,
        image_scale=None,
        image_adherence=None,
        cfg=None,
        use_instantid=None,
    ):
        self.concat_conditioning = (
            concat_conditioning
            if concat_conditioning
            else self.metadata.parameters["concat_conditionings"]["value"]
        )
        self.image_scale = (
            image_scale
            if image_scale
            else self.metadata.parameters["image_scale"]["value"]
        )
        self.image_adherence = (
            image_adherence
            if image_adherence
            else self.metadata.parameters["image_adherence"]["value"]
        )
        self.use_instantid = (
            use_instantid
            if use_instantid
            else self.metadata.parameters["use_instantid"]["value"]
        )

        if self.image_scale < 1.25:
            base_step = (30, 20)
            base_cfg = (8, 4)
        elif self.image_scale < 1.5:
            base_step = (25, 15)
            base_cfg = (8, 4)
        elif self.image_scale < 1.75:
            base_step = (20, 10)
            base_cfg = (6, 2)
        else:
            base_step = (15, 5)
            base_cfg = (4, 2)

        if self.ctx.type in ["Lightning", "Hyper4S"]:
            step_scale = 6
        elif self.ctx.type in ["Hyper8S", "Turbo"]:
            step_scale = 10
        elif self.ctx.type == "fewsteplora":
            step_scale = 8
        else:
            step_scale = 30

        cfg_scale = cfg if cfg else self.metadata.parameters["cfg"]["value"]

        self.steps = scale_steps(base_step, step_scale)
        self.cfg = self._scale_cfg(scale_cfg(base_cfg, cfg_scale))

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        positive = ctx.positive_conditioning
        if "Eyes" in self.concat_conditioning:
            positive = ConditioningConcat(positive, ctx.eyes_conditioning)
        if "Skin" in self.concat_conditioning:
            positive = ConditioningConcat(positive, ctx.skin_conditioning)
        if "Hair" in self.concat_conditioning:
            positive = ConditioningConcat(positive, ctx.hair_conditioning)
        if "Face" in self.concat_conditioning:
            positive = ConditioningConcat(positive, ctx.face_conditioning)

        if self.use_instantid:
            model, positive, negative = ApplyInstantIDAdvanced(
                instantid=ctx.instantid,
                insightface=ctx.faceanalysis,
                control_net=ctx.instantid_cn,
                image=ctx.face_image,
                model=ctx.lora_model,
                positive=positive,
                negative=ctx.negative_conditioning,
                ip_weight=0.8,
                cn_strength=0.5,
                start_at=0.7,
                end_at=1.0,
                noise=0.1,
                combine_embeds=ApplyInstantIDAdvanced.combine_embeds.average,
                image_kps=image,
            )
        else:
            model, positive, negative = (
                ctx.lora_model,
                positive,
                ctx.negative_conditioning,
            )

        positive, negative = ControlNetApplyAdvanced(
            positive=positive,
            negative=negative,
            control_net=ctx.cn,
            image=image,
            strength=self.image_adherence,
            start_percent=0,
            end_percent=1,
            vae=ctx.vae,
        )

        denoise = (-0.3 * self.image_adherence + 0.7, -0.4 * self.image_adherence + 0.6)

        if self.image_scale < 1.25:
            num_iter = 1
        elif self.image_scale < 1.5:
            num_iter = 2
        elif self.image_scale < 1.75:
            num_iter = 3
        else:
            num_iter = 4

        upscaled = self._iterative_image_upscale(
            image=image,
            scale=self.image_scale,
            model=model,
            positive=positive,
            negative=negative,
            steps=self.steps,
            cfg=self.cfg,
            denoise=denoise,
            num_iterations=num_iter,
            seed_offset=self.metadata.order,
            add_noise=False,
            sharpen=0.4,
            apply_cn=False,
        )

        image = ImageColorMatch(
            image=upscaled,
            reference=image,
            color_space=ImageColorMatch.color_space.LAB,
            factor=1,
        )

        new_width = BasicDataHandlingCastToInt(
            BasicDataHandlingFloatMultiply(
                BasicDataHandlingCastToFloat(state.width), self.image_scale
            )
        )
        new_height = BasicDataHandlingCastToInt(
            BasicDataHandlingFloatMultiply(
                BasicDataHandlingCastToFloat(state.height), self.image_scale
            )
        )

        image, _, _ = ImageResize_(
            image,
            new_width,
            new_height,
            interpolation=ImageResize_.interpolation.lanczos,
            method=ImageResize_.method.stretch,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(
            latent=latent, image=image, width=new_width, height=new_height
        )
