from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class LatentUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(
        label="Latent Upscale",
        order=1,
        parameters={
            "latent_scale": {
                "minimum": 1,
                "maximum": 2,
                "step": 0.05,
                "value": 1.6,
                "label": "Latent Upscale Scale",
                "type": "slider",
            },
            "latent_adherence": {
                "minimum": 0,
                "maximum": 1,
                "step": 0.05,
                "value": 0.3,
                "label": "Latent Upscale Adherence",
                "type": "slider",
            },
        },
    )

    def _init(self, latent_scale, latent_adherence):
        self.latent_scale = latent_scale
        self.latent_adherence = latent_adherence

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image
        latent = state.latent

        cn_positive, cn_negative = ControlNetApplyAdvanced(
            positive=ctx.positive_conditioning,
            negative=ctx.negative_conditioning,
            control_net=ctx.cn,
            image=image,
            strength=self.latent_adherence,
            start_percent=0,
            end_percent=1,
            vae=ctx.vae,
        )

        denoise = (
            -0.3 * self.latent_adherence + 0.8,
            -0.2 * self.latent_adherence + 0.4,
        )

        if self.latent_scale < 1.25:
            num_iter = 1
        elif self.latent_scale < 1.5:
            num_iter = 2
        elif self.latent_scale < 1.75:
            num_iter = 3
        else:
            num_iter = 4

        latent = self._iterative_latent_upscale(
            latent=latent,
            scale=self.latent_scale,
            model=ctx.lora_model,
            positive=cn_positive,
            negative=cn_negative,
            steps=ctx.steps["latent_upscale"],
            cfg=self._scale_cfg(ctx.cfg["latent_upscale"]),
            denoise=denoise,
            num_iterations=num_iter,
            seed_offset=self.metadata.order,
            apply_cn=False,
        )

        upscaled = VAEDecode(latent, ctx.vae)

        image = ImageColorMatch(
            image=upscaled,
            reference=image,
            color_space=ImageColorMatch.color_space.LAB,
            factor=1,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
