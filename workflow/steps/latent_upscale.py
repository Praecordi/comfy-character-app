from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class LatentUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Latent Upscale", order=1)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image
        latent = state.latent

        cn_positive, cn_negative = ControlNetApplyAdvanced(
            positive=ctx.positive_conditioning,
            negative=ctx.negative_conditioning,
            control_net=ctx.cn,
            image=image,
            strength=ctx.latent_adherence,
            start_percent=0,
            end_percent=1,
            vae=ctx.vae,
        )

        denoise = (-0.3 * ctx.latent_adherence + 0.8, -0.2 * ctx.latent_adherence + 0.4)

        if ctx.latent_scale < 1.25:
            num_iter = 1
        elif ctx.latent_scale < 1.5:
            num_iter = 2
        elif ctx.latent_scale < 1.75:
            num_iter = 3
        else:
            num_iter = 4

        latent = self._iterative_latent_upscale(
            latent=latent,
            scale=ctx.latent_scale,
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
