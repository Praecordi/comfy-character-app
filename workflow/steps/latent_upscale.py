from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class LatentUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Latent Upscale", order=1)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        latent = state.latent

        latent = self._iterative_latent_upscale(
            latent=latent,
            scale=1.6,
            model=ctx.model,
            positive=ctx.positive_conditioning,
            negative=ctx.negative_conditioning,
            steps=ctx.steps["latent_upscale"],
            cfg=self._scale_cfg(ctx.cfg["latent_upscale"]),
            denoise=(0.8, 0.4),
            num_iterations=3,
            seed_offset=1,
            apply_cn=True,
            cn_strength=0.3,
        )

        image = VAEDecode(latent, ctx.vae)

        return state.update(latent=latent, image=image)
