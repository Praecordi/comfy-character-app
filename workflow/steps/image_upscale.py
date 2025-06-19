from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class ImageUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Image Upscale", order=5)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        positive = ConditioningConcat(ctx.positive_conditioning, ctx.eyes_conditioning)
        positive = ConditioningConcat(positive, ctx.face_conditioning)
        positive = ConditioningConcat(positive, ctx.hair_conditioning)

        image = self._iterative_image_upscale(
            image=image,
            scale=1.3,
            model=ctx.model,
            positive=positive,
            negative=ctx.negative_conditioning,
            steps=ctx.steps["image_upscale"],
            cfg=self._scale_cfg(ctx.cfg["image_upscale"]),
            denoise=(0.6, 0.5),
            num_iterations=2,
            seed_offset=5,
            sharpen=0.4,
            apply_cn=True,
            cn_strength=0.9,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
