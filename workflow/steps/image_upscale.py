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

        if ctx.swap_method == "instantid":
            model, positive, negative = ApplyInstantIDAdvanced(
                instantid=ctx.instantid,
                insightface=ctx.faceanalysis,
                control_net=ctx.instantid_cn,
                image=ctx.face_image,
                model=ctx.model,
                positive=positive,
                negative=ctx.negative_conditioning,
                ip_weight=0.9,
                cn_strength=0.3,
                start_at=0,
                end_at=0.5,
                noise=0,
                combine_embeds=ApplyInstantIDAdvanced.combine_embeds.average,
                image_kps=image,
            )
        else:
            model, positive, negative = ctx.model, positive, ctx.negative_conditioning

        image = self._iterative_image_upscale(
            image=image,
            scale=1.3,
            model=model,
            positive=positive,
            negative=negative,
            steps=ctx.steps["image_upscale"],
            cfg=self._scale_cfg(ctx.cfg["image_upscale"]),
            denoise=(0.7, 0.6),
            num_iterations=2,
            seed_offset=5,
            sharpen=0.4,
            apply_cn=True,
            cn_strength=0.9,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
