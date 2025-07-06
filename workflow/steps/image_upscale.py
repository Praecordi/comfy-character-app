from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class ImageUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Image Upscale", order=6)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        positive = ConditioningConcat(ctx.positive_conditioning, ctx.eyes_conditioning)
        positive = ConditioningConcat(positive, ctx.hair_conditioning)
        positive = ConditioningConcat(positive, ctx.skin_conditioning)

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
                start_at=0.3,
                end_at=0.9,
                noise=0,
                combine_embeds=ApplyInstantIDAdvanced.combine_embeds.average,
                image_kps=image,
            )
        else:
            model, positive, negative = ctx.model, positive, ctx.negative_conditioning

        positive, negative = ControlNetApplyAdvanced(
            positive=positive,
            negative=negative,
            control_net=ctx.cn,
            image=image,
            strength=ctx.image_adherence,
            start_percent=0,
            end_percent=1,
            vae=ctx.vae,
        )

        denoise = (-0.3 * ctx.image_adherence + 0.7, -0.4 * ctx.image_adherence + 0.6)

        if ctx.image_scale < 1.25:
            num_iter = 1
        elif ctx.image_scale < 1.5:
            num_iter = 2
        elif ctx.image_scale < 1.75:
            num_iter = 3
        else:
            num_iter = 4

        upscaled = self._iterative_image_upscale(
            image=image,
            scale=ctx.image_scale,
            model=model,
            positive=positive,
            negative=negative,
            steps=ctx.steps["image_upscale"],
            cfg=self._scale_cfg(ctx.cfg["image_upscale"]),
            denoise=denoise,
            num_iterations=num_iter,
            seed_offset=self.metadata.order,
            sharpen=0.4,
            apply_cn=False,
        )

        image = ImageColorMatch(
            image=upscaled,
            reference=image,
            color_space=ImageColorMatch.color_space.RGB,
            factor=0.75,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
