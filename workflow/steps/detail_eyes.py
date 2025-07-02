from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailEyesStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Eyes Detail", order=5)
    usebbox = True
    applymask = True

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        _, mask = GroundingDinoSAMSegmentSegmentAnything(
            ctx.sam_model, ctx.gd_model, image, prompt="eye", threshold=0.5
        )

        if self.usebbox:
            image_width, image_height, _ = GetImageSize(image)
            _, _, x, y, width, height = MaskBoundingBox(mask)
            mask = MaskRectAreaAdvanced(x, y, width, height, image_width, image_height)

        segs = MaskToSEGS(
            mask=mask,
            combined=True,
            crop_factor=2,
            bbox_fill=False,
            drop_size=10,
            contour_fill=True,
        )
        segs = SetDefaultImageForSEGS(segs=segs, image=image, override=True)
        segs_header, seg_elt = ImpactDecomposeSEGS(segs)
        seg_elt, cropped_image, cropped_mask, _, _, _, _, _ = ImpactFromSEGELT(seg_elt)
        width, height, _ = GetImageSize(cropped_image)

        if ctx.upscale_model:
            cropped_image, _ = CRUpscaleImage(
                image=cropped_image,
                upscale_model=ctx.upscale_model_name,
                mode=CRUpscaleImage.mode.resize,
                resize_width=1024,
                resampling_method=CRUpscaleImage.resampling_method.lanczos,
            )
        else:
            cropped_image, _, _ = ImageResize_(
                image=cropped_image,
                width=1024,
                height=2048,
                interpolation=ImageResize_.interpolation.lanczos,
                method=ImageResize_.method.keep_proportion,
            )

        model = DifferentialDiffusion(ctx.model)

        cropped_mask = GrowMask(cropped_mask, expand=20)
        cropped_mask = MaskBlur(cropped_mask, amount=50)

        cropped_latent = VAEEncode(cropped_image, ctx.vae)

        cropped_latent = self._iterative_latent_upscale(
            latent=cropped_latent,
            scale=2,
            model=model,
            positive=ctx.eyes_conditioning,
            negative=ctx.negative_conditioning,
            steps=ctx.steps["detail_eyes"],
            cfg=self._scale_cfg(ctx.cfg["detail_eyes"]),
            denoise=(0.6, 0.5),
            num_iterations=2,
            seed_offset=self.metadata.order,
            optional_mask=cropped_mask if self.applymask else None,
            apply_cn=True,
            cn_strength=0.5,
        )

        cropped_image = VAEDecode(cropped_latent, ctx.vae)

        cropped_image, _, _ = ImageResize_(
            image=cropped_image,
            width=width,
            height=height,
            interpolation=ImageResize_.interpolation.lanczos,
            method=ImageResize_.method.keep_proportion,
        )

        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 10, 255)

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
