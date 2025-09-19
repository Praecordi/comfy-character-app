from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailSkinStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Skin Detail", order=2)
    usebbox = False
    applymask = True

    def _skin_segment(self, image):
        sam = LayerMaskSegmentAnythingUltraV2

        _, person_mask = sam(
            image,
            sam.sam_model.sam_vit_h_2_56GB,
            sam.grounding_dino_model.GroundingDINO_SwinB_938MB,
            detail_method=sam.detail_method.VITMatte,
            detail_erode=36,
            detail_dilate=6,
            prompt="person",
            threshold=0.5,
            cache_model=False,
        )
        _, clothes_mask = sam(
            image,
            sam.sam_model.sam_vit_h_2_56GB,
            sam.grounding_dino_model.GroundingDINO_SwinB_938MB,
            detail_method=sam.detail_method.VITMatte,
            detail_erode=6,
            detail_dilate=36,
            prompt="clothes",
            threshold=0.5,
            cache_model=False,
        )
        _, hair_mask = sam(
            image,
            sam.sam_model.sam_vit_h_2_56GB,
            sam.grounding_dino_model.GroundingDINO_SwinB_938MB,
            detail_method=sam.detail_method.VITMatte,
            detail_erode=6,
            detail_dilate=36,
            prompt="hair",
            threshold=0.5,
            cache_model=False,
        )

        mask = MasksSubtract(person_mask, clothes_mask)
        mask = MasksSubtract(mask, hair_mask)

        return mask

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        mask = self._skin_segment(image)

        cropped_image, cropped_mask, crop_box, _ = LayerUtilityCropByMaskV2(
            image,
            mask,
            detect="mask_area",
            top_reserve=100,
            bottom_reserve=100,
            left_reserve=100,
            right_reserve=100,
        )
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

        cropped_mask = GrowMask(cropped_mask, expand=30)
        cropped_mask = MaskBlur(cropped_mask, amount=70)

        cropped_latent = VAEEncode(cropped_image, ctx.vae)

        cropped_latent = self._iterative_latent_upscale(
            latent=cropped_latent,
            scale=1.6,
            model=model,
            positive=ctx.skin_conditioning,
            negative=ctx.negative_conditioning,
            steps=ctx.steps["detail_skin"],
            cfg=self._scale_cfg(ctx.cfg["detail_skin"]),
            denoise=(0.7, 0.6),
            num_iterations=2,
            seed_offset=self.metadata.order,
            optional_mask=cropped_mask if self.applymask else None,
            apply_cn=True,
            cn_strength=0.75,
        )

        cropped_image = VAEDecode(cropped_latent, ctx.vae)

        cropped_image, _, _ = ImageResize_(
            image=cropped_image,
            width=width,
            height=height,
            interpolation=ImageResize_.interpolation.lanczos,
            method=ImageResize_.method.stretch,
        )

        image, _ = LayerUtilityRestoreCropBox(
            image, cropped_image, False, crop_box, cropped_mask
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
