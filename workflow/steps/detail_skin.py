from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailSkinStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Skin Detail", order=3)
    usebbox = False
    applymask = True

    def _skin_segment(self, image):
        ctx = self.ctx
        gd_sam = GroundingDinoSAMSegmentSegmentAnything

        _, person_mask = gd_sam(
            ctx.sam_model, ctx.gd_model, image, prompt="person", threshold=0.5
        )
        _, clothes_mask = gd_sam(
            ctx.sam_model, ctx.gd_model, image, prompt="clothes", threshold=0.5
        )
        _, hair_mask = gd_sam(
            ctx.sam_model, ctx.gd_model, image, prompt="hair", threshold=0.5
        )

        person_mask = MaskErodeRegion(person_mask, 10)
        clothes_mask = MaskDilateRegion(clothes_mask, 10)
        hair_mask = MaskDilateRegion(hair_mask, 10)

        mask = SubtractMask(person_mask, clothes_mask)
        mask = SubtractMask(mask, hair_mask)

        mask = MaskDilateRegion(mask, 10)
        mask = MaskFillHoles(mask)

        return mask

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        mask = self._skin_segment(image)

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

        if ctx.swap_method == "instantid":
            model, positive, negative = ApplyInstantIDAdvanced(
                instantid=ctx.instantid,
                insightface=ctx.faceanalysis,
                control_net=ctx.instantid_cn,
                image=ctx.face_image,
                model=ctx.model,
                positive=ctx.skin_conditioning,
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
            model, positive, negative = (
                ctx.model,
                ctx.skin_conditioning,
                ctx.negative_conditioning,
            )

        model = DifferentialDiffusion(ctx.model)

        cropped_mask = GrowMask(cropped_mask, expand=30)
        cropped_mask = MaskBlur(cropped_mask, amount=70)

        cropped_latent = VAEEncode(cropped_image, ctx.vae)

        cropped_latent = self._iterative_latent_upscale(
            latent=cropped_latent,
            scale=1.6,
            model=model,
            positive=positive,
            negative=negative,
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
            method=ImageResize_.method.keep_proportion,
        )

        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 20, 255)

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
