from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailEyesStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Eyes Detail", order=4)
    usebbox = True
    applymask = True

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        bbox_detector, segm_detector = MediaPipeFaceMeshDetectorProviderInspire(
            max_faces=1,
            face=False,
            mouth=False,
            left_eyebrow=False,
            left_eye=True,
            left_pupil=False,
            right_eyebrow=False,
            right_eye=True,
            right_pupil=False,
        )

        common_detector_settings = {
            "image": image,
            "threshold": 0.5,
            "dilation": 30,
            "crop_factor": 2,
            "drop_size": 10,
        }

        if self.usebbox:
            segs = BboxDetectorSEGS(
                bbox_detector=bbox_detector, **common_detector_settings
            )
        else:
            segs = SegmDetectorSEGS(
                segm_detector=segm_detector, **common_detector_settings
            )

        mask = SegsToCombinedMask(segs)
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

        cropped_image, _ = CRUpscaleImage(
            image=cropped_image,
            upscale_model=ctx.upscale_model_name,
            mode=CRUpscaleImage.mode.resize,
            resize_width=1024,
            resampling_method=CRUpscaleImage.resampling_method.lanczos,
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
            seed_offset=4,
            optional_mask=cropped_mask if self.applymask else None,
            apply_cn=True,
            cn_strength=0.5,
        )

        cropped_image = VAEDecode(cropped_latent, ctx.vae)

        cropped_image = ImageResize(
            image=cropped_image,
            mode=ImageResize.mode.resize,
            resampling=ImageResize.resampling.lanczos,
            resize_width=width,
            resize_height=height,
        )

        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 10, 255)

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
