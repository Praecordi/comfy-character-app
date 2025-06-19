from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailFaceStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Face Detail", order=2)
    usebbox = True
    applymask = True

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        bbox_detector, segm_detector = MediaPipeFaceMeshDetectorProviderInspire(
            max_faces=1,
            face=True,
            mouth=False,
            left_eyebrow=False,
            left_eye=False,
            left_pupil=False,
            right_eyebrow=False,
            right_eye=False,
            right_pupil=False,
        )

        common_detector_settings = {
            "image": image,
            "threshold": 0.5,
            "dilation": 30,
            "crop_factor": 3,
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
            combined=False,
            crop_factor=3,
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
                interpolation=ImageResize_.interpolation.lanczos,
                method=ImageResize_.method.keep_proportion,
            )

        positive = ConditioningConcat(ctx.face_conditioning, ctx.positive_conditioning)

        if ctx.use_instantid:
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
                end_at=1,
                noise=0,
                combine_embeds=ApplyInstantIDAdvanced.combine_embeds.average,
                image_kps=cropped_image,
            )
        else:
            model, positive, negative = ctx.model, positive, ctx.negative_conditioning

        model = DifferentialDiffusion(model)

        cropped_mask = GrowMask(cropped_mask, expand=50)
        cropped_mask = MaskBlur(cropped_mask, amount=70)

        cropped_latent = VAEEncode(cropped_image, ctx.vae)

        cropped_latent = self._iterative_latent_upscale(
            latent=cropped_latent,
            scale=1.6,
            model=model,
            positive=positive,
            negative=negative,
            steps=ctx.steps["detail_face"],
            cfg=self._scale_cfg(ctx.cfg["detail_face"]),
            denoise=(0.7, 0.6),
            num_iterations=3,
            seed_offset=2,
            optional_mask=cropped_mask if self.applymask else None,
            apply_cn=True,
            cn_strength=0.5,
            cn_limits=(0, 0.5),
        )

        cropped_image = VAEDecode(cropped_latent, ctx.vae)

        cropped_image, _, _ = ImageResize_(
            image=cropped_image,
            width=width,
            height=height,
            interpolation=ImageResize_.interpolation.lanczos,
            method=ImageResize_.method.keep_proportion,
        )

        cropped_image = FaceRestoreCFWithModel(
            FaceRestoreModelLoader(FaceRestoreModelLoader.model_name.codeformer_v0_1_0),
            cropped_image,
            FaceRestoreCFWithModel.facedetection.retinaface_resnet50,
            0.5,
        )

        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 30, 255)

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
