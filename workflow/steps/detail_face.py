from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailFaceStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Face Detail", order=2)
    usebbox = False
    applymask = True

    def apply_face_sampling(self, state):
        ctx = self.ctx

        image = state.image

        _, mask = GroundingDinoSAMSegmentSegmentAnything(
            ctx.sam_model, ctx.gd_model, image, prompt="face", threshold=0.5
        )

        if self.usebbox:
            image_width, image_height, _ = GetImageSize(image)
            _, _, x, y, width, height = MaskBoundingBox(mask)
            mask = MaskRectAreaAdvanced(x, y, width, height, image_width, image_height)

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
                height=2048,
                interpolation=ImageResize_.interpolation.lanczos,
                method=ImageResize_.method.keep_proportion,
            )

        positive = ConditioningConcat(ctx.face_conditioning, ctx.positive_conditioning)

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
            seed_offset=self.metadata.order,
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

        cropped_image = ImageColorMatch(
            image=cropped_image,
            reference=image,
            color_space=ImageColorMatch.color_space.RGB,
            factor=0.5,
            reference_mask=mask,
        )

        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        detailed = SEGSPaste(image, segs, 30, 255)

        image = ReActorRestoreFace(
            detailed,
            ReActorRestoreFace.facedetection.retinaface_resnet50,
            model=ReActorRestoreFace.model.codeformer_v0_1_0,
            visibility=1,
            codeformer_weight=0.5,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)

    def apply_reactor(self, state):
        ctx = self.ctx

        image = state.image

        face_model = ReActorBuildFaceModel(
            False,
            False,
            "default",
            ReActorBuildFaceModel.compute_method.Mean,
            images=ctx.face_image,
        )

        booster = ReActorFaceBoost(
            enabled=True,
            boost_model=ReActorFaceBoost.boost_model.codeformer_v0_1_0,
            interpolation=ReActorFaceBoost.interpolation.Lanczos,
            visibility=1,
            codeformer_weight=0.5,
            restore_with_main_after=True,
        )

        image, _, _ = ReActorFaceSwap(
            enabled=True,
            input_image=image,
            face_model=face_model,
            face_boost=booster,
            swap_model=ReActorFaceSwap.swap_model.inswapper_128_onnx,
            facedetection=ReActorFaceSwap.facedetection.retinaface_resnet50,
            face_restore_model=ReActorFaceSwap.face_restore_model.codeformer_v0_1_0,
        )

        latent = VAEDecode(image, ctx.vae)

        return state.update(image=image, latent=latent)

    def run(self, state: WorkflowState) -> WorkflowState:
        if self.ctx.swap_method in ["instantid", "prompt"]:
            return self.apply_face_sampling(state)
        else:
            return self.apply_reactor(state)
