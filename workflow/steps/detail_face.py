from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailFaceStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Face Detail", order=3)
    applymask = True

    def apply_face_sampling(self, state):
        ctx = self.ctx

        image = state.image

        sam = LayerMaskSegmentAnythingUltraV2

        _, mask = sam(
            image,
            sam.sam_model.sam_vit_h_2_56GB,
            sam.grounding_dino_model.GroundingDINO_SwinB_938MB,
            detail_method=sam.detail_method.VITMatte,
            detail_erode=18,
            detail_dilate=18,
            prompt="face",
            threshold=0.5,
            cache_model=False,
        )

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
                ip_weight=0.8,
                cn_strength=0.5,
                start_at=0.8,
                end_at=1.0,
                noise=0.1,
                combine_embeds=ApplyInstantIDAdvanced.combine_embeds.average,
                image_kps=cropped_image,
            )
        else:
            model, positive, negative = ctx.model, positive, ctx.negative_conditioning

        model = DifferentialDiffusion(model)

        cropped_mask = GrowMask(cropped_mask, expand=50)
        cropped_mask = MaskBlur(cropped_mask, amount=70)

        cropped_image = self._iterative_image_upscale(
            image=cropped_image,
            scale=1.6,
            model=model,
            positive=positive,
            negative=negative,
            steps=ctx.steps["detail_face"],
            cfg=self._scale_cfg(ctx.cfg["detail_face"]),
            denoise=(0.8, 0.6),
            num_iterations=3,
            seed_offset=self.metadata.order,
            optional_mask=cropped_mask if self.applymask else None,
            apply_color_match=True,
            apply_cn=True,
            cn_strength=0.6,
            cn_limits=(0, 1),
        )

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
