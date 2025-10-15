from comfy_nodes import *

from utils import scale_cfg, scale_steps
from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class DetailFaceStep(WorkflowStep):
    metadata = WorkflowMetadata(
        label="Face Detail",
        order=3,
        parameters={
            "swap_method": {
                "type": "radio",
                "choices": [
                    ("Use InstantID & ReActor", "both"),
                    ("Use InstantID", "instantid"),
                    ("Use ReActor", "reactor"),
                    ("Use Prompt Only", "prompt"),
                ],
                "value": "instantid",
                "label": "Face Swap Method",
            },
            "strength": {
                "type": "slider",
                "minimum": 0,
                "maximum": 1,
                "step": 0.05,
                "value": 0.5,
                "label": "Swap Strength",
            },
            "cfg": {"type": "number", "label": "CFG", "value": 8},
        },
    )
    applymask = True

    def _init(self, swap_method=None, strength=None, cfg=None):
        self.swap_method = (
            swap_method
            if swap_method
            else self.metadata.parameters["swap_method"]["value"]
        )
        self.strength = (
            strength if strength else self.metadata.parameters["strength"]["value"]
        )
        base_step = 15
        base_cfg = 4

        if self.ctx.type in ["Lightning", "Hyper4S"]:
            step_scale = 6
        elif self.ctx.type in ["Hyper8S", "Turbo"]:
            step_scale = 10
        elif self.ctx.type == "fewsteplora":
            step_scale = 8
        else:
            step_scale = 30

        cfg_scale = cfg if cfg else self.metadata.parameters["cfg"]["value"]

        self.steps = scale_steps(base_step, step_scale)
        self.cfg = self._scale_cfg(scale_cfg(base_cfg, cfg_scale))

    def apply_face_sampling(self, state):
        image = state.image

        ctx = self.ctx

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
            device=sam.device.cpu,
        )

        cropped_image, cropped_mask, crop_box, _ = LayerUtilityCropByMaskV2(
            image,
            mask,
            detect="mask_area",
            top_reserve=250,
            bottom_reserve=250,
            left_reserve=250,
            right_reserve=250,
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

        positive = ctx.face_conditioning
        # positive = ConditioningConcat(ctx.face_conditioning, ctx.eyes_conditioning)

        if self.swap_method in ["instantid", "both"]:
            model, positive, negative = ApplyInstantIDAdvanced(
                instantid=ctx.instantid,
                insightface=ctx.faceanalysis,
                control_net=ctx.instantid_cn,
                image=ctx.face_image,
                model=ctx.model,
                positive=positive,
                negative=ctx.negative_conditioning,
                ip_weight=self.strength,
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
            steps=self.steps,
            cfg=self.cfg,
            denoise=(0.8, 0.6),
            num_iterations=3,
            seed_offset=self.metadata.order,
            optional_mask=cropped_mask if self.applymask else None,
            apply_color_match=True,
            apply_cn=True,
            cn_strength=1 - self.strength,
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

        return image

    def apply_reactor(self, state):
        image = state.image

        ctx = self.ctx

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

        options = ReActorOptions(restore_swapped_only=True)

        image, _, _ = ReActorFaceSwapOpt(
            enabled=True,
            input_image=image,
            face_model=face_model,
            face_boost=booster,
            swap_model=ReActorFaceSwapOpt.swap_model.inswapper_128_onnx,
            facedetection=ReActorFaceSwapOpt.facedetection.retinaface_resnet50,
            face_restore_model=ReActorFaceSwapOpt.face_restore_model.codeformer_v0_1_0,
            options=options,
        )

        return image

    def run(self, state: WorkflowState) -> WorkflowState:
        if self.swap_method in ["both", "reactor"]:
            image = self.apply_reactor(state)

        if self.swap_method in ["both", "instantid", "prompt"]:
            image = self.apply_face_sampling(state)

        latent = VAEDecode(image, self.ctx.vae)

        return state.update(image=image, latent=latent)
