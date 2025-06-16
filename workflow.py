import re
import random

from constants import comfyui_server_url, characters, app_constants

from comfy_script.runtime import *

load(comfyui_server_url)
from comfy_script.runtime.nodes import *

queue.watch_display(False)


def clean_and_separate(prompt):
    prompt = re.sub(r"\r\n|\n|\r", " ", prompt).strip()

    separated = prompt.split(",")
    separated = [x.strip() for x in separated]

    return separated


def gen_step(label, order, default=False):
    def decorator(func):
        func._is_gen_step = True
        func._step_label = label
        func._step_order = order
        func._default_step = default
        return func

    return decorator


def expand_iterations_linear(value, n, callback=lambda x: x):
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
        if n == 1:
            return [callback(end)] * n

        step_size = (end - start) / (n - 1)
        return [callback(start + i * step_size) for i in range(n)]
    elif isinstance(value, (int, float)):
        return [callback(value)] * n
    else:
        raise ValueError("Provide a value of int/float or tuple of size 2 to expand")


def expand_iterations_geometric(value, n, callback=lambda x: x):
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
        if n == 1:
            return [callback(end)] * n

        if start <= 0 or end <= 0:
            raise ValueError("Both start and end must be positive")

        ratio = (end / start) ** (1 / (n - 1))
        return [callback(start * (ratio**i)) for i in range(n)]
    elif isinstance(value, (int, float)):
        return [callback(value)] * n
    else:
        raise ValueError("Provide a value of int/float or tuple of size 2 to expand")


class CharacterWorkflow:
    def __init__(
        self,
        checkpoint,
        fewsteplora,
        resolution,
        upscaler,
        style_prompt,
        base_seed,
        perturb_seed,
        cn_image,
        cn_strength,
        style_image,
        style_strength,
        hair_prompt,
        eyes_prompt,
        face_image,
        pos_prompt,
        neg_prompt,
        character,
        preview_callback=None,
    ):
        self.preview_callback = preview_callback

        self.main_model, self.main_clip, self.main_vae = CheckpointLoaderSimple(
            checkpoint
        )

        if not cn_image is None:
            self.cn_image, _ = LoadImage(cn_image)
            self.cn_strength = cn_strength
        else:
            self.cn_image = None

        if not style_image is None:
            self.style_image, _ = LoadImage(style_image)
            self.style_strength = style_strength
            self.main_model, ipadapter = IPAdapterUnifiedLoader(
                self.main_model, IPAdapterUnifiedLoader.preset.STANDARD_medium_strength
            )
            style_negative = IPAdapterNoise(
                IPAdapterNoise.type.shuffle, 0.4, 10, self.style_image
            )
            self.main_model = IPAdapterPreciseStyleTransfer(
                self.main_model,
                ipadapter,
                self.style_image,
                self.style_strength / 100,
                1,
                IPAdapterPreciseStyleTransfer.combine_embeds.concat,
                0,
                1,
                IPAdapterPreciseStyleTransfer.embeds_scaling.K_V,
                style_negative,
            )
        else:
            self.style_image = None

        if "Lightning" in checkpoint:
            self.sampler = Samplers.dpmpp_2s_ancestral
            self.scheduler = Schedulers.normal
            self.steps = {
                "base_gen": 6,
                "latent_upscale": (6, 4),
                "detail_face": 4,
                "detail_hair": (6, 4),
                "detail_eyes": (6, 4),
                "image_upscale": (4, 2),
            }
            self.cfg = {
                "base_gen": 2,
                "latent_upscale": (1, 2),
                "detail_face": 1.5,
                "detail_hair": (1.5, 2),
                "detail_eyes": (1.5, 2),
                "image_upscale": (2, 2.5),
            }
        elif "Hyper4S" in checkpoint:
            self.sampler = Samplers.dpmpp_2s_ancestral
            self.scheduler = Schedulers.normal
            self.steps = {
                "base_gen": 6,
                "latent_upscale": (6, 4),
                "detail_face": 6,
                "detail_hair": (6, 4),
                "detail_eyes": (6, 4),
                "image_upscale": (4, 2),
            }
            self.cfg = {
                "base_gen": 2,
                "latent_upscale": (1, 2),
                "detail_face": 1.5,
                "detail_hair": (1.5, 2),
                "detail_eyes": (1.5, 2),
                "image_upscale": (2, 2.5),
            }
        elif "Hyper8S" in checkpoint:
            self.sampler = Samplers.dpmpp_2s_ancestral
            self.scheduler = Schedulers.normal
            self.steps = {
                "base_gen": 10,
                "latent_upscale": (8, 6),
                "detail_face": 8,
                "detail_hair": (8, 6),
                "detail_eyes": (8, 6),
                "image_upscale": (6, 4),
            }
            self.cfg = {
                "base_gen": 2,
                "latent_upscale": (1, 2),
                "detail_face": 1.5,
                "detail_hair": (1.5, 2),
                "detail_eyes": (1.5, 2),
                "image_upscale": (2, 2.5),
            }
        elif "Turbo" in checkpoint:
            self.sampler = Samplers.dpmpp_2s_ancestral
            self.scheduler = Schedulers.normal
            self.steps = {
                "base_gen": 8,
                "latent_upscale": (6, 4),
                "detail_face": 6,
                "detail_hair": (6, 4),
                "detail_eyes": (6, 4),
                "image_upscale": (4, 2),
            }
            self.cfg = {
                "base_gen": 2,
                "latent_upscale": (1, 2),
                "detail_face": 1.5,
                "detail_hair": (1.5, 2),
                "detail_eyes": (1.5, 2),
                "image_upscale": (2, 2.5),
            }
        else:
            if (
                fewsteplora == "lcm"
                or fewsteplora == "turbo"
                or fewsteplora == "dpo_turbo"
            ):
                if fewsteplora == "lcm":
                    self.main_model, self.main_clip = LoraLoader(
                        self.main_model, self.main_clip, app_constants["lcm_lora"], 1, 1
                    )
                elif fewsteplora == "turbo":
                    self.main_model, self.main_clip = LoraLoader(
                        self.main_model,
                        self.main_clip,
                        app_constants["turbo_lora"],
                        1,
                        1,
                    )
                else:
                    self.main_model, self.main_clip = LoraLoader(
                        self.main_model,
                        self.main_clip,
                        app_constants["dpo_turbo_lora"],
                        1,
                        1,
                    )
                self.sampler = Samplers.lcm
                self.scheduler = Schedulers.sgm_uniform
                self.steps = {
                    "base_gen": 8,
                    "latent_upscale": (6, 4),
                    "detail_face": 6,
                    "detail_hair": (6, 4),
                    "detail_eyes": (6, 4),
                    "image_upscale": (6, 4),
                }
                self.cfg = {
                    "base_gen": 2,
                    "latent_upscale": (1, 2),
                    "detail_face": 1.5,
                    "detail_hair": (1.5, 2),
                    "detail_eyes": (1.5, 2),
                    "image_upscale": (2, 2.5),
                }
            else:
                self.sampler = Samplers.dpmpp_2m_sde_gpu
                self.scheduler = Schedulers.karras
                self.steps = {
                    "base_gen": 30,
                    "latent_upscale": (20, 10),
                    "detail_face": 15,
                    "detail_hair": (15, 10),
                    "detail_eyes": (15, 10),
                    "image_upscale": (15, 10),
                }
                self.cfg = {
                    "base_gen": 8,
                    "latent_upscale": (4, 6),
                    "detail_face": 4,
                    "detail_hair": (6, 8),
                    "detail_eyes": (6, 8),
                    "image_upscale": (8, 10),
                }

        self.upscale_model_name = upscaler
        self.upscale_model = UpscaleModelLoader(self.upscale_model_name)

        self.cn = ControlNetLoader(app_constants["union_controlnet"])
        self.cn = SetUnionControlNetType(self.cn, "auto")

        self.resolution = resolution
        self.base_seed = (
            random.randint(0, 10000000000) if base_seed == -1 else base_seed
        )
        self.perturb_seed = (
            random.randint(0, 10000000000) if perturb_seed == -1 else perturb_seed
        )

        self.resolved_prompts = self._process_prompt(
            pos_prompt,
            neg_prompt,
            style_prompt=style_prompt,
            hair_prompt=hair_prompt,
            eyes_prompt=eyes_prompt,
            apply_character=character,
            apply_scores=checkpoint.startswith("pony"),
        )

        self.pos_condition = CLIPTextEncodeSDXL(
            self.main_clip,
            4096,
            4096,
            0,
            0,
            4096,
            4096,
            self.resolved_prompts["positive"],
            self.resolved_prompts["positive"],
        )

        self.neg_condition = CLIPTextEncodeSDXL(
            self.main_clip,
            4096,
            4096,
            0,
            0,
            4096,
            4096,
            self.resolved_prompts["negative"],
            self.resolved_prompts["negative"],
        )

        self.hair_condition = CLIPTextEncodeSDXL(
            self.main_clip,
            4096,
            4096,
            0,
            0,
            4096,
            4096,
            self.resolved_prompts["hair"],
            self.resolved_prompts["hair"],
        )

        self.eyes_condition = CLIPTextEncodeSDXL(
            self.main_clip,
            4096,
            4096,
            0,
            0,
            4096,
            4096,
            self.resolved_prompts["eyes"],
            self.resolved_prompts["eyes"],
        )

        self.face_image, _ = LoadImage(face_image)

        self.instantid = InstantIDModelLoader(app_constants["instantid_model"])
        self.faceanalysis = InstantIDFaceAnalysis("CPU")
        self.instantid_cn = ControlNetLoader(app_constants["instantid_controlnet"])

    @classmethod
    def get_steps(cls):
        steps = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if (
                callable(str)
                and getattr(attr, "_is_gen_step", False)
                and not getattr(attr, "_default_step", False)
            ):
                steps.append((attr._step_order, attr._step_label))

        steps = [x[1] for x in sorted(steps, key=lambda x: x[0])]

        return steps

    def _process_prompt(
        self,
        pos_prompt,
        neg_prompt,
        style_prompt=None,
        hair_prompt=None,
        eyes_prompt=None,
        apply_scores=False,
        apply_character=None,
    ):
        pos_separated = clean_and_separate(pos_prompt)
        neg_separated = clean_and_separate(neg_prompt)

        if style_prompt is not None:
            style_separated = clean_and_separate(style_prompt)
            pos_separated += style_separated

        if apply_scores:
            pos_separated = ["score_9", "score_8_up", "score_7_up"] + pos_separated
            neg_separated = ["score_1", "score_2", "score_3"] + neg_separated

        to_return = {
            "positive": ", ".join(pos_separated),
            "negative": ", ".join(neg_separated),
        }

        if hair_prompt is not None:
            hair_separated = clean_and_separate(hair_prompt)

            if apply_scores:
                hair_separated = ["score_9"] + hair_separated

            to_return["hair"] = ", ".join(hair_separated)

        if eyes_prompt is not None:
            eyes_separated = clean_and_separate(eyes_prompt)

            if apply_scores:
                eyes_separated = ["score_9"] + eyes_separated

            to_return["eyes"] = ", ".join(eyes_separated)

        if apply_character is not None and not apply_character == "Custom":
            for key, val in characters[apply_character.lower()].items():
                to_return["positive"] = to_return["positive"].replace(
                    "{{{0}}}".format(key), val
                )
                to_return["negative"] = to_return["negative"].replace(
                    "{{{0}}}".format(key), val
                )

        return to_return

    def _scale_cfg(self, cfg, scale_for_cn=False):
        scale = False
        if not self.style_image is None:
            scale = True
        else:
            if scale_for_cn:
                if not self.cn_image is None:
                    scale = True

        if scale:
            if isinstance(cfg, tuple):
                return tuple([(i + 1) / 2 for i in cfg])
            else:
                return (cfg + 1) / 2
        else:
            return cfg

    def _iterative_latent_upscale(
        self,
        latent,
        scale,
        pipe,
        steps,
        cfg,
        denoise,
        num_iterations=1,
        seed_offset=0,
        optional_mask=None,
    ):
        steps = expand_iterations_linear(steps, num_iterations)
        cfg = expand_iterations_linear(cfg, num_iterations)
        denoise = expand_iterations_linear(denoise, num_iterations)
        ratio = scale ** (1 / num_iterations)

        for i in range(num_iterations):
            latent = NNLatentUpscale(latent, "SDXL", ratio)
            if optional_mask is not None:
                latent = SetLatentNoiseMask(latent, optional_mask)
            _, latent, _ = ImpactKSamplerBasicPipe(
                pipe,
                latent_image=latent,
                seed=self.base_seed + seed_offset,
                steps=steps[i],
                cfg=cfg[i],
                denoise=denoise[i],
                sampler_name=self.sampler,
                scheduler=self.scheduler,
            )

        return latent

    def _iterative_image_upscale(
        self,
        image,
        scale,
        pipe,
        steps,
        cfg,
        denoise,
        num_iterations=1,
        seed_offset=0,
        optional_mask=None,
        sharpen=0.8,
    ):
        steps = expand_iterations_linear(
            steps, num_iterations, callback=lambda x: int(x)
        )
        cfg = expand_iterations_linear(
            cfg, num_iterations, callback=lambda x: round(x, 2)
        )
        denoise = expand_iterations_linear(
            denoise, num_iterations, callback=lambda x: round(x, 2)
        )
        ratio = scale ** (1 / num_iterations)

        for i in range(num_iterations):
            image, _ = CRUpscaleImage(
                image,
                self.upscale_model_name,
                "rescale",
                rescale_factor=ratio,
                resampling_method="lanczos",
            )
            image = ImageCASharpening(image, sharpen)
            latent = VAEEncode(image, self.main_vae)
            if optional_mask is not None:
                latent = SetLatentNoiseMask(latent, optional_mask)
            _, latent, _ = ImpactKSamplerBasicPipe(
                pipe,
                latent_image=latent,
                seed=self.base_seed + seed_offset,
                steps=steps[i],
                cfg=cfg[i],
                denoise=denoise[i],
                sampler_name=self.sampler,
                scheduler=self.scheduler,
            )
            image = VAEDecode(latent, self.main_vae)

        return image

    @gen_step(label="Base Generation", order=0, default=True)
    def base_gen(self, latent, image):
        if not self.cn_image is None:
            positive, negative = ControlNetApplyAdvanced(
                self.pos_condition,
                self.neg_condition,
                self.cn,
                self.cn_image,
                self.cn_strength / 100,
                0,
                1,
                self.main_vae,
            )
        else:
            positive, negative = self.pos_condition, self.neg_condition

        main_pipe = ToBasicPipe(
            self.main_model, self.main_clip, self.main_vae, positive, negative
        )

        perturb_noise = RandomNoise(self.perturb_seed)

        sigmas = BasicScheduler(
            self.main_model, self.scheduler, self.steps["base_gen"], 1
        )
        _, sigmas2 = SplitSigmas(sigmas, int(self.steps["base_gen"] * 0.65))

        latent = AddNoise(self.main_model, perturb_noise, sigmas2, latent)

        _, latent, _ = ImpactKSamplerBasicPipe(
            basic_pipe=main_pipe,
            seed=self.base_seed,
            steps=self.steps["base_gen"],
            cfg=self._scale_cfg(self.cfg["base_gen"], scale_for_cn=True),
            sampler_name=self.sampler,
            scheduler=self.scheduler,
            latent_image=latent,
            denoise=1,
        )

        image = VAEDecode(latent, self.main_vae)

        return latent, image

    @gen_step(label="Latent Upscale", order=1)
    def latent_upscale(self, latent, image):
        positive, negative = ControlNetApplyAdvanced(
            self.pos_condition,
            self.neg_condition,
            control_net=self.cn,
            image=image,
            strength=0.3,
            start_percent=0,
            end_percent=1,
            vae=self.main_vae,
        )

        if not self.style_image is None:
            model, ipadapter = IPAdapterUnifiedLoader(
                self.main_model, IPAdapterUnifiedLoader.preset.STANDARD_medium_strength
            )
            style_negative = IPAdapterNoise(
                IPAdapterNoise.type.shuffle, 0.4, 10, self.style_image
            )
            model = IPAdapterPreciseStyleTransfer(
                model,
                ipadapter,
                self.style_image,
                1,
                1,
                IPAdapterPreciseStyleTransfer.combine_embeds.concat,
                0,
                1,
                IPAdapterPreciseStyleTransfer.embeds_scaling.K_V,
                style_negative,
            )
        else:
            model = self.main_model

        main_pipe = ToBasicPipe(
            model, self.main_clip, self.main_vae, positive, negative
        )

        latent = self._iterative_latent_upscale(
            latent=latent,
            scale=1.6,
            pipe=main_pipe,
            steps=self.steps["latent_upscale"],
            cfg=self._scale_cfg(self.cfg["latent_upscale"]),
            denoise=(0.8, 0.6),
            num_iterations=3,
            seed_offset=1,
        )

        image = VAEDecode(latent, self.main_vae)

        return latent, image

    @gen_step(label="Face Detail", order=2)
    def detail_face(self, latent, image):
        bbox_detector, segm_detector = MediaPipeFaceMeshDetectorProviderInspire(
            1,
            face=True,
            mouth=False,
            left_eyebrow=False,
            left_eye=False,
            left_pupil=False,
            right_eyebrow=False,
            right_eye=False,
            right_pupil=False,
        )

        segs = BboxDetectorSEGS(bbox_detector, image, 0.5, 30, 2, 10)
        # segs = SegmDetectorSEGS(segm_detector, image, 0.5, 30, 2, 10)
        mask = SegsToCombinedMask(segs)
        segs = MaskToSEGS(mask, False, 2, False, 10, True)
        segs = SetDefaultImageForSEGS(segs, image, True)
        segs_header, seg_elt = ImpactDecomposeSEGS(segs)
        seg_elt, cropped_image, cropped_mask, _, _, _, _, _ = ImpactFromSEGELT(seg_elt)
        width, height, _ = GetImageSize(cropped_image)

        cropped_image, _ = CRUpscaleImage(
            cropped_image,
            upscale_model=self.upscale_model_name,
            mode="resize",
            resize_width=1024,
        )

        positive, negative = ControlNetApplyAdvanced(
            self.pos_condition,
            ConditioningZeroOut(self.neg_condition),
            control_net=self.cn,
            image=cropped_image,
            strength=0.5,
            start_percent=0,
            end_percent=1,
            vae=self.main_vae,
        )
        model, positive, negative = ApplyInstantIDAdvanced(
            instantid=self.instantid,
            insightface=self.faceanalysis,
            control_net=self.instantid_cn,
            image=self.face_image,
            model=self.main_model,
            positive=positive,
            negative=negative,
            ip_weight=0.7,
            cn_strength=0.5,
            start_at=0,
            end_at=1,
            noise=0,
            combine_embeds="average",
            image_kps=cropped_image,
        )
        model = DifferentialDiffusion(model)

        instantid_pipe = ToBasicPipe(
            model, self.main_clip, self.main_vae, positive, negative
        )

        latent_mask = GrowMask(cropped_mask, 50, True)
        latent_mask = MaskBlur(latent_mask, 70, "auto")

        cropped_image = self._iterative_image_upscale(
            image=cropped_image,
            scale=1.6,
            pipe=instantid_pipe,
            steps=self.steps["detail_face"],
            cfg=self._scale_cfg(self.cfg["detail_face"]),
            denoise=(0.4, 0.2),
            num_iterations=3,
            seed_offset=2,
            optional_mask=latent_mask,
            sharpen=0.6,
        )

        cropped_image, _, _ = ImageResize_(
            cropped_image, width, height, "lanczos", "keep proportion", "always", 0
        )
        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 30, 255)

        latent = VAEEncode(image, self.main_vae)

        return latent, image

    @gen_step(label="Hair Detail", order=3)
    def detail_hair(self, latent, image):
        _, segm_detector = UltralyticsDetectorProvider(app_constants["hair_seg_model"])

        segs = SegmDetectorSEGS(segm_detector, image, 0.5, 30, 2, 10)
        mask = SegsToCombinedMask(segs)
        segs = MaskToSEGS(mask, True, 2, False, 10, True)
        segs = SetDefaultImageForSEGS(segs, image, True)
        segs_header, seg_elt = ImpactDecomposeSEGS(segs)
        seg_elt, cropped_image, cropped_mask, _, _, _, _, _ = ImpactFromSEGELT(seg_elt)
        width, height, _ = GetImageSize(cropped_image)

        cropped_image, _ = CRUpscaleImage(
            cropped_image,
            upscale_model=self.upscale_model_name,
            mode="resize",
            resize_width=1024,
        )

        positive, negative = ControlNetApplyAdvanced(
            self.hair_condition,
            ConditioningZeroOut(self.neg_condition),
            control_net=self.cn,
            image=cropped_image,
            strength=0.5,
            start_percent=0,
            end_percent=1,
            vae=self.main_vae,
        )
        model = DifferentialDiffusion(self.main_model)

        main_pipe = ToBasicPipe(
            model, self.main_clip, self.main_vae, positive, negative
        )

        cropped_mask = GrowMask(cropped_mask, 30, True)
        cropped_mask = MaskBlur(cropped_mask, 70, "auto")
        cropped_latent = VAEEncode(cropped_image, self.main_vae)

        cropped_latent = self._iterative_latent_upscale(
            latent=cropped_latent,
            scale=1.6,
            pipe=main_pipe,
            steps=self.steps["detail_hair"],
            cfg=self._scale_cfg(self.cfg["detail_hair"]),
            denoise=(0.7, 0.5),
            num_iterations=2,
            seed_offset=3,
            optional_mask=cropped_mask,
        )

        cropped_image = VAEDecode(cropped_latent, self.main_vae)
        cropped_image, _, _ = ImageResize_(
            cropped_image, width, height, "lanczos", "keep proportion", "always", 0
        )
        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 20, 255)

        latent = VAEEncode(image, self.main_vae)

        return latent, image

    @gen_step(label="Eyes Detail", order=4)
    def detail_eyes(self, latent, image):
        bbox_detector, segm_detector = MediaPipeFaceMeshDetectorProviderInspire(
            1,
            face=False,
            mouth=False,
            left_eyebrow=False,
            left_eye=True,
            left_pupil=False,
            right_eyebrow=False,
            right_eye=True,
            right_pupil=False,
        )

        segs = BboxDetectorSEGS(bbox_detector, image, 0.5, 30, 2, 10)
        # segs = SegmDetectorSEGS(segm_detector, image, 0.5, 30, 2, 10)
        mask = SegsToCombinedMask(segs)
        segs = MaskToSEGS(mask, True, 2, False, 10, True)
        segs = SetDefaultImageForSEGS(segs, image, True)
        segs_header, seg_elt = ImpactDecomposeSEGS(segs)
        seg_elt, cropped_image, cropped_mask, _, _, _, _, _ = ImpactFromSEGELT(seg_elt)
        width, height, _ = GetImageSize(cropped_image)

        cropped_image, _ = CRUpscaleImage(
            cropped_image,
            upscale_model=self.upscale_model_name,
            mode="resize",
            resize_width=1024,
        )

        positive, negative = ControlNetApplyAdvanced(
            self.eyes_condition,
            ConditioningZeroOut(self.neg_condition),
            control_net=self.cn,
            image=cropped_image,
            strength=0.5,
            start_percent=0,
            end_percent=1,
            vae=self.main_vae,
        )
        model = DifferentialDiffusion(self.main_model)

        main_pipe = ToBasicPipe(
            model, self.main_clip, self.main_vae, positive, negative
        )

        latent_mask = GrowMask(cropped_mask, 20, True)
        latent_mask = MaskBlur(latent_mask, 50, "auto")
        cropped_latent = VAEEncode(cropped_image, self.main_vae)

        cropped_latent = self._iterative_latent_upscale(
            latent=cropped_latent,
            scale=2,
            pipe=main_pipe,
            steps=self.steps["detail_eyes"],
            cfg=self._scale_cfg(self.cfg["detail_eyes"]),
            denoise=(0.6, 0.4),
            num_iterations=2,
            seed_offset=4,
            optional_mask=cropped_mask,
        )

        cropped_image = VAEDecode(cropped_latent, self.main_vae)
        cropped_image, _, _ = ImageResize_(
            cropped_image, width, height, "lanczos", "keep proportion", "always", 0
        )
        seg_elt = ImpactEditSEGELT(seg_elt, cropped_image)
        segs = ImpactAssembleSEGS(segs_header, seg_elt)
        image = SEGSPaste(image, segs, 10, 255)

        latent = VAEEncode(image, self.main_vae)

        return latent, image

    @gen_step(label="Image Upscale", order=5)
    def image_upscale(self, latent, image):
        positive = ConditioningConcat(self.pos_condition, self.eyes_condition)
        positive = ConditioningConcat(positive, self.hair_condition)
        positive, negative = ControlNetApplyAdvanced(
            positive,
            self.neg_condition,
            control_net=self.cn,
            image=image,
            strength=0.7,
            start_percent=0,
            end_percent=1,
            vae=self.main_vae,
        )
        model, positive, negative = ApplyInstantIDAdvanced(
            instantid=self.instantid,
            insightface=self.faceanalysis,
            control_net=self.instantid_cn,
            image=self.face_image,
            model=self.main_model,
            positive=positive,
            negative=negative,
            ip_weight=0.5,
            cn_strength=0.3,
            start_at=0.4,
            end_at=0.9,
            noise=0,
            combine_embeds="average",
            image_kps=image,
        )

        main_pipe = ToBasicPipe(
            model, self.main_clip, self.main_vae, positive, negative
        )

        image = self._iterative_image_upscale(
            image=image,
            scale=1.3,
            pipe=main_pipe,
            steps=self.steps["image_upscale"],
            cfg=self._scale_cfg(self.cfg["image_upscale"]),
            denoise=(0.8, 0.7),
            num_iterations=2,
            seed_offset=5,
            sharpen=0.4,
        )
        # skin_model = UpscaleModelLoader(UpscaleModels._1x_ITF_SkinDiffDetail_Lite_v1)
        # image = ImageUpscaleWithModel(skin_model, image)

        latent = VAEEncode(image, self.main_vae)

        return latent, image

    @gen_step(label="Remove Background", order=6)
    def remove_background(self, latent, image):
        session = TransparentBGSession(TransparentBGSession.mode.base, use_jit=True)
        image, _ = ImageRemoveBackground(session, image)
        # image = ImageRembgRemoveBackground(
        #     image,
        #     False,
        #     ImageRembgRemoveBackground.model.silueta,
        #     alpha_matting=True,
        #     alpha_matting_foreground_threshold=240,
        #     alpha_matting_background_threshold=10,
        #     alpha_matting_erode_size=20,
        #     background_color="black",
        # )

        latent = VAEEncode(image, self.main_vae)

        return latent, image

    def iter_wf(self, controller):
        cls = type(self)

        steps = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)

            if callable(str) and getattr(attr, "_is_gen_step", False):
                method = getattr(self, attr_name)
                if attr._step_label in controller or attr._default_step:
                    steps.append((attr._step_order, method, attr._step_label))

        steps = [(x[1], x[2]) for x in sorted(steps, key=lambda x: x[0])]

        for method, label in steps:
            yield method, label

    def generate_preview_callback(self):

        def _preview_callback(task, node_id, image):
            if self.preview_callback:
                self.preview_callback(image)

        return _preview_callback

    def generate(self, controller):
        results = []
        with Workflow(queue=False) as wf:
            _, _, _, _, _, latent, _ = CRAspectRatio(aspect_ratio=self.resolution)
            image = VAEDecode(latent, self.main_vae)

            for func, label in self.iter_wf(controller):
                latent, image = func(latent, image)

                res = PreviewImage(image)
                results.append((res, label))

        wf.queue()
        [
            res.task.add_preview_callback(self.generate_preview_callback())
            for res, _ in results
        ]

        for result, label in results:
            yield result.wait(), label

    @staticmethod
    def cancel_current():
        queue.cancel_current()

    @staticmethod
    def cancel_remaining():
        queue.cancel_remaining()

    @staticmethod
    def cancel_all():
        queue.cancel_all()
