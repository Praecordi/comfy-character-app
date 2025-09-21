from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, Dict
import importlib
import pkgutil
import sys


import comfy_nodes as csn
from utils import expand_iterations_linear
from workflow.state import WorkflowState, WorkflowContext

_STEP_REGISTRY: Dict[str, Type["WorkflowStep"]] = {}


def register_step(cls: Type["WorkflowStep"]) -> Type["WorkflowStep"]:
    _STEP_REGISTRY[cls.metadata.label] = cls
    return cls


def get_steps():
    steps = []
    for label, cls in _STEP_REGISTRY.items():
        if not cls.metadata.default_enabled:
            steps.append((cls.metadata.order, cls.metadata.label))

    return [x[1] for x in sorted(steps, key=lambda x: x[0])]


@dataclass
class WorkflowMetadata:
    label: str
    order: int
    default_enabled: bool = False


class WorkflowStep(ABC):
    metadata: WorkflowMetadata

    def __init__(self, context: WorkflowContext):
        """
        context: The WorkflowContext instance
        This allows access to models, configs, VAE, etc.
        """
        self.ctx = context

    @abstractmethod
    def run(self, state: WorkflowState) -> WorkflowState:
        pass

    def _scale_cfg(self, cfg, scale_for_cn=False):
        scale = False
        if self.ctx.style_image is not None:
            scale = True
        else:
            if scale_for_cn:
                if self.ctx.cn_image is not None:
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
        model,
        positive,
        negative,
        steps,
        cfg,
        denoise,
        num_iterations=1,
        seed_offset=0,
        optional_mask=None,
        apply_cn=True,
        cn_strength=0.5,
        cn_limits=(0, 1),
    ):
        ctx = self.ctx
        steps = expand_iterations_linear(steps, num_iterations)
        cfg = expand_iterations_linear(cfg, num_iterations)
        denoise = expand_iterations_linear(denoise, num_iterations)
        ratio = scale ** (1 / num_iterations)

        base_noise = csn.RandomNoise(ctx.base_seed + seed_offset)
        for i in range(num_iterations):
            if apply_cn:
                cn_positive, cn_negative = csn.ControlNetApplyAdvanced(
                    positive=positive,
                    negative=negative,
                    control_net=ctx.cn,
                    image=csn.VAEDecode(latent, ctx.vae),
                    strength=cn_strength,
                    start_percent=cn_limits[0],
                    end_percent=cn_limits[1],
                    vae=ctx.vae,
                )
            else:
                cn_positive, cn_negative = positive, negative

            latent = csn.NNLatentUpscale(latent, "SDXL", ratio)
            if optional_mask is not None:
                latent = csn.SetLatentNoiseMask(latent, optional_mask)

            sigmas = csn.BasicScheduler(
                model=model,
                scheduler=ctx.scheduler_name,
                steps=steps[i],
                denoise=denoise[i],
            )

            guider = csn.CFGGuider(
                model=model, positive=cn_positive, negative=cn_negative, cfg=cfg[i]
            )

            _, latent = csn.SamplerCustomAdvanced(
                noise=base_noise,
                guider=guider,
                sampler=ctx.sampler,
                sigmas=sigmas,
                latent_image=latent,
            )

        return latent

    def _iterative_image_upscale(
        self,
        image,
        scale,
        model,
        positive,
        negative,
        steps,
        cfg,
        denoise,
        num_iterations=1,
        seed_offset=0,
        optional_mask=None,
        sharpen=0.8,
        apply_color_match=False,
        apply_cn=True,
        cn_strength=0.5,
        cn_limits=(0, 1),
    ):
        ctx = self.ctx
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
            if apply_cn:
                cn_positive, cn_negative = csn.ControlNetApplyAdvanced(
                    positive=positive,
                    negative=negative,
                    control_net=ctx.cn,
                    image=image,
                    strength=cn_strength,
                    start_percent=cn_limits[0],
                    end_percent=cn_limits[1],
                    vae=ctx.vae,
                )
            else:
                cn_positive, cn_negative = positive, negative

            if ctx.upscale_model:
                image, _ = csn.CRUpscaleImage(
                    image=image,
                    upscale_model=ctx.upscale_model_name,
                    mode=csn.CRUpscaleImage.mode.rescale,
                    rescale_factor=ratio,
                    resampling_method=csn.CRUpscaleImage.resampling_method.lanczos,
                )
            else:
                image = csn.ImageScaleBy(
                    image=image,
                    upscale_method=csn.ImageScaleBy.upscale_method.lanczos,
                    scale_by=ratio,
                )

            if sharpen > 0:
                image = csn.ImageCASharpening(image, sharpen)
                prev_image = image

            latent = csn.VAEEncode(image, ctx.vae)
            if optional_mask is not None:
                latent = csn.SetLatentNoiseMask(latent, optional_mask)

            base_noise = csn.RandomNoise(ctx.base_seed + seed_offset)

            sigmas = csn.BasicScheduler(
                model=model,
                scheduler=ctx.scheduler_name,
                steps=steps[i],
                denoise=denoise[i],
            )

            guider = csn.CFGGuider(
                model=model, positive=cn_positive, negative=cn_negative, cfg=cfg[i]
            )

            _, latent = csn.SamplerCustomAdvanced(
                noise=base_noise,
                guider=guider,
                sampler=ctx.sampler,
                sigmas=sigmas,
                latent_image=latent,
            )

            image = csn.VAEDecode(latent, ctx.vae)

            if apply_color_match:
                image = csn.ImageColorMatch(
                    image=image,
                    reference=prev_image,
                    color_space=csn.ImageColorMatch.color_space.LAB,
                    factor=0.5,
                )

        return image

    def __str__(self):
        return f"{self.__class__.__name__} ({self.metadata.label})"


for _, module_name, _ in pkgutil.iter_modules(__path__):
    if module_name.startswith("_") or module_name == "__init__":
        continue

    full_module_name = f"{__name__}.{module_name}"
    if full_module_name not in sys.modules:
        importlib.import_module(full_module_name)
