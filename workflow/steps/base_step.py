from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class BaseGenStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Base Generation", order=0, default_enabled=True)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        if ctx.input_image:
            image = ctx.input_image
            latent = VAEEncode(image, ctx.vae)

            return state.update(latent=latent, image=image)
        else:
            latent = state.latent

            if not ctx.cn_image is None:
                positive, negative = ControlNetApplyAdvanced(
                    ctx.positive_conditioning,
                    ctx.negative_conditioning,
                    ctx.cn,
                    ctx.cn_image,
                    ctx.cn_strength / 100,
                    0,
                    1,
                    ctx.vae,
                )
            else:
                positive, negative = (
                    ctx.positive_conditioning,
                    ctx.negative_conditioning,
                )

            base_noise = RandomNoise(ctx.base_seed)
            perturb_noise = RandomNoise(ctx.perturb_seed)

            sigmas = BasicScheduler(
                model=ctx.model,
                scheduler=ctx.scheduler_name,
                steps=ctx.steps["base_gen"],
                denoise=1,
            )
            _, sigmas2 = SplitSigmas(sigmas, int(ctx.steps["base_gen"] * 0.65))

            sampler = KSamplerSelect(ctx.sampler_name)

            guider = CFGGuider(
                model=ctx.model,
                positive=positive,
                negative=negative,
                cfg=self._scale_cfg(ctx.cfg["base_gen"], scale_for_cn=True),
            )

            latent = AddNoise(ctx.model, perturb_noise, sigmas2, latent)

            latent, _ = SamplerCustomAdvanced(
                noise=base_noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent,
            )

            image = VAEDecode(latent, ctx.vae)

            return state.update(latent=latent, image=image)
