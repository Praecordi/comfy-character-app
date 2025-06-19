from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class BaseGenStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Base Generation", order=0, default_enabled=True)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

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
            positive, negative = ctx.positive_conditioning, ctx.negative_conditioning

        perturb_noise = RandomNoise(ctx.perturb_seed)

        sigmas = BasicScheduler(ctx.model, ctx.scheduler, ctx.steps["base_gen"], 1)
        _, sigmas2 = SplitSigmas(sigmas, int(ctx.steps["base_gen"] * 0.65))

        latent = AddNoise(ctx.model, perturb_noise, sigmas2, latent)

        latent = KSampler(
            model=ctx.model,
            seed=ctx.base_seed,
            steps=ctx.steps["base_gen"],
            cfg=self._scale_cfg(ctx.cfg["base_gen"], scale_for_cn=True),
            denoise=1.0,
            sampler_name=ctx.sampler,
            scheduler=ctx.scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
        )

        image = VAEDecode(latent, ctx.vae)

        return state.update(latent=latent, image=image)
