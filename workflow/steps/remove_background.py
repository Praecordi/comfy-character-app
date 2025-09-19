from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class ImageUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Remove Background", order=7)

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        image, _ = LayerMaskRmBgUltraV2(
            image,
            LayerMaskRmBgUltraV2.detail_method.VITMatte,
            detail_dilate=18,
            detail_erode=18,
        )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
