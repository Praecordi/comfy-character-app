from comfy_nodes import *

from workflow.state import WorkflowState
from workflow.steps import WorkflowStep, register_step, WorkflowMetadata


@register_step
class ImageUpscaleStep(WorkflowStep):
    metadata = WorkflowMetadata(label="Remove Background", order=7)
    make_transparent = True

    def run(self, state: WorkflowState) -> WorkflowState:
        ctx = self.ctx

        image = state.image

        if self.make_transparent:
            session = TransparentBGSession(TransparentBGSession.mode.base, use_jit=True)
            image, _ = ImageRemoveBackground(session, image)
        else:
            image = ImageRembgRemoveBackground(
                image,
                transparency=False,
                model=ImageRembgRemoveBackground.model.silueta,
                alpha_matting=True,
                alpha_matting_background_threshold=240,
                alpha_matting_foreground_threshold=10,
                alpha_matting_erode_size=20,
                background_color=ImageRembgRemoveBackground.background_color.black,
            )

        latent = VAEEncode(image, ctx.vae)

        return state.update(latent=latent, image=image)
