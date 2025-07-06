import comfy_nodes as csn


class CaptionWorkflow:
    def __init__(self, ui_state: dict):
        if ui_state["input_image"] is not None:
            self.image, _ = csn.LoadImage(ui_state["input_image"])
        else:
            self.image = None

        self.florence_model = csn.Florence2ModelLoader(
            csn.Florence2ModelLoader.model.CogFlorence_2_2_Large,
            convert_to_safetensors=True,
        )

    async def generate(self):
        with csn.Workflow(queue=False) as wf:
            _, _, caption, _ = csn.Florence2Run(
                self.image,
                self.florence_model,
                task=csn.Florence2Run.task.more_detailed_caption,
                seed=69,
            )

            caption = csn.PreviewAny(caption)

        await wf._queue()

        try:
            return await caption
        except RuntimeError as e:
            print("Error in captioning")
            print(e)
