from typing import List, Tuple
import gradio as gr

PREVIEW_REFRESH_RATE = 0.5

from ui.runner import WorkflowRunner
from ui.layout import MainLayout
from ui.events import bind_events


class UI:
    def __init__(
        self,
        checkpoints: List[Tuple[str, str]],
        resolutions: List[Tuple[str, str]],
        upscalers: List[Tuple[str, str]],
    ):
        self.checkpoints = checkpoints
        self.resolutions = resolutions
        self.upscalers = upscalers
        self.components = {}

        self.runner = WorkflowRunner()

    def create_ui(self):
        with gr.Blocks(
            title="Character Generator", head_paths=["head.html"]
        ) as demo:
            gr.Markdown("# Character Generator")

            self.components = MainLayout.create(
                self.checkpoints, self.resolutions, self.upscalers
            )

            self.components["browser_state"] = gr.BrowserState(
                storage_key="ccw-ui-state", secret="ccw-secret"
            )

            bind_events(
                demo, self.components, self.runner, self.checkpoints, self.resolutions
            )

        return demo
