from typing import List, Tuple
import gradio as gr

PREVIEW_REFRESH_RATE = 0.5

from ui.runner import WorkflowRunner
from ui.layout import MainLayout, CharacterManagerLayout
from ui.generator_events import bind_events as bind_generator_events
from ui.cm_events import bind_events as bind_cm_events


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

        self.runner = WorkflowRunner()

    def create_ui(self):
        with gr.Blocks(title="Praecordi's Character Generator", head_paths=["head.html"]) as demo:
            gr.Markdown("# Praecordi's Character Generator")

            with gr.Tab("Generator"):
                main_components = MainLayout.create(
                    self.checkpoints, self.resolutions, self.upscalers
                )
            with gr.Tab("Character Manager"):
                cm_components = CharacterManagerLayout.create()

            main_components["browser_state"] = gr.BrowserState(
                storage_key="ccw-ui-state", secret="ccw-secret"
            )

            bind_generator_events(
                demo, main_components, self.runner, self.checkpoints, self.resolutions
            )

            bind_cm_events(demo, cm_components)

        return demo
