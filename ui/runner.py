import random
from threading import Event, Lock
import gradio as gr

from comfy_nodes import queue
from workflow.core import CharacterWorkflow
from workflow.utils import make_output_text


class WorkflowRunner:
    def __init__(self):
        self._preview_lock = Lock()
        self._latest_preview = None
        self._stop_event = Event()
        self._stop_wf = False

    def _push(self, task, node_id, image):
        with self._preview_lock:
            self._latest_preview = image

    def _reset_preview(self):
        with self._preview_lock:
            self._latest_preview = None

    def generate(self, in_state: dict):
        self._reset_preview()
        self._stop_event.clear()

        in_state["base_seed"] = (
            random.randint(0, 1e10)
            if in_state["base_seed"] == -1
            else in_state["base_seed"]
        )
        in_state["perturb_seed"] = (
            random.randint(0, 1e10)
            if in_state["perturb_seed"] == -1
            else in_state["perturb_seed"]
        )
        in_state["preview_callback"] = self._push

        wf = CharacterWorkflow(in_state)

        res_agg = []

        out_text = make_output_text(wf.ctx)

        yield gr.update(), out_text

        try:
            for result, label in wf.generate(in_state["process_controller"]):
                if result is not None:
                    image = result.wait()
                    image_size = image[0].size

                    res_agg += [
                        (image[0], f"{label} ({image_size[0]} X {image_size[1]})")
                    ]

                    yield gr.update(value=res_agg), gr.update()
                else:
                    yield gr.update(), gr.update()
        except IndexError:
            yield gr.update(value=res_agg), gr.update()
        finally:
            self._stop_event.set()

    def interrupt(self):
        queue.cancel_current()

    def get_preview(self):
        if self._stop_event.is_set():
            return (None, "stopped")

        with self._preview_lock:
            return (self._latest_preview, "running")
