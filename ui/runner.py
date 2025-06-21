import random
from queue import Queue, Empty
from threading import Thread, Event
import gradio as gr

from comfy_nodes import queue
from constants import characters
from workflow.core import CharacterWorkflow
from workflow.utils import make_output_text


class WorkflowRunner:
    def __init__(self):
        self._preview_queue = Queue()
        self._stop_event = Event()
        self._thread = None
        self._stop_wf = False

    def _push(self, task, node_id, image):
        self._preview_queue.put(image)

    def _reset_queue(self):
        self._preview_queue = Queue()

    def _start_preview_processor(self):
        self._stop_event.clear()
        self._thread = Thread(target=None, daemon=True)
        self._thread.start()

    def _stop_preview_processor(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def generate(self, in_state: dict):
        self._reset_queue()
        self._start_preview_processor()

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
            self._stop_preview_processor()

    def generate_all(self, in_state: dict):
        self._stop_wf = False
        self._reset_queue()
        self._start_preview_processor()

        character_keys = [key for key in characters if key != "Custom"]
        accumulated_images = []

        in_state["base_seed"] = (
            random.randint(0, 1e10)
            if in_state["base_seed"] == -1
            else in_state["base_seed"]
        )
        in_state["perturb_seed"] = (
            random.randint(0, 1e10)
            if in_state["perturb_seed"]
            else in_state["perturb_seed"]
        )
        in_state["preview_callback"] = self._push

        yield gr.update(value=accumulated_images), gr.update(
            value="Starting batch generation for all characters..."
        )

        try:
            for char_key in character_keys:
                char_dict = characters[char_key]
                char_name = char_key.capitalize()

                copy_state = in_state.copy()

                copy_state["face_prompt"] = char_dict["face"]
                copy_state["hair_prompt"] = char_dict["hair"]
                copy_state["eyes_prompt"] = char_dict["eyes"]
                copy_state["face_images"] = char_dict["face_reference"]
                copy_state["character"] = char_name

                wf = CharacterWorkflow(copy_state)

                for result, label in wf.generate(copy_state["process_controller"]):
                    if self.stop_wf:
                        raise InterruptedError("Workflow interrupted")

                    if result is not None:
                        image = result.wait()
                        image_size = image[0].size
                        label_str = (
                            f"{char_name}: {label} ({image_size[0]} X {image_size[1]})"
                        )
                        accumulated_images.append((image[0], label_str))

                    out_text = f"Generating: {char_name}\n" + make_output_text(wf.ctx)

                    yield gr.update(value=accumulated_images), gr.update(value=out_text)
        except InterruptedError:
            yield gr.update(value=accumulated_images), gr.update(
                value="Batch generation interrupted!"
            )
            self._stop_preview_processor()
            return

        yield gr.update(value=accumulated_images), gr.update(
            value="Batch generation completed!"
        )

    def interrupt(self):
        queue.cancel_current()

    def interrupt_all(self):
        queue.cancel_all()
        self.stop_wf = True

    def get_preview(self):
        if self._stop_event.is_set():
            return (None, "stopped")

        try:
            image = self._preview_queue.get_nowait()
            return (image, "running")
        except Empty:
            return (None, "running")
