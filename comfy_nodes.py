from constants import comfyui_server_url

from comfy_script.runtime import *

load(comfyui_server_url)
from comfy_script.runtime.nodes import *

queue.watch_display(False)
