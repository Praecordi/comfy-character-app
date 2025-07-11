import comfy_nodes as csn

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict


@dataclass
class WorkflowState:
    latent: Optional[csn.Latent]
    image: Optional[csn.Image]
    mask: Optional[csn.Mask] = None

    def update(self, **kwargs) -> "WorkflowState":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


@dataclass
class WorkflowContext:
    model: csn.Checkpoints = field(default=None)
    clip: csn.Clip = field(default=None)
    vae: csn.Vae = field(default=None)
    sampler_name: csn.Samplers = field(default=None)
    scheduler_name: csn.Schedulers = field(default=None)
    sampler: csn.Sampler = field(default=None)
    steps: Dict[str, Union[int, Tuple]] = field(default=None)
    cfg: Dict[str, Union[int, float, Tuple]] = field(default=None)
    resolution: str = field(default=None)

    latent_scale: float = field(default=None)
    latent_adherence: float = field(default=None)
    image_scale: float = field(default=None)
    image_adherence: float = field(default=None)

    sam_model: csn.SamModel = field(default=None)
    gd_model: csn.GroundingDinoModel = field(default=None)

    base_seed: int = field(default=None)
    perturb_seed: int = field(default=None)

    style_image: Optional[csn.Image] = field(default=None)
    style_strength: Optional[float] = field(default=None)
    cn_image: Optional[csn.Image] = field(default=None)
    cn_strength: Optional[float] = field(default=None)
    face_image: Optional[csn.Image] = field(default=None)

    cn: csn.ControlNet = field(default=None)
    upscale_model_name: str = field(default=None)
    upscale_model: Optional[csn.UpscaleModel] = field(default=None)

    input_image: Optional[csn.Image] = field(default=None)
    positive_prompt: str = field(default=None)
    negative_prompt: str = field(default=None)
    face_prompt: str = field(default=None)
    skin_prompt: str = field(default=None)
    hair_prompt: str = field(default=None)
    eyes_prompt: str = field(default=None)

    positive_conditioning: csn.Conditioning = field(default=None)
    negative_conditioning: csn.Conditioning = field(default=None)
    face_conditioning: csn.Conditioning = field(default=None)
    skin_conditioning: csn.Conditioning = field(default=None)
    hair_conditioning: csn.Conditioning = field(default=None)
    eyes_conditioning: csn.Conditioning = field(default=None)

    instantid: csn.Instantid = field(default=None)
    faceanalysis: csn.Faceanalysis = field(default=None)
    instantid_cn: csn.ControlNet = field(default=None)
    swap_method: str = field(default=None)

    def update(self, **kwargs) -> "WorkflowContext":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
