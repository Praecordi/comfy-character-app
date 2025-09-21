from typing import List, Optional, TYPE_CHECKING
import re

from constants import characters

if TYPE_CHECKING:
    from workflow.state import WorkflowContext


def make_output_text(ctx: "WorkflowContext"):
    text = """Base Seed: {bseed}

Perturb Seed: {pseed}

Positive Prompt: {pprompt}

Negative Prompt: {nprompt}

Face Prompt: {fprompt}

Skin Prompt: {sprompt}

Hair Prompt: {hprompt}

Eyes Prompt: {eprompt}""".format(
        bseed=ctx.base_seed,
        pseed=ctx.perturb_seed,
        pprompt=ctx.positive_prompt,
        nprompt=ctx.negative_prompt,
        fprompt=ctx.face_prompt,
        sprompt=ctx.skin_prompt,
        hprompt=ctx.hair_prompt,
        eprompt=ctx.eyes_prompt,
    )

    return text


def clean_prompt_string(prompt: str) -> List[str]:
    if not prompt:
        return []
    return [
        x.strip() for x in re.sub(r"\r\n|\n|\r", " ", prompt).split(",") if x.strip()
    ]


def build_conditioning_prompt(
    base: str, style: Optional[str] = None, score_boost: bool = False
) -> str:
    parts = clean_prompt_string(base)

    if score_boost:
        parts = ["score_9"] + parts

    if style:
        parts += clean_prompt_string(style)

    return ", ".join(parts)


def substitute_character_tokens(prompt: str, character: str) -> str:
    if make_key(character) == "custom":
        char_data = {}
    else:
        char_data = characters[make_key(character)]
    for token, val in char_data.items():
        if isinstance(val, list):
            val = ", ".join(val)
        prompt = prompt.replace(f"{{{token}}}", val)
    return prompt


def expand_iterations_linear(value, n, callback=lambda x: x):
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
        if n == 1:
            return [callback(end)] * n

        step_size = (end - start) / (n - 1)
        return [callback(start + i * step_size) for i in range(n)]
    elif isinstance(value, (int, float)):
        return [callback(value)] * n
    else:
        raise ValueError("Provide a value of int/float or tuple of size 2 to expand")


def expand_iterations_geometric(value, n, callback=lambda x: x):
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
        if n == 1:
            return [callback(end)] * n

        if start <= 0 or end <= 0:
            raise ValueError("Both start and end must be positive")

        ratio = (end / start) ** (1 / (n - 1))
        return [callback(start * (ratio**i)) for i in range(n)]
    elif isinstance(value, (int, float)):
        return [callback(value)] * n
    else:
        raise ValueError("Provide a value of int/float or tuple of size 2 to expand")


def make_character_description(character):
    char_key = make_key(character)
    if char_key == "custom" or char_key not in characters:
        return ""
    else:
        char_dict = characters[char_key]

        desc = "\n".join([f"- **{{{key}}}**" for key in char_dict.keys()])

        return desc


def make_key(name):
    return name.replace(" ", "_").lower()


def make_name(key):
    return key.replace("_", " ").title()
